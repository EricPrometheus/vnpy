"""1 日标签版遗传规划因子挖掘：修复"持有期错配"。

背景：run_gp_mining.py 用"未来 5 日收益"做标签挖出的因子，在 1 日尺度上
反转（Top5-Bottom5 尾部价差约 -0.19%/日），导致日度调仓组合巨亏。本脚本
把标签换成"未来 1 日收益"重新挖掘，其余流程（适应度、进化配置、名人堂
复检、表达式翻译）直接复用 run_gp_mining.py，保证两版唯一差异是标签。

输出隔离（重要）：结果只写 research/gp_1d/ 下的独立 JSON 与 report.md，
绝不触碰 research/mined_manifest.json 的 "gp" 段——那是 5 日版成果。

用法（仓库根目录运行；挖掘一轮约 7 分钟）：
    D:\\veighna_studio\\python.exe -X utf8 research/run_gp_mining_1d.py --seed 123
    D:\\veighna_studio\\python.exe -X utf8 research/run_gp_mining_1d.py --seed 42
    D:\\veighna_studio\\python.exe -X utf8 research/run_gp_mining_1d.py --report
"""

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import pandas as pd
import polars as pl
from gplearn.genetic import SymbolicTransformer

from factor_common import (
    FEATURES,
    FEATURE_NAMES,
    RESEARCH_ROOT,
    TRAIN,
    VALID,
    daily_rank_ic_series,
    load_manifest,
    load_panel,
    safe_join,
    segment_mask,
)
from run_gp_mining import (
    FUNCTION_SET,
    GENERATIONS,
    HALL_OF_FAME,
    MAX_MUTUAL_CORR,
    MIN_VALID_IC,
    N_KEEP,
    PARSIMONY,
    POPULATION_SIZE,
    TOP_K,
    TOURNAMENT_SIZE,
    SegmentArrays,
    _segment_arrays,
    make_ic_fitness,
    select_programs,
)

SOURCE_TAG: str = "gp1d"          # 独立来源标记，区别于清单里的 5 日版 "gp"
TEST_1D: tuple[str, str] = ("2026-01-01", "2026-09-21")  # 测试段（任务指定口径）
SEGMENTS: dict[str, tuple[str, str]] = {
    "train": TRAIN,
    "valid": VALID,
    "test": TEST_1D,
}
# 1 日标签的 vnpy 表达式（负 delay 表示未来，与 5 日版标签写法同款）
LABEL_1D_VNPY: str = "ts_delay(close, -1) / close - 1"
_LABEL_1D_EXPR: pl.Expr = (
    pl.col("close").shift(-1) / pl.col("close") - 1
).over("vt_symbol")
REPORT_SEEDS: list[int] = [123, 42]  # 报告汇总时依次读取的种子
TAIL_K: int = 5                     # 尾部价差的 Top/Bottom 截断数


def build_panel_1d() -> pl.DataFrame:
    """构造与 build_factor_table 同构的面板，唯一差异：标签为 1 日前向收益。"""
    df: pl.DataFrame = load_panel()
    for feature_name, (expr, _vnpy) in FEATURES.items():
        df = df.with_columns(expr.alias(feature_name))
    df = df.with_columns(_LABEL_1D_EXPR.alias("label"))
    df = df.drop_nulls(subset=[*FEATURE_NAMES, "label"])
    return df.sort(["datetime", "vt_symbol"])


def build_gp(seed: int, fitness: Callable) -> SymbolicTransformer:
    """与 run_gp_mining 完全一致的进化配置（常量复用，超参数不复制）。"""
    return SymbolicTransformer(
        generations=GENERATIONS,
        population_size=POPULATION_SIZE,
        tournament_size=TOURNAMENT_SIZE,
        function_set=FUNCTION_SET,
        metric=fitness,
        hall_of_fame=HALL_OF_FAME,
        n_components=N_KEEP,
        parsimony_coefficient=PARSIMONY,
        p_crossover=0.7,
        p_subtree_mutation=0.15,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        max_samples=1.0,
        verbose=1,
        random_state=seed,
        n_jobs=1,
    )


def tail_spread_pandas(values: pd.Series, target: pd.Series,
                       dates: pd.Series, top_k: int = TAIL_K) -> tuple[float, int]:
    """逐日 TopK-BottomK 尾部价差（多头日均收益 - 空头日均收益）。

    这是本次任务的成败判据：5 日版因子在 1 日尺度上该值为负（约 -0.19%/日），
    1 日版因子必须转正才算修复持有期错配。
    """
    frame: pd.DataFrame = pd.DataFrame(
        {"value": values.to_numpy(), "target": target.to_numpy()},
        index=pd.Index(dates.to_numpy(), name="date"),
    ).dropna()

    def _daily_spread(group: pd.DataFrame) -> float:
        if len(group) < 2 * top_k:
            return float("nan")
        ranked: pd.DataFrame = group.sort_values("value")
        top_ret: float = float(ranked["target"].tail(top_k).mean())
        bottom_ret: float = float(ranked["target"].head(top_k).mean())
        return top_ret - bottom_ret

    daily: pd.Series = frame.groupby(level="date").apply(_daily_spread).dropna()
    if daily.empty:
        return float("nan"), 0
    return float(daily.mean()), int(daily.size)


def prepare_eval_base(panel: pl.DataFrame) -> pl.DataFrame:
    """在完整面板上预计算 1 日标签列，供各因子评估复用（标签只算一次）。

    注意必须保留 open/high/low/close 等原始列：vnpy 引擎求值因子时要用。
    """
    from vnpy.alpha.dataset.utility import calculate_by_expression

    label_data: pl.Series = calculate_by_expression(panel, LABEL_1D_VNPY)["data"]
    return panel.with_columns(label_data.alias("target"))


def segment_stats(base_with_value: pl.DataFrame,
                  segment: tuple[str, str]) -> dict[str, float]:
    """某一段的引擎口径统计：日均 Rank IC + Top5-Bottom5 日均价差（%/日）。"""
    pdf: pd.DataFrame = (
        base_with_value.filter(segment_mask(segment)).to_pandas().dropna()
    )
    ic_series: pd.Series = daily_rank_ic_series(
        pdf["value"], pdf["target"], pdf["datetime"]
    )
    spread: float
    n_days: int
    spread, n_days = tail_spread_pandas(pdf["value"], pdf["target"], pdf["datetime"])
    return {
        "ic": round(float(ic_series.mean()), 6),
        "spread_pct": round(spread * 100, 4),
        "days": n_days,
    }


def evaluate_expression(eval_base: pl.DataFrame,
                        vnpy_expr: str) -> dict[str, dict]:
    """用 vnpy 公式引擎求值，返回三段 {ic, spread_pct, days}（引擎=实盘口径）。"""
    from vnpy.alpha.dataset.utility import calculate_by_expression

    value_data: pl.Series = calculate_by_expression(eval_base, vnpy_expr)["data"]
    base_with_value: pl.DataFrame = (
        eval_base.select(["datetime", "vt_symbol", "target"])
        .with_columns(value_data.alias("value"))
    )
    return {
        seg_name: segment_stats(base_with_value, seg)
        for seg_name, seg in SEGMENTS.items()
    }


def save_seed_result(seed: int, factors: list[dict]) -> Path:
    """把单种子结果写到 research/gp_1d/ 独立 JSON，不碰 mined_manifest.json。"""
    payload: dict = {
        "seed": seed,
        "source": SOURCE_TAG,
        "label_vnpy": LABEL_1D_VNPY,
        "segments": {name: list(seg) for name, seg in SEGMENTS.items()},
        "config": {
            "population": POPULATION_SIZE,
            "generations": GENERATIONS,
            "min_valid_ic": MIN_VALID_IC,
            "max_mutual_corr": MAX_MUTUAL_CORR,
            "top_k": TOP_K,
        },
        "factors": factors,
    }
    out_path: Path = safe_join(f"gp_1d/mine_s{seed}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"    已写入 {out_path}")
    return out_path


def mine_one_seed(seed: int) -> None:
    """单种子完整流程：1 日面板 -> 进化 -> 名人堂复检 -> 引擎评估 -> 写 JSON。"""
    t0: float = time.time()
    print(f"[0] research 根目录: {RESEARCH_ROOT}")
    panel: pl.DataFrame = build_panel_1d()
    print(f"[1] 1 日标签面板: {panel.height} 行, "
          f"{panel['vt_symbol'].n_unique()} 个币, "
          f"{panel['datetime'].n_unique()} 个交易日")
    segments: dict[str, SegmentArrays] = {
        seg_name: _segment_arrays(panel, seg) for seg_name, seg in SEGMENTS.items()
    }
    x_train, y_train, date_codes, day_bounds = segments["train"]
    print(f"[2] 遗传规划进化: 种群 {POPULATION_SIZE} x {GENERATIONS} 代, "
          f"seed={seed} ...")
    gp: SymbolicTransformer = build_gp(seed, make_ic_fitness(date_codes, day_bounds))
    gp.fit(x_train, y_train)
    candidates: list[dict] = select_programs(gp, segments)
    print(f"[3] 复检池 {N_KEEP} 个公式，入选 {len(candidates)} 个")
    print("[4] vnpy 引擎复算三段 IC 与尾部价差 ...")
    base: pl.DataFrame = prepare_eval_base(panel)
    factors: list[dict] = []
    for i, cand in enumerate(candidates, 1):
        engine_stats: dict[str, dict] = evaluate_expression(base, cand["vnpy_expr"])
        factors.append({
            "name": f"gp1d_s{seed}_{i:03d}",
            "source": SOURCE_TAG,
            "vnpy_expr": cand["vnpy_expr"],
            "formula_gp": cand["formula_gp"],
            "train_ic": round(cand["train_ic"], 6),
            "valid_ic": round(cand["valid_ic"], 6),
            "test_ic": round(cand["test_ic"], 6),
            "engine": engine_stats,
        })
        gap: float = abs(engine_stats["valid"]["ic"] - cand["valid_ic"])
        if gap > 0.015:
            print(f"    [警告] 引擎复算偏差较大: gplearn={cand['valid_ic']:.4f} "
                  f"vs 引擎={engine_stats['valid']['ic']:.4f}")
        print(f"  gp1d_s{seed}_{i:03d}  train={cand['train_ic']:+.4f} "
              f"valid={cand['valid_ic']:+.4f} test={cand['test_ic']:+.4f} "
              f"test_spread={engine_stats['test']['spread_pct']:+.3f}%/日")
    save_seed_result(seed, factors)
    print(f"[5] 总耗时 {time.time() - t0:.0f}s")


def load_seed_payloads() -> list[dict]:
    """按固定种子顺序读取 research/gp_1d/ 下的挖掘结果（缺失则报错提示）。"""
    payloads: list[dict] = []
    for seed in REPORT_SEEDS:
        path: Path = safe_join(f"gp_1d/mine_s{seed}.json")
        if not path.is_file():
            raise FileNotFoundError(f"缺少挖掘结果 {path}，请先运行 --seed {seed}")
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
    return payloads


def baseline_rows(base: pl.DataFrame) -> list[dict]:
    """对照组：清单 gp 段（5 日版因子）在同一把尺子下的 1 日尺度表现。"""
    rows: list[dict] = []
    for fac in load_manifest().get("factors", []):
        if fac.get("source") != "gp":
            continue
        rows.append({
            "name": fac["name"],
            "vnpy_expr": fac["vnpy_expr"],
            "stats": evaluate_expression(base, fac["vnpy_expr"]),
        })
    return rows


def _render_intro(panel: pl.DataFrame) -> str:
    """报告第一节：背景、方法与口径说明。"""
    return "\n".join([
        "# 1 日标签遗传规划因子挖掘报告",
        "",
        "## 一、背景与方法",
        "",
        "- 背景：5 日标签挖出的因子（mined_manifest.json 的 gp 段）在 1 日尺度反转，"
        "Top5-Bottom5 尾部价差约 -0.19%/日，日度调仓组合因此巨亏（持有期错配）。",
        "- 方法：与 run_gp_mining.py 完全同一套流程（gplearn 遗传规划、日均截面 Rank IC "
        "适应度、名人堂复检：验证段 IC>=0.02 + 符号一致 + 相关去重 0.7、vnpy 表达式翻译），"
        "唯一差异是把标签从 5 日前向收益换成 1 日前向收益。",
        f"- 数据：{panel['vt_symbol'].n_unique()} 个币种日线，"
        f"标签 = {LABEL_1D_VNPY}。",
        "- 分段：训练 2023-01-01~2024-12-31 / 验证 2025 / 测试 2026-01-01~2026-09-21。",
        "- 口径：表中 IC 与价差均为 vnpy 公式引擎复算值（与实盘表达式一致）；"
        "价差 = 每日按因子值排序后 Top5 均值收益 - Bottom5 均值收益，再对日期取平均，"
        "单位 %/日。",
        "- 清单隔离：结果只写 research/gp_1d/ 下的独立 JSON，"
        "mined_manifest.json 的 gp 段（5 日版成果）保持原样。",
        "",
    ])


def _render_seed_sections(payloads: list[dict]) -> str:
    """报告第二节：各种子的入选因子表（三段 IC 与尾部价差）。"""
    chunks: list[str] = ["## 二、挖掘结果\n"]
    for payload in payloads:
        factors: list[dict] = payload["factors"]
        chunks.append(f"### 种子 {payload['seed']}：入选 {len(factors)} 个因子\n")
        if not factors:
            chunks.append("没有公式通过复检门槛（验证段 IC>=0.02）。\n")
            continue
        chunks.append("| 因子 | 训练IC | 验证IC | 测试IC | "
                      "训练价差(%/日) | 验证价差(%/日) | 测试价差(%/日) |")
        chunks.append("| --- | --- | --- | --- | --- | --- | --- |")
        for fac in factors:
            eng: dict = fac["engine"]
            chunks.append(
                f"| {fac['name']} | {eng['train']['ic']:+.4f} "
                f"| {eng['valid']['ic']:+.4f} | {eng['test']['ic']:+.4f} "
                f"| {eng['train']['spread_pct']:+.3f} "
                f"| {eng['valid']['spread_pct']:+.3f} "
                f"| {eng['test']['spread_pct']:+.3f} |"
            )
        best: dict = max(factors, key=lambda f: f["engine"]["test"]["spread_pct"])
        chunks += ["", f"最优因子 **{best['name']}**（按测试段尾部价差）的公式"
                     "（vnpy 表达式 / gplearn 前缀式）：", "", "```",
                   best["vnpy_expr"], best["formula_gp"], "```", ""]
    return "\n".join(chunks)


def _render_baseline(baselines: list[dict]) -> str:
    """报告第三节：5 日版因子在 1 日尺度的对照复测。"""
    chunks: list[str] = [
        "## 三、对照：5 日版因子在 1 日尺度的表现\n",
        "对 mined_manifest.json gp 段的 5 日版因子，用同一套代码在同一 1 日标签"
        "面板上复测（持有期错配的直接证据）：\n",
        "| 因子 | 1日IC(训练/验证/测试) | 测试段尾部价差(%/日) |",
        "| --- | --- | --- |",
    ]
    for row in baselines:
        eng: dict = row["stats"]
        chunks.append(
            f"| {row['name']} | {eng['train']['ic']:+.4f} / "
            f"{eng['valid']['ic']:+.4f} / {eng['test']['ic']:+.4f} "
            f"| {eng['test']['spread_pct']:+.3f} |"
        )
    chunks.append("")
    return "\n".join(chunks)


def _render_verdict(payloads: list[dict], baselines: list[dict]) -> str:
    """报告第四节：成败判据——1 日版因子测试段尾部价差是否转正。"""
    spreads: list[float] = [
        fac["engine"]["test"]["spread_pct"]
        for payload in payloads for fac in payload["factors"]
    ]
    base_spreads: list[float] = [
        row["stats"]["test"]["spread_pct"] for row in baselines
    ]
    positive: list[float] = [s for s in spreads if s > 0]
    chunks: list[str] = ["## 四、结论：持有期错配是否修复\n", f"- 5 日版对照因子"
                         f"（{len(base_spreads)} 个）测试段尾部价差区间："
                         f"[{min(base_spreads):+.3f}, {max(base_spreads):+.3f}]%/日"]
    if not spreads:
        chunks.append("- 没有公式通过复检，无法判定（需放宽门槛重挖）。")
        return "\n".join(chunks)
    chunks.append(f"- 1 日版因子共 {len(spreads)} 个，测试段尾部价差为正的 "
                  f"{len(positive)} 个；价差区间 [{min(spreads):+.3f}, "
                  f"{max(spreads):+.3f}]%/日")
    if all(s > 0 for s in spreads):
        chunks.append("- **判定：持有期错配已修复**。1 日标签挖出的因子尾部价差全面转正，"
                      "与 5 日版因子在 1 日尺度的负价差形成鲜明对照，"
                      "可用于日度调仓组合。")
    elif positive:
        chunks.append("- **判定：部分修复**。仅部分因子尾部价差转正，"
                      "建议只取转正因子进入组合。")
    else:
        chunks.append("- **判定：未修复**。1 日标签仍挖不出尾部价差为正的因子。")
    return "\n".join(chunks)


def write_report() -> None:
    """汇总两种子结果 + 5 日版对照，生成 research/gp_1d/report.md。"""
    base: pl.DataFrame = prepare_eval_base(build_panel_1d())
    payloads: list[dict] = load_seed_payloads()
    baselines: list[dict] = baseline_rows(base)
    lines: list[str] = [
        _render_intro(base),
        _render_seed_sections(payloads),
        _render_baseline(baselines),
        _render_verdict(payloads, baselines),
        "",
    ]
    out_path: Path = safe_join("gp_1d/report.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已写入 {out_path}")


def main() -> None:
    """入口：--seed N 挖掘一轮；--report 汇总生成报告（不挖掘）。"""
    arg_parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="1 日标签 gplearn 因子挖掘（--seed 挖掘 / --report 出报告）"
    )
    arg_parser.add_argument("--seed", type=int, default=None,
                            help="随机种子（如 123、42）")
    arg_parser.add_argument("--report", action="store_true",
                            help="汇总 research/gp_1d/*.json 生成 report.md")
    cli_args: argparse.Namespace = arg_parser.parse_args()
    if cli_args.report:
        write_report()
    elif cli_args.seed is not None:
        mine_one_seed(cli_args.seed)
    else:
        arg_parser.error("请指定 --seed N（挖掘）或 --report（生成报告）")


if __name__ == "__main__":
    main()
