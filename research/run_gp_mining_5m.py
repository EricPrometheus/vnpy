"""1 分钟 -> 5 分钟币安永续数据的遗传规划挖矿：适应度 = 尾部价差。

与日线挖矿（run_gp_mining.py，日均截面 Rank IC 适应度）的本质区别：
- 战场：5 个一线币（BTC/ETH/BNB/SOL/XRP）币安永续 1m 线重采样的 5m bar；
- 目标：不再优化"整体排序"（Rank IC 不保证两端 extreme 排对），而是直接
  优化"因子高时做多、因子低时做空"的可交易结构——每币按因子值的
  时间序列分位数取前 20% 与后 20%，适应度 =
  mean(下一根 5m 收益 | 前 20%) - mean(下一根 5m 收益 | 后 20%)，5 币平均。

已知的简化（v1）：分位阈值取"该币在当前评估段内的全局分位"，不做滚动
窗口。后果：训练段存在轻微的段内前视（阈值用了全段分布）；测试段同口径
评估下阈值同样用了段内未来信息。严格实盘应改滚动窗口或训练段固定阈值——
脚本额外输出"训练段固定阈值"口径的测试段结果作为稳健性参照。

性能：适应度用 numpy 向量化（每币全局分位阈值一次切片），启动时自测
单次求值耗时（上限 500ms，实际约几十毫秒）。

本脚本不访问数据库、不写 mined_manifest.json（独立于日线因子项目）。
用法（仓库根目录运行）：
    D:\\veighna_studio\\python.exe -X utf8 research/run_gp_mining_5m.py --seed 123
    加 --smoke 用极小种群快速验证管线（结果仅供调试，不作正式结论）。
"""

import argparse
import ast
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
from gplearn.genetic import SymbolicTransformer
from gplearn.fitness import make_fitness

from factor_common import RESEARCH_ROOT, date_boundaries, safe_join
from run_gp_mining import _TOKEN_RE, _parse_node

# ---- 数据与输出路径（字面相对路径，经 safe_join 防穿越）----
SYMBOLS: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT")
MINUTE_DIR: str = "alpha_lab/minute"   # 1m 永续 parquet 目录（相对 research 根）
OUTPUT_DIR: str = "gp_5m"              # 本项目输出目录（相对 research 根）

# ---- 三段划分（注册后不得事后放宽）----
TRAIN: tuple[str, str] = ("2023-10-01", "2024-12-31")
VALID: tuple[str, str] = ("2025-01-01", "2025-12-31")
TEST: tuple[str, str] = ("2026-01-01", "2026-09-21")
SEGMENTS: dict[str, tuple[str, str]] = {"train": TRAIN, "valid": VALID, "test": TEST}

# ---- 尾部价差定义 ----
TAIL_Q_LO: float = 0.20        # 后 20% 分位（做空腿）
TAIL_Q_HI: float = 0.80        # 前 20% 分位（做多腿）
FEE_ROUNDTRIP: float = 0.0004  # 往返费用：maker 0.02% x 2 边

# ---- GP 超参数（正式跑约 10-30 分钟）----
POPULATION_SIZE: int = 500
GENERATIONS: int = 10
TOURNAMENT_SIZE: int = 20
HALL_OF_FAME: int = 40
N_COMPONENTS: int = 40         # 必须等于 HALL_OF_FAME，保证 transform 列与名人堂对齐
# 简约惩罚必须与适应度量纲匹配：5m 尾部价差 ~1e-5 量级，比日线 Rank IC
# （~1e-2 量级，对应 5e-4）小三个数量级；沿用 5e-4 会让惩罚淹没适应度信号，
# 进化退化为"选最短程序"（冒烟已复现：平均长度 17.8 -> 3.2）。
PARSIMONY: float = 1e-6
DEFAULT_SEED: int = 123
# 函数集与日线挖矿一致：四则运算 + abs，公式可无损翻译回特征组合文本
FUNCTION_SET: list[str] = ["add", "sub", "mul", "div", "abs"]

MAX_MUTUAL_CORR: float = 0.7   # 入选因子之间的最大相关系数（去重）
TOP_K: int = 8                 # 最多保留的因子数量
FITNESS_BUDGET_SEC: float = 0.5  # 单次适应度求值的硬约束上限

# ---- 基础特征（10 个，全部 5m bar 可算）----
# name -> (polars 表达式, 人类可读公式文本)；时序特征均按币内滚动
FEATURES_5M: dict[str, tuple[pl.Expr, str]] = {
    "ret1": (
        pl.col("close").pct_change(1).over("symbol"),
        "close/ts_delay(close,1)-1",
    ),
    "mom3": (
        pl.col("close").pct_change(3).over("symbol"),
        "close/ts_delay(close,3)-1",
    ),
    "mom12": (
        pl.col("close").pct_change(12).over("symbol"),
        "close/ts_delay(close,12)-1",
    ),
    "amp": (
        (pl.col("high") - pl.col("low")) / pl.col("close"),
        "(high-low)/close",
    ),
    "body": (
        (pl.col("close") - pl.col("open")) / pl.col("close"),
        "(close-open)/close",
    ),
    "upper_shadow": (
        (pl.col("high") - pl.max_horizontal("open", "close")) / pl.col("close"),
        "(high-max(open,close))/close",
    ),
    "lower_shadow": (
        (pl.min_horizontal("open", "close") - pl.col("low")) / pl.col("close"),
        "(min(open,close)-low)/close",
    ),
    "vola20": (
        pl.col("close").pct_change(1).rolling_std(20).over("symbol"),
        "ts_std(close/ts_delay(close,1)-1,20)",
    ),
    "volr288": (
        (pl.col("volume") / pl.col("volume").rolling_mean(288)).over("symbol"),
        "volume/ts_mean(volume,288)",
    ),
    "gap": (
        (pl.col("open") / pl.col("close").shift(1) - 1).over("symbol"),
        "open/ts_delay(close,1)-1",
    ),
}
FEATURE_NAMES_5M: list[str] = list(FEATURES_5M)

# 段数组：X 特征矩阵 / y 标签 / 每币 [起,止) 行边界（按 (symbol, datetime) 排序后连续）
SegmentArrays = tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]


def minute_file_path(symbol: str) -> Path:
    """拼出某币 1m 永续 parquet 的安全路径（白名单校验 + safe_join 防穿越）。"""
    if symbol not in SYMBOLS:
        raise ValueError(f"币种不在白名单内: {symbol}")
    return safe_join(f"{MINUTE_DIR}/{symbol}.BINANCE.UM.1m.parquet")


def resample_5m(df: pl.DataFrame, symbol: str) -> pl.DataFrame:
    """单币 1m -> 5m 重采样：OHLCV 标准聚合（first/max/min/last/sum）。"""
    tagged: pl.DataFrame = df.with_columns(pl.lit(symbol).alias("symbol"))
    return (
        tagged.sort(["symbol", "datetime"])
        .group_by_dynamic("datetime", every="5m", group_by="symbol")
        .agg(
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
        )
    )


def load_5m_panel() -> pl.DataFrame:
    """读取 5 币永续 1m 线并重采样，合并为 5m 长表。"""
    frames: list[pl.DataFrame] = []
    for symbol in SYMBOLS:
        one_min: pl.DataFrame = pl.read_parquet(minute_file_path(symbol))
        frames.append(resample_5m(one_min, symbol))
    return pl.concat(frames, how="vertical").sort(["symbol", "datetime"])


def build_factor_table() -> pl.DataFrame:
    """5m 长表 + 基础特征 + 标签（下一根 5m 收益），剔除预热/尾部无效行。"""
    df: pl.DataFrame = load_5m_panel()
    for name, (expr, _text) in FEATURES_5M.items():
        df = df.with_columns(expr.alias(name))
    # 标签与特征同口径按币分组：close.shift(-1)/close - 1
    label_expr: pl.Expr = (
        pl.col("close").shift(-1) / pl.col("close") - 1
    ).over("symbol")
    df = df.with_columns(label_expr.alias("label"))
    # 保险丝：极端行情可能产生 inf，统一转回 null 再剔除，保证 X 全有限
    for name in FEATURE_NAMES_5M:
        df = df.with_columns(
            pl.when(pl.col(name).is_finite()).then(pl.col(name)).otherwise(None).alias(name)
        )
    return df.drop_nulls(subset=[*FEATURE_NAMES_5M, "label"]).sort(["symbol", "datetime"])


def segment_filter(segment: tuple[str, str]) -> pl.Expr:
    """某时间段（含端点日全天）的行过滤表达式。"""
    start: datetime = datetime.strptime(segment[0], "%Y-%m-%d")
    end: datetime = datetime.strptime(segment[1], "%Y-%m-%d")
    return (pl.col("datetime") >= pl.datetime(start.year, start.month, start.day)) & (
        pl.col("datetime") <= pl.datetime(end.year, end.month, end.day, hour=23, minute=59)
    )


def segment_arrays(panel: pl.DataFrame, segment: tuple[str, str]) -> SegmentArrays:
    """取某时间段的 (特征矩阵 X, 标签 y, 每币行边界)。"""
    sub: pl.DataFrame = panel.filter(segment_filter(segment))
    x: np.ndarray = sub.select(FEATURE_NAMES_5M).to_numpy().astype(np.float64)
    y: np.ndarray = sub["label"].to_numpy().astype(np.float64)
    _codes, bounds = date_boundaries(sub["symbol"].to_numpy())
    return x, y, bounds


def coin_tail_means(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """单币尾部两腿均值：(前20%标签均值, 后20%标签均值)，全局分位阈值切片。"""
    if pred.size == 0:  # 空切片（gplearn 签名校验用玩具数组触发）按无差异处理
        return 0.0, 0.0
    thr_hi: float = float(np.quantile(pred, TAIL_Q_HI))
    thr_lo: float = float(np.quantile(pred, TAIL_Q_LO))
    top: np.ndarray = target[pred >= thr_hi]
    bottom: np.ndarray = target[pred <= thr_lo]
    if top.size == 0 or bottom.size == 0:  # 常数因子等退化情形，两腿无差异
        return 0.0, 0.0
    return float(top.mean()), float(bottom.mean())


def tail_legs(pred: np.ndarray, target: np.ndarray,
              bounds: list[tuple[int, int]]) -> tuple[float, float]:
    """5 币各自切尾部后取平均，返回 (做多腿均值, 做空腿均值)。"""
    pred = np.nan_to_num(pred, nan=0.0, posinf=1e9, neginf=-1e9)
    pairs: list[tuple[float, float]] = [
        coin_tail_means(pred[lo:hi], target[lo:hi]) for lo, hi in bounds
    ]
    top_leg: float = float(np.mean([pair[0] for pair in pairs]))
    bottom_leg: float = float(np.mean([pair[1] for pair in pairs]))
    return top_leg, bottom_leg


def tail_spread(pred: np.ndarray, target: np.ndarray,
                bounds: list[tuple[int, int]]) -> float:
    """尾部价差适应度：做多腿均值 - 做空腿均值（毛值，不含费用）。"""
    top_leg, bottom_leg = tail_legs(pred, target, bounds)
    return top_leg - bottom_leg


def make_tail_fitness(bounds: list[tuple[int, int]]) -> Callable:
    """构造 gplearn 自定义适应度：5 币平均尾部价差。

    注意 n_jobs 必须为 1：闭包依赖本进程内的分组边界，多进程会丢状态。
    """
    def _metric(y: np.ndarray, y_pred: np.ndarray, _weight: np.ndarray) -> float:
        return tail_spread(y_pred, y, bounds)

    return make_fitness(function=_metric, greater_is_better=True)


def measure_fitness_latency(segments: dict[str, SegmentArrays],
                            panel: pl.DataFrame) -> tuple[float, float]:
    """自测适应度单次求值耗时：训练段（实际使用）与全量 150 万行（上限口径）。"""
    rng: np.random.Generator = np.random.default_rng(DEFAULT_SEED)
    x_train, y_train, train_bounds = segments["train"]
    _codes, full_bounds = date_boundaries(panel["symbol"].to_numpy())
    fake_train: np.ndarray = rng.standard_normal(y_train.shape[0])
    fake_full: np.ndarray = rng.standard_normal(panel.height)
    t0: float = time.perf_counter()
    tail_spread(fake_train, y_train, train_bounds)
    train_sec: float = time.perf_counter() - t0
    t0 = time.perf_counter()
    tail_spread(fake_full, panel["label"].to_numpy().astype(np.float64), full_bounds)
    full_sec: float = time.perf_counter() - t0
    return train_sec, full_sec


def translate_to_vnpy_5m(formula: str) -> str:
    """gplearn 公式（如 sub(X0, mul(X1, X2))）-> 5m 特征定义的可读公式文本。

    复用 run_gp_mining 的前缀公式解析器，仅把占位符映射换成 5m 特征。
    """
    feature_asts: dict[str, ast.expr] = {
        f"X{i}": ast.parse(FEATURES_5M[name][1], mode="eval").body
        for i, name in enumerate(FEATURE_NAMES_5M)
    }
    tokens: list[str] = _TOKEN_RE.findall(formula)
    root, pos = _parse_node(tokens, 0, feature_asts)
    if pos != len(tokens):
        raise ValueError(f"公式翻译不完整: {formula}")
    return ast.unparse(ast.Expression(body=root))


def select_programs(gp: SymbolicTransformer, segments: dict[str, SegmentArrays],
                    matrices: dict[str, np.ndarray]) -> list[dict]:
    """名人堂复检：验证段毛价差必须为正 + 训练/验证符号一致 + 相关去重。"""
    train_matrix: np.ndarray = matrices["train"]
    mutual_corr: np.ndarray = np.nan_to_num(
        np.corrcoef(train_matrix, rowvar=False), nan=0.0
    )
    y_valid, bounds_valid = segments["valid"][1], segments["valid"][2]
    y_test, bounds_test = segments["test"][1], segments["test"][2]
    accepted_cols: list[int] = []
    results: list[dict] = []
    for col, prog in enumerate(gp._best_programs):
        train_spread: float = float(getattr(prog, "raw_fitness_", 0.0))
        valid_spread: float = tail_spread(matrices["valid"][:, col], y_valid, bounds_valid)
        test_spread: float = tail_spread(matrices["test"][:, col], y_test, bounds_test)
        if valid_spread <= 0.0:
            continue
        if train_spread * valid_spread <= 0:  # 训练/验证符号翻转，视为不稳定
            continue
        if any(abs(mutual_corr[col, kept]) > MAX_MUTUAL_CORR for kept in accepted_cols):
            continue
        accepted_cols.append(col)
        results.append({
            "hof_col": col,
            "formula_gp": str(prog),
            "vnpy_text": translate_to_vnpy_5m(str(prog)),
            "train_spread": train_spread,
            "valid_spread": valid_spread,
            "test_spread": test_spread,
            "test_net": test_spread - FEE_ROUNDTRIP,
        })
    results.sort(key=lambda item: item["valid_spread"], reverse=True)
    return results[:TOP_K]


def monthly_stats(test_panel: pl.DataFrame, factor: np.ndarray) -> dict[str, dict[str, float]]:
    """测试段（2026 年）逐月尾部价差：月内分位口径，毛值与净值。"""
    tagged: pl.DataFrame = test_panel.select(["symbol", "datetime", "label"]).with_columns(
        pl.Series("factor", factor),
        pl.col("datetime").dt.strftime("%Y-%m").alias("month"),
    )
    stats: dict[str, dict[str, float]] = {}
    for month in sorted(tagged["month"].unique().to_list()):
        sub: pl.DataFrame = tagged.filter(pl.col("month") == month)
        _codes, bounds = date_boundaries(sub["symbol"].to_numpy())
        gross: float = tail_spread(
            sub["factor"].to_numpy(), sub["label"].to_numpy().astype(np.float64), bounds
        )
        stats[month] = {"gross": round(gross, 8), "net": round(gross - FEE_ROUNDTRIP, 8)}
    return stats


def fixed_threshold_test_spread(train_arrays: SegmentArrays, test_arrays: SegmentArrays,
                                train_pred: np.ndarray, test_pred: np.ndarray) -> float:
    """稳健性口径：用训练段每币分位阈值原样套到测试段再算尾部价差。"""
    _x_tr, y_tr, bounds_tr = train_arrays
    _x_te, y_te, bounds_te = test_arrays
    pred_tr: np.ndarray = np.nan_to_num(train_pred, nan=0.0, posinf=1e9, neginf=-1e9)
    pred_te: np.ndarray = np.nan_to_num(test_pred, nan=0.0, posinf=1e9, neginf=-1e9)
    thresholds: list[tuple[float, float]] = [
        (float(np.quantile(pred_tr[lo:hi], TAIL_Q_LO)),
         float(np.quantile(pred_tr[lo:hi], TAIL_Q_HI)))
        for lo, hi in bounds_tr
    ]
    spreads: list[float] = []
    for (thr_lo, thr_hi), (lo, hi) in zip(thresholds, bounds_te):
        p: np.ndarray = pred_te[lo:hi]
        t: np.ndarray = y_te[lo:hi]
        top: np.ndarray = t[p >= thr_hi]
        bottom: np.ndarray = t[p <= thr_lo]
        if top.size == 0 or bottom.size == 0:
            spreads.append(0.0)
            continue
        spreads.append(float(top.mean() - bottom.mean()))
    return float(np.mean(spreads))


def enrich_candidates(candidates: list[dict], segments: dict[str, SegmentArrays],
                      matrices: dict[str, np.ndarray], test_panel: pl.DataFrame,
                      train_symbols: np.ndarray, test_symbols: np.ndarray) -> None:
    """为入选因子补齐：两腿均值、逐月稳定性、训练段固定阈值稳健口径。"""
    # 固定阈值口径要求两段币种集合一致（同 5 币，且按序对齐）
    if not np.array_equal(np.unique(train_symbols), np.unique(test_symbols)):
        raise ValueError("训练/测试段币种集合不一致，固定阈值口径不可比")
    y_valid, bounds_valid = segments["valid"][1], segments["valid"][2]
    y_test, bounds_test = segments["test"][1], segments["test"][2]
    for cand in candidates:
        col: int = cand["hof_col"]
        valid_top, valid_bottom = tail_legs(matrices["valid"][:, col], y_valid, bounds_valid)
        test_top, test_bottom = tail_legs(matrices["test"][:, col], y_test, bounds_test)
        cand["valid_top_leg"] = valid_top
        cand["valid_bottom_leg"] = valid_bottom
        cand["test_top_leg"] = test_top
        cand["test_bottom_leg"] = test_bottom
        cand["monthly_2026"] = monthly_stats(test_panel, matrices["test"][:, col])
        fixed: float = fixed_threshold_test_spread(
            segments["train"], segments["test"],
            matrices["train"][:, col], matrices["test"][:, col],
        )
        cand["test_spread_fixed_thr"] = fixed
        cand["test_net_fixed_thr"] = fixed - FEE_ROUNDTRIP


def write_result(result: dict, seed: int) -> Path:
    """把完整结果写入 research/gp_5m/mining_result_s{seed}.json（按种子区分防覆盖）。"""
    out_dir: Path = safe_join(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path: Path = out_dir / f"mining_result_s{seed}.json"
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    return out_path


def run_mining(seed: int, smoke: bool, panel: pl.DataFrame,
               segments: dict[str, SegmentArrays]) -> tuple[SymbolicTransformer, dict]:
    """gplearn 进化主流程，返回拟合好的 transformer 与段变换矩阵。"""
    population: int = 30 if smoke else POPULATION_SIZE
    generations: int = 2 if smoke else GENERATIONS
    hall_size: int = 10 if smoke else HALL_OF_FAME
    x_train, y_train, train_bounds = segments["train"]
    print(f"[2] 遗传规划进化: 种群 {population} x {generations} 代, 种子 {seed} ...")
    gp = SymbolicTransformer(
        generations=generations,
        population_size=population,
        tournament_size=TOURNAMENT_SIZE,
        function_set=FUNCTION_SET,
        metric=make_tail_fitness(train_bounds),
        hall_of_fame=hall_size,
        n_components=hall_size,
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
    gp.fit(x_train, y_train)
    # transform 的列 j 对应名人堂第 j 个公式（n_components == hall_of_fame 保证对齐）
    matrices: dict[str, np.ndarray] = {
        name: gp.transform(segments[name][0]) for name in SEGMENTS
    }
    return gp, matrices


def print_summary(candidates: list[dict], elapsed: float) -> None:
    """打印逐因子结果与预注册成败判据（测试段净值 > 0）。"""
    for i, cand in enumerate(candidates, 1):
        print(f"  #{i}  train={cand['train_spread'] * 100:+.4f}%  "
              f"valid={cand['valid_spread'] * 100:+.4f}%  "
              f"test={cand['test_spread'] * 100:+.4f}%  "
              f"net={cand['test_net'] * 100:+.4f}%")
        print(f"      {cand['formula_gp']}")
        print(f"      {cand['vnpy_text']}")
    if not candidates:
        print("[成败判据] 无因子通过复检（验证段毛价差均为非正），未达成")
        return
    best: dict = candidates[0]
    positive_months: int = sum(
        1 for m in best["monthly_2026"].values() if m["net"] > 0
    )
    achieved: bool = best["test_net"] > 0
    print(f"[成败判据] 测试段(2026-01-01~09-21)最优因子净尾部价差 = "
          f"{best['test_net'] * 100:+.4f}%（扣往返费 {FEE_ROUNDTRIP * 100:.2f}%），"
          f"{'达成' if achieved else '未达成'}")
    print(f"[逐月稳定] 2026 年净值>0 的月份: {positive_months}/{len(best['monthly_2026'])}")
    print(f"[固定阈值] 测试段(训练段阈值套用)净价差 = "
          f"{best['test_net_fixed_thr'] * 100:+.4f}%")
    print(f"[完成] 总耗时 {elapsed:.0f}s")


def main(seed: int = DEFAULT_SEED, smoke: bool = False) -> None:
    """完整挖掘流程：重采样 -> 特征/标签 -> 进化 -> 复检 -> 输出。"""
    t0: float = time.time()
    print(f"[0] research 根目录: {RESEARCH_ROOT}")
    panel: pl.DataFrame = build_factor_table()
    print(f"[1] 5m 长表: {panel.height} 行 x {panel['symbol'].n_unique()} 个币, "
          f"{panel['datetime'].min()} ~ {panel['datetime'].max()}")
    segments: dict[str, SegmentArrays] = {
        name: segment_arrays(panel, seg) for name, seg in SEGMENTS.items()
    }
    for name, (x_seg, _y, _b) in segments.items():
        print(f"    {name}: {x_seg.shape[0]} 行")
    train_sec, full_sec = measure_fitness_latency(segments, panel)
    print(f"[1.5] 适应度耗时自检: 训练段 {train_sec * 1000:.0f}ms, "
          f"全量 {full_sec * 1000:.0f}ms（上限 {FITNESS_BUDGET_SEC * 1000:.0f}ms）")
    if max(train_sec, full_sec) > FITNESS_BUDGET_SEC:
        raise RuntimeError("适应度求值超出 0.5s 硬约束，必须先优化实现")

    gp, matrices = run_mining(seed, smoke, panel, segments)
    print(f"[3] 名人堂 {len(gp._best_programs)} 个公式, 复检（验证段门槛+去重）...")
    candidates: list[dict] = select_programs(gp, segments, matrices)
    print(f"    入选 {len(candidates)} 个")

    test_panel: pl.DataFrame = panel.filter(segment_filter(TEST))
    enrich_candidates(
        candidates, segments, matrices, test_panel,
        panel.filter(segment_filter(TRAIN))["symbol"].to_numpy(),
        test_panel["symbol"].to_numpy(),
    )
    elapsed: float = time.time() - t0
    print_summary(candidates, elapsed)
    if not smoke:
        result: dict = {
            "meta": {
                "tool": "gplearn",
                "seed": seed,
                "population": POPULATION_SIZE,
                "generations": GENERATIONS,
                "hall_of_fame": HALL_OF_FAME,
                "function_set": FUNCTION_SET,
                "features": {n: FEATURES_5M[n][1] for n in FEATURE_NAMES_5M},
                "label": "close.shift(-1)/close - 1（按币分组，下一根 5m 收益）",
                "fitness": "尾部价差：每币因子值段内分位前/后 20% 标签均值之差，5 币平均",
                "segments": {k: list(v) for k, v in SEGMENTS.items()},
                "fee_roundtrip": FEE_ROUNDTRIP,
                "fitness_latency_sec": {"train": train_sec, "full": full_sec},
                "runtime_sec": round(elapsed, 1),
            },
            "candidates": candidates,
        }
        out_path: Path = write_result(result, seed)
        print(f"[输出] {out_path}")


if __name__ == "__main__":
    arg_parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="5m 尾部价差遗传规划挖矿（--smoke 快速验证管线）"
    )
    arg_parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                            help="随机种子（默认 123）")
    arg_parser.add_argument("--smoke", action="store_true",
                            help="极小种群快速验证管线，不写正式结果")
    cli_args: argparse.Namespace = arg_parser.parse_args()
    main(seed=cli_args.seed, smoke=cli_args.smoke)
