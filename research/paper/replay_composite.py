"""全清单合成信号 + MA30/MA50 择时的事件化模拟盘重放（backtest_replay.py 的扩展）。

把 run_ma_robustness 扫描出的"全清单 24 因子合成 × MA30/MA50"两个 2026 测试段配置，
从向量化权重网格搬到事件化逐日模拟撮合，验证两条路线是否对齐：
    1) 信号：manifest 全部 24 因子 -> compute_factor_table（自动注册算子并补 vwap 列）
       -> composite_signal（当日截面 rank / 按清单 weight 加权合成，与既有研究同口径）；
    2) 择时：BTCUSDT.OKX 收盘 > N 日均线（N=30/50，当日收盘判定，无未来函数）；
    3) 逐日重放：收盘判定择时 -> 算目标 Top5 -> 与昨日比对、仅目标币集变化时生成
       模拟订单，费率 = 单边换手 × 0.1% -> 次日收盘按新价格重估净值；
    4) 同口径向量化参照：同样的信号与择时走 long_top5_returns（每日等权再平衡的
       权重网格模型），输出重放 vs 向量化的年化/Sharpe/最大回撤对照与偏差。

复用 backtest_replay.py 的事件化重放机件（run_replay / execute_rebalance 等），
不修改任何既有文件。

输出：
- research/paper/replay_composite_equity_ma30.csv / _ma50.csv：逐日净值/持仓/订单/费用；
- research/paper/replay_composite_report.md：对照表与结论。

用法（在仓库根目录运行）：
    D:\\veighna_studio\\python.exe -X utf8 research/paper/replay_composite.py
"""

import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# 引导导入：research/ 根目录（研究模块）与同目录 paper/（复用 backtest_replay）
_PAPER_DIR: Path = Path(__file__).resolve().parent
_RESEARCH_DIR: Path = _PAPER_DIR.parent
for _extra_path in (str(_PAPER_DIR), str(_RESEARCH_DIR)):
    if _extra_path not in sys.path:
        sys.path.insert(0, _extra_path)

import pandas as pd
import polars as pl

from backtest_replay import (
    ReplayInputs,
    WARMUP_DAYS,
    build_target_sets,
    mask_test_segment,
    run_replay,
    write_replay_csv,
)
from eval_mined_factors import composite_signal, compute_factor_table
from factor_common import TEST, build_factor_table, load_manifest, safe_join
from run_final_study import (
    FEE_RATE,
    TOP_N,
    long_top5_returns,
    next_day_returns,
    perf_stats,
)
from run_ma_robustness import btc_timing_mask

MA_WINDOWS: list[int] = [30, 50]   # 待重放的均线窗口（对应 ma_robustness 扫描格）
REPORT_REL: str = "paper/replay_composite_report.md"
# run_ma_robustness/report.md 的 2026 测试段"全清单24因子"年化参照（向量化口径）
ROBUSTNESS_REFERENCE: dict[int, float] = {30: 0.2227, 50: 0.2087}
DEVIATION_TOLERANCE: float = 0.03  # 重放与向量化年化偏差超过 3 个百分点必须查明原因


@dataclass
class WindowOutcome:
    """单均线窗口配置的结果汇总：重放统计 + 向量化参照 + 双向偏差与调仓计数。"""

    window: int
    replay_stats: dict[str, float]
    vector_stats: dict[str, float]
    rebalance_total: int
    rebalance_in_test: int
    hold_ratio: float
    replay_vs_vector_pp: float
    vector_vs_reference_pp: float


def prepare_shared_inputs() -> tuple[pl.DataFrame, pl.DataFrame, pd.Series, pd.DataFrame]:
    """装载面板、计算全清单合成信号与次日收益宽表（两个均线窗口共用一份）。"""
    factors: list[dict] = load_manifest().get("factors", [])
    names: list[str] = [item["name"] for item in factors]
    print(f"[1] 清单因子共 {len(factors)} 个: {names}")
    panel: pl.DataFrame = build_factor_table()
    table: pl.DataFrame = compute_factor_table(panel, factors)
    composite: pd.Series = composite_signal(table, factors)
    next_wide: pd.DataFrame = next_day_returns(panel)
    print(f"[2] 面板就绪: {panel.height} 行，末端 {panel['datetime'].max()}，"
          f"合成信号 {composite.notna().sum()} 行")
    return panel, table, composite, next_wide


def build_window_inputs(panel: pl.DataFrame, table: pl.DataFrame,
                        composite: pd.Series, window: int) -> ReplayInputs:
    """单窗口重放输入：参数化择时序列 + 每日 Top5 目标集合 + 收盘价宽表。"""
    timing: pd.Series = btc_timing_mask(panel, window)
    closes_wide: pd.DataFrame = panel.select(
        ["datetime", "vt_symbol", "close"]
    ).to_pandas().pivot(index="datetime", columns="vt_symbol", values="close")
    return ReplayInputs(panel, composite, timing,
                        build_target_sets(table, composite), closes_wide)


def vectorized_reference(composite: pd.Series, panel: pl.DataFrame,
                         next_wide: pd.DataFrame,
                         timing: pd.Series) -> dict[str, float]:
    """同口径向量化参照：long_top5_returns 权重网格，切片到 2026 测试段统计。"""
    work_pdf: pd.DataFrame = panel.select(["datetime", "vt_symbol"]).to_pandas()
    source: dict[str, float] = long_top5_returns(composite, work_pdf, next_wide, timing)
    net: pd.Series = source["net_series"]
    return perf_stats(net[mask_test_segment(net.index)])


def test_segment_flag(dates: pd.Series) -> pd.Series:
    """布尔掩码：日期是否落在 2026 测试段（含端点，与 mask_test_segment 同口径）。"""
    upper: pd.Timestamp = pd.Timestamp(TEST[1]) + pd.Timedelta(hours=23, minutes=59)
    parsed: pd.Series = pd.to_datetime(dates)
    return (parsed >= pd.Timestamp(TEST[0])) & (parsed <= upper)


def replay_extras(daily: pd.DataFrame) -> tuple[int, int, float]:
    """重放统计补充：(全程调仓次数, 测试段调仓次数, 测试段持仓天数占比)。"""
    flags: pd.Series = daily["rebalanced"].astype(bool)
    in_test: pd.Series = test_segment_flag(daily["date"])
    hold_ratio: float = float(daily.loc[in_test, "hold_timing"].astype(bool).mean())
    return int(flags.sum()), int((flags & in_test).sum()), hold_ratio


def run_window(panel: pl.DataFrame, table: pl.DataFrame, composite: pd.Series,
               next_wide: pd.DataFrame, window: int,
               warmup_start: pd.Timestamp) -> WindowOutcome:
    """跑单个 MA 窗口的完整对照：事件化重放 + 向量化参照，并写逐日 CSV。"""
    inputs: ReplayInputs = build_window_inputs(panel, table, composite, window)
    result = run_replay(inputs, warmup_start, True)  # 主口径：仅目标币集变化时调仓
    net_series: pd.Series = result.net_series
    replay_stats: dict[str, float] = perf_stats(net_series[mask_test_segment(net_series.index)])
    vector_stats: dict[str, float] = vectorized_reference(
        composite, panel, next_wide, inputs.timing
    )
    rebalance_total, rebalance_in_test, hold_ratio = replay_extras(result.daily)
    write_replay_csv(result.daily, f"paper/replay_composite_equity_ma{window}.csv")
    replay_vs_vector_pp: float = round(
        (replay_stats["年化收益"] - vector_stats["年化收益"]) * 100, 2
    )
    vector_vs_reference_pp: float = round(
        (vector_stats["年化收益"] - ROBUSTNESS_REFERENCE[window]) * 100, 2
    )
    print(f"[3] MA{window} 重放: 年化 {replay_stats['年化收益']:+.2%} / "
          f"Sharpe {replay_stats['Sharpe']:+.3f} / 回撤 {replay_stats['最大回撤']:+.2%}；"
          f"向量化 {vector_stats['年化收益']:+.2%}，偏差 {replay_vs_vector_pp:+.2f}pp；"
          f"测试段调仓 {rebalance_in_test} 次")
    return WindowOutcome(window, replay_stats, vector_stats, rebalance_total,
                         rebalance_in_test, hold_ratio, replay_vs_vector_pp,
                         vector_vs_reference_pp)


def format_row(outcome: WindowOutcome) -> str:
    """把单个窗口结果格式化成对照表的一行（markdown 表格）。"""
    replay: dict[str, float] = outcome.replay_stats
    vector: dict[str, float] = outcome.vector_stats
    reference: float = ROBUSTNESS_REFERENCE[outcome.window]
    return ("| "
            f"MA{outcome.window} | {replay['年化收益']:+.2%} | {vector['年化收益']:+.2%} | "
            f"{outcome.replay_vs_vector_pp:+.2f} | {reference:+.2%} | "
            f"{outcome.vector_vs_reference_pp:+.2f} | {replay['Sharpe']:+.3f} | "
            f"{vector['Sharpe']:+.3f} | {replay['最大回撤']:+.2%} | "
            f"{vector['最大回撤']:+.2%} | {outcome.rebalance_in_test}/"
            f"{outcome.rebalance_total} | {outcome.hold_ratio:.2f} |")


def build_conclusion(outcomes: list[WindowOutcome]) -> list[str]:
    """汇总结论：重放 vs 向量化对齐判定 + 与 run_ma_robustness 参照的偏差归因。"""
    worst_pp: float = max(abs(item.replay_vs_vector_pp) for item in outcomes)
    aligned: bool = worst_pp <= DEVIATION_TOLERANCE * 100
    lines: list[str] = [
        f"- 重放 vs 向量化年化最大偏差 {worst_pp:+.2f}pp，"
        + ("在 3pp 容忍线内，事件化逐日重放与向量化回测对齐" if aligned
           else "超过容忍线，需对照 replay CSV 的 orders/fee 列查明差异来源")
    ]
    for item in outcomes:
        lines.append(
            f"- MA{item.window}: 重放年化 {item.replay_stats['年化收益']:+.2%} vs "
            f"向量化 {item.vector_stats['年化收益']:+.2%}（偏差 "
            f"{item.replay_vs_vector_pp:+.2f}pp），测试段调仓 {item.rebalance_in_test} 次"
            f"（全程 {item.rebalance_total} 次），测试段持仓占比 {item.hold_ratio:.2f}"
        )
    exact_match: bool = all(
        abs(item.vector_vs_reference_pp) < 0.005 for item in outcomes
    )
    attribution: str = (
        "向量化参照与重放走同一代码路径（复用 long_top5_returns / perf_stats），"
        "参照偏差为 0，数字完全复现" if exact_match else
        "自报告生成后数据已更新或统计切片略有差异，见表格'向量化-参照'列"
    )
    lines.append(f"- 与 run_ma_robustness 参照数字的偏差归因：{attribution}")
    return lines


def write_report(outcomes: list[WindowOutcome], panel_last: str) -> Path:
    """写出 markdown 对照报告（Path.write_text，路径经 safe_join 校验）。"""
    generated_at: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report: str = "\n".join([
        "# 全清单合成信号 × MA30/MA50 事件化重放对照报告",
        "",
        f"- 生成时间: {generated_at}；因子面板可用最后一天: {panel_last}"
        "（标签需未来 6 日数据，尾部若干天被裁掉，与原研究口径一致）",
        "- 信号: manifest 全部 24 因子合成（当日截面 rank、按清单 weight 加权），目标 Top5 等权",
        "- 择时: BTCUSDT.OKX 收盘 > N 日均线（当日收盘判定，无未来函数）；"
        "重放为'仅目标币集变化时调仓'，向量化参照为每日等权再平衡（权重网格）",
        "- 费用: 0.1%/单边换手，两口径一致；重放预热起点为测试段前 45 天",
        "- run_ma_robustness 报告（2026 测试段，全清单24因子）参照："
        "MA30 年化 +22.27%/Sharpe +0.775/回撤 -18.24%；MA50 年化 +20.87%/Sharpe +0.745/回撤 -22.33%",
        "",
        "| 配置 | 重放年化 | 向量化年化 | 重放-向量化(pp) | 稳健性参照 | 向量化-参照(pp) "
        "| 重放Sharpe | 向量化Sharpe | 重放回撤 | 向量化回撤 | 调仓次数(测试段/全程) | 持仓占比 |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *[format_row(item) for item in outcomes],
        "",
        "## 结论",
        "",
        *build_conclusion(outcomes),
        "",
    ])
    output: Path = safe_join(REPORT_REL)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    return output


def main() -> None:
    """入口：共用输入 -> 双窗口各跑（重放 + 向量化参照）-> 写 CSV 与对照报告。"""
    start_ms: float = time.time()
    panel, table, composite, next_wide = prepare_shared_inputs()
    warmup_start: pd.Timestamp = pd.Timestamp(TEST[0]) - pd.Timedelta(days=WARMUP_DAYS)
    print(f"[2] 重放预热起点 {warmup_start.date()}，费率 {FEE_RATE:.1%}，目标 Top{TOP_N}")
    outcomes: list[WindowOutcome] = [
        run_window(panel, table, composite, next_wide, window, warmup_start)
        for window in MA_WINDOWS
    ]
    output: Path = write_report(outcomes, str(panel["datetime"].max()))
    worst_pp: float = max(abs(item.replay_vs_vector_pp) for item in outcomes)
    verdict: str = "对齐" if worst_pp <= DEVIATION_TOLERANCE * 100 else "未对齐（需排查）"
    print(f"[4] 对照报告已写入 {output}；重放 vs 向量化最大偏差 {worst_pp:+.2f}pp -> {verdict}；"
          f"耗时 {time.time() - start_ms:.0f}s")


if __name__ == "__main__":
    main()
