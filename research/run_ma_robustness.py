"""BTC 均线窗口稳健性扫描：验证"+14.4% 年化"是否过拟合 MA20 单一参数。

背景：run_final_study 的结论是"gp_s123 单因子 + BTC>MA20 择时 Top5 轮动"
在 2026 测试段年化 +14.4% / Sharpe 0.538。本脚本回答两个问题：
1) 结果对均线窗口敏感吗？（扫描 10/20/30/50 日）
2) 在 2025 验证段也成立吗？（跨时段复检）

扫描矩阵：均线窗口 × 时段 × 信号（gp_s123 单因子 / 全清单 24 因子合成），
参照行含"不择时纯多头 Top5"与"买入持有 BTC"两个基准。
每格输出：年化收益、年化波动、Sharpe、最大回撤、持仓天数占比
（用于识别全程持仓 / 全程空仓的退化配置）。

方法与费用口径完全复用 run_final_study（换手 x 0.1%），不重写回测逻辑。

输出：research/ma_robustness/report.md + stdout 摘要。
用法（在仓库根目录运行）：
    D:\\veighna_studio\\python.exe -X utf8 research/run_ma_robustness.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from factor_common import TEST, VALID, build_factor_table, load_manifest
from eval_mined_factors import composite_signal, compute_factor_table
from run_final_study import long_top5_returns, next_day_returns, perf_stats

REPORT_PATH: Path = Path("research/ma_robustness/report.md")
BTC_SYMBOL: str = "BTCUSDT.OKX"
MA_WINDOWS: list[int] = [10, 20, 30, 50]
SEGMENTS: dict[str, tuple[str, str]] = {"2025验证段": VALID, "2026测试段": TEST}
GP_PREFIX: str = "gp_s123"
SIGNAL_NAMES: list[str] = ["单因子gp_s123", "全清单24因子"]
# 退化判定阈值：时段内持仓占比接近 0（全程空仓）或 1（全程持仓）视为退化
DEGENERATE_LOW: float = 0.05
DEGENERATE_HIGH: float = 0.95


def btc_timing_mask(panel: pl.DataFrame, window: int) -> pd.Series:
    """BTC 收盘 > N 日均线的择时序列（参数化版，替代写死的 MA20）。

    当日收盘判定、次日生效口径与原研究一致，无未来函数。
    """
    btc: pl.DataFrame = panel.filter(pl.col("vt_symbol") == BTC_SYMBOL).sort("datetime")
    btc = btc.with_columns(pl.col("close").rolling_mean(window).alias("ma"))
    btc = btc.with_columns(
        (pl.col("close") > pl.col("ma")).fill_null(False).alias("hold")
    )
    series: pd.Series = btc.select(["datetime", "hold"]).to_pandas().set_index(
        "datetime"
    )["hold"]
    return series


def segment_date_mask(dates: pd.DatetimeIndex,
                      segment: tuple[str, str]) -> np.ndarray:
    """布尔数组：宽表日期是否落在指定时段（含端点，与原研究切片口径一致）。"""
    return np.asarray((dates >= segment[0]) & (dates <= segment[1] + " 23:59:59"))


def holding_ratio(timing: pd.Series | None, seg_dates: pd.DatetimeIndex) -> float:
    """持仓天数占比：择时序列在时段内为真的比例；纯多头恒为 1。"""
    if timing is None:
        return 1.0
    aligned: pd.Series = timing.reindex(seg_dates).fillna(False)
    return float(aligned.astype(bool).mean())


def run_cell(composite: pd.Series, work_pdf: pd.DataFrame,
             next_wide: pd.DataFrame, timing: pd.Series | None,
             seg_mask: np.ndarray) -> dict[str, float]:
    """跑一个矩阵格：组合收益切片到时段 -> 绩效指标 + 持仓占比。"""
    source: dict[str, float] = long_top5_returns(
        composite, work_pdf, next_wide, timing
    )
    seg_series: pd.Series = source["net_series"][seg_mask]
    seg_dates: pd.DatetimeIndex = source["net_series"].index[seg_mask]
    stats: dict[str, float] = perf_stats(seg_series)
    return {**stats, "持仓占比": round(holding_ratio(timing, seg_dates), 2)}


def btc_buy_hold_row(next_wide: pd.DataFrame,
                     seg_mask: np.ndarray) -> dict[str, float]:
    """基准：买入持有 BTC（次日收益口径与组合同源，一次性买入不计费）。"""
    stats: dict[str, float] = perf_stats(next_wide[BTC_SYMBOL][seg_mask].dropna())
    return {**stats, "持仓占比": 1.0}


def build_signals(table: pl.DataFrame, factors: list[dict]) -> dict[str, pd.Series]:
    """两种合成信号：gp_s123 单因子、全清单 24 因子截面 rank 加权合成。"""
    gp_single: list[dict] = [f for f in factors if f["name"].startswith(GP_PREFIX)]
    return {
        SIGNAL_NAMES[0]: composite_signal(table, gp_single),
        SIGNAL_NAMES[1]: composite_signal(table, factors),
    }


def collect_matrix(signals: dict[str, pd.Series], work_pdf: pd.DataFrame,
                   next_wide: pd.DataFrame,
                   timings: dict[int, pd.Series]) -> pd.DataFrame:
    """组装完整扫描矩阵：基准行 + {信号 x 窗口} 择时行，按时段展开。"""
    rows: list[dict] = []
    dates: pd.DatetimeIndex = next_wide.index
    for seg_name, segment in SEGMENTS.items():
        seg_mask: np.ndarray = segment_date_mask(dates, segment)
        rows.append({"时段": seg_name, "信号": "基准_买入持有BTC", "窗口": "-",
                     **btc_buy_hold_row(next_wide, seg_mask)})
        for sig_name in SIGNAL_NAMES:
            rows.append({"时段": seg_name, "信号": f"基准_纯多头_{sig_name}",
                         "窗口": "-",
                         **run_cell(signals[sig_name], work_pdf, next_wide,
                                    None, seg_mask)})
        for sig_name in SIGNAL_NAMES:
            for window in MA_WINDOWS:
                rows.append({"时段": seg_name, "信号": sig_name,
                             "窗口": f"MA{window}",
                             **run_cell(signals[sig_name], work_pdf, next_wide,
                                        timings[window], seg_mask)})
    return pd.DataFrame(rows)


def _segment_summary(seg: pd.DataFrame) -> list[str]:
    """单个时段的择时配置小结：为正个数、Sharpe 范围、最优与退化配置。"""
    pos_count: int = int((seg["年化收益"] > 0).sum())
    total: int = len(seg)
    best: pd.Series = seg.loc[seg["Sharpe"].idxmax()]
    # 注意 apply 在空表上可能返回 Series，显式转 list 保证后续真值判断安全
    degenerate: list[str] = list(seg.loc[
        (seg["持仓占比"] <= DEGENERATE_LOW) | (seg["持仓占比"] >= DEGENERATE_HIGH)
    ].apply(lambda r: f"{r['信号']}|{r['窗口']}(持仓占比{r['持仓占比']})", axis=1))
    lines: list[str] = [
        f"- 择时配置为正比例：{pos_count}/{total}；"
        f"Sharpe 范围 [{seg['Sharpe'].min():+.3f}, {seg['Sharpe'].max():+.3f}]",
        f"- 该段最优配置：{best['信号']}|{best['窗口']}，"
        f"年化 {best['年化收益']:+.1%}，Sharpe {best['Sharpe']:+.3f}",
    ]
    if degenerate:
        lines.append(f"- 退化配置（接近全程持仓/空仓）：{'、'.join(degenerate)}")
    else:
        lines.append("- 无退化配置")
    return lines


def summarize(matrix: pd.DataFrame) -> list[str]:
    """汇总结论文本：分段小结 + 跨段对比 + 对"+14.4% 稳健性"的判定。"""
    timed: pd.DataFrame = matrix[matrix["窗口"] != "-"]
    lines: list[str] = []
    for seg_name in SEGMENTS:
        seg: pd.DataFrame = timed[timed["时段"] == seg_name]
        lines.append(f"### {seg_name}")
        lines.extend(_segment_summary(seg))
        gp: pd.DataFrame = seg[seg["信号"] == SIGNAL_NAMES[0]].set_index("窗口")
        lines.append(
            f"- gp_s123 各窗口年化："
            + "，".join(f"{w}:{gp.loc[w, '年化收益']:+.1%}" for w in gp.index)
        )
        lines.append("")
    # 跨段稳健性判定：单因子 gp_s123 在 4 个窗口 x 2 个时段中的表现为正数
    gp_all: pd.DataFrame = timed[timed["信号"] == SIGNAL_NAMES[0]]
    gp_pos: int = int((gp_all["年化收益"] > 0).sum())
    best_overall: pd.Series = timed.loc[timed["Sharpe"].idxmax()]
    lines.append("### 总体判定")
    lines.append(
        f"- gp_s123 择时在 4 窗口 x 2 时段共 {len(gp_all)} 格中 "
        f"{gp_pos} 格为正"
        + ("（稳健）" if gp_pos == len(gp_all)
           else "（存在反例，稳健性存疑）" if gp_pos < len(gp_all) / 2
           else "（多数为正，但依赖窗口/时段）")
    )
    lines.append(
        f"- 全部 16 个择时格 Sharpe 范围 "
        f"[{timed['Sharpe'].min():+.3f}, {timed['Sharpe'].max():+.3f}]，"
        f"跨格最优：{best_overall['信号']}|{best_overall['窗口']}"
        f"|{best_overall['时段']}（Sharpe {best_overall['Sharpe']:+.3f}）"
    )
    return lines


def write_report(matrix: pd.DataFrame, conclusion: list[str],
                 panel_last_date: str) -> None:
    """写出 markdown 报告（口径说明 + 完整矩阵 + 结论）。"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sections: list[str] = [
        "# BTC 均线窗口稳健性扫描（含费，Top5 等权日调）",
        "",
        f"- 因子面板可用最后一天：{panel_last_date}"
        "（标签需未来 6 日数据，尾部若干天被裁掉，与原研究口径一致）",
        "- 择时规则：BTC 收盘 > N 日均线则持仓 Top5，否则空仓；费率 0.1%/换手",
        "- 持仓占比：时段内择时为真的天数比例，≈0 全程空仓、≈1 全程持仓（退化）",
        "",
        matrix.to_markdown(index=False),
        "",
        "## 结论",
        "",
        *conclusion,
        "",
    ]
    REPORT_PATH.write_text("\n".join(sections), encoding="utf-8")


def main() -> None:
    """入口：加载数据 -> 算因子与信号 -> 扫描矩阵 -> 写报告 + stdout 摘要。"""
    t0: float = time.time()
    factors: list[dict] = load_manifest().get("factors", [])
    print(f"[1] 清单因子 {len(factors)} 个")
    panel: pl.DataFrame = build_factor_table()
    table: pl.DataFrame = compute_factor_table(panel, factors)
    work_pdf: pd.DataFrame = table.select(["datetime", "vt_symbol"]).to_pandas()
    next_wide: pd.DataFrame = next_day_returns(panel)
    timings: dict[int, pd.Series] = {
        window: btc_timing_mask(panel, window) for window in MA_WINDOWS
    }
    signals: dict[str, pd.Series] = build_signals(table, factors)
    print(f"[2] 因子表与择时序列就绪，面板末端 {panel['datetime'].max()}")

    matrix: pd.DataFrame = collect_matrix(signals, work_pdf, next_wide, timings)
    conclusion: list[str] = summarize(matrix)
    write_report(matrix, conclusion, str(panel["datetime"].max()))
    print("\n=============== 稳健性扫描矩阵（含费）===============")
    print(matrix.to_string(index=False))
    print("\n=============== 结论 ===============")
    print("\n".join(conclusion))
    print(f"\n报告已写入 {REPORT_PATH}，耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
