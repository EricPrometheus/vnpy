"""多空与择时组合构造研究：挖掘因子在 2026 年测试段的三种构造对比。

回答的问题：纯多头 Top5 轮动 2026 年亏损（市场整体大跌），
换成"多空对冲"或"BTC 均线择时"能否做出正收益？

方法论（避免常见回测偏差）：
- 信号：当日截面合成因子（复用 eval_mined_factors.composite_signal，
  rank/count 加权，清单 weight 为负即反向），只用当日收盘时点信息；
- 收益：次日单日前向收益 close.shift(-1)/close - 1（按 vt_symbol 分组），
  即 t 日定仓、承担 t -> t+1 一天的涨跌；不用 5 日重叠标签；
- 费用：日换手 = sum(|w_t - w_{t-1}|) / 2，费率 0.1%；多空两腿不共享币种，
  该统一公式天然等价于"双腿分别计费"；
- 三种构造：(a) 纯多头 Top5 等权；(b) 多头前 5 + 空头后 5 等权；
  (c) 仅当 BTCUSDT.OKX 收盘 > 其 20 日均线时持有 Top5，否则空仓。

用法（仓库根目录）：
    D:\\veighna_studio\\python.exe -X utf8 research/run_ls_timing_study.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

# 复用 research 共用层（build_factor_table / compute_factor_table / composite_signal）
sys.path.insert(0, "research")

from eval_mined_factors import composite_signal, compute_factor_table  # noqa: E402
from factor_common import TEST, build_factor_table, segment_mask  # noqa: E402

# ==================== 常量集中区 ====================
SNAPSHOT_PATH: str = "research/ls_timing_study/manifest_snapshot.json"  # 清单快照（防并行改写竞态）
RESULT_PATH: str = "research/ls_timing_study/result.md"  # 结果报告输出路径
BTC_SYMBOL: str = "BTCUSDT.OKX"  # 择时基准币种
MA_WINDOW: int = 20  # 择时均线窗口（日）
TOP_N: int = 5  # 多头 / 空头腿各持币数
FEE_RATE: float = 0.001  # 单边费率 0.1%（费用 = 换手 × 0.1%）
DAYS_PER_YEAR: int = 365  # 加密货币全年无休，按 365 天年化
SIGNAL_COLS: dict[str, str] = {"comp_all": "全部14因子", "comp_gp": "仅gp因子"}
SCHEME_NAMES: dict[str, str] = {
    "long_only": "a) 纯多头Top5",
    "long_short": "b) 多空(多5+空5)",
    "timed_long": "c) BTC择时多头",
}


def load_snapshot_factors() -> tuple[list[dict], list[dict]]:
    """读取清单快照，返回 (全部因子, 仅 gp 因子)；快照缺失直接抛错。"""
    snapshot_file: Path = Path(SNAPSHOT_PATH)
    if not snapshot_file.is_file():
        raise FileNotFoundError(f"清单快照不存在: {snapshot_file}，请先复制 mined_manifest.json")
    manifest: dict = json.loads(snapshot_file.read_text(encoding="utf-8"))
    all_factors: list[dict] = list(manifest.get("factors", []))
    gp_factors: list[dict] = [f for f in all_factors if f.get("source") == "gp"]
    if not all_factors or not gp_factors:
        raise ValueError(
            f"清单快照因子不足: 全部 {len(all_factors)} 个 / gp {len(gp_factors)} 个"
        )
    print(f"[快照] 共 {len(all_factors)} 个因子（gp {len(gp_factors)} 个）")
    return all_factors, gp_factors


def attach_composites(
    table: pl.DataFrame, all_factors: list[dict], gp_factors: list[dict]
) -> pl.DataFrame:
    """按清单权重合成两套信号（当日截面 rank/count），负权重即反向。"""
    comp_all: pd.Series = composite_signal(table, all_factors)
    comp_gp: pd.Series = composite_signal(table, gp_factors)
    return table.with_columns(
        pl.Series("comp_all", comp_all.to_numpy()),
        pl.Series("comp_gp", comp_gp.to_numpy()),
    )


def attach_price_columns(table: pl.DataFrame, panel: pl.DataFrame) -> pl.DataFrame:
    """补单日前向收益（按币种分组）与 BTC 站上 20 日均线的择时标记。

    都在完整面板上计算再按键连接，保证均线/前向收益不因测试段截断而失真。
    """
    with_fwd: pl.DataFrame = panel.with_columns(
        (pl.col("close").shift(-1) / pl.col("close") - 1)
        .over("vt_symbol")
        .alias("fwd_ret")
    )
    btc_flag: pl.DataFrame = (
        panel.filter(pl.col("vt_symbol") == BTC_SYMBOL)
        .with_columns(
            (pl.col("close") > pl.col("close").rolling_mean(MA_WINDOW).over("vt_symbol"))
            .alias("btc_ok")
        )
        .select(["datetime", "btc_ok"])
    )
    return (
        table.join(
            with_fwd.select(["datetime", "vt_symbol", "fwd_ret"]),
            on=["datetime", "vt_symbol"],
            how="left",
        )
        .join(btc_flag, on="datetime", how="left")
        .with_columns(pl.col("btc_ok").fill_null(False))
    )


def make_study_table(all_factors: list[dict], gp_factors: list[dict]) -> pl.DataFrame:
    """面板 -> 因子列 -> 两套合成信号 -> 前向收益与择时标记 -> 截取测试段。"""
    panel: pl.DataFrame = build_factor_table()
    table: pl.DataFrame = compute_factor_table(panel, all_factors)
    table = attach_composites(table, all_factors, gp_factors)
    table = attach_price_columns(table, panel)
    study: pl.DataFrame = table.filter(segment_mask(TEST)).sort(["datetime", "vt_symbol"])
    if study.is_empty():
        raise ValueError("测试段过滤后为空，请检查数据日期范围")
    return study


def study_to_pandas(study: pl.DataFrame) -> pd.DataFrame:
    """polars -> pandas，并把 datetime 归一为交易日（去掉 16:00 时点）。"""
    pdf: pd.DataFrame = study.to_pandas()
    pdf["trade_date"] = pd.to_datetime(pdf["datetime"]).dt.normalize()
    return pdf


def to_wide(study_pdf: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """长表转宽表：行 = 交易日，列 = 币种。"""
    return study_pdf.set_index(["trade_date", "vt_symbol"])[value_col].unstack("vt_symbol")


def build_wide_frames(study_pdf: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    """生成两套信号宽表 + 前向收益宽表 + BTC 择时日度标记。

    最后一天没有次日收益（前向收益全 NaN），从评估窗口中剔除。
    """
    fwd: pd.DataFrame = to_wide(study_pdf, "fwd_ret")
    keep: pd.Series = fwd.notna().any(axis=1)
    wide: dict[str, pd.DataFrame] = {
        "comp_all": to_wide(study_pdf, "comp_all").loc[keep],
        "comp_gp": to_wide(study_pdf, "comp_gp").loc[keep],
        "fwd": fwd.loc[keep],
    }
    btc_flags: pd.Series = (
        study_pdf.drop_duplicates("trade_date")
        .set_index("trade_date")["btc_ok"]
        .sort_index()
        .reindex(wide["fwd"].index)
        .fillna(False)
    )
    return wide, btc_flags


def leg_weights(signal_row: pd.Series, sign: float) -> pd.Series:
    """单腿等权仓位：sign=+1 取因子值最高的 TOP_N 个，sign=-1 取最低的 TOP_N 个。

    信号为 NaN 的币当日成立性不足，直接剔除（不计入 TOP_N）。
    """
    valid: pd.Series = signal_row.dropna()
    picked = valid.sort_values(ascending=(sign < 0)).head(TOP_N).index
    weights: pd.Series = pd.Series(0.0, index=signal_row.index)
    if len(picked) > 0:
        weights.loc[picked] = sign / len(picked)
    return weights


def scheme_weights(
    scheme: str, signal_wide: pd.DataFrame, btc_flags: pd.Series
) -> pd.DataFrame:
    """按构造方式把信号矩阵映射成仓位矩阵（行 = 日，列 = 币，含多空符号）。"""
    if scheme == "long_only":
        return signal_wide.apply(lambda row: leg_weights(row, 1.0), axis=1)
    if scheme == "long_short":
        return signal_wide.apply(
            lambda row: leg_weights(row, 1.0) + leg_weights(row, -1.0), axis=1
        )
    if scheme == "timed_long":
        rows: list[pd.Series] = []
        for trade_date, row in signal_wide.iterrows():
            if bool(btc_flags.get(trade_date, False)):
                rows.append(leg_weights(row, 1.0))
            else:
                rows.append(pd.Series(0.0, index=row.index))
        return pd.DataFrame(rows, index=signal_wide.index, columns=signal_wide.columns)
    raise ValueError(f"未知构造方式: {scheme}")


def daily_turnover(weights: pd.DataFrame) -> pd.Series:
    """日换手 = sum(|w_t - w_{t-1}|) / 2（首日相对空仓计）。"""
    previous: pd.DataFrame = weights.shift(1).fillna(0.0)
    return (weights - previous).abs().sum(axis=1) / 2.0


def perf_stats(daily_returns: pd.Series) -> dict[str, float]:
    """日收益序列 -> 年化收益 / 年化波动 / Sharpe / 最大回撤（按 365 天年化）。"""
    returns: pd.Series = daily_returns.dropna()
    if len(returns) < 2:
        raise ValueError("收益序列太短，无法计算绩效指标")
    ann_return: float = float(returns.mean() * DAYS_PER_YEAR)
    ann_vol: float = float(returns.std(ddof=1) * np.sqrt(DAYS_PER_YEAR))
    sharpe: float = ann_return / ann_vol if ann_vol > 0 else float("nan")
    equity: pd.Series = (1.0 + returns).cumprod()
    max_drawdown: float = float((1.0 - equity / equity.cummax()).max())
    return {"ann_ret": ann_return, "ann_vol": ann_vol, "sharpe": sharpe, "max_dd": max_drawdown}


def evaluate_cell(
    scheme: str, signal_col: str, wide: dict[str, pd.DataFrame], btc_flags: pd.Series
) -> dict:
    """单个 (构造 × 信号) 组合的评估：含费 / 不含费两套指标 + 换手与敞口。"""
    weights: pd.DataFrame = scheme_weights(scheme, wide[signal_col], btc_flags)
    returns_gross: pd.Series = (weights * wide["fwd"]).sum(axis=1)
    turnover: pd.Series = daily_turnover(weights)
    returns_net: pd.Series = returns_gross - turnover * FEE_RATE
    return {
        "scheme": scheme,
        "signal": signal_col,
        "gross": perf_stats(returns_gross),
        "net": perf_stats(returns_net),
        "turnover": float(turnover.mean()),
        "exposure": float((weights.abs().sum(axis=1) > 1e-12).mean()),
    }


def run_all() -> tuple[list[dict], pd.Series, dict[str, float]]:
    """主流程：快照 -> 面板 -> 信号 -> 逐格评估，返回结果行、择时标记与诊断。"""
    all_factors, gp_factors = load_snapshot_factors()
    study_pdf: pd.DataFrame = study_to_pandas(make_study_table(all_factors, gp_factors))
    wide, btc_flags = build_wide_frames(study_pdf)
    fwd_index: pd.Index = wide["fwd"].index
    min_coins: int = int(wide["fwd"].notna().sum(axis=1).min())
    print(
        f"[窗口] {fwd_index.min().date()} ~ {fwd_index.max().date()} "
        f"共 {len(fwd_index)} 个交易日 / {wide['fwd'].shape[1]} 个币 / "
        f"每日最少 {min_coins} 个币有信号"
    )
    diagnostics: dict[str, float] = signal_diagnostics(study_pdf)
    print(
        f"[诊断] 合成信号 1 日 IC = {diagnostics['ic1d']:+.4f}，"
        f"Top5-Bottom5 尾部价差 = {diagnostics['spread1d'] * 100:+.4f}%/日"
        f"（年化约 {diagnostics['spread1d'] * DAYS_PER_YEAR * 100:+.1f}%）"
    )
    results: list[dict] = []
    for signal_col, signal_label in SIGNAL_COLS.items():
        for scheme in SCHEME_NAMES:
            row: dict = evaluate_cell(scheme, signal_col, wide, btc_flags)
            row["signal_label"] = signal_label
            row["scheme_label"] = SCHEME_NAMES[scheme]
            results.append(row)
    return results, btc_flags, diagnostics


def _day_ic(valid: pd.DataFrame) -> float:
    """单日截面 Rank IC（合成信号 vs 1 日前向收益）。"""
    return float(valid["comp_all"].rank().corr(valid["fwd_ret"].rank()))


def _day_tail_spread(valid: pd.DataFrame) -> float:
    """单日 Top5 - Bottom5 的 1 日前向收益价差（多空毛价差）。"""
    ranked: pd.DataFrame = valid.sort_values("comp_all", ascending=False)
    top: float = float(ranked.head(TOP_N)["fwd_ret"].mean())
    bottom: float = float(ranked.tail(TOP_N)["fwd_ret"].mean())
    return top - bottom


def signal_diagnostics(study_pdf: pd.DataFrame) -> dict[str, float]:
    """诊断信号为何失灵：1 日 IC 与尾部价差（全部14因子口径，按日取均值）。"""
    value_cols: list[str] = ["comp_all", "fwd_ret"]
    grouped = study_pdf.dropna(subset=value_cols).groupby("trade_date")[value_cols]
    return {"ic1d": float(grouped.apply(_day_ic).mean()),
            "spread1d": float(grouped.apply(_day_tail_spread).mean())}


def build_result_rows(results: list[dict]) -> list[str]:
    """渲染 6 格 × 含费/不含费的主对比表（markdown 行）。"""
    lines: list[str] = [
        "| 信号 | 构造 | 费用 | 年化收益 | 年化波动 | Sharpe | 最大回撤 | 日均换手 | 持仓敞口 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in results:
        for fee_tag, stats in (("不含费", row["gross"]), ("含费", row["net"])):
            lines.append(
                f"| {row['signal_label']} | {row['scheme_label']} | {fee_tag} "
                f"| {stats['ann_ret'] * 100:+.1f}% | {stats['ann_vol'] * 100:.1f}% "
                f"| {stats['sharpe']:+.2f} | {stats['max_dd'] * 100:.1f}% "
                f"| {row['turnover'] * 100:.1f}% | {row['exposure'] * 100:.0f}% |"
            )
    return lines


def pick_best_net(results: list[dict]) -> dict:
    """按含费 Sharpe 选出最优组合。"""
    return max(results, key=lambda row: row["net"]["sharpe"])


def render_report(
    results: list[dict], btc_flags: pd.Series, diagnostics: dict[str, float]
) -> str:
    """生成 result.md 全文：方法说明 + 主对比表 + 诊断与结论 + 口径附注。"""
    best: dict = pick_best_net(results)
    hedge_rows: list[dict] = [r for r in results if r["scheme"] != "long_only"]
    best_hedge: dict = max(hedge_rows, key=lambda row: row["net"]["sharpe"])
    lines: list[str] = [
        "# 多空与择时组合构造研究（测试段 2026-01-01 ~ 2026-09-21）",
        "",
        "## 背景与方法",
        "",
        "- 因子：清单快照（14 个 = gp_001~004 + ag_001~010），"
        "合成信号为当日截面 rank/n 按 weight 加权求和（负权重即反向），"
        "分\"全部14因子\"与\"仅gp因子\"两套；",
        "- 收益：t 日收盘定仓，PnL 用次日单日前向收益 close.shift(-1)/close - 1"
        "（按 vt_symbol 分组），不含 5 日重叠标签；",
        "- 费用：日换手 = sum(|w_t - w_{t-1}|)/2，费率 0.1%；多空两腿不共享币种，"
        "统一公式等价于双腿分别计费；",
        "- 构造：a) 纯多头 Top5 等权；b) 多头前5 + 空头后5 等权；"
        "c) BTCUSDT.OKX 收盘 > 20 日均线时持有 Top5，否则空仓；",
        f"- 年化按 {DAYS_PER_YEAR} 天（加密货币全年无休）；多空为名义敞口口径"
        "（多头腿 +1、空头腿 -1，总敞口 2 倍）。",
        "",
        "## 主对比表（3 构造 × 2 信号 × 含费/不含费）",
        "",
    ]
    lines.extend(build_result_rows(results))
    lines.extend([
        "",
        "## 关键诊断：为什么多空亏损",
        "",
        f"- 合成信号 5 日标签 Rank IC 约 +0.06（复现清单口径），但 1 日前向 IC 仅 "
        f"{diagnostics['ic1d']:+.4f}；截面整体仍正相关，",
        f"- 然而 Top5−Bottom5 尾部 1 日价差均值为 "
        f"{diagnostics['spread1d'] * 100:+.4f}%/日（年化约 "
        f"{diagnostics['spread1d'] * DAYS_PER_YEAR * 100:+.1f}%）："
        "信号在截面两端尾部发生反转（信号按 5 日持有窗口挖掘，"
        "1 日换仓存在持有期错配），因此多空在该口径下大幅亏损，多头的 Top5 恰好是差的那一端。",
        "",
        "## 结论",
        "",
        f"- 最优构造（含费 Sharpe）：**{best['signal_label']} × {best['scheme_label']}**，"
        f"年化 {best['net']['ann_ret'] * 100:+.1f}%、Sharpe {best['net']['sharpe']:+.2f}、"
        f"最大回撤 {best['net']['max_dd'] * 100:.1f}%；",
        f"- 对冲/择时类（b、c）中最好：**{best_hedge['signal_label']} × "
        f"{best_hedge['scheme_label']}**，含费年化 "
        f"{best_hedge['net']['ann_ret'] * 100:+.1f}%、Sharpe {best_hedge['net']['sharpe']:+.2f}；",
        f"- BTC 择时在仓比例约 {btc_flags.mean() * 100:.0f}%"
        f"（{int(btc_flags.sum())}/{len(btc_flags)} 天）。",
        "",
        "## 口径附注",
        "",
        "- 因子清单使用快照 research/ls_timing_study/manifest_snapshot.json，"
        "避免与并行改写的 mined_manifest.json 产生竞态；",
        "- 择时均线用面板全历史滚动计算，2026 年首个交易日已有完整 20 日窗口；",
        "- 评估窗口为面板测试段减去最后一日（该日无次日收盘价）。",
        "",
    ])
    return "\n".join(lines)


def print_summary(results: list[dict]) -> None:
    """stdout 摘要表（含费口径优先展示）。"""
    print("\n================== 测试段组合构造对比（含费 / 不含费年化收益 | Sharpe）==================")
    header: str = f"{'信号':<10}{'构造':<16}{'年化(含费)':>12}{'Sharpe(含费)':>14}{'年化(不含费)':>13}{'最大回撤(含费)':>14}"
    print(header)
    for row in results:
        print(
            f"{row['signal_label']:<10}{row['scheme_label']:<16}"
            f"{row['net']['ann_ret'] * 100:>11.1f}%{row['net']['sharpe']:>14.2f}"
            f"{row['gross']['ann_ret'] * 100:>12.1f}%{row['net']['max_dd'] * 100:>13.1f}%"
        )
    print("=" * 90)


def main() -> None:
    """入口：跑全部 (构造 × 信号) 组合，写 result.md 并打印摘要。"""
    results, btc_flags, diagnostics = run_all()
    report: str = render_report(results, btc_flags, diagnostics)
    output_file: Path = Path(RESULT_PATH)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(report, encoding="utf-8")
    print(f"[输出] 报告已写入 {output_file}")
    print_summary(results)


if __name__ == "__main__":
    main()
