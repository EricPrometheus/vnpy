"""全家桶最终择时研究：单因子 vs 全清单合成 vs 全清单+链上市值。

背景：此前研究结论是"仅 gp_s123 信号 + BTC 20 日均线择时多头"为唯一
翻正构造（年化 +14.4% / Sharpe 0.54）。本次把后续测试线新入选的因子
（lt 低波族、alpha101 负向族、gtja191 回填 3 个）以及链上市值因子
（o_cap_log，链上列经 join 注入——轻量工程化）纳入合成信号，对比：

  A. gp_s123 单因子（原基线）
  B. 全清单合成（gp+lt+alpha101+gtja191 共 14 个）
  C. 全清单 + 链上市值（15 个）

每种信号跑两种构造：纯多头 Top5 等权日调、BTC>MA20 择时多头
（否则空仓）。费用：换手 × 0.1%，与前次研究口径一致。

输出：research/final_study/report.md + stdout 对比表。
用法（在仓库根目录运行）：
    D:\\veighna_studio\\python.exe -X utf8 research/run_final_study.py
"""

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from factor_common import (
    TEST,
    build_factor_table,
    load_manifest,
    segment_mask,
)
from eval_mined_factors import composite_signal, compute_factor_table

REPORT_PATH: Path = Path("research/final_study/report.md")
BTC_SYMBOL: str = "BTCUSDT.OKX"
BTC_MA_WINDOW: int = 20
TOP_N: int = 5
FEE_RATE: float = 0.001
SEGMENTS: dict[str, tuple[str, str]] = {"train": ("2023-01-01", "2024-12-31"),
                                        "valid": ("2025-01-01", "2025-12-31"),
                                        "test": TEST}


def add_onchain_caplog(table: pl.DataFrame) -> pl.DataFrame:
    """把链上 ln 市值因子按 (币, 日期) join 进因子表（轻量工程化）。"""
    from run_onchain_factors import load_onchain

    cm: pl.DataFrame = load_onchain().select(["coin", "date", "o_cap_log"])
    keyed = table.select(
        "datetime",
        "vt_symbol",
        pl.col("datetime").dt.date().alias("date"),
        pl.col("vt_symbol")
        .str.replace("USDT.OKX", "")
        .str.to_lowercase()
        .alias("coin"),
    )
    merged: pl.DataFrame = keyed.join(
        cm, on=["coin", "date"], how="left"
    ).select(["datetime", "vt_symbol", "o_cap_log"])
    joined: pl.DataFrame = table.join(
        merged, on=["datetime", "vt_symbol"], how="left"
    )
    # 关键：链上数据只覆盖部分币，缺失币若留空会在合成排名中沉底、
    # 被隐性踢出候选池——按当日截面中位数中性填充
    daily_median: pl.DataFrame = joined.group_by("datetime").agg(
        pl.col("o_cap_log").fill_nan(None).median().alias("_median_cap")
    )
    return joined.join(daily_median, on="datetime", how="left").with_columns(
        pl.col("o_cap_log").fill_nan(None).fill_null(pl.col("_median_cap"))
    ).drop("_median_cap")


def next_day_returns(panel: pl.DataFrame) -> pd.DataFrame:
    """次日单日前向收益宽表（按币分组 shift）。"""
    closes: pl.DataFrame = panel.select(
        ["datetime", "vt_symbol", "close"]
    ).with_columns(
        (pl.col("close").shift(-1) / pl.col("close") - 1)
        .over("vt_symbol")
        .alias("next_ret")
    )
    return closes.to_pandas().pivot(
        index="datetime", columns="vt_symbol", values="next_ret"
    )


def btc_timing_mask(panel: pl.DataFrame) -> pd.Series:
    """BTC 收盘 > 20 日均线（当日收盘判定，无未来函数）。"""
    btc: pl.DataFrame = panel.filter(
        pl.col("vt_symbol") == BTC_SYMBOL
    ).sort("datetime")
    btc = btc.with_columns(
        pl.col("close").rolling_mean(BTC_MA_WINDOW).alias("ma")
    )
    btc = btc.with_columns(
        (pl.col("close") > pl.col("ma")).fill_null(False).alias("hold")
    )
    series: pd.Series = btc.select(["datetime", "hold"]).to_pandas().set_index(
        "datetime"
    )["hold"]
    return series


def long_top5_returns(composite: pd.Series, pdf: pd.DataFrame,
                      next_wide: pd.DataFrame,
                      timing: pd.Series | None) -> dict[str, float]:
    """合成信号 -> Top5 等权日调组合收益（含费），可选择时过滤。"""
    work: pd.DataFrame = pdf[["datetime", "vt_symbol"]].copy()
    work["composite"] = composite.to_numpy()
    work["rank_desc"] = work.groupby("datetime")["composite"].rank(
        ascending=False, method="first"
    )
    work["weight"] = np.where(work["rank_desc"] <= TOP_N, 1.0 / TOP_N, 0.0)
    if timing is not None:
        hold: pd.Series = work["datetime"].map(timing).fillna(False)
        work["weight"] = work["weight"] * hold.astype(float)

    weights: pd.DataFrame = work.pivot(
        index="datetime", columns="vt_symbol", values="weight"
    ).reindex(columns=next_wide.columns).fillna(0.0)
    portfolio_gross: pd.Series = (weights * next_wide.loc[weights.index]).sum(axis=1)
    turnover: pd.Series = weights.diff().abs().sum(axis=1) / 2.0
    turnover.iloc[0] = weights.iloc[0].abs().sum() / 2.0
    portfolio_net: pd.Series = portfolio_gross - turnover * FEE_RATE
    return {
        "gross_mean": float(portfolio_gross.mean()),
        "net_mean": float(portfolio_net.mean()),
        "net_std": float(portfolio_net.std(ddof=1)),
        "net_series": portfolio_net,
        "avg_turnover": float(turnover.mean()),
    }


def perf_stats(net_series: pd.Series) -> dict[str, float]:
    """年化收益 / 波动 / Sharpe / 最大回撤（加密全年 365 日）。"""
    net: pd.Series = net_series.dropna()
    annual_return: float = float(net.mean() * 365)
    annual_vol: float = float(net.std(ddof=1) * np.sqrt(365))
    sharpe: float = annual_return / annual_vol if annual_vol > 0 else float("nan")
    equity: pd.Series = (1.0 + net).cumprod()
    drawdown: pd.Series = equity / equity.cummax() - 1.0
    return {
        "年化收益": round(annual_return, 4),
        "年化波动": round(annual_vol, 4),
        "Sharpe": round(sharpe, 3),
        "最大回撤": round(float(drawdown.min()), 4),
    }


def main() -> None:
    """构建三种信号 × 两种构造的最终对比。"""
    t0: float = time.time()
    manifest: dict = load_manifest()
    factors: list[dict] = manifest.get("factors", [])
    gp_single: list[dict] = [f for f in factors
                             if f["name"].startswith("gp_s123")]
    print(f"[1] 清单因子 {len(factors)} 个（单因子对照: {len(gp_single)} 个）")

    panel: pl.DataFrame = build_factor_table()
    table: pl.DataFrame = compute_factor_table(panel, factors)
    table_full: pl.DataFrame = add_onchain_caplog(table)
    onchain_factor: list[dict] = [{"name": "o_cap_log", "weight": 1.0,
                                   "source": "onchain"}]
    print(f"[2] 因子表就绪（含链上市值列: "
          f"{'o_cap_log' in table_full.columns}）")

    next_wide: pd.DataFrame = next_day_returns(panel)
    timing: pd.Series = btc_timing_mask(panel)
    dates: pd.DatetimeIndex = next_wide.index
    test_mask: pd.Series = pd.Series(
        (dates >= TEST[0]) & (dates <= TEST[1] + " 23:59:59"), index=dates
    )
    work_pdf: pd.DataFrame = table_full.select(
        ["datetime", "vt_symbol"]
    ).to_pandas()

    signals: dict[str, tuple[pd.Series, list[dict]]] = {
        "A_单因子gp_s123": (
            composite_signal(table_full, gp_single), gp_single),
        "B_全清单合成": (composite_signal(table_full, factors), factors),
        "C_全清单+链上": (
            composite_signal(
                table_full,
                [dict(f) for f in factors] + onchain_factor,
            ),
            [dict(f) for f in factors] + onchain_factor,
        ),
    }

    results: list[dict] = []
    for signal_name, (composite, factor_list) in signals.items():
        for scheme_name, mask in (("纯多头Top5", None), ("BTC择时多头", timing)):
            stats_src: dict[str, float] = long_top5_returns(
                composite, work_pdf, next_wide, mask
            )
            perf: dict[str, float] = perf_stats(
                stats_src["net_series"][test_mask.values]
            )
            results.append({
                "信号": signal_name,
                "构造": scheme_name,
                **perf,
                "日均换手": round(stats_src["avg_turnover"], 4),
                "全期净均值": round(stats_src["net_mean"], 6),
            })

    result_df: pd.DataFrame = pd.DataFrame(results)
    print("\n=============== 测试段（2026 样本外，含费）===============")
    print(result_df.to_string(index=False))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# 全家桶最终择时研究（测试段 2026，含费）\n\n"
        + result_df.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )
    print(f"\n报告已写入 {REPORT_PATH}，耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
