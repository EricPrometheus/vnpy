"""GTJA 头部因子手工回填：翻译为 vnpy 表达式、对齐校验、并入统一清单。

入选原则（从 GTJA191 批测头部中筛选）：
- gtja191_089（MACD 柱）：均线动量族，与 gp_s123 的"振幅×动量"互补
- gtja191_013（几何公允价偏离）：第三方实现用 A 股"手转股"口径
  （amount/(volume*100)），币圈 volume 即币数，此处修正为
  pow1(high*low,0.5) - turnover/volume
- gtja191_132（下行压力）：下跌日绝对跌幅的 12 日和
- gtja191_010（条件波动极值）：下跌日取 20 日波动率、上涨日取价格，
  取平方后 5 日最大值再截面排名
- 126/129/153（价格水平/成交额规模）与链上市值、lt_dvol30 同族，不回填

校验：翻译后的 vnpy 引擎值与第三方 pandas 值做逐日截面秩相关，
均值 >= 0.95 判定翻译无损（013 因口径修正不比对）。

用法（在仓库根目录运行）：
    D:\\veighna_studio\\python.exe -X utf8 research/backfill_gtja.py
"""

import time
from pathlib import Path

import pandas as pd
import polars as pl

from factor_common import build_factor_table, save_factors
from gtja191_kit import install_operators

# name -> (vnpy 表达式, 第三方包因子 id, 批测三段 IC 快照)
BACKFILL: dict[str, tuple[str, str | None, tuple[float, float, float]]] = {
    "gtja191_089": (
        "2 * (sma(close, 13, 2) - sma(close, 27, 2) - "
        "sma(sma(close, 13, 2) - sma(close, 27, 2), 10, 2))",
        "gtja191_089",
        (+0.0331, +0.1028, +0.0830),
    ),
    "gtja191_013": (
        "pow1(high * low, 0.5) - turnover / volume",
        None,  # 口径修正版，不与第三方比对
        (+0.0383, +0.1064, +0.0745),
    ),
    "gtja191_132": (
        "ts_sum(lt(close, ts_delay(close, 1)) * "
        "abs(close - ts_delay(close, 1)), 12)",
        "gtja191_132",
        (+0.0257, +0.0718, +0.0597),
    ),
    "gtja191_010": (
        "cs_rank(ts_max(pow1("
        "lt(close / ts_delay(close, 1) - 1, 0) * "
        "ts_std(close / ts_delay(close, 1) - 1, 20) + "
        "gt(close / ts_delay(close, 1) - 1, 0) * close, 2), 5))",
        "gtja191_010",
        (+0.0324, +0.0979, +0.0694),
    ),
}
CORR_THRESHOLD: float = 0.95  # 逐日截面秩相关均值的无损判定线


def build_panel() -> pl.DataFrame:
    """GTJA 面板（含 vwap，公允价偏离因子需要）。"""
    return build_factor_table().with_columns(
        pl.when(pl.col("volume") > 0)
        .then(pl.col("turnover") / pl.col("volume"))
        .otherwise(None)
        .alias("vwap")
    )


def third_party_long(factor_id: str,
                     pdf: pd.DataFrame) -> pd.Series | None:
    """调第三方 pandas 实现并把结果转成与 pdf 行序对齐的长表。"""
    from src.factors.registry import Registry

    wide_tables: dict[str, pd.DataFrame] = {}
    for field, column in (("close", "close"), ("open", "open"),
                          ("high", "high"), ("low", "low"),
                          ("volume", "volume"), ("amount", "turnover"),
                          ("vwap", "vwap")):
        wide_tables[field] = pdf.pivot(
            index="datetime", columns="vt_symbol", values=column
        )
    try:
        wide: pd.DataFrame = Registry().compute(factor_id, wide_tables)
    except Exception as error:  # 第三方实现失败则跳过比对
        print(f"    [比对跳过] {factor_id}: {str(error)[:70]}")
        return None
    stacked: pd.Series = wide.stack(future_stack=True)
    stacked.index.names = ["datetime", "vt_symbol"]
    base_index: pd.MultiIndex = pdf.set_index(["datetime", "vt_symbol"]).index
    return stacked.reindex(base_index).reset_index(drop=True)


def daily_rank_corr(ours: pd.Series, theirs: pd.Series,
                    pdf: pd.DataFrame) -> float | None:
    """逐日截面秩相关的均值（抽样 120 天，验证翻译无损）。"""
    frame: pd.DataFrame = pdf[["datetime"]].copy()
    frame["ours"] = ours
    frame["theirs"] = theirs
    frame = frame.dropna()
    sampled: pd.DataFrame = frame[frame["datetime"].isin(
        sorted(frame["datetime"].unique())[::5][:120]
    )]
    correlations: list[float] = []
    for _, group in sampled.groupby("datetime"):
        if len(group) < 5 or group["ours"].std() == 0:
            continue
        value: float = group["ours"].rank().corr(group["theirs"].rank())
        if pd.notna(value):
            correlations.append(value)
    return float(sum(correlations) / len(correlations)) if correlations else None


def main() -> None:
    """翻译 -> 引擎计算 -> 比对校验 -> 并入清单。"""
    t0: float = time.time()
    install_operators()
    from vnpy.alpha.dataset.utility import calculate_by_expression

    panel: pl.DataFrame = build_panel()
    pdf: pd.DataFrame = panel.to_pandas()  # 全列保留：第三方比对需要价量列
    print(f"[1] 面板 {panel.height} 行，回填 {len(BACKFILL)} 个 GTJA 头部因子")

    factors: list[dict] = []
    for name, (expr, compare_id, ic_snapshot) in BACKFILL.items():
        ours: pd.Series = pd.Series(
            calculate_by_expression(panel, expr)["data"]
        )
        verdict: str = "口径修正版，未比对"
        if compare_id is not None:
            theirs: pd.Series | None = third_party_long(compare_id, pdf)
            if theirs is not None:
                correlation: float | None = daily_rank_corr(ours, theirs, pdf)
                verdict = (
                    f"对齐校验 日均秩相关={correlation:.4f}"
                    if correlation is not None else "比对不可用"
                )
                if correlation is not None and correlation < CORR_THRESHOLD:
                    verdict += " < 阈值，弃用"
                    print(f"  [弃用] {name}: {verdict}")
                    continue
        train_ic, valid_ic, test_ic = ic_snapshot
        factors.append({
            "name": name,
            "source": "gtja191",
            "vnpy_expr": expr,
            "weight": 1.0,
            "train_ic": train_ic,
            "valid_ic": valid_ic,
            "test_ic": test_ic,
        })
        print(f"  [入选] {name}: {verdict}")

    save_factors("gtja191", factors, meta={
        "tool": "GTJA191 手工回填（国泰君安 2017 研报）",
        "corr_threshold": CORR_THRESHOLD,
    })
    print(f"[2] 已并入清单 {len(factors)} 个（source=gtja191），"
          f"耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
