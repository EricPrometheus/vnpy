"""统一评估：把 mined_manifest.json 里的因子放到与现有管线相同的口径下检验。

对清单中每个因子（gp / alphagen 来源通用）计算：
- 分段（训练/验证/测试）日均截面 Rank IC 与 ICIR（IC 信息比率）；
- 等权（按 weight）合成的复合信号的分段 IC。

可选参数：
- --backtest：用复合信号在测试段跑 Top5 币轮动回测（AlphaLab 引擎）；
- --baseline：对已存档的 lgb_alpha158_crypto_v1 信号跑同样的回测作对照。

用法（在仓库根目录运行）：
    D:\\veighna_studio\\python.exe -X utf8 research/eval_mined_factors.py
    D:\\veighna_studio\\python.exe -X utf8 research/eval_mined_factors.py --backtest --baseline
"""

import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl

from factor_common import (
    LABEL_VNPY,
    TEST,
    TRAIN,
    VALID,
    baseline_signal_path,
    build_factor_table,
    daily_dir,
    daily_rank_ic_series,
    load_manifest,
    segment_mask,
)

# 合约设置与 run_crypto_pipeline.py 保持一致：加密按"币"计，引擎费率归零
LONG_RATE: float = 0.0
SHORT_RATE: float = 0.0
SIZE: float = 1.0
PRICETICK: float = 0.01
CAPITAL: float = 1_000_000

SEGMENTS: dict[str, tuple[str, str]] = {"训练": TRAIN, "验证": VALID, "测试": TEST}


def compute_factor_table(panel: pl.DataFrame, factors: list[dict]) -> pl.DataFrame:
    """在完整面板上计算全部因子列与标签列（vnpy 引擎，口径与管线一致）。"""
    from vnpy.alpha.dataset.utility import calculate_by_expression

    from gtja191_kit import install_operators  # 清单含 sma/gt/lt 等注册算子

    install_operators()
    if "vwap" not in panel.columns:  # Alpha101/GTJA 部分公式引用 vwap
        panel = panel.with_columns(
            pl.when(pl.col("volume") > 0)
            .then(pl.col("turnover") / pl.col("volume"))
            .otherwise(None)
            .alias("vwap")
        )
    table: pl.DataFrame = panel.select(["datetime", "vt_symbol"])
    table = table.with_columns(
        calculate_by_expression(panel, LABEL_VNPY)["data"].alias("label")
    )
    for factor in factors:
        col = calculate_by_expression(panel, factor["vnpy_expr"])["data"]
        table = table.with_columns(col.alias(factor["name"]))
    return table


def _segment_ic(table: pl.DataFrame, column: str,
                segment: tuple[str, str]) -> tuple[float, float]:
    """某列在某时间段的 (日均 Rank IC, ICIR)。"""
    pdf: pd.DataFrame = table.filter(segment_mask(segment)).to_pandas()
    ic_series: pd.Series = daily_rank_ic_series(
        pdf[column], pdf["label"], pdf["datetime"]
    ).dropna()
    if ic_series.empty:
        return float("nan"), float("nan")
    icir: float = (
        float(ic_series.mean() / ic_series.std(ddof=1)) if len(ic_series) > 1 else float("nan")
    )
    return float(ic_series.mean()), icir


def print_ic_report(table: pl.DataFrame, factors: list[dict]) -> None:
    """打印分段 IC/ICIR 报告（含复合信号）。"""
    print("\n================== 因子分段 Rank IC 报告 ==================")
    header: str = f"{'因子':<14}{'来源':<10}" + "".join(
        f"{name + ' IC/ICIR':>24}" for name in SEGMENTS
    )
    print(header)
    for factor in factors:
        cells: list[str] = []
        for segment in SEGMENTS.values():
            ic_value, icir_value = _segment_ic(table, factor["name"], segment)
            cells.append(f"{ic_value:+.4f} / {icir_value:+.2f}")
        print(f"{factor['name']:<14}{factor.get('source', '?'):<10}"
              + "".join(f"{cell:>24}" for cell in cells))

    composite: pd.Series = composite_signal(table, factors)
    table_composite: pl.DataFrame = table.with_columns(
        pl.Series("composite", composite.to_numpy())
    )
    cells = []
    for segment in SEGMENTS.values():
        ic_value, icir_value = _segment_ic(table_composite, "composite", segment)
        cells.append(f"{ic_value:+.4f} / {icir_value:+.2f}")
    print(f"{'composite':<14}{'合成':<10}" + "".join(f"{cell:>24}" for cell in cells))
    print("=" * 90)


def composite_signal(table: pl.DataFrame, factors: list[dict]) -> pd.Series:
    """按 weight 加权合成信号：每个因子先做当日截面 rank 归一化再求和。"""
    pdf: pd.DataFrame = table.to_pandas()
    composite: pd.Series = pd.Series(0.0, index=pdf.index)
    for factor in factors:
        weight: float = float(factor.get("weight", 1.0))
        ranked: pd.Series = pdf.groupby("datetime")[factor["name"]].rank()
        count: pd.Series = pdf.groupby("datetime")[factor["name"]].transform("count")
        composite = composite + weight * (ranked / count)
    return composite


def run_backtest(signal_df: pl.DataFrame, title: str) -> None:
    """用给定信号在测试段跑 Top5 币轮动回测（复刻管线第 5 步）。"""
    from vnpy.alpha.lab import AlphaLab
    from vnpy.alpha.strategy.backtesting import BacktestingEngine
    from vnpy.alpha.strategy.strategies.equity_demo_strategy import EquityDemoStrategy
    from vnpy.trader.constant import Interval

    vt_symbols: list[str] = sorted(p.stem for p in daily_dir().glob("*.OKX.parquet"))
    lab: AlphaLab = AlphaLab("research/alpha_lab")
    for vt_symbol in vt_symbols:
        lab.add_contract_setting(vt_symbol, LONG_RATE, SHORT_RATE, SIZE, PRICETICK)

    engine: BacktestingEngine = BacktestingEngine(lab)
    engine.set_parameters(
        vt_symbols=vt_symbols,
        interval=Interval.DAILY,
        start=datetime.strptime(TEST[0], "%Y-%m-%d"),
        end=datetime.strptime(TEST[1], "%Y-%m-%d"),
        capital=CAPITAL,
    )
    engine.add_strategy(EquityDemoStrategy, {}, signal_df)
    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    stats: dict = engine.calculate_statistics()

    print(f"\n========== {title}（{TEST[0]} ~ {TEST[1]}）==========")
    for key, value in stats.items():
        print(f"{key}: {round(value, 4) if isinstance(value, float) else value}")


def backtest_composite(table: pl.DataFrame, factors: list[dict]) -> None:
    """复合信号 -> 测试段回测。"""
    composite: pd.Series = composite_signal(table, factors)
    signal_df: pl.DataFrame = table.select(["datetime", "vt_symbol"]).with_columns(
        pl.Series("signal", composite.to_numpy())
    ).filter(segment_mask(TEST))
    run_backtest(signal_df, "挖掘因子复合信号 Top5 轮动回测")


def backtest_baseline() -> None:
    """Alpha158+LightGBM 历史信号 -> 同样回测，作为对照组。"""
    if not baseline_signal_path().exists():
        print(f"[跳过] 未找到基线信号文件: {baseline_signal_path()}")
        return
    signal_df: pl.DataFrame = pl.read_parquet(baseline_signal_path()).filter(
        segment_mask(TEST)
    )
    run_backtest(signal_df, "基线对照：Alpha158 + LightGBM Top5 轮动回测")


def main() -> None:
    """入口：解析参数 -> 计算因子 -> IC 报告 -> 可选回测。"""
    parser = argparse.ArgumentParser(description="挖掘因子统一评估")
    parser.add_argument("--backtest", action="store_true", help="复合信号测试段回测")
    parser.add_argument("--baseline", action="store_true", help="附带 Alpha158 基线回测")
    args = parser.parse_args()

    manifest: dict = load_manifest()
    factors: list[dict] = manifest.get("factors", [])
    if not factors:
        print("因子清单为空，请先运行 run_gp_mining.py 或 alphagen 导出脚本。")
        return
    print(f"清单中共 {len(factors)} 个因子: "
          f"{[f['name'] for f in factors]}")

    panel: pl.DataFrame = build_factor_table()
    table: pl.DataFrame = compute_factor_table(panel, factors)
    print_ic_report(table, factors)

    if args.backtest:
        backtest_composite(table, factors)
    if args.baseline:
        backtest_baseline()


if __name__ == "__main__":
    main()
