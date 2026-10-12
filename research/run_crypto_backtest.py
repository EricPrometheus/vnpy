"""加密货币 Alpha158 选币回测（加载已生成信号）。

前置：run_crypto_pipeline.py 已生成信号 lgb_alpha158_crypto_v1。
本脚本只做组合回测（Top5 币轮动，2026 样本外）。

用法：
    D:\\veighna_studio\\python.exe -X utf8 research/run_crypto_backtest.py
"""

import os
from datetime import datetime
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)

import polars as pl

from vnpy.alpha.lab import AlphaLab
from vnpy.alpha.strategy.backtesting import BacktestingEngine
from vnpy.alpha.strategy.strategies.equity_demo_strategy import EquityDemoStrategy
from vnpy.trader.constant import Interval

LAB_PATH: Path = REPO_ROOT / "research" / "alpha_lab"

TEST_START: str = "2026-01-01"
TEST_END: str = "2026-09-28"
CAPITAL: int = 1_000_000

# 合约：size=1（币按枚算），引擎费率 0（策略内自算佣金）
LONG_RATE: float = 0.0
SHORT_RATE: float = 0.0
SIZE: float = 1.0
PRICETICK: float = 0.01


def setup_contracts(lab: AlphaLab, vt_symbols: list[str]) -> None:
    for vt_symbol in vt_symbols:
        lab.add_contract_setting(vt_symbol, LONG_RATE, SHORT_RATE, SIZE, PRICETICK)


def main() -> None:
    lab = AlphaLab(str(LAB_PATH))
    vt_symbols: list[str] = sorted(f.stem for f in (LAB_PATH / "daily").glob("*.OKX.parquet"))
    setup_contracts(lab, vt_symbols)
    signal_df: pl.DataFrame = lab.load_signal("lgb_alpha158_crypto_v1")
    print(f"信号 {signal_df.height} 条，币池 {len(vt_symbols)} 个")

    engine = BacktestingEngine(lab)
    engine.set_parameters(
        vt_symbols=vt_symbols,
        interval=Interval.DAILY,
        start=datetime.strptime(TEST_START, "%Y-%m-%d"),
        end=datetime.strptime(TEST_END, "%Y-%m-%d"),
        capital=CAPITAL,
    )
    engine.add_strategy(EquityDemoStrategy, {}, signal_df)
    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    stats: dict = engine.calculate_statistics()

    print("\n========== 加密 Alpha158 + LightGBM 选币回测报告（2026样本外） ==========")
    for k, v in stats.items():
        print(f"{k}: {round(v, 4) if isinstance(v, float) else v}")


if __name__ == "__main__":
    main()