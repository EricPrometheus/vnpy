"""批量回测：官方 8 个经典策略 × BTC 日线（2019~2026，OKX）。

对比 IF 股指版（run_backtest_batch.py），验证"趋势策略在 BTC 上更强"的假设。
合约假设：现货 OKX，1手=1BTC，手续费 taker 万10，滑点 1 USDT。

用法：
    D:\\veighna_studio\\python.exe -X utf8 research/run_backtest_btc.py
"""

import os
from datetime import datetime
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)

import pandas as pd

from vnpy.trader.constant import Interval
from vnpy_ctastrategy.backtesting import BacktestingEngine
from vnpy_ctastrategy.strategies.atr_rsi_strategy import AtrRsiStrategy
from vnpy_ctastrategy.strategies.boll_channel_strategy import BollChannelStrategy
from vnpy_ctastrategy.strategies.double_ma_strategy import DoubleMaStrategy
from vnpy_ctastrategy.strategies.dual_thrust_strategy import DualThrustStrategy
from vnpy_ctastrategy.strategies.king_keltner_strategy import KingKeltnerStrategy
from vnpy_ctastrategy.strategies.multi_signal_strategy import MultiSignalStrategy
from vnpy_ctastrategy.strategies.multi_timeframe_strategy import MultiTimeframeStrategy
from vnpy_ctastrategy.strategies.turtle_signal_strategy import TurtleSignalStrategy

# BTC 现货回测参数
VT_SYMBOL: str = "BTCUSDT.OKX"
INTERVAL: Interval = Interval.DAILY
START: datetime = datetime(2019, 1, 1)
END: datetime = datetime(2026, 9, 27)
RATE: float = 0.001       # OKX 现货 taker 0.1%
SLIPPAGE: float = 1.0     # 1 USDT
SIZE: int = 1             # 1手 = 1 BTC（约8%仓位）
PRICETICK: float = 0.1
CAPITAL: int = 1_000_000

STRATEGIES: list[tuple[type, dict]] = [
    (DoubleMaStrategy, {}),
    (TurtleSignalStrategy, {}),
    (AtrRsiStrategy, {}),
    (DualThrustStrategy, {}),
    (KingKeltnerStrategy, {}),
    (BollChannelStrategy, {}),
    (MultiSignalStrategy, {}),
    (MultiTimeframeStrategy, {}),
]

REPORT_KEYS: list[str] = [
    "total_return", "annual_return", "max_ddpercent",
    "sharpe_ratio", "total_trade_count", "total_commission",
]


def run_one(strategy_class: type, settings: dict) -> dict:
    """跑单个策略返回汇总行。"""
    name: str = strategy_class.__name__.replace("Strategy", "")
    engine = BacktestingEngine()
    engine.set_parameters(
        vt_symbol=VT_SYMBOL, interval=INTERVAL, start=START, end=END,
        rate=RATE, slippage=SLIPPAGE, size=SIZE,
        pricetick=PRICETICK, capital=CAPITAL,
    )
    engine.add_strategy(strategy_class, settings)
    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    stats: dict = engine.calculate_statistics(output=False)
    row: dict = {"策略": name}
    for k in REPORT_KEYS:
        v = stats.get(k)
        row[k] = round(v, 3) if isinstance(v, float) else v
    return row


def main() -> None:
    rows: list[dict] = []
    for cls, settings in STRATEGIES:
        print(f"回测中: {cls.__name__} ...")
        try:
            rows.append(run_one(cls, settings))
        except Exception as e:
            rows.append({"策略": cls.__name__.replace("Strategy", ""), "error": str(e)[:60]})

    df = pd.DataFrame(rows)
    if "total_return" in df.columns:
        df = df.sort_values("total_return", ascending=False)

    print("\n" + "=" * 95)
    print(f"{'BTC 策略大乱斗':^18s} BTC日线 2019-01 ~ 2026-09  初始100万  成本:千1+滑点1U")
    print("=" * 95)
    print(df.to_string(index=False))

    out = Path(__file__).parent / "backtest_btc_compare.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n对比表已保存: {out}")


if __name__ == "__main__":
    main()