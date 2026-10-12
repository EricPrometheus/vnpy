"""加密货币横截面因子管线：Alpha158 + LightGBM + 选币组合回测。

数据：OKX 主流币日线（research/alpha_lab/daily/*.OKX.parquet）。
划分：训练 2023-01 ~ 2024-12（保证全部币都有数据）| 验证 2025 | 测试 2026。
逻辑复用股票管线（单进程逐因子计算，避免内存爆炸）。

用法：
    D:\\veighna_studio\\python.exe -X utf8 research/run_crypto_pipeline.py
"""

import os
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)

import polars as pl

from vnpy.alpha.lab import AlphaLab
from vnpy.alpha.dataset.datasets.alpha_158 import Alpha158
from vnpy.alpha.dataset.utility import Segment, calculate_by_expression
from vnpy.alpha.model.models.lgb_model import LgbModel
from vnpy.alpha.strategy.backtesting import BacktestingEngine
from vnpy.alpha.strategy.strategies.equity_demo_strategy import EquityDemoStrategy
from vnpy.trader.constant import Interval

LAB_PATH: Path = REPO_ROOT / "research" / "alpha_lab"

# 数据划分（2023 起保证 22 个币都有完整数据）
TRAIN: tuple[str, str] = ("2023-01-01", "2024-12-31")
VALID: tuple[str, str] = ("2025-01-01", "2025-12-31")
TEST: tuple[str, str] = ("2026-01-01", "2026-09-28")

# 标签：未来 5 日收益（加密 7x24，同样用未来 N 日收益）
LABEL: str = "ts_delay(close, -6) / ts_delay(close, -1) - 1"

# 合约设置：加密按"币"算（size=1），引擎费率归零（策略内自算佣金）
LONG_RATE: float = 0.0
SHORT_RATE: float = 0.0
SIZE: float = 1.0
PRICETICK: float = 0.01


def setup_contracts(lab: AlphaLab, vt_symbols: list[str]) -> None:
    """写入合约交易配置。"""
    for vt_symbol in vt_symbols:
        lab.add_contract_setting(vt_symbol, LONG_RATE, SHORT_RATE, SIZE, PRICETICK)


def prepare_data_sequential(dataset: Alpha158) -> None:
    """单进程逐因子计算（内存友好版，复用股票管线经验）。"""
    t0: float = time.time()
    result_df: pl.DataFrame = dataset.df

    total: int = len(dataset.feature_expressions) + 1
    for idx, (name, expression) in enumerate(dataset.feature_expressions.items(), 1):
        result: pl.DataFrame = calculate_by_expression(result_df, expression)
        result_df = result_df.with_columns(result["data"].alias(name))
        if idx % 30 == 0:
            print(f"    因子 {idx}/{total - 1} ({time.time() - t0:.0f}s)")

    label_result: pl.DataFrame = calculate_by_expression(result_df, dataset.label_expression)
    result_df = result_df.with_columns(label_result["data"].alias("label"))
    print(f"    因子+标签计算完成: {total} 个，耗时 {time.time() - t0:.0f}s")

    raw_df: pl.DataFrame = result_df.fill_null(float("nan"))
    select_columns: list[str] = ["datetime", "vt_symbol"] + raw_df.columns[dataset.df.width:]
    dataset.result_df = result_df
    dataset.raw_df = raw_df.select(select_columns).sort(["datetime", "vt_symbol"])
    dataset.infer_df = dataset.raw_df
    dataset.learn_df = dataset.raw_df


def main() -> None:
    """完整加密管线。"""
    lab = AlphaLab(str(LAB_PATH))

    # 0. 币符号（从 parquet 文件名）
    vt_symbols: list[str] = sorted(
        f.stem for f in (LAB_PATH / "daily").glob("*.OKX.parquet")
    )
    print(f"[0] 加密资产池 {len(vt_symbols)} 个")
    setup_contracts(lab, vt_symbols)

    # 1. 横截面数据
    print("[1] 加载横截面数据 ...")
    df: pl.DataFrame = lab.load_bar_df(vt_symbols, Interval.DAILY, "2022-06-01", TEST[1], 100)
    print(f"    {df.height} 行 x {df.width} 列")

    # 2. Alpha158 因子
    print("[2] 逐因子计算 Alpha158 ...")
    dataset = Alpha158(df=df, train_period=TRAIN, valid_period=VALID, test_period=TEST)
    dataset.set_label(LABEL)
    prepare_data_sequential(dataset)
    lab.save_dataset("alpha158_crypto_v1", dataset)

    # 3. 训练
    print("[3] 训练 LightGBM ...")
    model = LgbModel(learning_rate=0.05, num_boost_round=500,
                     early_stopping_rounds=50, log_evaluation_period=50)
    model.fit(dataset)
    lab.save_model("lgb_alpha158_crypto_v1", model)

    # 4. 信号
    print("[4] 生成测试段信号 ...")
    pred = model.predict(dataset, Segment.TEST)
    infer_df: pl.DataFrame = dataset.fetch_infer(Segment.TEST)
    signal_df: pl.DataFrame = infer_df.select(["datetime", "vt_symbol"]).with_columns(
        pl.Series("signal", pred)
    )
    lab.save_signal("lgb_alpha158_crypto_v1", signal_df)
    print(f"    信号 {signal_df.height} 条 / {signal_df['datetime'].n_unique()} 个交易日")

    # 5. 组合回测（持 5 币轮动）
    print("[5] 组合回测（Top5 币轮动，2026样本外）...")
    engine = BacktestingEngine(lab)
    engine.set_parameters(
        vt_symbols=vt_symbols,
        interval=Interval.DAILY,
        start=datetime.strptime(TEST[0], "%Y-%m-%d"),
        end=datetime.strptime(TEST[1], "%Y-%m-%d"),
        capital=1_000_000,
    )
    engine.add_strategy(EquityDemoStrategy, {}, signal_df)
    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    stats: dict = engine.calculate_statistics()

    print("\n========== 加密 Alpha158 + LightGBM 选币回测报告 ==========")
    for k, v in stats.items():
        print(f"{k}: {round(v, 4) if isinstance(v, float) else v}")


if __name__ == "__main__":
    main()