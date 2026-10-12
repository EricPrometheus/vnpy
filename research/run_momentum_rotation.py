"""22 币动量轮动回测（加密经典策略：定期换仓持有近期最强币）。

策略规则（经典截面动量）：
- 每周最后一个交易日调仓
- 按过去 N 日收益率排名，卖出跌出前 K 名的，买入新进前 K 名的
- 等权持有 K 个币

数据：research/alpha_lab/daily/*.OKX.parquet（22 币，2019~今）
用法：D:\\veighna_studio\\python.exe -X utf8 research/run_momentum_rotation.py
"""

import os
from datetime import datetime
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)

import polars as pl

DATA_DIR: Path = REPO_ROOT / "research" / "alpha_lab" / "daily"

# ---- 策略参数 ----
LOOKBACK_DAYS: int = 7      # 动量回顾期（天）
TOP_K: int = 3              # 持有最强的 K 个币
REBALANCE_EVERY: int = 7    # 每 7 个交易日调仓
START_DATE: str = "2023-01-01"   # 保证大部分币已上市（22 币最晚 2023 年）
END_DATE: str = "2026-09-27"
CAPITAL: float = 100_000.0
FEE: float = 0.001          # 单边手续费 0.1%


def load_closes() -> pl.DataFrame:
    """读全部 OKX 币日线 -> 宽表（date x 每币收盘价）。"""
    frames = []
    for f in sorted(DATA_DIR.glob("*.OKX.parquet")):
        df = pl.read_parquet(f).select([
            pl.col("datetime").dt.date().alias("date"),
            pl.col("close"),
            pl.lit(f.stem).alias("coin"),
        ])
        frames.append(df)
    long_df = pl.concat(frames)
    wide = long_df.pivot(index="date", on="coin", values="close").sort("date")
    return wide.filter(
        (pl.col("date") >= datetime.strptime(START_DATE, "%Y-%m-%d").date())
        & (pl.col("date") <= datetime.strptime(END_DATE, "%Y-%m-%d").date())
    )


def main() -> None:
    wide = load_closes()
    coins: list[str] = [c for c in wide.columns if c != "date"]
    dates = wide["date"].to_list()
    print(f"币数 {len(coins)} | 交易日 {len(dates)} | {dates[0]} ~ {dates[-1]}")

    # 转成 python 列表便于逐日处理
    closes = {c: wide[c].to_list() for c in coins}
    n = len(dates)

    cash = CAPITAL
    holdings: dict[str, float] = {}   # coin -> 币数量
    last_rebalance = -1
    equity_curve: list[float] = []
    trade_count = 0

    for i in range(n):
        # 估值
        equity = cash + sum(qty * closes[c][i] for c, qty in holdings.items() if closes[c][i])
        equity_curve.append(equity)

        # 每 REBALANCE_EVERY 天调仓（且不是最后一天）
        if i - last_rebalance >= REBALANCE_EVERY and i < n - LOOKBACK_DAYS:
            last_rebalance = i

            # 动量排名：过去 LOOKBACK_DAYS 收益率（要求两端价格有效）
            momentum: list[tuple[str, float]] = []
            for c in coins:
                p_now, p_then = closes[c][i], closes[c][i - LOOKBACK_DAYS]
                if p_now and p_then:
                    momentum.append((c, p_now / p_then - 1))
            momentum.sort(key=lambda x: -x[1])
            target = {c for c, _ in momentum[:TOP_K]}

            # 全部清仓（简化：一次性换仓）
            for c, qty in list(holdings.items()):
                price = closes[c][i]
                if price:
                    cash += qty * price * (1 - FEE)
                    trade_count += 1
            holdings = {}

            # 买入目标币（等权，留一份现金零头防浮点）
            if target:
                per = cash * 0.98 / len(target)
                for c in target:
                    price = closes[c][i]
                    if price:
                        qty = per / price
                        holdings[c] = qty
                        cash -= qty * price
                        trade_count += 1

    # 绩效统计
    final = equity_curve[-1]
    total_ret = final / CAPITAL - 1
    years = (dates[-1] - dates[0]).days / 365
    annual = (final / CAPITAL) ** (1 / years) - 1
    peak = 0
    max_dd = 0.0
    for e in equity_curve:
        peak = max(peak, e)
        max_dd = min(max_dd, e / peak - 1)

    # 日收益率 -> 夏普（简化，无风险利率 0）
    rets = [equity_curve[i] / equity_curve[i-1] - 1 for i in range(1, len(equity_curve)) if equity_curve[i-1] > 0]
    import math
    mean_r = sum(rets) / len(rets)
    std_r = math.sqrt(sum((r - mean_r) ** 2 for r in rets) / len(rets))
    sharpe = mean_r / std_r * math.sqrt(365) if std_r > 0 else 0

    print(f"\n========== 22币动量轮动（{LOOKBACK_DAYS}日动量 Top{TOP_K}，每{REBALANCE_EVERY}日调仓） ==========")
    print(f"回测区间: {dates[0]} ~ {dates[-1]}（{years:.1f} 年）")
    print(f"总收益: {total_ret:+.2%}（{CAPITAL:,.0f} -> {final:,.0f}）")
    print(f"年化收益: {annual:+.2%}")
    print(f"最大回撤: {max_dd:.2%}")
    print(f"夏普比率: {sharpe:.2f}")
    print(f"成交笔数: {trade_count}")
    print(f"期末持仓: {list(holdings.keys())}")


if __name__ == "__main__":
    main()