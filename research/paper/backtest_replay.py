"""模式 B：本地模拟盘 —— gp_s123 选币 + BTC 20 日均线择时 + Top5 轮动逐日重放。

把研究管线验证过的策略（2026 样本外年化约 +14.4%，见 research/final_study/report.md）
封装成事件化的逐日模拟撮合，每个交易日收盘后依次执行：
    1) 择时判定：BTCUSDT.OKX 收盘价 > 其 20 日均线 -> 持仓，否则空仓；
    2) 目标持仓：持仓时按 gp_s123 因子值选 Top5 等权；
    3) 与当前持仓比对生成"模拟订单"（买/卖币数），按单边换手 × 0.1% 计费
       （与研究口径一致：fee = Σ|Δw| / 2 × 费率，见 run_final_study.py）；
    4) 次日收盘按新价格对持仓重新估值，滚动出净值曲线。

因子值与既有回测完全同源：复用 factor_common.build_factor_table() 装载日线面板，
用 vnpy 表达式引擎 calculate_by_expression 计算（eval_mined_factors.compute_factor_table，
内部会调用 gtja191_kit.install_operators 注册算子）。gp_s123 表达式只用 ts_delay/ts_mean
等后视算子、选币只做当日截面 rank，因此"全面板一次计算"与"逐日滚动计算"严格等价
（无未来函数），模式 A 实时盘可安全复用同一表达式。

输出：
- research/paper/replay_equity.csv：主口径（仅信号变化时调仓，即策略规则
  "换仓：信号变化时调整"）的逐日持仓（币数）+ 净值 + 模拟订单；
- research/paper/replay_equity_daily.csv：对照口径（每日再平衡到等权，更贴近
  既有回测的权重网格模型），用于交叉验证差异来源。

用法（在仓库根目录运行）：
    D:\\veighna_studio\\python.exe -X utf8 research/paper/backtest_replay.py
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# 引导导入 research/ 下的既有研究模块（本文件位于 research/paper/ 子目录）
_RESEARCH_DIR: Path = Path(__file__).resolve().parent.parent
if str(_RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_DIR))

import numpy as np
import pandas as pd
import polars as pl

from eval_mined_factors import CAPITAL, composite_signal, compute_factor_table
from factor_common import TEST, build_factor_table, load_manifest, safe_join
from run_final_study import (
    FEE_RATE,
    TOP_N,
    btc_timing_mask,
    long_top5_returns,
    next_day_returns,
    perf_stats,
)

REPLAY_CSV_REL: str = "paper/replay_equity.csv"
DAILY_CSV_REL: str = "paper/replay_equity_daily.csv"
WARMUP_DAYS: int = 45        # 测试段前预热天数：足够建仓并跑满因子/均线滚动窗口
GP_PREFIX: str = "gp_s123"   # 因子清单里的选币因子名前缀
REPORT_ANN: float = 0.1436   # 既有报告（final_study/report.md）的基准年化，用于偏差校验
REPORT_SHARPE: float = 0.538
DEVIATION_TOLERANCE: float = 0.03  # 年化偏差超过 3 个百分点必须查明原因


@dataclass
class PortfolioState:
    """模拟账户状态：现金（USDT）+ 各币持仓数量（单位：币）。"""

    cash: float = CAPITAL
    qty: dict[str, float] = field(default_factory=dict)


@dataclass
class TradeTotals:
    """一次调仓的成交汇总（名义额，用于计费与逐日报表）。"""

    buy: float = 0.0
    sell: float = 0.0
    fee: float = 0.0
    lines: list[str] = field(default_factory=list)


@dataclass
class ReplayResult:
    """逐日重放输出：日报表 + 按判定日索引的净收益序列。"""

    daily: pd.DataFrame
    net_series: pd.Series


@dataclass
class ReplayInputs:
    """重放所需的全部输入（面板、因子合成信号、择时、目标集合、收盘宽表）。"""

    panel: pl.DataFrame
    composite: pd.Series
    timing: pd.Series
    target_sets: dict[pd.Timestamp, frozenset[str]]
    closes_wide: pd.DataFrame


def pick_gp_factors(manifest: dict) -> list[dict]:
    """从因子清单取 gp_s123 前缀条目（策略选币信号）。"""
    factors: list[dict] = [
        item for item in manifest.get("factors", [])
        if str(item.get("name", "")).startswith(GP_PREFIX)
    ]
    if not factors:
        raise ValueError(f"因子清单中没有 {GP_PREFIX} 前缀条目，请检查 mined_manifest.json")
    return factors


def build_target_sets(table: pl.DataFrame, composite: pd.Series,
                      ) -> dict[pd.Timestamp, frozenset[str]]:
    """逐日选出因子 Top5 币集合（口径与 long_top5_returns 一致：降序 rank）。"""
    work: pd.DataFrame = table.select(["datetime", "vt_symbol"]).to_pandas()
    work["composite"] = composite.to_numpy()
    work["rank_desc"] = work.groupby("datetime")["composite"].rank(
        ascending=False, method="first"
    )
    picked: pd.DataFrame = work[work["rank_desc"] <= TOP_N]
    return {
        date: frozenset(group["vt_symbol"])
        for date, group in picked.groupby("datetime")
    }


def mark_prices(last_price: dict[str, float], prices: pd.Series) -> None:
    """用当日收盘价更新最新价表；缺失价沿用旧值（停牌/上市前的保守处理）。"""
    for symbol, price in prices.items():
        if pd.notna(price):
            last_price[str(symbol)] = float(price)


def decide_target(target_sets: dict[pd.Timestamp, frozenset[str]],
                  timing: pd.Series, date: pd.Timestamp) -> frozenset[str]:
    """当日目标组合：择时关闭则空仓，否则为当日 Top5 集合。"""
    if not bool(timing.get(date, False)):
        return frozenset()
    return target_sets.get(date, frozenset())


def execute_rebalance(state: PortfolioState, target: frozenset[str],
                      equity: float, last_price: dict[str, float]) -> TradeTotals:
    """按目标组合调仓：先卖后买，成交价 = 当日收盘价，费率按单边换手计。"""
    totals: TradeTotals = TradeTotals()

    # 1) 卖出不在目标里的持仓（整仓清零）
    for symbol in sorted(state.qty):
        volume: float = state.qty[symbol]
        if symbol in target or volume <= 0:
            continue
        notional: float = volume * last_price.get(symbol, 0.0)
        state.cash += notional
        totals.sell += notional
        del state.qty[symbol]
        totals.lines.append(f"SELL {symbol} {volume:.6f}")

    # 2) 买入/换入目标币，等权 equity / TOP_N（与既有回测的权重网格一致）
    per_coin_budget: float = equity / TOP_N
    for symbol in sorted(target):
        price: float = last_price.get(symbol, 0.0)
        if price <= 0:
            continue
        target_volume: float = per_coin_budget / price
        delta: float = target_volume - state.qty.get(symbol, 0.0)
        if delta > 0:
            state.cash -= delta * price
            totals.buy += delta * price
            totals.lines.append(f"BUY {symbol} {delta:.6f}@{price:.6g}")
        elif delta < 0:
            state.cash += -delta * price
            totals.sell += -delta * price
            totals.lines.append(f"SELL {symbol} {-delta:.6f}@{price:.6g}")
        state.qty[symbol] = target_volume

    # 单边换手计费：(买单额+卖单额)/2 × 费率，与研究口径 fee = Σ|Δw|/2 × 费率一致
    totals.fee = (totals.buy + totals.sell) / 2.0 * FEE_RATE
    state.cash -= totals.fee
    return totals


def daily_row(date: pd.Timestamp, target: frozenset[str], timing: pd.Series,
              totals: TradeTotals, state: PortfolioState,
              equity_post: float, rebalanced: bool) -> dict[str, object]:
    """组装单日记录行：信号、订单、费用、现金、净值与逐币持仓数量。"""
    row: dict[str, object] = {
        "date": date.date().isoformat(),
        "hold_timing": bool(timing.get(date, False)),
        "target_coins": "|".join(sorted(target)) if target else "(空仓)",
        "rebalanced": rebalanced,
        "buy_notional": round(totals.buy, 2),
        "sell_notional": round(totals.sell, 2),
        "fee": round(totals.fee, 2),
        "cash": round(state.cash, 2),
        "equity": round(equity_post, 2),
        "orders": "; ".join(totals.lines),
    }
    return row


def append_quantity_columns(records: list[dict[str, object]],
                            state: PortfolioState) -> None:
    """把逐币持仓数量补进当日记录（单独函数，避免 daily_row 超行数限制）。"""
    row: dict[str, object] = records[-1]
    for symbol, volume in state.qty.items():
        if volume > 0:
            row[f"qty_{symbol}"] = round(volume, 6)


def run_replay(inputs: ReplayInputs, warmup_start: pd.Timestamp,
               rebalance_on_signal_change: bool) -> ReplayResult:
    """事件化逐日重放主循环，返回日报表与净收益序列。"""
    state: PortfolioState = PortfolioState()
    last_price: dict[str, float] = {}
    prev_target: frozenset[str] = frozenset()
    records: list[dict[str, object]] = []
    equity_map: dict[pd.Timestamp, float] = {}

    for date in inputs.closes_wide.index:
        if date < warmup_start:
            continue
        mark_prices(last_price, inputs.closes_wide.loc[date])
        equity_pre: float = state.cash + holdings_value(state, last_price)
        target: frozenset[str] = decide_target(inputs.target_sets, inputs.timing, date)

        # 策略规则"换仓：信号变化时调整"；对照口径则每日再平衡
        need_rebalance: bool = (
            target != prev_target if rebalance_on_signal_change else True
        )
        totals: TradeTotals = TradeTotals()
        if need_rebalance:
            totals = execute_rebalance(state, target, equity_pre, last_price)
        prev_target = target

        equity_post: float = state.cash + holdings_value(state, last_price)
        records.append(daily_row(date, target, inputs.timing, totals, state,
                                 equity_post, need_rebalance))
        append_quantity_columns(records, state)
        equity_map[date] = equity_post

    equities: pd.Series = pd.Series(equity_map).sort_index()
    net_series: pd.Series = equities.shift(-1) / equities - 1
    return ReplayResult(pd.DataFrame(records).fillna({"orders": ""}), net_series)


def holdings_value(state: PortfolioState, last_price: dict[str, float]) -> float:
    """按最新价给持仓估值（缺失价沿用旧值，由 mark_prices 保证）。"""
    return sum(
        volume * last_price.get(symbol, 0.0)
        for symbol, volume in state.qty.items()
    )


def mask_test_segment(index: pd.DatetimeIndex) -> np.ndarray:
    """测试段（2026 样本外，含端点）的布尔掩码。"""
    upper: pd.Timestamp = pd.Timestamp(TEST[1]) + pd.Timedelta(hours=23, minutes=59)
    return np.asarray(
        (index >= pd.Timestamp(TEST[0])) & (index <= upper), dtype=bool
    )


def write_replay_csv(daily: pd.DataFrame, relative: str) -> Path:
    """写出逐日持仓与净值（Path.write_text，路径经 safe_join 校验）。"""
    output: Path = safe_join(relative)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(daily.to_csv(index=False), encoding="utf-8")
    return output


def prepare_inputs() -> ReplayInputs:
    """装载面板并用 vnpy 表达式引擎计算因子，产出重放所需的全部输入。"""
    manifest: dict = load_manifest()
    gp_factors: list[dict] = pick_gp_factors(manifest)
    print(f"[1] 选币因子: {[item['name'] for item in gp_factors]}")

    panel: pl.DataFrame = build_factor_table()
    table: pl.DataFrame = compute_factor_table(panel, gp_factors)
    composite: pd.Series = composite_signal(table, gp_factors)
    timing: pd.Series = btc_timing_mask(panel)
    closes_wide: pd.DataFrame = panel.select(
        ["datetime", "vt_symbol", "close"]
    ).to_pandas().pivot(index="datetime", columns="vt_symbol", values="close")
    target_sets: dict[pd.Timestamp, frozenset[str]] = build_target_sets(table, composite)
    print(f"[2] 面板就绪: {panel.height} 行 x {len(closes_wide.columns)} 币")
    return ReplayInputs(panel, composite, timing, target_sets, closes_wide)


def run_engine_baseline(inputs: ReplayInputs) -> dict[str, float]:
    """复算既有回测口径（run_final_study 的 A + BTC择时多头），作为对照基准。"""
    work_pdf: pd.DataFrame = inputs.panel.select(
        ["datetime", "vt_symbol"]
    ).to_pandas()
    baseline: dict[str, float] = long_top5_returns(
        inputs.composite, work_pdf, next_day_returns(inputs.panel), inputs.timing
    )
    net: pd.Series = baseline["net_series"]
    return perf_stats(net[mask_test_segment(net.index)])


def print_compare(label: str, stats: dict[str, float], baseline_ann: float) -> float:
    """打印单口径绩效并返回与既有报告的年化偏差（百分点）。"""
    deviation: float = float(stats["年化收益"]) - baseline_ann
    print(f"  {label}: 年化 {stats['年化收益']:+.4f} / Sharpe {stats['Sharpe']:.3f} "
          f"/ 回撤 {stats['最大回撤']:+.4f} / 与基准偏差 {deviation:+.4f}")
    return deviation


def main() -> None:
    """入口：准备输入 -> 引擎基准复算 -> 双口径重放 -> 输出 CSV 与对比。"""
    start_ms: float = time.time()
    inputs: ReplayInputs = prepare_inputs()

    print("[3] 引擎口径基准复算（期望约 年化+14.36% / Sharpe 0.538）")
    baseline_stats: dict[str, float] = run_engine_baseline(inputs)
    print(f"  基准: 年化 {baseline_stats['年化收益']:+.4f} / "
          f"Sharpe {baseline_stats['Sharpe']:.3f} / 回撤 {baseline_stats['最大回撤']:+.4f}")

    warmup_start: pd.Timestamp = pd.Timestamp(TEST[0]) - pd.Timedelta(days=WARMUP_DAYS)
    print(f"[4] 逐日重放（预热起点 {warmup_start.date()}，费率 {FEE_RATE:.1%}）")
    deviations: dict[str, float] = {}
    results: dict[str, ReplayResult] = {}
    for label, flag in (("主口径-仅信号变化调仓", True),
                        ("对照-每日等权再平衡", False)):
        result: ReplayResult = run_replay(inputs, warmup_start, flag)
        stats: dict[str, float] = perf_stats(
            result.net_series[mask_test_segment(result.net_series.index)]
        )
        deviations[label] = print_compare(label, stats, baseline_stats["年化收益"])
        results[label] = result

    write_replay_csv(results["主口径-仅信号变化调仓"].daily, REPLAY_CSV_REL)
    write_replay_csv(results["对照-每日等权再平衡"].daily, DAILY_CSV_REL)
    worst: float = max(abs(value) for value in deviations.values())
    if worst > DEVIATION_TOLERANCE:
        print(f"[!] 年化偏差 {worst:+.4f} 超过 {DEVIATION_TOLERANCE:.0%} 容忍线，"
              "请对照 replay CSV 的 orders/fee 列查明差异来源")
    else:
        print(f"[OK] 两种口径与基准年化偏差均在 {DEVIATION_TOLERANCE:.0%} 以内")
    print(f"[5] 逐日持仓与净值已写入 research/paper/{Path(REPLAY_CSV_REL).name} "
          f"与 {Path(DAILY_CSV_REL).name}，耗时 {time.time() - start_ms:.0f}s")


if __name__ == "__main__":
    main()
