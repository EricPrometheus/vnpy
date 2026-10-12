"""5 分钟尾部价差挖矿 v2：在 v1 因子上叠加"低换手"交易政策。

背景（v1 结论，见 research/gp_5m/report.md）：GP 因子方向真实但测试段
毛价差仅 +0.0018%/bar，逐 bar 全额换手下 maker 往返费 0.04%/bar 是其
20 倍，测试段净值 0/9 个月为正。打平数学：平均持仓 >= 110 分钟
（22 根 5m bar）。v2 只动交易结构，不动因子本身：

1. 公式复算：从 v1 的 mining_result JSON 读回三个种子的名人堂公式，
   在 5m 面板上用 numpy 复算（受保护除法语义与 gplearn 对齐），
   并按 v1 口径（段内全局分位尾部价差）对账，容差 1e-9；
2. 滚动分位归一：因子值 -> 相对过去 2016 根 bar（约一周）的滚动
   百分位，映射为 [-1,+1] 符号分（修复 v1 的段内全局分位前视简化；
   窗口不满 2016 根时为 NaN，保持空仓）；
3. 迟滞交易政策（施密特触发器，逐币路径模拟、全向量化）：
   空仓时 符号分 >= +Q_entry 开多 / <= -Q_entry 开空；
   持多时 符号分 < +Q_exit 平仓；持空时 符号分 > -Q_exit 平仓；
   波动过滤只挡开仓（vola20 同窗口滚动分位 > P50 / P70）。
   参数网格只在 2025 验证段上选（净收益最大化，并列取更长持仓），
   冻结后测试段只评一次。

费用口径：每笔成交 0.02%（maker 单边），仓位每变化 1 个单位计一次，
完整开+平往返 = 0.04%，与 v1 一致。每个评估段从空仓开始独立模拟
（滚动分位仍用全历史，只依赖过去，无前视）。

预注册成败判据（不得事后放宽）：测试段（2026-01-01~09-21）三个种子中
至少 2 个满足：净收益 > 0 且平均持仓时长 >= 110 分钟。

用法（仓库根目录运行）：
    D:\\veighna_studio\\python.exe -X utf8 research/run_gp_5m_v2.py
"""

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from factor_common import safe_join
from run_gp_mining_5m import (
    FEATURE_NAMES_5M,
    SEGMENTS,
    SYMBOLS,
    build_factor_table,
)

# ---- 政策与实验常量 ----
ROLL_WIN: int = 2016            # 滚动分位窗口：约一周的 5m bar
FEE_PER_SIDE: float = 0.0002    # maker 单边费率，往返 0.04% 与 v1 对齐
MIN_HOLD_MINUTES: float = 110.0  # 预注册判据：平均持仓时长下限
N_SEEDS_NEEDED: int = 2         # 预注册判据：至少几个种子通过
SEEDS: tuple[int, ...] = (123, 7, 2026)
Q_ENTRIES: tuple[int, ...] = (70, 80, 90)       # 入场分位（百分位）
Q_EXITS: tuple[int, ...] = (30, 40, 50)         # 出场分位（百分位）
VOL_FILTERS: dict[str, float | None] = {        # 波动过滤 -> vola20 滚动分位下限
    "none": None,
    "p50": 0.50,
    "p70": 0.70,
}
OUTPUT_DIR: str = "gp_5m_v2"
# 对账容差：gplearn 的 __str__ 用 '%.3f' 打印常数（见 gplearn/_program.py），
# 含常数项的公式（种子 7 的 -0.009）无法逐位还原，实测最大偏差 ~2e-07；
# 无常数公式（种子 123/2026）可对到 1e-9。取 1e-6（约为尾部价差的 1.5%）。
VERIFY_TOL: float = 1e-6

# 类型别名：逐币数组面板 / 一组政策参数
CoinPanel = dict[str, dict[str, np.ndarray]]
Combo = tuple[int, int, str]

# gplearn 前缀公式的记号：函数名 / 特征占位符 / 数值常量（含负号）
_FORMULA_TOKEN_RE = re.compile(
    r"add|sub|mul|div|abs|X\d+|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
)


def load_seed_formulas() -> dict[int, tuple[str, dict[str, float]]]:
    """从 v1 结果 JSON 读回三个种子的公式与三段尾部价差（用于对账）。"""
    loaded: dict[int, tuple[str, dict[str, float]]] = {}
    for seed in SEEDS:
        path: Path = safe_join(f"gp_5m/mining_result_s{seed}.json")
        payload: dict = json.loads(path.read_text(encoding="utf-8"))
        best: dict = payload["candidates"][0]
        spreads: dict[str, float] = {
            "train": best["train_spread"],
            "valid": best["valid_spread"],
            "test": best["test_spread"],
        }
        loaded[seed] = (best["formula_gp"], spreads)
    return loaded


def _apply_binary(name: str, left: np.ndarray | float,
                  right: np.ndarray | float) -> np.ndarray | float:
    """gplearn 二元算子的 numpy 复刻（除法必须用其受保护语义）。"""
    if name == "add":
        return np.add(left, right)
    if name == "sub":
        return np.subtract(left, right)
    if name == "mul":
        return np.multiply(left, right)
    # gplearn：|分母| <= 1e-3 时返回 1.0，不复刻会导致对账失败
    return np.where(np.abs(right) > 1e-3, np.divide(left, right), 1.0)


def _parse_formula_node(tokens: list[str], pos: int,
                        features: dict[str, np.ndarray]) -> tuple[np.ndarray | float, int]:
    """递归下降解析前缀公式，返回 (数值, 下一个未消费记号位置)。"""
    name: str = tokens[pos]
    if name in ("add", "sub", "mul", "div"):
        left: np.ndarray | float
        right: np.ndarray | float
        left, pos = _parse_formula_node(tokens, pos + 1, features)
        right, pos = _parse_formula_node(tokens, pos, features)
        return _apply_binary(name, left, right), pos
    if name == "abs":
        operand, pos = _parse_formula_node(tokens, pos + 1, features)
        return np.abs(operand), pos
    if name.startswith("X"):
        return features[name], pos + 1
    return float(name), pos + 1


def evaluate_formula(formula: str, features: dict[str, np.ndarray]) -> np.ndarray:
    """gplearn 前缀公式 -> 全历史因子值数组（逐行求值，与段划分无关）。"""
    tokens: list[str] = _FORMULA_TOKEN_RE.findall(formula)
    cleaned: str = re.sub(r"[\s,()]", "", formula)
    if "".join(tokens) != cleaned:
        raise ValueError(f"公式含无法识别的记号: {formula}")
    value, pos = _parse_formula_node(tokens, 0, features)
    if pos != len(tokens) or not isinstance(value, np.ndarray):
        raise ValueError(f"公式解析不完整或退化为常数: {formula}")
    return value


def prepare_panel() -> CoinPanel:
    """复用 v1 的面板构建（同重采样/特征/清洗口径），拆成逐币数组。"""
    panel: pl.DataFrame = build_factor_table()
    coins: CoinPanel = {}
    for symbol in SYMBOLS:
        sub: pl.DataFrame = panel.filter(pl.col("symbol") == symbol)
        coins[symbol] = {
            "datetime": sub["datetime"].to_numpy().astype("datetime64[ns]"),
            "X": sub.select(FEATURE_NAMES_5M).to_numpy().astype(np.float64),
            "label": sub["label"].to_numpy().astype(np.float64),
            "vola20": sub["vola20"].to_numpy().astype(np.float64),
        }
        print(f"    {symbol}: {sub.height} 行")
    return coins


def coin_factor(coins: CoinPanel, formula: str) -> dict[str, np.ndarray]:
    """逐币复算 v1 公式的全历史因子值（X0..X9 与 v1 的占位符顺序一致）。"""
    result: dict[str, np.ndarray] = {}
    for symbol, coin in coins.items():
        feature_map: dict[str, np.ndarray] = {
            f"X{i}": coin["X"][:, i] for i in range(len(FEATURE_NAMES_5M))
        }
        result[symbol] = evaluate_formula(formula, feature_map)
    return result


def rolling_score(values: np.ndarray, window: int) -> np.ndarray:
    """滚动百分位 -> 符号分 2*(百分位-0.5)；窗口不满时 NaN（策略保持空仓）。"""
    series: pd.Series = pd.Series(values)
    pct: pd.Series = series.rolling(window, min_periods=window).rank(
        method="average", pct=True
    )
    return 2.0 * (pct.to_numpy() - 0.5)


def segment_mask(datetimes: np.ndarray, segment: tuple[str, str]) -> np.ndarray:
    """某段（含端点日全天）的行掩码：[start 00:00, end+1day) 左闭右开。"""
    left: np.datetime64 = np.datetime64(f"{segment[0]}T00:00:00", "ns")
    right: np.datetime64 = np.datetime64(f"{segment[1]}T00:00:00", "ns") + np.timedelta64(1, "D")
    return (datetimes >= left) & (datetimes < right)


def v1_tail_spread_one(pred: np.ndarray, target: np.ndarray) -> float:
    """单币 v1 口径尾部价差：段内全局分位前/后 20% 标签均值之差。"""
    thr_hi: float = float(np.quantile(pred, 0.80))
    thr_lo: float = float(np.quantile(pred, 0.20))
    top: np.ndarray = target[pred >= thr_hi]
    bottom: np.ndarray = target[pred <= thr_lo]
    return float(top.mean() - bottom.mean())


def verify_against_v1(coins: CoinPanel, factor: dict[str, np.ndarray],
                      v1_spreads: dict[str, float]) -> dict[str, float]:
    """对账：复算三段 v1 尾部价差并与 JSON 记录比对，超容差直接报错。"""
    recomputed: dict[str, float] = {}
    for seg_name, seg_bounds in SEGMENTS.items():
        spreads: list[float] = []
        for symbol, coin in coins.items():
            mask: np.ndarray = segment_mask(coin["datetime"], seg_bounds)
            spreads.append(
                v1_tail_spread_one(factor[symbol][mask], coin["label"][mask])
            )
        recomputed[seg_name] = float(np.mean(spreads))
        diff: float = abs(recomputed[seg_name] - v1_spreads[seg_name])
        if diff > VERIFY_TOL:
            raise RuntimeError(
                f"v1 对账失败：{seg_name} 复算 {recomputed[seg_name]:.3e} "
                f"vs 记录 {v1_spreads[seg_name]:.3e}（|diff|={diff:.3e}）"
            )
    return recomputed


def gated_state(entry_event: np.ndarray, continue_cond: np.ndarray) -> np.ndarray:
    """向量化施密特触发器：state[t] = (state[t-1] | entry[t]) & cont[t]。

    段内重启的 OR：最近一次 cont 为 False 的位置之后，entry 只要出现过
    一次即置位；用前缀和 + 最近重置位置索引实现，严格等价于逐 bar 递推。
    """
    size: int = entry_event.size
    idx: np.ndarray = np.arange(size)
    reset_idx: np.ndarray = np.where(~continue_cond, idx, -1)
    last_reset: np.ndarray = np.maximum.accumulate(reset_idx)
    prefix: np.ndarray = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(entry_event.astype(np.int64)))
    )
    entered_since: np.ndarray = (prefix[idx + 1] - prefix[last_reset + 1]) > 0
    return entered_since & continue_cond


def simulate_coin(score: np.ndarray, vol_pct: np.ndarray, label: np.ndarray,
                  qe_thr: float, qx_thr: float,
                  vol_min: float | None) -> dict[str, object]:
    """单币迟滞政策模拟（一段内从空仓开始），返回统计与逐 bar 净损益。"""
    volpass: np.ndarray = (
        np.ones(score.size, dtype=bool) if vol_min is None else vol_pct > vol_min
    )
    long_state: np.ndarray = gated_state((score >= qe_thr) & volpass, score >= qx_thr)
    short_state: np.ndarray = gated_state((score <= -qe_thr) & volpass, score <= -qx_thr)
    position: np.ndarray = long_state.astype(np.int8) - short_state.astype(np.int8)
    changes: np.ndarray = np.diff(position, prepend=np.int8(0))
    fees: np.ndarray = FEE_PER_SIDE * np.abs(changes).astype(np.float64)
    gross: np.ndarray = position.astype(np.float64) * label
    pnl: np.ndarray = gross - fees
    prev: np.ndarray = np.concatenate((np.zeros(1, dtype=np.int8), position[:-1]))
    new_episode: np.ndarray = (position != 0) & (
        (prev == 0) | (np.sign(position) != np.sign(prev))
    )
    holding_bars: int = int(np.count_nonzero(position))
    episodes: int = int(np.count_nonzero(new_episode))
    return {
        "pnl": pnl,
        "net": float(pnl.sum()),
        "gross": float(gross.sum()),
        "holding_bars": holding_bars,
        "episodes": episodes,
        "trades": int(np.abs(changes).sum()),
        "bars": int(score.size),
    }


def aggregate_stats(per_coin: list[dict[str, object]]) -> dict[str, object]:
    """5 币等权汇总：净/毛收益（%）、平均持仓（分钟）、开仓与成交次数。"""
    n_coins: int = max(len(per_coin), 1)
    net_pct: float = 100.0 * sum(float(x["net"]) for x in per_coin) / n_coins
    gross_pct: float = 100.0 * sum(float(x["gross"]) for x in per_coin) / n_coins
    holding_bars: int = sum(int(x["holding_bars"]) for x in per_coin)
    episodes: int = sum(int(x["episodes"]) for x in per_coin)
    avg_hold: float = 5.0 * holding_bars / episodes if episodes else 0.0
    return {
        "net_pct": round(net_pct, 4),
        "gross_pct": round(gross_pct, 4),
        "avg_hold_min": round(avg_hold, 1),
        "episodes": episodes,
        "trades": sum(int(x["trades"]) for x in per_coin),
        "bars": max(int(x["bars"]) for x in per_coin),
        "in_market_ratio": round(holding_bars / max(sum(int(x["bars"]) for x in per_coin), 1), 4),
    }


def q_threshold(percentile: int) -> float:
    """百分位 -> 符号分阈值：70 分位 -> +0.4，30 分位 -> -0.4，50 分位 -> 0。"""
    return 2.0 * (percentile / 100.0 - 0.5)


def evaluate_segment(coins: CoinPanel, scores: dict[str, np.ndarray],
                     vol_scores: dict[str, np.ndarray], seg_name: str,
                     combo: Combo) -> dict[str, object]:
    """某段某组参数下的 5 币等权模拟汇总。"""
    qe: int
    qx: int
    vol_name: str
    qe, qx, vol_name = combo
    seg_bounds: tuple[str, str] = SEGMENTS[seg_name]
    per_coin: list[dict[str, object]] = []
    for symbol, coin in coins.items():
        mask: np.ndarray = segment_mask(coin["datetime"], seg_bounds)
        per_coin.append(simulate_coin(
            scores[symbol][mask], vol_scores[symbol][mask], coin["label"][mask],
            q_threshold(qe), q_threshold(qx), VOL_FILTERS[vol_name],
        ))
    return aggregate_stats(per_coin)


def is_better(stats: dict[str, object], best: dict[str, object] | None) -> bool:
    """验证段选参规则：净收益更大者优先；并列时取平均持仓更长者。"""
    if best is None:
        return True
    if stats["net_pct"] != best["net_pct"]:
        return bool(stats["net_pct"] > best["net_pct"])
    return bool(stats["avg_hold_min"] > best["avg_hold_min"])


def grid_search(coins: CoinPanel, scores: dict[str, np.ndarray],
                vol_scores: dict[str, np.ndarray]) -> tuple[Combo, dict[str, object], int]:
    """验证段参数网格：3x3x3 共 27 组，只看 2025 全年（冻结前）。

    第三个返回值是净收益 > 0 的组合数：若为 0，说明"最优"只是
    27 组里亏得最少的，选参本身已无经济意义（报告需如实披露）。
    """
    best_combo: Combo | None = None
    best_stats: dict[str, object] | None = None
    positive_combos: int = 0
    for qe in Q_ENTRIES:
        for qx in Q_EXITS:
            for vol_name in VOL_FILTERS:
                combo: Combo = (qe, qx, vol_name)
                stats: dict[str, object] = evaluate_segment(
                    coins, scores, vol_scores, "valid", combo
                )
                if float(stats["net_pct"]) > 0:
                    positive_combos += 1
                if is_better(stats, best_stats):
                    best_combo, best_stats = combo, stats
    assert best_combo is not None and best_stats is not None
    return best_combo, best_stats, positive_combos


def monthly_net(coins: CoinPanel, scores: dict[str, np.ndarray],
                vol_scores: dict[str, np.ndarray], combo: Combo) -> dict[str, float]:
    """测试段（2026）逐月净收益（%）：用冻结参数重新模拟后按月切分。"""
    qe: int
    qx: int
    vol_name: str
    qe, qx, vol_name = combo
    totals: dict[str, float] = {}
    for symbol, coin in coins.items():
        mask: np.ndarray = segment_mask(coin["datetime"], SEGMENTS["test"])
        sim: dict[str, object] = simulate_coin(
            scores[symbol][mask], vol_scores[symbol][mask], coin["label"][mask],
            q_threshold(qe), q_threshold(qx), VOL_FILTERS[vol_name],
        )
        months: np.ndarray = coin["datetime"][mask].astype("datetime64[M]").astype(str)
        pnl: np.ndarray = sim["pnl"]  # type: ignore[assignment]
        for month in np.unique(months):
            totals[month] = totals.get(month, 0.0) + float(pnl[months == month].sum())
    n_coins: int = max(len(coins), 1)
    return {month: round(100.0 * value / n_coins, 4)
            for month, value in sorted(totals.items())}


def score_autocorr(scores: dict[str, np.ndarray], lag: int = 1) -> float:
    """符号分的滞后自相关（5 币合并、剔除窗口预热 NaN）：量化信号逐 bar 持续性。

    若接近 0，说明归一分位信号逐 bar 近乎独立：出场分位阈值 Q_exit 对应
    的"每 bar 存续概率"约 = 1 - Q_exit/100，平均持仓被钳在
    100/(Q_exit) 根 bar 附近（Q_exit=30 -> ~3.3 根 ≈ 17 分钟），迟滞带
    拉不长持仓，这是低换手失败的机理证据。
    """
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    for values in scores.values():
        clean: np.ndarray = values[np.isfinite(values)]
        left.append(clean[:-lag])
        right.append(clean[lag:])
    merged_left: np.ndarray = np.concatenate(left)
    merged_right: np.ndarray = np.concatenate(right)
    return float(np.corrcoef(merged_left, merged_right)[0, 1])


def run_seed(seed: int, formula: str, v1_spreads: dict[str, float],
             coins: CoinPanel, vol_scores: dict[str, np.ndarray]) -> dict[str, object]:
    """单种子完整流程：对账 -> 归一 -> 验证段选参 -> 冻结 -> 测试段评一次。"""
    factor: dict[str, np.ndarray] = coin_factor(coins, formula)
    verified: dict[str, float] = verify_against_v1(coins, factor, v1_spreads)
    scores: dict[str, np.ndarray] = {
        symbol: rolling_score(values, ROLL_WIN) for symbol, values in factor.items()
    }
    best_combo: Combo
    valid_stats: dict[str, object]
    positive_combos: int
    best_combo, valid_stats, positive_combos = grid_search(coins, scores, vol_scores)
    test_stats: dict[str, object] = evaluate_segment(coins, scores, vol_scores, "test", best_combo)
    monthly: dict[str, float] = monthly_net(coins, scores, vol_scores, best_combo)
    passed: bool = (
        float(test_stats["net_pct"]) > 0
        and float(test_stats["avg_hold_min"]) >= MIN_HOLD_MINUTES
    )
    return {
        "formula_gp": formula,
        "verify_v1_spread": {k: round(v, 10) for k, v in verified.items()},
        "score_lag1_autocorr": round(score_autocorr(scores), 4),
        "frozen_combo": {"q_entry": best_combo[0], "q_exit": best_combo[1],
                         "vol_filter": best_combo[2]},
        "valid_positive_combos": positive_combos,
        "valid": valid_stats,
        "test": test_stats,
        "monthly_net_2026": monthly,
        "positive_months": sum(1 for v in monthly.values() if v > 0),
        "criteria_pass": passed,
    }


def write_result(result: dict[str, object]) -> Path:
    """完整结果写入 research/gp_5m_v2/v2_result.json。"""
    out_dir: Path = safe_join(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path: Path = out_dir / "v2_result.json"
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    return out_path


def main() -> None:
    """完整 v2 评估流程：面板 -> 逐种子对账/选参/出样本 -> 判据 -> 落盘。"""
    started: float = time.perf_counter()
    print(f"[0] research 根目录: {safe_join('.')}")
    print("[1] 构建 5m 面板（复用 v1 口径）...")
    coins: CoinPanel = prepare_panel()
    print("[2] 波动率滚动分位（vola20, 窗口 2016）...")
    vol_scores: dict[str, np.ndarray] = {
        symbol: rolling_score(coin["vola20"], ROLL_WIN) for symbol, coin in coins.items()
    }
    formulas: dict[int, tuple[str, dict[str, float]]] = load_seed_formulas()
    seed_results: dict[str, dict[str, object]] = {}
    for seed in SEEDS:
        wall = time.perf_counter()
        formula, v1_spreads = formulas[seed]
        print(f"[3] 种子 {seed}: 复算公式 + v1 对账 + 验证段网格（27 组）...")
        seed_results[str(seed)] = run_seed(seed, formula, v1_spreads, coins, vol_scores)
        result: dict[str, object] = seed_results[str(seed)]
        valid: dict = result["valid"]  # type: ignore[assignment]
        test: dict = result["test"]  # type: ignore[assignment]
        combo: dict = result["frozen_combo"]  # type: ignore[assignment]
        print(f"    冻结参数 Q_entry={combo['q_entry']} Q_exit={combo['q_exit']} "
              f"波动过滤={combo['vol_filter']}")
        print(f"    验证段: 净 {valid['net_pct']:+.3f}% 持仓 {valid['avg_hold_min']:.0f}min "
              f"开仓 {valid['episodes']} 次")
        print(f"    测试段: 净 {test['net_pct']:+.3f}% 持仓 {test['avg_hold_min']:.0f}min "
              f"开仓 {test['episodes']} 次  净>0月 {result['positive_months']}/9  "
              f"判据{'通过' if result['criteria_pass'] else '未通过'}"
              f"（耗时 {time.perf_counter() - wall:.0f}s）")
    passed_seeds: list[str] = [
        seed for seed, item in seed_results.items() if item["criteria_pass"]
    ]
    achieved: bool = len(passed_seeds) >= N_SEEDS_NEEDED
    elapsed: float = time.perf_counter() - started
    print(f"[判据] 通过种子 {len(passed_seeds)}/{len(SEEDS)}（需 >= {N_SEEDS_NEEDED}）: "
          f"{'达成' if achieved else '未达成'}，总耗时 {elapsed:.0f}s")
    outcome: dict[str, object] = {
        "meta": {
            "roll_win": ROLL_WIN,
            "fee_per_side": FEE_PER_SIDE,
            "fee_roundtrip": 2 * FEE_PER_SIDE,
            "grid": {"q_entry": list(Q_ENTRIES), "q_exit": list(Q_EXITS),
                     "vol_filter": list(VOL_FILTERS)},
            "selection_rule": "验证段(2025)净收益最大，并列取平均持仓更长；冻结后测试段只评一次",
            "segments": {k: list(v) for k, v in SEGMENTS.items()},
            "criteria": f"测试段至少 {N_SEEDS_NEEDED} 个种子：净>0 且 平均持仓>={MIN_HOLD_MINUTES}min",
        },
        "seeds": seed_results,
        "verdict": {"passed_seeds": passed_seeds, "achieved": achieved,
                    "runtime_sec": round(elapsed, 1)},
    }
    out_path: Path = write_result(outcome)
    print(f"[输出] {out_path}")


if __name__ == "__main__":
    main()
