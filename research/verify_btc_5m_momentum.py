"""独立验证：Polymarket BTC 5分钟押注 bot 的"前3分钟动量→5分钟方向"假说。

复现 ratrimaa/btc-5m-polymarket-bot 的核心逻辑（README 公开的方法）：
- 5 分钟窗口内，前 3 分钟动量（mom3）预测整个 5 分钟方向（Up=末价>=起始价）
- 用 1 分钟数据精确重采样 5 分钟窗口，分阈值统计胜率
- 差异：样本量是其 996 窗口的 20 倍以上（半年 ≈ 5 万窗口）

用法：
    D:\\veighna_studio\\python.exe -X utf8 research/verify_btc_5m_momentum.py
"""

import os
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)

import polars as pl

DATA: Path = REPO_ROOT / "research" / "crypto_data" / "btc_1m.parquet"

# ratrimaa 的阈值分组
THRESHOLDS: list[tuple[str, float, str]] = [
    ("强涨  > +0.10%", 0.0010, "up"),
    ("中涨  > +0.05%", 0.0005, "up"),
    ("中跌  < -0.05%", -0.0005, "down"),
    ("强跌  < -0.10%", -0.0010, "down"),
]


def build_windows(df: pl.DataFrame) -> pl.DataFrame:
    """1 分钟数据重采样为 5 分钟窗口，取出 起点价/第3分钟价/终点价。

    用分钟序号（0..4）提取每窗口第 0/2/4 根 1m 的价格，
    不依赖聚合 nth（polars 新版已改名）。
    """
    df = df.sort("datetime").with_columns([
        (pl.col("datetime").dt.epoch("s") // 300).alias("win_id"),
        (pl.col("datetime").dt.epoch("s") % 300 // 60).alias("min_in_win"),
    ])

    p0 = df.filter(pl.col("min_in_win") == 0).select(["win_id", "open"]).rename({"open": "p0"})
    p3 = df.filter(pl.col("min_in_win") == 2).select(["win_id", "close"]).rename({"close": "p3m"})
    p5 = df.filter(pl.col("min_in_win") == 4).select(["win_id", "close"]).rename({"close": "p5m"})

    win = p0.join(p3, on="win_id").join(p5, on="win_id")

    return win.with_columns([
        ((pl.col("p3m") / pl.col("p0") - 1) * 10000).alias("mom3_bp"),  # 动量，单位 0.01% (bp)
        ((pl.col("p5m") / pl.col("p0") - 1) * 10000).alias("ret5_bp"),
        (pl.col("p5m") >= pl.col("p0")).alias("is_up"),
    ])


def main() -> None:
    df: pl.DataFrame = pl.read_parquet(DATA)
    win = build_windows(df)
    total = win.height
    print(f"1分钟数据: {df.height} 根 | 5分钟窗口: {total} 个\n")

    # ---- 1. 全样本基线 ----
    up_ratio: float = win["is_up"].mean()
    print(f"全样本: Up 占比 {up_ratio:.4f}（随机≈0.5）\n")

    # ---- 2. 复现 ratrimaa 阈值分组 ----
    print("=" * 66)
    print(f"{'分组(前3分钟动量)':<18}{'窗口数':>8}{'Up占比':>10}{'Down占比':>10}{'方向':>6}")
    print("-" * 66)
    for label, thr, side in THRESHOLDS:
        sub = win.filter(pl.col("mom3_bp") * (1 if side == "up" else -1) > abs(thr) * 10000)
        n = sub.height
        up = sub["is_up"].mean() if n else float("nan")
        down = 1 - up
        direction = "Up" if up > 0.5 else "Down"
        print(f"{label:<18}{n:>8}{up:>10.4f}{down:>10.4f}{direction:>6}")
    flat = win.filter(pl.col("mom3_bp").abs() <= 5)
    print(f"{'平盘 ±0.05%':<18}{flat.height:>8}{flat['is_up'].mean():>10.4f}{1-flat['is_up'].mean():>10.4f}")
    print("=" * 66)

    # ---- 3. 分月稳定性（用强动量 >0.05%）----
    print("\n=== 分月稳定性（前3分钟 |动量|>0.05%，预测 Up/Down 的胜率） ===")
    strong = win.filter(pl.col("mom3_bp").abs() > 5).with_columns(
        pl.col("win_id").cast(pl.Utf8)
    )
    # 还原日期用于分月
    df_win = win.with_columns(
        pl.from_epoch(pl.col("win_id") * 300).alias("dt")
    )
    strong2 = df_win.filter(pl.col("mom3_bp").abs() > 5).with_columns([
        pl.col("dt").dt.strftime("%Y-%m").alias("month"),
        (pl.col("mom3_bp") > 0).alias("predict_up"),
        (pl.col("is_up") == (pl.col("mom3_bp") > 0)).alias("hit"),
    ])
    monthly = strong2.group_by("month").agg([
        pl.len().alias("n"),
        pl.col("hit").mean().alias("acc"),
    ]).sort("month")
    print(monthly)
    acc = monthly["acc"]
    print(f"\n月度准确率: 平均 {acc.mean():.4f} | 最低 {acc.min():.4f} | 最高 {acc.max():.4f}")

    # ---- 4. Polymarket 期望（赔率50/50、无手续费下注1单位）----
    print("\n=== Polymarket 二元下注期望（败赔1/胜赔1，下注1单位） ===")
    for label, thr, side in THRESHOLDS:
        sub = df_win.filter(pl.col("mom3_bp") * (1 if side == "up" else -1) > abs(thr) * 10000)
        n = sub.height
        if n < 50:
            continue
        winrate = sub["is_up"].mean() if side == "up" else 1 - sub["is_up"].mean()
        ev = 2 * winrate - 1                       # 每次下注期望
        daily_windows = 288                        # 每天 288 个 5 分钟窗口
        est_daily = ev * min(n, 50) * daily_windows / n if n else 0  # 保守：每天最多 50 次
        print(f"{label:<18} 预测胜率 {winrate:.4f} | 每注期望 {ev:+.4f} (盈亏平衡 0.5) | 窗口数 {n}")
    print("\n注: 未计 Polymarket 买卖点差(通常 1-3 美分/刀)与滑点，真实收益会低于此")


if __name__ == "__main__":
    main()