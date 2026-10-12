"""验证 Polymarket BTC 5 分钟市场是否"错价"（对强动量信号不定价）。

方法：
1. 用 OKX 1 分钟数据找出"前3分钟强动量(|mom3|>0.05%)"的 5 分钟窗口
2. 窗口时间戳(win_id*300) 直接映射 Polymarket event slug btc-updown-5m-<ts>
3. 拉该市场在信号时刻(第165~180秒)的 Up token 价格（CLOB prices-history）
4. 统计定价分布 + 按 Polymarket 实际赔率重算期望收益

用法（后台跑，约 3-5 分钟）：
    D:\\veighna_studio\\python.exe -X utf8 research/verify_polymarket_pricing.py
"""

import os
import random
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)

import polars as pl
import requests

DATA: Path = REPO_ROOT / "research" / "crypto_data" / "btc_1m.parquet"
OUT: Path = REPO_ROOT / "research" / "crypto_data" / "pm_pricing.parquet"

PROXY: str = "http://127.0.0.1:10809"
GAMMA: str = "https://gamma-api.polymarket.com"
CLOB: str = "https://clob.polymarket.com"

SAMPLE_N: int = 300          # 抽样窗口数（API 限速下的平衡）
MOM_THR_BP: float = 5.0      # 强动量阈值 0.05% = 5bp
SIGNAL_START: int = 165      # 信号时刻：窗口开始后 165~180 秒
SIGNAL_END: int = 180

SESSION = requests.Session()
SESSION.proxies = {"http": PROXY, "https": PROXY}
SESSION.headers.update({"User-Agent": "Mozilla/5.0"})

random.seed(42)  # 可复现抽样


def build_windows(df: pl.DataFrame) -> pl.DataFrame:
    """复用动量验证的窗口构建（OKX 1m -> 5 分钟窗口）。"""
    df = df.sort("datetime").with_columns([
        (pl.col("datetime").dt.epoch("s") // 300).alias("win_id"),
        (pl.col("datetime").dt.epoch("s") % 300 // 60).alias("min_in_win"),
    ])
    p0 = df.filter(pl.col("min_in_win") == 0).select(["win_id", "open"]).rename({"open": "p0"})
    p3 = df.filter(pl.col("min_in_win") == 2).select(["win_id", "close"]).rename({"close": "p3m"})
    p5 = df.filter(pl.col("min_in_win") == 4).select(["win_id", "close"]).rename({"close": "p5m"})
    win = p0.join(p3, on="win_id").join(p5, on="win_id")
    return win.with_columns([
        ((pl.col("p3m") / pl.col("p0") - 1) * 10000).alias("mom3_bp"),
        (pl.col("p5m") >= pl.col("p0")).alias("is_up"),
    ])


def get_up_token(event_slug: str) -> str | None:
    """从 gamma API 拿该 event 的 Up token id。"""
    resp = SESSION.get(f"{GAMMA}/events", params={"slug": event_slug}, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return None
    markets = data[0].get("markets", [])
    if not markets:
        return None
    tokens: list[str] = markets[0].get("clobTokenIds", [])
    outcomes: list[str] = markets[0].get("outcomes", [])
    if len(tokens) < 2 or "Up" not in outcomes:
        return None
    return tokens[outcomes.index("Up")]


def get_pm_price(token_id: str, start_ts: int, end_ts: int) -> float | None:
    """拉 CLOB prices-history 在信号时刻的 Up 价格（取区间末值=最新）。"""
    resp = SESSION.get(f"{CLOB}/prices-history", params={
        "market": token_id,
        "startTs": start_ts,
        "endTs": end_ts,
        "fidelity": 1,
    }, timeout=15)
    resp.raise_for_status()
    hist: list = resp.json().get("history", [])
    if not hist:
        return None
    return float(hist[-1]["p"])


def main() -> None:
    df = pl.read_parquet(DATA)
    win = build_windows(df)

    # 强动量窗口（含随机抽样）
    strong = win.filter(pl.col("mom3_bp").abs() > MOM_THR_BP)
    print(f"强动量窗口总数 {strong.height}，抽样 {SAMPLE_N} 个验证 Polymarket 定价")

    idx = list(range(strong.height))
    random.shuffle(idx)
    picked = strong[idx[:SAMPLE_N]]

    records: list[dict] = []
    t0 = time.time()

    for i, row in enumerate(picked.iter_rows(named=True), 1):
        ts: int = int(row["win_id"]) * 300
        slug: str = f"btc-updown-5m-{ts}"
        try:
            token = get_up_token(slug)
            if not token:
                continue
            price = get_pm_price(token, ts + SIGNAL_START, ts + SIGNAL_END)
            if price is None:
                continue
            records.append({
                "ts": ts,
                "mom3_bp": row["mom3_bp"],
                "is_up": row["is_up"],          # OKX 现货判定
                "pm_up_price": price,           # 信号时刻 Up token 价格
            })
        except Exception as e:
            print(f"  跳过 {slug}: {str(e)[:50]}")
        if i % 30 == 0:
            print(f"  进度 {i}/{SAMPLE_N} ({time.time() - t0:.0f}s), 有效 {len(records)}")
        time.sleep(0.3)

    out_df = pl.DataFrame(records)
    out_df.write_parquet(OUT)
    print(f"\n有效样本 {out_df.height} 个，已保存 {OUT}")

    # ---- 统计 ----
    if out_df.height < 50:
        print("样本不足，请检查 API 返回")
        return

    # 1) 信号时刻 Polymarket 定价分布（对"看涨信号"，Up 价格应 >0.5 若市场理性定价）
    mom_up = out_df.filter(pl.col("mom3_bp") > 0)
    mom_down = out_df.filter(pl.col("mom3_bp") < 0)
    print("\n=== 强动量时刻的 Polymarket Up-token 定价 ===")
    print(f"看涨信号(mom3>0): Up价 均值 {mom_up['pm_up_price'].mean():.3f} | "
          f"中位 {mom_up['pm_up_price'].median():.3f} | P25 {mom_up['pm_up_price'].quantile(0.25):.3f} | "
          f"P75 {mom_up['pm_up_price'].quantile(0.75):.3f}")
    print(f"看跌信号(mom3<0): Up价 均值 {mom_down['pm_up_price'].mean():.3f} | "
          f"中位 {mom_down['pm_up_price'].median():.3f}")

    # 2) 实际命中率（OKX 判定 vs 押注方向）
    hit_up = (mom_up["is_up"]).mean()
    hit_down = (~mom_down["is_up"]).mean()
    print(f"\nOKX 口径命中率: 看涨 {hit_up:.3f} | 看跌 {hit_down:.3f}")

    # 3) 按 Polymarket 实际价格买入的期望收益
    #    押 Up@p: E = hit*(1-p) - (1-hit)*p
    ev_up_rows = mom_up.with_columns(
        (pl.when(pl.col("is_up")).then(1 - pl.col("pm_up_price"))
         .otherwise(-pl.col("pm_up_price"))).alias("ev")
    )
    #    押 Down@q（q=1-p）: E = (1-hit)*... Down 命中 = not is_up
    ev_down_rows = mom_down.with_columns(
        (pl.when(~pl.col("is_up")).then(pl.col("pm_up_price"))
         .otherwise(-(1 - pl.col("pm_up_price")))).alias("ev")
    )
    print("\n=== 按 Polymarket 实际定价下注的每注期望 ===")
    print(f"看涨押Up:  {ev_up_rows['ev'].mean():+.4f}/注 (n={ev_up_rows.height})")
    print(f"看跌押Down: {ev_down_rows['ev'].mean():+.4f}/注 (n={ev_down_rows.height})")
    total = pl.concat([ev_up_rows, ev_down_rows])
    print(f"全部信号:  {total['ev'].mean():+.4f}/注 (n={total.height})")
    win_rate = (total["ev"] > 0).mean()
    print(f"单注盈利占比: {win_rate:.3f}")
    print("\n注: 未扣点差/滑点；结算基准为 Chainlink TWAP 与 OKX 现货略有差异")


if __name__ == "__main__":
    main()