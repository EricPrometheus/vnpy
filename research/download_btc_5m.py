"""下载 BTC/USDT 5 分钟 K 线（OKX，一年量级）用于验证 Polymarket 5 分钟押注 bot 的动量假说。

数据源：OKX history-candles（免费，bar=5m，走本地代理）。
输出：research/crypto_data/btc_5m.parquet
"""

import os
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)

import polars as pl
import requests

PROXY: str = "http://127.0.0.1:10809"
START_TS_MS: int = int(datetime(2025, 8, 1).timestamp() * 1000)   # 一年数据（2025-08 ~ 2026-09）
API_URL: str = "https://www.okx.com/api/v5/market/history-candles"
OUT_DIR: Path = REPO_ROOT / "research" / "crypto_data"
OUT_DIR.mkdir(exist_ok=True)

SESSION = requests.Session()
SESSION.proxies = {"http": PROXY, "https": PROXY}
SESSION.headers.update({"User-Agent": "Mozilla/5.0"})

INST_ID: str = "BTC-USDT"
BAR: str = "5m"


def fetch_page(limit: int = 300, after: str | None = None) -> list:
    params: dict = {"instId": INST_ID, "bar": BAR, "limit": limit}
    if after:
        params["after"] = after
    resp = SESSION.get(API_URL, params=params, timeout=15)
    resp.raise_for_status()
    data: dict = resp.json()
    if data.get("code") != "0":
        raise ValueError(f"OKX 错误: {data}")
    return data["data"]


def main() -> None:
    rows_all: list[list] = []
    cursor: str | None = None
    batch: int = 0

    while True:
        rows = fetch_page(after=cursor)
        if not rows:
            break
        rows_all.extend(rows)
        # OKX 返回最新在前；cursor 往前翻页
        cursor = rows[-1][0]
        batch += 1
        if int(rows[-1][0]) <= START_TS_MS:
            break
        if batch % 50 == 0:
            print(f"已拉取 {len(rows_all)} 根，游标时间 {datetime.fromtimestamp(int(cursor)/1000)}")
        time.sleep(0.25)

    # 翻转正序 + 过滤起止
    rows_all.reverse()
    rows_all = [r for r in rows_all if int(r[0]) >= START_TS_MS]

    df = pl.DataFrame({
        "datetime": [datetime.fromtimestamp(int(r[0]) / 1000) for r in rows_all],
        "open": [float(r[1]) for r in rows_all],
        "high": [float(r[2]) for r in rows_all],
        "low": [float(r[3]) for r in rows_all],
        "close": [float(r[4]) for r in rows_all],
        "volume": [float(r[5]) for r in rows_all],
    })
    out: Path = OUT_DIR / "btc_5m.parquet"
    df.write_parquet(out)
    print(f"\n完成: {len(df)} 根 5 分钟K线, {df['datetime'].min()} ~ {df['datetime'].max()}")
    print(f"保存: {out}")


if __name__ == "__main__":
    main()