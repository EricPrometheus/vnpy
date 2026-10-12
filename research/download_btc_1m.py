"""下载 BTC/USDT 1 分钟 K 线（OKX，半年）用于精确复现 5 分钟窗口动量因子。

为什么需要 1 分钟：验证 polymarket bot 的"前3分钟动量→5分钟方向"假说，
5 分钟 K 线拿不到窗口内第 3 分钟的价格，1 分钟数据可精确重采样。

用法：
    D:\\veighna_studio\\python.exe -X utf8 research/download_btc_1m.py
"""

import os
import time
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)

import polars as pl
import requests

PROXY: str = "http://127.0.0.1:10809"
DAYS: int = 180                                     # 半年
START_TS_MS: int = int(
    (datetime.now() - timedelta(days=DAYS)).timestamp() * 1000
)
API_URL: str = "https://www.okx.com/api/v5/market/history-candles"
OUT_DIR: Path = REPO_ROOT / "research" / "crypto_data"
OUT_DIR.mkdir(exist_ok=True)

SESSION = requests.Session()
SESSION.proxies = {"http": PROXY, "https": PROXY}
SESSION.headers.update({"User-Agent": "Mozilla/5.0"})

INST_ID: str = "BTC-USDT"
BAR: str = "1m"


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
        cursor = rows[-1][0]
        batch += 1
        if int(rows[-1][0]) <= START_TS_MS:
            break
        if batch % 100 == 0:
            print(f"已拉取 {len(rows_all)} 根, 游标 {datetime.fromtimestamp(int(cursor)/1000)}")
        time.sleep(0.2)

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
    out: Path = OUT_DIR / "btc_1m.parquet"
    df.write_parquet(out)
    print(f"\n完成: {len(df)} 根 1分钟K线, {df['datetime'].min()} ~ {df['datetime'].max()}")
    print(f"保存: {out}")


if __name__ == "__main__":
    main()