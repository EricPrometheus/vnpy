"""下载主流加密货币日线（OKX 现货）到 vnpy AlphaLab。

数据源：OKX 公共接口（免费、无需 key），走本地代理 10809（境外域名）。
输出：research/alpha_lab/daily/{COIN}USDT.OKX.parquet，每币一个文件。

用法：
    D:\\veighna_studio\\python.exe -X utf8 research/download_crypto_daily.py
"""

import os
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)

import requests

# 说明：vnpy 官方 Exchange 枚举没有加密货币交易所；Python 3.13 对 Enum
# 运行时扩展封得很死（extend_enum/__new__ 注入均失效）。由于 AlphaLab
# 的 save_bar_data 只用 exchange.value 拼文件名，而 BarData 是 dataclass
# 不校验类型，这里用一个带 .value 的占位对象即可绕开枚举（零风险）。
class CryptoExchange:
    """加密货币交易所占位（仅提供 .value，供 alpha_lab 拼 vt_symbol）。"""

    def __init__(self, name: str) -> None:
        self.value: str = name

from vnpy.alpha.lab import AlphaLab
from vnpy.trader.constant import Interval
from vnpy.trader.object import BarData

# ---- 配置 ----
PROXY: str = "http://127.0.0.1:10809"     # 本地代理（OKX 境外域名必须走代理）
START_TS_MS: int = int(datetime(2019, 1, 1).timestamp() * 1000)   # 从 2019-01-01 起

# 主流币（USDT 现货，覆盖不同上市时间）
COINS: list[str] = [
    "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "TRX",
    "LINK", "AVAX", "DOT", "LTC", "BCH", "UNI", "ATOM", "XLM",
    "NEAR", "APT", "ARB", "OP", "SUI", "PEPE",
]

OKX_BASE: str = "https://www.okx.com/api/v5/market/history-candles"
API_URL: str = "https://www.okx.com/api/v5/market/history-candles"
LAB_PATH: Path = REPO_ROOT / "research" / "alpha_lab"

SESSION: requests.Session = requests.Session()
SESSION.proxies = {"http": PROXY, "https": PROXY}
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

# enum.extend_enum 给 Exchange 注入 OKX 成员（旧 __new__ 黑科技在 3.12+ 已失效）。


def fetch_kline_page(inst_id: str, bar: str, limit: int = 300, after: str | None = None) -> list:
    """拉一页 K 线（OKX 返回最新在前）。"""
    params: dict = {"instId": inst_id, "bar": bar, "limit": limit}
    if after:
        params["after"] = after
    resp = SESSION.get(API_URL, params=params, timeout=15)
    resp.raise_for_status()
    data: dict = resp.json()
    if data.get("code") != "0":
        raise ValueError(f"OKX 错误: {data}")
    return data["data"]


def fetch_kline_history(inst_id: str, bar: str, start_ms: int) -> list[dict]:
    """分页拉取 start_ms 至今的全部 K 线（OKX 上限 300/次）。"""
    all_rows: list[dict] = []
    cursor: str | None = None
    while True:
        rows = fetch_kline_page(inst_id, bar, after=cursor)
        if not rows:
            break
        all_rows.extend(rows)
        # 最新时间戳作为下次分页游标（往前翻）
        cursor = rows[-1][0]
        if int(rows[-1][0]) <= start_ms:
            break
        time.sleep(0.25)  # 温和限速
    return all_rows


def to_bar(row: list, coin: str, exchange: CryptoExchange) -> BarData:
    """OKX 行 -> BarData。字段: [ts,o,h,l,c,vol,volCcy,volCcyQuote,confirm]"""
    dt: datetime = datetime.fromtimestamp(int(row[0]) / 1000, tz=timezone.utc)
    # AlphaLab 存 naive（内部会 replace(tzinfo=None)），统一用 UTC 时间
    dt = dt.replace(tzinfo=None)
    bar = BarData(
        symbol=f"{coin}USDT",
        exchange=exchange,
        datetime=dt,
        gateway_name="DB",
        interval=Interval.DAILY,
        open_price=float(row[1]),
        high_price=float(row[2]),
        low_price=float(row[3]),
        close_price=float(row[4]),
        volume=float(row[5]),
        turnover=float(row[6]),
    )
    return bar


def download_one(coin: str, lab: AlphaLab) -> tuple[str, int]:
    """下载单个币日线并入库，返回 (币名, 根数)。"""
    exchange: CryptoExchange = CryptoExchange("OKX")

    inst_id: str = f"{coin}-USDT"
    rows = fetch_kline_history(inst_id, "1D", START_TS_MS)
    # OKX 返回最新在前，翻转为时间正序
    rows.reverse()
    rows = [r for r in rows if int(r[0]) >= START_TS_MS]

    bars: list[BarData] = [to_bar(r, coin, exchange) for r in rows]
    if bars:
        lab.save_bar_data(bars)
    return coin, len(bars)


def main() -> None:
    """批量下载全部币种。"""
    lab = AlphaLab(str(LAB_PATH))
    print(f"下载 {len(COINS)} 个币种日线（OKX 现货，2019 至今）...")

    for coin in COINS:
        try:
            coin, n = download_one(coin, lab)
            print(f"  {coin}: {n} 根")
        except Exception as e:
            print(f"  {coin}: 失败 - {str(e)[:80]}")

    print("\n完成。数据目录:", LAB_PATH / "daily")


if __name__ == "__main__":
    main()