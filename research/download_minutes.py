"""下载主流币分钟级 K 线（OKX 现货）到 AlphaLab 分钟目录。支持断点续传。

数据源：OKX 公共接口 history-candles（免费、无需 key）。
网络策略：先直连，连接失败自动回退本地代理（环境变量 OKX_PROXY 可覆盖）；
连接重置类故障用长退避重试（5s 起指数退避，上限 120s）。
断点续传：每 120 页把已拉取数据覆盖写入目标 parquet；重启时从文件里
最早的时间戳继续往前拉，中途崩溃不丢进度。

用法（在仓库根目录运行）：
    D:\\veighna_studio\\python.exe -X utf8 research/download_minutes.py --bar 1m --days 1095 --coins BTC
"""

import argparse
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import requests

MINUTE_DIR: Path = Path("research/alpha_lab/minute")
API_URL: str = "https://www.okx.com/api/v5/market/history-candles"
DEFAULT_PROXY: str = "http://127.0.0.1:10809"

TIER1: list[str] = ["BTC", "ETH", "SOL", "BNB", "XRP"]
TIER2: list[str] = [
    "DOGE", "ADA", "TRX", "LINK", "AVAX", "DOT", "LTC", "BCH",
    "UNI", "ATOM", "XLM", "NEAR", "APT", "ARB", "OP", "SUI",
]
TIERS: dict[str, list[str]] = {"tier1": TIER1, "tier2": TIER2}

PAGE_LIMIT: int = 300
PAGE_SLEEP: float = 0.3          # 温和限速（秒/页）
CHECKPOINT_EVERY: int = 120      # 每 N 页落盘一次断点
BACKOFF_STEPS: list[float] = [5, 15, 40, 90, 120, 120, 120, 120, 120, 120, 120, 120]
MAX_CONSECUTIVE_FAILURES: int = len(BACKOFF_STEPS)


def build_session(proxy: str | None) -> requests.Session:
    """构造请求会话：proxy 为空则直连，否则走代理。"""
    session: requests.Session = requests.Session()
    if proxy:
        session.proxies = {"http": proxy, "https": proxy}
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    return session


def fetch_page_once(session: requests.Session, inst_id: str, bar: str,
                    after: str | None) -> list:
    """拉一页历史 K 线（OKX 返回最新在前），失败抛异常由上层退避重试。"""
    params: dict = {"instId": inst_id, "bar": bar, "limit": str(PAGE_LIMIT)}
    if after:
        params["after"] = after
    resp: requests.Response = session.get(API_URL, params=params, timeout=15)
    resp.raise_for_status()
    data: dict = resp.json()
    if data.get("code") != "0":
        raise ValueError(f"OKX 错误: {data.get('msg')}")
    return data["data"]


def load_checkpoint(path: Path) -> pl.DataFrame:
    """读取已有断点文件（不存在或损坏则返回空表）。"""
    if not path.exists():
        return pl.DataFrame()
    try:
        return pl.read_parquet(path)
    except Exception as error:  # 断点文件损坏则弃用，从头拉
        print(f"    [警告] 断点文件损坏，重新拉取: {error}", flush=True)
        return pl.DataFrame()


def save_checkpoint(path: Path, existing: pl.DataFrame,
                    new_rows: list[list]) -> pl.DataFrame:
    """合并新拉取的页并覆盖写断点，返回合并后的表。"""
    fresh: pl.DataFrame = rows_to_frame(new_rows)
    merged: pl.DataFrame = (
        pl.concat([existing, fresh], how="vertical")
        .unique(subset=["datetime"]).sort("datetime")
    )
    merged.write_parquet(path)
    return merged


def rows_to_frame(rows: list[list]) -> pl.DataFrame:
    """OKX 行 [ts,o,h,l,c,vol,volCcy,...] -> 与 AlphaLab 同构的表。"""
    return pl.DataFrame({
        "datetime": [
            datetime.fromtimestamp(int(r[0]) / 1000, tz=timezone.utc).replace(tzinfo=None)
            for r in rows
        ],
        "open": [float(r[1]) for r in rows],
        "high": [float(r[2]) for r in rows],
        "low": [float(r[3]) for r in rows],
        "close": [float(r[4]) for r in rows],
        "volume": [float(r[5]) for r in rows],
        "turnover": [float(r[6]) for r in rows],
        "open_interest": [0.0 for _ in rows],
    })


def fetch_history_resumable(session: requests.Session, inst_id: str,
                            bar: str, start_ms: int, out_path: Path) -> int:
    """断点续传式拉取：从断点文件最早时间继续向前翻页，返回总根数。"""
    merged: pl.DataFrame = load_checkpoint(out_path)
    if merged.is_empty():
        cursor_ms: int = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    else:
        earliest: datetime = merged["datetime"].min()
        cursor_ms: int = int(earliest.replace(tzinfo=timezone.utc).timestamp() * 1000) - 60_000
        print(f"    [续传] 已有 {merged.height} 根，从 {earliest} 继续向前", flush=True)

    buffer: list[list] = []
    pages: int = 0
    failures: int = 0
    while cursor_ms > start_ms:
        try:
            rows: list = fetch_page_once(session, inst_id, bar, str(cursor_ms))
        except (requests.RequestException, ValueError) as error:
            failures += 1
            if failures >= MAX_CONSECUTIVE_FAILURES:
                raise ConnectionError(
                    f"{inst_id} 连续 {failures} 次失败: {str(error)[:80]}"
                ) from error
            delay: float = BACKOFF_STEPS[failures - 1]
            print(f"    [重试 {failures}/{MAX_CONSECUTIVE_FAILURES}] "
                  f"{delay:.0f}s 后继续: {str(error)[:60]}", flush=True)
            time.sleep(delay)
            continue
        failures = 0
        if not rows:
            break  # 到达 OKX 历史库存的最早时间
        buffer.extend(rows)
        cursor_ms = int(rows[-1][0])
        pages += 1
        if pages % CHECKPOINT_EVERY == 0:
            merged = save_checkpoint(out_path, merged, buffer)
            buffer = []
            print(f"    [断点] 累计 {merged.height} 根 "
                  f"(最早 {merged['datetime'].min()})", flush=True)
        time.sleep(PAGE_SLEEP)

    if buffer:
        merged = save_checkpoint(out_path, merged, buffer)
    return merged.height


def download_coin(session: requests.Session, coin: str, bar: str,
                  start_ms: int) -> tuple[str, int]:
    """下载单个币（含续传），返回 (币名, 根数)。"""
    out_path: Path = MINUTE_DIR / f"{coin}USDT.OKX.{bar}.parquet"
    count: int = fetch_history_resumable(
        session, f"{coin}-USDT", bar, start_ms, out_path
    )
    return coin, count


def main() -> None:
    """入口：解析参数 -> 选币 -> 逐币下载（单币失败记录后继续）。"""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="下载主流币分钟级 K 线（OKX，支持断点续传）"
    )
    parser.add_argument("--bar", default="5m",
                        help="K 线粒度：1m/3m/5m/15m/30m/1H/2H/4H 等")
    parser.add_argument("--days", type=int, default=730, help="回溯天数")
    parser.add_argument("--tiers", default="tier1,tier2",
                        help="币种档位：tier1 / tier2 / tier1,tier2")
    parser.add_argument("--coins", default="",
                        help="显式币种列表（逗号分隔），指定后忽略 --tiers")
    args: argparse.Namespace = parser.parse_args()

    if args.coins:
        coins: list[str] = [c.strip().upper() for c in args.coins.split(",") if c.strip()]
    else:
        coins = []
        for tier in args.tiers.split(","):
            if tier not in TIERS:
                raise ValueError(f"未知档位: {tier}（可选 tier1/tier2）")
            coins.extend(TIERS[tier])

    MINUTE_DIR.mkdir(parents=True, exist_ok=True)
    start_ms: int = int(
        (datetime.now() - timedelta(days=args.days))
        .replace(tzinfo=timezone.utc).timestamp() * 1000
    )

    session: requests.Session = build_session(None)
    use_proxy: bool = False
    print(f"下载 {len(coins)} 个币 {args.bar} K 线（近 {args.days} 天）...", flush=True)

    for coin in coins:
        try:
            coin_name, count = download_coin(session, coin, args.bar, start_ms)
            print(f"[完成] {coin_name}: {count} 根 -> "
                  f"{MINUTE_DIR / f'{coin}USDT.OKX.{args.bar}.parquet'}", flush=True)
        except (ConnectionError, requests.RequestException) as error:
            print(f"[失败] {coin}: {str(error)[:100]}", flush=True)
            if not use_proxy:
                proxy: str = os.environ.get("OKX_PROXY", DEFAULT_PROXY)
                print(f"  直连多次失败，切换代理 {proxy} 重试 {coin}", flush=True)
                session = build_session(proxy)
                use_proxy = True
                try:
                    _name, count = download_coin(session, coin, args.bar, start_ms)
                    print(f"[完成-代理] {coin}: {count} 根", flush=True)
                except (ConnectionError, requests.RequestException) as retry_error:
                    print(f"[失败] {coin} (代理): {str(retry_error)[:100]}", flush=True)


if __name__ == "__main__":
    main()
