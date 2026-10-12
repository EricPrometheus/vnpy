"""从币安官方数据仓库（data.binance.vision）下载 K 线月度打包（现货/合约通吃）。

数据源 URL 结构：
    现货:       https://data.binance.vision/data/spot/monthly/klines/{SYMBOL}/{bar}/{SYMBOL}-{bar}-{YYYY-MM}.zip
    U本位合约:  https://data.binance.vision/data/futures/um/monthly/klines/{SYMBOL}/{bar}/{SYMBOL}-{bar}-{YYYY-MM}.zip
CSV 无表头，现货 12 列；合约老文件可能 11 列（少尾列），脚本按实际列数自适应。
注意 2025 年起币安把时间戳从毫秒改成微秒，自动检测归一。
输出：research/alpha_lab/minute/{COIN}USDT.BINANCE.{SPOT|UM}.{bar}.parquet。

用法（在仓库根目录运行）：
    D:\\veighna_studio\\python.exe -X utf8 research/download_binance_minutes.py --market spot --bar 1m --months 36
    D:\\veighna_studio\\python.exe -X utf8 research/download_binance_minutes.py --market um --bar 1m --months 36
"""

import argparse
import io
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import requests

OUT_DIR: Path = Path("research/alpha_lab/minute")
TMP_DIR: Path = Path("research/alpha_lab/minute/_tmp")
VISION_BASE: str = "https://data.binance.vision/data"
MARKET_PATHS: dict[str, str] = {"spot": "spot", "um": "futures/um"}
MARKET_TAGS: dict[str, str] = {"spot": "SPOT", "um": "UM"}
TIMEOUT: int = 30

COINS: list[str] = ["BTC", "ETH", "SOL", "BNB", "XRP"]

# 12 列版（合约 11 列文件会去掉最后的 ignore 列）
CSV_COLUMNS: list[str] = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]
CSV_DTYPES: dict[str, object] = {
    "open_time": pl.Int64, "open": pl.Float64, "high": pl.Float64,
    "low": pl.Float64, "close": pl.Float64, "volume": pl.Float64,
    "close_time": pl.Int64, "quote_volume": pl.Float64, "count": pl.Int64,
    "taker_buy_volume": pl.Float64, "taker_buy_quote_volume": pl.Float64,
    "ignore": pl.Utf8,
}


def month_list(start_month: datetime, end_month: datetime) -> list[str]:
    """生成 [起始月, 结束月] 闭区间的 'YYYY-MM' 列表。"""
    months: list[str] = []
    year: int = start_month.year
    month: int = start_month.month
    while (year, month) <= (end_month.year, end_month.month):
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month > 12:
            year, month = year + 1, 1
    return months


def normalize_timestamp(values: pl.Series) -> pl.Series:
    """币安 2025 起时间戳从毫秒改成微秒，按量级自动归一为毫秒。"""
    if values.len() == 0:
        return values
    return values // 1000 if values.max() > 10**15 else values


def parse_kline_csv(raw: bytes) -> pl.DataFrame:
    """解析 K 线 CSV：合约新文件带表头行、现货无表头，自适应处理。"""
    first_line: str = raw.split(b"\n")[0].decode().strip().lower()
    has_header: bool = first_line.startswith("open_time")
    width: int = first_line.count(",") + 1
    names: list[str] = CSV_COLUMNS[:width]
    dtypes: dict[str, object] = {k: CSV_DTYPES[k] for k in names}
    frame: pl.DataFrame = pl.read_csv(
        io.BytesIO(raw), has_header=has_header,
        new_columns=None if has_header else names,
        schema_overrides=dtypes,
    )
    if has_header:  # 用文件自带的表头读取后，按位置统一重命名
        frame = frame.rename(dict(zip(frame.columns, names)))
    open_ms: pl.Series = normalize_timestamp(frame["open_time"])
    return pl.DataFrame({
        "datetime": [
            datetime.fromtimestamp(ts / 1000, tz=timezone.utc).replace(tzinfo=None)
            for ts in open_ms
        ],
        "open": frame["open"], "high": frame["high"],
        "low": frame["low"], "close": frame["close"],
        "volume": frame["volume"], "turnover": frame["quote_volume"],
        "open_interest": pl.Series([0.0] * frame.height),
    })


def download_month(session: requests.Session, base_url: str, symbol: str,
                   bar: str, month: str, market_tag: str) -> pl.DataFrame | None:
    """下载并解析单月 K 线 zip；该月打包不存在（404）返回 None。"""
    url: str = f"{base_url}/{symbol}/{bar}/{symbol}-{bar}-{month}.zip"
    resp: requests.Response = session.get(url, timeout=TIMEOUT)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()

    zip_path: Path = TMP_DIR / f"{market_tag}-{symbol}-{bar}-{month}.zip"
    zip_path.write_bytes(resp.content)  # 落临时盘，避免整包进内存
    try:
        with zipfile.ZipFile(zip_path) as archive:
            csv_name: str = archive.namelist()[0]
            raw: bytes = archive.read(csv_name)
    finally:
        zip_path.unlink(missing_ok=True)
    return parse_kline_csv(raw)


def download_coin(session: requests.Session, base_url: str, coin: str,
                  bar: str, months: list[str], market_tag: str) -> None:
    """逐月下载单个币并合并落盘；缺失月跳过并记录。"""
    symbol: str = f"{coin}USDT"
    frames: list[pl.DataFrame] = []
    missing: list[str] = []
    for month in months:
        try:
            frame: pl.DataFrame | None = download_month(
                session, base_url, symbol, bar, month, market_tag
            )
        except requests.RequestException as error:
            print(f"  {coin} {month}: 下载失败 {str(error)[:60]}，重试一次",
                  flush=True)
            frame = download_month(session, base_url, symbol, bar, month)
        if frame is None:
            missing.append(month)
            continue
        frames.append(frame)
        print(f"  {coin} {month}: {frame.height} 根", flush=True)

    if not frames:
        raise ValueError(f"{coin}: 一整段时间都没有可用数据")
    merged: pl.DataFrame = (
        pl.concat(frames, how="vertical")
        .unique(subset=["datetime"]).sort("datetime")
    )
    out_path: Path = OUT_DIR / f"{symbol}.BINANCE.{market_tag}.{bar}.parquet"
    merged.write_parquet(out_path)
    size_mb: float = out_path.stat().st_size / 1024 / 1024
    print(f"[完成] {coin}: 共 {merged.height} 根 "
          f"({merged['datetime'].min()} ~ {merged['datetime'].max()}) "
          f"-> {out_path} ({size_mb:.1f} MB)"
          + (f"；缺失月: {missing}" if missing else ""), flush=True)


def main() -> None:
    """入口：解析参数 -> 逐币下载近 N 个月的月度打包。"""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="从币安数据仓库下载 K 线（月度打包，现货/合约通吃）"
    )
    parser.add_argument("--market", default="spot", choices=["spot", "um"],
                        help="spot=现货, um=U本位永续合约")
    parser.add_argument("--bar", default="1m",
                        help="K 线粒度：1s/1m/5m/15m/1h/4h/1d 等")
    parser.add_argument("--months", type=int, default=36, help="回溯月数")
    parser.add_argument("--coins", default=",".join(COINS),
                        help="币种列表（逗号分隔）")
    args: argparse.Namespace = parser.parse_args()

    coins: list[str] = [c.strip().upper() for c in args.coins.split(",") if c.strip()]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    now: datetime = datetime.now(tz=timezone.utc)
    year: int = now.year - args.months // 12
    month_now: int = now.month - args.months % 12
    if month_now <= 0:
        year, month_now = year - 1, month_now + 12
    start_month: datetime = datetime(year, month_now, 1, tzinfo=timezone.utc)
    end_month: datetime = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
    months: list[str] = month_list(start_month, end_month)

    base_url: str = f"{VISION_BASE}/{MARKET_PATHS[args.market]}/monthly/klines"
    print(f"下载 {len(coins)} 个币 [{args.market}] {args.bar} 月度打包 "
          f"({months[0]} ~ {months[-1]})...", flush=True)

    session: requests.Session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    market_tag: str = MARKET_TAGS[args.market]
    for coin in coins:
        download_coin(session, base_url, coin, args.bar, months, market_tag)
    print(f"\n全部完成。数据目录: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
