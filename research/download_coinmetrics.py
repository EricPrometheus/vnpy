"""下载 CoinMetrics 社区链上数据（GitHub coinmetrics/data，经 jsdelivr CDN 镜像）。

为什么是 CoinMetrics：唯一给因子研究带来"新信息维度"的免费开源数据集——
活跃地址数（AdrActCnt）、链上转账额（TxTfrValAdjUSD）、交易笔数（TxCnt）、
市值（CapMrktCurUSD）等链上指标，完全独立于价量 K 线。社区版免费无 key、日频。

网络路径：raw.githubusercontent.com 直连不通（本机环境常态），走 jsdelivr
CDN 镜像（cdn.jsdelivr.net/gh/coinmetrics/data@master/csv/{asset}.csv）。
jsdelivr 有缓存、至多滞后 master 一天，对日频研究无影响。

输出：research/coinmetrics/{asset}.parquet（date + 精选列；源文件几百列，
与本研究无关的列全部丢弃）。某币缺失的指标列补空值，保证下游 join 稳定。
用法（在仓库根目录运行）：
    D:\\veighna_studio\\python.exe -X utf8 research/download_coinmetrics.py
"""

import io
import time
from pathlib import Path

import polars as pl
import requests

OUT_DIR: Path = Path("research/coinmetrics")
CDN_TEMPLATE: str = "https://cdn.jsdelivr.net/gh/coinmetrics/data@master/csv/{asset}.csv"
# 22 币 -> CoinMetrics asset id（全小写；与日线面板币种一一对应）
ASSETS: list[str] = [
    "btc", "eth", "bnb", "sol", "xrp", "doge", "ada", "trx", "link",
    "avax", "dot", "ltc", "bch", "uni", "atom", "xlm", "near", "apt",
    "arb", "op", "sui", "pepe",
]
KEEP_COLS: list[str] = [
    "time", "AdrActCnt", "CapMrktCurUSD", "TxCnt", "PriceUSD",
]
# 社区版实测：TxTfrValAdjUSD/VtyDayRet30d 等已移入付费层（全空），不再下载
TIMEOUT_SECONDS: int = 90
FLOAT_COLS: list[str] = ["AdrActCnt", "CapMrktCurUSD", "TxCnt", "PriceUSD"]


def download_one(session: requests.Session, asset: str) -> tuple[int, list[str]]:
    """下载单币 CSV，保留关键列并落盘；返回 (行数, 缺失的指标列)。

    全列按字符串读入再转换（infer_schema_length=0）：供应量类列含
    10^18 级最小单位大数，polars 默认 i64 推断会直接解析失败。
    """
    response = session.get(CDN_TEMPLATE.format(asset=asset), timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    frame: pl.DataFrame = pl.read_csv(io.BytesIO(response.content), infer_schema_length=0)
    missing: list[str] = [c for c in FLOAT_COLS if c not in frame.columns]
    keep: list[str] = [c for c in ["time", *FLOAT_COLS] if c in frame.columns]
    frame = frame.select(keep)
    for col in FLOAT_COLS:  # 数值列：字符串 -> Float64（含千分位清理）
        if col in frame.columns:
            frame = frame.with_columns(
                pl.col(col).str.replace_all(",", "").cast(pl.Float64, strict=False)
            )
    for col in missing:  # 缺失指标补空列，保证下游 join 后列齐全
        frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias(col))
    frame = frame.with_columns(
        pl.col("time").str.slice(0, 10).str.to_date().alias("date")
    ).sort("date")
    out_path: Path = OUT_DIR / f"{asset}.parquet"
    frame.write_parquet(out_path)
    return frame.height, missing


def main() -> None:
    """批量下载 22 币链上数据，逐币容错。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    failed: list[str] = []
    t0: float = time.time()
    for index, asset in enumerate(ASSETS, 1):
        try:
            rows: int
            missing: list[str]
            rows, missing = download_one(session, asset)
            note: str = f"（缺列: {missing}）" if missing else ""
            print(f"  [{index}/{len(ASSETS)}] {asset}: {rows} 行 {note}")
        except Exception as error:  # 单币失败不阻断批次
            failed.append(asset)
            print(f"  [{index}/{len(ASSETS)}] {asset}: 失败 {str(error)[:80]}")
        time.sleep(0.3)  # 温和限速，尊重公共 CDN
    if failed:
        print(f"失败 {len(failed)} 个: {failed}")
    print(f"完成，输出目录 {OUT_DIR}，耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
