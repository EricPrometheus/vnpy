"""把 replay_composite_equity_ma30.csv 的重放历史回填进 SQLite 监控库。

目的：让 Grafana 看板开箱就有完整 309 天历史净值曲线（source='replay'），
策略上线后写入的实盘仿真行用 source='live'，同图异色区分。

回填口径（与回测 CSV 列一一对应）：
- daily_status：equity/cash 原样取自 CSV；holdings 由 target_coins 解析；
  n_trades 当日订单数；fees 为截至当日的累计费用（便于画累计费用曲线）；
- trades：CSV 的 orders 字段形如 "SELL X 123.4; BUY Y 1.2@8000.9"，
  逐单解析入库；市价卖出单重放里不含成交价，price 记 0.0；
  重放未拆分单笔费用，trade.fee 统一记 0.0（费用汇总在 daily_status.fees）。

运行（仓库根目录）：D:\\veighna_studio\\python.exe research/paper/backfill_db.py
脚本可重复执行（daily_status 按主键 UPSERT；trades 会重复追加，重灌前删 paper.db）。
"""

import re
from pathlib import Path

import polars as pl

import paper_store

CSV_FILENAME: str = "replay_composite_equity_ma30.csv"
CSV_REL: str = "research/paper/replay_composite_equity_ma30.csv"  # 文档提示用
EMPTY_HOLDINGS_TOKEN: str = "(空仓)"       # CSV 里空仓日的占位文本
TS_SUFFIX: str = "T16:05:00+00:00"        # 重放成交统一记在当日判定时刻（UTC）
ORDER_PATTERN: re.Pattern[str] = re.compile(
    r"(BUY|SELL)\s+([A-Z0-9.]+)\s+([0-9.]+)(?:@([0-9.]+))?"
)


def csv_path() -> Path:
    """重放 CSV 路径：本脚本同级目录（字面文件名，无动态拼接）。"""
    return Path(__file__).resolve().parent / CSV_FILENAME


def parse_holdings(target_coins: str) -> list[str]:
    """解析 target_coins 列："(空仓)" 返回空列表，否则按 | 拆分。"""
    text: str = str(target_coins).strip()
    if not text or text == EMPTY_HOLDINGS_TOKEN:
        return []
    return [coin.strip() for coin in text.split("|") if coin.strip()]


def parse_orders(orders: str) -> list[tuple[str, str, float, float]]:
    """解析 orders 列为 (方向, 符号, 数量, 价格) 列表；无 @ 价的市价单价格记 0。"""
    parsed: list[tuple[str, str, float, float]] = []
    for match in ORDER_PATTERN.finditer(str(orders or "")):
        price_text: str | None = match.group(4)
        parsed.append((
            match.group(1), match.group(2),
            float(match.group(3)), float(price_text) if price_text else 0.0,
        ))
    return parsed


def backfill() -> tuple[int, int]:
    """全量回填：返回 (daily_status 行数, trades 行数)。"""
    frame: pl.DataFrame = pl.read_csv(csv_path())
    paper_store.init_db()
    cumulative_fee: float = 0.0
    status_count: int = 0
    trade_count: int = 0
    for row in frame.iter_rows(named=True):
        day: str = str(row["date"])
        cumulative_fee += float(row["fee"])
        orders: list[tuple[str, str, float, float]] = parse_orders(
            str(row["orders"] or "")
        )
        paper_store.upsert_daily_status(
            day,
            str(row["hold_timing"]).strip().lower() == "true",
            parse_holdings(str(row["target_coins"])),
            float(row["equity"]), float(row["cash"]),
            len(orders), cumulative_fee, paper_store.SOURCE_REPLAY,
        )
        status_count += 1
        for direction, symbol, qty, price in orders:
            paper_store.insert_trade(
                day + TS_SUFFIX, symbol, direction, price, qty,
                0.0, paper_store.SOURCE_REPLAY,
            )
            trade_count += 1
    return status_count, trade_count


def main() -> int:
    """入口：回填并打印统计。"""
    try:
        status_count, trade_count = backfill()
    except (OSError, pl.exceptions.PolarsError) as exc:
        print(f"回填失败: {exc}")
        return 1
    print(f"已回填 daily_status {status_count} 行, trades {trade_count} 行 "
          f"-> {paper_store.db_path()}")
    print(f"统计: {paper_store.summary_counts()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
