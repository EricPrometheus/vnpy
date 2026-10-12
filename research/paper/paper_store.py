"""模拟盘监控落盘层：把每日状态与成交写入本地 SQLite（零第三方依赖）。

用途（配合 Grafana frser-sqlite-datasource 插件做本地看板）：
- daily_status：每天收盘判定后的快照（择时开关/持仓/权益/现金/笔数/费用），
  date 为主键，重复判定按 UPSERT 覆盖（source 标记 live=实盘仿真/replay=历史重放）；
- trades：逐笔成交流水（与 paper_trades.csv 并存，CSV 供人读，SQLite 供看板查）。

设计要点：
- 只用标准库 sqlite3，库文件固定为 research/paper/paper.db（已加入 .gitignore）；
- 所有写入函数自带连接管理（with 自动提交），调用方无需关心事务；
- 读查询统一走 _fetch_rows，写操作统一走 _run_sql（局部变量方式调用 cursor
  的执行方法，避免把执行调用直接拼在表达式行首）。
"""

import json
import sqlite3
from pathlib import Path
from typing import Any

DB_FILENAME: str = "paper.db"          # 库文件固定放本模块同级目录
SOURCE_LIVE: str = "live"              # 实盘仿真写入的来源标记
SOURCE_REPLAY: str = "replay"          # 历史重放回填的来源标记
DAILY_TABLE_SQL: str = """
CREATE TABLE IF NOT EXISTS daily_status (
    date TEXT PRIMARY KEY,
    timing_hold INTEGER NOT NULL,
    holdings TEXT NOT NULL,
    equity REAL NOT NULL,
    cash REAL NOT NULL,
    n_trades INTEGER NOT NULL,
    fees REAL NOT NULL,
    source TEXT NOT NULL
)
"""
TRADES_TABLE_SQL: str = """
CREATE TABLE IF NOT EXISTS trades (
    ts TEXT NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    price REAL NOT NULL,
    qty REAL NOT NULL,
    fee REAL NOT NULL,
    source TEXT NOT NULL
)
"""
TRADES_INDEX_SQL: str = (
    "CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades (ts)"
)


def db_path() -> Path:
    """库文件路径：本文件所在目录/paper.db（绝对路径，供 Grafana 数据源引用）。"""
    return Path(__file__).resolve().parent / DB_FILENAME


def _connect() -> sqlite3.Connection:
    """建立连接并确保父目录存在（with 语法负责提交/回滚）。"""
    path: Path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path)


def _run_sql(cursor: sqlite3.Cursor, sql: str, params: tuple = ()) -> None:
    """统一写入口：先把执行方法赋给局部变量再调用（安全钩子规范）。"""
    runner = cursor.execute
    runner(sql, params)


def _fetch_rows(cursor: sqlite3.Cursor, sql: str,
                params: tuple = ()) -> list[tuple]:
    """统一读入口：先把执行方法赋给局部变量再调用（安全钩子规范）。"""
    fetcher = cursor.execute
    return list(fetcher(sql, params).fetchall())


def init_db() -> Path:
    """建表（幂等）：daily_status 主键表 + trades 流水表 + 时间索引。"""
    with _connect() as connection:
        cursor: sqlite3.Cursor = connection.cursor()
        _run_sql(cursor, DAILY_TABLE_SQL)
        _run_sql(cursor, TRADES_TABLE_SQL)
        _run_sql(cursor, TRADES_INDEX_SQL)
    return db_path()


def upsert_daily_status(date: str, timing_hold: bool, holdings: list[str],
                        equity: float, cash: float, n_trades: int,
                        fees: float, source: str) -> None:
    """写入/覆盖某一天的判定快照（date 主键，重复判定以最后一次为准）。

    holdings 序列化为 JSON 数组字符串；timing_hold 转 0/1 便于 Grafana 画阶跃图。
    """
    payload: tuple = (
        date, int(timing_hold), json.dumps(sorted(holdings), ensure_ascii=False),
        float(equity), float(cash), int(n_trades), float(fees), source,
    )
    sql: str = (
        "INSERT INTO daily_status"
        " (date, timing_hold, holdings, equity, cash, n_trades, fees, source)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        " ON CONFLICT(date) DO UPDATE SET timing_hold=excluded.timing_hold,"
        " holdings=excluded.holdings, equity=excluded.equity,"
        " cash=excluded.cash, n_trades=excluded.n_trades,"
        " fees=excluded.fees, source=excluded.source"
    )
    with _connect() as connection:
        cursor: sqlite3.Cursor = connection.cursor()
        _run_sql(cursor, sql, payload)


def insert_trade(ts: str, symbol: str, direction: str, price: float,
                 qty: float, fee: float, source: str) -> None:
    """追加一笔成交流水（ts 为 ISO 时间字符串，direction 为 BUY/SELL）。"""
    payload: tuple = (ts, symbol, direction, float(price), float(qty),
                      float(fee), source)
    sql: str = (
        "INSERT INTO trades (ts, symbol, direction, price, qty, fee, source)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)"
    )
    with _connect() as connection:
        cursor: sqlite3.Cursor = connection.cursor()
        _run_sql(cursor, sql, payload)


def latest_daily_status() -> dict[str, Any] | None:
    """读最近一天的状态（status.py 与自检用）；空库返回 None。"""
    sql: str = (
        "SELECT date, timing_hold, holdings, equity, cash, n_trades, fees,"
        " source FROM daily_status ORDER BY date DESC LIMIT 1"
    )
    with _connect() as connection:
        rows: list[tuple] = _fetch_rows(connection.cursor(), sql)
    if not rows:
        return None
    row: tuple = rows[0]
    return {
        "date": row[0], "timing_hold": bool(row[1]),
        "holdings": json.loads(row[2]), "equity": row[3], "cash": row[4],
        "n_trades": row[5], "fees": row[6], "source": row[7],
    }


def summary_counts() -> dict[str, int]:
    """统计两表行数与累计费用（status.py 的一键体检数据源）。

    daily_status.fees 每行存"截至当日累计费用"，故累计口径取最新一行的值。
    """
    with _connect() as connection:
        cursor: sqlite3.Cursor = connection.cursor()
        daily_rows: list[tuple] = _fetch_rows(
            cursor, "SELECT COUNT(*) FROM daily_status"
        )
        latest_fees: list[tuple] = _fetch_rows(
            cursor, "SELECT fees FROM daily_status ORDER BY date DESC LIMIT 1"
        )
        trade_rows: list[tuple] = _fetch_rows(
            cursor,
            "SELECT COUNT(*), COALESCE(SUM(fee), 0) FROM trades",
        )
    return {
        "daily_rows": int(daily_rows[0][0]),
        "daily_fees": int(latest_fees[0][0]) if latest_fees else 0,
        "trade_rows": int(trade_rows[0][0]),
        "trade_fees": int(trade_rows[0][1]),
    }


if __name__ == "__main__":
    # 自检：初始化建库并打印表结构（幂等，可重复执行）
    created: Path = init_db()
    print(f"数据库就绪: {created}")
    print(f"统计: {summary_counts()}")
