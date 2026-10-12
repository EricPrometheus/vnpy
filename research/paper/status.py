"""一键状态 CLI：查看模拟盘当前持仓、最新净值、累计费用与监控库规模。

运行（仓库根目录）：D:\\veighna_studio\\python.exe research/paper/status.py
只读 paper.db（不连交易所、不写任何文件），适合挂在脚本/计划任务里日常巡检。
"""

import paper_store


def print_latest(latest: dict[str, object] | None) -> None:
    """打印最近一天的判定快照（库为空时给出回填提示）。"""
    if latest is None:
        print("库为空：请先运行 research/paper/backfill_db.py 回填重放历史")
        return
    source_label: str = "实盘仿真" if latest["source"] == "live" else "历史重放"
    print(f"最新日期 : {latest['date']}（来源: {source_label}）")
    print(f"择时状态 : {'持仓' if latest['timing_hold'] else '空仓'}")
    holdings: list[str] = list(latest["holdings"])
    print(f"当前持仓 : {len(holdings)} 只 "
          f"{'、'.join(holdings) if holdings else '(空仓)'}")
    print(f"最新净值 : {latest['equity']:.2f} USDT（现金 {latest['cash']:.2f}）")
    print(f"当日成交 : {latest['n_trades']} 笔")


def main() -> int:
    """入口：初始化建库（幂等）并汇总打印状态。"""
    db_file = paper_store.init_db()
    print(f"数据库   : {db_file}")
    print_latest(paper_store.latest_daily_status())
    counts: dict[str, int] = paper_store.summary_counts()
    print(f"daily_status : {counts['daily_rows']} 行, "
          f"累计费用 {counts['daily_fees']:.2f} USDT")
    print(f"trades       : {counts['trade_rows']} 行, "
          f"逐笔费用合计 {counts['trade_fees']:.2f} USDT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
