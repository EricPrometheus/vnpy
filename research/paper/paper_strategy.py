"""模式 A：gp_s123 选币 + BTC 20 日均线择时 Top5 轮动 —— vnpy 实时仿真盘策略。

设计要点（口径与 research/paper/backtest_replay.py 的模式 B、研究回测完全一致）：
- 行情：订阅 OKX 实时 tick（vnpy_okx 网关，现货合约命名为 BTCUSDT_SPOT_OKX，
  交易所为 GLOBAL，与研究数据 BTCUSDT.OKX 通过映射函数互转）；
- 切日：按 UTC 16:00 聚合日线（bar K 覆盖 [K 16:00, K+1 16:00) UTC），与
  research/alpha_lab/daily/*.parquet 的日线时间戳一致；
- 预热：历史日线从本地 parquet 读取（因子 ts_delay/ts_mean 与均线需要 20+ 根
  历史 bar，冷启动无法靠实时 tick 攒齐），之后由实时 tick 增量补齐最新 bar；
- 信号：每日收盘后（UTC 16:05 = 北京时间 00:05）触发一次判定——BTCUSDT.OKX
  收盘 > 20 日均线则持仓，按 gp_s123 因子表达式（vnpy 表达式引擎
  calculate_by_expression，与回测同口径）选 Top5 等权，否则空仓；
- 下单：目标组合与当前持仓不一致时先卖后买市价单；MainEngine.send_order 已被
  vnpy_paperaccount（模拟交易应用）接管，订单在本地仿真账户撮合，不会触碰真实交易所；
- 凭据：本模块不读取、不存储任何密钥（API Key 等一律由 run_paper.py 从环境变量
  或本地凭据文件注入网关）。

已知简化（相对实盘）：
- PaperAccount 不模拟手续费/资金占用，策略自行按 0.1% 估算费用并记录流水；
- tick 聚合的日线不累计成交量（gp_s123 表达式只用到 high/low/close，无需 volume）。
"""

import os
import sqlite3
import sys
from datetime import datetime, timedelta, time as dt_time, timezone
from pathlib import Path

# 引导导入 research/ 下的既有研究模块（本文件位于 research/paper/ 子目录）
_RESEARCH_DIR: Path = Path(__file__).resolve().parent.parent
if str(_RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_DIR))
# 引导导入本目录模块（paper_store 与策略同目录，保证任意入口可 import）
_PAPER_DIR: Path = Path(__file__).resolve().parent
if str(_PAPER_DIR) not in sys.path:
    sys.path.insert(0, str(_PAPER_DIR))

import paper_store

import polars as pl
from vnpy.event import Event, EventEngine
from vnpy.trader.constant import Direction, Exchange, Offset, OrderType
from vnpy.trader.engine import MainEngine
from vnpy.trader.event import (
    EVENT_CONTRACT,
    EVENT_LOG,
    EVENT_TICK,
    EVENT_TIMER,
    EVENT_TRADE,
)
from vnpy.trader.object import (
    ContractData,
    LogData,
    OrderRequest,
    SubscribeRequest,
    TickData,
    TradeData,
)

from factor_common import daily_dir, load_manifest, safe_join

BTC_SYMBOL: str = "BTCUSDT.OKX"
BTC_MA_WINDOW: int = 20          # 择时均线窗口（与研究口径一致）
TOP_N: int = 5                   # 持仓币数（等权）
FEE_RATE: float = 0.001          # 手续费率（PaperAccount 不扣费，策略自行估算）
GP_PREFIX: str = "gp_s123"       # 因子清单里的选币因子名前缀
DAILY_CUT_UTC_HOUR: int = 16     # 日线切日边界：UTC 16:00
DECISION_TIME: dt_time = dt_time(16, 5)   # 判定时刻：UTC 16:05（bar 收盘后留 5 分钟）
WARMUP_BARS: int = 120           # 预热历史日线根数（>= 因子/均线所需窗口）
MIN_BARS_REQUIRED: int = 25      # 参与选币所需的最少 bar 数（5 日延迟 + 20 日均值）
VOLUME_DECIMALS: int = 6         # 下单数量小数位（现货可任意拆分，取 6 位足够）
ENV_PAPER_CAPITAL: str = "PAPER_CAPITAL"      # 模拟资金（USDT），默认 100 万
DEFAULT_CAPITAL: float = 1_000_000.0
JOURNAL_REL: str = "paper/paper_trades.csv"   # 交易流水（相对 research 根目录）
JOURNAL_HEADER: str = "time,completed_bar,hold,target_coins,equity_est,realized_pnl,fee_est"
BAR_FIELDS: list[str] = ["datetime", "open", "high", "low", "close", "volume"]


def research_to_live(research_symbol: str) -> str:
    """研究符号转 OKX 实盘合约名：BTCUSDT.OKX -> BTCUSDT_SPOT_OKX。"""
    return research_symbol.split(".")[0] + "_SPOT_OKX"


def live_to_research(live_vt_symbol: str) -> str | None:
    """OKX 实盘合约转研究符号：BTCUSDT_SPOT_OKX.GLOBAL -> BTCUSDT.OKX。"""
    symbol: str = live_vt_symbol.split(".")[0]
    if not symbol.endswith("_SPOT_OKX"):
        return None
    return symbol[: -len("_SPOT_OKX")] + ".OKX"


def bar_date_key(tick_time: datetime) -> str:
    """按 UTC 16:00 切日的日线键：bar K 覆盖 [K 16:00, K+1 16:00) UTC。"""
    utc_time: datetime = (
        tick_time.astimezone(timezone.utc)
        if tick_time.tzinfo else tick_time.replace(tzinfo=timezone.utc)
    )
    return (utc_time - timedelta(hours=DAILY_CUT_UTC_HOUR)).date().isoformat()


def completed_bar_key(now_utc: datetime) -> str:
    """最近一根"已收盘"日线的键（bar K 在 (K+1) 16:00 UTC 收盘）。"""
    cutoff: datetime = now_utc - timedelta(hours=DAILY_CUT_UTC_HOUR)
    return (cutoff.date() - timedelta(days=1)).isoformat()


class GpS123PaperStrategy:
    """实时仿真盘策略：订阅行情 -> 聚合日线 -> 收盘判定 -> 模拟账户调仓。"""

    def __init__(self, main_engine: MainEngine, event_engine: EventEngine,
                 gateway_name: str = "OKX") -> None:
        self.main_engine: MainEngine = main_engine
        self.event_engine: EventEngine = event_engine
        self.gateway_name: str = gateway_name
        self.active: bool = False
        self.symbols: list[str] = sorted(
            path.stem for path in daily_dir().glob("*.OKX.parquet")
        )
        self.subscribed: set[str] = set()
        self.factor_expr: str = ""
        self.last_decision_key: str = ""
        # 研究符号 -> 历史日线列表（datetime 为 naive datetime，升序）
        self.history: dict[str, list[dict[str, object]]] = {}
        # 研究符号 -> 日线键 -> 实时累计 bar
        self.live_bars: dict[str, dict[str, dict[str, float]]] = {}
        self.last_price: dict[str, float] = {}
        # 自有持仓记账：研究符号 -> {"volume", "avg_price"}
        self.book: dict[str, dict[str, float]] = {}
        self.realized_pnl: float = 0.0
        self.fee_est: float = 0.0
        self.journal_rows: list[str] = self.load_journal()
        raw_capital: str = os.environ.get(ENV_PAPER_CAPITAL, str(DEFAULT_CAPITAL))
        self.capital: float = float(raw_capital)
        # 监控落盘状态（SQLite daily_status 当日行的可变字段）
        self.status_date: str = ""              # 最近一次判定的日线键
        self.trade_count_today: int = 0         # 当日成交笔数
        self.last_closes: dict[str, float] = {}  # 最近判定日各币收盘价
        self.last_hold: bool = False            # 最近判定择时开关
        self.last_target: list[str] = []        # 最近判定目标组合

    # ------------------------------------------------------------------ 生命周期
    def start(self) -> None:
        """启动策略：注册算子与事件 -> 预热历史日线 -> 等待合约到达后订阅。"""
        self.install_operators()
        self.factor_expr = self.load_factor_expr()
        self.load_warmup_history()
        # 初始化监控 SQLite（幂等建表；失败仅记日志，绝不影响交易主流程）
        try:
            db_file: Path = paper_store.init_db()
            self.log(f"监控数据库就绪: {db_file}")
        except (sqlite3.Error, OSError) as exc:
            self.log(f"监控数据库初始化失败（看板将无数据）: {exc}")
        self.event_engine.register(EVENT_CONTRACT, self.on_contract)
        self.event_engine.register(EVENT_TICK, self.on_tick)
        self.event_engine.register(EVENT_TIMER, self.on_timer)
        self.event_engine.register(EVENT_TRADE, self.on_trade)
        self.event_engine.register(EVENT_LOG, self.on_log)
        self.active = True
        self.log(f"策略启动: {len(self.symbols)} 个合约待订阅, "
                 f"初始资金 {self.capital:.0f}, 因子表达式已装载")

    def stop(self) -> None:
        """停止策略（vnpy 事件引擎不支持反注册，仅置位让处理器空转）。"""
        self.active = False
        self.log("策略已停止")

    def install_operators(self) -> None:
        """按研究管线口径注册表达式算子（gp_s123 只用内置算子，注册是保底）。"""
        try:
            from gtja191_kit import install_operators

            install_operators()
        except ImportError:
            self.log("未找到 gtja191_kit，跳过算子注册（gp_s123 仅需内置算子）")

    def load_factor_expr(self) -> str:
        """从因子清单读取 gp_s123 表达式（与回测同一条公式）。"""
        factors: list[dict] = [
            item for item in load_manifest().get("factors", [])
            if str(item.get("name", "")).startswith(GP_PREFIX)
        ]
        if not factors:
            raise ValueError(f"因子清单缺少 {GP_PREFIX} 前缀条目，无法启动")
        return str(factors[0]["vnpy_expr"])

    # ------------------------------------------------------------------ 数据准备
    def load_warmup_history(self) -> None:
        """从本地日线 parquet 预热（冷启动时因子/均线无法靠实时 tick 攒齐）。"""
        for research_symbol in self.symbols:
            parquet_path: Path = safe_join(
                f"alpha_lab/daily/{research_symbol}.parquet"
            )
            if not parquet_path.is_file():
                self.log(f"缺少预热数据: {research_symbol}，该币将不参与选币")
                continue
            frame: pl.DataFrame = pl.read_parquet(parquet_path).sort("datetime")
            bars: list[dict[str, object]] = frame.tail(WARMUP_BARS).rows(named=True)
            self.history[research_symbol] = [
                {field_name: bar[field_name] for field_name in BAR_FIELDS}
                for bar in bars
            ]

    def merge_bars(self, research_symbol: str, completed_key: str,
                   ) -> list[dict[str, object]] | None:
        """合并历史预热与实时累计 bar，返回某币截至 completed_key 的日线序列。

        只取日期 <= completed_key 的历史 bar，保证对任意历史键判定都无未来数据。
        """
        history: list[dict[str, object]] = self.history.get(research_symbol, [])
        bars: list[dict[str, object]] = [
            bar for bar in history if str(bar["datetime"])[:10] < completed_key
        ]
        same_day: list[dict[str, object]] = [
            bar for bar in history if str(bar["datetime"])[:10] == completed_key
        ]
        live_bar: dict[str, float] | None = self.live_bars.get(
            research_symbol, {}
        ).get(completed_key)
        if live_bar is not None:
            # 实时 bar 优先（含盘中最新累计），覆盖历史中的同日 bar
            full_bar: dict[str, object] = dict(live_bar)
            full_bar["datetime"] = datetime.fromisoformat(completed_key)
            bars.append(full_bar)
        else:
            # 预热数据已包含当日（parquet 较新）且尚无实时 bar 时，直接用历史 bar
            bars.extend(same_day)
        if len(bars) < MIN_BARS_REQUIRED:
            return None
        return bars[-WARMUP_BARS:]

    def build_decision_panel(self, completed_key: str) -> pl.DataFrame | None:
        """把所有币的日线序列拼成表达式引擎所需的面板（按币、日期排序）。"""
        rows: list[dict[str, object]] = []
        for research_symbol in self.symbols:
            bars: list[dict[str, object]] | None = self.merge_bars(
                research_symbol, completed_key
            )
            if bars is None:
                continue
            for bar in bars:
                row: dict[str, object] = dict(bar)
                row["vt_symbol"] = research_symbol  # 表达式引擎按该列分组
                rows.append(row)
        if not rows:
            self.log(f"bar {completed_key} 无可用日线数据，跳过判定")
            return None
        frame: pl.DataFrame = pl.DataFrame(rows).sort(["vt_symbol", "datetime"])
        return frame

    # ------------------------------------------------------------------ 事件处理
    def on_contract(self, event: Event) -> None:
        """合约到达后订阅对应行情（PaperAccount 需要先有合约映射才能路由订阅）。"""
        contract: ContractData = event.data
        research_symbol: str | None = live_to_research(contract.vt_symbol)
        if research_symbol not in self.symbols:
            return
        request: SubscribeRequest = SubscribeRequest(
            symbol=contract.symbol, exchange=Exchange.GLOBAL
        )
        self.main_engine.subscribe(request, self.gateway_name)
        self.subscribed.add(research_symbol)
        if len(self.subscribed) == len(self.symbols):
            self.log("全部合约订阅完成，等待收盘判定")

    def on_tick(self, event: Event) -> None:
        """实时 tick：更新最新价并按 UTC 16:00 切日累计当日日线。"""
        if not self.active:
            return
        tick: TickData = event.data
        research_symbol: str | None = live_to_research(tick.vt_symbol)
        if research_symbol is None or tick.last_price <= 0:
            return
        self.last_price[research_symbol] = tick.last_price
        bars: dict[str, dict[str, float]] = self.live_bars.setdefault(
            research_symbol, {}
        )
        key: str = bar_date_key(tick.datetime)
        bar: dict[str, float] | None = bars.get(key)
        if bar is None:
            bars[key] = {
                "open": tick.last_price,
                "high": tick.last_price,
                "low": tick.last_price,
                "close": tick.last_price,
                "volume": 0.0,
            }
            return
        bar["high"] = max(bar["high"], tick.last_price)
        bar["low"] = min(bar["low"], tick.last_price)
        bar["close"] = tick.last_price

    def on_timer(self, event: Event) -> None:
        """每秒定时器：到达判定时刻（UTC 16:05）后对最近收盘日线判定一次。"""
        if not self.active:
            return
        now_utc: datetime = datetime.now(timezone.utc)
        if now_utc.time() < DECISION_TIME:
            return
        key: str = completed_bar_key(now_utc)
        if key == self.last_decision_key:
            return
        self.run_decision(key)

    def on_trade(self, event: Event) -> None:
        """成交回报：更新自有持仓记账与费用/盈亏估算（PaperAccount 不扣费）。"""
        trade: TradeData = event.data
        research_symbol: str | None = live_to_research(trade.vt_symbol)
        if research_symbol is None:
            return
        position: dict[str, float] = self.book.setdefault(
            research_symbol, {"volume": 0.0, "avg_price": 0.0}
        )
        signed: float = (
            trade.volume if trade.direction == Direction.LONG else -trade.volume
        )
        if signed > 0:
            total_cost: float = (
                position["avg_price"] * position["volume"] + trade.price * signed
            )
            position["volume"] += signed
            position["avg_price"] = (
                total_cost / position["volume"] if position["volume"] else 0.0
            )
        else:
            self.realized_pnl += (trade.price - position["avg_price"]) * (-signed)
            position["volume"] = max(position["volume"] + signed, 0.0)
        self.fee_est += trade.price * trade.volume * FEE_RATE
        self.log(f"成交: {trade.direction.value} {research_symbol} "
                 f"{trade.volume}@{trade.price}")
        # 监控落盘：成交写 trades 流水并刷新当日快照（笔数/费用/权益），失败不影响交易
        try:
            paper_store.insert_trade(
                trade.datetime.isoformat(timespec="seconds"), research_symbol,
                trade.direction.value, trade.price, trade.volume,
                trade.price * trade.volume * FEE_RATE, paper_store.SOURCE_LIVE
            )
            self.refresh_daily_row()
        except sqlite3.Error as exc:
            self.log(f"成交写入监控库失败（不影响交易）: {exc}")

    def refresh_daily_row(self) -> None:
        """成交后刷新当日快照行：笔数 +1，费用与权益按最新记账重估。"""
        if not self.status_date:
            return
        self.trade_count_today += 1
        cash: float
        equity: float
        cash, equity = self.snapshot_equity(self.last_closes)
        paper_store.upsert_daily_status(
            self.status_date, self.last_hold, self.last_target, equity, cash,
            self.trade_count_today, self.fee_est, paper_store.SOURCE_LIVE
        )

    def on_log(self, event: Event) -> None:
        """转发网关/引擎日志到控制台，便于观察连接与撮合状态。"""
        log_data: LogData = event.data
        print(f"[vnpy][{log_data.gateway_name}] {log_data.msg}")

    # ------------------------------------------------------------------ 信号与调仓
    def run_decision(self, completed_key: str) -> None:
        """收盘判定：算因子与择时 -> 生成目标组合 -> 与持仓比对调仓。"""
        self.last_decision_key = completed_key  # 先记账，避免异常后每秒重试刷屏
        panel: pl.DataFrame | None = self.build_decision_panel(completed_key)
        if panel is None:
            return
        hold: bool
        top_coins: list[str]
        hold, top_coins = self.compute_signals(panel, completed_key)
        target: set[str] = set(top_coins) if hold else set()
        self.log(f"bar {completed_key} 判定: 择时={'持仓' if hold else '空仓'}, "
                 f"目标={'|'.join(sorted(target)) or '(空仓)'}")
        self.rebalance_to(target)
        self.write_journal(completed_key, hold, top_coins)
        self.save_daily_status(completed_key, hold, target, panel)

    def save_daily_status(self, completed_key: str, hold: bool,
                          target: set[str], panel: pl.DataFrame) -> None:
        """判定后（无论是否调仓）把当日快照 upsert 进监控库，供 Grafana 读。

        权益口径：现金 = 资金 + 已实现盈亏 - 累计费用 - 持仓成本，
        权益 = 现金 + Σ持仓 × 当日收盘（收盘价取自判定面板，与信号同源）。
        """
        self.status_date = completed_key
        self.trade_count_today = 0
        latest: pl.DataFrame = (
            panel.sort("datetime")
            .group_by("vt_symbol", maintain_order=True).last()
        )
        self.last_closes: dict[str, float] = {
            str(row[0]): float(row[1])
            for row in latest.select(["vt_symbol", "close"]).iter_rows()
        }
        self.last_hold = hold
        self.last_target = sorted(target)
        cash: float
        equity: float
        cash, equity = self.snapshot_equity(self.last_closes)
        try:
            paper_store.upsert_daily_status(
                completed_key, hold, self.last_target, equity, cash,
                self.trade_count_today, self.fee_est, paper_store.SOURCE_LIVE
            )
        except sqlite3.Error as exc:
            self.log(f"当日快照写入监控库失败（不影响交易）: {exc}")

    def snapshot_equity(self, closes: dict[str, float]) -> tuple[float, float]:
        """按当日收盘估值：返回 (现金, 权益)；缺收盘价的持仓按成本价保守估算。"""
        open_positions: list[tuple[str, dict[str, float]]] = [
            (symbol, position) for symbol, position in self.book.items()
            if position["volume"] > 0
        ]
        cost: float = sum(
            position["volume"] * position["avg_price"]
            for _, position in open_positions
        )
        market_value: float = sum(
            position["volume"] * closes.get(symbol, position["avg_price"])
            for symbol, position in open_positions
        )
        cash: float = self.capital + self.realized_pnl - self.fee_est - cost
        return cash, cash + market_value

    def compute_signals(self, panel: pl.DataFrame,
                        completed_key: str) -> tuple[bool, list[str]]:
        """用 vnpy 表达式引擎计算 gp_s123 因子并叠加 BTC 择时（与回测同口径）。"""
        from vnpy.alpha.dataset.utility import calculate_by_expression

        factor_frame: pl.DataFrame = calculate_by_expression(panel, self.factor_expr)
        latest: pl.DataFrame = (
            factor_frame.sort("datetime")
            .group_by("vt_symbol", maintain_order=True).last()
            .filter(pl.col("data").is_finite())
        )
        ranked: pl.DataFrame = latest.sort(
            ["data", "vt_symbol"], descending=[True, False]
        )
        # 面板行本身就用研究符号（BTCUSDT.OKX），无需再做合约名映射
        top_coins: list[str] = ranked.head(TOP_N)["vt_symbol"].to_list()
        hold: bool = self.btc_timing(panel, completed_key)
        return hold, top_coins

    def btc_timing(self, panel: pl.DataFrame, completed_key: str) -> bool:
        """BTC 择时：最新收盘 > 20 日均线（含当日，与研究 rolling_mean(20) 一致）。"""
        btc: pl.DataFrame = panel.filter(
            pl.col("vt_symbol") == BTC_SYMBOL
        ).sort("datetime")
        if btc.height < BTC_MA_WINDOW:
            self.log("BTC 日线不足 20 根，保守起见本轮空仓")
            return False
        closes: pl.Series = btc["close"]
        moving_average: float = float(closes[-BTC_MA_WINDOW:].mean())
        as_of: str = str(btc["datetime"][-1])[:10]
        if as_of != completed_key:
            self.log(f"警告: BTC 最新 bar 为 {as_of}（预期 {completed_key}），"
                     "可能预热数据过期，判定基于旧数据")
        return bool(closes[-1] > moving_average)

    def rebalance_to(self, target: set[str]) -> None:
        """调仓：目标组合与当前持仓一致则跳过（信号变化时才调整）。"""
        current: set[str] = {
            symbol for symbol, position in self.book.items()
            if position["volume"] > 0
        }
        if target == current:
            self.log("目标组合未变化，跳过调仓")
            return
        equity_est: float = self.capital + self.realized_pnl - self.fee_est
        # 先卖后买：卖出不在目标里的持仓，买入新入选币（等权 equity / TOP_N）
        for research_symbol in sorted(current - target):
            volume: float = self.book[research_symbol]["volume"]
            self.send_market(research_symbol, Direction.SHORT, volume)
        for research_symbol in sorted(target - current):
            price: float = self.last_price.get(research_symbol, 0.0)
            if price <= 0:
                self.log(f"缺少 {research_symbol} 最新价，暂缓买入")
                continue
            buy_volume: float = round(
                equity_est / TOP_N / price, VOLUME_DECIMALS
            )
            self.send_market(research_symbol, Direction.LONG, buy_volume)

    def send_market(self, research_symbol: str, direction: Direction,
                    volume: float) -> str:
        """发市价单（经 MainEngine，被 PaperAccount 接管后在本地仿真撮合）。"""
        if volume <= 0:
            return ""
        request: OrderRequest = OrderRequest(
            symbol=research_to_live(research_symbol),
            exchange=Exchange.GLOBAL,
            direction=direction,
            type=OrderType.MARKET,
            volume=volume,
            price=0.0,
            offset=Offset.OPEN,  # OKX 现货为净持仓模式，开平标志不影响
            reference="GpS123Paper",
        )
        vt_orderid: str = self.main_engine.send_order(request, self.gateway_name)
        self.log(f"委托: {direction.value} {research_symbol} {volume} "
                 f"-> {vt_orderid or '(被拒绝)'}")
        return vt_orderid

    # ------------------------------------------------------------------ 流水
    def load_journal(self) -> list[str]:
        """读取既有交易流水（Path.write_text 全量重写，先载入历史行）。"""
        journal_path: Path = safe_join(JOURNAL_REL)
        if journal_path.is_file():
            return journal_path.read_text(encoding="utf-8").splitlines()
        return [JOURNAL_HEADER]

    def write_journal(self, completed_key: str, hold: bool,
                      top_coins: list[str]) -> None:
        """把本轮判定追加写入交易流水 CSV（全量重写，禁止 open 追加）。"""
        equity_est: float = self.capital + self.realized_pnl - self.fee_est
        self.journal_rows.append(
            f"{datetime.now().isoformat(timespec='seconds')},{completed_key},"
            f"{hold},{'|'.join(sorted(top_coins)) or '(空仓)'},"
            f"{equity_est:.2f},{self.realized_pnl:.2f},{self.fee_est:.2f}"
        )
        journal_path: Path = safe_join(JOURNAL_REL)
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        journal_path.write_text(
            "\n".join(self.journal_rows) + "\n", encoding="utf-8"
        )

    def log(self, message: str) -> None:
        """通过主引擎记录策略日志（EVENT_LOG -> on_log 打印）。"""
        self.main_engine.write_log(message, source="GpS123Paper")
