"""GTJA191（国泰君安 191 因子）适配包：算子扩展 + 表达式翻译器 + 公式抓取。

出处：Alpha191 出自国泰君安 2017 年研报《数量化专题：基于短周期价量特征
的多因子选股体系》（常被误传为国信证券）。公式全文抓取自 BigQuant Wiki
的公开整理页，经 jsdelivr 无关、直连可达。

引擎扩展机制：vnpy.alpha.dataset.utility.register_functions() 是官方的
自定义算子注册入口，注册名在表达式求值时优先于内置算子（utility.py 中
d.update(EXPRESSION_FUNCTIONS) 位于内置 from-import 之后）。本模块注册：
- sma(x, n, m)：中国式递归平滑 S_t = (m*x_t + (n-m)*S_{t-1})/n，
  数学上等价 ewm_mean(alpha=m/n, adjust=False)
- wma(x, d)：权重 1..d 线性递增的加权移动平均
- gt/lt/ge/le/eq：逐元素比较，输出 1.0/0.0（供三元条件、SUMIF、COUNT 展开）

翻译规则（GTJA 大写算子 -> vnpy 引擎，见 translate_gtja）：
- 一一映射：DELAY->ts_delay、MEAN->ts_mean、CORR->ts_corr、RANK->cs_rank 等
- MAX/MIN 双义：第二参数为纯整数按滚动窗口 ts_max/ts_min，否则逐元素
  greater/less（vnpy 的 less/greater 即逐元素最小/最大）
- SUMIF(a,d,c) -> ts_sum(a*c, d)、COUNT(c,d) -> ts_sum(c, d)（c 为 1/0）
- REGBETA(x,y,d) -> ts_cov/(ts_std*ts_std)；REGBETA(x,d) -> ts_slope
- `c ? x : y` -> ((x)*c+(y)*(1-c))，c 为比较算子输出的 1/0
- 字段：VWAP->vwap（面板需补 turnover/volume 列）、AMOUNT->turnover、
  RET -> close/ts_delay(close,1)-1

约定：在仓库根目录运行；公式缓存以字面相对路径写入。
"""

import html
import json
import re
from pathlib import Path

import numpy as np
import polars as pl
import requests

from vnpy.alpha.dataset.utility import DataProxy, register_functions

GTJA_URL: str = "https://bigquant.com/wiki/doc/Pyf0TYya6H"
USER_AGENT: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
MIN_FORMULAS: int = 180  # 公式条数下限（略容忍个别解析失败）


# ---------------- 自定义算子（DataProxy 风格，与内置 ts_* 一致） ----------------

def _pair(left: DataProxy | float, right: DataProxy | float) -> pl.DataFrame:
    """把两个操作数对齐成带 data/data_right 两列的面板。"""
    left_df: pl.DataFrame = (
        left.df if isinstance(left, DataProxy)
        else pl.DataFrame(columns=["datetime", "vt_symbol", "data"])
    )
    if isinstance(right, DataProxy):
        return left_df.join(right.df, on=["datetime", "vt_symbol"], how="inner")
    return left_df.with_columns(pl.lit(float(right)).alias("data_right"))


def _compare(left: DataProxy | float, right: DataProxy | float,
             op: str) -> DataProxy:
    """逐元素比较，输出 1.0/0.0 的 DataProxy。"""
    merged: pl.DataFrame = _pair(left, right)
    ops: dict[str, pl.Expr] = {
        ">": pl.col("data") > pl.col("data_right"),
        "<": pl.col("data") < pl.col("data_right"),
        ">=": pl.col("data") >= pl.col("data_right"),
        "<=": pl.col("data") <= pl.col("data_right"),
        "==": (pl.col("data") - pl.col("data_right")).abs() < 1e-12,
    }
    df: pl.DataFrame = merged.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        ops[op].cast(pl.Float64).alias("data"),
    )
    return DataProxy(df)


def gt(a: DataProxy | float, b: DataProxy | float) -> DataProxy:
    """逐元素大于，输出 1.0/0.0。"""
    return _compare(a, b, ">")


def lt(a: DataProxy | float, b: DataProxy | float) -> DataProxy:
    """逐元素小于，输出 1.0/0.0。"""
    return _compare(a, b, "<")


def ge(a: DataProxy | float, b: DataProxy | float) -> DataProxy:
    """逐元素大于等于，输出 1.0/0.0。"""
    return _compare(a, b, ">=")


def le(a: DataProxy | float, b: DataProxy | float) -> DataProxy:
    """逐元素小于等于，输出 1.0/0.0。"""
    return _compare(a, b, "<=")


def eq(a: DataProxy | float, b: DataProxy | float) -> DataProxy:
    """逐元素（近似）相等，输出 1.0/0.0。"""
    return _compare(a, b, "==")


def sma(feature: DataProxy, n: float, m: float) -> DataProxy:
    """中国式递归平滑（研报口径），等价 ewm_mean(alpha=m/n)。"""
    alpha: float = float(m) / float(n)
    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").ewm_mean(alpha=alpha, adjust=False).over("vt_symbol").alias("data"),
    )
    return DataProxy(df)


def wma(feature: DataProxy, d: int) -> DataProxy:
    """权重 1..d 线性递增的加权移动平均（近期权重最大）。"""
    window: int = int(d)
    weights: np.ndarray = np.arange(1, window + 1, dtype=np.float64)

    def _wma(series: pl.Series) -> float:
        return float(np.average(series.to_numpy(), weights=weights))

    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").rolling_map(_wma, window_size=window, min_samples=window)
        .over("vt_symbol").alias("data"),
    )
    return DataProxy(df)


CUSTOM_OPERATORS: list = [gt, lt, ge, le, eq, sma, wma]


def install_operators() -> None:
    """把自定义算子注册进 vnpy 表达式引擎（官方入口，进程内生效）。"""
    register_functions(CUSTOM_OPERATORS)


# ---------------- GTJA -> vnpy 表达式翻译器 ----------------

FIELD_MAP: dict[str, str] = {
    "CLOSE": "close",
    "OPEN": "open",
    "HIGH": "high",
    "LOW": "low",
    "VOLUME": "volume",
    "VWAP": "vwap",
    "AMOUNT": "turnover",
    "RET": "(close / ts_delay(close, 1) - 1)",
}

SIMPLE_OPS: dict[str, str] = {
    "DELAY": "ts_delay",
    "MEAN": "ts_mean",
    "TSMean".upper(): "ts_mean",
    "STD": "ts_std",
    "SUM": "ts_sum",
    "CORR": "ts_corr",
    "COVIANCE": "ts_cov",
    "TSRANK": "ts_rank",
    "TSMAX": "ts_max",
    "TSMIN": "ts_min",
    "DELTA": "ts_delta",
    "DECAYLINEAR": "ts_decay_linear",
    "RANK": "cs_rank",
    "ABS": "abs",
    "LOG": "log",
    "SIGN": "sign",
    "SMA": "sma",
    "WMA": "wma",
    "REGRESI": "ts_resi",
}

_TOKEN_RE: re.Pattern = re.compile(
    r"\d+\.?\d*|[A-Za-z_]+|>=|<=|==|[()?,:<>+\-*/=]"
)


class GtjaTranslator:
    """递归下降翻译器：GTJA 大写表达式 -> vnpy 表达式文本。"""

    def __init__(self, text: str) -> None:
        self.tokens: list[str] = _TOKEN_RE.findall(text)
        self.pos: int = 0

    def _peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _next(self) -> str:
        token: str = self.tokens[self.pos]
        self.pos += 1
        return token

    def translate(self) -> str:
        text: str = self._ternary()
        if self.pos != len(self.tokens):
            raise ValueError(f"翻译不完整，剩余 token: {self.tokens[self.pos:]}")
        return text

    def _ternary(self) -> str:
        condition: str = self._compare()
        if self._peek() == "?":
            self._next()
            branch_true: str = self._ternary()
            self._next()  # 跳过 ':'
            branch_false: str = self._ternary()
            return (
                f"(({branch_true}) * ({condition}) "
                f"+ ({branch_false}) * (1 - ({condition})))"
            )
        return condition

    def _compare(self) -> str:
        left: str = self._sum()
        op: str | None = self._peek()
        if op in (">", "<", ">=", "<=", "=="):
            self._next()
            right: str = self._sum()
            return f"{op}({left}, {right})" if op in ("==",) else f"{op[:1]}({left}, {right})"
        return left

    def _sum(self) -> str:
        left: str = self._product()
        while self._peek() in ("+", "-"):
            op: str = self._next()
            left = f"({left} {op} {self._product()})"
        return left

    def _product(self) -> str:
        left: str = self._factor()
        while self._peek() in ("*", "/"):
            op: str = self._next()
            left = f"({left} {op} {self._factor()})"
        return left

    def _factor(self) -> str:
        if self._peek() == "-":
            self._next()
            return f"(-{self._factor()})"
        return self._atom()

    def _atom(self) -> str:
        token: str = self._next()
        if token == "(":
            inner: str = self._ternary()
            self._next()  # 跳过 ')'
            return f"({inner})"
        if re.fullmatch(r"\d+\.?\d*", token):
            return token
        if token in FIELD_MAP:
            return FIELD_MAP[token]
        if token.upper() in SIMPLE_OPS or token.upper() in ("MAX", "MIN", "SUMIF", "COUNT", "REGBETA"):
            return self._call(token.upper())
        raise ValueError(f"未知标识符: {token}")

    def _call(self, name: str) -> str:
        self._next()  # 跳过 '('
        args: list[str] = []
        if self._peek() != ")":
            args.append(self._ternary())
            while self._peek() == ",":
                self._next()
                args.append(self._ternary())
        self._next()  # 跳过 ')'
        return self._map_call(name, args)

    def _map_call(self, name: str, args: list[str]) -> str:
        """把 GTJA 算子调用映射为 vnpy 表达式片段。"""
        if name in ("MAX", "MIN"):
            rolling: str = "ts_max" if name == "MAX" else "ts_min"
            elementwise: str = "greater" if name == "MAX" else "less"
            if len(args) == 2 and re.fullmatch(r"\d+\.?\d*", args[1]):
                return f"{rolling}({args[0]}, {args[1]})"
            return f"{elementwise}({args[0]}, {args[1]})"
        if name == "SUMIF":  # 条件 c 为 1/0：条件内求和 = (a*c) 的滚动和
            return f"ts_sum(({args[0]}) * ({args[2]}), {args[1]})"
        if name == "COUNT":  # 条件为 1/0：计数 = 滚动和
            return f"ts_sum({args[0]}, {args[1]})"
        if name == "REGBETA":
            if len(args) == 2:  # 对时间回归的斜率
                return f"ts_slope({args[0]}, {args[1]})"
            return (
                f"(ts_cov({args[0]}, {args[1]}, {args[2]}) / "
                f"(ts_std({args[1]}, {args[2]}) * ts_std({args[1]}, {args[2]})))"
            )
        if name in SIMPLE_OPS:
            args_text: str = ", ".join(args)
            return f"{SIMPLE_OPS[name]}({args_text})"
        raise ValueError(f"未知算子: {name}")


def translate_gtja(formula: str) -> str:
    """GTJA 大写表达式 -> vnpy 表达式文本。"""
    return GtjaTranslator(formula).translate()


# ---------------- 公式抓取（BigQuant Wiki，带本地缓存） ----------------

def fetch_gtja_formulas(cache_path: Path) -> dict[str, str]:
    """抓取 191 条公式（HTML 转义 -> 去标签 -> 按 AlphaN: 切分），带缓存。"""
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    response: requests.Response = requests.get(
        GTJA_URL, headers={"User-Agent": USER_AGENT}, timeout=60
    )
    response.raise_for_status()
    text: str = html.unescape(response.text)
    text = re.sub(r"<[^>]+>", " ", text)  # 去 HTML 标签

    parts: list[str] = re.split(r"Alpha(\d+)\s*[:：]", text)
    formulas: dict[str, str] = {}
    for index in range(1, len(parts) - 1, 2):
        number: int = int(parts[index])
        body: str = parts[index + 1]
        body = body.replace("\\n", " ").replace("```", " ").replace("\\", " ")
        body = re.sub(r"\s+", " ", body).strip().strip("`*").strip()
        if not body:
            continue
        formulas[f"alpha{number}"] = body

    if len(formulas) < MIN_FORMULAS:
        raise ValueError(f"公式解析数量不足: {len(formulas)} < {MIN_FORMULAS}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(formulas, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return formulas
