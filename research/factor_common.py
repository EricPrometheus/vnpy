"""因子挖掘共用层：数据加载、基础特征、标签、IC 计算、因子清单读写。

被 run_gp_mining.py（gplearn 挖掘）和 eval_mined_factors.py（统一评估）共享，
保证两边的因子口径与 run_crypto_pipeline.py 完全一致（DRY）。

目录定位（防路径穿越）：
- 根目录 = 环境变量 VNPY_RESEARCH_ROOT，缺省回落到 "运行目录/research"；
- 所有文件读写都必须经过 safe_join()：显式拒绝 ".."、规范化
  （normpath/abspath）并校验最终路径仍位于根目录之内（commonpath）。

运行环境：D:\\veighna_studio\\python.exe（需 polars / numpy / pandas / vnpy）
"""

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ENV_RESEARCH_ROOT: str = "VNPY_RESEARCH_ROOT"


def _resolve_research_root() -> Path:
    """定位并校验 research 根目录：环境变量优先，其次"运行目录/research"。"""
    env_value: str | None = os.environ.get(ENV_RESEARCH_ROOT)
    candidate: Path
    if env_value:
        candidate = Path(env_value)
    else:
        candidate = Path.cwd() / "research"
    resolved: Path = Path(os.path.abspath(os.path.normpath(str(candidate))))
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"research 根目录不存在: {resolved}；"
            f"请在仓库根目录运行，或设置环境变量 {ENV_RESEARCH_ROOT}"
        )
    return resolved


RESEARCH_ROOT: Path = _resolve_research_root()


def safe_join(relative: str) -> Path:
    """把字面相对路径安全地拼到根目录下，拒绝任何形式的向上跳转。"""
    parts: list[str] = relative.replace("\\", "/").split("/")
    if any(part in ("..", "") for part in parts):
        raise ValueError(f"非法相对路径（含 .. 或空段）: {relative}")
    base_abs: str = os.path.abspath(os.path.normpath(str(RESEARCH_ROOT)))
    joined_abs: str = os.path.abspath(os.path.normpath(os.path.join(base_abs, *parts)))
    if os.path.commonpath([base_abs, joined_abs]) != base_abs:
        raise ValueError(f"路径越出根目录，已拒绝: {relative}")
    return Path(joined_abs)


def daily_dir() -> Path:
    """日线 parquet 目录（相对 research 根目录）。"""
    return safe_join("alpha_lab/daily")


def manifest_path() -> Path:
    """因子清单文件路径（相对 research 根目录）。"""
    return safe_join("mined_manifest.json")


def baseline_signal_path() -> Path:
    """Alpha158 基线信号文件路径（相对 research 根目录）。"""
    return safe_join("alpha_lab/signal/lgb_alpha158_crypto_v1.parquet")


# 三段划分与 run_crypto_pipeline.py 保持一致
TRAIN: tuple[str, str] = ("2023-01-01", "2024-12-31")
VALID: tuple[str, str] = ("2025-01-01", "2025-12-31")
TEST: tuple[str, str] = ("2026-01-01", "2026-09-28")

# 标签：未来 5 日收益，与现有管线同款表达式，保证口径可比
LABEL_VNPY: str = "ts_delay(close, -6) / ts_delay(close, -1) - 1"
_LABEL_EXPR: pl.Expr = (
    pl.col("close").shift(-6) / pl.col("close").shift(-1) - 1
).over("vt_symbol")

# 基础特征（挖掘的"原料"）：每个都必须能无损翻译成 vnpy 表达式，
# 这样挖出的公式才能回填现有管线的表达式引擎。
# name -> (polars 表达式, vnpy 表达式)
FEATURES: dict[str, tuple[pl.Expr, str]] = {
    "ret1": (
        pl.col("close").pct_change(1).over("vt_symbol"),
        "close/ts_delay(close,1)-1",
    ),
    "mom5": (
        pl.col("close").pct_change(5).over("vt_symbol"),
        "close/ts_delay(close,5)-1",
    ),
    "mom10": (
        pl.col("close").pct_change(10).over("vt_symbol"),
        "close/ts_delay(close,10)-1",
    ),
    "mom20": (
        pl.col("close").pct_change(20).over("vt_symbol"),
        "close/ts_delay(close,20)-1",
    ),
    "mom60": (
        pl.col("close").pct_change(60).over("vt_symbol"),
        "close/ts_delay(close,60)-1",
    ),
    "vola20": (
        pl.col("close").pct_change(1).rolling_std(20).over("vt_symbol"),
        "ts_std(close/ts_delay(close,1)-1,20)",
    ),
    "rng20": (
        ((pl.col("high") - pl.col("low")) / pl.col("close"))
        .rolling_mean(20)
        .over("vt_symbol"),
        "ts_mean((high-low)/close,20)",
    ),
    "vr20": (
        (pl.col("volume") / pl.col("volume").rolling_mean(20)).over("vt_symbol"),
        "volume/ts_mean(volume,20)",
    ),
}

FEATURE_NAMES: list[str] = list(FEATURES)


def load_panel() -> pl.DataFrame:
    """读取全部币种日线 parquet 并纵向合并，附 vt_symbol 列。"""
    data_dir: Path = daily_dir()
    files: list[Path] = sorted(data_dir.glob("*.OKX.parquet"))
    if not files:
        raise FileNotFoundError(f"未找到日线数据: {data_dir}")
    dfs: list[pl.DataFrame] = []
    for file_path in files:
        checked: Path = safe_join(str(file_path.relative_to(RESEARCH_ROOT)))
        if not checked.is_file():
            raise FileNotFoundError(f"数据文件校验失败: {file_path}")
        df: pl.DataFrame = pl.read_parquet(checked)
        dfs.append(df.with_columns(pl.lit(checked.stem).alias("vt_symbol")))
    return pl.concat(dfs, how="vertical").sort(["vt_symbol", "datetime"])


def build_factor_table() -> pl.DataFrame:
    """加载行情并计算基础特征与标签，去掉预热期/尾部无法计算的行。"""
    df: pl.DataFrame = load_panel()
    for name, (expr, _vnpy) in FEATURES.items():
        df = df.with_columns(expr.alias(name))
    df = df.with_columns(_LABEL_EXPR.alias("label"))
    df = df.drop_nulls(subset=[*FEATURE_NAMES, "label"])
    # 后续数组操作按 (日期, 币) 排序，保证同一天的数据在内存中连续
    return df.sort(["datetime", "vt_symbol"])


def date_boundaries(dates: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """把排序后的日期数组转成 (日期编码, 每天的 [起,止) 边界)。"""
    unique: np.ndarray
    inverse: np.ndarray
    unique, inverse = np.unique(dates, return_inverse=True)
    inverse = inverse.astype(np.int32)
    starts: np.ndarray = np.searchsorted(inverse, np.arange(len(unique)))
    ends: np.ndarray = np.searchsorted(inverse, np.arange(len(unique)), side="right")
    return inverse, list(zip(starts.tolist(), ends.tolist()))


def mean_rank_ic(pred: np.ndarray, target: np.ndarray,
                 codes: np.ndarray, boundaries: list[tuple[int, int]]) -> float:
    """日均截面 Rank IC：每天内因子值与标签的秩相关，再对日期取平均。

    这是因子预测能力的标准打分，比整体相关系数更贴近横截面选币场景。
    """
    pred_clean: np.ndarray = np.nan_to_num(pred, nan=0.0, posinf=1e9, neginf=-1e9)
    ics: list[float] = []
    for lo, hi in boundaries:
        if hi - lo < 3:
            continue
        p: np.ndarray = pred_clean[lo:hi]
        t: np.ndarray = target[lo:hi]
        if not np.isfinite(p).all():
            continue
        rank_p: np.ndarray = np.argsort(np.argsort(p)).astype(np.float64)
        rank_t: np.ndarray = np.argsort(np.argsort(t)).astype(np.float64)
        if rank_p.std() == 0:
            continue
        ics.append(float(np.corrcoef(rank_p, rank_t)[0, 1]))
    return float(np.mean(ics)) if ics else 0.0


def daily_rank_ic_series(values: pd.Series, target: pd.Series,
                         dates: pd.Series) -> pd.Series:
    """逐日截面 Rank IC 序列（pandas 版，供评估脚本算 ICIR）。"""
    frame: pd.DataFrame = pd.DataFrame(
        {"date": dates, "value": values, "target": target}
    ).dropna()

    def _ic(group: pd.DataFrame) -> float:
        if len(group) < 3 or group["value"].std() == 0:
            return np.nan
        return group["value"].rank().corr(group["target"].rank())

    return frame.groupby("date").apply(_ic)


def segment_mask(segment: tuple[str, str]) -> pl.Expr:
    """生成某时间段（含端点）的行过滤表达式。"""
    start: datetime = datetime.strptime(segment[0], "%Y-%m-%d")
    end: datetime = datetime.strptime(segment[1], "%Y-%m-%d")
    return (pl.col("datetime") >= pl.datetime(start.year, start.month, start.day)) & (
        pl.col("datetime") <= pl.datetime(end.year, end.month, end.day, hour=23)
    )


def load_manifest() -> dict:
    """读取因子清单（不存在则返回空结构）。"""
    path: Path = manifest_path()
    if not path.exists():
        return {"factors": []}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_factors(source: str, factors: list[dict], meta: dict) -> None:
    """把某一来源（gp / alphagen）的因子写入清单，覆盖同来源旧条目。"""
    manifest: dict = load_manifest()
    kept: list[dict] = [
        x for x in manifest.get("factors", []) if x.get("source") != source
    ]
    manifest["factors"] = kept + factors
    manifest["meta"] = {**manifest.get("meta", {}), source: meta}
    manifest["updated"] = datetime.now().isoformat(timespec="seconds")
    text: str = json.dumps(manifest, ensure_ascii=False, indent=2)
    manifest_path().write_text(text, encoding="utf-8")
