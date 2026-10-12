"""单因子批量 IC 测试共用层（从 run_alpha101_test.py 泛化抽取）。

任何"因子名 -> 因子值列"的批量测试都可复用：三段日均截面 Rank IC /
ICIR 评分、覆盖率过滤（质检布尔退化因子）、markdown 报告、达标因子
并入统一清单。Alpha101 / Liu-Tsyvinski / 链上 / Alpha191 测试共用（DRY）。

覆盖率质检的由来：布尔/条件式因子（如 Alpha101 的 alpha68）在多数
交易日对全部币输出同一个值，秩相关无定义，会在极少数有效日上产生
荒谬的高 IC 假信号——覆盖率（有效天数占比）低于门槛的因子一律不计。

约定：在仓库根目录运行；报告路径由调用方以字面相对路径给出。
运行环境：D:\\veighna_studio\\python.exe（需 polars / pandas / vnpy）
"""

from pathlib import Path

import pandas as pd
import polars as pl

from factor_common import TEST, TRAIN, VALID, daily_rank_ic_series, save_factors

SEGMENTS: dict[str, tuple[str, str]] = {
    "train": TRAIN,
    "valid": VALID,
    "test": TEST,
}
MIN_VALID_IC: float = 0.02    # 入选清单的验证段日均 Rank IC 门槛
MIN_COVERAGE: float = 0.6     # 有效天数占比门槛
TOP_K: int = 5                # 最多并入清单的因子数量


def score_factor(pdf: pd.DataFrame, name: str, mask: pd.Series) -> tuple[float, float, float]:
    """某因子在某时间段的（日均 Rank IC, ICIR, 覆盖率）。"""
    part: pd.DataFrame = pdf.loc[mask, ["datetime", "vt_symbol", "label", name]]
    total_days: int = int(part["datetime"].nunique())
    ic_series: pd.Series = daily_rank_ic_series(
        part[name], part["label"], part["datetime"]
    ).dropna()
    if ic_series.empty:
        return float("nan"), float("nan"), 0.0
    coverage: float = len(ic_series) / total_days if total_days else 0.0
    icir: float = (
        float(ic_series.mean() / ic_series.std(ddof=1))
        if len(ic_series) > 1 else float("nan")
    )
    return float(ic_series.mean()), icir, coverage


def score_all(pdf: pd.DataFrame, values: dict[str, pd.Series]) -> pd.DataFrame:
    """全部因子 × 三段评分，按测试段 IC 降序。

    pdf 需含 datetime / vt_symbol / label 三列；values 的每个 Series 与
    pdf 行对齐。
    """
    work: pd.DataFrame = pdf.copy()
    for name, series in values.items():
        work[name] = pd.Series(series)  # 兼容 polars Series / pandas Series / ndarray
    rows: list[dict] = []
    for name in values:
        entry: dict = {"factor": name}
        for seg_name, segment in SEGMENTS.items():
            mask: pd.Series = (work["datetime"] >= segment[0]) & (
                work["datetime"] <= segment[1] + " 23:59:59"
            )
            ic_value, icir_value, coverage = score_factor(work, name, mask)
            entry[f"{seg_name}_ic"] = round(ic_value, 6)
            entry[f"{seg_name}_icir"] = round(icir_value, 4)
            entry[f"{seg_name}_cov"] = round(coverage, 3)
        rows.append(entry)
    result: pd.DataFrame = pd.DataFrame(rows)
    return result.sort_values("test_ic", ascending=False, na_position="last")


def write_report(scores: pd.DataFrame, failed: list[str], title: str,
                 notes: list[str], report_path: Path) -> None:
    """评分表写成 markdown 报告（报告路径由调用方给定）。"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        f"# {title}",
        "",
        f"- 因子总数：{len(scores) + len(failed)}（计算失败 {len(failed)} 个：{failed}）",
        "- 标签：未来 5 日收益（全部测试线同口径）",
        f"- 分段：训练 {TRAIN[0]}~{TRAIN[1]} / 验证 {VALID[0]}~{VALID[1]} / 测试 {TEST[0]}~{TEST[1]}",
        *[f"- {note}" for note in notes],
        "",
        "| 因子 | 覆盖率 t/v/t | 训练 IC/ICIR | 验证 IC/ICIR | 测试 IC/ICIR |",
        "|---|---|---|---|---|",
    ]
    for _, row in scores.iterrows():
        coverage_text: str = (
            f"{row['train_cov']:.0%}/{row['valid_cov']:.0%}/{row['test_cov']:.0%}"
        )
        lines.append(
            f"| {row['factor']} | {coverage_text} "
            f"| {row['train_ic']:+.4f} / {row['train_icir']:+.2f} "
            f"| {row['valid_ic']:+.4f} / {row['valid_icir']:+.2f} "
            f"| {row['test_ic']:+.4f} / {row['test_icir']:+.2f} |"
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已写入 {report_path}")


def promote_factors(scores: pd.DataFrame, formulas: dict[str, str],
                    source: str, prefix: str) -> list[dict]:
    """验证段达标（IC 门槛 + 覆盖率门槛 + 测试段同号）的头部因子并入清单。"""
    qualified: pd.DataFrame = scores[
        (scores["valid_ic"].abs() >= MIN_VALID_IC)
        & (scores["valid_ic"] * scores["test_ic"] > 0)
        & (scores["valid_cov"] >= MIN_COVERAGE)
        & (scores["test_cov"] >= MIN_COVERAGE)
    ]
    # 按验证段 |IC| 取头部：负向因子（反向使用）与正向同等对待
    qualified = qualified.reindex(
        qualified["valid_ic"].abs().sort_values(ascending=False).index
    ).head(TOP_K)
    factors: list[dict] = []
    for _, row in qualified.iterrows():
        factor_name: str = f"{prefix}_{row['factor']}" if prefix else str(row["factor"])
        weight: float = 1.0 if row["valid_ic"] > 0 else -1.0
        factors.append({
            "name": factor_name,
            "source": source,
            "vnpy_expr": formulas[row["factor"]],
            "weight": weight,
            "train_ic": float(row["train_ic"]),
            "valid_ic": float(row["valid_ic"]),
            "test_ic": float(row["test_ic"]),
        })
        print(f"  入选 {row['factor']}: valid={row['valid_ic']:+.4f} "
              f"test={row['test_ic']:+.4f}")
    save_factors(source, factors, meta={
        "tool": source,
        "tested": int(len(scores)),
        "min_valid_ic": MIN_VALID_IC,
        "min_coverage": MIN_COVERAGE,
    })
    return factors


def print_top(scores: pd.DataFrame, top_n: int = 15) -> None:
    """stdout 打印头部因子。"""
    print(f"\n头部 {top_n} 名（按测试段 IC 降序）：")
    for _, row in scores.head(top_n).iterrows():
        print(f"  {row['factor']:<16} train={row['train_ic']:+.4f} "
              f"valid={row['valid_ic']:+.4f} test={row['test_ic']:+.4f} "
              f"cov={row['test_cov']:.0%}")
