"""Alpha101（WorldQuant 经典论文因子）在币圈的批量 IC 测试。

背景：Alpha101 因子库从未在加密货币数据上检验过（2026-10-07 首测，
结论：最好的真因子 test IC 仅 0.02~0.05，远弱于自挖因子）。
评分、报告、清单入选逻辑已抽取到 ic_test.py 共用层，本脚本只负责
三件事：构建面板（补 vwap 列）、逐因子计算、调用共用层。

关键适配：Alpha101 大量使用 vwap（成交量加权均价），行情面板没有这一列。
加密 7x24 连续交易、单根日线内无时段缺口，用 turnover/volume（成交额/
成交量）合成日频 vwap 口径成立；volume 为 0 的行保留空值。

输出：research/alpha101_test/report.md；达标因子并入统一清单。
用法（在仓库根目录运行）：
    D:\\veighna_studio\\python.exe -X utf8 research/run_alpha101_test.py
"""

import time
from pathlib import Path

import pandas as pd
import polars as pl

from factor_common import TEST, TRAIN, VALID, build_factor_table
from ic_test import print_top, promote_factors, score_all, write_report

REPORT_PATH: Path = Path("research/alpha101_test/report.md")


def build_alpha101_panel() -> pl.DataFrame:
    """Alpha101 专用面板：在共用面板基础上补 vwap 列。"""
    panel: pl.DataFrame = build_factor_table()
    return panel.with_columns(
        pl.when(pl.col("volume") > 0)
        .then(pl.col("turnover") / pl.col("volume"))
        .otherwise(None)
        .alias("vwap")
    )


def compute_factor_values(panel: pl.DataFrame,
                          formulas: dict[str, str]) -> dict[str, pl.Series]:
    """逐个因子用 vnpy 表达式引擎计算（单进程、内存友好）。"""
    from vnpy.alpha.dataset.utility import calculate_by_expression

    values: dict[str, pl.Series] = {}
    failed: list[str] = []
    t0: float = time.time()
    total: int = len(formulas)
    for index, (name, formula) in enumerate(formulas.items(), 1):
        try:
            values[name] = calculate_by_expression(panel, formula)["data"]
        except Exception as error:  # 单因子失败不阻断批次
            failed.append(name)
            print(f"  [失败] {name}: {str(error)[:80]}")
        if index % 20 == 0:
            print(f"  进度 {index}/{total}（{time.time() - t0:.0f}s）")
    if failed:
        print(f"共 {len(failed)} 个因子计算失败: {failed}")
    return values


def main() -> None:
    """完整流程：面板 -> 逐因子计算 -> 共用层评分/报告/入选。"""
    t0: float = time.time()
    panel: pl.DataFrame = build_alpha101_panel()
    print(f"[1] 面板 {panel.height} 行 x {panel['vt_symbol'].n_unique()} 币")

    from vnpy.alpha.dataset.datasets.alpha_101 import Alpha101
    dataset = Alpha101(
        df=panel, train_period=TRAIN, valid_period=VALID, test_period=TEST
    )
    formulas: dict[str, str] = dict(dataset.feature_expressions)
    print(f"[2] Alpha101 共 {len(formulas)} 个因子，逐个计算 ...")
    values: dict[str, pl.Series] = compute_factor_values(panel, formulas)

    print("[3] 三段 IC 评分 ...")
    pdf: pd.DataFrame = panel.select(["datetime", "vt_symbol", "label"]).to_pandas()
    scores: pd.DataFrame = score_all(
        pdf, {name: pd.Series(series) for name, series in values.items()}
    )
    failed: list[str] = [name for name in formulas if name not in values]
    write_report(
        scores, failed, "Alpha101 币圈批量 IC 测试报告",
        ["vwap 适配：turnover / volume（加密 7x24，单日无时段缺口）"],
        REPORT_PATH,
    )
    print_top(scores)

    print("[4] 达标因子并入清单 ...")
    promote_factors(scores, formulas, "alpha101", "a101")
    print(f"完成，总耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
