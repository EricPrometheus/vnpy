"""GTJA191（国泰君安 191 短周期价量因子）在币圈的批量 IC 测试。

出处更正：GTJA191 常被误传为国信证券因子，实为国泰君安 2017 年研报。
预期管理：A 股复现研究显示 2018-2025 年沪深 300 仅约 8 个因子仍保持
平均 IC > 0.02——测试目的是"有没有意外惊喜"，而非期待批量可用。

路线说明：表达式翻译路线（gtja191_kit.py）受公式源质量限制（公开页面
最多 52 条完整），改为调用 vibe-trading-ai 包内置的因子动物园——191 个
GTJA 因子均为现成 pandas 实现（compute(panel) 宽表进宽表出，第三方代码
只负责算值，不改不发布）。IC 评分/报告走 ic_test.py 共用层。

技术约定：该包的 panel 是"字段名 -> 宽表（date x 币）"的字典；我们用
vnpy 面板 pivot 出 7 张宽表（close/open/high/low/volume/amount/vwap），
算完的因子宽表再 stack 回长表与标签对齐。

输出：research/gtja191/report.md（不并入清单：第三方实现无 vnpy 表达式，
若证实有效需另行工程化回填）。
用法（在仓库根目录运行）：
    D:\\veighna_studio\\python.exe -X utf8 research/run_gtja191_test.py
"""

import glob
import time
from pathlib import Path

import pandas as pd
import polars as pl

from factor_common import build_factor_table
from ic_test import print_top, score_all, write_report

REPORT_PATH: Path = Path("research/gtja191/report.md")
PANEL_FIELDS: dict[str, str] = {
    # 包约定字段名 -> vnpy 面板列名（amount 即成交额 turnover）
    "close": "close",
    "open": "open",
    "high": "high",
    "low": "low",
    "volume": "volume",
    "amount": "turnover",
}
ID_PREFIX: str = "gtja191_"


def list_gtja_ids() -> list[str]:
    """从包目录枚举 gtja191 域全部因子 id（命名规律 gtja191_NNN）。"""
    import src.factors.registry as registry_module

    zoo: str = os_path_join(
        os_path_dirname(registry_module.__file__), "zoo", "gtja191"
    )
    ids: list[str] = []
    for path in sorted(glob.glob(os_path_join(zoo, "alpha_*.py"))):
        number: str = Path(path).stem.split("_")[-1]
        ids.append(f"{ID_PREFIX}{number}")
    return ids


def os_path_dirname(path: str) -> str:
    """os.path.dirname 的薄封装（避免顶部多一个 import 别名）。"""
    import os

    return os.path.dirname(path)


def os_path_join(*parts: str) -> str:
    """os.path.join 的薄封装。"""
    import os

    return os.path.join(*parts)


def build_wide_tables(panel: pl.DataFrame) -> dict[str, pd.DataFrame]:
    """长表面板 -> 包约定的宽表字典（index=datetime, columns=vt_symbol）。"""
    pdf: pd.DataFrame = panel.to_pandas()
    tables: dict[str, pd.DataFrame] = {}
    for field, column in {**PANEL_FIELDS, "vwap": "vwap"}.items():
        tables[field] = pdf.pivot(
            index="datetime", columns="vt_symbol", values=column
        )
    return tables


def to_long_series(wide: pd.DataFrame,
                   base_index: pd.MultiIndex) -> pd.Series:
    """因子宽表 -> 与标签行序对齐的长表 Series（reindex 对齐）。"""
    stacked: pd.Series = wide.stack(future_stack=True)
    stacked.index.names = ["datetime", "vt_symbol"]
    return stacked.reindex(base_index).reset_index(drop=True)


def main() -> None:
    """完整流程：宽表 -> 逐因子调用第三方实现 -> 共用层评分/报告。"""
    t0: float = time.time()
    from src.factors.registry import Registry

    registry = Registry()
    ids: list[str] = list_gtja_ids()
    print(f"[1] 注册表就绪，GTJA 因子 {len(ids)} 个")

    panel: pl.DataFrame = build_factor_table().with_columns(
        pl.when(pl.col("volume") > 0)
        .then(pl.col("turnover") / pl.col("volume"))
        .otherwise(None)
        .alias("vwap")
    )
    pdf: pd.DataFrame = panel.select(
        ["datetime", "vt_symbol", "label"]
    ).to_pandas()
    base_index: pd.MultiIndex = pdf.set_index(["datetime", "vt_symbol"]).index
    tables: dict[str, pd.DataFrame] = build_wide_tables(panel)
    print(f"[2] 宽表就绪 {tables['close'].shape}，逐因子计算 ...")

    values: dict[str, pd.Series] = {}
    failed: list[str] = []
    for index, alpha_id in enumerate(ids, 1):
        try:
            wide: pd.DataFrame = registry.compute(alpha_id, tables)
            values[alpha_id] = to_long_series(wide, base_index)
        except Exception as error:  # 单因子失败不阻断批次
            failed.append(alpha_id)
            print(f"  [失败] {alpha_id}: {str(error)[:70]}")
        if index % 40 == 0:
            print(f"  进度 {index}/{len(ids)}（{time.time() - t0:.0f}s）")
    if failed:
        print(f"共 {len(failed)} 个因子失败: {failed}")

    print("[3] 三段 IC 评分 ...")
    scores: pd.DataFrame = score_all(pdf, values)
    write_report(
        scores, failed, "GTJA191 币圈批量 IC 测试报告",
        ["出处：国泰君安 2017 研报；因子值由 vibe-trading-ai 包的 pandas 实现计算",
         "A 股 2018-2025 仅约 8 个因子仍有效（外部研究），预期已放低",
         "第三方实现无 vnpy 表达式，本测试不并入统一清单"],
        REPORT_PATH,
    )
    print_top(scores)
    print(f"完成，总耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
