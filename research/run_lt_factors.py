"""Liu-Tsyvinski 加密原生横截面因子族的批量 IC 测试。

出处：Liu & Tsyvinski, "Risks and Returns of Cryptocurrency"（RFS 2021）。
这是学术界直接在加密市场数据上提出并验证的横截面预测因子族：
短期反转（1周~1月，显著负）、中期动量（3~6月，弱正）、波动率、
Amihud 非流动性、成交额水平与放量、日内振幅、彩票偏好（MAX 效应）。
全部只用日线 OHLCV + 成交额即可构造，本次实现 17 个。

与美股搬来的现成库（Alpha101 实测全弱）相比，这一族的先验更强：
它就是在币圈数据上发现的规律；且我们 GP 挖出的"振幅×动量"主题
与它互相印证。评分/报告/入选走 ic_test.py 共用层。

输出：research/lt_factors/report.md；达标因子并入统一清单。
用法（在仓库根目录运行）：
    D:\\veighna_studio\\python.exe -X utf8 research/run_lt_factors.py
"""

import time
from pathlib import Path

import pandas as pd
import polars as pl

from factor_common import build_factor_table
from ic_test import print_top, promote_factors, score_all, write_report

REPORT_PATH: Path = Path("research/lt_factors/report.md")

# 因子族定义：name -> vnpy 表达式（全部日线可算，口径与管线一致）
LT_FACTORS: dict[str, str] = {
    # ---- 短期反转（论文核心结论：1 周~1 月显著为负）----
    "lt_ret7": "close/ts_delay(close,7)-1",
    "lt_ret14": "close/ts_delay(close,14)-1",
    "lt_ret21": "close/ts_delay(close,21)-1",
    "lt_ret30": "close/ts_delay(close,30)-1",
    # ---- 中期动量（论文：3~6 月弱正）----
    "lt_ret90": "close/ts_delay(close,90)-1",
    "lt_ret180": "close/ts_delay(close,180)-1",
    # ---- 波动率 ----
    "lt_vola7": "ts_std(close/ts_delay(close,1)-1,7)",
    "lt_vola30": "ts_std(close/ts_delay(close,1)-1,30)",
    # ---- Amihud 非流动性（|日收益|/成交额）----
    "lt_amihud7": "ts_mean(abs(close/ts_delay(close,1)-1)/turnover,7)",
    "lt_amihud30": "ts_mean(abs(close/ts_delay(close,1)-1)/turnover,30)",
    # ---- 成交额水平（流动性/规模代理）----
    "lt_dvol30": "ts_mean(turnover,30)",
    # ---- 放量程度 ----
    "lt_vratio": "ts_mean(turnover,7)/ts_mean(turnover,28)",
    "lt_vtrend": "ts_mean(turnover,28)/ts_mean(turnover,90)",
    # ---- 日内振幅（与 GP 挖出的 rng 主题对照）----
    "lt_range7": "ts_mean((high-low)/close,7)",
    "lt_range30": "ts_mean((high-low)/close,30)",
    # ---- 彩票偏好（MAX 效应）----
    "lt_maxret7": "ts_max(close/ts_delay(close,1)-1,7)",
    "lt_minret7": "ts_min(close/ts_delay(close,1)-1,7)",
}


def compute_lt_values(panel: pl.DataFrame) -> dict[str, pd.Series]:
    """逐个计算因子值（vnpy 表达式引擎，与其他测试线同口径）。"""
    from vnpy.alpha.dataset.utility import calculate_by_expression

    values: dict[str, pd.Series] = {}
    for name, formula in LT_FACTORS.items():
        values[name] = pd.Series(calculate_by_expression(panel, formula)["data"])
    return values


def main() -> None:
    """完整流程：面板 -> 因子计算 -> 共用层评分/报告/入选。"""
    t0: float = time.time()
    panel: pl.DataFrame = build_factor_table()
    print(f"[1] 面板 {panel.height} 行 x {panel['vt_symbol'].n_unique()} 币")

    print(f"[2] 计算 {len(LT_FACTORS)} 个 Liu-Tsyvinski 因子 ...")
    values: dict[str, pd.Series] = compute_lt_values(panel)

    print("[3] 三段 IC 评分 ...")
    pdf: pd.DataFrame = panel.select(["datetime", "vt_symbol", "label"]).to_pandas()
    scores: pd.DataFrame = score_all(pdf, values)
    write_report(
        scores, [], "Liu-Tsyvinski 加密原生因子族 IC 测试",
        ["出处：Liu & Tsyvinski (RFS 2021)，全部日线 OHLCV+成交额构造"],
        REPORT_PATH,
    )
    print_top(scores)

    print("[4] 达标因子并入清单 ...")
    promote_factors(scores, LT_FACTORS, "lt", "")
    print(f"完成，总耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
