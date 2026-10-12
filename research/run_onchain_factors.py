"""CoinMetrics 链上因子批量 IC 测试（唯一独立于价量的新信息维度）。

数据：research/coinmetrics/*.parquet（社区版核心指标：活跃地址数
AdrActCnt、交易笔数 TxCnt、市值 CapMrktCurUSD；转账额类已移入付费层，
故 NVT 因子用"笔均市值"代理）。

因子族（全部 polars 原生计算，与价量无共线性来源）：
- o_addr_chg7/28：活跃地址数 7/28 日变化率（网络用户扩张/收缩）
- o_addr_rel90：活跃度相对自身 90 日均值的水平
- o_tx_chg28：交易笔数 28 日变化率（链上使用强度）
- o_nvt30_log：ln(市值 / 30 日均笔数)——NVT 估值的笔数代理，类比"每用户市值"
- o_cap_log：ln(市值)——规模因子（已知加密市场存在小市值溢价）

注意：链上列不在 AlphaLab 日线面板中，vnpy 表达式引擎无法计算，
因此本测试只评分与报告，因子不并入 mined_manifest.json（若后续
证实有增量信息，再把链上列并入 AlphaLab 存储是工程项）。

输出：research/onchain_factors/report.md
用法（在仓库根目录运行）：
    D:\\veighna_studio\\python.exe -X utf8 research/run_onchain_factors.py
"""

from pathlib import Path

import pandas as pd
import polars as pl

from factor_common import build_factor_table
from ic_test import print_top, score_all, write_report

CM_DIR: Path = Path("research/coinmetrics")
REPORT_PATH: Path = Path("research/onchain_factors/report.md")


CM_COLS: list[str] = ["date", "AdrActCnt", "CapMrktCurUSD", "TxCnt", "PriceUSD"]


def load_onchain() -> pl.DataFrame:
    """读取全部链上 parquet 并计算因子列（按币分组滚动）。"""
    frames: list[pl.DataFrame] = []
    for path in sorted(CM_DIR.glob("*.parquet")):
        frame: pl.DataFrame = pl.read_parquet(path)
        for col in CM_COLS[1:]:  # 某些币缺列：统一补空，保证 concat 列序一致
            if col not in frame.columns:
                frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias(col))
        frames.append(
            frame.with_columns(pl.lit(path.stem).alias("coin")).select(
                ["coin", *CM_COLS]
            )
        )
    cm: pl.DataFrame = pl.concat(frames, how="vertical").sort(["coin", "date"])

    def safe_ratio(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
        """除法带零/空值保护，避免 inf 污染评分。"""
        return (
            pl.when(denominator.abs() > 1e-12)
            .then(numerator / denominator)
            .otherwise(None)
        )

    addr: pl.Expr = pl.col("AdrActCnt")
    tx: pl.Expr = pl.col("TxCnt")
    cap: pl.Expr = pl.col("CapMrktCurUSD")
    cm = cm.with_columns(
        (safe_ratio(addr, addr.shift(7).over("coin")) - 1).alias("o_addr_chg7"),
        (safe_ratio(addr, addr.shift(28).over("coin")) - 1).alias("o_addr_chg28"),
        safe_ratio(
            addr, addr.rolling_mean(90, min_samples=30).over("coin")
        ).alias("o_addr_rel90"),
        (safe_ratio(tx, tx.shift(28).over("coin")) - 1).alias("o_tx_chg28"),
        safe_ratio(
            cap, tx.rolling_mean(30, min_samples=15).over("coin")
        ).log1p().alias("o_nvt30_log"),
        cap.log1p().alias("o_cap_log"),
    )
    factor_cols: list[str] = [
        c for c in cm.columns if c.startswith("o_")
    ]
    return cm.select(["coin", "date", *factor_cols])


def join_with_label(cm: pl.DataFrame) -> pd.DataFrame:
    """链上因子与日线标签按 (币, 日期) 对齐。"""
    panel: pl.DataFrame = build_factor_table()
    labeled = panel.select(
        "datetime",
        pl.col("datetime").dt.date().alias("date"),
        "vt_symbol",
        pl.col("vt_symbol")
        .str.replace("USDT.OKX", "")
        .str.to_lowercase()
        .alias("coin"),
        "label",
    )
    joined: pl.DataFrame = labeled.join(cm, on=["coin", "date"], how="inner").drop(
        "date"
    )
    return joined.to_pandas()


def main() -> None:
    """链上因子 -> 与日线标签对齐 -> 共用层评分/报告。"""
    print("[1] 读取链上数据并计算因子 ...")
    cm: pl.DataFrame = load_onchain()
    print(f"    {cm.height} 行 x {cm['coin'].n_unique()} 币")
    print("[2] 与日线标签对齐 ...")
    pdf: pd.DataFrame = join_with_label(cm)
    factor_cols: list[str] = [c for c in pdf.columns if c.startswith("o_")]
    values: dict[str, pd.Series] = {name: pdf[name] for name in factor_cols}
    print(f"    对齐后 {len(pdf)} 行（链上数据实际覆盖 {pdf['coin'].nunique()} 币）")

    print("[3] 三段 IC 评分 ...")
    scores: pd.DataFrame = score_all(pdf, values)
    write_report(
        scores, [], "CoinMetrics 链上因子 IC 测试",
        ["社区版核心指标：活跃地址/交易笔数/市值；转账额类已移入付费层",
         "截面仅覆盖部分币种（约 13 个），IC 参考价值受限",
         "链上列不在 vnpy 面板中，仅评分报告，不并入因子清单"],
        REPORT_PATH,
    )
    print_top(scores)


if __name__ == "__main__":
    main()
