"""以 polars 直接读取 vnpy AlphaLab 日线 parquet，提供 alphagen 的 StockData 等价类。

为什么不用官方 qlib 数据层：pyqlib 在 Windows + 新版 Python 上没有安装包，
而 alphagen 对数据层的全部要求只是一个"天 x 特征 x 币"三维 float 张量
加日期/币种两个索引。本类与其鸭子类型兼容，qlib 因此完全不需要安装。

张量布局与 alphagen_qlib.StockData 完全一致：
    data: float32 张量，形状 (总天数, 特征数, 币数)；
    前 max_backtrack_days 天留作表达式回看窗口，后 max_future_days 天
    留作未来标签窗口；n_days = 总天数 - 两者之和。
币种未上市期间的缺失值保留 NaN——alphagen 的批量相关系数实现会
按天掩码跳过 NaN 行（见 alphagen/utils/correlation.py），无需预填充。

约定：在仓库根目录运行（数据路径为字面相对路径）。
运行环境：research/.venv-alphagen（需 torch / polars / pandas / numpy）
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch

sys.path.insert(0, "research/alphagen")  # alphagen 包（FeatureType 枚举所在）

from alphagen_qlib.stock_data import FeatureType  # noqa: E402

DATA_DIR: str = "research/alpha_lab/daily"
_DAY: np.timedelta64 = np.timedelta64(1, "D")


def _parse_date(text: str) -> np.datetime64:
    """'YYYY-MM-DD' -> numpy 日期时间（当天 00:00）。"""
    return np.datetime64(datetime.strptime(text, "%Y-%m-%d"))


class CryptoStockData:
    """与 alphagen_qlib.StockData 鸭子类型兼容的加密货币日线数据集。"""

    def __init__(
        self,
        start_time: str | None = None,
        end_time: str | None = None,
        max_backtrack_days: int = 100,
        max_future_days: int = 10,
        features: list[FeatureType] | None = None,
        device: torch.device = torch.device("cpu"),
        preloaded: tuple[torch.Tensor, pd.DatetimeIndex, pd.Index] | None = None,
        data_dir: str = DATA_DIR,
    ) -> None:
        self.max_backtrack_days = max_backtrack_days
        self.max_future_days = max_future_days
        self._features = features if features is not None else list(FeatureType)
        self.device = device
        if preloaded is not None:
            self.data, self._dates, self._stock_ids = preloaded
        else:
            self.data, self._dates, self._stock_ids = self._load(
                data_dir, start_time, end_time
            )

    # ---------------- 数据加载 ----------------

    def _load(
        self, data_dir: str, start_time: str | None, end_time: str | None
    ) -> tuple[torch.Tensor, pd.DatetimeIndex, pd.Index]:
        per_symbol: dict[str, pl.DataFrame] = {}
        for path in sorted(Path(data_dir).glob("*.OKX.parquet")):
            df: pl.DataFrame = pl.read_parquet(path).sort("datetime")
            if start_time is not None:
                df = df.filter(pl.col("datetime") >= _parse_date(start_time))
            if end_time is not None:
                df = df.filter(pl.col("datetime") < _parse_date(end_time) + _DAY)
            per_symbol[path.stem] = df
        if not per_symbol:
            raise FileNotFoundError(f"未找到日线数据: {data_dir}")

        all_dates: np.ndarray = np.array(sorted(
            {ts for df in per_symbol.values() for ts in df["datetime"].to_numpy()}
        ))
        symbols: list[str] = sorted(per_symbol)
        # 与 qlib 版语义一致：区间两侧各扩 backtrack/future 窗口
        start_idx: int = 0 if start_time is None else int(
            all_dates.searchsorted(_parse_date(start_time))
        )
        end_idx: int = len(all_dates) if end_time is None else int(
            all_dates.searchsorted(_parse_date(end_time) + _DAY)
        )
        real_start: int = max(0, start_idx - self.max_backtrack_days)
        real_end: int = min(len(all_dates), end_idx + self.max_future_days)

        values: np.ndarray = np.full(
            (real_end - real_start, len(self._features), len(symbols)),
            np.nan, dtype=np.float32,
        )
        row_of_date: dict[np.datetime64, int] = {
            ts: i - real_start for i, ts in enumerate(all_dates[real_start:real_end], real_start)
        }
        for col_idx, symbol in enumerate(symbols):
            self._fill_symbol(values, row_of_date, col_idx, per_symbol[symbol])

        dates_index: pd.DatetimeIndex = pd.DatetimeIndex(all_dates[real_start:real_end])
        tensor: torch.Tensor = torch.from_numpy(values).to(self.device)
        return tensor, dates_index, pd.Index(symbols)

    def _fill_symbol(
        self,
        values: np.ndarray,
        row_of_date: dict[np.datetime64, int],
        col_idx: int,
        df: pl.DataFrame,
    ) -> None:
        """把单币行情按日期行号写入张量；VWAP 由成交额/成交量推出。"""
        dates: np.ndarray = df["datetime"].to_numpy()
        rows: list[int] = [row_of_date[ts] for ts in dates if ts in row_of_date]
        mask: np.ndarray = np.array([ts in row_of_date for ts in dates])
        rows = [row_of_date[ts] for ts in dates[mask]]
        columns: dict[str, np.ndarray] = {
            name: df[name].to_numpy()[mask].astype(np.float32)
            for name in ("open", "high", "low", "close", "volume", "turnover")
        }
        channel: dict[FeatureType, np.ndarray] = {
            FeatureType.OPEN: columns["open"],
            FeatureType.HIGH: columns["high"],
            FeatureType.LOW: columns["low"],
            FeatureType.CLOSE: columns["close"],
            FeatureType.VOLUME: columns["volume"],
        }
        volume: np.ndarray = columns["volume"]
        vwap: np.ndarray = np.where(
            volume > 0, columns["turnover"] / np.maximum(volume, 1e-12), np.nan
        ).astype(np.float32)
        channel[FeatureType.VWAP] = vwap
        for feature_type, data_array in channel.items():
            values[rows, int(feature_type), col_idx] = data_array

    # ---------------- 与 alphagen_qlib.StockData 兼容的接口 ----------------

    def __getitem__(self, slc: slice) -> "CryptoStockData":
        """按可用日下标切片（语义照抄 qlib 版：两端各含 backtrack/future 余量）。"""
        if slc.step is not None:
            raise ValueError("只支持 step=None 的切片")
        if isinstance(slc.start, str):
            return self[self.find_date_slice(slc.start, slc.stop)]
        start: int = 0 if slc.start is None else max(0, slc.start)
        stop: int = (
            self.n_days if slc.stop is None else slc.stop
        ) + self.max_future_days + self.max_backtrack_days
        stop = min(self.data.shape[0], stop)
        data: torch.Tensor = self.data[start:stop]
        remaining: torch.Tensor = (
            data.isnan().reshape(-1, data.shape[-1]).all(dim=0)
            .logical_not().nonzero().flatten()
        )
        data = data[:, :, remaining]
        return CryptoStockData(
            start_time=self._dates[start + self.max_backtrack_days].strftime("%Y-%m-%d"),
            end_time=self._dates[stop - 1 - self.max_future_days].strftime("%Y-%m-%d"),
            max_backtrack_days=self.max_backtrack_days,
            max_future_days=self.max_future_days,
            features=self._features,
            device=self.device,
            preloaded=(
                data,
                pd.DatetimeIndex(self._dates[start:stop]),
                self._stock_ids[remaining.tolist()],
            ),
        )

    def find_date_index(self, date: str, exclusive: bool = False) -> int:
        """日期 -> 可用日下标（扣除 backtrack 余量）。"""
        ts: pd.Timestamp = pd.Timestamp(date)
        idx: int = int(self._dates.searchsorted(ts))
        if exclusive and self._dates[idx] == ts:
            idx += 1
        idx -= self.max_backtrack_days
        if idx < 0 or idx > self.n_days:
            raise ValueError(f"日期越界: {date}")
        return idx

    def find_date_slice(self, start_time: str | None = None,
                        end_time: str | None = None) -> slice:
        """日期区间 -> 下标切片（左闭右开）。"""
        start = None if start_time is None else self.find_date_index(start_time)
        stop = None if end_time is None else self.find_date_index(end_time)
        return slice(start, stop)

    @property
    def n_features(self) -> int:
        return len(self._features)

    @property
    def n_stocks(self) -> int:
        return self.data.shape[-1]

    @property
    def n_days(self) -> int:
        return self.data.shape[0] - self.max_backtrack_days - self.max_future_days

    @property
    def stock_ids(self) -> pd.Index:
        return self._stock_ids

    def make_dataframe(self, data: torch.Tensor,
                       columns: list[str] | None = None) -> pd.DataFrame:
        """(天, 币[, 列]) 张量 -> 长表（语义照抄 qlib 版）。"""
        if len(data.shape) == 2:
            data = data.unsqueeze(2)
        if columns is None:
            columns = [str(i) for i in range(data.shape[2])]
        n_days: int
        n_stocks: int
        n_columns: int
        n_days, n_stocks, n_columns = data.shape
        if self.n_days != n_days or self.n_stocks != n_stocks:
            raise ValueError("张量形状与当前数据集不匹配")
        if self.max_future_days == 0:
            date_index = self._dates[self.max_backtrack_days:]
        else:
            date_index = self._dates[self.max_backtrack_days:-self.max_future_days]
        index = pd.MultiIndex.from_product([date_index, self._stock_ids])
        flat = data.reshape(-1, n_columns)
        return pd.DataFrame(flat.detach().cpu().numpy(), index=index, columns=columns)
