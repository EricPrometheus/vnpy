"""alphagen 加密货币训练入口（qlib-free 适配版，CPU 可跑）。

与官方 scripts/rl.py 的差异：
- 数据层换成本目录的 CryptoStockData（polars 读 parquet，不依赖 qlib）；
- 算子表原地裁剪为可映射回 vnpy 表达式引擎的 15 个（其余算子无对应实现，
  挖出来也回填不了，索性不让 RL 生成）；
- 目标标签与现有管线对齐：未来 5 日收益 Ref($close,-6)/Ref($close,-1)-1；
- 设备默认 CPU（可用环境变量 ALPHAGEN_DEVICE 覆盖），步数参数化。

输出：research/alphagen_out/<steps>_pool.json（公式 + 权重 + 单因子 IC），
供 export_alphagen_factors.py 翻译回 vnpy 表达式并写入统一因子清单。

用法（在仓库根目录运行）：
    research/.venv-alphagen/Scripts/python.exe -X utf8 research/alphagen_adaptor/train_crypto.py --steps 20000
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "research/alphagen")
sys.path.insert(0, "research/alphagen_adaptor")

import torch  # noqa: E402

# 必须在 import alphagen.rl 之前原地裁剪算子表：
# wrapper/core 用 from-import 持有同一个 list 对象，替换内容即全局生效
from alphagen.config import OPERATORS  # noqa: E402
from alphagen.data.expression import (  # noqa: E402
    Abs, Add, Corr, Cov, Delta, Div, Log, Max, Mean, Min, Mul, Ref, Std, Sub, Sum,
)

KEEP_OPERATORS: list = [Abs, Log, Add, Sub, Mul, Div, Ref, Mean, Sum, Std,
                        Max, Min, Delta, Cov, Corr]
OPERATORS[:] = KEEP_OPERATORS

from sb3_contrib import MaskablePPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402

from alphagen.data.calculator import TensorAlphaCalculator  # noqa: E402
from alphagen.data.expression import Feature, FeatureType, Ref  # noqa: E402
from alphagen.models.linear_alpha_pool import MseAlphaPool  # noqa: E402
from alphagen.rl.env.wrapper import AlphaEnv  # noqa: E402
from alphagen.rl.policy import LSTMSharedNet  # noqa: E402
from alphagen.utils import reseed_everything  # noqa: E402
from alphagen.utils.pytorch_utils import normalize_by_day  # noqa: E402

from crypto_stock_data import CryptoStockData  # noqa: E402

# 数据段：训练 + 三个测试段（与官方 rl.py 的四段结构一致）
SEGMENTS: list[tuple[str, str]] = [
    ("2023-01-01", "2024-12-31"),
    ("2025-01-01", "2025-06-30"),
    ("2025-07-01", "2025-12-31"),
    ("2026-01-01", "2026-09-21"),
]
OUTPUT_DIR: str = "research/alphagen_out"
POOL_CAPACITY: int = 10
L1_ALPHA: float = 5e-3
MAX_BACKTRACK_DAYS: int = 100
MAX_FUTURE_DAYS: int = 10  # 目标最多看未来 6 天，留 10 天余量即可


def safe_output_path(save_dir: str, filename: str) -> str:
    """规范并校验输出路径必须位于 save_dir 之内（防路径穿越）。"""
    base_abs: str = os.path.abspath(os.path.normpath(save_dir))
    target_abs: str = os.path.abspath(
        os.path.normpath(os.path.join(base_abs, filename))
    )
    if os.path.commonpath([base_abs, target_abs]) != base_abs:
        raise ValueError(f"输出路径越出目录，已拒绝: {filename}")
    return target_abs


class CryptoStockDataCalculator(TensorAlphaCalculator):
    """照抄 QLibStockDataCalculator：日截面标准化后交给基类算 IC。"""

    def __init__(self, data: CryptoStockData, target_expr) -> None:
        super().__init__(normalize_by_day(target_expr.evaluate(data)))
        self.data = data

    def evaluate_alpha(self, expr):
        return normalize_by_day(expr.evaluate(self.data))

    @property
    def n_days(self) -> int:
        return self.data.n_days


class PoolCheckpointCallback(BaseCallback):
    """每个 rollout 结束时：记录测试段 IC 并把当前因子池存成 JSON。"""

    def __init__(self, save_dir: str, test_calculators: list) -> None:
        super().__init__(verbose=1)
        self.save_dir = save_dir
        self.test_calculators = test_calculators
        os.makedirs(save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        pool = self.training_env.envs[0].unwrapped.pool
        total_days: int = sum(calc.data.n_days for calc in self.test_calculators)
        weighted_ic: float = 0.0
        weighted_ric: float = 0.0
        for order, calc in enumerate(self.test_calculators, start=1):
            ic: float
            ric: float
            ic, ric = pool.test_ensemble(calc)
            weight: float = calc.data.n_days / total_days
            weighted_ic += ic * weight
            weighted_ric += ric * weight
            self.logger.record(f"test/ic_{order}", ic)
            self.logger.record(f"test/ric_{order}", ric)
        self.logger.record("test/ic_mean", weighted_ic)
        self.logger.record("test/ric_mean", weighted_ric)
        self._save_pool(pool, weighted_ic, weighted_ric)

    def _save_pool(self, pool, weighted_ic: float, weighted_ric: float) -> None:
        """导出当前因子池：公式文本 + 权重 + 单因子 IC（自定格式）。"""
        size: int = pool.size
        snapshot: dict = {
            "size": int(size),
            "exprs": [str(pool.exprs[i]) for i in range(size)],
            "weights": [float(w) for w in pool.weights],
            "single_ics": [float(x) for x in pool.single_ics[:size]],
            "train_ic_ret": float(pool.best_ic_ret),
            "test_ic_mean": weighted_ic,
            "test_ric_mean": weighted_ric,
        }
        path: str = safe_output_path(
            self.save_dir, f"{self.num_timesteps}_pool.json"
        )
        Path(path).write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if self.verbose > 0:
            print(f"[pool] size={size} train_ic={pool.best_ic_ret:.4f} "
                  f"test_ic={weighted_ic:.4f} test_ric={weighted_ric:.4f} -> {path}")


def parse_args() -> argparse.Namespace:
    """命令行参数。"""
    parser = argparse.ArgumentParser(description="alphagen 加密货币训练")
    parser.add_argument("--steps", type=int, default=20000, help="总训练步数")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--capacity", type=int, default=POOL_CAPACITY, help="因子池容量")
    return parser.parse_args()


def main() -> None:
    """训练主流程：数据 -> 因子池 -> MaskablePPO -> 周期性导出。"""
    args: argparse.Namespace = parse_args()
    reseed_everything(args.seed)
    device: torch.device = torch.device(os.environ.get("ALPHAGEN_DEVICE", "cpu"))
    print(f"[main] 设备={device} 步数={args.steps} 池容量={args.capacity} "
          f"算子数={len(OPERATORS)}")

    close = Feature(FeatureType.CLOSE)
    target_expr = Ref(close, -6) / Ref(close, -1) - 1  # 未来 5 日收益，与管线对齐

    datasets = [
        CryptoStockData(
            start_time=start, end_time=end, device=device,
            max_backtrack_days=MAX_BACKTRACK_DAYS, max_future_days=MAX_FUTURE_DAYS,
        )
        for start, end in SEGMENTS
    ]
    calculators = [CryptoStockDataCalculator(d, target_expr) for d in datasets]
    print(f"[main] 训练段 {datasets[0].n_days} 天 x {datasets[0].n_stocks} 币, "
          f"测试段 {[d.n_days for d in datasets[1:]]} 天")

    pool = MseAlphaPool(
        capacity=args.capacity, calculator=calculators[0],
        ic_lower_bound=None, l1_alpha=L1_ALPHA, device=device,
    )
    env = AlphaEnv(pool=pool, device=device, print_expr=True)
    callback = PoolCheckpointCallback(OUTPUT_DIR, calculators[1:])
    model = MaskablePPO(
        "MlpPolicy", env,
        policy_kwargs=dict(
            features_extractor_class=LSTMSharedNet,
            features_extractor_kwargs=dict(
                n_layers=2, d_model=128, dropout=0.1, device=device,
            ),
        ),
        gamma=1.0,
        ent_coef=0.01,
        batch_size=128,
        device=device,
        verbose=0,
    )
    model.learn(total_timesteps=args.steps, callback=callback)

    final_path: str = safe_output_path(OUTPUT_DIR, "final_pool.json")
    snapshot_path: str = safe_output_path(
        OUTPUT_DIR, f"{model.num_timesteps}_pool.json"
    )
    if os.path.exists(snapshot_path):
        snapshot = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
        Path(final_path).write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"[main] 训练完成，最终因子池: {final_path}")


if __name__ == "__main__":
    main()
