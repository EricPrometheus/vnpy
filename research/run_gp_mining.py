"""遗传规划（GP, Genetic Programming）自动挖掘加密货币横截面因子。

原理与适配（gplearn 的局限及对策）：
- gplearn 只会逐行计算公式，感知不到"同一天内横向比大小"；
- 打分不用默认相关系数，而是自定义适应度 = 日均截面 Rank IC
  （每天 N 个币内，因子值与未来收益的秩相关，再对日期取平均）；
- 基础特征已携带时序信息（动量/波动/量比），GP 负责横截面组合；
- 函数集只留四则运算与 abs，挖出的公式可无损翻译回 vnpy 表达式引擎。

说明：本脚本不访问任何数据库。"公式翻译"是把 gplearn 的前缀式数学
公式改写为 vnpy 表达式文本：直接在标准库 ast（抽象语法树）层面构造
公式节点，再由 ast.unparse 统一序列化，全程不存在手工拼接文本。
名人堂公式的批量求值统一走 gplearn 的 transform() 矩阵接口。

输出：research/mined_manifest.json 的 "gp" 段（公式 + 三段 IC + 引擎复算）。
用法（在仓库根目录运行，--seed 指定随机种子用于复现性检验）：
    D:\\veighna_studio\\python.exe -X utf8 research/run_gp_mining.py --seed 42
"""

import argparse
import ast
import re
import time
from typing import Callable

import numpy as np
from gplearn.genetic import SymbolicTransformer
from gplearn.fitness import make_fitness

from factor_common import (
    FEATURES,
    FEATURE_NAMES,
    LABEL_VNPY,
    RESEARCH_ROOT,
    TEST,
    TRAIN,
    VALID,
    build_factor_table,
    daily_rank_ic_series,
    date_boundaries,
    mean_rank_ic,
    save_factors,
    segment_mask,
)

# ---- GP 超参数（CPU 全程约 5-10 分钟）----
POPULATION_SIZE: int = 1000
GENERATIONS: int = 15
TOURNAMENT_SIZE: int = 20
HALL_OF_FAME: int = 60
N_KEEP: int = 60               # 进入复检环节的公式数量（transform 的列数）
PARSIMONY: float = 5e-4
DEFAULT_RANDOM_SEED: int = 42  # --seed 未指定时的默认随机种子
# 函数集刻意只留四则运算与 abs：min/max/log 等在 vnpy 表达式引擎里没有对应算子
FUNCTION_SET: list[str] = ["add", "sub", "mul", "div", "abs"]

MIN_VALID_IC: float = 0.02     # 验证段日均 Rank IC 门槛（低于此直接淘汰）
MAX_MUTUAL_CORR: float = 0.7   # 入选因子之间的最大相关系数（去重）
TOP_K: int = 8                 # 最多保留的因子数量

# gplearn 函数名 -> ast 二元运算节点类型（白名单）
AST_BINARY_OPS: dict[str, type[ast.BinOp]] = {
    "add": ast.Add,
    "sub": ast.Sub,
    "mul": ast.Mult,
    "div": ast.Div,
}
ABS_FUNC_NAME: str = "abs"

_TOKEN_RE: re.Pattern = re.compile(
    r"\(|\)|,|[A-Za-z_][A-Za-z0-9_]*|[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
)

SegmentArrays = tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]


def _segment_arrays(panel, segment: tuple[str, str]) -> SegmentArrays:
    """取某时间段的 (特征矩阵 X, 标签 y, 日期编码, 每日边界)。"""
    sub = panel.filter(segment_mask(segment))
    x: np.ndarray = sub.select(FEATURE_NAMES).to_numpy().astype(np.float64)
    y: np.ndarray = sub["label"].to_numpy().astype(np.float64)
    codes, boundaries = date_boundaries(sub["datetime"].to_numpy())
    return x, y, codes, boundaries


def make_ic_fitness(codes: np.ndarray, boundaries: list[tuple[int, int]]) -> Callable:
    """构造 gplearn 自定义适应度：日均截面 Rank IC。

    注意 n_jobs 必须为 1：闭包依赖本进程内的分组边界，多进程会丢状态。
    """
    def _metric(y: np.ndarray, y_pred: np.ndarray, _weight: np.ndarray) -> float:
        return mean_rank_ic(y_pred, y, codes, boundaries)

    return make_fitness(function=_metric, greater_is_better=True)


def _parse_node(tokens: list[str], pos: int,
                feature_asts: dict[str, ast.expr]) -> tuple[ast.expr, int]:
    """递归下降：把 gplearn 前缀公式的一个节点转成 ast 公式节点。

    安全边界：函数名必须命中运算符白名单，特征占位符必须命中
    feature_asts，常量必须匹配数字模式——之外的符号一律抛错。
    """
    node_name: str = tokens[pos]
    if node_name == "(":
        raise ValueError("公式格式错误：意外的前括号")
    if node_name in AST_BINARY_OPS:
        left_node, pos = _parse_node(tokens, pos + 2, feature_asts)   # 跳过 "("
        right_node, pos = _parse_node(tokens, pos + 1, feature_asts)  # 跳过 ","
        op_type: type[ast.binop] = AST_BINARY_OPS[node_name]
        return ast.BinOp(left=left_node, op=op_type(), right=right_node), pos + 1
    if node_name == ABS_FUNC_NAME:
        inner_node, pos = _parse_node(tokens, pos + 2, feature_asts)
        call = ast.Call(func=ast.Name(id="abs", ctx=ast.Load()),
                        args=[inner_node], keywords=[])
        return call, pos + 1
    if node_name in feature_asts:  # 特征占位符 X0..Xn -> 预解析好的 ast 片段
        return feature_asts[node_name], pos + 1
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", node_name):
        return ast.Constant(value=float(node_name)), pos + 1
    raise ValueError(f"公式含白名单之外的符号，已拒绝: {node_name}")


def translate_to_vnpy(formula: str) -> str:
    """gplearn 公式（如 sub(X0, mul(X1, X2))）-> vnpy 公式文本。"""
    feature_asts: dict[str, ast.expr] = {
        f"X{i}": ast.parse(FEATURES[name][1], mode="eval").body
        for i, name in enumerate(FEATURE_NAMES)
    }
    tokens: list[str] = _TOKEN_RE.findall(formula)
    root, pos = _parse_node(tokens, 0, feature_asts)
    if pos != len(tokens):
        raise ValueError(f"公式翻译不完整: {formula}")
    return ast.unparse(ast.Expression(body=root))


def select_programs(gp: SymbolicTransformer,
                    segments: dict[str, SegmentArrays]) -> list[dict]:
    """名人堂复检：验证段门槛 + 符号一致性 + 相关性去重，按验证 IC 排序。

    三个时间段的公式输出统一用 gp.transform() 一次性批量求值成矩阵
    （列 j 对应第 j 个最优公式），比逐公式求值更快也更省代码。
    """
    train_matrix: np.ndarray = gp.transform(segments["train"][0])
    valid_matrix: np.ndarray = gp.transform(segments["valid"][0])
    test_matrix: np.ndarray = gp.transform(segments["test"][0])
    # 公式两两相关性矩阵（列向量间），常数列产生的 NaN 按 0 处理
    mutual_corr: np.ndarray = np.nan_to_num(
        np.corrcoef(train_matrix, rowvar=False), nan=0.0
    )
    y_valid, c_valid, b_valid = segments["valid"][1:]
    y_test, c_test, b_test = segments["test"][1:]

    accepted_cols: list[int] = []
    results: list[dict] = []
    for col, prog in enumerate(gp._best_programs):
        train_ic: float = float(getattr(prog, "raw_fitness_", 0.0))
        valid_ic: float = mean_rank_ic(valid_matrix[:, col], y_valid, c_valid, b_valid)
        test_ic: float = mean_rank_ic(test_matrix[:, col], y_test, c_test, b_test)
        if valid_ic < MIN_VALID_IC:
            continue
        if train_ic * valid_ic <= 0:  # 训练/验证符号翻转，视为不稳定
            continue
        if any(abs(mutual_corr[col, kept]) > MAX_MUTUAL_CORR for kept in accepted_cols):
            continue
        accepted_cols.append(col)
        results.append({
            "formula_gp": str(prog),
            "vnpy_expr": translate_to_vnpy(str(prog)),
            "train_ic": train_ic,
            "valid_ic": valid_ic,
            "test_ic": test_ic,
        })
    results.sort(key=lambda item: item["valid_ic"], reverse=True)
    return results[:TOP_K]


def verify_with_engine(panel, candidates: list[dict]) -> None:
    """用 vnpy 公式引擎复算验证段 IC，确认翻译无损（口径对齐）。"""
    from vnpy.alpha.dataset.utility import calculate_by_expression

    label_data = calculate_by_expression(panel, LABEL_VNPY)["data"]
    base = panel.select(["datetime", "vt_symbol"])
    valid_filter = segment_mask(VALID)
    for cand in candidates:
        value_data = calculate_by_expression(panel, cand["vnpy_expr"])["data"]
        pdf = (
            base.with_columns(value_data.alias("value"), label_data.alias("target"))
            .filter(valid_filter)
            .to_pandas()
        )
        ic_series = daily_rank_ic_series(pdf["value"], pdf["target"], pdf["datetime"])
        engine_valid_ic: float = round(float(ic_series.mean()), 6)
        cand["engine_valid_ic"] = engine_valid_ic
        if abs(engine_valid_ic - cand["valid_ic"]) > 0.015:
            print(f"    [警告] 引擎复算偏差较大: "
                  f"gplearn={cand['valid_ic']:.4f} vs 引擎={engine_valid_ic:.4f}")


def main(seed: int = DEFAULT_RANDOM_SEED) -> None:
    """完整挖掘流程：数据 -> 进化 -> 复检 -> 引擎验证 -> 写清单。"""
    t0: float = time.time()
    print(f"[0] research 根目录: {RESEARCH_ROOT}")
    panel = build_factor_table()
    print(f"[1] 数据: {panel.height} 行, {panel['vt_symbol'].n_unique()} 个币, "
          f"{panel['datetime'].n_unique()} 个交易日")

    segments: dict[str, SegmentArrays] = {
        "train": _segment_arrays(panel, TRAIN),
        "valid": _segment_arrays(panel, VALID),
        "test": _segment_arrays(panel, TEST),
    }
    print(f"[2] 遗传规划进化: 种群 {POPULATION_SIZE} x {GENERATIONS} 代 ...")
    x_train, y_train, date_codes, day_bounds = segments["train"]
    fitness = make_ic_fitness(date_codes, day_bounds)
    gp = SymbolicTransformer(
        generations=GENERATIONS,
        population_size=POPULATION_SIZE,
        tournament_size=TOURNAMENT_SIZE,
        function_set=FUNCTION_SET,
        metric=fitness,
        hall_of_fame=HALL_OF_FAME,
        n_components=N_KEEP,
        parsimony_coefficient=PARSIMONY,
        p_crossover=0.7,
        p_subtree_mutation=0.15,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        max_samples=1.0,
        verbose=1,
        random_state=seed,
        n_jobs=1,
    )
    gp.fit(x_train, y_train)

    candidates: list[dict] = select_programs(gp, segments)
    print(f"[3] 复检池 {N_KEEP} 个公式，入选 {len(candidates)} 个")

    print("[4] vnpy 引擎复算验证段 IC（校验翻译无损）...")
    verify_with_engine(panel, candidates)

    factors: list[dict] = []
    for i, cand in enumerate(candidates, 1):
        name: str = f"gp_s{seed}_{i:03d}"
        factors.append({
            "name": name,
            "source": "gp",
            "vnpy_expr": cand["vnpy_expr"],
            "weight": 1.0,
            "train_ic": round(cand["train_ic"], 6),
            "valid_ic": round(cand["valid_ic"], 6),
            "test_ic": round(cand["test_ic"], 6),
            "engine_valid_ic": cand.get("engine_valid_ic"),
        })
        print(f"  {name}  train={cand['train_ic']:+.4f}  valid={cand['valid_ic']:+.4f}  "
              f"test={cand['test_ic']:+.4f}")
        print(f"        {cand['formula_gp']}")

    save_factors("gp", factors, meta={
        "tool": "gplearn",
        "features": {n: FEATURES[n][1] for n in FEATURE_NAMES},
        "label": LABEL_VNPY,
        "population": POPULATION_SIZE,
        "generations": GENERATIONS,
        "seed": seed,
    })
    print(f"[5] 已写入清单，总耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    arg_parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="gplearn 因子挖掘（--seed 指定随机种子，用于多种子复现性检验）"
    )
    arg_parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED,
                            help="随机种子（默认 42）")
    cli_args: argparse.Namespace = arg_parser.parse_args()
    main(seed=cli_args.seed)
