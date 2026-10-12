"""把 alphagen 训练产出的因子池翻译成 vnpy 表达式并写入统一因子清单。

输入：research/alphagen_out/final_pool.json（train_crypto.py 的输出，含
qlib 风格公式文本如 "Mul(Ref($close,1),Abs(Sub($close,$open)))" 与权重）。
翻译：在标准库 ast（抽象语法树）层面构造公式节点后由 ast.unparse 序列化，
算子按白名单映射表转换（Ref->ts_delay、Mean->ts_mean、Corr->ts_corr 等），
$vwap 字段翻译为 turnover/volume（vnpy 行情面板没有现成 vwap 列）。
输出：research/mined_manifest.json 的 "alphagen" 段。

用法（在仓库根目录运行，veighna 环境）：
    D:\\veighna_studio\\python.exe -X utf8 research/export_alphagen_factors.py
"""

import ast
import json
import re
from pathlib import Path

from factor_common import (
    RESEARCH_ROOT,
    load_manifest,
    save_factors,
)

POOL_PATH: Path = RESEARCH_ROOT / "alphagen_out" / "final_pool.json"

# alphagen 字段 -> vnpy 面板列；vwap 无现成列，用成交额/成交量推出
FIELD_NAMES: dict[str, str] = {
    "$open": "open",
    "$high": "high",
    "$low": "low",
    "$close": "close",
    "$volume": "volume",
    "$vwap": "turnover / volume",
}

# alphagen 算子白名单 -> (vnpy 函数名, 参数个数)；二元四则另列
ROLLING_CALLS: dict[str, tuple[str, int]] = {
    "Ref": ("ts_delay", 2),
    "Mean": ("ts_mean", 2),
    "Sum": ("ts_sum", 2),
    "Std": ("ts_std", 2),
    "Max": ("ts_max", 2),
    "Min": ("ts_min", 2),
    "Delta": ("ts_delta", 2),
    "Cov": ("ts_cov", 3),
    "Corr": ("ts_corr", 3),
}
UNARY_CALLS: dict[str, str] = {"Abs": "abs", "Log": "log"}
INFIX_OPS: dict[str, type[ast.BinOp]] = {
    "Add": ast.Add,
    "Sub": ast.Sub,
    "Mul": ast.Mult,
    "Div": ast.Div,
}

_TOKEN_RE: re.Pattern = re.compile(
    r"\(|\)|,|\$[a-z_]+|[A-Za-z_][A-Za-z0-9_]*|[+-]?(?:\d+\.?\d*|\.\d+)d?"
)


def _parse_node(tokens: list[str], pos: int) -> tuple[ast.expr, int]:
    """递归下降：把 qlib 风格公式的一个节点转成 ast 公式节点。

    安全边界：算子名必须命中映射表白名单，字段名必须命中 FIELD_NAMES，
    时间窗口必须是带 d 后缀的整数，常量必须是纯数字——之外一律抛错。
    """
    node: str = tokens[pos]
    if node == "(":
        raise ValueError("公式格式错误：意外的前括号")
    if node in INFIX_OPS:
        left, pos = _parse_node(tokens, pos + 2, )   # 跳过 "("
        right, pos = _parse_node(tokens, pos + 1)
        return ast.BinOp(left=left, op=INFIX_OPS[node](), right=right), pos + 1
    if node in UNARY_CALLS:
        inner, pos = _parse_node(tokens, pos + 2)
        return _make_call(UNARY_CALLS[node], [inner]), pos + 1
    if node in ROLLING_CALLS:
        func_name, arity = ROLLING_CALLS[node]
        args: list[ast.expr] = []
        pos += 2  # 跳过函数名和 "("
        for order in range(arity):
            if order > 0:
                pos += 1  # 跳过 ","
            arg_node, pos = _parse_node(tokens, pos)
            args.append(arg_node)
        if arity == 3:  # Corr/Dep 式：(x, y, 窗口)
            args = [args[0], args[1], args[2]]
        return _make_call(func_name, args), pos + 1  # 跳过 ")"
    if node in FIELD_NAMES:
        return ast.parse(FIELD_NAMES[node], mode="eval").body, pos + 1
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?d", node):  # 滚动窗口（DeltaTime）
        return ast.Constant(value=int(node.rstrip("d"))), pos + 1
    if re.fullmatch(r"[+-]?(?:\d+\.?\d*|\.\d+)", node):  # 常数
        return ast.Constant(value=float(node)), pos + 1
    raise ValueError(f"公式含白名单之外的符号，已拒绝: {node}")


def _make_call(func_name: str, args: list[ast.expr]) -> ast.Call:
    """构造一个普通的函数调用公式节点。"""
    return ast.Call(
        func=ast.Name(id=func_name, ctx=ast.Load()), args=args, keywords=[]
    )


def translate_to_vnpy(qlib_formula: str) -> str:
    """qlib 风格公式（Add($close,Ref($close,1))）-> vnpy 公式文本。"""
    tokens: list[str] = _TOKEN_RE.findall(qlib_formula)
    root, pos = _parse_node(tokens, 0)
    if pos != len(tokens):
        raise ValueError(f"公式翻译不完整: {qlib_formula}")
    return ast.unparse(ast.Expression(body=root))


def main() -> None:
    """读取因子池 -> 翻译 -> 写入统一清单。"""
    if not POOL_PATH.exists():
        raise FileNotFoundError(
            f"未找到因子池文件: {POOL_PATH}（请先运行 train_crypto.py）"
        )
    pool: dict = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    size: int = int(pool.get("size", 0))
    if size == 0:
        raise ValueError("因子池为空（RL 尚未产出任何公式），请先完成训练")

    factors: list[dict] = []
    for i in range(size):
        qlib_formula: str = pool["exprs"][i]
        vnpy_text: str = translate_to_vnpy(qlib_formula)
        factors.append({
            "name": f"ag_{i + 1:03d}",
            "source": "alphagen",
            "vnpy_expr": vnpy_text,
            "weight": float(pool["weights"][i]),
            "qlib_expr": qlib_formula,
            "train_ic": float(pool.get("single_ics", [0.0] * size)[i]),
        })
        print(f"  ag_{i + 1:03d}  weight={pool['weights'][i]:+.4f}  {qlib_formula}")

    save_factors("alphagen", factors, meta={
        "tool": "alphagen(RL)",
        "pool_train_ic": pool.get("train_ic_ret"),
        "pool_test_ic": pool.get("test_ic_mean"),
        "pool_test_ric": pool.get("test_ric_mean"),
    })
    print(f"已写入 {len(factors)} 个 alphagen 因子到清单")


if __name__ == "__main__":
    main()
