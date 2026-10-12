"""模式 A 启动脚本：MainEngine + vnpy_okx 网关 + PaperAccount 模拟交易 + 轮动策略。

组装顺序（顺序有讲究）：
    1) 设置 VNPY_RESEARCH_ROOT 并导入研究模块（必须在 MainEngine 之前——
       MainEngine 初始化会把工作目录切到 .vntrader，相对路径会失效）；
    2) MainEngine + EventEngine；
    3) add_app(PaperAccountApp)：模拟交易应用会接管 MainEngine.send_order，
       之后策略发出的所有订单只在本地仿真撮合，不会触碰真实交易所；
    4) add_gateway(OkxGateway) 并用凭据连接 OKX（行情用于聚合日线与撮合定价）；
    5) 启动策略，等待合约与行情；每天 UTC 16:05（北京时间 00:05）自动完成
       收盘判定与模拟调仓，流水记录在 research/paper/paper_trades.csv。

凭据来源（二选一，代码与 git 仓库中永不出现真实密钥）：
    a) 环境变量：OKX_API_KEY / OKX_API_SECRET / OKX_PASSPHRASE，
       可选 OKX_SERVER（REAL|DEMO）、OKX_PROXY_HOST、OKX_PROXY_PORT；
    b) 本地文件 research/paper/okx_setting.json（首次运行自动生成空值模板，
       该文件已被 .gitignore 忽略）。键名与 vnpy_okx 的 OkxGateway.default_setting
       一致：API Key / Secret Key / Passphrase / Server / Proxy Host / Proxy Port。

启用步骤（本机当前缺 OKX 凭据，以下为凭据就绪后的操作清单）：
    1) 安装 OKX 网关（本机当前未安装）：
       D:\\veighna_studio\\python.exe -m pip install vnpy_okx
    2) 按上述任一方式准备凭据（建议先用 OKX 模拟盘 DEMO 的 API Key 验证链路；
       即使 Server=REAL，订单也只会进入本地 PaperAccount，不会真实下单）；
    3) 确认 research/alpha_lab/daily/*.parquet 预热数据足够新（策略冷启动的
       历史日线来自这里，过期会收到警告日志）；
    4) 在仓库根目录运行：
       D:\\veighna_studio\\python.exe -X utf8 research/paper/run_paper.py
    5) 观察日志确认合约订阅完成；每天北京时间 00:05 后查看
       research/paper/paper_trades.csv 中的判定与委托流水。

静态检查（无凭据也可执行）：
    D:\\veighna_studio\\python.exe -m py_compile research/paper/paper_strategy.py research/paper/run_paper.py
"""

import json
import os
import sys
import time
from pathlib import Path

# 必须在导入研究模块之前设置根目录环境变量（factor_common 靠它定位 research/）
_HERE: Path = Path(__file__).resolve().parent
_RESEARCH_DIR: Path = _HERE.parent
os.environ.setdefault("VNPY_RESEARCH_ROOT", str(_RESEARCH_DIR))
if str(_RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_DIR))

from factor_common import safe_join  # noqa: E402  （需在设置环境变量之后导入）

GATEWAY_NAME: str = "OKX"
SETTING_FILE_ENV: str = "PAPER_OKX_SETTING"       # 凭据文件路径可被环境变量覆盖
SETTING_FILE_REL: str = "paper/okx_setting.json"  # 默认凭据文件（已被 .gitignore 忽略）
# OkxGateway.default_setting 的键名（与 vnpy_okx 源码一致）-> 环境变量名
ENV_KEYS: dict[str, str] = {
    "API Key": "OKX_API_KEY",
    "Secret Key": "OKX_API_SECRET",
    "Passphrase": "OKX_PASSPHRASE",
    "Server": "OKX_SERVER",
    "Proxy Host": "OKX_PROXY_HOST",
    "Proxy Port": "OKX_PROXY_PORT",
}
SETTING_TEMPLATE: dict[str, object] = {
    "API Key": "",
    "Secret Key": "",
    "Passphrase": "",
    "Server": "REAL",        # 真实行情；订单反正只进本地 PaperAccount
    "Proxy Host": "127.0.0.1",
    "Proxy Port": 10809,
    "Spread Trading": "False",
}


def setting_path() -> Path:
    """解析凭据文件路径（环境变量优先，默认在 research 根目录内并做穿越校验）。"""
    override: str | None = os.environ.get(SETTING_FILE_ENV)
    if override:
        return Path(override)
    return safe_join(SETTING_FILE_REL)


def build_setting_from_env() -> dict[str, object] | None:
    """从环境变量组装网关设置；三个必填项缺一不可。"""
    required_envs: list[str] = [
        ENV_KEYS["API Key"], ENV_KEYS["Secret Key"], ENV_KEYS["Passphrase"]
    ]
    if not all(os.environ.get(name) for name in required_envs):
        return None
    setting: dict[str, object] = {
        key: os.environ.get(env_name, "")
        for key, env_name in ENV_KEYS.items()
    }
    setting["Server"] = str(setting["Server"] or "REAL")  # 缺省真实行情
    setting["Proxy Port"] = int(setting["Proxy Port"] or 0)  # type: ignore[call-overload]
    print("已从环境变量读取 OKX 凭据")
    return setting


def build_setting_from_file() -> dict[str, object] | None:
    """从本地凭据文件读取设置；文件不存在时生成空值模板并退出。"""
    path: Path = setting_path()
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(SETTING_TEMPLATE, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"已生成凭据模板: {path}\n"
              "请填入 OKX API 凭据后重跑（该文件已被 .gitignore 忽略，勿提交）")
        return None
    setting: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    if not str(setting.get("API Key") or "").strip():
        print(f"{path} 中 API Key 为空，请先填入凭据")
        return None
    print(f"已从凭据文件读取 OKX 设置: {path}")
    return setting


def load_okx_setting() -> dict[str, object] | None:
    """凭据装载：环境变量优先，其次本地凭据文件。"""
    from_env: dict[str, object] | None = build_setting_from_env()
    if from_env is not None:
        return from_env
    return build_setting_from_file()


def main() -> int:
    """入口：检查凭据 -> 组装引擎/应用/网关 -> 连接并保持运行。"""
    setting: dict[str, object] | None = load_okx_setting()
    if setting is None:
        return 1
    try:
        from vnpy_okx import OkxGateway
    except ImportError:
        print("未安装 vnpy_okx 网关，请先执行：\n"
              "  D:\\veighna_studio\\python.exe -m pip install vnpy_okx")
        return 1

    from vnpy.trader.engine import MainEngine

    from paper_strategy import GpS123PaperStrategy
    from vnpy_paperaccount import PaperAccountApp

    main_engine: MainEngine = MainEngine()
    strategy: GpS123PaperStrategy | None = None
    try:
        # 模拟交易应用：接管 send_order，订单只在本地仿真撮合
        paper_engine = main_engine.add_app(PaperAccountApp)
        paper_engine.set_instant_trade(True)  # 有最新 tick 即撮合，贴近实盘成交
        main_engine.add_gateway(OkxGateway)

        strategy = GpS123PaperStrategy(
            main_engine, main_engine.event_engine, gateway_name=GATEWAY_NAME
        )
        strategy.start()
        main_engine.connect(setting, GATEWAY_NAME)
        print(f"已连接 {GATEWAY_NAME} 网关，等待行情；"
              "每天 UTC 16:05（北京时间 00:05）自动判定调仓，Ctrl+C 退出")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("收到退出信号，正在关闭……")
    finally:
        if strategy is not None:
            strategy.stop()
        main_engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
