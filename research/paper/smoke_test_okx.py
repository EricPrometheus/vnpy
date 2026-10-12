"""OKX 模拟盘密钥冒烟测试：验证网络链路、密钥有效性、模拟盘请求头需求。

做什么：
1. 读 okx_setting.json 里的三件套凭据
2. 按 OKX 官方签名规则（HMAC-SHA256 + Base64）调用私有接口
   GET /api/v5/account/balance（账户余额）
3. 对比"带 x-simulated-trading:1 头"与"不带"两种请求的结果：
   - 带头成功 => 模拟盘密钥有效，且确认 vnpy_okx 网关需要补这个头
   - 带头返回 401/密码短语错误 => 密钥三件套不完整（先检查 Passphrase）
   - 网络错误 => 代理链路问题

用法（在仓库根目录运行）：
    D:\\veighna_studio\\python.exe -X utf8 research/paper/smoke_test_okx.py
"""

import base64
import hashlib
import hmac
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

SETTING_PATH: Path = Path("research/paper/okx_setting.json")
BALANCE_URL: str = "https://www.okx.com/api/v5/account/balance"
PROXY: dict = {"http": "http://127.0.0.1:10809", "https": "http://127.0.0.1:10809"}
TIMEOUT_SECONDS: int = 20


def okx_timestamp() -> str:
    """OKX 要求的 ISO8601 UTC 时间戳（含毫秒与 Z 后缀）。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def okx_sign(secret: str, timestamp: str, method: str,
             request_path: str, body: str = "") -> str:
    """OKX 签名：Base64(HMAC-SHA256(secret, timestamp+method+path+body))。"""
    prehash: str = f"{timestamp}{method}{request_path}{body}"
    digest = hmac.new(secret.encode(), prehash.encode(), hashlib.sha256).digest()
    return base64.b64encode(digest).decode()


def call_balance(settings: dict, demo_header: bool) -> dict:
    """调余额接口，返回 OKX 响应 JSON。"""
    key: str = settings["API Key"]
    secret: str = settings["Secret Key"]
    passphrase: str = settings["Passphrase"]
    timestamp: str = okx_timestamp()
    headers: dict = {
        "OK-ACCESS-KEY": key,
        "OK-ACCESS-SIGN": okx_sign(secret, timestamp, "GET",
                                   "/api/v5/account/balance"),
        "OK-ACCESS-TIMESTAMP": timestamp,
        "OK-ACCESS-PASSPHRASE": passphrase,
        "Content-Type": "application/json",
    }
    if demo_header:
        headers["x-simulated-trading"] = "1"
    response = requests.get(
        BALANCE_URL, headers=headers, proxies=PROXY, timeout=TIMEOUT_SECONDS
    )
    return response.json()


def main() -> None:
    """读凭据 -> 带头请求 -> 不带头请求 -> 给出结论。"""
    settings: dict = json.loads(SETTING_PATH.read_text(encoding="utf-8"))
    if not settings.get("Passphrase"):
        print("[提示] Passphrase 为空：下面的 401 属预期，"
              "请先在 okx_setting.json 填入创建密钥时自己设定的密码短语")

    print("[1] 带模拟盘头（x-simulated-trading: 1）请求 ...")
    with_header: dict = call_balance(settings, demo_header=True)
    print(f"    响应: {json.dumps(with_header, ensure_ascii=False)[:300]}")

    print("[2] 不带模拟盘头请求（模拟 vnpy_okx 现状）...")
    without_header: dict = call_balance(settings, demo_header=False)
    print(f"    响应: {json.dumps(without_header, ensure_ascii=False)[:300]}")

    code_with: str = str(with_header.get("code"))
    code_without: str = str(without_header.get("code"))
    print("\n=============== 结论 ===============")
    if code_with == "0":
        print("模拟盘密钥有效（带头请求成功拿到账户数据）。")
    elif code_without == "0":
        print("密钥在真实环境有效——注意这是模拟盘密钥，unexpected，请检查 Server 设置。")
    elif code_with in ("5011", "5012", "401") or "passphrase" in json.dumps(
        with_header
    ).lower():
        print("三件套不完整或 Passphrase 不对：请检查 okx_setting.json。")
    else:
        print(f"带头请求返回 code={code_with}，按 OKX 错误码表排查。")
    if code_with == "0" and code_without != "0":
        print("确认：vnpy_okx 网关需要补 x-simulated-trading 头才能用模拟盘。")


if __name__ == "__main__":
    main()
