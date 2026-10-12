"""Grafana 监控栈一键配置：纯 HTTP API 创建 SQLite 数据源与四面板看板。

前提（一次性）：
    1) Grafana 已解压到 research/paper/grafana/grafana-13.2.3/；
    2) 插件已装好：cd 该目录后执行
       ./bin/grafana.exe cli plugins install frser-sqlite-datasource；
    3) Grafana 已在后台运行：./bin/grafana.exe server --homepath .

本脚本做三件事（幂等，可重复执行）：
    a) 创建/复用名为 paper-sqlite 的 SQLite 数据源（指向 research/paper/paper.db）；
    b) 调数据源 /health 验证连通性；
    c) 创建/覆盖看板 "OKX 模拟盘监控"（uid=paper-monitor），四个面板：
       净值曲线（replay/live 同图异色）/ 当前持仓明细 / 最近交易 / 择时状态阶跃图。

凭据：默认 admin/admin（Grafana 出厂默认）。改密后请用环境变量
GRAFANA_USER / GRAFANA_PASSWORD 覆盖；正式使用请务必改掉默认密码。
"""

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

GRAFANA_URL: str = os.environ.get("GRAFANA_URL", "http://localhost:3000")
GRAFANA_USER: str = os.environ.get("GRAFANA_USER", "admin")
GRAFANA_PASSWORD: str = os.environ.get("GRAFANA_PASSWORD", "admin")
DATASOURCE_NAME: str = "paper-sqlite"
# 库文件的字面绝对路径（Grafana 进程按此路径读 SQLite，相对路径不可靠）
DATASOURCE_DB_PATH: str = "D:/Code/Projects/vnpy/research/paper/paper.db"
DASHBOARD_UID: str = "paper-monitor"
PLUGIN_TYPE: str = "frser-sqlite-datasource"
HEALTH_TIMEOUT_S: int = 60


def api_request(method: str, path: str, payload: dict[str, Any] | None = None,
                ) -> tuple[int, dict[str, Any]]:
    """带 Basic Auth 的 Grafana API 调用，返回 (HTTP 状态码, 响应 JSON)。"""
    body: bytes | None = (
        json.dumps(payload).encode("utf-8") if payload is not None else None
    )
    request = urllib.request.Request(
        GRAFANA_URL + path, data=body, method=method,
        headers={"Content-Type": "application/json"},
    )
    import base64
    token: str = base64.b64encode(
        f"{GRAFANA_USER}:{GRAFANA_PASSWORD}".encode("utf-8")
    ).decode("ascii")
    request.add_header("Authorization", f"Basic {token}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            text: str = response.read().decode("utf-8")
            data: dict[str, Any] = json.loads(text) if text else {}
            return response.status, data
    except urllib.error.HTTPError as exc:
        return exc.code, {"message": exc.read().decode("utf-8", "replace")}


def wait_for_grafana() -> None:
    """轮询 /api/health 直到 Grafana 存活（超时抛错）。"""
    deadline: float = time.time() + HEALTH_TIMEOUT_S
    while time.time() < deadline:
        status, _ = api_request("GET", "/api/health")
        if status == 200:
            print("Grafana 存活: /api/health -> 200")
            return
        time.sleep(2)
    raise RuntimeError(f"Grafana 在 {HEALTH_TIMEOUT_S}s 内未就绪")


def ensure_datasource() -> str:
    """按名字查找数据源，不存在则创建；返回其 uid。"""
    status, found = api_request("GET", f"/api/datasources/name/{DATASOURCE_NAME}")
    if status == 200:
        print(f"数据源已存在: {DATASOURCE_NAME} (uid={found.get('uid')})")
        return str(found["uid"])
    status, created = api_request("POST", "/api/datasources", payload={
        "name": DATASOURCE_NAME,
        "type": PLUGIN_TYPE,
        "access": "proxy",
        "jsonData": {"path": DATASOURCE_DB_PATH},
    })
    if status not in (200, 201):
        raise RuntimeError(f"创建数据源失败({status}): {created}")
    print(f"数据源已创建: {DATASOURCE_NAME} -> {DATASOURCE_DB_PATH}")
    return str(created["datasource"]["uid"])


def check_datasource_health(uid: str) -> None:
    """调数据源 /health，确认 SQLite 文件可读。"""
    status, health = api_request("GET", f"/api/datasources/uid/{uid}/health")
    print(f"数据源健康检查({status}): {health}")
    if status != 200:
        raise RuntimeError(f"数据源健康检查失败({status}): {health}")


def equity_panel(ds_uid: str) -> dict[str, Any]:
    """面板 1：净值曲线，replay/live 两个查询同图异色。"""
    def equity_query(source: str, alias: str) -> dict[str, Any]:
        return {
            "refId": source.upper(),
            "format": "time_series",
            "queryText": (
                "SELECT CAST(strftime('%s', date) AS INTEGER) AS time, "
                f"equity AS \"{alias}\" FROM daily_status "
                f"WHERE source = '{source}' ORDER BY time"
            ),
            "datasource": {"type": PLUGIN_TYPE, "uid": ds_uid},
        }

    return {
        "id": 1, "type": "timeseries", "title": "净值曲线（蓝=重放历史, 橙=实盘仿真）",
        "gridPos": {"h": 10, "w": 14, "x": 0, "y": 0},
        "datasource": {"type": PLUGIN_TYPE, "uid": ds_uid},
        "fieldConfig": {"defaults": {
            "unit": "number", "custom": {
                "drawStyle": "line", "lineWidth": 2, "fillOpacity": 10,
                "showPoints": "never", "spanNulls": True,
            }},
            "overrides": [
                {"matcher": {"id": "byName", "options": "净值(replay)"},
                 "properties": [{"id": "color", "value": {
                     "mode": "fixed", "fixedColor": "#5794F2"}}]},
                {"matcher": {"id": "byName", "options": "净值(live)"},
                 "properties": [{"id": "color", "value": {
                     "mode": "fixed", "fixedColor": "#FF780A"}}]},
            ]},
        "options": {"legend": {"displayMode": "list", "placement": "bottom"},
                    "tooltip": {"mode": "multi"}},
        "targets": [
            equity_query("replay", "净值(replay)"),
            equity_query("live", "净值(live)"),
        ],
    }


def timing_panel(ds_uid: str) -> dict[str, Any]:
    """面板 4：择时状态阶跃图（0=空仓 1=持仓，阶梯后置）。"""
    return {
        "id": 4, "type": "timeseries", "title": "择时状态（1=持仓 0=空仓）",
        "gridPos": {"h": 10, "w": 10, "x": 14, "y": 0},
        "datasource": {"type": PLUGIN_TYPE, "uid": ds_uid},
        "fieldConfig": {"defaults": {
            "min": 0, "max": 1.2,
            "custom": {"drawStyle": "line", "lineWidth": 2, "fillOpacity": 25,
                       "showPoints": "never", "spanNulls": True,
                       "lineStyle": {"step": "after"}}},
            "overrides": [{"matcher": {"id": "byName", "options": "timing_hold"},
                           "properties": [{"id": "color", "value": {
                               "mode": "fixed", "fixedColor": "#73BF69"}}]}]},
        "options": {"legend": {"displayMode": "list", "placement": "bottom"},
                    "tooltip": {"mode": "single"}},
        "targets": [{
            "refId": "A", "format": "time_series",
            "queryText": (
                "SELECT CAST(strftime('%s', date) AS INTEGER) AS time, "
                "timing_hold AS timing_hold FROM daily_status ORDER BY time"
            ),
            "datasource": {"type": PLUGIN_TYPE, "uid": ds_uid},
        }],
    }


def holdings_panel(ds_uid: str) -> dict[str, Any]:
    """面板 2：当前持仓明细表（最新一天的 holdings JSON 展开）。"""
    return {
        "id": 2, "type": "table", "title": "当前持仓明细（最新判定日）",
        "gridPos": {"h": 10, "w": 8, "x": 0, "y": 10},
        "datasource": {"type": PLUGIN_TYPE, "uid": ds_uid},
        "fieldConfig": {"defaults": {}, "overrides": []},
        "options": {"showHeader": True},
        "targets": [{
            "refId": "A", "format": "table",
            "queryText": (
                "SELECT date AS 判定日, json_each.value AS 持仓合约 "
                "FROM daily_status, json_each(daily_status.holdings) "
                "WHERE date = (SELECT MAX(date) FROM daily_status)"
            ),
            "datasource": {"type": PLUGIN_TYPE, "uid": ds_uid},
        }],
    }


def trades_panel(ds_uid: str) -> dict[str, Any]:
    """面板 3：最近 50 笔交易列表（重放与实盘都有 source 标记）。"""
    return {
        "id": 3, "type": "table", "title": "最近交易（最多 50 笔）",
        "gridPos": {"h": 10, "w": 16, "x": 8, "y": 10},
        "datasource": {"type": PLUGIN_TYPE, "uid": ds_uid},
        "fieldConfig": {"defaults": {}, "overrides": []},
        "options": {"showHeader": True},
        "targets": [{
            "refId": "A", "format": "table",
            "queryText": (
                "SELECT ts AS 时间, symbol AS 合约, direction AS 方向, "
                "price AS 成交价, qty AS 数量, fee AS 手续费, source AS 来源 "
                "FROM trades ORDER BY ts DESC LIMIT 50"
            ),
            "datasource": {"type": PLUGIN_TYPE, "uid": ds_uid},
        }],
    }


def create_dashboard(ds_uid: str) -> None:
    """组装并创建/覆盖看板（overwrite=True 保证脚本幂等）。"""
    dashboard: dict[str, Any] = {
        "uid": DASHBOARD_UID,
        "title": "OKX 模拟盘监控",
        "tags": ["paper", "okx"],
        "timezone": "browser",
        "schemaVersion": 39,
        "refresh": "1m",
        "time": {"from": "now-13M", "to": "now"},
        "panels": [
            equity_panel(ds_uid), timing_panel(ds_uid),
            holdings_panel(ds_uid), trades_panel(ds_uid),
        ],
    }
    status, result = api_request("POST", "/api/dashboards/db", payload={
        "dashboard": dashboard, "overwrite": True,
    })
    if status != 200:
        raise RuntimeError(f"创建看板失败({status}): {result}")
    print(f"看板已就绪: {result.get('url')}")


def main() -> int:
    """入口：等待存活 -> 数据源 -> 健康检查 -> 看板。"""
    wait_for_grafana()
    ds_uid: str = ensure_datasource()
    check_datasource_health(ds_uid)
    create_dashboard(ds_uid)
    print(f"访问地址: {GRAFANA_URL}/d/{DASHBOARD_UID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
