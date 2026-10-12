# OKX 模拟盘本地监控栈（SQLite + Grafana，全开源零费用）

模拟盘策略（`paper_strategy.py` + `run_paper.py`）每天 UTC 16:05 判定一次；
本监控栈把"每日状态 + 成交流水"落到本地 SQLite（`paper.db`），并用本地
Grafana 出四个看板面板：净值曲线 / 当前持仓 / 最近交易 / 择时状态。

```
策略每日报判 ──> paper_store.py ──> research/paper/paper.db (SQLite)
                                        │
重放历史 309 天 ──> backfill_db.py ──────┘
                                        │
                              Grafana (localhost:3000)
                              数据源: frser-sqlite-datasource
```

## 一、三条启动命令

以下命令都在**仓库根目录**（`D:\Code\Projects\vnpy`）的 Git Bash 里执行。

### 1) 启动 Grafana（本地看板）

```bash
cd research/paper/grafana/grafana-13.2.3
./bin/grafana.exe server --homepath "$(cygpath -w "$(pwd)")" \
  --config "$(cygpath -w "$(pwd)")\\conf\\defaults.ini" --packaging=windows
```

想丢到后台常驻（Git Bash 写法）：

```bash
cd research/paper/grafana/grafana-13.2.3
nohup ./bin/grafana.exe server --homepath "$(cygpath -w "$(pwd)")" \
  --config "$(cygpath -w "$(pwd)")\\conf\\defaults.ini" \
  --packaging=windows > server.log 2>&1 &
```

看板地址：**<http://localhost:3000/d/paper-monitor>**（Grafana 首页搜
"OKX 模拟盘监控"也能找到）。
默认账号 `admin/admin`——**首次登录请立即改密**（或用环境变量
`GRAFANA_USER` / `GRAFANA_PASSWORD` 给 `setup_grafana.py` 传新凭据）。

### 2) 启动策略（常驻，连 OKX 模拟盘）

```bash
nohup D:/veighna_studio/python.exe -X utf8 research/paper/run_paper.py \
  > research/paper/paper_run.log 2>&1 &
```

前置条件见 `run_paper.py` 文件头（OKX 凭据放 `research/paper/okx_setting.json`，
该文件已 gitignore，永不入库）。策略启动时会自动建监控库并打印
"监控数据库就绪"；每天北京时间 00:05 判定后自动把当日快照写进 SQLite。

### 3) 查看一键状态

```bash
D:/veighna_studio/python.exe research/paper/status.py
```

输出：当前持仓、最新净值、当日成交、累计费用、两表行数。

## 二、实盘数据与重放历史在图上如何区分

`daily_status.source` 列标记数据来源：

| source    | 含义                     | 写入者          |
| --------- | ------------------------ | --------------- |
| `replay`  | 事件化重放的历史 309 天  | `backfill_db.py` |
| `live`    | 策略实时仿真写入的新数据 | `paper_strategy.py` |

净值面板（面板 1）里两段数据是**同一个查询不同过滤**：`replay` 段蓝色、
`live` 段橙色，同图直接对比"策略实盘表现 vs 回测重放"。实盘运行的第一天起，
橙色曲线就会从当天开始生长，与蓝色历史曲线无缝衔接（口径一致：现金 + Σ持仓 × 当日收盘）。
交易表（面板 3）的"来源"列同样用 `replay` / `live` 区分。

重放段的两个特殊口径（原样保留回测器输出，不做修饰）：
- `cash` 存的是回测器记录的**当日现金变动**（不是绝对现金余额），实盘段是真实现金；
- 重放 `trades.fee` 统一为 0（回测未拆分单笔费用，当日费用汇总在
  `daily_status.fees` 里）；实盘段逐笔费用按 0.1% 记录。

## 三、首次部署（本机已完成，换机时照做）

1. 解压 Grafana 到 `research/paper/grafana/`（zip 在同目录，已 gitignore）；
2. 装插件：
   ```bash
   cd research/paper/grafana/grafana-13.2.3
   export https_proxy=http://127.0.0.1:10809   # 需要代理时
   ./bin/grafana.exe cli plugins install frser-sqlite-datasource
   ```
   注意：CLI 可能把插件装到 `<上级目录>/data/plugins`，把它挪到
   `grafana-13.2.3/data/plugins/` 下即可；
3. 启动 Grafana（上面的命令 1）；
4. 配置数据源 + 看板（幂等，可重复执行）：
   ```bash
   cd ../../..   # 回仓库根目录
   D:/veighna_studio/python.exe research/paper/setup_grafana.py
   ```
5. 回填历史（幂等；重灌前删掉 `research/paper/paper.db` 即可全量重来）：
   ```bash
   D:/veighna_studio/python.exe research/paper/backfill_db.py
   ```

## 四、文件清单

| 文件 | 作用 |
| --- | --- |
| `paper_store.py` | SQLite 落盘层（标准库 sqlite3，建表/UPSERT/插入/统计） |
| `backfill_db.py` | 读 `replay_composite_equity_ma30.csv` 回填 309 天历史（source=replay） |
| `setup_grafana.py` | Grafana HTTP API 一键配置（数据源 + 四面板看板，幂等） |
| `status.py` | CLI 一键状态（持仓/净值/费用/行数） |
| `paper.db` | 监控库（gitignore；Grafana 数据源指向它） |
| `grafana/` | Grafana 本体 + 数据目录 + 插件（整个目录 gitignore） |

`daily_status` 表：`date` 主键、`timing_hold`(0/1)、`holdings`(JSON 数组)、
`equity`、`cash`、`n_trades`(当日成交笔数)、`fees`(截至当日累计费用)、`source`。
`trades` 表：`ts`、`symbol`、`direction`、`price`、`qty`、`fee`、`source`。

## 五、常见问题

- **看板无数据**：先跑 `status.py` 确认 `paper.db` 有行数；再确认 Grafana
  数据源健康（`setup_grafana.py` 会自动打印）；数据源里的库路径是字面绝对
  路径，换机器/换目录后需要重跑 `setup_grafana.py`。
- **改了库路径**：重跑 `setup_grafana.py`（会按脚本里的 `DATASOURCE_DB_PATH`
  覆盖更新数据源配置）。
- **安全提醒**：Grafana 默认 `admin/admin` 只适合本机自用；若监听非
  localhost 请务必改密并配置防火墙。插件允许 SQL 查询，默认禁止 ATTACH
  外部库，但 Grafana 用户能看到本机该 SQLite 的全部数据。
