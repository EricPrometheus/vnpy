**1. 时间线：整体可行，两处必脱期**
第1-3月没问题。第4-6月“纯手写回测引擎”最危险——pandas（Python表格数据处理库）的向量化思维是零基础最陡的坎，行业共识转行做数据工程也需6-18个月才可雇佣。建议该里程碑后移至第9-10月。另一隐性风险：现有管线是AI（人工智能）代写的“假熟练”，手写里程碑的设计是对的，必须严格执行。

**2. 雇用接受度：分层看**
国内头部量化私募卡名校理工硕博，自学+作品集基本进不去；中小私募、金融科技公司、加密机构对学历宽松，认真实项目。数据工程师岗远比量化研究员宽松，SQL（数据库查询语言）是硬通货，作品集可过HR（人事）筛选。最被认可的作品集：真实端到端数据管道（采集→入库→转换→测试→定时调度），而非“又一个回测脚本”——26万根BTC（比特币）1分钟数据正是现成素材。

**3. 学习顺序：两处调整**
SQL提前到第4月与pandas并行——先入库再分析，直击数据岗面试重点；统计学不单独学，嵌入回测指标（夏普比率、最大回撤、过拟合检验）中学，省1-2个月。语法期压缩到2个月，直接用自有行情数据边查边写，比刷书快。

**4. 每周安排（10-12小时）**
- 工作日每天1小时：手写代码，AI只做讲评、禁止代写
- 周末4-6小时：做项目
- 里程碑改为：第3月独立调试改造现有管线 → 第6月行情数据入DuckDB（嵌入式分析数据库）+SQL查询层 → 第9月手写回测引擎 → 第12月端到端管道+模拟盘
- 求职定位改为“量化数据开发/金融科技后端”，比硬投“量化开发”命中率高数倍；MBA同学 network（人脉内推）是该画像最大杠杆，别浪费

Sources: [DataEngineer Academy 转行时间线](https://dataengineeracademy.com/blog/how-long-does-it-take-to-switch-into-data-engineering) · [2026 数据工程作品集指南](https://medium.com/@aidelearning/the-data-engineering-portfolio-that-actually-gets-you-hired-in-2026-649a7352c876) · [r/quant 无学历现实讨论](https://www.reddit.com/r/quantfinance/comments/1hr4c5j/is_it_possible_to_self_teach_the_necessary_skills)
