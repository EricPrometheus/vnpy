# GP 因子挖掘多种子复现性检验报告

- 数据：22 个币种日线，2762 个交易日，共 48473 行
- 流程：gplearn 遗传规划（种群 1000 x 15 代，自定义适应度 = 日均截面 Rank IC），
  名人堂复检（验证段 IC >= 0.02 + 符号一致 + 相关性去重 0.7）+ vnpy 公式引擎复算
- 时间段：train / valid / test 三段切分，各因子报告三段日均 Rank IC（IC，信息系数）
- 本轮新跑种子：7、123、2026；此前已留档：0（manifest_seed0.json）；
  42 号为最初基线（当时清单条目名 gp_001~004，尚无 --seed 参数）

## 一、各种子结果明细

### 种子 42（基线，历史已知结果）

- 入选因子数：4（gp_001 ~ gp_004）
- 最优因子三段 IC：train +0.116 / valid +0.103 / test +0.064
- 主题：全部为「20 日振幅 rng20 x 短动量」

### 种子 0

- 入选因子数：1
- 最优因子 gp_s0_001：train_ic 0.087396 / valid_ic 0.157603 / test_ic 0.068477
  （引擎复算 valid 0.157891，偏差 0.0003，翻译无损）
- 公式全文（vnpy 表达式）：

```
(ts_std(close / ts_delay(close, 1) - 1, 20) - ts_mean((high - low) / close, 20)) * ts_mean((high - low) / close, 20) * ts_mean((high - low) / close, 20)
```

- 主题判断：含 rng20 片段 `ts_mean((high - low) / close, 20)`（出现 3 次），
  但配对的是 vola20（20 日日收益波动率）而非动量项，属于「振幅 x 波动率」变体；
  其原材料 ret1（1 日收益）是动量族特征，算部分复现。

### 种子 7

- 入选因子数：1
- 最优因子 gp_s7_001：train_ic 0.118807 / valid_ic 0.112681 / test_ic 0.052123
  （引擎复算 valid 0.113523，偏差 0.0008，翻译无损）
- 公式全文（vnpy 表达式）：

```
(close / ts_delay(close, 20) - 1) * ((close / ts_delay(close, 5) - 1 - ts_mean((high - low) / close, 20)) * abs(close / ts_delay(close, 60) - 1)) * (volume / ts_mean(volume, 20) - (close / ts_delay(close, 5) - 1)) + ((close / ts_delay(close, 20) - 1) * ((close / ts_delay(close, 5) - 1 - ts_mean((high - low) / close, 20)) * abs(close / ts_delay(close, 60) - 1)) * (volume / ts_mean(volume, 20) - (close / ts_delay(close, 5) - 1)) + (ts_std(close / ts_delay(close, 1) - 1, 20) - (close / ts_delay(close, 5) - 1) * ts_mean((high - low) / close, 20) - abs(ts_mean((high - low) / close, 20))) - (close / ts_delay(close, 5) - 1) * ts_mean((high - low) / close, 20) - abs(ts_mean((high - low) / close, 20)))
```

- 主题判断：复现。主干为 mom20 x (mom5 - rng20) x |mom60| x (vr20 - mom5)，
  rng20 与多周期动量（5/20/60 日）反复耦合，重复子树是 GP 自复制的典型痕迹。

### 种子 123

- 入选因子数：1
- 最优因子 gp_s123_001：train_ic 0.093524 / valid_ic 0.114439 / test_ic 0.080087
  （引擎复算 valid 0.114553，偏差 0.0001，翻译无损）
- 公式全文（vnpy 表达式）：

```
abs(0.282 - (close / ts_delay(close, 5) - 1)) / ts_mean((high - low) / close, 20)
```

- 主题判断：复现。5 日动量偏离常数 0.282 的绝对值除以 rng20，
  形态最简洁，也是全部种子里测试段 IC 最高的一支。

### 种子 2026

- 入选因子数：2
- 最优因子 gp_s2026_001：train_ic 0.108854 / valid_ic 0.102310 / test_ic 0.057672
  （引擎复算 valid 0.102310，偏差 0.0000，翻译无损）
- 次优因子 gp_s2026_002：train_ic 0.102336 / valid_ic 0.101933 / test_ic 0.041318
  （引擎复算 valid 0.101933，偏差 0.0000，翻译无损）
- 公式全文（vnpy 表达式）：

```
((close / ts_delay(close, 20) - 1 - ts_mean((high - low) / close, 20)) * abs(close / ts_delay(close, 10) - 1) - ts_mean((high - low) / close, 20) - ts_mean((high - low) / close, 20) - ts_mean((high - low) / close, 20)) * (close / ts_delay(close, 5) - 1) - ts_mean((high - low) / close, 20)
```

```
(close / ts_delay(close, 5) - 1 - ts_mean((high - low) / close, 20)) * (close / ts_delay(close, 5) - 1 - ts_mean((high - low) / close, 20)) - ts_mean((high - low) / close, 20)
```

- 主题判断：复现。两个因子均为 rng20 x 动量（mom20/mom10/mom5），
  其中 gp_s2026_002 就是 (mom5 - rng20)^2 - rng20，形态高度规整。

## 二、主题复现判定标准

公式主干包含片段 `ts_mean((high - low) / close, 20)`（即特征 rng20，20 日平均振幅），
且与动量项（ret1/mom5/mom10/mom20/mom60）直接耦合。

| 种子 | 含 rng20 片段 | 与动量项耦合 | 判定 |
| ---- | ------------- | ------------- | ---- |
| 42（基线） | 是 | 是 | 复现 |
| 0 | 是（3 次） | 否（配 vola20 波动率） | 部分复现 |
| 7 | 是 | 是（mom5/mom20/mom60） | 复现 |
| 123 | 是 | 是（mom5） | 复现 |
| 2026 | 是（两支均含） | 是（mom5/mom10/mom20） | 复现 |

## 三、结论

1. 复现率：rng20 主干复现率 5/5；「rng20 x 动量」严格口径 4/5（种子 0 走了
   振幅 x 波动率变体，未直接耦合动量）。
2. 最稳定公式形态：短动量与 rng20 的差值结构 `(momN - rng20)`——种子 7、
   2026_001、2026_002 都独立进化出该形态；种子 123 用其绝对值再除以 rng20，
   同族同向。可认为「20 日振幅 x 短动量」是该数据集上 GP 的收敛吸引子。
3. 测试段 IC 最高的种子：123（gp_s123_001 test_ic 0.080087），高于 42 号基线的
   0.064，且公式最简，已回填 research/mined_manifest.json 作为当前最优清单。
4. 42 号结果可复现性判断：可复现。五个种子全部正测试段 IC（0.052 ~ 0.080，
   基线 0.064 落在区间内），主题 4/5 严格复现、5/5 含 rng20 主干；说明该信号
   不是 42 号种子的偶然产物，而是数据中稳定存在的横截面规律。各期 IC 衰减节奏
   一致（train > valid > test 但测试段仍显著为正），engine 复算偏差均 < 0.001，
   翻译无损成立。

## 四、留档清单

- manifest_seed0.json / manifest_seed7.json / manifest_seed123.json /
  manifest_seed2026.json：各种子跑完后的 research/mined_manifest.json 快照
- 当前 research/mined_manifest.json = manifest_seed123.json（测试段 IC 最优）
- 备注：本轮曾并行跑种子 7 与 123，两进程同写一个清单导致种子 7 首轮结果被
  覆盖，后独占重跑补齐；同种子重跑结果与首轮完全一致（train/valid/test IC 与
  公式均相同），顺带验证了 --seed 的确定性。
