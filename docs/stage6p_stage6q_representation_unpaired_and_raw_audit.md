# Stage 6P/6Q：Representation × Unpaired Release 与 Waymo raw interaction audit

## 1. 本轮问题

本轮不训练新checkpoint、不扩大Waymo、不重跑nuPlan simulation，回答两个问题：

1. Stage6L配对纯纵向实验中明显更强的ego13，进入真正log-disjoint、异场景unpaired release后，是否仍比full64可靠；
2. Stage6O的`intermittent-following=0`究竟是Waymo full51原始数据缺失，还是正式builder/window sampling结构性过滤。

Stage6O v1保持原状态`FROZEN_BLOCKED_WAYMO_COVERAGE_NOT_TRAINING`，其配置和输出没有修改或覆盖。

## 2. Stage 6P 冻结设计

- 复用Stage6H的800 pair、489 log和原样2400个release split；四种representation逐trial共用相同日志和场景。
- 比较full learned64、ego kinematic13D、handcrafted46D；neighbor-zero64D仅作diagnostic。
- 每种representation、每个n=200/250/300/400分别用200个`AA_CALIBRATION` trial冻结q95，再在独立200个`AA_EVALUATION`和200个`AB_EVALUATION`上计算FPR与检出率。
- 每种representation使用自己的冻结median-heuristic RBF bandwidth。禁止跨representation比较raw MMD²；只比较FPR、检出率、`detection-FPR`和同trial告警差异。

## 3. Stage 6P 结果

| representation | n=200 FPR / detection | n=250 FPR / detection | n=300 FPR / detection | n=400 FPR / detection |
|---|---:|---:|---:|---:|
| full64 | 7.5% / 31.5% | 4.5% / 30.0% | 2.0% / 26.0% | 4.5% / 63.5% |
| ego13 | 2.0% / 100.0% | 4.0% / 100.0% | 3.5% / 100.0% | 1.5% / 100.0% |
| handcrafted46 | 3.5% / 68.0% | 5.0% / 93.5% | 6.5% / 94.5% | 3.5% / 100.0% |
| neighbor-zero64 diagnostic | 4.0% / 65.0% | 5.5% / 91.0% | 2.5% / 93.5% | 1.0% / 100.0% |

n=400时，ego13相对full64检出率差为`+36.5pp`；200个相同A/B release中有73个仅ego13告警、0个仅full64告警，McNemar exact双侧`p=2.12e-22`。两个方向ego13均为100/100检出，full64分别为65/100和62/100。

因此答案是：**ego13在真正unpaired release条件下仍显著比当前full64可靠**。这不是因为ego13的raw MMD²更大，而是它在各自独立A/A calibration后同时取得更低FPR与更高A/B detection。

这个结果不表示neighbor/context无用。它只说明当前checkpoint的64D表示对这组受控纵向版本差异没有充分保留ego运动学信号。论文定义保持：

> behavior style = ego response conditioned on traffic / interaction context

## 4. Stage 6Q raw-source审计设计

- 逐条读取生成full51的Waymo原始TFRecord 00000–00050，共51文件、24872 scenario；不调用正式builder的neighbor-valid `>=0.8`筛选。
- 以逐帧ego坐标系几何proxy动态选择最近前车：纵向0–120m、横向主规则3m、航向差不超过45度；同时审计2m和4m敏感性。
- 审计全部合格vehicle窗口、正式前64 target sampling、正式builder retained inventory三层漏斗。
- 事件包括lead entry、lead exit、`intermittent <0.8`、`intermittent <0.5`、front identity switch、free-flow→closing→following、following→free-flow。
- 几何动态lead只用于coverage漏斗，不替代正式lane-aware语义。

## 5. Stage 6Q 结果与根因

3m主规则结果：

| 漏斗 | 窗口 | lead entry | lead exit | intermittent <0.8 | intermittent <0.5 | identity switch | free→closing→following | following→free |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw全部合格vehicle | 182837 | 31555 | 44349 | 54829 | 31275 | 29103 | 3465 | 43153 |
| 正式前64 target sampling | 168191 | 28423 | 40730 | 50045 | 28622 | 25994 | 3018 | 39676 |
| 正式builder retained | 164871 | 27321 | 39814 | 48240 | 27284 | 25639 | 2850 | 38700 |

2m/4m敏感性下，raw `intermittent <0.8`仍分别为53448/51109，均远高于Stage6O冻结门槛5000。故根因不是原始full51缺数据，而是builder的结构性语义：

1. `assign_stage5d_slots`只在窗口参考帧执行一次；
2. 选中的固定front track随后必须通过整窗`min_valid_ratio=0.8`的sanitize；
3. 输出front slot因此天然偏向“全空”或“至少80%持续有效”，动态entry/exit和identity switch无法被真实保留；
4. Stage6O基于这个静态slot mask分类，得到`intermittent=0`是预期后果。

## 6. 决策

下一步是**优先修builder，不扩大Waymo，也不开始Interaction-aware v2训练**：

1. 新建版本化builder，支持逐帧dynamic front assignment、front identity序列和显式有效mask；
2. 为entry/exit处增加temporal continuity与短缺失容忍，但不降低Stage6O的5000门槛；
3. 生成全新的Waymo数据版本，旧full51与Stage6O v1保持不变；
4. 对新数据重新执行质量审计与Stage6O freeze；
5. 只有新版本通过coverage、防泄漏、finite/shape和域内非劣性准备门后，才继续Interaction-aware v2训练准备。

权威输出：

- `outputs/stage6p_representation_unpaired_release_v1/`
- `outputs/stage6q_waymo_raw_interaction_coverage_v1/`
