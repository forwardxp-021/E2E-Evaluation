# Stage 6S-v2 interaction benchmark 开发与 confirmation 冻结报告

## 1. 本阶段边界

本阶段对应 Issue #261。目标不是证明新模型有效，而是在不读取任何 embedding、BDD/MMD、
不训练新 checkpoint 的前提下，开发一个能够约束“模型不能退化成复杂版 ego13”的
interaction-dominant nuPlan benchmark，并冻结独立 confirmation roster。

本阶段未复用 Stage6S-v1 的24个场景。pre-treatment inventory 从扩大的 Pittsburgh nuPlan DB
中审计15,779个候选，使用 `scenario_tag.agent_track_token` 对持续front exposure、初始gap、
ego有效速度和closing pressure做筛选，共得到301个eligible场景、19个日志。筛选过程未读取
planner outcome、embedding或BDD/MMD。

## 2. 最终 interaction treatment

两个PDM Closed planner的speed schedule、fallback desired speed、最大加/减速度和lateral offsets
完全一致。唯一差异如下：

| 参数 | short-headway v2 | long-headway v2 |
|---|---:|---:|
| `idm_policies.headway_time` | 0.8 s | 2.2 s |
| `idm_policies.min_gap_to_lead_agent` | 0.5 m | 2.5 m |

因此本benchmark操纵的是following interaction response，而不是横向风格、速度上限或动力学能力。
development roster包含24个same-scenario pair、4个日志，每日志6个场景；24个场景均为
`following_lane_with_slow_lead`，与Stage6S-v1场景token重叠为0。48条official nuPlan rollout
全部成功，平均每pair 32.786秒。

## 3. Development realized mechanism

THW从第一版分析开始就只保留有限且物理有效的 `0 < THW < 20 s`，显式排除999、cap、
非有限值；先在每个pair内取median，再对24个pair取median。

短headway减长headway的development结果为：

| 指标 | 中位差 | 方向一致pair比例 | 门禁 |
|---|---:|---:|---|
| mean speed | +0.259 m/s | — | `|Δ| <= 1.0`，通过 |
| RMS accel | +0.225 m/s² | — | `|Δ| <= 0.75`，通过 |
| median front gap | -4.284 m | 91.7% | 通过 |
| median finite THW | -2.660 s | 100.0% | 通过 |
| closing accel response | +0.045 m/s² | 50.0% | 不通过 |
| following-pressure accel response | +0.045 m/s² | 50.0% | 不通过 |

24/24 pair均有有效front。预冻结规则要求四项interaction指标至少两项通过，实际front gap和
finite THW两项通过；同时mean speed与RMS accel差异均保持在小差异门槛内。因此状态为
`DEVELOPMENT_MECHANISM_PASS_CONFIRMATION_FREEZE_ALLOWED`。这说明PDM/nuPlan能够构造本研究所需的
受控interaction benchmark，但不能声称本批已稳定操纵closing/following acceleration response。

## 4. Confirmation set 冻结与独立性

confirmation在development机制通过后冻结，但选择时只读取同一份pre-treatment inventory。
development结果只作为“允许冻结”的布尔授权，不进入scenario排序。固定内容包括planner参数、
scenario selection rule、interaction metrics、mechanism gates、THW处理和统计口径。
confirmation仍按同一机制门禁判定，并预冻结按`log_name`聚类的10,000次bootstrap percentile 95%
区间（seed 620261）；这样不会把同一日志内多个场景误当作完全独立样本。区间不用于事后改门槛。

最终roster为80个complete-pair目标场景、15个日志，其中77个slow-lead、3个lead。独立性审计为：

- 与development的log重叠：0；
- 与development的scenario token重叠：0；
- 与Stage6S-v1的scenario token重叠：0；
- confirmation outcome、embedding、BDD/MMD读取：否；
- confirmation rollout启动：否；
- checkpoint训练或新模型评估启动：否。

roster SHA-256为
`a2360bf38f8c60fc481226d580475cb651cf503cb46c5b6167ad1d916d50d1e2`，冻结后不可根据未来
old64、ego13或new64结果修改场景、planner、指标或门槛。

## 5. 结论与训练授权边界

Stage6S-v2已经达到本阶段的三个结束条件：development机制门禁稳定通过、独立confirmation
roster冻结、benchmark设计不依赖任何embedding结果。结合Stage6O-v2数据门禁已经通过，当前已
具备**准备启动** Interaction-aware v2正式训练的条件。

这不是已经启动训练的声明，也不是新64D有效的证据。下一代64D仍须以恢复old64丢失的纵向风格
敏感性为第一目标，使unpaired detection从约63.5%向ego13的100%靠近；Stage6S-v2 confirmation
用于验证这种提升没有让64D退化为只依赖ego运动学的复杂版ego13。正式训练及后续模型评估必须
另行授权，且confirmation一旦开始仍不得按模型结果回调benchmark。
