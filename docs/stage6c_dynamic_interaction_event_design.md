# Stage 6C：Dynamic Interaction Exposure 与 Event-specific Style Diagnosis

Stage 6C 是 Stage 6A / Stage 6B 之后新增的一层诊断协议，不重写 Stage 6A 的 unpaired BDD，也不替代 Stage 6B 的 static map ODD / behavior-event bin 报告。它的目标是把 Stage 6B 中仍未解释的 drift 进一步拆成：**动态交互暴露** 与 **事件内风格响应**。

## 1. 为什么 Stage 6B 之后需要 Stage 6C

Stage 6A 已经建立三类 unpaired BDD 对照：

- `negative_control_random`：同分布随机拆分不应产生显著漂移；
- `pseudo_agg_vs_cons`：伪 aggressive vs conservative 风格应产生高 BDD；
- `scene_confounding`：构造的场景混杂拆分也会产生高 BDD。

Stage 6B 进一步加入两层分析：

1. **Static Map ODD bins**：`odd_map_complexity_bin`、`odd_lane_count_bin`、`odd_curvature_bin`，用于检查静态道路几何/HD-map 上下文是否解释 drift；
2. **Behavior-event bins**：`event_following_bin`、`event_cut_in_bin`、`event_lane_change_bin`、`event_yielding_bin`、`event_lateral_activity_bin`，用于定位 drift 出现在哪类粗行为事件中。

Stage 6B 的关键发现是：static map ODD balancing 不能解释掉 `scene_confounding` drift，`pseudo_agg_vs_cons` 在 static ODD balancing 后仍保持高 BDD，而粗 behavior-event decomposition 显示 pseudo drift 在 high lateral activity / lane-change 中最强。这说明 remaining drift 可能来自动态交互暴露差异、行为代理特征 mismatch，或真实风格响应差异。

因此 Stage 6C 聚焦：

> Dynamic interaction exposure + event-specific style diagnosis.

## 2. 三类概念必须分离

### A. Static Map ODD

Static Map ODD 指静态道路几何和 HD-map 上下文，例如：

- map complexity；
- lane density / lane count；
- curvature；
- crosswalk；
- stop sign。

它是 Stage 6B 的主控制对象，用于公平性控制：A/B 是否面对相似道路几何。

### B. Dynamic Interaction Exposure

Dynamic Interaction Exposure 指 ego/target vehicle 暴露在什么交通交互情境中，例如：

- following pressure；
- cut-in exposure；
- overtake opportunity；
- dense traffic；
- front gap pressure；
- side gap pressure；
- yield conflict。

Stage 6C 的 `exposure_*` bins 属于这一类。它们可以作为后续 matching/control 的候选变量，但当前第一版仍是 proxy 定义。

### C. Behavior Outcome / Style

Behavior Outcome / Style 指驾驶员或模型实际做了什么，例如：

- ego lane change；
- hard brake；
- late braking；
- hesitation；
- assertive interaction；
- overtake executed；
- lateral unstable。

Stage 6C 的 `outcome_*` bins 属于这一类。它们主要用于 report/localization，不应当被当作纯 scenario-control 变量。原因是 outcome 本身可能由策略或驾驶风格生成；如果拿 outcome 做主匹配，可能会把真实风格差异抵消掉。

## 3. Embedding BDD 与 handcrafted features 的互补关系

Handcrafted features are not replaced by embedding.

- embedding / BDD 是 measurement layer；
- event-specific features 是 explanation layer。

Feature statistics 回答：

> Which known metric changed?

Embedding BDD 回答：

> Whether the overall behavior distribution changed, how large the drift is, and which cases are most representative.

Event-specific features then explain the BDD direction.

推荐论文表述：

> Embedding-based BDD provides a unified behavior distribution metric across heterogeneous driving events, while event-specific features provide semantic diagnosis of the detected drift.

## 4. Stage 6C 输出文件

### 4.1 Dynamic event bins

脚本：`tools/stage6c_build_dynamic_event_bins.py`

输出：

- `dynamic_event_bins.csv`
- `dynamic_event_bins.npy`
- `dynamic_event_bin_schema.json`
- `dynamic_event_bin_report.md`
- `dynamic_event_bin_warnings.json`

行对齐规则：`global_row` 按 `shard_manifest.json` 的 shard 顺序从 0 开始递增，`local_row` 是 shard 内行号。

### 4.2 Event style metrics

脚本：`tools/stage6c_build_event_style_metrics.py`

输出：

- `event_style_metrics.csv`
- `event_style_metrics.npy`
- `event_style_metric_schema.json`
- `event_style_metric_report.md`
- `event_style_metric_warnings.json`

缺失代理特征时对应 metric 为 `NaN`，不会静默填 0。

### 4.3 Event style report

脚本：`tools/stage6c_event_style_report.py`

输出：

- `event_bdd_summary.csv`
- `event_style_delta.csv`
- `event_report_card.md`
- `top_event_drift_cases.csv`
- `warnings.json`
- 可选图：`plots/event_bdd_bar.png`、`plots/event_style_delta_bar.png`

## 5. Dynamic exposure bins 定义

| Bin | 正类标签 | 负类标签 | 主要代理特征 |
|---|---|---|---|
| `exposure_following` | `following` | `not_following` | THW、front distance、front pressure、front relative speed |
| `exposure_cut_in` | `cut_in_exposure` | `no_cut_in_exposure` | cut-in count proxy、front pressure、front/side gap、yielding proxy |
| `exposure_overtake_opportunity` | `overtake_opportunity` | `no_overtake_opportunity` | front vehicle present、front relative speed、ego speed、front pressure |
| `exposure_dense_traffic` | `dense_traffic` | `normal_traffic` | neighbor count、interaction density、front/side/rear gaps |
| `exposure_front_pressure` | `high_front_pressure` | `low_front_pressure` | front pressure、front distance、THW、relative speed |
| `exposure_side_pressure` | `high_side_pressure` | `low_side_pressure` | left/right front/rear min gap |
| `exposure_gap_pressure` | `small_gap` | `normal_gap` | front/rear/left/right gap proxies |
| `exposure_yield_conflict` | `yield_conflict` | `no_yield_conflict` | yielding proxy、front pressure、gap pressure、cut-in/side pressure |
| `exposure_free_cruising` | `free_cruising` | `not_free_cruising` | low following/cut-in/front pressure/side pressure/lane-change exposure |

如果所需代理特征完全不可解析，输出 `unknown`。

## 6. Behavior outcome bins 定义

| Bin | 正类标签 | 负类标签 | 主要代理特征 |
|---|---|---|---|
| `outcome_ego_lane_change` | `lane_change` | `no_lane_change` | lane-change count/rate、yaw rate、heading change |
| `outcome_overtake_executed` | `overtake_executed` | `no_overtake_executed` | lane change、speed gain、acceleration |
| `outcome_hard_brake` | `hard_brake` | `no_hard_brake` | max decel/min acc、max abs accel、jerk、brake count |
| `outcome_late_brake` | `late_brake` | `not_late_brake` | hard brake under small THW/front gap/high pressure/low TTC |
| `outcome_hesitation` | `hesitation` | `no_hesitation` | lane-change duration、yaw sign changes、speed oscillation |
| `outcome_assertive_interaction` | `assertive_interaction` | `non_assertive_interaction` | acceleration/speed under gap/front/yield pressure |
| `outcome_stop_go` | `stop_go` | `not_stop_go` | low speed、stop count、jerk/speed oscillation |
| `outcome_lateral_unstable` | `lateral_unstable` | `lateral_stable_or_low_activity` | yaw rate、curvature、heading change、duration/oscillation |

Outcome bins 不能被解释为纯场景变量；它们是行为结果或风格表现。

## 7. Event-specific style metrics

Stage 6C 第一版计算以下 metric group：

- Following：THW、front distance、peak decel、jerk、front pressure；
- Cut-in response：reaction delay proxy、peak decel、min TTC、front gap、jerk after proxy；
- Lane-change：yaw rate、curvature、heading change、duration、front/rear gap、gap acceptance、lateral sharpness；
- Overtake：opportunity score、execution score、peak accel、jerk、speed gain；
- Yielding / assertiveness：yielding score、gap pressure、assertiveness、conflict accel、small-gap speed maintain；
- Hesitation：hesitation score、duration、yaw/speed oscillation、abort-like proxy；
- Hard braking / comfort：hard-brake score、peak decel、jerk、brake comfort；
- Free cruising：speed/acc/jerk/yaw-rate proxy、stability score。

## 8. 报告解释逻辑

`event_report_card.md` 会把 event-level BDD 与 metric delta 结合成自然语言结论。例如：

- following exposure 中，如果 B 的 min THW/front distance 更低，peak decel/jerk 更高，则解释为 closer-following and more abrupt braking；
- cut-in exposure 中，如果 B 的 min TTC 更低、peak decel 更高，则解释为 later reaction and harder braking；
- lane-change / side-pressure 中，如果 yaw rate 更高、duration 更短、accepted gap 更小，则解释为 sharper and more assertive lane changes；
- overtake opportunity 中，如果 execution score、peak accel、jerk 更高，则解释为更愿意 overtake 且加速更强；
- free cruising 中如果 BDD 很低，则解释为基础巡航稳定，drift 集中在交互事件。

## 9. 限制

1. 当前事件定义是 first-pass proxy，不是人工标注 ground truth。
2. `feature_schema.json` 中缺失代理特征时，只能输出 `unknown` 或 `NaN`。
3. 不同数据版本的 feature name 可能不同，因此脚本使用 alias resolution；但语义仍取决于原始特征质量。
4. 当前 Stage 6C 是 diagnosis layer，不自动执行 dynamic matching。
5. event bins 中 A/B 样本量可能严重不平衡，因此小 bin 的 BDD 会被 `--min_bin_size` 跳过。
6. top drift cases 基于 embedding 到 opposite centroid 的距离，是代表性检索 proxy，不等同于因果解释。

## 10. Future work

- 将 `exposure_*` bins 扩展为 dynamic matching keys；
- 使用 propensity weighting 控制 dynamic interaction exposure；
- 用更精确的 trajectory-level cut-in、merge、yield、overtake detector 替代 proxy；
- 在 report 中加入 case-level visualization；
- 区分 human data、model rollout、closed-loop policy 的 event-specific response。
