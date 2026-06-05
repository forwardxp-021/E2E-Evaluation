# Stage 6B：Baseline Comparison 与 Scenario-Controlled Drift Analysis

## 1. 为什么 Stage 6A 后还需要 Stage 6B
Stage 6A 已证明 BDD 能区分随机对照与伪风格漂移，但也显示 scene/proxy shift 会造成高漂移值。这里需要更精确区分：static map ODD mismatch can be reduced by map-derived ODD balancing, but dynamic interaction-exposure mismatch may remain. Stage 6B 用于回答：
- learned embedding/BDD 相比简单手工统计有什么增益；
- 在 unpaired A/B 中如何减少 scenario/ODD 混杂。

## 2. 为什么 feature mean baseline 不够
单变量均值差异会丢失：
- 多变量联合结构；
- 非线性边界；
- 时序与交互动态。
因此仅做 mean delta 容易欠检或误检。

## 3. 为什么增加 feature MMD 与 PCA-feature MMD
- **feature MMD** 在原始特征空间比较分布整体形状；
- **PCA-feature MMD** 在降噪后子空间比较主变化模式；
- 两者作为强基线，可和 embedding BDD 同台校准。

## 4. learned embedding BDD 的持续价值
- 编码时序-交互行为表示；
- 可支持 top-drift case retrieval；
- 提供统一 behavior space，便于横向实验对比。

## 5. 为什么需要 scenario balancing
unpaired log 对比通常存在 ODD 构成差异。若 A/B 在 proxy scene 分布不一致，raw BDD 可能混合“风格差异 + 场景差异”。按 proxy bin 做平衡能降低这种偏差。

## 6. raw / sliced / balanced 的区别
- **raw BDD**：直接比较原始 A/B；
- **scenario-sliced BDD**：分 bin 分别比较；
- **scenario-balanced BDD**：按 bin 下采样对齐后再比较整体。

## 7. 三类实验解释
- **negative_control_random**：同分布 sanity check，应无显著漂移；
- **pseudo_agg_vs_cons**：已知伪风格漂移，应显著；
- **scene_confounding**：proxy 场景漂移，应作为 confounding warning 解读。

## 8. 当前限制
- scene proxies 仍有限；
- speed proxy 可能在当前 schema 缺失或不稳定；
- 真正 ODD matching 需要 Stage 6C/6D 引入更丰富元数据。


## Stage 6B Map ODD Validation Results

| Experiment | ODD control bin | Raw BDD | ODD-balanced BDD | Reduction | Balanced N | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| scene_confounding | odd_map_complexity_bin | 0.1205 | 0.1233 | -2.28% | 4087 | static map complexity does not explain drift |
| scene_confounding | odd_lane_count_bin | 0.1205 | 0.1195 | +0.83% | 4144 | lane-density balancing has negligible effect |
| scene_confounding | odd_curvature_bin | 0.1205 | 0.1219 | -1.10% | 4672 | curvature balancing does not explain drift |
| pseudo_agg_vs_cons | odd_map_complexity_bin | 0.1696 | 0.1621 | +4.40% | 3711 | pseudo style drift remains high |
| pseudo_agg_vs_cons | odd_lane_count_bin | 0.1696 | 0.1547 | +8.75% | 3723 | pseudo style drift remains high |
| pseudo_agg_vs_cons | odd_curvature_bin | 0.1696 | 0.1602 | +5.54% | 4218 | pseudo style drift remains high |

All six ODD-balanced BDD runs used:
- odd_bins_total_rows = 164871
- odd_bins_valid_rows = 164871
- n_A_raw = 4917
- n_B_raw = 4917
- p_value = 0.009900990099009901 for both raw and balanced BDD
- no bins skipped

## Main conclusion

Map-derived ODD balancing does not explain away the scene_confounding drift. Across map complexity, lane-count context, and curvature controls, scene_confounding BDD remains around 0.12 and changes by only -2.28% to +0.83%. This indicates that the constructed scene_confounding split is not primarily driven by static map ODD mismatch. Instead, it is more likely driven by dynamic interaction-exposure and behavior-proxy factors, such as lateral activity, interaction pressure, and gap size.

In contrast, pseudo_agg_vs_cons remains strongly separated after all three static ODD controls. Although ODD balancing slightly reduces BDD by about 4.4% to 8.75%, the balanced BDD remains high, around 0.155 to 0.162, with significant permutation p-values. This supports the claim that pseudo aggressive-vs-conservative drift is robust to static map ODD balancing.

## Recommended Stage 6B interpretation

- Use raw BDD as the initial observed drift.
- Use static ODD-balanced BDD to test whether drift is explainable by static map context.
- If BDD remains high after static ODD balancing, the remaining drift may reflect behavior style shift or dynamic interaction-exposure shift.
- For pseudo_agg_vs_cons, the high remaining BDD supports style-shift robustness.
- For scene_confounding, the high remaining BDD suggests the split captures dynamic interaction/behavior-proxy confounding rather than static map ODD mismatch.

## Limitations

- Current map ODD bins control static HD-map context only.
- They do not directly control traffic density, cut-in exposure, front vehicle behavior, or other dynamic agent interactions.
- Dynamic interaction-exposure matching should be added in a later Stage 6C/6D.
- odd_intersection_bin currently needs refinement because it marks nearly all rows as intersection_like in the current dataset.


## Stage 6B Behavior-event BDD Decomposition

### pseudo_agg_vs_cons

| Behavior-event bin | Bin value | n_A | n_B | BDD_MMD | Interpretation |
|---|---:|---:|---:|---:|---|
| event_following_bin | following_proxy | 4917 | 4917 | 0.1661 | following style drift remains strong |
| event_cut_in_bin | cut_in_proxy | 2908 | 429 | 0.1538 | cut-in proxy drift exists but A/B count is imbalanced |
| event_cut_in_bin | no_cut_in_proxy | 2009 | 4488 | 0.1350 | drift also exists outside cut-in proxy |
| event_lane_change_bin | lane_change | 675 | 2756 | 0.2314 | strongest lane-change-related drift |
| event_lane_change_bin | no_lane_change | 4242 | 2161 | 0.1331 | lower drift without lane-change |
| event_yielding_bin | non_yielding_like | 1931 | 4454 | 0.1460 | yielding-related proxy shift |
| event_yielding_bin | yielding_like | 2986 | 463 | 0.1398 | yielding proxy drift but imbalanced |
| event_lateral_activity_bin | high | 487 | 2827 | 0.2403 | strongest high-lateral-activity drift |
| event_lateral_activity_bin | mid | 1393 | 1693 | 0.1860 | medium lateral activity drift |
| event_lateral_activity_bin | low | 3037 | 397 | 0.1277 | lowest lateral activity drift |

### scene_confounding

| Behavior-event bin | Bin value | n_A | n_B | BDD_MMD | Interpretation |
|---|---:|---:|---:|---:|---|
| event_following_bin | following_proxy | 4917 | 4917 | 0.1246 | following drift close to overall scene_confounding BDD |
| event_cut_in_bin | cut_in_proxy | 278 | 2576 | 0.1475 | cut-in proxy exposure is strongly imbalanced |
| event_cut_in_bin | no_cut_in_proxy | 4639 | 2341 | 0.2240 | strong drift outside cut-in proxy |
| event_lane_change_bin | lane_change | 602 | 2575 | 0.1936 | lane-change-related drift |
| event_lane_change_bin | no_lane_change | 4315 | 2342 | 0.1858 | drift also persists without lane-change |
| event_yielding_bin | non_yielding_like | 4524 | 2415 | 0.2169 | strong non-yielding-like drift |
| event_yielding_bin | yielding_like | 393 | 2502 | 0.1837 | yielding proxy exposure is imbalanced |
| event_lateral_activity_bin | high | 430 | 2621 | 0.1881 | high lateral activity drift |
| event_lateral_activity_bin | mid | 1375 | 1333 | 0.1399 | moderate drift |
| event_lateral_activity_bin | low | 3112 | 963 | 0.2931 | strongest drift in low lateral activity bin |

Notes:
- `event_low_speed_bin` and `event_high_speed_bin` are currently unknown/unavailable in both experiments.
- All reported p-values are significant at approximately `0.0099` in the current run.
- Behavior-event bins are diagnostic/reporting bins, not primary fairness-control bins.

## Main interpretation update

Behavior-event decomposition confirms that pseudo_agg_vs_cons drift is behaviorally interpretable. Its largest BDD values occur in high-lateral-activity and lane-change bins, where BDD reaches approximately 0.2403 and 0.2314. This is consistent with the pseudo aggressive-vs-conservative construction and shows that the behavior-event report layer can localize where style drift is expressed.

For scene_confounding, behavior-event decomposition shows a different pattern. The drift is not explained by static map ODD balancing, and the largest behavior-event BDD values appear in low lateral activity, no-cut-in proxy, non-yielding-like, and lane-change/no-lane-change bins. This suggests that the scene_confounding split captures dynamic interaction-exposure and behavior-proxy differences rather than pure static map ODD mismatch.

## Conceptual clarification

Stage 6 now distinguishes three layers:

1. Static Map ODD control
   - map complexity
   - lane-count context
   - curvature
   - used for fairness control

2. Dynamic interaction / behavior-exposure diagnostics
   - following
   - cut-in proxy
   - yielding proxy
   - lane-change
   - lateral activity
   - used for drift localization

3. Overall behavior distribution drift
   - raw BDD
   - ODD-balanced BDD
   - behavior-event BDD

Map ODD bins and behavior-event bins should not be confused. Static ODD bins test whether A/B faced similar road geometry. Behavior-event bins explain which driving tasks or interaction modes contain the drift.

## Added limitations

- low/high speed bins are unavailable in the current feature schema.
- cut-in and yielding bins are proxy definitions, not ground-truth event annotations.
- several behavior-event bins are highly A/B imbalanced, so they should be interpreted as localization signals rather than causal proof.
- dynamic interaction exposure matching should be developed in a later Stage 6C/6D.

## Stage 6C pointer

Static ODD control does not remove all dynamic interaction exposure confounding. Stage 6C introduces dynamic event diagnosis by separating static map ODD, dynamic interaction exposure, and behavior outcome/style, then reporting event-level BDD with event-specific style metric deltas.
