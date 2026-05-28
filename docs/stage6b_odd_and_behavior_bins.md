# Stage 6B：ODD bins 与 Behavior-event bins 分层设计

Stage 6B 明确分离两层分箱：
- **Map-derived ODD bins**：用于 A/B 公平性控制（外部路况可比性）。
- **Behavior-event bins**：用于工程定位（漂移出现在哪类驾驶任务）。

`lateral_activity` 是行为污染变量（会被驾驶风格直接影响），因此只能作为 `event_lateral_activity_bin` 报告维度，不可作为主 ODD 控制变量。

## ODD bins 含义

> 说明：`odd_map_complexity_bin`、`odd_lane_count_bin`、`odd_curvature_bin` 是当前 Stage 6B 推荐的 **static map ODD 主控制 bins**。`odd_intersection_bin` 在当前 full51 数据上分布极不均衡（几乎全为 intersection_like），暂不推荐作为 primary balancing key。
- `odd_crosswalk_bin`：附近是否有人行横道。
- `odd_stop_sign_bin`：附近是否有 stop sign。
- `odd_curvature_bin`：直道/中等曲率/高曲率。
- `odd_intersection_bin`：路口样复杂场景代理。
- `odd_map_complexity_bin`：低/中/高地图复杂度。
- `odd_lane_count_bin`：简单/多车道/高密度车道上下文。

## Behavior-event bins 含义
- `event_following_bin`
- `event_cut_in_bin`（无显式标注时为 proxy）
- `event_lane_change_bin`
- `event_low_speed_bin` / `event_high_speed_bin`
- `event_yielding_bin`
- `event_lateral_activity_bin`

## 指标解释
- `BDD_overall`：未控制总体差异。
- `BDD_odd_balanced`：按 ODD bins 平衡后的总体差异（Stage 6B 主结论优先）。
- `BDD_following` / `BDD_cut_in` / `BDD_lane_change`：行为事件定位指标。
- `BDD_intersection`：路口样场景下差异。

## 为什么行为事件不能做主控制
行为事件（尤其换道、横摆、跟车压迫）可能由策略本身产生；若用于主匹配，会抵消真实风格差异。

> 结论：behavior-event bins 用于 localization/reporting，不用于 fairness control。

## 当前限制
1. 需要原始 Waymo scenario 文件才能提取 map ODD。
2. speed/traffic-light 在部分数据中缺失或不稳定。
3. cut-in 可能只能用 proxy。
4. 车道拓扑精确匹配仍可继续增强。
5. 当前 map ODD bins 仅控制静态 HD-map 上下文，不能直接控制 traffic density、cut-in exposure、前车行为等动态交互。
6. dynamic interaction-exposure matching 需要在后续 Stage 6C/6D 增加。
7. `odd_intersection_bin` 在当前数据集上几乎全为 `intersection_like`，仍需进一步细化。


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

## Stage 6B 结论（Map ODD 解释力）

Map-derived ODD balancing does not explain away the scene_confounding drift. Across map complexity, lane-count context, and curvature controls, scene_confounding BDD remains around 0.12 and changes by only -2.28% to +0.83%. This indicates that the constructed scene_confounding split is not primarily driven by static map ODD mismatch. Instead, it is more likely driven by dynamic interaction-exposure and behavior-proxy factors, such as lateral activity, interaction pressure, and gap size.

In contrast, pseudo_agg_vs_cons remains strongly separated after all three static ODD controls. Although ODD balancing slightly reduces BDD by about 4.4% to 8.75%, the balanced BDD remains high, around 0.155 to 0.162, with significant permutation p-values. This supports the claim that pseudo aggressive-vs-conservative drift is robust to static map ODD balancing.

## 概念澄清：Stage 6B 区分三类 shift

1. Behavior style shift
   - Example: pseudo_agg_vs_cons.
   - Remains high after static ODD balancing.

2. Static map ODD shift
   - Example controls: map complexity, lane-count context, curvature.
   - Can be controlled by map-derived ODD bins.

3. Dynamic interaction-exposure shift
   - Example: scene_confounding split.
   - May involve front pressure, gap size, lateral interaction, cut-in exposure, or other traffic-agent interaction conditions.
   - Not fully controlled by static map ODD bins.

This distinction is important. ODD balancing should not be overclaimed as removing all confounding. It controls static map context only. Dynamic interaction exposure requires additional bins or matching logic in later stages.

## Recommended Stage 6B interpretation

- Use raw BDD as the initial observed drift.
- Use static ODD-balanced BDD to test whether drift is explainable by static map context.
- If BDD remains high after static ODD balancing, the remaining drift may reflect behavior style shift or dynamic interaction-exposure shift.
- For pseudo_agg_vs_cons, the high remaining BDD supports style-shift robustness.
- For scene_confounding, the high remaining BDD suggests the split captures dynamic interaction/behavior-proxy confounding rather than static map ODD mismatch.


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
