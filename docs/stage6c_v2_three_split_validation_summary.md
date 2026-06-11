# Stage 6C v2 three-split validation summary

## 1. Purpose

Stage 6C v2 is validated with three complementary splits:

- `negative_control_random`: a same-distribution random split used as a sanity check.
- `pseudo_agg_vs_cons`: a pseudo aggressive-vs-conservative positive-control split.
- `scene_confounding`: a scene / dynamic interaction exposure split used to diagnose confounding.

Together these splits test whether task-conditioned behavior-event BDD avoids artificial drift under random partitioning, detects known behavior-style differences, and reveals scenario-induced embedding distribution shifts.

## 2. Experimental Setup

Behavior-event inputs:

- `outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_bins_v2.csv`
- `outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv`

Embedding manifest:

- `outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2/embeddings/embedding_manifest.json`

Split directories:

- `outputs/stage6A_splits/negative_control_random/`
- `outputs/stage6A_splits/pseudo_agg_vs_cons/`
- `outputs/stage6A_splits/scene_confounding/`

Final result directories:

- `outputs/stage6C_task_bdd/negative_control_random_v2_final/`
- `outputs/stage6C_task_bdd/pseudo_agg_vs_cons_v2_final/`
- `outputs/stage6C_task_bdd/scene_confounding_v2_final/`

Cross-experiment summary:

- `outputs/stage6C_task_bdd/stage6c_v2_three_split_summary/`

Formal parameters:

- `num_bootstrap = 50`
- `num_permutation = 100`
- `max_mmd_samples = 2000`
- `min_bin_size = 100`

## 3. Main Cross-Experiment BDD Table

| task_key | BDD negative | BDD pseudo | BDD scene | pseudo_minus_negative | scene_minus_negative | detector tier |
|---|---:|---:|---:|---:|---:|---|
| `task_cutin_response` | 0.000966 | 0.210890 | 0.219525 | 0.209924 | 0.218559 | auxiliary_proxy |
| `task_following` | 0.000307 | 0.136325 | 0.165289 | 0.136019 | 0.164983 | primary |
| `task_hesitation` | 0.000462 | 0.176213 | 0.100805 | 0.175751 | 0.100342 | primary |
| `task_lane_change` | 0.000286 | 0.224173 | 0.178129 | 0.223887 | 0.177843 | primary |
| `task_lead_brake_response` | 0.000394 | 0.135319 | 0.165798 | 0.134924 | 0.165404 | auxiliary_proxy |
| `task_overtake_opportunity` | 0.002682 | 0.134527 | skipped | 0.131845 | NA | auxiliary_proxy_sample_limited |
| `task_queue_approach` | 0.000305 | 0.146251 | 0.177014 | 0.145947 | 0.176709 | auxiliary_proxy |
| `task_yield_conflict` | 0.000412 | 0.138207 | 0.145184 | 0.137795 | 0.144772 | primary |

`task_overtake_executed` is skipped due to below-`min_bin_size` sample counts and should be reported as sample-limited, not as no drift.

## 4. Negative-Control Result

The negative-control random split produces near-zero BDD values and non-significant p-values across task slices. Representative values include:

- `task_following`: BDD = 0.000307, p = 0.772277.
- `task_lane_change`: BDD = 0.000286, p = 0.871287.
- `task_yield_conflict`: BDD = 0.000412, p = 0.435644.

This validates that Stage 6C v2 does not spuriously report behavior-style drift under a random same-distribution partition.

## 5. Positive-Control Result

The pseudo aggressive-vs-conservative split yields significant task-conditioned BDD across all reported task slices. All reported rows have p approximately 0.00990099 and `observed_in_bootstrap_ci = True`.

The strongest primary drift is `task_lane_change` with BDD = 0.224173. Other primary task results also support the aggressive-style interpretation:

- `task_following`: BDD = 0.136325.
- `task_yield_conflict`: BDD = 0.138207.
- `task_hesitation`: BDD = 0.176213.

Metric-level deltas are directionally consistent with a more aggressive behavior profile in group B:

- Following: lower THW, higher deceleration, higher jerk, higher aggressiveness score.
- Lane change: higher yaw rate, lateral speed, lateral acceleration, and sharpness.
- Yield conflict: higher assertiveness and conflict acceleration.
- Cut-in response: lower TTC / THW and higher jerk / deceleration, with proxy-only caveat.
- Hesitation: significant BDD, but interpret as hesitation-like / prolonged maneuver behavior rather than psychological hesitation.

## 6. Scene-Confounding Result

The scene-confounding split also yields significant task-conditioned BDD. This confirms that scenario composition and dynamic interaction exposure can strongly affect embedding distributions.

Representative scene BDD values include:

- `task_cutin_response`: BDD = 0.219525.
- `task_lane_change`: BDD = 0.178129.
- `task_queue_approach`: BDD = 0.177014.
- `task_lead_brake_response`: BDD = 0.165798.
- `task_following`: BDD = 0.165289.
- `task_yield_conflict`: BDD = 0.145184.
- `task_hesitation`: BDD = 0.100805.

Scene style directions indicate stronger interaction pressure:

- Following: smaller front distance, lower THW, higher deceleration, and higher jerk.
- Lane change: stronger lateral motion and smaller target-lane gaps.
- Yield conflict: higher assertiveness and conflict acceleration.
- Queue / lead-brake / cut-in: consistent with higher interaction pressure, but proxy-heavy.

Therefore, overall BDD alone is insufficient. Task-conditioned diagnosis and detector reliability are necessary to separate behavior-style drift from scenario-induced confounding.

## 7. Detector Reliability and Limitations

Primary conclusions should emphasize strong-detector tasks:

- `task_following`
- `task_lane_change`
- `task_yield_conflict`
- `task_hesitation`

Auxiliary / proxy-heavy diagnostics:

- `task_cutin_response`
- `task_queue_approach`
- `task_lead_brake_response`
- `task_overtake_opportunity`
- `task_overtake_executed`

Limitations:

- Cut-in is proxy-only.
- Queue approach and lead-brake response are proxy-dominant.
- Overtake tasks are proxy and / or sample-limited.
- Skipped tasks should be reported as sample-limited, not as no drift.
- This is still public-data / pseudo-split validation, not real E2E model A/B data.

## 8. Suggested Paper Wording

These three splits jointly validate the Stage 6C v2 task-conditioned BDD framework. The negative-control split produces near-zero BDD, demonstrating that the metric does not spuriously report drift under random same-distribution partitioning. The pseudo aggressive-vs-conservative split produces significant task-conditioned BDD with metric-level shifts consistent with aggressive behavior. The scene-confounding split also produces significant BDD, confirming that scenario composition and dynamic interaction exposure can induce embedding distribution shifts. Therefore, task-conditioned diagnosis and detector reliability are necessary to distinguish behavior-style drift from scenario-induced confounding.
