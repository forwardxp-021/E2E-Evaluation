# Stage 6C v2 pseudo positive-control summary

## 1. Purpose

`pseudo_agg_vs_cons` is a Stage 6C v2 positive-control experiment. It uses a pseudo aggressive-vs-conservative split to validate whether task-conditioned behavior-event BDD can detect known behavior-style differences within the same driving task / behavior-event slices.

This is not a real model A/B result. Its role is to confirm that the current Stage 6C v2 protocol is sensitive to behavior-style drift before comparing negative-control and scene-confounding final experiments.

## 2. Inputs

- Behavior-event inputs:
  - `outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_bins_v2.csv`
  - `outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv`
- Pseudo split inputs:
  - `outputs/stage6A_splits/pseudo_agg_vs_cons/a_indices.npy`
  - `outputs/stage6A_splits/pseudo_agg_vs_cons/b_indices.npy`
- Embedding manifest:
  - `outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2/embeddings/embedding_manifest.json`
- Result directory:
  - `outputs/stage6C_task_bdd/pseudo_agg_vs_cons_v2_final/`

Formal parameters:

- `num_bootstrap = 50`
- `num_permutation = 100`
- `max_mmd_samples = 2000`
- `min_bin_size = 100`
- `top_k = 20`
- `seed = 42`

All reported task-conditioned BDD rows have `p_value = 0.00990099` and `observed_in_bootstrap_ci = True`, so the positive-control signal is stable under the current formal settings.

## 3. Main BDD Results

| task_key | BDD | p_value | detector_strength | n_A | n_B | interpretation_tier |
|---|---:|---:|---|---:|---:|---|
| `task_lane_change` | 0.224173 | 0.00990099 | strong | 719 | 2847 | primary |
| `task_cutin_response` | 0.210890 | 0.00990099 | proxy | 824 | 114 | auxiliary_proxy |
| `task_hesitation` | 0.176213 | 0.00990099 | strong | 1112 | 1966 | primary |
| `task_queue_approach` | 0.146251 | 0.00990099 | proxy-dominant | 2863 | 431 | auxiliary_proxy |
| `task_yield_conflict` | 0.138207 | 0.00990099 | strong | 2365 | 1084 | primary |
| `task_following` | 0.136325 | 0.00990099 | strong | 3184 | 472 | primary |
| `task_lead_brake_response` | 0.135319 | 0.00990099 | proxy-dominant | 2439 | 432 | auxiliary_proxy |
| `task_overtake_opportunity` | 0.134527 | 0.00990099 | proxy | 547 | 107 | auxiliary_proxy |
| `task_overtake_executed` | skipped | NA | sample-limited | 80 | 47 | skipped_sample_limited |

Primary result candidates:

- `task_lane_change`
- `task_following`
- `task_yield_conflict`
- `task_hesitation`

Auxiliary / proxy diagnostics:

- `task_cutin_response`
- `task_queue_approach`
- `task_lead_brake_response`
- `task_overtake_opportunity`
- `task_overtake_executed`

`task_overtake_executed` was skipped because it was below `min_bin_size`; this should not be interpreted as evidence of no drift.

## 4. Style Explanation

### Following

Group B has lower THW and higher jerk / deceleration:

- `following_mean_thw`: B lower than A.
- `following_min_thw`: B lower than A.
- `following_peak_decel`: B higher than A.
- `following_rms_jerk` and `following_max_abs_jerk`: B higher than A.

Interpretation: group B is closer-following, harsher-braking, and less comfortable / more aggressive.

### Lane Change

Group B has sharper and more assertive lateral maneuver behavior:

- `lc_sharpness_score`: B higher than A.
- `lc_rms_lateral_accel`: B higher than A.
- `lc_max_lateral_speed`: B higher than A.
- `lc_rms_yaw_rate`: B higher than A.

Interpretation: group B shows sharper lateral maneuvers and stronger lane-change dynamics.

### Yield Conflict

Group B is more assertive under interaction pressure:

- `assertiveness_score`: B higher than A.
- `conflict_accel_score`: B higher than A.
- `yield_conflict_score`: B higher than A.

Interpretation: group B behaves more competitively / assertively in yield-conflict slices.

### Hesitation

Group B has longer lane-change / maneuver duration:

- `hesitation_lc_duration`: B higher than A.

Interpretation should be cautious. This supports a longer maneuver-execution or hesitation-like behavior explanation, but should not be overclaimed as psychological hesitation.

### Cut-In Response

Group B has lower TTC / THW and higher jerk / deceleration after the cut-in proxy:

- `cutin_min_ttc`: B lower than A.
- `cutin_min_thw`: B lower than A.
- `cutin_jerk_after_cutin`: B higher than A.
- `cutin_peak_decel_after_cutin`: B higher than A.

Interpretation: this supports auxiliary diagnosis, but cut-in remains proxy-only because it is still based on gap-drop / front-appearance logic rather than a true side-to-front slot-ID transition detector.

## 5. Conclusion

The pseudo aggressive-vs-conservative positive control validates that Stage 6C v2 task-conditioned BDD is sensitive to behavior-style drift. The strongest primary drift appears in lane-change, following, yield-conflict, and hesitation-like maneuver slices. Metric-level deltas are directionally consistent with a more aggressive behavior profile in group B. Proxy-heavy tasks such as cut-in, queue approach, lead-brake response, and overtake should be treated as auxiliary diagnostics.

## 6. Limitations

- This is a pseudo split, not a real model A/B comparison.
- Cut-in and overtake are proxy detectors.
- Queue approach and lead-brake response are partly proxy-dominant.
- `task_overtake_executed` is sample-limited and was skipped.
- Full validation still requires final comparison against `negative_control_random` and `scene_confounding` results.
