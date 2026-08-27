# R1 Phase A 科学负责人批准记录 v0.1

状态：`SCIENTIFIC_OWNER_APPROVED_FOR_IMPLEMENTATION_SMOKE`。

本记录把 R1 Phase A 的 A--H 决策绑定为实现与严格隔离技术烟雾测试的输入；它不冻结整个 R1 protocol，不创建正式 R1 development roster，不授权 RBR 训练、表示评估或 R4。

## A. 通用 pre-treatment 合同

- `T_PRE_CONTEXT=[t_anchor-1.0 s,t_anchor)`，采样 `dt=0.1 s`，必须恰好有 10 个有效 history frames。
- 优先 `OFFICIAL_HISTORY_BUFFER`；仅在 runtime 无法提供完整 1.0 s official history 时，可使用 `CONDITION_IDENTICAL_1S_WARMUP`，不得缩短窗口。
- 在 `dt=0.1 s` 下，`t_diverge=t_anchor+0.1 s`。未来采样率改变时仍保持物理时间定义、重新离散化并版本化。
- paired arms 必须有同一 scenario token、map version、route fingerprint、initial-state fingerprint、original background replay、history 和 `t<t_diverge` 的 generator prefix。

## B. HLC context

批准的变量为 `map_location`、`road_class`、`log_id`、`intended_lane_change_direction`、`initial_speed_mps`、`initial_lane_offset_m`、`traffic_density`、`neighbor_availability_pattern`、`target_lane_initial_front_gap_m` 与 `target_lane_initial_rear_gap_m`。

- target front/rear 分别使用 `TARGET_FRONT_PRESENT/ABSENT` 与 `TARGET_REAR_PRESENT/ABSENT`。
- `ABSENT` 时 gap 为 `NOT_APPLICABLE_BY_FROZEN_ABSENCE_STATE`；`PRESENT` 时需至少 8/10 有效帧、稳定 track ID，并取 arc gap 中位数。
- traffic density 为每帧合法可投影 dynamic vehicle 数的 10 帧中位数。单个暂失 track 不淘汰 scenario；只有 ego/map/current required lane 无法在完整 pre-context 中定义时才不 eligible。
- 同 scenario arms 的 primary equivalence 为相同 `pre_context_raw_hash` 和相同 `canonical_context_json_hash`。

## C. HLC mechanism：OPTION_B

三项均采用 `median3 p(t)`：`commit_latency_s`、`hesitation_retreat_count`、`monotonic_transition_fraction`。

- `p_depart=0.10`，`p_commit=0.75`，commit persistence=0.5 s。
- retreat：导数 `<=-0.10 /s` 持续 0.3 s、累计回落 `>=0.08`、episode 至少 0.4 s、间隔 `<0.4 s` 合并。
- monotonic 使用 OPTION_B displacement-penalty formula。
- pair qualification：baseline retreat=0；treatment retreat>=1；treatment-baseline commit latency>=0.5 s；treatment monotonic<=baseline-0.10。三项同时成立为 `HLC_MECHANISM_PAIR_PASS`。这些阈值不是人类可感知性阈值。

## D. TSB context

批准变量为 `map_location`、`road_class`、`log_id`、`initial_speed_mps`、`initial_front_gap_m`、`initial_lead_relative_speed_mps`、`initial_thw_s`、`traffic_density`、`neighbor_availability_pattern` 与 `planned_stop_or_hazard_class`。

- 使用 `FRONT_PRESENT/ABSENT`。`ABSENT` 时 front gap、lead relative speed、THW 均是 `NOT_APPLICABLE_BY_FROZEN_ABSENCE_STATE`，不得使用数值 sentinel。
- hazard priority 为 `ROUTE_SIGNAL_RED_OR_YELLOW > STATIC_STOP_CONTROL_AHEAD > OBSERVED_SLOW_LEAD > NONE_OBSERVED`，同时保留 multi-hot audit。
- runtime 无法合法暴露 pre-treatment traffic-light/route-control API 时，不能以 scenario type 静默替代；只可在正式 roster freeze 前单独 SHA 绑定 `SCENARIO_ELIGIBILITY_ALTERNATIVE_TSB_LEAD_FOLLOWING_ONLY`。

## E. TSB mechanism：OPTION_A

三项均采用 `median3` longitudinal speed 后的 timestamp-aware finite-difference acceleration：`brake_phase_count`、`interstage_release_fraction`、`second_brake_peak_ratio`。

- brake onset `a<=-0.80 m/s²`，release `a>=-0.20 m/s²`，brake/release 最短均 0.3 s，phase merge gap `<0.3 s`，保留 `LOW_SPEED_ENDSTOP`。
- pair qualification：baseline phase=1；treatment phase=2；release fraction>=0.15；second peak ratio>=0.50。全部成立为 `TSB_MECHANISM_PAIR_PASS`；不得放宽为 `>=2`。

## F--H. generator、规模与 R-IP

- HLC 复用 Stage7L deterministic external-planner architecture；TSB 使用同接口的 `PiecewiseLongitudinalProfileGenerator`。二者仅获技术烟雾实现授权，均不是正式 generator freeze。
- 未来 planning target：HLC 48 scenarios、>=20 logs、96 endpoint rollouts；TSB 58、>=20、116。现在不是 roster freeze。
- R-IP 维持 `SECONDARY_CONDITIONAL_NOT_REQUIRED_FOR_INITIAL_ENABLEMENT`，在 D2 attribution 与 interaction anchor 独立解决前不启动。

## 不可变边界

烟雾开始后，不得因为通过率、机制门或任何结果改变上述阈值。若发现实现定义错误，只能以版本化 amendment 修正并说明原因；不得按有利结果重设 gate。
