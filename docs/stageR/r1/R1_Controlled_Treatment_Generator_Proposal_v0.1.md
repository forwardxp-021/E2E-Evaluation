# R1 controlled treatment generator proposal v0.1

## 状态和非执行声明

状态：`DRAFT_FOR_SCIENTIFIC_OWNER_REVIEW`。本文只定义 implementation route；没有创建
planner、没有运行 simulation 或 rollout，也没有读取 representation/BDD/probe outcome。
建议优先复用既有 deterministic external-planner 架构，而不是引入新的 E2E planner。

## 共同实现合同

每个 prospective roster item 在 rollout 前冻结：`scenario_token`、`log_id`、map version、
route roadblock IDs、source/target lane IDs（HLC）、initial-state fingerprint、original
background replay、`t_anchor`、`t_diverge` 和 arm parameter JSON 的 SHA。两个 arms 在
`t<t_diverge` 必须有相同 route、history、initial state、background mode 和 generator
prefix；所有 context 只从 `T_PRE_CONTEXT` 读取。

未通过 map adjacency、route、initial-state fingerprint、history completeness 或 trajectory
finite preflight 的 item，在 roster freeze 前记录为 technical eligibility exclusion。rollout
之后的 collision、off-road、incompletion、weak mechanism 或 safety failure 不是删除或
替换理由。

## R-HLC：decisive lane change 与 hesitation/retreat/recommit

### 推荐实现路线

以 `tools/stage7l_pure_lateral_execution_planner.py` 为结构起点：它已经有 frozen
source/target centerline、common canonical longitudinal progress、quintic blend、native lane
adjacency、route fingerprint 和 initial-state fingerprint 检查。R1 不复用旧 Stage7L roster，
而是在 owner 批准后为新 prospective roster 生成独立 manifest。

### Arm 定义

|项目|BASELINE|TREATMENT|
|---|---|---|
|lateral profile|`p(t)=q(u(t))` 的连续 decisive transition|`p(t)` 依次经过 departure → partial advance → hold → retreat → recommit → target settle；每段使用 quintic join 保持位置/速度/加速度连续。|
|可控 knobs|`transition_length_m`、`t_diverge`|共同 baseline knobs 加 `p_hold`、`hold_duration_s`、`retreat_delta_p`、`retreat_duration_s`、`recommit_duration_s`。|
|纵向通道|相同 canonical longitudinal progress、target speed、acceleration limit|必须与 baseline 字节级相同；不得因 treatment 改动纵向 controller。|
|预期 mechanism|小 latency、0 retreat、高 monotonic fraction|更长 commit latency、至少一个 retreat episode、较低 monotonic fraction。|

建议初始 owner-review parameter family 为 `p_hold=0.35`、`retreat_delta_p=0.15`、
`hold_duration_s=0.5`、`retreat_duration_s=0.5`、`recommit_duration_s=1.0`；这些是待审查
的 geometry/time knobs，不是冻结值。treatment 在 `t_diverge` 后才开始，且必须给
`T_PRE_CONTEXT` 保留 1.0 s 完全相同的 history。

### nuisance 与安全

- 最大设计风险：retreat/recommit 会改变 `heading_change_abs_total` 与 `path_length`。
  未来 outcome-blind trajectory-only preflight 必须检查 R0 frozen HLC F_match 全部
  caliper；不能根据 representation 结果重选参数或 scenario。
- 必须检查 map projection、lane adjacency、route progress 不越界、trajectory finite、
  dynamic consistency 和官方 vehicle/simulation safety constraints。任何以后定义的
  numeric safety limit 要在 rollout 前批准和绑定。
- failure conditions：source/target lane object 缺失、source/target 不再 native adjacent、
  route/fingerprint mismatch、common prefix mismatch、nonfinite state、trajectory超出参考
  polyline、官方 runtime failure。

## R-TSB：continuous braking 与 brake-release-second-brake

### 推荐实现路线

在同一 external-planner interface 中新增 **待实现** 的
`PiecewiseLongitudinalProfileGenerator`，复用现有 canonical progress generator 的
timestamp、initial-state、route 和 audit 结构。它是 rule-based deterministic profile，
不是 PDM parameter sweep 或新 E2E planner；lateral route centerline、background replay 和
initial state 在两 arm 固定。

### Arm 定义

|项目|BASELINE|TREATMENT|
|---|---|---|
|speed reference|单一连续 braking segment：`a_base<0`，随后 settle|`a_1<0` → `a_release>=0` → `a_2<0` → settle。|
|可控 knobs|`a_base`、`brake_duration_s`、`settle_target_speed_mps`|`a_1`、`duration_1_s`、`a_release`、`release_duration_s`、`a_2`、`duration_2_s`、共同 settle target。|
|lateral 通道|冻结 route centerline 和 lateral controller 参数|与 baseline 相同。|
|预期 mechanism|一段 braking、无 interstage release|两段 braking、正 release fraction、可定义的 second-brake peak ratio。|

参数求解必须仅依赖 trajectory physics：强制相同 initial speed、相同 horizon、近似相同
terminal speed 和 path-length integral，并在 R0 frozen TSB F_match caliper 内检查
`mean_speed`、`end_minus_start_speed`、`mean_abs_accel`、`path_length`。其中
`mean_abs_accel` 是最难保持的 descriptor，因为 release 会增加 total speed variation；
若没有 outcome-blind solution，必须报告该 generator design 不可行，而非降低 mechanism
或放宽 R0 caliper。

### nuisance 与安全

- route/tangent projection 失败、speed reference 非单调时间、speed 变负、jerk/acceleration
  state 非有限、common prefix 不同、官方 runtime failure 都是预冻结 technical failure 类。
- treatment-mechanism weak、碰撞、off-road、终止或完成失败属于 whole-roster outcome，
  不能在 representation 前删除。
- low-speed/end-stop scenario 应在 pre-treatment eligibility 中单列；不是通过事后过滤
  brake phase 来消除。

## 可复现性与运行量级

未来 manifest 必须记录 implementation code SHA、parameter JSON SHA、map version、route
fingerprint、initial-state fingerprint、background configuration SHA、t_anchor/t_diverge、
pre-context hash 和完整 roster hash。既有 rules-only 链的技术记录为约 17.23 秒/rollout；
R1 的真实运行时间在 implementation smoke 后重新记录，不能把该估计当作 performance
claim。

当前没有 generator 获批、冻结或运行。所有选择见 scientific owner decision sheet。
