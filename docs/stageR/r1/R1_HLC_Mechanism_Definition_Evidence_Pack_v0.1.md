# R1 HLC mechanism definition evidence pack v0.1

## 状态与用途

状态为 `DRAFT_FOR_SCIENTIFIC_OWNER_REVIEW`。本 pack 为未来 R-HLC controlled benchmark
提出可审查算法；不冻结阈值、不重算 R0 D4，也不读取 embedding、BDD、probe、detection
或 RBR outcome。全部选项必须在 prospective roster/rollout 前由 scientific owner 选择并
单独 SHA 绑定。

## sign-corrected lane-transition progress 提案

对每个 future roster item，先冻结 source lane 与 native adjacent target lane 的中心线
`c_S(s)`、`c_T(s)`、source/target arc 起点、map version、route 和 initial-state
fingerprint。对每个 valid ego rear-axle point `x(t)`：

1. 在 source reference 上求唯一的共同 route progress `s(t)`；若 projection residual
   超过 1.0 m、progress 不唯一、source/target lane 不再邻接或 lane separation 非正，则该
   frame 无效，不能以 nearest arbitrary lane 替代。
2. 令 `d(t)=c_T(s(t))-c_S(s(t))`，并定义
   `p_raw(t)=dot(x(t)-c_S(s(t)); d(t))/||d(t)||^2`。这使 source center 为 0、target
   center 为 1；LEFT/RIGHT 的符号由 target lane 绑定而非观察到的 response 决定。
3. `p(t)=clip(p_raw(t);0;1)`。OPTION_A 直接使用它；OPTION_B 使用三帧 centered
   median；OPTION_C 使用五帧 centered median。边界帧使用可用的对称窗口；不足三帧时
   标记 `EDGE_WINDOW_INSUFFICIENT`。

lane width/geometry 不是常数假定：每帧使用 map 中的 source-target separation。该规则能
处理曲线 lane、不同 lane width 和 LEFT/RIGHT；却不接受 lane assignment ambiguity。
低速时仍可计算位置 progress，但 `speed < 1.0 m/s` 持续超过 0.5 s 的 transition 要标记
`LOW_SPEED_TRANSITION`，作为 mechanism reporting stratum，不能静默删除。

## 数值依据与敏感性边界

`r1_phasea_raw_trajectory_evidence_v0.1.json` 仅从预先列明的完整 raw arrays 汇总，
`dt=0.1 s`。R-HLC reference 的 lateral speed q95/q99 为 0.496/0.879 m/s；因此下表的
0.10--0.20 progress/s 阈值在 3--4 m lane separation 下相当于 0.30--0.80 m/s。它们是
车辆/lane geometry、0.1 s resolution 与 raw measurement-scale 的提案，不是 outcome 调参。

所有 minimum duration 都是 `dt` 的整数倍；若 future runtime 的 sampling interval 不是
0.1 s，必须按物理秒数重新离散化、重新走 owner approval，不能仅复制 frame count。

## 变量定义候选

详表见 `r1_hlc_mechanism_definition_options_v0.1.csv`。三个变量均给出 A/B/C：

- `commit_latency_s`：从已冻结 `t_anchor` 到首次 stable target commitment 的时间。
- `hesitation_retreat_count`：从 departure 到首次 commitment 前、达到最小向后 progress
  深度的分离 retreat episode 数。
- `monotonic_transition_fraction`：departure 到 commitment 期间正向 progress 的比例；
  它是连续描述量，不以单一 episode 作替代。

所有选项均需报告 frame-validity、projection ambiguity、low-speed 和 unfinished transition
数量。没有 valid `p(t)` 序列时，正式值为 `NOT_EVALUABLE_MECHANISM_VARIABLE`，而不是
用 raw33 proxy、whole-window lateral speed 或人工目测替代。
