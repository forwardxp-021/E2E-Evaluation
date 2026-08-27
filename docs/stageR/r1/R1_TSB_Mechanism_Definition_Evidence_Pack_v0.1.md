# R1 TSB mechanism definition evidence pack v0.1

## 状态与用途

状态为 `DRAFT_FOR_SCIENTIFIC_OWNER_REVIEW`。本 pack 为 R-TSB future controlled benchmark
提出 brake-release-brake 的分段规则；不冻结阈值、不运行 rollout、不读取任何
representation、BDD、probe、detection 或 RBR outcome。

## 共用 longitudinal signal 与 phase segmentation 提案

从 official rollout ego rear-axle state 的 source-lane/route tangent 定义
`v_s(t)=dot(v(t); tangent_route(t))`。处理顺序固定为：finite/time monotonic validation →
filter OPTION → `a_s(t)=dv_s/dt` → hysteretic phase labelling → phase merge。不得把
whole-window raw33 acceleration、jerk 或 future response-derived THW 当作 mechanism
变量的替代。

每个 OPTION 的 filter、onset/release threshold、duration 和 merge gap 都在 CSV 中明确。
共同规则如下：

- 一个 brake phase 是达到 onset threshold 并满足 minimum duration 的 maximal interval；
  相邻 phase 若间隔小于 proposal merge gap，合并为一个 phase。
- interstage release 只在至少两个有效 brake phase 存在时计算；否则对应值为
  `NOT_EVALUABLE_MECHANISM_VARIABLE`，不把零当作 release。
- `v_s < 1.0 m/s` 连续达到 0.5 s 时标记 `LOW_SPEED_ENDSTOP`；在该 stratum 内，末端
  零速停靠不会被解释为第二 brake phase 或 release。
- edge filter 无法形成完整窗口、速度/route tangent 非有限、或 map route projection
  ambiguity 都必须显式报告，不能以较短数组 silently pad。

## 数值依据与敏感性边界

raw-only audit 在 `dt=0.1 s` 下得到 Stage6J longitudinal acceleration q05/q01 为
-0.923/-2.215 m/s²，Stage6K dose25 为 -0.806/-1.944 m/s²；对应 jerk q95 为
0.759/0.588 m/s³。下表的 onset -0.6 至 -1.0 m/s²、release -0.2 至 0.0 m/s²和
0.3 至 0.5 s duration 是 physical braking semantics、0.1 s resolution 与这些
未筛选 raw distributions 的提案。它们不是 embedding/BDD detection 结果，且均待审批。

任一 future runtime 若不是 0.1 s 采样，filter span、duration、merge gap 与 derivative
必须按 physical seconds 重建并重新送审；不可照搬 frame count。

## 变量定义候选

详表见 `r1_tsb_mechanism_definition_options_v0.1.csv`：

- `brake_phase_count`：完成 hysteretic segmentation 后的分离 brake phase 数。
- `interstage_release_fraction`：第一与第二 brake phase 间的 speed release 相对于第一
  phase speed loss 的比例。
- `second_brake_peak_ratio`：第二 phase 的 peak braking magnitude 相对于第一 phase。

所有变量共同报告 filter choice、low-speed/end-stop flags、phase validity 和 merged interval
数量。没有合法的 phase 分段时，不得推断为 mechanism absence。
