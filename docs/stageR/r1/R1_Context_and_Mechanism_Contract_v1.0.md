# R1 Context 与 Mechanism Measurement Contract v1.0

状态：`R1_CONTEXT_MECHANISM_CONTRACT_V1_FROZEN`。

此冻结仅覆盖 R1 context 与 mechanism measurement。它不冻结完整 R1 protocol、正式 generator 或 roster，也不授权 representation evaluation、RBR training 或 R4。

## 绑定

- scientific-owner approval：`r1_phaseA_scientific_owner_approval_v0.1.json`，SHA-256 `27aac073d2323aadd8d1a89b96d959fcdcb41e7b913d53e5d8acbc59b6dbc12c`。
- source context proposal：`r1_context_anchor_definition_proposal_v0.1.csv`，SHA-256 `83b02a71e49d6e96ec87970e8bd301f909b4b04cb23309bce29b98ffb5dccaf1`。
- machine contracts：`r1_context_contract_v1.0.json`、`r1_hlc_mechanism_contract_v1.0.json`、`r1_tsb_mechanism_contract_v1.0.json`。

## 时间、arm 与 context

`T_PRE_CONTEXT=[t_anchor-1.0s,t_anchor)`、`dt=0.1s`、恰好 10 个有效帧；优先 official history，runtime 无法提供完整窗口时才允许 condition-identical 1s warm-up。`t_diverge=t_anchor+0.1s`。同 scenario arm 在 rollout 前须证明 raw history hash 与 canonical context JSON hash 均相同。

canonical record 至少保留 context variables、missingness state、frame-valid coverage、map/source IDs、slot track IDs 与 query version。HLC front/rear 和 TSB front 的 `ABSENT` 是合法 canonical state，所有相应数值均为 `NOT_APPLICABLE_BY_FROZEN_ABSENCE_STATE`，不得用数值 sentinel。

## HLC 与 TSB mechanisms

HLC 固定为 Option B：median3 `p(t)`，`p_depart=.10`、`p_commit=.75`、0.5s persistence；retreat 与 Option-B displacement penalty 的阈值见 machine contract。其 pair pass 是 baseline retreat=0、treatment retreat>=1、commit latency delta>=.5s 和 treatment monotonic 至少低 .10。

TSB 固定为 Option A：median3 speed、timestamp-aware finite difference；brake/release 为 -.80/-.20 m/s²，brake/release duration=.3s，merge gap<.3s，并保留 low-speed endstop。其 pair pass 是 1 phase vs 2 phases、release fraction>=.15、second peak ratio>=.50。

技术烟雾开始后不得因通过率、机制或任何 outcome 调整这些定义或阈值。实现定义 bug 只能以独立版本化 amendment 处理。
