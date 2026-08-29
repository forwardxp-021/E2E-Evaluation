# R1 官方合规技术 Smoke 范围修订 v1.0

状态：`FROZEN_PROSPECTIVE_BEFORE_FRESH_CANDIDATE_ENUMERATION`。

本修订在 fresh candidate enumeration 与第一条 official closed-loop run 之前完成。它只修订 technical-smoke 的 candidate enumeration：HLC 与 TSB 的 generator 均已冻结为单一 treatment，因此不再存在三候选 treatment 的枚举需要。

## 唯一合法范围

| family | fresh scenarios | 每场景唯一 arms | official closed-loop runs |
|---|---:|---|---:|
| R-HLC | 12 | `HLC_BASELINE_DECISIVE_MONOTONIC_LANE_CHANGE` + `HLC_TREATMENT_HLC_GEN_V2_OPTION_B` | 24 |
| R-TSB | 12 | `TSB_BASELINE_SINGLE_CONTINUOUS_BRAKING` + `TSB_TREATMENT_TSB_GEN_V2_OPTION_A` | 24 |
| 合计 | 24 | 每场景 2 arms | 48 |

旧的 `HLC_MILD/NOMINAL/STRONG` 与 `TSB_MILD/NOMINAL/STRONG` 均不得进入本轮执行日程。

## 前瞻性与边界

修订理由仅为 `HLC_GEN_V2_OPTION_B` 和 `TSB_GEN_V2_OPTION_A` 已在本轮 fresh enumeration 前科学冻结；没有引用任何 fresh smoke outcome。本修订不改变 R0 protocol、R1 context/mechanism gate、F_match caliper、HLC endpoint、TSB mechanism 或训练授权。

任何 scope、arm、run cap 或 frozen generator SHA 不匹配，必须在 claim 前停止，且保持 `0/48`。
