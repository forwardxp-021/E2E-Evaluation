# R1 B2.7 Official Smoke 授权请求

## 当前状态

`READY_FOR_SCIENTIFIC_OWNER_SMOKE_AUTHORIZATION`

已完成 B2.7-R1 全局去重修正与一次性 outcome-blind roster freeze。当前仍然**没有** official smoke 授权：`OFFICIAL_SMOKE_AUTHORIZED=false`、`NEW_RUN_BUDGET=0`。本文件不是 rollout 授权。

## 冻结证据

- corrected selector SHA256：`b830476149ce284e0f36a9d9a3328dbb25d97f96eb74a3afecc063d17c85b32a`
- selector contract SHA256：`338ee58b9cfaa96b513b506bf5484119bb9f347ba5578826b45205d50f40864c`
- global dedup audit SHA256：`2c33d5ab53e2cd9eae19b57f47145d3841f53ad22b1aeff9211dc408b40c86e8`
- roster SHA256：`af672c0aa47eadebc1799dfac611016abad5b280ddd2cd56ab8ed02b605a219f`
- 48-run schedule SHA256：`d449db48aa915b5d605d51ee587aa4d6ee5fa40029eb911fbb2af5b0721fc8c5`
- execution bindings canonical SHA256：`005cc00218ed9131a13745aad60898c7a54bf04ad578e1b82ffb2d406f6fec86`

## 完成情况

- source occurrence / duplicate / global unique：`5,405,672 / 19,097 / 5,386,575`，闭合成立。
- 全局 log 数：`1,621`；其中 token-bearing candidate log 数为 `1,576`。
- duplicate identity conflict：`0`。
- leakage audit：`PASS_NO_OUTCOME_INPUTS`。
- frozen roster：HLC 12、TSB 12，合计 24 个唯一 scenario token 与 24 个唯一 log。
- schedule：48 个唯一 run ID、24 个 pair ID；第 49 个预算 claim 被拒绝。
- zero-run roster preflight：24/24 通过。

## 请求的唯一决策

是否授权冻结 roster v2.0 的一次性 48-run official compliant technical smoke？

在新的 Scientific Owner 授权前，禁止启动 `run_simulation.py`、official simulation、48-run smoke 或 RBR A/B/C。
