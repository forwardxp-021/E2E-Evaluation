# R0 Future R4 Reserved Pool Proposal v0.1

## 1. 当前决策

```text
FUTURE_R4_RESERVED_POOL = NOT_AVAILABLE
RULESET_DRAFT_READY_FOR_REVIEW
RBR_TRAINING_NOT_AUTHORIZED
```

机器候选：`docs/stageR/r0/manifests/r0_future_r4_reserved_pool_candidate_v0.1.csv`。

本文件冻结的是候选规则草案，不声称当前已有可用 reserved pool。既有 source 的“未使用 remainder”尚无 authoritative identity ledger；prospective controlled planner 路线尚未绑定具体 source、token roster 与 SHA。

## 2. Route A — 未使用 existing source/token pool

只有同时满足以下条件才可升级为 READY：

1. 绑定 dataset release、map set、source/log/scenario/token identity 与 SHA；
2. 对 Waymo train/val/historical-test、Stage6/Stage7/Stage7L、R0 development/audit ledger 做完整 overlap 检查；
3. prior historical use、model selection、representation evaluation 均为 false；
4. 使用 hash-sorted fixed seed `2026082601` 选择；
5. exclusions 仅含 identity overlap、缺失 source/log metadata、预先已知技术 unrunnability 与预注册规则；
6. scenario allocation log-disjoint；若存在 driver identity，进一步 driver-disjoint；
7. 不读取 old64/A/B/C/ego13/RBR embedding、probe、BDD 或 detection outcome。

在 unused identity ledger 完成前：

```text
R4A_EXISTING_UNUSED_POOL = NOT_AVAILABLE
```

## 3. Route B — Prospective controlled planner

在运行任何 treatment rollout 以前必须锁定：

### Scenario source

- dataset release、map set、log roster、scenario token roster；
- source/version/config SHA；
- 与全部历史和 R0 角色的 identity overlap=0；
- scenario family 与 pre-treatment context coverage。

### Token selection

- 按 pre-treatment eligibility 过滤；
- hash-sort eligible tokens；
- seed `2026082601`；
- 一次性形成 whole roster；
- 不按 representation outcome、预计模型难度或 realized mechanism 排序。

### Treatment family

- `R-HLC`：hesitation/commitment planner rule；
- `R-TSB`：continuous vs two-stage braking；
- `R-IP`：在相似 context 下的 wait-for-gap vs early-gap-acceptance probe；
- 每个 family 的 software/config SHA 在 rollout 前绑定。

### Dose/parameter family

- 每个 treatment 使用 bounded parameter grid；
- grid、dose、方向和 fallback 在 rollout 前冻结；
- 不因 realized effect 弱而追加/删除 dose；
- 不根据 future representation outcome 选择参数。

### Exclusions

- pre-treatment identity overlap；
- 缺失必要 context/map/config；
- 在 frozen preflight 中已知且可复现的技术不可运行；
- 预注册的 safety/config incompatibility；
- realized mechanism 弱、方向不理想或 representation 不显著不能作为 exclusion。

### Runnability 与 independence

- runnability 只由 import/config/map availability 和 treatment 前 smoke/preflight 决定；
- independent unit 为 scenario；
- 多 scenario 同 log 时以 log cluster；
- allocation 必须 log-disjoint；
- Primary estimand 是 whole-frozen-roster / intention-to-evaluate。

## 4. Rollout 后顺序

```text
source/token/roster freeze
→ treatment rollout
→ whole-roster mechanism gate
→ whole-roster Primary embedding confirmation
→ pre-specified mechanism-success sensitivity（可选、Secondary）
```

不得删除 mechanism 弱的 scenario 后，在 survivor subset 上重新执行 Primary embedding confirmation。技术失败按 rollout 前 frozen failure/missingness policy 处理；弱 mechanism 不是技术失败。

## 5. 升级 READY 的硬条件

必须同时存在：

- exact source/token roster manifest；
- source/config/code SHA；
- historical overlap=0 证明；
- treatment/dose/exclusion/runnability/independence rule；
- whole-roster mechanism/Primary order；
- 未读取 future outcome 的 command ledger；
- independent-unit 与 power feasibility；
- scientific owner/governance approval。

当前只满足规则草案，未满足资产绑定，因此 final status 保持 `NOT_AVAILABLE`。
