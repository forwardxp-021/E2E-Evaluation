# R1 Phase A 科学负责人决策单 v0.1

状态：`DRAFT_FOR_SCIENTIFIC_OWNER_REVIEW`。以下每项均为
`REQUIRES_SCIENTIFIC_OWNER_APPROVAL`；本文件不表示任何批准、冻结、rollout 或训练授权。

|编号|待审批项目|提案|需要明确选择/签署的内容|当前状态|
|---|---|---|---|---|
|A|common pre-treatment anchor/window|`t_anchor` 在首次 control divergence 前；`T_PRE_CONTEXT=[t_anchor-1.0s;t_anchor)`；10 个 0.1s 有效 history frames|采用 official history 还是 identical warm-up option；anchor manifest 字段和完整性门|REQUIRES_SCIENTIFIC_OWNER_APPROVAL|
|B|HLC context definitions|采用 context CSV 中 map/lane/slot/target-gap 的 exact query；无完整 target gap 则 pre-rollout exclusion|road_class enum；target-gap missingness；禁止 geometric fallback 的 exact-parity 规则|REQUIRES_SCIENTIFIC_OWNER_APPROVAL|
|C|HLC mechanism option|为三项 HLC mechanism 选择同一版本族 OPTION_A/B/C 或退回重新设计|p smoothing；commit threshold/dwell；retreat rate/depth；monotonic formula|REQUIRES_SCIENTIFIC_OWNER_APPROVAL|
|D|TSB context definitions|采用 context CSV 中 current-lane front/THW/signal-route hazard 的 exact query|planned_stop_or_hazard_class enum；无 signal API 时的 scenario-eligibility alternative|REQUIRES_SCIENTIFIC_OWNER_APPROVAL|
|E|TSB mechanism option|为三项 TSB mechanism 选择同一版本族 OPTION_A/B/C 或退回重新设计|filter；brake/release threshold；duration；merge gap；end-stop处理|REQUIRES_SCIENTIFIC_OWNER_APPROVAL|
|F|controlled generator choice|HLC 采用 Stage7L external deterministic architecture；TSB 采用同接口的 piecewise longitudinal extension|parameter family；preflight safety bounds；implementation owner；smoke authorization另行提出|REQUIRES_SCIENTIFIC_OWNER_APPROVAL|
|G|initial development sample scale|HLC 与 TSB 初始建议均为 RECOMMENDED|是否采用 48/58 total roster scenario；unique log minimum；pre-freeze reserve policy|REQUIRES_SCIENTIFIC_OWNER_APPROVAL|
|H|R-IP defer/activate|初始 defer；保留为 secondary conditional family|继续 defer 或只在 D2 attribution 与 interaction anchor 独立解决后启动 Phase A|REQUIRES_SCIENTIFIC_OWNER_APPROVAL|

## 许可边界

即使上述条目获批，下一步也只能按独立授权执行 implementation/smoke 或 roster preparation；
不得自动启动 planner rollout、benchmark rollout、representation evaluation 或 RBR training。
`r0_training_authorization_manifest_v1.0.json` 保持不变：RBR-A、RBR-B、RBR-C 均为
`NOT_AUTHORIZED`。
