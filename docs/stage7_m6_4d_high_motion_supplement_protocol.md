# Stage 7 M6.4D High-Motion Supplemental Protocol Amendment

## Status

`COMPLETED_OUTCOME_BLIND_SUPPLEMENT`（2026-08-08）

对应 GitHub Issue：[#239](https://github.com/forwardxp-021/E2E-Evaluation/issues/239)。

## 背景与范围

M6.4B 的375个 frozen primary 完成后，M6.4C 通过2个 quoted-token primary retry
和20个 frozen reserve 将有效完整 pairs 提升到305。任务计数为 following=60、
lane-change=60、stop-go=67、high-motion=55、dense/vulnerable=63。原 M6.4
high-motion primary/reserve 已耗尽，但 high-motion 距预先冻结的60对要求仍缺5对。

本 amendment 只补充该技术缺口，不改变 planner treatment、任务定义、primary
estimand、统计检验、Holm correction 或 power target。不得用 planner behavior、
embedding、BDD、effect size 或 trajectory metric 选择补充场景。

## Outcome-blind 冻结规则

补充选择工具为 `tools/stage7_m6_4d_freeze_high_motion_supplement.py`。输入为原 M6.4
eligible candidate inventory、development metadata、原375 primary/75 reserve、
M6.4B 技术状态、M6.4C 技术恢复状态和 nuPlan SQLite DB。只允许使用：

1. 原先已冻结的 high-motion scenario-type mapping；
2. token、log、DB 和 scenario metadata；
3. nuPlan 官方 scene position 技术可运行性；
4. M6.4B/M6.4C 的成功/失败技术状态，用于确认缺口为5，不读取 planner outcome。

排除全部 development token/log 和原 M6.4 primary/reserve token/log。新集合内部每个
log 最多1个场景。固定 salt 为
`stage7-m6.4d-high-motion-supplement-v1`，候选 probe limit 为2048；按
`SHA256(salt:task:log_name:scenario_token)` 升序检查。只有满足 nuPlan 官方 scene
position 条件的候选可以进入集合。

冻结5个 supplemental primary 和5个 supplemental technical reserve。5个 primary
必须全部按冻结顺序尝试；reserve 只允许替换 documented technical failure，且
primary 无技术失败时 runner 必须拒绝 reserve。

## 冻结结果

输出目录：`outputs/stage7_m6_4d_high_motion_supplement_freeze_v1/`。

前16个固定顺序候选中，10个通过并被选中，4个因 official scene position 无效排除，
2个因补充集合内 log 重复排除。最终10个 token/log 与 development 和原450条集合的
overlap 均为0。

```text
supplement manifest file SHA-256:
3dc11ab70c71479191bb4c789782e5ebe78dd7e43efdaec55651451b99041c2f

primary canonical SHA-256:
e63634711345e590de8db038c44a0fbe890700cd197e4de01156f338481113bb
```

Stage7C 和 M6.4B runner 继续保持冻结 SHA-256：

```text
076b35d2112e126008eec5c96bf3e7b159ded75a40be7999212956423cb3e530
ef0026b3cc20942846035ac23d0d16d616a3d7dd6675e9a0f9c2612871d7fb06
```

## 执行安全门

`tools/stage7_m6_4d_run_locked_supplement.py` 默认 dry-run。真实执行需要同时提供
`--execute`、supplement manifest 文件 SHA-256 和 source canonical SHA-256。
Runner 复核 selection/Stage7C/batch tool hashes、planner fingerprints、nuPlan 与
tuPlan commits、runtime paths、timeout、CSV/canonical hashes 和每个 token 的 SQLite
技术可运行性。每个结果必须通过2/2 official success、trajectory pair completeness、
same-log 与 strict-token alignment。

Hydra numeric-like token 会在执行前主动使用可穿过 `shlex.split` 的转义引号，但原始
`scenario_token` 保持不变，用于严格身份校验。

## 实际结果

5个 supplemental primary 全部成功，0失败；端到端耗时范围29.42–34.93秒，均值
31.14秒，中位数30.54秒。每个场景均为2/2 official success、完整 trajectory pair，
same-log 和 strict-token alignment 通过。5个 frozen reserve 未执行。

最终完整 pairs：

```text
following_interaction:             60 / 60
lane_change:                       60 / 60
stop_go_control:                   67 / 60
high_motion_dynamics:              60 / 60
dense_or_vulnerable_interaction:   63 / 60
overall:                          310
```

M6.4D 只完成样本与技术完整性要求。后续统计确认必须使用预冻结的 M6.2/M6.3 方法，
不能因为补充结果而修改 embedding、BDD、effect-size estimator 或显著性规则。
