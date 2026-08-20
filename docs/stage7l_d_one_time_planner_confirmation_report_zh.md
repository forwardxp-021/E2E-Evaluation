# Stage7L-D 一次性 Planner-Level Confirmation 报告

## 最终状态

```text
STAGE7L_D_PLANNER_LEVEL_CONFIRMATION_PASSED
STAGE7L_E_REPRESENTATION_EVALUATION_UNLOCKED
```

前瞻冻结的纯横向执行 Treatment 产生了预注册方向的横向机制变化，同时满足预冻结的纵向 nuisance 与安全/有效性门禁。本阶段只完成 planner-level confirmation；没有读取 checkpoint/embedding，没有计算 BDD/MMD，也没有执行 Stage7L-E。

## 1. 冻结 provenance

- 分支：`20260611_stage7_conclusion`
- C2远端冻结基线：`c2d6ff0225967244422e749c677de548c3b8c1cf`
- execution start commit：`63080ed8547966c572fe55e7819f93e7b99c44ea`
- 最终运行实现代次：`runtime_manifest_adapter_v3_hydra_searchpath`
- protocol SHA256：`f5a8b2df5ed60c0384e8181feceab33f3c6f048780e95aab851184e49247490a`
- confirmation roster SHA256：`90ec9b427636cefc59e6d7ace2507ac8364747e2a38964124be08fdc2a10acf9`
- blind authorization SHA256：`f2d46f586a3131144aa2d20b6d86a1cfdd2bfe77dccf2b224a99e8de66acfb5e`
- planner code SHA256：`284d60263621a99b3e57f63f3092797e44a9c393ec6975f419ed02ecb64885d0`
- 冻结source maneuver manifest SHA256：`ad0abdb3ebfaf8f06c92987538254a79455ad3d7b03a42e609462b49596de725`
- runtime adapter manifest SHA256：`9bf1777f94f70239b69a5af1cb5106d923558ceda5d26fad69861ddea5a4dce1`
- roster：80 scenarios，15 left，65 right，79 unique logs；development scenario/log overlap均为0。
- treatment：transition length `60/58.5/57/55.5/54 m`，trigger 12 m，15 s horizon，0.1 s sampling，target speed 5 m/s，accel limit 1 m/s²。
- background：`closed_loop_nonreactive_agents`。
- C2 task masks只作provenance：`LAT.LANE_CHANGE=80`、`LAT.DYNAMICS=38`，本阶段未用其计算BDD。

冻结source manifest在首条有效rollout前被证明缺少planner dataclass所需的4个接口字段。按C2允许的`demonstrated code non-executability`规则，运行时adapter仅补入协议已冻结的15 s horizon、non-reactive background agent/config及五档profile IDs；未改roster、几何、dose、planner或gate。第二代adapter随后在0条有效rollout时暴露Hydra searchpath不一致，最终实现代次只恢复历史已验证searchpath。所有失败attempt均永久保留。

## 2. Official rollout inventory与missingness

| 项目 | 结果 |
|---|---:|
| N_design | 80 |
| planned cells | 400 |
| successful official rollout cells | 400 |
| final failed cells | 0 |
| N_complete_all_five_doses | 80 |
| dose0 / dose25 / dose50 / dose75 / dose100 success | 80 / 80 / 80 / 80 / 80 |
| total attempts | 412 |
| replacement | 0 |

12条结果前基础设施attempt未生成有效official rollout：第一实现代次5条`exit=2`加1条中断，根因是冻结manifest接口字段缺失；第二实现代次5条`exit=2`加1条中断，根因是Hydra searchpath缺少`ego_controller`。最终400个固定cell全部由同一scenario、同一dose、参数不变的有效attempt完成，因此最终infrastructure failed cell为0。

Treatment/trajectory outcome共8个rollout cell：同一场景五档均出现responsible collision（5格），另两个场景共3格出现off-road；lane-change incomplete、route failure和invalid trajectory均为0。这8格是保留的正式结果，不是基础设施失败，也没有触发重跑或删除。

Execution gate：`80 >= 76`，通过。

## 3. Primary mechanism gate（dose100 − dose0）

所有delta统一为Target（Sharp dose100）减Behavior Reference（Gentle dose0）。CI为按`log_name`聚类的10,000次percentile bootstrap（seed 620272），只用于展示。

| 指标 | paired median Δ | 方向一致率 | 95% CI | 冻结门槛 | 结果 |
|---|---:|---:|---:|---:|---|
| lane-change duration | −0.200160 s | 88.75% | [−0.200415, −0.199672] | Δ<0且≥70% | PASS |
| RMS lateral accel | +0.055832 m/s² | 100.00% | [0.044971, 0.062866] | Δ>0且≥80% | PASS |
| peak yaw rate | +0.014404 rad/s | 96.25% | [0.013369, 0.014992] | Δ>0且≥80% | PASS |

三项primary mechanism全部通过：`STAGE7L_D_MECHANISM_GATE_PASSED`。

## 4. Secondary dose-response（planner-level描述）

| Target − dose0 | Δ duration (s) | Δ RMS lateral accel (m/s²) | Δ peak yaw rate (rad/s) |
|---|---:|---:|---:|
| dose25 | −0.000083 | +0.012297 | +0.003151 |
| dose50 | −0.099988 | +0.025780 | +0.006682 |
| dose75 | −0.100261 | +0.040193 | +0.010392 |
| dose100 | −0.200160 | +0.055832 | +0.014404 |

RMS横向加速度与峰值横摆角速度随dose增强；换道时长总体缩短，但冻结协议不要求中间dose严格单调。

## 5. Longitudinal nuisance gate（dose100 − dose0）

| 指标 | median absolute Δ | p90 absolute Δ | 冻结阈值（median/p90） | 结果 |
|---|---:|---:|---:|---|
| mean speed | 0.000951 m/s | 0.000987 m/s | 0.02 / 0.02 | PASS |
| RMS longitudinal accel | 0.000202 m/s² | 0.000809 m/s² | 0.05 / 0.05 | PASS |
| RMS longitudinal jerk | 0.000230 m/s³ | 0.000710 m/s³ | 0.10 / 0.10 | PASS |
| route progress | 0.013880 m | 0.020229 m | 0.25 / 0.25 | PASS |

四项median与p90均在预冻结阈值内：`STAGE7L_D_LONGITUDINAL_NUISANCE_GATE_PASSED`。

## 6. Safety / validity

冻结口径是全部80个场景的scenario-level conservative aggregation：official success和completion要求五档全部成功；任一dose发生off-road或responsible collision即把该场景记为发生。未做post-treatment deletion。

| 指标 | 场景数 / 80 | 比率 | 冻结门槛 | 结果 |
|---|---:|---:|---:|---|
| official success | 80 | 100.00% | ≥95% | PASS |
| lane-change completion | 80 | 100.00% | ≥95% | PASS |
| off-road | 2 | 2.50% | ≤5% | PASS |
| responsible collision | 1 | 1.25% | ≤5% | PASS |
| invalid / incomplete / route failure | 0 / 0 / 0 | 0% | 描述 | — |

`any collision`为`N/A`：official结果包只暴露at-fault/responsible collision，未提供可独立审计的all-collision字段。Safety gate通过：`STAGE7L_D_SAFETY_VALIDITY_GATE_PASSED`。

## 7. Canonical treatment purity与geometry validity

- canonical identity pass：80/80；mismatch：0。
- 五档的canonical `s_route`、source/target、trigger、longitudinal config及generator SHA保持一致。
- lane-change completion：400/400 rollout。
- final target center offset与settling time作为secondary geometry描述保留，不构成新增gate。
- off-road、责任碰撞等不良geometry/safety outcome均保留在全80 design population中。

## 8. 最终机器门禁

| Gate | 状态 |
|---|---|
| execution | `STAGE7L_D_CONFIRMATION_EXECUTION_SUFFICIENT` |
| canonical identity | PASS |
| mechanism | `STAGE7L_D_MECHANISM_GATE_PASSED` |
| longitudinal nuisance | `STAGE7L_D_LONGITUDINAL_NUISANCE_GATE_PASSED` |
| safety / validity | `STAGE7L_D_SAFETY_VALIDITY_GATE_PASSED` |
| Stage7L-D total | `STAGE7L_D_PLANNER_LEVEL_CONFIRMATION_PASSED` |
| representation authorization | `STAGE7L_E_REPRESENTATION_EVALUATION_UNLOCKED` |

## 9. Blind boundary与claim boundary

- embedding read：No
- checkpoint inference：No
- BDD/MMD computed：No
- Stage7L-E executed：No
- model training/retraining：No

本阶段只支持以下表述：

> 前瞻冻结的纯横向执行Treatment产生了预注册方向的横向机制变化，同时满足预冻结的纵向nuisance与安全/有效性门禁。

它尚不支持“BDD已检出横向风格漂移”“Candidate B validated”或任何representation排名。Stage6V的`NO_ABC_CANDIDATE_QUALIFIES_UNDER_PRE_FROZEN_RULE`保持不变。Stage7L-E虽已解锁，但必须等待下一次独立授权。

## 10. 用户要求的28项核对

1. 冻结80场景：是。
2. 方向分布：15 left / 65 right。
3. replacement：0。
4. planned rollouts：400。
5. successful rollouts：400。
6. N_complete_all_five_doses：80。
7. 各dose success：80 / 80 / 80 / 80 / 80。
8. infrastructure failures：12条结果前失败attempt（两代各5条exit=2+1条中断）；分别由manifest接口缺失和Hydra searchpath缺失`ego_controller`导致；最终failed cell=0。
9. treatment outcome failures：8个rollout cell（5格responsible collision、3格off-road），对应3个场景；均作为结果保留。
10. duration：−0.200160 s；88.75%。
11. RMS lateral accel：+0.055832 m/s²；100%。
12. peak yaw：+0.014404 rad/s；96.25%。
13. 三项mechanism：全部PASS。
14. 四项nuisance median/p90：speed 0.000951/0.000987 m/s；RMS longitudinal accel 0.000202/0.000809 m/s²；RMS longitudinal jerk 0.000230/0.000710 m/s³；route progress 0.013880/0.020229 m。
15. nuisance gate：PASS。
16. official success rate：100%。
17. completion rate：100%。
18. off-road rate：2.50%。
19. responsible collision rate：1.25%。
20. safety gate：PASS。
21. canonical identity：PASS（80/80，mismatch 0）。
22. execution gate：PASS（80≥76）。
23. Stage7L-D总gate：PASS。
24. representation解锁：Yes — `STAGE7L_E_REPRESENTATION_EVALUATION_UNLOCKED`。
25. 读取embedding：No。
26. 计算BDD/MMD：No。
27. commit SHA：以本报告最终提交的Git记录为准；execution start commit为`63080ed8547966c572fe55e7819f93e7b99c44ea`。
28. push远端：以本报告最终提交后的push结果为准。
