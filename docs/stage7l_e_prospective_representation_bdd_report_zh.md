# Stage7L-E Prospective Representation / BDD 最终中文报告

> 最终状态：`STAGE7L_E_PROSPECTIVE_REPRESENTATION_EVALUATION_COMPLETE`
> Primary状态：`STAGE7L_E_PRIMARY_BDD_FAILED`
> 本报告接受预注册失败结果；没有换checkpoint、换task、调kernel、重训或rescue experiment。

## 1. Frozen provenance

- Stage7L-D冻结commit：`6279bc742ad527246a945a4b6d5d7090fab591ea`。
- Stage7L protocol SHA：`f5a8b2df5ed60c0384e8181feceab33f3c6f048780e95aab851184e49247490a`。
- roster SHA：`90ec9b427636cefc59e6d7ace2507ac8364747e2a38964124be08fdc2a10acf9`。
- E2 preformal commit：`a85314a34518aaec627dca7baf5b73b15483553c`。
- 400条Stage7L-D official rollout原样复用；planner rerun=No，replacement=0，outcome filtering=No。

## 2. Stage7L-D unlock evidence

execution、canonical identity、mechanism、longitudinal nuisance、safety/validity和representation unlock六项均为PASS。80/80场景五档完整，400/400 official rollout成功。

## 3. Representation contracts

仅使用old64、A3407、B3407、C3407、ego13。四个learned representation复用Stage6V/W冻结83D→64D inference；ego13复用冻结13D scaler。Primary固定B3407，没有按结果换seed。

## 4. Pair populations与输入合同

五档context均为`[80,150,83]`、float32、finite、scenario order一致。`LAT.LANE_CHANGE`四档均80 pair/79 log；`LAT.DYNAMICS`四档均38 pair/38 log，后者继续是pre-treatment high-motion `MIXED_PROXY`。

## 5. 预注册Primary B结果

| Rep | Contrast | Task | N | raw MMD² | null mean | null SD | null q95 | BDD/q95 | Z_BDD | plus-one p | 结论 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B3407 | dose100 − dose0 | LAT.LANE_CHANGE | 80 | 0.001075041 | 0.001118377 | 0.000666342 | 0.002466807 | 0.436× | -0.065 | 0.411906 | FAIL |

planner-level pure-lateral treatment confirmation成功，但Candidate B未通过预先指定的prospective paired BDD endpoint。BDD failure不否定物理mechanism；它否定的是B在该冻结任务上的预注册检测主张。

## 6. dose100五representation对照

| Representation | raw MMD²（仅本rep内解释） | null q95 | BDD/q95 | Z_BDD | p | 身份 |
|---|---|---|---|---|---|---|
| old64 | 0.000566707 | 0.000976291 | 0.580× | -0.646 | 0.719943 | Secondary |
| A | 0.000861210 | 0.001873710 | 0.460× | -0.021 | 0.378166 | Secondary |
| B | 0.001075041 | 0.002466807 | 0.436× | -0.065 | 0.411906 | Primary |
| C | 0.001292183 | 0.002369465 | 0.545× | 0.373 | 0.262247 | Secondary |
| ego13 | 0.044692576 | 0.003415018 | 13.087× | 40.201 | 9.9999e-06 | Secondary |

ego13具有该Treatment下最高within-null标准化敏感度。该treatment直接改变ego横向运动学，因此不能写成ego13全局最佳，也不能据此否定neighbor/context。raw MMD²没有跨representation排序。

## 7. LAT.LANE_CHANGE四档dose curve

| Representation | Dose | N | BDD/q95 | Z_BDD | raw p | Holm p | Status |
|---|---|---|---|---|---|---|---|
| old64 | dose25 | 80 | 0.710× | -0.069 | 0.494915 | 1 | PROSPECTIVE_SECONDARY_HOLM_NOT_SIGNIFICANT |
| old64 | dose50 | 80 | 0.589× | -0.721 | 0.744213 | 1 | PROSPECTIVE_SECONDARY_HOLM_NOT_SIGNIFICANT |
| old64 | dose75 | 80 | 0.577× | -0.786 | 0.778912 | 1 | PROSPECTIVE_SECONDARY_HOLM_NOT_SIGNIFICANT |
| old64 | dose100 | 80 | 0.580× | -0.646 | 0.719943 | 1 | PROSPECTIVE_SECONDARY_HOLM_NOT_SIGNIFICANT |
| A | dose25 | 80 | 0.226× | -1.298 | 0.97959 | 1 | PROSPECTIVE_SECONDARY_HOLM_NOT_SIGNIFICANT |
| A | dose50 | 80 | 0.264× | -0.671 | 0.699623 | 1 | PROSPECTIVE_SECONDARY_HOLM_NOT_SIGNIFICANT |
| A | dose75 | 80 | 0.515× | 0.420 | 0.234398 | 1 | PROSPECTIVE_SECONDARY_HOLM_NOT_SIGNIFICANT |
| A | dose100 | 80 | 0.460× | -0.021 | 0.378166 | 1 | PROSPECTIVE_SECONDARY_HOLM_NOT_SIGNIFICANT |
| B | dose25 | 80 | 0.371× | -0.914 | 0.772042 | 1 | PROSPECTIVE_SECONDARY_HOLM_NOT_SIGNIFICANT |
| B | dose50 | 80 | 0.256× | -0.818 | 0.799182 | 1 | PROSPECTIVE_SECONDARY_HOLM_NOT_SIGNIFICANT |
| B | dose75 | 80 | 0.580× | 0.433 | 0.268717 | 1 | PROSPECTIVE_SECONDARY_HOLM_NOT_SIGNIFICANT |
| B | dose100 | 80 | 0.436× | -0.065 | 0.411906 | Primary—excluded | PROSPECTIVE_PRE_REGISTERED_PRIMARY_FAILED |
| C | dose25 | 80 | 0.181× | -1.434 | 0.96976 | 1 | PROSPECTIVE_SECONDARY_HOLM_NOT_SIGNIFICANT |
| C | dose50 | 80 | 0.284× | -0.693 | 0.724033 | 1 | PROSPECTIVE_SECONDARY_HOLM_NOT_SIGNIFICANT |
| C | dose75 | 80 | 0.600× | 0.615 | 0.206478 | 1 | PROSPECTIVE_SECONDARY_HOLM_NOT_SIGNIFICANT |
| C | dose100 | 80 | 0.545× | 0.373 | 0.262247 | 1 | PROSPECTIVE_SECONDARY_HOLM_NOT_SIGNIFICANT |
| ego13 | dose25 | 80 | 12.520× | 39.387 | 9.9999e-06 | 0.000389996 | PROSPECTIVE_SECONDARY_HOLM_SIGNIFICANT |
| ego13 | dose50 | 80 | 12.636× | 39.336 | 9.9999e-06 | 0.000389996 | PROSPECTIVE_SECONDARY_HOLM_SIGNIFICANT |
| ego13 | dose75 | 80 | 12.859× | 39.881 | 9.9999e-06 | 0.000389996 | PROSPECTIVE_SECONDARY_HOLM_SIGNIFICANT |
| ego13 | dose100 | 80 | 13.087× | 40.201 | 9.9999e-06 | 0.000389996 | PROSPECTIVE_SECONDARY_HOLM_SIGNIFICANT |

中间dose不要求严格单调。learned64四种表示均未在该任务获得显著secondary或Primary结果；ego13四档均通过Holm。

## 8. LAT.DYNAMICS secondary

| Representation | Dose | N | BDD/q95 | Z_BDD | raw p | Holm p | Status |
|---|---|---|---|---|---|---|---|
| old64 | dose25 | 38 | 0.584× | -1.010 | 0.817022 | 1 | PROSPECTIVE_SECONDARY_LOW_N_MIXED_PROXY_HOLM_NOT_SIGNIFICANT |
| old64 | dose50 | 38 | 0.329× | -1.869 | 0.9935 | 1 | PROSPECTIVE_SECONDARY_LOW_N_MIXED_PROXY_HOLM_NOT_SIGNIFICANT |
| old64 | dose75 | 38 | 0.644× | -0.445 | 0.647054 | 1 | PROSPECTIVE_SECONDARY_LOW_N_MIXED_PROXY_HOLM_NOT_SIGNIFICANT |
| old64 | dose100 | 38 | 0.691× | 0.048 | 0.471985 | 1 | PROSPECTIVE_SECONDARY_LOW_N_MIXED_PROXY_HOLM_NOT_SIGNIFICANT |
| A | dose25 | 38 | 0.970× | 1.029 | 0.182068 | 1 | PROSPECTIVE_SECONDARY_LOW_N_MIXED_PROXY_HOLM_NOT_SIGNIFICANT |
| A | dose50 | 38 | 0.448× | -0.562 | 0.521515 | 1 | PROSPECTIVE_SECONDARY_LOW_N_MIXED_PROXY_HOLM_NOT_SIGNIFICANT |
| A | dose75 | 38 | 1.033× | 1.797 | 0.0253197 | 0.784912 | PROSPECTIVE_SECONDARY_LOW_N_MIXED_PROXY_HOLM_NOT_SIGNIFICANT |
| A | dose100 | 38 | 1.028× | 2.000 | 0.0330997 | 0.99299 | PROSPECTIVE_SECONDARY_LOW_N_MIXED_PROXY_HOLM_NOT_SIGNIFICANT |
| B | dose25 | 38 | 0.997× | 1.919 | 0.0514995 | 1 | PROSPECTIVE_SECONDARY_LOW_N_MIXED_PROXY_HOLM_NOT_SIGNIFICANT |
| B | dose50 | 38 | 0.312× | -1.094 | 0.883631 | 1 | PROSPECTIVE_SECONDARY_LOW_N_MIXED_PROXY_HOLM_NOT_SIGNIFICANT |
| B | dose75 | 38 | 0.921× | 1.574 | 0.0832392 | 1 | PROSPECTIVE_SECONDARY_LOW_N_MIXED_PROXY_HOLM_NOT_SIGNIFICANT |
| B | dose100 | 38 | 0.856× | 1.514 | 0.110339 | 1 | PROSPECTIVE_SECONDARY_LOW_N_MIXED_PROXY_HOLM_NOT_SIGNIFICANT |
| C | dose25 | 38 | 0.974× | 1.623 | 0.0848192 | 1 | PROSPECTIVE_SECONDARY_LOW_N_MIXED_PROXY_HOLM_NOT_SIGNIFICANT |
| C | dose50 | 38 | 0.513× | -1.107 | 0.871301 | 1 | PROSPECTIVE_SECONDARY_LOW_N_MIXED_PROXY_HOLM_NOT_SIGNIFICANT |
| C | dose75 | 38 | 0.941× | 1.700 | 0.0834392 | 1 | PROSPECTIVE_SECONDARY_LOW_N_MIXED_PROXY_HOLM_NOT_SIGNIFICANT |
| C | dose100 | 38 | 1.038× | 2.219 | 0.0376796 | 1 | PROSPECTIVE_SECONDARY_LOW_N_MIXED_PROXY_HOLM_NOT_SIGNIFICANT |
| ego13 | dose25 | 38 | 6.275× | 19.662 | 9.9999e-06 | 0.000389996 | PROSPECTIVE_SECONDARY_LOW_N_MIXED_PROXY_HOLM_SIGNIFICANT |
| ego13 | dose50 | 38 | 6.262× | 19.611 | 9.9999e-06 | 0.000389996 | PROSPECTIVE_SECONDARY_LOW_N_MIXED_PROXY_HOLM_SIGNIFICANT |
| ego13 | dose75 | 38 | 6.427× | 19.788 | 9.9999e-06 | 0.000389996 | PROSPECTIVE_SECONDARY_LOW_N_MIXED_PROXY_HOLM_SIGNIFICANT |
| ego13 | dose100 | 38 | 6.624× | 20.269 | 9.9999e-06 | 0.000389996 | PROSPECTIVE_SECONDARY_LOW_N_MIXED_PROXY_HOLM_SIGNIFICANT |

该38场景slice全部标记`LOW_N_SECONDARY_DIAGNOSTIC`。A/C的部分raw p低于0.05，但39-test Holm后均不显著；只有ego13四档通过。不得把该slice称为pure lateral dynamics ground truth。

## 9. 固定39-test Holm family

理论矩阵为5 representations×4 doses×2 tasks=40格；唯一B×dose100×LAT.LANE_CHANGE Primary只排除一次，secondary family固定39格。Holm通过8格，全部来自ego13；NOT_COMPUTABLE=0，LOW_N=20。

## 10. Semantic mechanism与BDD方向分离

所有semantic delta为Sharp dose100−Gentle dose0，方向来自Stage7L-D trajectory mechanism，而不是MMD正值：

| Metric | paired median Δ | direction consistency | 95% log-cluster CI | Gate |
|---|---|---|---|---|
| lane_change_duration_s | -0.200160 | 88.75% | [-0.20041489601135254, -0.19967222213745117] | PASS |
| rms_lateral_accel_mps2 | +0.055832 | 100.00% | [0.044970850180745814, 0.06286575735295391] | PASS |
| peak_yaw_rate_radps | +0.014404 | 96.25% | [0.013368515271013237, 0.014992355302704041] | PASS |

纵向nuisance gate仍为True。因此正确组合结论是：planner-level横向机制positive，但B representation Primary detection negative。

## 11. Positive / negative findings

- Positive：prospective pure-lateral planner treatment在80个冻结场景上产生方向正确、纵向副作用极小的横向机制差异。
- Positive：ego13在两个task的四档secondary均显著，证明冻结统计管线能够检出运动学处置。
- Negative：B的唯一Primary未通过；old64/A/C在dose100 lane-change secondary也未通过。
- Negative：learned64没有在本前瞻pure-lateral任务中证明稳定检测能力。

## 12. Stage6V compatibility与claim boundary

Stage6V联合结论继续是`NO_ABC_CANDIDATE_QUALIFIES_UNDER_PRE_FROZEN_RULE`。B仍是最简单、最强的release-level learned工程候选，但Stage7L没有为其增加pure-lateral paired成功证据。A仍是dynamic-data repair ablation；C仍是dual-branch ablation，Stage7L不是interaction-specific benchmark，不能改写Stage6S-v3的context增量负结果。

允许写：经前瞻冻结的pure-lateral planner treatment得到物理确认，但B未通过预注册BDD主端点；ego13在该运动学处置中最敏感。禁止写：B完成横向验证、ego13全局最佳、context无价值、新64D全面优于old64。

## 13. Thesis implication

Stage7L补齐的是一个可信的确认性负结果：框架成功区分了‘行为确实变化’与‘指定representation能否检出’。这直接支撑task-conditioned representation qualification，而不是削弱论文主线。Stage7L实验链到此关闭，后续只允许论文写作、图表和claim cleanup。

## 14. 任务要求29项核对

1. Stage7L-D unlock：已验证，六项required gate均PASS。
2. 原400条rollout：是；未重新仿真。
3. input contract：通过，五档均`[80,150,83]`且finite。
4. old64/A/B/C/ego13：全部推理成功。
5. LAT.LANE_CHANGE N_pair：25/50/75/100均80。
6. LAT.DYNAMICS N_pair：25/50/75/100均38。
7. Primary raw MMD²：`0.0010750406060657607`。
8. Primary null q95：`0.00246680739100163`。
9. Primary BDD/q95：`0.4358024100249059`。
10. Primary Z_BDD：`-0.06503666023600715`。
11. Primary plus-one p：`0.41190588094119057`。
12. Primary：FAIL。
13. old64 dose100：ratio/Z/p=`0.580469/-0.645764/0.719943`。
14. A dose100：ratio/Z/p=`0.459629/-0.021421/0.378166`。
15. B dose100：ratio/Z/p=`0.435802/-0.065037/0.411906`。
16. C dose100：ratio/Z/p=`0.545348/0.373013/0.262247`。
17. ego13 dose100：ratio/Z/p=`13.087068/40.201025/9.9999e-06`。
18. 最高标准化敏感度：ego13；仅限该kinematic-heavy treatment。
19. B Z曲线：`-0.914/-0.818/0.433/-0.065`。
20. 五representation曲线：完整列于第7节；learned64不显著，ego13四档显著。
21. LAT.DYNAMICS：38场景/38 log，完整列于第8节。
22. Holm通过：8/39。
23. NOT_COMPUTABLE：0。
24. LOW_N：20。
25. 跨representation比较raw MMD²：No。
26. Stage6V qualification改变：No。
27. Stage7L-E最终状态：`STAGE7L_E_PROSPECTIVE_REPRESENTATION_EVALUATION_COMPLETE`。
28. 最终commit SHA：由E3提交后写入Git历史；manifest绑定提交前全部证据SHA。
29. 远端同步：由E3提交后完成并在最终回复确认。

`STAGE7L_E_PROSPECTIVE_REPRESENTATION_EVALUATION_COMPLETE`
