# Stage6V 一次性盲测中文总报告

## 1. 最终结论

本轮状态为`FROZEN_STAGE6V_ONE_TIME_BLIND_EVALUATION_COMPLETE`。预冻结最终决策是：

`NO_ABC_CANDIDATE_QUALIFIES_UNDER_PRE_FROZEN_RULE`

最强正结果来自Stage6P：n=400的context-balanced非配对发布检出率从old64的66.5%提升到
A/B/C的90.5%/100%/99.5%，且A/A FPR保持在3.0%/5.0%/6.5%。这证明新数据与训练目标确实恢复了
release-level纵向信号。

但是，A/B/C都没有同时通过Waymo primary longitudinal门禁和Stage6J/K paired完整门禁；Stage6S-v2
冻结roster还因场景可运行性建榜遗漏而未完成。因此不能依据单项Stage6P优异结果，把A、B或C指定为符合
全部预冻结条件的论文主模型。

## 2. 盲测授权与不可变性

Blind Evaluation Authorization在任何Waymo test与新checkpoint nuPlan结果出现前创建，绑定：

- Stage6T protocol fingerprint：`0f33a7939030397488dd53179fa7821e3b0c9a81721f01235a380ee058f5b1da`；
- Stage6U implementation freeze SHA：`4160c599aab64144d525949f63d6847792a7d6fe668c8aa5208eea580e0a817c`；
- formal training authorization SHA：`0e501742401897a62f05e691a22cd2851c9e9b85f0aa0678008f20a59242924c`；
- checkpoint ledger SHA：`e87c74527d3702de49bc68bebd47ebb485f3ced2a143cd5724cc3c12d59e7ab5`；
- 9个best checkpoint SHA、primary seed 3407，以及80-pair Stage6S-v2 roster SHA
  `a2360bf38f8c60fc481226d580475cb651cf503cb46c5b6167ad1d916d50d1e2`。

授权manifest SHA为`c7f945b3236856b4bb0ee9c8e888c2eca83856dd6201d4c4c957fae9dacef5bd`，明确记录：
`evaluation results cannot trigger retraining or protocol changes`。本轮没有训练返工、换seed、换epoch、改loss、
改architecture或修改既有benchmark。

## 3. Waymo Dynamic-v2 test

测试使用16784行、2446个scenario cluster；primary结果固定使用seed 3407，3408/3409只用于seed stability。
所有差值均相对同一个Dynamic-v2 test上重新计算的old64。

| Candidate | longitudinal delta（95% CI） | following/lateral/behavior/retrieval非劣性 | 完整Waymo门禁 |
|---|---:|---|---|
| A-3407 | -0.0232（-0.0289, -0.0176） | 通过 | 未通过 |
| B-3407 | +0.0248（+0.0181, +0.0314） | 通过 | 未通过 |
| C-3407 | +0.0159（+0.0096, +0.0223） | 通过 | 未通过 |

A的纵向指标显著下降。B/C有正向改善，但primary effect没有达到冻结幅度。三个candidate的3/3 seed均通过
综合非劣性；只有B-3409通过全部Waymo门禁，但primary seed已在结果出现前固定为3407，所以不能事后换seed。
A与B/C的candidate-specific total loss含义不同，没有直接横向比较。

## 4. Stage6J/K纵向paired能力

盲测严格复用183个same-scenario pair、25/50/75/100%四剂量、各representation独立bandwidth/null、
100000次pair内交换和Holm task×dose协议。

| Representation | overall Holm | task×dose Holm | 最小检出剂量 | median Z_BDD | 冻结门禁 |
|---|---:|---:|---:|---:|---|
| old64 | 4/4 | 7/12 | 25% | 7.539 | 未通过 |
| A | 4/4 | 7/12 | 25% | 8.630 | 未通过 |
| B | 3/4 | 2/12 | 50% | 6.015 | 未通过 |
| C | 3/4 | 2/12 | 50% | 5.189 | 未通过 |
| ego13 | 4/4 | 12/12 | 25% | 21.115 | 通过 |

结论：ego13明确最好；learned64中A最好，但只与old64具有相同的overall/task coverage。B/C没有在冻结paired
benchmark中恢复纵向敏感性。没有跨representation比较raw MMD²。

## 5. Stage6P非配对发布

复用原800 pair、489 log和2400 split；每个representation×样本量×方法独立完成A/A calibration。

| Representation | n=400 A/A FPR | context-balanced A/B detection | 两方向 | detection−FPR | 冻结门禁 |
|---|---:|---:|---:|---:|---|
| old64 | 5.0% | 66.5% | 62% / 71% | 61.5 pp | 未通过 |
| A-3407 | 3.0% | 90.5% | 91% / 90% | 87.5 pp | 通过 |
| B-3407 | 5.0% | 100% | 100% / 100% | 95.0 pp | 通过 |
| C-3407 | 6.5% | 99.5% | 100% / 99% | 93.0 pp | 通过 |
| ego13 | 2.0% | 100% | 100% / 100% | 98.0 pp | 通过 |

C满足整体≥80%、双方向各≥75%、FPR≤7.5%以及raw detection≥75%的全部冻结门槛。C三个seed的
context-balanced detection均为99.5%，FPR为6.5%/5.5%/6.0%；B三个seed均为100% detection，FPR为
5.0%/6.0%/7.5%。因此提升具有seed稳定性。

与old64使用相同release trials做配对比较，A/B/C的context-balanced detection分别提高24.0/33.5/33.0
个百分点，95% CI分别为[18.0, 30.5]/[27.0, 40.0]/[26.5, 40.0]个百分点。这是本轮最有力的工程正结果。

## 6. Stage6S-v2 confirmation

80个冻结scenario中61个official rollout成功，19个失败；全部失败token在原场景上重试一次后仍失败。19项均为
`NUPLAN_VALID_SCENES_BOUNDARY_EXCLUSION`：token真实存在于对应DB，但属于第一个或倒数第二个scene。
nuPlan官方查询只接受按scene name排序后满足`row_num >= 3 AND row_num < scene_count - 1`的scene，Stage6S-v2
pre-treatment inventory建榜时没有复用这一规则。

这是confirmation roster构造/执行完整性失败，不是模型或planner机制失败。由于80-pair roster已经冻结，不能在
看到结果后替换19个场景，也不能把61个成功子集事后定义为新的confirmation set。因此：

- trajectory mechanism gate未评估；
- interaction representation evaluation未解锁；
- 没有读取Stage6S-v2 embedding/BDD；
- C full-context相对C neighbor-zero的null-standardized ΔZ与log-cluster bootstrap CI未计算。

所以对“C是否有增量interaction信息”的正确结论是**不可判定**，不是“没有增量”。

## 7. 最终模型决策

- C：Stage6P通过，但Waymo primary、Stage6J/K和Stage6S-v2完整证据链未通过，不入选；
- B：Stage6P最强且跨seed稳定，但Waymo primary和Stage6J/K未通过，不能按“更简单模型优先”规则入选；
- A：Stage6P明显改善，但Waymo longitudinal下降，paired不优于old64，不入选。

因此A/B/C没有任何一个满足预冻结联合规则。old64继续作为冻结历史baseline，ego13继续作为纵向敏感性参考
上界；ego13的结果不能被解释为interaction/context无用。

## 8. 论文可写结论与限制

可以写入论文：

1. Dynamic Builder v2与新训练目标使64D representation在受控nuPlan纵向版本差异的异log/异场景
   release-level检出中取得大幅、跨seed稳定的提升；
2. B/C把old64的66.5% context-balanced detection提高到100%/99.5%，FPR仍受独立A/A calibration控制；
3. B/C在Waymo following/lateral/behavior/retrieval指标上保持冻结非劣性，说明unpaired提升不是以这些指标的
   明显退化换取的；
4. paired与unpaired成绩出现明显分离，说明两者测量的是不同operating condition，不能只用单一BDD值或单一
   benchmark定义模型成功。

必须作为限制或负结果：

1. A/B/C均未通过完整Waymo primary longitudinal与Stage6J/K task-coverage联合门禁；
2. B/C没有在paired dose benchmark上恢复old64缺失的完整纵向敏感性；
3. Stage6S-v2 confirmation因pre-treatment roster遗漏官方scene可运行性规则而失败，无法证明C相对neighbor-zero
   有增量interaction信息；
4. 本轮不能选出满足全部预注册条件的最终64D论文主模型；
5. 结论不能外推为通用BDD阈值、真实整车厂发布可靠性、安全有效性或无需A/A标定的部署方案。

机器可审计manifest和完整输出位于`outputs/stage6v_one_time_blind_evaluation_final_v1/`。
