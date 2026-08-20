# Stage7L-E 前瞻性 Representation / BDD 执行就绪报告（E1）

## 1. 本段范围

Stage7L-E总任务预计需要3–5小时，已按不超过2小时的单段工作拆分。本文件冻结第1段E1：只完成
Stage7L-D解锁复核、400条既有official rollout的输入视图、Stage5D `[N,T,83]` context、C2 task mask重放、
统一A/B/C推理工具和paired BDD统计实现测试。E1没有读取任何checkpoint或embedding，没有计算正式BDD/MMD，
也没有重新运行nuPlan。

当前状态：

`FROZEN_READY_FOR_STAGE7L_E_PROSPECTIVE_BDD_EXECUTION_NOT_RUN`

## 2. 解锁与冻结来源

- Stage7L-D状态：`STAGE7L_D_PLANNER_LEVEL_CONFIRMATION_PASSED`。
- Representation状态：`STAGE7L_E_REPRESENTATION_EVALUATION_UNLOCKED`。
- execution、canonical identity、mechanism、longitudinal nuisance、safety/validity、representation unlock均为true。
- Stage7L-D冻结提交`6279bc742ad527246a945a4b6d5d7090fab591ea`是当前HEAD祖先。
- protocol SHA256：`f5a8b2df5ed60c0384e8181feceab33f3c6f048780e95aab851184e49247490a`。
- roster SHA256：`90ec9b427636cefc59e6d7ace2507ac8364747e2a38964124be08fdc2a10acf9`。
- Stage7L-D manifest SHA256：`eb8c04795d1ee2c10419dbd9db3f9145ece9f9add618ba46241e53374ff461ba`。

## 3. 输入合同

五档各复用80条Stage7L-D成功trajectory，共400条；没有replacement、outcome filtering或planner rerun。原始数组
包含149和150步，完全沿用历史M6.5合同：不足150步只在右侧补零，新增mask为false。五档context均为
`[80,150,83]`，ego为`[80,150,8]`，mask为`[80,150]`；token/log/roster顺序逐档一致，全部finite。

C2 pre-treatment mask只重放冻结roster和`official_scenario_types_json`：

- `LAT.LANE_CHANGE = 80/80`；
- `LAT.DYNAMICS = 38/80`；
- task mask SHA256：`74206af6b0d7bc5be4b16e8ef8343feebad6ced722d1d3232e4e72c6396a3ec3`。

碰撞、off-road及高运动学结果没有被删除。E2正常预期四档pair count分别为80和38。

## 4. 低覆盖slot诊断

在读取任何checkpoint/embedding/BDD之前，通用builder的默认5%全局符号诊断发现`right_front`覆盖率约5.4%，
并在这个65/80为右变道的总体上出现聚合`rel_y`符号失败。分方向检查表明该符号随lane-change总体构成变化；shape、
finite、token、map解析、lane-aware assignment和feature公式均没有失败。

该诊断没有用于修改83D值、删除场景或切换assignment。最终构建仍使用原
`lane_aware_with_geometric_fallback`，仅把低于6%的聚合slot符号检查按builder既有机制归为低覆盖diagnostic；五档均PASS。
这一执行阈值只控制审计分类，不进入模型输入，也不改变任何representation或BDD统计。该限制必须保留在最终报告中。

## 5. 统计实现冻结

Stage7L-E直接复用Stage6V/W实际运行过的标准实现：

- exact pooled positive off-diagonal median bandwidth；
- biased single-RBF MMD² V-statistic；
- same-scenario pair内部label swap的等价random-sign quadratic form；
- 每格100,000次；
- plus-one upper-tail p；
- null mean/SD/q95、BDD/q95和`Z_BDD=(observed-null mean)/null SD`；
- 基准随机种子`2026081301`及Stage6V的representation/dose/task确定性offset；
- 理论40格中只排除一次`B_seed3407 × dose100 × LAT.LANE_CHANGE`，其余固定39格一次性Holm；
- non-computable格仍以p=1进入family；
- raw MMD²禁止跨representation排序。

synthetic测试验证了vectorized pair swap的数学支持、plus-one p、q95、Z、确定性null、Holm39、Primary唯一排除、
non-computable p=1和A/B/C/old64的83D→64D forward合同。

## 6. 下一段

E2单独授权后才会首次读取四个锁定checkpoint及ego13 scaler，导出五档embedding并计算Primary和39格secondary BDD。
预计E2需60–100分钟。E2结束后冻结机器结果，不更新论文结论；E3再用45–75分钟生成中文总报告、统一BDD矩阵更新并提交远端。
