# B1-IR3 Native Route Infrastructure Closure v1

主状态：`B1_IR3_READY_FOR_RESUME_OWNER_REVIEW`。

正式 B1 科学状态仍是 `B1_TSB_BENCHMARK_QUALIFICATION_INCOMPLETE_INFRASTRUCTURE`。本轮闭合的是可提交 Owner 审核的路由基础设施条件，不是 B1 PASS，也不是恢复执行授权。

## 闭合结果

旧生产 builder 在 native lane 分支上贪心选择最小字符串 ID，缺少失败后的回溯。进入 connector 19512 时，它选择 `20202 → 18519`；18519 无法通往 frozen route 下一段 19339。官方地图已经存在合法分支 `20203 → 18520 → 19953`，对应完全相同的 ordered roadblocks `19000 → 19512 → 18999 → 19339`，所有接点距离精确为 0 米。

新增版本 `build_native_route_reference_v1_2_native_successor_backtracking` 首先调用不可变 v1.1。原来成功的查询原样返回，包括所有 ndarray bytes、标量、排序、metadata 和旧 builder_version。只有原始错误明确为 missing outgoing successor 时才搜索已有原生分支；保持当前 edge、cursor、required distance、原 ordered route 和坐标拼接规则。不增加地图边，不改 route，不缩短规划窗口，不生成 trajectory。

| 审计项 | 结果 |
|---|---:|
| B1-01～17 全部历史有限查询 | 2,720/2,720 全返回逐字节一致 |
| B1-06/10/14 重点查询（上述子集） | 480/480 逐字节一致 |
| B1-18 失败前成功查询 | 53/53 逐字节一致 |
| 原历史成功查询总计 | 2,773/2,773 逐字节一致 |
| B1-18 iteration 53 原失败输入 | 修复，合法 native reference，接点间隙 0 m |
| 原始 20 pair / 40 arm 静态 gate | 40/40 COMPATIBLE |
| 新 scientific execution | 0 |

所有历史 B1 raw 产物、原 manifest、trajectory、recorder、恢复官方 metric 和 IR/IR2 文档在审计前后 hash 相同。前 17 对 mechanism、F_match、official safety、full-contract scientific PASS 均继续 17/17，不重算、不重跑。

## B1-20 准入依据

它与 B1-18 的远端故障来自同一 greedy dead end。初始有限生产查询通过；修复后的共享 resolver 从原生产初始 edge/cursor 沿完整剩余 ordered route 解析到 19339，全部 native adjacency 存在，全部接点间隙为 0 m。其路由状态据此为 `COMPATIBLE`，不再依赖仅有的初始 715 m 容量判断，也没有把 unknown 直接改名为 compatible。

该证据满足“初始有限生产查询 + 原生剩余 route 可解析 + 已暴露真实故障修复 + 历史成功 reference 不变”的工程准入范围。不声称对所有理论未来 ego state、脱离 route、route 终点外请求或未来完整闭环必然成功作证明。未来真实运行异常仍必须停止和保留，不能作为科学失败的 retry-rescue。

## 科学代码边界

旧 route builder、TSB generator、参数、measurement/analyzer、safety adapter、serializer、passive recorder 均逐字节不变。planner 仅改变 route builder 的 import binding，所有函数体 AST 相同。B1 executor 只改变 gate import 并新增新 route 源码必须纳入授权 source_file_hashes 的关闭式校验；runner、controller、采样和科学分析流程不变。

当前仓库未找到 v2.4 文件，当前依据为 v2.3 和已批准冻结合同。v2.3 状态为受控验证 freeze draft；S1R Owner approval 和后续 B1 冻结记录提供具体执行边界。

## 恢复与停止

01～17 NO_RERUN；18 baseline 待首次执行，treatment 原 attempt 1 永久保留，attempt 2 仅在 Owner 授权后可使用；19/20 只在 Owner 授权后首次执行。原分母仍为 20，最大 technical attempts 仍为每 arm 2，科学失败不能 retry。

旧授权的 executor/planner hash 不匹配新栈，且未绑定新路由源码，因此不可执行。后续授权必须固定新源码 commit/hash、原身份、attempt namespace 和剩余预算；本轮不改写原账本、不实现覆盖 attempt 1 的捷径。新版恢复计划只是 prospective 计划。

`B1_RESUME_EXECUTION=NOT_AUTHORIZED; RBR_TRAINING=NOT_AUTHORIZED; PRIMARY_EVALUATION=NOT_AUTHORIZED`。

`SIMULATION=0; RUNNER_RUN=0; TSB_ROLLOUT=0; HLC_ROLLOUT=0; RBR_TRAINING=0; PRIMARY_EVALUATION=0; V_EXECUTION=0; C_EXECUTION=0; NEW_SCIENTIFIC_OUTCOME_EXPOSURE=0`。

下一阶段仅为 **Scientific Owner review → B1 Resume Authorization**。本轮完成后停止。

## 已运行验证

最终定向 suite：`tests/test_b1_ir3_route_closure.py`、IR2、IR 和 B1 regression，共 **41 passed，0 skipped**。测试覆盖原生分支修复、旧成功字节恒等、不存在的连接必须失败、其他异常保留、production/gate 同源、旧授权及漏绑新源码拒绝、identity 拒绝、所有历史输入、所有历史文件 hash、全部报告/CSV/JSON/manifest 的确定性逐字节复现。runner、planner trajectory generation、controller update、LQR track 和 motion-model propagation 在测试中全部禁止调用。

全仓 `python3 -m py_compile tools/*.py`（Python 3.14.4）、nuPlan Python 3.9 对全部修改工具/测试的编译、`check_no_tmp_dependencies.py` 和 `git diff --check` 均通过。仅有第三方 matplotlib/pyparsing deprecation 与 nuPlan map cast warning，没有失败或跳过项。
