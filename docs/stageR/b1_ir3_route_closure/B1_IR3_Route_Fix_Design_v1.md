# B1-IR3 Route Fix Design v1

Implementation: `tools/b1_native_route_infrastructure.py`。

Production successor: `build_native_route_reference_v1_2_native_successor_backtracking`。

## 算法合同

1. 优先调用不可变的 `build_native_route_reference_v1_1`。如果成功，直接返回该对象；不更改坐标、shape、dtype、metadata、排序、current arc 或有效范围。
2. 只有原错误为 `NATIVE_ROUTE_FAIL: no native outgoing successor into ...` 时启用修复。其他异常和 route 终点覆盖不足继续失败，不被重试吞掉。
3. 使用 v1.1 的零距离查询取得完全相同的初始 edge、route occurrence cursor 和 current arc。
4. 在原 ordered route 的下一 roadblock 上，按原字符串 ID 排序搜索现有 outgoing edges。下游失败则回溯早先分支。每层 cursor 严格加一，因此搜索由冻结 route 长度界定，无循环探索、无无穷 horizon。
5. 有限 production 请求在原 required distance 满足时结束；静态完整 suffix 检查使用同一个 path resolver 的 route-end 模式。两者共享搜索实现，不另写近似 checker。
6. reference 由原 map baseline coordinates 按旧拼接规则构造，保留原采样、点序和 arclength 检查。没有修改地图或创建邻接关系。

## 生产接线

`R2BControllerAwarePlannerV1` 仅将其 route import alias 绑定到 v1.2；所有函数体 AST 与 IR2 commit 相同，TSB profile/distance/state-generation 函数未变。B1 executor gate alias 绑定到同模块 precheck，并要求授权 source_file_hashes 必须包含新 route 模块。

旧的 v1.1 和 IR2 tooling 保留，可继续复现原失败；它们不是新 B1 生产路径。新 planner 与新 executor precheck 直接共享 v1.2。旧授权不能执行新栈；仅重绑 executor/planner 而漏掉新 route 源码也会拒绝。

## 等价规则

在修改后结果检查前确定的比较规则：`reference_xy` 比较 dtype、shape、C-order 原始 ndarray bytes；所有其余返回字段作排序、无 rounding 的 canonical JSON 精确比较。没有数值容差，没有“只要不抛异常即等价”的规则。

原本失败的查询不存在旧 reference，因此只判定修复成功、native 邻接、ordered roadblocks、接点几何和 required distance 保持；不虚称其 pre/post reference 相等。2773 个旧成功查询全部满足完整返回 byte equality。

## 禁止事项

不跳过 roadblock、不制造地图边、不缩短 required distance、不提前结束 episode、不修改干预时钟、不训练或评价任何模型。找不到合法 native path 必须失败。科学失败永远不属于技术 retry eligible 类别。
