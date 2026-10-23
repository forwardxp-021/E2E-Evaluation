# B1-IR3 B1-20 Compatibility v1

`B1_20_ROUTE_STATUS = COMPATIBLE`，范围为本轮声明的静态 native-route 工程准入条件。两 arm 仍为 NOT_RUN，执行仍未授权。

IR2 的初始有限请求通过、初始可解析距离约 715.175 m、所需距离约 85.113 m 的事实保持不变。该距离差没有被当成全 episode 成功证明。

## 新增闭合证据

1. B1-20 的 ordered route 在远端也经过 `19000 → 19512 → 18999 → 19339`，原无限 greedy 诊断暴露的 dead end 与 B1-18 相同。
2. 新 production successor 的初始有限查询通过。
3. 同一个共享 native path kernel 能从 v1.1 选择的原初始 edge/cursor，解析完整剩余 route suffix：

```text
Roadblocks:
19045 → 19463 → 19074 → 19404 → 14541 → 19445 →
19000 → 19512 → 18999 → 19339

Native edges:
18524 → 20156 → 18516 → 18409 → 19802 → 20141 →
19812 → 20203 → 18520 → 19953
```

初始 frozen route 的 19516 位于当前 cursor 之前；它没有被删除或改写。route hash 仍绑定原完整 ordered IDs。所有相邻 native edges 由官方 outgoing_edges 直接连接，全部九个接点 gap 精确为 0 m。

4. B1-18 已有真实 failure input 证明该共享 kernel 能修复这一类故障；B1-01～17 所有已有成功查询证明旧成功行为保持原样。

因此这里消除的是已识别的 greedy successor 解析不确定性，没有用新仿真结果决定准入，没有替换 SESSION，也没有改变初始 route identity。

## 证据边界

没有 B1-20 realized trace，未伪造这种输入。full-route suffix 证据是静态 graph 解析，不是未来轨迹预测或 controller rollout。正常 runtime 风险仍包括离开 route、其他 lane 选择、未来请求超过 route 终点或环境故障；本轮不要求对所有理论未来车辆状态证明绝对兼容性，也不声称闭环必然成功。

未来首次执行仍须绑定新源码版本、完整 pre-run gate 和 Owner 授权。任何实际失败保留在原 20-pair 分母内；不能因此换样本、缩短 horizon 或绕过 retry 上限。
