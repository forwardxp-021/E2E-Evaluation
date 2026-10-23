# B1-IR3 B1-18 Root Cause v1

精确根因：`GREEDY_LANE_SUCCESSOR_SELECTION_WITHOUT_BACKTRACKING`。19339 及其合法连接没有缺失；旧实现未在下游匹配失败时回溯早先 lane-level 分支。

## 官方原生拓扑

| Roadblock | Native edge | 对 frozen route 有关的 outgoing |
|---|---|---|
| 19000 | 19812 | 20202、20203，均属于 19512 |
| 19512 | 20202 | 18519，属于 18999 |
| 19512 | 20203 | 18520，属于 18999 |
| 18999 | 18519 | 18359→19338；19951→19337；没有 19339 |
| 18999 | 18520 | 19952→19337；19953→19339 |
| 19339 | 19953 | 18285→19062 |

v1.1 每层按字符串 edge ID 排序取第一个，故选中 20202。只有在前向请求触及 19339 时才发现该分支不能满足 ordered route；它没有尝试早先同 roadblock 的另一条既有连接 20203。

## 冻结失败输入

来源：B1-18 treatment 原 trace iteration 53，nominal 5.3 s。

- rear axle：x=588079.9128488203，y=4475752.550421184，heading=-2.479414586631137。
- speed=10.409060120214438 m/s；time_us=1630329270049697。
- required_forward_m=82.231574950694，保持原冻结距离函数结果。
- 原 greedy path 的可解析距离约 81.612 m。

v1.2 对同一输入先复现 v1.1 的原错误，然后回溯得到 `19812 → 20203 → 18520 → 19953`。该序列每一条相邻关系均来自原 map API outgoing_edges，baseline 接点逐个完全相等；没有合成 bridge 或 extrapolation。

地图端点证据和 map.gpkg SHA256 在 `B1_IR3_Static_Evidence_v1.json`。每个 pre/post 查询的输入 hash、原 trace hash、route/precontext identity 和输出 hash 在 `B1_IR3_Pre_Post_Equivalence_Audit_v1.csv`。

## 合法性和限制

这是情况 A：识别并使用地图中已有合法连接，不属于 `SCIENTIFIC_CONTRACT_CHANGE_REQUIRED`。route sequence、初始 state、required distance、TSB、episode、采样和干预时钟全部保持原样。旧算法已经成功的 2,773 个查询没有任何返回变化。

这仅证明原故障输入修复成立；B1-18 treatment 没有重跑。它的 attempt 1 仍是 INFRASTRUCTURE_FAILURE，不升级为 PASS。后续 attempt 2 必须由 Owner 明确授权，所有新异常和科学结果必须保留。
