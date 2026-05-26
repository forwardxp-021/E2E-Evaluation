# Stage 6B：ODD bins 与 Behavior-event bins 分层设计

Stage 6B 明确分离两层分箱：
- **Map-derived ODD bins**：用于 A/B 公平性控制（外部路况可比性）。
- **Behavior-event bins**：用于工程定位（漂移出现在哪类驾驶任务）。

`lateral_activity` 是行为污染变量（会被驾驶风格直接影响），因此只能作为 `event_lateral_activity_bin` 报告维度，不可作为主 ODD 控制变量。

## ODD bins 含义
- `odd_crosswalk_bin`：附近是否有人行横道。
- `odd_stop_sign_bin`：附近是否有 stop sign。
- `odd_curvature_bin`：直道/中等曲率/高曲率。
- `odd_intersection_bin`：路口样复杂场景代理。
- `odd_map_complexity_bin`：低/中/高地图复杂度。
- `odd_lane_count_bin`：简单/多车道/高密度车道上下文。

## Behavior-event bins 含义
- `event_following_bin`
- `event_cut_in_bin`（无显式标注时为 proxy）
- `event_lane_change_bin`
- `event_low_speed_bin` / `event_high_speed_bin`
- `event_yielding_bin`
- `event_lateral_activity_bin`

## 指标解释
- `BDD_overall`：未控制总体差异。
- `BDD_odd_balanced`：按 ODD bins 平衡后的总体差异（Stage 6B 主结论优先）。
- `BDD_following` / `BDD_cut_in` / `BDD_lane_change`：行为事件定位指标。
- `BDD_intersection`：路口样场景下差异。

## 为什么行为事件不能做主控制
行为事件（尤其换道、横摆、跟车压迫）可能由策略本身产生；若用于主匹配，会抵消真实风格差异。

## 当前限制
1. 需要原始 Waymo scenario 文件才能提取 map ODD。
2. speed/traffic-light 在部分数据中缺失或不稳定。
3. cut-in 可能只能用 proxy。
4. 车道拓扑精确匹配仍可继续增强。
