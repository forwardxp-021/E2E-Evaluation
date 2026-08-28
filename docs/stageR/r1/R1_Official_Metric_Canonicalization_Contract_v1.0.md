# R1 官方 Metric Canonicalization 合同 v1.0

本合同只定义官方 nuPlan Parquet 的确定性比较表示，不创建新的 scientific metric，也不解释任何
collision 或 drivable outcome。

## 冻结来源

语义直接继承既有 `tools/stage7l_extract_confirmation_metrics.py`：collision 使用
`number_of_all_at_fault_collisions_stat_value` 的整数计数；drivable 使用
`drivable_area_compliance_stat_value` 的二值语义。该来源工具 SHA 为
`eb834af609ac621881dda0202dae711a16406be57035ee59e42b9674b515c23f`。

## 精确文件与 canonical payload

- collision 只能来自恰好一个 `no_ego_at_fault_collisions.parquet`，canonical field 为
  `number_of_all_at_fault_collisions_stat_value`（integer）。
- drivable-area 只能来自恰好一个 `drivable_area_compliance.parquet`，canonical field 为
  `drivable_area_compliance_stat_value`（boolean；官方数值二值编码只能是有限的 0/1）。

canonical payload 固定为：

```json
{
  "collision": {"number_of_all_at_fault_collisions_stat_value": 0},
  "drivable_area": {"drivable_area_compliance_stat_value": true}
}
```

示例值只说明类型，不是 V2/V3 outcome，也不构成阈值。

## 比较与 fail-closed

Primary comparison 是 canonical JSON 的精确相等；Parquet container SHA 只记录 artifact provenance，
绝不作为 determinism primary。缺失、重复、不可读、空表、列缺失、非有限/非法值、类型不兼容或多行
scenario 均为 `TECHNICAL_FAILURE`；不允许模糊文件名替代或静默回退。
