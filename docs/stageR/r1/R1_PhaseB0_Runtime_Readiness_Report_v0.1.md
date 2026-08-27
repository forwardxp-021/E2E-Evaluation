# R1 Phase B0 官方 nuPlan 运行时就绪报告 v0.1

状态：`NOT_READY`。本报告只审计环境、版本、文件存在性与接口，不运行 scenario、planner、background replay 或安全指标。

## 逐项结论

|项目|状态|证据与边界|
|---|---|---|
|nuPlan import|`READY`|专用 Python 3.9 环境配合本地 `nuplan-devkit`/`tuplan_garage` 路径可导入；`waymo_dev` 不是本任务运行环境。|
|exact nuPlan version|`READY`|`nuplan-devkit==1.2.2`，`hydra-core==1.1.0rc1`，`omegaconf==2.1.0rc1`。|
|database access|`NOT_READY`|候选 cache 根存在但为 0 B；没有建立可执行 scenario 数据库访问。|
|scenario database files|`NOT_FOUND`|未绑定或打开 fresh-smoke DB 文件。|
|map root|`READY`|`nuplan/dataset/maps` 存在，约 1.3 GiB。|
|official history buffer|`READY`|`Simulation` 向 `PlannerInput` 传入官方 history buffer；仅接口验证。|
|original background replay|`NOT_READY`|代码接口存在，但 DB 与 fresh identity 不可用；未执行 replay。|
|external planner integration|`READY`|`AbstractPlanner`、`PlannerInput`、`PlannerInitialization` 及本地外部 planner 接口可导入；未做闭环执行。|
|traffic-light API|`READY`|`Simulation` 从 scenario 读取灯态并传入 `PlannerInput`；仅接口验证。|
|route-control API|`READY`|`PlannerInitialization.route_roadblock_ids` 存在；仅接口验证。|
|collision/off-road evaluation|`READY`|已有官方 collision 与 drivable-area parquet 提取路径；本阶段未计算指标。|
|scenario token/log identity|`NOT_READY`|未来 roster 按要求未选择，且 DB 不可用。|
|deterministic replay seed|`VERSION_AMBIGUOUS`|未定位到面向下一次 compliant smoke 的独立冻结 seed/replay 合同。|

## 阻断条件

在以下条件全部关闭前，不建议授权新的 48-call official compliant smoke：非空可访问的官方 scenario DB、fresh token/log 绑定、版本化 replay seed 合同、科学负责人对 generator/amendment 的批准。

`trajectory-only core` 的合成单测不能称为官方 nuPlan runtime 验证。本阶段没有运行真实数据测试，原因是任务明确禁止新 smoke，且候选 DB 根为空。

## 复核命令

```bash
PYTHONPATH=../nuplan-devkit:../tuplan_garage:$PWD PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python /Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 -c "import nuplan, hydra, omegaconf, tuplan_garage"
du -sh /Users/liuqing/Projects/01_E2E_QA_Code/nuplan/dataset/data/cache/locked_pool_expanded_v1 /Users/liuqing/Projects/01_E2E_QA_Code/nuplan/dataset/maps
```

通过标准不是“命令能导入”，而是上述 13 项中所有 official execution 必需项均为 `READY`；当前未达到。
