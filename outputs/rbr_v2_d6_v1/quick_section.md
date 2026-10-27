

## RBR-V2-D6：具体任务总体与真实AA可执行性（NO-GO）

### 1. 命令

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d6_static.py
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d6_numeric.py
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d6_finalize.py audit
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d6_finalize.py verify
```

### 2. 期望行为

只读原冻结静态来源及初始帧SQLite/地图，不访问C raw或V/C科学结果；用标量解析式与已有D轨迹作算子核对，不进行车辆传播/模型推理。四单元q*不允许空单元重新归一化。verify核对直接相关来源元数据、模型及本轮文件SHA。重生成结果后需package及manifest封存；候选合同没有科学runner入口。

### 3. 通过标准与实际结果

6项fixture/解析测试通过，D6_UNPAIRED_TASK_BDD_NOT_FEASIBLE_UNDER_CURRENT_RESOURCES。224原静态适格源仅40落入q、严格未来候选29，两个单元零支持，现有同版本q轨迹仅6源；源域CI/power与新数值尺度均未认证。不得启动新D、240源计划、Primary/V/C或下载。见docs/stageR/rbr_v2_d6_v1/Scientific_Report_CN_v1.md及D6_Method_Contract_PROPOSED_v1.md。
