

## BDD-P0：真实正常驾驶日志、多场景A/A与构成变化诊断

### 1. 命令

在项目根目录运行只读交付核验（不重新提取、不重新拟合或计算报警）：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/bdd_p0_delivery.py verify
/Users/liuqing/miniconda3/envs/nuplan/bin/python -m pytest -q tests/test_bdd_p0_statistics.py tests/test_bdd_p0_physical.py
```

已执行的生产入口如下，记录用于复现溯源。当前固定产物不应覆盖；复现须在独立副本保留同一协议和完整来源，不能重新选窗口/来源获得有利结果：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/bdd_p0_data.py roster
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/bdd_p0_data.py extract
outputs/rbr_v2_smoke_v1/runtime/bin/python tools/bdd_p0_analysis.py encode
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/bdd_p0_analysis.py statistics
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/bdd_p0_diagnostics.py
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/bdd_p0_delivery.py audit
outputs/rbr_v2_smoke_v1/runtime/bin/python tools/bdd_p0_delivery.py reload
```

### 2. 期望行为

读取既有nuPlan原数据库/地图，按事先固定的来源及vehicle-date依赖块生成80×83正常驾驶窗口。使用冻结epoch30、Waymo normalizer和完整H30，外部CAL与开发留出互斥；同一释放两侧40个SESSION不同，两表示使用相同身份。新CAL与留出本轮均登记E3，不能再当同方法的独立确认样本。只读verify检查交付SHA和历史模型/合同绑定，不下载、训练、传播车辆或执行Primary/V/C。

### 3. 通过标准与实际结果

- 实际190/190窗口、190×64 RBR与190×30 H30；拟合57、外部校准68、开发留出65个不同采集块。
- 6项物理/统计测试通过；独立进程重载16个真实输入，z64最大差0。
- RAW A/A报警RBR6.16%、H30 2.84%；活动匹配A/A为3.94%、5.60%；人为活动构成变化为33.10%、25.87%。这些是重叠释放的有限池诊断，不是来源总体FPR或Primary优势。
- 完整拓扑配额因CAL仅9个JUNCTION、要求至少10个而NOT_ASSESSABLE；来源bootstrap支持不足保留，来源级CI/覆盖率/功效NOT_ESTABLISHED。
- 结论为BDD_P0_REAL_DATA_RESEARCH_ENTRY_ESTABLISHED_WITH_LIMITATIONS。真实E2E策略版本比较NOT_ESTABLISHED；历史SDC正控制和D6 NO-GO不变。
- 详细结果：`docs/stageR/bdd_p0_v1/Final_Report_CN_v1.md`；来源：`outputs/bdd_p0_v1/Source_Manifest.json`；完整性：`docs/stageR/bdd_p0_v1/Manifest_v1.json`。
