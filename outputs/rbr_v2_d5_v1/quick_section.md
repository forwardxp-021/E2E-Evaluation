

## RBR-V2-D5：核心主张与统计路线决策（未激活修约）

### 1. 命令

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d5_evidence.py
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d5_package.py verify
```

### 2. 期望行为

读取已暴露D的原数组及冻结计划，重现任务混合/条件null合成反例和三种物理单位诊断；不重算旧D报警、不运行车辆模型或神经网络。verify只核验本轮及直接相关历史来源SHA。重新生成D5输出须运行package、manifest后再verify封存，不得把PROPOSED_ONLY草案交正式runner执行。

### 3. 通过标准与实际结论

3项性质测试和真实数组只读诊断通过。D5_SCIENTIFIC_ROUTE_RECOMMENDED_PENDING_OWNER_APPROVAL：优先任务标准化的不配对版本比较及外部AA校准；D4仅为辅助受控检验，原H30尺度风险未消除；不追加240源，不启动Primary。当前双制动RBR正增量确认投入缺少依据，不能把双方100%报警当优势。完整决策及唯一SAP草案见docs/stageR/rbr_v2_d5_v1/。
