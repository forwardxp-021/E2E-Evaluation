

## RBR-V2-D4：配对零假设与跨来源标定分析（仅D方法开发）

### 1. 命令

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d4_analysis.py real
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d4_analysis.py synthetic
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d4_theory.py
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d4_report.py report
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d4_report.py verify
```

### 2. 期望行为

只读取已冻结D表示与已有Canonical83输入，重现数学、8000组合成和已暴露D的探索性诊断。verify只核对本轮Manifest绑定的直接相关文件SHA。重生成分析后须重新运行report及manifest封存。本轮不调用车辆传播、模型推理、训练、下载或Primary/V/C。

### 3. 通过标准与实际结果

数学穷举、4项统计测试通过；D4代码Python3.9编译及全tools Python3.13编译通过，旧代码在Python3.9的已有语法不兼容另行保留日志。结论D4_PROSPECTIVE_SAP_AMENDMENT_REQUIRED；原D3阈值/结果不变。条件随机化候选仅为修约提案，D3数据全部是E3方法开发，不能重新称独立验证；候选两表示探索性饱和不构成RBR优势。详见docs/stageR/rbr_v2_d4_v1/Scientific_Report_CN_v1.md。
