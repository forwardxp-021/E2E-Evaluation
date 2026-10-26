

## RBR-V2-D1：冻结模型公平接口与SESSION容量审计

### 1. 命令

在仓库根目录运行。extract拒绝覆盖已物化输入；scan与D_E3检查拒绝重写冻结名单，不能删除历史目录获取新实验身份。

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d1_interface.py extract
outputs/rbr_v2_smoke_v1/runtime/bin/python tools/rbr_v2_d1_interface.py encode
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d1_capacity.py inventory
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d1_capacity.py scan
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d1_d_capacity.py
/Users/liuqing/miniconda3/envs/nuplan/bin/python -m unittest tests.test_rbr_v2_d1_statistics -v
PYTHONPATH=outputs/rbr_v2_uonly_v1/runtime_packages:../nuplan-devkit:. /Users/liuqing/miniconda3/envs/nuplan/bin/python -m unittest tests.test_rbr_v2_uonly.PhysicalTests -v
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d1_synthetic.py
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d1_report.py
```

### 2. 期望行为

仅使用配置事前绑定的SDC三开发SESSION/六个真实历史arm；原Canonical83和Waymo normalizer经唯一epoch30模型生成z64，同80帧计算未改动H30，记录mask和来源。D_fit只拟合三baseline的工程scaler和median带宽。统计模块执行无偏MMD²、同身份release抽样、阈值、SESSION权重bootstrap和原SAP判定；纯合成A/A不代表实际检测性能。目录清点、原规则扩展和旧E3的D-only静态检查不调用模型传播，E3不恢复V/C资格。不能以旧科学执行许可运行新仿真。

### 3. 通过标准

模型/normalizer/H/Canonical83哈希不变；六个arm均同身份80帧、[6,64]/[6,30]有限表示、重复eval一致；7项输入/来源测试（含物理oracle）和8项统计测试通过；原受保护CSV及用户前缀不变。接口通过不等于D阶段Ready：必须另满足足够合法互斥SESSION、真实D功效/覆盖/FPR与角色防火墙。当前完整结论见docs/stageR/rbr_v2_d1_v1/Final_Report_CN_v1.md。本轮无重训、新仿真、真实检测优劣比较或Primary/V/C执行。
