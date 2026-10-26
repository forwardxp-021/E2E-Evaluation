

## RBR-V2-PROD：完整Canonical83与固定epoch30正式U-only候选

### 1. 命令

在仓库根目录运行。既有plan/preflight拒绝重写；produce自动核验并复用已认证分片；execute仅允许同一身份的基础设施恢复，数值失败和预算停止不能重启。

```bash
PYTHONPATH=outputs/rbr_v2_uonly_v1/runtime_packages:. /Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_prod_data.py plan
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=outputs/rbr_v2_uonly_v1/runtime_packages:. /Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_prod_data.py produce --workers 6
outputs/rbr_v2_smoke_v1/runtime/bin/python tools/rbr_v2_prod_train.py preflight
# 首个optimizer step前，必须先提交执行冻结、完整样本清单、normalizer、源码及配置。
outputs/rbr_v2_smoke_v1/runtime/bin/python tools/rbr_v2_prod_train.py execute
# 本轮已启动以下结束依赖流程：只等待训练正常完成后重载/出报告，不启动训练。勿与手工export/report并行重复运行。
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_prod_postprocess.py
# 两个独立进程核验最终epoch30模型：
outputs/rbr_v2_smoke_v1/runtime/bin/python tools/rbr_v2_prod_train.py export
outputs/rbr_v2_smoke_v1/runtime/bin/python tools/rbr_v2_prod_train.py export
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_prod_report.py
PYTHONPATH=outputs/rbr_v2_uonly_v1/runtime_packages:. /Users/liuqing/miniconda3/envs/nuplan/bin/python -m unittest tests.test_rbr_v2_prod_features -v
```

### 2. 期望行为

全部candidate_train合法窗口按原观察依赖图隔离candidate/legacy test关联组，真实重建Canonical83+mask及Semantic13。按Scenario共享投影、按源文件提交可恢复mmap分片；不每epoch重解码，不使用旧错误83维。唯一normalizer仅fit全部训练窗口。模型复用GRU128共享z64与双Head，从随机初始化seed3407开始，batch128、AdamW lr0.001/weight_decay0.0001、双loss1:1、clip1。

无认证独立U_val，事前选择FIXED_EPOCH_30_NO_U_VAL_CHECKPOINT_SELECTION；256个in-sample诊断窗口不选择checkpoint。每epoch首批额外检查两loss共享梯度，普通批次一次总loss backward；遮挡/标准化及梯度等价性必须先通过。完整optimizer/RNG/epoch顺序/next offset保存，累计训练设备预算86400秒，持续外推若超预算停止。

### 3. 通过标准

完整实际训练分母与冻结清单一致、30个完整epoch、真实参数更新、epoch30唯一交付模型、全部数据/model/normalizer/source hash一致，独立进程真实输入导出[N,64]有限表示且重复一致。附加U_SOURCE_COMPLETE_INDEPENDENCE_NOT_ESTABLISHED；不宣称独立U验证或检测优势，不运行下游H30/BDD/Primary/V/C或新仿真。完整中文结果见docs/stageR/rbr_v2_prod_v1/Final_Report_CN_v1.md。
