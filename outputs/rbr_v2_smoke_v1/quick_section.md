

## RBR-V2-SMOKE：真实Waymo工程训练链路（非Primary模型）

### 1. 命令

原Canonical83/模型/正式训练入口不修改。当前版本已有冻结文件后，freeze/materialize/execute拒绝重建或重启；不要删除账本重跑。以下为本轮实际工作流：

```bash
PYTHONPATH=outputs/rbr_v2_uonly_v1/runtime_packages:. /Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_smoke_data.py freeze
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=outputs/rbr_v2_uonly_v1/runtime_packages:. /Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_smoke_data.py materialize --workers 4
# 独立环境，不覆盖nuPlan环境；首次建立时使用：
python -m venv outputs/rbr_v2_smoke_v1/runtime
outputs/rbr_v2_smoke_v1/runtime/bin/python -m pip install --proxy http://127.0.0.1:7890 torch==2.9.1 numpy==2.2.6
outputs/rbr_v2_smoke_v1/runtime/bin/python tools/rbr_v2_smoke_train.py probe
outputs/rbr_v2_smoke_v1/runtime/bin/python tools/rbr_v2_smoke_train.py preflight
outputs/rbr_v2_smoke_v1/runtime/bin/python tools/rbr_v2_smoke_train.py execute
outputs/rbr_v2_smoke_v1/runtime/bin/python tools/rbr_v2_smoke_train.py reload-export
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_smoke_report.py
PYTHONPATH=outputs/rbr_v2_uonly_v1/runtime_packages:. /Users/liuqing/miniconda3/envs/nuplan/bin/python -m unittest tests.test_rbr_v2_smoke -v
```

### 2. 期望行为

固定身份hash抽取2048训练+256诊断窗口，仅来自候选train且排除连接test的观察依赖组；真实原始payload经既有Canonical83重新物化。normalizer只在2048训练窗口拟合。MPS兼容则使用MPS；原GRU128共享z64与双Head、AdamW及loss不变。唯一seed3407、batch128、2epoch、32次实际optimizer.step，最后epoch2保存Smoke checkpoint，新进程重载后导出固定256×64表示。训练子进程墙钟硬限1800秒，单run独立账本。

### 3. 通过标准

必须有实际两epoch/32次更新、两loss到共享encoder的非零有限梯度、两Head实际参数变化、checkpoint重载及有限z64；不能以单元测试代替。工程状态永久为RBR_V2_ENGINEERING_SMOKE_ONLY，checkpoint标记SMOKE_ONLY_NOT_PRIMARY_CHECKPOINT，来源互斥未知与历史training_certified=false保留。诊断loss不选epoch、无best checkpoint。中文总报告：docs/stageR/rbr_v2_smoke_v1/Final_Report_CN_v1.md。
