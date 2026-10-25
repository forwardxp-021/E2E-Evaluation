

## RBR-V2 U-only canonical83 v1（来源阻断，零训练交付）

### 1. 命令

在仓库根目录执行。原nuplan环境不覆盖，protobuf单独安装；已有target可跳过安装。

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python -m pip install --target outputs/rbr_v2_uonly_v1/runtime_packages --no-deps protobuf==4.25.8
PYTHONPATH=outputs/rbr_v2_uonly_v1/runtime_packages:.:../nuplan-devkit /Users/liuqing/miniconda3/envs/nuplan/bin/python -m unittest tests.test_rbr_v2_uonly -v
# 以下为完整可重建的只读审计。使用全新目录，禁止覆盖本轮账本：
PYTHONPATH=outputs/rbr_v2_uonly_v1/runtime_packages:. /Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_u_data.py census --out outputs/rbr_v2_uonly_reproduction
PYTHONPATH=outputs/rbr_v2_uonly_v1/runtime_packages:. /Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_u_data.py quality --out outputs/rbr_v2_uonly_reproduction
PYTHONPATH=outputs/rbr_v2_uonly_v1/runtime_packages:. /Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_u_data.py materialize --out outputs/rbr_v2_uonly_reproduction --limit 16
# 对既有16份真实输入重新核验，不覆盖原始数组：
PYTHONPATH=outputs/rbr_v2_uonly_v1/runtime_packages:. /Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_u_data.py verify
# 打包本轮既有数据与最终报告，不训练、不运行仿真：
PYTHONPATH=outputs/rbr_v2_uonly_v1/runtime_packages:.:../nuplan-devkit /Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_finalize.py
# 预期退出码2：来源未认证，正式训练必须拒绝。
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_train.py gate --manifest outputs/rbr_v2_uonly_v1/data_manifest.json --config configs/rbr_v2_uonly_v1/seed3407.json
```

### 2. 期望行为

原始51份Waymo文件及旧窗口身份只读；新版本先建立移动世界轨迹依赖组、再按固定哈希划分，并审查全部168700窗口的真实自车有效性。16个固定窗口演示新83维处理器，不更换无效样本。物理oracle经Waymo和官方nuPlan合成对象两接口验证；不调用仿真。模型两个Head分别验证编码器梯度，不做optimizer step。完整中文结论见docs/stageR/rbr_v2_uonly_v1/Final_Report_CN_v1.md。

### 3. 通过标准

10项测试通过；完整源文件hash匹配；观察依赖组跨新split为0，但不得据此认证原始片段独立。raw segment、全局对象与绝对时间映射缺失时，入口输出TRAINING_ENTRY_BLOCKED且不生成seed3407训练目录或checkpoint。实际结果B：来源阻断；训练0、Primary0、下游比较0。不能把data_manifest的false手改为true。
