

## RBR-V2-D3：端点弧长兼容修复与有界D验证

### 1. 命令

以下命令使用独立D3空间，禁止删除历史目录、复活D2账本或重跑已领取科学身份。

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d3_diagnostic.py --namespace <新的零运行诊断名> --patched
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d3_compatibility.py
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d3_execute.py --prepare-engineering
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d3_preflight.py ENGINEERING
# 提交精确工程协议、源码、输入SHA和预检后执行；最多12个ENGINEERING arm。
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d3_execute.py --execute ENGINEERING
# 真实工程通过且科学兼容身份明确后，单独冻结SCIENCE协议和D40名单再运行：
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d3_preflight.py SCIENCE
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d3_execute.py --execute SCIENCE
```

### 2. 期望行为

只对当前车道中心线端点截断、与同一原生拼接参考线弧长不一致的情况修复坐标绑定；不更换道路、LQR、车辆模型或纵向命令。历史正确参考逐位保持，受影响的旧D来源另做工程验证，未经兼容认证不得混入正式D统计。工程12arm与科学80首次arm分别入账，科学CPU4小时、单arm20分钟、科学重试0。D2失败永久保留；V/C、模型训练、额外下载均不运行。

### 3. 通过标准

根因原帧8.3218569rad/s可确定性复现；修复参考首步前进；工程全79次传播、80状态、官方安全、原机制/F_match及观测映射完整。科学阶段同一冻结版本、完整D40分母与所有失败保留。RBR/H30只能用相同80帧和冻结模型，m20、alpha、pi和SAP不改变；不支持的FPR、功效、覆盖或增益结论如实NOT_ESTABLISHED。
