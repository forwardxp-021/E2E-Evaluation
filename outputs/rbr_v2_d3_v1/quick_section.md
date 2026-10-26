

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


### 4. 实际收口与只读核验

D3已完成工程12arm、科学80个首次arm和40对资格检查，全部配对通过。第13对Treatment原监护暂停记录永久保留；其既有完整产物经只读核验恢复，没有重跑，随后仅运行原名单剩余54首次身份。`SCIENCE_analysis_view.json`是链接账本的只读分析聚合，不能用作重置执行预算的账本。

真实D paired-swap结果：RBR AA=1、AB=0；H30 AA=0.0212、AB=0.2462。原始报警差−0.2462不构成有效FPR控制下的检测优劣证明。H30未饱和；RBR跨源标定失败，Primary/源域CI/功效/覆盖未认证。不重训、不改核或阈值规则、不下载新源。

只读检查当前封存交付：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d3_verify.py
```

归档的离线处理顺序（当前输出已封存，不删除后重做；复算应在独立副本保留原产物）：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d3_analysis_view.py view
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d3_interface.py extract
PYTHONPATH="$PWD/outputs/rbr_v2_d3_v1/inference_dependency_overlay" outputs/rbr_v2_smoke_v1/runtime/bin/python tools/rbr_v2_d3_interface.py encode
# 必须用链接账本的只读聚合入口，不能将原SCIENCE暂停账本当作完整40对结果。
PYTHONPATH="$PWD/outputs/rbr_v2_d3_v1/inference_dependency_overlay" outputs/rbr_v2_smoke_v1/runtime/bin/python tools/rbr_v2_d3_analysis_view.py statistics
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d3_calibration_audit.py
```

推理覆盖层只含本机原有requests、urllib3、idna、certifi、charset_normalizer纯Python文件，源码/版本SHA见对应overlay manifest。没有网络安装或修改原环境；不替换NumPy、Torch、checkpoint、normalizer。当前工程与科学arm额度已用完，旧账本均停止，上述执行命令仅记录历史入口，不授权重跑。
