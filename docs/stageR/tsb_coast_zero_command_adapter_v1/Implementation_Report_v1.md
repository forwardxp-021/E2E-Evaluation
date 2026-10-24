# 零纵向请求滑行适配器 v1 — 已实现，未运行闭环

日期：2026-10-24（Asia/Shanghai）。状态：`ZERO_LONGITUDINAL_REQUEST_ADAPTER_IMPLEMENTED_SYNTHETIC_TESTS_PASS`。

本次依据用户“停用纵向跟踪，向运动模型明确传入零纵向请求，继续推进状态，可以实现吗”的新请求，实现独立工程模块。它不改官方控制器/模型，不是恢复P1条件授权，也不将C0代理能力判定升级为真实自由滑行证据。

## 实现内容

`tools/tsb_coast_zero_command_tracker_v1.py::ZeroLongitudinalRequestTrackerV1`实现官方AbstractTracker接口，包装一个原版LQRTracker。

- 滑行模式：不调用原`track_trajectory`、纵向LQR或停车P控制器；忽略未来参考速度对纵向控制的影响，返回严格0.0的纵向加速度请求。
- 转向保留：继续提取当前横向误差和路线曲率，调用原横向LQR。其纵向预测使用当前速度的恒速近似，不沿用未屏蔽的制动/加速预测；它尚未验证包含实际模型滤波尾巴的横向闭环误差。
- 非滑行模式：直接委托原LQR，保留其返回对象本身，滑行结束恢复原控制。
- 模式按显式episode整数微秒锚点与半开区间[start,end)判断，不按未来预瞄时间判断。时间不一致、负相对时间、重叠/非法区间、跨边界而未对齐的控制步均拒绝。
- 不重置初始速度/加速度，不跳过车辆更新。原TwoStageController可以接收该tracker，继续把返回请求送给原运动模型；其滤波与物理简化保持原样。
- `last_command_evidence`只描述该次返回的请求包、模式和物理时间。它不是轮端扭矩观测，也不是模型滤波后的实际加速度。

没有内置科学SESSION、默认滑行时长、候选搜索或新制动幅值。区间和episode锚点由将来经授权的绑定显式传入。本轮测试用历史C2区间只是纯接口fixture，不是新的执行配置或名单。

## 未来接入方式（示意，不是执行命令）

```python
tracker = ZeroLongitudinalRequestTrackerV1(
    tracker=existing_lqr,
    episode_start_time_us=explicit_episode_anchor,
    coast_intervals_us=approved_intervals,
)
controller = TwoStageController(
    scenario=existing_scenario,
    tracker=tracker,
    motion_model=existing_official_motion_model,
)
```

这只是依赖注入方式。本轮未将适配器注册到旧Hydra配置、production factory或P1执行入口。原recorder对LQR类名/绑定有检查，不能直接把它当已经兼容新adapter；若未来接入科学生产，需独立版本记录与授权检查，不能修改旧检查绕过。P1仍`TSB_COAST_P1_PREFLIGHT_BLOCKED`，M1误差问题未解除，P2/训练/Primary/V/C均未授权。

## 实际验证

21项测试通过。覆盖：高/低未来目标速度都不会触发纵向求解；低速也不启用停车控制；转向请求可非零；起止边界和恢复；输入状态不被重置；时钟、步长跨边界、形状和非有限值拒绝；异常不遗留上一包成功记录。

两类集成fixture明确分开：

1. 调用真实官方TwoStageController，使用**运动模型测试替身**。证明其确实把零纵向请求交给`propagate_state`接口，并采用该接口返回的下一时刻状态；未调用官方运动模型传播。
2. 构造纯合成直线poses，调用真实LQR的横向误差/曲率/横向求解辅助函数；纵向求解函数设为禁止调用。证明该adapter可用真实nuPlan数据类型和数学辅助接口工作，不是只测Mock返回值。该fixture不是nuPlan场景或闭环rollout。

Python编译、临时依赖检查和diff检查通过。测试存在原依赖的15条弃用警告（pyparsing和官方横向LQR的NumPy array→scalar转换），不影响本次21项结果，未为此修改历史依赖。

实际科学仿真、runner、官方motion_model传播、训练/推理、H/BDD/MMD、Primary/V/C均0；未新增SESSION名单或真实车辆曲线。没有声称“真实闭环已形成滑行”。真实数据最小检查仅为冻结历史数据/源码hash核对，未做被禁止的科学smoke。

## 复现命令与限制

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python /Users/liuqing/miniconda3/envs/nuplan/bin/python -m pytest -q tests/test_tsb_coast_zero_command_tracker_v1.py
python -m py_compile tools/tsb_coast_zero_command_tracker_v1.py tests/test_tsb_coast_zero_command_tracker_v1.py
python tools/check_no_tmp_dependencies.py
```

当前模型无滚阻/风阻/坡度力，因此本模块实现的是“零纵向请求+官方残余滤波”的抽象滑行，不是零踏板或零轮端力的真实动力总成证明。恒速横向预测与实际残余纵向状态的闭环影响尚未验证；这不通过额外试跑在本轮解决。

旧B1、P0/P1/C0及S1/S1R保持只读；保护基线、代码/依赖hash及验证记录见本目录Manifest。工程模块实现完成后停止，不启动科学执行。
