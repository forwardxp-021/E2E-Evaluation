# P1R 官方模型零请求响应 v1

证据范围：**隔离模型测试，非nuPlan正式闭环实验**。基线、源码SHA256和版本见 `P1R_Manifest_v1.json`；官方devkit HEAD `e9241677997dd86bfc0bcd44817ab04fe631405b`，Python3.9.19、numpy1.26.4。官方模型与参数未改。

## 官方状态方程

KinematicBicycleModel `_update_commands` 对纵向请求u使用：

`a_next = a_before + dt/(dt+tau) × (u−a_before)`，`tau=0.2s`。

`propagate_state` 使用更新后的a推进 `vx_next=vx_before+dt×a_next`；位置用推进前速度及当前heading积分。零请求不等于重置实际加速度。在dt=.1时，u=0给出 `a_next=(2/3) a_before`。模型没有气动阻力、滚阻、坡度、发动机制动或回收扭矩，本结论严格是 `ABSTRACT_COAST_PROXY_ONLY`。

单步测试从v=10m/s、a=−1m/s²进入零请求，输出a=−2/3、v=9.933333333m/s，t增加100000微秒，位置前进；输入原对象a仍为−1。非零转向率请求得到有限且非零的实际转向响应。

## 多步真实模型证据

`tools/tsb_coast_p1r_model_fixture.py` 构建真实官方TwoStageController、LQR、既有适配器和KinematicBicycleModel，以**纯合成直线状态、单位幅值参考**传播79次，得到80状态。没有SESSION、观察者、runner或正式C2 generator。测试起点已有a=−1仅用于验证残余状态保留，不能将图中0秒衰减解读为C2第一次制动。

输出：`evidence/isolated_model/Official_Model_Fixture_v1.json`、`official_model_transitions.jsonl`。

| 当前控制时刻s | 模式 | 请求m/s² | 输入状态a | 输出状态a | 输出速度m/s |
|---:|---|---:|---:|---:|---:|
| 2.0 | BRAKING | −0.954545 | −0.930016 | −0.938193 | 9.033093 |
| 2.1 | COAST | 0 | −0.938193 | −0.625462 | 8.970547 |
| 2.2 | COAST | 0 | −0.625462 | −0.416974 | 8.928850 |
| 3.8 | COAST | 0 | −0.000952 | −0.000635 | 8.845582 |
| 3.9 | BRAKING | −0.954545 | −0.000635 | −0.318605 | 8.813721 |

18个coast请求均为精确零且到达真实model；model连续推进；coast期间longitudinal LQR和stopping solver调用为0；3.9秒恢复原版LQR。两次独立新对象运行结果逐值相同，仅支持该合成用例的确定性，不证明完整真实runner的确定性。

图：`P1R_Isolated_Model_Response_v1.png`。请求、模型状态加速度、raw central speed gradient分别绘制，后者不是model状态a。后向速度差与model输出a一致，而central gradient混合相邻区间；没有为通过机制改变F的导数。

## 横向预测的剩余简化

coast横向LQR使用当前恒速预测，实际速度还会因残余a下降。若入口加速度a0且之后零请求，H步速度预测误差为 `dt×|a0|×r×(1−r^H)/(1−r)`，r=2/3；10步约0.1965×|a0|m/s，极限0.2×|a0|m/s。该解析量不是新的通过阈值。

横向控制继续工作不等于真实道路安全已通过。新记录器保留转向请求；未来首对必须检查官方collision/drivable-area和实际响应，失败即停止，不通过改gain补救。本轮未引入新动力学预测器或扩大工程。

## 通过与不能声称的事项

单步、79步、边界进出、状态对象不被清零、转向有限、实际命令到model、重复/缺步拒绝、shadow一致性及故障注入测试通过。原adapter边界跨步拒绝测试保留。此为接口/模型能力证据，**不是C2真实双制动、F_match、安全或完整episode路由PASS**；当前真实日志时钟另有明确阻断。
