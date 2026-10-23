# TSB-Coast P1 真实加速度与速度报告 v1 — NOT_RUN

本轮真实闭环完整pair=0，完成arm=0，科学attempt=0。**尚未在nuPlan实际闭环中观察到C2的“制动—近似滑行—再次制动”。** 前置测量gate阻断，不是“没有形成该行为”的实验证据。

| 预留预算slot | SESSION | 实际两次峰值 | 实际near-zero core/漂移 | 实际正恢复 | 真实曲线/trace |
|---|---|---|---|---|---|
| 1 | 未选定 | NOT_RUN | NOT_RUN | NOT_RUN | 不存在 |
| 2 | 未选定 | NOT_RUN | NOT_RUN | NOT_RUN | 不存在 |
| 3 | 未选定 | NOT_RUN | NOT_RUN | NOT_RUN | 不存在 |

没有实际完成的pair，因此不生成PNG/SVG或所谓“真实绘图数据”。解析fixture只用于测量测试，明确标为DETERMINISTIC_MEASUREMENT_FIXTURES_NOT_CLOSED_LOOP，不能将其速度/加速度图冒充车辆响应。没有用旧B1正恢复曲线填充Coast结果。

实际F_match、全窗负a剂量、制动峰平衡、pre/post运动状态、collision、drivable及low-speed均NOT_EVALUATED，不填0、不填PASS。22项纯测试中规范输入可被算法识别，不代表任何闭环样本通过。

若以后有新授权并实际运行，无论通过或失败，都必须按Owner要求保留真实raw/measurement速度与加速度、reference对照、零线、阶段和core标注以及原始trace SHA；本轮没有相应科学数据。
