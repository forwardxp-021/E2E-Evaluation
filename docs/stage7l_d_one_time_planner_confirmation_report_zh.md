# Stage7L-D 一次性 Planner-Level Confirmation 报告

## 当前状态

`STAGE7L_D_ONE_TIME_CONFIRMATION_IN_PROGRESS`

本报告对应最终C2冻结协议。Stage7L-D只运行80个冻结场景×5档pure-lateral treatment，共400个official nuPlan closed-loop rollout；在机器gate完成前不读取checkpoint、embedding，也不计算BDD/MMD。

## 冻结 provenance

- protocol SHA256：`f5a8b2df5ed60c0384e8181feceab33f3c6f048780e95aab851184e49247490a`
- confirmation roster SHA256：`90ec9b427636cefc59e6d7ace2507ac8364747e2a38964124be08fdc2a10acf9`
- roster：80 scenarios，15 left，65 right，79 unique logs
- treatment：transition length `60/58.5/57/55.5/54 m`，trigger 12 m，15 s horizon，0.1 s sampling
- background：`closed_loop_nonreactive_agents`

## 执行与门禁

正式数值将在400格official inventory完成后，由`tools/stage7l_extract_confirmation_metrics.py`和`tools/stage7l_evaluate_confirmation_gates.py`从planner-level资产一次性写入。安全口径在结果前固定为全部80场景的scenario-level conservative aggregation；不做post-treatment deletion或replacement。

## Blind boundary

- embedding read：No
- checkpoint inference：No
- BDD/MMD：No
- Stage7L-E executed：No

本文件当前仅记录执行开始状态，不预写pass/fail结论。
