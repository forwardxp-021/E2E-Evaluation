# P1R-D 首对资格结论 v1

`P1R_D_FIRST_PAIR_SCIENTIFIC_FAIL`。

Baseline技术/arm科学/实际观察审计通过后，才启动Treatment一次。Treatment官方生命周期完整、安全通过，但冻结机制reason为TWO_BRAKE_PHASES_NOT_REALIZED。两次真实减速候选的峰值和剂量不满足冻结下限，合格段数0。

依据本轮Owner第十节，只有两个arm均满足arm级条件才进入pair联合判断；此先决条件不满足。因此：

- Pair F_match：`NOT_EVALUATED_ARM_GATE_FAILED`。
- 负加速度剂量差及逐峰平衡：`NOT_EVALUATED_ARM_GATE_FAILED`。
- 不调用pair analyzer，不计算配对差值或伪报F_MATCH_PASS/F_MATCH_FAIL。
- CSV仅列实际单臂F描述符和既有caliper，差值留空；不能据此宣称匹配成功。
- 两臂官方安全各自PASS；安全不抵消机制失败。

实际闭环确认了“减速—零纵向请求并继续推进—再次减速”的简化代理现象，但尚未实现冻结Development合同要求的合格双制动基准。本结果只属于B-development，不是P2资格或Primary干预。

失败后已停，无DEV02/03、retry、样本替换、参数调整、放宽门槛、导数切换、H/RBR/BDD计算。旧B1 20/20 PASS不变；LOW_ORDER_NUISANCE_ELIMINATED和TSB_CLEAN_RESIDUAL_TASK仍NOT_ESTABLISHED。
