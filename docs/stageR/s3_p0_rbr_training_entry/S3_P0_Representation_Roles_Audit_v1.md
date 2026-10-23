# S3-P0 Representation Roles Audit v1

现行角色继承S1.1已批准设计，S1R改变身份暴露规则而不削弱H或改Primary。

| 表示 | 真实实现/入口 | 研究角色与边界 |
|---|---|---|
| Raw33 | `tools/interaction_context_features.py::aggregate_interaction_features`，feature_schema列序 | 第一代历史人工行为/交互特征；33维监督/检索基线。包含proxy、dt缺陷，不能整组当V2真实物理标签 |
| ego13 / F0_project | `tools/stage6l_prepare_context_representation_ablation.py::ego_kinematic_features` | 常规人工运动学摘要、历史Secondary；不是四项F_match，也不是所有企业KPI |
| F_match | `r1_closed_loop_benchmark_v2_1.trajectory_descriptors_timestamp_aware`前述四项 | TSB开发/资格匹配：mean_speed、end−start_speed、path_length、mean_abs_accel。不是完整H；匹配通过不证明nuisance全消除 |
| H | `tools/s1_protocol_schema.py::handcrafted_h` | 正式Primary唯一强人工挑战者，30列，DEVELOPMENT_INFORMED_STRONG_CHALLENGER；结构已批准，不能因容易检测TSB而删除 |
| O | frozen Option-A mechanism evaluator：phase count、release fraction、second-brake peak ratio及valid/status | 机制阳性控制，不是H，也不是encoder训练标签；RBR不需要胜O |
| RBR64 | 当前未来共享z64设计，历史old64不能冒充新checkpoint | 学习式共享行为表示，最终Primary直接用z64；训练语义Head预测与真值都不替代z64 |

## H30精确结构复核（只读源码，无新比较）

0–12：原ego13原公式；13：timestamp-aware F的mean_abs_accel；14–21：8个固定[0,1)…[7,8)秒 signed acceleration均值；22–24：lag5/10/20 autocorrelation；25–26：负加速度时间加权质心与spread；27–28：ACF、braking-mass validity；29：max|physical dt−.1| cadence。

ego13以固定dt=.1对valid压缩序列差分，不回改；F用raw speed的physical-time gradient edge_order2；其余时间特征用median3 speed后physical gradient。ACF固定分母，全零/小分母给0并valid=false；制动mass用trapezoidal权重；空bin/非有限按合同invalid，不选择删除相应科学样本。不加入mechanism phase count，不做数据驱动列删除。

H/ z64的BDD preprocessing各自在D_fit baseline拟合mean/SD（常数scale1），相同RBF median-distance数学规则，单个scaler+单个带宽规则，零监督核/投影搜索。数字带宽不同是合理的，不需要相同数值；禁止用原始MMD²大小跨表示比较。

## 三种“13”与评价标签风险

历史ego13、Raw33中的某些语义、V2建议canonical13概念有重合，公式/列序/valid规则并不等同。V2 semantic13具体建议见Semantic文档，只从U物理轨迹产生并通过梯度训练encoder；不是把人工13直接拼入z64，也不追求人工距离几何一致。通用表示head负责U中完整序列结构，不拿O或H输出当监督。

用同一语义定义训练和检索只能评价已知语义保持，不能自证未知增量。正式Primary仍只按独立角色、固定误报和预算比较RBR与强H；H+RBR预测增量是Secondary，不选择encoder，不替代Primary。没有在本轮计算新的H向量、RBR embedding或RBR/H检测结果。
