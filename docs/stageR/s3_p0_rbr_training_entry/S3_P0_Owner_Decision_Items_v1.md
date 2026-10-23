# S3-P0 Owner Decision Items v1

以下是尚未批准且必须由Scientific Owner决定的新增事项。本文件为可审查建议，不请求重选已批准Primary或让Owner补做工程事实判断。

| ID | 待决定事项 | 建议及可审查依据 | 决定前状态 |
|---|---|---|---|
| O1 | 新版本canonical83与适配变更的科学合同 | 接受Input83 proposal的center/full velocity、正确轴顺序、physical-time derivatives、明确absolute neighbor yaw、radial closing/TTC/THW及mask/normalizer；确认新增插值/valid支持规则。需版本化而非覆盖旧数组；若拒绝，应指定修订方向，不能直接宣称现状同义 | BLOCKED_INPUT_CONTRACT |
| O2 | 未来U来源/依赖分组/去重及旧证据影响处置 | 授权最小metadata修复：恢复上游segment来源，按依赖图完整分组后冻结新U；至少处理已证实train/test重复ego。旧35分片缺失导致历史影响未认证；接受如实limitations，任何重评需单独授权 | BLOCKED_U_DATA_INDEPENDENCE |
| O3 | 具体训练协议预注册 | 审查单一GRU→shared z64、canonical13语义回归+masked temporal reconstruction、λ=1/1、mask/selection/early stopping及随机流。原则已批准，这里只批准尚未冻结的层、目标、超参数和规则 | BLOCKED_TRAINING_PROTOCOL |
| O4 | 训练资源与一次性预算、授权边界 | 修复验收后决定固定Primary seed3407及3408/3409重复性运行、每run30epoch/总24设备小时的建议。先绑定新U计数、源码/config/runtime/hash，超预算停止；真正训练仍需独立明确授权manifest | NOT_AUTHORIZED |

v2.4工作区草案同步是行政待办，已记录hash与`ROADMAP_V2_4_REPOSITORY_SYNC_PENDING`，不作为O1–O4替代，不把文件存在当已批准。

**不重新提出的既有决定**：BDD唯一Primary、ΔBDD、α=.05、m20、π.50、δ*.10、H30、U-only selection、SESSION inference、E3/E4排除、unpaired preferred/预注册paired fallback、完整分母与无survivor、PRIMARY_FREEZE_POINT。这些已批准。

最终V/C样本量、calibration/AA/AB内部身份、coverage/power和执行预算是未来Primary冻结材料，不在本轮据估算要求Owner“选一档就开跑”。先走原SAP决策流程；本轮不选择成员、不申请额外benchmark或平台开发。

本轮结束后停止。只建议针对O1/O2/O3/O4阻断的最小修复任务，完成前不训练、不新增仿真。
