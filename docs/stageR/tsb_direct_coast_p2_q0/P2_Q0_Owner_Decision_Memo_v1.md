# P2-Q0 — Scientific Owner执行审查备忘录 v1

## A. Overall Verdict

**P2_Q0_READY_FOR_OWNER_EXECUTION_REVIEW**。

此状态只表示固定20名单与40arm零运行准备齐全，不是P2_QUALIFICATION_PASS，也没有激活科学执行。Authorization始终PROPOSED_ONLY、scientific_owner_approved=false，Q0入口没有科学launcher；即使手工改成AUTHORIZED，Q0函数仍拒绝执行。未来须独立Owner最终批准，并把新的20对调度/账本治理绑定到已冻结F2科学链。

## B. 合法独立样本容量

248SESSION普查中214为原静态适格；应用预先冻结初速≥5、旧B1全部20/Coast DEV01/02/03排除、E3/E4/UNKNOWN排除和E5保护后，剩168个候选。按固定SHA256(salt|SESSION)排名冻结前20，样本足够。完整248行筛选、168名排序及原20名单全部保存；没根据预检或任何科学结果换人。

20个不同SESSION、20个LOG、20个database log token、20个实际完整DB SHA均唯一，实际DB log表与原始源元数据一致。同SESSION两臂完整初态、地图、route、precontext与观察映射一致，非不同token冒充独立SESSION。

## C. F2科学合同冻结

原ScriptedLongitudinalRequestTrackerV1、官方model、横向LQR、请求值/整数时序、80状态/79推进、测量与安全全部逐值/逐文件SHA保持。Baseline −1.45@[2.1,3.9)，Treatment −1.305@[1.1,2.1)及[3.9,4.9)，其余0。未新增F3、参数优化、model或controller改动；四F、额外剂量/逐峰balance、低速及停止规则不变。

## D. 40arm生产零运行预检

原名单40/40 PASS，0 BLOCKED；每arm使用原F2工厂构建fresh官方完整对象图，初始有限route、tracker/model、clock、实际命令recorder、16项官方指标、serializer/callback/finalizer装配及fresh namespace检查通过。guard所有科学入口调用0，history/recorder0，没有机制、F_match或官方安全结果。不同SESSION时钟偏差各自核查，最大绝对1443us，原same-index局限完整披露。

在第一次40arm生产预检之前，新Q0 metadata入口曾混淆历史metadata fingerprint与file SHA；初次20行错误阻断记录完整保留，正确域核对另存Verified Arm Specs。只修新准备工具，没有改F2或原20名单；原记录没有被删除或当作科学失败。实际生产预检只执行一次，无替补。

## E. E0–E5与LOG-disjoint

所选16 E1、4 E2，E3/E4/未知交集0；与旧B1、Coast DEV01/02/03及C候选交集0。全部5个E5及其LOG来源保护给Primary C，不新建/占用最终C名单。S1R后续正式批准优先于早期PROPOSED草案；不恢复历史绝对未使用容量规则。

这只是B Benchmark资格角色。零运行访问不洗去E1/E2，未来科学开始后按真实启动记录E2暴露，不能冒称E5或自动Claim-A未见SESSION泛化。SESSION独立单位不等于driver/城市完全独立。

## F. 待批预算、成功与停止

固定20对、40首次arm，冻结pair顺序，各Baseline通过后Treatment。建议CPU≤16小时/57600秒，每arm墙钟≤20分钟/1200秒。最多42累计attempt只是待批天花板，当前技术retry授权0；任何技术故障停，单独Owner裁定，不能自动用余下2次。

首次真实科学失败立即停，不调参、不放宽阈值、不换人、不继续找成功者；保存所有启动attempt、结果、未运行成员和完整分母。只有原20对全完成且机制/匹配/额外balance/安全/完整性全部满足批准合同才可P2_QUALIFICATION_PASS；20/20不是总体100%概率或Primary功效证明。预算与成功/停止规则必须在第一次runner之前经独立Owner最终批准。本轮没创建可claim的P2账本，旧C2/F2停止账本原样保护。

## G. 对未来Primary的边界

P2不训练/选择encoder，不做H30/RBR/BDD/MMD，不执行V/C，不修改Primary统计合同或成功规则。不得根据P2结果调整后续Primary样本或标准，失败/成功子集不能拿去重新筛未来cohort。P2的20对不是每个release arm20独立log的Primary容量/功效认证，D内部隔离、calibration、V/C容量和PRIMARY_FREEZE_POINT仍须单独闭合。

## H. 测试、Git与历史资产

7项测试覆盖hash排序及输入排列不变、伪科学结果不影响选择、E3/E5/未知及source alias排除、重复SESSION拒绝、20SESSION/LOG唯一与原排名、旧/提案授权拒绝、原始metadata指纹与full-file SHA域区分、零科学调用源码检查。原6项和修正后7项日志均保留，F2不变；另用真实B1/B1-R2/C2/F2及本轮提案，逐一拒绝全部40 P2 run IDs。

源码、数据、协议、名单、预检、zero-run配置及历史保护SHA见Manifest/Final Verification；只有本轮零运行入口、测试与文档提交，用户已有未提交修改与历史资产保留。选择规则先提交，再提交20名单，然后在首次生产预检前提交准备入口修正；准确最终commit和push远端核对在交付回复中报告，避免自引用。

## I. 唯一建议下一阶段

**Scientific Owner终审固定20名单、F2科学/来源绑定、40arm预检、预算与停止规则 → 决定是否给予独立P2科学执行授权。**

本轮已停止。当前不激活提案、不领取attempt，不启动任何科学仿真、训练、检测比较或V/C。
