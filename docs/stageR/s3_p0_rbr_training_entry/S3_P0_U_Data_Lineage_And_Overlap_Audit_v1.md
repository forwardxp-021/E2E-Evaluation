# S3-P0 U Data Lineage and Overlap Audit v1

结论：`S3_P0_BLOCKED_U_DATA_INDEPENDENCE`；`RAW_SEGMENT_DISJOINTNESS_NOT_ESTABLISHED`。已确认至少一组跨train/test重复的**自车物理轨迹**，不是仅凭Scenario ID或相同列名猜测泄漏。没有改划分、删除数据或重新训练。

## 审计对象与覆盖

未来U尚未正式指定。本轮审计两个历史候选资产，不把当前候选自动升级为RBR-V2训练集。

| 资产 | 元数据/覆盖 | 结论 |
|---|---|---|
| 第一代static full51 | shard_manifest声明35分片、164,871窗口；old64 summary为train131,998 / val16,481，余test16,392 | 本机35分片的meta均缺失；配置、summary、标准化文件和checkpoint存在。不能重新认证历史窗口级独立性 |
| Dynamic v2 full51 | 36分片全部读取meta/split；168,700窗口；全部ego_seq逐行hash；每片前2行context/neighbor独立布局抽样 | train135,046 / val16,870 / test16,784；retained Scenario24,548（train19,616 / val2,486 / test2,446） |
| 原始Waymo | 51 TFRecord，23,245,316,267 bytes，全文件SHA256重新计算；24,872唯一Scenario，全部原始record身份/时间索引 | retained24,548全部可定位原始文件/record offset；324个原始Scenario未产生保留窗口，不是漏计训练Scenario |

根manifest的`scenario_count=24872`是raw processed数，不能误作retained数。原始文件均包含不同split的Scenario；TFRecord容器不是独立驾驶片段，不应把“文件互斥”当充分或必要的真实源片段互斥。

## 实际构建顺序

`build_waymo_5neighbor_context_dataset.py::split_of_sid`以MD5(Scenario ID)前8 hex / 0xFFFFFFFF分成<.8 train、<.9 val、其余test。`process_one_scenario`先确定该Scenario的split，再逐vehicle/stride窗口sanitize、静止过滤、邻车赋槽、写出meta。Dynamic v2继承同一split函数，无全局raw-segment grouping或跨Scenario轨迹去重；后续merge/全局标准化不重新独立分组。

原始Scenario有91帧，生产窗口80、stride20，因此这些原始记录通常只产生start=0窗口；“stride20”不意味着本批每个actor必有多个窗口。meta含scenario_id、scenario内target_agent_id、start、window_len、split、赋槽质量；缺raw_segment_id、全局对象ID、原始绝对采集时间。通过本轮新索引已补充**只读可追溯关联**到原文件与record offset，未回写旧meta。原始时间是相对秒（起点0），不能比较两个Scenario的绝对时间区间。

## 已运行的独立性检查

- Scenario→split多值数=0；同一Scenario/actor/start/length重复identity窗口=0；同一Scenario/actor相交窗口跨split=0。这些只是Scenario内保证。
- 全部168,700个ego_seq的精确byte hash发现1种跨split重复自车序列；未进行全库模糊/错位/部分重叠搜索，不能据此声称只有一处风险。
- 回查两个原始TFRecord payload，按官方Scenario/Track/ObjectState字段读取世界x/y、vx/vy、heading、valid：80帧全部有效，六个字段逐元素完全相同，路径长度各19.9864276946m；原始91帧时间数组hash也相同。两个窗口的完整83维context并不完全相同，因邻车上下文可能不同。结论限定为**重复ego轨迹跨train/test**，不把它扩写成所有输入/标签完全重复或test分数虚高某个数值。

| split | Scenario | actor（仅Scenario内ID） | 原始文件编号 | record offset |
|---|---|---|---|---|
| test | 249bd8c0ba9102c7 | 1252 | 00030-of-01000 | 94749254 |
| train | fd4ef9f02c34f649 | 3925 | 00048-of-01000 | 70657735 |

相同local ego序列SHA256：`1f498b4eb2afa2ca928fd35d0d2dcfa4a02bf13c30edd714627ada14efeb3b8a`。原始坐标首点均[-3216.7745203292043, -763.011210230745]；详见`evidence/U_Local_Duplicate_Raw_Check_v1.json`。本轮没有读两个样本的BDD/RBR/H结果。

原始proto的Track ID不提供跨Scenario全局车辆身份，Scenario schema不含足够的raw segment来源映射。官方字段依据已存入`evidence/waymo_scenario.proto`并hash绑定；未安装TensorFlow/Waymo包，也未借缺依赖跳过原始元数据核查。最小wire reader仅解析身份/时间与上述重复候选，拒绝截断/未知wire type；TFRecord CRC未单独验证，完整字节由SHA绑定。

## 历史影响边界

Dynamic v2训练及使用其test数据的历史泛化评价存在已证实的ego重复风险，相关test结果不能直接当完全独立的泛化证据；当前未复算任何历史分数。发现于train/test；没有由此证明train/val已有同样的精确重复，也没有排除其部分重叠。

第一代static同源51文件，故具有同类来源风险；其缺失分片meta使本轮无法确定这两个actor是否被第一代过滤，不能直接宣称旧Clean V2具体样本已泄漏或给出受影响比例。保留历史结果及其原合同范围，加注独立性未认证，不作事后删除/重评。

## 最小补救方案（未执行）

1. 恢复upstream Scenario→原始采集segment/session/全局时间映射（数据持有者），恢复旧35分片只为历史追溯，禁止覆盖旧模型/数据。若确实不可获得，上报剩余不确定性，不能以Scenario互斥关闭G3。
2. 针对未来U生成**新版本**lineage index：原文件hash+record offset+Scenario+actor+时间范围+world轨迹fingerprint；同一源片段、共享车辆时间交集、精确及高重叠world轨迹形成依赖图连通分量。局部序列hash可作为候选检索，必须回到world coordinates与物理时间复核；不能仅因匀速动作相似合并无关车辆。
3. 先建立依赖分组、去重政策和固定seed，再按整个组分train/val/test，最后从组内切窗和过滤、只用train拟合所有normalizer/semantic统计。窗口跨split overlap审计应包括错位时间及近似重叠，不仅byte equality。冻结原始来源、组映射、分母、排除理由与新版本SHA；不把重复组中的test成员挪到val以保留好结果。
4. 若原segment缺失，执行保守的多actor世界轨迹/时间重叠恢复并量化剩余覆盖；不足以证明组独立时继续BLOCKED，或由Owner另行决定科学主张范围，不虚构PASS。TFR文件级split本身无法补救已发现跨文件重复。
5. 新split与canonical83适配必须另立最小修复授权，零训练回归后再复核G2/G3/G4/G6/G11。本轮未自动重分、重导出、去重或训练。

可复现入口：`tools/s3_p0_training_entry_audit.py lineage`。逐分片meta/split/ego/neighbor/context SHA及原始51文件SHA、24,872行record索引见`S3_P0_U_Data_Split_Manifest_v1.json`与`evidence/U_Raw_Record_Index_v1.csv`。没有拼接大数组。
