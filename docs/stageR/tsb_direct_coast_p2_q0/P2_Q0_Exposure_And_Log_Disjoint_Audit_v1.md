# P2-Q0 — Exposure与LOG独立性审计 v1

结论：原排名20个SESSION、20个raw LOG、20个database log token与20个实际完整数据库file SHA均唯一；同一SESSION两arm身份/地图/route/precontext/原始观察映射相同。actual log表只读查询与冻结元数据逐值一致，逐行来源见evidence/P2_Q0_Actual_LOG_And_Exposure_Audit_v1.csv。独立单位为SESSION，不把不同scenario token当新SESSION，也不声称独立driver/城市或所有交通相关性已消除。

248SESSION治理框由已批准S1R提供，E3=31/E4=0/UNKNOWN=0；后续B1 R2 exposure rollforward后E0=1/E1=146/E2=65/E5=5。本次完整静态eligible为214，应用初速≥5与预先冻结排除条件后168；固定hash前20为16 E1_UNRELATED_HISTORICAL_USE、4 E2_BENCHMARK_ENGINEERING。没有通过试跑、机制、F_match、安全或H/RBR结果决定候选。

20个旧B1资格SESSION及全部Coast DEV01/02/03严格排除，F2与科学C2 DEV01因同一SESSION被排除。与E3/E4交集0、未知身份/暴露来源交集0；旧B1、DEV及被保护C候选的交集均0。完整248行筛选及168名排序全保留，原20名单先于metadata/40arm预检冻结，未替换任何SESSION/token。

所有5个E5 SESSION及其raw LOG fingerprint保守保护给Primary C；S2R批准角色记录尚未物化最终V/C名单，这里不假称新建/完成了最终C冻结。原R0 reserved pool处于NOT_FROZEN状态，不把历史提案误称当前已批准C名单。此举额外保护全部低暴露C候选，所选20均无C角色冲突。

本轮仅记录E0性质的metadata/装配访问事件，operative标签仍保留16 E1+4 E2；不改写成E5。未来若实际开始P2资格，按真实已启动暴露追加E2_BENCHMARK_QUALIFICATION，未运行成员保留原历史标签及B名单预留，不能用“未跑/失败/成功”筛后续科学样本。P2是B benchmark验证，不是自动V/C、D detector比较或Claim-A unseen-SESSION泛化证据。

原始指纹的域须区分：frozen roster的source_database_sha256字段是S2结构化metadata fingerprint，包含relative_path/size/mtime/schema/token-set散列，不是完整file SHA。初次新增Q0入口错误混用两个域的记录完整保留；修正只涉及新零运行准备工具，不改F2/名单。原历史指纹按原定义核对，actual完整file SHA另列；Verified Arm Specs绑定43项DB/shard/map来源。不能把初次工具错误报作数据库损坏或F2科学失败。

40个arm正式生产零运行预检一次完成，40 PASS/0 BLOCKED，guard调用总数0、history/recorder均0。各SESSION clock偏差逐一记录，本次全集范围-1014～+1443us，最大绝对1443us；不能统一套用DEV01的330us。raw same-index/token不插值、不换帧，DISCLOSED_OFFLINE_SAME_INDEX_REPLAY_NOT_STRICT_EVENT_TIME_CAUSALITY。未读取未来机制/匹配/安全结果，初始有限route通过也不保证完整episode成功。

暴露判断仅针对有来源支持的当前S1R claim policy；不把历史科学使用等同未见/从未曝光。未来发现影响当前Primary方法的E3/E4信息须按原合同处理，不用本名单准入覆盖新证据。Primary的U-only、H30、统计、独立calibration/power/role容量和freeze规则不变。
