# S2-PRE Readiness Report

日期：2026-09-17。基线及 Owner 冻结的 S1.1 提交：`9764ea46a91b6495ec9063616c0ec1ce6b1045f3`。

入口文件审计：任务指定的 `handover0917.md` 在当前仓库、当前分支及可见 Git 历史中均不存在。仓库只有 `handover.md`，其 SHA256 为 `f3905893aa2462346083b261507bcb74fdd8bf05b5ed4a4761a757960bd8bfc4`，与冻结 S1 manifest 绑定一致，但内容日期为 2026-09-05，不能冒充 0917 handover。本报告以任务中明确给出的最新 Scientific Owner 状态、冻结 S1 文件及后续授权提交为准；此缺失被公开记录，不进行猜测或静默修正。

**最终状态：S2_PREFLIGHT_BLOCKED_Q20_CAPACITY。** 全量身份分母已枚举并绑定，完整科学资格仍未认证；已知冲突后的容量上界只有 5 个 SESSION，不能提供 20 对，也不能启动 Q12 fallback。Q20/Q12 roster 均未生成。S2 不启动。

## Owner 要求的 10 个明确答案

| 问题 | 答案 |
|---|---|
| 1. frozen source universe candidate 数 | 原始来源 5,386,575 token；冻结早期排除后的 S2-PRE denominator 为 5,338,021 token、1,564 logs、248 sessions |
| 2. metadata-only funnel | 已声明历史结果暴露排除 5,290,386 token / 242 sessions；暴露与保留冲突并集排除 5,301,514 token / 243 sessions；最多剩 36,507 token / 13 logs / 5 sessions，其他资格仍 UNKNOWN |
| 3. 最终 eligible independent units | 完整认证的 eligible units 为 0；真实准确数 UNKNOWN；由已知冲突得到的容量上界为 5，不能把 0 或 5 写成已证明的真实 eligible 数 |
| 4. independence unit 与证据 | `SESSION`；同采集前缀的分段共享数据库 log timestamp、vehicle、date，208/248 sessions 含多个 log chunks；driver ID 不可得，跨 session route dependency 尚未闭合 |
| 5. Q20 capacity | `Q20_CAPACITY_NOT_AVAILABLE`，因为 independent-unit 上界 5 < 20；同样 <12，但不自动切换 Q12 |
| 6. Q20 NOT_RUN roster hash | N/A；容量不足，roster 与 reserve order 均未创建 |
| 7. production execution binding | BLOCKED；组件文件可绑定，但 TSB Q20 executor、resolved config、serializer callback、主结果 manifest 与 active budget ledger 未绑定 |
| 8. full-arm reset | `RESET_CONTRACT=FAIL`，readiness=BLOCKED；关键 mutable states 均 `NOT_PROVEN` |
| 9. precontext identity | BLOCKED；字段分类与未来检查已冻结，真实 resolved lifecycle/capture 尚未绑定 |
| 10. protected CSV | PASS；SHA256=`e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8` |

## 交付内容与缺口

| 审计 | 状态 | 依据 |
|---|---|---|
| 冻结来源全量枚举 | PASS | 5,338,021 token、1,564 log；来源与集合指纹全部匹配 |
| 独立性审计 | SESSION | 248 会话；组内 timestamp/vehicle/date 一致 |
| Q20 容量 | BLOCKED | 243 会话有已知暴露或保留冲突；剩余最多 5 |
| 完整 eligibility | BLOCKED | 正式初始状态、原生 route/replay 和角色证明未闭合；不把 UNKNOWN 计作通过 |
| Production execution binding | BLOCKED | 历史生产入口绑定 HLC roster 和两次工程预算；不存在已绑定的 TSB Q20 executor/config/serializer callback |
| Full arm reset | BLOCKED | fresh object graph、RNG、callbacks、lazy caches、filesystem 尚无绑定的 TSB 工厂证明 |
| Precontext identity | BLOCKED | 字段和未来比较规则已列出；实际配置与捕获边界尚未绑定 |
| 未来预算草案 | CLOSED | future maximum 40；active=false；当前预算与消耗均 0 |
| RBR 原则边界 | PRESERVED | 冻结 S1 文件字节不变；不实现或训练 RBR |

执行组件文件 SHA 已绑定，但“文件存在并有 SHA”不等于生产集成 PASS。原有 `zero_run_observability_preflight` 虽不调用 run，却会构造 Simulation；本阶段没有调用它。没有为了证明 reset 而构造模拟器。

组件 binding status 仅使用 `BOUND / UNKNOWN / AMBIGUOUS / BLOCKED / NOT_FOUND`。`BOUND` 只表示该文件、符号及 SHA 被精确绑定；不会把单个组件的 BOUND 推导为整条 production chain PASS。reset 各项仅使用 `FRESH_CONSTRUCT / PROVEN_RESET / IMMUTABLE_SHARED / NOT_PROVEN / AMBIGUOUS`；当前关键可变状态均为 `NOT_PROVEN`，所以 `RESET_CONTRACT=FAIL`，对外 readiness 仍记为 BLOCKED。

## 执行与重置边界

未来链路仍要求一个 TSB executor 连接官方 lifecycle、冻结 planner/generator、实际 controller/LQR、被动 recorder、canonical serializer、生产 dispatcher、安全产物和一个主结果 manifest。历史 HLC wrapper 只作为审计证据，不作为当前 TSB 执行绑定。

`TwoStageController.reset` 仅清除 `_current_state`；不重建 tracker、motion model 或场景缓存。默认 builder 可接受共享 planner/callback；官方 seed 初始化位于 run_simulation，直接调用 builder 不证明已播种。每 arm 需要新进程及新对象图，或等效的逐状态证明。未证明只读非干扰之前不允许共享 map/cache。完整状态列表见 reset contract。

Precontext contract 区分 DIRECTLY_BINDABLE、HASH_VERIFIABLE、NOT_AVAILABLE。原始转向角不可得；官方构造器硬编码 0.0，不能称为测量值。其他字段包括精确提取起点、EgoState、路由/replay、加速度、可用角速度、控制器配置、seed、时间原点与 divergence 前 0..10 行。哈希相等必要但不足以证明完整 reset。

预算需在 simulator entry 前原子持久化 claim，异常消耗、无退款/重试/替补；首个科学失败停止，基础设施失败为 INCOMPLETE。跨进程 claim 与正式 TSB executor 尚未绑定。草案没有可激活的执行入口，仅编辑布尔值不能授权执行。只允许未来单独 Owner 批准对已有完整产物做离线恢复。

## 冻结科学语义

TSB candidate、参数、机制/F_match/安全 evaluator、S1 SAP、H 与 schema 均保持字节不变。F0_project=ego13、H 为唯一 Primary comparator、O 为机制正控；π=.50 为条件性合成基准。训练原则保持 X→temporal encoder/GRU→shared z64→Human Semantic Head 与 generic representation heads；U-only semantic supervision 反传到 encoder，不使用 TSB/O/Q/D/E 标签、不做 ego13 几何对齐；BDD 直接使用 z64。RBR_TRAINING=NOT_AUTHORIZED。

## 验证

- 定向 pytest：**26 passed**，14 个既有第三方 deprecation warnings。包含 7 项新 preflight tests、S1 schema/生产分析器数值 fixture，以及被动 recorder 对象身份/命令/持久化失败测试。
- 覆盖 SHA 篡改拒绝、closed budget/错键拒绝、重复 token/会话分组、无自动 Q12、输出目录新鲜度与 symlink 拒绝、80/79、错键 fail-closed、mechanism/F_match/safety 原结果语义。
- Fixture 的数值 pass 只是单元测试期望值；未生成科学 qualification、Q 结果或真实 trace。preflight validator 明确返回 scientific_qualification=false、execution_possible=false。
- 新增两个工具及测试文件 `py_compile`：PASS。按仓库要求尝试 `tools/*.py` 整体编译，遇到未修改旧文件 `tools/build_unified_bdd_posttraining_report.py:550` 在当前 Python 3.9 的 f-string 语法错误；该文件与基线一致，不在本阶段修复。
- `tools/check_no_tmp_dependencies.py`：PASS。
- 全量来源 token 数/集合 SHA、每数据库冻结指纹、日志数：PASS。完整压缩账本与索引一致性、最终 manifest 校验另由最终 manifest 记录。

复现/验证命令与通过标准见 QUICK_REFERENCE 的 S2-PRE 段。全量 census 重跑只能写新目录，不覆盖冻结账本。

## 文件与差异范围

相对 S1.1 基线 `9764ea46…`，S2-PRE 包创建 1,564 个 gzip census shards、9 个顶层 preflight artifacts、2 个工具、1 个测试文件，并更新 QUICK_REFERENCE，共 1,577 个文件。完整数据采用现有 v1 convention：`S2_Preflight_Eligibility_Census_v1.json` 索引 + SHA-bound `census/log_*.json.gz` 分片；不再重复生成数 GB 的平铺 CSV。由于 Q20 容量不足，两个 roster 文件按协议不存在。

本轮 Owner 指令复核只修改 package 内的状态词、零运行计数字段、两份报告、manifest、静态验证器及对应测试；未修改 S1、TSB candidate、阈值、历史结果或 census shards。最终 Git diff/commit/remote SHA 在交付回复中报告。

## 零运行与保留项

simulation=0；runner.run=0；new scientific outcome exposure=0；RBR training=0；E access=0；Simulation objects constructed=0；Q→D activation=0。

受保护历史 CSV SHA256 保持 `e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8`。其原有工作区改动及其他历史未跟踪输出不纳入本次提交。原 S1 文件不改写，Owner 冻结声明记录在本阶段文件中。

最终提交 SHA 与远端 SHA 在交付回复中核对报告，manifest 不绑定自身或未来 commit，避免循环引用。STOP；等待 Owner 处理容量和绑定缺口，不能将当前包解释为 S2 授权。
