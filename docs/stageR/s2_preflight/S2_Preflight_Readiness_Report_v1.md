# S2-PRE Readiness Report

日期：2026-09-17。基线及 Owner 冻结的 S1.1 提交：`9764ea46a91b6495ec9063616c0ec1ce6b1045f3`。

**最终状态：S2_PREFLIGHT_BLOCKED_Q20_CAPACITY。** 全量身份分母已枚举并绑定，完整科学资格仍未认证；已知冲突后的容量上界只有 5 个 SESSION，不能提供 20 对，也不能启动 Q12 fallback。Q20/Q12 roster 均未生成。S2 不启动。

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

## 零运行与保留项

simulation=0；runner.run=0；new scientific outcome exposure=0；RBR training=0；E access=0；Simulation objects constructed=0；Q→D activation=0。

受保护历史 CSV SHA256 保持 `e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8`。其原有工作区改动及其他历史未跟踪输出不纳入本次提交。原 S1 文件不改写，Owner 冻结声明记录在本阶段文件中。

最终提交 SHA 与远端 SHA 在交付回复中核对报告，manifest 不绑定自身或未来 commit，避免循环引用。STOP；等待 Owner 处理容量和绑定缺口，不能将当前包解释为 S2 授权。
