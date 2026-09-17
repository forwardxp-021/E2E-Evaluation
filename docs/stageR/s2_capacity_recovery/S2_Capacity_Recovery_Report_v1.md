# Stage S2 Capacity Recovery — Fresh Independent Source Discovery

日期：2026-09-17（Asia/Shanghai）

分支：`20260825_stageR_new`

审计基线 HEAD：`0fb928de457c4fec88beb07dfd2982ce7a995dc9`

最终容量状态：**CAPACITY_RECOVERY_NONE**

S2 状态：**S2_PREFLIGHT_BLOCKED_Q20_CAPACITY**

## 结论

本轮完成了 metadata-only、zero-simulation 的独立来源发现。没有发现冻结 `mini/train_pittsburgh` 全集之外、已物化且能够证明来源与独立性的 nuPlan 原始会话。三个看似额外的本地池均为符号链接子集；Git 可达历史只有现有 Waymo LFS 指针，没有 nuPlan DB/归档；下载暂存区没有数据；可见挂载点没有外部数据集。历史 `outputs/` 仅按目录名登记并受 outcome firewall 保护，未打开其中的科学结果。

因此新增容量三层均为 **0**：Level A raw fresh sessions = 0；Level B metadata-ready sessions = 0；Level C prospectively eligible upper bound sessions = 0。既有冻结全集在暴露/保留冲突后仍只有 **5 sessions / 13 logs / 36,507 tokens** 的资格上界，且其正式初始速度、official native route/reference、forward support 与 mapping 仍未完整认证。合并新增来源后，Q20 上界仍为 5，Q12 上界仍为 5，二者均不可用。

`CAPACITY_RECOVERY_NONE` 表示本轮没有找到可提出 amendment 的新池。它不把 UNKNOWN 写成失败，也不声称现有 5 个会话科学合格。没有生成 Q/E roster、替补顺序或 `PROPOSED_SOURCE_UNIVERSE_AMENDMENT`。

## 搜索覆盖与证据

| 范围 | 结果 |
|---|---|
| 本地 nuPlan raw cache | 1,624 DB 路径；沿用 S2-PRE 去重结果 1,621 logs / 248 sessions；全部属于当前冻结全集 |
| 本地 alias pools | `locked_pool_expanded_v1`=1621、`locked_pool_v1`=64、`mini_non_pittsburgh_v1`=61；全部指向冻结 raw cache |
| 未物化 split/download 路径 | split 目录无 payload；download staging 仅有空日志，未发起下载 |
| Git refs / LFS | 106 个 named refs；可达 raw 路径只有现有 Waymo TFRecord，非冻结 nuPlan Q replay 来源 |
| 本地归档 | 排除 result trees 后检索 zip/tar/tar.gz/tgz/7z/rar/db.gz；发现 6 个候选 payload |
| 外接/挂载路径 | 可见卷仅 `Macintosh HD`；无外部数据集挂载 |
| 历史输出树 | 仅枚举目录名；内容 `BLOCKED_BY_OUTCOME_FIREWALL`，且派生结果不构成 fresh raw source |

完整逐来源证据见 `S2_Fresh_Source_Inventory_v1.csv`。目录证据 SHA 绑定路径名、类型、大小、mtime 与符号链接目标；它不是 DB 内容哈希，也没有读取轨迹。Git 检索覆盖本地已存在的 heads、remotes 与 tags，不声称覆盖未获取的远端对象或当前不可见的物理存储。

## 三层容量与独立性

| 容量层 | 新来源 | 合并既有冻结剩余池 | 判定 |
|---|---:|---:|---|
| A：raw fresh sessions | 0 | 既有 5 不属于新来源 | 无容量恢复 |
| B：metadata-ready | 0 | 0 已完整认证；5 仍有 route/reference/forward/mapping UNKNOWN | 不可进入 roster |
| C：prospectively eligible upper bound | 0 | ≤5 | Q20/Q12 均不足 |

独立单位保持 **SESSION**。解析器继续使用 `YYYY.MM.DD.hh.mm.ss_veh-XX` 采集前缀；没有引入新命名格式，也没有把分段 LOG 重定义成独立单位。车辆 ID 不等同于驾驶员 ID，跨会话更高层依赖仍可能进一步降低容量。

`S2_Fresh_Session_Registry_v1.csv` 只有表头，因为没有任何来源通过 Level A；空表是零发现的显式结果，不是漏写。历史冲突表列出 243 个冻结 conflict sessions。其 stage 字段保守记为 frozen aggregate，因为现有冻结账本不提供逐 source-to-session stage join；为避免重开历史科学结果，本轮不猜测更细 stage。聚合暴露/保留分类本身为高置信度。

容量情景中的新来源 `N=0`：预留 Q20 后 `max(N-20, 0)=0`，容量缺口为 20；预留 Q12 后 `max(N-12, 0)=0`，容量缺口为 12。由于两个 Q 情景都无法组成，0 只表示没有剩余 potential future E reserve，不是已定义的 E sample size。

## 15 个明确答案

| # | 问题 | 答案 |
|---:|---|---|
| 1 | 当前 frozen universe 外是否发现新的 source reservoir？ | **否。** 本地候选均为冻结源别名、空目录、非 nuPlan 原始数据或结果树。 |
| 2 | 一共发现多少新的 raw candidate sessions？ | **0。** |
| 3 | 其中多少 session 可以证明未历史 exposure？ | **0。** 没有新的 session identity 可进入 freshness 证明。 |
| 4 | 多少 session freshness 为 UNKNOWN？ | **0 个已识别 session。** 若干未物化/不可见 source 的 provenance 为 UNKNOWN，但它们没有可计数身份，不能换算成 session。 |
| 5 | 多少 session metadata 足够支持 prospective filtering？ | **0。** Level B metadata-ready pool 为空。 |
| 6 | 按冻结规则 eligible upper bound 是多少？ | 新来源 Level C 为 **0**；与既有冻结剩余池合并仍为 **≤5 sessions**。`prospectively eligible != scientifically qualified TSB pair`。 |
| 7 | 是否存在 Q12 capacity？ | **否。** 合并上界 5 < 12；`Q12_CAPACITY_NOT_AVAILABLE`。 |
| 8 | 是否存在 Q20 capacity？ | **否。** 合并上界 5 < 20；`Q20_CAPACITY_NOT_AVAILABLE`。 |
| 9 | 如果预留 20 个 session 给 Q，未来还剩多少 fresh SESSION？ | 新来源 `N=0`，所以可剩 **0**，同时存在 20 个 session 的 Q 容量缺口；Q20 实际不可预留。 |
| 10 | 是否存在显著 potential future E reserve？ | **否。** Q20 本身不可组成；E 数量尚未定义、E 未构造且未访问。 |
| 11 | 是否需要 source universe amendment？ | **本轮不需要。** 没有找到可提案的新池。 |
| 12 | amendment 是否仅为 proposal？ | 本轮没有生成 amendment；若未来发现新池，只能 `PROPOSED_ONLY` 且必须 Owner 批准。 |
| 13 | 有哪些 UNKNOWN / AMBIGUOUS / BLOCKED provenance？ | 未物化 split/download、未获取远端与当前不可见存储为 UNKNOWN；outputs 为 `BLOCKED_BY_OUTCOME_FIREWALL`；没有把这些来源计成 session。 |
| 14 | 是否读取了任何 scientific outcome？ | **否。** 只读取冻结身份账本与路径元数据；历史结果 payload 未打开。 |
| 15 | simulation / runner / training / E access 是否全部保持 0？ | **是。** 六项硬计数全部为 0。 |

## 硬边界与计数

```text
SIMULATION_RUNS=0
RUNNER_RUN_CALLS=0
NEW_SCIENTIFIC_OUTCOME_EXPOSURE=0
RBR_TRAINING=0
E_CONSTRUCTION=0
E_ACCESS=0
```

本轮没有执行 S2、Q20/Q12、reset/precontext 修复或 RBR；也没有修改 S1/S1.1 冻结文件。`handover0917.md` 现已由用户提供并作为本轮权威入口保存；它不会追溯改写既有 S2-PRE manifest 中当时“未找到该文件”的历史记录。

Protected CSV 保持 SHA256=`e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8`。最终唯一状态为 `CAPACITY_RECOVERY_NONE`，并单独保持 `S2_STATUS = S2_PREFLIGHT_BLOCKED_Q20_CAPACITY`。
