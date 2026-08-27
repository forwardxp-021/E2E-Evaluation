# R1 官方 nuPlan DB 就绪性报告 v0.1

DB 层状态：`READY`。整体 official replay 状态仍为 `VERSION_AMBIGUOUS / NOT_READY`，两者不得混同。

## 只读盘点结果

|项目|结果|
|---|---|
|cache root|`/Users/liuqing/Projects/01_E2E_QA_Code/nuplan/dataset/data/cache`|
|分区|`mini` 64 个 DB；`train_pittsburgh` 1,560 个 DB|
|DB 总数 / 可读数|1,624 / 1,624|
|总大小|70,077,571,072 bytes|
|scenario_tag 行数|9,727,375|
|去重 scenario token|5,386,575|
|去重 log|1,621|
|token-set SHA|`66a0d95ef424afd379ec174059d4831beff56fb0cf2938edc1541aeaea4f54c7`|
|source-root fingerprint|`5b53ad42497fe6926c73936970658a3717d1c2cc51077812d5284a57fd242489`|
|inventory CSV SHA|`e4acb94fc66888ce514464ad0597993e120150a5a5a8fb85dc08d40bb5928643`|

每个 DB 均以 SQLite read-only URI 打开，记录绝对路径、大小、mtime、schema SHA、DB fingerprint、scenario/tag 数、token-set SHA、log/location/map version 与状态。所有 DB 均非零、可读，且其 map version 可在绑定 map root 中解析；token/log identity collision 为 0。

cache root 下没有单独的非空 validation 分区；三个 locked-pool / mini-non-Pittsburgh 候选目录均为 0 B，因此未把它们伪记为可用 DB。当前绑定明确只覆盖既有 `mini` 与 `train_pittsburgh`。

## 地图绑定

map root 为 `/Users/liuqing/Projects/01_E2E_QA_Code/nuplan/dataset/maps`，8 个文件共 1,426,603,899 bytes，root fingerprint 为 `a85e17eba18e5fdd65148705844b8f189bb4d4373a1d82805e1f8ffd4ae8afb3`。可用 map version：Singapore One North、Boston、Las Vegas Strip、Pittsburgh Hazelwood。

## 相对 B0 的修正

B0 只检查了空的 locked-pool 候选目录，因此当时的 `database_access = NOT_READY` 对那个候选路径成立。本次按任务要求扩展到既有授权项目路径，定位到 `mini` 与 `train_pittsburgh` 的完整缓存。这是 inventory evidence 更新，不改写 B0 历史记录，也不代表 closed-loop replay 已运行或已证明确定性。

本阶段没有下载数据、运行 planner、生成 roster 或读取 representation/BDD/probe/RBR。
