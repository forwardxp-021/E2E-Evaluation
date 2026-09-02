# R1 B2.9-E Scientific Owner 48-Run Reauthorization Request v0.1

## 请求事项

B2.9-E 已仅修复 post-run callback lifecycle，并将修复后的新版本执行包冻结。现请求 Scientific Owner 判断：是否针对下述新 manifest 授权一次全新的 48-run official scientific smoke。

- final execution manifest SHA256：`99dfafaa719c5b2ea454b46f637132e5d6e9c755c972bf5a15ec37442057c006`
- 当前 `OFFICIAL_SMOKE_AUTHORIZED = false`
- 当前 `NEW_RUN_BUDGET = 0`
- 当前 `RBR_A/B/C = NOT_AUTHORIZED`

## 冻结资产与语义

- roster v3.0 SHA256：`efe8e9d680ca0bcacb367bc9b616610ca78c260195e53b8f025a7bd1d92c23e6`，identity 完全未变。
- schedule v3.1 SHA256：`99f44095c27319b746921376d2549a00186303298b5266ff45dd008a98c08455`；48/48 与 v3.0 scientific semantics 完全相同，仅使用全新 `R1B29E-...` run/pair references。
- pair bindings v2.1 SHA256：`a606a87b01cd1fdd340070fca7e77170b6e0782aafa1e7c19ab6c91228cc9fa6`；24/24 scientific semantics 完全相同，仅机械更新 run/pair references 和 package provenance。
- selector 未调用，source universe 未扫描，未重新 rank，未替换 identity。

## 生命周期修复与验证

- 新 shared lifecycle helper SHA256：`a330a8a9319bb0e395fb07caf6273aa40a22bb190045f693d3c2a2e842e07d2b`。
- 新 executor SHA256：`b7525038388317be7c0d957c0b94e44bee2e3716d089025459b0a2918114e22c`；executor 不直接调用 `SimulationRunner.run()`，只经 shared helper 使用 nuPlan `run_runners(...)`。
- exact-executor engineering canary：HLC 2/2、TSB 2/2 technical complete；4/4 exact Primary80 trace、metric parquet、runner report 与 safety adapter complete；2/2 dispatcher complete；scientific outcome 仅 descriptive。
- 48/48 zero-run construction PASS；`runner.run = 0`、`run_runners = 0`。
- 24/24 frozen pair structural dispatcher PASS。
- callback 与完整 transitive SHA closure：PASS，共 84 个组件。

## 旧 Attempt 隔离

B2.9-D once authorization 已消费，两个旧 official attempts 永久保留为 `ATTEMPT_HISTORY_ONLY`。旧输出未删除、覆盖、append、补 callback、补 parquet或补 scientific evaluation，也未作为新 pair input。新授权若批准，其预算语义是新版本 package 的 `NEW_RUN_BUDGET = 48`，不是复用旧预算的 46。

## Scientific Owner 唯一待决问题

是否对 final manifest SHA256 `99dfafaa719c5b2ea454b46f637132e5d6e9c755c972bf5a15ec37442057c006` 重新授权一次冻结的新版本 48-run official scientific smoke？在收到与该 SHA 精确匹配的显式授权前，executor 保持 fail-closed。
