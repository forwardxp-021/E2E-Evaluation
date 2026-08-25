# R0 Local Codex Verification Handoff v0.2

## Mission

只读核验本地 `E2E-Evaluation` 的真实资产与实现合同。当前 StageR active branch 应为 `20260825_stageR_new`，为 R0 v1.0 freeze 提供事实。**不得训练、不得仿真、不得修改历史代码/输出、不得清理仓库。**

最终只需要回传两个核心机器文件：

```text
r0_local_contract_verification.json
r0_local_asset_inventory.csv
```

并保存命令日志。

## Hard prohibitions

```text
NO git reset --hard
NO git clean
NO bulk delete outputs
NO checkpoint retraining
NO simulation
NO regeneration/overwrite of historical outputs
NO blind git add/commit
```

## Step 0 — Repo identity and preservation

在 repo 根目录只读执行并记录输出：

```bash
git branch --show-current
git rev-parse HEAD
git remote -v
git status --short
git rev-list --left-right --count origin/20260825_stageR_new...HEAD
git merge-base HEAD 20260611_stage7_conclusion
```


Branch handling rule：

- 预期 active branch：`20260825_stageR_new`；
- 若当前不是该 branch：记录 `BRANCH_MISMATCH` 并停止任何会写文件的后续操作，不要自动 checkout/switch；
- 若本地 HEAD 与远端不同：记录 ahead/behind 与 SHA，不自动 pull/rebase/reset；
- Generation-1 historical branch `20260611_stage7_conclusion` 仅作为追溯 reference。

若以下文件存在，先记录：

```bash
sha256sum outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv
```

不要修改该文件。

## Step 1 — Enumerate A/B/C checkpoint assets

搜索而不移动/重命名文件：

```bash
find . -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' \) -print
```

重点确认 A/B/C seed `3407/3408/3409`。对每个真实 checkpoint 记录：

- absolute/relative path；
- size；
- mtime；
- SHA256；
- seed；
- candidate；
- associated config/log if identifiable。

**若3408/3409不存在，记录 MISSING；不得重新训练补齐。**

## Step 2 — Read checkpoint metadata safely

只读取 metadata/state_dict key/shape，不运行训练。记录：

- encoder class/signature；
- input_dim；
- hidden_dim；
- latent_dim；
- GRU layers/directions；
- projection heads；
- saved scaler/config refs；
- state_dict shape signature。

若 checkpoint 反序列化存在不可信 pickle 风险，优先使用项目原生 loader 或 PyTorch 支持的安全/weights-only方式；不要执行来源不明代码。

## Step 3 — Locate actual training contract

定位 A/B/C 的 training CLI/config/log/split manifest，确认每个 seed：

```text
actual input T
actual input D
normalization/scaler
losses/weights
sampling
train/val/test split
checkpoint selection criterion
encoder forward/pooling
```

不要从 handover 推断，优先实际 config/log/tensor。

## Step 4 — Inspect Waymo tensors

对真正用于 A/B/C 的 `.npy/.npz` 等资产只读记录：

```text
path
sha256
shape
dtype
finite rate
min/max (sampled if huge)
valid-mask channels/statistics
number of examples
```

关键问题：实际训练输入究竟 `[N,80,83]`、`[N,150,83]`，还是其他。

## Step 5 — Inspect Stage7L input and inference chain

定位 Stage7L 实际 context array / manifest / inference invocation，记录：

```text
exact tensor shape
dtype
feature schema
normalization
T actually fed into B/C
mask/padding behavior
encoder forward()
last-hidden vs other pooling
embedding output shape
```

必须区分“builder输出150帧”与“模型实际消费150帧”。

## Step 6 — Trace ego13 exact implementation

定位生成 Stage6/Stage7 ego13 的源代码和 schema，输出 ordered 13-feature list、公式/单位、聚合窗口与 SHA。

不要用研究讨论中的近似列表替代正式 schema。

## Step 7 — Trace BDD/MMD exact implementation

定位历史正式 BDD 路径并记录：

```text
biased/unbiased MMD²
kernel family
bandwidth rule and fit data
paired vs unpaired mode
null construction
permutation/randomization unit
number of permutations
seed
q95 / p-value calculation
```

同时定位 Stage6P 与 Stage7L 实际 CLI/config/result metadata。

## Step 8 — Data-role classification

把发现的 dataset/manifest 分类为：

```text
R0_DEVELOPMENT
R0_AUDIT_HOLDOUT_CANDIDATE
FUTURE_R4_RESERVED_POOL_CANDIDATE
HISTORICAL_DEVELOPMENT_ONLY
```

注意：Waymo historical test、Stage6/Stage7/Stage7L 已经解盲，不能标为 untouched confirmation。

此步骤只基于 token/log/scenario identity 与历史使用情况，不看任何新模型 outcome。

## Step 9 — Required outputs

从提供的模板生成：

```text
r0_local_contract_verification.json
r0_local_asset_inventory.csv
r0_local_command_ledger.jsonl
```

每条事实附：source path、SHA/commit、verification command 或 provenance。未知项必须写：

```text
UNKNOWN / NOT_FOUND / AMBIGUOUS
```

禁止猜测。

## Stop condition

完成事实核验后停止。不要启动 R0 statistical audits，不要训练 RBR-A/B/C。把输出文件交回 ChatGPT Work / 本项目对话，用于合并到 v1.0 Operational Freeze。


## Output location

本轮只读核验结果建议写到：

```text
outputs/stageR/r0_local_audit/
```

至少输出：`r0_local_contract_verification.json`、`r0_local_asset_inventory.csv`、`r0_local_command_ledger.jsonl`。不要覆盖 `docs/stageR/r0/manifests/` 中的预冻结 master 文件；由 Work review 后再合并。
