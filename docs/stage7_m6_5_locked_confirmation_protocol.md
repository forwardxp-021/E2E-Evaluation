# Stage 7 M6.5：310-pair locked confirmation protocol

Issue: [#240](https://github.com/forwardxp-021/E2E-Evaluation/issues/240)

## 目标与冻结边界

M6.5 使用 M6.4B/C/D 已锁定并成功完成的 310 个 planner pairs，执行 M6.1
总体 paired BDD、M6.2 pre-treatment task family、M6.1 质量敏感性，以及既定
interaction / trajectory representation controls。确认集相对 45-pair development
set 必须同时满足 scenario token 与 log name 零重叠。

冻结边界位于任何确认 embedding 或效应统计被读取之前。分析锁位于：

`outputs/stage7_m6_5_locked_analysis_freeze_v1/m6_5_confirmation_analysis_lock.json`

锁内固定 310 pairs / 620 rows、五个 task、两个 planner fingerprints、100000 次
within-pair swaps、plus-one p、exact pooled median bandwidth、Holm families、质量
阈值、checkpoint 与所有入口工具 SHA256。M6.1/M6.2 frozen tools 均未修改。

## 固定样本构成

| source | complete pairs |
| --- | ---: |
| M6.4B locked primary | 283 |
| M6.4C quoted-token primary recovery | 2 |
| M6.4C frozen technical reserves | 20 |
| M6.4D pre-frozen high-motion supplement | 5 |
| total | 310 |

Task 构成为 following 60、lane change 60、stop/go 67、high motion 60、
dense/vulnerable interaction 63。质量只能定义预先固定的 sensitivity subsets，不能
删除或替换 full-primary 场景。

## 统一 Stage7C 视图

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/stage7_m6_5_prepare_locked_confirmation.py \
  --output_dir outputs/stage7_m6_5_locked_confirmation_view_v1
```

工具逐场景重跑 frozen Stage7C audit，验证 planner metadata 一致性，生成 310 行
ledger。149-step rollout 只在末尾补零到 150，并把新增 mask 设为 false。small arrays
合并为 `[310,2,150,8]`；official run trees 不复制，每个 global scenario 使用只读
symlink 指向原始 `scenario_0`。

## Stage5D context

Mac 环境必须显式加入本地 tuPlan Garage；否则 nuPlan pickle 反序列化会因
`ModuleNotFoundError: tuplan_garage` 失败并退化为空 history。

```bash
env PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/tuplan_garage \
  caffeinate -dimsu \
  /Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/build_nuplan_5neighbor_context_dataset.py \
  --sim_dir outputs/stage7_m6_5_locked_confirmation_view_v1 \
  --output_dir outputs/stage7_m6_5_locked_confirmation_context_v1 \
  --assignment_mode lane_aware_with_geometric_fallback \
  --nuplan_map_root /Users/liuqing/Projects/01_E2E_QA_Code/nuplan/dataset/maps \
  --nuplan_db_root /Users/liuqing/Projects/01_E2E_QA_Code/nuplan/dataset/data/cache/locked_pool_expanded_v1 \
  --write_projection_debug \
  --write_strict_filter_diagnostic \
  --strict_filter_min_laneaware_ratio 0.8 \
  --strict_filter_ratio_sweep 1.0 0.9 0.8 0.7 0.6
```

第一次未设置 `PYTHONPATH` 的退化输出未用于统计，完整保存在
`outputs/stage7_m6_5_invalid_preflight_missing_tuplan_import/` 作为失败审计证据。

## 下游与锁定检验

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/stage7e_embed_stage6_dataset.py \
  --context_dataset_dir outputs/stage7_m6_5_locked_confirmation_context_v1 \
  --checkpoint outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2/best_model.pt \
  --output_dir outputs/stage7_m6_5_locked_confirmation_embeddings_v1 \
  --device cpu

/Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/stage7f_aggressive_conservative_paired_delta.py \
  --embedding_dir outputs/stage7_m6_5_locked_confirmation_embeddings_v1 \
  --context_dataset_dir outputs/stage7_m6_5_locked_confirmation_context_v1 \
  --stage7f_dir outputs/stage7_m6_5_locked_confirmation_stage7f_v1 \
  --planner_a pdm_closed_assertive_v1 \
  --planner_b pdm_closed_conservative_v1 \
  --output_dir outputs/stage7_m6_5_locked_confirmation_stage7f_v1/paired_delta_assertive_minus_conservative

/Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/stage7_m6_5_run_locked_confirmation.py run \
  --lock_manifest outputs/stage7_m6_5_locked_analysis_freeze_v1/m6_5_confirmation_analysis_lock.json \
  --embedding_path outputs/stage7_m6_5_locked_confirmation_embeddings_v1/embedding.npy \
  --metadata_csv outputs/stage7_m6_5_locked_confirmation_embeddings_v1/metadata.csv \
  --paired_delta_csv outputs/stage7_m6_5_locked_confirmation_stage7f_v1/paired_delta_assertive_minus_conservative/paired_delta_by_scenario.csv \
  --row_quality_csv outputs/stage7_m6_5_locked_confirmation_quality_v1/row_quality_tiers.csv \
  --pair_quality_csv outputs/stage7_m6_5_locked_confirmation_quality_v1/paired_quality_gate.csv \
  --interaction_representation_path outputs/stage7_m6_5_locked_confirmation_representations_v1/interaction_features.npy \
  --trajectory_representation_path outputs/stage7_m6_5_locked_confirmation_representations_v1/trajectory_summary.npy \
  --output_dir outputs/stage7_m6_5_locked_confirmation_analysis_v1
```

## 结果

310/310 pairs 通过 pair audit，620 rows embedding 全部 finite，development overlap
为0，planner fingerprints 与冻结 treatment 完全一致，M6.3 总体和逐 task power
target validation 通过。

总体原始 64D embedding primary：MMD²=`0.0044693963`，100000 个 null swaps 中
0个达到 observed，plus-one p=`9.9999e-6`。因此在冻结 alpha=0.05 下拒绝“两个
planner 在该 paired confirmation population 上具有相同 embedding distribution”的
null。Residual secondary 同样为 p=`9.9999e-6`。

| pre-treatment task | pairs | MMD² | raw p | Holm p |
| --- | ---: | ---: | ---: | ---: |
| following interaction | 60 | 0.024780 | 0.00006 | 0.00030 |
| lane change | 60 | 0.028784 | 0.00009 | 0.00036 |
| stop/go control | 67 | 0.005230 | 0.01820 | 0.01820 |
| high-motion dynamics | 60 | 0.014453 | 0.00014 | 0.00042 |
| dense/vulnerable interaction | 63 | 0.013792 | 0.00129 | 0.00258 |

五个 learned-embedding tasks 全部通过 Holm 0.05。Interaction 与 trajectory controls
在五个 task 上的 raw p 也均小于0.05，但其协议角色为 mechanism controls，不纳入
learned-embedding confirmatory Holm family，也不替代 primary。

质量分层为 Tier A=58、Tier B=77、Tier C=175。原始 embedding 的 Tier A 与
Tier A+B sensitivities 均通过预定 Holm（adjusted p 均为`0.0182`）。Residual Tier
A+B 通过 Holm（`0.0008`），但 Tier A residual 不显著（raw/Holm p=`0.126249`）。
另外，全局 fallback rate=`10.59%`，超过旧 M2B scale-readiness 的5%门；fallback
与 pair embedding distance 明显相关（max-pair fallback Spearman rho=`0.5088`，
Holm p=`2.45e-21`；pair fallback delta rho=`0.5704`，Holm p=`1.52e-27`）。

因此结果支持“冻结的 assertive/conservative planner treatment 在新 log/scenario-
disjoint population 上产生了可检测、跨五类 task 的行为 representation 分布差异”。
它不证明安全性或某个 planner 更优；质量相关性和 Tier A residual 不显著要求把
lane-context fallback 作为明确限制，而不能把全部 effect 解释为纯 planner mechanism。
