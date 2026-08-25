# Stage 7 M6.6 confirmation evidence protocol

## 1. Purpose and analysis boundary

M6.6 converts the completed M6.5 locked confirmation into a reproducible paper-evidence
package. It does not alter the confirmation population, planner treatment, embedding,
bandwidth, permutation count, correction family, or any M6.5 result. The full 310-pair
original-embedding endpoint remains the sole overall confirmatory endpoint.

Lane assignment, fallback, and ambiguity are realized after rollout. All M6.6 associations
between those measures and embedding distance are therefore labelled
`descriptive_exploratory_post_treatment`. They are not covariate adjustment, mediation,
causal attribution, or replacement endpoints.

## 2. Locked inputs and fail-closed checks

The builder reads:

- `outputs/stage7_m6_5_locked_analysis_freeze_v1/m6_5_confirmation_analysis_lock.json`;
- `outputs/stage7_m6_5_locked_confirmation_analysis_v1/` summary, pair audit, quality table,
  task summary, and task BDD table;
- `outputs/stage7_m6_5_locked_confirmation_quality_v1/milestone2b_summary.json`;
- `outputs/stage7_m6_5_locked_confirmation_embeddings_v1/metadata.csv`;
- the frozen M6.5 paired-delta table.

Before writing output it recomputes every hash recorded by the M6.5 analysis lock and result
summary, then requires 310 unique pairs, 620 unique embedding rows, equal valid horizons,
finite pair embeddings, zero development token/log overlap, passing sample targets, the
frozen 58/135 quality counts, and the completed five-task Holm result. Any mismatch fails
closed. The output directory must not already exist.

## 3. Evidence generated

The package preserves, without recomputation, the overall primary and five learned-embedding
task results. It also exports:

- locked sample, treatment, task BDD, quality sensitivity, and mechanism-control tables;
- descriptive paired kinematic mean deltas with fixed-seed 10,000-replicate bootstrap CIs;
- quality-tier distance and task-by-tier composition tables;
- Spearman associations between four post-treatment lane-quality measures and within-pair
  embedding L2 distance, shown overall, within each frozen pre-treatment task, and after
  residualizing ranks on the five task indicators;
- stratified, fixed-seed 10,000-replicate bootstrap CIs for overall and task-adjusted
  associations;
- six publication-oriented figures in PNG and PDF, a concise report, bilingual manuscript
  text, machine-readable summary, and full input/tool provenance.

Raw exploratory p-values are reported for transparency. They are not included in either the
frozen five-task Holm family or the frozen quality-sensitivity family.

## 4. Reproduction command

```bash
MPLCONFIGDIR=/tmp/mpl-m6-6 \
/Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/stage7_m6_6_build_confirmation_evidence.py \
  --analysis_dir outputs/stage7_m6_5_locked_confirmation_analysis_v1 \
  --analysis_lock outputs/stage7_m6_5_locked_analysis_freeze_v1/m6_5_confirmation_analysis_lock.json \
  --quality_summary outputs/stage7_m6_5_locked_confirmation_quality_v1/milestone2b_summary.json \
  --metadata_csv outputs/stage7_m6_5_locked_confirmation_embeddings_v1/metadata.csv \
  --paired_delta_csv outputs/stage7_m6_5_locked_confirmation_stage7f_v1/paired_delta_assertive_minus_conservative/paired_delta_by_scenario.csv \
  --bootstrap_repetitions 10000 \
  --seed 20260808 \
  --output_dir outputs/stage7_m6_6_confirmation_evidence_v1
```

## 5. Completed result

The evidence-package status is `PASS_WITH_QUALITY_LIMITATIONS`. All eight sample checks pass.
The locked overall result remains MMD²=`0.0044693963`, 0/100000 exceedances, plus-one
p=`9.9999e-6`; all five learned-embedding tasks retain Holm significance.

Maximum paired fallback rate has overall Spearman rho=`0.5088`, with task-stratified
bootstrap 95% CI `[0.4086, 0.6035]`. After removing between-task rank differences, rho remains
`0.4499`, with 95% CI `[0.3842, 0.5719]`. This material post-treatment association strengthens
the quality limitation: M6.5 supports a planner-conditioned behavior-distribution difference,
but not safety, planner superiority, or a claim that the entire difference is a pure planner
mechanism independent of lane-context quality.

## 6. Validation

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python -m pytest -q \
  tests/test_stage7_m6_6_build_confirmation_evidence.py
/Users/liuqing/miniconda3/envs/nuplan/bin/python -m py_compile tools/*.py
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/check_no_tmp_dependencies.py
```
