#!/usr/bin/env python3
"""Conditionally evaluate Stage6S-v3 representations after mechanism PASS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import stage6v_evaluate_stage6s_v2_representations as frozen  # noqa: E402

PROTOCOL = ROOT / "configs/stage6t_training_evaluation_protocol.json"
LEDGER = ROOT / "outputs/stage6u_abc_formal_training_v1/checkpoint_lock/stage6u_formal_checkpoint_ledger.json"
FREEZE = ROOT / "outputs/stage6s_v3_confirmation_freeze_v1/stage6s_v3_confirmation_freeze_manifest.json"
FREEZE_SHA = "7105940bd822f02d643ed4f5cb9a8321b3827ca6117be289914057e3fe8a26c6"
PLANNERS = frozen.PLANNERS
REPS = frozen.REPS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism_summary", type=Path, required=True)
    parser.add_argument("--context_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Refusing to overwrite {out}")
    if frozen.sha256(FREEZE) != FREEZE_SHA:
        raise RuntimeError("Stage6S-v3 freeze changed")
    mechanism = frozen.read_json(args.mechanism_summary)
    if mechanism.get("status") != "STAGE6S_V3_MECHANISM_GATE_PASS_REPRESENTATION_EVALUATION_AUTHORIZED" or mechanism.get("mechanism_gate_passed") is not True:
        raise RuntimeError("Stage6S-v3 mechanism gate did not authorize representation evaluation")
    if mechanism.get("embedding_or_bdd_read") is not False:
        raise RuntimeError("Stage6S-v3 mechanism blind state invalid")
    protocol = frozen.read_json(PROTOCOL)
    scorecard = protocol["stage6s_v2_interaction_scorecard"]
    ledger = frozen.read_json(LEDGER)
    meta = pd.read_csv(args.context_dir / "metadata.csv").sort_values("global_row").reset_index(drop=True)
    pairs, logs = [], []
    for scenario, frame in meta.groupby("scenario_index", sort=True):
        short = frame[frame.planner_name == PLANNERS[0]]
        long = frame[frame.planner_name == PLANNERS[1]]
        if len(short) != 1 or len(long) != 1:
            raise RuntimeError(f"incomplete Stage6S-v3 pair {scenario}")
        pairs.append((int(short.iloc[0].global_row), int(long.iloc[0].global_row)))
        logs.append(str(short.iloc[0].log_name))
    pairs = np.asarray(pairs, dtype=np.int64)
    if pairs.shape != (80, 2):
        raise RuntimeError(f"expected 80 Stage6S-v3 pairs, got {pairs.shape}")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    representations = frozen.build_representations(args.context_dir.resolve(), ledger, device)
    out.mkdir(parents=True)
    rep_dir = out / "representations"
    rep_dir.mkdir()
    rows, nulls, contrasts = [], {}, {}
    null_cfg = scorecard["paired_null"]
    for representation in REPS:
        path = rep_dir / f"{representation}.npy"
        np.save(path, representations[representation].astype(np.float32))
        row, samples, contrast = frozen.analyze(
            representations[representation], pairs,
            int(null_cfg["permutations"]), int(null_cfg["seed"]),
        )
        row = {"representation": representation, **row}
        row["candidate_detection_gate_pass"] = bool(
            row["raw_p"] < 0.05 and row["null_standardized_z_bdd"] > 1.645
        )
        rows.append(row)
        nulls[representation] = samples.astype(np.float32)
        contrasts[representation] = contrast
    results = pd.DataFrame(rows)
    results_path = out / "stage6s_v3_confirmation_representation_results.csv"
    results.to_csv(results_path, index=False)
    null_path = out / "stage6s_v3_confirmation_null_samples.npz"
    np.savez_compressed(null_path, **nulls)
    by_rep = results.set_index("representation")
    delta = float(by_rep.loc["C", "null_standardized_z_bdd"] - by_rep.loc["C_neighbor_zero", "null_standardized_z_bdd"])
    log_values = np.asarray(logs, dtype=str)
    unique = np.unique(log_values)
    rng = np.random.default_rng(int(scorecard["c_context_increment"]["bootstrap_seed"]))
    bootstrap = np.empty(10_000, dtype=float)
    c_mean, c_sd = float(np.mean(nulls["C"])), float(np.std(nulls["C"], ddof=1))
    z_mean, z_sd = float(np.mean(nulls["C_neighbor_zero"])), float(np.std(nulls["C_neighbor_zero"], ddof=1))
    for repetition in range(10_000):
        selected_logs = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(log_values == log) for log in selected_logs])
        c_observed = float(np.mean(contrasts["C"][np.ix_(indices, indices)]))
        zero_observed = float(np.mean(contrasts["C_neighbor_zero"][np.ix_(indices, indices)]))
        bootstrap[repetition] = (c_observed - c_mean) / c_sd - (zero_observed - z_mean) / z_sd
    lower, upper = float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))
    increment = {
        "comparison": "C_full_minus_C_neighbor_zero", "delta_z_bdd": delta,
        "log_cluster_bootstrap95_lower": lower, "log_cluster_bootstrap95_upper": upper,
        "bootstrap_repetitions": 10000,
        "bootstrap_seed": int(scorecard["c_context_increment"]["bootstrap_seed"]),
        "incremental_interaction_information_pass": bool(delta > 0 and lower > 0),
        "raw_mmd2_difference_computed": False,
    }
    increment_path = out / "stage6s_v3_c_context_increment.json"
    frozen.write_json(increment_path, increment)
    files = [results_path.name, null_path.name, increment_path.name]
    manifest = {
        "schema_version": "stage6s_v3_representation_evaluation_v1",
        "status": "FROZEN_STAGE6S_V3_REPRESENTATION_EVALUATION_COMPLETE",
        "mechanism_summary_sha256": frozen.sha256(args.mechanism_summary),
        "stage6s_v3_freeze_sha256": FREEZE_SHA, "primary_seed": 3407,
        "representations": REPS, "common_swap_vectors": True,
        "permutations": int(null_cfg["permutations"]), "seed": int(null_cfg["seed"]),
        "cross_representation_raw_mmd2_comparison_performed": False,
        "training_or_protocol_modified": False,
        "result_files": {name: frozen.sha256(out / name) for name in files},
        "representation_sha256": {name: frozen.sha256(rep_dir / f"{name}.npy") for name in REPS},
        "frozen_stage6s_v2_representation_implementation_sha256": frozen.sha256(Path(frozen.__file__)),
    }
    manifest_path = out / "stage6s_v3_representation_result_manifest.json"
    frozen.write_json(manifest_path, manifest)
    lines = [
        "# Stage6S-v3 prospective confirmation representation 报告", "",
        f"- 状态：`{manifest['status']}`",
        "- 机制门禁先验通过后才解锁；各representation独立bandwidth/null；不跨representation比较raw MMD²。",
        "", "## Null-standardized结果", "", results.to_markdown(index=False),
        "", "## C上下文增量", "", pd.DataFrame([increment]).to_markdown(index=False),
    ]
    (out / "stage6s_v3_representation_report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": manifest["status"], "manifest_sha256": frozen.sha256(manifest_path),
        "results": rows, "c_context_increment": increment,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
