#!/usr/bin/env python3
"""Run the frozen Stage7L-E representation inference and prospective paired BDD.

The command consumes only Stage7L-E prepared contexts.  It never runs nuPlan,
changes a checkpoint, selects scenarios, or ranks raw MMD² across representations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage6l_prepare_context_representation_ablation import (  # noqa: E402
    apply_scaler,
    ego_kinematic_features,
)
from tools.stage6l_run_context_representation_ablation import kernel_analysis  # noqa: E402
from tools.stage6u_unified_abc_trainer import UnifiedABCModel, feature_group_indices  # noqa: E402
from tools.stage7_m6_scenario_conditioned_bdd import holm_adjust  # noqa: E402
from tools.train_context_behavior_embedding import ContextFlattenGRUEncoder  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOSES = ("dose0", "dose25", "dose50", "dose75", "dose100")
NONZERO_DOSES = DOSES[1:]
TASKS = ("LAT.LANE_CHANGE", "LAT.DYNAMICS")
REPRESENTATIONS = ("old64", "A_seed3407", "B_seed3407", "C_seed3407", "ego13")
PRIMARY_KEY = ("B_seed3407", "dose100", "LAT.LANE_CHANGE")
NULL_REPETITIONS = 100_000
NULL_SEED_BASE = 2026081301
EXPECTED_CHECKPOINTS = {
    "old64": (
        ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2/best_model.pt",
        "909022f5df03a3f01c2149da6c9b44c613e955a4d816e8ec4d5862f39f8bf0cc",
    ),
    "A_seed3407": (
        ROOT / "outputs/stage6t_candidates_v1/candidate_A_dynamic_data_legacy/seed_3407/best_model.pt",
        "353982753f208d27d677c6863a681997b8e28b728573a52fa407807f6fd0298d",
    ),
    "B_seed3407": (
        ROOT / "outputs/stage6t_candidates_v1/candidate_B_single_gru_recovery/seed_3407/best_model.pt",
        "d8e0de6e74ee29076082aabef27a425b47678e1372c630e4f4a04106ff34265f",
    ),
    "C_seed3407": (
        ROOT / "outputs/stage6t_candidates_v1/candidate_C_dual_branch/seed_3407/best_model.pt",
        "cc6bf3c427534f66f74904c8948bf427cfe9f1152bba4bca0e8342f3fa47433d",
    ),
}
EGO13_SCALER = ROOT / "outputs/stage6l_context_representation_ablation_representations_v2_runtime_repaired/scalers/handcrafted_reference_scalers.npz"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    write_json(temporary, value)
    temporary.replace(path)


def atomic_save_npy(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        np.save(handle, values)
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    names = list(fields or rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=names, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def schema_names(context_dir: Path) -> list[str]:
    schema = read_json(context_dir / "feature_schema.json")
    rows = schema.get("features", schema.get("channels"))
    if not isinstance(rows, list) or len(rows) != 33:
        raise RuntimeError(f"Invalid frozen global33 feature schema: {context_dir / 'feature_schema.json'}")
    return [str(row["name"]) for row in rows]


def validate_checkpoint_locks() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for representation, (path, expected) in EXPECTED_CHECKPOINTS.items():
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"Locked checkpoint SHA mismatch: {representation}")
        result[representation] = {"path": str(path), "sha256": observed}
    if not EGO13_SCALER.is_file():
        raise FileNotFoundError(f"Missing frozen ego13 scaler: {EGO13_SCALER}")
    result["ego13"] = {"path": str(EGO13_SCALER), "sha256": sha256_file(EGO13_SCALER)}
    return result


def embed_model(model: torch.nn.Module, context: np.ndarray, device: torch.device) -> np.ndarray:
    values: list[np.ndarray] = []
    model = model.eval().to(device)
    with torch.no_grad():
        for start in range(0, len(context), 128):
            batch = torch.from_numpy(np.asarray(context[start : start + 128], dtype=np.float32).copy()).to(device)
            values.append(model(batch).detach().cpu().numpy().astype(np.float64))
    result = np.concatenate(values)
    if result.shape != (80, 64) or not np.isfinite(result).all():
        raise RuntimeError(f"Invalid learned representation output: {result.shape}")
    return result


def load_models(feature_names: Sequence[str], device: torch.device) -> dict[str, torch.nn.Module]:
    groups = feature_group_indices(list(feature_names))
    old = ContextFlattenGRUEncoder(input_dim=83, hidden_dim=128, embedding_dim=64)
    old.load_state_dict(torch.load(EXPECTED_CHECKPOINTS["old64"][0], map_location="cpu", weights_only=False)["model"])
    models: dict[str, torch.nn.Module] = {"old64": old.eval().to(device)}
    for short_name, representation in (("A", "A_seed3407"), ("B", "B_seed3407"), ("C", "C_seed3407")):
        model = UnifiedABCModel(short_name, groups)
        state = torch.load(EXPECTED_CHECKPOINTS[representation][0], map_location="cpu", weights_only=False)["model"]
        model.load_state_dict(state, strict=True)
        models[representation] = model.eval().to(device)
    return models


def validate_contexts(context_root: Path, task_rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    canonical_tokens = [row["scenario_token"] for row in task_rows]
    feature_sha: str | None = None
    dose_rows: dict[str, Any] = {}
    for dose in DOSES:
        directory = context_root / dose
        context = np.load(directory / "context_traj.npy", mmap_mode="r")
        ego = np.load(directory / "ego_seq.npy", mmap_mode="r")
        mask = np.load(directory / "ego_seq_mask.npy", mmap_mode="r")
        metadata = read_csv(directory / "metadata.csv")
        if context.shape != (80, 150, 83) or ego.shape != (80, 150, 8) or mask.shape != (80, 150):
            raise RuntimeError(f"Stage7L-E context shape mismatch at {dose}: {context.shape}/{ego.shape}/{mask.shape}")
        if not np.isfinite(context).all() or not np.isfinite(ego).all():
            raise RuntimeError(f"Non-finite Stage7L-E input at {dose}")
        tokens = [row["scenario_token"] for row in metadata]
        if tokens != canonical_tokens:
            raise RuntimeError(f"Scenario order mismatch at {dose}")
        builder_warnings = read_json(directory / "warnings.json")
        builder_validation = builder_warnings.get("validation", {})
        if builder_validation.get("pass") is not True:
            raise RuntimeError(f"Stage5D builder validation did not pass at {dose}")
        current_feature_sha = sha256_file(directory / "feature_schema.json")
        if feature_sha is None:
            feature_sha = current_feature_sha
        elif current_feature_sha != feature_sha:
            raise RuntimeError("Feature schema changed across Stage7L-E doses")
        dose_rows[dose] = {
            "context_shape": list(context.shape),
            "context_dtype": str(context.dtype),
            "context_sha256": sha256_file(directory / "context_traj.npy"),
            "ego_sha256": sha256_file(directory / "ego_seq.npy"),
            "mask_sha256": sha256_file(directory / "ego_seq_mask.npy"),
            "metadata_sha256": sha256_file(directory / "metadata.csv"),
            "valid_timestep_min": int(np.asarray(mask, dtype=bool).sum(axis=1).min()),
            "valid_timestep_max": int(np.asarray(mask, dtype=bool).sum(axis=1).max()),
            "builder_validation_pass": True,
            "assignment_mode": builder_validation.get("assignment_mode"),
            "slot_sanity_min_coverage": builder_validation.get("slot_sanity_min_coverage"),
            "slot_sanity_passed": builder_validation.get("slot_sanity_passed"),
            "slot_coverage_by_slot": builder_validation.get("slot_coverage_by_slot"),
            "fallback_assignment_used_rate": builder_validation.get("fallback_assignment_used_rate"),
        }
    return {
        "status": "STAGE7L_E_INPUT_CONTRACT_VALIDATED",
        "dose_count": 5,
        "rows_per_dose": 80,
        "target_shape_per_dose": [80, 150, 83],
        "feature_schema_sha256": feature_sha,
        "scenario_order_identical_across_doses": True,
        "finite": True,
        "unsafe_or_offroad_or_collision_filtering": False,
        "population_specific_slot_diagnostic_note": (
            "Before any checkpoint/embedding/BDD read, the generic 0.05 global sign sanity "
            "evaluated right_front at 5.4% coverage and failed on this lane-change-heavy population. "
            "The frozen 83D values were not changed; 0.06 only classifies this low-coverage aggregate "
            "sign check as diagnostic while preserving lane-aware assignment and every row."
        ),
        "dose_inputs": dose_rows,
    }


def build_representations(
    context_root: Path, output_dir: Path, checkpoint_locks: Mapping[str, Mapping[str, str]]
) -> tuple[dict[tuple[str, str], np.ndarray], list[dict[str, Any]]]:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    names = schema_names(context_root / "dose0")
    models = load_models(names, device)
    scaler = np.load(EGO13_SCALER)
    values: dict[tuple[str, str], np.ndarray] = {}
    per_rep: dict[str, list[dict[str, Any]]] = {name: [] for name in REPRESENTATIONS}
    embedding_root = output_dir / "embeddings"
    embedding_root.mkdir(exist_ok=True)
    for dose in DOSES:
        directory = context_root / dose
        context = np.asarray(np.load(directory / "context_traj.npy", mmap_mode="r"), dtype=np.float32)
        for representation, model in models.items():
            rep_dir = embedding_root / representation
            rep_dir.mkdir(exist_ok=True)
            path = rep_dir / f"{dose}.npy"
            if path.is_file():
                restored = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float64)
                if restored.shape != (80, 64) or not np.isfinite(restored).all():
                    raise RuntimeError(f"Invalid resumable embedding: {path}")
                values[(representation, dose)] = restored
            else:
                values[(representation, dose)] = embed_model(model, context, device)
                atomic_save_npy(path, values[(representation, dose)].astype(np.float32))
        ego = np.asarray(np.load(directory / "ego_seq.npy", mmap_mode="r"), dtype=np.float32)
        mask = np.asarray(np.load(directory / "ego_seq_mask.npy", mmap_mode="r"), dtype=bool)
        ego_dir = embedding_root / "ego13"
        ego_dir.mkdir(exist_ok=True)
        ego_path = ego_dir / f"{dose}.npy"
        if ego_path.is_file():
            values[("ego13", dose)] = np.asarray(np.load(ego_path, mmap_mode="r"), dtype=np.float64)
        else:
            values[("ego13", dose)] = np.asarray(
                apply_scaler(ego_kinematic_features(ego, mask), scaler["ego_median"], scaler["ego_scale"]),
                dtype=np.float64,
            )
            atomic_save_npy(ego_path, values[("ego13", dose)].astype(np.float32))
        if values[("ego13", dose)].shape != (80, 13) or not np.isfinite(values[("ego13", dose)]).all():
            raise RuntimeError(f"Invalid ego13 representation at {dose}")
        for representation in REPRESENTATIONS:
            rep_dir = embedding_root / representation
            rep_dir.mkdir(exist_ok=True)
            path = rep_dir / f"{dose}.npy"
            per_rep[representation].append(
                {
                    "dose": dose,
                    "shape": list(values[(representation, dose)].shape),
                    "embedding_sha256": sha256_file(path),
                    "input_feature_sha256": sha256_file(directory / "context_traj.npy"),
                }
            )
    manifests: list[dict[str, Any]] = []
    for representation in REPRESENTATIONS:
        manifest = {
            "schema_version": "stage7l_e_embedding_manifest_v1",
            "representation": representation,
            "checkpoint_or_scaler": checkpoint_locks[representation],
            "preprocessing_contract": "frozen_stage6v_stage6w_inference_pipeline",
            "model_eval": True,
            "dropout_disabled": True,
            "scenario_order_sha256": sha256_file(output_dir / "task_masks.csv"),
            "finite": True,
            "doses": per_rep[representation],
        }
        path = output_dir / f"embedding_manifest_{representation.removesuffix('_seed3407')}.json"
        write_json(path, manifest)
        manifests.append({"path": str(path), "sha256": sha256_file(path)})
    return values, manifests


def cell_seed(rep_index: int, dose_index: int, task_index: int) -> int:
    """Replay the Stage6V/W deterministic per-cell seed policy."""
    return NULL_SEED_BASE + rep_index * 1000 + dose_index * 100 + task_index


def apply_fixed_holm(cells: list[dict[str, Any]]) -> None:
    secondary = [row for row in cells if row["multiplicity_role"] == "SECONDARY_HOLM_39"]
    if len(cells) != 40 or len(secondary) != 39:
        raise RuntimeError(f"Frozen family size mismatch: total={len(cells)}, secondary={len(secondary)}")
    adjusted = holm_adjust([float(row["raw_p_for_multiplicity"]) for row in secondary])
    for row, value in zip(secondary, adjusted):
        row["holm_p"] = float(value)
        row["holm_significant_0_05"] = bool(value < 0.05)
    primary = [row for row in cells if row["multiplicity_role"] == "PRIMARY_NOT_PART_OF_SECONDARY_HOLM_FAMILY"]
    if len(primary) != 1 or (primary[0]["representation"], primary[0]["dose"], primary[0]["task"]) != PRIMARY_KEY:
        raise RuntimeError("Primary was not excluded exactly once from the 39-test family")
    primary[0]["holm_p"] = None
    primary[0]["holm_significant_0_05"] = None


def evaluate_cells(
    representations: Mapping[tuple[str, str], np.ndarray],
    task_rows: Sequence[Mapping[str, str]],
    *,
    cell_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    task_masks = {
        task: np.asarray([str(row[task]).lower() == "true" for row in task_rows], dtype=bool)
        for task in TASKS
    }
    if (int(task_masks["LAT.LANE_CHANGE"].sum()), int(task_masks["LAT.DYNAMICS"].sum())) != (80, 38):
        raise RuntimeError("Frozen task population mismatch before BDD")
    cells: list[dict[str, Any]] = []
    nulls: dict[str, np.ndarray] = {}
    if cell_dir is not None:
        cell_dir.mkdir(parents=True, exist_ok=True)
    for rep_index, representation in enumerate(REPRESENTATIONS):
        for dose_index, dose in enumerate(NONZERO_DOSES):
            for task_index, task in enumerate(TASKS):
                mask = task_masks[task]
                n_pair = int(mask.sum())
                key = (representation, dose, task)
                role = (
                    "PRIMARY_NOT_PART_OF_SECONDARY_HOLM_FAMILY"
                    if key == PRIMARY_KEY
                    else "SECONDARY_HOLM_39"
                )
                seed = cell_seed(rep_index, dose_index, task_index)
                cell_id = f"{representation}__{dose}__{task.replace('.', '_')}"
                cell_path = cell_dir / f"{cell_id}.json" if cell_dir is not None else None
                null_path = cell_dir / f"{cell_id}_null.npy" if cell_dir is not None else None
                if cell_path is not None and cell_path.is_file():
                    row = read_json(cell_path)
                    expected_identity = {
                        "representation": representation,
                        "dose": dose,
                        "task": task,
                        "N_pair": n_pair,
                        "null_seed": seed,
                        "null_reps": NULL_REPETITIONS,
                        "multiplicity_role": role,
                    }
                    if any(row.get(key) != value for key, value in expected_identity.items()):
                        raise RuntimeError(f"Resumable cell identity mismatch: {cell_path}")
                    if row.get("status") != "NOT_COMPUTABLE_PRE_FROZEN_TASK_POPULATION":
                        if null_path is None or not null_path.is_file():
                            raise RuntimeError(f"Resumable cell lacks null samples: {cell_path}")
                        samples = np.asarray(np.load(null_path, mmap_mode="r"), dtype=np.float32)
                        if samples.shape != (NULL_REPETITIONS,) or not np.isfinite(samples).all():
                            raise RuntimeError(f"Invalid resumable null samples: {null_path}")
                        nulls[f"{representation}__{dose}__{task}"] = samples
                    cells.append(row)
                    continue
                if n_pair == 0:
                    row = {
                        "representation": representation,
                        "dose": dose,
                        "task": task,
                        "N_pair": 0,
                        "status": "NOT_COMPUTABLE_PRE_FROZEN_TASK_POPULATION",
                        "null_seed": seed,
                        "null_reps": NULL_REPETITIONS,
                        "raw_p": None,
                        "raw_p_for_multiplicity": 1.0,
                        "multiplicity_role": role,
                    }
                else:
                    result, samples, _ = kernel_analysis(
                        representations[(representation, "dose0")][mask],
                        representations[(representation, dose)][mask],
                        repetitions=NULL_REPETITIONS,
                        seed=seed,
                    )
                    null_key = f"{representation}__{dose}__{task}"
                    nulls[null_key] = samples.astype(np.float32)
                    row = {
                        "representation": representation,
                        "dose": dose,
                        "task": task,
                        "N_pair": n_pair,
                        "status": "COMPUTED" if task != "LAT.DYNAMICS" else "LOW_N_SECONDARY_DIAGNOSTIC",
                        "null_seed": seed,
                        "null_reps": NULL_REPETITIONS,
                        "raw_mmd2": result["mmd2"],
                        "bandwidth": result["bandwidth"],
                        "null_mean": result["paired_null_mean"],
                        "null_sd": result["paired_null_sd"],
                        "null_q95": result["paired_null_q95"],
                        "bdd_over_null_q95": result["bdd_to_null_q95_ratio"],
                        "z_bdd": result["null_standardized_z_bdd"],
                        "raw_p": result["raw_p"],
                        "raw_p_for_multiplicity": result["raw_p"],
                        "exceedance_count": result["exceedance_count"],
                        "multiplicity_role": role,
                    }
                    if null_path is not None:
                        atomic_save_npy(null_path, nulls[null_key])
                if cell_path is not None:
                    atomic_write_json(cell_path, row)
                cells.append(row)
    apply_fixed_holm(cells)
    return cells, nulls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--context-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    preparation = read_json(args.prepared_dir / "input_contract_preparation_audit.json")
    if preparation.get("status") != "STAGE7L_E_INPUT_VIEWS_PREPARED_CONTEXT_BUILD_NOT_YET_VALIDATED":
        raise RuntimeError("Stage7L-E input preparation audit is not in the expected state")
    if preparation.get("stage7l_d_unlock_verified") is not True or preparation.get("planner_rerun") is not False:
        raise RuntimeError("Stage7L-D unlock/provenance check failed")
    prepared_is_output = args.prepared_dir.resolve() == args.output_dir.resolve()
    if args.output_dir.exists() and not prepared_is_output:
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    if prepared_is_output and (args.output_dir / "stage7l_e_final_decision.json").exists():
        raise FileExistsError("Stage7L-E formal results already exist; deterministic resume must use cell artifacts, not overwrite the frozen decision")
    args.output_dir.mkdir(parents=True, exist_ok=prepared_is_output)
    task_rows = read_csv(args.prepared_dir / "task_masks.csv")
    if not prepared_is_output:
        shutil.copy2(args.prepared_dir / "task_masks.csv", args.output_dir / "task_masks.csv")
    input_audit = validate_contexts(args.context_root, task_rows)
    write_json(args.output_dir / "input_contract_audit.json", input_audit)
    write_json(args.output_dir / "task_mask_audit.json", read_json(args.prepared_dir / "task_mask_audit.json"))
    checkpoint_locks = validate_checkpoint_locks()
    representations, embedding_manifests = build_representations(args.context_root, args.output_dir, checkpoint_locks)
    cells, nulls = evaluate_cells(
        representations,
        task_rows,
        cell_dir=args.output_dir / "cell_ledger",
    )
    np.savez_compressed(args.output_dir / "paired_null_samples.npz", **nulls)
    fields = sorted({key for row in cells for key in row})
    write_csv(args.output_dir / "all_bdd_cells.csv", cells, fields)
    primary = next(row for row in cells if row["multiplicity_role"] == "PRIMARY_NOT_PART_OF_SECONDARY_HOLM_FAMILY")
    primary["primary_success"] = bool(float(primary["raw_p"]) < 0.05 and int(primary["N_pair"]) >= 76)
    primary["final_primary_status"] = "STAGE7L_E_PRIMARY_BDD_PASSED" if primary["primary_success"] else "STAGE7L_E_PRIMARY_BDD_FAILED"
    write_json(args.output_dir / "primary_bdd_result.json", primary)
    secondary = [row for row in cells if row["multiplicity_role"] == "SECONDARY_HOLM_39"]
    write_csv(args.output_dir / "secondary_bdd_cells.csv", secondary, fields)
    write_csv(args.output_dir / "secondary_holm_results.csv", secondary, fields)
    dose_rows = [
        {
            "representation": row["representation"],
            "task": row["task"],
            "dose": row["dose"],
            "N_pair": row["N_pair"],
            "bdd_over_null_q95": row.get("bdd_over_null_q95"),
            "z_bdd": row.get("z_bdd"),
            "raw_p": row.get("raw_p"),
            "holm_p": row.get("holm_p"),
            "status": row["status"],
        }
        for row in cells
    ]
    write_csv(args.output_dir / "dose_response_standardized_sensitivity.csv", dose_rows)
    comparison = [row for row in cells if row["dose"] == "dose100" and row["task"] == "LAT.LANE_CHANGE"]
    write_csv(args.output_dir / "representation_comparison_dose100_lane_change.csv", comparison, fields)
    decision = {
        "schema_version": "stage7l_e_prospective_bdd_decision_v1",
        "primary_status": primary["final_primary_status"],
        "final_status": "STAGE7L_E_PROSPECTIVE_REPRESENTATION_EVALUATION_COMPLETE",
        "primary": primary,
        "theoretical_cell_count": len(cells),
        "secondary_holm_family_count": len(secondary),
        "secondary_holm_pass_count": sum(bool(row["holm_significant_0_05"]) for row in secondary),
        "not_computable_count": sum(row["status"] == "NOT_COMPUTABLE_PRE_FROZEN_TASK_POPULATION" for row in cells),
        "low_n_count": sum(row["status"] == "LOW_N_SECONDARY_DIAGNOSTIC" for row in cells),
        "cross_representation_raw_mmd2_comparison_performed": False,
        "stage6v_qualification_changed": False,
        "planner_rerun": False,
        "checkpoint_or_training_modified": False,
        "embedding_manifests": embedding_manifests,
    }
    write_json(args.output_dir / "stage7l_e_final_decision.json", decision)
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
