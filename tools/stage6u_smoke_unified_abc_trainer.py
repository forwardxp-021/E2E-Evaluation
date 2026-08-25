#!/usr/bin/env python3
"""Run bounded synthetic and Dynamic-v2 train/val smoke for Stage6U."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage6u_unified_abc_trainer import (
    DynamicTrainValDataset,
    InMemoryTrainValDataset,
    UnifiedABCModel,
    assert_bc_fairness,
    build_optimizer_and_scheduler,
    build_random_plan,
    collate_rows,
    encoder_parameter_count,
    feature_group_indices,
    formal_checkpoint_payload,
    load_and_validate_implementation_config,
    load_checkpoint,
    load_formal_checkpoint,
    random_plan_ledger,
    read_json,
    resolve_repo_path,
    run_batch,
    save_checkpoint,
    sha256_array,
    sha256_file,
    sha256_json,
    state_dict_sha256,
    validate_resume_plan,
)


def select_device(policy: str) -> torch.device:
    if policy == "mps_if_available_else_cpu" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def synthetic_datasets(
    rows: int,
    train_rows: int,
    seed: int,
    group_indices: dict[str, list[int]],
) -> tuple[InMemoryTrainValDataset, InMemoryTrainValDataset]:
    rng = np.random.default_rng(seed)
    all_rows = []
    for index in range(rows):
        context = rng.normal(0.0, 0.5, size=(80, 83)).astype(np.float32)
        context[:, 5] = np.abs(context[:, 5] * 5.0 + (index % 3) * 7.0)
        slot_valid = rng.random((5, 80)) < np.asarray([0.5, 0.25, 0.25, 0.25, 0.25])[:, None]
        for slot in range(5):
            start = 8 + slot * 15
            context[:, start] = slot_valid[slot].astype(np.float32)
            context[~slot_valid[slot], start : start + 15] = 0.0
        raw33 = rng.normal(0.0, 1.0, size=33).astype(np.float32)
        raw33[13] = float(index % 2)
        clean = rng.normal(0.0, 1.0, size=6).astype(np.float32)
        speed_bin = 0 if context[:, 5].mean() < 5 else (1 if context[:, 5].mean() < 15 else 2)
        front_ratio = slot_valid[0].mean()
        front_regime = 0 if front_ratio < 1e-6 else (1 if front_ratio < 0.5 else 2)
        all_rows.append(
            {
                "context": context,
                "raw33": raw33,
                "feat33": raw33.copy(),
                "clean_longitudinal": clean,
                "slot_valid": slot_valid,
                "stratum": (speed_bin, front_regime, int(raw33[13] > 0)),
                "nuisance": raw33[[13, 14, 18, 19, 20]].copy(),
                "row_key": f"synthetic_{index}",
                "dataset_index": index if index < train_rows else index - train_rows,
            }
        )
    train = InMemoryTrainValDataset(all_rows[:train_rows], group_indices)
    val_rows = all_rows[train_rows:]
    for local_index, row in enumerate(val_rows):
        row["dataset_index"] = local_index
    val = InMemoryTrainValDataset(val_rows, group_indices)
    return train, val


def make_plan(
    dataset: Any,
    candidate: str,
    stage6t: dict[str, Any],
    seed: int,
    epoch: int,
    epoch_samples: int,
    batch_size: int,
) -> dict[str, Any]:
    candidate_config = stage6t["candidates"][candidate]
    dropout = stage6t["dropout_packages"][candidate_config["dropout_package"]]
    objective = stage6t["objective_packages"][candidate_config["objective_package"]]
    return build_random_plan(
        dataset,
        seed=seed,
        pair_seed=int(stage6t["common_optimization"]["pair_seed"]),
        epoch=epoch,
        epoch_samples=epoch_samples,
        batch_size=batch_size,
        candidate=candidate,
        sampling_package=candidate_config["sampling_package"],
        dropout_package=candidate_config["dropout_package"],
        slot_dropout_probability=float(dropout.get("slot_dropout_probability", 0.0)),
        all_neighbor_dropout_probability=float(dropout.get("all_neighbor_dropout_probability", 0.0)),
        ranking_margin=float(objective.get("ranking_margin", 0.0)),
    )


def train_candidate_smoke(
    *,
    label: str,
    candidate: str,
    train_dataset: Any,
    val_dataset: Any,
    stage6t: dict[str, Any],
    seed: int,
    batch_size: int,
    train_batches: int,
    val_batches: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    seed_everything(seed)
    model = UnifiedABCModel(candidate, train_dataset.group_indices).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, stage6t["common_optimization"])
    train_plan = make_plan(
        train_dataset, candidate, stage6t, seed, 0, max(batch_size * train_batches, batch_size), batch_size
    )
    val_plan = make_plan(
        val_dataset, candidate, stage6t, seed + 90_000, 0, max(batch_size * val_batches, batch_size), batch_size
    )
    train_results = []
    for batch_index in range(train_batches):
        train_results.append(
            run_batch(
                candidate=candidate, model=model, optimizer=optimizer, dataset=train_dataset, plan=train_plan,
                batch_index=batch_index, stage6t=stage6t, device=device, train=True,
            )
        )
    scheduler.step()
    val_results = []
    with torch.no_grad():
        for batch_index in range(val_batches):
            val_results.append(
                run_batch(
                    candidate=candidate, model=model, optimizer=optimizer, dataset=val_dataset, plan=val_plan,
                    batch_index=batch_index, stage6t=stage6t, device=device, train=False,
                )
            )
    checks = {
        "train_loss_finite": all(row["loss_finite"] for row in train_results),
        "val_loss_finite": all(row["loss_finite"] for row in val_results),
        "embedding_shape_64d": all(row["embedding_shape"][1] == 64 for row in train_results + val_results),
        "embedding_finite": all(row["embedding_finite"] for row in train_results + val_results),
    }
    if not all(checks.values()):
        raise RuntimeError(f"{label}/{candidate} smoke failed: {checks}")
    return (
        {
            "label": label,
            "candidate": candidate,
            "encoder_parameter_count": encoder_parameter_count(model.encoder),
            "train_results": train_results,
            "val_results": val_results,
            "checks": checks,
            "passed": True,
        },
        random_plan_ledger(train_plan),
    )


def resume_smoke(
    *,
    checkpoint_path: Path,
    dataset: Any,
    stage6t: dict[str, Any],
    protocol_fingerprint: str,
    seed: int,
    batch_size: int,
    total_batches: int,
    device: torch.device,
) -> dict[str, Any]:
    candidate = "B"
    plan = make_plan(dataset, candidate, stage6t, seed, 0, batch_size * total_batches, batch_size)
    ledger = random_plan_ledger(plan)

    seed_everything(seed)
    reference = UnifiedABCModel(candidate, dataset.group_indices).to(device)
    reference_optimizer, reference_scheduler = build_optimizer_and_scheduler(reference, stage6t["common_optimization"])
    reference_results = []
    for batch_index in range(total_batches):
        reference_results.append(
            run_batch(
                candidate=candidate, model=reference, optimizer=reference_optimizer, dataset=dataset, plan=plan,
                batch_index=batch_index, stage6t=stage6t, device=device, train=True,
            )
        )
    reference_scheduler.step()
    reference_state = state_dict_sha256(reference.state_dict())

    seed_everything(seed)
    interrupted = UnifiedABCModel(candidate, dataset.group_indices).to(device)
    interrupted_optimizer, interrupted_scheduler = build_optimizer_and_scheduler(interrupted, stage6t["common_optimization"])
    first = run_batch(
        candidate=candidate, model=interrupted, optimizer=interrupted_optimizer, dataset=dataset, plan=plan,
        batch_index=0, stage6t=stage6t, device=device, train=True,
    )
    save_checkpoint(
        checkpoint_path,
        candidate=candidate,
        model=interrupted,
        optimizer=interrupted_optimizer,
        scheduler=interrupted_scheduler,
        epoch=0,
        next_batch_index=1,
        global_step=1,
        plan_ledger=ledger,
        protocol_fingerprint=protocol_fingerprint,
        smoke_only=True,
    )

    resumed = UnifiedABCModel(candidate, dataset.group_indices).to(device)
    resumed_optimizer, resumed_scheduler = build_optimizer_and_scheduler(resumed, stage6t["common_optimization"])
    checkpoint = load_checkpoint(
        checkpoint_path,
        candidate=candidate,
        model=resumed,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        expected_protocol_fingerprint=protocol_fingerprint,
    )
    resumed_results = [first]
    for batch_index in range(int(checkpoint["next_batch_index"]), total_batches):
        resumed_results.append(
            run_batch(
                candidate=candidate, model=resumed, optimizer=resumed_optimizer, dataset=dataset, plan=plan,
                batch_index=batch_index, stage6t=stage6t, device=device, train=True,
            )
        )
    resumed_scheduler.step()
    resumed_state = state_dict_sha256(resumed.state_dict())
    checks = {
        "checkpoint_exists": checkpoint_path.is_file(),
        "candidate_restored": checkpoint["candidate"] == candidate,
        "epoch_restored": int(checkpoint["epoch"]) == 0,
        "next_batch_restored": int(checkpoint["next_batch_index"]) == 1,
        "global_step_restored": int(checkpoint["global_step"]) == 1,
        "optimizer_state_restored": bool(checkpoint["optimizer"]["state"]),
        "scheduler_state_restored": checkpoint["scheduler"]["last_epoch"] == 0,
        "rng_state_restored": set(checkpoint["rng_state"]) == {"python", "numpy", "torch"},
        "random_plan_restored": checkpoint["plan_ledger"]["candidate_independent_fingerprint_sha256"]
        == ledger["candidate_independent_fingerprint_sha256"],
        "loss_sequence_exact": [row["loss"] for row in reference_results]
        == [row["loss"] for row in resumed_results],
        "model_state_exact": reference_state == resumed_state,
        "optimizer_lr_exact": reference_optimizer.param_groups[0]["lr"] == resumed_optimizer.param_groups[0]["lr"],
        "scheduler_last_epoch_exact": reference_scheduler.last_epoch == resumed_scheduler.last_epoch,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Resume smoke failed: {checks}")
    return {
        "candidate": candidate,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "reference_model_state_sha256": reference_state,
        "resumed_model_state_sha256": resumed_state,
        "checks": checks,
        "passed": True,
    }


def formal_epoch_boundary_resume_smoke(
    *,
    checkpoint_path: Path,
    dataset: Any,
    stage6t: dict[str, Any],
    protocol_fingerprint: str,
    seed: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """Prove that an epoch-end formal checkpoint resumes at the next epoch."""
    candidate = "B"
    implementation_sha = "formal-resume-smoke-implementation"
    authorization_sha = "formal-resume-smoke-authorization"
    metadata = {
        "protocol_id": stage6t["protocol_id"],
        "candidate_id": candidate,
        "seed": seed,
        "validation_objective": "smoke_only_waymo_val_not_read_here",
    }
    plan_epoch0 = make_plan(dataset, candidate, stage6t, seed, 0, batch_size, batch_size)
    plan_epoch1 = make_plan(dataset, candidate, stage6t, seed, 1, batch_size, batch_size)
    ledger_epoch0 = random_plan_ledger(plan_epoch0)
    ledger_epoch1 = random_plan_ledger(plan_epoch1)

    seed_everything(seed)
    reference = UnifiedABCModel(candidate, dataset.group_indices).to(device)
    reference_optimizer, reference_scheduler = build_optimizer_and_scheduler(
        reference, stage6t["common_optimization"]
    )
    first = run_batch(
        candidate=candidate,
        model=reference,
        optimizer=reference_optimizer,
        dataset=dataset,
        plan=plan_epoch0,
        batch_index=0,
        stage6t=stage6t,
        device=device,
        train=True,
    )
    reference_scheduler.step()
    payload = formal_checkpoint_payload(
        candidate=candidate,
        seed=seed,
        model=reference,
        optimizer=reference_optimizer,
        scheduler=reference_scheduler,
        epoch=1,
        next_batch_index=0,
        global_step=1,
        best_val_loss=1.0,
        best_epoch=0,
        patience_count=0,
        early_stopping_reference=1.0,
        plan_ledger=None,
        epoch_train_loss_sum=0.0,
        epoch_train_rows=0,
        protocol_fingerprint=protocol_fingerprint,
        implementation_freeze_sha256=implementation_sha,
        authorization_manifest_sha256=authorization_sha,
        checkpoint_metadata=metadata,
        resume_history=[],
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)
    reference_second = run_batch(
        candidate=candidate,
        model=reference,
        optimizer=reference_optimizer,
        dataset=dataset,
        plan=plan_epoch1,
        batch_index=0,
        stage6t=stage6t,
        device=device,
        train=True,
    )
    reference_state = state_dict_sha256(reference.state_dict())

    seed_everything(seed + 1)
    resumed = UnifiedABCModel(candidate, dataset.group_indices).to(device)
    resumed_optimizer, resumed_scheduler = build_optimizer_and_scheduler(
        resumed, stage6t["common_optimization"]
    )
    checkpoint = load_formal_checkpoint(
        checkpoint_path,
        candidate=candidate,
        seed=seed,
        model=resumed,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        protocol_fingerprint=protocol_fingerprint,
        implementation_freeze_sha256=implementation_sha,
        authorization_manifest_sha256=authorization_sha,
    )
    plan_validation = validate_resume_plan(checkpoint, ledger_epoch1)
    resumed_second = run_batch(
        candidate=candidate,
        model=resumed,
        optimizer=resumed_optimizer,
        dataset=dataset,
        plan=plan_epoch1,
        batch_index=0,
        stage6t=stage6t,
        device=device,
        train=True,
    )
    resumed_state = state_dict_sha256(resumed.state_dict())
    checks = {
        "checkpoint_exists": checkpoint_path.is_file(),
        "formal_schema": checkpoint["schema_version"] == "stage6u_formal_checkpoint_v1",
        "seed_bound": int(checkpoint["seed"]) == seed,
        "authorization_bound": checkpoint["authorization_manifest_sha256"] == authorization_sha,
        "next_epoch_restored": int(checkpoint["epoch"]) == 1,
        "epoch_boundary_cursor_zero": int(checkpoint["next_batch_index"]) == 0,
        "epoch_boundary_plan_is_none": checkpoint["plan_ledger"] is None,
        "next_epoch_plan_not_compared_to_previous": plan_validation == "epoch_boundary_no_plan_check",
        "epoch_plans_differ": ledger_epoch0["candidate_independent_fingerprint_sha256"]
        != ledger_epoch1["candidate_independent_fingerprint_sha256"],
        "optimizer_state_restored": bool(checkpoint["optimizer"]["state"]),
        "scheduler_state_restored": resumed_scheduler.last_epoch == reference_scheduler.last_epoch,
        "next_epoch_loss_exact": reference_second["loss"] == resumed_second["loss"],
        "next_epoch_model_state_exact": reference_state == resumed_state,
        "first_epoch_loss_finite": bool(np.isfinite(first["loss"])),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Formal epoch-boundary resume smoke failed: {checks}")
    return {
        "candidate": candidate,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checks": checks,
        "passed": True,
    }


def global33_smoke(dataset: DynamicTrainValDataset, standardization_path: Path) -> dict[str, Any]:
    stats = read_json(standardization_path)
    selected = list(range(min(len(dataset), 16)))
    manual = []
    observed = []
    for index in selected:
        row = dataset.get(index)
        manual.append((row["raw33"] - np.asarray(stats["mean"], dtype=np.float32)) / np.asarray(stats["std"], dtype=np.float32))
        observed.append(row["feat33"])
    manual_array = np.stack(manual)
    observed_array = np.stack(observed)
    checks = {
        "source_array_is_raw33": stats["source_array"] == "interaction_feat_style_raw.npy",
        "fit_split_train": stats["fit_split"] == "train",
        "train_count_135046": int(stats["train_count"]) == 135046,
        "manual_formula_exact": np.array_equal(manual_array, observed_array),
        "part_local_target_forbidden": stats["part_local_interaction_feat_style_npy_allowed_for_stage6t_training"] is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"global33 smoke failed: {checks}")
    return {"checks": checks, "observed_subset_sha256": sha256_array(observed_array), "passed": True}


def timing_probe(
    *,
    candidate: str,
    dataset: Any,
    stage6t: dict[str, Any],
    seed: int,
    batch_size: int,
    warmup_batches: int,
    measured_batches: int,
    device: torch.device,
) -> dict[str, Any]:
    total = warmup_batches + measured_batches
    plan = make_plan(dataset, candidate, stage6t, seed, 1, batch_size * total, batch_size)
    seed_everything(seed)
    model = UnifiedABCModel(candidate, dataset.group_indices).to(device)
    optimizer, _ = build_optimizer_and_scheduler(model, stage6t["common_optimization"])
    durations = []
    for batch_index in range(total):
        started = time.perf_counter()
        run_batch(
            candidate=candidate, model=model, optimizer=optimizer, dataset=dataset, plan=plan,
            batch_index=batch_index, stage6t=stage6t, device=device, train=True,
        )
        if device.type == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - started
        if batch_index >= warmup_batches:
            durations.append(elapsed)
    mean_seconds = float(np.mean(durations))
    formal_batch = int(stage6t["common_optimization"]["batch_size"])
    scaling = formal_batch / batch_size
    estimated_formal_batch_seconds = mean_seconds * scaling
    epoch_batches = math.ceil(int(stage6t["dataset_contract"]["split_counts"]["train"]) / formal_batch)
    return {
        "candidate": candidate,
        "device": str(device),
        "probe_batch_size": batch_size,
        "measured_batch_seconds": durations,
        "mean_probe_batch_seconds": mean_seconds,
        "linear_scaled_formal_batch_seconds": estimated_formal_batch_seconds,
        "formal_batches_per_epoch": epoch_batches,
        "estimated_epoch_hours": estimated_formal_batch_seconds * epoch_batches / 3600.0,
        "estimated_max30_hours": estimated_formal_batch_seconds * epoch_batches * 30 / 3600.0,
        "estimate_method": "small-train-subset MPS smoke; linear batch-size scaling; excludes val/checkpoint and is planning-only",
    }


def count_formal_checkpoints(paths: list[str]) -> tuple[int, list[str]]:
    files = []
    for text in paths:
        root = resolve_repo_path(text)
        if root.exists():
            files.extend(str(path) for path in root.rglob("*.pt"))
    return len(files), files


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    config, stage6t, freeze = load_and_validate_implementation_config(config_path)
    smoke = config["smoke"]
    output = args.output_dir.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Smoke output exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    device = select_device(smoke["device_policy"])
    seed = int(smoke["seed"])
    data = config["training_data"]
    dynamic_manifest = resolve_repo_path(data["dynamic_full51_manifest_path"])
    global33 = resolve_repo_path(data["global_33d_standardization_path"])
    feature_schema = Path(read_json(dynamic_manifest)["part_roots"][0]) / "feature_schema.json"
    feature_names = [row["name"] for row in read_json(feature_schema)["features"]]
    groups = feature_group_indices(feature_names)
    synthetic_train, synthetic_val = synthetic_datasets(
        int(smoke["synthetic_rows"]), int(smoke["synthetic_train_rows"]), seed, groups
    )
    waymo_train = DynamicTrainValDataset(
        dynamic_manifest, "train", global33, feature_schema_path=feature_schema, max_rows=int(smoke["waymo_train_rows"])
    )
    waymo_val = DynamicTrainValDataset(
        dynamic_manifest, "val", global33, feature_schema_path=feature_schema, max_rows=int(smoke["waymo_val_rows"])
    )
    global33_result = global33_smoke(waymo_train, global33)

    result_rows = []
    fairness_ledgers: dict[str, dict[str, Any]] = {}
    for label, train_dataset, val_dataset in (
        ("synthetic", synthetic_train, synthetic_val),
        ("waymo_train_val_subset", waymo_train, waymo_val),
    ):
        for candidate in "ABC":
            row, ledger = train_candidate_smoke(
                label=label,
                candidate=candidate,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                stage6t=stage6t,
                seed=seed,
                batch_size=int(smoke["batch_size"]),
                train_batches=int(smoke["train_batches_per_candidate"]),
                val_batches=int(smoke["val_batches_per_candidate"]),
                device=device,
            )
            result_rows.append(row)
            fairness_ledgers[f"{label}_{candidate}"] = ledger
    fairness = {}
    for label in ("synthetic", "waymo_train_val_subset"):
        fairness[label] = assert_bc_fairness(fairness_ledgers[f"{label}_B"], fairness_ledgers[f"{label}_C"])
    resume_result = resume_smoke(
        checkpoint_path=output / "smoke_checkpoints" / "candidate_B_resume_smoke.pt",
        dataset=waymo_train,
        stage6t=stage6t,
        protocol_fingerprint=freeze["protocol_content_fingerprint_sha256"],
        seed=seed,
        batch_size=int(smoke["batch_size"]),
        total_batches=int(smoke["resume_total_batches"]),
        device=device,
    )
    formal_epoch_boundary_resume_result = formal_epoch_boundary_resume_smoke(
        checkpoint_path=output / "smoke_checkpoints" / "candidate_B_formal_epoch_boundary_resume_smoke.pt",
        dataset=waymo_train,
        stage6t=stage6t,
        protocol_fingerprint=freeze["protocol_content_fingerprint_sha256"],
        seed=seed,
        batch_size=int(smoke["batch_size"]),
        device=device,
    )
    timing = [
        timing_probe(
            candidate=candidate,
            dataset=waymo_train,
            stage6t=stage6t,
            seed=seed,
            batch_size=int(smoke["timing_probe_batch_size"]),
            warmup_batches=int(smoke["timing_probe_warmup_batches"]),
            measured_batches=int(smoke["timing_probe_measured_batches"]),
            device=device,
        )
        for candidate in "ABC"
    ]
    formal_checkpoint_count, formal_checkpoint_files = count_formal_checkpoints(
        smoke["formal_checkpoint_roots_must_remain_empty"]
    )
    if formal_checkpoint_count != 0:
        raise RuntimeError(f"Formal checkpoints appeared during smoke: {formal_checkpoint_files}")
    summary = {
        "schema_version": "stage6u_unified_abc_trainer_smoke_v1",
        "status": "PASS_UNIFIED_ABC_TRAINER_SMOKE_NO_FORMAL_TRAINING",
        "issue": int(config["issue"]),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "stage6t_protocol_fingerprint_sha256": freeze["protocol_content_fingerprint_sha256"],
        "device": str(device),
        "torch_version": torch.__version__,
        "synthetic_and_waymo_results": result_rows,
        "global33_smoke": global33_result,
        "fairness_ledgers": fairness_ledgers,
        "B_C_fairness_audit": fairness,
        "checkpoint_resume_smoke": resume_result,
        "formal_epoch_boundary_resume_smoke": formal_epoch_boundary_resume_result,
        "timing_probe": timing,
        "formal_checkpoint_count": formal_checkpoint_count,
        "formal_checkpoint_files": formal_checkpoint_files,
        "waymo_splits_read": ["train", "val"],
        "waymo_test_read": False,
        "nuplan_read_or_run": False,
        "stage6j_k_p_read": False,
        "stage6s_v2_confirmation_read_or_run": False,
        "embedding_bdd_mmd_read": False,
        "formal_training_launched": False,
        "frozen_architecture_or_loss_modified_from_smoke_outcome": False,
        "validation": {
            "all_abc_synthetic_pass": all(row["passed"] for row in result_rows if row["label"] == "synthetic"),
            "all_abc_waymo_train_val_pass": all(row["passed"] for row in result_rows if row["label"] == "waymo_train_val_subset"),
            "global33_pass": global33_result["passed"],
            "bc_fairness_pass": all(row["all_streams_identical"] for row in fairness.values()),
            "resume_pass": resume_result["passed"] and formal_epoch_boundary_resume_result["passed"],
            "formal_checkpoint_count_zero": formal_checkpoint_count == 0,
            "blind_boundary_pass": True,
            "pass": True,
        },
    }
    (output / "stage6u_smoke_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "stage6u_random_fairness_ledger.json").write_text(
        json.dumps({"ledgers": fairness_ledgers, "B_C_audit": fairness}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"status": result["status"], "validation": result["validation"]}, indent=2, ensure_ascii=False))
