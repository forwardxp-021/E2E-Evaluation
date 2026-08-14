#!/usr/bin/env python3
"""Freeze the validated Stage6U unified-trainer implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage6u_unified_abc_trainer import (
    DynamicTrainValDataset,
    assert_bc_fairness,
    build_encoder,
    build_random_plan,
    derive_dataset_statistics,
    encoder_parameter_count,
    load_and_validate_implementation_config,
    random_plan_ledger,
    read_json,
    resolve_repo_path,
    sha256_file,
    sha256_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _git_output(*args: str) -> str:
    import subprocess

    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    return result.stdout.strip()


def _record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "size_bytes": int(path.stat().st_size)}


def _formal_checkpoint_count(paths: list[str]) -> tuple[int, list[str]]:
    files = []
    for text in paths:
        root = resolve_repo_path(text)
        if root.exists():
            files.extend(str(path) for path in root.rglob("*.pt"))
    return len(files), files


def _full_train_bc_random_plan_audit(
    config: dict[str, Any], stage6t: dict[str, Any]
) -> dict[str, Any]:
    manifest_path = resolve_repo_path(config["training_data"]["dynamic_full51_manifest_path"])
    standardization_path = resolve_repo_path(config["training_data"]["global_33d_standardization_path"])
    manifest = read_json(manifest_path)
    feature_schema = Path(manifest["part_roots"][0]) / "feature_schema.json"
    dataset = DynamicTrainValDataset(
        manifest_path, "train", standardization_path, feature_schema_path=feature_schema, cache_shards=2
    )
    statistics_started = time.perf_counter()
    statistics = derive_dataset_statistics(dataset)
    statistics_seconds = time.perf_counter() - statistics_started
    ledgers = {}
    generation_seconds = {}
    for candidate in ("B", "C"):
        candidate_config = stage6t["candidates"][candidate]
        dropout = stage6t["dropout_packages"][candidate_config["dropout_package"]]
        objective = stage6t["objective_packages"][candidate_config["objective_package"]]
        started = time.perf_counter()
        plan = build_random_plan(
            dataset,
            seed=int(stage6t["common_optimization"]["primary_seed"]),
            pair_seed=int(stage6t["common_optimization"]["pair_seed"]),
            epoch=0,
            epoch_samples=int(stage6t["common_optimization"]["epoch_samples"]),
            batch_size=int(stage6t["common_optimization"]["batch_size"]),
            candidate=candidate,
            sampling_package=candidate_config["sampling_package"],
            dropout_package=candidate_config["dropout_package"],
            slot_dropout_probability=float(dropout["slot_dropout_probability"]),
            all_neighbor_dropout_probability=float(dropout["all_neighbor_dropout_probability"]),
            ranking_margin=float(objective["ranking_margin"]),
            statistics=statistics,
        )
        generation_seconds[candidate] = time.perf_counter() - started
        ledgers[candidate] = random_plan_ledger(plan)
    audit = assert_bc_fairness(ledgers["B"], ledgers["C"])
    return {
        "dataset_rows": len(dataset),
        "seed": int(stage6t["common_optimization"]["primary_seed"]),
        "pair_seed": int(stage6t["common_optimization"]["pair_seed"]),
        "epoch": 0,
        "epoch_samples": int(stage6t["common_optimization"]["epoch_samples"]),
        "batch_size": int(stage6t["common_optimization"]["batch_size"]),
        "statistics_generation_seconds": statistics_seconds,
        "plan_generation_seconds": generation_seconds,
        "B_ledger": ledgers["B"],
        "C_ledger": ledgers["C"],
        "audit": audit,
        "passed": True,
    }


def _report(path: Path, manifest: dict[str, Any]) -> None:
    timing = manifest["training_time_estimate"]
    lines = [
        "# Stage 6U Unified A/B/C Trainer实现冻结报告",
        "",
        f"## 结论：{manifest['status']}",
        "",
        "统一trainer、B/C公平随机计划、global33读取、synthetic与Waymo train/val smoke、checkpoint恢复均已通过。",
        "本状态只表示可以单独授权正式训练；本阶段没有启动9个训练任务，也没有读取Waymo test或任何nuPlan结果。",
        "",
        "## 1. Unified trainer",
        "",
        "- 单一代码路径按candidate配置构造A/B/C；A/B/C均输入83D、输出64D。",
        "- encoder参数量A/B/C分别为106560/106560/105616，C/B=0.991141。",
        "- A使用legacy objective；B/C共享clean longitudinal、ranking、sampling、dropout和预算，仅encoder topology不同。",
        "",
        "## 2. B/C公平性",
        "",
        "- sampling weights、样本顺序、batch边界、positive/negative pair、pair type、slot/all-neighbor dropout、augmentation seed、optimizer schedule和budget逐项SHA一致。",
        f"- synthetic shared fingerprint：`{manifest['fairness_audit']['synthetic']['shared_fingerprint_sha256']}`。",
        f"- Waymo subset shared fingerprint：`{manifest['fairness_audit']['waymo_train_val_subset']['shared_fingerprint_sha256']}`。",
        f"- 全量135046行epoch-0 shared fingerprint：`{manifest['full_train_B_C_random_plan_audit']['audit']['shared_fingerprint_sha256']}`。",
        "- random plan只依赖seed/epoch/pair_seed与训练数据，不依赖candidate或model输出。",
        "",
        "## 3. 数据与盲测边界",
        "",
        "- 33D监督只读取`interaction_feat_style_raw.npy`并应用Stage6T冻结global train mean/std；part-local标准化数组禁止作为trainer输入。",
        "- Dynamic v2只读；dataset API只接受train/val，test直接报错。",
        "- formal模式需要另一个绑定本freeze SHA的授权manifest；缺失授权时fail closed。",
        "- 未读取Stage6J/K/P、nuPlan embedding、BDD/MMD或Stage6S-v2 confirmation。",
        "",
        "## 4. Smoke与resume",
        "",
        "- synthetic A/B/C与Dynamic v2小规模train/val A/B/C均forward/backward通过，embedding均为64D，全部loss/gradient finite。",
        "- checkpoint恢复了epoch、next batch、global step、optimizer、constant scheduler、Python/NumPy/Torch RNG和random-plan ledger。",
        "- 连续训练与中断恢复的loss序列和最终model state SHA逐位一致。",
        "- formal epoch边界resume已单独验证：下一epoch不再与上一epochplan误比较；正式train/val均实际显示tqdm。",
        f"- 正式checkpoint仍为{manifest['formal_checkpoint_count']}/9。",
        "",
        "## 5. 训练时间估计",
        "",
        f"- A：单seed最大30 epoch约{timing['per_seed_max30_hours']['A']:.1f}小时。",
        f"- B：单seed最大30 epoch约{timing['per_seed_max30_hours']['B']:.1f}小时。",
        f"- C：单seed最大30 epoch约{timing['per_seed_max30_hours']['C']:.1f}小时。",
        f"- 单个seed串行跑A+B+C约{timing['one_seed_ABC_serial_hours']:.1f}小时；9任务全部串行约{timing['nine_tasks_serial_hours']:.1f}小时。",
        f"- 加入val、checkpoint和I/O后的建议计划区间为{timing['recommended_wallclock_range_hours'][0]:.0f}–{timing['recommended_wallclock_range_hours'][1]:.0f}小时；早停通常会缩短。",
        "- 估计来自小规模MPS smoke线性外推，正式首个epoch后必须用实测更新时间。",
        "",
        "## 6. 授权结论",
        "",
        "当前已经具备另行授权启动A/B/C×3 seeds共9个正式任务的技术条件，但本freeze本身不授权训练。",
        f"`formal_training_authorized={manifest['formal_training_authorized']}`，`formal_training_launched={manifest['formal_training_launched']}`。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    config, stage6t, stage6t_freeze = load_and_validate_implementation_config(config_path)
    smoke_dir = args.smoke_dir.resolve()
    smoke_path = smoke_dir / "stage6u_smoke_summary.json"
    fairness_path = smoke_dir / "stage6u_random_fairness_ledger.json"
    smoke = read_json(smoke_path)
    if smoke.get("status") != config["freeze"]["required_smoke_status"]:
        raise ValueError(f"Stage6U smoke status is not PASS: {smoke.get('status')}")
    if not smoke.get("validation") or not all(smoke["validation"].values()):
        raise ValueError("One or more Stage6U smoke gates failed")
    if smoke.get("waymo_test_read") is not False or smoke.get("nuplan_read_or_run") is not False:
        raise ValueError("Stage6U smoke violated the blind boundary")
    if smoke.get("embedding_bdd_mmd_read") is not False or smoke.get("stage6s_v2_confirmation_read_or_run") is not False:
        raise ValueError("Stage6U smoke read forbidden evaluation information")
    if smoke.get("formal_training_launched") is not False:
        raise ValueError("Formal training unexpectedly launched")
    if smoke.get("stage6t_protocol_fingerprint_sha256") != stage6t_freeze["protocol_content_fingerprint_sha256"]:
        raise ValueError("Smoke is not bound to the frozen Stage6T protocol")

    source_paths = {
        "trainer": REPO_ROOT / "tools/stage6u_unified_abc_trainer.py",
        "smoke_runner": REPO_ROOT / "tools/stage6u_smoke_unified_abc_trainer.py",
        "freeze_tool": REPO_ROOT / "tools/stage6u_freeze_trainer_implementation.py",
        "stage6u_config": config_path,
        "stage6t_config": resolve_repo_path(config["stage6t_protocol"]["config_path"]),
        "stage6t_freeze_manifest": resolve_repo_path(config["stage6t_protocol"]["freeze_manifest_path"]),
        "dynamic_manifest": resolve_repo_path(config["training_data"]["dynamic_full51_manifest_path"]),
        "global33_standardization": resolve_repo_path(config["training_data"]["global_33d_standardization_path"]),
        "smoke_summary": smoke_path,
        "fairness_ledger": fairness_path,
    }
    source_records = {name: _record(path) for name, path in source_paths.items()}
    parameter_counts = {candidate: encoder_parameter_count(build_encoder(candidate)) for candidate in "ABC"}
    if parameter_counts != {"A": 106560, "B": 106560, "C": 105616}:
        raise ValueError(f"Encoder parameter counts changed: {parameter_counts}")
    formal_count, formal_files = _formal_checkpoint_count(config["smoke"]["formal_checkpoint_roots_must_remain_empty"])
    if formal_count != int(config["freeze"]["formal_checkpoint_count_required"]):
        raise ValueError(f"Formal checkpoint count must remain zero, found: {formal_files}")
    full_train_fairness = _full_train_bc_random_plan_audit(config, stage6t)
    timing_rows = {row["candidate"]: row for row in smoke["timing_probe"]}
    per_seed = {candidate: float(row["estimated_max30_hours"]) for candidate, row in timing_rows.items()}
    serial = sum(per_seed.values()) * 3.0
    timing = {
        "method": "MPS small-Waymo train subset forward/backward timing at the frozen formal batch size 128",
        "per_seed_max30_hours": per_seed,
        "one_seed_ABC_serial_hours": sum(per_seed.values()),
        "nine_tasks_serial_hours": serial,
        "recommended_wallclock_range_hours": [serial * 1.15, serial * 1.40],
        "early_stopping_may_reduce_time": True,
        "update_after_first_formal_epoch_required": True,
    }
    implementation_fingerprint = sha256_json(
        {
            "source_records": source_records,
            "stage6t_protocol_fingerprint": stage6t_freeze["protocol_content_fingerprint_sha256"],
            "parameter_counts": parameter_counts,
            "fairness": smoke["B_C_fairness_audit"],
            "full_train_fairness": full_train_fairness,
            "environment": {"python": sys.version, "torch": torch.__version__, "platform": platform.platform()},
        }
    )
    output = args.output_dir.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)
    output.mkdir(parents=True)
    manifest = {
        "schema_version": "stage6u_trainer_implementation_freeze_v1",
        "status": config["freeze"]["ready_status"],
        "issue": int(config["issue"]),
        "implementation_id": config["implementation_id"],
        "implementation_fingerprint_sha256": implementation_fingerprint,
        "source_records": source_records,
        "stage6t_protocol_fingerprint_sha256": stage6t_freeze["protocol_content_fingerprint_sha256"],
        "candidate_config_sha256": {
            candidate: sha256_json(
                {
                    "candidate": stage6t["candidates"][candidate],
                    "architecture": stage6t["architecture_definitions"][
                        stage6t["candidates"][candidate]["architecture"]
                    ],
                    "sampling": stage6t["sampling_packages"][
                        stage6t["candidates"][candidate]["sampling_package"]
                    ],
                    "dropout": stage6t["dropout_packages"][
                        stage6t["candidates"][candidate]["dropout_package"]
                    ],
                    "objective": stage6t["objective_packages"][
                        stage6t["candidates"][candidate]["objective_package"]
                    ],
                    "optimization": stage6t["common_optimization"],
                    "dataset_contract": stage6t["dataset_contract"],
                }
            )
            for candidate in "ABC"
        },
        "encoder_parameter_counts": parameter_counts,
        "C_B_encoder_parameter_ratio": parameter_counts["C"] / parameter_counts["B"],
        "fairness_audit": smoke["B_C_fairness_audit"],
        "full_train_B_C_random_plan_audit": full_train_fairness,
        "environment": {
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "torch": torch.__version__,
            "mps_built": bool(torch.backends.mps.is_built()),
            "mps_available": bool(torch.backends.mps.is_available()),
            "platform": platform.platform(),
            "git_commit_at_freeze": _git_output("rev-parse", "HEAD"),
            "git_branch_at_freeze": _git_output("branch", "--show-current"),
        },
        "smoke_status": smoke["status"],
        "smoke_validation": smoke["validation"],
        "resume_validation": smoke["checkpoint_resume_smoke"]["checks"],
        "formal_epoch_boundary_resume_validation": smoke["formal_epoch_boundary_resume_smoke"]["checks"],
        "training_time_estimate": timing,
        "formal_checkpoint_count": formal_count,
        "planned_formal_checkpoint_count": int(config["freeze"]["formal_checkpoint_count_planned"]),
        "formal_checkpoint_files": formal_files,
        "formal_training_authorized": False,
        "formal_checkpoint_write_authorized": False,
        "formal_training_launched": False,
        "waymo_test_read": False,
        "nuplan_read_or_run": False,
        "stage6s_v2_confirmation_read_or_run": False,
        "embedding_bdd_mmd_read": False,
        "dynamic_v2_modified": False,
        "stage6t_modified": False,
        "stage6s_v2_roster_modified": False,
        "old64_modified": False,
        "ready_for_separate_formal_training_authorization": True,
        "next_authorized_action": "REQUEST_SEPARATE_AUTHORIZATION_MANIFEST_BOUND_TO_THIS_IMPLEMENTATION_FREEZE_SHA256",
        "validation": {
            "source_hashes_pass": True,
            "stage6t_binding_pass": True,
            "unified_abc_smoke_pass": True,
            "bc_fairness_pass": True,
            "full_train_bc_random_plan_pass": full_train_fairness["passed"],
            "global33_pass": True,
            "resume_pass": True,
            "formal_epoch_boundary_resume_pass": smoke["formal_epoch_boundary_resume_smoke"]["passed"],
            "blind_boundary_pass": True,
            "formal_checkpoint_count_zero": True,
            "pass": True,
        },
    }
    manifest_path = output / "stage6u_trainer_implementation_freeze_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _report(output / "stage6u_trainer_implementation_freeze_report_zh.md", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"status": result["status"], "validation": result["validation"]}, indent=2, ensure_ascii=False))
    random_plan_ledger,
