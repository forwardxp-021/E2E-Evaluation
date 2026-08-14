#!/usr/bin/env python3
"""Unified Stage6U A/B/C trainer with candidate-independent random plans.

The same implementation handles all candidates. Formal mode fails closed
unless a separate authorization manifest explicitly binds the frozen Stage6U
implementation. Smoke mode accepts only train/val rows and never reads test,
nuPlan, embeddings, BDD, or MMD.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage5d_context_core import NEIGHBOR_CHANNELS, SLOT_NAMES
from tools.train_context_behavior_embedding import FEATURE_GROUPS


REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_SPLITS = {"train", "val"}
FORBIDDEN_INFORMATION_TOKENS = ("nuplan", "bdd", "mmd", "stage6j", "stage6k", "stage6p", "stage6s")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _require_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected={expected}, actual={actual}, path={path}")


def load_and_validate_implementation_config(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = read_json(config_path)
    if config.get("schema_version") != "stage6u_unified_abc_trainer_implementation_v1":
        raise ValueError("Unexpected Stage6U implementation schema")
    if int(config.get("issue", -1)) != 263:
        raise ValueError("Stage6U must reference Issue #263")
    auth = config.get("authorization", {})
    for key in (
        "formal_training_authorized", "formal_checkpoint_write_authorized", "waymo_test_authorized",
        "nuplan_authorized", "stage6s_v2_confirmation_authorized",
    ):
        if auth.get(key) is not False:
            raise ValueError(f"authorization.{key} must remain false during implementation freeze")
    stage6t_record = config["stage6t_protocol"]
    stage6t_config_path = resolve_repo_path(stage6t_record["config_path"])
    freeze_path = resolve_repo_path(stage6t_record["freeze_manifest_path"])
    _require_sha(stage6t_config_path, stage6t_record["config_sha256"], "Stage6T config")
    _require_sha(freeze_path, stage6t_record["freeze_manifest_sha256"], "Stage6T freeze manifest")
    stage6t = read_json(stage6t_config_path)
    freeze = read_json(freeze_path)
    if freeze.get("protocol_content_fingerprint_sha256") != stage6t_record["protocol_content_fingerprint_sha256"]:
        raise ValueError("Stage6T protocol fingerprint changed")
    if freeze.get("status") != "FROZEN_READY_FOR_ABC_TRAINER_IMPLEMENTATION_NOT_TRAINING":
        raise ValueError("Stage6T is not ready for trainer implementation")
    if freeze.get("training_authorized") is not False or freeze.get("waymo_test_authorized") is not False:
        raise ValueError("Stage6T implementation boundary changed")
    data = config["training_data"]
    manifest_path = resolve_repo_path(data["dynamic_full51_manifest_path"])
    standardization_path = resolve_repo_path(data["global_33d_standardization_path"])
    _require_sha(manifest_path, data["dynamic_full51_manifest_sha256"], "Dynamic v2 manifest")
    _require_sha(standardization_path, data["global_33d_standardization_sha256"], "Stage6T global33 standardization")
    if set(data.get("allowed_splits", [])) != ALLOWED_SPLITS or data.get("forbidden_split") != "test":
        raise ValueError("Trainer must allow only train/val and forbid test")
    if data.get("allowed_33d_source") != "interaction_feat_style_raw.npy":
        raise ValueError("Only interaction_feat_style_raw.npy is allowed")
    if data.get("forbidden_33d_source") != "interaction_feat_style.npy":
        raise ValueError("Part-local interaction_feat_style.npy must be explicitly forbidden")
    if data.get("modify_or_rewrite_shards") is not False:
        raise ValueError("Dynamic v2 shards must remain read-only")
    if config["operational_semantics"].get("B_C_random_plan_candidate_independent") is not True:
        raise ValueError("B/C random plans must be candidate-independent")
    return config, stage6t, freeze


def assert_blind_path(path: Path, *, allow_stage6t_protocol: bool = False) -> None:
    text = str(path).lower()
    for token in FORBIDDEN_INFORMATION_TOKENS:
        if token in text and not (allow_stage6t_protocol and "stage6t" in text):
            raise ValueError(f"Forbidden blind-evaluation path token '{token}' in trainer input: {path}")


class LegacySingleGRUEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(83, 128, batch_first=True)
        self.proj = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 64))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(context)
        return self.proj(hidden[-1])


class PartitionedSingleGRUEncoder(LegacySingleGRUEncoder):
    """Same topology as legacy, with frozen training views z[:16]/z[16:]."""


class DualBranchEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ego_gru = nn.GRU(8, 48, batch_first=True)
        self.ego_proj = nn.Sequential(nn.Linear(48, 48), nn.ReLU(), nn.Linear(48, 16))
        self.context_gru = nn.GRU(83, 120, batch_first=True)
        self.context_proj = nn.Sequential(nn.Linear(120, 120), nn.ReLU(), nn.Linear(120, 48))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        _, ego_hidden = self.ego_gru(context[:, :, :8])
        _, context_hidden = self.context_gru(context)
        return torch.cat((self.ego_proj(ego_hidden[-1]), self.context_proj(context_hidden[-1])), dim=1)


def encoder_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def build_encoder(candidate: str) -> nn.Module:
    if candidate == "A":
        return LegacySingleGRUEncoder()
    if candidate == "B":
        return PartitionedSingleGRUEncoder()
    if candidate == "C":
        return DualBranchEncoder()
    raise ValueError(f"Unknown candidate: {candidate}; expected A/B/C")


def _feature_names(schema_path: Path) -> list[str]:
    schema = read_json(schema_path)
    names = schema.get("feature_names")
    if not names:
        names = [row["name"] for row in schema.get("features", [])]
    if len(names) != 33:
        raise ValueError(f"Expected 33 feature names, got {len(names)} from {schema_path}")
    return [str(name) for name in names]


def feature_group_indices(feature_names: Sequence[str]) -> dict[str, list[int]]:
    index = {name: position for position, name in enumerate(feature_names)}
    result: dict[str, list[int]] = {}
    for group, names in FEATURE_GROUPS.items():
        missing = [name for name in names if name not in index]
        if missing:
            raise ValueError(f"Feature schema missing {group}: {missing}")
        result[group] = [index[name] for name in names]
    return result


def pairwise_euclidean(values: torch.Tensor, *, squared: bool = False) -> torch.Tensor:
    """MPS-safe equivalent of torch.cdist(values, values, p=2)."""
    squared_norm = torch.sum(values * values, dim=1, keepdim=True)
    distance_squared = torch.clamp(squared_norm + squared_norm.T - 2.0 * (values @ values.T), min=0.0)
    if squared:
        return distance_squared
    distance = torch.sqrt(distance_squared + 1e-12)
    diagonal_mask = 1.0 - torch.eye(len(values), device=values.device, dtype=values.dtype)
    return distance * diagonal_mask


def soft_contrastive_loss(
    z: torch.Tensor,
    features: torch.Tensor,
    temperature: float = 0.1,
    feature_temperature: float = 1.0,
) -> torch.Tensor:
    normalized_z = F.normalize(z, dim=1, eps=1e-8)
    normalized_features = F.normalize(features, dim=1, eps=1e-8)
    logits = (normalized_z @ normalized_z.T) / temperature
    target_logits = -pairwise_euclidean(normalized_features, squared=True) / feature_temperature
    diagonal = torch.eye(z.shape[0], device=z.device, dtype=torch.bool)
    logits = logits.masked_fill(diagonal, -1e9)
    target_logits = target_logits.masked_fill(diagonal, -1e9)
    return -(F.softmax(target_logits, dim=1) * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def metric_alignment_loss(z: torch.Tensor, target: torch.Tensor, loss_type: str = "huber") -> torch.Tensor:
    distance_z = pairwise_euclidean(F.normalize(z, dim=1, eps=1e-8))
    distance_target = pairwise_euclidean(target)
    off_diagonal = ~torch.eye(z.size(0), dtype=torch.bool, device=z.device)
    distance_z = distance_z[off_diagonal]
    distance_target = distance_target[off_diagonal]
    distance_z = (distance_z - distance_z.mean()) / (distance_z.std(unbiased=False) + 1e-8)
    distance_target = (distance_target - distance_target.mean()) / (distance_target.std(unbiased=False) + 1e-8)
    if loss_type == "mse":
        return F.mse_loss(distance_z, distance_target)
    if loss_type == "huber":
        return F.smooth_l1_loss(distance_z, distance_target)
    raise ValueError(f"Unsupported metric loss type: {loss_type}")


class UnifiedABCModel(nn.Module):
    def __init__(self, candidate: str, group_indices: Mapping[str, Sequence[int]]) -> None:
        super().__init__()
        self.candidate = candidate
        self.encoder = build_encoder(candidate)
        if candidate == "A":
            input_dims = {key: 64 for key in ("longitudinal", "following", "lateral_dynamics", "lateral_gap", "behavior_proxy")}
        else:
            input_dims = {
                "longitudinal": 16,
                "following": 48,
                "lateral_dynamics": 48,
                "lateral_gap": 48,
                "behavior_proxy": 64,
                "clean_longitudinal": 16,
            }
        target_dims = {
            "longitudinal": len(group_indices["longitudinal_comfort"]),
            "following": len(group_indices["following_interaction"]),
            "lateral_dynamics": len(group_indices["lateral_dynamics"]),
            "lateral_gap": len(group_indices["lateral_gap_interaction"]),
            "behavior_proxy": len(group_indices["behavior_proxy"]),
            "clean_longitudinal": 6,
        }
        self.heads = nn.ModuleDict(
            {name: nn.Linear(input_dim, target_dims[name]) for name, input_dim in input_dims.items()}
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        z = self.encoder(context)
        if z.ndim != 2 or z.shape[1] != 64:
            raise RuntimeError(f"Candidate {self.candidate} must export [B,64], got {list(z.shape)}")
        return z

    @staticmethod
    def longitudinal_view(z: torch.Tensor) -> torch.Tensor:
        return z[:, :16]

    @staticmethod
    def context_view(z: torch.Tensor) -> torch.Tensor:
        return z[:, 16:64]


@dataclass(frozen=True)
class RowReference:
    shard_id: int
    local_index: int
    scenario_id: str
    target_agent_id: str
    start: int


class DynamicTrainValDataset:
    """Read-only sharded dataset that never opens test or part-local 33D targets."""

    def __init__(
        self,
        manifest_path: Path,
        split: str,
        standardization_path: Path,
        *,
        feature_schema_path: Path | None = None,
        max_rows: int | None = None,
        cache_shards: int = 1,
    ) -> None:
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"Trainer split must be train or val; refusing '{split}'")
        assert_blind_path(manifest_path)
        assert_blind_path(standardization_path, allow_stage6t_protocol=True)
        self.manifest_path = manifest_path.resolve()
        self.manifest = read_json(self.manifest_path)
        self.split = split
        self.shards = [Path(path) for path in self.manifest["shard_paths"]]
        self.cache_shards = max(1, int(cache_shards))
        self._cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        stats = read_json(standardization_path)
        if stats.get("source_array") != "interaction_feat_style_raw.npy" or stats.get("fit_split") != "train":
            raise ValueError("Global33 standardization provenance is invalid")
        if stats.get("part_local_interaction_feat_style_npy_allowed_for_stage6t_training") is not False:
            raise ValueError("Global33 record does not forbid part-local standardized targets")
        self.raw33_mean = np.asarray(stats["mean"], dtype=np.float32)
        self.raw33_std = np.maximum(np.asarray(stats["std"], dtype=np.float32), float(stats["epsilon_floor"]))
        if self.raw33_mean.shape != (33,) or self.raw33_std.shape != (33,):
            raise ValueError("Global33 standardization must have 33 mean/std entries")
        self.rows: list[RowReference] = []
        for shard_id, shard in enumerate(self.shards):
            split_path = shard / "split.npy"
            meta_path = shard / "meta.npy"
            split_values = np.load(split_path, allow_pickle=True).astype(str)
            meta = np.load(meta_path, allow_pickle=True)
            if not np.array_equal(split_values, meta["split"].astype(str)):
                raise ValueError(f"Split mismatch in {shard}")
            selected = np.flatnonzero(split_values == split)
            for local_index in selected:
                row = meta[int(local_index)]
                self.rows.append(
                    RowReference(
                        shard_id=shard_id,
                        local_index=int(local_index),
                        scenario_id=str(row["scenario_id"]),
                        target_agent_id=str(row["target_agent_id"]),
                        start=int(row["start"]),
                    )
                )
                if max_rows is not None and len(self.rows) >= int(max_rows):
                    break
            if max_rows is not None and len(self.rows) >= int(max_rows):
                break
        if not self.rows:
            raise ValueError(f"No {split} rows found in Dynamic v2")
        if feature_schema_path is None:
            feature_schema_path = Path(self.manifest["part_roots"][0]) / "feature_schema.json"
        self.feature_schema_path = feature_schema_path.resolve()
        self.feature_names = _feature_names(self.feature_schema_path)
        self.group_indices = feature_group_indices(self.feature_names)

    def __len__(self) -> int:
        return len(self.rows)

    def _load_shard(self, shard_id: int) -> dict[str, np.ndarray]:
        if shard_id in self._cache:
            self._cache.move_to_end(shard_id)
            return self._cache[shard_id]
        shard = self.shards[shard_id]
        forbidden = shard / "interaction_feat_style.npy"
        if not forbidden.is_file():
            raise FileNotFoundError(f"Expected frozen diagnostic target is missing: {forbidden}")
        arrays = {
            "context": np.load(shard / "context_traj.npy", mmap_mode="r"),
            "raw33": np.load(shard / "interaction_feat_style_raw.npy", mmap_mode="r"),
            "longitudinal": np.load(shard / "longitudinal_supervision_v2.npy", mmap_mode="r"),
            "slot_valid": np.load(shard / "slot_valid_mask.npy", mmap_mode="r"),
        }
        self._cache[shard_id] = arrays
        if len(self._cache) > self.cache_shards:
            self._cache.popitem(last=False)
        return arrays

    @staticmethod
    def _clean_longitudinal_window_target(longitudinal: np.ndarray) -> np.ndarray:
        speed = longitudinal[:, 0]
        accel = longitudinal[:, 1]
        jerk = longitudinal[:, 2]
        return np.asarray(
            [
                np.mean(speed), np.std(speed), np.sqrt(np.mean(np.square(accel))),
                np.quantile(np.abs(accel), 0.90), np.sqrt(np.mean(np.square(jerk))),
                np.quantile(np.abs(jerk), 0.90),
            ],
            dtype=np.float32,
        )

    def get(self, index: int) -> dict[str, Any]:
        ref = self.rows[int(index)]
        arrays = self._load_shard(ref.shard_id)
        context = np.asarray(arrays["context"][ref.local_index], dtype=np.float32)
        raw33 = np.asarray(arrays["raw33"][ref.local_index], dtype=np.float32)
        longitudinal = np.asarray(arrays["longitudinal"][ref.local_index], dtype=np.float32)
        slot_valid = np.asarray(arrays["slot_valid"][ref.local_index], dtype=bool)
        if context.shape != (80, 83) or raw33.shape != (33,) or longitudinal.shape != (80, 3) or slot_valid.shape != (5, 80):
            raise ValueError(
                f"Dynamic row shape mismatch: context={context.shape}, raw33={raw33.shape}, "
                f"longitudinal={longitudinal.shape}, slot_valid={slot_valid.shape}"
            )
        if not all(np.isfinite(value).all() for value in (context, raw33, longitudinal)):
            raise ValueError(f"Non-finite Stage6U input at dataset index {index}")
        standardized33 = (raw33 - self.raw33_mean) / self.raw33_std
        clean_target = self._clean_longitudinal_window_target(longitudinal)
        speed_mean = float(np.mean(context[:, 5]))
        speed_bin = 0 if speed_mean < 5.0 else (1 if speed_mean < 15.0 else 2)
        front_ratio = float(slot_valid[0].mean())
        front_regime = 0 if front_ratio < 1e-6 else (1 if front_ratio < 0.5 else 2)
        lateral_nuisance = int(raw33[13] > 0.0)
        nuisance = np.asarray(
            [standardized33[13], standardized33[14], standardized33[18], standardized33[19], standardized33[20]],
            dtype=np.float32,
        )
        return {
            "context": context,
            "raw33": raw33,
            "feat33": standardized33.astype(np.float32),
            "clean_longitudinal": clean_target,
            "slot_valid": slot_valid,
            "stratum": (speed_bin, front_regime, lateral_nuisance),
            "nuisance": nuisance,
            "row_key": f"{ref.scenario_id}|{ref.target_agent_id}|{ref.start}",
            "dataset_index": int(index),
        }


class InMemoryTrainValDataset:
    def __init__(self, rows: list[dict[str, Any]], group_indices: Mapping[str, Sequence[int]]) -> None:
        self.rows = rows
        self.group_indices = {key: list(value) for key, value in group_indices.items()}

    def __len__(self) -> int:
        return len(self.rows)

    def get(self, index: int) -> dict[str, Any]:
        return self.rows[int(index)]


def collate_rows(dataset: Any, indices: np.ndarray) -> dict[str, Any]:
    rows = [dataset.get(int(index)) for index in indices]
    return {
        "context": torch.from_numpy(np.stack([row["context"] for row in rows])),
        "feat33": torch.from_numpy(np.stack([row["feat33"] for row in rows])),
        "clean_longitudinal": torch.from_numpy(np.stack([row["clean_longitudinal"] for row in rows])),
        "slot_valid": np.stack([row["slot_valid"] for row in rows]),
        "nuisance": np.stack([row["nuisance"] for row in rows]),
        "strata": np.asarray([row["stratum"] for row in rows], dtype=np.int64),
        "row_keys": [row["row_key"] for row in rows],
        "dataset_indices": np.asarray([row["dataset_index"] for row in rows], dtype=np.int64),
    }


def derive_dataset_statistics(dataset: Any) -> dict[str, np.ndarray]:
    clean = []
    nuisance = []
    strata = []
    row_keys = []
    for index in range(len(dataset)):
        row = dataset.get(index)
        clean.append(row["clean_longitudinal"])
        nuisance.append(row["nuisance"])
        strata.append(row["stratum"])
        row_keys.append(row["row_key"])
    clean_array = np.asarray(clean, dtype=np.float32)
    nuisance_array = np.asarray(nuisance, dtype=np.float32)
    strata_array = np.asarray(strata, dtype=np.int64)
    counts = Counter(map(tuple, strata_array.tolist()))
    raw_weights = np.asarray([1.0 / math.sqrt(counts[tuple(row)]) for row in strata_array], dtype=np.float64)
    raw_weights /= raw_weights.mean()
    weights = np.clip(raw_weights, 0.25, 4.0)
    weights /= weights.sum()
    stratum_members: dict[tuple[int, ...], np.ndarray] = {}
    for stratum in sorted(set(map(tuple, strata_array.tolist()))):
        members = np.flatnonzero(np.all(strata_array == np.asarray(stratum), axis=1))
        priorities = np.asarray(
            [hashlib.sha256(f"93407|{row_keys[int(index)]}".encode("utf-8")).hexdigest() for index in members]
        )
        stratum_members[stratum] = members[np.argsort(priorities, kind="stable")]
    return {
        "clean": clean_array,
        "nuisance": nuisance_array,
        "strata": strata_array,
        "weights": weights,
        "stratum_members": stratum_members,
    }


def _candidate_pool(
    anchor: int,
    strata: np.ndarray,
    stratum_members: Mapping[tuple[int, ...], np.ndarray],
    max_candidates: int,
) -> np.ndarray:
    same = stratum_members[tuple(strata[anchor].tolist())]
    same = same[same != anchor]
    if len(same) == 0:
        same = np.arange(len(strata), dtype=np.int64)
        same = same[same != anchor]
    return np.asarray(same[:max_candidates], dtype=np.int64)


def build_random_plan(
    dataset: Any,
    *,
    seed: int,
    pair_seed: int,
    epoch: int,
    epoch_samples: int,
    batch_size: int,
    candidate: str,
    sampling_package: str,
    dropout_package: str,
    slot_dropout_probability: float,
    all_neighbor_dropout_probability: float,
    ranking_margin: float,
    statistics: Mapping[str, np.ndarray] | None = None,
    sample_without_replacement: bool = False,
) -> dict[str, np.ndarray | int | float | str]:
    if candidate not in {"A", "B", "C"}:
        raise ValueError(candidate)
    statistics = derive_dataset_statistics(dataset) if statistics is None else statistics
    plan_seed = int(seed) * 1_000_003 + int(epoch) * 10_007 + int(pair_seed)
    rng = np.random.default_rng(plan_seed)
    if sample_without_replacement:
        if epoch_samples > len(dataset):
            raise ValueError("Without-replacement plan cannot exceed dataset size")
        sample_indices = rng.permutation(len(dataset))[:epoch_samples].astype(np.int64)
        sampling_weights = np.asarray(statistics["weights"], dtype=np.float64)
    elif sampling_package == "legacy_uniform_v1":
        repeats = math.ceil(epoch_samples / len(dataset))
        orders = [rng.permutation(len(dataset)) for _ in range(repeats)]
        sample_indices = np.concatenate(orders)[:epoch_samples].astype(np.int64)
        sampling_weights = np.full(len(dataset), 1.0 / len(dataset), dtype=np.float64)
    elif sampling_package == "dynamic_longitudinal_v2":
        sampling_weights = statistics["weights"]
        sample_indices = rng.choice(len(dataset), size=epoch_samples, replace=True, p=sampling_weights).astype(np.int64)
    else:
        raise ValueError(f"Unknown sampling package: {sampling_package}")
    batch_count = math.ceil(epoch_samples / batch_size)
    batch_offsets = np.arange(0, batch_count * batch_size + 1, batch_size, dtype=np.int64)
    batch_offsets[-1] = epoch_samples
    augmentation_seeds = rng.integers(0, np.iinfo(np.int64).max, size=batch_count, dtype=np.int64)
    slot_masks = np.zeros((epoch_samples, 5), dtype=bool)
    all_neighbor_masks = np.zeros(epoch_samples, dtype=bool)
    positives = sample_indices.copy()
    negatives = sample_indices.copy()
    pair_types = np.zeros(epoch_samples, dtype=np.int8)
    if dropout_package == "dynamic_mask_aware_v2":
        slot_masks = rng.random((epoch_samples, 5)) < slot_dropout_probability
        all_neighbor_masks = rng.random(epoch_samples) < all_neighbor_dropout_probability
        slot_masks[all_neighbor_masks] = True
    elif dropout_package != "none":
        raise ValueError(f"Unknown dropout package: {dropout_package}")
    if sampling_package == "dynamic_longitudinal_v2":
        clean = statistics["clean"]
        nuisance = statistics["nuisance"]
        strata = statistics["strata"]
        hard_count = epoch_samples // 2
        near_count = epoch_samples // 4
        pair_types = np.concatenate(
            (
                np.zeros(hard_count, dtype=np.int8),
                np.ones(near_count, dtype=np.int8),
                np.full(epoch_samples - hard_count - near_count, 2, dtype=np.int8),
            )
        )
        rng.shuffle(pair_types)
        for position, anchor in enumerate(sample_indices):
            pool = _candidate_pool(int(anchor), strata, statistics["stratum_members"], 64)
            longitudinal_distance = np.linalg.norm(clean[pool] - clean[int(anchor)], axis=1)
            nuisance_distance = np.linalg.norm(nuisance[pool] - nuisance[int(anchor)], axis=1)
            positives[position] = pool[int(np.lexsort((pool, nuisance_distance, longitudinal_distance))[0])]
            if pair_types[position] == 0:
                q75 = np.quantile(longitudinal_distance, 0.75)
                q25 = np.quantile(nuisance_distance, 0.25)
                valid = np.flatnonzero((longitudinal_distance >= q75) & (nuisance_distance <= q25))
                if len(valid):
                    local_order = np.lexsort(
                        (pool[valid], -longitudinal_distance[valid], nuisance_distance[valid])
                    )
                    chosen = int(valid[int(local_order[0])])
                else:
                    chosen = int(np.lexsort((pool, -longitudinal_distance, nuisance_distance))[0])
            elif pair_types[position] == 1:
                q40, q60 = np.quantile(longitudinal_distance, [0.40, 0.60])
                valid = np.flatnonzero((longitudinal_distance >= q40) & (longitudinal_distance <= q60))
                if len(valid):
                    local = np.lexsort((pool[valid], nuisance_distance[valid], longitudinal_distance[valid]))
                    chosen = int(valid[int(local[0])])
                else:
                    chosen = int(np.argmin(np.abs(longitudinal_distance - np.median(longitudinal_distance))))
            else:
                chosen = int(rng.integers(0, len(pool)))
            negatives[position] = pool[chosen]
    plan = {
        "candidate": candidate,
        "seed": int(seed),
        "pair_seed": int(pair_seed),
        "epoch": int(epoch),
        "plan_seed": int(plan_seed),
        "epoch_samples": int(epoch_samples),
        "batch_size": int(batch_size),
        "ranking_margin": float(ranking_margin),
        "sample_indices": sample_indices,
        "sampling_weights": sampling_weights,
        "batch_offsets": batch_offsets,
        "positive_indices": positives,
        "negative_indices": negatives,
        "pair_types": pair_types,
        "slot_dropout_masks": slot_masks,
        "all_neighbor_dropout_masks": all_neighbor_masks,
        "augmentation_seeds": augmentation_seeds,
    }
    return plan


FAIRNESS_FIELDS = (
    "sampling_weights", "sample_indices", "batch_offsets", "positive_indices", "negative_indices",
    "pair_types", "slot_dropout_masks", "all_neighbor_dropout_masks", "augmentation_seeds",
)


def random_plan_ledger(plan: Mapping[str, Any]) -> dict[str, Any]:
    fields = {name: sha256_array(np.asarray(plan[name])) for name in FAIRNESS_FIELDS}
    fields["optimizer_schedule"] = sha256_json({"scheduler": "constant", "factor": 1.0})
    fields["training_budget"] = sha256_json(
        {"epoch_samples": plan["epoch_samples"], "batch_size": plan["batch_size"], "batch_count": len(plan["batch_offsets"]) - 1}
    )
    pair_counts = np.bincount(np.asarray(plan["pair_types"], dtype=np.int64), minlength=3)
    sampling_weights = np.asarray(plan["sampling_weights"], dtype=np.float64)
    return {
        "candidate": plan["candidate"],
        "seed": plan["seed"],
        "pair_seed": plan["pair_seed"],
        "epoch": plan["epoch"],
        "plan_seed": plan["plan_seed"],
        "field_sha256": fields,
        "audit_summary": {
            "epoch_sample_count": int(len(plan["sample_indices"])),
            "batch_count": int(len(plan["batch_offsets"]) - 1),
            "pair_type_counts": {
                "hard_negative": int(pair_counts[0]),
                "near_boundary": int(pair_counts[1]),
                "uniform": int(pair_counts[2]),
            },
            "sampling_weight_min": float(np.min(sampling_weights)),
            "sampling_weight_max": float(np.max(sampling_weights)),
            "sampling_weight_sum": float(np.sum(sampling_weights)),
            "slot_dropout_true_count": int(np.sum(plan["slot_dropout_masks"])),
            "all_neighbor_dropout_true_count": int(np.sum(plan["all_neighbor_dropout_masks"])),
            "augmentation_seed_count": int(len(plan["augmentation_seeds"])),
        },
        "candidate_independent_fingerprint_sha256": sha256_json(fields),
    }


def assert_bc_fairness(ledger_b: Mapping[str, Any], ledger_c: Mapping[str, Any]) -> dict[str, Any]:
    comparisons = {
        key: ledger_b["field_sha256"].get(key) == ledger_c["field_sha256"].get(key)
        for key in sorted(set(ledger_b["field_sha256"]) | set(ledger_c["field_sha256"]))
    }
    passed = all(comparisons.values()) and (
        ledger_b["candidate_independent_fingerprint_sha256"] == ledger_c["candidate_independent_fingerprint_sha256"]
    )
    if not passed:
        raise ValueError(f"B/C fairness ledger mismatch: {comparisons}")
    return {
        "all_streams_identical": True,
        "stream_comparisons": comparisons,
        "shared_fingerprint_sha256": ledger_b["candidate_independent_fingerprint_sha256"],
    }


def apply_neighbor_dropout(context: torch.Tensor, slot_masks: np.ndarray) -> torch.Tensor:
    result = context.clone()
    if slot_masks.shape != (len(context), len(SLOT_NAMES)):
        raise ValueError(f"Expected dropout mask [{len(context)},{len(SLOT_NAMES)}], got {slot_masks.shape}")
    for row in range(len(context)):
        for slot in range(len(SLOT_NAMES)):
            if bool(slot_masks[row, slot]):
                start = 8 + slot * len(NEIGHBOR_CHANNELS)
                result[row, :, start : start + len(NEIGHBOR_CHANNELS)] = 0.0
    return result


def _legacy_losses(
    model: UnifiedABCModel,
    z: torch.Tensor,
    feat33: torch.Tensor,
    groups: Mapping[str, Sequence[int]],
    weights: Mapping[str, float],
    objective: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    loss = float(weights["style_soft_contrastive"]) * soft_contrastive_loss(
        z, feat33, float(objective["temperature"]), float(objective["feature_temperature"])
    )
    parts = {"global_style_soft_contrastive": float(loss.detach())}
    mapping = {
        "longitudinal": ("longitudinal_comfort", "aux_longitudinal", "metric_longitudinal"),
        "following": ("following_interaction", "aux_following", "metric_following"),
        "lateral_dynamics": ("lateral_dynamics", "aux_lateral_dynamics", "metric_lateral_dynamics"),
        "lateral_gap": ("lateral_gap_interaction", "aux_lateral_gap", "metric_lateral_gap"),
        "behavior_proxy": ("behavior_proxy", "aux_behavior_proxy", "metric_behavior_proxy"),
    }
    for head, (group, aux_weight, metric_weight) in mapping.items():
        target = feat33[:, groups[group]]
        auxiliary = F.smooth_l1_loss(model.heads[head](z), target)
        metric = metric_alignment_loss(z, target, objective["metric_loss_type"])
        loss = loss + float(weights[aux_weight]) * auxiliary + float(weights[metric_weight]) * metric
        parts[f"aux_{head}"] = float(auxiliary.detach())
        parts[f"metric_{head}"] = float(metric.detach())
    return loss, parts


def _recovery_losses(
    model: UnifiedABCModel,
    z: torch.Tensor,
    clean_z: torch.Tensor,
    augmented_z: torch.Tensor,
    positive_z: torch.Tensor,
    negative_z: torch.Tensor,
    feat33: torch.Tensor,
    clean_target: torch.Tensor,
    groups: Mapping[str, Sequence[int]],
    objective: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = objective["weights"]
    long_z = model.longitudinal_view(z)
    context_z = model.context_view(z)
    components: dict[str, torch.Tensor] = {
        "global_style_soft_contrastive": soft_contrastive_loss(z, feat33, 0.1, 1.0),
        "ego_longitudinal_aux_huber": F.smooth_l1_loss(model.heads["clean_longitudinal"](long_z), clean_target),
        "ego_longitudinal_metric_alignment": metric_alignment_loss(long_z, clean_target, objective["metric_loss_type"]),
        "following_interaction_aux": F.smooth_l1_loss(
            model.heads["following"](context_z), feat33[:, groups["following_interaction"]]
        ),
        "lateral_dynamics_aux": F.smooth_l1_loss(
            model.heads["lateral_dynamics"](context_z), feat33[:, groups["lateral_dynamics"]]
        ),
        "lateral_gap_aux": F.smooth_l1_loss(
            model.heads["lateral_gap"](context_z), feat33[:, groups["lateral_gap_interaction"]]
        ),
        "behavior_proxy_aux": F.smooth_l1_loss(
            model.heads["behavior_proxy"](z), feat33[:, groups["behavior_proxy"]]
        ),
        "neighbor_dropout_consistency": F.smooth_l1_loss(
            F.normalize(clean_z, dim=1, eps=1e-8), F.normalize(augmented_z, dim=1, eps=1e-8)
        ),
    }
    anchor = F.normalize(model.longitudinal_view(z), dim=1, eps=1e-8)
    positive = F.normalize(model.longitudinal_view(positive_z), dim=1, eps=1e-8)
    negative = F.normalize(model.longitudinal_view(negative_z), dim=1, eps=1e-8)
    positive_distance = 1.0 - torch.sum(anchor * positive, dim=1)
    negative_distance = 1.0 - torch.sum(anchor * negative, dim=1)
    components["ego_longitudinal_pair_ranking"] = torch.relu(
        float(objective["ranking_margin"]) + positive_distance - negative_distance
    ).mean()
    loss = sum(float(weights[name]) * component for name, component in components.items())
    return loss, {name: float(value.detach()) for name, value in components.items()}


def build_optimizer_and_scheduler(model: nn.Module, optimization: Mapping[str, Any]) -> tuple[Any, Any]:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    return optimizer, scheduler


def run_batch(
    *,
    candidate: str,
    model: UnifiedABCModel,
    optimizer: torch.optim.Optimizer,
    dataset: Any,
    plan: Mapping[str, Any],
    batch_index: int,
    stage6t: Mapping[str, Any],
    device: torch.device,
    train: bool,
) -> dict[str, Any]:
    offsets = np.asarray(plan["batch_offsets"])
    start, end = int(offsets[batch_index]), int(offsets[batch_index + 1])
    sample_indices = np.asarray(plan["sample_indices"])[start:end]
    positive_indices = np.asarray(plan["positive_indices"])[start:end]
    negative_indices = np.asarray(plan["negative_indices"])[start:end]
    batch = collate_rows(dataset, sample_indices)
    positive = collate_rows(dataset, positive_indices)
    negative = collate_rows(dataset, negative_indices)
    context = batch["context"].float().to(device)
    feat33 = batch["feat33"].float().to(device)
    clean_target = batch["clean_longitudinal"].float().to(device)
    model.train(train)
    if train:
        optimizer.zero_grad(set_to_none=True)
    z = model(context)
    if candidate == "A":
        objective = stage6t["objective_packages"]["legacy_stage5d_balanced_v2_exact"]
        loss, components = _legacy_losses(model, z, feat33, dataset.group_indices, objective["weights"], objective)
        dropout_hash = sha256_array(np.zeros((len(context), 5), dtype=bool))
    else:
        slot_masks = np.asarray(plan["slot_dropout_masks"])[start:end]
        augmented_context = apply_neighbor_dropout(context, slot_masks)
        augmented_z = model(augmented_context)
        positive_z = model(positive["context"].float().to(device))
        negative_z = model(negative["context"].float().to(device))
        objective = stage6t["objective_packages"]["longitudinal_recovery_v2"]
        loss, components = _recovery_losses(
            model, z, z, augmented_z, positive_z, negative_z, feat33, clean_target,
            dataset.group_indices, objective,
        )
        dropout_hash = sha256_array(slot_masks)
    if not torch.isfinite(loss):
        raise RuntimeError(f"Candidate {candidate} non-finite loss at batch {batch_index}")
    if train:
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError(f"Candidate {candidate} has missing or non-finite gradients")
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(stage6t["common_optimization"]["gradient_clip_norm"]))
        optimizer.step()
    return {
        "candidate": candidate,
        "batch_index": int(batch_index),
        "train": bool(train),
        "loss": float(loss.detach().cpu()),
        "loss_finite": bool(torch.isfinite(loss)),
        "embedding_shape": list(z.shape),
        "embedding_finite": bool(torch.isfinite(z).all()),
        "components": components,
        "sample_indices_sha256": sha256_array(sample_indices),
        "positive_indices_sha256": sha256_array(positive_indices),
        "negative_indices_sha256": sha256_array(negative_indices),
        "dropout_mask_sha256": dropout_hash,
        "row_keys_sha256": sha256_json(batch["row_keys"]),
    }


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


def save_checkpoint(
    path: Path,
    *,
    candidate: str,
    model: UnifiedABCModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    next_batch_index: int,
    global_step: int,
    plan_ledger: Mapping[str, Any],
    protocol_fingerprint: str,
    smoke_only: bool,
) -> None:
    if not smoke_only:
        raise PermissionError("Stage6U implementation phase can only write smoke checkpoints")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "stage6u_smoke_checkpoint_v1",
            "smoke_only": True,
            "candidate": candidate,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": int(epoch),
            "next_batch_index": int(next_batch_index),
            "global_step": int(global_step),
            "rng_state": capture_rng_state(),
            "plan_ledger": dict(plan_ledger),
            "protocol_fingerprint": protocol_fingerprint,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    *,
    candidate: str,
    model: UnifiedABCModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    expected_protocol_fingerprint: str,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != "stage6u_smoke_checkpoint_v1" or checkpoint.get("smoke_only") is not True:
        raise ValueError("Only a Stage6U smoke checkpoint may be loaded during implementation freeze")
    if checkpoint.get("candidate") != candidate:
        raise ValueError("Checkpoint candidate mismatch")
    if checkpoint.get("protocol_fingerprint") != expected_protocol_fingerprint:
        raise ValueError("Checkpoint Stage6T fingerprint mismatch")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    restore_rng_state(checkpoint["rng_state"])
    return checkpoint


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def validate_formal_authorization(
    config: Mapping[str, Any],
    authorization_manifest_path: Path | None,
    implementation_freeze_sha256: str | None,
    output_dir: Path,
    resume_requested: bool = False,
    candidate: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    guard = config["formal_training_guard"]
    if authorization_manifest_path is None or implementation_freeze_sha256 is None:
        raise PermissionError("Formal mode requires a separate authorization manifest and implementation freeze SHA-256")
    if not authorization_manifest_path.is_file():
        raise FileNotFoundError(authorization_manifest_path)
    authorization = read_json(authorization_manifest_path)
    if authorization.get("status") != guard["required_authorization_status"]:
        raise PermissionError("Formal training authorization status is invalid")
    if authorization.get("training_authorized") is not True:
        raise PermissionError("Formal training authorization.training_authorized must be true")
    if authorization.get("implementation_freeze_sha256") != implementation_freeze_sha256:
        raise PermissionError("Formal authorization does not bind the requested implementation freeze SHA-256")
    freeze_manifest_text = authorization.get("implementation_freeze_manifest_path")
    if not freeze_manifest_text:
        raise PermissionError("Formal authorization must record the implementation freeze manifest path")
    freeze_manifest_path = resolve_repo_path(str(freeze_manifest_text))
    if sha256_file(freeze_manifest_path) != implementation_freeze_sha256:
        raise PermissionError("Supplied implementation freeze SHA-256 does not match the freeze manifest file")
    freeze_manifest = read_json(freeze_manifest_path)
    if freeze_manifest.get("status") != "FROZEN_READY_FOR_ABC_FORMAL_TRAINING":
        raise PermissionError("Implementation freeze is not ready for formal training")
    current_trainer_sha = sha256_file(Path(__file__).resolve())
    if freeze_manifest.get("source_records", {}).get("trainer", {}).get("sha256") != current_trainer_sha:
        raise PermissionError("Current trainer SHA-256 differs from the implementation freeze")
    current_config_path = REPO_ROOT / "configs/stage6u_unified_abc_trainer.json"
    if freeze_manifest.get("source_records", {}).get("stage6u_config", {}).get("sha256") != sha256_file(current_config_path):
        raise PermissionError("Current Stage6U config SHA-256 differs from the implementation freeze")
    if candidate is None or seed is None:
        raise PermissionError("Formal authorization validation requires candidate and seed")
    if authorization.get("authorized_candidates") != guard["authorized_candidates_required"]:
        raise PermissionError("Formal authorization candidate set or order changed")
    if candidate not in authorization.get("authorized_candidates", []):
        raise PermissionError(f"Candidate {candidate} is not authorized")
    authorized_seeds = [int(value) for value in authorization.get("authorized_seeds", [])]
    if authorized_seeds != [int(value) for value in guard["authorized_seeds_required"]]:
        raise PermissionError("Formal authorization seed set or order changed")
    if int(seed) not in authorized_seeds:
        raise PermissionError(f"Seed {seed} is not authorized")
    stage6t_path = resolve_repo_path(config["stage6t_protocol"]["config_path"])
    stage6t = read_json(stage6t_path)
    expected_root = resolve_repo_path(stage6t["candidates"][candidate]["output_root"]) / f"seed_{seed}"
    if output_dir.resolve() != expected_root.resolve():
        raise PermissionError(f"Formal output must be exactly {expected_root}, got {output_dir}")
    matching_runs = [
        row
        for row in authorization.get("training_order", [])
        if row.get("candidate") == candidate and int(row.get("seed", -1)) == int(seed)
    ]
    if len(matching_runs) != 1:
        raise PermissionError(f"Formal authorization must contain exactly one ordered run for {candidate}/{seed}")
    authorized_output = resolve_repo_path(str(matching_runs[0].get("output_dir", "")))
    if authorized_output.resolve() != expected_root.resolve():
        raise PermissionError(f"Authorized run output differs from frozen Stage6T output: {authorized_output}")
    expected_order = [
        (candidate_name, frozen_seed)
        for candidate_name in guard["authorized_candidates_required"]
        for frozen_seed in guard["authorized_seeds_required"]
    ]
    observed_order = [
        (str(row.get("candidate")), int(row.get("seed", -1)))
        for row in authorization.get("training_order", [])
    ]
    if observed_order != expected_order:
        raise PermissionError(f"Formal authorization training order changed: {observed_order}")
    if authorization.get("single_device_serial_execution") is not True:
        raise PermissionError("Formal authorization must require single-device serial execution")
    forbidden = authorization.get("forbidden_evaluation_boundary", {})
    required_forbidden = (
        "waymo_test", "stage6j_k_p", "nuplan", "embedding_bdd_mmd", "stage6s_v2_confirmation"
    )
    if not all(forbidden.get(key) is True for key in required_forbidden):
        raise PermissionError("Formal authorization must preserve every Stage6U blind-evaluation boundary")
    if output_dir.exists() and not resume_requested:
        raise FileExistsError(f"Formal candidate output must not exist: {output_dir}")
    if resume_requested and not output_dir.is_dir():
        raise FileNotFoundError(f"Resume output directory does not exist: {output_dir}")
    return authorization


def formal_checkpoint_payload(
    *,
    candidate: str,
    seed: int,
    model: UnifiedABCModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    next_batch_index: int,
    global_step: int,
    best_val_loss: float,
    best_epoch: int,
    patience_count: int,
    early_stopping_reference: float,
    plan_ledger: Mapping[str, Any] | None,
    epoch_train_loss_sum: float,
    epoch_train_rows: int,
    protocol_fingerprint: str,
    implementation_freeze_sha256: str,
    authorization_manifest_sha256: str,
    checkpoint_metadata: Mapping[str, Any],
    resume_history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "stage6u_formal_checkpoint_v1",
        "smoke_only": False,
        "candidate": candidate,
        "seed": int(seed),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": int(epoch),
        "next_batch_index": int(next_batch_index),
        "global_step": int(global_step),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "patience_count": int(patience_count),
        "early_stopping_reference": float(early_stopping_reference),
        "epoch_train_loss_sum": float(epoch_train_loss_sum),
        "epoch_train_rows": int(epoch_train_rows),
        "rng_state": capture_rng_state(),
        "plan_ledger": dict(plan_ledger) if plan_ledger is not None else None,
        "protocol_fingerprint": protocol_fingerprint,
        "implementation_freeze_sha256": implementation_freeze_sha256,
        "authorization_manifest_sha256": authorization_manifest_sha256,
        "checkpoint_metadata": dict(checkpoint_metadata),
        "resume_history": [dict(row) for row in resume_history],
    }


def save_formal_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def load_formal_checkpoint(
    path: Path,
    *,
    candidate: str,
    seed: int,
    model: UnifiedABCModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    protocol_fingerprint: str,
    implementation_freeze_sha256: str,
    authorization_manifest_sha256: str,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != "stage6u_formal_checkpoint_v1" or checkpoint.get("smoke_only") is not False:
        raise ValueError("Not a Stage6U formal checkpoint")
    checks = {
        "candidate": checkpoint.get("candidate") == candidate,
        "seed": int(checkpoint.get("seed", -1)) == int(seed),
        "protocol": checkpoint.get("protocol_fingerprint") == protocol_fingerprint,
        "implementation": checkpoint.get("implementation_freeze_sha256") == implementation_freeze_sha256,
        "authorization": checkpoint.get("authorization_manifest_sha256") == authorization_manifest_sha256,
    }
    if not all(checks.values()):
        raise ValueError(f"Formal checkpoint binding mismatch: {checks}")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    restore_rng_state(checkpoint["rng_state"])
    return checkpoint


def validate_resume_plan(
    resume_payload: Mapping[str, Any],
    train_ledger: Mapping[str, Any],
) -> str:
    """Validate a random plan only for a checkpoint inside an unfinished epoch."""
    next_batch_index = int(resume_payload.get("next_batch_index", 0))
    if next_batch_index == 0:
        return "epoch_boundary_no_plan_check"
    checkpoint_ledger = resume_payload.get("plan_ledger")
    if not isinstance(checkpoint_ledger, Mapping):
        raise ValueError("Mid-epoch resume checkpoint is missing its random-plan ledger")
    expected = checkpoint_ledger.get("candidate_independent_fingerprint_sha256")
    observed = train_ledger.get("candidate_independent_fingerprint_sha256")
    if observed != expected:
        raise ValueError("Resume random plan differs from checkpoint")
    return "mid_epoch_plan_match"


def update_validation_selection(
    *,
    val_loss: float,
    epoch: int,
    best_val_loss: float,
    best_epoch: int,
    early_stopping_reference: float,
    patience_count: int,
    min_delta: float,
) -> dict[str, Any]:
    """Separate exact best-val selection from min-delta early stopping."""
    best_improved = val_loss < best_val_loss
    if best_improved:
        best_val_loss = val_loss
        best_epoch = epoch
    patience_improved = val_loss < early_stopping_reference - min_delta
    if patience_improved:
        early_stopping_reference = val_loss
        patience_count = 0
    else:
        patience_count += 1
    return {
        "best_improved": best_improved,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "patience_improved": patience_improved,
        "early_stopping_reference": early_stopping_reference,
        "patience_count": patience_count,
    }


def _formal_plan(
    *,
    dataset: DynamicTrainValDataset,
    statistics: Mapping[str, np.ndarray],
    candidate: str,
    stage6t: Mapping[str, Any],
    seed: int,
    epoch: int,
    epoch_samples: int,
    batch_size: int,
    validation: bool = False,
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
        statistics=statistics,
        sample_without_replacement=validation,
    )


def run_formal_training(
    *,
    config: Mapping[str, Any],
    stage6t: Mapping[str, Any],
    stage6t_freeze: Mapping[str, Any],
    candidate: str,
    seed: int,
    output_dir: Path,
    implementation_freeze_sha256: str,
    authorization_manifest_path: Path,
    authorization_manifest_sha256: str,
    resume_checkpoint: Path | None,
) -> dict[str, Any]:
    if seed not in [int(value) for value in stage6t["common_optimization"]["seeds"]]:
        raise ValueError(f"Seed {seed} is not in the Stage6T frozen seed set")
    data = config["training_data"]
    manifest_path = resolve_repo_path(data["dynamic_full51_manifest_path"])
    standardization_path = resolve_repo_path(data["global_33d_standardization_path"])
    manifest = read_json(manifest_path)
    feature_schema = Path(manifest["part_roots"][0]) / "feature_schema.json"
    train_dataset = DynamicTrainValDataset(
        manifest_path, "train", standardization_path, feature_schema_path=feature_schema, cache_shards=2
    )
    val_dataset = DynamicTrainValDataset(
        manifest_path, "val", standardization_path, feature_schema_path=feature_schema, cache_shards=2
    )
    expected_counts = stage6t["dataset_contract"]["split_counts"]
    if len(train_dataset) != int(expected_counts["train"]) or len(val_dataset) != int(expected_counts["val"]):
        raise ValueError(f"Formal train/val counts differ: train={len(train_dataset)}, val={len(val_dataset)}")
    device_policy = stage6t["common_optimization"]["device_policy"]
    device = torch.device("mps" if device_policy == "mps_if_available_else_cpu" and torch.backends.mps.is_available() else "cpu")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = UnifiedABCModel(candidate, train_dataset.group_indices).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, stage6t["common_optimization"])
    train_statistics = derive_dataset_statistics(train_dataset)
    val_statistics = derive_dataset_statistics(val_dataset)
    optimization = stage6t["common_optimization"]
    batch_size = int(optimization["batch_size"])
    epoch_samples = int(optimization["epoch_samples"])
    maximum_epochs = int(optimization["max_epochs"])
    max_steps = int(optimization["max_total_optimizer_steps_per_seed"])
    patience_limit = int(optimization["early_stopping_patience_epochs"])
    min_delta = float(optimization["early_stopping_min_delta"])
    start_epoch = 0
    next_batch_index = 0
    global_step = 0
    best_val_loss = float("inf")
    best_epoch = -1
    patience_count = 0
    early_stopping_reference = float("inf")
    resume_payload = None
    resume_history: list[dict[str, Any]] = []
    if resume_checkpoint is not None:
        resume_payload = load_formal_checkpoint(
            resume_checkpoint,
            candidate=candidate,
            seed=seed,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            protocol_fingerprint=stage6t_freeze["protocol_content_fingerprint_sha256"],
            implementation_freeze_sha256=implementation_freeze_sha256,
            authorization_manifest_sha256=authorization_manifest_sha256,
        )
        start_epoch = int(resume_payload["epoch"])
        next_batch_index = int(resume_payload["next_batch_index"])
        global_step = int(resume_payload["global_step"])
        best_val_loss = float(resume_payload["best_val_loss"])
        best_epoch = int(resume_payload["best_epoch"])
        patience_count = int(resume_payload["patience_count"])
        early_stopping_reference = float(resume_payload["early_stopping_reference"])
        resume_history = [dict(row) for row in resume_payload.get("resume_history", [])]
        resume_history.append(
            {
                "resumed_at_utc": utc_now_iso(),
                "checkpoint_path": str(resume_checkpoint),
                "checkpoint_sha256": sha256_file(resume_checkpoint),
                "epoch": start_epoch,
                "next_batch_index": next_batch_index,
                "global_step": global_step,
            }
        )
    output_dir.mkdir(parents=True, exist_ok=resume_checkpoint is not None)
    progress_path = output_dir / "progress.jsonl"
    log_path = output_dir / "train_log.csv"
    trainer_sha256 = sha256_file(Path(__file__).resolve())
    stage6u_config_sha256 = sha256_file(REPO_ROOT / "configs/stage6u_unified_abc_trainer.json")
    checkpoint_metadata = {
        "protocol_id": stage6t["protocol_id"],
        "stage6t_freeze_manifest_sha256": config["stage6t_protocol"]["freeze_manifest_sha256"],
        "candidate_id": candidate,
        "seed": seed,
        "architecture_id": stage6t["candidates"][candidate]["architecture"],
        "sampling_package_id": stage6t["candidates"][candidate]["sampling_package"],
        "objective_package_id": stage6t["candidates"][candidate]["objective_package"],
        "dataset_content_signature_sha256": manifest["content_signature_sha256"],
        "git_commit": git_revision(),
        "training_environment": {
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": str(device),
            "mps_available": bool(torch.backends.mps.is_available()),
        },
        "validation_objective": "frozen_candidate_specific_total_loss_on_waymo_val",
        "trainer_sha256": trainer_sha256,
        "stage6u_config_sha256": stage6u_config_sha256,
        "implementation_freeze_sha256": implementation_freeze_sha256,
        "authorization_manifest_path": str(authorization_manifest_path),
        "authorization_manifest_sha256": authorization_manifest_sha256,
    }
    if resume_checkpoint is None:
        bootstrap_payload = formal_checkpoint_payload(
            candidate=candidate,
            seed=seed,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=0,
            next_batch_index=0,
            global_step=0,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            patience_count=patience_count,
            early_stopping_reference=early_stopping_reference,
            plan_ledger=None,
            epoch_train_loss_sum=0.0,
            epoch_train_rows=0,
            protocol_fingerprint=stage6t_freeze["protocol_content_fingerprint_sha256"],
            implementation_freeze_sha256=implementation_freeze_sha256,
            authorization_manifest_sha256=authorization_manifest_sha256,
            checkpoint_metadata=checkpoint_metadata,
            resume_history=resume_history,
        )
        save_formal_checkpoint(output_dir / "resume_model.pt", bootstrap_payload)
    config_snapshot = {
        "candidate": candidate,
        "seed": seed,
        "stage6t_protocol_fingerprint_sha256": stage6t_freeze["protocol_content_fingerprint_sha256"],
        "implementation_freeze_sha256": implementation_freeze_sha256,
        "stage6t_candidate": stage6t["candidates"][candidate],
        "optimization": optimization,
        "train_rows": len(train_dataset),
        "val_rows": len(val_dataset),
        "device": str(device),
        "checkpoint_metadata": checkpoint_metadata,
    }
    config_snapshot_path = output_dir / "formal_training_config.json"
    if config_snapshot_path.is_file():
        if read_json(config_snapshot_path) != config_snapshot:
            raise ValueError("Resume formal training config differs from the existing frozen snapshot")
    else:
        config_snapshot_path.write_text(
            json.dumps(config_snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    if not log_path.exists():
        with log_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(["epoch", "train_loss", "val_loss", "global_step", "elapsed_seconds"])
    run_started = time.monotonic()
    stopped_reason = "max_epochs"
    for epoch in range(start_epoch, maximum_epochs):
        if resume_payload is not None and epoch == start_epoch and next_batch_index == 0:
            if patience_count >= patience_limit:
                stopped_reason = "early_stopping"
                break
            if global_step >= max_steps:
                stopped_reason = "max_optimizer_steps"
                break
        train_plan = _formal_plan(
            dataset=train_dataset,
            statistics=train_statistics,
            candidate=candidate,
            stage6t=stage6t,
            seed=seed,
            epoch=epoch,
            epoch_samples=epoch_samples,
            batch_size=batch_size,
        )
        train_ledger = random_plan_ledger(train_plan)
        if resume_payload is not None and epoch == start_epoch:
            validate_resume_plan(resume_payload, train_ledger)
        train_loss_sum = (
            float(resume_payload.get("epoch_train_loss_sum", 0.0))
            if resume_payload is not None and epoch == start_epoch
            else 0.0
        )
        train_rows = (
            int(resume_payload.get("epoch_train_rows", 0))
            if resume_payload is not None and epoch == start_epoch
            else 0
        )
        batch_count = len(train_plan["batch_offsets"]) - 1
        first_batch = next_batch_index if epoch == start_epoch else 0
        initial_payload = formal_checkpoint_payload(
            candidate=candidate,
            seed=seed,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            next_batch_index=first_batch,
            global_step=global_step,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            patience_count=patience_count,
            early_stopping_reference=early_stopping_reference,
            plan_ledger=train_ledger,
            epoch_train_loss_sum=train_loss_sum,
            epoch_train_rows=train_rows,
            protocol_fingerprint=stage6t_freeze["protocol_content_fingerprint_sha256"],
            implementation_freeze_sha256=implementation_freeze_sha256,
            authorization_manifest_sha256=authorization_manifest_sha256,
            checkpoint_metadata=checkpoint_metadata,
            resume_history=resume_history,
        )
        save_formal_checkpoint(output_dir / "resume_model.pt", initial_payload)
        train_progress = tqdm(
            range(first_batch, batch_count),
            total=batch_count,
            initial=first_batch,
            desc=f"Stage6U {candidate}/{seed} train epoch {epoch + 1}/{maximum_epochs}",
            unit="batch",
            dynamic_ncols=True,
        )
        completed_batch_index = first_batch
        for batch_index in train_progress:
            if global_step >= max_steps:
                stopped_reason = "max_optimizer_steps"
                break
            result = run_batch(
                candidate=candidate,
                model=model,
                optimizer=optimizer,
                dataset=train_dataset,
                plan=train_plan,
                batch_index=batch_index,
                stage6t=stage6t,
                device=device,
                train=True,
            )
            start, end = train_plan["batch_offsets"][batch_index : batch_index + 2]
            current_rows = int(end - start)
            train_loss_sum += result["loss"] * current_rows
            train_rows += current_rows
            global_step += 1
            completed_batch_index = batch_index + 1
            train_progress.set_postfix(loss=f"{result['loss']:.5f}", step=global_step, refresh=False)
            if global_step % 100 == 0:
                with progress_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "event": "heartbeat",
                                "candidate": candidate,
                                "seed": seed,
                                "epoch": epoch,
                                "batch_index": batch_index,
                                "global_step": global_step,
                                "loss": result["loss"],
                                "elapsed_seconds": time.monotonic() - run_started,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                heartbeat_payload = formal_checkpoint_payload(
                    candidate=candidate,
                    seed=seed,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    next_batch_index=batch_index + 1,
                    global_step=global_step,
                    best_val_loss=best_val_loss,
                    best_epoch=best_epoch,
                    patience_count=patience_count,
                    early_stopping_reference=early_stopping_reference,
                    plan_ledger=train_ledger,
                    epoch_train_loss_sum=train_loss_sum,
                    epoch_train_rows=train_rows,
                    protocol_fingerprint=stage6t_freeze["protocol_content_fingerprint_sha256"],
                    implementation_freeze_sha256=implementation_freeze_sha256,
                    authorization_manifest_sha256=authorization_manifest_sha256,
                    checkpoint_metadata=checkpoint_metadata,
                    resume_history=resume_history,
                )
                save_formal_checkpoint(output_dir / "resume_model.pt", heartbeat_payload)
            if global_step >= max_steps:
                stopped_reason = "max_optimizer_steps"
                break
        pre_validation_payload = formal_checkpoint_payload(
            candidate=candidate,
            seed=seed,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            next_batch_index=min(batch_count, completed_batch_index),
            global_step=global_step,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            patience_count=patience_count,
            early_stopping_reference=early_stopping_reference,
            plan_ledger=train_ledger,
            epoch_train_loss_sum=train_loss_sum,
            epoch_train_rows=train_rows,
            protocol_fingerprint=stage6t_freeze["protocol_content_fingerprint_sha256"],
            implementation_freeze_sha256=implementation_freeze_sha256,
            authorization_manifest_sha256=authorization_manifest_sha256,
            checkpoint_metadata=checkpoint_metadata,
            resume_history=resume_history,
        )
        save_formal_checkpoint(output_dir / "resume_model.pt", pre_validation_payload)
        scheduler.step()
        val_plan = _formal_plan(
            dataset=val_dataset,
            statistics=val_statistics,
            candidate=candidate,
            stage6t=stage6t,
            seed=seed + 90_000,
            epoch=epoch,
            epoch_samples=len(val_dataset),
            batch_size=batch_size,
            validation=True,
        )
        val_loss_sum = 0.0
        val_rows = 0
        val_batch_count = len(val_plan["batch_offsets"]) - 1
        with torch.no_grad():
            val_progress = tqdm(
                range(val_batch_count),
                total=val_batch_count,
                desc=f"Stage6U {candidate}/{seed} val epoch {epoch + 1}/{maximum_epochs}",
                unit="batch",
                dynamic_ncols=True,
            )
            for batch_index in val_progress:
                result = run_batch(
                    candidate=candidate,
                    model=model,
                    optimizer=optimizer,
                    dataset=val_dataset,
                    plan=val_plan,
                    batch_index=batch_index,
                    stage6t=stage6t,
                    device=device,
                    train=False,
                )
                start, end = val_plan["batch_offsets"][batch_index : batch_index + 2]
                current_rows = int(end - start)
                val_loss_sum += result["loss"] * current_rows
                val_rows += current_rows
                val_progress.set_postfix(loss=f"{result['loss']:.5f}", refresh=False)
        train_loss = train_loss_sum / max(train_rows, 1)
        val_loss = val_loss_sum / max(val_rows, 1)
        selection = update_validation_selection(
            val_loss=val_loss,
            epoch=epoch,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            early_stopping_reference=early_stopping_reference,
            patience_count=patience_count,
            min_delta=min_delta,
        )
        best_improved = bool(selection["best_improved"])
        best_val_loss = float(selection["best_val_loss"])
        best_epoch = int(selection["best_epoch"])
        early_stopping_reference = float(selection["early_stopping_reference"])
        patience_count = int(selection["patience_count"])
        payload = formal_checkpoint_payload(
            candidate=candidate,
            seed=seed,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            next_batch_index=0,
            global_step=global_step,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            patience_count=patience_count,
            early_stopping_reference=early_stopping_reference,
            plan_ledger=None,
            epoch_train_loss_sum=0.0,
            epoch_train_rows=0,
            protocol_fingerprint=stage6t_freeze["protocol_content_fingerprint_sha256"],
            implementation_freeze_sha256=implementation_freeze_sha256,
            authorization_manifest_sha256=authorization_manifest_sha256,
            checkpoint_metadata=checkpoint_metadata,
            resume_history=resume_history,
        )
        save_formal_checkpoint(output_dir / "last_model.pt", payload)
        save_formal_checkpoint(output_dir / "resume_model.pt", payload)
        if best_improved:
            save_formal_checkpoint(output_dir / "best_model.pt", payload)
        with log_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [epoch + 1, train_loss, val_loss, global_step, time.monotonic() - run_started]
            )
        print(
            f"candidate={candidate} seed={seed} epoch={epoch + 1}/{maximum_epochs} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} global_step={global_step}",
            flush=True,
        )
        next_batch_index = 0
        resume_payload = None
        if patience_count >= patience_limit:
            stopped_reason = "early_stopping"
            break
        if global_step >= max_steps:
            break
    best_path = output_dir / "best_model.pt"
    if not best_path.is_file():
        raise RuntimeError("Formal training ended without a validation-selected checkpoint")
    summary = {
        "schema_version": "stage6u_formal_training_summary_v1",
        "candidate": candidate,
        "seed": seed,
        "best_epoch": best_epoch + 1,
        "best_val_loss": best_val_loss,
        "global_step": global_step,
        "stopped_reason": stopped_reason,
        "best_checkpoint_sha256": sha256_file(best_path),
        "last_checkpoint_sha256": sha256_file(output_dir / "last_model.pt"),
        "training_config_sha256": sha256_file(config_snapshot_path),
        "trainer_sha256": trainer_sha256,
        "stage6u_config_sha256": stage6u_config_sha256,
        "implementation_freeze_sha256": implementation_freeze_sha256,
        "authorization_manifest_sha256": authorization_manifest_sha256,
        "checkpoint_metadata": checkpoint_metadata,
        "resume_history": resume_history,
        "training_complete": True,
        "waymo_splits_read": ["train", "val"],
        "waymo_test_read": False,
        "stage6j_k_p_read_or_run": False,
        "nuplan_read_or_run": False,
        "stage6s_v2_confirmation_read_or_run": False,
        "embedding_bdd_mmd_read": False,
    }
    (output_dir / "formal_training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate", choices=["A", "B", "C"], required=True)
    parser.add_argument("--mode", choices=["smoke", "formal"], default="smoke")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--authorization_manifest", type=Path)
    parser.add_argument("--implementation_freeze_sha256")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume_checkpoint", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, stage6t, stage6t_freeze = load_and_validate_implementation_config(args.config.resolve())
    if args.mode == "formal":
        authorization = validate_formal_authorization(
            config,
            args.authorization_manifest,
            args.implementation_freeze_sha256,
            args.output_dir.resolve(),
            resume_requested=args.resume_checkpoint is not None,
            candidate=args.candidate,
            seed=args.seed,
        )
        if args.seed is None:
            raise ValueError("Formal mode requires --seed from the frozen Stage6T seed set")
        authorization_manifest_path = args.authorization_manifest.resolve()
        authorization_manifest_sha256 = sha256_file(authorization_manifest_path)
        summary = run_formal_training(
            config=config,
            stage6t=stage6t,
            stage6t_freeze=stage6t_freeze,
            candidate=args.candidate,
            seed=args.seed,
            output_dir=args.output_dir.resolve(),
            implementation_freeze_sha256=str(args.implementation_freeze_sha256),
            authorization_manifest_path=authorization_manifest_path,
            authorization_manifest_sha256=authorization_manifest_sha256,
            resume_checkpoint=args.resume_checkpoint.resolve() if args.resume_checkpoint else None,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return
    raise RuntimeError(
        "Use tools/stage6u_smoke_unified_abc_trainer.py for bounded smoke execution; "
        "this CLI never defaults to formal training"
    )


if __name__ == "__main__":
    main()
