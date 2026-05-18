#!/usr/bin/env python3
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


SPLIT_TO_ID = {"train": 0, "val": 1, "test": 2}
ID_TO_SPLIT = {v: k for k, v in SPLIT_TO_ID.items()}


def _split_to_str(split_arr: np.ndarray) -> np.ndarray:
    if split_arr.dtype.kind in {"U", "S", "O"}:
        return split_arr.astype(str)
    return np.array([ID_TO_SPLIT.get(int(x), str(int(x))) for x in split_arr], dtype=object)


def inspect_shard_manifest(manifest_path):
    manifest_path = Path(manifest_path)
    obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard_entries = obj.get("shards", obj.get("shard_infos", []))
    if not shard_entries:
        raise RuntimeError(f"No shard entries in manifest: {manifest_path}")
    first_dir = manifest_path.parent / shard_entries[0]["shard_path"]

    first_shapes = {}
    for name in ["context_traj.npy", "context_mask.npy", "context_mask_window.npy", "interaction_feat_style.npy", "split.npy", "meta.npy"]:
        p = first_dir / name
        if p.exists():
            first_shapes[name] = list(np.load(p, mmap_mode="r").shape)

    context_dim = first_shapes.get("context_traj.npy", [None, None, None])[-1]
    feature_dim = first_shapes.get("interaction_feat_style.npy", [None, None])[-1]
    split_counts = obj.get("split_counts", {})
    total_rows = int(obj.get("n_windows_kept", sum(int(x.get("n_rows", 0)) for x in shard_entries)))
    report = {
        "total_shards": len(shard_entries),
        "total_rows": total_rows,
        "split_counts": split_counts,
        "context_dim": context_dim,
        "feature_dim": feature_dim,
        "first_shard_shapes": first_shapes,
    }
    return report


class ContextShardDataset(Dataset):
    def __init__(
        self,
        shard_manifest,
        split: str = "train",
        use_standardized_features: bool = True,
        max_samples: Optional[int] = None,
        cache_shards: int = 2,
        strict: bool = True,
    ):
        self.manifest_path = Path(shard_manifest)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.split = split
        self.use_standardized_features = use_standardized_features
        self.cache_shards = max(1, int(cache_shards))
        self.strict = strict

        self.shards = self.manifest.get("shards", self.manifest.get("shard_infos", []))
        if not self.shards:
            raise RuntimeError(f"No shards found in {self.manifest_path}")

        self.global_index: List[Dict[str, int]] = []
        for shard_id, shard_info in enumerate(self.shards):
            split_path = self._shard_dir(shard_info) / "split.npy"
            split_arr = _split_to_str(np.load(split_path, mmap_mode="r"))
            for local_idx, s in enumerate(split_arr):
                if self.split == "all" or s == self.split:
                    self.global_index.append({"global_index": len(self.global_index), "shard_id": shard_id, "local_index": int(local_idx)})

        if max_samples is not None:
            self.global_index = self.global_index[: int(max_samples)]

        self._cache = OrderedDict()

    def _shard_dir(self, shard_info):
        return self.manifest_path.parent / shard_info["shard_path"]

    def _load_shard(self, shard_id: int):
        if shard_id in self._cache:
            self._cache.move_to_end(shard_id)
            return self._cache[shard_id]

        d = self._shard_dir(self.shards[shard_id])
        arrays = {
            "context": np.load(d / "context_traj.npy", mmap_mode="r"),
            "context_mask": np.load(d / "context_mask.npy", mmap_mode="r") if (d / "context_mask.npy").exists() else None,
            "feat": np.load(d / ("interaction_feat_style.npy" if self.use_standardized_features else "interaction_feat_style_raw.npy"), mmap_mode="r"),
        }
        self._cache[shard_id] = arrays
        if len(self._cache) > self.cache_shards:
            self._cache.popitem(last=False)
        return arrays

    def __len__(self):
        return len(self.global_index)

    def __getitem__(self, idx):
        rec = self.global_index[idx]
        sid, lid = rec["shard_id"], rec["local_index"]
        arr = self._load_shard(sid)
        context = np.asarray(arr["context"][lid], dtype=np.float32)
        feat = np.asarray(arr["feat"][lid], dtype=np.float32)
        c_mask = np.asarray(arr["context_mask"][lid], dtype=np.float32) if arr["context_mask"] is not None else None

        if self.strict:
            if not np.isfinite(context).all():
                raise RuntimeError(f"Non-finite context at dataset_idx={idx}, shard={sid}, local={lid}")
            if not np.isfinite(feat).all():
                raise RuntimeError(f"Non-finite feature at dataset_idx={idx}, shard={sid}, local={lid}")

        item = {
            "context": torch.from_numpy(context),
            "feat": torch.from_numpy(feat),
            "meta": {
                "global_index": rec["global_index"],
                "shard_id": sid,
                "local_index": lid,
            },
        }
        item["context_mask"] = torch.from_numpy(c_mask) if c_mask is not None else torch.zeros((context.shape[0], 5), dtype=torch.float32)
        return item
