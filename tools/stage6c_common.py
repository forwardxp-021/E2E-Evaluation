#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def get_tqdm():
    try:
        from tqdm import tqdm
        return tqdm
    except Exception:
        def tqdm(x, **kwargs):
            return x
        return tqdm


def iter_progress(iterable, enabled=True, **kwargs):
    if not enabled:
        return iterable
    return get_tqdm()(iterable, **kwargs)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_schema_names(path: str) -> List[str]:
    obj = read_json(path)
    feats = obj.get("features", [])
    if feats:
        return [f["name"] for f in sorted(feats, key=lambda x: int(x.get("index", 0)))]
    return obj.get("feature_names", [])


def _norm_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


class FeatureResolver:
    def __init__(self, names: Sequence[str]):
        self.names = list(names)
        self.feature_map = {n: i for i, n in enumerate(self.names)}
        self.norm_map = {_norm_name(n): n for n in self.names}
        self.resolved: Dict[str, Optional[str]] = {}
        self.missing: Dict[str, List[str]] = {}

    def resolve(self, logical_name: str, candidates: Sequence[str]) -> Optional[str]:
        for cand in candidates:
            if cand in self.feature_map:
                self.resolved[logical_name] = cand
                return cand
        for cand in candidates:
            got = self.norm_map.get(_norm_name(cand))
            if got is not None:
                self.resolved[logical_name] = got
                return got
        norm_items = list(self.norm_map.items())
        for cand in candidates:
            cn = _norm_name(cand)
            for nn, actual in norm_items:
                if cn and (cn in nn or nn in cn):
                    self.resolved[logical_name] = actual
                    return actual
        self.resolved[logical_name] = None
        self.missing[logical_name] = list(candidates)
        return None

    def get(self, X: np.ndarray, logical_name: str, candidates: Sequence[str]) -> Optional[np.ndarray]:
        name = self.resolve(logical_name, candidates)
        if name is None:
            return None
        return np.asarray(X[:, self.feature_map[name]], dtype=float)


SHARD_METADATA_COLUMNS = [
    "scenario_id",
    "target_agent_id",
    "start",
    "window_len",
    "split",
    "assignment_mode",
    "lane_assignment_success",
    "fallback_used",
    "lane_context_quality",
]


def _metadata_frame_from_npy(path: Path) -> pd.DataFrame:
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.lib.npyio.NpzFile):
        return pd.DataFrame({k: arr[k] for k in arr.files})
    if isinstance(arr, np.ndarray) and arr.dtype.names:
        return pd.DataFrame(arr)
    if isinstance(arr, np.ndarray) and arr.shape == ():
        obj = arr.item()
        if isinstance(obj, dict):
            return pd.DataFrame(obj)
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        vals = arr.tolist()
        if isinstance(vals, list) and (not vals or isinstance(vals[0], dict)):
            return pd.DataFrame(vals)
    return pd.DataFrame(arr)


def _load_optional_shard_metadata(shard_dir: Path, rows: int, shard_id: int, shard_path: str):
    for name in ["metadata.csv", "meta.csv", "meta.npy"]:
        path = shard_dir / name
        if not path.exists():
            continue
        if path.suffix == ".csv":
            raw = pd.read_csv(path)
        else:
            raw = _metadata_frame_from_npy(path)
        if len(raw) != rows:
            return None, {
                "shard_id": int(shard_id),
                "shard_path": str(shard_path),
                "metadata_path": str(path),
                "reason": "row_count_mismatch",
                "feature_rows": int(rows),
                "metadata_rows": int(len(raw)),
            }
        safe_cols = [c for c in SHARD_METADATA_COLUMNS if c in raw.columns]
        if not safe_cols:
            return None, {
                "shard_id": int(shard_id),
                "shard_path": str(shard_path),
                "metadata_path": str(path),
                "reason": "no_safe_metadata_columns",
                "available_columns": [str(c) for c in raw.columns],
            }
        return raw[safe_cols].reset_index(drop=True), None
    return None, {
        "shard_id": int(shard_id),
        "shard_path": str(shard_path),
        "reason": "metadata_missing",
        "searched_files": ["metadata.csv", "meta.csv", "meta.npy"],
    }


def load_shard_paths(shard_manifest: str) -> Tuple[Path, List[str]]:
    manifest = read_json(shard_manifest)
    base = Path(shard_manifest).parent
    shards = manifest.get("shards", manifest.get("shard_infos", []))
    if shards:
        paths = [s["shard_path"] for s in shards]
    else:
        paths = manifest.get("shard_paths", [])
    if not paths:
        raise ValueError(f"No shard paths found in shard manifest: {shard_manifest}")
    return base, paths


def resolve_path(base: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else base / pp


def resolve_existing_path(candidates: Sequence[Path], raw_path: str, purpose: str) -> Path:
    tried = []
    for candidate in candidates:
        tried.append(candidate)
        if candidate.exists():
            return candidate
    tried_text = "\n".join(f"  - {p}" for p in tried)
    raise FileNotFoundError(f"Missing {purpose}: {raw_path}\nTried:\n{tried_text}")


def load_feature_rows(shard_manifest: str, progress_enabled: bool = True, include_shard_metadata: bool = False):
    base, shard_paths = load_shard_paths(shard_manifest)
    feats = []
    meta = []
    metadata_warnings = []
    metadata_loaded_shards = []
    metadata_missing_shards = []
    global_offset = 0
    for shard_id, sp in enumerate(iter_progress(shard_paths, enabled=progress_enabled, desc="loading shards", unit="shard")):
        shard_dir = resolve_path(base, sp)
        feat_path = shard_dir / "interaction_feat_style.npy"
        if not feat_path.exists():
            raise FileNotFoundError(f"Missing interaction feature file: {feat_path}")
        arr = np.load(feat_path, mmap_mode="r")
        feats.append(np.asarray(arr))
        rows = arr.shape[0]
        frame = pd.DataFrame({
            "global_row": np.arange(global_offset, global_offset + rows, dtype=np.int64),
            "shard_id": shard_id,
            "local_row": np.arange(rows, dtype=np.int64),
        })
        if include_shard_metadata:
            shard_meta, warning = _load_optional_shard_metadata(shard_dir, rows, shard_id, sp)
            if shard_meta is not None:
                frame = pd.concat([frame.reset_index(drop=True), shard_meta.reset_index(drop=True)], axis=1)
                metadata_loaded_shards.append({
                    "shard_id": int(shard_id),
                    "shard_path": str(sp),
                    "columns": [str(c) for c in shard_meta.columns],
                })
            else:
                metadata_warnings.append(warning)
                metadata_missing_shards.append({
                    "shard_id": int(shard_id),
                    "shard_path": str(sp),
                    "reason": warning.get("reason") if warning else "metadata_unavailable",
                })
        meta.append(frame)
        global_offset += rows
    if not feats:
        raise ValueError("No feature shards loaded")
    meta_df = pd.concat(meta, ignore_index=True)
    meta_df.attrs["shard_metadata"] = {
        "include_shard_metadata": bool(include_shard_metadata),
        "metadata_loaded_shards": metadata_loaded_shards,
        "metadata_missing_shards": metadata_missing_shards,
        "metadata_warnings": metadata_warnings,
        "safe_metadata_columns": SHARD_METADATA_COLUMNS,
    }
    return np.concatenate(feats, axis=0), meta_df, len(shard_paths)


def load_embeddings(shard_manifest: str, embedding_manifest: str, progress_enabled: bool = True):
    base, shard_paths = load_shard_paths(shard_manifest)
    emb_obj = read_json(embedding_manifest)
    emb_paths = emb_obj.get("embedding_shard_paths", [])
    if len(emb_paths) != len(shard_paths):
        raise ValueError(
            f"feature/embedding shard count mismatch: {len(shard_paths)} feature shards vs {len(emb_paths)} embedding shards"
        )
    embs = []
    meta = []
    global_offset = 0
    emb_base = Path(embedding_manifest).parent
    for shard_id, (sp, ep) in enumerate(iter_progress(list(zip(shard_paths, emb_paths)), enabled=progress_enabled, desc="loading embedding shards", unit="shard")):
        shard_dir = resolve_path(base, sp)
        feat_path = shard_dir / "interaction_feat_style.npy"
        raw_emb_path = Path(ep)
        emb_candidates = [raw_emb_path] if raw_emb_path.is_absolute() else [
            raw_emb_path,
            resolve_path(emb_base, ep),
            resolve_path(base, ep),
        ]
        emb_path = resolve_existing_path(emb_candidates, ep, "embedding shard file")
        z = np.load(emb_path, mmap_mode="r")
        if feat_path.exists():
            f = np.load(feat_path, mmap_mode="r")
            if f.shape[0] != z.shape[0]:
                raise ValueError(f"Row count mismatch for shard {sp}: features={f.shape[0]}, embeddings={z.shape[0]}")
        embs.append(np.asarray(z))
        rows = z.shape[0]
        meta.append(pd.DataFrame({
            "global_row": np.arange(global_offset, global_offset + rows, dtype=np.int64),
            "shard_id": shard_id,
            "local_row": np.arange(rows, dtype=np.int64),
        }))
        global_offset += rows
    return np.concatenate(embs, axis=0), pd.concat(meta, ignore_index=True)


def finite_quantile(x, q, default=np.nan):
    arr = np.asarray(x, dtype=float)
    ok = np.isfinite(arr)
    if ok.sum() == 0:
        return default
    return float(np.quantile(arr[ok], q))


def high_mask(x, q=0.67):
    if x is None:
        return None
    thr = finite_quantile(x, q)
    if not np.isfinite(thr):
        return np.zeros(len(x), dtype=bool)
    return np.asarray(x, float) >= thr


def low_mask(x, q=0.33):
    if x is None:
        return None
    thr = finite_quantile(x, q)
    if not np.isfinite(thr):
        return np.zeros(len(x), dtype=bool)
    return np.asarray(x, float) <= thr


def present_mask(x, q=0.67, smaller_is_present=True):
    if x is None:
        return None
    arr = np.asarray(x, float)
    ok = np.isfinite(arr)
    if ok.sum() == 0:
        return np.zeros(len(arr), dtype=bool)
    thr = finite_quantile(arr, 1.0 - q if smaller_is_present else q)
    return ok & ((arr <= thr) if smaller_is_present else (arr >= thr))


def label_from_mask(mask, true_label: str, false_label: str, n: int) -> np.ndarray:
    if mask is None:
        return np.array(["unknown"] * n, dtype=object)
    mask = np.asarray(mask, dtype=bool)
    return np.where(mask, true_label, false_label).astype(object)


def combine_or(masks: Iterable[Optional[np.ndarray]], n: int) -> Optional[np.ndarray]:
    valid = [np.asarray(m, dtype=bool) for m in masks if m is not None]
    if not valid:
        return None
    out = np.zeros(n, dtype=bool)
    for m in valid:
        out |= m
    return out


def combine_and(masks: Iterable[Optional[np.ndarray]], n: int) -> Optional[np.ndarray]:
    valid = [np.asarray(m, dtype=bool) for m in masks if m is not None]
    if not valid:
        return None
    out = np.ones(n, dtype=bool)
    for m in valid:
        out &= m
    return out


def row_nan_metric(n: int):
    return np.full(n, np.nan, dtype=float)


def safe_array(x, n: int):
    return row_nan_metric(n) if x is None else np.asarray(x, dtype=float)


def min_available(arrays: Sequence[Optional[np.ndarray]], n: int):
    vals = [np.asarray(a, dtype=float) for a in arrays if a is not None]
    if not vals:
        return None
    return np.nanmin(np.vstack(vals), axis=0)


def max_available(arrays: Sequence[Optional[np.ndarray]], n: int):
    vals = [np.asarray(a, dtype=float) for a in arrays if a is not None]
    if not vals:
        return None
    return np.nanmax(np.vstack(vals), axis=0)


def mean_available(arrays: Sequence[Optional[np.ndarray]], n: int):
    vals = [np.asarray(a, dtype=float) for a in arrays if a is not None]
    if not vals:
        return None
    return np.nanmean(np.vstack(vals), axis=0)


def robust_score(x, higher_is_more=True, scale_floor: float = 1e-3, clip: float = 10.0):
    if x is None:
        return None
    arr = np.asarray(x, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    ok = np.isfinite(arr)
    if ok.sum() == 0:
        return out
    med = finite_quantile(arr, 0.5, 0.0)
    q25 = finite_quantile(arr, 0.25, med)
    q75 = finite_quantile(arr, 0.75, med)
    scale = max(q75 - q25, float(scale_floor))
    z = (arr - med) / scale
    if not higher_is_more:
        z = -z
    out[ok] = np.clip(z[ok], -float(clip), float(clip))
    return out


def score_from_parts(parts: Sequence[Optional[np.ndarray]], n: int):
    vals = [np.asarray(p, dtype=float) for p in parts if p is not None]
    if not vals:
        return None
    return np.nanmean(np.vstack(vals), axis=0)
