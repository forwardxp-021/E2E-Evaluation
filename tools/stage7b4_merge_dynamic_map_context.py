#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

ALIGNMENT_KEY_CANDIDATE_SETS = [
    ["db_name", "scene_token", "sample_id", "start_frame_index", "end_frame_index"],
    ["scenario_id", "sample_id", "start_frame_index", "end_frame_index"],
    ["scenario_id", "sample_id"],
    ["db_name", "scene_token", "start_frame_index", "end_frame_index"],
]
INVALID_KEY_STRINGS = {"", "nan", "NaN", "None", "null"}
DYNAMIC_ARRAYS = ["ego_seq.npy", "neighbor_seq.npy", "context_traj.npy", "context_mask.npy", "interaction_feat_style.npy"]


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"Required metadata CSV does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Metadata CSV has no header: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def manifest_shard_paths(dynamic_dir: Path) -> List[Path]:
    manifest_path = dynamic_dir / "shard_manifest.json"
    manifest = read_json(manifest_path)
    raw = []
    if manifest.get("shards"):
        raw = [s.get("shard_path") for s in manifest["shards"]]
    elif manifest.get("shard_infos"):
        raw = [s.get("shard_path") for s in manifest["shard_infos"]]
    else:
        raw = manifest.get("shard_paths", [])
    raw = [p for p in raw if p]
    if not raw:
        raise ValueError(f"No shard paths found in dynamic shard manifest: {manifest_path}")
    return [(Path(p) if Path(p).is_absolute() else dynamic_dir / p) for p in raw]


def load_dynamic(dynamic_dir: Path) -> Tuple[Dict[str, np.ndarray], List[Dict[str, str]], List[str], Path]:
    shards = manifest_shard_paths(dynamic_dir)
    arrays_by_name: Dict[str, List[np.ndarray]] = {name: [] for name in DYNAMIC_ARRAYS}
    meta_rows: List[Dict[str, str]] = []
    meta_fields: List[str] = []
    for shard in shards:
        if not shard.exists():
            raise FileNotFoundError(f"Shard path listed in manifest does not exist: {shard}")
        for name in DYNAMIC_ARRAYS:
            path = shard / name
            if not path.exists():
                raise FileNotFoundError(f"Required dynamic array missing: {path}")
            arrays_by_name[name].append(np.load(path, mmap_mode="r"))
        fields, rows = load_csv(shard / "metadata.csv")
        if not meta_fields:
            meta_fields = fields
        else:
            for field in fields:
                if field not in meta_fields:
                    meta_fields.append(field)
        meta_rows.extend(rows)
    arrays = {name: np.concatenate(parts, axis=0) if len(parts) > 1 else np.asarray(parts[0]) for name, parts in arrays_by_name.items()}
    n = arrays["interaction_feat_style.npy"].shape[0]
    for name, arr in arrays.items():
        if arr.shape[0] != n:
            raise ValueError(f"Dynamic array row mismatch: {name} has {arr.shape[0]} rows, expected {n}")
    if len(meta_rows) != n:
        raise ValueError(f"Dynamic metadata rows ({len(meta_rows)}) do not match dynamic arrays ({n})")
    return arrays, meta_rows, meta_fields, dynamic_dir / "feature_schema.json"


def is_empty_key_value(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return str(value).strip() in INVALID_KEY_STRINGS


def key_tuple(row: Dict[str, str], keys: Sequence[str]) -> Tuple[str, ...]:
    return tuple(str(row.get(k, "")).strip() for k in keys)


def duplicate_key_count(rows: List[Dict[str, str]], keys: Sequence[str]) -> int:
    seen = set()
    duplicates = set()
    for row in rows:
        key = key_tuple(row, keys)
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    return len(duplicates)


def validate_alignment_candidate(dynamic_rows: List[Dict[str, str]], map_rows: List[Dict[str, str]], dyn_fields: Sequence[str], map_fields: Sequence[str], keys: List[str]) -> Tuple[bool, dict]:
    result = {
        "keys": keys,
        "passed": False,
        "reason": "",
        "dynamic_unique_key_count": 0,
        "map_odd_unique_key_count": 0,
        "missing_key_count": 0,
        "duplicate_dynamic_key_count": 0,
        "duplicate_map_odd_key_count": 0,
    }
    missing_dyn = [k for k in keys if k not in dyn_fields]
    missing_map = [k for k in keys if k not in map_fields]
    if missing_dyn or missing_map:
        result["reason"] = f"missing fields: dynamic={missing_dyn}, map_odd={missing_map}"
        return False, result
    if len(dynamic_rows) != len(map_rows):
        result["reason"] = f"row count mismatch: dynamic={len(dynamic_rows)}, map_odd={len(map_rows)}"
        return False, result
    for side, rows in (("dynamic", dynamic_rows), ("map_odd", map_rows)):
        bad = []
        for i, row in enumerate(rows):
            invalid_fields = [k for k in keys if is_empty_key_value(row.get(k))]
            if invalid_fields:
                bad.append({"row_index": i, "fields": invalid_fields})
                if len(bad) >= 5:
                    break
        if bad:
            result["reason"] = f"{side} has empty/null alignment key values; examples={bad}"
            return False, result
    dyn_keys = [key_tuple(r, keys) for r in dynamic_rows]
    map_keys = [key_tuple(r, keys) for r in map_rows]
    result["dynamic_unique_key_count"] = len(set(dyn_keys))
    result["map_odd_unique_key_count"] = len(set(map_keys))
    result["duplicate_dynamic_key_count"] = duplicate_key_count(dynamic_rows, keys)
    result["duplicate_map_odd_key_count"] = duplicate_key_count(map_rows, keys)
    missing_keys = sorted(set(dyn_keys) - set(map_keys))
    result["missing_key_count"] = len(missing_keys)
    if result["duplicate_dynamic_key_count"]:
        result["reason"] = f"duplicate dynamic keys: duplicate_count={result['duplicate_dynamic_key_count']}"
        return False, result
    if result["duplicate_map_odd_key_count"]:
        result["reason"] = f"duplicate map/ODD keys: duplicate_count={result['duplicate_map_odd_key_count']}"
        return False, result
    if missing_keys:
        result["reason"] = f"dynamic keys absent from map/ODD metadata: missing_count={len(missing_keys)}, examples={missing_keys[:5]}"
        return False, result
    result["passed"] = True
    result["reason"] = "passed"
    return True, result


def select_alignment_keys(dynamic_rows: List[Dict[str, str]], map_rows: List[Dict[str, str]], dyn_fields: Sequence[str], map_fields: Sequence[str]) -> Tuple[List[str], dict]:
    attempts = []
    for keys in ALIGNMENT_KEY_CANDIDATE_SETS:
        ok, detail = validate_alignment_candidate(dynamic_rows, map_rows, dyn_fields, map_fields, keys)
        attempts.append(detail)
        if ok:
            dyn_keys = [key_tuple(r, keys) for r in dynamic_rows]
            map_keys = [key_tuple(r, keys) for r in map_rows]
            selected = dict(detail)
            selected.update({
                "selected_alignment_keys": keys,
                "alignment_key_candidates_tried": attempts,
                "row_order_already_aligned": dyn_keys == map_keys,
                "reindexing_needed": dyn_keys != map_keys,
            })
            return keys, selected
    raise ValueError(
        "No strict Stage 7B.4 alignment key candidate passed. "
        f"dynamic_columns={list(dyn_fields)}; map_odd_columns={list(map_fields)}; "
        f"candidate_key_sets_tried={attempts}"
    )


def align_indices(dynamic_rows: List[Dict[str, str]], map_rows: List[Dict[str, str]], keys: List[str]) -> Tuple[np.ndarray, bool]:
    dynamic_keys = [key_tuple(r, keys) for r in dynamic_rows]
    map_keys = [key_tuple(r, keys) for r in map_rows]
    dup_dyn = duplicate_key_count(dynamic_rows, keys)
    dup_map = duplicate_key_count(map_rows, keys)
    if dup_dyn:
        raise ValueError(f"Cannot safely align rows: duplicate dynamic keys for columns {keys}; duplicate_count={dup_dyn}")
    if dup_map:
        raise ValueError(f"Cannot safely align rows: duplicate map/ODD keys for columns {keys}; duplicate_count={dup_map}")
    if dynamic_keys == map_keys:
        return np.arange(len(dynamic_rows)), True
    lookup = {k: i for i, k in enumerate(map_keys)}
    missing = [k for k in dynamic_keys if k not in lookup]
    if missing:
        raise ValueError(f"Cannot safely align rows: {len(missing)} dynamic keys are absent from map/ODD metadata using columns {keys}")
    return np.asarray([lookup[k] for k in dynamic_keys], dtype=np.int64), False


def schema_feature_names(schema: dict, feature_dim: int, source: str, schema_path: Path) -> Tuple[List[str], dict]:
    required_key = "interaction_features" if source == "dynamic" else "feature_names"
    names = schema.get(required_key)
    validation = {
        f"{source}_names_source": required_key,
        f"{source}_name_count": len(names) if isinstance(names, list) else None,
        "expected_count": int(feature_dim),
        "passed": False,
    }
    if not isinstance(names, list):
        raise ValueError(f"Required feature names key '{required_key}' missing or not a list in {schema_path}")
    if len(names) != feature_dim:
        raise ValueError(f"Feature name count mismatch in {schema_path}: key='{required_key}' has {len(names)} names, expected feature_dim={feature_dim}")
    validation["passed"] = True
    return [str(n) for n in names], validation


def finite_summary(arrays: Dict[str, np.ndarray]) -> Dict[str, bool]:
    return {name: bool(np.isfinite(arr).all()) for name, arr in arrays.items()}


def make_report(args, n_dyn, n_map, alignment_validation, feature_name_validation, shapes, finite_validation, warnings, passed):
    keys = alignment_validation.get("selected_alignment_keys", [])
    lines = [
        "# Stage 7B.4 Dynamic Context + Map/ODD Merge Alignment Report",
        "",
        "## 1. Stage and purpose",
        "- stage: 7B.4",
        "- purpose: strictly merge Stage 7B.2 dynamic context with Stage 7B.3 map/ODD features. No Stage 7C/7D, rollout, training, BDD, or policy simulation is performed.",
        "",
        "## 2. Input dirs",
        f"- dynamic_context_dir: `{args.dynamic_context_dir}`",
        f"- map_odd_dir: `{args.map_odd_dir}`",
        "",
        "## 3. Output dir",
        f"- output_dir: `{args.output_dir}`",
        "",
        "## 4. Dynamic rows",
        f"- dynamic rows: {n_dyn}",
        "",
        "## 5. Map/ODD rows",
        f"- map/ODD rows: {n_map}",
        "",
        "## 6. Selected alignment keys",
        f"- selected alignment keys: {', '.join(keys)}",
        "",
        "## 7. Candidate key sets tried and failure reasons",
    ]
    for candidate in alignment_validation.get("alignment_key_candidates_tried", []):
        lines.append(f"- {candidate.get('keys')}: passed={candidate.get('passed')}, reason={candidate.get('reason')}")
    lines += [
        "",
        "## 8. Dynamic unique key count",
        f"- dynamic unique key count: {alignment_validation.get('dynamic_unique_key_count')}",
        f"- duplicate dynamic key count: {alignment_validation.get('duplicate_dynamic_key_count')}",
        "",
        "## 9. Map/ODD unique key count",
        f"- map/ODD unique key count: {alignment_validation.get('map_odd_unique_key_count')}",
        f"- duplicate map/ODD key count: {alignment_validation.get('duplicate_map_odd_key_count')}",
        f"- missing key count: {alignment_validation.get('missing_key_count')}",
        "",
        "## 10. Row order already aligned",
        f"- row order already aligned: {str(alignment_validation.get('row_order_already_aligned')).lower()}",
        "",
        "## 11. Reindexing needed",
        f"- reindexing needed: {str(alignment_validation.get('reindexing_needed')).lower()}",
        "",
        "## 12. Output shapes",
    ]
    for name, shape in shapes.items():
        lines.append(f"- {name}: {shape}")
    lines += [
        "",
        "## 13. Feature name validation summary",
        f"- dynamic names source: {feature_name_validation.get('dynamic_names_source')}",
        f"- map/ODD names source: {feature_name_validation.get('map_odd_names_source')}",
        f"- dynamic name count: {feature_name_validation.get('dynamic_name_count')}",
        f"- map/ODD name count: {feature_name_validation.get('map_odd_name_count')}",
        f"- merged name count: {feature_name_validation.get('merged_name_count')}",
        f"- passed: {str(feature_name_validation.get('passed')).lower()}",
        "",
        "## 14. Finite checks for all arrays",
    ]
    for name, detail in finite_validation.get("arrays", {}).items():
        lines.append(f"- {name}: finite={str(detail.get('finite')).lower()}, shape={detail.get('shape')}")
    lines += [
        "",
        "## 15. Warning summary",
        f"- warning_count: {len(warnings)}",
    ]
    for w in warnings:
        lines.append(f"- {w}")
    lines += [
        "",
        "## 16. PASS/FAIL summary",
        f"- status: {'PASS' if passed else 'FAIL'}",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="Stage 7B.4 merge dynamic context and map/ODD-lite features with row alignment checks.")
    p.add_argument("--dynamic_context_dir", required=True)
    p.add_argument("--map_odd_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    dynamic_dir = Path(args.dynamic_context_dir)
    map_dir = Path(args.map_odd_dir)
    out = Path(args.output_dir)
    if out.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory exists; pass --overwrite to replace it: {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    warnings: List[dict] = []
    dyn_arrays, dyn_rows, dyn_fields, dyn_schema_path = load_dynamic(dynamic_dir)
    map_feat_path = map_dir / "map_odd_feat.npy"
    if not map_feat_path.exists():
        raise FileNotFoundError(f"Required map/ODD feature array missing: {map_feat_path}")
    map_feat = np.load(map_feat_path, mmap_mode="r")
    map_fields, map_rows = load_csv(map_dir / "map_odd_meta.csv")
    map_schema_path = map_dir / "map_odd_feature_schema.json"
    dyn_schema = read_json(dyn_schema_path)
    map_schema = read_json(map_schema_path)

    n_dyn = int(dyn_arrays["interaction_feat_style.npy"].shape[0])
    n_map = int(map_feat.shape[0])
    if len(map_rows) != n_map:
        raise ValueError(f"map_odd_meta.csv rows ({len(map_rows)}) do not match map_odd_feat rows ({n_map})")
    if n_dyn != n_map:
        raise ValueError(f"Row count mismatch: dynamic rows={n_dyn}, map/ODD rows={n_map}; safe Stage 7B.4 merge requires equal row counts")
    common_keys, alignment_validation = select_alignment_keys(dyn_rows, map_rows, dyn_fields, map_fields)
    map_indices, already_aligned = align_indices(dyn_rows, map_rows, common_keys)
    reindexed = not already_aligned
    alignment_validation["row_order_already_aligned"] = bool(already_aligned)
    alignment_validation["reindexing_needed"] = bool(reindexed)
    if reindexed:
        warnings.append({"type": "map_odd_rows_reindexed", "message": "map_odd_feat/map_odd_meta row order was reindexed to match dynamic metadata order.", "alignment_keys": common_keys})
    map_feat_aligned = np.asarray(map_feat)[map_indices]
    map_rows_aligned = [map_rows[int(i)] for i in map_indices]

    dynamic_feat = np.asarray(dyn_arrays["interaction_feat_style.npy"])
    if dynamic_feat.ndim != 2 or map_feat_aligned.ndim != 2:
        raise ValueError(f"Features must be 2D: interaction_feat_style shape={dynamic_feat.shape}, map_odd_feat shape={map_feat_aligned.shape}")
    merged = np.concatenate([dynamic_feat, map_feat_aligned], axis=1).astype(np.float32, copy=False)
    output_arrays = {
        "ego_seq": np.asarray(dyn_arrays["ego_seq.npy"]),
        "neighbor_seq": np.asarray(dyn_arrays["neighbor_seq.npy"]),
        "context_traj": np.asarray(dyn_arrays["context_traj.npy"]),
        "context_mask": np.asarray(dyn_arrays["context_mask.npy"]),
        "dynamic_feat_style": dynamic_feat,
        "map_odd_feat": map_feat_aligned,
        "merged_context_feat": merged,
    }
    finite = finite_summary(output_arrays)
    finite_validation = {"arrays": {name: {"finite": ok, "shape": list(output_arrays[name].shape)} for name, ok in finite.items()}, "passed": bool(all(finite.values()))}
    non_finite = [name for name, ok in finite.items() if not ok]
    if non_finite:
        raise ValueError(f"Non-finite values found in Stage 7B.4 output arrays {non_finite}; refusing to write invalid outputs")

    dyn_names, dyn_name_validation = schema_feature_names(dyn_schema, dynamic_feat.shape[1], "dynamic", dyn_schema_path)
    map_names, map_name_validation = schema_feature_names(map_schema, map_feat_aligned.shape[1], "map_odd", map_schema_path)
    merged_names = [f"dynamic::{n}" for n in dyn_names] + [f"map_odd::{n}" for n in map_names]
    feature_name_validation = {
        "dynamic_names_source": dyn_name_validation["dynamic_names_source"],
        "map_odd_names_source": map_name_validation["map_odd_names_source"],
        "dynamic_name_count": len(dyn_names),
        "map_odd_name_count": len(map_names),
        "merged_name_count": len(merged_names),
        "passed": bool(len(dyn_names) == dynamic_feat.shape[1] and len(map_names) == map_feat_aligned.shape[1] and len(merged_names) == merged.shape[1]),
    }
    merged_schema = {
        "stage": "7B.4",
        "feature_type": "nuplan_dynamic_plus_map_odd_context",
        "num_dynamic_features": int(dynamic_feat.shape[1]),
        "num_map_odd_features": int(map_feat_aligned.shape[1]),
        "num_merged_features": int(merged.shape[1]),
        "dynamic_feature_schema_source": str(dyn_schema_path),
        "map_odd_feature_schema_source": str(map_schema_path),
        "dynamic_feature_names": dyn_names,
        "map_odd_feature_names": map_names,
        "merged_feature_names": merged_names,
        "feature_slices": {"dynamic": [0, int(dynamic_feat.shape[1])], "map_odd": [int(dynamic_feat.shape[1]), int(merged.shape[1])]},
        "alignment_keys_used": common_keys,
        "row_order_already_aligned": bool(already_aligned),
        "reindexing_needed": bool(reindexed),
        "feature_name_validation": feature_name_validation,
        "notes": [],
    }
    write_json(out / "merged_feature_schema.json", merged_schema)

    for name, arr in output_arrays.items():
        np.save(out / f"{name}.npy", arr)

    merged_rows = []
    fieldnames = list(dyn_fields)
    for f in map_fields:
        pf = f"map_odd::{f}"
        if pf not in fieldnames:
            fieldnames.append(pf)
    for dr, mr in zip(dyn_rows, map_rows_aligned):
        row = dict(dr)
        for f in map_fields:
            row[f"map_odd::{f}"] = mr.get(f, "")
        merged_rows.append(row)
    write_csv(out / "merged_metadata.csv", merged_rows, fieldnames)
    shapes = {name: list(arr.shape) for name, arr in output_arrays.items()}
    passed = bool(
        n_dyn == n_map
        and alignment_validation.get("passed")
        and feature_name_validation.get("passed")
        and finite_validation.get("passed")
        and merged.shape[1] == dynamic_feat.shape[1] + map_feat_aligned.shape[1]
        and len(merged_names) == merged.shape[1]
    )
    write_json(out / "warnings.json", {"warnings": warnings, "alignment_validation": alignment_validation, "feature_name_validation": feature_name_validation, "finite_validation": finite_validation})
    (out / "alignment_report.md").write_text(make_report(args, n_dyn, n_map, alignment_validation, feature_name_validation, shapes, finite_validation, warnings, passed), encoding="utf-8")
    print(f"Wrote Stage 7B.4 merged context to {out}: merged_context_feat shape={list(merged.shape)}, alignment_keys={common_keys}, row_order_already_aligned={already_aligned}, warnings={len(warnings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
