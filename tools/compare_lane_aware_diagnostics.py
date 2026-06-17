#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from tools.lane_aware_assignment import SLOT_NAMES



def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_rate(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def load_shard_paths(dataset_dir: Path) -> List[Path]:
    manifest = read_json(dataset_dir / "shard_manifest.json")
    out: List[Path] = []
    for row in manifest.get("shards", []) if isinstance(manifest.get("shards"), list) else []:
        sp = row.get("shard_path") if isinstance(row, dict) else None
        if sp:
            p = Path(sp)
            out.append(p if p.is_absolute() else dataset_dir / p)
    if out:
        return out
    shards_dir = dataset_dir / "shards"
    if shards_dir.exists():
        return sorted(p for p in shards_dir.iterdir() if p.is_dir())
    return [dataset_dir]


def slot_coverage_and_switches(dataset_dir: Path, max_rows: Optional[int] = None) -> Dict[str, Any]:
    try:
        import numpy as np
    except ModuleNotFoundError:
        return {
            "slot_coverage_by_slot": {},
            "slot_switch_rate_by_slot": {},
            "slot_switch_count_by_slot": {},
            "array_scan_warning": "numpy is unavailable; skipped neighbor_seq.npy and neighbor_slot_ids.npy array diagnostics",
        }
    valid_counts = Counter()
    total_counts = Counter()
    transitions = Counter()
    switches = Counter()
    rows_seen = 0
    for shard in load_shard_paths(dataset_dir):
        nbr_path = shard / "neighbor_seq.npy"
        ids_path = shard / "neighbor_slot_ids.npy"
        if nbr_path.exists():
            nbr = np.load(nbr_path, mmap_mode="r")
            take = nbr.shape[0] if max_rows is None else max(0, min(nbr.shape[0], max_rows - rows_seen))
            if take > 0:
                for si, sn in enumerate(SLOT_NAMES):
                    valid = nbr[:take, si, :, 0] > 0.5
                    valid_counts[sn] += int(np.sum(valid))
                    total_counts[sn] += int(valid.size)
                rows_seen += take
        if ids_path.exists():
            ids = np.load(ids_path, allow_pickle=True)
            take = ids.shape[0] if max_rows is None else min(ids.shape[0], max_rows)
            for row in ids[:take]:
                for si, sn in enumerate(SLOT_NAMES):
                    seq = list(row[si]) if np.asarray(row[si]).ndim else [str(row[si])]
                    prev = "-1"
                    for tok in seq:
                        tok = str(tok)
                        if tok in {"", "None", "nan"}:
                            tok = "-1"
                        if tok != "-1" and prev != "-1":
                            transitions[sn] += 1
                            if tok != prev:
                                switches[sn] += 1
                        prev = tok
    return {
        "slot_coverage_by_slot": {sn: safe_rate(valid_counts[sn], total_counts[sn]) for sn in SLOT_NAMES},
        "slot_switch_rate_by_slot": {sn: safe_rate(switches[sn], transitions[sn]) for sn in SLOT_NAMES},
        "slot_switch_count_by_slot": {sn: int(switches[sn]) for sn in SLOT_NAMES},
    }


def summarize_lane_debug_csv(dataset_dir: Path) -> Dict[str, Any]:
    method_counts = {sn: Counter() for sn in SLOT_NAMES}
    reason_counts = Counter()
    for shard in load_shard_paths(dataset_dir):
        csv_path = shard / "lane_assignment_debug.csv"
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                sn = row.get("slot_name", "")
                if sn in method_counts:
                    method = row.get("assignment_method") or row.get("method") or "unknown"
                    method_counts[sn][method] += 1
                reason = row.get("rejection_reason") or row.get("fallback_reason") or row.get("reason") or ""
                if reason:
                    reason_counts[reason] += 1
    return {
        "assignment_method_counts_by_slot_from_debug_csv": {sn: dict(method_counts[sn]) for sn in SLOT_NAMES},
        "debug_csv_reason_counts": dict(reason_counts),
    }


def summarize_waymo(dataset_dir: Path, max_rows: Optional[int]) -> Dict[str, Any]:
    summary = read_json(dataset_dir / "build_summary.json") or read_json(dataset_dir / "neighbor_context_summary.json")
    n = int(summary.get("n_windows_kept", 0) or summary.get("n_windows", 0) or 0)
    metrics = {
        "dataset": "waymo_stage5",
        "path": str(dataset_dir),
        "n_rows": n,
        "lane_assignment_available": bool(summary.get("lane_assignment_success_rate", 0.0) > 0 or summary.get("lane_assignment_success_count_kept", 0) > 0),
        "lane_assignment_available_rate": as_float(summary.get("lane_assignment_success_rate"), 0.0),
        "fallback_assignment_used_rate": as_float(summary.get("fallback_assignment_rate"), as_float(summary.get("fallback_assignment_used_rate"), None)),
        "candidate_projection_success_rate": as_float(summary.get("lane_projection_success_rate"), None),
        "adjacency_source_counts": summary.get("adjacency_source_counts", {}),
        "lane_context_quality_counts": summary.get("lane_context_quality_counts", {}),
        "lane_context_quality_reason_counts": summary.get("lane_context_quality_reason_counts", {}),
        "rejection_reason_counts": summary.get("slot_rejection_reason_counts", {}),
        "slot_coverage_by_slot": summary.get("slot_valid_ratio") or {sn: 1.0 - as_float((summary.get("empty_slot_ratio_by_slot") or {}).get(sn), 1.0) for sn in SLOT_NAMES},
        "slot_switch_rate_by_slot": summary.get("slot_id_switch_rate_by_slot", {}),
        "source_files": [str(dataset_dir / "build_summary.json"), str(dataset_dir / "neighbor_context_summary.json")],
    }
    metrics.update(slot_coverage_and_switches(dataset_dir, max_rows=max_rows))
    metrics.update(summarize_lane_debug_csv(dataset_dir))
    return metrics


def flatten_rejection_counts(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    out = Counter()
    for d in rows:
        for by_slot in (d.get("slot_rejection_reason_counts") or {}).values():
            if isinstance(by_slot, dict):
                out.update({str(k): int(v) for k, v in by_slot.items()})
    return dict(out)


def summarize_nuplan(dataset_dir: Path, max_rows: Optional[int]) -> Dict[str, Any]:
    warnings = read_json(dataset_dir / "warnings.json")
    validation = warnings.get("validation", {}) if isinstance(warnings.get("validation"), dict) else {}
    assign_rows = read_json(dataset_dir / "assignment_debug.json")
    rows = assign_rows if isinstance(assign_rows, list) else []
    metrics = {
        "dataset": "nuplan_stage7_adapter",
        "path": str(dataset_dir),
        "n_rows": len(rows) or validation.get("n_rows"),
        "lane_assignment_available": bool(validation.get("lane_assignment_available", False)),
        "lane_assignment_available_rate": as_float(validation.get("ego_lane_projection_success_rate"), None),
        "fallback_assignment_used_rate": as_float(validation.get("fallback_assignment_used_rate"), None),
        "candidate_projection_success_rate": as_float(validation.get("candidate_lane_projection_success_rate"), None),
        "adjacency_source_counts": validation.get("adjacency_source_counts", warnings.get("adjacency_source_counts", {})),
        "lane_context_quality_counts": dict(Counter(d.get("lane_context_quality", "unknown") for d in rows)) if rows else {},
        "lane_context_quality_reason_counts": dict(Counter(r for d in rows for r in d.get("lane_context_quality_reasons", []))) if rows else {},
        "rejection_reason_counts": flatten_rejection_counts(rows),
        "slot_coverage_by_slot": validation.get("slot_coverage_by_slot", {}),
        "slot_switch_rate_by_slot": validation.get("slot_id_switch_rate_by_slot", {}),
        "source_files": [str(dataset_dir / "warnings.json"), str(dataset_dir / "assignment_debug.json"), str(dataset_dir / "slot_assignment_report.md")],
    }
    if not metrics["lane_context_quality_counts"]:
        metrics.update(summarize_lane_debug_csv(dataset_dir))
    cov_switch = slot_coverage_and_switches(dataset_dir, max_rows=max_rows)
    if cov_switch["slot_coverage_by_slot"]:
        metrics["slot_coverage_by_slot_from_arrays"] = cov_switch["slot_coverage_by_slot"]
    if cov_switch["slot_switch_rate_by_slot"]:
        metrics["slot_switch_rate_by_slot_from_arrays"] = cov_switch["slot_switch_rate_by_slot"]
    return metrics


def diagnose(waymo: Dict[str, Any], nuplan: Dict[str, Any], fallback_gap_threshold: float) -> Dict[str, Any]:
    wf = as_float(waymo.get("fallback_assignment_used_rate"), None)
    nf = as_float(nuplan.get("fallback_assignment_used_rate"), None)
    wc = as_float(waymo.get("candidate_projection_success_rate"), None)
    nc = as_float(nuplan.get("candidate_projection_success_rate"), None)
    fallback_comparable = wf is not None and nf is not None
    projection_comparable = wc is not None and nc is not None
    missing_waymo_metrics = {
        "fallback_assignment_used_rate": wf is None,
        "candidate_projection_success_rate": wc is None,
    }

    if wf is None and wc is None:
        verdict = "inconclusive_missing_waymo_metrics"
        reason = "Waymo fallback/projection metrics are unavailable, so nuPlan cannot be compared conservatively under the shared Stage5D assignment interface."
    elif projection_comparable and nc + 0.20 < wc:
        verdict = "nuplan_adapter_or_map_projection_issue"
        reason = "nuPlan candidate projection success is substantially lower than Waymo under the shared Stage5D assignment interface."
    elif fallback_comparable and nf > wf + fallback_gap_threshold:
        verdict = "nuplan_adapter_or_map_projection_issue"
        reason = "nuPlan geometric fallback rate is substantially higher than Waymo under the same Stage5D assignment interface."
    else:
        verdict = "generic_stage5_lane_aware_limitation_or_inconclusive"
        reason = "nuPlan is not clearly worse than Waymo on comparable fallback/projection metrics, or comparable metrics are incomplete."
    return {
        "verdict": verdict,
        "reason": reason,
        "waymo_fallback_rate": wf,
        "nuplan_fallback_rate": nf,
        "waymo_candidate_projection_success_rate": wc,
        "nuplan_candidate_projection_success_rate": nc,
        "fallback_rate_comparable": fallback_comparable,
        "candidate_projection_success_comparable": projection_comparable,
        "missing_waymo_metrics": missing_waymo_metrics,
    }


def render_markdown(payload: Dict[str, Any]) -> str:
    diagnosis = payload["diagnosis"]
    lines = [
        "# Stage5D Lane-Aware Assignment Cross-Dataset Diagnostics",
        "",
        "## Architecture constraint",
        "",
        "- Stage5D CORE / `tools.lane_aware_assignment.py` remains the only lane-aware assignment implementation.",
        "- Stage7 nuPlan is treated only as an adapter that emits Stage5-compatible `LaneInfo` and candidate states.",
        "",
        "## Diagnosis",
        "",
        f"- verdict: `{diagnosis['verdict']}`",
        f"- reason: {diagnosis['reason']}",
        f"- fallback_rate_comparable: `{diagnosis.get('fallback_rate_comparable')}`",
        f"- candidate_projection_success_comparable: `{diagnosis.get('candidate_projection_success_comparable')}`",
        f"- missing_waymo_metrics: `{diagnosis.get('missing_waymo_metrics')}`",
        "",
        "## Comparable metrics",
    ]
    for name in ("waymo", "nuplan"):
        m = payload[name]
        lines += ["", f"### {name}", "", f"- path: `{m.get('path')}`", f"- lane_assignment_available: `{m.get('lane_assignment_available')}`", f"- lane_assignment_available_rate: `{m.get('lane_assignment_available_rate')}`", f"- fallback_assignment_used_rate: `{m.get('fallback_assignment_used_rate')}`", f"- candidate_projection_success_rate: `{m.get('candidate_projection_success_rate')}`", f"- adjacency_source_counts: `{m.get('adjacency_source_counts')}`", f"- lane_context_quality counts: `{m.get('lane_context_quality_counts')}`", f"- rejection reason counts: `{m.get('rejection_reason_counts')}`", f"- slot coverage by slot: `{m.get('slot_coverage_by_slot')}`", f"- slot switch rate by slot: `{m.get('slot_switch_rate_by_slot')}`"]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare shared Stage5D lane-aware assignment diagnostics between Waymo Stage5 output and nuPlan Stage7 adapter output.")
    p.add_argument("--waymo_dir", type=Path, required=True, help="Waymo Stage5/Stage5D output directory containing build_summary.json and/or shards.")
    p.add_argument("--nuplan_dir", type=Path, required=True, help="nuPlan Stage7E/Stage5D-compatible output directory containing warnings.json and optional assignment_debug.json.")
    p.add_argument("--out_dir", type=Path, required=True, help="Directory for lane_aware_diagnostic_comparison.json/.md.")
    p.add_argument("--max_rows", type=int, default=None, help="Optional cap when scanning neighbor arrays for coverage/switch diagnostics.")
    p.add_argument("--fallback_gap_threshold", type=float, default=0.20, help="Minimum nuPlan-minus-Waymo fallback-rate gap used for adapter-issue diagnosis.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.waymo_dir.exists():
        raise FileNotFoundError(f"Waymo output directory does not exist: {args.waymo_dir}")
    if not args.nuplan_dir.exists():
        raise FileNotFoundError(f"nuPlan output directory does not exist: {args.nuplan_dir}")
    waymo = summarize_waymo(args.waymo_dir, args.max_rows)
    nuplan = summarize_nuplan(args.nuplan_dir, args.max_rows)
    payload = {"waymo": waymo, "nuplan": nuplan, "diagnosis": diagnose(waymo, nuplan, args.fallback_gap_threshold)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "lane_aware_diagnostic_comparison.json", payload)
    (args.out_dir / "lane_aware_diagnostic_comparison.md").write_text(render_markdown(payload), encoding="utf-8")
    print(f"Wrote diagnostics: {args.out_dir / 'lane_aware_diagnostic_comparison.json'}")
    print(f"Wrote report: {args.out_dir / 'lane_aware_diagnostic_comparison.md'}")


if __name__ == "__main__":
    main()
