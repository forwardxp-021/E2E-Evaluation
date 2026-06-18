#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import ast
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")
CLASS_NAMES = ["PDMClosedPlanner", "AbstractPDMClosedPlanner", "AbstractPDMPlanner", "PDMGenerator", "PDMProposal", "PDMProposalManager", "PDMSimulator", "PDMScorer"]
GROUP_RULES = [
    ("proposal generation / trajectory sampling", ["trajectory_sampling", "proposal_sampling", "proposal", "sample", "trajectory", "horizon", "interval", "num_poses"]),
    ("longitudinal / speed / progress", ["idm_policies", "speed", "velocity", "accel", "decel", "progress", "idm", "headway", "gap"]),
    ("lateral / offset / lane-change-like behavior", ["lateral_offsets", "lateral", "offset", "lane_change", "steer", "yaw"]),
    ("route/path following", ["map_radius", "route", "path", "centerline", "lane", "map"]),
    ("scoring / weights", ["score", "weight", "cost"]),
    ("comfort", ["comfort", "jerk"]),
    ("collision / safety / emergency brake", ["collision", "safety", "brake", "emergency", "ttc"]),
    ("simulator / dynamics", ["simulator", "simulation", "dynamics", "vehicle"]),
]


def strip_inline_comment(text: str) -> str:
    in_single = in_double = False
    bracket_depth = 0
    out = []
    prev = ""
    for ch in text:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif not in_single and not in_double:
            if ch in "[({":
                bracket_depth += 1
            elif ch in "])}" and bracket_depth > 0:
                bracket_depth -= 1
            elif ch == "#" and (prev == "" or prev.isspace()):
                break
        out.append(ch)
        prev = ch
    return "".join(out).strip()


def parse_scalar(text: str) -> Any:
    raw = strip_inline_comment(text).strip().strip('"\'')
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    if raw.lower() in {"null", "none"}:
        return None
    if NUM_RE.match(raw):
        return float(raw) if any(c in raw for c in ".eE") else int(raw)
    if raw.startswith("[") and raw.endswith("]"):
        try:
            value = ast.literal_eval(raw)
            return value
        except (SyntaxError, ValueError):
            return [parse_scalar(v.strip()) for v in raw[1:-1].split(",") if v.strip()]
    return raw


def kind_of(value: Any, name: str = "") -> str:
    if name.endswith("._target_") or name == "_target_":
        return "target_path"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "numeric_scalar"
    if isinstance(value, list) and value and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
        return "numeric_list"
    if isinstance(value, str):
        return "string"
    return "unknown"


def parse_simple_yaml(path: Path) -> List[Dict[str, Any]]:
    """Small dependency-free YAML subset parser with clean inline-comment handling and line numbers."""
    rows: List[Dict[str, Any]] = []
    stack: List[Tuple[int, str]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, value = line.strip().split(":", 1)
        key = key.strip()
        while stack and stack[-1][0] >= indent:
            stack.pop()
        full_key = ".".join([item[1] for item in stack] + [key])
        clean_value = strip_inline_comment(value)
        if clean_value == "":
            stack.append((indent, key))
            continue
        parsed = parse_scalar(clean_value)
        rows.append({"source": "yaml", "path": str(path), "line": lineno, "name": full_key, "value": parsed, "kind": kind_of(parsed, full_key), "group": classify(full_key)})
    return rows


def classify(name: str) -> str:
    lower = name.lower()
    for group, keys in GROUP_RULES:
        if any(k in lower for k in keys):
            return group
    return "unknown"


def iter_py_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return root.rglob("*.py")


def inspect_class_signatures(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    wanted = set(CLASS_NAMES)
    for path in iter_py_files(root):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in wanted:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                        args = item.args.args
                        defaults = [None] * (len(args) - len(item.args.defaults)) + list(item.args.defaults)
                        for arg, default in zip(args, defaults):
                            if arg.arg == "self":
                                continue
                            try:
                                default_value = ast.literal_eval(default) if default is not None else "<required>"
                            except (ValueError, TypeError):
                                default_value = ast.unparse(default) if default is not None else "<required>"
                            rows.append({"source": "class_signature", "path": str(path), "line": item.lineno, "class": node.name, "name": f"{node.name}.__init__.{arg.arg}", "value": default_value, "kind": "verified_class_arg", "group": classify(arg.arg)})
    return rows


def verified_config_rows(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {r["name"]: r for r in rows if r.get("source") == "yaml"}


def find_key(rows_by_name: Dict[str, Dict[str, Any]], suffix: str) -> str | None:
    if suffix in rows_by_name:
        return suffix
    matches = [name for name in rows_by_name if name.endswith("." + suffix)]
    return matches[0] if matches else None


def blueprint_override_candidates(rows: List[Dict[str, Any]]) -> Dict[str, List[Tuple[str, Any, str]]]:
    by = verified_config_rows(rows)
    def add_if(name: str, value: Any, note: str, out: List[Tuple[str, Any, str]]):
        key = find_key(by, name)
        if key:
            out.append((key, value, note))
    conservative: List[Tuple[str, Any, str]] = []
    assertive: List[Tuple[str, Any, str]] = []
    comfort: List[Tuple[str, Any, str]] = []
    add_if("speed_limit_fraction", [0.2, 0.4, 0.6, 0.8], "verified_config_key; remove highest speed fraction candidate", conservative)
    add_if("fallback_target_velocity", 10.0, "verified_config_key; lower fallback speed candidate", conservative)
    add_if("min_gap_to_lead_agent", 2.0, "verified_config_key; larger IDM gap candidate", conservative)
    add_if("headway_time", 2.0, "verified_config_key; larger time headway candidate", conservative)
    add_if("accel_max", 1.0, "verified_config_key; lower acceleration candidate", conservative)
    add_if("decel_max", 2.0, "verified_config_key; caution: lower decel may improve comfort but reduce emergency authority", conservative)
    add_if("lateral_offsets", [-0.5, 0.5], "verified_config_key; narrower lateral offsets candidate", conservative)
    add_if("speed_limit_fraction", [0.2, 0.4, 0.6, 0.8, 1.0], "verified_config_key; keep highest speed fraction candidate", assertive)
    add_if("fallback_target_velocity", 18.0, "verified_config_key; higher fallback speed candidate", assertive)
    add_if("min_gap_to_lead_agent", 0.5, "verified_config_key; smaller IDM gap candidate", assertive)
    add_if("headway_time", 1.0, "verified_config_key; smaller time headway candidate", assertive)
    add_if("accel_max", 2.0, "verified_config_key; higher acceleration candidate", assertive)
    add_if("lateral_offsets", [-1.5, 1.5], "verified_config_key; wider lateral offsets candidate", assertive)
    add_if("accel_max", 1.0, "verified_config_key; lower acceleration comfort candidate", comfort)
    add_if("decel_max", 2.0, "verified_config_key; lower deceleration comfort candidate if safety constraints remain satisfied", comfort)
    add_if("headway_time", 2.0, "verified_config_key; larger headway comfort candidate", comfort)
    add_if("lateral_offsets", [-0.5, 0.5], "verified_config_key; narrow lateral offsets comfort candidate", comfort)
    return {"pdm_closed_conservative_candidate": conservative, "pdm_closed_assertive_candidate": assertive, "pdm_closed_comfort_candidate": comfort}


def write_outputs(rows: List[Dict[str, Any]], out_dir: Path, config_path: Path) -> None:
    summary = {"planner_config": str(config_path), "parameter_count": len(rows), "groups": {}}
    for row in rows:
        summary["groups"].setdefault(row["group"], 0)
        summary["groups"][row["group"]] += 1
    (out_dir / "pdm_closed_parameter_summary.json").write_text(json.dumps({"summary": summary, "parameters": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = ["source", "path", "line", "class", "name", "value", "kind", "group", "override_status"]
    with (out_dir / "pdm_closed_parameter_table.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {**row, "value": json.dumps(row.get("value"), ensure_ascii=False), "override_status": "verified_config_key" if row["source"] == "yaml" else "verified_class_arg"}
            writer.writerow(out)
    lines = ["# Stage7P PDM Closed Parameter Report", "", f"- config: `{config_path}`", f"- parameters: `{len(rows)}`", "- parser: dependency-free YAML subset parser with inline-comment stripping and preserved source line numbers", "", "## Parameters"]
    for row in rows:
        status = "verified_config_key" if row["source"] == "yaml" else "verified_class_arg"
        lines.append(f"- `{row['name']}` = `{row.get('value')}` ({row['kind']}, {row['group']}, {status}, source={row['source']}, line={row.get('line')})")
    (out_dir / "pdm_closed_parameter_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    blueprint = ["# PDM Closed Variant Blueprint", "", "Concrete override candidates below are emitted only for `verified_config_key` YAML keys discovered in the closed planner config.", "PDM open/hybrid variants are intentionally not marked runnable here because they require `checkpoint_path`.", "Labels used: `verified_config_key`, `verified_class_arg`, `inferred_candidate`, `unsafe_unknown`.", "", "## pdm_closed_default", "- status: verified_config_key baseline; runnable if the local PDM closed planner environment is installed.", ""]
    candidates = blueprint_override_candidates(rows)
    for name, items in candidates.items():
        blueprint.extend([f"## {name}", "- status: inferred_candidate", "- concrete command overrides (all listed keys are verified_config_key):"])
        if items:
            for key, value, note in items:
                blueprint.append(f"  - `+planner.{key}={json.dumps(value, ensure_ascii=False)}` — {note}; inferred_candidate")
        else:
            blueprint.append("  - no concrete verified_config_key overrides found")
        blueprint.append("")
    blueprint.extend(["## open_or_hybrid_pdm_variants", "- status: unsafe_unknown", "- not marked runnable because open/hybrid variants require `checkpoint_path` and additional validation.", ""])
    (out_dir / "pdm_closed_variant_blueprint.md").write_text("\n".join(blueprint), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    root = Path(args.tuplan_garage_root).expanduser().resolve()
    config_path = root / "tuplan_garage" / "planning" / "script" / "config" / "simulation" / "planner" / f"{args.planner_config_name}.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing PDM planner config: {config_path}")
    out_dir = Path(args.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output_dir exists and is not empty: {out_dir}. Use --overwrite.")
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = parse_simple_yaml(config_path) + inspect_class_signatures(root / "tuplan_garage")
    write_outputs(rows, out_dir, config_path)
    print(json.dumps({"output_dir": str(out_dir), "parameter_count": len(rows)}, indent=2, ensure_ascii=False))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage7P PDM closed planner config parameter discovery report.")
    parser.add_argument("--tuplan_garage_root", required=True)
    parser.add_argument("--planner_config_name", default="pdm_closed_planner")
    parser.add_argument("--output_dir", default="outputs/stage7p_pdm_closed_config_params_v1")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
