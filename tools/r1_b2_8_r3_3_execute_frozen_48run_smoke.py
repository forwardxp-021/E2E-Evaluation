#!/usr/bin/env python3
"""R3.3: R3.2 execution semantics with recursive R3.1 SHA authorization."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import r1_b2_8_r3_2_execute_frozen_48run_smoke as r3_2  # noqa: E402

_BASE_AUTHORIZE = r3_2._authorize

ROOT = Path(__file__).resolve().parents[1]
R1 = ROOT / "docs/stageR/r1"
FINAL_MANIFEST = R1 / "r1_b2_8_r3_3_final_execution_binding_manifest_v1.2.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping): raise ValueError("STOP_BEFORE_SIMULATION_MANIFEST_NOT_OBJECT")
    return value


def _component_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _verify_inherited_r3_1(manifest: Mapping[str, Any]) -> int:
    inherited = manifest.get("inherits_r3_1")
    if not isinstance(inherited, Mapping) or not inherited.get("path") or not inherited.get("sha256"):
        raise PermissionError("STOP_BEFORE_SIMULATION_INHERITED_MANIFEST_SHA_MISMATCH")
    inherited_path = _component_path(str(inherited["path"]))
    if not inherited_path.is_file() or sha(inherited_path) != str(inherited["sha256"]):
        raise PermissionError("STOP_BEFORE_SIMULATION_INHERITED_MANIFEST_SHA_MISMATCH")
    payload = read(inherited_path)
    if payload.get("status") != "FROZEN_READY_FOR_SCIENTIFIC_OWNER_48_RUN_AUTHORIZATION":
        raise PermissionError("STOP_BEFORE_SIMULATION_INHERITED_MANIFEST_NOT_READY")
    components = payload.get("future_execution_components_sha256")
    if not isinstance(components, Mapping) or not components:
        raise PermissionError("STOP_BEFORE_SIMULATION_INHERITED_RUNTIME_CLOSURE_INVALID")
    for path_text, expected in components.items():
        component = _component_path(str(path_text))
        if not component.is_file() or sha(component) != str(expected):
            raise PermissionError(f"STOP_BEFORE_SIMULATION_INHERITED_RUNTIME_COMPONENT_SHA_MISMATCH:{path_text}")
    return len(components)


def authorize(manifest: Path, authorization: Path | None) -> int:
    """Verify Owner → R3.3/R3.2 → inherited R3.1 complete closure."""
    _BASE_AUTHORIZE(manifest, authorization)
    return _verify_inherited_r3_1(read(manifest))


def run(*, execute: bool, output_root: Path | None = None, authorization: Path | None = None, manifest: Path = FINAL_MANIFEST) -> dict[str, Any]:
    if execute:
        # R3.2 run is intentionally reused without modifications.  Rebinding
        # only its authorization symbol preserves all scheduling/run semantics.
        original = r3_2._authorize
        try:
            r3_2._authorize = authorize
            return r3_2.run(execute=True, output_root=output_root, authorization=authorization, manifest=manifest)
        finally:
            r3_2._authorize = original
    return r3_2.run(execute=False, output_root=output_root, authorization=None, manifest=manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true"); parser.add_argument("--authorization", type=Path)
    parser.add_argument("--output-root", type=Path); parser.add_argument("--manifest", type=Path, default=FINAL_MANIFEST)
    args = parser.parse_args()
    result = run(execute=args.execute, output_root=args.output_root, authorization=args.authorization, manifest=args.manifest)
    print(json.dumps({"status": result["status"], "simulation_started": result["simulation_started"]}, ensure_ascii=False))


if __name__ == "__main__": main()
