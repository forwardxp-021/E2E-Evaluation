#!/usr/bin/env python3
"""Freeze Stage 6L numeric Go/No-Go rules before representation BDD is read."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage7_m6_4b_run_locked_rollouts import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--addendum_json", type=Path, required=True)
    parser.add_argument("--freeze_manifest", type=Path, required=True)
    parser.add_argument("--representation_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def run(args: argparse.Namespace) -> dict:
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"output_dir exists: {output_dir}; pass --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    addendum = read_json(args.addendum_json.resolve())
    freeze = read_json(args.freeze_manifest.resolve())
    representations = read_json(args.representation_manifest.resolve())
    if addendum.get("schema_version") != "stage6l_preanalysis_decision_addendum_v1":
        raise ValueError("Unexpected addendum schema")
    if addendum.get("frozen_before_representation_bdd_read") is not True:
        raise ValueError("Addendum is not marked as pre-BDD")
    if freeze.get("status") != "FROZEN_BEFORE_STAGE6L_REPRESENTATION_ABLATION":
        raise ValueError("Invalid Stage 6L input freeze")
    if representations.get("status") != "STAGE6L_A_D_REPRESENTATIONS_READY":
        raise ValueError("Representations are not ready")
    manifest = {
        "schema_version": "stage6l_preanalysis_decision_addendum_freeze_v1",
        "status": "FROZEN_BEFORE_STAGE6L_REPRESENTATION_BDD_READ",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "issue": 253,
        "representation_bdd_read": False,
        "addendum": addendum,
        "addendum_sha256": sha256_file(args.addendum_json.resolve()),
        "input_freeze_manifest_sha256": sha256_file(args.freeze_manifest.resolve()),
        "representation_manifest_sha256": sha256_file(args.representation_manifest.resolve()),
        "tool_sha256": sha256_file(Path(__file__).resolve()),
    }
    path = output_dir / "stage6l_preanalysis_decision_addendum_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "stage6l_preanalysis_decision_addendum_report_zh.md").write_text(
        "# Stage 6L 解盲前 Go/No-Go 决策补充冻结\n\n"
        "- 状态：`FROZEN_BEFORE_STAGE6L_REPRESENTATION_BDD_READ`\n"
        "- representation BDD已读取：`false`\n"
        "- 规则用途：决定是否优先做context-v2或准备新checkpoint协议；GO不自动授权重训。\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
