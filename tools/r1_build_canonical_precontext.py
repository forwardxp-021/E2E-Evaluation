#!/usr/bin/env python3
"""Build one independent frozen R1 canonical pre-context record from JSON input."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.r1_context_mechanism_core import build_canonical_context_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze an exact 10-frame R1 pre-context as canonical JSON.")
    parser.add_argument("--input", type=Path, required=True, help="Input JSON containing one context payload and exactly ten frames.")
    parser.add_argument("--output", type=Path, required=True, help="New canonical context JSON path; refuses overwrite.")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing context record: {args.output}")
    with args.input.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    record = build_canonical_context_record(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "eligible": record["eligible"], "pre_context_raw_hash": record["pre_context_raw_hash"], "canonical_context_json_hash": record["canonical_context_json_hash"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
