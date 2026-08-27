#!/usr/bin/env python3
"""Export the read-only R1 nuPlan DB inventory rows to a deterministic CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "docs/stageR/r1/r1_official_nuplan_db_inventory_rows_v0.1.json"
DEFAULT_OUTPUT = ROOT / "docs/stageR/r1/r1_official_nuplan_db_inventory_v0.1.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite inventory CSV: {args.output}")
    payload = json.loads(args.source.read_text(encoding="utf-8"))
    columns = [str(value) for value in payload["columns"]]
    rows = payload["rows"]
    if not rows or any(set(row) != set(columns) for row in rows):
        raise ValueError("inventory rows are empty or do not match the declared schema")
    with args.output.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
