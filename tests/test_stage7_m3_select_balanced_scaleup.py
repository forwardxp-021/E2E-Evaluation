import csv
import json
from pathlib import Path

from tools.stage7_m3_select_balanced_scaleup import (
    DEFAULT_QUOTAS,
    fill_quotas,
    load_successful_seed_rows,
    manifest_hash,
)


def test_load_successful_seed_rows_preserves_noncontiguous_source_axis(tmp_path: Path) -> None:
    context = tmp_path / "merged_metadata.csv"
    rows = [
        {
            "log_name": f"log_{index}",
            "scenario_token": f"{index:016x}",
            "scene_token": f"{index:016x}",
            "scenario_type": "following_lane_with_slow_lead",
            "bucket": "following_slow_lead",
        }
        for index in range(4)
    ]
    with context.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    sim_dir = tmp_path / "sim"
    sim_dir.mkdir()
    (sim_dir / "simulated_ego_seq_index.json").write_text(
        json.dumps({"scenario_axis": ["0", "2", "3"]}),
        encoding="utf-8",
    )
    selected, failed = load_successful_seed_rows(context, sim_dir)
    assert [row["source_scenario_index"] for row in selected] == ["0", "2", "3"]
    assert failed == ["0000000000000001"]


def test_fill_quotas_is_deterministic_and_respects_log_cap() -> None:
    quotas = {
        "actual_verified_lane_change": 1,
        "following_interaction": 2,
    }
    seeds = [{
        "log_name": "seed_log",
        "scenario_token": "seed",
        "scenario_type": "changing_lane",
        "bucket": "actual_verified_lane_change",
    }]
    candidates = {
        "following_interaction": [
            {
                "log_name": "log_a",
                "scenario_token": "a",
                "scenario_type": "following_lane_with_lead",
                "bucket": "following_interaction",
            },
            {
                "log_name": "log_a",
                "scenario_token": "b",
                "scenario_type": "following_lane_with_lead",
                "bucket": "following_interaction",
            },
            {
                "log_name": "log_b",
                "scenario_token": "c",
                "scenario_type": "following_lane_with_lead",
                "bucket": "following_interaction",
            },
        ]
    }
    selected = fill_quotas(seeds, candidates, quotas, max_per_log=1)
    assert [row["scenario_token"] for row in selected] == ["seed", "a", "c"]
    assert manifest_hash(selected) == manifest_hash(selected)


def test_default_m3_quota_totals_fifty() -> None:
    assert sum(DEFAULT_QUOTAS.values()) == 50
