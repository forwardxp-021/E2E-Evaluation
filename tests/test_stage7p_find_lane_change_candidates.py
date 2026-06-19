import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools.stage7p_find_lane_change_candidates as finder


def test_find_lane_change_candidates_from_metadata_and_events(tmp_path: Path):
    ctx = tmp_path / "ctx"
    events = ctx / "behavior_events_v2"
    out = tmp_path / "out"
    events.mkdir(parents=True)
    (ctx / "merged_metadata.csv").write_text(
        "scenario_id,scenario_type,log_name\n"
        "s0,following,log_a\n"
        "s1,changing_lane,log_b\n"
        "s2,unknown,cut_in_merge_log\n",
        encoding="utf-8",
    )
    (events / "behavior_event_bins_v2.csv").write_text(
        "global_row,task_lane_change\n0,0\n1,0\n2,1\n",
        encoding="utf-8",
    )

    rc = finder.run(SimpleNamespace(context_dir=str(ctx), output_dir=str(out), top_k=20, behavior_events_dir=""))

    assert rc == 0
    summary = json.loads((out / "lane_change_candidate_summary.json").read_text(encoding="utf-8"))
    assert summary["metadata_rows"] == 3
    assert summary["candidate_rows"] == 2
    assert summary["behavior_events"]["available"] is True
    rows = list(csv.DictReader((out / "lane_change_candidate_metadata.csv").open(encoding="utf-8")))
    assert [row["scenario_id"] for row in rows] == ["s2", "s1"]
    assert "behavior_event_bins_v2:task_lane_change" in rows[0]["match_sources"]
    assert (out / "lane_change_candidate_report.md").is_file()
