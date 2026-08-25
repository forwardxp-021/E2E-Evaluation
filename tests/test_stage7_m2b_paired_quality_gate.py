from pathlib import Path

from tools.stage7_m2b_build_paired_quality_gate import build_quality_gate, classify_quality_tier


def _row(index, scenario, planner, fallback, ambiguous, bad=0.0):
    return {
        "global_row": str(index),
        "scenario_index": str(scenario),
        "planner_name": planner,
        "fallback_rate": str(fallback),
        "ambiguous_frame_rate": str(ambiguous),
        "bad_frame_rate": str(bad),
        "quality_eligible_frame_rate": str(1.0 - fallback - ambiguous),
    }


def test_quality_tiers_use_frame_rates() -> None:
    kwargs = dict(
        tier_a_max_fallback=0.05,
        tier_a_max_ambiguous=0.05,
        tier_b_max_fallback=0.20,
        tier_b_max_ambiguous=0.20,
    )
    assert classify_quality_tier(_row(0, 0, "a", 0.01, 0.02), **kwargs)[0] == "A"
    assert classify_quality_tier(_row(0, 0, "a", 0.10, 0.02), **kwargs)[0] == "B"
    assert classify_quality_tier(_row(0, 0, "a", 0.01, 0.30), **kwargs)[0] == "C"
    assert classify_quality_tier(_row(0, 0, "a", 0.01, 0.02, 0.01), **kwargs)[0] == "C"


def test_pair_gate_uses_worst_planner_and_preserves_noncontiguous_scenarios() -> None:
    rows = [
        _row(0, 0, "a", 0.01, 0.01),
        _row(1, 0, "b", 0.02, 0.02),
        _row(2, 14, "a", 0.01, 0.01),
        _row(3, 14, "b", 0.10, 0.02),
        _row(4, 18, "a", 0.01, 0.30),
        _row(5, 18, "b", 0.01, 0.01),
    ]
    _, pairs, indices = build_quality_gate(
        rows,
        tier_a_max_fallback=0.05,
        tier_a_max_ambiguous=0.05,
        tier_b_max_fallback=0.20,
        tier_b_max_ambiguous=0.20,
    )
    assert [row["scenario_index"] for row in pairs] == [0, 14, 18]
    assert [row["pair_quality_tier"] for row in pairs] == ["A", "B", "C"]
    assert indices["tier_a_a"] == [0]
    assert indices["tier_a_b"] == [1]
    assert indices["tier_b_inclusive_a"] == [0, 2]
    assert indices["tier_b_inclusive_b"] == [1, 3]


def test_pair_gate_rejects_missing_planner_row() -> None:
    rows = [
        _row(0, 0, "a", 0.01, 0.01),
        _row(1, 0, "b", 0.01, 0.01),
        _row(2, 14, "a", 0.01, 0.01),
    ]
    try:
        build_quality_gate(
            rows,
            tier_a_max_fallback=0.05,
            tier_a_max_ambiguous=0.05,
            tier_b_max_fallback=0.20,
            tier_b_max_ambiguous=0.20,
        )
    except ValueError as exc:
        assert "planner set mismatch" in str(exc)
    else:
        raise AssertionError("missing planner row must fail")


def test_primary_dataset_label_is_dynamic_for_scaleup() -> None:
    source = Path("tools/stage7_m2b_build_paired_quality_gate.py").read_text(encoding="utf-8")
    assert "full_17_planner_paired_scenarios" not in source
    assert 'f"full_{full_pairs}_planner_paired_scenarios"' in source
