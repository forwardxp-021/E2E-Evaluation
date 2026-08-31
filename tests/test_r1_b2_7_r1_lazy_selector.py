from __future__ import annotations

from pathlib import Path

import pytest

from tools import r1_future_compliant_smoke_selector_v1_1 as selector


def _candidate(index: int, passed: bool = True, *, token: str | None = None, log: str | None = None) -> selector.RankedIdentity:
    token = token or f"token-{index:03d}"
    log = log or f"log-{index:03d}"
    return selector.RankedIdentity("R-HLC", token, log, index, f"{index:064x}", {"passed": passed})


def _eligibility(candidate: selector.RankedIdentity):
    return bool(candidate.payload["passed"]), {"failure_reason": "SYNTHETIC_FAIL"}


@pytest.mark.parametrize(
    "passing",
    [
        [True] * 24,
        [False, False, True, False, True, True, False, True, True, True, True, True, True, True],
    ],
)
def test_lazy_exactly_matches_exhaustive_for_high_and_low_pass_rates(passing):
    candidates = [_candidate(index, value) for index, value in enumerate(passing)]
    exhaustive = selector.exhaustive_ranked_select(candidates, _eligibility, 6, set(), set())
    lazy, accounting = selector.lazy_rank_ordered_select(iter(candidates), _eligibility, 6, set(), set())
    assert [row[0].scenario_token for row in lazy] == [row[0].scenario_token for row in exhaustive]
    assert accounting["status"] == "TOP_K_DETERMINED_BY_FROZEN_RANK_ORDER"


def test_lazy_matches_exhaustive_with_blacklist_and_duplicate_token_log_shape():
    # The source preflight collapses duplicates before this stage; this list
    # models the resulting unique identities and a blacklisted removal.
    candidates = [_candidate(0, True), _candidate(2, True), _candidate(3, False), _candidate(4, True)]
    candidates = [row for row in candidates if row.scenario_token != "token-002"]
    exhaustive = selector.exhaustive_ranked_select(candidates, _eligibility, 2, set(), set())
    lazy, _ = selector.lazy_rank_ordered_select(candidates, _eligibility, 2, set(), set())
    assert [row[0].scenario_token for row in lazy] == [row[0].scenario_token for row in exhaustive] == ["token-000", "token-004"]


def test_cross_family_log_conflict_is_skipped_in_same_rank_order():
    candidates = [_candidate(0, True, log="used-log"), _candidate(1, True), _candidate(2, True)]
    exhaustive = selector.exhaustive_ranked_select(candidates, _eligibility, 2, {"prior-token"}, {"used-log"})
    lazy, accounting = selector.lazy_rank_ordered_select(candidates, _eligibility, 2, {"prior-token"}, {"used-log"})
    assert [row[0].scenario_token for row in lazy] == [row[0].scenario_token for row in exhaustive] == ["token-001", "token-002"]
    assert accounting["pre_evaluation_cross_family_conflicts"] == 1


def test_tie_like_rank_order_uses_frozen_secondary_fields():
    left = selector.RankedIdentity("R-HLC", "a", "z", 2, "0" * 64, {"passed": True})
    right = selector.RankedIdentity("R-HLC", "b", "a", 1, "0" * 64, {"passed": True})
    ordered = sorted([right, left], key=lambda row: row.rank_key)
    lazy, _ = selector.lazy_rank_ordered_select(ordered, _eligibility, 2, set(), set())
    assert [row[0].scenario_token for row in lazy] == ["a", "b"]


def test_insufficient_pool_is_explicit_stop_status():
    candidates = [_candidate(0, False), _candidate(1, True)]
    lazy, accounting = selector.lazy_rank_ordered_select(candidates, _eligibility, 2, set(), set())
    assert len(lazy) == 1
    assert accounting["status"] == "INSUFFICIENT_ELIGIBLE_FRESH_IDENTITIES"


def test_rank_payload_is_byte_exact_and_no_simulation_terms_are_called():
    salt = "617331678ef4573be11b5408a1dde2c910c8614177541dd51c650c08bc24baf9"
    assert selector.rank_digest(salt, "R-TSB", "token", "log") == "55d04a6e40a05a23509555882d72c77811717c54324f9ee29beee1d0347b4307"
    source = Path(selector.__file__).read_text(encoding="utf-8")
    assert "run_simulation" not in source
    assert "compute_trajectory" not in source
