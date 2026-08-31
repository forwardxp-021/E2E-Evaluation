#!/usr/bin/env python3
"""Frozen B2.7-R1 source identity and lazy rank-selection primitives.

This module contains no planner outcome, simulation, or representation input.
It implements only the corrected global-identity semantics and the provably
equivalent rank-ordered execution order authorized for B2.7-R1.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, MutableSet, Sequence


@dataclass(frozen=True)
class RankedIdentity:
    """One globally unique source identity in frozen rank order."""

    family: str
    scenario_token: str
    log_id: str
    timestamp: int
    rank_sha256: str
    payload: Mapping[str, Any]

    @property
    def rank_key(self) -> tuple[str, str, str, int]:
        return (self.rank_sha256, self.scenario_token, self.log_id, self.timestamp)


def rank_digest(salt_sha256: str, family: str, scenario_token: str, log_id: str) -> str:
    """Return the frozen, byte-exact B2.7 ranking digest."""
    return hashlib.sha256(f"{salt_sha256}|{family}|{scenario_token}|{log_id}".encode("utf-8")).hexdigest()


def lazy_rank_ordered_select(
    ranked_candidates: Iterable[RankedIdentity],
    eligibility: Callable[[RankedIdentity], tuple[bool, Mapping[str, Any]]],
    target_count: int,
    used_tokens: MutableSet[str],
    used_logs: MutableSet[str],
) -> tuple[list[tuple[RankedIdentity, Mapping[str, Any]]], dict[str, Any]]:
    """Select the first K eligible non-conflicting identities in frozen order.

    The result is exactly the first K entries of an exhaustive eligible-and-rank
    ordering, because every unvisited entry has a rank no lower than the last
    selected entry.  Conflicting identities are skipped before expensive
    evaluation only when the already-frozen earlier family makes them
    unselectable by definition.
    """
    if target_count <= 0:
        raise ValueError("TARGET_COUNT_MUST_BE_POSITIVE")
    selected: list[tuple[RankedIdentity, Mapping[str, Any]]] = []
    failures: dict[str, int] = {}
    expensive_evaluations = 0
    pre_evaluation_cross_family_conflicts = 0
    last_rank: str | None = None
    for candidate in ranked_candidates:
        if candidate.scenario_token in used_tokens or candidate.log_id in used_logs:
            pre_evaluation_cross_family_conflicts += 1
            continue
        expensive_evaluations += 1
        passed, audit = eligibility(candidate)
        if not passed:
            reason = str(audit.get("failure_reason", "FROZEN_ELIGIBILITY_FAIL"))
            failures[reason] = failures.get(reason, 0) + 1
            continue
        selected.append((candidate, audit))
        used_tokens.add(candidate.scenario_token)
        used_logs.add(candidate.log_id)
        last_rank = candidate.rank_sha256
        if len(selected) == target_count:
            return selected, {
                "status": "TOP_K_DETERMINED_BY_FROZEN_RANK_ORDER",
                "expensive_eligibility_evaluations_performed": expensive_evaluations,
                "eligible_candidates_encountered_before_roster_closure": len(selected),
                "failure_counts_among_evaluated_candidates": failures,
                "pre_evaluation_cross_family_conflicts": pre_evaluation_cross_family_conflicts,
                "stopping_rank_sha256": last_rank,
            }
    return selected, {
        "status": "INSUFFICIENT_ELIGIBLE_FRESH_IDENTITIES",
        "expensive_eligibility_evaluations_performed": expensive_evaluations,
        "eligible_candidates_encountered_before_roster_closure": len(selected),
        "failure_counts_among_evaluated_candidates": failures,
        "pre_evaluation_cross_family_conflicts": pre_evaluation_cross_family_conflicts,
        "stopping_rank_sha256": last_rank,
    }


def exhaustive_ranked_select(
    candidates: Sequence[RankedIdentity],
    eligibility: Callable[[RankedIdentity], tuple[bool, Mapping[str, Any]]],
    target_count: int,
    used_tokens: MutableSet[str],
    used_logs: MutableSet[str],
) -> list[tuple[RankedIdentity, Mapping[str, Any]]]:
    """Reference implementation used only for equivalence tests."""
    passing: list[tuple[RankedIdentity, Mapping[str, Any]]] = []
    for candidate in candidates:
        passed, audit = eligibility(candidate)
        if passed:
            passing.append((candidate, audit))
    selected: list[tuple[RankedIdentity, Mapping[str, Any]]] = []
    for candidate, audit in sorted(passing, key=lambda item: item[0].rank_key):
        if candidate.scenario_token in used_tokens or candidate.log_id in used_logs:
            continue
        selected.append((candidate, audit))
        used_tokens.add(candidate.scenario_token)
        used_logs.add(candidate.log_id)
        if len(selected) == target_count:
            break
    return selected
