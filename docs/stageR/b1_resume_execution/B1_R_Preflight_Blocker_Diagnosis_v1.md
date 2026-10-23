# B1-R Preflight Blocker Diagnosis v1

Terminal stage verdict: **B1_R_PREFLIGHT_BLOCKED**. New simulator entries = 0, new attempts consumed = 0, cumulative attempts = 35. No attempt namespace or scientific execution manifest was created for a real run; the six archived records are explicitly NOT_RUN. The only new raw folder is the first factory's zero-run diagnostic namespace.

## Observed failure and responsibility

The formal preflight constructed the first official factory, then stopped at `tools/b1_resume_execution.py:267`:

```text
RuntimeError: RECORDER_CALLBACK_NOT_BOUND
require(arm.recorder is not None and len(arm.callbacks) > 0, ...)
```

This check was introduced in B1-R by Codex. It incorrectly treats the shared callback list returned by `build_simulation_callbacks` as the complete per-simulation callback graph. Its emptiness does **not** prove that the real official recorder, metric callback or serializer is missing. The unit tests did not cover this real-factory distinction. This is a preflight implementation defect, not evidence of a new route failure or scientific failure.

Read-only inspection of the exact hash-bound official source explains the distinction:

- `nuplan-devkit/nuplan/planning/script/builders/simulation_callback_builder.py:60–63` deliberately skips stateful SimulationLogCallback and MetricCallback when constructing the shared list.
- `nuplan-devkit/nuplan/planning/script/builders/simulation_builder.py:106–116` constructs per-simulation metric and simulation-log callbacks; lines 126–129 attach `MultiCallback(callbacks + stateful_callbacks)` to the actual simulation.
- `tools/s2r_production_executor.py` stores the shared list as `FreshArm.callbacks`, which can therefore legitimately be empty even when the actual simulation has stateful callbacks.

This diagnosis is based on source inspection after stopping. We did not rebuild factories, initialize or run planners, or perform a second formal preflight. The first factory's runtime graph was not retained in the evidence before the failing assertion, so this report does not claim that all six runtime graphs were audited successfully.

## Preflight coverage and stopping decision

Checks 01–07 passed: committed source/hash binding, original20/40 identities, IR2 original17 traceability and historical hashes, original18 treatment failure, five not-run identities, unchanged IR3 route components, all40 route/precontext identities. The first factory's `validate_pre_run_lifecycle` returned before the failing shared-list assertion, but no all-six route/callback result was materialized. Requirements 08–14 must not be described as fully passed. No partial scientific run occurred.

The Owner instruction explicitly says any failed formal preflight must stop, with no automatic repair-and-continue. Consequently the source, authorization, formal failed evidence and attempt ledger remain unchanged. No patch to this assertion, authorization hash update, repeat preflight, retry, additional attempt or training was performed.

## Legal next action

Owner review of this blocker is required before any new work that would reopen execution. A future instruction could authorize an engineering-only correction to inspect the actual simulation callback graph, persist callback/recorder/serializer evidence before final gate assertions, and test the real official zero-run factory. It would require a new versioned source/authorization/preflight decision rather than silently rebinding the failed stage. This report does not itself authorize that correction or any of the six runs.

The original17 pairs remain valid and unchanged. B1 qualification as a whole remains historically INCOMPLETE_INFRASTRUCTURE; this B1-R stage ends at PREFLIGHT_BLOCKED, with no new scientific outcome. The route repair and the frozen scientific contract are not contradicted by this preflight defect.
