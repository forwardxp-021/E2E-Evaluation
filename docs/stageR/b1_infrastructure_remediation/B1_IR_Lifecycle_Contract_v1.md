# B1-IR Arm and Pair Lifecycle Contract v1

## Pre-run gate

Before budget claim and `runner.run()`, the executor requires: active exact authorization; pair/arm/spec identity; frozen precontext and route identity; `route_precheck_status=COMPATIBLE`; a fresh arm and fresh namespace; complete callback chain with official `MetricFileCallback`; manifest writer; and available budget. Any failed item is `DO_NOT_ENTER_RUNNER`.

## Post-run state machine

The only legal order is:

1. `RUNNER_COMPLETE`
2. `RECORDER_COMPLETE`
3. `OFFICIAL_METRICS_COMPLETE`
4. `SERIALIZER_COMPLETE`
5. `MANIFEST_COMPLETE`
6. `ARTIFACT_HASHES_VALIDATED`

A missing or out-of-order state is `TECHNICAL_INCOMPLETE`.

`ARM_COMPLETE` means all six states are present. `PAIR_COMPLETE` means both arms are `ARM_COMPLETE`, the frozen pair/precontext/route identities match, and neither arm has a disqualifying technical state. Joint mechanism, F_match and safety evaluation is allowed only after `PAIR_COMPLETE` (or an explicitly audited offline-recovered historical equivalent).

The 34 historical manifests are preserved and are not rewritten. Their official metric recovery and hashes live in the separate B1-IR audit chain.
