# R0 nuPlan Unused Pool Audit Report v0.1

## Decision

`CLEAN_UNUSED_POOL_EXISTS`，但 `R0_AUDIT_HOLDOUT_NOT_FEASIBLE_FROM_CURRENT_NUPLAN`。

## Identity-only accounting

- canonical global logs: 1621
- canonical runnable scenario tokens: 6310755
- runnable tagged tokens (per-log distinct sum): 4155661
- identity manifests scanned: 4738
- historical token identifiers observed: 273462 unique / 650361 source-level occurrences
- historical runnable tokens matched to current global pool: 232966
- historical/current logs excluded by whole-log rule: 1510
- runnable tokens excluded with those logs: 6297950
- clean unused logs: 111 identity-clean, of which 19 contain at least one runnable token
- clean unused runnable tokens: 12805
- clean-vs-used log overlap: 0; matched historical runnable-token overlap: 0 by whole-log exclusion

Global ledger is deliberately compact: one row per canonical log, with the complete sorted runnable-token set bound by count/min/max/SHA-256, source path/version, schema SHA, map/time/type metadata. It is not a multi-million-row token materialization. Subtraction itself streamed every runnable token and conservatively removed the entire log when any historical token or log identity matched.

The local source path follows nuPlan v1.1 naming, but the SQLite files do not embed an independently attestable release checksum. Therefore the ledger records reproducible path, size, mtime, SQLite/schema/page metadata, and token-set SHA rather than claiming an unavailable upstream file SHA.

## Clean coverage

Top clean tagged families (distinct-token sums; tags may overlap): stationary=11112, on_carpark=2800, stationary_at_traffic_light_with_lead=767, on_stopline_traffic_light=401, stationary_at_traffic_light_without_lead=401, on_intersection=401, on_traffic_light_intersection=401, low_magnitude_speed=196, near_pedestrian_on_crosswalk=143, stopping_at_crosswalk=10.

Potential runnable independent cluster units are the 19 clean logs with at least one runnable token; the remaining 92 identity-clean logs have zero tokens under the frozen official-compatible scene-boundary rule. The conservative D0 plan requires 150 runnable logs (10 scenarios/log, planning ICC 0.10, design effect 1.90), so at least 131 new identity-clean runnable logs are required before an audit holdout can be frozen. No representation, BDD, treatment rollout, or outcome was read for selection.
