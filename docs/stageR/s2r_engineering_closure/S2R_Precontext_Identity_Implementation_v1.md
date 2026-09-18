
# S2R Precontext Identity Implementation v1

Status: `PRECONTEXT_IDENTITY_CONTRACT = PASS`.

`PRECONTEXT_ID = SHA256(canonical_json(s2r_precontext_v1))` is now a required arm-spec and execution-manifest field. It binds SESSION/log/scenario identity, official simulation initial lidar token and timestamp, official ego initial state, route and map identity, raw traffic-agent/traffic-light precontext hash, history-buffer source hash, initialization contract, and the 1.1 s intervention boundary. The official constant zero steering value is explicitly labeled as an API source rather than a measured field.

Before execution, the pair factory requires exact baseline/treatment equality of both `PRECONTEXT_ID` and `ROUTE_ID`; mismatch yields `PAIR_INVALID_PRECONTEXT` or `PAIR_INVALID_ROUTE`. The production planner's frozen common-preintervention rule uses the complete baseline trajectory for both arms before 1.1 s. No rollout was used to form these hashes.
