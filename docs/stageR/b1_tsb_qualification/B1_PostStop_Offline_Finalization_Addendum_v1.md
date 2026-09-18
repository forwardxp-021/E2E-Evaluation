# B1 Post-stop Offline Finalization Addendum v1

Status: `REPORTING_ONLY_AFTER_INFRASTRUCTURE_STOP`.

The frozen execution stopped at authorized arm 35, `B1-TSB-18-TREATMENT`, after the official runner entered and raised `ValueError: NATIVE_ROUTE_FAIL: no native outgoing successor into 19339`. The arm budget was consumed and marked `TECHNICAL_INCOMPLETE`. Arms 36–40 remain `NOT_RUN`.

No retry, replacement, new scenario selection, parameter change, threshold change, or further simulator entry is permitted. This addendum authorizes only deterministic offline finalization of already complete pairs and materialization of `NOT_RUN` records for unstarted arms. It does not synthesize missing states or convert the infrastructure stop into a scientific result.

The post-stop code change only makes final reporting accept a partial execution ledger, archive existing manifests, generate explicit `NOT_RUN` arm records, and preserve the exact infrastructure reason. The frozen analyzer and all scientific numerics remain unchanged.
