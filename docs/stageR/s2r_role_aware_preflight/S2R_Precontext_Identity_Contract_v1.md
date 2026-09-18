
# S2R Precontext Identity Contract v1

Status: `PRECONTEXT_IDENTITY_CONTRACT = BLOCKED`.

For every future paired baseline/treatment execution:

```text
BASELINE_PRECONTEXT_ID == TREATMENT_PRECONTEXT_ID
PRECONTEXT_ID = sha256(UTF8(canonical_json(precontext_v1)))
```

`canonical_json` means sorted keys, UTF-8, no insignificant whitespace, explicit units, finite numeric values, and no fallback key guessing. `precontext_v1` must be captured from the final resolved production path and contain only fields that path can actually provide:

- schema version; SESSION, log/database token, scenario token, source database fingerprint;
- exact first extracted lidar token and timestamp;
- ego pose, velocity, acceleration, angular velocity/rate fields available from the official initial EgoState;
- official route roadblock IDs, map name/version, and hash of the derived rolling native reference;
- pre-intervention tracked-object/traffic-light replay token sequence hash;
- planner warmup/history-buffer contents and initialization parameters;
- controller, tracker, motion-model configuration hashes and random seed/state policy;
- ordered pre-intervention callback configuration and state hash.

Unavailable fields must be marked unavailable and fail closed; they must not be invented. The existing metadata anchor speed is not necessarily the execution initial speed. The official API's constant steering value is not a measured initial steering state and must be labeled separately.

The current runner/callback chain does not materialize this schema, so equality cannot yet be verified. Precontext hash equality is necessary but does not substitute for the full-arm reset contract.
