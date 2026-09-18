
# S2R Static V Capacity v1

```text
248 total SESSION clusters
↓ exclude E3=31 / E4=0
217 raw V candidates
↓ execution initial speed >= 3.61 m/s
214 sessions with a deterministically ranked static-eligible identity
↓ native route/reference and map support
214
↓ metadata/precontext complete
214
↓ canonical production executor compatible
214
↓ STATIC_CERTIFIED_V_POOL
214
```

The selector ranks identities by SHA256 of a fixed salt, SESSION, scenario token, and log ID, then chooses the first identity satisfying the pre-outcome static contract. At most 256 rank-leading identities were map-evaluated per SESSION; sessions with no success in that prefix remain ambiguous rather than being declared ineligible. This is a capacity census and candidate identity evidence, not a final Primary roster. A later frozen roster must use an Owner-approved deterministic draw from this pool and must prohibit replacement after execution begins.

The 3.61 m/s gate remains `NOMINAL_MEASURABILITY_SCREEN / NOT_A_CLOSED_LOOP_GUARANTEE`. Native route completeness uses the same official builder and conservative `initial_speed × 7.9 s` reference requirement already used by the frozen zero-run route preflight.
