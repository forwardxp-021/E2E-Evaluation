# B1 Stop Policy Addendum v1

Owner decision: `COMPLETE_PRE_REGISTERED_ROSTER_ON_SCIENTIFIC_FAILURE` and `STOP_ON_INFRASTRUCTURE_FAILURE`.

This prospective addendum is materialized from the Scientific Owner instruction dated 2026-09-18 before any B1 rollout. It supersedes the old strict-fresh Q rule `FIRST_SCIENTIFIC_FAILURE_STOP` for B1 only. It does not weaken the old all-pairs qualification threshold.

The purpose is to estimate realized coverage of the already-frozen TSB across the complete 20-SESSION B cohort. A scientific failure is retained in the denominator and execution proceeds to the next preregistered arm. A runner, callback, recorder, serializer, schema, environment, or required-artifact failure stops all subsequent arms. No failed pair is retried or replaced.

The B1 success decision remains:

```text
20/20 joint pair PASS → B1_TSB_BENCHMARK_QUALIFICATION_PASS
any scientific or measurement failure → B1_TSB_BENCHMARK_QUALIFICATION_FAIL
any infrastructure stop/incomplete arm → B1_TSB_BENCHMARK_QUALIFICATION_INCOMPLETE_INFRASTRUCTURE
```
