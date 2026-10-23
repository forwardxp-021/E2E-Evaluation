# B1-R2 Authorization Audit v1

Owner instruction SHA256 `0090dab4e7ade26e397b644cd493e8b36d91cd9474adc7764a2157fe13f7ab33`; authorization SHA256 `9189d1ca1f8b4ec2c22cc52771936127fb2bfdca724af34796aa283db9a1caeb`.

EXECUTABLE_SOURCE_BASE_COMMIT = `498aa18fd1e22ad5edd7cfd7b6c3352fd012c7f4`. Inherited409 bindings remain unchanged; R2 adds only a versioned authorization/accounting/report entry around the same factory, sole B1 runner/lifecycle and frozen evaluators.

AUTHORIZATION_RECORD_COMMIT is resolved from committed authorization path history, recorded separately in final preflight and each attempt. It is not embedded back into its own file. Later documentation commits do not change the source base. Source is validated by exact file hashes and Git blobs, never HEAD equality.

Authorization permits only the original ordered six attempts; cumulative ceiling41. Original17 outcomes and all old budgets/failures remain immutable. Any final preflight or runtime infrastructure failure terminates this stage; no repair, rebind, rerun or additional retry. Scientific failure retains denominator and continues.
