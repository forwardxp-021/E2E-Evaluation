# B1-IR Native Route Precheck v1

Status: **PASS — gate implemented and B1-18 pre-detected**.

The old census requested only the initial short forward horizon. The runtime recomputed the route from each realized ego state, so a later roadblock transition could fail after runner entry. The new zero-run gate invokes the exact production function `build_native_route_reference_v1_1` from the frozen initial state with an infinite forward request. This forces the production function itself to traverse every frozen successor. Its terminal `insufficient native forward coverage` means all listed transitions resolved; any earlier missing successor is an incompatibility. No parallel approximation was introduced.

Production source SHA256: `592c755390f565db229a765901aa4a2af50de78f895ef564dd985b71480f6dbd`.

| Pair | Status | Missing successor | Failure code |
|---|---|---:|---|
| B1-TSB-01 | COMPATIBLE |  | `` |
| B1-TSB-02 | COMPATIBLE |  | `` |
| B1-TSB-03 | COMPATIBLE |  | `` |
| B1-TSB-04 | COMPATIBLE |  | `` |
| B1-TSB-05 | COMPATIBLE |  | `` |
| B1-TSB-06 | INCOMPATIBLE | 19339 | `NATIVE_ROUTE_FAIL: no native outgoing successor into 19339` |
| B1-TSB-07 | COMPATIBLE |  | `` |
| B1-TSB-08 | COMPATIBLE |  | `` |
| B1-TSB-09 | COMPATIBLE |  | `` |
| B1-TSB-10 | INCOMPATIBLE | 19339 | `NATIVE_ROUTE_FAIL: no native outgoing successor into 19339` |
| B1-TSB-11 | COMPATIBLE |  | `` |
| B1-TSB-12 | COMPATIBLE |  | `` |
| B1-TSB-13 | COMPATIBLE |  | `` |
| B1-TSB-14 | INCOMPATIBLE | 19339 | `NATIVE_ROUTE_FAIL: no native outgoing successor into 19339` |
| B1-TSB-15 | COMPATIBLE |  | `` |
| B1-TSB-16 | COMPATIBLE |  | `` |
| B1-TSB-17 | COMPATIBLE |  | `` |
| B1-TSB-18 | INCOMPATIBLE | 19339 | `NATIVE_ROUTE_FAIL: no native outgoing successor into 19339` |
| B1-TSB-19 | COMPATIBLE |  | `` |
| B1-TSB-20 | INCOMPATIBLE | 19339 | `NATIVE_ROUTE_FAIL: no native outgoing successor into 19339` |

The gate checked 20 pairs/40 arms: 15 compatible and 5 incompatible. B1-18 deterministically reports `NATIVE_ROUTE_FAIL: no native outgoing successor into 19339` before budget claim or runner entry. B1-06, B1-10 and B1-14 already have preserved completed attempts and need no rerun; B1-20 remains NOT_RUN and is fail-closed until route infrastructure and Owner authorization are resolved.
