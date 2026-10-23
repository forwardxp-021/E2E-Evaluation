# B1-R2 Validation Record v1

Before source freeze: 53 passed, 0 skipped, 0 deselected in R2, RP real-factory, original resume, B1 qualification and infrastructure suites. Includes actual production factory configuration equality with the pre-frozen RP normalized hash, callback fault regression, exact six-attempt ordering, duplicate/attempt3 rejection, scientific-failure continuation and terminal infrastructure stop. No scientific simulation was executed in these tests.

Full tools py_compile (Python3.14), check_no_tmp_dependencies.py and diff whitespace checks passed. Existing409 source/config/test bindings remain byte-identical; the three new R2 entry/report/test files are additionally pinned to the executable source base commit. The sole production runner and all scientific algorithms/configuration remain unchanged.

Final authorized preflight and scientific execution results are recorded separately. Authorization documents and report commits cannot rebind source hashes or scientific parameters.

After all six completed: the same 53 tests passed again, 0 skipped/deselected; tools compilation, forbidden temporary dependency and whitespace checks passed. Artifact completeness audit: six distinct process IDs, 80 trace rows + 80 planner rows + 79 actual-controller transitions each, all six complete lifecycle manifests, 96 official metric parquets and six official nonempty simulation logs; every required artifact checksum matches.

Final safety table: all40 completed arms collision=0 and drivable-area=true; all20 pairs PASS. Zero LOW_SPEED_ENDSTOP. No scientific or infrastructure failure in the six new executions. Original18 treatment attempt1 remains independently frozen. Original raw logs are retained; report log copies remove trailing whitespace only.
