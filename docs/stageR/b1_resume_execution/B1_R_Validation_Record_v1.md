# B1-R Validation Record v1

Before source freeze: nuPlan Python 3.9 targeted regression suite **59 passed, 2 deselected**, zero scientific execution. Suite: B1-R, original B1, IR, IR2 and IR3. The two deselected IR3 tests reproduce its entire historical source snapshot and require its original executor bytes; they remain unchanged and are replaced for this successor by explicit historical commit-blob validation plus new authorization/source audits. Native-route fixtures, original source binding refusal, frozen scientific code checks and budget/lifecycle/stop-policy tests are included.

Commands:

```bash
PYTHONPATH="$PWD:../nuplan-devkit" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python /Users/liuqing/miniconda3/envs/nuplan/bin/python -m pytest -q --disable-warnings tests/test_b1_resume_execution.py tests/test_b1_tsb_qualification.py tests/test_b1_infrastructure_remediation.py tests/test_b1_ir2_adjudication.py tests/test_b1_ir3_route_closure.py -k 'not reproduction_and_all_historical_references and not final_manifest_hashes_and_unchanged_historical_assets'
python3 -m py_compile tools/*.py
python3 tools/check_no_tmp_dependencies.py
git diff --check
```

Full tools compilation, forbidden temporary dependency check and whitespace check passed. Third-party deprecation/map cast warnings remain. Formal preflight and postrun results are separately recorded; this document grants no additional execution authority.
