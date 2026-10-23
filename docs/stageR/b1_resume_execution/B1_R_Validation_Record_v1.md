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

正式真实数据 zero-run preflight：BLOCKED；首个factory构建成功，但新增共享callback非空断言不符合官方stateful callback构建方式。未完成六个factory的整体预检；simulator entry=0、attempt consumption=0。该断言保持原样，未在失败后修复或重做预检。Postrun hash audit PASS，910个历史文件、406个源文件、24648个依赖文件和5个关键数据文件保持绑定。

停止后的同一59项定向测试再次通过（59 passed、2 deselected）；全tools编译、临时依赖检查、diff whitespace检查通过。该结果不代表真实六工厂预检通过；正式状态仍为BLOCKED。
