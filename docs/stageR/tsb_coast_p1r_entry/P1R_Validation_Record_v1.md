# P1R 验证记录 v1

2026-10-24；基线a6447c2；源码和依赖见Source_Binding/Runtime/Manifest。本轮新增6个tools模块、1个测试模块及独立planner配置，文档/manifest/候选和零运行证据位于本目录；QUICK_REFERENCE仅提交本轮新增章节。

## 实际执行

- nuplan Python3.9.19，设置 `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH=.:../nuplan-devkit`；运行 `pytest -q tests/test_tsb_coast_p1r.py tests/test_tsb_coast_zero_command_tracker_v1.py`：**49 passed**（28新P1R用例及21原adapter用例），437 warnings，详见Tests_v1.log/xml。警告来自官方/历史依赖中的numpy标量转换及弃用接口，没有为压警告修改官方源码。
- 同一环境运行真实官方模型的合成单步/79步、模式切换、不同命令/模型状态/速度导数、确定性、记录器缺失/异常故障、真实未初始化factory及callback/model绑定故障、日志物理时钟阻断、旧授权/待批合同拒绝、累计预算/parent ticket和无仿真子进程monitor成功失败fixture。
- 正式六臂factory在禁止run/initialize/planner/propagation/metric/finalizer的guard下构建。初次和接口修复后的构建证据保留；完整clock audit读取官方scenario的81个时间点，6/6阻断。没有把早期component PASS当最终准入PASS。
- `tools/tsb_coast_p1r_model_fixture.py --output-dir docs/stageR/tsb_coast_p1r_entry/evidence/isolated_model`：持久化80合成状态、79真实模型传播；不使用正式SESSION、不进入runner。图已检查布局及明确fixture标签，见P1R_Isolated_Model_Response。
- `tools/tsb_coast_p1r_analysis.py --metrology-audit .../evidence/FMatch_Metrology_v1.json`：只读40条冻结B1速度来源与hash，3200值JSON精确往返；新增16种固定相位的解析压力配对诊断；没有重算B1科学结论。
- 本轮新增tools与tests在nuplan Python3.9 `py_compile`：PASS。
- 仓库全部 `tools/*.py` 用系统Python3.13.13 `py_compile`：PASS。
- 同样全仓编译用nuplan Python3.9：FAIL，定位旧`tools/build_unified_bdd_posttraining_report.py:550`，`f-string expression part cannot include a backslash`。该文件与HEAD SHA256完全相同：`e25a7debfcbd8dd82fd775839f3ab9236f09169b0bd6b2b62464f0607b674be2`。与新栈无依赖，未改动；明确保留此兼容性限制，不报告全仓3.9通过。
- `tools/check_no_tmp_dependencies.py`：PASS；Git whitespace检查PASS。
- `evidence/Protected_Verification_v1.json`：3367受保护文件不变，3个选定DB及元数据shard hash不变，所有绑定source hash一致。protected CSV SHA仍`e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8`。

## 严格证据范围

本地真实数据存在，已做元数据、finite-route及真实未初始化factory最小检查；没有以“无数据”为由跳过。真实科学runner未运行，是Owner限制及已查明的clock阻断，不是模型测试通过后的默认下一步。

`SCIENTIFIC_SIMULATION=0; TSB_COAST_SCIENTIFIC_ATTEMPTS=0; RBR_TRAINING=0; RBR_INFERENCE=0; H30_EVALUATION=0; MMD=0; BDD=0; PRIMARY_EVALUATION=0; V_EXECUTION=0; C_EXECUTION=0`。

纯模型测试实际调用官方propagate，多次pytest还做了额外合成传播；这些明确属于授权的隔离测试，不能把它们写成“模型完全没运行”，也不算科学rollout。0仅指科学执行/受guard保护的生产调用。

正式callback完成、真实Coast机制/配对F/安全、完整episode route、完整runner数值重现及科学资源消耗全部NOT_RUN/NOT_ESTABLISHED。当前停在Owner D1–D3和物理时钟阻断，不自动启动后续任务。
