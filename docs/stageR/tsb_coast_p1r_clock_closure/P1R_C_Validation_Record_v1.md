# P1R-C 验证与复现记录 v1

本轮新增5个tools模块（clock、production、executor、preflight、fixture），1个测试模块，1个独立Hydra clock配置，以及本目录合同/报告/证据。旧P1R及官方源文件只读。每个实际source/config/data/output的SHA256见Execution_Source_Binding/Manifest。

## 实际运行命令与结果

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH=.:../nuplan-devkit python -m pytest -q tests/test_tsb_coast_p1r_c.py tests/test_tsb_coast_p1r.py tests/test_tsb_coast_zero_command_tracker_v1.py tests/test_tsb_coast_p1_measurement.py
python -m py_compile tools/tsb_coast_p1r_c_*.py tests/test_tsb_coast_p1r_c.py
python tools/check_no_tmp_dependencies.py
```

nuplan Python3.9.19测试 **96 passed**，598 warnings。新模块编译/no-tmp均PASS；全仓tools另用系统Python3.13.13编译PASS。旧全仓Python3.9在未改的build_unified_bdd_posttraining_report.py:550存在f-string兼容问题，继承原记录，不为此改历史工具。依赖警告来自官方numpy标量转换、pyparsing/Pillow/Bokeh等弃用接口，不改官方代码消除它们。

实际模型隔离命令：`tools/tsb_coast_p1r_c_fixture.py --output-dir .../evidence/isolated_model_verified`，Baseline/Treatment分别79个真实官方model传播，80合成状态。模拟时间全部来自新clock。禁止正式SESSION/runner，纯合成reference是单位测试信号，不是C2实际曲线。

真实生产零运行命令：`tools/tsb_coast_p1r_c_preflight.py --output-root outputs/stageR/tsb_coast_p1r_c_final_zero_run_v1 --evidence-dir docs/stageR/tsb_coast_p1r_clock_closure/evidence/production_verified`。六对象全部PASS；两个旧审计过程亦保留，最终以production_verified为准。此工具只能固定原六spec，不包含重新选SESSION或运行科学runner的选项。

## 覆盖与范围

- 81个clock接口、80帧/79transition官方结束语义、reset/t0、固定dt、21/39边界。
- 原日志不可变、正负偏差、唯一最近帧单元；重排、重复token、短场景、非10Hz、半周期歧义、非整数timestamp均拒绝。
- 官方TracksObservation在sentinel scenario上0..79同index同对象返回；无未来index、漏帧或插值。
- 官方controller/model合成传播、零请求到model、残余a滤波及LQR恢复；原adapter转向/跨界拒绝回归。
- planner 80次合成完整future grid与common precontext；每call一条realized-current trace。
- 官方metric纯时间提取器读取canonical状态；实际metrics/callback/serializer仅装配验证，未执行compute/finalize。
- F原四项定义/epoch平移和既有22项测量测试，压力反例仍可复现。
- 旧clock、新clock权威混淆、旧授权、PROPOSED_ONLY、单独Owner批准记录、缺失clock源码绑定、不同合同/预检、未确认时间差均不能进入科学执行；首对scope/spec篡改拒绝。
- 新旧预算monitor成功/失败分支用无仿真Python子进程fixture验证，真实工作账本始终0；没有启动正式child。

## 历史保护

最终Protected_Verification核对3485个源文件/合同/历史输出，旧B1、P0/P1/P1R、S1/S1R、旧R1时间控制器及官方源码均不变。所选DB、地图、静态shard及原spec/roster完整hash核对。CSV保持`e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8`。

只提交本轮范围，QUICK_REFERENCE通过单独章节暂存保护用户原127行修改；Stage5原17行和CSV工作区修改不提交。日志的新包空白规范化另记录原始/规范化hash，不修改旧证据。
