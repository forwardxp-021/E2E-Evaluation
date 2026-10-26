

## RBR-V2-D2：官方nuPlan增量来源与受限D-only验证

### 1. 命令

在仓库根目录运行。源目录和科学账本版本化；已完成的科学目录不得删除、重置或复跑。下面是本轮入口和执行顺序，已有冻结plan/binding/preflight会拒绝覆盖。

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d2_download.py catalog
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d2_download.py plan
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d2_round2.py plan
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d2_fast_acquire.py
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d2_round2.py run
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d2_support.py
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d2_prepare.py
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d2_execute.py --preflight
# 第一个D科学arm前，先提交完整身份/源码/数据/预检冻结。
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d2_execute.py --execute
# 本轮已启动postprocess依赖流程；勿同时手工重复extract/encode/statistics。
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d2_postprocess.py
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d2_roles.py
/Users/liuqing/miniconda3/envs/nuplan/bin/python -m unittest tests.test_rbr_v2_d1_statistics tests.test_rbr_v2_d2_download tests.test_rbr_v2_d2_statistics tests.test_rbr_v2_d2_cache tests.test_rbr_v2_d2_fast_download -v
```

### 2. 期望行为

只从AWS官方公开v1.1 train/val ZIP分段获取SQLite，不读取test/sensor blobs，不重下已认证日志。共享50,000,000,000字节硬预算，磁盘保留20GiB；所有请求/失败/恢复计入账本。原SDC-v1物理、控制、机制、F_match和官方安全不变；新来源按事前成本/完整性规则选择LOG并固定最多16个anchor。第二来源框对所有具备来源的SESSION一致应用，不按首轮成功者挑选；同SESSION多chunk仍只计一次。

原C五个及新增二十个索引保留C不读原始轨迹。40个D新来源首次两臂最多80arm，技术重试0，每arm20分钟、累计科学CPU4小时。已有20个SDC基准SESSION显式转为D-swap校准、现有三D baseline用于固定scaler；全60个D SESSION与未来V/C隔离。冻结epoch30及Waymo normalizer、Canonical83和完整H30不变；D数值是paired controlled-swap诊断，不能冒充运营A/A误报、真实独立release或Primary结果。

### 3. 通过标准

实际数据库SESSION/LOG身份、官方split/ZIP ETag/CRC/SHA闭合；不以chunk和重复抽样增加独立分母。完整D名单、失败和CPU账本保留；同80帧输出z64/H30，负MMD²和原阈值/alpha/m/pi规则保持。容量、真实功效/覆盖/FPR任一必要条件未成立则如实BLOCKED；静态适格及最低规划数不自动产生Ready。所有历史结果、受保护CSV及用户Quick前缀不变。任务结束不执行V/C、Primary或模型重训。


### 4. 本次真实停止与只读收口

本次原始D执行器首arm缺失生命周期实参，失败已永久保留；随后通过已提交的`rbr_v2_d2_continue.py`只调用原函数的正确签名，对原剩余身份首次执行。首Treatment在0.7秒失去原生路线关联，继续账本也已停止。两个已领取身份均不得重跑，余78arm未执行；不得用这些目录复用、重置或建立替补名单。上面的科学入口是历史复现说明，不是当前可再次执行的命令。

```bash
# 只读已有8个状态、7次控制和地图；不运行runner、传播模型或重新判定安全。
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d2_route_diagnostic.py
# 两个来源框达到终止状态后，重算全部保留源/无损缓存的SHA，再生成最终报告。
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d2_closeout.py verify-sources
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d2_closeout.py finalize
# 只核验历史资产与用户原Quick前缀。
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d2_closeout.py protected
```

`finalize`拒绝覆盖已关闭报告；重现时核验现有Manifest，不能删除历史产物来绕过保护。当前真实A/A、检测率、ΔBDD、饱和、功效与覆盖均NOT_ESTABLISHED。来源缓存恢复命令：`python tools/rbr_v2_d2_round2.py restore --path <原始SQLite绝对路径>`，独立核对原始SHA后恢复；不下载旧源、不修改科学合同。
