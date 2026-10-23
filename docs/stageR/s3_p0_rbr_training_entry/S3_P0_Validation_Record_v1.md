# S3-P0 Validation Record v1

日期2026-10-23；base commit9e781d596371781d00c326c57dc686b79b9596fe；Python科学环境见`evidence/Runtime_v1.json`。本轮只新增一个tools文件和一个测试文件；没有修改或运行旧模型训练/科学执行入口。

## 实际执行与结果

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/s3_p0_training_entry_audit.py snapshot
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python /Users/liuqing/miniconda3/envs/nuplan/bin/python tools/s3_p0_training_entry_audit.py parity
/Users/liuqing/miniconda3/envs/nuplan/bin/python -u tools/s3_p0_training_entry_audit.py lineage
python3 -m py_compile tools/s3_p0_training_entry_audit.py tests/test_s3_p0_training_entry_audit.py
python3 tools/check_no_tmp_dependencies.py
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python /Users/liuqing/miniconda3/envs/nuplan/bin/python -m pytest -q tests/test_s3_p0_training_entry_audit.py
git diff --check
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/s3_p0_training_entry_audit.py manifest
```

- Snapshot保护2185文件，CSV初始SHA正确；基线记录HEAD、tracked dirty状态与基线时165,987个untracked条目（含刚新增的audit入口）的列表摘要hash，不复制/提交原outputs。
- Parity执行14类×3实际processing APIs；数值证据及每case输入hash写Golden JSON。mask/轴顺序/closing等发现是**阻断**；不把两条错误链一致误当PASS。
- Lineage实际跑全36分片元数据与168,700行ego精确hash；原始51文件共23,245,316,267字节完整SHA及24,872 record元数据索引；retained24,548全部匹配。重复候选回查原始payload，证实80帧world ego重复，完整context不相同。没有大数组merge。
- 回归测试最终 **4 passed**，无skip；独立轴顺序oracle、生产链及物理oracle、wire解析截断失败、fixture确定性。测试输出使用pytest隔离目录，不改冻结报告。
- py_compile PASS；check_no_tmp_dependencies PASS；diff whitespace check PASS。没有以训练step或新仿真替代smoke。
- Manifest集成验收：12 gates恰好G1–G12，83行恰好index0–82，必需文档齐全；历史raw+已读数组hash与旧ledger核对结果在`evidence/Historical_Data_Hash_Comparison_v1.json`，保护验证在`evidence/Protected_Verification_v1.json`。整个审计源码、配置/来源、runtime与输出SHA在S3 Manifest；数据SHA在U Split Manifest。

## 构建中修正与限制

fixture构造三车道时首次LaneInfo参数缺失导致TypeError，已仅修复新增fixture构造、重新运行通过；没有修改生产pipeline。环境探测首次发现torch1.9无MPS API，最终报告明确记录`MPS_API_UNAVAILABLE`，没有假设GPU可用或安装新运行环境。此类audit工具修正没有进行科学retry、训练、模型选择或改变冻结结果。

本轮新观察的是U数据质量/物理fixture问题，不是V/C模型比较结果。未运行全库近似/时间错位world-track dedupe，未恢复缺失raw segment/global timestamps，不宣称所有重叠已被排除。旧35个static shard meta本机不存在；不宣称已认证旧模型所见每个样本。原始CRC未单独验证；file SHA精确绑定原始字节。

官方Waymo wire字段来源：[scenario.proto](https://raw.githubusercontent.com/waymo-research/waymo-open-dataset/master/src/waymo_open_dataset/protos/scenario.proto)，本轮下载的源副本/许可证及SHA留在evidence。只作为字段解析依据，未获取任何新训练数据。仓库外原始数据仍在原位置，索引可按filename+offset重放；跨机器复现应绑定相同hash的数据并配置其位置，不伪造不存在路径。

本包不含训练checkpoint或推理结果；`RBR_TRAINING=0; NEW_NUPLAN_SIMULATION=0; PRIMARY_EVALUATION=0; V_EXECUTION=0; C_EXECUTION=0`。所有科学禁令保持。
