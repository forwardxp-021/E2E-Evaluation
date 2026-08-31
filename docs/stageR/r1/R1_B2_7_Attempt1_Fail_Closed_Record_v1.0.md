# R1 B2.7 Attempt-1 失败关闭记录

## 结论

Attempt-1 在任何正式结果写入前以 fail-closed 方式终止。没有科学结果、没有冻结 roster、没有阈值变化、没有人工选择，也没有 protocol deviation。

## 根因

冻结 source contract 的身份单位是全局唯一 scenario token：5,386,575 个 token、1,621 个 log。Attempt-1 错将逐 DB token occurrence 相加为 5,405,672；mini 与 train_pittsburgh 中有三组 byte-identical DB 重复，合计 19,097 次重复 occurrence。因此：

`5,405,672 - 19,097 = 5,386,575`。

三组重复 DB、SHA256 与 canonical representative 见 `r1_b2_7_attempt1_fail_closed_record_v1.0.json`。

## 科学边界

- 未启动 official simulation 或 `run_simulation.py`。
- 未执行 48-run smoke。
- 未训练或执行 RBR A/B/C。
- B2.6 和既有历史文件未被覆盖。
