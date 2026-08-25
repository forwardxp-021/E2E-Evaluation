# Stage 6U Unified A/B/C Trainer实现与冻结

关联Issue：[GitHub #263](https://github.com/forwardxp-021/E2E-Evaluation/issues/263)。

## 1. 结论与授权边界

最终implementation freeze状态为：

`FROZEN_READY_FOR_ABC_FORMAL_TRAINING`

这表示统一trainer、B/C公平随机流、Dynamic v2/global33输入、synthetic与Waymo train/val smoke、
checkpoint/resume和完整formal epoch loop已通过验证。它只表示已经具备“另行授权”的技术条件。

本阶段仍然：

- `formal_training_authorized=false`；
- `formal_checkpoint_count=0/9`；
- 未读取Waymo test；
- 未读取Stage6J/K/P结果；
- 未运行或读取nuPlan、BDD/MMD和Stage6S-v2 confirmation。

正式模式要求一个独立授权manifest，其状态必须为`AUTHORIZED_STAGE6U_ABC_FORMAL_TRAINING`，并用
SHA-256绑定本次implementation freeze。缺失授权、SHA不匹配、seed不属于冻结seed集合或输出目录已存在
时，formal CLI都会fail closed。

## 2. Unified trainer如何实现A/B/C

`tools/stage6u_unified_abc_trainer.py`是唯一训练实现，通过`--candidate A|B|C`切换模型和冻结package，
没有复制三套训练循环。

- A：Dynamic v2 + legacy single-GRU + legacy Stage5D objective；
- B：Dynamic v2 + 同一single-GRU topology + clean longitudinal supervision/ranking/sampling/dropout；
- C：与B共享全部数据、loss、sampling、dropout、seed和预算，仅encoder变为ego16+context48双分支；
- A/B/C输入均为`[B,80,83]`，输出均为`[B,64]`。

encoder参数量为A=106560、B=106560、C=105616，C/B=0.991141，与Stage6T完全一致。正式循环使用
Adam、冻结LR/weight decay、batch=128、最多30 epoch/31680 optimizer steps、patience=5、constant
LambdaLR、每100 steps JSONL heartbeat、Waymo val早停与best/last checkpoint。

## 3. B/C公平随机流

每个epoch先构建候选无关random plan。plan只依赖：

- frozen Dynamic v2 train rows；
- seed、epoch、pair_seed；
- Stage6T sampling/dropout package。

它不读取candidate、encoder输出、loss或任何评估结果。B/C同seed逐项共享：

- sampling weights；
- epoch sample indices；
- batch offsets/composition；
- ranking positive/negative indices与pair type；
- slot dropout与all-neighbor dropout masks；
- augmentation seed stream；
- optimizer schedule和budget。

synthetic和Waymo subset的所有字段SHA逐项一致，且单元测试会主动篡改dropout SHA，确认ledger能
fail closed。全量135046行的B/C epoch-0 plan也做了实现级性能探针：两者各约6.1秒生成，fingerprint
完全相同。非dropout augmentation目前冻结为none，但其candidate-independent seed stream仍写入ledger，
防止后续静默引入不公平增强。

## 4. Dynamic v2与global33

Trainer dataset API只接受`train`和`val`；传入`test`直接抛错。它只打开：

- `context_traj.npy`；
- `interaction_feat_style_raw.npy`；
- `longitudinal_supervision_v2.npy`；
- `slot_valid_mask.npy`。

33D supervision严格使用Stage6T冻结的全体train 135046行global mean/std。smoke对16个真实Waymo train
row逐项手算`(raw33-mean)/std`，与trainer输出逐位一致。part-local `interaction_feat_style.npy`仅检查
冻结资产存在，永不载入为训练数组；Dynamic v2 shard不会被改写。

## 5. Smoke与resume验证

synthetic与真实Dynamic v2小规模train/val subset上，A/B/C均通过：

- forward/backward；
- embedding shape `[B,64]`；
- loss、embedding与gradient finite；
- optimizer step与constant scheduler；
- checkpoint save/load。

Candidate B resume smoke分别执行连续3 batch，以及1 batch后保存、重新构造model/optimizer/scheduler、
load后继续2 batch。恢复内容包括epoch、next batch、global step、optimizer、scheduler、Python RNG、
NumPy RNG、Torch RNG和random-plan ledger。两条路径的loss序列完全相同，最终model state SHA也完全相同。

PyTorch 2.5.1 MPS不支持`torch.cdist` backward；实现用基础矩阵运算计算相同的Euclidean pairwise
distance，CPU单元测试验证其off-diagonal结果与`torch.cdist`一致，没有启用隐式CPU fallback，也没有
更改Stage6T loss定义或权重。

## 6. 训练时间估计

最终MPS timing probe使用冻结formal batch=128，warmup 2 batch、测量5 batch。按1056 batch/epoch、
最多30 epoch外推：

- A：单seed约1.0小时；
- B：单seed约1.9小时；
- C：单seed约3.5小时；
- A+B+C单个seed串行约6.5小时；
- 9个任务完全串行纯训练约19.4小时；
- 加上val、checkpoint、数据I/O和计划生成，建议准备约22–27小时墙钟时间。

这是小subset缓存条件下的计划估计，可能偏乐观。正式启动后必须用每个candidate第一个完整epoch实测更新
ETA。early stopping可能缩短总时间；本机单MPS设备不建议并发多个训练任务。

## 7. 冻结产物

- 配置：`configs/stage6u_unified_abc_trainer.json`；
- trainer：`tools/stage6u_unified_abc_trainer.py`；
- smoke runner：`tools/stage6u_smoke_unified_abc_trainer.py`；
- freeze tool：`tools/stage6u_freeze_trainer_implementation.py`；
- smoke summary：`outputs/stage6u_unified_abc_trainer_smoke_v1/stage6u_smoke_summary.json`；
- fairness ledger：`outputs/stage6u_unified_abc_trainer_smoke_v1/stage6u_random_fairness_ledger.json`；
- implementation manifest：`outputs/stage6u_trainer_implementation_freeze_v1/stage6u_trainer_implementation_freeze_manifest.json`；
- 中文报告：`outputs/stage6u_trainer_implementation_freeze_v1/stage6u_trainer_implementation_freeze_report_zh.md`。

由于freeze tool源码本身也记录在manifest中，最终文档和代码完成后必须重跑freeze，以最终manifest SHA作为
以后正式授权绑定值；不能使用早期运行产生的SHA。

## 8. 下一步

下一步只有在用户明确授权后，才创建独立formal authorization manifest，绑定最终implementation freeze
manifest SHA，然后按固定seed 3407/3408/3409串行启动A/B/C共9个任务。完成9个checkpoint锁定前不得读取
Waymo test，之后仍必须遵守Stage6T冻结的盲测顺序。

## 9. 2026-08-12正式训练前实现复核

用户正式授权后，启动前复核确认原implementation freeze SHA
`6d1032b47f7dfaf4329a83db63105bedbeabf5a88ecbbc309ca77714a4d938fb`与当时trainer/config完全匹配，
但formal loop存在三个工程一致性缺口：`tqdm`已导入却未实际包裹train/val循环；epoch末checkpoint保存下一
epoch游标却携带上一epoch plan ledger；checkpoint元数据不足以建立用户要求的9任务锁定链。另发现best epoch
选择错误地复用了early-stopping `min_delta`，可能漏掉更低但改善小于`1e-4`的Waymo val epoch。

这些问题在正式训练启动前修复，未改变Stage6T的architecture、loss、sampling、dropout、seed、optimizer、
batch size、epoch或早停门槛。修复后的语义为：

- 每个formal train/val epoch均实际显示`tqdm`，并保留每100 optimizer steps的JSONL heartbeat；
- `resume_model.pt`在epoch开始、每100 steps和epoch边界原子写入；epoch中恢复校验相同plan并恢复累计loss，
  epoch边界恢复不再把上一epoch plan与下一epoch比较；
- best checkpoint严格取candidate-specific Waymo val objective的最低值，并列时保留最早epoch；patience仍以
  已冻结`min_delta=1e-4`单独计算；
- checkpoint/summary绑定candidate、seed、Stage6T、implementation freeze、formal authorization、trainer、
  config、Dynamic v2 content signature、package IDs、环境和resume history；
- 新增正式epoch边界resume smoke，连续路径与恢复路径的下一epoch loss和model state完全一致。

因此原`6d103...`保留为历史freeze，但不得用于本轮正式训练。trainer变更后必须重跑完整smoke与
implementation freeze，并让formal authorization绑定新的最终SHA。
