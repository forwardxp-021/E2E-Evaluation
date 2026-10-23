# S3-P0 RBR-V2 Training Architecture Proposal v1

状态：`PROPOSED_NOT_OWNER_FROZEN_NOT_AUTHORIZED`。原则继承已批准S1.1；以下具体规格是U-only前瞻建议，不是既有Owner批准，不是训练产物。数据/输入阻断未关闭，不得启动。

## 已批准边界

`canonical X + masks → temporal encoder/GRU → shared z64 → Human Semantic Head + Generic Temporal Representation Head`。两个Head的loss均必须通过z64对共享encoder反向传播；不detach、不在训练后另拟合head冒充训练监督。Primary BDD直接使用未拼接语义真值或预测值的冻结z64。没有强制`[s; h−φ(s)]`结构，也没有语义独立性保证。

严格排除TSB机制/treatment/B1/H/BDD/Primary outcome选择或监督；禁止人工feature距离、ego13几何对齐、mechanism-specific相位目标。历史Stage5D/6U的人工距离loss不直接继承。H30维及Primary已批准SAP不随本提案改变。

## 单一可冻结候选

| 项目 | 本轮建议 |
|---|---|
| 输入 | 80×83 canonical v1 float32；源时间另以float64/整数us保留；mask/ID独立sidecar，版本严格匹配 |
| 数据缺失 | 数值normalize后无效填0；channel mask屏蔽loss；无效ego timestep跳过GRU状态更新，不压缩跨洞时间；至少64/80 ego有效且至少2个连续有效差分，否则固定input-invalid全量记账。64帧门槛尚待Owner批准，不移植为B1/H过滤 |
| mask接入 | 输入是83数值+83 channel-valid bits再附加物理时间t/7.9，encoder内部167维；外部物理合同仍为83维。neighbor presence在原83中保留；derivative-valid通过channel-valid表达，不重复增加物理字段 |
| temporal encoder | 一层单向GRU，input167、hidden128、dropout0；按ego_valid显式保留/更新hidden；固定80个物理time positions，不pack跨洞时间 |
| shared z64 | last valid recurrent state→Linear128→128→ReLU→Linear128→64，无真值语义拼接，无L2强制单位化；输出[N,64] float32有限值 |
| Semantic Head | z64→Linear64→64→ReLU→Linear64→13，直接回归13项物理摘要；定义见Semantic Target文档 |
| Generic Temporal Head | [z64, t/7.9]→Linear65→128→ReLU→Linear128→83，逐时间点重建输入；训练时对有效非valid-bit通道施加固定15%遮挡（2帧连续span；可重叠，先按固定RNG打乱合法(start_time,channel) span列表，依序取尚未遮挡的有效单元至ceil(.15×有效连续单元数)，末span可截短；从目标和mask确定，拒绝把invalid当0目标） |
| L_repr | 只在被遮挡且target-valid的连续通道SmoothL1(beta=1)平均；5个presence通道不作为回归目标；heading用wrapped residual/π，其余按U_train frozen scaler归一；每样本先平均有效单元再batch均值，无有效单元为协议错误，不跳过有利样本 |
| L_sem | 13个train-standardized摘要的masked SmoothL1(beta=1)，先每目标对有效样本平均，再对13目标等权平均；无目标支持则fail closed，不能按val重新加权 |
| 总loss | `1.0*L_repr + 1.0*L_sem`，无可选aux项、无特征距离/样本对挖掘/三元组/TSB特定项 |
| optimizer | AdamW(lr=.001, betas=.9/.999, eps=1e-8, weight_decay=.0001)，constant LR，无scheduler搜索，global grad norm clip1.0 |
| batch/epochs | batch128；train每epoch全部训练窗口固定seed shuffle一次；validation全部一次不shuffle；30epoch上限；尾batch保留；无邻车有效也允许，依presence mask |
| checkpoint规则 | 最低U_val总loss，严格改善>1e-6才更新；相差≤1e-6取最早epoch；patience5；每epoch必评价；最少1个完整epoch，不使用U_test/B/V/C选checkpoint |
| validation loss | 同一train-fitted normalizer/target统计与相同loss权重；遮挡计划由hash(dataset version, row identity, seed2026102301)预先固定，所有epoch相同；无随机验证有利seed |
| seeds | Primary encoder seed3407事先指定；3408、3409仅预先声明重复性运行，不能在三者择优或平均替代Primary；各自按U_val同一规则选epoch。numpy/Python/torch/loader/masking RNG流分别记录；验证mask seed固定 |
| 划分 | 未来新U版本按依赖连通组80/10/10建议；分组后固定seed2026102302分配；以组为单位避免同源跨split。实际比例/计数由新manifest冻结，不复用当前Scenario-only认证 |
| 训练输入标准化 | 只在U_train有效连续值拟合population mean/std，std≤1e-6置1；heading/heading_rel除π、presence不变；无效值和TTC/THW sentinel不进入拟合；保存精确统计、顺序和hash |
| target标准化 | 只在U_train有效target拟合mean/std（std≤1e-6置1），train/val/test同一组；不是H的D_fit scaler |

该masked reconstruction目标是通用序列结构压缩，不承诺发现新语义或RBR胜H。遮挡只在U产生，不用nuPlan真实/TSB轨迹选择span或loss权重。为什么采用GRU128：延续历史可运行模型规模，控制实现与预算；不是基于B1机制强弱决定。

## U-only选择及接口验收

未来trainer必须只接受一个hash-bound U manifest，schema/来源/partition严格白名单；训练/选择进程禁止接入nuPlan B/V/C、detector/H结果路径。U_test不用于选择；本轮为独立性审计读test的身份/自车数组，未作表示评价，下一轮需要对历史暴露和新的U版本关系如实登记。

未来训练前零训练验收：shape/mask/normalizer一致、相同fixture跨源正确、仅Semantic loss的梯度能到encoder、仅repr loss也能到encoder、禁止数据路径拒绝、selection tie/early-stop/resume规则。**这些是未执行的修复验收项**；本轮没有构造模型、backward或optimizer。当前旧trainer没有上述新合同的完整实现，G6/G8保持阻断。

输出z64及每行source/row/model/input-contract/normalizer hash，遇非有限或invalid整行保持失败分母，不在exporter默默nan_to_num。对z64不作semantic-head替换、不用H选择projection。以后D_fit分别给z64/H fit baseline mean/std及已批准median bandwidth；训练输入scaler不能代替BDD scaler。

## 资源与预算建议

本机候选执行环境为现有nuplan Python3.9（精确版本见runtime evidence），CPU为复现基线；若后续授权GPU则设备/driver/CUDA/torch版本先绑定，无法确定性重现的kernel要披露。本机16GiB RAM、CPU可用，torch1.9未提供MPS/CUDA；拟定进程RSS上限8GiB、单设备；按shard mmap最多cache2，不合并context大数组。

最多3个固定seed、每个30epoch，共90epoch；没有architecture/loss/LR/seed搜索预算。以当前135,046训练行仅作资源算例，ceil(N/128)=1056 batch/epoch，最大95,040 optimizer steps；新U去重/分组后必须重绑定实际N和精确step ceiling，不能把当前N当最终训练承诺。预估总墙钟6–24小时、上限24设备小时/3 runs（估算，未实测、未启动计时训练；若不足即停，不自动扩额）。每epoch日志、数据吞吐、训练/验证loss、梯度有限性、walltime、RNG/loader状态、数据及代码hash全保留。Owner批准预算前不得以估算启动。

NaN/Inf loss/gradient、身份/hash变更、mask/normalizer错配立即停止该run，保留全部日志。仅基础设施中断允许在未来授权预算内从最后完整epoch原state/RNG恢复，不换seed；没有完整state则本run记未完成并请求Owner裁定，禁止多跑挑好者。模型不收敛或语义任务无支持不是偷偷改loss的理由；登记失败，任何新方案需新的U-only前瞻版本。

本轮提供可审查规格，**尚无完整RBR-V2 trainer/config/data版本/normalizer或checkpoint**。这些缺项与输入、独立性修复共同构成`S3_P0_BLOCKED_TRAINING_PROTOCOL`，不能由文字规格替代执行绑定。
