# S3-P0 Minimal Repair Plan v1 — NOT EXECUTED

本轮发现多项阻断；这份变更方案不产生实施或训练授权，不把工作范围重新扩展为TSB平台开发。

1. **Input83合同与适配**：Owner冻结O1后，新建versioned canonical builder/adapter与schema（保留原Stage5D/6R/7E）；纠正time/slot layout、anchor、velocity、time/heading/continuity导数、closing与mask；在现有14 fixtures之外增加弯道/交叉口地图与缺token的适配测试。通过独立物理oracle及两源numerical parity后，零训练生成小量新版本U样本。对未来nuPlan只做fixture/metadata接口测试，不开启rollout。验收G4/G8，旧文件SHA不变。
2. **U依赖分组**：恢复source segment/时间映射与原始轨迹索引；通过已确证重复pair作positive-control，加入错位80帧与近似world-track交集搜索。完成全量分组图、去重规则、跨split leakage audit，再在新目录冻结source-group split；无法恢复足够独立证据则继续BLOCKED，不能仅文件级随机划分。验收G2/G3。
3. **训练协议可执行绑定**：按O3/O4审查结果实现一个受U manifest白名单约束的trainer/config/normalizer/export interface；零optimizer测试shape、mask、train-only scaler、semantic/repr梯度连通、U-only path guard、deterministic selection/resume；完整source/runtime/data hash。没有科学训练或checkpoint选择。验收G6/G8/G10/G11并重新跑12 gates。

只有以上修复、所有mandatory gates与Owner审查完成后才提出U-only training authorization；不得凭本方案执行训练。未来Primary的D-only power/coverage与role capacity仍依原SAP独立授权，不能把B1扩充、H削弱或新sim当本次修复。
