# F2 实现与入口预检 v1

状态：ISOLATED_MODEL_PASS + PRODUCTION_ZERO_RUN_PREFLIGHT_PASS。不是正式Development PASS。

5项接口测试通过：整数边界/错误输入拒绝、两臂命令冲量与重心、两臂均绕过全部纵向solver且保留横向调用/不改输入状态、错误时钟拒绝、Owner参数与继承资格精确一致。测试中没有科学runner或model rollout。

隔离验证固定初态为(x,y,heading)=(0,0,0)、vx14 m/s、a0、time_us1000000000；两个独立官方model和tracker，各79次真实模型更新。Baseline一有效制动段；Treatment两段，峰约1.27671/1.27673 m/s²，支持各.9秒，剂量约1.00606/1.00614 m/s。18次中段请求严格0；原横向LQR各79调用，传播/滤波和运动量接近检查通过。完整158区间在evidence/F2_Isolated_Model_Validation_v1.json，不当作独立科学结果。其generic helper字段属于历史诊断schema，批准与否以本轮F2_Owner_Protocol为准。

生产预检构建两个fresh官方完整对象图；runner/history/controller/model/metric/serializer/finalizer guard调用全部0。准确原始初态逐值一致，precontext/route一致，81个metadata时间/token映射与历史DEV01原件相同、实际80帧范围−330～+323us；没有插值或换index。两臂使用同一个新adapter，官方model、16项指标配置、metric finalizer、serializer与实际命令记录器绑定检查通过。有限初始route查询通过，不冒称未来完整episode已执行。

源码及依赖按新F2 source commit及Execution Source Binding冻结；正式授权独立记录，只有两run ID。真实执行会重新构造所有对象，并比较resolved config与本预检，除输出路径和run-start timestamp之外必须相同。正式安全/科学结果仍未知。
