# R1 官方 nuPlan Replay 合同 v0.1

总状态：`VERSION_AMBIGUOUS`；执行授权：`false`。

## 固定 seed 与传播

`MASTER_SEED = 2026082701`。Python `random`、NumPy、Torch（若使用）、Hydra/config seed 必须接收该值；scenario 与 selector 按 `SHA256(selector_version|MASTER_SEED|family|scenario_token|log_id)` 稳定排序。绑定 planner 当前不消费随机数，记录为 `DETERMINISTIC_NO_SEED_CONSUMPTION_IN_BOUND_PLANNER`。

## 版本与文件绑定

- nuPlan devkit `1.2.2`；Hydra `1.1.0rc1`；OmegaConf `2.1.0rc1`。
- map root fingerprint：`a85e17eba18e5fdd65148705844b8f189bb4d4373a1d82805e1f8ffd4ae8afb3`。
- planner SHA：`284d60263621a99b3e57f63f3092797e44a9c393ec6975f419ed02ecb64885d0`。
- generator preparation SHA：`b2f50a826923e91a393be0be026939e45f62192535b741d997ac7ef31a65644e`。
- simulation config SHA：`94a642bd277f77caf0e8af275fe0b43780c0dd198ae9b784a2f59286f0124003`。

## 未关闭项

尚未在获授权的 official closed-loop replay 中证明背景交通与全部 simulation 组件的确定性。因此 seed 合同本身为 `READY`，但 background replay 为 `VERSION_AMBIGUOUS`，official replay 为 `NOT_READY`；本合同不授权 smoke 或 planner rollout。
