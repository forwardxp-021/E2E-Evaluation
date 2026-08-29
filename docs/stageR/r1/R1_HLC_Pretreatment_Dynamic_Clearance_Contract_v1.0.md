# R1 HLC Pretreatment Dynamic Clearance Contract v1.0

状态：`FROZEN_PROSPECTIVE_PRETREATMENT_COMMON_ENVELOPE`。

冻结 numerics：horizon 8.0 s、nominal query dt 0.1 s、maximum actor interpolation gap 0.25 s、longitudinal buffer 3.0 m、lateral buffer 0.5 m。0.25/3.0/0.5 来源于既有 Stage7L treatment-independent engineering clearance：`tools/stage7l_dynamic_clearance.py` SHA-256 `50253b75eed8473b1141b3b76d51ce755f1a8df1f7a2f0690a4075b35a3df129`，不是 R1 outcome-derived threshold。

ego footprint 必须来自 official runtime vehicle parameters，禁止 fallback；actor footprint 必须来自 official track dimensions。缺失即 `NOT_ELIGIBLE`。map 与 actor 均禁止 extrapolation。

同一 scenario 的共同包络必须同时覆盖 HLC baseline 与 Option-B treatment，不得按 arm 改变。唯一动态来源是 original replay tracks。禁止读取 planner/mechanism/F_match/safety outcome、representation、BDD、probe、checkpoint 或 RBR。
