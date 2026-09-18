
# S2R Production Executor Contract v1

Status: `PRODUCTION_EXECUTION_BINDING = PASS`.

The only future production entrypoint is `tools/s2r_production_executor.py --execute-authorized-arm`. It is closed unless an exact Owner authorization binds the executor SHA, one run ID, and a one-arm budget. A pair orchestrator must launch baseline and treatment as two fresh processes using `isolated_arm_command`; no alternative or legacy executor is allowed.

The unique stack is: official nuPlan `SimulationRunner` → `R2BControllerAwarePlannerV1` with frozen round-0 TSB parameters → `TwoStageController` / `LQRTracker` / `KinematicBicycleModel` → sequential callback chain → `PassiveActualLQRRecorderV1` → `s1_protocol_schema` serializer → `r1_b2_8_r3_2_post_run_evaluator_dispatcher.evaluate_frozen_pair` analyzer. The resolved Hydra override contract fixes exact scenario token, Primary80 controller, sequential worker, seed 2026091801, metrics, and output namespace.

Execution manifest schema `s2r_execution_manifest_v1` binds all 25 required fields before `runner.run()`. Technical failures remain recorded and cannot trigger scenario replacement. Audited source SHA is `0268d4b2bafb6ec8cc7f76d5a385aaa93f7d90f2`; the final commit SHA must be rebound by Owner authorization before any future execution.
