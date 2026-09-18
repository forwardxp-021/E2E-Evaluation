
# S2R Production Execution Binding v1

Status: `PRODUCTION_EXECUTION_BINDING = BLOCKED`.

The zero-run re-audit verified every previously bound component byte-for-byte. The exact code files are identifiable, but the future production chain is not uniquely resolved: there is no dedicated role-aware TSB executor, resolved Hydra configuration, installed production serializer/recorder callback chain, or single primary result manifest. The historical launcher and recorder adapter are HLC-specific and cannot be repurposed by assumption.

| Role | Path | Symbol/config | SHA256 | Component status |
|---|---|---|---|---|
| TSB_candidate | `docs/stageR/r2/r2_bh_tsb_family_development_candidate_v1.0.json` | `frozen candidate` | `7c37fdd2d939e9282adafcd98a76571c0ce9c0812e618c758b004098e5e09538` | BOUND |
| candidate_parameters | `docs/stageR/r2/r2_b_calibration_rounds/r2_b_tsb_round_0_parameters_v1.0.json` | `global_parameters` | `1833b245e2b2f74bc19aad7013f6339f554d9d700cc3077151ab0474169c716d` | BOUND |
| TSB_generator | `tools/r2_b_controller_aware_generator_v1.py` | `tsb_controller_aware_acceleration` | `d166c4746d0a70668b8e26a532890c641b5166f55661a2d34d7f796a9830e3eb` | BOUND |
| TSB_planner | `tools/r2_b_controller_aware_planner_v1.py` | `R2BControllerAwarePlannerV1` | `0b4020703b629e25ea7fde9e8759ae8fa5d03093c27e86a8dd91aba1ca036c7c` | BOUND |
| native_reference | `tools/r1_closed_loop_benchmark_v2_1.py` | `build_native_route_reference_v1_1` | `592c755390f565db229a765901aa4a2af50de78f895ef564dd985b71480f6dbd` | BOUND |
| passive_actual_LQR_recorder | `tools/r2_bj_b0_2_passive_actual_lqr_recorder.py` | `PassiveActualLQRRecorderV1` | `07f6ac5baec4ddf13dae53b379cb838e1374a38a1446eea4549fd17ea4c684b9` | BOUND |
| serializer_and_analyzer_adapter | `tools/s1_protocol_schema.py` | `serialize_trace / analyze_pair` | `6ba7b96b61be2e4c729c1d21e678e34610ef647f1a43b488b750ff42338f17a0` | BLOCKED |
| production_analyzer_dispatcher | `tools/r1_b2_8_r3_2_post_run_evaluator_dispatcher.py` | `evaluate_frozen_pair` | `82870a4b42c9343eb1ec22b20901566f3787964bfa7464ae48ea67e9b23396e1` | BOUND |
| mechanism_evaluator | `tools/r1_context_mechanism_core.py` | `qualify_tsb_pair` | `81be3d676e55ebdb5615883902f025d4408611fc0994dda602b1554023c0653e` | BOUND |
| F_match_evaluator | `tools/r1_closed_loop_benchmark_v2_1.py` | `prospective_primary_f_match` | `592c755390f565db229a765901aa4a2af50de78f895ef564dd985b71480f6dbd` | BOUND |
| official_safety_adapter | `tools/r1_b2_8_r3_1_official_safety_adapter.py` | `official safety artifact adapter` | `7f2e0c1f72b6545ca7de5843528e9dfd6c36d099bf9072f5430b66ae982195db` | BOUND |
| canonical_metric_parser | `tools/r1_official_metric_canonicalizer.py` | `canonical metric parsing` | `78f34dddc49cab2960b05fdfae57e812b20e35b83c43bde9a29a837c7f519985` | BOUND |
| primary80_time_controller | `tools/r1_primary80_scientific_time_controller_v1.py` | `R1Primary80ScientificTimeControllerV1` | `3b78f87593c01ea80e37d5a639a99355ff25a002b2e317409a029766c005f4d1` | BOUND |
| historical_executor_only | `tools/r2_bj_b0_1_production_canary_launcher.py` | `build_production_runner / _run_one` | `d16f215d73b7ce292ea33bb978e506bc399d1fbf13040ad8fb68c27c317fd0d0` | BLOCKED |
| historical_recorder_adapter_only | `tools/r2_bj_b0_2_production_launcher_adapter.py` | `_install_recorder` | `6c7bbc132a336179ffdead29e68775330c3641f826f3110a0c6691a019dd18bc` | BLOCKED |
| runner_entrypoint | `/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit/nuplan/planning/simulation/runner/simulations_runner.py` | `SimulationRunner` | `ef832d01e5d169da7afdc3bddcc494340f6a21c02e29e0dc72334a6328796784` | BLOCKED |
| scenario_builder | `/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit/nuplan/planning/scenario_builder/nuplan_db/nuplan_scenario_builder.py` | `NuPlanScenarioBuilder` | `05a97e0550c9700c068faadded8bdd9982a4671078181ac5c648c0c501ecc0f9` | BLOCKED |
| scenario | `/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit/nuplan/planning/scenario_builder/nuplan_db/nuplan_scenario.py` | `NuPlanScenario` | `28cb9004ab629bc89e8b33c9cdd5ea3d8e437cc3687fdc9a181909797a64eb7a` | BLOCKED |
| scenario_queries | `/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit/nuplan/database/nuplan_db/nuplan_scenario_queries.py` | `get_scenarios_from_db / get_ego_state_for_lidarpc_token_from_db` | `f3430350de5c4392b053106e994f79c1d50bb0c9db39b056dd8bb51fb17b8a71` | BLOCKED |
| scenario_extraction | `/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit/nuplan/planning/scenario_builder/nuplan_db/nuplan_scenario_utils.py` | `ScenarioMapping / extract_sensor_tokens_as_scenario` | `857b59554dc4087eac45f0a808c19468c799b93cdd25a000a61fdf0973ac5853` | BLOCKED |
| simulation_builder | `/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit/nuplan/planning/script/builders/simulation_builder.py` | `build_simulations` | `6d76b60251d4aca471a76cf112b1efe5b884d82ec05d9cac99d19d63f3493e0d` | BLOCKED |
| simulation_setup | `/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit/nuplan/planning/simulation/simulation_setup.py` | `SimulationSetup` | `cb378f5b705cd2965017248a7e3a9efe0a386a298a98a60137325b3debdc47c9` | BLOCKED |
| simulation | `/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit/nuplan/planning/simulation/simulation.py` | `Simulation` | `115a96a9f12d55ca4fc5d1201e6833a3f509ebac393fad32cdb9ed275cd04da3` | BLOCKED |
| controller | `/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit/nuplan/planning/simulation/controller/two_stage_controller.py` | `TwoStageController` | `22797e59f0f13a61ced983a62f8750bb22febac549e1e7ab80f82547c6cadbdd` | BLOCKED |
| tracker | `/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit/nuplan/planning/simulation/controller/tracker/lqr.py` | `LQRTracker` | `34c2ca40a6111824b6b3865520df145a860f71fda0b416108d7c62a0fa3fad5b` | BLOCKED |
| motion_model | `/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit/nuplan/planning/simulation/controller/motion_model/kinematic_bicycle.py` | `KinematicBicycleModel` | `df48a288aed0a88f0484184560b184c28d499bbb6fb967600abf55a59a44ba2d` | BLOCKED |
| history_buffer | `/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit/nuplan/planning/simulation/history/simulation_history_buffer.py` | `SimulationHistoryBuffer` | `2bf67ef5ae19b1d1da44544ed4b43cb3eeef9c21e38633da99bcd7783ad12c3a` | BLOCKED |
| callbacks | `/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit/nuplan/planning/script/builders/simulation_callback_builder.py` | `build_simulation_callbacks` | `cdd381f8f2fd5d11756a7c134e3bdd6798931443c99bbbd462c0bf2d79108a95` | BLOCKED |
| official_seed_initialization | `/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit/nuplan/planning/script/run_simulation.py` | `run_simulation` | `d1f0380a73bc625816aa3d74b270305feb5d4f723ac04ea2a84a23710c928d21` | BLOCKED |
| scenario_config | `/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit/nuplan/planning/script/config/common/scenario_builder/nuplan_mini.yaml` | `Hydra defaults` | `08901ced89ff4e190816e597fbb7d79bf5179f6b8a3b5fb6839181fceb8bb14f` | BLOCKED |
| scenario_mapping_config | `/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit/nuplan/planning/script/config/common/scenario_builder/scenario_mapping/nuplan_scenario_mapping.yaml` | `subsample .5; mapped extraction -3s` | `19eb18ef109c116e90f81c66c242c7d65c1449402803b08beda664311acfcbd2` | BLOCKED |
| controller_config | `/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit/nuplan/planning/script/config/simulation/ego_controller/two_stage_controller.yaml` | `tracker / motion_model` | `b876fc8be316a50860eefaf39f1e6b422816d51c72518868e4034e68ccbb24d7` | BLOCKED |
| tracker_config | `/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit/nuplan/planning/script/config/simulation/ego_controller/tracker/lqr_tracker.yaml` | `LQR numeric configuration` | `79d9baa95cf6b4046f0f230d5b4576c13d13b50a2871f2af5f1d2d5d45f6a14c` | BLOCKED |
| motion_model_config | `/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit/nuplan/planning/script/config/simulation/ego_controller/motion_model/kinematic_bicycle_model.yaml` | `motion model numeric configuration` | `f635af6aeae0f050dda2f2d8713b148ecd82def0fc12a86141dc813bedb79f7a` | BLOCKED |

Exact audited git SHA: `9ee519fbcb497570b4f2aab13486d4f3cc49ad57`. Branch: `20260825_stageR_new`.

Runtime: `CPython 3.13.13` at `/Users/liuqing/miniconda3/bin/python`; platform `macOS-26.6.2-arm64-arm-64bit-Mach-O`.

Exact preflight artifacts are under `docs/stageR/s2r_role_aware_preflight/`. Closure requires one production entrypoint plus resolved configuration that instantiates the frozen planner/generator, actual controller/LQR, passive recorder, canonical serializer, analyzer, safety artifacts, and one primary result manifest without fallback dispatch.
