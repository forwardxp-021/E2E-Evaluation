# B1-IR Official Metric Lifecycle Fix v1

## Root cause

`tools/b1_tsb_qualification_executor.py` previously called `arm.runner.run()` and `arm.recorder.validate_complete()`, then immediately wrote `TECHNICAL_COMPLETE_PENDING_PAIR_ANALYSIS`. It omitted nuPlan's post-run `save_runner_reports(...)` and `common_builder.multi_main_callback.on_run_simulation_end()` sequence. The callback chain was already started by `set_up_common_builder`; `MetricFileCallback` owned the missing integration from per-scenario `.pickle.temp` into metric-specific parquet files. Recorder completion therefore occurred before metric finalization, and exceptions/early returns skipped finalization.

## Fix

The future executor now requires the official `MetricFileCallback` binding before budget claim and runner entry. After a successful runner it enforces recorder validation, runner-report serialization, official main-callback finalization, serializer/manifest completion, required-artifact checks and SHA256 validation. Any failure becomes `TECHNICAL_INCOMPLETE`; `ARM_COMPLETE` is impossible until every lifecycle state is complete. The changed executor SHA invalidates the old execution authorization, so it cannot run without a new Owner record.

Expected official outputs are `raw/metrics/<metric>.parquet`, including `no_ego_at_fault_collisions.parquet` and `drivable_area_compliance.parquet`.

## Offline recovery

Each of the 34 runner/recorder-complete arms retained one trusted `.pickle.temp` containing all 16 official nuPlan metric rows. `tools/b1_offline_metric_finalize.py` calls the official nuPlan `MetricFileCallback` with deletion disabled. Recovery does not construct a simulator or invoke a planner/controller. All 34 arms were finalized and input/output hashes are frozen in `B1_IR_Offline_Finalize_Audit_v1.json`.
