# B1 TSB Qualification Execution Report v1

Main status: `B1_TSB_BENCHMARK_QUALIFICATION_INCOMPLETE_INFRASTRUCTURE`.

The frozen B1 roster contained 20 independent SESSION pairs and 40 authorized arms. Attempted arms: 35; runner/recorder complete arms: 34; full-contract technical complete arms: 0. Executed pairs: 18; full-contract technical complete pairs: 0; scientifically evaluable pairs: 0.

- Baseline one-phase success: `17/20`
- Treatment two-stage success: `17/20`
- Joint mechanism success: `17/20`
- F_match pass: `17/20`
- Official safety pass: `0/0` evaluable pairs; required official safety artifacts were unavailable for all 17 trace-complete pairs, and 3 pairs were not fully run
- LOW_SPEED_ENDSTOP: `0`
- Full-contract PASS count: `0/20`; descriptive full-denominator Wilson bound `[0.000000, 0.161125]`. Because scientifically evaluable pairs equal 0, this is not an estimable scientific qualification rate.
- Frozen all-20 success rule satisfied: `FALSE`

Failure ledger:
- `B1-TSB-01` — `TECHNICAL_INCOMPLETE`: MetricCanonicalizationError:collision:MISSING_EXPECTED_METRIC_FILE:no_ego_at_fault_collisions.parquet
- `B1-TSB-02` — `TECHNICAL_INCOMPLETE`: MetricCanonicalizationError:collision:MISSING_EXPECTED_METRIC_FILE:no_ego_at_fault_collisions.parquet
- `B1-TSB-03` — `TECHNICAL_INCOMPLETE`: MetricCanonicalizationError:collision:MISSING_EXPECTED_METRIC_FILE:no_ego_at_fault_collisions.parquet
- `B1-TSB-04` — `TECHNICAL_INCOMPLETE`: MetricCanonicalizationError:collision:MISSING_EXPECTED_METRIC_FILE:no_ego_at_fault_collisions.parquet
- `B1-TSB-05` — `TECHNICAL_INCOMPLETE`: MetricCanonicalizationError:collision:MISSING_EXPECTED_METRIC_FILE:no_ego_at_fault_collisions.parquet
- `B1-TSB-06` — `TECHNICAL_INCOMPLETE`: MetricCanonicalizationError:collision:MISSING_EXPECTED_METRIC_FILE:no_ego_at_fault_collisions.parquet
- `B1-TSB-07` — `TECHNICAL_INCOMPLETE`: MetricCanonicalizationError:collision:MISSING_EXPECTED_METRIC_FILE:no_ego_at_fault_collisions.parquet
- `B1-TSB-08` — `TECHNICAL_INCOMPLETE`: MetricCanonicalizationError:collision:MISSING_EXPECTED_METRIC_FILE:no_ego_at_fault_collisions.parquet
- `B1-TSB-09` — `TECHNICAL_INCOMPLETE`: MetricCanonicalizationError:collision:MISSING_EXPECTED_METRIC_FILE:no_ego_at_fault_collisions.parquet
- `B1-TSB-10` — `TECHNICAL_INCOMPLETE`: MetricCanonicalizationError:collision:MISSING_EXPECTED_METRIC_FILE:no_ego_at_fault_collisions.parquet
- `B1-TSB-11` — `TECHNICAL_INCOMPLETE`: MetricCanonicalizationError:collision:MISSING_EXPECTED_METRIC_FILE:no_ego_at_fault_collisions.parquet
- `B1-TSB-12` — `TECHNICAL_INCOMPLETE`: MetricCanonicalizationError:collision:MISSING_EXPECTED_METRIC_FILE:no_ego_at_fault_collisions.parquet
- `B1-TSB-13` — `TECHNICAL_INCOMPLETE`: MetricCanonicalizationError:collision:MISSING_EXPECTED_METRIC_FILE:no_ego_at_fault_collisions.parquet
- `B1-TSB-14` — `TECHNICAL_INCOMPLETE`: MetricCanonicalizationError:collision:MISSING_EXPECTED_METRIC_FILE:no_ego_at_fault_collisions.parquet
- `B1-TSB-15` — `TECHNICAL_INCOMPLETE`: MetricCanonicalizationError:collision:MISSING_EXPECTED_METRIC_FILE:no_ego_at_fault_collisions.parquet
- `B1-TSB-16` — `TECHNICAL_INCOMPLETE`: MetricCanonicalizationError:collision:MISSING_EXPECTED_METRIC_FILE:no_ego_at_fault_collisions.parquet
- `B1-TSB-17` — `TECHNICAL_INCOMPLETE`: MetricCanonicalizationError:collision:MISSING_EXPECTED_METRIC_FILE:no_ego_at_fault_collisions.parquet
- `B1-TSB-18` — `TECHNICAL_INCOMPLETE`: RuntimeError:ARM_BUDGET_STATUS:BASELINE=NOT_RUN,TREATMENT=TECHNICAL_INCOMPLETE;INFRASTRUCTURE_STOP=ValueError: NATIVE_ROUTE_FAIL: no native outgoing successor into 19339
- `B1-TSB-19` — `TECHNICAL_INCOMPLETE`: RuntimeError:ARM_BUDGET_STATUS:BASELINE=NOT_RUN,TREATMENT=NOT_RUN
- `B1-TSB-20` — `TECHNICAL_INCOMPLETE`: RuntimeError:ARM_BUDGET_STATUS:BASELINE=NOT_RUN,TREATMENT=NOT_RUN

B1 exposure was E2=17, E1=3, E5=0. RBR, H, BDD, z64, MMD, detector performance, and Primary comparison were not read or computed. `LOW_ORDER_NUISANCE_ELIMINATED = NOT_ESTABLISHED`; `TSB_CLEAN_RESIDUAL_TASK = NOT_ESTABLISHED`.
