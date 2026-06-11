# Stage 5A 分片合并报告

- 分片总数：35
- n_windows_kept：164871
- split_counts：{"test": 16392, "train": 131998, "val": 16481}
- slot occupied window ratio：{"front": 0.26709366717009053, "left_front": 0.14126195631736327, "left_rear": 0.15141534896980063, "right_front": 0.1579962516148989, "right_rear": 0.15896670730449866}
- slot valid frame ratio：{"front": 0.26406660358704687, "left_front": 0.13965577633422493, "left_rear": 0.15014829776006697, "right_front": 0.15628028883187461, "right_rear": 0.15772429657125875}
- empty slot ratio：{"front": 0.7329063328299095, "left_front": 0.8587380436826367, "left_rear": 0.8485846510301994, "right_front": 0.8420037483851011, "right_rear": 0.8410332926955013}
- lane_context_quality_counts：{"good": 163220, "ambiguous_intersection": 1651}
- fallback_assignment_rate：0.00000000
- nonfinite_output_detected：0
- global standardization train_count：131998
- 说明：本次仅进行清单与统计汇总，大型张量（例如 ego_seq / neighbor_seq / context_traj）保持分片存储，不进行全量拼接。
