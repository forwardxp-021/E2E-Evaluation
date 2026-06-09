# Stage 5C Context Embedding Evaluation
- Strict schema mode used: **True**
- feature_schema.json loaded: **yes**
- No fallback feature index was used: **yes**
- mean_speed and std_rel_speed are not part of the Stage 5 schema and were not evaluated.
- p95_rel_speed is used instead of std_rel_speed.
- Paper-grade valid: **yes**

## Retrieval Results
| representation            |   k |   mean_neighbor_feature_distance |   median_neighbor_feature_distance |   hit_at_1 |   hit_at_5 |   mean_same_label_fraction_at_5 |   mean_same_label_fraction_at_10 |
|:--------------------------|----:|---------------------------------:|-----------------------------------:|-----------:|-----------:|--------------------------------:|---------------------------------:|
| learned_context_embedding |  10 |                          1.92786 |                           1.30228  |  0.197962  |  0.507565  |                       0.179917  |                        0.166673  |
| raw_feature               |  10 |                          1.27145 |                           0.767944 |  0.266166  |  0.586018  |                       0.233919  |                        0.21652   |
| pca_feature               |  10 |                          1.33179 |                           0.823785 |  0.266776  |  0.5959    |                       0.236298  |                        0.22004   |
| context_l2                |  10 |                          3.58161 |                           2.67388  |  0.0857735 |  0.267509  |                       0.0809053 |                        0.076336  |
| random                    |  10 |                          6.86718 |                           6.45809  |  0.01092   |  0.0601513 |                       0.0123963 |                        0.0123475 |

## Style-distance Correlation
| representation            | target_feature                            |   spearman_corr |      p_value |   n_pairs |
|:--------------------------|:------------------------------------------|----------------:|-------------:|----------:|
| learned_context_embedding | rms_accel_delta                           |     0.163131    | 3.3569e-295  |     49996 |
| learned_context_embedding | rms_jerk_delta                            |     0.15779     | 4.39132e-276 |     49996 |
| learned_context_embedding | max_abs_accel_delta                       |     0.149394    | 2.19069e-247 |     49996 |
| learned_context_embedding | max_abs_jerk_delta                        |     0.151372    | 5.3513e-254  |     49996 |
| learned_context_embedding | mean_thw_delta                            |     0.462977    | 0            |     49996 |
| learned_context_embedding | min_thw_delta                             |     0.505599    | 0            |     49996 |
| learned_context_embedding | mean_front_distance_delta                 |     0.568356    | 0            |     49996 |
| learned_context_embedding | min_front_distance_delta                  |     0.560617    | 0            |     49996 |
| learned_context_embedding | mean_rel_speed_delta                      |     0.567182    | 0            |     49996 |
| learned_context_embedding | p95_rel_speed_delta                       |     0.569674    | 0            |     49996 |
| learned_context_embedding | front_pressure_score_delta                |     0.418097    | 0            |     49996 |
| learned_context_embedding | rear_vehicle_pressure_proxy_delta         |     0.317999    | 0            |     49996 |
| learned_context_embedding | rms_yaw_rate_delta                        |     0.22791     | 0            |     49996 |
| learned_context_embedding | rms_curvature_delta                       |    -0.004996    | 0.263963     |     49996 |
| learned_context_embedding | heading_change_total_delta                |     0.229383    | 0            |     49996 |
| learned_context_embedding | lane_change_count_proxy_delta             |     0.435796    | 0            |     49996 |
| learned_context_embedding | lane_change_rate_proxy_delta              |     0.435796    | 0            |     49996 |
| learned_context_embedding | max_lateral_speed_delta                   |     0.293975    | 0            |     49996 |
| learned_context_embedding | rms_lateral_accel_delta                   |     0.188734    | 0            |     49996 |
| learned_context_embedding | lane_change_oscillation_score_proxy_delta |     0.434306    | 0            |     49996 |
| learned_context_embedding | left_front_min_gap_delta                  |     0.274638    | 0            |     49996 |
| learned_context_embedding | left_rear_min_gap_delta                   |     0.203474    | 0            |     49996 |
| learned_context_embedding | right_front_min_gap_delta                 |     0.262206    | 0            |     49996 |
| learned_context_embedding | right_rear_min_gap_delta                  |     0.21095     | 0            |     49996 |
| learned_context_embedding | left_gap_min_delta                        |     0.187631    | 0            |     49996 |
| learned_context_embedding | right_gap_min_delta                       |     0.19311     | 0            |     49996 |
| learned_context_embedding | left_gap_acceptance_proxy_delta           |     0.27061     | 0            |     49996 |
| learned_context_embedding | right_gap_acceptance_proxy_delta          |     0.262107    | 0            |     49996 |
| learned_context_embedding | yielding_score_proxy_delta                |     0.548384    | 0            |     49996 |
| learned_context_embedding | assertiveness_score_proxy_delta           |     0.0853137   | 2.0714e-81   |     49996 |
| raw_feature               | rms_accel_delta                           |     0.179187    | 0            |     49996 |
| raw_feature               | rms_jerk_delta                            |     0.172254    | 0            |     49996 |
| raw_feature               | max_abs_accel_delta                       |     0.169303    | 0            |     49996 |
| raw_feature               | max_abs_jerk_delta                        |     0.169325    | 0            |     49996 |
| raw_feature               | mean_thw_delta                            |     0.449228    | 0            |     49996 |
| raw_feature               | min_thw_delta                             |     0.478452    | 0            |     49996 |
| raw_feature               | mean_front_distance_delta                 |     0.533407    | 0            |     49996 |
| raw_feature               | min_front_distance_delta                  |     0.530416    | 0            |     49996 |
| raw_feature               | mean_rel_speed_delta                      |     0.531463    | 0            |     49996 |
| raw_feature               | p95_rel_speed_delta                       |     0.528296    | 0            |     49996 |
| raw_feature               | front_pressure_score_delta                |     0.385566    | 0            |     49996 |
| raw_feature               | rear_vehicle_pressure_proxy_delta         |     0.32164     | 0            |     49996 |
| raw_feature               | rms_yaw_rate_delta                        |     0.173724    | 0            |     49996 |
| raw_feature               | rms_curvature_delta                       |    -0.0141871   | 0.00151248   |     49996 |
| raw_feature               | heading_change_total_delta                |     0.175171    | 0            |     49996 |
| raw_feature               | lane_change_count_proxy_delta             |     0.364624    | 0            |     49996 |
| raw_feature               | lane_change_rate_proxy_delta              |     0.364624    | 0            |     49996 |
| raw_feature               | max_lateral_speed_delta                   |     0.213197    | 0            |     49996 |
| raw_feature               | rms_lateral_accel_delta                   |     0.16685     | 6.6553e-309  |     49996 |
| raw_feature               | lane_change_oscillation_score_proxy_delta |     0.367263    | 0            |     49996 |
| raw_feature               | left_front_min_gap_delta                  |     0.337829    | 0            |     49996 |
| raw_feature               | left_rear_min_gap_delta                   |     0.239112    | 0            |     49996 |
| raw_feature               | right_front_min_gap_delta                 |     0.307821    | 0            |     49996 |
| raw_feature               | right_rear_min_gap_delta                  |     0.229847    | 0            |     49996 |
| raw_feature               | left_gap_min_delta                        |     0.249746    | 0            |     49996 |
| raw_feature               | right_gap_min_delta                       |     0.241126    | 0            |     49996 |
| raw_feature               | left_gap_acceptance_proxy_delta           |     0.320848    | 0            |     49996 |
| raw_feature               | right_gap_acceptance_proxy_delta          |     0.291956    | 0            |     49996 |
| raw_feature               | yielding_score_proxy_delta                |     0.509074    | 0            |     49996 |
| raw_feature               | assertiveness_score_proxy_delta           |     0.08438     | 1.13117e-79  |     49996 |
| pca_feature               | rms_accel_delta                           |     0.180609    | 0            |     49996 |
| pca_feature               | rms_jerk_delta                            |     0.173972    | 0            |     49996 |
| pca_feature               | max_abs_accel_delta                       |     0.170984    | 0            |     49996 |
| pca_feature               | max_abs_jerk_delta                        |     0.171106    | 0            |     49996 |
| pca_feature               | mean_thw_delta                            |     0.446952    | 0            |     49996 |
| pca_feature               | min_thw_delta                             |     0.476728    | 0            |     49996 |
| pca_feature               | mean_front_distance_delta                 |     0.532849    | 0            |     49996 |
| pca_feature               | min_front_distance_delta                  |     0.530272    | 0            |     49996 |
| pca_feature               | mean_rel_speed_delta                      |     0.530839    | 0            |     49996 |
| pca_feature               | p95_rel_speed_delta                       |     0.528101    | 0            |     49996 |
| pca_feature               | front_pressure_score_delta                |     0.383895    | 0            |     49996 |
| pca_feature               | rear_vehicle_pressure_proxy_delta         |     0.314882    | 0            |     49996 |
| pca_feature               | rms_yaw_rate_delta                        |     0.179147    | 0            |     49996 |
| pca_feature               | rms_curvature_delta                       |    -0.0102046   | 0.0225057    |     49996 |
| pca_feature               | heading_change_total_delta                |     0.180236    | 0            |     49996 |
| pca_feature               | lane_change_count_proxy_delta             |     0.378403    | 0            |     49996 |
| pca_feature               | lane_change_rate_proxy_delta              |     0.378403    | 0            |     49996 |
| pca_feature               | max_lateral_speed_delta                   |     0.218191    | 0            |     49996 |
| pca_feature               | rms_lateral_accel_delta                   |     0.169181    | 0            |     49996 |
| pca_feature               | lane_change_oscillation_score_proxy_delta |     0.380289    | 0            |     49996 |
| pca_feature               | left_front_min_gap_delta                  |     0.325488    | 0            |     49996 |
| pca_feature               | left_rear_min_gap_delta                   |     0.234778    | 0            |     49996 |
| pca_feature               | right_front_min_gap_delta                 |     0.295425    | 0            |     49996 |
| pca_feature               | right_rear_min_gap_delta                  |     0.226002    | 0            |     49996 |
| pca_feature               | left_gap_min_delta                        |     0.245342    | 0            |     49996 |
| pca_feature               | right_gap_min_delta                       |     0.238063    | 0            |     49996 |
| pca_feature               | left_gap_acceptance_proxy_delta           |     0.307421    | 0            |     49996 |
| pca_feature               | right_gap_acceptance_proxy_delta          |     0.278396    | 0            |     49996 |
| pca_feature               | yielding_score_proxy_delta                |     0.508539    | 0            |     49996 |
| pca_feature               | assertiveness_score_proxy_delta           |     0.0893705   | 3.49722e-89  |     49996 |
| context_l2                | rms_accel_delta                           |    -0.0175211   | 8.93265e-05  |     49996 |
| context_l2                | rms_jerk_delta                            |    -0.012341    | 0.00578958   |     49996 |
| context_l2                | max_abs_accel_delta                       |    -0.0354897   | 2.05941e-15  |     49996 |
| context_l2                | max_abs_jerk_delta                        |    -0.0277123   | 5.73864e-10  |     49996 |
| context_l2                | mean_thw_delta                            |     0.542722    | 0            |     49996 |
| context_l2                | min_thw_delta                             |     0.483359    | 0            |     49996 |
| context_l2                | mean_front_distance_delta                 |     0.403448    | 0            |     49996 |
| context_l2                | min_front_distance_delta                  |     0.399326    | 0            |     49996 |
| context_l2                | mean_rel_speed_delta                      |     0.350992    | 0            |     49996 |
| context_l2                | p95_rel_speed_delta                       |     0.371127    | 0            |     49996 |
| context_l2                | front_pressure_score_delta                |     0.444318    | 0            |     49996 |
| context_l2                | rear_vehicle_pressure_proxy_delta         |     0.327375    | 0            |     49996 |
| context_l2                | rms_yaw_rate_delta                        |    -0.059441    | 2.24444e-40  |     49996 |
| context_l2                | rms_curvature_delta                       |     0.211846    | 0            |     49996 |
| context_l2                | heading_change_total_delta                |    -0.0602909   | 1.72316e-41  |     49996 |
| context_l2                | lane_change_count_proxy_delta             |    -0.0522322   | 1.49065e-31  |     49996 |
| context_l2                | lane_change_rate_proxy_delta              |    -0.0522322   | 1.49065e-31  |     49996 |
| context_l2                | max_lateral_speed_delta                   |    -0.109723    | 1.05167e-133 |     49996 |
| context_l2                | rms_lateral_accel_delta                   |    -0.0474466   | 2.54711e-26  |     49996 |
| context_l2                | lane_change_oscillation_score_proxy_delta |    -0.0541098   | 9.64964e-34  |     49996 |
| context_l2                | left_front_min_gap_delta                  |     0.300433    | 0            |     49996 |
| context_l2                | left_rear_min_gap_delta                   |     0.312687    | 0            |     49996 |
| context_l2                | right_front_min_gap_delta                 |     0.304107    | 0            |     49996 |
| context_l2                | right_rear_min_gap_delta                  |     0.326779    | 0            |     49996 |
| context_l2                | left_gap_min_delta                        |     0.229279    | 0            |     49996 |
| context_l2                | right_gap_min_delta                       |     0.242358    | 0            |     49996 |
| context_l2                | left_gap_acceptance_proxy_delta           |     0.246518    | 0            |     49996 |
| context_l2                | right_gap_acceptance_proxy_delta          |     0.246378    | 0            |     49996 |
| context_l2                | yielding_score_proxy_delta                |     0.349041    | 0            |     49996 |
| context_l2                | assertiveness_score_proxy_delta           |     0.0550268   | 7.7197e-35   |     49996 |
| random                    | rms_accel_delta                           |     0.00122868  | 0.783528     |     49996 |
| random                    | rms_jerk_delta                            |     0.00191199  | 0.66901      |     49996 |
| random                    | max_abs_accel_delta                       |     0.00218168  | 0.625686     |     49996 |
| random                    | max_abs_jerk_delta                        |     0.00266885  | 0.550684     |     49996 |
| random                    | mean_thw_delta                            |    -0.00302776  | 0.498415     |     49996 |
| random                    | min_thw_delta                             |    -0.00182662  | 0.682966     |     49996 |
| random                    | mean_front_distance_delta                 |     0.00182884  | 0.682602     |     49996 |
| random                    | min_front_distance_delta                  |     0.0012863   | 0.773647     |     49996 |
| random                    | mean_rel_speed_delta                      |     0.00258467  | 0.563323     |     49996 |
| random                    | p95_rel_speed_delta                       |     0.00297907  | 0.505348     |     49996 |
| random                    | front_pressure_score_delta                |     0.0010635   | 0.812044     |     49996 |
| random                    | rear_vehicle_pressure_proxy_delta         |    -0.00314683  | 0.481676     |     49996 |
| random                    | rms_yaw_rate_delta                        |     0.00144308  | 0.746951     |     49996 |
| random                    | rms_curvature_delta                       |     0.00305943  | 0.493933     |     49996 |
| random                    | heading_change_total_delta                |     4.36872e-05 | 0.992206     |     49996 |
| random                    | lane_change_count_proxy_delta             |    -0.00476239  | 0.286949     |     49996 |
| random                    | lane_change_rate_proxy_delta              |    -0.00476239  | 0.286949     |     49996 |
| random                    | max_lateral_speed_delta                   |     0.00213891  | 0.632476     |     49996 |
| random                    | rms_lateral_accel_delta                   |     0.00148975  | 0.739061     |     49996 |
| random                    | lane_change_oscillation_score_proxy_delta |    -0.00355681  | 0.426452     |     49996 |
| random                    | left_front_min_gap_delta                  |     0.000543901 | 0.903206     |     49996 |
| random                    | left_rear_min_gap_delta                   |     0.000954232 | 0.831047     |     49996 |
| random                    | right_front_min_gap_delta                 |     0.00419693  | 0.348036     |     49996 |
| random                    | right_rear_min_gap_delta                  |    -0.0129303   | 0.00383741   |     49996 |
| random                    | left_gap_min_delta                        |    -0.00384528  | 0.389912     |     49996 |
| random                    | right_gap_min_delta                       |    -0.00252692  | 0.572073     |     49996 |
| random                    | left_gap_acceptance_proxy_delta           |     0.000971561 | 0.828026     |     49996 |
| random                    | right_gap_acceptance_proxy_delta          |     0.00732479  | 0.101466     |     49996 |
| random                    | yielding_score_proxy_delta                |     0.00238648  | 0.593618     |     49996 |
| random                    | assertiveness_score_proxy_delta           |     0.00486539  | 0.276652     |     49996 |

## Context Sensitivity
| representation            | context_variable            | metric_name             |   metric_value |
|:--------------------------|:----------------------------|:------------------------|---------------:|
| learned_context_embedding | mean_thw                    | mean_abs_neighbor_delta |    0.119745    |
| learned_context_embedding | mean_thw                    | nn_value_spearman_corr  |    0.9961      |
| raw_feature               | mean_thw                    | mean_abs_neighbor_delta |    0.0661558   |
| raw_feature               | mean_thw                    | nn_value_spearman_corr  |    0.993084    |
| pca_feature               | mean_thw                    | mean_abs_neighbor_delta |    0.0640526   |
| pca_feature               | mean_thw                    | nn_value_spearman_corr  |    0.992915    |
| context_l2                | mean_thw                    | mean_abs_neighbor_delta |    0.144117    |
| context_l2                | mean_thw                    | nn_value_spearman_corr  |    0.883764    |
| random                    | mean_thw                    | mean_abs_neighbor_delta |    0.397997    |
| random                    | mean_thw                    | nn_value_spearman_corr  |    0.00866133  |
| learned_context_embedding | min_thw                     | mean_abs_neighbor_delta |    0.138442    |
| learned_context_embedding | min_thw                     | nn_value_spearman_corr  |    0.995033    |
| raw_feature               | min_thw                     | mean_abs_neighbor_delta |    0.112039    |
| raw_feature               | min_thw                     | nn_value_spearman_corr  |    0.993734    |
| pca_feature               | min_thw                     | mean_abs_neighbor_delta |    0.123246    |
| pca_feature               | min_thw                     | nn_value_spearman_corr  |    0.993238    |
| context_l2                | min_thw                     | mean_abs_neighbor_delta |    0.262643    |
| context_l2                | min_thw                     | nn_value_spearman_corr  |    0.886237    |
| random                    | min_thw                     | mean_abs_neighbor_delta |    0.767927    |
| random                    | min_thw                     | nn_value_spearman_corr  |   -0.0012544   |
| learned_context_embedding | mean_front_distance         | mean_abs_neighbor_delta |    0.109999    |
| learned_context_embedding | mean_front_distance         | nn_value_spearman_corr  |    0.996305    |
| raw_feature               | mean_front_distance         | mean_abs_neighbor_delta |    0.0929258   |
| raw_feature               | mean_front_distance         | nn_value_spearman_corr  |    0.994401    |
| pca_feature               | mean_front_distance         | mean_abs_neighbor_delta |    0.0919337   |
| pca_feature               | mean_front_distance         | nn_value_spearman_corr  |    0.994952    |
| context_l2                | mean_front_distance         | mean_abs_neighbor_delta |    0.209667    |
| context_l2                | mean_front_distance         | nn_value_spearman_corr  |    0.893102    |
| random                    | mean_front_distance         | mean_abs_neighbor_delta |    0.808067    |
| random                    | mean_front_distance         | nn_value_spearman_corr  |    4.04034e-05 |
| learned_context_embedding | min_front_distance          | mean_abs_neighbor_delta |    0.125179    |
| learned_context_embedding | min_front_distance          | nn_value_spearman_corr  |    0.994982    |
| raw_feature               | min_front_distance          | mean_abs_neighbor_delta |    0.110346    |
| raw_feature               | min_front_distance          | nn_value_spearman_corr  |    0.992966    |
| pca_feature               | min_front_distance          | mean_abs_neighbor_delta |    0.116893    |
| pca_feature               | min_front_distance          | nn_value_spearman_corr  |    0.993097    |
| context_l2                | min_front_distance          | mean_abs_neighbor_delta |    0.21883     |
| context_l2                | min_front_distance          | nn_value_spearman_corr  |    0.891769    |
| random                    | min_front_distance          | mean_abs_neighbor_delta |    0.776332    |
| random                    | min_front_distance          | nn_value_spearman_corr  |    0.00131827  |
| learned_context_embedding | mean_rel_speed              | mean_abs_neighbor_delta |    0.113927    |
| learned_context_embedding | mean_rel_speed              | nn_value_spearman_corr  |    0.947829    |
| raw_feature               | mean_rel_speed              | mean_abs_neighbor_delta |    0.109941    |
| raw_feature               | mean_rel_speed              | nn_value_spearman_corr  |    0.95055     |
| pca_feature               | mean_rel_speed              | mean_abs_neighbor_delta |    0.108425    |
| pca_feature               | mean_rel_speed              | nn_value_spearman_corr  |    0.949736    |
| context_l2                | mean_rel_speed              | mean_abs_neighbor_delta |    0.169327    |
| context_l2                | mean_rel_speed              | nn_value_spearman_corr  |    0.857754    |
| random                    | mean_rel_speed              | mean_abs_neighbor_delta |    0.739739    |
| random                    | mean_rel_speed              | nn_value_spearman_corr  |   -0.000880523 |
| learned_context_embedding | p95_rel_speed               | mean_abs_neighbor_delta |    0.118875    |
| learned_context_embedding | p95_rel_speed               | nn_value_spearman_corr  |    0.99076     |
| raw_feature               | p95_rel_speed               | mean_abs_neighbor_delta |    0.1139      |
| raw_feature               | p95_rel_speed               | nn_value_spearman_corr  |    0.988745    |
| pca_feature               | p95_rel_speed               | mean_abs_neighbor_delta |    0.121692    |
| pca_feature               | p95_rel_speed               | nn_value_spearman_corr  |    0.988926    |
| context_l2                | p95_rel_speed               | mean_abs_neighbor_delta |    0.217883    |
| context_l2                | p95_rel_speed               | nn_value_spearman_corr  |    0.880431    |
| random                    | p95_rel_speed               | mean_abs_neighbor_delta |    0.80984     |
| random                    | p95_rel_speed               | nn_value_spearman_corr  |   -5.43297e-05 |
| learned_context_embedding | front_pressure_score        | mean_abs_neighbor_delta |    0.128815    |
| learned_context_embedding | front_pressure_score        | nn_value_spearman_corr  |    0.969296    |
| raw_feature               | front_pressure_score        | mean_abs_neighbor_delta |    0.103988    |
| raw_feature               | front_pressure_score        | nn_value_spearman_corr  |    0.970063    |
| pca_feature               | front_pressure_score        | mean_abs_neighbor_delta |    0.0988521   |
| pca_feature               | front_pressure_score        | nn_value_spearman_corr  |    0.968642    |
| context_l2                | front_pressure_score        | mean_abs_neighbor_delta |    0.288521    |
| context_l2                | front_pressure_score        | nn_value_spearman_corr  |    0.842809    |
| random                    | front_pressure_score        | mean_abs_neighbor_delta |    0.78966     |
| random                    | front_pressure_score        | nn_value_spearman_corr  |    0.00832238  |
| learned_context_embedding | rear_vehicle_pressure_proxy | mean_abs_neighbor_delta |    0.143766    |
| learned_context_embedding | rear_vehicle_pressure_proxy | nn_value_spearman_corr  |    0.944899    |
| raw_feature               | rear_vehicle_pressure_proxy | mean_abs_neighbor_delta |    0.147982    |
| raw_feature               | rear_vehicle_pressure_proxy | nn_value_spearman_corr  |    0.924913    |
| pca_feature               | rear_vehicle_pressure_proxy | mean_abs_neighbor_delta |    0.169784    |
| pca_feature               | rear_vehicle_pressure_proxy | nn_value_spearman_corr  |    0.92594     |
| context_l2                | rear_vehicle_pressure_proxy | mean_abs_neighbor_delta |    0.161782    |
| context_l2                | rear_vehicle_pressure_proxy | nn_value_spearman_corr  |    0.894339    |
| random                    | rear_vehicle_pressure_proxy | mean_abs_neighbor_delta |    0.813107    |
| random                    | rear_vehicle_pressure_proxy | nn_value_spearman_corr  |   -0.0111211   |
| learned_context_embedding | left_front_min_gap          | mean_abs_neighbor_delta |    0.160257    |
| learned_context_embedding | left_front_min_gap          | nn_value_spearman_corr  |    0.958431    |
| raw_feature               | left_front_min_gap          | mean_abs_neighbor_delta |    0.0947568   |
| raw_feature               | left_front_min_gap          | nn_value_spearman_corr  |    0.969654    |
| pca_feature               | left_front_min_gap          | mean_abs_neighbor_delta |    0.109386    |
| pca_feature               | left_front_min_gap          | nn_value_spearman_corr  |    0.971933    |
| context_l2                | left_front_min_gap          | mean_abs_neighbor_delta |    0.177938    |
| context_l2                | left_front_min_gap          | nn_value_spearman_corr  |    0.821741    |
| random                    | left_front_min_gap          | mean_abs_neighbor_delta |    0.5285      |
| random                    | left_front_min_gap          | nn_value_spearman_corr  |   -0.0105178   |
| learned_context_embedding | left_rear_min_gap           | mean_abs_neighbor_delta |    0.142223    |
| learned_context_embedding | left_rear_min_gap           | nn_value_spearman_corr  |    0.852525    |
| raw_feature               | left_rear_min_gap           | mean_abs_neighbor_delta |    0.094583    |
| raw_feature               | left_rear_min_gap           | nn_value_spearman_corr  |    0.837308    |
| pca_feature               | left_rear_min_gap           | mean_abs_neighbor_delta |    0.0948808   |
| pca_feature               | left_rear_min_gap           | nn_value_spearman_corr  |    0.840327    |
| context_l2                | left_rear_min_gap           | mean_abs_neighbor_delta |    0.155959    |
| context_l2                | left_rear_min_gap           | nn_value_spearman_corr  |    0.853241    |
| random                    | left_rear_min_gap           | mean_abs_neighbor_delta |    0.510998    |
| random                    | left_rear_min_gap           | nn_value_spearman_corr  |    0.00676952  |
| learned_context_embedding | right_front_min_gap         | mean_abs_neighbor_delta |    0.154185    |
| learned_context_embedding | right_front_min_gap         | nn_value_spearman_corr  |    0.957066    |
| raw_feature               | right_front_min_gap         | mean_abs_neighbor_delta |    0.0938751   |
| raw_feature               | right_front_min_gap         | nn_value_spearman_corr  |    0.976501    |
| pca_feature               | right_front_min_gap         | mean_abs_neighbor_delta |    0.104567    |
| pca_feature               | right_front_min_gap         | nn_value_spearman_corr  |    0.974467    |
| context_l2                | right_front_min_gap         | mean_abs_neighbor_delta |    0.185126    |
| context_l2                | right_front_min_gap         | nn_value_spearman_corr  |    0.82718     |
| random                    | right_front_min_gap         | mean_abs_neighbor_delta |    0.552436    |
| random                    | right_front_min_gap         | nn_value_spearman_corr  |   -0.000774352 |
| learned_context_embedding | right_rear_min_gap          | mean_abs_neighbor_delta |    0.132498    |
| learned_context_embedding | right_rear_min_gap          | nn_value_spearman_corr  |    0.94489     |
| raw_feature               | right_rear_min_gap          | mean_abs_neighbor_delta |    0.0991736   |
| raw_feature               | right_rear_min_gap          | nn_value_spearman_corr  |    0.862202    |
| pca_feature               | right_rear_min_gap          | mean_abs_neighbor_delta |    0.0997981   |
| pca_feature               | right_rear_min_gap          | nn_value_spearman_corr  |    0.864488    |
| context_l2                | right_rear_min_gap          | mean_abs_neighbor_delta |    0.167226    |
| context_l2                | right_rear_min_gap          | nn_value_spearman_corr  |    0.842219    |
| random                    | right_rear_min_gap          | mean_abs_neighbor_delta |    0.559607    |
| random                    | right_rear_min_gap          | nn_value_spearman_corr  |    0.00113464  |
| learned_context_embedding | left_gap_min                | mean_abs_neighbor_delta |    0.101503    |
| learned_context_embedding | left_gap_min                | nn_value_spearman_corr  |    0.877477    |
| raw_feature               | left_gap_min                | mean_abs_neighbor_delta |    0.0527465   |
| raw_feature               | left_gap_min                | nn_value_spearman_corr  |    0.948694    |
| pca_feature               | left_gap_min                | mean_abs_neighbor_delta |    0.0526802   |
| pca_feature               | left_gap_min                | nn_value_spearman_corr  |    0.947939    |
| context_l2                | left_gap_min                | mean_abs_neighbor_delta |    0.161688    |
| context_l2                | left_gap_min                | nn_value_spearman_corr  |    0.698271    |
| random                    | left_gap_min                | mean_abs_neighbor_delta |    0.406704    |
| random                    | left_gap_min                | nn_value_spearman_corr  |    0.000842788 |
| learned_context_embedding | right_gap_min               | mean_abs_neighbor_delta |    0.0782617   |
| learned_context_embedding | right_gap_min               | nn_value_spearman_corr  |    0.934354    |
| raw_feature               | right_gap_min               | mean_abs_neighbor_delta |    0.0504791   |
| raw_feature               | right_gap_min               | nn_value_spearman_corr  |    0.946897    |
| pca_feature               | right_gap_min               | mean_abs_neighbor_delta |    0.0523086   |
| pca_feature               | right_gap_min               | nn_value_spearman_corr  |    0.949351    |
| context_l2                | right_gap_min               | mean_abs_neighbor_delta |    0.157822    |
| context_l2                | right_gap_min               | nn_value_spearman_corr  |    0.665552    |
| random                    | right_gap_min               | mean_abs_neighbor_delta |    0.418309    |
| random                    | right_gap_min               | nn_value_spearman_corr  |    0.000105847 |
| learned_context_embedding | yielding_score_proxy        | mean_abs_neighbor_delta |    0.119867    |
| learned_context_embedding | yielding_score_proxy        | nn_value_spearman_corr  |    0.941711    |
| raw_feature               | yielding_score_proxy        | mean_abs_neighbor_delta |    0.108554    |
| raw_feature               | yielding_score_proxy        | nn_value_spearman_corr  |    0.939301    |
| pca_feature               | yielding_score_proxy        | mean_abs_neighbor_delta |    0.108767    |
| pca_feature               | yielding_score_proxy        | nn_value_spearman_corr  |    0.941141    |
| context_l2                | yielding_score_proxy        | mean_abs_neighbor_delta |    0.209161    |
| context_l2                | yielding_score_proxy        | nn_value_spearman_corr  |    0.841565    |
| random                    | yielding_score_proxy        | mean_abs_neighbor_delta |    0.687786    |
| random                    | yielding_score_proxy        | nn_value_spearman_corr  |    0.000313971 |
| learned_context_embedding | assertiveness_score_proxy   | mean_abs_neighbor_delta |    0.609751    |
| learned_context_embedding | assertiveness_score_proxy   | nn_value_spearman_corr  |    0.771892    |
| raw_feature               | assertiveness_score_proxy   | mean_abs_neighbor_delta |    0.333243    |
| raw_feature               | assertiveness_score_proxy   | nn_value_spearman_corr  |    0.929802    |
| pca_feature               | assertiveness_score_proxy   | mean_abs_neighbor_delta |    0.302577    |
| pca_feature               | assertiveness_score_proxy   | nn_value_spearman_corr  |    0.940282    |
| context_l2                | assertiveness_score_proxy   | mean_abs_neighbor_delta |    0.822687    |
| context_l2                | assertiveness_score_proxy   | nn_value_spearman_corr  |    0.522895    |
| random                    | assertiveness_score_proxy   | mean_abs_neighbor_delta |    1.1136      |
| random                    | assertiveness_score_proxy   | nn_value_spearman_corr  |   -0.0234114   |

## Category-wise Correlation Summary
| category              | representation            |   mean_spearman_corr |   median_spearman_corr |   number_of_features |
|:----------------------|:--------------------------|---------------------:|-----------------------:|---------------------:|
| longitudinal_comfort  | learned_context_embedding |          0.155422    |            0.154581    |                    4 |
| longitudinal_comfort  | raw_feature               |          0.172517    |            0.17079     |                    4 |
| longitudinal_comfort  | pca_feature               |          0.174168    |            0.172539    |                    4 |
| longitudinal_comfort  | context_l2                |         -0.023266    |           -0.0226167   |                    4 |
| longitudinal_comfort  | random                    |          0.0019978   |            0.00204684  |                    4 |
| following_interaction | learned_context_embedding |          0.496313    |            0.533108    |                    8 |
| following_interaction | raw_feature               |          0.469809    |            0.503374    |                    8 |
| following_interaction | pca_feature               |          0.468065    |            0.502415    |                    8 |
| following_interaction | context_l2                |          0.415333    |            0.401387    |                    8 |
| following_interaction | random                    |          0.000217646 |            0.0011749   |                    8 |
| lateral_lane_dynamics | learned_context_embedding |          0.256602    |            0.245745    |                   16 |
| lateral_lane_dynamics | raw_feature               |          0.251847    |            0.245436    |                   16 |
| lateral_lane_dynamics | pca_feature               |          0.251535    |            0.241702    |                   16 |
| lateral_lane_dynamics | context_l2                |          0.124057    |            0.220562    |                   16 |
| lateral_lane_dynamics | random                    |         -0.00063861  |            0.000749066 |                   16 |
| behavior_proxy        | learned_context_embedding |          0.316849    |            0.316849    |                    2 |
| behavior_proxy        | raw_feature               |          0.296727    |            0.296727    |                    2 |
| behavior_proxy        | pca_feature               |          0.298955    |            0.298955    |                    2 |
| behavior_proxy        | context_l2                |          0.202034    |            0.202034    |                    2 |
| behavior_proxy        | random                    |          0.00362594  |            0.00362594  |                    2 |

## Dynamic Evaluation Conclusions
- Global retrieval verdict: **lose**
- Longitudinal comfort verdict: **lose**
- Following interaction verdict: **win**
- Lateral/lane dynamics verdict: **win**
- Behavior proxy verdict: **win**
- Overall recommendation: **learned_context_embedding_best_tradeoff_so_far**

Global retrieval (hit@5): learned_context_embedding is lose vs best feature baseline. longitudinal comfort: learned_context_embedding is lose vs best feature baseline. following interaction: learned_context_embedding is win vs best feature baseline. lateral lane dynamics: learned_context_embedding is win vs best feature baseline. behavior proxy: learned_context_embedding is win vs best feature baseline. Feature-level learned wins observed on 16 targets (see learned_win_features.csv). Overall: learned_context_embedding is the best current trade-off learned representation, but not a full global retrieval win over handcrafted baselines.

## Warnings and Limitations
- None
