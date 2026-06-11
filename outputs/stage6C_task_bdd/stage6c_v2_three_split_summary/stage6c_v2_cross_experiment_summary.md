# Stage 6C v2 Cross-Experiment Summary

## Interpretation Guide

- `negative`: sanity check; task-conditioned BDD should be low and non-systematic.
- `pseudo`: positive control; should show strong BDD in behavior-style tasks.
- `scene`: confounding diagnosis; BDD may appear, but the pattern should differ from pseudo.

## Reliability Tier

- Primary tasks: `task_following`, `task_lane_change`, `task_yield_conflict`, `task_hesitation`.
- Auxiliary proxy tasks: `task_cutin_response`, `task_queue_approach`, `task_lead_brake_response`, `task_overtake_opportunity`, `task_overtake_executed`.
- Do not interpret skipped tasks, especially `task_overtake_executed` when it is skipped due to sample size.

## BDD Pivot

| task_key | negative | pseudo | scene |
| --- | --- | --- | --- |
| task_cutin_response | 0.0009656507880644 | 0.2108897139347606 | 0.2195245581039945 |
| task_following | 0.0003066249999998 | 0.1363251687060112 | 0.1652892121274223 |
| task_hesitation | 0.0004623125 | 0.1762132275199099 | 0.1008047275032382 |
| task_lane_change | 0.0002862812499999 | 0.2241732280120623 | 0.1781293436179647 |
| task_lead_brake_response | 0.000394265304664 | 0.1353185325815542 | 0.1657980448729801 |
| task_overtake_opportunity | 0.0026821377798387 | 0.1345269470179251 |  |
| task_queue_approach | 0.0003046382289071 | 0.1462511575980023 | 0.1770139337198735 |
| task_yield_conflict | 0.0004122187499999 | 0.1382068066836814 | 0.1451844997219822 |

## Delta vs Negative

| task_key | bdd_negative | bdd_pseudo | bdd_scene | pseudo_minus_negative | scene_minus_negative | strongest_experiment |
| --- | --- | --- | --- | --- | --- | --- |
| task_cutin_response | 0.0009656507880644 | 0.2108897139347606 | 0.2195245581039945 | 0.2099240631466962 | 0.21855890731593008 | scene |
| task_following | 0.0003066249999998 | 0.1363251687060112 | 0.1652892121274223 | 0.13601854370601138 | 0.1649825871274225 | scene |
| task_hesitation | 0.0004623125 | 0.1762132275199099 | 0.1008047275032382 | 0.1757509150199099 | 0.10034241500323819 | pseudo |
| task_lane_change | 0.0002862812499999 | 0.2241732280120623 | 0.1781293436179647 | 0.2238869467620624 | 0.1778430623679648 | pseudo |
| task_lead_brake_response | 0.000394265304664 | 0.1353185325815542 | 0.1657980448729801 | 0.13492426727689022 | 0.16540377956831612 | scene |
| task_overtake_opportunity | 0.0026821377798387 | 0.1345269470179251 |  | 0.13184480923808642 |  | pseudo |
| task_queue_approach | 0.0003046382289071 | 0.1462511575980023 | 0.1770139337198735 | 0.1459465193690952 | 0.1767092954909664 | scene |
| task_yield_conflict | 0.0004122187499999 | 0.1382068066836814 | 0.1451844997219822 | 0.13779458793368152 | 0.1447722809719823 | scene |

## Warnings

- No summarizer warnings.
