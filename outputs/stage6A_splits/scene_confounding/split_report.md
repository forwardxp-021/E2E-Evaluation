# Stage6 A/B Split

- mode: scene_confounding_control
- eval_split: test
- n_A: 4917
- n_B: 4917
- criteria: ['quantile based score split, q_low=0.3, q_high=0.7, min_group_size=500', 'A=easy_scene_like, B=complex_scene_like']
- overlap_removed: 0
