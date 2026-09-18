
# S2R Full-arm Reset Proof v1

Status: `FULL_ARM_RESET_CONTRACT = PASS`.

`build_fresh_arm` constructs a new runner, planner, simulation, controller, tracker, motion model, callback graph, recorder, random-state object, history-buffer owner, and artifact namespace for each arm. `assert_fresh_pair` fails closed if any mutable object identity or output root is shared. Future execution additionally requires one fresh operating-system process per arm, eliminating Python module/singleton carryover.

The official zero-run fixture constructed both arms from one real metadata-only scenario and proved independent objects for: `callbacks, controller, motion_model, planner, random_state, recorder, runner, simulation, tracker`. Recorder state and callback state were separately constructed. Both runners remained unstarted; runner, planner-compute, and simulation-advance counts were zero.
