
# S2R Full-arm Reset Contract v1

Status: `FULL_ARM_RESET_CONTRACT = BLOCKED` (prior evidence status: `BLOCKED`).

Code/schema inspection cannot prove that baseline and treatment arms receive independent fresh instances of every mutable component. `TwoStageController.reset()` clears current state but does not establish reconstruction of tracker and motion-model state; the official builder may reuse planner/callback objects; and no role-aware arm factory binds fresh planner, controller, internal buffers, callbacks, recorder, random/stateful components, and history for each arm.

Required future invariant:

```text
for each (scenario, arm):
  planner, controller, buffers, callbacks, recorder, RNG/stateful components = fresh factory products
  no mutable object identity is shared across arms
  reset/reconstruction evidence is written before outcome access
```

The deterministic preflight verified hashes and schemas only. It created no simulation object and ran no scientific rollout. A future fixture must instantiate the final production factory twice and assert disjoint mutable identities and clean initial state before this contract can pass.
