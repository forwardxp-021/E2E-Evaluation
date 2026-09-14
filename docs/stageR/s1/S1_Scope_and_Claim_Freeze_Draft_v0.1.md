# S1 Scope and Claim — Draft v0.1

Status: **S1_PROTOCOL_READY_FOR_OWNER_FREEZE**. Authoritative forward roadmap: repository-root `RBR-64_博士研究总体方案_v2.2_S1正式入口版.md`, read completely. Scientific Owner approved the scope/claims and S1.1 repairs; final protocol freeze remains an Owner action. No S2 authorization is issued.

1. **Level 1:** closed-loop behavior drift evaluation must separate intervention intent, realized mechanism, representation readability and distributional detection utility. Success at one layer does not imply success at the next. Stage6/7/7L, R0 and the HLC/TSB development chain support this contribution independently of a future positive RBR result.
2. **Conditional Level 2:** only if the preregistered Primary succeeds, a frozen RBR-BDD pipeline improves detection probability relative to the single development-informed H challenger at the same nominal FPR, sample budget and operating point, within the qualified TSB domain. This is a pipeline utility claim; it neither proves unique information nor establishes latent geometry as the cause.
3. **Application vision:** model-first discovery → feature diagnosis → human confirmation. Current Route A tests discovery/BDD plus mechanism validity. Unknown-family discovery, automatic naming, human-perceptual value and production release validation remain unvalidated.
4. **Residual-to-F0_project:** independently confirmed closed-loop behavioral structure difference remaining under the specified domain, Primary80 window and fixed project-predefined/routine handcrafted representation F0_project (= existing ego13). This is not residual to every possible handcrafted feature, not classifier independence, not general physical equivalence, and not a guarantee that F0_project is unable to detect the drift. The inherited F_match calipers originated as development balance tolerances; their pass does not establish low-order nuisance elimination.
5. **Use boundary:** unpaired Primary, if feasible, supports release emulation under this frozen public-data/nonreactive simulation setting. Paired prospective evidence supports controlled same-scenario sensitivity only. A/A calibration and independent-log uncertainty cannot be borrowed across these designs.
6. **Alarm semantics:** BEHAVIOR DISTRIBUTION CHANGE DETECTED. No automatic anomaly, degradation, safety superiority or release-blocking interpretation.
7. **HLC:** `CLOSED_BY_SCOPE_AFTER_ENGINEERING_NONCONVERGENCE`; V4 rejected; V5 and remaining runs unauthorized; scientific impossibility NOT_ESTABLISHED. V4's nominal morphology did not pass the reviewed ideal-tracking monotonic gate; terminal overshoot and residual lateral motion, endpoint and official safety failures remain. A rolling future reference endpoint is not the realized Primary80 endpoint. Different development identities do not make evolving scientific decisions independent architecture replications.
8. **Historical corrections, interpretation only:** R0 D3 = INCONCLUSIVE (full64 dilution was not established as sufficient explanation); Stage7L B seed3407 = Primary, old64/A/C = supporting/secondary; V4 offline feasibility ≠ closed-loop Primary80 feasibility. Preserve original files and PASS/FAIL states, including B1 infrastructure stop and B1.1 recovered scientific failure.

TSB remains `FROZEN_DEVELOPMENT_CANDIDATE_PENDING_FRESH_QUALIFICATION`. `TSB_LOW_ORDER_NUISANCE_ELIMINATION=NOT_ESTABLISHED`. `RBR_TRAINING=NOT_AUTHORIZED`. TSB-only is a disclosed post-development scope amendment, not success of the original combined HLC+TSB G_R2 program.

Evidence: R0 final report and `r0_final_scientific_execution_status_v1.1.json`; Stage7L-C/E protocols/reports; R2-B development report; R2-BK closure and scope disposition; B1/B1.1 reports. The design manifest binds exact source hashes.


## RBR_TRAINING_ARCHITECTURE_BOUNDARY — S1.1 Owner repair

Conceptually: trajectory/context X → temporal encoder / GRU → shared z64.
Training-time branches from z64 are (A) Human Semantic Head h_sem(z), predicting
pre-frozen human-defined driving semantics, and (B) generic representation /
temporal objective head(s). Conceptual loss:

`L = lambda_repr * L_repr + lambda_sem * L_sem + optional pre-frozen auxiliary terms`.

The Human Semantic Head MUST backpropagate through z64 into the encoder.
Its purpose is to retain known human-defined driving semantics in z64 while
generic representation objectives preserve additional trajectory structure.
This is not a specification of a purely self-supervised GRU plus post-hoc probes.

Semantic targets come only from U-authorized training data. No TSB treatment or
mechanism labels, no O labels, and no Q/D/E outcomes may supervise this head.
No handcrafted feature-distance or ego13 geometry alignment loss is permitted.
The semantic head supplies training-time supervision; it is not the Primary BDD
input. Primary BDD consumes frozen z64 directly, with the approved scaler/kernel
contract. Post-hoc semantic probes remain distinct Secondary diagnostics.
Encoder architecture/checkpoint selection remains U-only.

Repository precedent: `docs/stage6t_training_evaluation_protocol_zh.md` §4.1
already requires training-time auxiliary heads; Stage6T's auxiliary objectives
are part of the project lineage. This boundary is not a new TSB-driven mechanism
head. Exact future network layers, loss weights, SSL objective and any optional
auxiliary terms remain for the future pre-training protocol; S1 freezes none of
those implementation choices and authorizes no RBR training.
