# Method-code map

This is a source release, not a new experiment or a retroactive registration.
The `reference/` tree preserves the separation of the original experiments.
Path relocation does not change doses, readouts, cohorts, weighting, seeds,
intervention windows, or numerical acceptance criteria.

| Question | Reference implementation |
|---|---|
| Physical counterfactuals and native factor profiles | `reference/workspace/experiments/wam_control_state_v3/generate_group1_donors.py`, `run_group1_model.py` |
| Architecture-defined interface exchange | `reference/workspace/experiments/wam_control_state_v3/run_group2_{direct,joint,idm,imagewam}.py` |
| Strict current/future consumer control | `reference/workspace/joint_strict_consumption_audit_A_work/run_experiment_a.py` |
| Source-grouped attribution analysis | `reference/workspace/joint_strict_consumption_audit_A_work/analyze_experiment_a.py`, `s1_statistics.py` |
| Propagated-future restoration, not natural-donor future | `reference/workspace/joint_strict_consumption_audit_A_work/phase2_new_six_cells.py`, `phase2_new_statistics.py` |
| Explicit gripper execution-state restoration | `reference/workspace/joint_strict_consumption_audit_A_work/c_restore_v2.py` |
| Joint reduced-world computation | `reference/workspace/joint_strict_consumption_audit_A_work/run_c_early_stop.py`; RoboTwin `reference/workspace/FastWAM/experiments/robotwin_reduced_compute/early_stop.py` |
| Cross-fitted shared gain | `reference/workspace/joint_strict_consumption_audit_A_work/phase1_scaling_analysis.py` |
| RoboTwin interface × command-state factor | `reference/workspace/FastWAM/experiments/robotwin_interface_proprio/worker.py`, `analyze.py` |
| Strict interaction post-processing | `reference/workspace/FastWAM/results/strict_interaction_J/analyze.py` |
| Propagated-future post-processing | `reference/workspace/FastWAM/results/propagated_future_component/run_analysis.py` |

## Interpretation boundaries

- Interfaces are architecture-defined. Their discovery is not a contribution.
- Propagation-allowed node influence and strict consumer-edge effects differ.
  Node-minus-strict-current is not a natural indirect effect or an information share.
- With `delta10=A10-A00`, `delta01=A01-A00`, `delta11=A11-A00`,
  `J=delta11-delta10-delta01`; therefore `delta11=delta10+delta01+J`.
- Full-interface replay is a technical coverage/closure control, not by itself a
  mechanistic discovery. The propagated future P is not interchangeable with F_D.
- RoboTwin `obs['joint_action']['vector']` is the registered 14-dimensional
  joint-drive-target/gripper-command field, not measured articulation qpos.
- FK-based EEF target responses are predicted command readouts, not actual
  executed displacement. No predicted action is executed by figure reproduction.
- F1, F3, and F3-G are not interchangeable. Gripper readouts and translation
  projections do not share physical units.
- Repeated doses and action positions are not independent trajectories.
- Original confirmation, technical development, and post-hoc coverage cohorts
  retain their distinct identities. The original 34 and extended 42 RoboTwin
  sources are overlapping sets; the additional sources retain ancestry limits.
- Policy-computation reductions are not robot task wall-clock improvements,
  equivalence, safety guarantees, or evidence that future dynamics reasoning is preserved.

## Do not launch reference scripts blindly

Several original entry points load frozen manifests at import time, check exact
source hashes, or include historical acquisition/runner code. They are included
for method inspection, not enabled by the CPU-only quick start. Install the
appropriate upstream deployment stack and supply licensed assets and a new,
explicit execution configuration before any fresh experiment. Original frozen
hash gates are deliberately not relaxed to make an incomplete export run.
