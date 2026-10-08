#!/usr/bin/env python3
"""Generate preregistered Joint sensitivity/CKA features on new trajectories."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK))
import confirmatory_common as Q  # noqa: E402
import run_joint_baseline_diagnostics as D  # noqa: E402
from capture import make_capture  # noqa: E402

OUT = Q.JOINT / "new_trajectory_features_pre_label"
SPEC = Q.JOINT / "joint_augmented_baseline_frozen_spec.json"


def axes() -> dict[str, np.ndarray]:
    donors = pd.read_csv(Q.JOINT / "f3g_donor_registry.csv")
    result = {}
    for candidate, group in donors.groupby("candidate_id"):
        row = group.iloc[0]
        vector = np.asarray(json.loads(row.actual_goal_translation_m), float)
        dose_m = float(row.signed_dose_cm) / 100.0
        axis = -vector / dose_m
        axis /= np.linalg.norm(axis)
        result[str(candidate)] = axis
    return result


def worker(gpu: int, shard: int, shards: int) -> None:
    if not SPEC.exists():
        raise RuntimeError("augmented baseline parameters must be frozen before new features")
    rows = Q.registry("joint")
    own = [row for index, row in enumerate(rows) if index % shards == shard]
    amplitude = float(json.loads((D.OUT / "ordinary_sensitivity_amplitude_freeze.json").read_text())["selected_amplitude_rms"])
    axis_map = axes()
    D.obs_path = lambda row: Path(str(row["observation_path"]))
    D.seed = lambda row: int(row["policy_seed"])
    runner = make_capture("joint", gpu)
    sensitivity = []
    cka = []
    costs = []
    for index, row in enumerate(own, 1):
        a, b, c = D.one(runner, row, (amplitude,), axis_map[str(row["candidate_id"])])
        sensitivity += a
        cka += b
        costs.append(c)
        print(json.dumps({"stage": "NEW_FEATURES_PRE_LABEL", "shard": shard, "state": index,
                          "total": len(own), "candidate": row["candidate_id"]}), flush=True)
    D.write(OUT / f"shards/shard_{shard:02d}_sensitivity.csv", sensitivity)
    D.write(OUT / f"shards/shard_{shard:02d}_cka.csv", cka)
    D.write(OUT / f"shards/shard_{shard:02d}_cost.csv", costs)
    D.dump(OUT / f"shards/shard_{shard:02d}.json", {
        "status": "COMPLETE", "shard": shard, "shards": shards, "logical_gpu": gpu,
        "states": len(own), "baseline_spec_sha256": Q.sha(SPEC),
        "no_K8_K5_donor_damage_label_generated": True, "code_sha256": Q.sha(Path(__file__)),
    })


def finalize() -> None:
    manifests = sorted((OUT / "shards").glob("shard_*.json"))
    if len(manifests) != 3:
        raise RuntimeError(f"expected 3 shard manifests, found {len(manifests)}")
    sensitivity = pd.concat([pd.read_csv(path) for path in sorted((OUT / "shards").glob("*_sensitivity.csv"))], ignore_index=True)
    cka = pd.concat([pd.read_csv(path) for path in sorted((OUT / "shards").glob("*_cka.csv"))], ignore_index=True)
    costs = pd.concat([pd.read_csv(path) for path in sorted((OUT / "shards").glob("*_cost.csv"))], ignore_index=True)
    sensitivity.to_csv(OUT / "joint_new_ordinary_sensitivity_raw.csv", index=False)
    cka.to_csv(OUT / "joint_new_future_kv_linear_cka_raw.csv", index=False)
    costs.to_csv(OUT / "joint_new_diagnostic_cost.csv", index=False)
    sens = sensitivity.groupby(["candidate_id", "task_id", "trajectory_id", "policy_call", "split"], as_index=False).agg(
        ordinary_sensitivity_radial_l2=("radial_l2", "mean"),
        ordinary_sensitivity_translation_l2=("translation_l2", "mean"),
        ordinary_sensitivity_rotation_l2=("rotation_l2", "mean"),
        ordinary_sensitivity_gripper_l2=("gripper_l2", "mean"),
    )
    agg = cka[(cka.layer.astype(str) == "AGGREGATE") & (cka.component == "MEAN_LAYER_COMPONENT")]
    features = sens.merge(agg[["candidate_id", "K", "linear_CKA"]].rename(columns={"linear_CKA": "future_kv_linear_cka"}),
                          on="candidate_id", validate="one_to_many")
    features.to_csv(OUT / "joint_new_state_baseline_features_partial.csv", index=False)
    report = {
        "status": "JOINT_NEW_TRAJECTORY_PARTIAL_FEATURES_FROZEN_PRE_DAMAGE_LABEL",
        "created_unix": time.time(), "states": int(sens.candidate_id.nunique()),
        "state_K_rows": int(len(features)), "expected_states": 50,
        "baseline_spec_sha256": Q.sha(SPEC),
        "sensitivity_sha256": Q.sha(OUT / "joint_new_ordinary_sensitivity_raw.csv"),
        "cka_sha256": Q.sha(OUT / "joint_new_future_kv_linear_cka_raw.csv"),
        "features_sha256": Q.sha(OUT / "joint_new_state_baseline_features_partial.csv"),
        "cost_sha256": Q.sha(OUT / "joint_new_diagnostic_cost.csv"),
        "no_K8_K5_donor_damage_label_generated": True,
        "code_sha256": Q.sha(Path(__file__)),
    }
    D.dump(OUT / "joint_new_feature_manifest.json", report)
    print(json.dumps(report))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["worker", "finalize"], required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=3)
    args = parser.parse_args()
    if args.mode == "worker":
        worker(args.gpu, args.shard, args.shards)
    else:
        finalize()


if __name__ == "__main__":
    main()
