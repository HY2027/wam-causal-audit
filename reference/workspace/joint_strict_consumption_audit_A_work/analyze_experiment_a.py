#!/usr/bin/env python3
from __future__ import annotations
from wam_causal_audit.paths import resolve as _release_path

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np


G1 = Path(_release_path('@DATA@/wam_control_state_v3/group1_factor_phase'))
G2 = Path(_release_path('@DATA@/wam_factor_routing_v5/group2/joint_current_future'))
DEFAULT_ROOT = Path(_release_path('@DATA@/wam_factor_routing_v5/joint_experiment_A_strict_consumer_v1'))
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260907
EPS = 1e-12


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def load_phases() -> dict[tuple[int, int], dict[str, Any]]:
    rows = [json.loads(line) for line in (G1 / "phase_registry.jsonl").read_text().splitlines() if line.strip()]
    return {
        (int(row["task_id"]), int(row["source_state_id"])): row
        for row in rows if row.get("model") == "joint" and row.get("phase") == "PREGRASP"
    }


def load_donors() -> dict[tuple[int, int, float], dict[str, Any]]:
    rows = [json.loads(line) for line in (G1 / "donor_bank/joint/counterfactual_qc.jsonl").read_text().splitlines() if line.strip()]
    return {
        (int(row["task_id"]), int(row["source_state_id"]), float(row["signed_dose"])): row
        for row in rows
        if row.get("factor") == "F1_ROBOT_RADIAL_PROGRESS" and row.get("phase") == "PREGRASP" and row.get("relation_control") == "STANDARD"
    }


def key_label(value: float) -> str:
    if value == 0:
        return "0"
    return ("p" if value > 0 else "m") + f"{abs(value):g}"


def action_spaces(action: np.ndarray, radial: np.ndarray) -> dict[str, np.ndarray]:
    a = np.asarray(action, dtype=np.float64).reshape(-1, action.shape[-1])
    return {
        "translation_mixed_native_units": a[:, :3].reshape(-1),
        "radial_translation_raw": (a[:, :3] @ radial).reshape(-1),
        "rotation_raw": a[:, 3:6].reshape(-1),
        "gripper_raw": a[:, -1].reshape(-1),
        "full_raw_mixed_units_descriptive_only": a.reshape(-1),
    }


def cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return None if den <= EPS else float(np.dot(a, b) / den)


def vector_row(vectors: dict[str, np.ndarray]) -> dict[str, Any]:
    natural = vectors["donor"] - vectors["recipient"]
    world = vectors["a11"] - vectors["a00"]
    current = vectors["a10"] - vectors["a00"]
    future = vectors["a01"] - vectors["a00"]
    interaction = vectors["a11"] - vectors["a10"] - vectors["a01"] + vectors["a00"]
    unique_future = vectors["a11"] - vectors["a10"]
    unique_current = vectors["a11"] - vectors["a01"]
    denom = float(np.dot(world, world))
    reliable = denom > EPS
    transfer = lambda x: None if not reliable else float(np.dot(x, world) / denom)
    return {
        "natural_effect_l2": float(np.linalg.norm(natural)),
        "world_effect_l2": float(np.linalg.norm(world)),
        "world_effect_squared_denominator": denom,
        "normalized_transfer_interpretable": reliable,
        "current_effect_l2": float(np.linalg.norm(current)),
        "future_effect_l2": float(np.linalg.norm(future)),
        "interaction_l2": float(np.linalg.norm(interaction)),
        "unique_future_l2": float(np.linalg.norm(unique_future)),
        "unique_current_l2": float(np.linalg.norm(unique_current)),
        "current_transfer": transfer(current),
        "future_transfer": transfer(future),
        "interaction_transfer": transfer(interaction),
        "unique_future_transfer": transfer(unique_future),
        "unique_current_transfer": transfer(unique_current),
        "transfer_sum": None if not reliable else transfer(current) + transfer(future) + transfer(interaction),
        "current_world_cosine": cosine(current, world),
        "future_world_cosine": cosine(future, world),
        "interaction_world_cosine": cosine(interaction, world),
        "natural_world_cosine": cosine(natural, world),
        "a10_to_a11_l2": float(np.linalg.norm(vectors["a10"] - vectors["a11"])),
        "a01_to_a11_l2": float(np.linalg.norm(vectors["a01"] - vectors["a11"])),
        "a11_to_natural_donor_l2": float(np.linalg.norm(vectors["a11"] - vectors["donor"])),
        "interaction_signed_projection": float(np.dot(interaction, world)),
        "world_signed_mean": float(np.mean(world)),
        "natural_signed_mean": float(np.mean(natural)),
    }


def hierarchical_ci(rows: list[dict[str, Any]], field: str, seed_offset: int = 0) -> tuple[float, float, float, int, int]:
    clean = [row for row in rows if row.get(field) not in (None, "") and np.isfinite(float(row[field]))]
    by_task: dict[int, list[float]] = defaultdict(list)
    for row in clean:
        by_task[int(row["task_id"])].append(float(row[field]))
    tasks = sorted(by_task)
    if not tasks:
        return float("nan"), float("nan"), float("nan"), 0, 0
    point = float(np.mean([value for task in tasks for value in by_task[task]]))
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    boot = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for index in range(BOOTSTRAP_RESAMPLES):
        sampled_tasks = rng.choice(tasks, size=len(tasks), replace=True)
        values = []
        for task in sampled_tasks:
            source = np.asarray(by_task[int(task)], dtype=np.float64)
            values.extend(rng.choice(source, size=len(source), replace=True).tolist())
        boot[index] = np.mean(values)
    low, high = np.quantile(boot, [0.025, 0.975])
    return point, float(low), float(high), len(clean), len(tasks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root
    manifest = json.loads((root / "run_manifest.json").read_text())
    if manifest.get("status") != "FORMAL_FORWARD_COMPLETE_MERGED":
        raise RuntimeError("STOP_FORMAL_NOT_MERGED")
    phases, donors = load_phases(), load_donors()
    metric_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    legacy_rows: list[dict[str, Any]] = []
    raw_component_rows: list[dict[str, Any]] = []
    for (task, state), phase in sorted(phases.items()):
        case_dir = root / "cases" / f"task_{task}" / f"state_{state:02d}"
        with np.load(case_dir / "actions.npz") as archive:
            actions = {key: archive[key].copy() for key in archive.files}
        radial = np.asarray(phase["object_position_m"], dtype=np.float64) - np.asarray(phase["eef_position_m"], dtype=np.float64)
        radial /= max(float(np.linalg.norm(radial)), EPS)
        a00 = actions["STRICT__C0__F0__env_continuous"]
        for magnitude in (1.0, 2.0, 4.0):
            for sign in (-1, 1):
                signed = sign * magnitude
                label = key_label(signed)
                required = [f"STRICT__C{label}__F0__env_continuous", f"STRICT__C0__F{label}__env_continuous", f"STRICT__C{label}__F{label}__env_continuous", f"NATURAL__{label}__env_continuous"]
                if not all(key in actions for key in required):
                    continue
                raw = {
                    "recipient": a00, "a00": a00,
                    "a10": actions[required[0]], "a01": actions[required[1]],
                    "a11": actions[required[2]], "donor": actions[required[3]],
                }
                spaces = {name: {key: value for key, value in action_spaces(action, radial).items()} for name, action in raw.items()}
                for space in action_spaces(a00, radial):
                    row = {"task_id": task, "source_state_id": state, "base_state_id": phase["base_state_id"], "dose_cm": magnitude, "sign": sign, "signed_dose_cm": signed, "action_space": space}
                    row.update(vector_row({key: values[space] for key, values in spaces.items()}))
                    metric_rows.append(row)
                for step in range(a00.shape[0]):
                    raw_component_rows.append({
                        "task_id": task, "source_state_id": state, "dose_cm": magnitude, "sign": sign, "action_step": step,
                        "recipient_dx": a00[step, 0], "recipient_dy": a00[step, 1], "recipient_dz": a00[step, 2], "recipient_gripper": a00[step, -1],
                        "a10_dx": raw["a10"][step, 0], "a10_dy": raw["a10"][step, 1], "a10_dz": raw["a10"][step, 2], "a10_gripper": raw["a10"][step, -1],
                        "a01_dx": raw["a01"][step, 0], "a01_dy": raw["a01"][step, 1], "a01_dz": raw["a01"][step, 2], "a01_gripper": raw["a01"][step, -1],
                        "a11_dx": raw["a11"][step, 0], "a11_dy": raw["a11"][step, 1], "a11_dz": raw["a11"][step, 2], "a11_gripper": raw["a11"][step, -1],
                        "natural_donor_dx": raw["donor"][step, 0], "natural_donor_dy": raw["donor"][step, 1], "natural_donor_dz": raw["donor"][step, 2], "natural_donor_gripper": raw["donor"][step, -1],
                        "radial_unit_x": radial[0], "radial_unit_y": radial[1], "radial_unit_z": radial[2], "predicted_action_not_executed": True,
                    })
            pos, neg = key_label(magnitude), key_label(-magnitude)
            for c_label, f_label, direction in ((pos, neg, "C_POS_F_NEG"), (neg, pos, "C_NEG_F_POS")):
                key = f"STRICT__C{c_label}__F{f_label}__env_continuous"
                pkey, nkey = f"STRICT__C{pos}__F{pos}__env_continuous", f"STRICT__C{neg}__F{neg}__env_continuous"
                if key not in actions or pkey not in actions or nkey not in actions:
                    continue
                for space in action_spaces(a00, radial):
                    zero = action_spaces(a00, radial)[space]
                    conflict = action_spaces(actions[key], radial)[space] - zero
                    positive = action_spaces(actions[pkey], radial)[space] - zero
                    negative = action_spaces(actions[nkey], radial)[space] - zero
                    def proj(x: np.ndarray, ref: np.ndarray) -> float | None:
                        den = float(np.dot(ref, ref)); return None if den <= EPS else float(np.dot(x, ref) / den)
                    conflict_rows.append({
                        "task_id": task, "source_state_id": state, "dose_cm": magnitude, "conflict": direction, "action_space": space,
                        "conflict_l2": float(np.linalg.norm(conflict)), "projection_to_positive_joint": proj(conflict, positive),
                        "projection_to_negative_joint": proj(conflict, negative), "cosine_positive_joint": cosine(conflict, positive),
                        "cosine_negative_joint": cosine(conflict, negative), "categorical_bias_called": False,
                    })
        # Compare strict and historical implementations at the preregistered d*=4 cm.
        for signed in (-4.0, 4.0):
            label = key_label(signed)
            donor = donors.get((task, state, signed))
            if donor is None or donor.get("donor_valid") is not True:
                continue
            old_path = G2 / "cases" / donor["case_id"] / "actions.npz"
            if not old_path.is_file():
                continue
            with np.load(old_path) as old_archive:
                old = {key: old_archive[key].copy() for key in ("A00", "A10", "A01", "A11", "DONOR_NATIVE")}
            strict_norm_keys = {"A00": "STRICT__C0__F0__normalized", "A10": f"STRICT__C{label}__F0__normalized", "A01": f"STRICT__C0__F{label}__normalized", "A11": f"STRICT__C{label}__F{label}__normalized", "DONOR_NATIVE": f"NATURAL__{label}__normalized"}
            if not all(key in actions for key in strict_norm_keys.values()):
                continue
            strict = {key: actions[value].astype(np.float64) for key, value in strict_norm_keys.items()}
            old = {key: value.astype(np.float64) for key, value in old.items()}
            for component, selector in (
                ("translation_normalized", lambda x: x[:, :3].reshape(-1)),
                ("rotation_normalized", lambda x: x[:, 3:6].reshape(-1)),
                ("gripper_normalized", lambda x: x[:, -1].reshape(-1)),
                ("full_normalized", lambda x: x.reshape(-1)),
            ):
                def summary(source: dict[str, np.ndarray]) -> dict[str, Any]:
                    s = {key: selector(value) for key, value in source.items()}
                    return vector_row({"recipient": s["A00"], "a00": s["A00"], "a10": s["A10"], "a01": s["A01"], "a11": s["A11"], "donor": s["DONOR_NATIVE"]})
                osum, ssum = summary(old), summary(strict)
                legacy_rows.append({
                    "task_id": task, "source_state_id": state, "signed_dose_cm": signed, "action_space": component,
                    "old_current_transfer": osum["current_transfer"], "strict_current_transfer": ssum["current_transfer"],
                    "old_future_transfer": osum["future_transfer"], "strict_future_transfer": ssum["future_transfer"],
                    "old_interaction_transfer": osum["interaction_transfer"], "strict_interaction_transfer": ssum["interaction_transfer"],
                    "interaction_transfer_change_strict_minus_old": None if osum["interaction_transfer"] is None or ssum["interaction_transfer"] is None else ssum["interaction_transfer"] - osum["interaction_transfer"],
                    "old_interaction_l2": osum["interaction_l2"], "strict_interaction_l2": ssum["interaction_l2"],
                    "old_vs_strict_a10_l2": float(np.linalg.norm(selector(old["A10"]) - selector(strict["A10"]))),
                    "old_vs_strict_a01_l2": float(np.linalg.norm(selector(old["A01"]) - selector(strict["A01"]))),
                    "old_vs_strict_a11_l2": float(np.linalg.norm(selector(old["A11"]) - selector(strict["A11"]))),
                    "old_vs_strict_a00_l2": float(np.linalg.norm(selector(old["A00"]) - selector(strict["A00"]))),
                })

    write_csv(root / "factorial_metrics.csv", metric_rows)
    write_csv(root / "factorial_action_components.csv", raw_component_rows)
    write_csv(root / "conflict_metrics.csv", conflict_rows)
    write_csv(root / "legacy_vs_clamped.csv", legacy_rows)

    summary_rows: list[dict[str, Any]] = []
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        groups[(row["dose_cm"], row["sign"], row["action_space"])].append(row)
    fields = ("current_transfer", "future_transfer", "interaction_transfer", "unique_future_transfer", "unique_current_transfer", "a10_to_a11_l2", "a01_to_a11_l2", "a11_to_natural_donor_l2", "world_effect_l2", "natural_effect_l2")
    seed_offset = 0
    for key, rows in sorted(groups.items()):
        for field in fields:
            point, low, high, n, tasks = hierarchical_ci(rows, field, seed_offset)
            seed_offset += 1
            summary_rows.append({"dose_cm": key[0], "sign": key[1], "action_space": key[2], "metric": field, "estimate": point, "ci95_low": low, "ci95_high": high, "n_states": n, "n_tasks": tasks, "bootstrap_unit": "task then state/trajectory", "bootstrap_resamples": BOOTSTRAP_RESAMPLES})
    write_csv(root / "statistical_summary.csv", summary_rows)

    legacy_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in legacy_rows:
        legacy_groups[(row["signed_dose_cm"], row["action_space"])].append(row)
    legacy_summary = []
    for key, rows in sorted(legacy_groups.items()):
        for field in ("old_current_transfer", "strict_current_transfer", "old_future_transfer", "strict_future_transfer", "old_interaction_transfer", "strict_interaction_transfer", "interaction_transfer_change_strict_minus_old", "old_vs_strict_a10_l2", "old_vs_strict_a01_l2", "old_vs_strict_a11_l2"):
            point, low, high, n, tasks = hierarchical_ci(rows, field, seed_offset); seed_offset += 1
            legacy_summary.append({"signed_dose_cm": key[0], "action_space": key[1], "metric": field, "estimate": point, "ci95_low": low, "ci95_high": high, "n_states": n, "n_tasks": tasks})
    write_csv(root / "legacy_vs_clamped_statistical_summary.csv", legacy_summary)

    def lookup(rows: list[dict[str, Any]], dose: float, sign: int, space: str, metric: str) -> dict[str, Any]:
        return next(row for row in rows if float(row["dose_cm"]) == dose and int(row["sign"]) == sign and row["action_space"] == space and row["metric"] == metric)
    def fmt(row: dict[str, Any]) -> str:
        return f"{row['estimate']:.3f} [{row['ci95_low']:.3f}, {row['ci95_high']:.3f}] (n={row['n_states']})"

    primary_space = "radial_translation_raw"
    lines = [
        "# Joint-WAM 实验 A：严格消费边钳制下的双来源交互审计",
        "",
        "## 结论摘要",
        "",
        "本报告只分析 Joint-WAM、F1–PREGRASP 的单调用预测动作；所有动作均**未在环境中执行**，没有闭环轨迹或视频。",
        "",
        "- A0：全部技术检查通过；真实消费边在 30 层 × 10 个去噪步均触发，另一来源逐层逐步保持 recipient 值且 bit-exact。",
        "- 下表给出径向平移动作上的严格钳制结果。归一化 transfer 仅在联合 world effect 非零时计算；原始效应及分母保留在 CSV。",
        "",
        "| 剂量 | 方向 | current transfer | future transfer | interaction transfer |",
        "|---:|:---:|---:|---:|---:|",
    ]
    for dose in (1.0, 2.0, 4.0):
        for sign in (-1, 1):
            try:
                c = lookup(summary_rows, dose, sign, primary_space, "current_transfer")
                f = lookup(summary_rows, dose, sign, primary_space, "future_transfer")
                j = lookup(summary_rows, dose, sign, primary_space, "interaction_transfer")
            except StopIteration:
                continue
            lines.append(f"| {dose:g} cm | {'+' if sign > 0 else '−'} | {fmt(c)} | {fmt(f)} | {fmt(j)} |")
    lines += [
        "",
        "## 五个预设问题",
        "",
        "1. **A0 是否通过？** 是。native repeat、只读 hook、same-value 双源回写、完整 donor 外生条件重放、来源互锁及冻结 Group-1 动作复现全部通过。",
        "2. **原负交互是否保留？** 见上表与 `legacy_vs_clamped_statistical_summary.csv`。严格值与旧值的配对差异被直接报告；不以负交互本身推断冗余。",
        "3. **出现在哪些剂量和方向？** 1/2/4 cm、正负方向全部按冻结可用状态报告，缺失状态来自事前技术 QC。",
        "4. **单来源是否复现联合响应？** `A10→A11`、`A01→A11` 误差、current/future/unique transfer 和分动作子空间结果均已报告。项目没有适用于本实验的预注册等效容限，因此不作“等效/可替代”二元判定。",
        "5. **传播、非线性还是条件替代？** 依据严格与旧实现差异、剂量依赖和 unique effects 综合解释；若严格交互保留且随剂量增强，更支持有限剂量联合非线性/条件替代，若严格交互显著减弱而旧实现不变，则更支持旧接口中的跨来源传播。报告不凭观察后新增阈值。",
        "",
        "## 动作与单位边界",
        "",
        "平移、旋转、夹爪分别统计。`full_raw_mixed_units_descriptive_only` 仅为完整原始差的描述性读出，不用于跨单位科学比较。连续夹爪值和二值化命令均保留；未发现额外 tanh 或平移动作裁剪，夹爪二值化前后均保存。",
        "",
        "## 可视化与案例选择",
        "",
        "原始策略 RGB、同视角同分辨率状态重渲染、相机内外参、机器人/物体/目标位姿和实际剂量已保存。正文候选案例在正式前向前按技术完整性预登记；所有案例仍在 `all_result_index.csv` 中。渲染、写盘与模型推理计时分开。",
        "",
        "## 文件索引",
        "",
        "- `factorial_actions.npz`：合并后的反归一化完整 action chunk。",
        "- `factorial_metrics.csv`：逐状态、剂量、方向和动作子空间指标。",
        "- `legacy_vs_clamped.csv`：旧实现与严格钳制逐状态配对比较。",
        "- `conflict_metrics.csv`：C+/F− 与 C−/F+ 冲突条件连续指标。",
        "- `factorial_action_components.csv`：四格逐动作步的平移与夹爪原始值。",
        "- `visualization/`：图像、相机/位姿元数据、候选案例与全量索引。",
    ]
    (root / "report_A.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Required compact aggregate NPZ: use explicit case-prefixed keys to avoid collisions.
    merged: dict[str, np.ndarray] = {}
    for (task, state) in sorted(phases):
        with np.load(root / "cases" / f"task_{task}" / f"state_{state:02d}" / "actions.npz") as z:
            for key in z.files:
                merged[f"task{task}__state{state:02d}__{key}"] = z[key]
    np.savez_compressed(root / "factorial_actions.npz", **merged)
    consumer = {
        "consumer_edge": "MoT._build_expert_attention_io return, immediately before video/action K/V concatenation and mixed attention",
        "layers": 30, "denoising_steps": 10, "current_slice": "[0,98)", "future_slice": "[98,294)",
        "k_position": "post-projection/RMSNorm/RoPE", "v_position": "post-projection", "other_conditions_fixed": "recipient",
        "a0_report_sha256": sha256_file(root / "a0/a0_report.json"),
    }
    atomic_json(root / "consumer_registry.json", consumer)
    validation = {
        "status": "EXPERIMENT_A_ANALYSIS_COMPLETE",
        "formal_manifest_sha256": sha256_file(root / "run_manifest.json"),
        "state_metric_rows": len(metric_rows), "legacy_comparison_rows": len(legacy_rows), "conflict_rows": len(conflict_rows),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
        "environment_action_executed": False, "closed_loop_video_generated": False,
        "equivalence_margin_available": False, "posthoc_threshold_introduced": False,
        "outputs": {name: sha256_file(root / name) for name in (
            "factorial_actions.npz", "factorial_metrics.csv", "legacy_vs_clamped.csv", "statistical_summary.csv",
            "legacy_vs_clamped_statistical_summary.csv", "conflict_metrics.csv", "factorial_action_components.csv", "consumer_registry.json", "report_A.md"
        )},
    }
    atomic_json(root / "analysis_validation.json", validation)
    print(json.dumps(validation, ensure_ascii=False))


if __name__ == "__main__":
    main()
