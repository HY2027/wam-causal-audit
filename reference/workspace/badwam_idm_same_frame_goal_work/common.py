from __future__ import annotations
from wam_causal_audit.paths import resolve as _release_path

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

REPO = Path(_release_path('@DATA@/BadWAM'))
AUDIT_WORK = Path(_release_path('@WORKSPACE@/badwam_idm_causal_audit_work'))
for extra in (
    REPO,
    REPO / "experiments/libero",
    REPO / "experiments/first_grasp_lock",
    AUDIT_WORK,
    Path(_release_path('@WORKSPACE@/LIBERO')),
    Path(_release_path('@WORKSPACE@/evaluation/d2_ghost_pilot')),
):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import run_first_grasp_lock as fgl  # noqa: E402
from imagination_intervention import ImaginationIntervention, tensor_sha256  # noqa: E402
from run_smoke import action_from_latent, action_noise, generate_video_latent, prepare  # noqa: E402


TASKS = (0, 1, 2, 3, 4)
TARGETS = {
    0: "alphabet_soup_1",
    1: "cream_cheese_1",
    2: "salad_dressing_1",
    3: "bbq_sauce_1",
    4: "ketchup_1",
}
DONOR_TASK = {0: 1, 1: 0, 2: 4, 3: 2, 4: 3}
ALT_INSTRUCTION = {
    0: "pick up the cream cheese and place it in the basket",
    1: "pick up the alphabet soup and place it in the basket",
    2: "pick up the ketchup and place it in the basket",
    3: "pick up the salad dressing and place it in the basket",
    4: "pick up the bbq sauce and place it in the basket",
}
SCENE_OBJECTS = {
    0: ("alphabet_soup_1", "salad_dressing_1", "cream_cheese_1", "milk_1", "tomato_sauce_1", "butter_1"),
    1: ("cream_cheese_1", "alphabet_soup_1", "milk_1", "tomato_sauce_1", "butter_1", "orange_juice_1"),
    2: ("salad_dressing_1", "ketchup_1", "alphabet_soup_1", "cream_cheese_1", "milk_1", "tomato_sauce_1"),
    3: ("bbq_sauce_1", "chocolate_pudding_1", "ketchup_1", "salad_dressing_1", "alphabet_soup_1", "cream_cheese_1"),
    4: ("ketchup_1", "bbq_sauce_1", "salad_dressing_1", "alphabet_soup_1", "cream_cheese_1", "milk_1"),
}
SEED_BASE = 410_000
SETTLE_STEPS = 30
MAX_CONTROL_STEPS = 500
MAX_INTERVENTION_CALLS = 12
COMMIT_RADIUS_M = 0.15


def seed_for(task: int, pair: int, call: int = 0) -> int:
    return SEED_BASE + int(task) * 1000 + int(pair) * 20 + int(call)


def make_env(task_id: int, state_index: int) -> tuple[Any, Any, dict[str, Any]]:
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    task = suite.get_task(int(task_id))
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=fgl.LIBERO_ENV_RESOLUTION,
        camera_widths=fgl.LIBERO_ENV_RESOLUTION,
    )
    env.seed(fgl.ENVIRONMENT_SEED)
    env.reset()
    obs = dict(env.set_init_state(suite.get_task_init_states(int(task_id))[int(state_index)]))
    for _ in range(SETTLE_STEPS):
        obs, _, _, _ = env.step(fgl.get_libero_dummy_action())
        obs = dict(obs)
    return env, task, obs


def body_name(env: Any, entity: str) -> str:
    sim = fgl.get_sim(env)
    names = [str(sim.model.body_id2name(index) or "") for index in range(int(sim.model.nbody))]
    for candidate in (f"{entity}_main", entity):
        if candidate in names:
            return candidate
    match = next((name for name in names if entity.lower() in name.lower()), None)
    if match is None:
        raise KeyError(f"Cannot resolve body for {entity}")
    return match


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def normalized_to_env(action: torch.Tensor, processor: Any) -> np.ndarray:
    processed = fgl._denormalize_action(action, processor)[0]
    processed[..., -1] = processed[..., -1] * 2 - 1
    processed = fgl.invert_gripper_action(processed)
    processed[..., -1] = np.sign(processed[..., -1])
    return np.asarray(processed, dtype=np.float32)


def transfer_metrics(source: torch.Tensor, donor: torch.Tensor, patched: torch.Tensor) -> dict[str, float]:
    def one(a: torch.Tensor, b: torch.Tensor, p: torch.Tensor) -> tuple[float, float, float]:
        dp = (p - a).reshape(-1).double()
        dg = (b - a).reshape(-1).double()
        pn = float(torch.linalg.vector_norm(dp))
        gn = float(torch.linalg.vector_norm(dg))
        dot = float(torch.dot(dp, dg))
        return pn, dot / (gn * gn + 1e-12), 0.0 if pn == 0 or gn == 0 else dot / (pn * gn)

    f_l2, f_transfer, f_cosine = one(source[0], donor[0], patched[0])
    a_l2, a_transfer, a_cosine = one(source, donor, patched)
    e_l2, e_transfer, e_cosine = one(source[0, :3], donor[0, :3], patched[0, :3])
    return {
        "first_step_action_l2": f_l2,
        "first_step_transfer": f_transfer,
        "first_step_cosine": f_cosine,
        "action_l2": a_l2,
        "transfer": a_transfer,
        "cosine": a_cosine,
        "eef_action_l2": e_l2,
        "eef_transfer": e_transfer,
        "eef_cosine": e_cosine,
    }


def per_dimension_change(source: torch.Tensor, patched: torch.Tensor) -> dict[str, float]:
    delta = (patched - source).abs().mean(dim=0)
    return {f"action_dim_{index}_mean_abs_change": float(value) for index, value in enumerate(delta)}


def create_policy(gpu_id: int) -> fgl.BadWAMPolicy:
    return fgl.BadWAMPolicy("idm", gpu_id)


def model_bundle(policy: fgl.BadWAMPolicy, prepared: Mapping[str, Any], instruction: str, seed: int) -> tuple[dict[str, Any], torch.Tensor]:
    selected = prepared if prepared["instruction"] == instruction else {**prepared, "instruction": instruction}
    video = generate_video_latent(policy, selected, seed=seed)
    noise = action_noise(policy, seed)
    return video, noise


def patch_future(source: torch.Tensor, donor: torch.Tensor, condition: str, seed: int) -> tuple[torch.Tensor, dict[str, Any]]:
    if condition in {"clean", "context_only"}:
        hook = ImaginationIntervention(mode="identity", temporal_groups=(1, 2))
        return hook.patch_latent(source, source), hook.debug
    if condition in {"future_only", "both", "future_both"}:
        hook = ImaginationIntervention(mode="semantic_donor", temporal_groups=(1, 2))
        return hook.patch_latent(source, donor), hook.debug
    if condition == "future_group_1":
        hook = ImaginationIntervention(mode="semantic_donor", temporal_groups=(1,))
        return hook.patch_latent(source, donor), hook.debug
    if condition == "future_group_2":
        hook = ImaginationIntervention(mode="semantic_donor", temporal_groups=(2,))
        return hook.patch_latent(source, donor), hook.debug
    if condition == "shuffled_future":
        hook = ImaginationIntervention(mode="shuffled", temporal_groups=(1, 2), random_seed=seed + 999)
        return hook.patch_latent(source, source), hook.debug
    if condition == "future_swap_content":
        patched = source.clone()
        patched[:, :, 1] = source[:, :, 2]
        patched[:, :, 2] = source[:, :, 1]
        return patched, {
            "mode": "swap_content",
            "selected_temporal_groups": [1, 2],
            "source_sha256": tensor_sha256(source),
            "injected_sha256": tensor_sha256(patched),
            "content_swapped_between_temporal_slots": True,
            "positional_slots_unchanged": True,
            "current_group_preserved_bit_exact": bool(torch.equal(patched[:, :, 0], source[:, :, 0])),
        }
    raise ValueError(condition)


def finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value.float()).all())
