#!/usr/bin/env python3
from __future__ import annotations
from wam_causal_audit.paths import resolve as _release_path

import argparse
import csv
import json
import math
import sys
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from diffusers.video_processor import VideoProcessor
from PIL import Image, ImageDraw


LINGBOT_ROOT = Path(_release_path('@DATA@/lingbot-va'))
WAN_VA_ROOT = LINGBOT_ROOT / "wan_va"
if str(WAN_VA_ROOT) not in sys.path:
    sys.path.insert(0, str(WAN_VA_ROOT))

from modules.utils import load_vae  # noqa: E402


MODEL_NAME = "LingBot-VA"
DEFAULT_ROLLOUT_ROOT = Path(_release_path('@WORKSPACE@/results/d2_ghost_pilot_v2/dynaprobe_rollouts_v1/DYN'))
DEFAULT_OUT_ROOT = Path(_release_path('@WORKSPACE@/results/d2_ghost_pilot_v2/dynaprobe_motion_readout_lingbot'))
DEFAULT_VAE_PATH = Path(_release_path('@DATA@/checkpoints/lingbot-va-posttrain-libero-long/vae'))


@dataclass
class ProjectionRefs:
    a_px: np.ndarray
    c_px: np.ndarray
    d_px: np.ndarray


@dataclass
class Detection:
    ok: bool
    xy: np.ndarray | None
    reason: str
    area: int = 0
    distance_to_prior: float | None = None
    score: float | None = None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def iter_rollouts(root: Path, task_id: int, conditions: list[str]) -> list[Path]:
    task = f"libero_task{task_id:02d}"
    dirs: list[Path] = []
    for condition in conditions:
        dirs.extend((root / MODEL_NAME / task).glob(f"seed_*/{condition}"))
    return sorted(dirs)


def rollout_invalid_reason(rollout_dir: Path) -> str | None:
    try:
        event = load_json(rollout_dir / "event.json")
    except Exception:
        return None
    validation = event.get("validation") if isinstance(event.get("validation"), dict) else {}
    if bool(validation.get("invalid")):
        return str(validation.get("invalid_reason") or event.get("invalid_reason") or "invalid")
    return None


def camera_for_step(meta: dict[str, Any], step: int) -> dict[str, Any]:
    rows = meta.get("camera_poses_by_env_step") or []
    by_step = {int(row["env_step"]): row["cameras"]["agentview"] for row in rows}
    if step in by_step:
        return by_step[step]
    nearest = min(by_step, key=lambda s: abs(s - step))
    return by_step[nearest]


def project_world_to_agentview(world_xyz: Iterable[float], camera: dict[str, Any]) -> np.ndarray:
    point = np.array([*list(world_xyz), 1.0], dtype=np.float64)
    camera_from_world = np.array(camera["camera_from_world"], dtype=np.float64)
    intrinsic = np.array(camera["intrinsic"]["matrix"], dtype=np.float64)
    pc = camera_from_world @ point
    depth = -float(pc[2])
    if depth <= 1e-6:
        return np.array([np.nan, np.nan], dtype=np.float64)
    u = intrinsic[0, 0] * (pc[0] / depth) + intrinsic[0, 2]
    v = intrinsic[1, 2] - intrinsic[1, 1] * (pc[1] / depth)
    return np.array([u, v], dtype=np.float64)


def corrected_dense13_envsteps(row: dict[str, Any], decoded_frames: int) -> list[int]:
    policy_step = int(row.get("policy_call_step", row["env_step"]))
    if bool(row.get("lingbot_first_chunk_skipped_i0")):
        start = policy_step
    else:
        start = policy_step + 1
    return [start + i for i in range(decoded_frames)]


def connected_components(mask: np.ndarray) -> list[dict[str, Any]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[dict[str, Any]] = []
    for y0 in range(h):
        for x0 in range(w):
            if seen[y0, x0] or not mask[y0, x0]:
                continue
            q: deque[tuple[int, int]] = deque([(y0, x0)])
            seen[y0, x0] = True
            xs: list[int] = []
            ys: list[int] = []
            while q:
                y, x = q.popleft()
                xs.append(x)
                ys.append(y)
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    yy = y + dy
                    xx = x + dx
                    if 0 <= yy < h and 0 <= xx < w and not seen[yy, xx] and mask[yy, xx]:
                        seen[yy, xx] = True
                        q.append((yy, xx))
            xs_arr = np.array(xs, dtype=np.float64)
            ys_arr = np.array(ys, dtype=np.float64)
            comps.append(
                {
                    "area": len(xs),
                    "centroid": np.array([float(xs_arr.mean()), float(ys_arr.mean())], dtype=np.float64),
                    "bbox": (int(xs_arr.min()), int(ys_arr.min()), int(xs_arr.max()), int(ys_arr.max())),
                }
            )
    return comps


def hsv_arrays(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = rgb.astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx = np.max(arr, axis=-1)
    mn = np.min(arr, axis=-1)
    diff = mx - mn
    hue = np.zeros_like(mx)
    nz = diff > 1e-6
    idx = nz & (mx == r)
    hue[idx] = ((g[idx] - b[idx]) / diff[idx]) % 6
    idx = nz & (mx == g)
    hue[idx] = ((b[idx] - r[idx]) / diff[idx]) + 2
    idx = nz & (mx == b)
    hue[idx] = ((r[idx] - g[idx]) / diff[idx]) + 4
    hue *= 60.0
    sat = np.where(mx <= 1e-6, 0.0, diff / mx)
    return hue, sat, mx


def component_score(comp: dict[str, Any], priors: list[np.ndarray], task_id: int) -> float | None:
    x0, y0, x1, y1 = comp["bbox"]
    width = x1 - x0 + 1
    height = y1 - y0 + 1
    area = int(comp["area"])
    centroid = comp["centroid"]
    finite_priors = [p for p in priors if np.isfinite(p).all()]
    prior_center = np.mean(finite_priors, axis=0) if finite_priors else np.array([64.0, 64.0])
    dist = float(np.linalg.norm(centroid - prior_center))
    if task_id == 5:
        if not (18 <= area <= 700 and 4 <= width <= 38 and 8 <= height <= 58 and centroid[0] >= 48 and centroid[1] >= 34):
            return None
        slender_bonus = min(height / max(width, 1), 3.0)
        return -dist + 4.0 * slender_bonus + 0.015 * area
    if task_id == 9:
        if not (8 <= area <= 650 and 3 <= width <= 42 and 3 <= height <= 42 and centroid[0] >= 30 and centroid[1] >= 32):
            return None
        compact_bonus = 1.0 / max(1.0, math.sqrt(width * height))
        return -dist + 0.02 * area + 18.0 * compact_bonus
    raise ValueError(f"unsupported task id {task_id}")


def detect_target(rgb: np.ndarray, task_id: int, priors: list[np.ndarray]) -> Detection:
    rgb = np.asarray(rgb, dtype=np.uint8)
    h, w, _ = rgb.shape
    yy, xx = np.mgrid[:h, :w]
    finite_priors = [p for p in priors if np.isfinite(p).all()]
    if finite_priors:
        prior_stack = np.stack(finite_priors, axis=0)
        xmin = max(0, int(np.floor(np.nanmin(prior_stack[:, 0]) - 40)))
        xmax = min(w - 1, int(np.ceil(np.nanmax(prior_stack[:, 0]) + 40)))
        ymin = max(0, int(np.floor(np.nanmin(prior_stack[:, 1]) - 40)))
        ymax = min(h - 1, int(np.ceil(np.nanmax(prior_stack[:, 1]) + 40)))
        roi = (xx >= xmin) & (xx <= xmax) & (yy >= ymin) & (yy <= ymax)
    else:
        roi = np.ones((h, w), dtype=bool)

    if task_id == 5:
        gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])
        mask = (gray < 72) & (yy >= 32) & roi
    elif task_id == 9:
        hue, sat, val = hsv_arrays(rgb)
        yellow = (hue >= 25) & (hue <= 85) & (sat >= 0.18) & (val >= 0.28)
        greenish = (rgb[..., 1].astype(int) - rgb[..., 2].astype(int) > 12) & (rgb[..., 0] > 45) & (rgb[..., 1] > 55)
        mask = (yellow | greenish) & (yy >= 32) & roi
    else:
        raise ValueError(f"unsupported task id {task_id}")

    comps = connected_components(mask)
    scored: list[tuple[float, dict[str, Any]]] = []
    for comp in comps:
        score = component_score(comp, priors, task_id)
        if score is not None:
            scored.append((score, comp))
    if not scored:
        return Detection(ok=False, xy=None, reason="no_candidate")
    scored.sort(key=lambda x: x[0], reverse=True)
    comp = scored[0][1]
    centroid = comp["centroid"]
    if finite_priors:
        distance_to_prior = float(min(np.linalg.norm(centroid - p) for p in finite_priors))
        if distance_to_prior > 45.0:
            return Detection(ok=False, xy=centroid, reason="candidate_too_far_from_projection_band", area=int(comp["area"]), distance_to_prior=distance_to_prior)
    else:
        distance_to_prior = None
    return Detection(ok=True, xy=centroid, reason="", area=int(comp["area"]), distance_to_prior=distance_to_prior)


def crop_template(rgb: np.ndarray, center: np.ndarray, size: int) -> np.ndarray:
    half = size // 2
    h, w, _ = rgb.shape
    cx = int(round(float(center[0])))
    cy = int(round(float(center[1])))
    x0 = max(0, min(w - size, cx - half))
    y0 = max(0, min(h - size, cy - half))
    return np.asarray(rgb[y0 : y0 + size, x0 : x0 + size, :], dtype=np.uint8)


def to_gray(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float32) / 255.0
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


def match_template(
    rgb: np.ndarray,
    template_rgb: np.ndarray,
    priors: list[np.ndarray],
    device: str,
    *,
    search_margin_px: float = 38.0,
    min_score: float = 0.32,
) -> Detection:
    image = torch.from_numpy(to_gray(rgb))[None, None].to(device=device, dtype=torch.float32)
    template = torch.from_numpy(to_gray(template_rgb))[None, None].to(device=device, dtype=torch.float32)
    _, _, th, tw = template.shape
    if image.shape[-2] < th or image.shape[-1] < tw:
        return Detection(ok=False, xy=None, reason="template_larger_than_image")
    tpl = template - template.mean()
    tpl_energy = torch.sum(tpl * tpl).clamp_min(1e-8)
    ones = torch.ones_like(tpl)
    numerator = torch.nn.functional.conv2d(image, tpl)
    patch_sum = torch.nn.functional.conv2d(image, ones)
    patch_sum2 = torch.nn.functional.conv2d(image * image, ones)
    n = float(th * tw)
    patch_energy = (patch_sum2 - (patch_sum * patch_sum) / n).clamp_min(1e-8)
    score = numerator / torch.sqrt(patch_energy * tpl_energy)
    score_np = score[0, 0].detach().cpu().numpy()

    finite_priors = [p for p in priors if np.isfinite(p).all()]
    if finite_priors:
        prior_stack = np.stack(finite_priors, axis=0)
        centers_x = np.arange(score_np.shape[1], dtype=np.float32) + (tw - 1) / 2.0
        centers_y = np.arange(score_np.shape[0], dtype=np.float32) + (th - 1) / 2.0
        cx_grid, cy_grid = np.meshgrid(centers_x, centers_y)
        xmin = float(np.nanmin(prior_stack[:, 0]) - search_margin_px)
        xmax = float(np.nanmax(prior_stack[:, 0]) + search_margin_px)
        ymin = float(np.nanmin(prior_stack[:, 1]) - search_margin_px)
        ymax = float(np.nanmax(prior_stack[:, 1]) + search_margin_px)
        valid = (cx_grid >= xmin) & (cx_grid <= xmax) & (cy_grid >= ymin) & (cy_grid <= ymax)
        if not np.any(valid):
            return Detection(ok=False, xy=None, reason="empty_template_search_band")
        masked = np.where(valid, score_np, -np.inf)
    else:
        masked = score_np
    flat_idx = int(np.argmax(masked))
    best_score = float(masked.flat[flat_idx])
    if not np.isfinite(best_score):
        return Detection(ok=False, xy=None, reason="no_template_candidate")
    y, x = np.unravel_index(flat_idx, masked.shape)
    xy = np.array([x + (tw - 1) / 2.0, y + (th - 1) / 2.0], dtype=np.float64)
    distance = float(min(np.linalg.norm(xy - p) for p in finite_priors)) if finite_priors else None
    if best_score < min_score:
        return Detection(ok=False, xy=xy, reason="low_template_score", distance_to_prior=distance, score=best_score)
    return Detection(ok=True, xy=xy, reason="", distance_to_prior=distance, score=best_score)


def decode_latent(vae: Any, video_processor: VideoProcessor, latent_path: Path, device: str, dtype: torch.dtype) -> np.ndarray:
    latents = torch.load(latent_path, map_location="cpu").to(device=device, dtype=dtype)
    with torch.no_grad():
        mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype)
        std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype)
        video = vae.decode(latents / std + mean, return_dict=False)[0]
        arr = video_processor.postprocess_video(video, output_type="np")[0]
    if arr.max() <= 1.5:
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    else:
        arr = arr.clip(0, 255).astype(np.uint8)
    return arr[:, :, :128, :]


def reference_points_for_step(
    *,
    step: int,
    meta: dict[str, Any],
    steps_by_step: dict[int, np.ndarray],
    event: dict[str, Any],
    offset_px: np.ndarray,
) -> ProjectionRefs:
    trace = event.get("target_qpos_by_env_step") or []
    if trace and isinstance(trace[0], dict) and trace[0].get("target_center") is not None:
        a_pos = np.array(trace[0]["target_center"][:3], dtype=np.float64)
    else:
        t0 = int(event["t0_env_step"])
        a_pos = np.array(steps_by_step.get(t0 - 1, steps_by_step[min(steps_by_step)])[:3], dtype=np.float64)
    t0 = int(event["t0_env_step"])
    transform = event["tuple"]["transform"]
    n_slide = int(transform["n_slide_steps"])
    delta = np.array([*transform["delta_xy_m"], 0.0], dtype=np.float64)
    per_step = delta / float(n_slide)
    progress = max(0, int(step) - t0 + 1)
    d_pos = a_pos + per_step * progress
    c_pos = np.array(steps_by_step[int(step)][:3], dtype=np.float64)
    camera = camera_for_step(meta, int(step))
    return ProjectionRefs(
        a_px=project_world_to_agentview(a_pos, camera) + offset_px,
        c_px=project_world_to_agentview(c_pos, camera) + offset_px,
        d_px=project_world_to_agentview(d_pos, camera) + offset_px,
    )


def classify_detection(det: Detection, refs: ProjectionRefs, threshold_px: float, min_margin_px: float) -> tuple[str, dict[str, float]]:
    if not det.ok or det.xy is None:
        return "collapse", {"d_A": math.nan, "d_C": math.nan, "d_D": math.nan, "margin": math.nan}
    distances = {
        "lag": float(np.linalg.norm(det.xy - refs.a_px)),
        "snapshot": float(np.linalg.norm(det.xy - refs.c_px)),
        "extrapolate": float(np.linalg.norm(det.xy - refs.d_px)),
    }
    ordered = sorted(distances.items(), key=lambda kv: kv[1])
    margin = ordered[1][1] - ordered[0][1]
    metrics = {"d_A": distances["lag"], "d_C": distances["snapshot"], "d_D": distances["extrapolate"], "margin": margin}
    if ordered[0][1] > threshold_px:
        return "collapse", metrics
    if margin < min_margin_px:
        if distances["snapshot"] <= threshold_px and distances["extrapolate"] <= threshold_px:
            return "snapshot", metrics
        return "collapse", metrics
    return ordered[0][0], metrics


def draw_debug_frame(rgb: np.ndarray, det: Detection, refs: ProjectionRefs, out_path: Path) -> None:
    im = Image.fromarray(rgb)
    d = ImageDraw.Draw(im)
    colors = {"A": "yellow", "C": "lime", "D": "cyan"}
    for label, pt in (("A", refs.a_px), ("C", refs.c_px), ("D", refs.d_px)):
        if np.isfinite(pt).all():
            x, y = float(pt[0]), float(pt[1])
            d.ellipse((x - 3, y - 3, x + 3, y + 3), outline=colors[label], width=2)
            d.text((x + 4, y - 4), label, fill=colors[label])
    if det.xy is not None:
        x, y = float(det.xy[0]), float(det.xy[1])
        d.rectangle((x - 3, y - 3, x + 3, y + 3), outline="red", width=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path, quality=95)


def analyze_rollout(
    rollout_dir: Path,
    task_id: int,
    vae: Any,
    video_processor: VideoProcessor,
    device: str,
    dtype: torch.dtype,
    debug_dir: Path,
) -> list[dict[str, Any]]:
    meta = load_json(rollout_dir / "meta.json")
    event = load_json(rollout_dir / "event.json")
    policy_rows = load_jsonl(rollout_dir / "policy_calls.jsonl")
    steps_df = pd.read_parquet(rollout_dir / "steps.parquet")
    steps_by_step = {int(row.env_step): np.array(row.target_obj_pose[:3], dtype=np.float64) for row in steps_df.itertuples()}
    seed = int(meta["seed"])
    condition = str(meta.get("condition") or rollout_dir.name)

    frame_items: list[dict[str, Any]] = []
    debug_written = False
    for row in policy_rows:
        if row.get("latent_window") is False or not row.get("latent_path"):
            continue
        latent_path = Path(row["latent_path"])
        if not latent_path.exists():
            continue
        frames = decode_latent(vae, video_processor, latent_path, device, dtype)
        env_steps = corrected_dense13_envsteps(row, int(frames.shape[0]))
        for frame_idx, (rgb, env_step) in enumerate(zip(frames, env_steps)):
            if int(env_step) not in steps_by_step:
                continue
            refs_unshifted = reference_points_for_step(
                step=int(env_step),
                meta=meta,
                steps_by_step=steps_by_step,
                event=event,
                offset_px=np.zeros(2, dtype=np.float64),
            )
            frame_items.append(
                {
                    "rollout_dir": str(rollout_dir),
                    "task_id": int(task_id),
                    "seed": seed,
                    "condition": condition,
                    "call_idx": int(row["call_idx"]),
                    "frame_idx": int(frame_idx),
                    "env_step": int(env_step),
                    "rgb": rgb,
                    "refs_unshifted": refs_unshifted,
                }
            )

    if not frame_items:
        return []
    template_size = 31 if task_id == 5 else 25
    first_refs = frame_items[0]["refs_unshifted"]
    template = crop_template(frame_items[0]["rgb"], first_refs.c_px, template_size)
    min_score = 0.30 if task_id == 5 else 0.24

    records: list[dict[str, Any]] = []
    for item in frame_items:
        refs = item["refs_unshifted"]
        det = match_template(
            item["rgb"],
            template,
            [refs.a_px, refs.c_px, refs.d_px],
            device,
            search_margin_px=40.0,
            min_score=min_score,
        )
        label, metrics = classify_detection(det, refs, threshold_px=16.0, min_margin_px=2.0)
        det_xy = det.xy if det.xy is not None else np.array([math.nan, math.nan], dtype=np.float64)
        records.append(
            {
                "model": MODEL_NAME,
                "task_id": int(task_id),
                "seed": seed,
                "condition": condition,
                "call_idx": int(item["call_idx"]),
                "frame_idx": int(item["frame_idx"]),
                "env_step": int(item["env_step"]),
                "detected": bool(det.ok),
                "detect_reason": det.reason,
                "det_area": int(det.area),
                "det_score": float(det.score) if det.score is not None else math.nan,
                "det_x": float(det_xy[0]),
                "det_y": float(det_xy[1]),
                "A_x": float(refs.a_px[0]),
                "A_y": float(refs.a_px[1]),
                "C_x": float(refs.c_px[0]),
                "C_y": float(refs.c_px[1]),
                "D_x": float(refs.d_px[0]),
                "D_y": float(refs.d_px[1]),
                "class": label,
                "d_A": metrics["d_A"],
                "d_C": metrics["d_C"],
                "d_D": metrics["d_D"],
                "margin": metrics["margin"],
                "offset_x": 0.0,
                "offset_y": 0.0,
            }
        )
        if not debug_written and item["call_idx"] in {0, 3} and item["frame_idx"] in {0, 6, 12}:
            debug_path = debug_dir / condition / f"task{task_id:02d}_seed{seed:03d}_call{item['call_idx']:02d}_frame{item['frame_idx']:02d}_{label}.jpg"
            draw_debug_frame(item["rgb"], det, refs, debug_path)
            debug_written = True
    return records


def plot_task(records: list[dict[str, Any]], task_id: int, condition: str, out_path: Path) -> None:
    rows = [r for r in records if int(r["task_id"]) == int(task_id) and str(r["condition"]) == str(condition)]
    if not rows:
        return
    rows = sorted(rows, key=lambda r: (r["seed"], r["env_step"], r["call_idx"], r["frame_idx"]))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for seed in sorted({r["seed"] for r in rows}):
        sr = [r for r in rows if r["seed"] == seed and r["detected"]]
        if not sr:
            continue
        env = np.array([r["env_step"] for r in sr], dtype=float)
        axes[0].scatter(env, [r["det_x"] for r in sr], s=8, alpha=0.28, color="tab:red")
        axes[1].scatter(env, [r["det_y"] for r in sr], s=8, alpha=0.28, color="tab:red")
        axes[2].scatter([r["det_x"] for r in sr], [r["det_y"] for r in sr], s=8, alpha=0.25, color="tab:red")
    by_step: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        by_step.setdefault(int(r["env_step"]), []).append(r)
    steps = np.array(sorted(by_step), dtype=float)
    for key, color, label in (("C", "tab:green", "true C"), ("D", "tab:blue", "extrap D")):
        xs = []
        ys = []
        for step in steps:
            vals = by_step[int(step)]
            xs.append(float(np.mean([r[f"{key}_x"] for r in vals])))
            ys.append(float(np.mean([r[f"{key}_y"] for r in vals])))
        axes[0].plot(steps, xs, color=color, label=label, linewidth=2)
        axes[1].plot(steps, ys, color=color, label=label, linewidth=2)
        axes[2].plot(xs, ys, color=color, label=label, linewidth=2)
    axes[0].set_title(f"LingBot task{task_id:02d} {condition}: x over env_step")
    axes[1].set_title(f"LingBot task{task_id:02d} {condition}: y over env_step")
    axes[2].set_title(f"LingBot task{task_id:02d} {condition}: pixel trajectory")
    axes[0].set_xlabel("env_step")
    axes[1].set_xlabel("env_step")
    axes[0].set_ylabel("agentview pixel x")
    axes[1].set_ylabel("agentview pixel y")
    axes[2].set_xlabel("pixel x")
    axes[2].set_ylabel("pixel y")
    axes[2].invert_yaxis()
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    keys = sorted({(str(r["condition"]), int(r["task_id"])) for r in records})
    for condition, task_id in keys:
        rows = [r for r in records if int(r["task_id"]) == task_id and str(r["condition"]) == condition]
        total = len(rows)
        detected = sum(1 for r in rows if r["detected"])
        classes = Counter(r["class"] for r in rows)
        summaries.append(
            {
                "model": MODEL_NAME,
                "condition": condition,
                "task_id": task_id,
                "frames": total,
                "detected": detected,
                "detection_failure_rate": 1.0 - (detected / total if total else 0.0),
                "extrapolate": classes["extrapolate"],
                "snapshot": classes["snapshot"],
                "lag": classes["lag"],
                "collapse": classes["collapse"],
                "extrapolate_rate": classes["extrapolate"] / total if total else 0.0,
                "snapshot_rate": classes["snapshot"] / total if total else 0.0,
                "lag_rate": classes["lag"] / total if total else 0.0,
                "collapse_rate": classes["collapse"] / total if total else 0.0,
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline LingBot-VA DynaProbe motion-imagination readout.")
    parser.add_argument("--rollout-root", type=Path, default=DEFAULT_ROLLOUT_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--vae-path", type=Path, default=DEFAULT_VAE_PATH)
    parser.add_argument("--tasks", nargs="+", type=int, default=[5, 9])
    parser.add_argument("--condition", default=None, help="Single condition label to analyze.")
    parser.add_argument("--conditions", nargs="+", default=None, help="One or more condition labels to analyze.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--max-rollouts-per-task", type=int, default=None)
    parser.add_argument("--include-invalid", action="store_true", help="Include rollouts whose event.validation.invalid is true.")
    args = parser.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    out_root = args.out_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    debug_dir = out_root / "debug_frames"
    conditions = args.conditions if args.conditions is not None else [args.condition or "measure1_slide_precheck"]

    vae = load_vae(str(args.vae_path), torch_dtype=dtype, torch_device=args.device).eval()
    video_processor = VideoProcessor(vae_scale_factor=1)

    all_records: list[dict[str, Any]] = []
    skipped_invalid: list[dict[str, Any]] = []
    for task_id in args.tasks:
        rollout_dirs = iter_rollouts(args.rollout_root, task_id, conditions)
        if args.max_rollouts_per_task is not None:
            rollout_dirs = rollout_dirs[: args.max_rollouts_per_task]
        print(f"[task{task_id:02d}] rollouts={len(rollout_dirs)}")
        for idx, rollout_dir in enumerate(rollout_dirs, 1):
            invalid_reason = rollout_invalid_reason(rollout_dir)
            if invalid_reason and not args.include_invalid:
                print(f"  [{idx}/{len(rollout_dirs)}] SKIP invalid {rollout_dir} reason={invalid_reason}")
                skipped_invalid.append(
                    {
                        "task_id": int(task_id),
                        "condition": str(rollout_dir.name),
                        "seed": int(rollout_dir.parent.name.split("_")[-1]),
                        "reason": invalid_reason,
                        "rollout_dir": str(rollout_dir),
                    }
                )
                continue
            print(f"  [{idx}/{len(rollout_dirs)}] {rollout_dir}")
            all_records.extend(analyze_rollout(rollout_dir, task_id, vae, video_processor, args.device, dtype, debug_dir))

    frame_csv = out_root / "lingbot_motion_frame_classifications.csv"
    frame_fields = [
        "model",
        "condition",
        "task_id",
        "seed",
        "call_idx",
        "frame_idx",
        "env_step",
        "detected",
        "detect_reason",
        "det_area",
        "det_score",
        "det_x",
        "det_y",
        "A_x",
        "A_y",
        "C_x",
        "C_y",
        "D_x",
        "D_y",
        "class",
        "d_A",
        "d_C",
        "d_D",
        "margin",
        "offset_x",
        "offset_y",
    ]
    with frame_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=frame_fields)
        writer.writeheader()
        for row in all_records:
            writer.writerow({field: row.get(field, "") for field in frame_fields})

    summaries = summarize(all_records)
    summary_csv = out_root / "lingbot_motion_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()) if summaries else ["model", "task_id"])
        writer.writeheader()
        writer.writerows(summaries)

    plots: list[str] = []
    for task_id in args.tasks:
        for condition in conditions:
            plot_path = out_root / f"lingbot_task{task_id:02d}_{condition}_three_trajectory.png"
            plot_task(all_records, task_id, condition, plot_path)
            if plot_path.exists():
                plots.append(str(plot_path))

    manifest = {
        "model": MODEL_NAME,
        "input_root": str(args.rollout_root),
        "out_root": str(out_root),
        "tasks": [int(x) for x in args.tasks],
        "conditions": list(conditions),
        "frame_mapping": {
            "status": "analysis_side_correction",
            "reason": "LingBot T=4 latent decodes to 13 dense RGB frames. Original post-call policy_calls rows contain 16 action-step entries after call0.",
            "rule": "call0 -> policy_call_step + [0..12]; later calls -> policy_call_step + [1..13]",
        },
        "projection": {
            "depth_axis": "-camera_z",
            "image_y": "cy - fy * camera_y / depth",
            "visual_center_offset": "none; template is centered on projected C in the first decoded frame, so matched centers use the same projection anchor convention",
        },
        "detector": {
            "task05": "per-rollout normalized template matching; template cropped around projected C in first decoded frame",
            "task09": "per-rollout normalized template matching; template cropped around projected C in first decoded frame",
            "classification_threshold_px": 16.0,
            "min_nearest_margin_px": 2.0,
        },
        "outputs": {
            "frame_csv": str(frame_csv),
            "summary_csv": str(summary_csv),
            "plots": plots,
            "debug_frames": str(debug_dir),
        },
        "include_invalid": bool(args.include_invalid),
        "skipped_invalid": skipped_invalid,
        "summary": summaries,
    }
    write_json(out_root / "manifest.json", manifest)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"[done] {out_root}")


if __name__ == "__main__":
    main()
