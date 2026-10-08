from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import imageio.v2 as imageio
import numpy as np

from d2_ghost_scheduler import body_pos, get_sim, jsonable, name_is_related
from dynaprobe_rollout_v1 import camera_snapshot

try:
    from analyze_lingbot_motion_readout import project_world_to_agentview as _project_world_to_camera
except Exception:  # pragma: no cover - fallback keeps rollout runnable if analysis deps are absent
    _project_world_to_camera = None


CAMERA_OBS_KEYS = {
    "agentview": "agentview_image",
    "robot0_eye_in_hand": "robot0_eye_in_hand_image",
}


def _image_convention_stride() -> int:
    try:
        from robosuite import macros
        from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING

        return int(IMAGE_CONVENTION_MAPPING.get(macros.IMAGE_CONVENTION, 1))
    except Exception:
        return 1


def _geom_name(model: Any, geom_id: int) -> str:
    for attr in ("geom_id2name",):
        fn = getattr(model, attr, None)
        if callable(fn):
            try:
                value = fn(int(geom_id))
                if value:
                    return str(value)
            except Exception:
                pass
    try:
        import mujoco

        value = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
        return "" if value is None else str(value)
    except Exception:
        return ""


def _body_name(model: Any, body_id: int) -> str:
    for attr in ("body_id2name",):
        fn = getattr(model, attr, None)
        if callable(fn):
            try:
                value = fn(int(body_id))
                if value:
                    return str(value)
            except Exception:
                pass
    try:
        import mujoco

        value = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body_id))
        return "" if value is None else str(value)
    except Exception:
        return ""


def _rle_bool_mask(mask: np.ndarray) -> dict[str, Any]:
    flat = np.asarray(mask, dtype=np.uint8).reshape(-1)
    if flat.size == 0:
        return {"shape": list(mask.shape), "start_value": 0, "counts": []}
    counts: list[int] = []
    value = int(flat[0])
    count = 1
    for item in flat[1:]:
        item_i = int(item)
        if item_i == value:
            count += 1
            continue
        counts.append(count)
        value = item_i
        count = 1
    counts.append(count)
    return {"shape": list(mask.shape), "start_value": int(flat[0]), "counts": counts}


class OcclusionMaskScheduler:
    """Observation-layer PERM occluder composed with TranslationInjectionScheduler.

    The wrapped translation scheduler owns triggering and target qpos movement.
    This class only masks camera observations that are about to enter a policy.
    """

    def __init__(
        self,
        *,
        translation_scheduler: Any,
        target_object_name: str,
        target_body_name: str,
        occlusion_steps: int,
        camera_obs_keys: Mapping[str, str] | None = None,
        bbox_padding_px: int = 6,
        bbox_padding_frac: float = 0.12,
        projection_radius_m: float = 0.12,
        sanity_dir: str | Path | None = None,
        sanity_max_frames: int = 8,
    ) -> None:
        steps = int(occlusion_steps)
        if steps < 1:
            raise ValueError(f"occlusion_steps must be >= 1, got {occlusion_steps}")
        self.translation_scheduler = translation_scheduler
        self.target_object_name = str(target_object_name)
        self.target_body_name = str(target_body_name)
        self.occlusion_steps = steps
        self.camera_obs_keys = dict(camera_obs_keys or CAMERA_OBS_KEYS)
        self.bbox_padding_px = int(bbox_padding_px)
        self.bbox_padding_frac = float(bbox_padding_frac)
        self.projection_radius_m = float(projection_radius_m)
        self.sanity_dir = None if sanity_dir is None else Path(sanity_dir)
        self.sanity_max_frames = int(sanity_max_frames)
        self.records: list[dict[str, Any]] = []
        self.failures: list[str] = []
        self._target_geom_ids: set[int] | None = None
        self._sanity_saved = 0
        self.multisample_summary: dict[str, Any] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.translation_scheduler, name)

    @property
    def t0_env_step(self) -> int | None:
        return self.translation_scheduler.t0_env_step

    @property
    def occlusion_span(self) -> list[int] | None:
        t0 = self.t0_env_step
        if t0 is None:
            return None
        return [int(t0), int(t0) + self.occlusion_steps - 1]

    def is_active_step(self, env_step: int | None) -> bool:
        t0 = self.t0_env_step
        if t0 is None or env_step is None:
            return False
        step = int(env_step)
        return int(t0) <= step < int(t0) + self.occlusion_steps

    def apply_to_raw_obs(
        self,
        env: Any,
        raw_obs: Mapping[str, Any],
        *,
        env_step: int,
        policy_call_idx: int | None,
        role: str,
        record: bool = True,
        write_sanity: bool = True,
    ) -> dict[str, Any]:
        if not self.is_active_step(env_step):
            return dict(raw_obs)

        out = dict(raw_obs)
        record_entry = {
            "env_step": int(env_step),
            "policy_call_idx": None if policy_call_idx is None else int(policy_call_idx),
            "role": str(role),
            "active": True,
            "cameras": {},
        } if record else None
        for camera_name, obs_key in self.camera_obs_keys.items():
            if obs_key not in raw_obs:
                raise KeyError(f"PERM occlusion cannot find observation key {obs_key!r} for camera {camera_name!r}")
            before = np.asarray(raw_obs[obs_key])
            after, camera_record = self._mask_camera(env, before, camera_name=str(camera_name), env_step=int(env_step))
            out[obs_key] = np.ascontiguousarray(after)
            if record_entry is not None:
                record_entry["cameras"][str(camera_name)] = camera_record
            if write_sanity:
                self._maybe_write_sanity(camera_name=str(camera_name), env_step=int(env_step), before=before, after=after)
        if record_entry is not None:
            self.records.append(record_entry)
        return out

    def _target_geom_ids_for_sim(self, sim: Any) -> set[int]:
        if self._target_geom_ids is not None:
            return self._target_geom_ids
        model = sim.model
        body_matched: set[int] = set()
        name_matched: set[int] = set()
        geom_bodyid = np.asarray(getattr(model, "geom_bodyid", []), dtype=np.int64).reshape(-1)
        for geom_id in range(int(getattr(model, "ngeom", 0) or 0)):
            geom_name = _geom_name(model, geom_id)
            body_name = _body_name(model, int(geom_bodyid[geom_id])) if geom_id < geom_bodyid.size else ""
            if name_is_related(body_name, self.target_body_name):
                body_matched.add(int(geom_id))
                continue
            if body_name.lower() != "world" and (
                name_is_related(geom_name, self.target_object_name)
                or name_is_related(geom_name, self.target_body_name)
                or name_is_related(body_name, self.target_object_name)
            ):
                name_matched.add(int(geom_id))
        self._target_geom_ids = body_matched or name_matched
        return self._target_geom_ids

    def _mask_camera(
        self,
        env: Any,
        image: np.ndarray,
        *,
        camera_name: str,
        env_step: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if image.ndim != 3 or image.shape[-1] < 3:
            raise ValueError(f"Expected RGB image for {camera_name}, got shape {image.shape}")
        h, w = int(image.shape[0]), int(image.shape[1])
        sim = get_sim(env)
        source = "segmentation"
        mask = self._segmentation_mask(sim, camera_name=camera_name, width=w, height=h)
        if mask is None or int(mask.sum()) <= 0:
            source = "projection_fallback"
            mask = self._projection_mask(env, camera_name=camera_name, width=w, height=h)
        if mask is None or int(mask.sum()) <= 0:
            reason = f"PERM occlusion bbox unavailable at env_step={env_step}, camera={camera_name}"
            self.failures.append(reason)
            raise RuntimeError(reason)

        occlusion_mask = self._expanded_mask(mask)
        bbox = self._expanded_bbox(occlusion_mask, width=w, height=h)
        x0, y0, x1, y1 = bbox
        out, fill_method = self._background_fill_image(image, bbox, occlusion_mask)
        fill_pixels = np.asarray(out[..., :3], dtype=np.float32)[occlusion_mask]
        fill = np.mean(fill_pixels, axis=0) if fill_pixels.size else np.zeros(3, dtype=np.float32)

        crop_mask = mask[y0 : y1 + 1, x0 : x1 + 1]
        crop_occlusion_mask = occlusion_mask[y0 : y1 + 1, x0 : x1 + 1]
        return out, {
            "bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
            "image_hw": [h, w],
            "mask_source": source,
            "mask_area_px": int(mask.sum()),
            "occluded_area_px": int(occlusion_mask.sum()),
            "mask_rle_crop": _rle_bool_mask(crop_mask),
            "occlusion_mask_rle_crop": _rle_bool_mask(crop_occlusion_mask),
            "fill_method": fill_method,
            "fill_rgb": [float(x) for x in np.asarray(fill, dtype=np.float32).reshape(-1)[:3]],
        }

    def _segmentation_mask(self, sim: Any, *, camera_name: str, width: int, height: int) -> np.ndarray | None:
        target_geom_ids = self._target_geom_ids_for_sim(sim)
        if not target_geom_ids:
            return None
        try:
            seg = sim.render(camera_name=camera_name, width=width, height=height, depth=False, segmentation=True)
        except TypeError:
            try:
                seg = sim.render(width=width, height=height, camera_name=camera_name, depth=False, segmentation=True)
            except Exception:
                return None
        except Exception:
            # Older robosuite + NumPy 2 can overflow while encoding geom id
            # 256 into a uint8 segmentation buffer.  This is precisely the
            # unstable-segmentation case covered by the projection fallback.
            return None
        if isinstance(seg, tuple):
            seg = seg[0]
        arr = np.asarray(seg)
        if arr.ndim != 3 or arr.shape[-1] < 2:
            return None
        stride = _image_convention_stride()
        # MuJoCo geom ids can exceed uint8 even when a renderer returns an
        # uint8 segmentation buffer.  Promote before building the target-id
        # array so ids such as 256 do not raise or wrap to zero.
        geom_ids = np.asarray(arr[::stride, :, 1], dtype=np.int64)
        return np.isin(geom_ids, np.asarray(sorted(target_geom_ids), dtype=np.int64))

    def _projection_mask(self, env: Any, *, camera_name: str, width: int, height: int) -> np.ndarray | None:
        sim = get_sim(env)
        points = self._target_geometry_bbox_points(sim)
        if not points:
            center = body_pos(sim, self.target_body_name).astype(np.float64)
            radius = float(self.projection_radius_m)
            offsets = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [radius, 0.0, 0.0],
                    [-radius, 0.0, 0.0],
                    [0.0, radius, 0.0],
                    [0.0, -radius, 0.0],
                    [0.0, 0.0, radius],
                    [0.0, 0.0, -radius],
                ],
                dtype=np.float64,
            )
            points = [point for point in center.reshape(1, 3) + offsets]
        cameras = camera_snapshot(env, width=width, height=height, camera_names=[camera_name])
        camera = cameras[camera_name]
        pixels = []
        for point in points:
            if _project_world_to_camera is not None:
                px = np.asarray(_project_world_to_camera(point, camera), dtype=np.float64).reshape(2)
            else:
                px = self._project_world_to_camera(point, camera)
            if np.all(np.isfinite(px)):
                # camera_from_world/intrinsic yields conventional top-left
                # image coordinates, while raw LIBERO observations are still
                # in MuJoCo's bottom-up convention at this hook point.
                px = px.copy()
                px[1] = float(height - 1) - px[1]
                pixels.append(px)
        if not pixels:
            return None
        arr = np.asarray(pixels, dtype=np.float64)
        rounded = np.rint(arr).astype(np.int32)
        rounded[:, 0] = np.clip(rounded[:, 0], 0, width - 1)
        rounded[:, 1] = np.clip(rounded[:, 1], 0, height - 1)
        mask = np.zeros((height, width), dtype=np.uint8)
        try:
            import cv2

            hull = cv2.convexHull(rounded.reshape(-1, 1, 2))
            cv2.fillConvexPoly(mask, hull, 1)
        except Exception:
            x0, y0 = np.min(rounded, axis=0)
            x1, y1 = np.max(rounded, axis=0)
            mask[int(y0) : int(y1) + 1, int(x0) : int(x1) + 1] = 1
        return mask.astype(bool)

    def _target_geometry_bbox_points(self, sim: Any) -> list[np.ndarray]:
        """World-space corners of target geoms for segmentation fallback.

        This is the requested world-to-pixel + object-size fallback, using
        MuJoCo's live geom pose rather than a task-specific colour detector.
        """
        try:
            geom_ids = self._target_geom_ids_for_sim(sim)
            geom_pos = np.asarray(sim.data.geom_xpos, dtype=np.float64)
            geom_mat = np.asarray(sim.data.geom_xmat, dtype=np.float64).reshape(-1, 3, 3)
            geom_size = np.asarray(sim.model.geom_size, dtype=np.float64)
        except Exception:
            return []
        corners: list[np.ndarray] = []
        signs = np.asarray(
            [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
            dtype=np.float64,
        )
        for geom_id in geom_ids:
            if geom_id < 0 or geom_id >= len(geom_pos) or geom_id >= len(geom_size):
                continue
            half_extent = np.asarray(geom_size[geom_id], dtype=np.float64).reshape(-1)[:3]
            if half_extent.size < 3:
                continue
            # Some geom types expose only radius / half-length; a box using
            # their maximum extent is conservative and still much tighter
            # than a fixed 12 cm sphere.
            half_extent = np.maximum(half_extent, float(self.bbox_padding_px) * 0.0005)
            local = signs * half_extent.reshape(1, 3)
            center = geom_pos[geom_id].reshape(3)
            rotation = geom_mat[geom_id]
            corners.extend([center + rotation @ point for point in local])
        return corners

    @staticmethod
    def _project_world_to_camera(world_xyz: np.ndarray, camera: Mapping[str, Any]) -> np.ndarray:
        point = np.array([float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2]), 1.0], dtype=np.float64)
        camera_from_world = np.array(camera["camera_from_world"], dtype=np.float64)
        intrinsic = np.array(camera["intrinsic"]["matrix"], dtype=np.float64)
        pc = camera_from_world @ point
        depth = -float(pc[2])
        if depth <= 1e-6:
            return np.array([np.nan, np.nan], dtype=np.float64)
        u = intrinsic[0, 0] * (pc[0] / depth) + intrinsic[0, 2]
        v = intrinsic[1, 2] - intrinsic[1, 1] * (pc[1] / depth)
        return np.array([u, v], dtype=np.float64)

    def _expanded_bbox(self, mask: np.ndarray, *, width: int, height: int) -> tuple[int, int, int, int]:
        ys, xs = np.nonzero(mask)
        x0 = int(xs.min())
        x1 = int(xs.max())
        y0 = int(ys.min())
        y1 = int(ys.max())
        return (
            max(0, x0),
            max(0, y0),
            min(width - 1, x1),
            min(height - 1, y1),
        )

    def _expanded_mask(self, mask: np.ndarray) -> np.ndarray:
        ys, xs = np.nonzero(mask)
        if not len(xs):
            return np.asarray(mask, dtype=bool)
        span = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
        pad = max(int(self.bbox_padding_px), int(np.ceil(span * self.bbox_padding_frac)))
        binary = np.asarray(mask, dtype=np.uint8)
        try:
            import cv2

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
            return cv2.dilate(binary, kernel, iterations=1).astype(bool)
        except Exception:
            out = binary.astype(bool).copy()
            for dy in range(-pad, pad + 1):
                for dx in range(-pad, pad + 1):
                    if dx * dx + dy * dy > pad * pad:
                        continue
                    shifted = np.zeros_like(out)
                    y0, y1 = max(0, dy), min(out.shape[0], out.shape[0] + dy)
                    x0, x1 = max(0, dx), min(out.shape[1], out.shape[1] + dx)
                    shifted[y0:y1, x0:x1] = binary[y0 - dy : y1 - dy, x0 - dx : x1 - dx]
                    out |= shifted
            return out

    @staticmethod
    def _background_fill_image(
        image: np.ndarray, bbox: tuple[int, int, int, int], fill_mask: np.ndarray
    ) -> tuple[np.ndarray, str]:
        x0, y0, x1, y1 = bbox
        h, w = image.shape[:2]
        ring = max(14, int(round(0.30 * max(x1 - x0 + 1, y1 - y0 + 1))))
        rx0 = max(0, x0 - ring)
        ry0 = max(0, y0 - ring)
        rx1 = min(w - 1, x1 + ring)
        ry1 = min(h - 1, y1 + ring)
        region = np.asarray(image[ry0 : ry1 + 1, rx0 : rx1 + 1, :3], dtype=np.float32)
        ring_mask = ~np.asarray(fill_mask[ry0 : ry1 + 1, rx0 : rx1 + 1], dtype=bool)
        samples = region[ring_mask]
        if samples.size == 0:
            samples = np.asarray(image[..., :3], dtype=np.float32).reshape(-1, 3)

        yy, xx = np.mgrid[ry0 : ry1 + 1, rx0 : rx1 + 1]
        coords = np.stack([np.ones_like(xx), xx, yy], axis=-1)[ring_mask].astype(np.float64)
        values = region[ring_mask].astype(np.float64)
        # Dark robot / object pixels are common near the wrist-view boundary.
        # Keep the brighter table cluster, then robustly fit its local mean
        # plane so the replacement follows illumination and wood gradients.
        brightness = np.mean(values, axis=1)
        wood_like = (
            (values[:, 0] > values[:, 1] + 8.0)
            & (values[:, 1] > values[:, 2] + 8.0)
            & (brightness > 50.0)
        )
        keep = wood_like if int(wood_like.sum()) >= 12 else brightness >= np.percentile(brightness, 45.0)
        coords_fit = coords[keep]
        values_fit = values[keep]
        if len(coords_fit) < 12:
            coords_fit, values_fit = coords, values
        coeff = np.linalg.lstsq(coords_fit, values_fit, rcond=None)[0]
        for _ in range(2):
            residual = np.linalg.norm(values_fit - coords_fit @ coeff, axis=1)
            cutoff = np.percentile(residual, 70.0)
            inliers = residual <= cutoff
            if int(inliers.sum()) < 12:
                break
            coeff = np.linalg.lstsq(coords_fit[inliers], values_fit[inliers], rcond=None)[0]

        fy, fx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
        fill_coords = np.stack([np.ones_like(fx), fx, fy], axis=-1).astype(np.float64)
        fill_patch = fill_coords @ coeff
        fill_patch = np.clip(fill_patch, 0.0, 255.0).astype(np.float32)

        # Reuse only the high-frequency component of a nearby table patch.
        # This keeps the ring-derived mean / lighting plane while avoiding a
        # conspicuously textureless rectangle.
        ph, pw = fill_patch.shape[:2]
        rgb_full = np.asarray(image[..., :3], dtype=np.float32)
        target_mean = np.mean(fill_patch.reshape(-1, 3), axis=0)
        candidates: list[tuple[float, np.ndarray]] = []
        stride_y = max(6, ph // 4)
        stride_x = max(6, pw // 4)
        for sy in range(0, max(1, h - ph + 1), stride_y):
            for sx in range(0, max(1, w - pw + 1), stride_x):
                if not (sx + pw <= x0 or sx > x1 or sy + ph <= y0 or sy > y1):
                    continue
                candidate = rgb_full[sy : sy + ph, sx : sx + pw]
                if candidate.shape[:2] != (ph, pw):
                    continue
                cand_brightness = np.mean(candidate, axis=2)
                dark_fraction = float(np.mean(cand_brightness < 55.0))
                color_mismatch = float(np.linalg.norm(np.mean(candidate.reshape(-1, 3), axis=0) - target_mean))
                gx = np.mean(np.abs(np.diff(candidate, axis=1))) if pw > 1 else 0.0
                gy = np.mean(np.abs(np.diff(candidate, axis=0))) if ph > 1 else 0.0
                edge_score = float(gx + gy)
                score = 180.0 * dark_fraction + color_mismatch + 2.5 * edge_score
                candidates.append((score, candidate))
        if candidates:
            source = min(candidates, key=lambda item: item[0])[1]
            try:
                import cv2

                # A wide blur keeps the broad illumination correction in the
                # fitted plane but preserves the table's visible grain.
                smooth = cv2.GaussianBlur(source, (0, 0), sigmaX=12.0, sigmaY=12.0)
            except Exception:
                smooth = np.broadcast_to(np.mean(source, axis=(0, 1), keepdims=True), source.shape)
            texture = np.clip(source - smooth, -18.0, 18.0)
            fill_patch = np.clip(fill_patch + 0.9 * texture, 0.0, 255.0)

        # Feather only inside the padded border.  The exact segmentation mask
        # lies beyond this band, so no target pixels remain visible.
        border = max(1, min(4, min(ph, pw) // 5))
        distance = np.minimum.reduce(
            [
                np.arange(ph)[:, None] + np.zeros((ph, pw)),
                np.arange(ph - 1, -1, -1)[:, None] + np.zeros((ph, pw)),
                np.arange(pw)[None, :] + np.zeros((ph, pw)),
                np.arange(pw - 1, -1, -1)[None, :] + np.zeros((ph, pw)),
            ]
        )
        alpha = np.clip((distance + 1.0) / float(border), 0.0, 1.0)[..., None]
        out = np.asarray(image).copy()
        original = np.asarray(out[y0 : y1 + 1, x0 : x1 + 1, :3], dtype=np.float32)
        local_mask = np.asarray(fill_mask[y0 : y1 + 1, x0 : x1 + 1], dtype=bool)
        try:
            import cv2

            distance = cv2.distanceTransform(local_mask.astype(np.uint8), cv2.DIST_L2, 3)[..., None]
            alpha = np.clip(distance / float(border), 0.0, 1.0)
        except Exception:
            alpha = local_mask[..., None].astype(np.float32)
        blended = alpha * fill_patch + (1.0 - alpha) * original
        result = original.copy()
        result[local_mask] = blended[local_mask]
        out[y0 : y1 + 1, x0 : x1 + 1, :3] = np.clip(np.round(result), 0, 255).astype(image.dtype)
        return out, "surrounding_table_robust_mean_plane_texture_feathered"

    def _maybe_write_sanity(self, *, camera_name: str, env_step: int, before: np.ndarray, after: np.ndarray) -> None:
        if self.sanity_dir is None or self._sanity_saved >= self.sanity_max_frames:
            return
        self.sanity_dir.mkdir(parents=True, exist_ok=True)
        prefix = self.sanity_dir / f"envstep_{int(env_step):04d}_{camera_name}"
        imageio.imwrite(prefix.with_name(prefix.name + "_before.png"), np.asarray(before)[::-1])
        imageio.imwrite(prefix.with_name(prefix.name + "_after.png"), np.asarray(after)[::-1])
        self._sanity_saved += 1

    def to_dynaprobe_event(
        self,
        *,
        invalid: bool | None = None,
        invalid_reason: str | None = None,
        delta_record: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        base = self.translation_scheduler.to_dynaprobe_event(
            invalid=invalid,
            invalid_reason=invalid_reason,
            delta_record=delta_record,
        )
        base["occlusion_span"] = self.occlusion_span
        base["dose_obs_count"] = int(self.translation_scheduler.b_observation_calls)
        base["target_qpos_by_env_step"] = list(getattr(self.translation_scheduler, "_qpos_trace", []))
        base["occlusion"] = {
            "implementation": "observation_layer_background_fill",
            "camera_obs_keys": dict(self.camera_obs_keys),
            "occlusion_steps": int(self.occlusion_steps),
            "bbox_padding_px": int(self.bbox_padding_px),
            "bbox_padding_frac": float(self.bbox_padding_frac),
            "projection_radius_m": float(self.projection_radius_m),
            "records": list(self.records),
            "failures": list(self.failures),
            "sanity_dir": None if self.sanity_dir is None else str(self.sanity_dir),
        }
        transform = base.setdefault("tuple", {}).setdefault("transform", {})
        transform["occlusion"] = {
            "type": "observation_layer_target_bbox_background_fill",
            "cameras": list(self.camera_obs_keys.keys()),
        }
        base["tuple"]["visibility"] = "target_occluded_during_translation"
        base["tuple"]["transition"] = "occluded_translation_then_visible_static_endpoint"
        validation = base.setdefault("validation", {})
        validation["occlusion_trigger_hit"] = self.occlusion_span is not None
        validation["occlusion_records"] = int(len(self.records))
        validation["occlusion_failures"] = int(len(self.failures))
        validation["occlusion_failure_reason"] = ";".join(self.failures)
        qpos_trace = list(getattr(self.translation_scheduler, "_qpos_trace", []))
        qposes = [np.asarray(row.get("actual_qpos", []), dtype=np.float64) for row in qpos_trace]
        qposes = [item for item in qposes if item.size >= 3 and np.isfinite(item[:3]).all()]
        stationary_max = (
            0.0
            if len(qposes) < 2
            else float(max(np.linalg.norm(item[:3] - qposes[0][:3]) for item in qposes[1:]))
        )
        validation["target_stationary_max_displacement_m"] = stationary_max
        validation["target_stationary"] = bool(stationary_max <= 1e-6)
        if self.multisample_summary is not None:
            base["multisample"] = dict(self.multisample_summary)
            validation["multisample_primary_exact"] = bool(
                self.multisample_summary.get("validation", {}).get("all_primary_reference_exact", False)
            )
        base["severity_level"] = "L2"
        return jsonable(base)
