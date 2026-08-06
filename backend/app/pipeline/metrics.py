from __future__ import annotations

import math

import numpy as np
from scipy.signal import savgol_filter

from app.pipeline.pose_track import TrackBuffer, TrackedPose
from app.schemas import AnalysisResult, MetricPoint, PoseFrame

# COCO indices used for limb speed / energy / pose features
KEY_INDICES = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
REACH_INDICES = [0, 9, 10, 15, 16]  # nose, wrists, ankles
# COCO left↔right pairs for mirror across the vertical axis
MIRROR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
# Bone chains for angle/unit-vector features (parent → child)
BONE_PAIRS = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]
TRAVEL_WINDOW_S = 0.8
# Below this window path (in torso heights), directness is undefined
DIRECTNESS_MIN_PATH = 0.08
# Per-frame torso floor as fraction of track median (prevents near-zero divisors)
SCALE_FLOOR_FRAC = 0.50
DEFAULT_ASPECT = 16.0 / 9.0
TORSO_CONF = 0.2
# Centred Savitzky–Golay ~83 ms (~5 frames @ 60 fps); scale window by fps
SMOOTH_TARGET_MS = 83.0
SMOOTH_POLYORDER = 2
# Joint-confidence gate for limb speed / energy (slightly stricter than torso)
LIMB_CONF = 0.3


def frame_aspect(frame_width: int, frame_height: int) -> float:
    """width/height for square-pixel correction; default 16:9 if unknown."""
    if frame_width > 0 and frame_height > 0:
        return float(frame_width) / float(frame_height)
    return DEFAULT_ASPECT


def to_square_pixels(keypoints: np.ndarray, aspect: float) -> np.ndarray:
    """
    Convert height-normalized image coords to square-pixel space.
    Stored keypoints are x/W, y/H; multiply x by W/H so 1 unit = 1 frame-height.
    """
    out = np.asarray(keypoints, dtype=np.float64).copy()
    if out.ndim != 2 or out.shape[1] < 2:
        return out
    out[:, 0] = out[:, 0] * float(aspect)
    return out


def mirror_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """Reflect pose across vertical axis through torso centre; swap L/R joints."""
    if keypoints.shape[0] < 17:
        return keypoints.copy()
    out = keypoints.copy()
    ls, rs = out[5], out[6]
    lh, rh = out[11], out[12]
    if min(float(ls[2]), float(rs[2]), float(lh[2]), float(rh[2])) < TORSO_CONF:
        cx = float(np.mean(out[:, 0]))
    else:
        cx = float((ls[0] + rs[0] + lh[0] + rh[0]) / 4)
    out[:, 0] = (2.0 * cx) - out[:, 0]
    for a, b in MIRROR_PAIRS:
        tmp = out[a].copy()
        out[a] = out[b]
        out[b] = tmp
    return out


def _torso_parts(sq: np.ndarray) -> tuple[float, float, float, float, float] | None:
    """
    Square-pixel keypoints → (cx, cy, torso_h, shoulder_w, hip_y) or None.
    torso_h = |shoulder_mid − hip_mid|; shoulder_w for fallback only.
    """
    if sq.shape[0] < 17:
        return None
    ls, rs = sq[5], sq[6]
    lh, rh = sq[11], sq[12]
    if min(float(ls[2]), float(rs[2]), float(lh[2]), float(rh[2])) < TORSO_CONF:
        return None
    sh_x = 0.5 * (float(ls[0]) + float(rs[0]))
    sh_y = 0.5 * (float(ls[1]) + float(rs[1]))
    hip_x = 0.5 * (float(lh[0]) + float(rh[0]))
    hip_y = 0.5 * (float(lh[1]) + float(rh[1]))
    torso_h = float(np.hypot(sh_x - hip_x, sh_y - hip_y))
    shoulder_w = float(np.hypot(float(ls[0] - rs[0]), float(ls[1] - rs[1])))
    cx = float((ls[0] + rs[0] + lh[0] + rh[0]) / 4)
    cy = float((ls[1] + rs[1] + lh[1] + rh[1]) / 4)
    return cx, cy, torso_h, shoulder_w, hip_y


def measure_scale(
    keypoints: np.ndarray,
    *,
    aspect: float,
    scale_ref: float | None = None,
) -> tuple[float, float, float, float] | None:
    """
    Return (cx, cy, scale, hip_y) in square-pixel space, or None if degenerate.
    scale = max(torso_h, floor); floor from track median when available.
    Falls back to shoulder width (also floored) when hips unusable but shoulders ok.
    """
    sq = to_square_pixels(keypoints, aspect)
    parts = _torso_parts(sq)
    if parts is None:
        # Shoulder-only fallback
        if sq.shape[0] < 7:
            return None
        ls, rs = sq[5], sq[6]
        if min(float(ls[2]), float(rs[2])) < TORSO_CONF:
            return None
        cx = 0.5 * (float(ls[0]) + float(rs[0]))
        cy = 0.5 * (float(ls[1]) + float(rs[1]))
        shoulder_w = float(np.hypot(float(ls[0] - rs[0]), float(ls[1] - rs[1])))
        floor = (SCALE_FLOOR_FRAC * scale_ref) if scale_ref and scale_ref > 0 else 1e-3
        scale = max(shoulder_w, floor)
        if scale < floor * 0.99 and (scale_ref is None or scale_ref <= 0):
            return None
        return cx, cy, scale, cy
    cx, cy, torso_h, shoulder_w, hip_y = parts
    floor = (SCALE_FLOOR_FRAC * scale_ref) if scale_ref and scale_ref > 0 else max(torso_h * 0.5, 1e-3)
    if torso_h >= floor * 0.5:  # usable torso
        scale = max(torso_h, floor)
    else:
        # Near-collapsed torso — try floored shoulder, else reject
        scale = max(shoulder_w, floor)
        if scale <= floor and torso_h < 1e-4:
            return None
        scale = max(scale, floor)
    return cx, cy, scale, hip_y


def track_scale_ref(poses: list, aspect: float) -> float:
    """Median torso height over the track (square-pixel); fallback shoulder median."""
    torsos: list[float] = []
    shoulders: list[float] = []
    for p in poses:
        k = getattr(p, "keypoints", p)
        sq = to_square_pixels(k, aspect)
        parts = _torso_parts(sq)
        if parts is None:
            continue
        _cx, _cy, torso_h, shoulder_w, _hip = parts
        if torso_h > 1e-4:
            torsos.append(torso_h)
        if shoulder_w > 1e-4:
            shoulders.append(shoulder_w)
    if torsos:
        return float(max(np.median(np.asarray(torsos, dtype=np.float64)), 1e-3))
    if shoulders:
        return float(max(np.median(np.asarray(shoulders, dtype=np.float64)), 1e-3))
    return 0.1  # last resort in square-pixel height units


def pose_feature(
    keypoints: np.ndarray,
    *,
    aspect: float = DEFAULT_ASPECT,
    scale_ref: float | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Torso-normalized XY feature + per-dim visibility mask.
    Occluded joints are masked False (coords left 0 — never treated as torso-centre evidence).
    """
    if keypoints.shape[0] < 17:
        return None
    measured = measure_scale(keypoints, aspect=aspect, scale_ref=scale_ref)
    if measured is None:
        return None
    cx, cy, scale, _hip = measured
    sq = to_square_pixels(keypoints, aspect)

    feats: list[float] = []
    mask: list[bool] = []
    for i in KEY_INDICES:
        p = sq[i]
        if float(p[2]) < TORSO_CONF:
            feats.extend([0.0, 0.0])
            mask.extend([False, False])
        else:
            feats.extend([(float(p[0]) - cx) / scale, (float(p[1]) - cy) / scale])
            mask.extend([True, True])
    return (
        np.asarray(feats, dtype=np.float32),
        np.asarray(mask, dtype=bool),
    )


def pose_feature_mirrored(
    keypoints: np.ndarray,
    *,
    aspect: float = DEFAULT_ASPECT,
    scale_ref: float | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Left↔right mirror of pose_feature (same gesture on the other side)."""
    return pose_feature(mirror_keypoints(keypoints), aspect=aspect, scale_ref=scale_ref)


def pose_feature_rich(
    keypoints: np.ndarray,
    *,
    aspect: float = DEFAULT_ASPECT,
    scale_ref: float | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Torso-normalized limb XY + unit bone vectors, with per-dim visibility.
    Used as DTW frame features (not L2-normalized as a whole).
    """
    base = pose_feature(keypoints, aspect=aspect, scale_ref=scale_ref)
    if base is None:
        return None
    base_feat, base_mask = base
    measured = measure_scale(keypoints, aspect=aspect, scale_ref=scale_ref)
    if measured is None:
        return None
    cx, cy, scale, _hip = measured
    sq = to_square_pixels(keypoints, aspect)

    bones: list[float] = []
    bone_mask: list[bool] = []
    for a, b in BONE_PAIRS:
        pa, pb = sq[a], sq[b]
        if float(pa[2]) < TORSO_CONF or float(pb[2]) < TORSO_CONF:
            bones.extend([0.0, 0.0])
            bone_mask.extend([False, False])
            continue
        dx = (float(pb[0]) - float(pa[0])) / scale
        dy = (float(pb[1]) - float(pa[1])) / scale
        n = float(np.hypot(dx, dy)) or 1.0
        bones.extend([dx / n, dy / n])
        bone_mask.extend([True, True])
    return (
        np.concatenate([base_feat, np.asarray(bones, dtype=np.float32)]),
        np.concatenate([base_mask, np.asarray(bone_mask, dtype=bool)]),
    )


def pose_feature_rich_mirrored(
    keypoints: np.ndarray,
    *,
    aspect: float = DEFAULT_ASPECT,
    scale_ref: float | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    return pose_feature_rich(
        mirror_keypoints(keypoints), aspect=aspect, scale_ref=scale_ref
    )


def estimate_fps(times: list[float] | np.ndarray) -> float:
    """Median 1/Δt from a time series; default 30 if unknown."""
    t = np.asarray(times, dtype=np.float64)
    if len(t) < 2:
        return 30.0
    dts = np.diff(t)
    dts = dts[dts > 1e-4]
    if len(dts) == 0:
        return 30.0
    return float(1.0 / np.median(dts))


def smooth_window_len(n: int, fps: float) -> int:
    """Odd Savitzky–Golay window ≈ SMOOTH_TARGET_MS, clamped to series length."""
    if n < SMOOTH_POLYORDER + 2:
        return 0
    w = int(round(float(fps) * SMOOTH_TARGET_MS / 1000.0))
    if w % 2 == 0:
        w += 1
    w = max(5, w)
    max_odd = n if n % 2 == 1 else n - 1
    w = min(w, max_odd)
    if w <= SMOOTH_POLYORDER:
        return 0
    return int(w)


def smooth_series(values: list[float] | np.ndarray, *, fps: float) -> np.ndarray:
    """
    Centred Savitzky–Golay on a 1-D metric curve.
    Same filter is used for displayed charts and motif segmentation (Wave 2).
    """
    x = np.asarray(values, dtype=np.float64)
    w = smooth_window_len(len(x), fps)
    if w == 0:
        return x.copy()
    return savgol_filter(x, window_length=w, polyorder=SMOOTH_POLYORDER)


def limb_energy_and_speed(
    prev_kpts: np.ndarray,
    cur_kpts: np.ndarray,
    dt: float,
    *,
    aspect: float,
    scale: float,
) -> tuple[float, float]:
    """
    Scale-normalized energy and speed over KEY_INDICES only.

    Energy accumulates (v/scale)^2 — unit torso²/s² before per-actor peak
    normalization. Scale-normalization makes cross-clip comparison possible later;
    charts still display energy relative to the actor's own peak.
    Speed is mean |v|/scale in torso/s.
    """
    dt = max(dt, 1e-3)
    scale = max(float(scale), 1e-3)
    prev_sq = to_square_pixels(prev_kpts, aspect)
    cur_sq = to_square_pixels(cur_kpts, aspect)
    energy = 0.0
    speed_sum = 0.0
    speed_n = 0
    n = min(len(prev_sq), len(cur_sq))
    for j in KEY_INDICES:
        if j >= n:
            continue
        a = prev_sq[j]
        b = cur_sq[j]
        if float(a[2]) < LIMB_CONF or float(b[2]) < LIMB_CONF:
            continue
        v = float(np.hypot(float(b[0] - a[0]), float(b[1] - a[1])) / dt)
        v_n = v / scale
        energy += v_n * v_n
        speed_sum += v_n
        speed_n += 1
    speed = speed_sum / speed_n if speed_n else 0.0
    return float(energy), float(speed)


def _reach_raw(sq: np.ndarray, cx: float, cy: float, scale: float) -> float:
    best = 0.0
    for i in REACH_INDICES:
        p = sq[i]
        if float(p[2]) < TORSO_CONF:
            continue
        d = float(np.hypot(float(p[0]) - cx, float(p[1]) - cy)) / scale
        best = max(best, d)
    return best


def _foot_y_norm(keypoints: np.ndarray, fallback_y: float) -> float:
    """Foot y in original normalized image coords (for floor map)."""
    if keypoints.shape[0] < 17:
        return fallback_y
    candidates: list[tuple[float, float]] = []
    for i in (15, 16):
        p = keypoints[i]
        conf = float(p[2])
        if conf >= TORSO_CONF:
            candidates.append((float(p[1]), conf))
    if not candidates:
        return fallback_y
    candidates.sort(key=lambda t: (t[0], t[1]))
    return candidates[-1][0]


def _leg_length_sq(sq: np.ndarray, hip_y: float) -> float | None:
    """Mean |ankle_y - hip_y| in square-pixel space."""
    if sq.shape[0] < 17:
        return None
    lengths: list[float] = []
    for i in (15, 16):
        p = sq[i]
        if float(p[2]) < TORSO_CONF:
            continue
        lengths.append(abs(float(p[1]) - hip_y))
    if not lengths:
        return None
    return float(sum(lengths) / len(lengths))


def compute_metrics(
    buffer: TrackBuffer,
    *,
    aspect: float = DEFAULT_ASPECT,
) -> list[MetricPoint]:
    poses = buffer.poses
    if len(poses) < 2:
        return []

    scale_ref = track_scale_ref(poses, aspect)

    times: list[float] = []
    energy_r: list[float] = []
    speed_th: list[float] = []  # torso heights / second
    reach_th: list[float] = []  # torso multiples
    leg_r: list[float] = []
    pos_xs: list[float] = []  # original normalized (floor map)
    pos_ys: list[float] = []
    scales: list[float] = []
    valid_flags: list[bool] = []

    last_reach = 1.0
    last_leg: float | None = None
    last_pos = (0.5, 0.5)
    last_scale = scale_ref
    last_speed = 0.0

    for i in range(1, len(poses)):
        prev = poses[i - 1]
        cur = poses[i]
        dt = max(cur.time - prev.time, 1e-3)

        cur_sq = to_square_pixels(cur.keypoints, aspect)

        measured = measure_scale(cur.keypoints, aspect=aspect, scale_ref=scale_ref)
        if measured is None:
            reach = last_reach
            leg = last_leg
            pos_x, pos_y = last_pos
            scale = last_scale
            energy, speed_n_out = limb_energy_and_speed(
                prev.keypoints,
                cur.keypoints,
                dt,
                aspect=aspect,
                scale=last_scale,
            )
            # Carry last speed on fully degenerate frames; keep energy for valleys
            if last_scale <= 0:
                speed_n_out = last_speed
            valid = False
        else:
            cx, cy, scale, hip_y = measured
            reach = _reach_raw(cur_sq, cx, cy, scale)
            leg = _leg_length_sq(cur_sq, hip_y)
            # Floor-map positions stay in original normalized coords
            ls, rs = cur.keypoints[5], cur.keypoints[6]
            lh, rh = cur.keypoints[11], cur.keypoints[12]
            pos_x = float((ls[0] + rs[0] + lh[0] + rh[0]) / 4)
            pos_y = _foot_y_norm(cur.keypoints, float((ls[1] + rs[1] + lh[1] + rh[1]) / 4))
            energy, speed_n_out = limb_energy_and_speed(
                prev.keypoints,
                cur.keypoints,
                dt,
                aspect=aspect,
                scale=scale,
            )
            last_reach = reach
            if leg is not None:
                last_leg = leg
            last_pos = (pos_x, pos_y)
            last_scale = scale
            last_speed = speed_n_out
            valid = True

        times.append(cur.time)
        energy_r.append(float(energy))
        speed_th.append(float(speed_n_out))
        reach_th.append(float(reach))
        leg_r.append(float(leg) if leg is not None else float("nan"))
        pos_xs.append(float(np.clip(pos_x, 0.0, 1.0)))
        pos_ys.append(float(np.clip(pos_y, 0.0, 1.0)))
        scales.append(float(max(scale, 1e-3)))
        valid_flags.append(valid)

    fps = estimate_fps(times)
    energy_r = smooth_series(energy_r, fps=fps).tolist()
    speed_th = smooth_series(speed_th, fps=fps).tolist()

    valid_legs = [v for v in leg_r if not math.isnan(v)]
    if valid_legs:
        leg_ref = float(np.percentile(np.asarray(valid_legs, dtype=np.float64), 90))
        leg_ref = max(leg_ref, 1e-4)
    else:
        leg_ref = 1.0

    level_n: list[float] = []
    last_level = 1.0
    for v in leg_r:
        if math.isnan(v):
            level_n.append(last_level)
        else:
            lv = float(np.clip(v / leg_ref, 0.0, 1.0))
            last_level = lv
            level_n.append(lv)

    # Travel / directness on foot track — path in square-pixel space
    travel_th_s: list[float] = []
    direct_r: list[float] = []
    for i, t in enumerate(times):
        t0 = t - TRAVEL_WINDOW_S
        j0 = i
        while j0 > 0 and times[j0 - 1] >= t0:
            j0 -= 1
        path = 0.0
        for k in range(j0 + 1, i + 1):
            # pos stored normalized; convert delta x by aspect for path length
            dx = (pos_xs[k] - pos_xs[k - 1]) * aspect
            dy = pos_ys[k] - pos_ys[k - 1]
            path += float(np.hypot(dx, dy))
        net = float(
            np.hypot(
                (pos_xs[i] - pos_xs[j0]) * aspect,
                pos_ys[i] - pos_ys[j0],
            )
        )
        win_scale = float(np.mean(scales[j0 : i + 1])) if i >= j0 else scales[i]
        win_scale = max(win_scale, SCALE_FLOOR_FRAC * scale_ref, 1e-3)
        win_dt = max(times[i] - times[j0], 1e-3) if i > j0 else TRAVEL_WINDOW_S
        path_th = path / win_scale
        travel_th_s.append(float(path_th / win_dt))
        if path_th < DIRECTNESS_MIN_PATH:
            direct_r.append(float("nan"))
        else:
            direct_r.append(float(np.clip(net / path, 0.0, 1.0)))

    e_max = max(energy_r) if energy_r else 0.0
    if e_max <= 1e-9:
        energy_n = [0.0] * len(energy_r)
    else:
        # Peak-relative for display; absolute (torso²/s²) available before this step
        energy_n = [float(v / e_max) for v in energy_r]

    return [
        MetricPoint(
            time=times[i],
            energy=energy_n[i],
            speed=speed_th[i],
            reach=reach_th[i],
            level=level_n[i],
            travel=travel_th_s[i],
            directness=direct_r[i],
            pos_x=pos_xs[i],
            pos_y=pos_ys[i],
        )
        for i in range(len(times))
    ]


def _actor_to_buffer(actor_id: str, poses: list[PoseFrame]) -> TrackBuffer:
    tid = 0
    if actor_id.startswith("track_"):
        try:
            tid = int(actor_id.split("_", 1)[1])
        except ValueError:
            tid = 0
    buf = TrackBuffer(track_id=tid)
    for pf in poses:
        kpts = np.asarray(pf.keypoints, dtype=np.float32)
        if kpts.ndim != 2 or kpts.shape[1] < 3:
            continue
        if kpts.shape[1] > 3:
            kpts = kpts[:, :3]
        buf.poses.append(
            TrackedPose(
                time=float(pf.time),
                track_id=tid,
                keypoints=kpts,
                xyxy=np.zeros(4, dtype=np.float32),
            )
        )
    return buf


def enrich_metrics(result: AnalysisResult) -> AnalysisResult:
    """Recompute all metrics (incl. space + position) from stored poses. Mutates actors."""
    aspect = frame_aspect(int(result.frame_width or 0), int(result.frame_height or 0))
    for actor in result.actors:
        if not actor.poses:
            continue
        buf = _actor_to_buffer(actor.id, actor.poses)
        if len(buf.poses) < 2:
            continue
        actor.metrics = compute_metrics(buf, aspect=aspect)
    return result
