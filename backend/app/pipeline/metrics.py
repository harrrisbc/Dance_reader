from __future__ import annotations

import math

import numpy as np
from scipy.signal import savgol_filter

from app.pipeline.pose_track import TrackBuffer, TrackedPose
from app.pipeline.smpl_skeleton import smpl_keypoints_to_coco17
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

# SMPL-24 (gvhmr_world, Y-up) — parallel sets for smpl3d metrics / motif features
SMPL_PELVIS = 0
SMPL_L_HIP, SMPL_R_HIP = 1, 2
SMPL_L_KNEE, SMPL_R_KNEE = 4, 5
SMPL_L_ANKLE, SMPL_R_ANKLE = 7, 8
SMPL_HEAD = 15
SMPL_L_SHOULDER, SMPL_R_SHOULDER = 16, 17
SMPL_L_ELBOW, SMPL_R_ELBOW = 18, 19
SMPL_L_WRIST, SMPL_R_WRIST = 20, 21
SMPL_KEY_INDICES = [
    SMPL_L_SHOULDER,
    SMPL_R_SHOULDER,
    SMPL_L_ELBOW,
    SMPL_R_ELBOW,
    SMPL_L_WRIST,
    SMPL_R_WRIST,
    SMPL_L_HIP,
    SMPL_R_HIP,
    SMPL_L_KNEE,
    SMPL_R_KNEE,
    SMPL_L_ANKLE,
    SMPL_R_ANKLE,
]
SMPL_REACH_INDICES = [SMPL_HEAD, SMPL_L_WRIST, SMPL_R_WRIST, SMPL_L_ANKLE, SMPL_R_ANKLE]
SMPL_MIRROR_PAIRS = [
    (SMPL_L_HIP, SMPL_R_HIP),
    (SMPL_L_KNEE, SMPL_R_KNEE),
    (SMPL_L_ANKLE, SMPL_R_ANKLE),
    (SMPL_L_SHOULDER, SMPL_R_SHOULDER),
    (SMPL_L_ELBOW, SMPL_R_ELBOW),
    (SMPL_L_WRIST, SMPL_R_WRIST),
]
SMPL_BONE_PAIRS = [
    (SMPL_L_SHOULDER, SMPL_L_ELBOW),
    (SMPL_L_ELBOW, SMPL_L_WRIST),
    (SMPL_R_SHOULDER, SMPL_R_ELBOW),
    (SMPL_R_ELBOW, SMPL_R_WRIST),
    (SMPL_L_HIP, SMPL_L_KNEE),
    (SMPL_L_KNEE, SMPL_L_ANKLE),
    (SMPL_R_HIP, SMPL_R_KNEE),
    (SMPL_R_KNEE, SMPL_R_ANKLE),
]
TRAVEL_WINDOW_S = 0.8
# Level bands from 武拜廟 smpl3d (n=968): bow min 1.22, p50 1.46, standing p75 1.52, max 1.59.
# No dedicated deep-plié clip — values below the bow trough are "low".
LEVEL_BAND_LOW = 1.25
LEVEL_BAND_HIGH = 1.50
LEVEL_CHART_YLIM = (1.0, 1.75)
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
    """Deprecated alias — use level_stance_ratio_2d. Kept for callers/tests."""
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


def level_stance_ratio_2d(sq: np.ndarray, scale: float) -> float | None:
    """Hip → supporting (lower) ankle vertical drop / torso scale (image Y down)."""
    if scale <= 1e-6 or sq.shape[0] < 17:
        return None
    lh, rh = sq[11], sq[12]
    if min(float(lh[2]), float(rh[2])) < TORSO_CONF:
        return None
    hip_y = 0.5 * (float(lh[1]) + float(rh[1]))
    ankles: list[float] = []
    for i in (15, 16):
        if float(sq[i, 2]) >= TORSO_CONF:
            ankles.append(float(sq[i, 1]))
    if not ankles:
        return None
    # Supporting = lower in the frame = larger image Y
    drop = abs(max(ankles) - hip_y)
    return float(drop / scale)


def _joints3d_arr(j: np.ndarray | list) -> np.ndarray | None:
    a = np.asarray(j, dtype=np.float64)
    if a.ndim != 2 or a.shape != (24, 3):
        return None
    return a


def torso_scale_3d(j: np.ndarray) -> float | None:
    """Shoulder-mid → hip-mid length in metres."""
    sh = 0.5 * (j[SMPL_L_SHOULDER] + j[SMPL_R_SHOULDER])
    hip = 0.5 * (j[SMPL_L_HIP] + j[SMPL_R_HIP])
    s = float(np.linalg.norm(sh - hip))
    if s < 1e-4:
        return None
    return s


def track_scale_ref_3d(poses: list[TrackedPose]) -> float:
    scales: list[float] = []
    for p in poses:
        if p.joints3d is None:
            continue
        j = _joints3d_arr(p.joints3d)
        if j is None:
            continue
        s = torso_scale_3d(j)
        if s is not None:
            scales.append(s)
    if not scales:
        return 0.5
    return float(max(np.median(np.asarray(scales, dtype=np.float64)), 1e-3))


def level_stance_ratio_3d(j: np.ndarray, scale: float) -> float | None:
    """Y-up: (hip_y − supporting_ankle_y) / S."""
    if scale <= 1e-6:
        return None
    hip_y = 0.5 * (float(j[SMPL_L_HIP, 1]) + float(j[SMPL_R_HIP, 1]))
    ay_l = float(j[SMPL_L_ANKLE, 1])
    ay_r = float(j[SMPL_R_ANKLE, 1])
    supporting_y = min(ay_l, ay_r)
    drop = hip_y - supporting_y
    return float(drop / scale)


def supporting_ankle_xz(j: np.ndarray) -> tuple[float, float]:
    """World XZ of the lower ankle (Y-up)."""
    if float(j[SMPL_L_ANKLE, 1]) <= float(j[SMPL_R_ANKLE, 1]):
        a = j[SMPL_L_ANKLE]
    else:
        a = j[SMPL_R_ANKLE]
    return float(a[0]), float(a[2])


def mirror_joints3d(j: np.ndarray) -> np.ndarray:
    """Reflect across vertical plane through pelvis; swap L/R."""
    out = np.asarray(j, dtype=np.float64).copy()
    cx = float(out[SMPL_PELVIS, 0])
    out[:, 0] = (2.0 * cx) - out[:, 0]
    for a, b in SMPL_MIRROR_PAIRS:
        tmp = out[a].copy()
        out[a] = out[b]
        out[b] = tmp
    return out


def limb_energy_and_speed_3d(
    prev_j: np.ndarray,
    cur_j: np.ndarray,
    dt: float,
    *,
    scale: float,
) -> tuple[float, float]:
    if scale <= 1e-6 or dt <= 1e-6:
        return 0.0, 0.0
    energy = 0.0
    speed_sum = 0.0
    n = 0
    for i in SMPL_KEY_INDICES:
        v = float(np.linalg.norm(cur_j[i] - prev_j[i]) / dt)
        v_n = v / scale
        energy += v_n * v_n
        speed_sum += v_n
        n += 1
    speed = speed_sum / n if n else 0.0
    return float(energy), float(speed)


def _reach_raw_3d(j: np.ndarray, centre: np.ndarray, scale: float) -> float:
    best = 0.0
    for i in SMPL_REACH_INDICES:
        d = float(np.linalg.norm(j[i] - centre)) / scale
        best = max(best, d)
    return best


def pose_feature_3d(
    joints: np.ndarray,
    *,
    scale_ref: float | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Torso-normalized XYZ for SMPL_KEY_INDICES + visibility (always True if finite)."""
    j = _joints3d_arr(joints)
    if j is None:
        return None
    s = torso_scale_3d(j)
    if s is None:
        return None
    if scale_ref is not None and scale_ref > 0:
        s = max(s, SCALE_FLOOR_FRAC * float(scale_ref))
    centre = 0.25 * (
        j[SMPL_L_SHOULDER] + j[SMPL_R_SHOULDER] + j[SMPL_L_HIP] + j[SMPL_R_HIP]
    )
    feats: list[float] = []
    mask: list[bool] = []
    for i in SMPL_KEY_INDICES:
        rel = (j[i] - centre) / s
        if not np.all(np.isfinite(rel)):
            feats.extend([0.0, 0.0, 0.0])
            mask.extend([False, False, False])
        else:
            feats.extend([float(rel[0]), float(rel[1]), float(rel[2])])
            mask.extend([True, True, True])
    return (
        np.asarray(feats, dtype=np.float32),
        np.asarray(mask, dtype=bool),
    )


def pose_feature_rich_3d(
    joints: np.ndarray,
    *,
    scale_ref: float | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    base = pose_feature_3d(joints, scale_ref=scale_ref)
    if base is None:
        return None
    base_feat, base_mask = base
    j = _joints3d_arr(joints)
    assert j is not None
    s = torso_scale_3d(j)
    if s is None:
        return None
    if scale_ref is not None and scale_ref > 0:
        s = max(s, SCALE_FLOOR_FRAC * float(scale_ref))
    bones: list[float] = []
    bone_mask: list[bool] = []
    for a, b in SMPL_BONE_PAIRS:
        d = j[b] - j[a]
        n = float(np.linalg.norm(d)) or 1.0
        u = d / n
        bones.extend([float(u[0]), float(u[1]), float(u[2])])
        bone_mask.extend([True, True, True])
    return (
        np.concatenate([base_feat, np.asarray(bones, dtype=np.float32)]),
        np.concatenate([base_mask, np.asarray(bone_mask, dtype=bool)]),
    )


def pose_feature_rich_3d_mirrored(
    joints: np.ndarray,
    *,
    scale_ref: float | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    j = _joints3d_arr(joints)
    if j is None:
        return None
    return pose_feature_rich_3d(mirror_joints3d(j), scale_ref=scale_ref)


def _travel_directness_series(
    times: list[float],
    xs: list[float],
    zs: list[float],
    scales: list[float],
    scale_ref: float,
    *,
    x_scale: float = 1.0,
) -> tuple[list[float], list[float]]:
    """Rolling path travel (torso/s) and directness on a 2D ground track."""
    travel_th_s: list[float] = []
    direct_r: list[float] = []
    for i, t in enumerate(times):
        t0 = t - TRAVEL_WINDOW_S
        j0 = i
        while j0 > 0 and times[j0 - 1] >= t0:
            j0 -= 1
        path = 0.0
        for k in range(j0 + 1, i + 1):
            dx = (xs[k] - xs[k - 1]) * x_scale
            dz = zs[k] - zs[k - 1]
            path += float(np.hypot(dx, dz))
        net = float(
            np.hypot((xs[i] - xs[j0]) * x_scale, zs[i] - zs[j0])
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
    return travel_th_s, direct_r


def compute_metrics(
    buffer: TrackBuffer,
    *,
    aspect: float = DEFAULT_ASPECT,
) -> list[MetricPoint]:
    """2D / coco2d metrics from image keypoints."""
    poses = buffer.poses
    if len(poses) < 2:
        return []

    scale_ref = track_scale_ref(poses, aspect)

    times: list[float] = []
    energy_r: list[float] = []
    speed_th: list[float] = []
    reach_th: list[float] = []
    level_n: list[float] = []
    pos_xs: list[float] = []
    pos_ys: list[float] = []
    pos_foot_xs: list[float] = []
    pos_foot_ys: list[float] = []
    scales: list[float] = []

    last_reach = 1.0
    last_level = 1.0
    last_pos = (0.5, 0.5)
    last_foot = (0.5, 0.5)
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
            level = last_level
            pos_x, pos_y = last_pos
            foot_x, foot_y = last_foot
            scale = last_scale
            energy, speed_n_out = limb_energy_and_speed(
                prev.keypoints, cur.keypoints, dt, aspect=aspect, scale=last_scale
            )
            if last_scale <= 0:
                speed_n_out = last_speed
        else:
            cx, cy, scale, _hip = measured
            reach = _reach_raw(cur_sq, cx, cy, scale)
            lv = level_stance_ratio_2d(cur_sq, scale)
            level = float(lv) if lv is not None else last_level
            ls, rs = cur.keypoints[5], cur.keypoints[6]
            lh, rh = cur.keypoints[11], cur.keypoints[12]
            pos_x = float((ls[0] + rs[0] + lh[0] + rh[0]) / 4)
            pos_y = _foot_y_norm(
                cur.keypoints, float((ls[1] + rs[1] + lh[1] + rh[1]) / 4)
            )
            foot_x, foot_y = pos_x, pos_y
            ank: list[tuple[float, float, float]] = []
            for ai in (15, 16):
                if float(cur.keypoints[ai, 2]) >= TORSO_CONF:
                    ank.append(
                        (
                            float(cur.keypoints[ai, 0]),
                            float(cur.keypoints[ai, 1]),
                            float(cur.keypoints[ai, 2]),
                        )
                    )
            if ank:
                ank.sort(key=lambda t: (t[1], t[2]))
                foot_x, foot_y = ank[-1][0], ank[-1][1]
            energy, speed_n_out = limb_energy_and_speed(
                prev.keypoints, cur.keypoints, dt, aspect=aspect, scale=scale
            )
            last_reach = reach
            last_level = level
            last_pos = (pos_x, pos_y)
            last_foot = (foot_x, foot_y)
            last_scale = scale
            last_speed = speed_n_out

        times.append(cur.time)
        energy_r.append(float(energy))
        speed_th.append(float(speed_n_out))
        reach_th.append(float(reach))
        level_n.append(float(level))
        pos_xs.append(float(np.clip(pos_x, 0.0, 1.0)))
        pos_ys.append(float(np.clip(pos_y, 0.0, 1.0)))
        pos_foot_xs.append(float(np.clip(foot_x, 0.0, 1.0)))
        pos_foot_ys.append(float(np.clip(foot_y, 0.0, 1.0)))
        scales.append(float(max(scale, 1e-3)))

    fps = estimate_fps(times)
    energy_r = smooth_series(energy_r, fps=fps).tolist()
    speed_th = smooth_series(speed_th, fps=fps).tolist()
    travel_th_s, direct_r = _travel_directness_series(
        times, pos_xs, pos_ys, scales, scale_ref, x_scale=aspect
    )
    e_max = max(energy_r) if energy_r else 0.0
    energy_n = (
        [0.0] * len(energy_r)
        if e_max <= 1e-9
        else [float(v / e_max) for v in energy_r]
    )
    return [
        MetricPoint(
            time=times[i],
            energy=energy_n[i],
            energy_abs=float(energy_r[i]),
            speed=speed_th[i],
            reach=reach_th[i],
            level=level_n[i],
            travel=travel_th_s[i],
            directness=direct_r[i],
            pos_x=pos_xs[i],
            pos_y=pos_ys[i],
            pos_foot_x=pos_foot_xs[i],
            pos_foot_y=pos_foot_ys[i],
        )
        for i in range(len(times))
    ]


def compute_metrics_smpl3d(buffer: TrackBuffer) -> list[MetricPoint]:
    """World-metre metrics from TrackedPose.joints3d (Y-up gvhmr_world)."""
    poses = [p for p in buffer.poses if p.joints3d is not None]
    if len(poses) < 2:
        return []

    scale_ref = track_scale_ref_3d(poses)
    times: list[float] = []
    energy_r: list[float] = []
    speed_th: list[float] = []
    reach_th: list[float] = []
    level_n: list[float] = []
    pos_xs: list[float] = []
    pos_ys: list[float] = []
    pos_foot_xs: list[float] = []
    pos_foot_ys: list[float] = []
    scales: list[float] = []

    last_reach = 1.0
    last_level = 1.0
    last_pos = (0.0, 0.0)
    last_foot = (0.0, 0.0)
    last_scale = scale_ref
    last_speed = 0.0

    for i in range(1, len(poses)):
        prev, cur = poses[i - 1], poses[i]
        dt = max(cur.time - prev.time, 1e-3)
        prev_j = _joints3d_arr(prev.joints3d)
        cur_j = _joints3d_arr(cur.joints3d)
        if prev_j is None or cur_j is None:
            continue
        s = torso_scale_3d(cur_j)
        if s is None:
            scale = last_scale
            reach = last_reach
            level = last_level
            px, pz = last_pos
            fx, fz = last_foot
            energy, speed_n_out = limb_energy_and_speed_3d(
                prev_j, cur_j, dt, scale=last_scale
            )
            if last_scale <= 0:
                speed_n_out = last_speed
        else:
            scale = max(s, SCALE_FLOOR_FRAC * scale_ref)
            centre = 0.25 * (
                cur_j[SMPL_L_SHOULDER]
                + cur_j[SMPL_R_SHOULDER]
                + cur_j[SMPL_L_HIP]
                + cur_j[SMPL_R_HIP]
            )
            reach = _reach_raw_3d(cur_j, centre, scale)
            lv = level_stance_ratio_3d(cur_j, scale)
            level = float(lv) if lv is not None else last_level
            px = float(cur_j[SMPL_PELVIS, 0])
            pz = float(cur_j[SMPL_PELVIS, 2])
            fx, fz = supporting_ankle_xz(cur_j)
            energy, speed_n_out = limb_energy_and_speed_3d(
                prev_j, cur_j, dt, scale=scale
            )
            last_reach = reach
            last_level = level
            last_pos = (px, pz)
            last_foot = (fx, fz)
            last_scale = scale
            last_speed = speed_n_out

        times.append(cur.time)
        energy_r.append(float(energy))
        speed_th.append(float(speed_n_out))
        reach_th.append(float(reach))
        level_n.append(float(level))
        pos_xs.append(float(px))
        pos_ys.append(float(pz))
        pos_foot_xs.append(float(fx))
        pos_foot_ys.append(float(fz))
        scales.append(float(max(scale, 1e-3)))

    if len(times) < 1:
        return []

    fps = estimate_fps(times)
    energy_r = smooth_series(energy_r, fps=fps).tolist()
    speed_th = smooth_series(speed_th, fps=fps).tolist()
    travel_th_s, direct_r = _travel_directness_series(
        times, pos_xs, pos_ys, scales, scale_ref, x_scale=1.0
    )
    # Floor origin = first pelvis so the map starts at (0, 0). Deltas unchanged.
    ox, oz = pos_xs[0], pos_ys[0]
    pos_xs = [x - ox for x in pos_xs]
    pos_ys = [z - oz for z in pos_ys]
    pos_foot_xs = [x - ox for x in pos_foot_xs]
    pos_foot_ys = [z - oz for z in pos_foot_ys]
    e_max = max(energy_r) if energy_r else 0.0
    energy_n = (
        [0.0] * len(energy_r)
        if e_max <= 1e-9
        else [float(v / e_max) for v in energy_r]
    )
    return [
        MetricPoint(
            time=times[i],
            energy=energy_n[i],
            energy_abs=float(energy_r[i]),
            speed=speed_th[i],
            reach=reach_th[i],
            level=level_n[i],
            travel=travel_th_s[i],
            directness=direct_r[i],
            pos_x=pos_xs[i],
            pos_y=pos_ys[i],
            pos_foot_x=pos_foot_xs[i],
            pos_foot_y=pos_foot_ys[i],
        )
        for i in range(len(times))
    ]


def actor_has_joints3d(poses: list[PoseFrame]) -> bool:
    return any(
        p.joints3d is not None and len(p.joints3d) == 24 for p in poses
    )


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
        j3d = None
        if pf.joints3d is not None and len(pf.joints3d) == 24:
            j3d = np.asarray(pf.joints3d, dtype=np.float32)
        elif kpts.shape[0] == 24:
            # Legacy 24-point without joints3d: bridge for coco2d only
            kpts = np.asarray(smpl_keypoints_to_coco17(kpts.tolist()), dtype=np.float32)
        buf.poses.append(
            TrackedPose(
                time=float(pf.time),
                track_id=tid,
                keypoints=kpts,
                xyxy=np.zeros(4, dtype=np.float32),
                joints3d=j3d,
            )
        )
    return buf


def enrich_metrics(result: AnalysisResult) -> AnalysisResult:
    """Recompute all metrics from stored poses. Sets metricsSpace. Mutates actors."""
    aspect = frame_aspect(int(result.frame_width or 0), int(result.frame_height or 0))
    any_3d = False
    for actor in result.actors:
        if not actor.poses:
            continue
        buf = _actor_to_buffer(actor.id, actor.poses)
        if len(buf.poses) < 2:
            continue
        if actor_has_joints3d(actor.poses):
            any_3d = True
            actor.metrics = compute_metrics_smpl3d(buf)
        else:
            actor.metrics = compute_metrics(buf, aspect=aspect)
    result.metricsSpace = "smpl3d" if any_3d else "coco2d"
    return result
