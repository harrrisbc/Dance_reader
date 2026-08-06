from __future__ import annotations

import math

import numpy as np

from app.pipeline.pose_track import TrackBuffer, TrackedPose
from app.schemas import AnalysisResult, MetricPoint, PoseFrame

# COCO indices used for limb speed
KEY_INDICES = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
REACH_INDICES = [0, 9, 10, 15, 16]  # nose, wrists, ankles
TRAVEL_WINDOW_S = 0.8
# Below this window path (in shoulder widths), directness is undefined
DIRECTNESS_MIN_PATH_SW = 0.08


def _torso_frame(
    keypoints: np.ndarray,
) -> tuple[float, float, float, float] | None:
    """Return (cx, cy, shoulder_scale, hip_y) or None if torso invisible."""
    if keypoints.shape[0] < 17:
        return None
    ls, rs = keypoints[5], keypoints[6]
    lh, rh = keypoints[11], keypoints[12]
    if min(float(ls[2]), float(rs[2]), float(lh[2]), float(rh[2])) < 0.2:
        return None
    cx = float((ls[0] + rs[0] + lh[0] + rh[0]) / 4)
    cy = float((ls[1] + rs[1] + lh[1] + rh[1]) / 4)
    scale = float(np.hypot(float(ls[0] - rs[0]), float(ls[1] - rs[1]))) or 1e-3
    hip_y = float((lh[1] + rh[1]) / 2)
    return cx, cy, scale, hip_y


def _reach_raw(keypoints: np.ndarray, cx: float, cy: float, scale: float) -> float:
    best = 0.0
    for i in REACH_INDICES:
        p = keypoints[i]
        if float(p[2]) < 0.2:
            continue
        d = float(np.hypot(float(p[0]) - cx, float(p[1]) - cy)) / scale
        best = max(best, d)
    return best


def _foot_y(keypoints: np.ndarray, fallback_y: float) -> float:
    """Lower (more downstage / larger y) confident ankle; else fallback."""
    if keypoints.shape[0] < 17:
        return fallback_y
    candidates: list[tuple[float, float]] = []
    for i in (15, 16):
        p = keypoints[i]
        conf = float(p[2])
        if conf >= 0.2:
            candidates.append((float(p[1]), conf))
    if not candidates:
        return fallback_y
    # Prefer lower ankle (larger y); if tie, higher confidence
    candidates.sort(key=lambda t: (t[0], t[1]))
    return candidates[-1][0]


def _leg_length(keypoints: np.ndarray, hip_y: float) -> float | None:
    """Mean |ankle_y - hip_y| over confident ankles."""
    if keypoints.shape[0] < 17:
        return None
    lengths: list[float] = []
    for i in (15, 16):
        p = keypoints[i]
        if float(p[2]) < 0.2:
            continue
        lengths.append(abs(float(p[1]) - hip_y))
    if not lengths:
        return None
    return float(sum(lengths) / len(lengths))


def compute_metrics(buffer: TrackBuffer) -> list[MetricPoint]:
    poses = buffer.poses
    if len(poses) < 2:
        return []

    times: list[float] = []
    energy_r: list[float] = []
    speed_sw: list[float] = []  # shoulder widths / second
    reach_sw: list[float] = []  # shoulder multiples (absolute)
    leg_r: list[float] = []  # raw leg length; nan if missing
    pos_xs: list[float] = []
    pos_ys: list[float] = []
    scales: list[float] = []

    last_reach = 0.0
    last_leg: float | None = None
    last_pos = (0.5, 0.5)
    last_scale = 1e-3

    for i in range(1, len(poses)):
        prev = poses[i - 1]
        cur = poses[i]
        dt = max(cur.time - prev.time, 1e-3)

        energy = 0.0
        speed_sum = 0.0
        speed_n = 0
        n = min(len(prev.keypoints), len(cur.keypoints))
        for j in range(n):
            a = prev.keypoints[j]
            b = cur.keypoints[j]
            if a[2] < 0.3 or b[2] < 0.3:
                continue
            dx = float(b[0] - a[0])
            dy = float(b[1] - a[1])
            v = float(np.hypot(dx, dy) / dt)
            energy += v * v
            if j in KEY_INDICES:
                speed_sum += v
                speed_n += 1
        speed_frame = speed_sum / speed_n if speed_n else 0.0

        torso = _torso_frame(cur.keypoints)
        if torso is None:
            # Carry forward — do not write 0.0 (false spikes / contracted)
            reach = last_reach
            leg = last_leg
            pos_x, pos_y = last_pos
            scale = last_scale
        else:
            cx, cy, scale, hip_y = torso
            reach = _reach_raw(cur.keypoints, cx, cy, scale)
            leg = _leg_length(cur.keypoints, hip_y)
            pos_x = cx
            pos_y = _foot_y(cur.keypoints, cy)
            last_reach = reach
            if leg is not None:
                last_leg = leg
            last_pos = (pos_x, pos_y)
            last_scale = scale

        times.append(cur.time)
        energy_r.append(energy)
        # Absolute speed in shoulder widths / second
        speed_sw.append(float(speed_frame / max(scale, 1e-3)))
        reach_sw.append(float(reach))
        leg_r.append(float(leg) if leg is not None else float("nan"))
        pos_xs.append(float(np.clip(pos_x, 0.0, 1.0)))
        pos_ys.append(float(np.clip(pos_y, 0.0, 1.0)))
        scales.append(float(max(scale, 1e-3)))

    # Posture level: leg / 90th percentile of leg over track
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

    # Travel / directness on foot track (pos_x, pos_y)
    travel_sw_s: list[float] = []
    direct_r: list[float] = []
    for i, t in enumerate(times):
        t0 = t - TRAVEL_WINDOW_S
        j0 = i
        while j0 > 0 and times[j0 - 1] >= t0:
            j0 -= 1
        path = 0.0
        for k in range(j0 + 1, i + 1):
            path += float(
                np.hypot(pos_xs[k] - pos_xs[k - 1], pos_ys[k] - pos_ys[k - 1])
            )
        net = float(
            np.hypot(pos_xs[i] - pos_xs[j0], pos_ys[i] - pos_ys[j0])
        )
        # Mean shoulder scale in window for unit conversion
        win_scale = float(np.mean(scales[j0 : i + 1])) if i >= j0 else scales[i]
        win_scale = max(win_scale, 1e-3)
        win_dt = max(times[i] - times[j0], 1e-3) if i > j0 else TRAVEL_WINDOW_S
        path_sw = path / win_scale
        travel_sw_s.append(float(path_sw / win_dt))
        if path_sw < DIRECTNESS_MIN_PATH_SW:
            direct_r.append(float("nan"))
        else:
            direct_r.append(float(np.clip(net / path, 0.0, 1.0)))

    # Energy remains relative to this actor's peak
    e_max = max(energy_r) if energy_r else 0.0
    if e_max <= 1e-9:
        energy_n = [0.0] * len(energy_r)
    else:
        energy_n = [float(v / e_max) for v in energy_r]

    return [
        MetricPoint(
            time=times[i],
            energy=energy_n[i],
            speed=speed_sw[i],
            reach=reach_sw[i],
            level=level_n[i],
            travel=travel_sw_s[i],
            directness=direct_r[i],
            pos_x=pos_xs[i],
            pos_y=pos_ys[i],
        )
        for i in range(len(times))
    ]


def pose_feature(keypoints: np.ndarray) -> np.ndarray | None:
    """Torso-normalized XY feature from COCO-17 keypoints."""
    if keypoints.shape[0] < 17:
        return None
    ls, rs = keypoints[5], keypoints[6]
    lh, rh = keypoints[11], keypoints[12]
    if min(ls[2], rs[2], lh[2], rh[2]) < 0.2:
        return None

    cx = float((ls[0] + rs[0] + lh[0] + rh[0]) / 4)
    cy = float((ls[1] + rs[1] + lh[1] + rh[1]) / 4)
    scale = float(np.hypot(ls[0] - rs[0], ls[1] - rs[1])) or 1.0

    feats: list[float] = []
    for i in KEY_INDICES:
        p = keypoints[i]
        if p[2] < 0.2:
            feats.extend([0.0, 0.0])
        else:
            feats.extend([(float(p[0]) - cx) / scale, (float(p[1]) - cy) / scale])
    return np.asarray(feats, dtype=np.float32)


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
    for actor in result.actors:
        if not actor.poses:
            continue
        buf = _actor_to_buffer(actor.id, actor.poses)
        if len(buf.poses) < 2:
            continue
        actor.metrics = compute_metrics(buf)
    return result
