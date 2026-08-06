from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from app.pipeline.metrics import KEY_INDICES, pose_feature
from app.pipeline.pose_track import TrackBuffer
from app.schemas import MotifOccurrence, MotifType

TYPE_COLORS = [
    "#c45c4a",
    "#e8c56e",
    "#78b4a0",
    "#a67c52",
    "#d4a0a0",
    "#8fbc8f",
    "#cd853f",
    "#b8860b",
]

# Fallback sliding windows
WINDOW_S = 1.2
STEP_S = 0.4
# Phrase duration clamps for valley cuts
PHRASE_MIN_S = 0.6
PHRASE_MAX_S = 2.5
EMBED_SAMPLES = 10
MOTION_SPEED_MIN = 0.08
MOTION_REL_FRAC = 0.22
CLUSTER_DISTANCE = 0.48
NMS_IOU = 0.45
# Query-by-example: cosine similarity threshold
SIMILARITY_MIN = 0.82
ENERGY_SMOOTH = 5


@dataclass
class Segment:
    actor_id: str
    start: float
    end: float
    embedding: np.ndarray
    motion: float


def _type_letter(i: int) -> str:
    if i < 26:
        return chr(ord("A") + i)
    return f"A{i}"


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return v.astype(np.float32)
    return (v / n).astype(np.float32)


def _frame_speed(prev_kpts: np.ndarray, cur_kpts: np.ndarray, dt: float) -> float:
    dt = max(dt, 1e-3)
    speed_sum = 0.0
    speed_n = 0
    n = min(len(prev_kpts), len(cur_kpts))
    for j in KEY_INDICES:
        if j >= n:
            continue
        a = prev_kpts[j]
        b = cur_kpts[j]
        if a[2] < 0.3 or b[2] < 0.3:
            continue
        v = float(np.hypot(float(b[0] - a[0]), float(b[1] - a[1])) / dt)
        speed_sum += v
        speed_n += 1
    return speed_sum / speed_n if speed_n else 0.0


def _window_motion(poses: list, idxs: np.ndarray) -> float:
    if len(idxs) < 2:
        return 0.0
    speeds: list[float] = []
    for k in range(1, len(idxs)):
        prev = poses[int(idxs[k - 1])]
        cur = poses[int(idxs[k])]
        speeds.append(_frame_speed(prev.keypoints, cur.keypoints, cur.time - prev.time))
    return float(np.mean(speeds)) if speeds else 0.0


def _trajectory_embedding(poses: list, idxs: np.ndarray) -> np.ndarray | None:
    """Sample pose+velocity features along the phrase; fixed length, L2-normalize."""
    if len(idxs) < 2:
        return None
    pick = np.linspace(0, len(idxs) - 1, EMBED_SAMPLES)
    pick = np.clip(np.round(pick).astype(int), 0, len(idxs) - 1)
    feats: list[float] = []
    prev_f: np.ndarray | None = None
    feat_dim = len(KEY_INDICES) * 2
    for pi in pick:
        i = int(idxs[int(pi)])
        f = pose_feature(poses[i].keypoints)
        if f is None:
            f = np.zeros(feat_dim, dtype=np.float32)
        if prev_f is None:
            vel = np.zeros_like(f)
        else:
            vel = f - prev_f
        prev_f = f
        feats.extend(f.tolist())
        feats.extend(vel.tolist())
    if not feats:
        return None
    return _l2_normalize(np.asarray(feats, dtype=np.float32))


def _energy_series(poses: list) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame limb speed series (time, energy)."""
    if len(poses) < 2:
        return np.array([]), np.array([])
    times: list[float] = []
    energy: list[float] = []
    for i in range(1, len(poses)):
        prev, cur = poses[i - 1], poses[i]
        times.append(cur.time)
        energy.append(_frame_speed(prev.keypoints, cur.keypoints, cur.time - prev.time))
    t = np.asarray(times, dtype=np.float64)
    e = np.asarray(energy, dtype=np.float64)
    if len(e) >= ENERGY_SMOOTH:
        kernel = np.ones(ENERGY_SMOOTH, dtype=np.float64) / ENERGY_SMOOTH
        e = np.convolve(e, kernel, mode="same")
    return t, e


def _valley_boundaries(times: np.ndarray, energy: np.ndarray) -> list[float]:
    """Local minima of energy used as phrase cuts."""
    if len(energy) < 5:
        return []
    cuts: list[float] = [float(times[0])]
    for i in range(2, len(energy) - 2):
        if energy[i] <= energy[i - 1] and energy[i] <= energy[i + 1]:
            if energy[i] <= energy[i - 2] and energy[i] <= energy[i + 2]:
                cuts.append(float(times[i]))
    cuts.append(float(times[-1]))
    # Deduplicate close cuts
    merged: list[float] = [cuts[0]]
    for c in cuts[1:]:
        if c - merged[-1] >= PHRASE_MIN_S * 0.5:
            merged.append(c)
    return merged


def _make_segment(
    actor_id: str,
    poses: list,
    times: np.ndarray,
    start: float,
    end: float,
) -> Segment | None:
    if end - start < PHRASE_MIN_S * 0.8:
        return None
    end = min(end, start + PHRASE_MAX_S)
    idxs = np.where((times >= start) & (times <= end))[0]
    if len(idxs) < 3:
        return None
    motion = _window_motion(poses, idxs)
    emb = _trajectory_embedding(poses, idxs)
    if emb is None:
        return None
    return Segment(
        actor_id=actor_id,
        start=float(start),
        end=float(end),
        embedding=emb,
        motion=motion,
    )


def _valley_segments(actor_id: str, buffer: TrackBuffer) -> list[Segment]:
    poses = buffer.poses
    if len(poses) < 8:
        return []
    times = np.array([p.time for p in poses], dtype=np.float64)
    t_e, energy = _energy_series(poses)
    if len(energy) < 5:
        return []
    cuts = _valley_boundaries(t_e, energy)
    if len(cuts) < 2:
        return []

    candidates: list[Segment] = []
    for i in range(len(cuts) - 1):
        start, end = cuts[i], cuts[i + 1]
        # Split long gaps into PHRASE_MAX chunks
        while end - start > PHRASE_MAX_S + 1e-6:
            seg = _make_segment(actor_id, poses, times, start, start + PHRASE_MAX_S)
            if seg:
                candidates.append(seg)
            start += PHRASE_MAX_S * 0.85
        seg = _make_segment(actor_id, poses, times, start, end)
        if seg:
            candidates.append(seg)
    return candidates


def _sliding_segments(actor_id: str, buffer: TrackBuffer) -> list[Segment]:
    poses = buffer.poses
    if len(poses) < 6:
        return []
    times = np.array([p.time for p in poses], dtype=np.float64)
    t0, t1 = float(times[0]), float(times[-1])
    if t1 - t0 < WINDOW_S * 0.8:
        return []
    candidates: list[Segment] = []
    t = t0
    while t + WINDOW_S <= t1 + 1e-6:
        seg = _make_segment(actor_id, poses, times, t, t + WINDOW_S)
        if seg:
            candidates.append(seg)
        t += STEP_S
    return candidates


def _gate_motion(candidates: list[Segment]) -> list[Segment]:
    if not candidates:
        return []
    peak = max(s.motion for s in candidates)
    thresh = max(MOTION_SPEED_MIN, peak * MOTION_REL_FRAC)
    return [s for s in candidates if s.motion >= thresh]


def build_segments(actors: dict[str, TrackBuffer]) -> list[Segment]:
    """Valley phrases first; fall back to sliding windows if too few."""
    all_segments: list[Segment] = []
    for actor_id, buf in actors.items():
        valley = _gate_motion(_valley_segments(actor_id, buf))
        if len(valley) >= 2:
            all_segments.extend(valley)
        else:
            all_segments.extend(_gate_motion(_sliding_segments(actor_id, buf)))
    return all_segments


def _cluster_segments(
    segments: list[Segment],
    *,
    distance_threshold: float = CLUSTER_DISTANCE,
) -> list[int]:
    if not segments:
        return []
    if len(segments) == 1:
        return [0]

    X = np.stack([s.embedding for s in segments], axis=0)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="euclidean",
        linkage="average",
        distance_threshold=distance_threshold,
    )
    labels = clustering.fit_predict(X)
    return labels.tolist()


def _nms_occurrences(
    occs: list[MotifOccurrence],
    *,
    iou_thresh: float = NMS_IOU,
) -> list[MotifOccurrence]:
    kept: list[MotifOccurrence] = []
    for o in sorted(occs, key=lambda x: x.start):
        drop = False
        for k in kept:
            if k.actorId != o.actorId:
                continue
            inter = max(0.0, min(k.end, o.end) - max(k.start, o.start))
            union = max(k.end, o.end) - min(k.start, o.start)
            if union > 0 and inter / union >= iou_thresh:
                drop = True
                break
        if not drop:
            kept.append(o)
    return kept


def _segments_to_motif_types(
    segments: list[Segment],
    *,
    labels: list[int] | None = None,
    name_prefix: str = "類型",
    start_type_idx: int = 0,
    start_occ: int = 0,
) -> list[MotifType]:
    if not segments:
        return []
    if labels is None:
        labels = _cluster_segments(segments)

    clusters: dict[int, list[int]] = {}
    for i, lab in enumerate(labels):
        clusters.setdefault(int(lab), []).append(i)

    ranked = sorted(clusters.items(), key=lambda kv: -len(kv[1]))
    motif_types: list[MotifType] = []
    type_idx = start_type_idx
    occ_counter = start_occ

    for _, idxs in ranked:
        if len(idxs) < 2:
            continue
        if type_idx - start_type_idx >= len(TYPE_COLORS) * 2:
            break
        mean_motion = float(np.mean([segments[si].motion for si in idxs]))
        if mean_motion < MOTION_SPEED_MIN:
            continue

        occs: list[MotifOccurrence] = []
        for si in idxs:
            seg = segments[si]
            occ_counter += 1
            occs.append(
                MotifOccurrence(
                    id=f"occ_{occ_counter}",
                    actorId=seg.actor_id,
                    start=seg.start,
                    end=seg.end,
                )
            )
        occs = _nms_occurrences(occs)
        if len(occs) < 2:
            continue

        letter = _type_letter(type_idx)
        motif_types.append(
            MotifType(
                id=f"type_{letter.lower()}",
                name=f"{name_prefix} {letter}",
                color=TYPE_COLORS[type_idx % len(TYPE_COLORS)],
                occurrences=occs,
            )
        )
        type_idx += 1

    return motif_types


def detect_motif_types(
    actors: dict[str, TrackBuffer],
) -> list[MotifType]:
    """
    Cut phrases at energy valleys, embed pose+velocity trajectories,
    cluster into 類型 A/B/C… Low-motion segments are dropped.
    """
    all_segments = build_segments(actors)
    if len(all_segments) < 2:
        return []
    return _segments_to_motif_types(all_segments)


def segment_near_time(
    actors: dict[str, TrackBuffer],
    actor_id: str,
    time_s: float,
) -> Segment | None:
    """Best motion-gated segment for actor covering / nearest to time_s."""
    buf = actors.get(actor_id)
    if buf is None:
        return None
    segs = _gate_motion(_valley_segments(actor_id, buf))
    if len(segs) < 1:
        segs = _gate_motion(_sliding_segments(actor_id, buf))
    if not segs:
        # Force a window around playhead
        times = np.array([p.time for p in buf.poses], dtype=np.float64)
        start = max(float(times[0]), time_s - WINDOW_S / 2)
        return _make_segment(actor_id, buf.poses, times, start, start + WINDOW_S)

    covering = [s for s in segs if s.start <= time_s <= s.end]
    if covering:
        return min(covering, key=lambda s: abs((s.start + s.end) / 2 - time_s))
    return min(segs, key=lambda s: min(abs(s.start - time_s), abs(s.end - time_s)))


def find_similar_motif(
    actors: dict[str, TrackBuffer],
    query: Segment,
    *,
    similarity_min: float = SIMILARITY_MIN,
    name: str = "相似 1",
    type_id: str = "type_sim_1",
    color: str = "#78b4a0",
) -> MotifType | None:
    """
    Find all motion segments similar to query (cosine on trajectory embeddings).
    Always includes the query span itself.
    """
    pool = build_segments(actors)
    if not pool:
        pool = [query]

    q = _l2_normalize(query.embedding)
    hits: list[Segment] = [query]
    for seg in pool:
        sim = float(np.dot(q, _l2_normalize(seg.embedding)))
        if sim >= similarity_min:
            hits.append(seg)

    occ_counter = 0
    occs: list[MotifOccurrence] = []
    for seg in hits:
        occ_counter += 1
        occs.append(
            MotifOccurrence(
                id=f"sim_occ_{occ_counter}",
                actorId=seg.actor_id,
                start=seg.start,
                end=seg.end,
            )
        )
    occs = _nms_occurrences(occs)
    if len(occs) < 1:
        return None
    return MotifType(
        id=type_id,
        name=name,
        color=color,
        occurrences=occs,
    )
