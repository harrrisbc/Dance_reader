from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering

from app.pipeline.metrics import (
    DEFAULT_ASPECT,
    KEY_INDICES,
    estimate_fps,
    limb_energy_and_speed,
    measure_scale,
    pose_feature_rich,
    pose_feature_rich_mirrored,
    smooth_series,
    to_square_pixels,
    track_scale_ref,
)
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

# Fallback sliding windows (only if valleys fail)
WINDOW_S = 1.2
STEP_S = 0.45
PHRASE_MIN_S = 0.65
PHRASE_MAX_S = 2.8
SEQ_FRAMES = 14
EMBED_SAMPLES = 10
# Clustering / matching gates in **raw mask-aware DTW distance**.
# Distance = mean per-visible-dim RMS in torso units. Lower = more similar.
# Auto discovery: precision > recall (see IMPLEMENTATION_NOTES_MOTIF_OVERMERGE).
# Find Similar uses SUBSEQ_* — do not loosen those to compensate discovery cuts.
CLUSTER_MAX_DIST = 0.24  # agglomerative cut (0.22–0.24 band)
INTRA_MAX_DIST = 0.26  # mean pairwise distance within a kept cluster
CLUSTER_LINKAGE = "complete"  # max pairwise must clear cut (stops A~B~C chains)
MATCH_MAX_DIST = 0.28  # legacy name; segment gates / docs

# Find-similar subsequence DTW accept if d <= this (slightly above segment MATCH)
SUBSEQ_MATCH_MAX_DIST = 0.30
# Display-only: d=0 → 100%, d>=DISPLAY_DIST_ZERO → 0%. Never used as a gate.
DISPLAY_DIST_ZERO = 0.50
# Motif type needs this many occurrences after NMS (lets short no-repeat clips say "no")
MOTIF_MIN_OCCURRENCES = 3
# Find-similar: reject hits whose duration / query duration is outside [1/r, r]
TEMPO_RATIO_MAX = 2.0
# Clear subsequence matches (always keep); softer hits kept sparsely up to SUBSEQ_MATCH_MAX_DIST
SUBSEQ_STRONG_DIST = 0.16
SUBSEQ_SOFT_MAX_EXTRA = 6
SUBSEQ_SOFT_SEP_S = 1.5  # min centre separation when keeping soft hits
# Subsequence DTW local-minima: peak must beat neighbors by this fraction of MATCH_MAX_DIST
SUBSEQ_MINIMA_REL = 0.02
# Cheap cosine pre-filter before DTW verify (reject only — do not invent scores)
COSINE_PREFILTER = 0.55
# Stricter motion gate — prefer missing idle over false merges
MOTION_SPEED_MIN = 0.11
MOTION_REL_FRAC = 0.30
NMS_IOU = 0.45
ENERGY_SMOOTH = 5  # legacy; Wave 2 uses Savitzky–Golay via smooth_series
VALLEY_FRAC = 0.55  # valley must be below this fraction of local peak
# Drop phrases where too many frames fail torso scale (blown features otherwise)
DEGENERATE_FRAME_FRAC = 0.35
# Mask-aware DTW: below this shared KEY_INDICES joint count, impute ≥ median cost
MIN_SHARED_VISIBLE_JOINTS = 4
N_XY_DIMS = len(KEY_INDICES) * 2
# Cosine-reject sentinel for clustering matrix (must exceed any real DTW)
COSINE_REJECT_DIST = 1e3

# Geometry context for feature extraction (set by detect / find-similar)
_CTX_ASPECT: float = DEFAULT_ASPECT
_CTX_SCALE_REFS: dict[str, float] = {}


def _set_geom_context(actors: dict[str, TrackBuffer], aspect: float) -> None:
    global _CTX_ASPECT, _CTX_SCALE_REFS
    _CTX_ASPECT = float(aspect) if aspect > 0 else DEFAULT_ASPECT
    _CTX_SCALE_REFS = {
        aid: track_scale_ref(buf.poses, _CTX_ASPECT) for aid, buf in actors.items()
    }


def _scale_for(actor_id: str) -> float | None:
    return _CTX_SCALE_REFS.get(actor_id)


@dataclass
class MotifDetectStats:
    """Per-gate drop counts for empty-result diagnosis."""

    phrases_raw: int = 0
    phrases: int = 0  # after motion gate
    dropped_motion: int = 0
    clusters: int = 0
    dropped_size: int = 0
    dropped_motion_mean: int = 0
    dropped_intra: int = 0
    dropped_nms: int = 0
    types_kept: int = 0
    # Mean pairwise DTW per kept type (UI / calibration honesty)
    type_mean_intras: list[float] = field(default_factory=list)

    def summary(self) -> str:
        base = (
            f"{self.phrases} phrases · {self.clusters} clusters · "
            f"{self.types_kept} passed distance {INTRA_MAX_DIST:.2f}"
        )
        if self.type_mean_intras:
            bits = ", ".join(f"{d:.2f}" for d in self.type_mean_intras)
            base += f" · mean_intra=[{bits}]"
        if self.types_kept > 0 or self.phrases < 2:
            return base
        # Explain emptiness — which gate wiped the result
        gates = [
            ("motion", self.dropped_motion),
            ("size<2", self.dropped_size),
            ("motion-mean", self.dropped_motion_mean),
            (f"intra>{INTRA_MAX_DIST:.2f}", self.dropped_intra),
            (f"occ<{MOTIF_MIN_OCCURRENCES}/nms", self.dropped_nms),
        ]
        killers = [f"{name}={n}" for name, n in gates if n > 0]
        if not killers:
            return base + " (nothing survived)"
        return base + " — emptied by " + ", ".join(killers)

    def detail(self) -> str:
        return (
            f"{self.summary()} "
            f"(drop motion={self.dropped_motion} size={self.dropped_size} "
            f"intra={self.dropped_intra} nms={self.dropped_nms})"
        )


LAST_DETECT_STATS = MotifDetectStats()


@dataclass
class Segment:
    actor_id: str
    start: float
    end: float
    embedding: np.ndarray
    embedding_mirror: np.ndarray
    seq: np.ndarray  # (T, D) rich features
    seq_mirror: np.ndarray
    seq_mask: np.ndarray  # (T, D) bool visibility
    seq_mirror_mask: np.ndarray
    motion: float
    reach_mean: float = 0.0
    level_mean: float = 0.0
    score: float = 1.0  # filled by find-similar


def _type_letter(i: int) -> str:
    if i < 26:
        return chr(ord("A") + i)
    return f"A{i}"


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return v.astype(np.float32)
    return (v / n).astype(np.float32)


def _frame_speed(
    prev_kpts: np.ndarray,
    cur_kpts: np.ndarray,
    dt: float,
    *,
    aspect: float,
    scale: float,
) -> float:
    """Scale-normalized mean limb speed (torso/s) — same joints as chart Speed."""
    _energy, speed = limb_energy_and_speed(
        prev_kpts, cur_kpts, dt, aspect=aspect, scale=scale
    )
    return speed


def _window_motion(poses: list, idxs: np.ndarray, *, actor_id: str) -> float:
    if len(idxs) < 2:
        return 0.0
    scale_ref = _scale_for(actor_id) or 0.1
    speeds: list[float] = []
    for k in range(1, len(idxs)):
        prev = poses[int(idxs[k - 1])]
        cur = poses[int(idxs[k])]
        measured = measure_scale(
            cur.keypoints, aspect=_CTX_ASPECT, scale_ref=scale_ref
        )
        scale = measured[2] if measured is not None else scale_ref
        speeds.append(
            _frame_speed(
                prev.keypoints,
                cur.keypoints,
                cur.time - prev.time,
                aspect=_CTX_ASPECT,
                scale=scale,
            )
        )
    return float(np.mean(speeds)) if speeds else 0.0


def _reach_level_means(
    poses: list,
    idxs: np.ndarray,
    *,
    actor_id: str,
) -> tuple[float, float]:
    """Lightweight phrase descriptors using torso-height scale."""
    reaches: list[float] = []
    levels: list[float] = []
    scale_ref = _scale_for(actor_id)
    for i in idxs:
        k = poses[int(i)].keypoints
        measured = measure_scale(k, aspect=_CTX_ASPECT, scale_ref=scale_ref)
        if measured is None:
            continue
        cx, cy, scale, hip_y = measured
        sq = to_square_pixels(k, _CTX_ASPECT)
        best = 0.0
        for j in (0, 9, 10, 15, 16):
            p = sq[j]
            if float(p[2]) < 0.2:
                continue
            best = max(
                best,
                float(np.hypot(float(p[0]) - cx, float(p[1]) - cy)) / scale,
            )
        reaches.append(best)
        ankles = []
        for j in (15, 16):
            if float(sq[j][2]) >= 0.2:
                ankles.append(abs(float(sq[j][1]) - hip_y))
        if ankles:
            levels.append(float(np.mean(ankles)) / scale)
    return (
        float(np.mean(reaches)) if reaches else 0.0,
        float(np.mean(levels)) if levels else 0.0,
    )


def _sample_sequence(
    poses: list,
    idxs: np.ndarray,
    *,
    actor_id: str,
    mirrored: bool = False,
    n_frames: int = SEQ_FRAMES,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (seq[T,D], mask[T,D]). Padded / degenerate frames have zero visibility."""
    if len(idxs) < 2:
        return None
    scale_ref = _scale_for(actor_id)
    pick = np.linspace(0, len(idxs) - 1, n_frames)
    pick = np.clip(np.round(pick).astype(int), 0, len(idxs) - 1)
    frames: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    feat_dim: int | None = None
    missing = 0
    for pi in pick:
        i = int(idxs[int(pi)])
        k = poses[i].keypoints
        packed = (
            pose_feature_rich_mirrored(k, aspect=_CTX_ASPECT, scale_ref=scale_ref)
            if mirrored
            else pose_feature_rich(k, aspect=_CTX_ASPECT, scale_ref=scale_ref)
        )
        if packed is None:
            missing += 1
            if feat_dim is None:
                continue
            frames.append(np.zeros(feat_dim, dtype=np.float32))
            masks.append(np.zeros(feat_dim, dtype=bool))
        else:
            f, m = packed
            feat_dim = int(f.shape[0])
            frames.append(f.astype(np.float32))
            masks.append(np.asarray(m, dtype=bool))
    if missing / max(len(pick), 1) > DEGENERATE_FRAME_FRAC:
        return None
    if len(frames) < max(4, n_frames // 2):
        return None
    while len(frames) < n_frames and feat_dim:
        # Pad with zero-visibility — do not repeat last frame as DTW evidence
        frames.append(np.zeros(feat_dim, dtype=np.float32))
        masks.append(np.zeros(feat_dim, dtype=bool))
    return (
        np.stack(frames[:n_frames], axis=0),
        np.stack(masks[:n_frames], axis=0),
    )


def _trajectory_embedding(seq: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Fixed-length L2 embedding from pose sequence + deltas (cheap prefilter)."""
    pick = np.linspace(0, len(seq) - 1, EMBED_SAMPLES)
    pick = np.clip(np.round(pick).astype(int), 0, len(seq) - 1)
    feats: list[float] = []
    prev: np.ndarray | None = None
    for pi in pick:
        idx = int(pi)
        f = seq[idx]
        # Use only first KEY_INDICES*2 dims (XY) for compact cosine prefilter
        xy = f[:N_XY_DIMS].copy()
        if mask is not None:
            # Occluded dims stay 0 — never fabricate torso-centre as visible pose
            m = mask[idx][:N_XY_DIMS]
            xy = np.where(m, xy, 0.0)
        if prev is None:
            vel = np.zeros_like(xy)
        else:
            vel = xy - prev
        prev = xy
        feats.extend(xy.tolist())
        feats.extend(vel.tolist())
    return _l2_normalize(np.asarray(feats, dtype=np.float32))


def _shared_visible_joints(mask_a: np.ndarray, mask_b: np.ndarray) -> int:
    """Count KEY_INDICES joints whose XY dims are visible in both frames."""
    n = 0
    lim = min(N_XY_DIMS, len(mask_a), len(mask_b))
    for k in range(0, lim, 2):
        if (
            bool(mask_a[k])
            and bool(mask_a[k + 1])
            and bool(mask_b[k])
            and bool(mask_b[k + 1])
        ):
            n += 1
    return n


def _masked_frame_distance(
    ai: np.ndarray,
    bi: np.ndarray,
    ma: np.ndarray,
    mb: np.ndarray,
) -> tuple[float, int]:
    """RMS per jointly-visible dim + shared KEY joint count."""
    shared = np.asarray(ma, dtype=bool) & np.asarray(mb, dtype=bool)
    n_dims = int(shared.sum())
    n_joints = _shared_visible_joints(ma, mb)
    if n_dims == 0:
        return float("nan"), n_joints
    diff = ai[shared] - bi[shared]
    return float(np.sqrt(np.sum(diff * diff) / n_dims)), n_joints


def _masked_dist_matrix(
    a: np.ndarray,
    b: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Frame-distance matrix with low-visibility imputation.

    Well-observed pairs (shared joints ≥ MIN_SHARED_VISIBLE_JOINTS) use RMS
    per visible dim. Low-evidence pairs get max(partial, median_good) so DTW
    never routes through occlusion for a cheap path.
    """
    n, m = len(a), len(b)
    dists = np.full((n, m), np.nan, dtype=np.float64)
    good: list[float] = []
    for i in range(n):
        for j in range(m):
            d, n_j = _masked_frame_distance(a[i], b[j], mask_a[i], mask_b[j])
            dists[i, j] = d
            if n_j >= MIN_SHARED_VISIBLE_JOINTS and np.isfinite(d):
                good.append(d)
    median = float(np.median(np.asarray(good, dtype=np.float64))) if good else 1.0
    for i in range(n):
        for j in range(m):
            n_j = _shared_visible_joints(mask_a[i], mask_b[j])
            d = dists[i, j]
            if n_j < MIN_SHARED_VISIBLE_JOINTS:
                if np.isfinite(d):
                    dists[i, j] = max(float(d), median)
                else:
                    dists[i, j] = median
            elif not np.isfinite(d):
                dists[i, j] = median
    return dists, median


def _dtw_from_dists(dists: np.ndarray) -> float:
    """Banded DTW over a precomputed (n, m) frame-distance matrix."""
    n, m = dists.shape
    if n == 0 or m == 0:
        return 1e6
    band = max(2, int(0.35 * max(n, m)))
    inf = 1e9
    cost = np.full((n + 1, m + 1), inf, dtype=np.float64)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        j0 = max(1, i - band)
        j1 = min(m, i + band)
        di = dists[i - 1]
        for j in range(j0, j1 + 1):
            cost[i, j] = float(di[j - 1]) + min(
                cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1]
            )
    path = cost[n, m]
    if not np.isfinite(path) or path >= inf / 2:
        return 1e3
    return float(path / (n + m))


def _dtw_path_from_dists(dists: np.ndarray) -> list[tuple[int, int]]:
    """Backtrack Sakoe-Chiba banded path as (i, j) frame indices."""
    n, m = dists.shape
    if n == 0 or m == 0:
        return []
    band = max(2, int(0.35 * max(n, m)))
    inf = 1e9
    cost = np.full((n + 1, m + 1), inf, dtype=np.float64)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        j0 = max(1, i - band)
        j1 = min(m, i + band)
        di = dists[i - 1]
        for j in range(j0, j1 + 1):
            cost[i, j] = float(di[j - 1]) + min(
                cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1]
            )
    if cost[n, m] >= inf / 2:
        return []
    path: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        cand = [
            (cost[i - 1, j - 1], i - 1, j - 1),
            (cost[i - 1, j], i - 1, j),
            (cost[i, j - 1], i, j - 1),
        ]
        _c, i, j = min(cand, key=lambda t: t[0])
    path.reverse()
    return path


def _dtw_distance_naive(a: np.ndarray, b: np.ndarray) -> float:
    """Reference banded DTW (Python loops, unmasked). Kept for W1 identity tests."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 1e6
    band = max(2, int(0.35 * max(n, m)))
    inf = 1e9
    cost = np.full((n + 1, m + 1), inf, dtype=np.float64)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        j0 = max(1, i - band)
        j1 = min(m, i + band)
        ai = a[i - 1]
        for j in range(j0, j1 + 1):
            d = float(np.linalg.norm(ai - b[j - 1]))
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    path = cost[n, m]
    if not np.isfinite(path) or path >= inf / 2:
        return 1e3
    return float(path / (n + m))


def _dtw_distance(
    a: np.ndarray,
    b: np.ndarray,
    mask_a: np.ndarray | None = None,
    mask_b: np.ndarray | None = None,
) -> float:
    """
    Normalized DTW distance. With masks: mean-per-visible-dim frame costs and
    low-visibility imputation. Without masks: cdist Euclidean (W1 path).
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 1e6
    if mask_a is None or mask_b is None:
        aa = np.asarray(a, dtype=np.float64)
        bb = np.asarray(b, dtype=np.float64)
        dists = cdist(aa, bb, metric="euclidean")
        return _dtw_from_dists(dists)
    dists, _med = _masked_dist_matrix(
        np.asarray(a, dtype=np.float64),
        np.asarray(b, dtype=np.float64),
        np.asarray(mask_a, dtype=bool),
        np.asarray(mask_b, dtype=bool),
    )
    return _dtw_from_dists(dists)


def segment_raw_dtw_distance(a: Segment, b: Segment) -> float:
    """Mirror-aware raw DTW distance (no cosine gate). Mask-aware when masks present."""
    d_same = _dtw_distance(a.seq, b.seq, a.seq_mask, b.seq_mask)
    d_mir = _dtw_distance(a.seq, b.seq_mirror, a.seq_mask, b.seq_mirror_mask)
    return float(min(d_same, d_mir))


def segment_path_visibility(a: Segment, b: Segment) -> dict[str, float]:
    """
    Diagnostics: fraction of warping-path cells with low shared visibility.
    Prefer the mirror orientation that yields the smaller DTW distance.
    """
    candidates = [
        (a.seq, b.seq, a.seq_mask, b.seq_mask),
        (a.seq, b.seq_mirror, a.seq_mask, b.seq_mirror_mask),
    ]
    best: dict[str, float] | None = None
    best_d = 1e9
    for seq_a, seq_b, ma, mb in candidates:
        dists, median = _masked_dist_matrix(
            np.asarray(seq_a, dtype=np.float64),
            np.asarray(seq_b, dtype=np.float64),
            np.asarray(ma, dtype=bool),
            np.asarray(mb, dtype=bool),
        )
        d = _dtw_from_dists(dists)
        path = _dtw_path_from_dists(dists)
        if not path:
            continue
        low = 0
        for i, j in path:
            if _shared_visible_joints(ma[i], mb[j]) < MIN_SHARED_VISIBLE_JOINTS:
                low += 1
        stats = {
            "dtw": float(d),
            "path_len": float(len(path)),
            "low_vis_frac": float(low / len(path)),
            "median_frame": float(median),
        }
        if d < best_d:
            best_d = d
            best = stats
    return best or {"dtw": 1e3, "path_len": 0.0, "low_vis_frac": 0.0, "median_frame": 1.0}


def segment_pair_distance(a: Segment, b: Segment) -> float:
    """Decision distance: cosine gate then mask-aware raw DTW. Lower = closer."""
    cos_same = float(np.dot(a.embedding, b.embedding))
    cos_mir = float(np.dot(a.embedding, b.embedding_mirror))
    if max(cos_same, cos_mir) < COSINE_PREFILTER:
        return float(COSINE_REJECT_DIST)
    return segment_raw_dtw_distance(a, b)


def distance_to_match_score(dist: float) -> float:
    """
    Display-only 0..1 match figure (higher = better).
    Not used by any gate — decisions stay in raw distance units.
    """
    if dist >= COSINE_REJECT_DIST / 2:
        return 0.0
    return float(np.clip(1.0 - float(dist) / DISPLAY_DIST_ZERO, 0.0, 1.0))


def segment_similarity(a: Segment, b: Segment) -> float:
    """Display-only match score from decision distance. Gates must use distance."""
    return distance_to_match_score(segment_pair_distance(a, b))


def _energy_series(poses: list) -> tuple[np.ndarray, np.ndarray]:
    """
    Scale-normalized energy (KEY_INDICES), Savitzky–Golay smoothed — same recipe
    as displayed Energy before peak-normalization.
    """
    if len(poses) < 2:
        return np.array([]), np.array([])
    scale_ref = track_scale_ref(poses, _CTX_ASPECT)
    times: list[float] = []
    energy: list[float] = []
    last_scale = scale_ref
    for i in range(1, len(poses)):
        prev, cur = poses[i - 1], poses[i]
        measured = measure_scale(
            cur.keypoints, aspect=_CTX_ASPECT, scale_ref=scale_ref
        )
        scale = measured[2] if measured is not None else last_scale
        last_scale = scale
        e, _s = limb_energy_and_speed(
            prev.keypoints,
            cur.keypoints,
            cur.time - prev.time,
            aspect=_CTX_ASPECT,
            scale=scale,
        )
        times.append(cur.time)
        energy.append(e)
    t = np.asarray(times, dtype=np.float64)
    fps = estimate_fps(times)
    e = smooth_series(energy, fps=fps)
    return t, e


def _valley_boundaries(times: np.ndarray, energy: np.ndarray) -> list[float]:
    """Local minima that are true rests (below nearby peak * VALLEY_FRAC)."""
    if len(energy) < 5:
        return []
    cuts: list[float] = [float(times[0])]
    for i in range(2, len(energy) - 2):
        if not (
            energy[i] <= energy[i - 1]
            and energy[i] <= energy[i + 1]
            and energy[i] <= energy[i - 2]
            and energy[i] <= energy[i + 2]
        ):
            continue
        lo = max(0, i - 12)
        hi = min(len(energy), i + 13)
        local_peak = float(np.max(energy[lo:hi]))
        if local_peak < 1e-6:
            continue
        if energy[i] > local_peak * VALLEY_FRAC:
            continue
        cuts.append(float(times[i]))
    cuts.append(float(times[-1]))
    merged: list[float] = [cuts[0]]
    for c in cuts[1:]:
        if c - merged[-1] >= PHRASE_MIN_S * 0.55:
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
    # Soften hard chops: allow slightly longer continuous accents
    if end - start > PHRASE_MAX_S:
        end = start + PHRASE_MAX_S
    idxs = np.where((times >= start) & (times <= end))[0]
    if len(idxs) < 4:
        return None
    motion = _window_motion(poses, idxs, actor_id=actor_id)
    sampled = _sample_sequence(poses, idxs, actor_id=actor_id, mirrored=False)
    sampled_m = _sample_sequence(poses, idxs, actor_id=actor_id, mirrored=True)
    if sampled is None or sampled_m is None:
        return None
    seq, seq_mask = sampled
    seq_m, seq_m_mask = sampled_m
    emb = _trajectory_embedding(seq, seq_mask)
    emb_m = _trajectory_embedding(seq_m, seq_m_mask)
    reach_m, level_m = _reach_level_means(poses, idxs, actor_id=actor_id)
    return Segment(
        actor_id=actor_id,
        start=float(start),
        end=float(end),
        embedding=emb,
        embedding_mirror=emb_m,
        seq=seq,
        seq_mirror=seq_m,
        seq_mask=seq_mask,
        seq_mirror_mask=seq_m_mask,
        motion=motion,
        reach_mean=reach_m,
        level_mean=level_m,
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
        # Soft overlap when splitting long spans — avoid bisecting accents mid-peak
        while end - start > PHRASE_MAX_S + 1e-6:
            seg = _make_segment(actor_id, poses, times, start, start + PHRASE_MAX_S)
            if seg:
                candidates.append(seg)
            start += PHRASE_MAX_S * 0.9
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


def build_segments(
    actors: dict[str, TrackBuffer],
    *,
    stats: MotifDetectStats | None = None,
    aspect: float | None = None,
) -> list[Segment]:
    if aspect is not None:
        _set_geom_context(actors, aspect)
    elif not _CTX_SCALE_REFS:
        _set_geom_context(actors, DEFAULT_ASPECT)
    all_segments: list[Segment] = []
    raw_n = 0
    for actor_id, buf in actors.items():
        valley = _valley_segments(actor_id, buf)
        valley_g = _gate_motion(valley)
        if len(valley_g) >= 2:
            raw_n += len(valley)
            all_segments.extend(valley_g)
        else:
            sliding = _sliding_segments(actor_id, buf)
            raw_n += len(sliding)
            all_segments.extend(_gate_motion(sliding))
    if stats is not None:
        stats.phrases_raw = raw_n
        stats.phrases = len(all_segments)
        stats.dropped_motion = max(0, raw_n - len(all_segments))
    return all_segments


def _pairwise_distance_matrix(segments: list[Segment]) -> np.ndarray:
    """Precomputed raw DTW distances for clustering (not 1-sim)."""
    n = len(segments)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = segment_pair_distance(segments[i], segments[j])
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _cluster_segments(
    segments: list[Segment],
    *,
    distance_threshold: float = CLUSTER_MAX_DIST,
    linkage: str = CLUSTER_LINKAGE,
) -> list[int]:
    if not segments:
        return []
    if len(segments) == 1:
        return [0]

    dist = _pairwise_distance_matrix(segments)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage=linkage,
        distance_threshold=distance_threshold,
    )
    labels = clustering.fit_predict(dist)
    return labels.tolist()


def _nms_occurrences(
    occs: list[MotifOccurrence],
    *,
    iou_thresh: float = NMS_IOU,
) -> list[MotifOccurrence]:
    kept: list[MotifOccurrence] = []
    for o in sorted(occs, key=lambda x: (-(x.score or 0.0), x.start)):
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


def _mean_intra_dist(segments: list[Segment], idxs: list[int]) -> float:
    if len(idxs) < 2:
        return 0.0
    dists: list[float] = []
    for a in range(len(idxs)):
        for b in range(a + 1, len(idxs)):
            dists.append(segment_pair_distance(segments[idxs[a]], segments[idxs[b]]))
    return float(np.mean(dists)) if dists else 0.0


def _segments_to_motif_types(
    segments: list[Segment],
    *,
    labels: list[int] | None = None,
    name_prefix: str = "Type",
    start_type_idx: int = 0,
    start_occ: int = 0,
    stats: MotifDetectStats | None = None,
    intra_max_dist: float = INTRA_MAX_DIST,
    min_occurrences: int = MOTIF_MIN_OCCURRENCES,
) -> list[MotifType]:
    if stats is None:
        stats = MotifDetectStats()
    if not segments:
        return []
    if labels is None:
        labels = _cluster_segments(segments)

    clusters: dict[int, list[int]] = {}
    for i, lab in enumerate(labels):
        clusters.setdefault(int(lab), []).append(i)

    ranked = sorted(clusters.items(), key=lambda kv: -len(kv[1]))
    stats.clusters = len(ranked)
    motif_types: list[MotifType] = []
    type_idx = start_type_idx
    occ_counter = start_occ

    for _, idxs in ranked:
        if len(idxs) < 2:
            stats.dropped_size += 1
            continue
        if type_idx - start_type_idx >= len(TYPE_COLORS) * 2:
            break
        mean_motion = float(np.mean([segments[si].motion for si in idxs]))
        if mean_motion < MOTION_SPEED_MIN:
            stats.dropped_motion_mean += 1
            continue
        mean_intra = _mean_intra_dist(segments, idxs)
        if mean_intra > intra_max_dist:
            stats.dropped_intra += 1
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
        if len(occs) < min_occurrences:
            stats.dropped_nms += 1
            continue

        letter = _type_letter(type_idx)
        motif_types.append(
            MotifType(
                id=f"type_{letter.lower()}",
                name=f"{name_prefix} {letter}",
                color=TYPE_COLORS[type_idx % len(TYPE_COLORS)],
                occurrences=occs,
                meanIntraDist=round(mean_intra, 4),
            )
        )
        stats.type_mean_intras.append(round(mean_intra, 4))
        type_idx += 1

    stats.types_kept = len(motif_types)
    return motif_types


def detect_motif_types(
    actors: dict[str, TrackBuffer],
    *,
    aspect: float | None = None,
) -> list[MotifType]:
    """
    Valley phrases → mirror-aware DTW similarity → clusters.
    Prefer fewer clean types over polluted merges.
    """
    asp = float(aspect) if aspect and aspect > 0 else DEFAULT_ASPECT
    _set_geom_context(actors, asp)
    stats = MotifDetectStats()
    all_segments = build_segments(actors, stats=stats, aspect=asp)
    if len(all_segments) < 2:
        for f in MotifDetectStats.__dataclass_fields__:
            setattr(LAST_DETECT_STATS, f, getattr(stats, f))
        return []
    types = _segments_to_motif_types(all_segments, stats=stats)
    for f in MotifDetectStats.__dataclass_fields__:
        setattr(LAST_DETECT_STATS, f, getattr(stats, f))
    return types


def segment_near_time(
    actors: dict[str, TrackBuffer],
    actor_id: str,
    time_s: float,
    *,
    aspect: float | None = None,
) -> Segment | None:
    if aspect is not None:
        _set_geom_context(actors, aspect)
    elif not _CTX_SCALE_REFS:
        _set_geom_context(actors, DEFAULT_ASPECT)
    buf = actors.get(actor_id)
    if buf is None:
        return None
    segs = _gate_motion(_valley_segments(actor_id, buf))
    if len(segs) < 1:
        segs = _gate_motion(_sliding_segments(actor_id, buf))
    if not segs:
        times = np.array([p.time for p in buf.poses], dtype=np.float64)
        start = max(float(times[0]), time_s - WINDOW_S / 2)
        return _make_segment(actor_id, buf.poses, times, start, start + WINDOW_S)

    covering = [s for s in segs if s.start <= time_s <= s.end]
    if covering:
        return min(covering, key=lambda s: abs((s.start + s.end) / 2 - time_s))
    return min(segs, key=lambda s: min(abs(s.start - time_s), abs(s.end - time_s)))


def _poses_feature_series(
    poses: list,
    idxs: np.ndarray,
    *,
    actor_id: str,
    mirrored: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    One rich feature (+ mask) per pose index — no SEQ_FRAMES resampling.
    Returns (seq[T,D], mask[T,D], times[T]) or None if too thin.
    """
    if len(idxs) < 2:
        return None
    scale_ref = _scale_for(actor_id)
    frames: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    times: list[float] = []
    feat_dim: int | None = None
    missing = 0
    for ii in idxs:
        i = int(ii)
        k = poses[i].keypoints
        packed = (
            pose_feature_rich_mirrored(k, aspect=_CTX_ASPECT, scale_ref=scale_ref)
            if mirrored
            else pose_feature_rich(k, aspect=_CTX_ASPECT, scale_ref=scale_ref)
        )
        if packed is None:
            missing += 1
            if feat_dim is None:
                continue
            frames.append(np.zeros(feat_dim, dtype=np.float32))
            masks.append(np.zeros(feat_dim, dtype=bool))
        else:
            f, m = packed
            feat_dim = int(f.shape[0])
            frames.append(f.astype(np.float32))
            masks.append(np.asarray(m, dtype=bool))
        times.append(float(poses[i].time))
    if not frames or feat_dim is None:
        return None
    if missing / max(len(frames), 1) > DEGENERATE_FRAME_FRAC:
        return None
    if len(frames) < 4:
        return None
    return (
        np.stack(frames, axis=0),
        np.stack(masks, axis=0),
        np.asarray(times, dtype=np.float64),
    )


def _subsequence_end_costs(
    query: np.ndarray,
    query_mask: np.ndarray,
    ref: np.ndarray,
    ref_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Open-begin subsequence DTW.

    Returns (end_cost[m], start_index[m]) where end_cost[j] is the normalized
    path cost ending at reference frame j, and start_index[j] is the backtracked
    start (or -1 if unreachable). Normalization matches full DTW: path/(n+span).
    """
    n, m = len(query), len(ref)
    if n == 0 or m == 0 or m < max(4, n // 2):
        return (
            np.full(max(m, 1), 1e3, dtype=np.float64),
            np.full(max(m, 1), -1, dtype=np.int32),
        )

    dists, _med = _masked_dist_matrix(
        np.asarray(query, dtype=np.float64),
        np.asarray(ref, dtype=np.float64),
        np.asarray(query_mask, dtype=bool),
        np.asarray(ref_mask, dtype=bool),
    )
    inf = 1e9
    cost = np.full((n + 1, m + 1), inf, dtype=np.float64)
    # Free begin: matching can start at any reference frame
    cost[0, :] = 0.0

    for i in range(1, n + 1):
        di = dists[i - 1]
        for j in range(1, m + 1):
            d = float(di[j - 1])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    end_cost = np.full(m, 1e3, dtype=np.float64)
    start_idx = np.full(m, -1, dtype=np.int32)
    for j in range(1, m + 1):
        path = cost[n, j]
        if not np.isfinite(path) or path >= inf / 2:
            continue
        ii, jj = n, j
        while ii > 0 and jj > 0:
            opts = [
                (cost[ii - 1, jj - 1], ii - 1, jj - 1),
                (cost[ii - 1, jj], ii - 1, jj),
                (cost[ii, jj - 1], ii, jj - 1),
            ]
            _c, ii, jj = min(opts, key=lambda t: t[0])
        # ii == 0 → free-begin landed; start ref frame is jj (0-based)
        s = int(np.clip(jj, 0, j - 1))
        span = max(j - s, 1)
        end_cost[j - 1] = float(path / (n + span))
        start_idx[j - 1] = s
    return end_cost, start_idx


def _local_minima_hits(
    end_cost: np.ndarray,
    start_idx: np.ndarray,
    times: np.ndarray,
    *,
    match_max_dist: float,
    query_start: float | None = None,
    query_end: float | None = None,
    query_dur: float | None = None,
) -> list[tuple[float, float, float]]:
    """
    One hit per contiguous below-threshold valley (argmin inside the run).

    End positions whose match span overlaps the query seed are suppressed so the
    self-match plateau cannot swallow nearby true repeats into one giant valley.
    """
    m = len(end_cost)
    if m < 3:
        return []
    qdur = float(query_dur) if query_dur and query_dur > 0 else PHRASE_MIN_S
    min_dur = max(0.20, qdur / TEMPO_RATIO_MAX)
    cost = end_cost.copy()
    if query_start is not None and query_end is not None:
        for j in range(m):
            if start_idx[j] < 0:
                continue
            s = int(start_idx[j])
            t0 = float(times[s])
            t1 = float(times[j])
            inter = max(0.0, min(query_end, t1) - max(query_start, t0))
            union = max(query_end, t1) - min(query_start, t0)
            if union > 0 and inter / union >= 0.25:
                cost[j] = 1e3
    below = (cost <= match_max_dist) & (start_idx >= 0)
    hits: list[tuple[float, float, float]] = []
    i = 0
    while i < m:
        if not bool(below[i]):
            i += 1
            continue
        j = i + 1
        while j < m and bool(below[j]):
            j += 1
        k = i + int(np.argmin(cost[i:j]))
        s = int(start_idx[k])
        t0 = float(times[s])
        t1 = float(times[k])
        # If DTW collapsed the span, expand to query duration about the midpoint
        # so tempo ratio stays meaningful and short true hits survive.
        if (t1 - t0) < min_dur:
            mid = 0.5 * (t0 + t1)
            half = 0.5 * qdur
            t0 = max(float(times[0]), mid - half)
            t1 = min(float(times[-1]), mid + half)
        if (t1 - t0) < min_dur:
            i = j
            continue
        hits.append((t0, t1, float(cost[k])))
        i = j
    return hits


def _tempo_ratio(hit_dur: float, query_dur: float) -> float:
    q = max(float(query_dur), 1e-3)
    return float(hit_dur) / q


def _tempo_in_bounds(ratio: float) -> bool:
    r = float(ratio)
    if r <= 0:
        return False
    return (1.0 / TEMPO_RATIO_MAX) <= r <= TEMPO_RATIO_MAX


def format_tempo_label(ratio: float | None) -> str:
    """English UI fragment, e.g. '1.4× slower' / '1.2× faster' / ''."""
    if ratio is None:
        return ""
    r = float(ratio)
    if r >= 1.08:
        return f"{r:.1f}× slower"
    if r <= 0.92:
        return f"{1.0 / r:.1f}× faster"
    return ""


def query_segment_at_time(
    actors: dict[str, TrackBuffer],
    actor_id: str,
    time_s: float,
    *,
    start: float | None = None,
    end: float | None = None,
    window_s: float = WINDOW_S,
    aspect: float | None = None,
) -> Segment | None:
    """
    Build a Find-similar query window without valley boundaries.

    Prefer explicit [start, end] (selected occurrence). Otherwise a window of
    ``window_s`` centred on ``time_s``, clipped to the track.
    """
    if aspect is not None:
        _set_geom_context(actors, aspect)
    elif not _CTX_SCALE_REFS:
        _set_geom_context(actors, DEFAULT_ASPECT)
    buf = actors.get(actor_id)
    if buf is None or len(buf.poses) < 4:
        return None
    times = np.asarray([p.time for p in buf.poses], dtype=np.float64)
    t_lo, t_hi = float(times[0]), float(times[-1])
    if start is not None and end is not None and end > start:
        q0, q1 = float(start), float(end)
    else:
        half = window_s * 0.5
        q0 = max(t_lo, float(time_s) - half)
        q1 = min(t_hi, q0 + window_s)
        if q1 - q0 < window_s * 0.8:
            q0 = max(t_lo, q1 - window_s)
    seg = _make_segment(actor_id, buf.poses, times, q0, q1)
    if seg is not None:
        return seg
    # Fallback: build a minimal segment even if motion/feature gates fail
    idxs = np.where((times >= q0) & (times <= q1))[0]
    if len(idxs) < 4:
        return None
    packed = _poses_feature_series(
        buf.poses, idxs, actor_id=actor_id, mirrored=False
    )
    packed_m = _poses_feature_series(
        buf.poses, idxs, actor_id=actor_id, mirrored=True
    )
    if packed is None or packed_m is None:
        return None
    seq, mask, _ = packed
    seq_m, mask_m, _ = packed_m
    emb = _trajectory_embedding(seq, mask)
    emb_m = _trajectory_embedding(seq_m, mask_m)
    return Segment(
        actor_id=actor_id,
        start=float(q0),
        end=float(q1),
        embedding=emb,
        embedding_mirror=emb_m,
        seq=seq,
        seq_mirror=seq_m,
        seq_mask=mask,
        seq_mirror_mask=mask_m,
        motion=1.0,
    )


def find_similar_motif(
    actors: dict[str, TrackBuffer],
    query: Segment,
    *,
    match_max_dist: float = SUBSEQ_MATCH_MAX_DIST,
    name: str = "Similar 1",
    type_id: str = "type_sim_1",
    color: str = "#78b4a0",
    aspect: float | None = None,
) -> MotifType | None:
    """
    Find Similar via open-begin/open-end subsequence DTW on the full track.

    Tempo-invariant warping; duration ratio is reported and bounded by
    TEMPO_RATIO_MAX. Auto motif discovery is unchanged (still segment-based).
    """
    asp = float(aspect) if aspect and aspect > 0 else _CTX_ASPECT or DEFAULT_ASPECT
    _set_geom_context(actors, asp)
    buf = actors.get(query.actor_id)
    if buf is None or len(buf.poses) < 4:
        return None

    poses = buf.poses
    times_all = np.asarray([p.time for p in poses], dtype=np.float64)
    track_idxs = np.arange(len(poses), dtype=np.int64)
    track = _poses_feature_series(
        poses, track_idxs, actor_id=query.actor_id, mirrored=False
    )
    track_m = _poses_feature_series(
        poses, track_idxs, actor_id=query.actor_id, mirrored=True
    )
    if track is None:
        return None
    ref_seq, ref_mask, ref_times = track

    q_idxs = np.where((times_all >= query.start) & (times_all <= query.end))[0]
    if len(q_idxs) < 4:
        # Fall back to nearest indices spanning the query window
        i0 = int(np.argmin(np.abs(times_all - query.start)))
        i1 = int(np.argmin(np.abs(times_all - query.end)))
        if i1 < i0:
            i0, i1 = i1, i0
        q_idxs = np.arange(i0, max(i0 + 1, i1 + 1), dtype=np.int64)
    q_pack = _poses_feature_series(
        poses, q_idxs, actor_id=query.actor_id, mirrored=False
    )
    if q_pack is None:
        return None
    q_seq, q_mask, _q_times = q_pack
    query_dur = max(float(query.end - query.start), 1e-3)

    # Same-side + mirrored reference (query stays unmirrored; mirror the track)
    candidates: list[tuple[float, float, float]] = []
    for ref_s, ref_mk in (
        (ref_seq, ref_mask),
        (track_m[0], track_m[1]) if track_m is not None else (None, None),
    ):
        if ref_s is None or ref_mk is None:
            continue
        end_cost, start_idx = _subsequence_end_costs(q_seq, q_mask, ref_s, ref_mk)
        candidates.extend(
            _local_minima_hits(
                end_cost,
                start_idx,
                ref_times,
                match_max_dist=match_max_dist,
                query_start=query.start,
                query_end=query.end,
                query_dur=query_dur,
            )
        )

    # Dedup near-identical (start,end), keep best distance
    candidates.sort(key=lambda h: h[2])
    filtered: list[tuple[float, float, float, float]] = []  # start, end, dist, ratio
    for t0, t1, dist in candidates:
        ratio = _tempo_ratio(t1 - t0, query_dur)
        if not _tempo_in_bounds(ratio):
            continue
        # Skip heavy overlap with the query seed
        inter = max(0.0, min(query.end, t1) - max(query.start, t0))
        union = max(query.end, t1) - min(query.start, t0)
        if union > 0 and inter / union >= NMS_IOU:
            continue
        filtered.append((t0, t1, dist, ratio))

    # NMS among hits (best distance first)
    kept: list[tuple[float, float, float, float]] = []
    for t0, t1, dist, ratio in sorted(filtered, key=lambda h: h[2]):
        drop = False
        for k0, k1, _kd, _kr in kept:
            inter = max(0.0, min(k1, t1) - max(k0, t0))
            union = max(k1, t1) - min(k0, t0)
            if union > 0 and inter / union >= NMS_IOU:
                drop = True
                break
        if not drop:
            kept.append((t0, t1, dist, ratio))

    # Tiered retention: all strong matches + a few best soft ones (time-diverse)
    strong = [h for h in kept if h[2] <= SUBSEQ_STRONG_DIST]
    soft = [h for h in kept if h[2] > SUBSEQ_STRONG_DIST]
    soft_kept: list[tuple[float, float, float, float]] = []
    for h in sorted(soft, key=lambda x: x[2]):
        mid = 0.5 * (h[0] + h[1])
        if any(
            abs(mid - 0.5 * (k[0] + k[1])) < SUBSEQ_SOFT_SEP_S
            for k in strong + soft_kept
        ):
            continue
        soft_kept.append(h)
        if len(soft_kept) >= SUBSEQ_SOFT_MAX_EXTRA:
            break
    kept = strong + soft_kept

    occs: list[MotifOccurrence] = [
        MotifOccurrence(
            id="sim_occ_query",
            actorId=query.actor_id,
            start=float(query.start),
            end=float(query.end),
            score=1.0,
            tempo_ratio=1.0,
            is_query=True,
        )
    ]
    for i, (t0, t1, dist, ratio) in enumerate(kept, start=1):
        occs.append(
            MotifOccurrence(
                id=f"sim_occ_{i}",
                actorId=query.actor_id,
                start=float(t0),
                end=float(t1),
                score=distance_to_match_score(dist),
                tempo_ratio=float(ratio),
                is_query=False,
            )
        )

    # Best-first after the query seed
    head, tail = occs[:1], occs[1:]
    tail.sort(key=lambda o: (-(o.score or 0.0), o.start))
    occs = head + tail

    if len(occs) < 2:
        return None
    return MotifType(
        id=type_id,
        name=name,
        color=color,
        occurrences=occs,
    )
