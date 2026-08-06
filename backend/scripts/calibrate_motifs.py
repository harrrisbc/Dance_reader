"""Calibrate motif distance thresholds against a real .dance.json project.

Usage:
  backend\\.venv\\Scripts\\python.exe backend\\scripts\\calibrate_motifs.py path\\to\\file.dance.json

Prints pairwise raw DTW-distance distributions and how many pairs/clusters
pass the current distance gates. Do not use pixel-identical synthetics.

  --negative   No-repeat clip mode: skip time probes; emphasize type-count sweep.

Precision criterion (auto discovery): prefer missing a borderline repeat over
a mega-type that mixes different moves. False-merge smell: 1 type with ≥6 occ
on a long clip that also contains walking.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pipeline.metrics import enrich_metrics, frame_aspect  # noqa: E402
from app.pipeline.motifs import (  # noqa: E402
    CLUSTER_LINKAGE,
    CLUSTER_MAX_DIST,
    COSINE_PREFILTER,
    COSINE_REJECT_DIST,
    INTRA_MAX_DIST,
    LAST_DETECT_STATS,
    MATCH_MAX_DIST,
    MOTIF_MIN_OCCURRENCES,
    MotifDetectStats,
    _cluster_segments,
    _segments_to_motif_types,
    build_segments,
    detect_motif_types,
    distance_to_match_score,
    segment_near_time,
    segment_pair_distance,
    segment_path_visibility,
    segment_raw_dtw_distance,
)
from app.project_io import buffers_from_analysis, load_project  # noqa: E402
from app.schemas import AnalysisResult  # noqa: E402

# Locked chart ranges after torso-height normalizer (must contain real data)
REACH_YLIM = (0.5, 4.0)
SPEED_YLIM = (0.0, 7.0)
TRAVEL_YLIM = (0.0, 2.5)

# Candidate raw-DTW distance cuts for sweep reporting
DIST_THRESHOLDS = (0.12, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.35)

# False-merge smell: single mega-type on a clip long enough to contain walk + gesture
MEGA_TYPE_OCC_SMELL = 6
MEGA_TYPE_DURATION_S = 12.0


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    return float(np.percentile(np.asarray(xs, dtype=np.float64), p))


def _print_dist(label: str, xs: list[float]) -> None:
    if not xs:
        print(f"{label}: (empty)")
        return
    xs_sorted = sorted(xs)
    print(f"{label} (n={len(xs)}):")
    print(f"  min    {xs_sorted[0]:.4f}")
    print(f"  p25    {_percentile(xs, 25):.4f}")
    print(f"  median {_percentile(xs, 50):.4f}")
    print(f"  p75    {_percentile(xs, 75):.4f}")
    print(f"  p95    {_percentile(xs, 95):.4f}")
    print(f"  max    {xs_sorted[-1]:.4f}")


def _metric_stats(analysis: AnalysisResult) -> None:
    reaches: list[float] = []
    speeds: list[float] = []
    travels: list[float] = []
    for actor in analysis.actors:
        for m in actor.metrics or []:
            reaches.append(float(m.reach))
            speeds.append(float(m.speed))
            travels.append(float(m.travel))
    if not reaches:
        print("No metrics after enrich.")
        return
    r = np.asarray(reaches, dtype=np.float64)
    print("Reach (× torso) after enrich:")
    print(
        f"  min={r.min():.2f} median={np.median(r):.2f} "
        f"p95={np.percentile(r, 95):.2f} max={r.max():.2f}"
    )
    outside = float(np.mean((r < REACH_YLIM[0]) | (r > REACH_YLIM[1])))
    print(f"  fraction outside locked [{REACH_YLIM[0]}, {REACH_YLIM[1]}]: {outside:.1%}")
    s = np.asarray(speeds, dtype=np.float64)
    t = np.asarray(travels, dtype=np.float64)
    print(
        f"Speed torso/s: median={np.median(s):.2f} p95={np.percentile(s, 95):.2f} max={s.max():.2f}"
    )
    print(
        f"Travel torso/s: median={np.median(t):.2f} p95={np.percentile(t, 95):.2f} max={t.max():.2f}"
    )


def _false_merge_flags(
    types: list,
    *,
    duration_s: float,
    negative: bool,
) -> list[str]:
    """Precision-oriented smells — do not treat 'all probes in one type' as success."""
    flags: list[str] = []
    if negative:
        if len(types) > 0:
            flags.append(f"NEG_HAS_TYPES={len(types)} (want 0)")
        return flags
    n_types = len(types)
    if n_types == 0:
        flags.append("ZERO_TYPES (may be over-tight)")
    if n_types == 1 and duration_s >= MEGA_TYPE_DURATION_S:
        n_occ = len(types[0].occurrences)
        if n_occ >= MEGA_TYPE_OCC_SMELL:
            flags.append(
                f"MEGA_TYPE smell: 1 type with {n_occ} occ on {duration_s:.1f}s clip "
                f"(prefer ≥2 types or fewer occ)"
            )
    if n_types >= 2:
        flags.append(f"SPLIT_OK: {n_types} types (precision-friendly cardinality)")
    return flags


def _type_sweep(segs: list, *, negative: bool, duration_s: float, linkage: str) -> None:
    print(
        "Types at candidate distance thresholds "
        f"(linkage={linkage} / cluster_max / intra_max / "
        f"min_occ={MOTIF_MIN_OCCURRENCES}):"
    )
    print(
        "  Criterion: precision > recall — flag MEGA_TYPE; "
        "neg want 0 types; prefer ≥2 types on mixed clips."
    )
    for cluster_d in DIST_THRESHOLDS:
        for intra in (cluster_d, round(cluster_d + 0.02, 2)):
            labels = _cluster_segments(
                segs, distance_threshold=cluster_d, linkage=linkage
            )
            stats = MotifDetectStats()
            types = _segments_to_motif_types(
                segs,
                labels=labels,
                stats=stats,
                intra_max_dist=intra,
                min_occurrences=MOTIF_MIN_OCCURRENCES,
            )
            mark = ""
            if (
                abs(cluster_d - CLUSTER_MAX_DIST) < 1e-9
                and abs(intra - INTRA_MAX_DIST) < 1e-9
                and linkage == CLUSTER_LINKAGE
            ):
                mark = " ← current"
            smell = ""
            if (
                not negative
                and len(types) == 1
                and duration_s >= MEGA_TYPE_DURATION_S
                and len(types[0].occurrences) >= MEGA_TYPE_OCC_SMELL
            ):
                smell = " MEGA"
            print(
                f"  cluster≤{cluster_d:.2f} intra≤{intra:.2f}: "
                f"{len(types)} type(s) "
                f"(clusters={stats.clusters} drop_intra={stats.dropped_intra} "
                f"drop_nms={stats.dropped_nms}){smell}{mark}"
            )
    if negative:
        print("  (negative fixture: want 0 types at the chosen gate)")


def _print_type_membership(types: list) -> None:
    for t in types:
        spans = ", ".join(f"{o.start:.1f}–{o.end:.1f}" for o in t.occurrences)
        intra = (
            f" mean_intra={t.meanIntraDist:.3f}"
            if t.meanIntraDist is not None
            else ""
        )
        print(f"  - {t.name}: {len(t.occurrences)} occ{intra} [{spans}]")


def main() -> int:
    parser = argparse.ArgumentParser(description="Motif distance-threshold calibration")
    parser.add_argument("project", type=Path, help=".dance.json or analysis.json path")
    parser.add_argument(
        "--probe",
        type=float,
        nargs="*",
        default=None,
        help="Times (s) to compare as a known repeated phrase "
        "(default 8 10 12; empty list disables; ignored with --negative). "
        "Named false-merge pairs: require different types after retune when supplied.",
    )
    parser.add_argument(
        "--negative",
        action="store_true",
        help="No-repeat fixture mode: skip time probes; emphasize type-count sweep",
    )
    parser.add_argument(
        "--linkage",
        choices=("average", "complete"),
        default=None,
        help=f"Agglomerative linkage for sweep (default: current {CLUSTER_LINKAGE})",
    )
    args = parser.parse_args()
    path = args.project
    if not path.is_file():
        print(f"Not found: {path}", file=sys.stderr)
        return 1

    linkage = args.linkage or CLUSTER_LINKAGE

    data = json.loads(path.read_text(encoding="utf-8"))
    if "analysis" in data:
        project = load_project(path)
        analysis = project.analysis
    else:
        analysis = AnalysisResult.model_validate(data)

    enrich_metrics(analysis)
    aspect = frame_aspect(int(analysis.frame_width or 0), int(analysis.frame_height or 0))
    print(f"Project: {path.name}")
    if args.negative:
        print("Mode: NEGATIVE (expect no repeated phrase)")
    print(f"Duration: {analysis.duration:.2f}s  actors: {len(analysis.actors)}")
    print(f"Aspect W/H: {aspect:.4f} (frame {analysis.frame_width}×{analysis.frame_height})")
    _metric_stats(analysis)
    print()

    actors = buffers_from_analysis(analysis)
    if not actors:
        print("No actors / poses in project.", file=sys.stderr)
        return 1

    stats = MotifDetectStats()
    segs = build_segments(actors, stats=stats, aspect=aspect)
    print(f"Actors: {len(actors)}")
    print(
        f"Phrases: raw={stats.phrases_raw} after_motion={stats.phrases} "
        f"(dropped_motion={stats.dropped_motion})"
    )
    print()
    print("Current auto-discovery distance gates:")
    print(f"  CLUSTER_MAX_DIST     = {CLUSTER_MAX_DIST}")
    print(f"  INTRA_MAX_DIST       = {INTRA_MAX_DIST}")
    print(f"  CLUSTER_LINKAGE      = {CLUSTER_LINKAGE}")
    print(f"  MATCH_MAX_DIST       = {MATCH_MAX_DIST} (legacy; Find Similar uses SUBSEQ_*)")
    print(f"  MOTIF_MIN_OCCURRENCES = {MOTIF_MIN_OCCURRENCES}")
    print(f"  COSINE_PREFILTER     = {COSINE_PREFILTER}")
    print(f"  Sweep linkage override = {linkage}")
    print()

    if len(segs) < 2:
        print("Fewer than 2 phrases — cannot form pairwise distribution.")
        types = detect_motif_types(actors, aspect=aspect)
        print(f"detect_motif_types → {len(types)} types")
        print(f"  {LAST_DETECT_STATS.summary()}")
        return 0

    raw_dists: list[float] = []
    raw_dists_cos_pass: list[float] = []
    cos_pass = 0
    cos_reject = 0
    n = len(segs)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = segs[i], segs[j]
            d = segment_pair_distance(a, b)
            raw = segment_raw_dtw_distance(a, b)
            raw_dists.append(raw)
            if d >= COSINE_REJECT_DIST / 2:
                cos_reject += 1
            else:
                cos_pass += 1
                raw_dists_cos_pass.append(raw)

    _print_dist("Pairwise raw DTW distances (all pairs)", raw_dists)
    print()
    _print_dist("Pairwise raw DTW distances (cosine-pass only)", raw_dists_cos_pass)
    print()
    print("Pairs passing threshold (raw DTW distance, cosine-pass only):")
    for d_thr in DIST_THRESHOLDS:
        k = sum(1 for d in raw_dists_cos_pass if d <= d_thr)
        print(f"  dist <= {d_thr:.2f}: {k} / {len(raw_dists_cos_pass)}")
    print()
    print(f"Cosine prefilter: pass={cos_pass} reject={cos_reject}")
    print()

    if args.negative:
        probe_times: list[float] = []
    elif args.probe is None:
        probe_times = [8.0, 10.0, 12.0]
    else:
        probe_times = list(args.probe)

    probe_segs: list[tuple[float, object]] = []
    if probe_times and len(probe_times) >= 2:
        print("Probe phrases at", probe_times)
        aid = next(iter(actors))
        for t in probe_times:
            seg = segment_near_time(actors, aid, float(t), aspect=aspect)
            if seg:
                probe_segs.append((t, seg))
                print(f"  t={t:.1f}s → [{seg.start:.2f}, {seg.end:.2f}]")
            else:
                print(f"  t={t:.1f}s → no segment")
        for i in range(len(probe_segs)):
            for j in range(i + 1, len(probe_segs)):
                ti, a = probe_segs[i]
                tj, b = probe_segs[j]
                dist = segment_pair_distance(a, b)
                vis = segment_path_visibility(a, b)
                score = distance_to_match_score(dist)
                print(
                    f"  d({ti:.0f}s,{tj:.0f}s) = {dist:.4f}  "
                    f"display={score * 100:.0f}%  "
                    f"path_low_vis={vis['low_vis_frac']:.0%}"
                )
        print()

    _type_sweep(
        segs,
        negative=bool(args.negative),
        duration_s=float(analysis.duration),
        linkage=linkage,
    )
    print()

    types = detect_motif_types(actors, aspect=aspect)
    print(f"detect_motif_types → {len(types)} type(s) (production gates)")
    print(f"  {LAST_DETECT_STATS.summary()}")
    print(f"  {LAST_DETECT_STATS.detail()}")
    _print_type_membership(types)

    flags = _false_merge_flags(
        types, duration_s=float(analysis.duration), negative=bool(args.negative)
    )
    print()
    print("False-merge / precision check:")
    if flags:
        for f in flags:
            print(f"  · {f}")
    else:
        print("  · (no flags)")

    # If user named probes that should stay split, report shared type membership
    if probe_segs and len(types) >= 1 and not args.negative:
        print()
        print("Probe co-membership (same type?):")
        aid = next(iter(actors))

        def _type_for(t: float) -> str | None:
            for mt in types:
                for o in mt.occurrences:
                    if o.start - 0.05 <= t <= o.end + 0.05 or abs(
                        (o.start + o.end) / 2 - t
                    ) < 0.8:
                        # prefer covering span
                        if o.start - 0.15 <= t <= o.end + 0.15:
                            return mt.name
            # fall back: segment-near centres
            for mt in types:
                for o in mt.occurrences:
                    if abs((o.start + o.end) / 2 - t) < 1.0:
                        return mt.name
            return None

        # Use segment windows from probes for membership
        for t, seg in probe_segs:
            mid = (seg.start + seg.end) / 2
            owner = None
            for mt in types:
                for o in mt.occurrences:
                    if abs(o.start - seg.start) < 0.15 and abs(o.end - seg.end) < 0.15:
                        owner = mt.name
                        break
                if owner:
                    break
            if owner is None:
                owner = _type_for(mid)
            print(f"  t≈{t:.1f}s [{seg.start:.2f},{seg.end:.2f}] → {owner or 'no type'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
