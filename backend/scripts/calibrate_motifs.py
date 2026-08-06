"""Calibrate motif distance thresholds against a real .dance.json project.

Usage:
  backend\\.venv\\Scripts\\python.exe backend\\scripts\\calibrate_motifs.py path\\to\\file.dance.json

Prints pairwise raw DTW-distance distributions and how many pairs/clusters
pass the current distance gates. Do not use pixel-identical synthetics.

  --negative   No-repeat clip mode: skip time probes; emphasize type-count sweep.
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


def _type_sweep(segs: list, *, negative: bool) -> None:
    print(
        "Types at candidate distance thresholds "
        f"(cluster_max / intra_max / min_occ={MOTIF_MIN_OCCURRENCES}):"
    )
    for cluster_d in DIST_THRESHOLDS:
        for intra in (cluster_d, round(cluster_d + 0.02, 2)):
            labels = _cluster_segments(segs, distance_threshold=cluster_d)
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
            ):
                mark = " ← current"
            print(
                f"  cluster≤{cluster_d:.2f} intra≤{intra:.2f}: "
                f"{len(types)} type(s) "
                f"(clusters={stats.clusters} drop_intra={stats.dropped_intra} "
                f"drop_nms={stats.dropped_nms}){mark}"
            )
    if negative:
        print(
            "  (negative fixture: want 0 types at the chosen gate)"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Motif distance-threshold calibration")
    parser.add_argument("project", type=Path, help=".dance.json or analysis.json path")
    parser.add_argument(
        "--probe",
        type=float,
        nargs="*",
        default=None,
        help="Times (s) to compare as a known repeated phrase "
        "(default 8 10 12; empty list disables; ignored with --negative)",
    )
    parser.add_argument(
        "--negative",
        action="store_true",
        help="No-repeat fixture mode: skip time probes; emphasize type-count sweep",
    )
    args = parser.parse_args()
    path = args.project
    if not path.is_file():
        print(f"Not found: {path}", file=sys.stderr)
        return 1

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
    print("Current distance gates (Wave 3):")
    print(f"  CLUSTER_MAX_DIST     = {CLUSTER_MAX_DIST}")
    print(f"  INTRA_MAX_DIST       = {INTRA_MAX_DIST}")
    print(f"  MATCH_MAX_DIST       = {MATCH_MAX_DIST}")
    print(f"  MOTIF_MIN_OCCURRENCES = {MOTIF_MIN_OCCURRENCES}")
    print(f"  COSINE_PREFILTER     = {COSINE_PREFILTER}")
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

    if probe_times and len(probe_times) >= 2:
        print("Probe phrases at", probe_times)
        probes = []
        aid = next(iter(actors))
        for t in probe_times:
            seg = segment_near_time(actors, aid, float(t), aspect=aspect)
            if seg:
                probes.append((t, seg))
                print(f"  t={t:.1f}s → [{seg.start:.2f}, {seg.end:.2f}]")
            else:
                print(f"  t={t:.1f}s → no segment")
        for i in range(len(probes)):
            for j in range(i + 1, len(probes)):
                ti, a = probes[i]
                tj, b = probes[j]
                dist = segment_pair_distance(a, b)
                vis = segment_path_visibility(a, b)
                score = distance_to_match_score(dist)
                print(
                    f"  d({ti:.0f}s,{tj:.0f}s) = {dist:.4f}  "
                    f"display={score * 100:.0f}%  "
                    f"path_low_vis={vis['low_vis_frac']:.0%}"
                )
        print()

    _type_sweep(segs, negative=bool(args.negative))
    print()

    types = detect_motif_types(actors, aspect=aspect)
    print(f"detect_motif_types → {len(types)} type(s)")
    print(f"  {LAST_DETECT_STATS.summary()}")
    print(f"  {LAST_DETECT_STATS.detail()}")
    for t in types:
        spans = ", ".join(f"{o.start:.1f}–{o.end:.1f}" for o in t.occurrences[:8])
        print(f"  - {t.name}: {len(t.occurrences)} occ [{spans}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
