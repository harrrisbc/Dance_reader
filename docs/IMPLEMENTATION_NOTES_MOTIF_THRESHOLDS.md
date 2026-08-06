# Implementation notes — motif threshold recalibration

## Problem

DTW similarity `sim = 1/(1+d)` peaks ~0.76 on noisy synthetic repeats and ~0.62 on a
real rehearsal clip, but gates were still at cosine-era **0.86**. Everything failed.

## Chosen thresholds (after `calibrate_motifs.py` on real data)

| Constant | Was | Hand-off start | **Landed** | Notes |
|----------|-----|----------------|------------|--------|
| `SIMILARITY_MIN` | 0.86 | 0.62 | **0.55** | Real near-repeat topped ~0.619 on `girl.dance.json` |
| `CLUSTER_DISTANCE` | 0.20 | 0.40 | **0.45** | Needs sim ≥ ~0.55 |
| `INTRA_SIM_MIN` | 0.86 | 0.60 | **0.50** | Same band as cluster |
| `COSINE_PREFILTER` | 0.55 | keep | **0.55** | Kept; rejects return **0.0** (no fabricated half-cosine) |

Girl clip calibration (1 dancer, 75 phrases after motion gate):

- Pairwise max sim **0.619**; only 1 pair ≥ 0.55
- `detect_motif_types` → **1 type / 2 occurrences**
- Unrelated bulk of cosine-pass pairs sits median ~0.12 — still well below gates

## Other fixes

- Cosine prefilter miss → `return 0.0` (reject), not `0.5 * cosine`
- `MotifDetectStats` + `motifDiag` on `AnalysisResult`; Qt status shows  
  `N phrases · M clusters · K passed similarity …`
- `find_similar` requires ≥ 2 NMS occurrences (query alone is not a hit)
- `LAST_DETECT_STATS` updated **in place** so importers see live stats

## Tool

```powershell
.\backend\.venv\Scripts\python.exe backend\scripts\calibrate_motifs.py path\to\file.dance.json
```

Also accepts bare `analysis.json`.
