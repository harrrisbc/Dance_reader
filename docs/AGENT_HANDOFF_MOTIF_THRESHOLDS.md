# WORK ORDER — motif/find-similar detects nothing (threshold scale bug)

**To:** programming agent
**From:** Design agent
**Date:** 2026-08-06
**Severity:** P0 — both auto motif types and 搵相似 return empty on every clip.

## Why

The motif-accuracy pass replaced cosine similarity with **mirror-aware DTW**, mapped as
`sim = 1 / (1 + dtw_dist)`. That is a different numeric scale from the old cosine — but the
thresholds were left at cosine-era values (`0.82` → `0.86`). Nothing can reach them.

Measured with synthetic gestures (clean, low noise — real footage scores **lower** than this):

| Pair | Measured sim |
|------|--------------|
| Identical repeat (pixel-perfect) | 1.000 |
| Same gesture repeated, small noise + timing/amplitude variation | **0.758** |
| Same gesture mirrored (left side vs right side) | **0.789** |
| Different gesture (arm raise vs kick) | 0.348 |

Current gates:

| Constant | Value | Requires sim ≥ |
|----------|-------|----------------|
| `SIMILARITY_MIN` (搵相似) | 0.86 | 0.86 |
| `CLUSTER_DISTANCE` 0.20 (dist = 1 − sim) | — | 0.80 |
| `INTRA_SIM_MIN` (keep a type) | 0.86 | 0.86 |

**Every threshold sits above the entire realistic range.** A near-perfect synthetic repeat at
0.758 already fails all three. Hence zero types and zero similar hits.

Note the discrimination itself is **fine** — good repeats ≈ 0.76–0.79, unrelated ≈ 0.35, and
mirror invariance works (0.789 for L-vs-R, verified). Only the cut points are wrong.

## Changes

### 1. Recalibrate thresholds (P0)

Starting values from measured data, midpoint biased toward precision:

```python
SIMILARITY_MIN = 0.62      # was 0.86  — 搵相似
CLUSTER_DISTANCE = 0.40    # was 0.20  — i.e. cluster needs sim >= 0.60
INTRA_SIM_MIN = 0.60       # was 0.86
```

These are a **starting point, not final**. Tune on 2–3 real rehearsal clips: the target is that
an obvious repeated phrase merges while an unrelated phrase does not. Record the values you land
on in the implementation notes.

### 2. Add a calibration script (P0 — prevents this recurring)

`backend/scripts/calibrate_motifs.py` (or similar): load a `.dance.json`, build segments, print
the **distribution** of pairwise similarities — min / p25 / median / p75 / max, plus how many
pairs pass each threshold, and how many clusters survive each gate.

This is what was missing. The previous pass validated against a *pixel-identical* synthetic
mirror (sim = 1.0), which cannot reveal a threshold set above the real range.

### 3. Fix the cosine prefilter fallback (P1)

```python
if max(cos_same, cos_mir) < COSINE_PREFILTER:
    return float(max(0.0, max(cos_same, cos_mir)) * 0.5)
```

Two problems:

- It returns a **fabricated score** on a different scale than the DTW branch, so the two
  branches are not comparable.
- Anything failing the prefilter can never exceed `0.275`, so a marginal prefilter miss becomes
  a permanent exclusion — DTW never gets to judge it.

Return `0.0` and treat it as a plain reject. Also verify `COSINE_PREFILTER = 0.55` does not
reject true matches: in testing, genuine same/mirror pairs had cosine 0.90–0.99, so 0.55 looks
safe, but confirm against the calibration distribution before keeping it.

### 4. Explain empty results in the UI (P1)

When zero types are produced, the user currently sees a blank list with no reason. Surface a
short diagnostic in the Qt status bar, e.g.:

> `18 phrases · 4 clusters · 0 passed similarity 0.60`

Log per-gate drop counts (motion gate, cluster size < 2, intra-sim, post-NMS < 2) so the next
failure is diagnosable without a bisect.

## Acceptance

- On a real clip with an obvious repeated phrase, at least one motif type appears.
- 搵相似 on a clear repeat returns more than just the query segment itself.
- A left-side and right-side version of the same phrase land in the same type.
- An unrelated gesture does not join that type.
- Calibration script prints a similarity distribution and per-gate survival counts.
- Chosen thresholds recorded in `docs/IMPLEMENTATION_NOTES_MOTIF_THRESHOLDS.md`.

## Out of scope

- Changing the similarity metric itself (`1/(1+d)` is fine once thresholds match it)
- Learned embeddings / ActionCLIP
- Web review UI (being retired — see index)
