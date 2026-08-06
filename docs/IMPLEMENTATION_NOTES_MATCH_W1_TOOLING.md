# Implementation notes — W1 calibration tooling

**Handoff:** [AGENT_HANDOFF_MATCH_W1_TOOLING.md](./AGENT_HANDOFF_MATCH_W1_TOOLING.md)  
**Date:** 2026-08-06

## What shipped

### 1. Vectorised `_dtw_distance` (result-identical)

- `backend/app/pipeline/motifs.py`: precompute frame distances with `scipy.spatial.distance.cdist`, keep the same Sakoe-Chiba band and `path / (n + m)` normalisation.
- Reference loop kept as `_dtw_distance_naive` for identity tests only.
- `segment_raw_dtw_distance()` added (mirror-aware raw DTW, no cosine / no reach-level penalty) so calibration can speak distance units without changing `segment_similarity`.

**No behaviour change:** `SEQ_FRAMES`, band, thresholds, features untouched. Similarity values match the naive path to float64 noise (`rtol=1e-9`, `atol=1e-7`).

### 2. Negative fixture

| | |
|--|--|
| Source | `C:\Users\USER\Downloads\hi.mp4` |
| Project | `C:\Users\USER\Downloads\hi.dance.json` (copy: `backend/uploads/hi.dance.json`) |
| Duration | **7.60 s** |
| Frame | 1920×1080 @ 25 fps |
| Dancers | **1** (S model) |
| Spot-check | Continuous stage walk / preparation (Prix de Lausanne wide shot). **No clear repeated choreographic phrase** by eye across evenly sampled frames. |

**Current detector note (expected until W3):** under today's gates the clip still yields **1 motif type** (3 overlapping walk phrases that look alike). That is exactly why this fixture exists — Wave 3 must be able to answer zero here without killing the positive fixture.

### 3. Calibration script

`backend/scripts/calibrate_motifs.py` now prints:

- Pairwise **similarity** distribution (as before)
- Pairwise **raw DTW distance** distribution (all pairs + cosine-pass)
- Gate survival in **both** units
- Type-count sweep over candidate `(cluster_distance, intra_sim_min)` pairs
- `--negative` skips the default 8/10/12 s probes and labels the type sweep for a no-repeat clip
- Probe lines also print `raw_dtw` beside `sim`

```text
backend\.venv\Scripts\python.exe backend\scripts\calibrate_motifs.py path\to\hi.dance.json --negative
backend\.venv\Scripts\python.exe backend\scripts\calibrate_motifs.py path\to\07….dance.json
```

### 4. Tests

`backend/tests/test_dtw_w1.py` (unittest):

- Random sequences: new ≡ naive within FP tolerance
- Every fixture pair (same + mirror): identity
- Pairwise speed budget &lt; 5 s (measured ~0.01 s on the positive fixture)
- `detect_motif_types` still returns **4** types on the positive fixture

```text
cd backend
set PYTHONPATH=.
..\.venv\Scripts\python.exe tests\test_dtw_w1.py -v
```

(On this machine: `backend\.venv\Scripts\python.exe`.)

## Timing (positive fixture, 15 phrases)

| Path | Pairwise (incl. mirror) |
|------|-------------------------|
| Vectorised | ~0.012 s |
| Naive reference | ~0.043 s |

(Full pairwise is already well under the &lt;2 s acceptance; the historical ~20 s figure was from a much larger segment count on other clips.)

## Out of scope (as ordered)

W2–W4 not started. No feature / threshold / `SEQ_FRAMES` changes.
