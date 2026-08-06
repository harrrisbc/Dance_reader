# WORK ORDER — Wave 1: make calibration cheap and honest (no behaviour change)

**To:** programming agent
**From:** Design agent
**Date:** 2026-08-06
**Answers:** [PROPOSAL_METRIC_MATH.md](./PROPOSAL_METRIC_MATH.md) items B3 (speedup half) + verification gaps.

## Why

Waves 2–4 each change the similarity distribution and each needs a calibration pass. Calibration
is currently a ~20 s pairwise run, which is why thresholds have been re-tuned twice by hand rather
than swept properly. Fix the tooling **before** changing any math, so every later wave is cheap to
verify and hard to fudge.

Nothing in this wave may change a single similarity value.

## Changes

### 1. Vectorise `_dtw_distance` — result-identical (P0)

`motifs.py:_dtw_distance` is a Python double loop calling `np.linalg.norm` per cell. Precompute the
pair's frame-distance matrix with `scipy.spatial.distance.cdist` (or an einsum equivalent) and run
the DP over that array. Keep the Sakoe-Chiba band and the `path / (n + m)` normalisation exactly
as they are.

**Acceptance is numerical identity**, not "close": assert the new implementation reproduces the old
one to within floating-point tolerance across every pair on the fixture. Land the assertion as a
test so a later refactor cannot silently change results.

Do **not** raise `SEQ_FRAMES` or retune the band in this wave. That is a separate decision once the
budget exists.

### 2. Add a negative fixture (P0)

Every threshold discussion so far has only had a clip that *does* contain repeats. There is no test
that the system can correctly answer **"no repeats here"**, which is the failure mode that
clip-relative thresholds would introduce in Wave 3.

**Source video supplied by user (2026-08-06):** `C:\Users\USER\Downloads\hi.mp4`
(~3.7 MB). Intended as a clip with **no repeated phrase**.

Analyze it with the current pipeline (default S model is fine unless detection is bad), save the
project as a second fixture next to the positive one — suggested name `hi.dance.json` — and
document the path + duration + dancer count in `docs/AGENT_HANDOFF_INDEX.md`.

Before treating it as the negative fixture, **spot-check by eye** that it really has no clear
repeated phrase. If it does contain an obvious repeat, stop and ask design; do not redefine the
fixture quietly.

### 3. Report in distance units too (P1)

`backend/scripts/calibrate_motifs.py` currently reports similarity. Also report the underlying
**raw DTW distance** distribution (min / p25 / median / p75 / p95 / max), and per-gate survival
counts in both units. Wave 3 will move thresholds into distance units; the script needs to speak
that language first.

Add a `--probe` line for the negative fixture showing how many types would be produced at a range
of candidate thresholds.

## Acceptance

- New DTW matches the old to floating-point tolerance on every fixture pair, enforced by a test.
- Full pairwise pass on the fixture drops from ~20 s to under ~2 s.
- `detect_motif_types()` on the fixture still returns the same 4 types with the same occurrences.
- Calibration script prints both similarity and raw-distance distributions.
- A negative fixture exists and is documented.

## Out of scope

- Any change to features, segmentation, thresholds, or `SEQ_FRAMES`
- Occlusion handling (Wave 2)
