# WORK ORDER — Wave 3: thresholds in distance units, and the ability to say "no"

**To:** programming agent
**From:** Design agent
**Date:** 2026-08-06
**Answers:** [PROPOSAL_METRIC_MATH.md](./PROPOSAL_METRIC_MATH.md) item B1 — design decision made below.
**Depends on:** Waves 1 and 2.

## Why

`sim = 1 / (1 + d)` compresses the whole decision range into about a tenth of the scale. On the
fixture the pairwise range is 0.324–0.774 while all three gates sit at 0.40 / 0.45 / 0.50 — every
decision crammed just above the median. That is why thresholds have been retuned twice and why they
will not transfer to the next clip.

## Decision

**Option 1: set thresholds in raw DTW distance units and delete the reciprocal mapping.**

Reasoning:

- After Wave 2's mask-aware normalisation, the distance is a **mean per-dimension deviation in
  torso-height units**. That is dimensionless, interpretable, and comparable across clips. A
  threshold like "0.5 torso heights of average deviation" is something a human can reason about.
- The reciprocal is a monotonic squash that adds no information and destroys interpretability.
  Removing it changes no rankings.
- **Option 2 (clip-relative thresholds) is rejected.** Its failure mode is fatal for this product:
  relative thresholds always find something. A tool whose job is answering "is this phrase
  repeated?" must be able to answer **no**. Manufacturing motif types from the best-of-a-bad-lot on
  a clip with no repeats is worse than returning nothing.

## Changes

### 1. Decide in distance, display in score

Move `SIMILARITY_MIN`, `CLUSTER_DISTANCE`, and `INTRA_SIM_MIN` to distance-space constants with
names that say so (e.g. `MATCH_MAX_DIST`). All gate comparisons use raw distance.

The UI shows users a match score where higher is better, so keep a **display-only** mapping from
distance to a 0–100% match figure. Decision variable and display variable are separate concerns;
do not let the display mapping leak back into any gate.

### 2. Calibrate by sweep, not by hand

With Wave 1's fast DTW, sweep candidate thresholds across both fixtures and pick from the resulting
curve rather than by trial. Report, for each candidate:

- Types produced on the positive fixture, and whether the three known repeats group
- Types produced on the **negative fixture** — this is the constraint that stops the threshold
  drifting loose

Choose the value with the widest margin between "finds the known repeats" and "finds nothing on the
negative fixture". Record the sweep table, not just the chosen number.

### 3. Say why when nothing is found

When zero types survive, the Qt status line should say which gate emptied it, e.g.
`14 phrases · 5 clusters · 0 passed distance 0.55`. Silent emptiness is what made the original bug
take this long to find.

## Acceptance

- No reciprocal mapping remains in any decision path.
- Positive fixture: the ~8 s / ~10.3 s / ~12 s repeats land in one type, and the clip still yields
  a sane number of types rather than collapsing into one.
- **Negative fixture: zero or near-zero motif types.**
- Threshold sweep table recorded in the implementation notes.
- Empty results explain themselves in the UI.

## Out of scope

- Subsequence matching (Wave 4)
- Changing features again
