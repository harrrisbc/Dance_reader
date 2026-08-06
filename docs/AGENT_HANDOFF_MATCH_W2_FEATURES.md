# WORK ORDER — Wave 2: feature correctness (batched, one calibration at the end)

**To:** programming agent
**From:** Design agent
**Date:** 2026-08-06
**Answers:** [PROPOSAL_METRIC_MATH.md](./PROPOSAL_METRIC_MATH.md) items A1, A2, A3, A4.
**Depends on:** Wave 1 (fast DTW + negative fixture).

## Why

Four separate defects all change the feature or segmentation scale. Shipping them one at a time
means four calibration passes and four sets of throwaway thresholds — the pattern that has already
burned two rounds. **Batch all four, then calibrate once.**

Do not tune thresholds inside this wave. Report what the distribution becomes; Wave 3 sets the cuts.

## Changes

### 1. A1 — stop fabricating occluded joints (P0, the important one)

`metrics.py` `pose_feature` writes `[0.0, 0.0]` for any keypoint below `TORSO_CONF`. In
torso-normalized space that is **exactly the torso centre**, so an occluded wrist reads as "wrist
held at the belly" rather than "unknown". `pose_feature_rich` inherits it for bone vectors, where a
zero-length "unit vector" is meaningless. `_dtw_distance` then sums these fabrications as evidence.

Carry a per-frame **visibility mask** alongside the feature vector. Make the frame distance
mask-aware: sum over jointly-visible dimensions only and divide by the visible-dimension count.

Two details the proposal left open, both of which matter:

**a. Occluded frames must never be cheap.** If a low-evidence frame pair yields a small distance,
DTW will route its warping path straight through the occluded region to buy a cheap alignment.
Below a minimum shared-visible-joint count, do not emit a small distance — impute a cost at or
above the running median frame distance for that pair. Verify on the fixture that warping paths do
not preferentially pass through low-visibility frames.

**b. Same problem in `_sample_sequence`.** It pads short sequences by repeating the last frame
(`frames.append(frames[-1])`). That is another fabrication DTW treats as real evidence. Mark padded
frames as zero-visibility so the mask logic covers them.

Dividing by the visible-dimension count also makes the distance a **mean per-dimension deviation**,
which is dimensionless and comparable across clips. Wave 3 depends on that property — it is what
makes absolute distance thresholds defensible.

### 2. A2 — scale-normalize energy, and use the same joints as speed (P0)

`metrics.py` `energy += v * v` uses raw square-pixel velocity while speed divides by torso scale.
Within one clip a dancer walking toward the camera therefore gains "energy" for identical movement
— and the fixture is a walking clip. Divide velocity by frame scale before squaring.

Energy also sums all 17 keypoints while speed uses `KEY_INDICES`. Face and head keypoints are noisy;
use the same joint set as speed.

This shifts the energy curve, which moves the segmentation valleys, which moves every segment. That
is expected and is precisely why this is batched here rather than shipped alone.

Keep the displayed unit as **relative to the actor's own peak**. Absolute energy would be
torso²/s², which is not a quantity a choreographer can read. Note in the code that scale
normalization now makes cross-clip comparison possible if we ever want it.

### 3. A3 — smooth speed and energy (P1)

Raw 60 fps frame differences with no low-pass. On the fixture, Speed p95 is 3.5 torso/s against a
max of 6.75 — the top of the range is spikes, not dancing.

Apply a short centred filter. Prefer **Savitzky-Golay** over a moving average: it preserves peak
height, and accent height is exactly what a choreographer reads off the Energy curve.

Window ~5 frames at 60 fps (~83 ms), below the ~100 ms threshold at which dance accents become
perceptually distinct. Scale the window by frame rate rather than hard-coding a frame count.

**Design decision — smooth the displayed curve and the segmenter input identically.** A valley
visible on the chart that the segmenter did not use, or vice versa, makes every future
phrase-boundary complaint undebuggable. If a later need for different smoothing appears, make it
explicit and show both.

### 4. A4 — delete the stale reach/level penalty (P1)

`motifs.py` adds `0.08 × Δreach + 0.15 × Δlevel` onto the DTW distance. Those weights were picked
when reach could hit 168×; the units have since changed by two orders of magnitude and the weights
were never re-derived.

**Decision: delete them.** The pose features already encode limb extension and leg configuration,
so this is double-counting via two magic numbers that will go stale again the next time units move.
Simpler is the right default.

If the Wave 2 calibration shows measurably worse separation without them, say so with numbers and
we will reintroduce derived weights — but do not keep them "just in case".

### 5. Re-check the motion gate

`MOTION_SPEED_MIN` / `MOTION_REL_FRAC` operate on raw speed, which A2 and A3 both change. Report
how many segments survive the gate before and after; flag if it moves a lot.

## Calibration (once, at the end)

Run `calibrate_motifs.py` on both fixtures. Report, in the implementation notes:

- Pairwise distribution in **raw distance** and similarity, before and after this wave
- Segment count and motion-gate survival, before and after
- The three known repeats (~8 s / ~10.3 s / ~12 s) — distance and similarity for each pair
- Whether occluded frames still attract DTW warping paths

State the expected direction of each change **before** running it, per standing lesson 1.

## Acceptance

- No feature vector contains a fabricated coordinate for an unseen joint.
- Padded sequence frames are excluded from the distance by the mask.
- Low-visibility frame pairs are not cheaper than well-observed ones.
- Energy is scale-normalized and uses `KEY_INDICES`.
- Displayed and segmenter curves use identical smoothing.
- Reach/level penalty removed (or retained with derived weights and stated evidence).
- Both fixtures still run end to end; results reported, thresholds untouched.

## Out of scope

- Setting thresholds (Wave 3)
- Subsequence matching (Wave 4)
- `level` redefinition (separate handoff)
- Raising `SEQ_FRAMES`
