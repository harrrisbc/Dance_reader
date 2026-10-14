# WORK ORDER — redefine `level` (metric semantics)

**To:** programming agent  
**From:** Design agent  
**Date:** 2026-08-06  
**Status:** done — shipped with GVHMR Phase 2 ([2026-10-15 design](./superpowers/specs/2026-10-15-gvhmr-3d-metrics-phase2-design.md); notes [IMPLEMENTATION_NOTES_GVHMR_3D_METRICS.md](./IMPLEMENTATION_NOTES_GVHMR_3D_METRICS.md))  
**Answers:** [PROPOSAL_METRIC_MATH.md](./PROPOSAL_METRIC_MATH.md) item A5 — design decision made below.

## Why

`level` is currently leg length divided by that track's 90th-percentile leg length. Two defects:

1. **Self-referential.** If the dancer never fully straightens, the reference is a bent leg and
   everything reads "high".
2. **Confuses foreshortening with bending.** Leg *length* in the image shortens when the dancer
   faces the camera exactly as it does when they bend.

## Decision

**Use the vertical drop from hip to supporting ankle, divided by torso height.**

```
level = (ankle_y − hip_y) / torso_height      # square-pixel space, supporting (lower) ankle
```

Why this and not the alternatives:

- **Vertical drop is invariant to the rotation that breaks the current definition.** Rotating about
  the vertical axis changes the horizontal and depth components and leaves the vertical one
  untouched, so turning no longer masquerades as crouching. This is the whole reason to prefer it
  over leg length. Camera pitch and forward lean still affect it — second-order, accept for now.
- **Torso height as the reference is anatomical, not self-referential.** Standing lands near a
  consistent value regardless of how straight this particular dancer ever got, so the number
  transfers across dancers and clips.
- **Knee angle is rejected as the definition of `level`.** It measures *how* a dancer is low, not
  *how low* they are. A choreographer reading "level" means where the body sits in vertical space —
  Laban's low / middle / high. Knee angle is a legitimate separate metric later; it is not this one.

Use the **supporting** ankle — the lower of the two — for the same reason `pos_y` does. A lifted
knee must not read as the whole body rising.

## Changes

1. Implement the definition above in `metrics.py`, replacing the percentile reference.
2. Derive the low / middle / high band boundaries from real data rather than carrying over
   0.45 / 0.8. Report what standing, plié, deep crouch, and floor work actually measure on the
   fixture, and set the bands from that.
3. Re-lock the chart Y range to contain the observed data, and verify the fraction outside the
   range is near zero — standing lesson 3.
4. Update the axis label if the unit changes.

## Acceptance

- A dancer turning without changing height produces a flat `level` curve.
- Crouching produces a clear drop; standing sits near the top of the middle-to-high band.
- A lifted knee does not raise `level`.
- Band boundaries derived from measured values, with the measurements recorded.
- Chart range contains the data.

## Out of scope

- Knee angle as an additional metric
- Jump / airborne detection
