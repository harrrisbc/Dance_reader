# WORK ORDER — Wave 4: Find Similar by subsequence DTW

**To:** programming agent
**From:** Design agent
**Date:** 2026-08-06
**Answers:** [PROPOSAL_METRIC_MATH.md](./PROPOSAL_METRIC_MATH.md) item B2, including the tempo design call.
**Depends on:** Waves 1–3.

## Why

Phrase boundaries currently decide the answer. The fixture shows it cleanly — all three are the
same move by eye:

| Pair | Segment lengths | Similarity |
|---|---|---|
| 8 s ↔ 10.3 s | 0.55 s vs 0.58 s | 0.646 |
| 8 s ↔ 12 s | 0.55 s vs 1.71 s | 0.419 |

The third scores near the floor purely because the segmenter handed it a 1.7 s box. This is the
largest accuracy win available, and it lands on **Find Similar** — the feature the user invokes by
hand and therefore judges most harshly.

## Changes

### 1. Subsequence DTW for Find Similar (P0)

Replace segment-to-segment matching with **open-start, open-end subsequence DTW**: slide the query
across the whole track, take local minima of the accumulated cost, apply NMS to overlapping hits.
Phrase boundaries stop mattering for this path entirely.

Keep segment-based clustering for **automatic motif type discovery** — it needs discrete units to
cluster. Multi-scale windows there are a later question, not this handoff.

### 2. Tempo — match it, but tell the user (design call)

The proposal asks whether tempo difference should count against a match. Every segment is currently
resampled to `SEQ_FRAMES = 14`, so tempo is discarded entirely.

**Decision: keep tempo-invariant matching, and surface the duration ratio.**

Performing a phrase slower or faster is a deliberate compositional device, not noise — a tool that
finds those variations is more useful than one that misses them. But the user must be able to tell
an exact repeat from a stretched one, so show the ratio in the occurrence list, e.g.
`12.0–13.7s · 92% · 1.4× slower`.

**Bound it.** Unbounded tempo invariance lets a 0.5 s flick match a 3 s slow phrase. Reject matches
whose duration ratio exceeds roughly 2× in either direction, and make that a named constant.

### 3. Scores and ordering

Occurrences sorted best-first, each showing its match score (display mapping from Wave 3) and the
duration ratio. The query's own span should be identifiable rather than appearing as a mysterious
perfect match.

## Acceptance

- Find Similar from the ~8 s phrase returns the ~10.3 s **and** the ~12 s occurrences, despite the
  1.7 s segmentation box that currently defeats the ~12 s one.
- Results do not change materially when phrase boundaries shift — verify by nudging valley
  parameters and confirming the same hits.
- A deliberately half-speed repeat is found and labelled with its ratio.
- A match beyond the 2× ratio bound is rejected.
- Negative fixture: Find Similar from an arbitrary phrase returns few or no hits.
- Auto motif discovery unchanged by this handoff.

## Out of scope

- Multi-scale windows for automatic discovery
- Raising `SEQ_FRAMES` (separate decision after Wave 1's speedup)
- Cross-dancer unison / canon detection — noted as a future idea, not now
