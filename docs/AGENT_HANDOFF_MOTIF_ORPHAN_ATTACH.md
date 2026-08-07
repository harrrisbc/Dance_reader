# WORK ORDER — Attach orphan phrases to motif types via subsequence match

**To:** programming agent
**From:** Design agent
**Date:** 2026-08-07
**Priority:** P1 — user: same motif missed on `07 走路+武拜廟.dance 2.dance.json`

## Why

On the positive fixture / user’s re-export:

| Region | Valley segment | In Type A? |
|--------|----------------|------------|
| ~8 s | `[6.90, 7.52]` | **No** |
| ~10.3 s | `[10.56, 11.16]` | Yes |
| ~12 s | `[11.91, 12.44]` | Yes |

Probe distances: `d(8,10)≈0.27`, `d(8,12)≈0.27`, `d(10,12)≈0.22`.  
Auto gates (`complete` @ 0.24 / 0.26) correctly keep 10+12 (+ early 1.2) and **exclude** the ~8 s valley box — that was the over-merge trade-off.

But by eye and by **Find Similar** from playhead ~8.2 s (`query [7.92, 8.47]`), hits include ~10.2 and ~12.2. So the repeat **is** matchable; auto discovery never gets a second chance after clustering.

**Do not** loosen `CLUSTER_MAX_DIST` back toward 0.28 — that recreates mega-types.  
**Do** give orphans a Find-Similar-style second pass against kept type exemplars.

## Scope

- `backend/app/pipeline/motifs.py` — post-cluster attach step inside `detect_motif_types` / `_segments_to_motif_types`
- Tests on the fixture JSON (path in index / Downloads)
- Short implementation notes; update index

## Changes

1. After motif types are kept (current gates unchanged), collect **orphan** phrases: motion-gated segments not already in any kept type’s occurrence list (IoU / time overlap).
2. For each kept type, pick an **exemplar** (e.g. occurrence closest to the type’s median centre, or lowest mean distance to siblings).
3. Run **subsequence DTW** (reuse W4 path / helpers) of exemplar vs full track **or** vs each orphan window — prefer matching orphans with `d ≤ SUBSEQ_STRONG_DIST` (0.16) first; optionally allow up to a named `ORPHAN_ATTACH_MAX_DIST` ≤ `SUBSEQ_MATCH_MAX_DIST` if strong-only is too thin on the fixture.
4. Attach orphan → type only if:
   - distance clears the attach gate, and
   - tempo ratio within `TEMPO_RATIO_MAX`, and
   - NMS against existing occs in that type.
5. Recompute `meanIntraDist` after attaches (segment-pair among valley members still in the type; orphans attached only via subseq may be excluded from intra mean, or included with a note — pick one and document).
6. **Do not** create new types from orphans alone in this handoff.
7. Status / `motifDiag` should mention attaches if easy, e.g. `+2 orphan attaches`.

## Acceptance

- [ ] On `07 走路+武拜廟…dance.json` (or `…dance 2.dance.json`): Type that owns ~10.6 and ~11.9 also gains an occurrence covering the **~8 s** gesture (centre near 8 s, not only the `[6.90,7.52]` mis-cut if subseq finds a better span).
- [ ] `hi.dance.json` still **0** types (no new false types from orphan logic).
- [ ] No return of MEGA_TYPE (1 type · ≥6 walk-mixed occ) on the positive fixture under production gates.
- [ ] Find Similar behaviour unchanged.
- [ ] Cluster / intra constants **unchanged**.

## Out of scope

- Loosening cluster cuts
- Redesigning valley segmentation (separate; note that probe at t=8 maps to `[6.90,7.52]` which is part of the miss)
- Feature / level math
