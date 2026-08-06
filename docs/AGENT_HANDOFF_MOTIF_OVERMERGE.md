# WORK ORDER — Motif auto-discovery: stop over-merging “looks like”

**To:** programming agent
**From:** Design agent
**Date:** 2026-08-06
**Priority:** P0 — user-visible regression after W1–W4

## Why

User report: automatic motif types now **group phrases that look vaguely similar but are
different moves** into one type. That matches a known W3 trade-off, not a mysterious new bug.

After W2, pairwise raw DTW on the positive fixture compressed to roughly **0.18–0.32**.
True repeats sit at **~0.22–0.27**. Unrelated but locomotion-similar pairs (negative `hi`) sit
at **~0.088** — *closer* than the true repeats. Absolute distance alone cannot separate
“same choreography” from “same kind of walking”.

W3 then chose `CLUSTER_MAX_DIST = 0.28` / `INTRA_MAX_DIST = 0.30` so the three probe regions
(~8 / ~10 / ~12) would co-cluster on one fixture. The sweep notes already say **0.24 was the
saner cardinality** and **0.28 is a merge plateau** (1 mega-type with 8 occ). W4 fixed Find
Similar via subsequence DTW; **auto discovery was left on the loose segment cluster**. That is
what the user is looking at.

**Product decision now: prefer precision over recall for automatic motif types.**
Missing a borderline repeat is better than inventing a type that mixes different moves.
True repeats that segment boundaries split can still be found with **Find Similar** (W4).

## Scope

- `backend/app/pipeline/motifs.py` — cluster / intra / min-occ gates; possibly linkage
- `backend/scripts/calibrate_motifs.py` — re-sweep with a **false-merge** criterion
- Qt status / empty messaging only if gate names change
- Docs: index + short implementation notes

## Changes

### 1. Retighten auto-discovery cuts (P0)

Re-sweep **only** `detect_motif_types` gates on both fixtures. Starting candidates from the W3
table (do not invent numbers without the table):

| cluster | intra | Expected shape (from W3 notes) |
|---------|-------|--------------------------------|
| 0.20 | 0.22–0.25 | Fewer merges; may miss co-clustering 8 with 10/12 |
| **0.22–0.24** | ≈cluster+0.02 | **Preferred band** — saner type count on positive |
| 0.28 | 0.30 | Current — reject for auto discovery |

**Default direction:** move `CLUSTER_MAX_DIST` / `INTRA_MAX_DIST` into the **0.22–0.24** band
unless the sweep shows an equally precise cut that still keeps clearly identical repeats
together on the positive fixture.

Do **not** loosen Find Similar (`SUBSEQ_*`) to compensate. Auto discovery and Find Similar are
allowed to disagree: discovery = conservative catalogue; Find Similar = user-directed recall.

### 2. Add a false-merge acceptance check (P0)

Calibration today only asks “do the three known regions land in one type?” That question
**forces** over-merge. Add a second bar:

- On the positive fixture, report type count and membership of every kept type (start–end list).
- Flag any kept type whose members span **visually distinct** regions if the user has named
  them; until then, treat **type count = 1 with ≥6 occ on a 16 s clip that also contains walking**
  as a smell — prefer ≥2 types when the sweep allows it without bringing `hi` back to >0 types.

Keep `hi` at **0 types** (or document why not). If tightening alone cannot get `hi→0` without
`MOTIF_MIN_OCCURRENCES=3`, keep min_occ=3 and say so — that hack stays until features separate
locomotion from gesture (out of scope here).

### 3. Consider complete linkage (P1, only if 0.22–0.24 still mushy)

Average linkage lets long chains of “somewhat similar” segments glue into one type (A~B, B~C ⇒
A with C). If after retightening the user still sees mixed types, switch agglomerative
`linkage` from `"average"` to `"complete"` (max pairwise inside a cluster must clear the cut)
and re-sweep. Do not change linkage and thresholds in the same commit without a before/after
table.

### 4. UI honesty (P1)

When a type has many occurrences, the status / motif list should make pairwise spread visible
enough to debug — e.g. show mean intra-distance next to the type, or keep printing it in
`LAST_DETECT_STATS`. Silent mega-types are how this regressed past the user.

## Acceptance

- [ ] Auto discovery no longer collapses the positive fixture into a single mega-type **unless**
      every occurrence is still the same move by the probe list (document membership).
- [ ] Preferred outcome on positive: **≥2 types** with the clear 10/12 (and ideally ~8) repeats
      still grouped with their true partners — not with unrelated walk/prep if the sweep can
      split them.
- [ ] Negative `hi`: still **0** motif types (or justified exception in notes).
- [ ] Find Similar on ~8 s still recovers ~10.3 and ~12 (W4 acceptance must not regress).
- [ ] Sweep table recorded in implementation notes with the **precision** criterion explicit.
- [ ] Index updated; this handoff marked done when shipped.

## Out of scope

- Redesigning features so walks score farther than true repeats (real fix long-term; needs its
  own proposal — richer gesture dims / motion direction / rejecting pure locomotion phrases)
- Raising `SEQ_FRAMES`
- Floor homography / `level` redefinition
- Changing display `%` mapping unless gates rename

## User follow-up (attach when available)

Ask for one concrete false-merge example: clip path + approximate seconds of two occurrences that
were wrongly put in the same type. If supplied, add them as named probes in
`calibrate_motifs.py --probe` and require they land in **different** types after the retune.
