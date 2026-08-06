# Implementation notes — Motif auto-discovery over-merge

**Handoff:** [AGENT_HANDOFF_MOTIF_OVERMERGE.md](./AGENT_HANDOFF_MOTIF_OVERMERGE.md)  
**Date:** 2026-08-06  
**Depends on:** W3 distance gates + W4 Find Similar

## Decision

**Precision > recall** for automatic motif types. Missing a borderline repeat is better
than one mega-type that mixes walk / prep / gesture. True partners split by valley cuts
remain recoverable via **Find Similar** (unchanged `SUBSEQ_*`).

| Constant | Was (W3) | Now | Role |
|----------|----------|-----|------|
| `CLUSTER_MAX_DIST` | 0.28 | **0.24** | Preferred 0.22–0.24 band |
| `INTRA_MAX_DIST` | 0.30 | **0.26** | ≈ cluster + 0.02 |
| `CLUSTER_LINKAGE` | `"average"` | **`"complete"`** | Stops A~B~C chains |
| `MOTIF_MIN_OCCURRENCES` | 3 | **3** | Still required so `hi` → 0 |
| `SUBSEQ_*` | — | **unchanged** | Find Similar recall |

### Why complete linkage (same change, with table)

Average linkage at 0.24 already left the mega-type plateau, but still produced a **4-occ
type** gluing ~6.4–7.5 with late walk ~14.6–15.7 — still “looks like” mush. Complete
linkage at the same cut keeps only the tight **1.2 + 10.6 + 11.9** cluster (mean_intra
0.21) and drops size-2 pairs under `min_occ=3`.

## Sweep table (precision criterion)

**Criterion:** no `MEGA_TYPE` (1 type · ≥6 occ on ≥12 s clip); `hi` → 0 types; prefer
clean 10/12 co-membership over forcing ~8 into the same type.

Positive fixture · `min_occ=3`:

| linkage | cluster | intra | types | Membership / notes |
|---------|---------|-------|-------|--------------------|
| average | 0.28 | 0.30 | **1 MEGA** | 8 occ — W3 plateau (reject) |
| average | 0.24 | 0.24–0.26 | 2 | Type A 4 occ (~6.4–7.5+14.6–15.7 mush); Type B 10/12+1.2 |
| average | 0.22 | 0.22–0.24 | 1 | Only 10/12+1.2; pairs of 2 fail min_occ |
| **complete** | **0.24** | **0.26** | **1** | **Chosen** — Type A: 1.2–1.8, 10.6–11.2, 11.9–12.4; mean_intra=0.21; **no MEGA** |
| complete | 0.22 | 0.24 | 0 | Over-tight |
| complete | 0.32 | 0.32 | 1 MEGA | Back to plateau |

Negative `hi`: **0 types** at chosen gate (`emptied by occ<3/nms=1`). Absolute distance
still cannot separate walk pairs (d≈0.088) from true repeats; keep min_occ=3 until
features improve (out of scope).

## Acceptance

| Criterion | Result |
|-----------|--------|
| No single mega-type on positive | Yes — 1 type · 3 occ (not ≥6) |
| 10/12 grouped with true partners | Yes — both in Type A |
| ~8 with 10/12 via auto discovery | No — segment at 6.9 fails min_occ alone; **Find Similar** recovers ~10.3 and ~12 |
| `hi` → 0 types | Yes |
| W4 Find Similar regression | Tests green (`test_subseq_w4`, `test_overmerge`) |
| Intra-distance visible | `MotifType.meanIntraDist` + Qt `d̄=…` + `LAST_DETECT_STATS` mean_intra list |

## Code touchpoints

- `backend/app/pipeline/motifs.py` — gates, `CLUSTER_LINKAGE`, stats / mean intra on types
- `backend/app/schemas.py` — `MotifType.meanIntraDist`
- `desktop/window.py` — type list shows `d̄=`
- `backend/scripts/calibrate_motifs.py` — MEGA_TYPE smell, `--linkage`, membership dump
- `backend/tests/test_overmerge.py` — regression
- `README.md` — threshold blurb

## Out of scope (unchanged)

Feature redesign so walks score farther than true repeats; raising `SEQ_FRAMES`; floor
homography; display-% remapping.
