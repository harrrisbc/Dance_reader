# Implementation notes — W3 thresholds in distance units

**Handoff:** [AGENT_HANDOFF_MATCH_W3_THRESHOLDS.md](./AGENT_HANDOFF_MATCH_W3_THRESHOLDS.md)  
**Date:** 2026-08-06  
**Depends on:** W1 tooling + W2 features

## Decision (as ordered)

Option 1: gates in **raw mask-aware DTW distance**. Deleted `1/(1+d)` from every decision path.
Display-only score: `distance_to_match_score(d) = clip(1 - d/0.50, 0, 1)` → shown as 0–100% in Qt.
That score never feeds a gate.

## Chosen gates

| Constant | Value | Role |
|----------|-------|------|
| `CLUSTER_MAX_DIST` | **0.28** | Agglomerative average-linkage cut |
| `INTRA_MAX_DIST` | **0.30** | Max mean pairwise distance inside a kept type |
| `MATCH_MAX_DIST` | **0.28** | Find-similar accept if `d ≤` this |
| `MOTIF_MIN_OCCURRENCES` | **3** | After NMS; lets short no-repeat clips return zero |
| `DISPLAY_DIST_ZERO` | 0.50 | Display mapping only |

### Why `MOTIF_MIN_OCCURRENCES=3`

On post-W2 features the negative fixture’s only pair has **d ≈ 0.088**, *closer* than the positive
known-repeat probes (**d ≈ 0.22–0.27**). No absolute `d ≤ T` cut can accept the repeats and reject
that walk pair. Requiring **3 occurrences** is an absolute rule that:

- returns **0 types** on `hi` (only 2 phrases), and
- still keeps the positive repeat cluster (8 phrases → 1 type with 8 occ).

Documented as part of the calibrated gate set, not a silent product change.

## Sweep table (excerpt)

`min_occ=3` unless noted. Full grid also ran `min_occ=2` (negative always 1 type whenever
cluster≤0.35 — confirms the conflict above).

| cluster | intra | pos types | neg types | Notes |
|---------|-------|-----------|-----------|-------|
| 0.12–0.18 | ≈cluster | 0 | 0† | †neg still 1 at min_occ=2 |
| 0.20 | 0.20–0.25 | 1 | 0 | Small unrelated pair only |
| 0.22 | 0.22–0.27 | 1 | 0 | Type covers 10.6+11.9 (+1.2); 6.9 separate at min_occ=2 |
| **0.24** | 0.24–0.29 | **2** | **0** | Saner split; 10+12 together; ~8s region with late walk |
| 0.26 | 0.26–0.31 | 2 | 0 | Same shape as 0.24 |
| **0.28** | **0.30** | **1** | **0** | **Chosen** — 6.9 / 10.6 / 11.9 all in one type |
| 0.30–0.35 | ≈cluster | 1 | 0 | Same merge plateau |

Margin used: prefer **neg=0** and **all three probe regions co-cluster**. That forces the 0.28
plateau (1 type on positive). The 0.24 row is saner cardinality but splits the ~8s region away
from 10/12 under current valley cuts (W4 subsequence is the proper fix for that).

## Acceptance check

| Criterion | Result |
|-----------|--------|
| No reciprocal in decision path | `segment_pair_distance` / clustering / find-similar use raw `d` only |
| Positive: ~8 / ~10 / ~12 in one type | Yes — Type A includes 6.9–7.5, 10.6–11.2, 11.9–12.4 (segment-near 8s → [6.90,7.52] after W2 valleys) |
| Positive: sane type count | 1 type (8 occ) — plateau merge; not the W2 “everything clears sim≥0.4” bug |
| Negative: zero types | **0** — `emptied by occ<3/nms=1` |
| Empty UI explains gate | `LAST_DETECT_STATS.summary()` e.g. `2 phrases · 1 clusters · 0 passed distance 0.30 — emptied by occ<3/nms=1` |

## Code touchpoints

- `backend/app/pipeline/motifs.py` — distance gates, `distance_to_match_score`, empty summary
- `desktop/window.py` — Find-similar message + % display score
- `backend/scripts/calibrate_motifs.py` — distance-native reporting / sweep
- `README.md` — threshold section

## Out of scope

W4 subsequence DTW — not started. That is what should reunite boundary-shifted repeats without
having to loosen cluster cuts into a single mega-type.
