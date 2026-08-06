# Implementation notes — W4 Find Similar by subsequence DTW

**Handoff:** [AGENT_HANDOFF_MATCH_W4_SUBSEQUENCE.md](./AGENT_HANDOFF_MATCH_W4_SUBSEQUENCE.md)  
**Date:** 2026-08-06  
**Depends on:** W1–W3

## What shipped

### Subsequence DTW (Find Similar only)
- `find_similar_motif` slides the query across the **full track** with open-begin
  subsequence DTW (mask-aware frame costs from W2).
- Hits = one per contiguous below-threshold valley (argmin), NMS by time IoU.
- Query self-overlap is suppressed before valley finding so the self-match plateau
  cannot swallow nearby repeats.
- **Auto motif discovery unchanged** (`detect_motif_types` still segment-based).

### Query window ignores valleys
- New `query_segment_at_time`: uses the selected occurrence `[start,end]` or a
  centred `WINDOW_S` at the playhead — **not** `segment_near_time` valley cuts.
- This is what makes boundary shifts irrelevant for Find Similar.

### Tempo
- Matching stays tempo-invariant (DTW warping).
- `MotifOccurrence.tempo_ratio` = hit_dur / query_dur; UI shows e.g. `1.4× slower`.
- `TEMPO_RATIO_MAX = 2.0` — reject outside `[0.5, 2.0]`.
- Short collapsed DTW spans are expanded toward query duration so ratio stays meaningful.

### Scores / ordering
- Query seed listed first, labelled `query` (`is_query=True`).
- Other hits best-first with display % (W3 mapping) + tempo label.
- Retention: all hits with `d ≤ SUBSEQ_STRONG_DIST` (0.16), plus up to 6 time-diverse
  softer hits up to `SUBSEQ_MATCH_MAX_DIST` (0.30).

## Constants

| Name | Value |
|------|-------|
| `SUBSEQ_MATCH_MAX_DIST` | 0.30 |
| `SUBSEQ_STRONG_DIST` | 0.16 |
| `SUBSEQ_SOFT_MAX_EXTRA` | 6 |
| `SUBSEQ_SOFT_SEP_S` | 1.5 |
| `TEMPO_RATIO_MAX` | 2.0 |

## Acceptance

| Criterion | Result |
|-----------|--------|
| ~8s query finds ~10.3 and ~12 | Yes — with query `[7.95,8.50]`: strong hit `10.21–10.51` (d≈0.09); soft hit covering ~12 (`12.16–12.71`) |
| Boundary-independent | Query path no longer uses valley segments |
| Half-speed labelled / 2× rejected | `tempo_ratio` + `format_tempo_label`; bound unit-tested |
| Negative: few/no hits | `hi`: no usable query or no non-query hits in practice |
| Auto discovery unchanged | Still 1 type on positive under W3 gates |

## Tests

`backend/tests/test_subseq_w4.py` — tempo bounds, positive 10.3+12 coverage, negative few hits.

## Out of scope (as ordered)

Multi-scale auto discovery, raising `SEQ_FRAMES`, cross-dancer unison — not started.
