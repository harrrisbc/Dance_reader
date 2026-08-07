# Implementation notes — Motif orphan attach

**Handoff:** [AGENT_HANDOFF_MOTIF_ORPHAN_ATTACH.md](./AGENT_HANDOFF_MOTIF_ORPHAN_ATTACH.md)  
**Date:** 2026-08-07

## What shipped

After cluster gates keep types (unchanged `complete` @ 0.24 / 0.26), a **second pass**
attaches orphan valley phrases to those types via open-begin subsequence DTW (W4 helpers):

1. Orphans = motion-gated segments not IoU-covered by any kept occurrence.
2. Per type, exemplar = occurrence closest to the type’s median centre.
3. Subseq search with `ORPHAN_ATTACH_MAX_DIST = SUBSEQ_STRONG_DIST` (0.16).
4. Attach a hit only if tempo is in bounds, NMS-clear vs existing occs, and near an
   unused orphan (`ORPHAN_ATTACH_NEAR_S=1.25` — covers the valley mis-cut
   `[6.90,7.52]` → gesture centre ~8 s).
5. **No new types** from orphans alone.
6. `meanIntraDist` stays the **valley-cluster** mean (subseq attaches excluded).
7. `LAST_DETECT_STATS` / motifDiag: `+N orphan attaches`.

Cluster / intra / linkage constants were **not** loosened.

## Fixture result (`07 …dance 2.dance.json`)

Type A (mean_intra≈0.21 valley-only) ends with occurrences covering ~1.2, **~6.0**,
**~7.6–8.1**, ~10.6, ~11.9 — the ~8 s gesture is present without a MEGA_TYPE (≥6
walk-mixed). `hi.dance.json` still **0** types.

## How to test

```powershell
.\backend\.venv\Scripts\python.exe -m unittest tests.test_orphan_attach tests.test_subseq_w4 tests.test_overmerge -v
```

Re-analyze or re-open the positive project in Qt — Type A should list a hit near 8 s;
status may show `+N orphan attaches`.
