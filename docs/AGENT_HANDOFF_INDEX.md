# Agent handoff index

Design agent maintains this list. Programming agent: work **top to bottom** unless the user reprioritizes.

| Status | Topic | Handoff | Notes |
|--------|--------|---------|--------|
| done | **P1** Motif orphan attach | [AGENT_HANDOFF_MOTIF_ORPHAN_ATTACH.md](./AGENT_HANDOFF_MOTIF_ORPHAN_ATTACH.md) | Strong subseq attach of orphans to kept types; cluster 0.24 untouched |
| done | **P1** Always-visible time readout | [AGENT_HANDOFF_TIME_READOUT.md](./AGENT_HANDOFF_TIME_READOUT.md) | `0:08.42 / 0:16.20` left of slider; clock before chart throttle |
| done | **P0** Floor map live during play | [AGENT_HANDOFF_FLOOR_LIVE.md](./AGENT_HANDOFF_FLOOR_LIVE.md) | Floor dot throttled ~10 Hz while playing; no freeze-until-pause |
| done | **P0** Motif over-merge | [AGENT_HANDOFF_MOTIF_OVERMERGE.md](./AGENT_HANDOFF_MOTIF_OVERMERGE.md) | complete @ 0.24/0.26; no mega-type; precision > recall |
| done | **W1** Calibration tooling | [AGENT_HANDOFF_MATCH_W1_TOOLING.md](./AGENT_HANDOFF_MATCH_W1_TOOLING.md) | Vectorised DTW identity-tested; `hi.dance.json` negative fixture; distance-unit calibration |
| done | **W2** Feature correctness | [AGENT_HANDOFF_MATCH_W2_FEATURES.md](./AGENT_HANDOFF_MATCH_W2_FEATURES.md) | Masks + energy scale + SG smooth + drop penalty; distances compressed → W3 must retune |
| done | **W3** Thresholds in distance units | [AGENT_HANDOFF_MATCH_W3_THRESHOLDS.md](./AGENT_HANDOFF_MATCH_W3_THRESHOLDS.md) | Raw DTW gates 0.28/0.30 + min_occ=3; neg→0; display % only |
| done | **W4** Find Similar by subsequence DTW | [AGENT_HANDOFF_MATCH_W4_SUBSEQUENCE.md](./AGENT_HANDOFF_MATCH_W4_SUBSEQUENCE.md) | Open-begin subseq DTW; tempo ratio ≤2×; query≠valley |
| pending | `level` redefinition | [AGENT_HANDOFF_LEVEL_DEFINITION.md](./AGENT_HANDOFF_LEVEL_DEFINITION.md) | Hip→supporting-ankle vertical drop / torso height. Independent of W1–W4 |
| answered | Metric & matching math proposal | [PROPOSAL_METRIC_MATH.md](./PROPOSAL_METRIC_MATH.md) | From programming agent. All items decided → W1–W4 + level. B4 floor homography deferred |
| done | Analyze model picker S / M | [AGENT_HANDOFF_POSE_MODEL_PICKER.md](./AGENT_HANDOFF_POSE_MODEL_PICKER.md) | Qt Model combo; default S; M for hard clips |
| done | Pose normalizer numerically broken | [AGENT_HANDOFF_VIEWPOINT_ROBUSTNESS.md](./AGENT_HANDOFF_VIEWPOINT_ROBUSTNESS.md) | Aspect + torso scale; Reach max 2.2 on fixture; 4 motif types |
| done | Motif / find-similar thresholds | [AGENT_HANDOFF_MOTIF_THRESHOLDS.md](./AGENT_HANDOFF_MOTIF_THRESHOLDS.md) | Re-tuned after normalizer: 0.40 / 0.50 / 0.45 |
| done | Retire web UI, Qt is the product | [AGENT_HANDOFF_RETIRE_WEB_UI.md](./AGENT_HANDOFF_RETIRE_WEB_UI.md) | Default `python -m desktop` → Qt; legacy `--web` |
| ~~cancelled~~ | ~~Web skeleton/timeline/chart polish~~ | [AGENT_HANDOFF_WEB_TIMELINE_POLISH.md](./AGENT_HANDOFF_WEB_TIMELINE_POLISH.md) | **Superseded — do not implement.** Qt replaces web |
| done | Space position metrics | [AGENT_HANDOFF_SPACE_POSITION.md](./AGENT_HANDOFF_SPACE_POSITION.md) | pos_x/y, stacked charts, floor map |
| done | Fluent review UI | [AGENT_HANDOFF_FLUENT_UI.md](./AGENT_HANDOFF_FLUENT_UI.md) | Outcome: Qt won; web retired |
| done | Playback smoothness | [AGENT_HANDOFF_PLAYBACK.md](./AGENT_HANDOFF_PLAYBACK.md) | Decode off UI thread, clock sync — user confirms smooth |
| partial | Motif / find-similar accuracy | [AGENT_HANDOFF_MOTIF_ACCURACY.md](./AGENT_HANDOFF_MOTIF_ACCURACY.md) | Mirror + DTW shipped and verified working; thresholds broken → see P0 above |
| done | English UI | [AGENT_HANDOFF_UI_ENGLISH_AND_TIMELINE.md](./AGENT_HANDOFF_UI_ENGLISH_AND_TIMELINE.md) | EN strings shipped |

Legacy: [AGENT_HANDOFF.md](./AGENT_HANDOFF.md) (Python desktop pivot — superseded).

**Quick reference:** [CHEATSHEET_METRICS_MOTIFS.md](./CHEATSHEET_METRICS_MOTIFS.md) (EN) · [CHEATSHEET_METRICS_MOTIFS_YUE.md](./CHEATSHEET_METRICS_MOTIFS_YUE.md) (粵)

**Analysis agent:** [ANALYSIS_AGENT_GUIDE.md](./ANALYSIS_AGENT_GUIDE.md) · rule `.cursor/rules/analysis-agent.mdc` · outputs in `docs/analysis/`

**Status values:** `pending` · `in progress` · `done` · `partial` · `cancelled` · `answered`

## Deferred (design decided, not scheduled)

- **Floor homography** (proposal B4) — the floor map stays an orientation aid, not a measuring
  instrument. It needs per-clip corner-clicking from the user and nobody has yet asked for distances
  in metres. Revisit when someone does.
- **Raising `SEQ_FRAMES`** — only after W1 frees the compute budget, and as its own decision.
- **Cross-dancer unison / canon detection** — genuinely interesting for choreography; out of scope
  until the single-dancer matching is trustworthy.
- **Knee angle** as a metric separate from `level`.

## Regression fixtures

- **Positive:** `07 走路+武拜廟.dance.json` (user-supplied, 16.2 s, 1 dancer, repeated move at
  ~8 s / ~10 s / ~12 s, includes real turning and near-degenerate shoulder frames).
  Paths: `C:\Users\USER\Downloads\07 走路+武拜廟.dance.json` ·
  `backend/uploads/review_project_07 走路+武拜廟.dance.json`.
- **Negative:** `hi.dance.json` from `C:\Users\USER\Downloads\hi.mp4` — **7.60 s**, **1 dancer**,
  1920×1080 @ 25 fps. Spot-checked: continuous stage walk / prep, **no clear repeated phrase**.
  Paths: `C:\Users\USER\Downloads\hi.dance.json` · `backend/uploads/hi.dance.json`.
  Note: current gates still emit 1 type (similar walk segments); fixing that is W3's job.

## Standing lessons

1. Do not calibrate thresholds against pixel-identical synthetic data. A perfect synthetic match
   scores 1.0 and hides thresholds that no real footage can reach — this is how the motif
   detector shipped returning zero results.
2. Never divide by a raw per-frame measurement without a floor. Shoulder width reaches 0.002 on
   real footage and inflated features by 85×, which silently destroyed all matching.
3. Check that locked chart ranges actually contain the data. Reach was locked to 1.0–3.5 while
   100% of real values exceeded it, so the panel looked plausible and was meaningless.
4. Batch every change that moves the feature or segmentation scale, then calibrate **once**.
   Thresholds have already been retuned twice because fixes landed one at a time, each invalidating
   the previous calibration.
5. Never write a missing measurement as a valid value. `[0.0, 0.0]` for an occluded joint is not
   "unknown", it is "joint at the torso centre", and every downstream consumer believed it.
