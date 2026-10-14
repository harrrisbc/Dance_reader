# GVHMR Phase 2 — 3D metrics, dual floor map, level redefine

Date: 2026-10-15  
Parent: [2026-10-14-gvhmr-mesh-viewer-metrics-design.md](./2026-10-14-gvhmr-mesh-viewer-metrics-design.md) (Phase 1 shipped on `main`)  
Level decision: [AGENT_HANDOFF_LEVEL_DEFINITION.md](../../AGENT_HANDOFF_LEVEL_DEFINITION.md) (folded into this phase)

## Goal

When a track has SMPL-24 `joints3d` (meters, `coordFrame: "gvhmr_world"`):

1. Compute **all** chart metrics and motif features from **3D joints** (not the COCO-17 bridge).
2. Show a **world-ground floor map** with pelvis + supporting-foot trajectories.
3. Set `metricsSpace: "smpl3d"`.
4. Redefine `level` (both 2D and 3D paths) as hip → supporting-ankle vertical drop / torso height.
5. Retune motif / metric gates on the real positive/negative fixtures before marking done.

YOLO-only / 17-point projects stay on `metricsSpace: "coco2d"` (new level formula only).

## Decisions locked with the user (2026-10-15)

| Topic | Choice |
|-------|--------|
| Implementation shape | **A** — branch inside `metrics.py` (and motif feature builders); no separate metrics3d package |
| Level | New definition on **both** `coco2d` and `smpl3d`; fold pending level handoff into Phase 2 |
| Done bar | Formulas + UI + **fixture retune** (武拜廟 positive + `hi` negative) in one phase |
| Floor map (`smpl3d`) | **Both** tracks: pelvis XZ solid + supporting-foot XZ dashed; axis in **metres (world)** |
| All metrics in 3D | Yes for `smpl3d` (speed, energy, reach, travel, level, directness, floor). Not inventable for YOLO-only |
| Mesh bake | Failure must **not** block metrics (joints already in JSON) |
| Live Link / FBX | Still Phase 3 — out of scope |

## Coordinate / accuracy caveats (document in UI tooltip)

- Export still uses `static_cam=True` (no DPVO). World XZ is gravity-aligned relative to the clip’s GV frame, **not** room GPS or surveyed stage metres.
- Moving camera → world trajectory less trustworthy.
- Multi-person: each track’s world frame is recovered independently; do not assume a shared absolute origin across dancers without a later design.

---

## Metric formulas

Shared: torso scale `S` = shoulder-mid → hip-mid length, with track-level floor. Missing measurements carry last good value. Energy charts remain **relative to that dancer’s peak** (0–1).

### Level (both paths) — replaces p90 leg-length

```
level = vertical_drop(hip → supporting_ankle) / S
```

| Path | Vertical drop |
|------|----------------|
| `coco2d` | Square-pixel space; image Y down → use absolute vertical separation of supporting (lower) ankle vs hip mid, ÷ `S` |
| `smpl3d` | Y-up metres → `(hip_y − supporting_ankle_y) / S` |

- Supporting ankle = lower of the two ankles on the up/vertical axis.
- A lifted free leg must not raise `level`.
- Low / middle / high bands and chart Y limits: **measure on fixtures**, then lock (do not copy 0.45 / 0.8 blindly).

### `smpl3d` (24 × `joints3d`)

| Field | Formula | Chart unit |
|-------|---------|------------|
| Speed | Mean limb-joint `‖v‖ / S` | torso heights / s |
| Energy | `Σ (v/S)²` then ÷ peak | rel. peak |
| Reach | Max 3D distance of head / wrists / ankles from torso centre ÷ `S` | × torso |
| Travel | Pelvis (joint 0) path length on **ground XZ** ÷ `S` ÷ window Δt | torso heights / s |
| Directness | `net / path` on that XZ window; NaN if nearly still | 0–1 |
| `pos_x`, `pos_y` | Pelvis world **X** and **Z** (metres) | m (world) |
| `pos_foot_x`, `pos_foot_y` | Supporting ankle world **X** and **Z** (metres) | m (world) |

Motif pose features: 3D joints relative to torso / `S`, plus unit bone vectors (mirror L↔R). **No** `smpl_keypoints_to_coco17` on this path.

### `coco2d` (17-point / no `joints3d`)

Keep existing speed, energy, reach, travel, directness, screen `pos_*` formulas. **Only level** switches to the new definition. Floor map stays single screen trajectory (foot / COM as today). Fill `pos_foot_*` with the screen supporting-foot coords (or same as today’s foot channel) so the schema is uniform.

---

## Schema

| Field | Change |
|-------|--------|
| `AnalysisResult.metricsSpace` | `"smpl3d"` \| `"coco2d"`. If mixed actors in one result, prefer `"smpl3d"` when any actor used 3D; document in notes |
| `MetricPoint.pos_x` / `pos_y` | **Semantic change when `metricsSpace=="smpl3d"`**: pelvis world XZ (m). Old JSON values are discarded on `enrich_metrics` |
| `MetricPoint.pos_foot_x` / `pos_foot_y` | New; supporting-foot floor coords (world m or screen 0–1) |

Pydantic defaults must keep old files loadable.

---

## Code path

1. `enrich_metrics` / GVHMR `run.py`: if poses are 24-point with `joints3d` → 3D compute + motif buffers from 3D; else 2D path + new level.
2. Delete `smpl_keypoints_to_coco17` when nothing references it.
3. Motif DTW gates / motion gates: retune on fixtures; record new constants and a short calibration note in implementation notes.
4. Desktop floor plot: if `metricsSpace=="smpl3d"`, draw pelvis solid + foot dashed, metres axes, tooltip about static-cam world frame; else legacy screen map.
5. Chart labels / ylim for level and pos series follow the new semantics after fixture measurement.
6. Optional status / label showing `metricsSpace`.

### Expected touch list

`backend/app/pipeline/metrics.py`, `run.py`, `motifs.py`, `smpl_skeleton.py`, `backend/app/schemas.py`, `desktop/window.py`, `backend/tests/`, cheatsheets, handoff / implementation notes.

---

## Testing

**CI (no GPU):**

- Fixture with `joints3d` → `metricsSpace=="smpl3d"`; bridge unused.
- 17-point fixture → `coco2d`; new level formula (not p90 leg).
- Schema accepts `pos_foot_*` and `metricsSpace`.
- Turning-in-place synthetic 3D: level flat; crouch: level drops; lifted free leg: level does not rise.

**Manual:**

- Positive: `07 走路+武拜廟` — usable motif types after retune.
- Negative: `hi.dance.json` — no false mega-type if gates honest.
- Floor map: pelvis solid + foot dashed move in world metres on a GVHMR clip.
- Mesh bake failure: charts still populate from `joints3d`.

---

## Out of scope

- Live Link, BVH/FBX, UE axis conversion (Phase 3)
- Metrics from dense mesh vertices
- Moving-camera / DPVO world recovery
- Floor homography / surveyed stage metres
- Knee angle as a separate metric
- Changing YOLO’s role inside GVHMR preprocess

---

## Supersedes

- Phase 2 summary bullets in the 2026-10-14 parent design (this doc is authoritative for Phase 2 detail).
- Pending scheduling of [AGENT_HANDOFF_LEVEL_DEFINITION.md](../../AGENT_HANDOFF_LEVEL_DEFINITION.md) — semantics absorbed here; mark that handoff done when Phase 2 lands.
