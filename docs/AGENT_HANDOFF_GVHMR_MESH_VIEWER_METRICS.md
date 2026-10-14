# WORK ORDER — GVHMR mesh overlay, 3D viewer, and 3D metrics

**Status:** Phase 1+2 implemented — see [IMPLEMENTATION_NOTES_GVHMR_3D_METRICS.md](./IMPLEMENTATION_NOTES_GVHMR_3D_METRICS.md)  
 
**Phase 1 spec:** [superpowers/specs/2026-10-14-gvhmr-mesh-viewer-metrics-design.md](./superpowers/specs/2026-10-14-gvhmr-mesh-viewer-metrics-design.md)  
**Phase 2 spec:** [superpowers/specs/2026-10-15-gvhmr-3d-metrics-phase2-design.md](./superpowers/specs/2026-10-15-gvhmr-3d-metrics-phase2-design.md)  
**Phase 1 plan:** [superpowers/plans/2026-10-14-gvhmr-mesh-viewer-phase1.md](./superpowers/plans/2026-10-14-gvhmr-mesh-viewer-phase1.md)  
**Phase 2 plan:** [superpowers/plans/2026-10-15-gvhmr-3d-metrics-phase2.md](./superpowers/plans/2026-10-15-gvhmr-3d-metrics-phase2.md)  
**Depends on:** Phase 1 mesh/viewer on `main`; user authorized starting Phase 2 without blocking on remaining Phase 1 manual checkboxes

Implement Phase 2 next. Phase 3 (Live Link / FBX) stays documentation only.

---

## 1. Why

Slice 1 + Phase 1 give wire + mesh + 3D viewer. Charts/motifs still go through a temporary COCO-17 bridge from projected joints. The user wants meter-accurate 3D metrics, a world floor map (pelvis + foot), and the redesigned `level` metric.

## 2. Scope (Phase 2)

| Area | Files / modules (expected) |
|------|----------------------------|
| Schema | `backend/app/schemas.py` — `metricsSpace`, `pos_foot_x` / `pos_foot_y` |
| Metrics | `backend/app/pipeline/metrics.py` — 2D/3D branch; new level both paths |
| Analyze | `backend/app/pipeline/run.py` — stop COCO bridge for GVHMR 24-point |
| Motifs | `backend/app/pipeline/motifs.py` — 3D features; retune gates |
| Bridge | `backend/app/pipeline/smpl_skeleton.py` — delete `smpl_keypoints_to_coco17` when unused |
| Desktop | `desktop/window.py` — dual floor map, chart labels/ylim, optional `metricsSpace` label |
| Docs | Cheatsheets, `IMPLEMENTATION_NOTES` (metrics Phase 2), level handoff → done |
| Tests | `backend/tests/` — branch, level, schema; no GPU |

## 3. Changes (Phase 2)

1. `metricsSpace: "smpl3d"` when poses have 24×`joints3d`; else `"coco2d"`.
2. All `smpl3d` charts from world joints (speed/energy/reach/travel/level/directness).
3. Floor: pelvis XZ solid + supporting-foot XZ dashed; metres; static-cam tooltip.
4. New `level` = hip → supporting ankle vertical drop / torso — **both** spaces.
5. Remove COCO-17 bridge from 24-point metric/motif path; delete helper when unused.
6. Retune motif/metric gates on 武拜廟 (positive) + `hi` (negative); record constants.
7. Mesh bake failure must not block metrics.
8. Mark [AGENT_HANDOFF_LEVEL_DEFINITION.md](./AGENT_HANDOFF_LEVEL_DEFINITION.md) done when shipped.

## 4. Acceptance

### Phase 1 (historical — code on `main`)

- [x] Analyze produces JSON + params sidecar; joints/wire still work.
- [x] Bake on load/analyze; reopen reuses cache.
- [x] Wire and mesh independent toggles.
- [x] 3D viewer orbits and follows playhead (floor / up-axis UX follow-up landed).
- [x] Bake failure does not kill wire; status explains failure.
- [x] CI tests for sidecar/mesh schema without GPU.
- [ ] Informal manual GPU checklist (optional; not a Phase 2 blocker).

### Phase 2

- [x] GVHMR project: all metrics + dual floor from 3D; `metricsSpace: "smpl3d"`.
- [x] YOLO/17-point: `coco2d` charts; new level definition.
- [x] COCO bridge unused when `joints3d` present (kept only for legacy 24 w/o 3D).
- [x] Motifs usable on fixtures; gates unchanged (武拜廟 2 types / `hi` 0 types).
- [x] Mesh bake failure does not block metrics.
- [x] Level: supporting ankle; unit tests for turn / lifted leg.
- [x] Cheatsheets + implementation notes updated.

## 5. Out of scope

- Live Link, FBX, BVH, MetaHuman, UE axis conversion (Phase 3)
- Vendoring GVHMR / SMPL weights
- Dense mesh-vertex metrics
- Crowds > 2, DPVO / moving camera
- Floor homography / surveyed stage metres
- Knee angle as a separate metric

## Notes for programming

- Spec is authoritative: `docs/superpowers/specs/2026-10-15-gvhmr-3d-metrics-phase2-design.md`.
- Do not copy old DTW gates; calibrate once after feature change (standing lesson 4).
- `pos_x`/`pos_y` semantic change on `smpl3d` — always recompute via `enrich_metrics` on load.
- Prefer small commits: schema → level both paths → smpl3d metrics → motifs/retune → desktop floor/charts → docs.
