# WORK ORDER — GVHMR mesh overlay, 3D viewer, and 3D metrics

**Status:** Phase 1 — implementation complete, **manual acceptance pending** (Phase 2 not started)  
**Spec:** [superpowers/specs/2026-10-14-gvhmr-mesh-viewer-metrics-design.md](./superpowers/specs/2026-10-14-gvhmr-mesh-viewer-metrics-design.md)  
**Phase 1 plan:** [superpowers/plans/2026-10-14-gvhmr-mesh-viewer-phase1.md](./superpowers/plans/2026-10-14-gvhmr-mesh-viewer-phase1.md)  
**Depends on:** GVHMR slice 1 (bones + export) already on local `main`

Implement in **two phases**. Do not start Phase 2 until Phase 1 acceptance passes unless the user says otherwise. Phase 3 is documentation only in the spec — do not implement Live Link / FBX here.

---

## 1. Why

Slice 1 only draws a wire SMPL tree on video. The user wants:

- A rendered **mesh** on the video
- A rotatable **3D viewer** synced to playback
- Analysis charts/motifs from **3D joints** (more accurate than the temporary COCO-17 bridge)

YOLO remains for GVHMR person boxes only; it is not the skeleton source.

## 2. Scope

| Area | Files / modules (expected) |
|------|----------------------------|
| Export | `backend/tools/gvhmr_export.py` — also write params sidecar |
| Bake | New `backend/tools/gvhmr_mesh_bake.py` (name OK to adjust) — GVHMR venv only |
| Runner | `backend/app/pipeline/gvhmr_run.py` / schemas — sidecar paths, bake invoke |
| Desktop | `desktop/playback.py`, `desktop/window.py` — split view, toggles, composite, viewer |
| Metrics | `backend/app/pipeline/metrics.py`, motif path, `smpl_skeleton.py` bridge removal in Phase 2 |
| Tests | `backend/tests/` — schema, metric branch, no-GPU |

App venv: **no** torch / hmr4d for viewing. Mesh bake and SMPL forward stay in `DANCE_GVHMR_PYTHON`.

## 3. Changes

### Phase 1 — See the person

1. Extend GVHMR export to write `smpl_params_global`, `smpl_params_incam`, and `K_fullimg` into a sidecar `*.dance.smpl.npz` (or equivalent); JSON gets `smplSidecar` + hash. Keep existing `joints3d` / `keypoints`.
2. Add on project load / after Analyze bake subprocess: verts_incam, verts_world, faces, times/mask → `*.dance.mesh.npz`; prefer soft-rendered translucent overlay for video composite.
3. Desktop: left video + right 3D viewer; **Mesh overlay** and **3D viewer** checkboxes independent of Skeleton/Joints/Off.
4. Composite baked overlay (or projected-edge fallback) on decoded frames; PyVista/pyvistaqt (or approved Qt GL) for `verts_world`.
5. Status strings: `Building mesh cache…`; clear errors if `SMPL_NEUTRAL.pkl` missing for preferred render path.
6. Old JSON without sidecar: wire only; suggest Re-analyze for mesh/viewer.

### Phase 2 — Measure in 3D

7. For frames with 24 `joints3d`, compute energy/speed/reach/travel/level/directness and motifs from meters; set `metricsSpace: "smpl3d"`.
8. Stop calling `smpl_keypoints_to_coco17` on that path; delete bridge when nothing else needs it.
9. Keep 17-point / no-`joints3d` on `"coco2d"` formulas.
10. Recalibrate motif/metric thresholds on real clips after the feature change (do not copy old gates blindly).

### Phase 3 — Do not implement

11. Spec already notes Live Link + BVH/FBX; no code in this handoff.

## 4. Acceptance

### Phase 1

- [ ] Analyze produces JSON + params sidecar; joints/wire still work.
- [ ] On project load / after Analyze, missing cache triggers bake; reopen reuses cache (no full re-bake).
- [ ] Wire and mesh can both be on; either can be off.
- [ ] 3D viewer orbits and follows the playhead.
- [ ] Bake failure does not kill wire playback; status explains the failure.
- [ ] CI tests for sidecar/mesh schema without GPU.

### Phase 2

- [ ] 24-point GVHMR project uses 3D metrics; units are meters (or m/s) in UI/docs.
- [ ] 17-point legacy project still charts with old 2D path.
- [ ] COCO bridge unused for 24-point metric path.
- [ ] Motifs still return usable types after retune on a real positive/negative pair.
- [ ] Mesh bake failure does not block metrics.

## 5. Out of scope

- Live Link, FBX, BVH, MetaHuman, UE axis conversion (Phase 3 later)
- Vendoring GVHMR / SMPL weights into git
- Replacing Qt with a web 3D stack
- Crowds &gt; 2, moving camera / DPVO
- Metrics from dense mesh vertices
- Changing YOLO’s role beyond current GVHMR box tracking
- Pretending Task 8 (manual slice-1 video checklist) was done

## Notes for programming

- **2026-10-14 (Task 8):** Desktop wiring landed — splitter + `MeshWorldView`, mesh/3D toggles, background bake on load/analyze, status strings, playhead sync. See `docs/IMPLEMENTATION_NOTES_GVHMR_MESH_VIEWER.md`. Run Phase 1 acceptance checklist before Phase 2 metrics work.
- Prefer small PRs / commits: export+sidecar → bake tool → desktop viewer → Phase 2 metrics.
- User still may not have `SMPL_NEUTRAL.pkl`; prefer renderer when present, edge fallback when not.
- Existing torch `weights_only` workaround in `gvhmr_export.py` should be reused in the bake script.
- After Phase 1, write `docs/IMPLEMENTATION_NOTES_GVHMR_MESH_VIEWER.md`; after Phase 2, extend or add metrics notes.
