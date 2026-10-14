# GVHMR mesh overlay and 3D viewer — Phase 1 implementation notes

## What shipped (Phase 1)

- **SMPL params sidecar** (`*.dance.smpl.npz`) written by GVHMR export; referenced from `AnalysisResult.smplSidecar` + hash.
- **Mesh bake** via `run_mesh_bake` / `backend/tools/gvhmr_mesh_bake.py` in the GVHMR venv (`DANCE_GVHMR_PYTHON`, `DANCE_GVHMR_ROOT`).
- **Mesh cache** (`*.dance.mesh.npz`) with `verts_world`, `verts_incam`, `faces`, per-track `times` / `mask`; optional baked overlay mp4 beside the cache.
- **Desktop playback**: `VideoDecodeWorker` composes mesh (overlay video or projected edge fallback) before wire skeleton; **Mesh overlay** checkbox.
- **Desktop layout**: horizontal splitter — video (left) and `MeshWorldView` (right); **3D viewer** checkbox shows/hides the right pane.
- **3D viewer UX**: ground plane/grid under the mesh; camera framed once to the full-sequence AABB (in-place point updates — no per-frame zoom); **Up axis** combo (`Y up (GVHMR)` / `Z up`).
- **Bake-on-demand**: on project load / after Analyze, if a sidecar exists and mesh cache is missing or invalid, a background thread runs the bake, updates `meshCache` / `meshCacheSha256`, persists beside the project when saved, and pushes `set_mesh_layer` + viewer data. Status shows `Building mesh cache…` (not overwritten by “Loaded project” while baking); on success restores the prior Done/Loaded message or `Mesh cache ready`; on failure shows `Mesh bake: …`. Re-analyze or open another project cancels stale bake workers (generation + sidecar guards). Wire/skeleton playback is unchanged on bake failure.
- **Legacy projects** without a sidecar: mesh and 3D viewer checkboxes disabled; tooltip `Re-analyze to enable mesh`.

Phase 2 (3D metrics, `metricsSpace: "smpl3d"`) is **not** implemented here.

## Key files

| Area | Path |
|------|------|
| Schemas | `backend/app/schemas.py` |
| Sidecar / cache I/O | `backend/app/pipeline/mesh_cache.py` |
| Bake runner | `backend/app/pipeline/gvhmr_run.py` |
| Projection helpers | `backend/app/pipeline/mesh_project.py` |
| Project save/load | `backend/app/project_io.py` |
| Playback compositing | `desktop/playback.py` |
| Main window wiring | `desktop/window.py` |
| 3D widget | `desktop/mesh_view.py` |
| CI tests | `backend/tests/test_gvhmr_mesh_phase1.py` |

## How to test manually

1. Configure `DANCE_GVHMR_PYTHON` and `DANCE_GVHMR_ROOT`; run **Analyze** on a clip with GVHMR enabled.
2. Confirm status `Building mesh cache…` then completion; enable **Mesh overlay** and/or **3D viewer** independently of skeleton style.
3. **Save project**, close, **Open project** — mesh cache should load without re-bake (watch status; no long bake wait).
4. Scrub/play — 3D mesh frame should track the playhead.
5. Break bake env (wrong Python) — wire skeleton still draws; status shows bake error.
6. Open an old `.dance.json` without sidecar — mesh toggles disabled with re-analyze tooltip.

## Dependencies

- App venv: `pyvista`, `pyvistaqt` (optional; viewer shows install hint if missing).
- Mesh forward / render: GVHMR venv only (no torch in app venv for bake).
