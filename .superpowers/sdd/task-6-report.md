# Task 6 Report: Playback mesh overlay + toggles

## Status

**DONE**

## Summary

Added `backend/app/pipeline/mesh_project.py` with `project_verts_to_pixels` and `face_edge_pairs`. `VideoDecodeWorker` handles `set_mesh_overlay` / `set_mesh_layer` (baked overlay mp4 path or edge-fallback dict). Compositing runs after downscale, before wire skeleton. Desktop toolbar includes a **Mesh overlay** checkbox wired to the worker.

## TDD Evidence

### RED

`TestProjectVerts` failed before `mesh_project.py` existed (`ModuleNotFoundError`).

### GREEN

```powershell
cd backend
.\.venv\Scripts\python.exe -m unittest tests.test_gvhmr_mesh_phase1 -v
```

14 tests, OK (includes `TestProjectVerts.test_shape`).

## Commits

- `9ea7d2e` — Composite mesh overlay on desktop video playback.

Author/committer: `H.arimadion <harrrisbc@users.noreply.github.com>`.

## Self-review

- Draw order: decoded frame → mesh (alpha-composited baked layer or edge polylines) → skeleton overlay.
- Baked overlay uses a second `VideoCapture`, frame-index aligned with main video; composited via `blend_mesh_overlay`.
- Edge fallback subsamples faces with `MESH_EDGE_STRIDE=8`; translucent BGR lines.
- No torch in app venv; projection tested in CI without Qt.

## Concerns

- `set_mesh_layer` from project/mesh cache is Task 8; checkbox toggles enable only until layer is pushed.

## Fix: alpha-composite baked overlay (spec)

**Problem:** Playback returned the baked overlay mp4 frame as a full replacement of the decoded video.

**Change:**

- `gvhmr_mesh_bake._render_incam_overlay`: render mesh on **black** background (mesh-only mp4).
- `mesh_project.blend_mesh_overlay`: mask non-black overlay pixels and lerp onto the decoded frame (`MESH_EDGE_ALPHA` in playback).
- Edge-fallback path unchanged (`_draw_mesh_edges`).

### Tests (after fix)

```powershell
cd backend
.\.venv\Scripts\python.exe -m unittest tests.test_gvhmr_mesh_phase1 -v
```

16 tests, OK (includes `TestBlendMeshOverlay`).

## Files changed

- `backend/app/pipeline/mesh_project.py` (created)
- `backend/tests/test_gvhmr_mesh_phase1.py`
- `desktop/playback.py`
- `desktop/window.py`
