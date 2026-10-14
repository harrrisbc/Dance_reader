# Task 4 Report: Mesh bake script (GVHMR venv)

## Status
**DONE** — `save_mesh_cache` / roundtrip test; `backend/tools/gvhmr_mesh_bake.py` CLI complete.

## Changes
| File | Change |
|------|--------|
| `backend/app/pipeline/mesh_cache.py` | Added `save_mesh_cache` (+ optional `overlay_relpath`) |
| `backend/tests/test_gvhmr_mesh_phase1.py` | `TestSaveMeshCache`, `TestOverlayPaths` |
| `backend/tools/gvhmr_mesh_bake.py` | New bake CLI: sidecar → verts, optional incam overlay mp4 |

## TDD evidence

### RED (pre-implementation)
`TestSaveMeshCache` imported missing `save_mesh_cache` → `ImportError`.

### GREEN
```text
> python -m unittest tests.test_gvhmr_mesh_phase1.TestSaveMeshCache tests.test_gvhmr_mesh_phase1.TestOverlayPaths -v
test_roundtrip_faces ... ok
test_default_overlay_beside_mesh ... ok
OK
```

`gvhmr_mesh_bake.py --help` runs without importing torch/hmr4d.

## Bake behavior
- Patches `torch.load` (`weights_only=False`) like `gvhmr_export.py`.
- `make_smplx("supermotion")` + `smplx2smpl_sparse.pt`; per-track `g{i}_*` / `c{i}_*` → `verts_world` / `verts_incam`.
- `times` = frame index / video fps (default 30); `mask` = all True (sidecar has no detection mask).
- Optional overlay: `render_incam`-style path when `--video`, SMPL faces load, and CUDA render succeeds; else stderr warning, exit 0 after mesh write.

## Concerns
- Sidecar-only bake cannot recover sparse detection mask from adapter JSON; playback edge fallback may need mask from a later task.
- Overlay uses track 0 only (matches single-subject demo).
- Manual GVHMR smoke not run in this session.

## Commit
Message: `Add GVHMR mesh bake tool and mesh cache writer.`

---

## Critical fix (2026-10-14)

**Issue:** `_verts` used `np.float32` without importing `numpy` in that function → `NameError` at bake time.

**Fix:**
- `import numpy as np` inside `_verts` (keeps module `--help` free of numpy).
- Overlay writer uses the same `fps` as mesh `times` (video CAP_PROP_FPS or 30), not hardcoded 30.

### Tests (post-fix)
```text
> python -m unittest tests.test_gvhmr_mesh_phase1.TestSaveMeshCache tests.test_gvhmr_mesh_phase1.TestOverlayPaths -v
test_roundtrip_faces ... ok
test_default_overlay_beside_mesh ... ok
OK
```

### Commit (fix)
Message: `Fix mesh bake _verts numpy import and overlay fps.`
Branch: `cursor/gvhmr-mesh-viewer-phase1` (tip commit for this fix).
