# Task 8 Report: Window split, bake-on-first-play, status

## Status
**DONE** (code + docs); Phase 1 manual acceptance **left for user**.

## Changes
| File | Change |
|------|--------|
| `desktop/window.py` | QSplitter video \| `MeshWorldView`; **3D viewer** checkbox; `MeshBakeWorker`; bake on load/analyze; status strings; playhead → `set_time`; mesh controls when no sidecar |
| `backend/app/pipeline/mesh_cache.py` | `persist_mesh_cache_with_project`, `mesh_cache_rel_name` |
| `backend/app/project_io.py` | `resolve_mesh_cache_beside_project`; `save_project(..., mesh_cache_src=)` |
| `docs/IMPLEMENTATION_NOTES_GVHMR_MESH_VIEWER.md` | Phase 1 summary (English) |
| `docs/AGENT_HANDOFF_GVHMR_MESH_VIEWER_METRICS.md` | Phase 1 progress note |

## Tests (automated)
```text
python -m unittest tests.test_gvhmr_mesh_phase1 -v  → 16 OK
from desktop.window import MainWindow  → ok
```

## Manual checklist
| Item | Verified |
|------|----------|
| Sidecar project, first session builds cache | **User** (needs GVHMR env + clip) |
| Second open reuses cache | **User** |
| Wire + mesh independent toggles | **User** |
| Viewer follows playhead | **User** |
| Bake failure, wire still works | **User** |

## Concerns
- Bake starts on **load/analyze** when cache missing (not deferred to first Play only); matches task brief wire list.
- Multi-dancer: overlay/viewer use selected dancer’s track index; overlay mp4 from bake is track 0 only.
- `save_project` now copies mesh cache when `mesh_cache_src` is passed from the save dialog.

## Commit
Message: `Wire mesh bake, overlay, and 3D viewer into the desktop UI.`

---

## Review fix (Task 8 Important)

**Status:** DONE

### Changes
- `desktop/window.py`: Open project no longer overwrites `Building mesh cache…` with Loaded message; `_status_after_bake` restores Loaded on bake OK. `_cancel_mesh_bake_worker` + `_mesh_bake_gen` / active sidecar checks ignore stale worker signals; cancel on re-analyze, open project, clear video.
- `docs/AGENT_HANDOFF_GVHMR_MESH_VIEWER_METRICS.md`, `docs/IMPLEMENTATION_NOTES_GVHMR_MESH_VIEWER.md`: acceptance wording → on project load / after Analyze (not first play).

### Tests
```text
backend: python -m unittest tests.test_gvhmr_mesh_phase1 -v  → 16 OK
PYTHONPATH=backend: from desktop.window import MainWindow  → ok
```

### Commit
Message: `Fix mesh bake status and stale worker on re-analyze.`

---

## Final review fix (mesh cache naming + overlay bake)

**Status:** DONE

### Changes
- `backend/app/pipeline/mesh_cache.py`: `mesh_cache_rel_name` → `{stem}.mesh.npz` (not double `.dance`); shared `mesh_cache_path_beside_sidecar`.
- `desktop/window.py`: use shared sidecar→mesh path; overlay toggle calls `_ensure_mesh_cache` when cache missing/invalid (same as 3D viewer).
- `backend/tests/test_gvhmr_mesh_phase1.py`: `TestMeshCacheRelName`.

### Tests
```text
backend: python -m unittest tests.test_gvhmr_mesh_phase1 -v  → 17 run, 15 OK, 2 ERROR (scipy missing in env; ResolveSidecarBesideProject imports)
```
