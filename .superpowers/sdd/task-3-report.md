# Task 3 Report: Durable sidecar beside `.dance.json`

## Status

**DONE**

## Commits

- `c0a0002` — Persist GVHMR SMPL sidecar next to the project JSON.

## Test summary

`backend/.venv/Scripts/python.exe -m unittest tests.test_gvhmr_mesh_phase1 tests.test_gvhmr_slice1 -v` — 23 tests, OK.

## Implementation notes

- **`persist_sidecar_with_project`**: Copies source NPZ beside the project JSON as `{stem}.smpl.npz` (e.g. `proj.dance.json` → `proj.dance.smpl.npz`); optional sha256 verify after copy.
- **`run_gvhmr_export`**: Optional `work_dir` writes `gvhmr.json` + sidecar in place; without `work_dir`, sidecar is copied to a persistent temp file before temp dir cleanup. Returns `(tracks, ignored, sidecar_path, sha)`.
- **`analyze_video`**: Uses `artifacts_dir/job_id` as `work_dir`; sets `AnalysisResult.smplSidecar` / `smplSidecarSha256`.
- **`save_project`**: Optional `smpl_sidecar_src`; renames/copies beside saved JSON and updates analysis fields.
- **Desktop**: Analyze worker emits artifact sidecar path; Save project passes it into `save_project`.

## Concerns

- No sidecar when analyze runs without `artifacts_dir`/`job_id` (tests mock export with `None` sidecar — OK).
- Temp-dir export leaves sidecar in `mkstemp` path until save or process exit; desktop always uses artifacts.
- Re-save without `smpl_sidecar_src` does not recopy if sidecar already beside JSON (fields already set).

## Review fix (open project sidecar source)

**Status:** DONE

- **`resolve_smpl_sidecar_beside_project`** (`project_io.py`): Pure helper — `Path(project_json).parent / smplSidecar` when file exists, else `None`.
- **Desktop `open_project`**: Sets `_smpl_sidecar_src` after load so Save As / re-save can pass a valid source into `persist_sidecar_with_project`.
- **Desktop `save_project_dialog`**: Uses same helper (DRY with open).
- **`analyze_video`**: Comment only — without `artifacts_dir`/`job_id`, sidecar remains on `mkstemp` until first `save_project`; desktop always uses artifacts.

### Tests

`backend/.venv/Scripts/python.exe -m unittest tests.test_gvhmr_mesh_phase1 tests.test_gvhmr_slice1 -v` — **25 tests, OK** (added `TestResolveSidecarBesideProject`).
