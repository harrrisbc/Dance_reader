# GVHMR mesh overlay + 3D viewer (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After a GVHMR Analyze, the desktop can bake an SMPL mesh cache on first play, composite a mesh on the video, and show an orbitable world-space 3D viewer synced to the playhead—without changing metric formulas yet.

**Architecture:** Analyze still subprocesses GVHMR. Export additionally writes a params sidecar (`*.dance.smpl.npz`) next to a durable adapter JSON; the app copies that sidecar beside the saved `.dance.json`. First play subprocesses a bake script in the GVHMR env to write `*.dance.mesh.npz` (+ optional soft overlay). The app venv only reads numpy caches and composites/views them—no torch/hmr4d.

**Tech Stack:** Python 3, Pydantic, NumPy, PySide6, OpenCV, unittest; bake uses GVHMR venv (`torch`, `hmr4d`, `pytorch3d` renderer when available). Optional app dep: `pyvista` + `pyvistaqt` for the 3D pane.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-10-14-gvhmr-mesh-viewer-metrics-design.md` Phase 1 only.
- Do **not** implement Phase 2 (3D metrics) or Phase 3 (Live Link / FBX) in this plan.
- Do not commit GVHMR or SMPL weights. Do not import `hmr4d` / torch in the app venv.
- Keep wire skeleton toggles; add **independent** Mesh overlay + 3D viewer toggles.
- Bake failure must not kill wire playback. Status must name the failure.
- Prefer soft-rendered translucent overlay when `SMPL_NEUTRAL.pkl` exists; else projected mesh-edge fallback from verts.
- Reuse the `torch.load` `weights_only=False` patch from `gvhmr_export.py` inside the bake script.
- Tests from `backend/` with `.\.venv\Scripts\python.exe -m unittest …`. CI needs no GPU.
- PowerShell: use `;` not `&&`. Git author for commits (if user asked to commit): env `GIT_AUTHOR_NAME`/`COMMITTER` = `H.arimadion`, email `harrrisbc@users.noreply.github.com`. Do not `git config`. Do not push unless asked.

## File map

| File | Responsibility |
|------|----------------|
| `backend/app/schemas.py` | `smplSidecar`, `smplSidecarSha256`, `meshCache`, `meshCacheSha256` on `AnalysisResult` |
| `backend/app/pipeline/mesh_cache.py` | Pure app-side load/validate of `.dance.smpl.npz` / `.dance.mesh.npz` (numpy only) |
| `backend/tools/gvhmr_export.py` | Also write params sidecar + JSON pointers |
| `backend/app/pipeline/gvhmr_run.py` | Durable export dir; return sidecar path; later invoke bake |
| `backend/app/pipeline/run.py` / `project_io.py` | Persist sidecar beside `.dance.json` |
| `backend/tools/gvhmr_mesh_bake.py` | GVHMR-venv bake: verts + faces + optional RGBA overlay |
| `desktop/mesh_view.py` | PyVista/Qt 3D widget driven by playhead |
| `desktop/playback.py` | Mesh overlay composite + toggles |
| `desktop/window.py` | Split layout, checkboxes, bake-on-open trigger, status |
| `backend/tests/test_gvhmr_mesh_phase1.py` | Schema + cache IO tests |

**Phase 2** (metrics from `joints3d`) is a **separate plan** after this one lands: `docs/superpowers/plans/2026-10-14-gvhmr-3d-metrics-phase2.md` (create when starting Phase 2).

---

### Task 1: AnalysisResult sidecar fields + mesh_cache IO

**Files:**
- Modify: `backend/app/schemas.py`
- Create: `backend/app/pipeline/mesh_cache.py`
- Test: `backend/tests/test_gvhmr_mesh_phase1.py`

**Interfaces:**
- Produces:
  - `AnalysisResult.smplSidecar: str | None = None` (relative filename, e.g. `project.dance.smpl.npz`)
  - `AnalysisResult.smplSidecarSha256: str | None = None`
  - `AnalysisResult.meshCache: str | None = None`
  - `AnalysisResult.meshCacheSha256: str | None = None`
  - `load_smpl_sidecar(path: Path) -> dict[str, Any]` raises `ValueError` if required keys missing
  - `load_mesh_cache(path: Path) -> dict[str, Any]` raises `ValueError` if required keys missing
  - Required smpl keys: `track_count`, `smpl_params_global`, `smpl_params_incam`, `K_fullimg` (each track indexed `0..N-1` or stacked—pick one layout and stick to it; recommended: `n_tracks: int` and per-track groups `t{i}_*` OR object arrays—**use flat keys** `params_global_{i}`, `params_incam_{i}`, `K_{i}`, `n_frames_{i}` with numpy arrays)
  - Required mesh keys: `faces` `(F,3) int`, `n_tracks`, per track `verts_world_{i}` `(T,V,3)`, `verts_incam_{i}` `(T,V,3)`, `times_{i}` `(T,)`, `mask_{i}` `(T,) bool`

- [ ] **Step 1: Write the failing test**

```python
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.pipeline.mesh_cache import load_mesh_cache, load_smpl_sidecar
from app.schemas import AnalysisResult


class TestMeshCacheSchema(unittest.TestCase):
    def test_analysis_result_defaults_none(self) -> None:
        r = AnalysisResult(duration=1.0, fps=30.0)
        self.assertIsNone(r.smplSidecar)
        self.assertIsNone(r.meshCache)

    def test_load_smpl_sidecar_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.dance.smpl.npz"
            np.savez_compressed(
                p,
                n_tracks=np.int32(1),
                n_frames_0=np.int32(2),
                params_global_0=np.zeros((2, 3), np.float32),  # placeholder shape OK for IO test
                params_incam_0=np.zeros((2, 3), np.float32),
                K_0=np.eye(3, dtype=np.float32).reshape(1, 3, 3).repeat(2, axis=0),
            )
            data = load_smpl_sidecar(p)
            self.assertEqual(int(data["n_tracks"]), 1)

    def test_load_mesh_cache_requires_faces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.dance.mesh.npz"
            np.savez_compressed(p, n_tracks=np.int32(0))
            with self.assertRaises(ValueError):
                load_mesh_cache(p)
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
cd C:\dev\Dance_reader\Dance_reader\backend
.\.venv\Scripts\python.exe -m unittest tests.test_gvhmr_mesh_phase1.TestMeshCacheSchema -v
```

Expected: FAIL (`mesh_cache` missing and/or new fields missing).

- [ ] **Step 3: Minimal implementation**

Add the four optional fields on `AnalysisResult`. Implement `load_smpl_sidecar` / `load_mesh_cache` checking the required keys listed above (for the smpl IO test, accept the placeholder arrays; document real param dict layout in Task 2).

For real params, Task 2 will store each SMPL param tensor as separate arrays, e.g. `global_body_pose_0`, `global_betas_0`, … **OR** one pickled dict under a single `track_0` object—**prefer separate float32 arrays** listed in a `PARAM_KEYS` constant shared with export/bake so nothing is pickled.

Define in `mesh_cache.py`:

```python
SMPL_PARAM_KEYS = (
    "body_pose",
    "betas",
    "global_orient",
    "transl",
)  # extend to match what gvhmr_export actually saves from pred["smpl_params_*"]
```

Inspect `pred["smpl_params_global"]` keys during Task 2 and lock `SMPL_PARAM_KEYS` to exactly those keys (typically also hand/expression fields GVHMR sets—save whatever the predict dict contains).

- [ ] **Step 4: Re-run tests — PASS**

- [ ] **Step 5: Commit** (only if user asked to commit in this session)

```powershell
git add backend/app/schemas.py backend/app/pipeline/mesh_cache.py backend/tests/test_gvhmr_mesh_phase1.py
git commit -m "Add mesh/smpl sidecar schema fields and numpy cache loaders."
```

---

### Task 2: Export writes params sidecar

**Files:**
- Modify: `backend/tools/gvhmr_export.py`
- Modify: `backend/app/pipeline/mesh_cache.py` (`SMPL_PARAM_KEYS` to match reality)
- Test: extend `backend/tests/test_gvhmr_mesh_phase1.py` with a **unit test of a pure helper** that builds the npz from fake tensors (no GPU)—extract `write_smpl_sidecar(out_npz: Path, tracks_params: list[dict]) -> Path` into a small module importable without hmr4d, e.g. `backend/app/pipeline/smpl_sidecar_write.py` used by the export script via path insert, **or** keep helper inside export and test by subprocess `--help` only plus a duplicated pure function in `mesh_cache.py`: `save_smpl_sidecar(path, tracks: list[dict[str, dict[str, np.ndarray]]]) -> None`.

**Interfaces:**
- Produces: beside `--out` JSON, file `Path(out).with_suffix("").as_posix()` carefully:
  - If `--out` is `…/gvhmr.json`, write `…/gvhmr.dance.smpl.npz` **or** same stem: prefer `out_path.with_name(out_path.stem + ".dance.smpl.npz")` when stem is `gvhmr`, simpler rule:

```text
json:  <dir>/<name>.json
smpl:  <dir>/<name>.dance.smpl.npz
```

  JSON gains:
  ```json
  "smplSidecar": "<name>.dance.smpl.npz",
  "smplSidecarSha256": "<hex>"
  ```
- Each track’s `smpl_params_global` / `smpl_params_incam` values saved as float32 arrays under `SMPL_PARAM_KEYS` with prefixes `g{i}_` and `c{i}_`, plus `K_{i}` `(T,3,3)`, `n_frames_{i}`.

- [ ] **Step 1: Failing test for `save_smpl_sidecar` + sha helper**

```python
import hashlib
from app.pipeline.mesh_cache import save_smpl_sidecar, sha256_file

class TestSaveSmplSidecar(unittest.TestCase):
    def test_save_and_reload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "clip.dance.smpl.npz"
            tracks = [{
                "global": {"body_pose": np.zeros((2, 63), np.float32), "betas": np.zeros((2, 10), np.float32),
                           "global_orient": np.zeros((2, 3), np.float32), "transl": np.zeros((2, 3), np.float32)},
                "incam": {"body_pose": np.zeros((2, 63), np.float32), "betas": np.zeros((2, 10), np.float32),
                          "global_orient": np.zeros((2, 3), np.float32), "transl": np.zeros((2, 3), np.float32)},
                "K": np.eye(3, dtype=np.float32).reshape(1, 3, 3).repeat(2, 0),
            }]
            save_smpl_sidecar(p, tracks)
            self.assertTrue(p.is_file())
            self.assertEqual(len(sha256_file(p)), 64)
            load_smpl_sidecar(p)
```

Adjust key sets after inspecting one real `pred["smpl_params_global"]` on the user’s machine (run a tiny debug once during implementation).

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Implement `save_smpl_sidecar` / `sha256_file`; call from `write_export` after building tracks, before writing JSON. Keep writing joints/keypoints frames as today.**

Also apply the existing torch.load patch before any checkpoint load (already present).

- [ ] **Step 4: Tests PASS**

- [ ] **Step 5: Commit** (if requested)

```powershell
git commit -m "Write GVHMR SMPL params sidecar beside adapter JSON."
```

---

### Task 3: Durable sidecar beside `.dance.json`

**Files:**
- Modify: `backend/app/pipeline/gvhmr_run.py`
- Modify: `backend/app/pipeline/run.py`
- Modify: `backend/app/project_io.py` (or desktop save path—follow wherever `.dance.json` is written from Analyze in Qt)
- Test: `backend/tests/test_gvhmr_mesh_phase1.py`

**Problem today:** `run_gvhmr_export` uses `tempfile.TemporaryDirectory()`, so the sidecar would be deleted.

**Interfaces:**
- Change `run_gvhmr_export` to:

```python
def run_gvhmr_export(
    video: Path,
    *,
    max_people: int = 2,
    work_dir: Path | None = None,
) -> tuple[list[list[PoseFrame]], int, Path | None, str | None]:
    """Returns tracks, ignored, smpl_sidecar_path | None, sha256 | None."""
```

- If `work_dir` is set, write `work_dir / "gvhmr.json"` and sidecar there; else keep temp behavior but **copy** sidecar out before temp cleanup when present.
- `analyze_video` / desktop Analyze must pass `artifacts_dir/job_id` or a project folder and set `result.smplSidecar` + hash; when user saves `.dance.json`, copy sidecar next to it and keep relative name.

- [ ] **Step 1: Test that a fake export layout is copied**

```python
class TestPersistSidecar(unittest.TestCase):
    def test_copy_sidecar_next_to_dance_json(self) -> None:
        from app.pipeline.mesh_cache import persist_sidecar_with_project
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "work" / "gvhmr.dance.smpl.npz"
            src.parent.mkdir()
            np.savez_compressed(src, n_tracks=np.int32(0))
            dance = Path(tmp) / "proj.dance.json"
            dance.write_text("{}", encoding="utf-8")
            rel = persist_sidecar_with_project(dance, src, sha256_file(src))
            self.assertEqual(rel, "proj.dance.smpl.npz")
            self.assertTrue((dance.parent / rel).is_file())
```

- [ ] **Step 2: FAIL then implement `persist_sidecar_with_project`**

- [ ] **Step 3: Wire `gvhmr_run` + Analyze save path so `AnalysisResult` carries the relative name**

- [ ] **Step 4: unittest PASS**

- [ ] **Step 5: Commit** (if requested)

```powershell
git commit -m "Persist GVHMR SMPL sidecar next to the project JSON."
```

---

### Task 4: Mesh bake script (GVHMR venv)

**Files:**
- Create: `backend/tools/gvhmr_mesh_bake.py`
- Manual check only for full bake; unit-test a pure `overlay_paths` helper if extracted.

**CLI:**

```text
python gvhmr_mesh_bake.py --smpl <sidecar.npz> --out <mesh.npz> [--video <path>] [--overlay <overlay.mp4|dir>]
```

**Behavior:**
1. Patch `torch.load` weights_only like export.
2. Load sidecar; `make_smplx("supermotion")` + `smplx2smpl_sparse.pt` + J regressor as in export `_joints`.
3. For each track, forward global/incam params → verts; write mesh npz via `save_mesh_cache` in `mesh_cache.py` (numpy save must be callable from bake by duplicating save logic in the script **or** adding `backend` to `sys.path`—prefer implementing `save_mesh_cache` in `mesh_cache.py` and running bake with `PYTHONPATH=backend`).
4. If `SMPL_NEUTRAL.pkl` / `make_smplx("smpl")` works and `--video` given, render soft mesh frames (copy patterns from `C:\dev\GVHMR\tools\demo\demo.py` `render_incam`) into `--overlay` mp4; record `overlay` relative name inside a small JSON companion or as mesh npz attr `overlay_relpath` string saved in a sidecars txt—simplest: write `out.with_name(out.stem + ".overlay.mp4")` and put key `overlay_relpath` as npz `allow_pickle` string array length 1.
5. On render failure, still write verts+faces; exit 0 if verts written; print warning to stderr.

- [ ] **Step 1: Implement script + `save_mesh_cache` with a failing unittest on save/load faces**

```python
class TestSaveMeshCache(unittest.TestCase):
    def test_roundtrip_faces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "m.dance.mesh.npz"
            faces = np.array([[0, 1, 2]], np.int32)
            save_mesh_cache(
                p,
                faces=faces,
                tracks=[{
                    "verts_world": np.zeros((2, 3, 3), np.float32),
                    "verts_incam": np.zeros((2, 3, 3), np.float32),
                    "times": np.array([0.0, 0.1], np.float64),
                    "mask": np.array([True, True]),
                }],
            )
            d = load_mesh_cache(p)
            np.testing.assert_array_equal(d["faces"], faces)
```

- [ ] **Step 2–4: TDD for `save_mesh_cache`; implement bake script; smoke on one short clip manually**

Manual (not CI):

```powershell
cd C:\dev\GVHMR
$env:PIP_CONFIG_FILE = "nul"
# PYTHONPATH to Dance Reader backend for mesh_cache
$env:PYTHONPATH = "C:\dev\Dance_reader\Dance_reader\backend"
C:\Users\USER\miniconda3\envs\gvhmr\python.exe C:\dev\Dance_reader\Dance_reader\backend\tools\gvhmr_mesh_bake.py --smpl <sidecar> --out <mesh.npz> --video <mp4>
```

- [ ] **Step 5: Commit** (if requested)

```powershell
git commit -m "Add GVHMR mesh bake tool and mesh cache writer."
```

---

### Task 5: Bake runner from the app

**Files:**
- Modify: `backend/app/pipeline/gvhmr_run.py` (or new `mesh_bake_run.py`)
- Test: unittest with mocked `subprocess.run`

**Interfaces:**

```python
class MeshBakeError(RuntimeError):
    pass

def run_mesh_bake(*, smpl_sidecar: Path, mesh_out: Path, video: Path | None) -> Path:
    """Subprocess DANCE_GVHMR_PYTHON on gvhmr_mesh_bake.py; cwd=DANCE_GVHMR_ROOT."""
```

- Reuse `_require_paths()` (YOLO weight preflight stays).
- On non-zero exit: raise `MeshBakeError` with stderr tail.
- Desktop catches this and sets status; does not clear poses.

- [ ] **Step 1: Failing test with `unittest.mock.patch("subprocess.run")`**

- [ ] **Step 2–4: Implement + PASS**

- [ ] **Step 5: Commit** (if requested)

```powershell
git commit -m "Invoke mesh bake through the GVHMR Python env."
```

---

### Task 6: Playback mesh overlay + toggles

**Files:**
- Modify: `desktop/playback.py`
- Modify: `desktop/window.py` (checkbox only; layout split in Task 8)

**Interfaces:**
- Worker commands: `set_mesh_overlay(enabled: bool)`, `set_mesh_layer(payload)` where payload is either:
  - pre-baked overlay video path, or
  - `{times, verts_incam, faces, K, width, height}` for edge fallback
- Draw order: video frame → mesh (alpha or polylines) → wire skeleton (existing).

Edge fallback: project `verts_incam` with `K` to pixels; draw a **subset** of edges (e.g. sample face edges with stride) in a translucent color so FPS stays acceptable.

- [ ] **Step 1: Add a pure function test in `backend/tests` or `desktop` if tested:**

```python
# backend/tests/test_gvhmr_mesh_phase1.py
from app.pipeline.mesh_project import project_verts_to_pixels

class TestProjectVerts(unittest.TestCase):
    def test_shape(self) -> None:
        V = np.array([[0.0, 0.0, 2.0]], np.float32)
        K = np.array([[1000.0, 0, 500.0], [0, 1000.0, 500.0], [0, 0, 1]], np.float32)
        uv = project_verts_to_pixels(V, K, width=1000, height=1000)
        self.assertEqual(uv.shape, (1, 2))
```

Put `project_verts_to_pixels` in `backend/app/pipeline/mesh_project.py` so CI can test without Qt.

- [ ] **Step 2–4: Implement project helper + hook playback worker**

- [ ] **Step 5: Commit** (if requested)

```powershell
git commit -m "Composite mesh overlay on desktop video playback."
```

---

### Task 7: 3D viewer widget

**Files:**
- Create: `desktop/mesh_view.py`
- Modify: `backend/requirements.txt` — add `pyvista` and `pyvistaqt` if chosen

**Interfaces:**

```python
class MeshWorldView(QWidget):
    def set_mesh(self, verts_world: np.ndarray, faces: np.ndarray) -> None: ...
    def set_time(self, t: float, times: np.ndarray) -> None: ...  # pick nearest frame
    def clear(self) -> None: ...
```

- Orbit enabled; background dark neutral (not purple glow clutter).
- If pyvista import fails, show QLabel with install hint; do not crash Analyze.

- [ ] **Step 1: Smoke import test optional; manual verification required**

- [ ] **Step 2: Implement widget**

- [ ] **Step 3: Commit** (if requested)

```powershell
git commit -m "Add PyVista world mesh viewer widget for desktop."
```

---

### Task 8: Window split, bake-on-first-play, status

**Files:**
- Modify: `desktop/window.py`

**Behavior:**
1. Splitter: video | `MeshWorldView`.
2. Checkboxes: `Mesh overlay`, `3D viewer` (independent of skeleton combo).
3. On project load / after Analyze: if `smplSidecar` present and `meshCache` missing → background bake → set `meshCache` + hash → refresh overlay/viewer.
4. Status: `Building mesh cache…` then clear or show error text.
5. If no sidecar: disable mesh/viewer checkboxes; tooltip “Re-analyze to enable mesh”.

- [ ] **Step 1: Manual checklist (Task acceptance)**

- [ ] Open project with sidecar, first play builds cache.
- [ ] Second open reuses cache (no bake).
- [ ] Wire + mesh both on; either off.
- [ ] Viewer follows playhead.
- [ ] Kill bake env → wire still works + error status.

- [ ] **Step 2: Write `docs/IMPLEMENTATION_NOTES_GVHMR_MESH_VIEWER.md`** (English) summarizing Phase 1.

- [ ] **Step 3: Commit** (if requested)

```powershell
git commit -m "Wire mesh bake, overlay, and 3D viewer into the desktop UI."
```

- [ ] **Step 4: Update handoff index** — mark Phase 1 in-progress/done notes on `AGENT_HANDOFF_GVHMR_MESH_VIEWER_METRICS.md` when finished.

---

## Phase 1 done criteria

Matches handoff Phase 1 acceptance checkboxes. Then create the Phase 2 plan file before touching metrics.

## Self-review (plan vs spec)

| Spec Phase 1 item | Task |
|-------------------|------|
| Params sidecar + hash in JSON | 1–3 |
| First-play bake verts/faces/overlay | 4–5, 8 |
| Video mesh + independent toggles | 6, 8 |
| Side 3D viewer synced | 7–8 |
| App venv no torch | all tasks |
| Bake fail ≠ lose wire | 5, 8 |
| Old projects without sidecar | 8 |
| Live Link / 3D metrics | **out of this plan** |
