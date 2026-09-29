# GVHMR 3D bones (slice 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Analyze a recorded video with GVHMR (up to two people, static camera) and store SMPL-24 joints so playback draws those bones, without falling back to YOLO Pose.

**Architecture:** A separate GVHMR venv writes a JSON file. The app parses it, stores 24-joint poses, and feeds existing metrics through an in-memory COCO-17 map. Overlay picks COCO edges or the SMPL tree from the keypoint count.

**Tech Stack:** Python 3, Pydantic, PySide6, OpenCV, unittest. GVHMR (`hmr4d`) runs only in the other venv.

## Global Constraints

- Personal/research SMPL use. Do not commit GVHMR or SMPL weights.
- Static camera only (`static_cam=True`). No Live Link, FBX, BVH, MetaHuman, or UE axis conversion.
- Do not recompute metric formulas from 3D. Do not fall back to `yolo11*-pose` or `track_video`.
- At most 2 tracks, ranked by frame count, then earlier first sample.
- A present sample is always 24 joints. A missing moment is an omitted sample, never zeros.
- Stored `keypoints` stay SMPL-24 projections. COCO-17 exists only inside the metric bridge.
- Adapter machine-readable output is `--out` JSON. Logs go to stderr.
- Progress text while the subprocess runs: `Running GVHMR…`.
- Tests run from `backend/` with `.\.venv\Scripts\python.exe -m unittest …`. CI must not need a GPU.

---

### Task 1: SMPL skeleton edges

**Files:**
- Create: `backend/app/pipeline/smpl_skeleton.py`
- Test: `backend/tests/test_gvhmr_slice1.py`

**Interfaces:**
- Consumes: nothing
- Produces: `SMPL_PARENTS: tuple[int, ...]` length 24; `skeleton_edges(n_keypoints: int) -> list[tuple[int, int]]`

- [ ] **Step 1: Write the failing test**

```python
import unittest

from app.pipeline.pose_track import COCO_EDGES
from app.pipeline.smpl_skeleton import SMPL_PARENTS, skeleton_edges


class TestSkeletonEdges(unittest.TestCase):
    def test_coco_17_uses_coco_edges(self) -> None:
        self.assertEqual(skeleton_edges(17), list(COCO_EDGES))

    def test_smpl_24_uses_parent_tree(self) -> None:
        edges = skeleton_edges(24)
        self.assertEqual(len(SMPL_PARENTS), 24)
        self.assertEqual(SMPL_PARENTS[0], -1)
        self.assertIn((1, 0), edges)  # left hip → pelvis
        self.assertNotIn((0, -1), edges)
        self.assertEqual(len(edges), 23)

    def test_other_counts_draw_nothing(self) -> None:
        self.assertEqual(skeleton_edges(0), [])
        self.assertEqual(skeleton_edges(10), [])
```

- [ ] **Step 2: Run test to verify it fails**

Run from `backend/`:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gvhmr_slice1.TestSkeletonEdges -v
```

Expected: FAIL with `ModuleNotFoundError: app.pipeline.smpl_skeleton`

- [ ] **Step 3: Write minimal implementation**

```python
"""SMPL-24 kinematic tree for overlay. COCO-17 stays the old path."""

from __future__ import annotations

from app.pipeline.pose_track import COCO_EDGES

# Parent of each SMPL joint. -1 is the pelvis root.
SMPL_PARENTS: tuple[int, ...] = (
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21,
)


def skeleton_edges(n_keypoints: int) -> list[tuple[int, int]]:
    if n_keypoints == 17:
        return list(COCO_EDGES)
    if n_keypoints == 24:
        return [(i, parent) for i, parent in enumerate(SMPL_PARENTS) if parent >= 0]
    return []
```

- [ ] **Step 4: Run test to verify it passes**

Same command. Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```powershell
git add backend/app/pipeline/smpl_skeleton.py backend/tests/test_gvhmr_slice1.py
git commit -m "Add SMPL-24 overlay edges beside the COCO-17 skeleton."
```

### Task 2: In-memory COCO-17 bridge

**Files:**
- Modify: `backend/app/pipeline/smpl_skeleton.py`
- Test: `backend/tests/test_gvhmr_slice1.py`

**Interfaces:**
- Consumes: nothing from Task 1 except the same module
- Produces: `smpl_keypoints_to_coco17(keypoints: list[list[float]]) -> list[list[float]]` — 17×3, nose/eyes/ears from SMPL head 15, eyes/ears confidence 0

- [ ] **Step 1: Write the failing test**

```python
from app.pipeline.smpl_skeleton import smpl_keypoints_to_coco17


class TestCocoBridge(unittest.TestCase):
    def test_maps_body_and_zeros_face_confidence(self) -> None:
        smpl = [[float(i), float(i) + 0.1, 1.0] for i in range(24)]
        coco = smpl_keypoints_to_coco17(smpl)
        self.assertEqual(len(coco), 17)
        self.assertEqual(coco[0], [15.0, 15.1, 1.0])  # nose ← head
        self.assertEqual(coco[1][2], 0.0)  # eye confidence
        self.assertEqual(coco[5], [16.0, 16.1, 1.0])  # L shoulder
        self.assertEqual(coco[6], [17.0, 17.1, 1.0])  # R shoulder
        self.assertEqual(coco[11], [1.0, 1.1, 1.0])  # L hip
        self.assertEqual(coco[15], [7.0, 7.1, 1.0])  # L ankle
        self.assertEqual(coco[16], [8.0, 8.1, 1.0])  # R ankle

    def test_rejects_wrong_length(self) -> None:
        with self.assertRaises(ValueError):
            smpl_keypoints_to_coco17([[0.0, 0.0, 1.0]])
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gvhmr_slice1.TestCocoBridge -v
```

Expected: FAIL with `ImportError` on `smpl_keypoints_to_coco17`.

- [ ] **Step 3: Write minimal implementation**

Append to `smpl_skeleton.py`:

```python
# COCO index → (SMPL index, confidence override or None to keep source conf)
_COCO_FROM_SMPL: tuple[tuple[int, float | None], ...] = (
    (15, None),  # nose ← head
    (15, 0.0),
    (15, 0.0),
    (15, 0.0),
    (15, 0.0),  # eyes, ears
    (16, None),
    (17, None),
    (18, None),
    (19, None),
    (20, None),
    (21, None),
    (1, None),
    (2, None),
    (4, None),
    (5, None),
    (7, None),
    (8, None),
)


def smpl_keypoints_to_coco17(keypoints: list[list[float]]) -> list[list[float]]:
    if len(keypoints) != 24:
        raise ValueError(f"expected 24 SMPL keypoints, got {len(keypoints)}")
    out: list[list[float]] = []
    for smpl_i, conf_override in _COCO_FROM_SMPL:
        x, y, conf = keypoints[smpl_i][:3]
        out.append([float(x), float(y), float(conf if conf_override is None else conf_override)])
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Same command. Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add backend/app/pipeline/smpl_skeleton.py backend/tests/test_gvhmr_slice1.py
git commit -m "Map projected SMPL joints to COCO-17 for the metric bridge."
```

### Task 3: Parse GVHMR JSON and rank tracks

**Files:**
- Create: `backend/app/pipeline/gvhmr_io.py`
- Modify: `backend/app/schemas.py` (`PoseFrame`, `AnalysisResult`)
- Test: `backend/tests/test_gvhmr_slice1.py`

**Interfaces:**
- Consumes: nothing from the skeleton module
- Produces:
  - `PoseFrame.joints3d: list[list[float]] | None = None`
  - `AnalysisResult.poseBackend: str = "yolo"`, `coordFrame: str = ""`, `staticCamera: bool = False`, `ignoredTracks: int = 0`
  - `parse_gvhmr_document(data: dict) -> tuple[list[list[PoseFrame]], int]`
  - Raises `ValueError` if any present frame is not 24×3 joints and 24×3 keypoints
  - Returns tracks sorted by `(-frame_count, first_time)`, and the file's `ignoredTracks`

- [ ] **Step 1: Write the failing test**

```python
from app.pipeline.gvhmr_io import parse_gvhmr_document
from app.schemas import AnalysisResult, PoseFrame


def _frame(t: float, n: int = 24) -> dict:
    return {
        "time": t,
        "joints3d": [[0.1, 0.2, 0.3] for _ in range(n)],
        "keypoints": [[0.4, 0.5, 1.0] for _ in range(n)],
    }


class TestParseGvhmr(unittest.TestCase):
    def test_sorts_longest_then_earlier_and_keeps_gap(self) -> None:
        doc = {
            "staticCamera": True,
            "coordFrame": "gvhmr_world",
            "ignoredTracks": 1,
            "tracks": [
                {"frames": [_frame(1.0), _frame(1.1)]},
                {"frames": [_frame(0.0), _frame(0.5), _frame(1.0)]},
            ],
        }
        tracks, ignored = parse_gvhmr_document(doc)
        self.assertEqual(ignored, 1)
        self.assertEqual(len(tracks), 2)
        self.assertEqual([p.time for p in tracks[0]], [0.0, 0.5, 1.0])
        self.assertEqual(len(tracks[0][0].joints3d), 24)
        self.assertEqual(len(tracks[0][0].keypoints), 24)
        self.assertNotIn(0.25, [p.time for p in tracks[0]])

    def test_tie_uses_earlier_first_sample(self) -> None:
        doc = {
            "ignoredTracks": 0,
            "tracks": [
                {"frames": [_frame(2.0), _frame(3.0)]},
                {"frames": [_frame(0.5), _frame(1.5)]},
            ],
        }
        tracks, _ = parse_gvhmr_document(doc)
        self.assertEqual(tracks[0][0].time, 0.5)

    def test_rejects_short_joints(self) -> None:
        doc = {"ignoredTracks": 0, "tracks": [{"frames": [_frame(0.0, n=3)]}]}
        with self.assertRaises(ValueError):
            parse_gvhmr_document(doc)

    def test_old_analysis_json_still_loads(self) -> None:
        old = {
            "version": 1,
            "duration": 1.0,
            "fps": 30.0,
            "actors": [{"id": "track_0", "label": "Dancer 1", "poses": [{"time": 0.0, "keypoints": [[0, 0, 1]]}]}],
        }
        result = AnalysisResult.model_validate(old)
        self.assertEqual(result.poseBackend, "yolo")
        self.assertIsNone(result.actors[0].poses[0].joints3d)
        self.assertFalse(result.staticCamera)
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gvhmr_slice1.TestParseGvhmr -v
```

Expected: FAIL importing `parse_gvhmr_document`.

- [ ] **Step 3: Write minimal implementation**

`PoseFrame` gains:

```python
joints3d: list[list[float]] | None = None  # SMPL-24 meters, gvhmr_world; None on old files
```

`AnalysisResult` gains, all with defaults so old files load:

```python
poseBackend: str = "yolo"
coordFrame: str = ""
staticCamera: bool = False
ignoredTracks: int = 0
```

`gvhmr_io.py`:

```python
"""Parse the GVHMR adapter JSON. No torch import — safe in the app venv."""

from __future__ import annotations

from app.schemas import PoseFrame


def _as_pose(frame: dict) -> PoseFrame:
    joints = frame.get("joints3d") or []
    kpts = frame.get("keypoints") or []
    if len(joints) != 24 or any(len(j) != 3 for j in joints):
        raise ValueError("joints3d must be 24x3")
    if len(kpts) != 24 or any(len(k) != 3 for k in kpts):
        raise ValueError("keypoints must be 24x3")
    return PoseFrame(
        time=float(frame["time"]),
        joints3d=[[float(v) for v in j] for j in joints],
        keypoints=[[float(v) for v in k] for k in kpts],
    )


def parse_gvhmr_document(data: dict) -> tuple[list[list[PoseFrame]], int]:
    raw_tracks = data.get("tracks") or []
    parsed: list[list[PoseFrame]] = []
    for track in raw_tracks:
        frames = [_as_pose(f) for f in (track.get("frames") or [])]
        frames.sort(key=lambda p: p.time)
        if frames:
            parsed.append(frames)
    parsed.sort(key=lambda frames: (-len(frames), frames[0].time))
    ignored = int(data.get("ignoredTracks") or 0)
    return parsed, ignored
```

- [ ] **Step 4: Run test to verify it passes**

Same command. Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```powershell
git add backend/app/pipeline/gvhmr_io.py backend/app/schemas.py backend/tests/test_gvhmr_slice1.py
git commit -m "Parse GVHMR track JSON into SMPL pose frames."
```

### Task 4: Metric bridge on load without replacing stored joints

**Files:**
- Modify: `backend/app/pipeline/metrics.py` (`_actor_to_buffer`, `enrich_metrics`)
- Test: `backend/tests/test_gvhmr_slice1.py`

**Interfaces:**
- Consumes: `smpl_keypoints_to_coco17`
- Produces: `enrich_metrics` leaves `actor.poses[*].keypoints` at length 24 and still fills `actor.metrics` from a COCO-17 buffer

- [ ] **Step 1: Write the failing test**

```python
import numpy as np

from app.pipeline.metrics import enrich_metrics
from app.schemas import Actor, AnalysisResult, PoseFrame


class TestEnrichKeepsSmplKeypoints(unittest.TestCase):
    def test_metrics_compute_but_stored_pose_stays_24(self) -> None:
        kpts = [[0.5, 0.5, 1.0] for _ in range(24)]
        # Give the head and ankles distinct image positions so level/reach are finite.
        kpts[15] = [0.5, 0.2, 1.0]
        kpts[7] = [0.45, 0.9, 1.0]
        kpts[8] = [0.55, 0.9, 1.0]
        poses = [
            PoseFrame(time=0.0, keypoints=kpts, joints3d=[[0.0, 0.0, 0.0] for _ in range(24)]),
            PoseFrame(time=0.1, keypoints=kpts, joints3d=[[0.0, 0.0, 0.0] for _ in range(24)]),
        ]
        result = AnalysisResult(
            duration=0.1,
            fps=10.0,
            frame_width=100,
            frame_height=100,
            poseBackend="gvhmr",
            actors=[Actor(id="track_0", label="Dancer 1", poses=poses)],
        )
        enrich_metrics(result)
        self.assertEqual(len(result.actors[0].poses[0].keypoints), 24)
        self.assertGreaterEqual(len(result.actors[0].metrics), 1)
        self.assertTrue(np.isfinite(result.actors[0].metrics[0].energy))
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gvhmr_slice1.TestEnrichKeepsSmplKeypoints -v
```

Expected: FAIL (index error inside `compute_metrics`, or metrics empty because 24-point buffers are skipped).

- [ ] **Step 3: Write minimal implementation**

In `_actor_to_buffer`, after building `kpts`, if `kpts.shape[0] == 24`, replace the array used for `TrackedPose` only:

```python
from app.pipeline.smpl_skeleton import smpl_keypoints_to_coco17

# inside the pose loop, after kpts is an (N, 3) array:
if kpts.shape[0] == 24:
    kpts = np.asarray(smpl_keypoints_to_coco17(kpts.tolist()), dtype=np.float32)
```

Do not write that array back onto `PoseFrame`. `enrich_metrics` already assigns only `actor.metrics`.

- [ ] **Step 4: Run test to verify it passes**

Same command. Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add backend/app/pipeline/metrics.py backend/tests/test_gvhmr_slice1.py
git commit -m "Feed metrics from COCO-17 while storing SMPL keypoints."
```

### Task 5: Subprocess runner fails closed

**Files:**
- Create: `backend/app/pipeline/gvhmr_run.py`
- Test: `backend/tests/test_gvhmr_slice1.py`

**Interfaces:**
- Consumes: `parse_gvhmr_document`
- Produces:
  - `class GvhmrNotConfigured(RuntimeError)`
  - `run_gvhmr_export(video: Path, *, max_people: int = 2) -> tuple[list[list[PoseFrame]], int]`
  - Reads `DANCE_GVHMR_PYTHON` and `DANCE_GVHMR_ROOT`
  - Command: `[python, <repo>/backend/tools/gvhmr_export.py, --video, video, --out, tmp, --max-people, str(max_people)]`
  - `cwd` is `DANCE_GVHMR_ROOT`
  - Missing env, missing python, missing `inputs/checkpoints/yolo/yolov8x.pt`, or non-zero exit raises `GvhmrNotConfigured` whose message names the missing path or includes the last 20 stderr lines
  - Does not import ultralytics or hmr4d

- [ ] **Step 1: Write the failing test**

Add `import os`, `import sys`, `import tempfile`, and `from unittest import mock` to the test module.

```python
class TestGvhmrRunner(unittest.TestCase):
    def test_missing_python_names_the_env_var(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(GvhmrNotConfigured) as ctx:
                run_gvhmr_export(Path("clip.mp4"))
        self.assertIn("DANCE_GVHMR_PYTHON", str(ctx.exception))

    def test_nonzero_exit_includes_stderr_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "gvhmr"
            weight = root / "inputs" / "checkpoints" / "yolo" / "yolov8x.pt"
            weight.parent.mkdir(parents=True)
            weight.write_bytes(b"x")
            script = Path(tmp) / "fail.py"
            script.write_text(
                "import sys\nsys.stderr.write('boom-from-gvhmr\\n')\nsys.exit(2)\n",
                encoding="utf-8",
            )
            env = {
                "DANCE_GVHMR_PYTHON": sys.executable,
                "DANCE_GVHMR_ROOT": str(root),
            }
            with mock.patch.dict(os.environ, env, clear=True):
                with mock.patch("app.pipeline.gvhmr_run._EXPORT_SCRIPT", script):
                    with self.assertRaises(GvhmrNotConfigured) as ctx:
                        run_gvhmr_export(Path(tmp) / "clip.mp4")
        self.assertIn("boom-from-gvhmr", str(ctx.exception))
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gvhmr_slice1.TestGvhmrRunner -v
```

Expected: FAIL importing `gvhmr_run`.

- [ ] **Step 3: Write minimal implementation**

```python
"""Call the GVHMR venv. This module must not import hmr4d."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from app.pipeline.gvhmr_io import parse_gvhmr_document
from app.schemas import PoseFrame

_EXPORT_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "gvhmr_export.py"


class GvhmrNotConfigured(RuntimeError):
    pass


def _require_paths() -> tuple[Path, Path]:
    py = os.environ.get("DANCE_GVHMR_PYTHON", "").strip()
    root = os.environ.get("DANCE_GVHMR_ROOT", "").strip()
    if not py or not Path(py).is_file():
        raise GvhmrNotConfigured("DANCE_GVHMR_PYTHON is missing or not a file")
    if not root or not Path(root).is_dir():
        raise GvhmrNotConfigured("DANCE_GVHMR_ROOT is missing or not a directory")
    weight = Path(root) / "inputs" / "checkpoints" / "yolo" / "yolov8x.pt"
    if not weight.is_file():
        raise GvhmrNotConfigured(f"GVHMR weight missing: {weight}")
    return Path(py), Path(root)


def run_gvhmr_export(video: Path, *, max_people: int = 2) -> tuple[list[list[PoseFrame]], int]:
    python, root = _require_paths()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "gvhmr.json"
        proc = subprocess.run(
            [
                str(python),
                str(_EXPORT_SCRIPT),
                "--video",
                str(video),
                "--out",
                str(out),
                "--max-people",
                str(max_people),
            ],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0 or not out.is_file():
            tail = "\n".join(proc.stderr.splitlines()[-20:])
            raise GvhmrNotConfigured(tail or f"GVHMR exited {proc.returncode}")
        import json
        data = json.loads(out.read_text(encoding="utf-8"))
    return parse_gvhmr_document(data)
```

- [ ] **Step 4: Run test to verify it passes**

Same command. Expected: PASS. Also run the missing-python test from Step 1 once it is in the file.

- [ ] **Step 5: Commit**

```powershell
git add backend/app/pipeline/gvhmr_run.py backend/tests/test_gvhmr_slice1.py
git commit -m "Fail Analyze when the GVHMR environment is missing."
```

### Task 6: Adapter script in the GVHMR repo's process

**Files:**
- Create: `backend/tools/gvhmr_export.py`
- Test: `backend/tests/test_gvhmr_slice1.py` (JSON writer only; do not import hmr4d)

**Interfaces:**
- Consumes: GVHMR's `Tracker.track`, `VitPoseExtractor`, `Extractor`, `DemoPL.predict`, `make_smplx`, `smpl_neutral_J_regressor.pt`
- Produces: the `--out` file described in the spec. `ignoredTracks` is how many YOLO tracks were dropped. Frames where that id was absent are omitted.

The script is not imported by the app tests. Add one test that the file exists and its argparse help mentions `--out`. Do not execute it in CI.

- [ ] **Step 1: Write the failing test**

```python
class TestExportScriptExists(unittest.TestCase):
    def test_help_mentions_out(self) -> None:
        script = Path(__file__).resolve().parents[1] / "tools" / "gvhmr_export.py"
        proc = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("--out", proc.stdout)
```

`--help` must work without importing `hmr4d`. Parse args before the GVHMR import.

- [ ] **Step 2: Run test to verify it fails**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gvhmr_slice1.TestExportScriptExists -v
```

Expected: FAIL because the script does not exist (exit code 2).

- [ ] **Step 3: Write the script**

`backend/tools/gvhmr_export.py` must parse `--help` before any `hmr4d` import. After a real run, `cwd` is the GVHMR root (the app sets it). Use this body:

```python
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--max-people", type=int, default=2)
    args = parser.parse_args()
    # imports of hmr4d, torch, hydra stay below this line
    write_export(Path(args.video), Path(args.out), max_people=args.max_people)


def write_export(video: Path, out: Path, *, max_people: int) -> None:
    from hmr4d.utils.preproc import Tracker, Extractor, VitPoseExtractor
    # 1. history = Tracker().track(str(video))
    # 2. Group ids. Rank by (-len(frame_ids), min(frame_ids)).
    #    kept = rank[:max_people]; ignoredTracks = max(0, len(ids) - max_people)
    # 3. For each kept id, build a dense box with the same interpolate+smooth
    #    steps as Tracker.get_one_track, and a bool mask that is True only on
    #    frames where that id was detected (before interpolation).
    # 4. Per track, new preprocess dir. static_cam=True so R_w2c = eye(3)
    #    repeated over length, matching demo.load_data_dict. Then
    #    VitPoseExtractor, Extractor, DemoPL.predict(data, static_cam=True).
    # 5. Global joints, no recenter:
    #    verts = smplx2smpl @ smplx(**pred["smpl_params_global"]).vertices
    #    joints_world = J_regressor @ verts   # (L, 24, 3)
    # 6. In-camera joints from pred["smpl_params_incam"], project with
    #    K = pred["K_fullimg"][frame]: u, v = (K @ X)[:2] / z
    #    keypoints x,y are u/width, v/height, conf 1.
    # 7. time = frame_index / cv2 fps of the original video.
    # 8. Append a frame only when the detection mask is True.
    payload = {
        "staticCamera": True,
        "coordFrame": "gvhmr_world",
        "ignoredTracks": ignored,
        "tracks": tracks,
    }
    out.write_text(json.dumps(payload), encoding="utf-8")
```

The numbered comments are the required behavior. Implement those calls. Do not call `render_incam`, `render_global`, or `move_to_start_point_face_z`. Two tracks must not share one `paths.bbx` file.

- [ ] **Step 4: Run the help test**

Same command. Expected: PASS. Do not run a real video in CI.

- [ ] **Step 5: Commit**

```powershell
git add backend/tools/gvhmr_export.py backend/tests/test_gvhmr_slice1.py
git commit -m "Add the GVHMR export script that writes SMPL track JSON."
```

### Task 7: Analyze, overlay, and hide the YOLO model combo

**Files:**
- Modify: `backend/app/pipeline/run.py` (`analyze_video`)
- Modify: `desktop/playback.py` (`_draw_poses`)
- Modify: `desktop/window.py` (hide Model combo, status text)
- Test: `backend/tests/test_gvhmr_slice1.py`

**Interfaces:**
- Consumes: `run_gvhmr_export`, `skeleton_edges`, `AnalysisResult` fields from Task 3
- Produces: `analyze_video` sets `poseBackend="gvhmr"`, `coordFrame="gvhmr_world"`, `staticCamera=True`, `ignoredTracks`, and actor poses of length 24. Returned track buffers used for motifs are COCO-17 (via `_actor_to_buffer`). Progress callback receives `Running GVHMR…` before the subprocess returns.

- [ ] **Step 1: Write the failing test**

```python
class TestAnalyzeUsesGvhmr(unittest.TestCase):
    def test_analyze_stores_smpl_and_does_not_call_yolo(self) -> None:
        poses = [
            PoseFrame(
                time=0.0,
                keypoints=[[0.5, 0.5, 1.0] for _ in range(24)],
                joints3d=[[0.0, 1.0, 0.0] for _ in range(24)],
            ),
            PoseFrame(
                time=0.1,
                keypoints=[[0.5, 0.5, 1.0] for _ in range(24)],
                joints3d=[[0.0, 1.0, 0.0] for _ in range(24)],
            ),
        ]
        messages: list[str] = []

        def fake_export(video, *, max_people=2):
            return [poses, poses], 3

        with mock.patch("app.pipeline.run.run_gvhmr_export", fake_export):
            with mock.patch("app.pipeline.run.load_model") as load_model:
                with mock.patch("app.pipeline.run.track_video") as track_video:
                    result, _overlay, buffers = analyze_video(
                        Path("unused.mp4"),
                        progress_cb=lambda _p, msg: messages.append(msg),
                    )
        load_model.assert_not_called()
        track_video.assert_not_called()
        self.assertEqual(result.poseBackend, "gvhmr")
        self.assertEqual(result.coordFrame, "gvhmr_world")
        self.assertTrue(result.staticCamera)
        self.assertEqual(result.ignoredTracks, 3)
        self.assertEqual(len(result.actors[0].poses[0].keypoints), 24)
        self.assertIn("Running GVHMR…", messages)
        # Motif buffers are the bridge, not the stored 24.
        first = next(iter(buffers.values()))
        self.assertEqual(first.poses[0].keypoints.shape[0], 17)
```

`analyze_video` still needs duration, fps, width, height. Read them with `cv2.VideoCapture` inside `analyze_video`. In this test, patch `cv2.VideoCapture` with a fake:

```python
class _Cap:
    def isOpened(self):
        return True
    def get(self, prop):
        return {cv2.CAP_PROP_FPS: 10.0, cv2.CAP_PROP_FRAME_COUNT: 2.0,
                cv2.CAP_PROP_FRAME_WIDTH: 100.0, cv2.CAP_PROP_FRAME_HEIGHT: 100.0}[prop]
    def release(self):
        return None
```

Overlay still receives only the selected dancer from `sync_overlay`. Do not draw every actor in this task.

- [ ] **Step 2: Run test to verify it fails**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gvhmr_slice1.TestAnalyzeUsesGvhmr -v
```

Expected: FAIL because `analyze_video` still calls `load_model`.

- [ ] **Step 3: Write minimal implementation**

In `analyze_video`:

- Delete the `load_model` / `track_video` calls.
- `progress(0.1, "Running GVHMR…")` then `tracks, ignored = run_gvhmr_export(video_path, max_people=min(max_people, 2))`.
- Read fps, frame count, width, height from the capture. `duration = frame_count / fps`.
- Build `TrackBuffer`s whose `TrackedPose.keypoints` are the stored 24-point arrays. Pass them through `_actor_to_buffer` (Task 4 converts to COCO-17 for metrics). Then replace each actor's `poses` with the original 24-point `PoseFrame`s, including `joints3d`.
- Set `poseBackend`, `coordFrame`, `staticCamera`, `ignoredTracks` on `AnalysisResult`.
- Keep `detect_motif_types` and `enrich_metrics`. `enrich_metrics` will see 24-point stored poses and bridge again. That is correct.
- If `tracks` is empty, return an analysis with no actors and do not raise.
- On `GvhmrNotConfigured`, let it propagate. `run_pipeline` already stores `str(exc)`.

`desktop/playback.py` `_draw_poses`: replace `for a, b in COCO_EDGES` with `for a, b in skeleton_edges(len(pts))`.

`desktop/window.py`:

- `self.model_label = QLabel("Model")`, then `self.model_combo.hide()` and `self.model_label.hide()`.
- `start_analyze` initial status: `Running GVHMR…`.
- `on_analyze_done`: if `result.actors` is empty, set status to `No person was tracked.` If `result.ignoredTracks > 0`, append ` · ignored {n} extra tracks`.

- [ ] **Step 4: Run the slice-1 tests**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gvhmr_slice1 -v
```

Expected: PASS. Then run the existing motif tests so the COCO bridge did not break 17-point projects:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_dtw_w1 tests.test_features_w2 tests.test_orphan_attach -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add backend/app/pipeline/run.py desktop/playback.py desktop/window.py backend/tests/test_gvhmr_slice1.py
git commit -m "Analyze with GVHMR and draw the SMPL skeleton."
```

### Task 8: Manual check (not CI)

On a machine where `DANCE_GVHMR_PYTHON` and `DANCE_GVHMR_ROOT` are set and `inputs/checkpoints/yolo/yolov8x.pt` exists:

- [ ] Unset the env vars, Analyze, and confirm the status names `DANCE_GVHMR_PYTHON` or `DANCE_GVHMR_ROOT`. The app must not report a YOLO success.
- [ ] One-dancer clip: bones stay on the body during playback. Stored poses have 24 keypoints.
- [ ] Two-dancer clip: two actors. A clip with a third person shows `ignored N extra tracks`.
- [ ] Open an old `.dance.json`: 17-point COCO skeleton still draws.

No commit unless a bugfix from this pass changes code. If it does, commit that fix on its own with the test that failed.
