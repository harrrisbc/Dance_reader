# GVHMR Phase 2 — 3D metrics + dual floor + level Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `metricsSpace: "smpl3d"` (all charts + dual world floor from `joints3d`), redefine `level` on both paths, remove the COCO-17 bridge for 24-point tracks, and retune motif gates on the real fixtures.

**Architecture:** Branch inside `metrics.py` / motif feature builders: 24×`joints3d` → world-metre formulas; else existing 2D path with new level only. `enrich_metrics` always recomputes curves on load. Desktop floor map reads `metricsSpace` and draws pelvis + foot when `smpl3d`.

**Tech Stack:** Python, Pydantic schemas, NumPy, existing `TrackBuffer` / motif DTW, PySide6 + matplotlib desktop charts.

## Global Constraints

- UI copy and docs in this plan’s product surfaces: **English**
- Chat with user: Cantonese (agents); commit messages: English
- No GPU / GVHMR weights required for CI tests — use synthetic `joints3d` fixtures
- Do not copy old DTW gates after feature change; calibrate once on 武拜廟 + `hi`
- Mesh bake failure must not block metrics
- `static_cam` world XZ is not surveyed stage metres — tooltip must say so
- Commit only when the user asks (or when executing this plan under an explicit commit instruction)

**Spec:** `docs/superpowers/specs/2026-10-15-gvhmr-3d-metrics-phase2-design.md`

---

## File map

| File | Responsibility |
|------|----------------|
| `backend/app/schemas.py` | `metricsSpace`, `pos_foot_x` / `pos_foot_y` |
| `backend/app/pipeline/metrics.py` | Level helper; `compute_metrics` (2D); `compute_metrics_smpl3d`; `enrich_metrics` branch |
| `backend/app/pipeline/run.py` | GVHMR analyze: no COCO bridge; 3D motif buffers |
| `backend/app/pipeline/motifs.py` | 3D pose features if buffer has `joints3d`; gate constants after retune |
| `backend/app/pipeline/smpl_skeleton.py` | Remove `smpl_keypoints_to_coco17` when unused |
| `desktop/window.py` | Dual floor map; chart labels/ylim; optional space label |
| `backend/tests/test_metrics_phase2.py` | New CI tests |
| Cheatsheets + `IMPLEMENTATION_NOTES_GVHMR_3D_METRICS.md` | Docs |

SMPL-24 indices used in this plan: `0` pelvis, `1`/`2` L/R hip, `7`/`8` L/R ankle, `12` neck, `15` head, `16`/`17` L/R shoulder, `20`/`21` L/R wrist. COCO ankles `15`/`16`, hips `11`/`12`, shoulders `5`/`6`.

---

### Task 1: Schema — `metricsSpace` + foot floor fields

**Files:**
- Modify: `backend/app/schemas.py`
- Test: `backend/tests/test_metrics_phase2.py` (create)

**Interfaces:**
- Produces: `AnalysisResult.metricsSpace: str` default `"coco2d"`; `MetricPoint.pos_foot_x: float = 0.0`, `pos_foot_y: float = 0.0`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_metrics_phase2.py
import unittest
from app.schemas import AnalysisResult, MetricPoint

class TestMetricsPhase2Schema(unittest.TestCase):
    def test_defaults(self) -> None:
        r = AnalysisResult(duration=1.0, fps=30.0)
        self.assertEqual(r.metricsSpace, "coco2d")
        p = MetricPoint(time=0.0, energy=0.0, speed=0.0)
        self.assertEqual(p.pos_foot_x, 0.0)
        self.assertEqual(p.pos_foot_y, 0.0)

    def test_roundtrip_smpl3d(self) -> None:
        r = AnalysisResult(
            duration=1.0, fps=30.0, metricsSpace="smpl3d",
            actors=[],
        )
        d = r.model_dump()
        self.assertEqual(AnalysisResult.model_validate(d).metricsSpace, "smpl3d")
```

- [ ] **Step 2: Run test — expect FAIL** (no `metricsSpace`)

```bash
.\backend\.venv\Scripts\python.exe -m unittest backend.tests.test_metrics_phase2.TestMetricsPhase2Schema -v
```

- [ ] **Step 3: Add fields to schemas**

```python
# AnalysisResult
metricsSpace: str = "coco2d"  # "smpl3d" | "coco2d"

# MetricPoint
pos_foot_x: float = 0.0
pos_foot_y: float = 0.0
# Update pos_x / pos_y comments: screen 0–1 for coco2d; pelvis world X/Z metres for smpl3d
```

- [ ] **Step 4: Re-run tests — PASS**

- [ ] **Step 5: Commit** (when user allows): `Add metricsSpace and foot floor fields to schema.`

---

### Task 2: New `level` on the 2D path

**Files:**
- Modify: `backend/app/pipeline/metrics.py`
- Test: `backend/tests/test_metrics_phase2.py`

**Interfaces:**
- Produces: `level_stance_ratio_2d(sq, scale) -> float | None` using supporting (lower) ankle vs hip mid ÷ `S`
- Removes use of track p90 leg length for level

- [ ] **Step 1: Failing tests**

```python
import numpy as np
from app.pipeline.metrics import level_stance_ratio_2d

class TestLevel2d(unittest.TestCase):
    def test_supporting_ankle_not_lifted_leg(self) -> None:
        # square-pixel COCO-shaped (17,3); hip mid y=100, left ankle y=180, right y=120 (lifted)
        sq = np.zeros((17, 3), np.float32)
        sq[:, 2] = 1.0
        sq[5] = [50, 40, 1]; sq[6] = [70, 40, 1]   # shoulders
        sq[11] = [50, 100, 1]; sq[12] = [70, 100, 1]  # hips
        sq[15] = [55, 180, 1]; sq[16] = [65, 120, 1]  # ankles
        S = 60.0  # torso
        lv = level_stance_ratio_2d(sq, S)
        # drop uses lower ankle y=180 → |180-100|/60 = 1.333…
        self.assertAlmostEqual(lv, abs(180 - 100) / 60.0, places=3)
```

- [ ] **Step 2: Run — FAIL** (helper missing)

- [ ] **Step 3: Implement helper; wire into `compute_metrics` replacing p90 leg path**

```python
def level_stance_ratio_2d(sq: np.ndarray, scale: float) -> float | None:
    if scale <= 1e-6 or sq.shape[0] < 17:
        return None
    hip_y = 0.5 * (float(sq[11, 1]) + float(sq[12, 1]))
    ankles = []
    for i in (15, 16):
        if float(sq[i, 2]) >= TORSO_CONF:
            ankles.append(float(sq[i, 1]))
    if not ankles:
        return None
    # image Y down: supporting = max y
    drop = abs(max(ankles) - hip_y)
    return float(drop / scale)
```

Replace the block that fills `level_n` from `leg_r` / p90 with per-frame `level_stance_ratio_2d` + carry-forward.

Also set `pos_foot_x` / `pos_foot_y` from supporting foot screen coords (same source as today’s foot `pos_y`, with matching x).

- [ ] **Step 4: Run unit test + any existing metrics tests**

```bash
.\backend\.venv\Scripts\python.exe -m unittest backend.tests.test_metrics_phase2.TestLevel2d -v
```

- [ ] **Step 5: Commit:** `Redefine 2D level as hip-to-supporting-ankle over torso.`

---

### Task 3: `compute_metrics_smpl3d`

**Files:**
- Modify: `backend/app/pipeline/metrics.py`
- Test: `backend/tests/test_metrics_phase2.py`

**Interfaces:**
- Consumes: `TrackBuffer` whose `TrackedPose` carries `joints3d` (T×24×3) — extend buffer pose type if needed, or read from parallel list on `PoseFrame` via a new helper that builds a 3D-aware buffer
- Produces: `compute_metrics_smpl3d(buffer, poses_with_joints3d) -> list[MetricPoint]` with world `pos_*` / `pos_foot_*`

**Preferred buffer approach:** Keep `TrackBuffer` keypoints as projected 24 for overlay; add optional `joints3d: np.ndarray | None` on each `TrackedPose` (or a parallel array on the buffer). GVHMR `run.py` fills it. Motifs read `joints3d` when present.

- [ ] **Step 1: Failing tests**

```python
class TestMetricsSmpl3d(unittest.TestCase):
    def test_travel_uses_pelvis_xz(self) -> None:
        # Two frames: pelvis moves 1m on X, Y constant; expect travel ~ 1/S /dt in torso/s
        ...

    def test_level_flat_on_turn(self) -> None:
        # Same hip/ankle Y, rotate XZ → level nearly constant

    def test_lifted_leg_ignored(self) -> None:
        # Free ankle higher; supporting lower → level uses lower ankle
```

Build minimal synthetic buffers (2–5 frames) with known metres.

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Implement**

Key helpers (names locked):

```python
SMPL_L_ANKLE, SMPL_R_ANKLE = 7, 8
SMPL_L_HIP, SMPL_R_HIP = 1, 2
SMPL_PELVIS = 0

def torso_scale_3d(j: np.ndarray) -> float | None: ...
def level_stance_ratio_3d(j: np.ndarray, scale: float) -> float | None: ...
def compute_metrics_smpl3d(buffer: TrackBuffer) -> list[MetricPoint]: ...
```

Formulas per spec. Travel/directness window ≈ existing ~0.8 s. Smooth energy/speed with existing `smooth_series`.

- [ ] **Step 4: Tests PASS**

- [ ] **Step 5: Commit:** `Add smpl3d metric path from joints3d.`

---

### Task 4: `enrich_metrics` + GVHMR `run.py` branch; delete bridge

**Files:**
- Modify: `backend/app/pipeline/metrics.py` (`enrich_metrics`)
- Modify: `backend/app/pipeline/run.py`
- Modify: `backend/app/pipeline/smpl_skeleton.py` (delete bridge when unused)
- Test: `backend/tests/test_metrics_phase2.py`

**Interfaces:**
- Produces: `enrich_metrics` sets `result.metricsSpace` and per-actor metrics via the correct path; GVHMR analyze stores 24-point poses with `joints3d` and builds motif buffers **without** `smpl_keypoints_to_coco17`

- [ ] **Step 1: Test** — AnalysisResult with one actor of 24 poses + `joints3d` → after `enrich_metrics`, `metricsSpace=="smpl3d"` and `pos_x` not clipped to 0–1 like screen. 17-point actor → `"coco2d"`.

- [ ] **Step 2: Implement branch**

```python
def _actor_has_joints3d(actor: Actor) -> bool:
    return any(
        p.joints3d is not None and len(p.joints3d) == 24
        for p in actor.poses
    )
```

- [ ] **Step 3: Grep and remove `smpl_keypoints_to_coco17` call sites; delete function if unused**

```bash
rg smpl_keypoints_to_coco17
```

- [ ] **Step 4: Full phase2 + existing gvhmr mesh tests**

```bash
.\backend\.venv\Scripts\python.exe -m unittest backend.tests.test_metrics_phase2 backend.tests.test_gvhmr_mesh_phase1 -v
```

- [ ] **Step 5: Commit:** `Route enrich/analyze to smpl3d and drop COCO bridge.`

---

### Task 5: Motif features from 3D

**Files:**
- Modify: `backend/app/pipeline/motifs.py`
- Test: `backend/tests/test_metrics_phase2.py` (or existing motif tests)

**Interfaces:**
- When buffer poses have `joints3d`, `pose_feature` / resample path uses 3D relative coords ÷ `S` + bone unit vectors; mirror swaps L/R SMPL pairs
- Phrase cutting still uses smoothed energy from the same metric path

- [ ] **Step 1: Test** — 3D feature length stable; mirror flips left/right wrists; distance of identical phrase ≈ 0

- [ ] **Step 2: Implement 3D feature branch** (keep 2D path for coco buffers)

- [ ] **Step 3: Tests PASS**

- [ ] **Step 4: Commit:** `Build motif pose features from joints3d when present.`

---

### Task 6: Retune gates on fixtures

**Files:**
- Modify: `backend/app/pipeline/motifs.py` (constants)
- Create/update: `docs/IMPLEMENTATION_NOTES_GVHMR_3D_METRICS.md` (calibration table)
- Fixtures (read-only): paths in `AGENT_HANDOFF_INDEX.md` (武拜廟 positive, `hi` negative)

**Interfaces:**
- Produces: updated `CLUSTER_MAX_DIST` / related gates **only after** measuring raw DTW on fixtures with the new features; document before/after

- [ ] **Step 1: Script or one-off notebook/commands** — load both `.dance.json`, `enrich_metrics`, run motif detect, print phrase counts, cluster cut histogram, types kept

- [ ] **Step 2: Adjust gates** so positive keeps repeated types; negative does not invent a mega-type (precision > recall)

- [ ] **Step 3: Record measurements + final constants in implementation notes

- [ ] **Step 4: Commit:** `Retune motif gates for smpl3d features.`

---

### Task 7: Desktop floor map + chart labels

**Files:**
- Modify: `desktop/window.py` (floor plot + `CHART_SERIES` ylim/ylabel for pos when `smpl3d`)

**Interfaces:**
- Consumes: `result.metricsSpace`, `MetricPoint.pos_*`, `pos_foot_*`
- Produces: dual trajectory when `smpl3d`; tooltip English string about static-cam world metres

- [ ] **Step 1: Branch floor renderer**

```python
if result.metricsSpace == "smpl3d":
    # solid pelvis (pos_x, pos_y), dashed foot (pos_foot_x, pos_foot_y)
    # equal aspect; xlabel/ylabel "world X (m)" / "world Z (m)"
    # do NOT invert_yaxis (not image space)
else:
    # existing screen floor map
```

- [ ] **Step 2: Pos chart series** — ylabel `m (world)` vs `screen`; autoscale or measured ylim for smpl3d

- [ ] **Step 3: Optional label near charts: `Metrics: smpl3d` / `coco2d`

- [ ] **Step 4: Manual smoke** — open GVHMR project, enable floor map, scrub; open YOLO-only project, confirm single screen trail

- [ ] **Step 5: Commit:** `Show dual world floor map for smpl3d metrics.`

---

### Task 8: Docs + handoff closeout

**Files:**
- Modify: `docs/CHEATSHEET_METRICS_MOTIFS.md`, `docs/CHEATSHEET_METRICS_MOTIFS_YUE.md`
- Create: `docs/IMPLEMENTATION_NOTES_GVHMR_3D_METRICS.md` (if not fully written in Task 6)
- Modify: `docs/AGENT_HANDOFF_GVHMR_MESH_VIEWER_METRICS.md`, `docs/AGENT_HANDOFF_LEVEL_DEFINITION.md`, `docs/AGENT_HANDOFF_INDEX.md`

- [ ] **Step 1: Update cheatsheets** for new level, smpl3d formulas, dual floor

- [ ] **Step 2: Mark Phase 2 acceptance checkboxes; level handoff `done`; index status**

- [ ] **Step 3: Commit:** `Document Phase 2 3D metrics and close level handoff.`

---

## Spec coverage checklist

| Spec requirement | Task |
|------------------|------|
| `metricsSpace` smpl3d/coco2d | 1, 4 |
| All smpl3d charts from joints3d | 3, 4 |
| Dual floor pelvis + foot | 3, 7 |
| New level both paths | 2, 3 |
| Delete COCO bridge | 4 |
| Motif 3D features + retune | 5, 6 |
| Bake fail ≠ block metrics | 4 (no bake dependency) |
| CI fixtures | 1–5 |
| Cheatsheets / notes | 6, 8 |

## Placeholder scan

No TBD steps; calibration Task 6 requires real fixture runs (commands + recorded table, not invented thresholds).
