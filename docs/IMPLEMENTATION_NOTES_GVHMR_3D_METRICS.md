# GVHMR Phase 2 — 3D metrics implementation notes

## What shipped

- `metricsSpace`: `"smpl3d"` when any actor has 24×`joints3d`; else `"coco2d"`.
- **smpl3d:** speed / energy / reach / travel / level / directness from world joints; floor `pos_*` = pelvis XZ (m); `pos_foot_*` = supporting ankle XZ (m).
- **coco2d:** prior formulas except **new level** (hip → supporting ankle / torso).
- COCO-17 bridge unused when `joints3d` present; `buffers_from_analysis` keeps `joints3d` for motifs.
- Desktop floor map: dual path (pelvis solid + foot dashed) in world metres when `smpl3d`.
- Motif features / energy valleys use 3D when buffers carry `joints3d`.

## Follow-ups (same phase)

- **Level bands** from 武拜廟 smpl3d (n=968): low &lt; 1.25 (bow trough 1.22), mid 1.25–1.50, high &gt; 1.50 (standing p75 = 1.52). Chart ylim `(1.0, 1.75)`. No deep-plié fixture yet.
- **`energy_abs`:** smoothed `Σ(v/S)²` before ÷ peak, so clips can be compared. Chart “Energy (abs)” is off by default. `energy` stays 0–1 relative.
- **Floor origin:** smpl3d `pos_*` / `pos_foot_*` subtracted by the first pelvis XZ (travel deltas unchanged).

## Calibration (2026-10-15)

Fixtures after `enrich_metrics` + fixed buffers:

| Clip | Space | Result with gates `CLUSTER_MAX_DIST=0.24`, `INTRA_MAX_DIST=0.26`, `min_occ=3` |
|------|-------|-------------------------------------------------------------------------------|
| `07 走路+武拜廟.dance.json` | smpl3d | **2 types** kept (mean_intra ≈ 0.17 / 0.21); +1 orphan attach |
| `hi.dance.json` | coco2d | **0 types** (cluster had &lt;3 occ after NMS) |

**No gate constant change** required for this feature cut. Revisit if a GVHMR-analyzed `hi` (smpl3d) false-positive appears.

## How to test

1. Open a GVHMR `.dance.json` with `joints3d` — charts should show world-scale travel/pos; floor map dual trails; status/`metricsSpace` = smpl3d after enrich.
2. Open YOLO-only project — screen floor; new level formula; `coco2d`.
3. CI: `python -m unittest backend.tests.test_metrics_phase2 -v`

## Key files

`schemas.py`, `metrics.py`, `motifs.py`, `run.py`, `project_io.py`, `desktop/window.py`, `backend/tests/test_metrics_phase2.py`
