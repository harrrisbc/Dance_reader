# GVHMR mesh, 3D viewer, and 3D metrics — design

Date: 2026-10-14  
Depends on: [2026-09-29-gvhmr-3d-bones-design.md](./2026-09-29-gvhmr-3d-bones-design.md) (slice 1 shipped)

## Goal

After Analyze, the user can:

1. See an SMPL **mesh** overlaid on the recorded video (in addition to the wire skeleton).
2. Orbit a **3D viewer** of the same motion in world space, synced to the playhead.
3. Get charts and motifs from **true 3D joints** (meters), not the temporary COCO-17 bridge.

Delivery is **one product direction, two implementation phases**, plus a documented Phase 3 that is not built yet.

## Decisions locked with the user

| Topic | Choice |
|-------|--------|
| Surfaces | Both: video mesh overlay **and** side 3D viewer |
| When mesh is built | Analyze stores SMPL params; **first play** bakes a reusable cache |
| Wire vs mesh | Independent toggles (both may be on) |
| Live Link / FBX | Specified as Phase 3; **not** implemented in this work |
| Metrics | Phase 2 removes the COCO-17 bridge and uses `joints3d` |
| YOLO | Still required for **person box tracking** inside GVHMR. Not used as the pose backbone |

## Phase plan

| Phase | Ships | Does not ship |
|-------|--------|----------------|
| **1** | Params sidecar, first-play mesh bake, video mesh composite, Qt 3D viewer, wire+mesh toggles | Metric formula changes, Live Link |
| **2** | 3D metrics + motifs from `joints3d`; delete bridge; `metricsSpace` label | Mesh work beyond Phase 1 fixes |
| **3** (spec only) | — | Live Link to UE5, BVH/FBX export, UE axis conversion |

Programming executes Phase 1, then Phase 2, unless the user reprioritizes.

---

## Phase 1 — Mesh overlay + 3D viewer

### Analyze output

Extend `backend/tools/gvhmr_export.py` so each kept track still writes `frames[]` with `time`, `joints3d` (24×3), `keypoints` (24×3), and **also** persists SMPL parameters needed to rebuild the mesh without re-running GVHMR prediction:

| Field | Role |
|-------|------|
| `smpl_params_global` | World mesh / 3D viewer |
| `smpl_params_incam` | In-camera mesh for video overlay |
| `K_fullimg` | Projection (per frame or broadcast) |

**Storage:** do **not** dump full params into `.dance.json` (too large). Write a sidecar next to the analysis file, e.g. `*.dance.smpl.npz`, and record a relative path + content hash in the JSON (`smplSidecar`, `smplSidecarSha256`). JSON remains the machine-readable contract the app opens first.

App venv still must not import `hmr4d` / torch for Analyze orchestration beyond the existing subprocess.

### First-play bake

When the user opens a project that has a params sidecar but **no** mesh cache:

1. Desktop status: `Building mesh cache…`
2. Subprocess via `DANCE_GVHMR_PYTHON` + `DANCE_GVHMR_ROOT` runs a bake script (new tool under `backend/tools/`, same pattern as `gvhmr_export.py`).
3. Bake writes `*.dance.mesh.npz` (or equivalent) containing at least:
   - `verts_incam` `(T, V, 3)`
   - `verts_world` `(T, V, 3)`
   - `faces` `(F, 3)` shared
   - `times` / detection mask aligned with stored frames
4. Preferred overlay quality: bake also produces a soft-rendered translucent mesh layer (RGBA frames or a short overlay video) using the same renderer path as GVHMR’s demo, then the desktop **alpha-composites** onto the decoded video.
5. `faces` are stored in the cache so later playback does not need body-model files in the app venv.

**`SMPL_NEUTRAL.pkl`:** required for the preferred SMPL-face renderer path during bake. If missing, bake still writes verts when possible; video mesh falls back to projected mesh edges; 3D viewer still works from `verts_world` + faces if faces can be obtained another way, otherwise show a clear status.

Wire overlay and Phase 2 metrics must **not** block on bake success.

### Playback UI (`desktop/`)

- Split preview: **left** video (existing decode worker), **right** 3D viewer.
- Keep Skeleton / Joints / Off combo.
- Add **Mesh overlay** checkbox (independent).
- Add **3D viewer** checkbox (show/hide right pane).
- Sync all three to the same playhead time / frame index.

**3D viewer tech:** read numpy verts from the mesh cache in the **app** venv. Prefer **PyVista + pyvistaqt** (or equivalent Qt-friendly OpenGL) with orbit/zoom. No Unreal, no Live Link in Phase 1. No torch in the app venv for viewing.

**Old projects** without a params sidecar: no mesh, no 3D viewer; status suggests Re-analyze. Existing wire playback unchanged.

### Errors (Phase 1)

| Case | Behavior |
|------|----------|
| First play, no cache | `Building mesh cache…`; wire still works |
| Bake fails (missing SMPL pickle, OOM, etc.) | Status names the cause; edge-projection fallback if verts exist; viewer if verts+faces exist |
| Missing sidecar | No mesh / no viewer; suggest Re-analyze |
| GVHMR env missing | Same fail-closed Analyze behavior as slice 1 |

---

## Phase 2 — Metrics from `joints3d`

**Authoritative detail (2026-10-15):**  
[2026-10-15-gvhmr-3d-metrics-phase2-design.md](./2026-10-15-gvhmr-3d-metrics-phase2-design.md)

Summary locked with the user:

- Branch in `metrics.py`: 24×`joints3d` → all charts + motifs in world 3D (`metricsSpace: "smpl3d"`); else `coco2d`.
- Delete COCO-17 bridge on the 24-point path.
- Floor map (`smpl3d`): pelvis XZ solid + supporting-foot XZ dashed (metres, static-cam world).
- Redefine `level` on **both** paths (hip → supporting ankle / torso); absorb level handoff.
- Retune motif gates on 武拜廟 + `hi` before Phase 2 is done.
- Mesh bake failure must not block metrics.
---

## Phase 3 — Spec only (not implemented here)

- Stream the skeleton (and later mesh-driven bones) to a UE5 Mannequin over Live Link during playback.
- Write BVH and/or FBX for offline use.
- Axis / unit conversion for Unreal is a separate design decision when Phase 3 is scheduled.

---

## Testing

**CI (no GPU / no weights):**

- Parser accepts sidecar pointers and rejects malformed mesh/params schema.
- Phase 1: overlay toggles do not crash when cache is absent (graceful empty).
- Phase 2: fixture with `joints3d` takes the 3D metric path; 17-point fixture stays on 2D; bridge is unused for 24-point.

**Manual (machine with GVHMR env):**

- Analyze → first play builds cache → second play reuses it.
- Wire and mesh both visible when both enabled; either can be off.
- 3D viewer orbits and follows playhead.
- Phase 2: charts show meter-based ranges; motif list still usable after retune.

---

## Non-goals (all phases in this doc)

- Replacing Qt with a web viewer
- Vendoring GVHMR or SMPL weights into git
- Crowds above 2 people
- Moving-camera / DPVO path (stay `staticCamera: true` unless a later design reopens it)
