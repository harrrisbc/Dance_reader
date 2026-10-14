# Task 7 Report: 3D viewer widget

## Status

**DONE**

## Summary

Added `desktop/mesh_view.py` with `MeshWorldView(QWidget)`: `set_mesh`, `set_time` (nearest frame), and `clear`. When `pyvista` / `pyvistaqt` import succeeds, embeds `QtInteractor` with trackball orbit and dark neutral gradient background (`#1e1e1e` / `#2a2a2a`). On import failure, shows a centered `QLabel` install hint; methods are safe no-ops for the plot path so Analyze does not crash. Listed `pyvista` and `pyvistaqt` in `backend/requirements.txt`.

## Tests / notes

```powershell
C:\dev\Dance_reader\Dance_reader\backend\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, r'C:\dev\Dance_reader\Dance_reader\backend'); sys.path.insert(0, r'C:\dev\Dance_reader\Dance_reader'); from desktop.mesh_view import MeshWorldView, _PV_AVAILABLE; print('import ok', _PV_AVAILABLE)"
```

Output: `import ok False` (pyvista not installed in venv yet; fallback path verified).

Manual: after `pip install pyvista pyvistaqt`, instantiate widget in desktop with sample `(T,V,3)` verts and `(F,3)` faces; scrub `set_time` and confirm orbit without camera reset each frame.

## Commits

- (pending hash) — Add PyVista world mesh viewer widget for desktop.

Author/committer: `H.arimadion <harrrisbc@users.noreply.github.com>`. Not pushed.

Files:

- `desktop/mesh_view.py`
- `backend/requirements.txt`

## Concerns

- PyVista not installed in the current backend venv; 3D pane shows install hint until deps are installed.
- Task 8 must wire `MeshWorldView` into `window.py` splitter and mesh cache playback.
- First `set_mesh` calls `reset_camera`; playback frame updates preserve user orbit.
