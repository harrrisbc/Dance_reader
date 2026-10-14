"""World-space SMPL mesh viewer for the desktop app (PyVista when available)."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

_INSTALL_HINT = (
    "3D viewer requires pyvista and pyvistaqt in the app venv.\n"
    "Install:\n"
    r".\backend\.venv\Scripts\python.exe -m pip install pyvista pyvistaqt"
)

_PV_AVAILABLE = False
_pv = None
_QtInteractor = None
_PV_IMPORT_ERROR: str | None = None

try:
    import pyvista as _pv
    from pyvistaqt import QtInteractor as _QtInteractor

    _PV_AVAILABLE = True
except Exception as exc:  # ImportError, VTK binary mismatch, etc.
    _PV_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def _faces_for_pyvista(faces: np.ndarray) -> np.ndarray:
    """Convert (F, 3) triangle indices to PyVista face array."""
    f = np.asarray(faces, dtype=np.int64)
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"faces must be (F, 3), got {f.shape}")
    return np.hstack(
        [np.full((f.shape[0], 1), 3, dtype=np.int64), f]
    ).ravel()


def _nearest_frame_index(t: float, times: np.ndarray) -> int:
    ts = np.asarray(times, dtype=np.float64).ravel()
    if ts.size == 0:
        return 0
    return int(np.argmin(np.abs(ts - float(t))))


class MeshWorldView(QWidget):
    """Orbitable 3D view of verts_world synced to playhead time."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._verts: np.ndarray | None = None
        self._faces: np.ndarray | None = None
        self._times: np.ndarray | None = None
        self._frame_idx = 0
        self._plotter = None
        self._mesh_actor = None
        self._camera_set = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if not _PV_AVAILABLE:
            text = _INSTALL_HINT
            if _PV_IMPORT_ERROR:
                text = f"{_INSTALL_HINT}\n\n({_PV_IMPORT_ERROR})"
            hint = QLabel(text, self)
            hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
            hint.setWordWrap(True)
            hint.setStyleSheet("color: #aaa; padding: 12px;")
            layout.addWidget(hint)
            return

        self._plotter = _QtInteractor(self)
        self._plotter.set_background("#1e1e1e", top="#2a2a2a")
        self._plotter.enable_trackball_style()
        self._plotter.show_axes()
        layout.addWidget(self._plotter.interactor)

    def set_mesh(self, verts_world: np.ndarray, faces: np.ndarray) -> None:
        """Store sequence (T, V, 3) and shared triangle faces (F, 3)."""
        v = np.asarray(verts_world, dtype=np.float32)
        if v.ndim == 2:
            v = v[np.newaxis, ...]
        if v.ndim != 3 or v.shape[2] != 3:
            raise ValueError(f"verts_world must be (T, V, 3) or (V, 3), got {v.shape}")
        self._verts = v
        self._faces = np.asarray(faces, dtype=np.int64)
        self._frame_idx = 0
        self._camera_set = False
        self._refresh_mesh(reset_camera=True)

    def set_time(self, t: float, times: np.ndarray) -> None:
        """Show the frame nearest to playhead time t."""
        if self._verts is None:
            return
        self._times = np.asarray(times, dtype=np.float64)
        idx = _nearest_frame_index(t, self._times)
        if idx == self._frame_idx and self._mesh_actor is not None:
            return
        self._frame_idx = idx
        self._refresh_mesh()

    def clear(self) -> None:
        self._verts = None
        self._faces = None
        self._times = None
        self._frame_idx = 0
        self._camera_set = False
        if self._plotter is None:
            return
        if self._mesh_actor is not None:
            self._plotter.remove_actor(self._mesh_actor)
            self._mesh_actor = None
        self._plotter.render()

    def _refresh_mesh(self, *, reset_camera: bool = False) -> None:
        if self._plotter is None or self._verts is None or self._faces is None:
            return
        if self._faces.size == 0 or self._faces.shape[0] == 0:
            # Empty faces (old bake without SMPL/SMPLX) — skip to avoid VTK spam.
            return
        idx = min(self._frame_idx, self._verts.shape[0] - 1)
        pts = self._verts[idx]
        try:
            pv_faces = _faces_for_pyvista(self._faces)
            mesh = _pv.PolyData(pts, pv_faces)
            mesh.compute_normals(inplace=True)
        except Exception:
            return

        if self._mesh_actor is not None:
            self._plotter.remove_actor(self._mesh_actor)

        self._mesh_actor = self._plotter.add_mesh(
            mesh,
            color="#c8c8c8",
            smooth_shading=True,
            specular=0.35,
            specular_power=15,
            show_edges=False,
        )
        if reset_camera or not self._camera_set:
            self._plotter.reset_camera()
            self._camera_set = True
        self._plotter.render()
