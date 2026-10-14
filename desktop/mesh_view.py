"""World-space SMPL mesh viewer for the desktop app (PyVista when available)."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

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

# Source verts from GVHMR bake are Y-up (gravity axis = Y).
_UP_Y = "y_up"
_UP_Z = "z_up"


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


def apply_up_axis(verts: np.ndarray, up_axis: str) -> np.ndarray:
    """Map Y-up source verts to the chosen display up axis.

    ``y_up``: identity. ``z_up``: (x, y, z) -> (x, z, y) so former Y becomes Z.
    """
    v = np.asarray(verts, dtype=np.float32)
    if up_axis == _UP_Y:
        return v
    if up_axis != _UP_Z:
        raise ValueError(f"up_axis must be '{_UP_Y}' or '{_UP_Z}', got {up_axis!r}")
    out = np.empty_like(v)
    out[..., 0] = v[..., 0]
    out[..., 1] = v[..., 2]
    out[..., 2] = v[..., 1]
    return out


def _sequence_bounds(verts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned min/max over all frames and vertices. verts: (T, V, 3)."""
    flat = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    return flat.min(axis=0), flat.max(axis=0)


def _floor_height(verts_display: np.ndarray, up_axis: str) -> float:
    """Place floor slightly below the lowest point on the up axis."""
    axis = 1 if up_axis == _UP_Y else 2
    lo = float(np.min(verts_display[..., axis]))
    return lo - 0.02


class MeshWorldView(QWidget):
    """Orbitable 3D view of verts_world synced to playhead time."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._verts_src: np.ndarray | None = None  # Y-up source (T, V, 3)
        self._verts_disp: np.ndarray | None = None
        self._faces: np.ndarray | None = None
        self._times: np.ndarray | None = None
        self._frame_idx = 0
        self._plotter = None
        self._polydata = None
        self._mesh_actor = None
        self._floor_actor = None
        self._camera_set = False
        self._up_axis = _UP_Y

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        bar = QHBoxLayout()
        bar.setContentsMargins(6, 4, 6, 4)
        bar.addWidget(QLabel("Up axis"))
        self._up_combo = QComboBox(self)
        self._up_combo.addItem("Y up (GVHMR)", _UP_Y)
        self._up_combo.addItem("Z up", _UP_Z)
        self._up_combo.setToolTip(
            "GVHMR world mesh is Y-up. Switch to Z-up if your orbit habit expects Z."
        )
        self._up_combo.currentIndexChanged.connect(self._on_up_axis_changed)
        bar.addWidget(self._up_combo)
        bar.addStretch(1)
        layout.addLayout(bar)

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
        self._verts_src = v
        self._faces = np.asarray(faces, dtype=np.int64)
        self._frame_idx = 0
        self._camera_set = False
        self._rebuild_display(reset_camera=True)

    def set_time(self, t: float, times: np.ndarray) -> None:
        """Show the frame nearest to playhead time t."""
        if self._verts_disp is None:
            return
        self._times = np.asarray(times, dtype=np.float64)
        idx = _nearest_frame_index(t, self._times)
        if idx == self._frame_idx and self._mesh_actor is not None:
            return
        self._frame_idx = idx
        self._update_frame_points()

    def clear(self) -> None:
        self._verts_src = None
        self._verts_disp = None
        self._faces = None
        self._times = None
        self._frame_idx = 0
        self._camera_set = False
        self._polydata = None
        if self._plotter is None:
            return
        if self._mesh_actor is not None:
            self._plotter.remove_actor(self._mesh_actor)
            self._mesh_actor = None
        if self._floor_actor is not None:
            self._plotter.remove_actor(self._floor_actor)
            self._floor_actor = None
        self._plotter.render()

    def _on_up_axis_changed(self) -> None:
        self._up_axis = str(self._up_combo.currentData())
        if self._verts_src is None:
            return
        self._camera_set = False
        self._rebuild_display(reset_camera=True)

    def _rebuild_display(self, *, reset_camera: bool) -> None:
        if self._verts_src is None or self._faces is None:
            return
        self._verts_disp = apply_up_axis(self._verts_src, self._up_axis)
        self._ensure_floor()
        self._ensure_mesh_actor()
        self._update_frame_points()
        if reset_camera or not self._camera_set:
            self._frame_camera_to_sequence()
            self._camera_set = True
        if self._plotter is not None:
            self._plotter.render()

    def _ensure_mesh_actor(self) -> None:
        if self._plotter is None or self._verts_disp is None or self._faces is None:
            return
        if self._faces.size == 0 or self._faces.shape[0] == 0:
            return
        idx = min(self._frame_idx, self._verts_disp.shape[0] - 1)
        pts = self._verts_disp[idx]
        try:
            pv_faces = _faces_for_pyvista(self._faces)
            self._polydata = _pv.PolyData(pts.copy(), pv_faces)
            self._polydata.compute_normals(inplace=True)
        except Exception:
            self._polydata = None
            return

        if self._mesh_actor is not None:
            self._plotter.remove_actor(self._mesh_actor)
            self._mesh_actor = None

        self._mesh_actor = self._plotter.add_mesh(
            self._polydata,
            color="#c8c8c8",
            smooth_shading=True,
            specular=0.35,
            specular_power=15,
            show_edges=False,
            reset_camera=False,
        )

    def _update_frame_points(self) -> None:
        """Update vertex positions only — do not recreate actors (keeps camera stable)."""
        if (
            self._plotter is None
            or self._verts_disp is None
            or self._polydata is None
            or self._mesh_actor is None
        ):
            return
        idx = min(self._frame_idx, self._verts_disp.shape[0] - 1)
        pts = self._verts_disp[idx]
        try:
            self._polydata.points = pts
            # Normals from a static rest pose are fine for silhouette; skip recompute
            # every frame to avoid hitching. Soft shading still looks ok.
        except Exception:
            return
        self._plotter.render()

    def _ensure_floor(self) -> None:
        if self._plotter is None or self._verts_disp is None:
            return
        if self._floor_actor is not None:
            self._plotter.remove_actor(self._floor_actor)
            self._floor_actor = None

        lo, hi = _sequence_bounds(self._verts_disp)
        span = hi - lo
        pad = max(float(np.max(span)) * 0.35, 0.5)
        floor_h = _floor_height(self._verts_disp, self._up_axis)

        # Horizontal extents on the two non-up axes.
        if self._up_axis == _UP_Y:
            cx = 0.5 * (lo[0] + hi[0])
            cz = 0.5 * (lo[2] + hi[2])
            i_size = float(hi[0] - lo[0]) + 2 * pad
            j_size = float(hi[2] - lo[2]) + 2 * pad
            center = (cx, floor_h, cz)
            # Plane default is XY; rotate so normal is +Y.
            plane = _pv.Plane(
                center=center,
                direction=(0.0, 1.0, 0.0),
                i_size=i_size,
                j_size=j_size,
                i_resolution=12,
                j_resolution=12,
            )
            cam_up = (0.0, 1.0, 0.0)
        else:
            cx = 0.5 * (lo[0] + hi[0])
            cy = 0.5 * (lo[1] + hi[1])
            i_size = float(hi[0] - lo[0]) + 2 * pad
            j_size = float(hi[1] - lo[1]) + 2 * pad
            center = (cx, cy, floor_h)
            plane = _pv.Plane(
                center=center,
                direction=(0.0, 0.0, 1.0),
                i_size=i_size,
                j_size=j_size,
                i_resolution=12,
                j_resolution=12,
            )
            cam_up = (0.0, 0.0, 1.0)

        self._floor_actor = self._plotter.add_mesh(
            plane,
            color="#3a3a3a",
            opacity=0.85,
            show_edges=True,
            edge_color="#555555",
            reset_camera=False,
        )
        try:
            self._plotter.camera.up = cam_up
        except Exception:
            pass

    def _frame_camera_to_sequence(self) -> None:
        """Fit camera once to the full motion AABB so playback does not re-zoom."""
        if self._plotter is None or self._verts_disp is None:
            return
        lo, hi = _sequence_bounds(self._verts_disp)
        # Include floor pad in bounds so reset_camera has room.
        pad = max(float(np.max(hi - lo)) * 0.2, 0.3)
        lo = lo - pad
        hi = hi + pad
        try:
            self._plotter.reset_camera(bounds=(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
            # Generous clipping so walking toward/away from camera does not clip/pop zoom.
            cam = self._plotter.camera
            dist = float(np.linalg.norm(np.asarray(cam.position) - np.asarray(cam.focal_point)))
            cam.clipping_range = (max(dist * 0.01, 0.05), max(dist * 20.0, 50.0))
        except Exception:
            self._plotter.reset_camera()
