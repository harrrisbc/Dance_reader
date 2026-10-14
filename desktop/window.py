from __future__ import annotations

import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.pipeline.metrics import (
    LEVEL_BAND_HIGH,
    LEVEL_BAND_LOW,
    LEVEL_CHART_YLIM,
    frame_aspect,
)
from app.pipeline.motifs import (
    LAST_DETECT_STATS,
    SUBSEQ_MATCH_MAX_DIST,
    find_similar_motif,
    format_tempo_label,
    query_segment_at_time,
)
from app.pipeline.pose_track import (
    DEFAULT_MAX_PEOPLE,
    DEFAULT_MODEL,
    DEFAULT_STRIDE,
    POSE_MODELS,
    TrackBuffer,
    resolve_pose_model,
)
from app.pipeline.run import analyze_video
from app.pipeline.mesh_cache import (
    load_mesh_cache,
    load_smpl_sidecar,
    mesh_cache_path_beside_sidecar,
    persist_mesh_cache_with_project,
    sha256_file,
)
from app.project_io import (
    DanceProject,
    buffers_from_analysis,
    load_project,
    resolve_mesh_cache_beside_project,
    resolve_smpl_sidecar_beside_project,
    save_project,
)
from app.schemas import AnalysisResult, Actor, MotifOccurrence, MotifType
from desktop.mesh_view import MeshWorldView
from desktop.playback import OverlayPose, VideoDecodeWorker

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
ARTIFACTS = BACKEND / "artifacts"
ZOOM_MIN_WINDOW_S = 2.0
PLAYHEAD_MIN_INTERVAL_S = 0.1  # ~10 Hz chart/floor updates while playing
TIME_READOUT_EMPTY = "--:--.-- / --:--.--"
MESH_DISABLED_TIP = "Re-analyze to enable mesh"


def format_timecode(seconds: float) -> str:
    """Minutes:seconds with two decimal places, e.g. 0:08.42."""
    s = max(0.0, float(seconds))
    minutes = int(s // 60)
    rem = s - minutes * 60
    return f"{minutes}:{rem:05.2f}"


def format_time_readout(current_s: float, duration_s: float) -> str:
    return f"{format_timecode(current_s)} / {format_timecode(duration_s)}"


# Chart series: attr, label, color, default_on, kind, ylabel, ylim
# kind: area | line | level | travel | ribbon | pos
CHART_SERIES = (
    ("energy", "Energy", "#4aa3ff", True, "area", "rel. peak", (0.0, 1.0)),
    ("energy_abs", "Energy (abs)", "#2f6fbf", False, "line", "torso²/s²", (0.0, 40.0)),
    ("speed", "Speed", "#f0a040", True, "line", "torso/s", (0.0, 7.0)),
    ("reach", "Reach", "#78b4a0", True, "reach", "× torso", (0.5, 4.0)),
    ("level", "Level", "#c45c4a", False, "level", "drop / torso", LEVEL_CHART_YLIM),
    ("travel", "Travel", "#e8c56e", False, "area", "torso/s", (0.0, 2.5)),
    ("directness", "Directness", "#a67c52", False, "ribbon", "", (0.0, 1.0)),
    ("pos_x", "L–R", "#8f7cc0", False, "pos", "screen", (0.0, 1.0)),
    ("pos_y", "F–B", "#c07fa8", False, "pos", "screen", (0.0, 1.0)),
)


def _configure_matplotlib() -> None:
    matplotlib.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "sans-serif"]
    matplotlib.rcParams["axes.unicode_minus"] = False


_configure_matplotlib()


class AnalyzeWorker(QThread):
    progress = Signal(float, str)
    finished_ok = Signal(object, object, object)  # AnalysisResult, keyed buffers, sidecar src
    finished_err = Signal(str)

    def __init__(
        self,
        video_path: Path,
        stride: int = DEFAULT_STRIDE,
        model_name: str = DEFAULT_MODEL,
        max_people: int = DEFAULT_MAX_PEOPLE,
    ) -> None:
        super().__init__()
        self.video_path = video_path
        self.stride = stride
        self.model_name = model_name
        self.max_people = max_people

    def run(self) -> None:
        try:
            job_id = f"desktop_{int(cv2.getTickCount())}"

            def cb(p: float, msg: str) -> None:
                self.progress.emit(p, msg)

            result, _overlay, buffers = analyze_video(
                self.video_path,
                stride=self.stride,
                model_name=self.model_name,
                max_people=self.max_people,
                write_overlay_video=False,
                artifacts_dir=ARTIFACTS,
                job_id=job_id,
                progress_cb=cb,
            )
            keyed = {f"track_{tid}": buf for tid, buf in buffers.items()}
            sidecar_src = None
            if result.smplSidecar:
                candidate = ARTIFACTS / job_id / result.smplSidecar
                if candidate.is_file():
                    sidecar_src = candidate
            self.finished_ok.emit(result, keyed, sidecar_src)
        except Exception as exc:
            self.finished_err.emit(str(exc))


class MeshBakeWorker(QThread):
    finished_ok = Signal(object)
    finished_err = Signal(str)

    def __init__(
        self,
        *,
        sidecar: Path,
        mesh_out: Path,
        video: Path | None,
    ) -> None:
        super().__init__()
        self._sidecar = sidecar
        self._mesh_out = mesh_out
        self._video = video

    def run(self) -> None:
        from app.pipeline.gvhmr_run import MeshBakeError, run_mesh_bake

        try:
            run_mesh_bake(
                smpl_sidecar=self._sidecar,
                mesh_out=self._mesh_out,
                video=self._video,
            )
            self.finished_ok.emit(self._mesh_out)
        except MeshBakeError as exc:
            self.finished_err.emit(str(exc))
        except Exception as exc:
            self.finished_err.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Dance Analyser")
        self.resize(1100, 720)

        self.video_path: Path | None = None
        self.fps = 30.0
        self.frame_count = 0
        self.playing = False
        self._display_frame = 0
        self.analysis: AnalysisResult | None = None
        self.track_buffers: dict[str, TrackBuffer] = {}
        self.worker: AnalyzeWorker | None = None
        self._smpl_sidecar_src: Path | None = None
        self._mesh_bake_worker: MeshBakeWorker | None = None
        self._mesh_bake_gen = 0
        self._mesh_bake_active_sidecar: Path | None = None
        self._mesh_times: np.ndarray | None = None
        self._status_before_bake = ""
        self._status_after_bake: str | None = None
        self.selected_type_id: str | None = None
        self.selected_occ_id: str | None = None
        self._sim_counter = 0
        self._playhead_lines: list = []
        self._chart_axes: list = []
        self._floor_playhead = None
        self._floor_ax = None
        self.view_start = 0.0
        self.view_end = 1.0
        self._duration_s = 1.0
        self._dirty = False
        self._project_path: Path | None = None
        self._last_playhead_wall = 0.0
        self._video_ready = False
        self._chart_bg = None

        self.decode = VideoDecodeWorker()
        self.decode.frame_ready.connect(self.on_decode_frame)
        self.decode.opened.connect(self.on_decode_opened)
        self.decode.ended.connect(self.on_decode_ended)
        self.decode.error.connect(self.on_decode_error)
        self.decode.start()

        self.video_label = QLabel("Open a local video to begin")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumHeight(360)
        self.video_label.setStyleSheet("background:#111; color:#ccc;")

        self.status = QLabel("Ready")
        self.actor_combo = QComboBox()
        self.actor_combo.currentIndexChanged.connect(self.refresh_chart)

        self.max_people_spin = QSpinBox()
        self.max_people_spin.setRange(1, 4)
        self.max_people_spin.setValue(DEFAULT_MAX_PEOPLE)
        self.max_people_spin.setToolTip("Max dancers to track during analysis (IDs may be reused after a drop)")

        self.model_combo = QComboBox()
        self.model_combo.addItem("S — fast (default)", POSE_MODELS["S"])
        self.model_combo.addItem("M — more accurate", POSE_MODELS["M"])
        self.model_combo.setCurrentIndex(0)
        self.model_combo.setToolTip(
            "M is slower but often better when fans or side views hide joints. "
            "Does not replace re-opening projects after the torso-scale fix."
        )
        self.model_label = QLabel("Model")
        self.model_combo.hide()
        self.model_label.hide()
        self._last_pose_model = DEFAULT_MODEL

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.sliderMoved.connect(self.on_seek)
        self.slider.sliderReleased.connect(self.on_seek_release)

        self.time_label = QLabel(TIME_READOUT_EMPTY)
        self.time_label.setObjectName("timeReadout")
        self.time_label.setToolTip("Current time / duration")
        mono = QFont("Consolas")
        if not mono.exactMatch():
            mono = QFont("Courier New")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(10)
        self.time_label.setFont(mono)
        self.time_label.setMinimumWidth(128)
        self.time_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        self.motif_list = QListWidget()
        self.motif_list.itemClicked.connect(self.on_motif_clicked)
        self.motif_list.itemDoubleClicked.connect(self.on_motif_rename)

        self.occ_list = QListWidget()
        self.occ_list.itemClicked.connect(self.on_occ_clicked)

        self.prev_occ_btn = QPushButton("Prev")
        self.prev_occ_btn.clicked.connect(lambda: self.step_occurrence(-1))
        self.next_occ_btn = QPushButton("Next")
        self.next_occ_btn.clicked.connect(lambda: self.step_occurrence(1))

        self.fig = Figure(figsize=(5, 3.5))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setMinimumHeight(220)
        self.canvas.mpl_connect("button_press_event", self.on_chart_click)
        self.canvas.mpl_connect("scroll_event", self.on_chart_scroll)
        # Keep figure inches in sync with widget pixels (HiDPI click accuracy)
        _orig_resize = self.canvas.resizeEvent

        def _canvas_resize(event) -> None:  # noqa: ANN001
            _orig_resize(event)
            self._sync_chart_figsize()
            if self._chart_axes:
                self.fig.subplots_adjust(
                    left=0.12, right=0.98, top=0.92, bottom=0.12, hspace=0.45
                )
                self._chart_bg = None
                self.canvas.draw_idle()

        self.canvas.resizeEvent = _canvas_resize  # type: ignore[method-assign]

        self.floor_fig = Figure(figsize=(5, 2.4))
        self.floor_canvas = FigureCanvasQTAgg(self.floor_fig)
        self.floor_canvas.setMinimumHeight(160)

        self.skel_style = QComboBox()
        self.skel_style.addItem("Skeleton", "skeleton")
        self.skel_style.addItem("Joints only", "joints")
        self.skel_style.addItem("Off", "off")
        self.skel_style.setToolTip(
            "Lines may cross when limbs overlap in the camera view."
        )
        self.skel_style.currentIndexChanged.connect(self.sync_overlay)

        self.mesh_overlay_cb = QCheckBox("Mesh overlay")
        self.mesh_overlay_cb.setToolTip(
            "Soft-rendered mesh video or projected mesh edges on playback."
        )
        self.mesh_overlay_cb.stateChanged.connect(self._on_mesh_overlay_toggle)

        self.mesh_viewer_cb = QCheckBox("3D viewer")
        self.mesh_viewer_cb.setToolTip(
            "Show orbitable world-space mesh beside the video."
        )
        self.mesh_viewer_cb.stateChanged.connect(self._on_mesh_viewer_toggle)

        self.mesh_view = MeshWorldView()
        self.mesh_view.setMinimumWidth(280)
        self.mesh_view.hide()

        self.floor_toggle = QCheckBox("Floor map")
        self.floor_toggle.setChecked(True)
        self.floor_toggle.setToolTip(
            "coco2d: screen coords. smpl3d: world XZ metres (pelvis solid, supporting foot dashed), "
            "origin = first pelvis. Static-cam / clip-relative, not surveyed stage GPS."
        )
        self.floor_toggle.stateChanged.connect(self._on_floor_toggle)

        open_btn = QPushButton("Open video")
        open_btn.clicked.connect(self.open_video)
        open_project_btn = QPushButton("Open project")
        open_project_btn.clicked.connect(self.open_project)
        save_project_btn = QPushButton("Save project")
        save_project_btn.clicked.connect(self.save_project_dialog)
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        analyze_btn = QPushButton("Analyze")
        analyze_btn.clicked.connect(self.start_analyze)
        self.find_similar_btn = QPushButton("Find similar")
        self.find_similar_btn.clicked.connect(self.find_similar)
        rename_btn = QPushButton("Rename type")
        rename_btn.clicked.connect(self.rename_selected_motif)

        zoom_in_btn = QPushButton("Zoom in")
        zoom_in_btn.clicked.connect(lambda: self.zoom_timeline(0.7))
        zoom_out_btn = QPushButton("Zoom out")
        zoom_out_btn.clicked.connect(lambda: self.zoom_timeline(1.4))
        zoom_reset_btn = QPushButton("Reset")
        zoom_reset_btn.clicked.connect(self.reset_timeline_zoom)

        top = QHBoxLayout()
        top.addWidget(open_btn)
        top.addWidget(open_project_btn)
        top.addWidget(save_project_btn)
        top.addWidget(self.play_btn)
        top.addWidget(analyze_btn)
        top.addWidget(self.find_similar_btn)
        top.addWidget(QLabel("Dancer"))
        top.addWidget(self.actor_combo)
        top.addWidget(QLabel("Overlay"))
        top.addWidget(self.skel_style)
        top.addWidget(self.mesh_overlay_cb)
        top.addWidget(self.mesh_viewer_cb)
        top.addWidget(QLabel("Max dancers"))
        top.addWidget(self.max_people_spin)
        top.addWidget(self.model_label)
        top.addWidget(self.model_combo)
        top.addWidget(rename_btn)
        top.addStretch(1)

        occ_nav = QHBoxLayout()
        occ_nav.addWidget(self.prev_occ_btn)
        occ_nav.addWidget(self.next_occ_btn)

        zoom_row = QHBoxLayout()
        zoom_row.addWidget(QLabel("Timeline"))
        zoom_row.addWidget(zoom_in_btn)
        zoom_row.addWidget(zoom_out_btn)
        zoom_row.addWidget(zoom_reset_btn)
        zoom_row.addStretch(1)

        series_row = QHBoxLayout()
        series_row.addWidget(QLabel("Curves"))
        self.series_checks: dict[str, QCheckBox] = {}
        for attr, label, _color, default_on, *_rest in CHART_SERIES:
            cb = QCheckBox(label)
            cb.setChecked(default_on)
            cb.stateChanged.connect(self.refresh_chart)
            self.series_checks[attr] = cb
            series_row.addWidget(cb)
        series_row.addStretch(1)

        right = QVBoxLayout()
        right.addWidget(QLabel("Motif types (double-click to rename)"))
        right.addWidget(self.motif_list, stretch=1)
        right.addWidget(QLabel("Occurrences (click to seek)"))
        right.addWidget(self.occ_list, stretch=1)
        right.addLayout(occ_nav)
        right.addLayout(zoom_row)
        right.addLayout(series_row)
        right.addWidget(QLabel("Click chart to seek · scroll to zoom"))
        right.addWidget(self.canvas, stretch=3)
        right.addWidget(self.floor_toggle)
        right.addWidget(self.floor_canvas, stretch=2)

        video_col = QVBoxLayout()
        video_col.addWidget(self.video_label, stretch=1)
        transport = QHBoxLayout()
        transport.addWidget(self.time_label)
        transport.addWidget(self.slider, stretch=1)
        video_col.addLayout(transport)
        video_host = QWidget()
        video_host.setLayout(video_col)

        self._video_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._video_splitter.addWidget(video_host)
        self._video_splitter.addWidget(self.mesh_view)
        self._video_splitter.setStretchFactor(0, 3)
        self._video_splitter.setStretchFactor(1, 2)
        self._video_splitter.setCollapsible(1, True)

        body = QHBoxLayout()
        body.addWidget(self._video_splitter, stretch=3)
        body.addLayout(right, stretch=2)

        root = QVBoxLayout()
        root.addLayout(top)
        root.addLayout(body, stretch=1)
        root.addWidget(self.status)

        w = QWidget()
        w.setLayout(root)
        self.setCentralWidget(w)

        self.actor_combo.currentIndexChanged.connect(self.sync_overlay)
        self.actor_combo.currentIndexChanged.connect(self._refresh_mesh_track_layer)
        self._sync_mesh_controls()

    def _resolve_sidecar_path(self) -> Path | None:
        if self._smpl_sidecar_src is not None and self._smpl_sidecar_src.is_file():
            return self._smpl_sidecar_src
        if self._project_path and self.analysis:
            found = resolve_smpl_sidecar_beside_project(
                self._project_path, self.analysis.smplSidecar
            )
            if found is not None:
                return found
            # Fallback: conventional name beside the project JSON.
            stem_sidecar = self._project_path.parent / f"{self._project_path.stem}.smpl.npz"
            if stem_sidecar.is_file():
                return stem_sidecar
        return None

    def _resolve_mesh_cache_path(self) -> Path | None:
        if not self.analysis or not self.analysis.meshCache:
            return None
        if self._project_path:
            beside = resolve_mesh_cache_beside_project(
                self._project_path, self.analysis.meshCache
            )
            if beside is not None:
                return beside
        sidecar = self._resolve_sidecar_path()
        if sidecar is not None:
            sibling = mesh_cache_path_beside_sidecar(sidecar)
            if sibling.is_file():
                return sibling
        return None

    def _mesh_cache_is_valid(self, path: Path) -> bool:
        if not path.is_file():
            return False
        try:
            mesh_data = load_mesh_cache(path)
            faces = np.asarray(mesh_data["faces"])
            if faces.size == 0 or int(faces.shape[0]) == 0:
                return False  # old bake without body-model faces — force re-bake
        except (OSError, ValueError, KeyError):
            return False
        expected = self.analysis.meshCacheSha256 if self.analysis else None
        if not expected:
            return True
        try:
            return sha256_file(path) == expected
        except OSError:
            return False

    def _mesh_track_index(self) -> int:
        actor = self.selected_actor()
        if actor is None:
            return 0
        if actor.id.startswith("track_"):
            try:
                return int(actor.id.split("_", 1)[1])
            except ValueError:
                pass
        return 0

    def _mesh_bake_out_path(self, sidecar: Path) -> Path:
        if self._project_path is not None:
            from app.pipeline.mesh_cache import mesh_cache_rel_name

            return self._project_path.parent / mesh_cache_rel_name(self._project_path)
        return mesh_cache_path_beside_sidecar(sidecar)

    def _sync_mesh_controls(self) -> None:
        has_sidecar = self._resolve_sidecar_path() is not None
        tip = (
            "Soft-rendered mesh video or projected mesh edges on playback."
            if has_sidecar
            else MESH_DISABLED_TIP
        )
        self.mesh_overlay_cb.setToolTip(tip)
        self.mesh_viewer_cb.setToolTip(
            "Show orbitable world-space mesh beside the video."
            if has_sidecar
            else MESH_DISABLED_TIP
        )
        self.mesh_overlay_cb.setEnabled(has_sidecar)
        self.mesh_viewer_cb.setEnabled(has_sidecar)
        if not has_sidecar:
            self.mesh_overlay_cb.blockSignals(True)
            self.mesh_overlay_cb.setChecked(False)
            self.mesh_overlay_cb.blockSignals(False)
            self.mesh_viewer_cb.blockSignals(True)
            self.mesh_viewer_cb.setChecked(False)
            self.mesh_viewer_cb.blockSignals(False)
            self.mesh_view.hide()
            self._mesh_times = None
            self.mesh_view.clear()
            self.decode.request("set_mesh_overlay", False)
            self.decode.request("set_mesh_layer", None)

    def _cancel_mesh_bake_worker(self) -> None:
        worker = self._mesh_bake_worker
        if worker is None:
            return
        try:
            worker.finished_ok.disconnect(self._on_mesh_bake_ok)
        except (RuntimeError, TypeError):
            pass
        try:
            worker.finished_err.disconnect(self._on_mesh_bake_err)
        except (RuntimeError, TypeError):
            pass
        self._mesh_bake_worker = None
        self._mesh_bake_active_sidecar = None
        self._mesh_bake_gen += 1

    def _mesh_bake_status_label(self, *, suffix: str = "") -> str:
        base = "Building mesh cache…"
        if suffix:
            return f"{base} ({suffix})"
        return base

    def _ensure_mesh_cache(self) -> bool:
        """Start background bake if needed. Returns True if a bake is running."""
        sidecar = self._resolve_sidecar_path()
        if sidecar is None or self.analysis is None:
            self._sync_mesh_controls()
            return False
        self._sync_mesh_controls()
        existing = self._resolve_mesh_cache_path()
        if existing is not None and self._mesh_cache_is_valid(existing):
            self._apply_mesh_from_path(existing)
            return False
        worker = self._mesh_bake_worker
        if worker is not None and worker.isRunning():
            if self._mesh_bake_active_sidecar == sidecar:
                return True
            self._cancel_mesh_bake_worker()
        mesh_out = self._mesh_bake_out_path(sidecar)
        self._status_before_bake = self.status.text()
        suffix = ""
        if self._project_path is not None:
            suffix = self._project_path.name
        self.status.setText(self._mesh_bake_status_label(suffix=suffix))
        bake_gen = self._mesh_bake_gen + 1
        self._mesh_bake_gen = bake_gen
        self._mesh_bake_active_sidecar = sidecar
        worker = MeshBakeWorker(
            sidecar=sidecar,
            mesh_out=mesh_out,
            video=self.video_path,
        )
        worker.finished_ok.connect(
            lambda path, g=bake_gen, sc=sidecar: self._on_mesh_bake_ok(path, g, sc)
        )
        worker.finished_err.connect(
            lambda msg, g=bake_gen, sc=sidecar: self._on_mesh_bake_err(msg, g, sc)
        )
        self._mesh_bake_worker = worker
        worker.start()
        return True

    def _on_mesh_bake_ok(
        self, mesh_path: object, bake_gen: int, sidecar: Path
    ) -> None:
        if bake_gen != self._mesh_bake_gen:
            return
        active = self._resolve_sidecar_path()
        if active is None or active.resolve() != sidecar.resolve():
            return
        if not isinstance(mesh_path, Path) or self.analysis is None:
            return
        try:
            digest = sha256_file(mesh_path)
        except OSError as exc:
            self._set_mesh_bake_status_error(str(exc))
            return
        rel = mesh_path.name
        resolved = mesh_path
        if self._project_path is not None:
            rel = persist_mesh_cache_with_project(self._project_path, mesh_path, digest)
            resolved = self._project_path.parent / rel
            self._persist_analysis_json()
        self.analysis.meshCache = rel
        self.analysis.meshCacheSha256 = digest
        self._dirty = True
        self._apply_mesh_from_path(resolved)
        after = self._status_after_bake
        if after:
            self.status.setText(after)
            self._status_after_bake = None
        else:
            prior = self._status_before_bake or ""
            self.status.setText(prior if prior else "Mesh cache ready")
        self._mesh_bake_active_sidecar = None
        if self._mesh_bake_worker is not None and not self._mesh_bake_worker.isRunning():
            self._mesh_bake_worker = None

    def _set_mesh_bake_status_error(self, msg: str) -> None:
        text = msg.strip()
        if text:
            line = text.splitlines()[-1].strip()
        else:
            line = "Mesh bake failed"
        if len(line) > 220:
            line = line[:217] + "..."
        self.status.setText(f"Mesh bake: {line}")

    def _on_mesh_bake_err(self, msg: str, bake_gen: int, sidecar: Path) -> None:
        if bake_gen != self._mesh_bake_gen:
            return
        active = self._resolve_sidecar_path()
        if active is None or active.resolve() != sidecar.resolve():
            return
        self._status_after_bake = None
        self._mesh_bake_active_sidecar = None
        if self._mesh_bake_worker is not None and not self._mesh_bake_worker.isRunning():
            self._mesh_bake_worker = None
        self._set_mesh_bake_status_error(msg)

    def _persist_analysis_json(self) -> None:
        if self._project_path is None or self.analysis is None or self.video_path is None:
            return
        project = DanceProject(
            video_path=str(self.video_path.resolve()),
            analysis=self.analysis,
            max_people=int(self.max_people_spin.value()),
            pose_model=self.selected_pose_model(),
        )
        save_project(self._project_path, project)

    def _apply_mesh_from_path(self, mesh_path: Path) -> None:
        sidecar = self._resolve_sidecar_path()
        if sidecar is None or not mesh_path.is_file():
            return
        try:
            mesh_data = load_mesh_cache(mesh_path)
            smpl = load_smpl_sidecar(sidecar)
        except (OSError, ValueError) as exc:
            self._set_mesh_bake_status_error(str(exc))
            return

        idx = self._mesh_track_index()
        n_tracks = int(mesh_data.get("n_tracks", 1))
        idx = max(0, min(idx, max(0, n_tracks - 1)))

        faces = np.asarray(mesh_data["faces"])
        times = np.asarray(mesh_data[f"times_{idx}"], dtype=np.float64)
        verts_world = np.asarray(mesh_data[f"verts_world_{idx}"])
        verts_incam = np.asarray(mesh_data[f"verts_incam_{idx}"])
        self._mesh_times = times

        if faces.size == 0 or int(faces.shape[0]) == 0:
            self._set_mesh_bake_status_error(
                "Mesh cache has no faces. Delete the .mesh.npz and re-open to re-bake "
                "(needs SMPLX_NEUTRAL.npz)."
            )
            return

        if self.mesh_viewer_cb.isChecked():
            self.mesh_view.set_mesh(verts_world, faces)
            self.mesh_view.set_time(self.current_time(), times)

        overlay_path: Path | None = None
        if "overlay_relpath" in mesh_data:
            rel = str(np.asarray(mesh_data["overlay_relpath"]).ravel()[0])
            candidate = mesh_path.parent / rel
            if candidate.is_file():
                overlay_path = candidate

        if overlay_path is not None:
            self.decode.request("set_mesh_layer", overlay_path)
        else:
            fw = int(self.analysis.frame_width or 0) if self.analysis else 0
            fh = int(self.analysis.frame_height or 0) if self.analysis else 0
            k_arr = np.asarray(smpl[f"K_{idx}"], dtype=np.float32)
            payload = {
                "times": times,
                "verts_incam": verts_incam,
                "faces": faces,
                "K": k_arr,
                "width": fw,
                "height": fh,
            }
            self.decode.request("set_mesh_layer", payload)

        if self.mesh_overlay_cb.isChecked():
            self.decode.request("set_mesh_overlay", True)

    def _refresh_mesh_track_layer(self) -> None:
        path = self._resolve_mesh_cache_path()
        if path is not None and self._mesh_cache_is_valid(path):
            self._apply_mesh_from_path(path)

    def _on_mesh_viewer_toggle(self) -> None:
        show = self.mesh_viewer_cb.isChecked()
        self.mesh_view.setVisible(show)
        if show:
            path = self._resolve_mesh_cache_path()
            if path is not None and self._mesh_cache_is_valid(path):
                self._apply_mesh_from_path(path)
            elif self._resolve_sidecar_path() is not None:
                self._ensure_mesh_cache()
        else:
            self.mesh_view.clear()

    def open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select video",
            "",
            "Video (*.mp4 *.mov *.avi *.mkv *.webm);;All (*.*)",
        )
        if not path:
            return
        self.load_video(Path(path))

    def load_video(self, path: Path, *, clear_analysis: bool = True) -> None:
        path = Path(path)
        # Re-opening the same video should keep Analyze / mesh (sidecar) state.
        same_video = (
            self.video_path is not None
            and path.resolve() == Path(self.video_path).resolve()
            and self.analysis is not None
        )
        if clear_analysis and same_video:
            clear_analysis = False
            self.status.setText(f"Reloaded video (analysis kept): {path.name}")

        self.video_path = path
        if clear_analysis:
            self.analysis = None
            self.track_buffers = {}
            self.selected_type_id = None
            self.selected_occ_id = None
            self._dirty = False
            self._project_path = None
            self.motif_list.clear()
            self.occ_list.clear()
            self.actor_combo.clear()
            self.fig.clear()
            self._chart_axes = []
            self._playhead_lines = []
            self.floor_fig.clear()
            self._floor_ax = None
            self._floor_playhead = None
            self.canvas.draw_idle()
            self.floor_canvas.draw_idle()
            self.sync_overlay()
            self._smpl_sidecar_src = None
            self._cancel_mesh_bake_worker()
            self._status_after_bake = None
            self._mesh_times = None
            self.mesh_view.clear()
            self.decode.request("set_mesh_layer", None)
            self.decode.request("set_mesh_overlay", False)
            self._sync_mesh_controls()
        self._display_frame = 0
        self._video_ready = False
        self.playing = False
        self.play_btn.setText("Play")
        if clear_analysis:
            self.status.setText(str(path.name))
        self._set_time_readout(None)
        self.decode.request("open", path)
        if not clear_analysis and self.analysis is not None:
            # Re-enable mesh controls after same-video reload / project open.
            self._sync_mesh_controls()
            if self._resolve_sidecar_path() is not None:
                self._ensure_mesh_cache()

    def on_decode_opened(self, fps: float, frame_count: int) -> None:
        self.fps = float(fps) or 30.0
        self.frame_count = int(frame_count)
        self._duration_s = self.frame_count / max(self.fps, 1e-6)
        self.view_start = 0.0
        self.view_end = max(self._duration_s, 1.0)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(0, self.frame_count - 1))
        self.slider.setValue(0)
        self._video_ready = True
        self._set_time_readout(0.0)
        self.sync_overlay()

    def on_decode_error(self, msg: str) -> None:
        self._video_ready = False
        self._set_time_readout(None)
        QMessageBox.warning(self, "Error", msg)

    def on_decode_ended(self) -> None:
        self.playing = False
        self.play_btn.setText("Play")
        self.update_playhead(self.current_time(), force=True)

    def on_decode_frame(
        self, qimg: object, frame_idx: int, t: float, playing: bool
    ) -> None:
        if not isinstance(qimg, QImage):
            return
        self._display_frame = int(frame_idx)
        if not playing and self.playing:
            self.playing = False
            self.play_btn.setText("Play")
        xform = (
            Qt.TransformationMode.FastTransformation
            if self.playing
            else Qt.TransformationMode.SmoothTransformation
        )
        pix = QPixmap.fromImage(qimg).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            xform,
        )
        self.video_label.setPixmap(pix)
        self.slider.blockSignals(True)
        self.slider.setValue(int(frame_idx))
        self.slider.blockSignals(False)
        self.update_playhead(float(t), force=not self.playing)

    def current_time(self) -> float:
        if not self._video_ready:
            return 0.0
        return self._display_frame / max(self.fps, 1e-6)

    def _set_time_readout(self, current_s: float | None) -> None:
        if current_s is None or not self._video_ready:
            self.time_label.setText(TIME_READOUT_EMPTY)
            return
        dur = float(getattr(self, "_duration_s", 0.0) or 0.0)
        self.time_label.setText(format_time_readout(current_s, dur))

    def seek_time(self, t: float) -> None:
        if not self._video_ready or self.frame_count <= 0:
            return
        frame = int(round(t * self.fps))
        frame = max(0, min(frame, self.frame_count - 1))
        self.playing = False
        self.play_btn.setText("Play")
        self.decode.request("seek", frame)

    def sync_overlay(self) -> None:
        """Push selected-actor poses to the decode worker (selected only)."""
        poses: list[OverlayPose] = []
        actor = self.selected_actor()
        if actor and actor.poses:
            poses = [
                OverlayPose(time=float(p.time), keypoints=p.keypoints) for p in actor.poses
            ]
        style = self.skel_style.currentData()
        if style is None:
            style = "skeleton"
        self.decode.request("set_overlay", (poses, str(style)))

    def _sync_chart_figsize(self) -> None:
        w = max(1, int(self.canvas.width()))
        h = max(1, int(self.canvas.height()))
        dpi = float(self.fig.get_dpi() or 100.0)
        self.fig.set_size_inches(w / dpi, h / dpi, forward=False)

    def toggle_play(self) -> None:
        if not self._video_ready:
            return
        if self.playing:
            self.playing = False
            self.play_btn.setText("Play")
            self.decode.request("pause", self._display_frame)
        else:
            self.playing = True
            self.play_btn.setText("Pause")
            self._last_playhead_wall = 0.0
            self.decode.request("play", self._display_frame)

    def on_seek(self, value: int) -> None:
        self.playing = False
        self.play_btn.setText("Play")
        self.decode.request("seek", int(value))

    def on_seek_release(self) -> None:
        self.decode.request("seek", int(self.slider.value()))

    def selected_pose_model(self) -> str:
        data = self.model_combo.currentData()
        return resolve_pose_model(str(data) if data else DEFAULT_MODEL)

    def set_pose_model_combo(self, model: str | None) -> None:
        resolved = resolve_pose_model(model)
        for i in range(self.model_combo.count()):
            if self.model_combo.itemData(i) == resolved:
                self.model_combo.setCurrentIndex(i)
                return
        self.model_combo.setCurrentIndex(0)

    def start_analyze(self) -> None:
        if self.video_path is None:
            QMessageBox.information(self, "Note", "Open a video first")
            return
        if self.worker and self.worker.isRunning():
            return
        self._cancel_mesh_bake_worker()
        self._status_after_bake = None
        n = int(self.max_people_spin.value())
        model_name = self.selected_pose_model()
        self._last_pose_model = model_name
        self.status.setText("Running GVHMR…")
        self.worker = AnalyzeWorker(
            self.video_path, max_people=n, model_name=model_name
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.finished_ok.connect(self.on_analyze_done)
        self.worker.finished_err.connect(self.on_analyze_err)
        self.worker.start()

    def on_progress(self, p: float, msg: str) -> None:
        self.status.setText(f"{msg}  {int(p * 100)}%")

    def on_analyze_done(self, result: object, keyed: object, sidecar_src: object = None) -> None:
        assert isinstance(result, AnalysisResult)
        self.analysis = result
        self.track_buffers = keyed if isinstance(keyed, dict) else {}
        self._smpl_sidecar_src = sidecar_src if isinstance(sidecar_src, Path) else None
        self._dirty = True
        self._project_path = None
        self.apply_analysis_to_ui(seek_first_motif=True)
        n_types = len(result.motifTypes)
        diag = result.motifDiag or LAST_DETECT_STATS.summary()
        model_bit = Path(self._last_pose_model).stem
        if not result.actors:
            status = "No person was tracked."
        elif n_types == 0:
            status = f"Done: no motif types ({model_bit}) — {diag}"
        else:
            status = (
                f"Done: {len(result.actors)} dancers, {n_types} motif types "
                f"({model_bit}) — unsaved — {diag}"
            )
        if result.ignoredTracks > 0:
            status += f" · ignored {result.ignoredTracks} extra tracks"
        self.status.setText(status)
        self.sync_overlay()
        self._ensure_mesh_cache()
        self.decode.request("seek", int(self.slider.value()))

    def selected_actor(self) -> Actor | None:
        if not self.analysis or self.actor_combo.count() == 0:
            return None
        idx = self.actor_combo.currentIndex()
        if idx < 0 or idx >= len(self.analysis.actors):
            return None
        return self.analysis.actors[idx]

    def selected_motif_type(self) -> MotifType | None:
        if not self.analysis or not self.selected_type_id:
            return None
        return next(
            (m for m in self.analysis.motifTypes if m.id == self.selected_type_id),
            None,
        )

    def apply_analysis_to_ui(self, *, seek_first_motif: bool = False) -> None:
        if not self.analysis:
            return
        result = self.analysis
        self.selected_type_id = None
        self.selected_occ_id = None
        self._duration_s = max(float(result.duration), self._duration_s, 1.0)
        self.view_start = 0.0
        self.view_end = self._duration_s
        self.actor_combo.blockSignals(True)
        self.actor_combo.clear()
        for a in result.actors:
            self.actor_combo.addItem(a.label, a.id)
        self.actor_combo.blockSignals(False)
        self.motif_list.clear()
        self.occ_list.clear()
        for m in result.motifTypes:
            item = QListWidgetItem(self._motif_type_label(m))
            item.setData(Qt.ItemDataRole.UserRole, m.id)
            self.motif_list.addItem(item)
        if result.motifTypes:
            self.motif_list.setCurrentRow(0)
            self.select_motif_type(
                result.motifTypes[0].id,
                seek_first=seek_first_motif,
            )
        else:
            self.refresh_chart()
        self.sync_overlay()
        self._set_time_readout(self.current_time() if self._video_ready else None)

    @staticmethod
    def _motif_type_label(m: MotifType) -> str:
        label = f"{m.name}  ·  {len(m.occurrences)}×"
        if m.meanIntraDist is not None:
            label += f"  ·  d̄={m.meanIntraDist:.2f}"
        return label

    def save_project_dialog(self) -> None:
        if not self.analysis or self.video_path is None:
            QMessageBox.information(self, "Note", "Open a video and finish analysis before saving a project.")
            return
        suggested = self._project_path or self.video_path.with_suffix(".dance.json")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save project",
            str(suggested),
            "Dance project (*.dance.json);;JSON (*.json)",
        )
        if not path:
            return
        out = Path(path)
        if not str(out).endswith(".dance.json") and out.suffix.lower() != ".json":
            out = out.with_suffix(".dance.json")
        try:
            project = DanceProject(
                video_path=str(self.video_path.resolve()),
                analysis=self.analysis,
                max_people=int(self.max_people_spin.value()),
                pose_model=self.selected_pose_model(),
            )
            mesh_src = self._resolve_mesh_cache_path()
            save_project(
                out,
                project,
                smpl_sidecar_src=self._smpl_sidecar_src,
                mesh_cache_src=mesh_src,
            )
            self.analysis = project.analysis
            self._smpl_sidecar_src = resolve_smpl_sidecar_beside_project(
                out, project.analysis.smplSidecar
            )
            self._project_path = out
            self._dirty = False
            self.status.setText(f"Saved: {out.name}")
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open project",
            "",
            "Dance project (*.dance.json);;JSON (*.json);;All (*.*)",
        )
        if not path:
            return
        try:
            project = load_project(Path(path))
        except Exception as exc:
            QMessageBox.critical(self, "Open failed", str(exc))
            return

        video = Path(project.video_path)
        if not video.is_file():
            QMessageBox.warning(
                self,
                "Video not found",
                f"Project video path does not exist:\n{project.video_path}\n\nPlease select the video file.",
            )
            picked, _ = QFileDialog.getOpenFileName(
                self,
                "Select project video",
                "",
                "Video (*.mp4 *.mov *.avi *.mkv *.webm);;All (*.*)",
            )
            if not picked:
                return
            video = Path(picked)

        if project.max_people:
            self.max_people_spin.setValue(max(1, min(4, int(project.max_people))))
        self.set_pose_model_combo(project.pose_model)
        if project.pose_model:
            self._last_pose_model = resolve_pose_model(project.pose_model)

        project_json = Path(path)
        self._cancel_mesh_bake_worker()
        self.analysis = project.analysis
        self.track_buffers = buffers_from_analysis(project.analysis)
        self._project_path = project_json
        self._smpl_sidecar_src = resolve_smpl_sidecar_beside_project(
            project_json, project.analysis.smplSidecar
        )
        self._dirty = False
        self.load_video(video, clear_analysis=False)
        self.apply_analysis_to_ui(seek_first_motif=False)
        self.sync_overlay()
        loaded_msg = (
            f"Loaded project: {Path(path).name} ({len(project.analysis.actors)} dancers, "
            f"{len(project.analysis.motifTypes)} types) — no re-analysis needed"
        )
        self._status_after_bake = loaded_msg
        baking = self._ensure_mesh_cache()
        if not baking:
            self._status_after_bake = None
            self.status.setText(loaded_msg)

    def find_similar(self) -> None:
        if not self.analysis or not self.track_buffers:
            QMessageBox.information(self, "Note", "Finish analysis first")
            return
        actor = self.selected_actor()
        if actor is None:
            QMessageBox.information(self, "Note", "Select a dancer")
            return

        query_time = self.current_time()
        q_start: float | None = None
        q_end: float | None = None
        mtype = self.selected_motif_type()
        if mtype and self.selected_occ_id:
            occ = next((o for o in mtype.occurrences if o.id == self.selected_occ_id), None)
            if occ:
                query_time = (occ.start + occ.end) / 2
                q_start, q_end = float(occ.start), float(occ.end)
                actor_id = occ.actorId
            else:
                actor_id = actor.id
        else:
            actor_id = actor.id

        aspect = frame_aspect(
            int(self.analysis.frame_width or 0),
            int(self.analysis.frame_height or 0),
        )
        query = query_segment_at_time(
            self.track_buffers,
            actor_id,
            query_time,
            start=q_start,
            end=q_end,
            aspect=aspect,
        )
        if query is None:
            QMessageBox.warning(self, "Find similar", "Not enough motion here — seek to a clearer move and try again.")
            return

        self._sim_counter += 1
        n = self._sim_counter
        found = find_similar_motif(
            self.track_buffers,
            query,
            name=f"Similar {n}",
            type_id=f"type_sim_{n}",
            color="#78b4a0" if n % 2 else "#a67c52",
            aspect=aspect,
        )
        if found is None or not found.occurrences:
            QMessageBox.information(
                self,
                "Find similar",
                f"No similar moves found (need DTW distance ≤ {SUBSEQ_MATCH_MAX_DIST:.2f} "
                f"besides the query).",
            )
            return

        remaining = [m for m in self.analysis.motifTypes if m.id != found.id]
        remaining.append(found)
        self.analysis.motifTypes = remaining

        self.motif_list.clear()
        for m in self.analysis.motifTypes:
            item = QListWidgetItem(self._motif_type_label(m))
            item.setData(Qt.ItemDataRole.UserRole, m.id)
            self.motif_list.addItem(item)
        for row in range(self.motif_list.count()):
            if self.motif_list.item(row).data(Qt.ItemDataRole.UserRole) == found.id:
                self.motif_list.setCurrentRow(row)
                break
        self.select_motif_type(found.id, seek_first=True)
        self._dirty = True
        best = found.occurrences[0].score
        best_txt = f", best {best * 100:.0f}%" if best is not None else ""
        self.status.setText(
            f"{found.name}: {len(found.occurrences)} hits{best_txt} (unsaved)"
        )

    def on_analyze_err(self, err: str) -> None:
        self.status.setText("Analysis failed")
        QMessageBox.critical(self, "Analysis failed", err)

    def reset_timeline_zoom(self) -> None:
        self.view_start = 0.0
        self.view_end = max(self._duration_s, 1.0)
        self.refresh_chart()

    def zoom_timeline(self, factor: float) -> None:
        dur = max(self._duration_s, 1.0)
        center = self.current_time()
        if center < self.view_start or center > self.view_end:
            center = 0.5 * (self.view_start + self.view_end)
        width = max(self.view_end - self.view_start, ZOOM_MIN_WINDOW_S)
        new_w = max(ZOOM_MIN_WINDOW_S, min(dur, width * factor))
        start = center - new_w * 0.5
        end = center + new_w * 0.5
        if start < 0:
            end -= start
            start = 0.0
        if end > dur:
            start -= end - dur
            end = dur
        self.view_start = max(0.0, start)
        self.view_end = min(dur, end)
        if self.view_end - self.view_start < ZOOM_MIN_WINDOW_S:
            self.view_end = min(dur, self.view_start + ZOOM_MIN_WINDOW_S)
        self.refresh_chart()

    def on_chart_scroll(self, event) -> None:
        if event.inaxes is None:
            return
        # scroll up → zoom in
        if event.step > 0:
            self.zoom_timeline(0.8)
        elif event.step < 0:
            self.zoom_timeline(1.25)

    def _on_mesh_overlay_toggle(self) -> None:
        checked = self.mesh_overlay_cb.isChecked()
        self.decode.request("set_mesh_overlay", checked)
        if not checked:
            return
        path = self._resolve_mesh_cache_path()
        if path is not None and self._mesh_cache_is_valid(path):
            self._apply_mesh_from_path(path)
        elif self._resolve_sidecar_path() is not None:
            self._ensure_mesh_cache()

    def _on_floor_toggle(self) -> None:
        self.floor_canvas.setVisible(self.floor_toggle.isChecked())
        if self.floor_toggle.isChecked():
            self.refresh_floor_map()

    def _enabled_series(self):
        out = []
        for row in CHART_SERIES:
            attr = row[0]
            cb = self.series_checks.get(attr)
            if cb is not None and cb.isChecked():
                out.append(row)
        return out

    def _draw_motif_spans(self, ax, actor_id: str) -> None:
        if not self.analysis:
            return
        for mtype in self.analysis.motifTypes:
            for occ in mtype.occurrences:
                if occ.actorId != actor_id:
                    continue
                if occ.end < self.view_start or occ.start > self.view_end:
                    continue
                alpha = 0.45 if occ.id == self.selected_occ_id else 0.22
                ax.axvspan(occ.start, occ.end, color=mtype.color, alpha=alpha)

    def _plot_series_row(self, ax, times, ys, *, kind: str, color: str, ylabel: str, ylim):
        y0, y1 = ylim
        # Expand only if data exceeds locked range
        finite = [v for v in ys if v == v]  # drop NaN
        if finite:
            data_max = max(finite)
            data_min = min(finite)
            if data_max > y1:
                y1 = data_max * 1.05
            if data_min < y0 and kind != "reach":
                y0 = min(y0, data_min)

        if kind == "area":
            ax.fill_between(times, ys, color=color, alpha=0.35, linewidth=0)
            ax.plot(times, ys, color=color, linewidth=1.2)
        elif kind == "reach":
            ax.plot(times, ys, color=color, linewidth=1.3)
            ax.axhline(1.2, color="#999999", linewidth=0.7, linestyle="--", alpha=0.7)
            ax.axhline(2.5, color="#999999", linewidth=0.7, linestyle="--", alpha=0.7)
            ax.text(
                self.view_end,
                1.2,
                " closed",
                va="bottom",
                ha="right",
                fontsize=7,
                color="#666666",
                clip_on=False,
            )
            ax.text(
                self.view_end,
                2.5,
                " open",
                va="bottom",
                ha="right",
                fontsize=7,
                color="#666666",
                clip_on=False,
            )
        elif kind == "level":
            ax.axhspan(y0, LEVEL_BAND_LOW, color="#c45c4a", alpha=0.10, zorder=0)
            ax.axhspan(LEVEL_BAND_LOW, LEVEL_BAND_HIGH, color="#e8c56e", alpha=0.10, zorder=0)
            ax.axhspan(LEVEL_BAND_HIGH, y1, color="#78b4a0", alpha=0.10, zorder=0)
            ax.plot(times, ys, color=color, linewidth=1.3)
            ax.text(self.view_end, LEVEL_BAND_LOW, " low", va="bottom", ha="right", fontsize=7, color="#666666")
            ax.text(self.view_end, LEVEL_BAND_HIGH, " high", va="bottom", ha="right", fontsize=7, color="#666666")
        elif kind == "ribbon":
            # Color by directness; gaps (NaN) where dancer is still
            ax.set_yticks([])
            for i in range(1, len(times)):
                a, b = ys[i - 1], ys[i]
                if a != a or b != b:  # NaN
                    continue
                # 1 = straight (cool), 0 = winding (warm)
                t = float(np.clip(0.5 * (a + b), 0.0, 1.0))
                r = int(196 + (80 - 196) * t)
                g = int(92 + (180 - 92) * t)
                bcol = int(74 + (160 - 74) * t)
                ax.axvspan(
                    times[i - 1],
                    times[i],
                    ymin=0.25,
                    ymax=0.75,
                    color=f"#{r:02x}{g:02x}{bcol:02x}",
                    alpha=0.85,
                    linewidth=0,
                )
            ax.set_ylim(0, 1)
            return
        elif kind in ("line", "pos"):
            ax.plot(times, ys, color=color, linewidth=1.3)
        else:
            ax.plot(times, ys, color=color, linewidth=1.3)

        ax.set_ylim(y0, y1)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=8)

    def refresh_chart(self) -> None:
        self.fig.clear()
        self._playhead_lines = []
        self._chart_axes = []
        self._chart_bg = None
        actor = self.selected_actor()
        enabled = self._enabled_series()
        n = len(enabled)

        if not actor or not actor.metrics:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "No analysis data", ha="center", va="center")
            ax.set_axis_off()
            self.canvas.draw_idle()
            self.refresh_floor_map()
            return

        if n == 0:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "Select at least one curve", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            self.canvas.draw_idle()
            self.refresh_floor_map()
            return

        # Match figure inches to canvas widget pixels (HiDPI-safe hit testing)
        self._sync_chart_figsize()
        axes = self.fig.subplots(n, 1, sharex=True, squeeze=False)
        axes = [axes[i, 0] for i in range(n)]
        self._chart_axes = axes

        times = [m.time for m in actor.metrics]
        t_now = self.current_time()

        space = getattr(self.analysis, "metricsSpace", "coco2d") if self.analysis else "coco2d"
        for i, (attr, label, color, _on, kind, ylabel, ylim) in enumerate(enabled):
            ax = axes[i]
            ys = [float(getattr(m, attr, 0.0)) for m in actor.metrics]
            row_ylabel, row_ylim = ylabel, ylim
            if space == "smpl3d" and attr in ("pos_x", "pos_y"):
                row_ylabel = "m (world)"
                if ys:
                    lo, hi = min(ys), max(ys)
                    pad = max(0.1 * (hi - lo), 0.2)
                    row_ylim = (lo - pad, hi + pad)
            self._plot_series_row(
                ax,
                times,
                ys,
                kind=kind,
                color=color,
                ylabel=row_ylabel or label,
                ylim=row_ylim,
            )
            ax.set_xlim(self.view_start, self.view_end)
            ax.tick_params(axis="y", labelsize=7)
            # Row title on the left
            ax.set_title(label, loc="left", fontsize=8, pad=2)
            self._draw_motif_spans(ax, actor.id)
            ph = ax.axvline(t_now, color="#222222", linewidth=1.4, zorder=5, animated=True)
            self._playhead_lines.append(ph)
            if i < n - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("Time (s)", fontsize=8)

        self.fig.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.12, hspace=0.45)
        self.canvas.draw()  # full bake — capture background for playhead blit
        try:
            self._chart_bg = self.canvas.copy_from_bbox(self.fig.bbox)
            for line in self._playhead_lines:
                line.set_animated(True)
                line.axes.draw_artist(line)
            self.canvas.blit(self.fig.bbox)
        except Exception:
            self._chart_bg = None
            self.canvas.draw_idle()
        self.refresh_floor_map()

    def refresh_floor_map(self) -> None:
        self.floor_fig.clear()
        self._floor_playhead = None
        self._floor_ax = None
        if not self.floor_toggle.isChecked():
            self.floor_canvas.draw_idle()
            return

        ax = self.floor_fig.add_subplot(111)
        self._floor_ax = ax
        actor = self.selected_actor()
        if not actor or not actor.metrics:
            ax.text(0.5, 0.5, "No floor-map data", ha="center", va="center")
            ax.set_axis_off()
            self.floor_canvas.draw_idle()
            return

        space = (
            getattr(self.analysis, "metricsSpace", "coco2d") if self.analysis else "coco2d"
        )
        world = space == "smpl3d"

        # Points inside current zoom window
        pts = [
            (m.pos_x, m.pos_y, m.time, m.pos_foot_x, m.pos_foot_y)
            for m in actor.metrics
            if self.view_start <= m.time <= self.view_end
        ]
        if len(pts) < 1:
            ax.text(0.5, 0.5, "No points in this time window", ha="center", va="center")
            ax.set_axis_off()
            self.floor_canvas.draw_idle()
            return

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        fxs = [p[3] for p in pts]
        fys = [p[4] for p in pts]
        ts = [p[2] for p in pts]
        t0, t1 = ts[0], ts[-1]
        span = max(t1 - t0, 1e-6)

        # Time-colored pelvis path (solid)
        for i in range(1, len(pts)):
            u = ((ts[i - 1] + ts[i]) * 0.5 - t0) / span
            g = int(216 + (51 - 216) * u)
            col = f"#{g:02x}{g:02x}{g:02x}"
            ax.plot(
                [xs[i - 1], xs[i]],
                [ys[i - 1], ys[i]],
                color=col,
                linewidth=2.0,
                solid_capstyle="round",
                linestyle="-",
            )
        if world:
            ax.plot(
                fxs,
                fys,
                color="#888888",
                linewidth=1.2,
                linestyle="--",
            )

        t_now = self.current_time()
        best = min(pts, key=lambda p: abs(p[2] - t_now))
        self._floor_playhead = ax.scatter(
            [best[0]],
            [best[1]],
            c="#c45c4a",
            s=48,
            zorder=5,
            edgecolors="white",
            linewidths=0.6,
        )

        if world:
            # Extra X pad so a mostly N–S path is not a hairline strip under labels.
            pad_y = 0.12
            all_x = xs + fxs
            all_y = ys + fys
            xmin, xmax = min(all_x), max(all_x)
            ymin, ymax = min(all_y), max(all_y)
            span_x = max(xmax - xmin, 1e-3)
            span_y = max(ymax - ymin, 1e-3)
            # Match at least ~35% of Z span on X so the panel stays readable.
            min_x_span = max(0.35 * span_y, 0.8)
            if span_x < min_x_span:
                mid = 0.5 * (xmin + xmax)
                xmin, xmax = mid - 0.5 * min_x_span, mid + 0.5 * min_x_span
                span_x = min_x_span
            ax.set_xlim(xmin - 0.08 * span_x, xmax + 0.08 * span_x)
            ax.set_ylim(ymin - pad_y * span_y, ymax + pad_y * span_y)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xlabel("world X (m)", fontsize=8)
            ax.set_ylabel("world Z (m)", fontsize=8)
            # Key in title — legend boxes cover narrow equal-aspect trails.
            ax.set_title(
                "Floor map — solid pelvis · dashed foot (m, static cam)",
                fontsize=8,
            )
            ax.tick_params(labelsize=7)
        else:
            fw = int(self.analysis.frame_width) if self.analysis else 0
            fh = int(self.analysis.frame_height) if self.analysis else 0
            if fw <= 0 or fh <= 0:
                fw, fh = 16, 9
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.invert_yaxis()
            ax.set_box_aspect(fh / float(fw))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("Floor map (screen coords)", fontsize=9)

        for spine in ax.spines.values():
            spine.set_color("#aaaaaa")
        self.floor_fig.tight_layout()
        self.floor_canvas.draw_idle()

    def update_playhead(self, t: float, *, force: bool = False) -> None:
        # Cheap; update every call so the clock stays accurate even when chart/floor throttle
        self._set_time_readout(float(t) if self._video_ready else None)

        if (
            self.mesh_viewer_cb.isChecked()
            and self._mesh_times is not None
            and self._mesh_times.size > 0
        ):
            self.mesh_view.set_time(float(t), self._mesh_times)

        if self.playing and not force:
            now = time.perf_counter()
            if now - self._last_playhead_wall < PLAYHEAD_MIN_INTERVAL_S:
                return
            self._last_playhead_wall = now

        if self._playhead_lines:
            for line in self._playhead_lines:
                line.set_xdata([t, t])
            # While playing: blit over baked background (no full replot)
            if self.playing and self._chart_bg is not None and not force:
                try:
                    self.canvas.restore_region(self._chart_bg)
                    for line in self._playhead_lines:
                        line.axes.draw_artist(line)
                    self.canvas.blit(self.fig.bbox)
                except Exception:
                    self.canvas.draw_idle()
            else:
                self.canvas.draw_idle()
                # Re-bake after forced/paused redraw
                if not self.playing:
                    try:
                        self.canvas.draw()
                        self._chart_bg = self.canvas.copy_from_bbox(self.fig.bbox)
                    except Exception:
                        self._chart_bg = None

        # Floor map: same ~10 Hz throttle as chart (top of this method); cheap offsets only
        if self._floor_ax is not None and self.floor_toggle.isChecked():
            actor = self.selected_actor()
            if actor and actor.metrics:
                pts = [
                    (m.pos_x, m.pos_y, m.time)
                    for m in actor.metrics
                    if self.view_start <= m.time <= self.view_end
                ]
                if pts:
                    best = min(pts, key=lambda p: abs(p[2] - t))
                    if self._floor_playhead is not None:
                        self._floor_playhead.set_offsets([[best[0], best[1]]])
                        self.floor_canvas.draw_idle()

    def on_chart_click(self, event) -> None:
        if not self._video_ready or self.frame_count <= 0:
            return
        if event.x is None or event.y is None or not self._chart_axes:
            return

        t: float | None = None
        # Prefer display→data mapping so HiDPI / margin clicks stay aligned
        for ax in self._chart_axes:
            bbox = ax.get_window_extent()
            if event.y < bbox.y0 or event.y > bbox.y1:
                continue
            try:
                xdata, _ydata = ax.transData.inverted().transform((event.x, event.y))
            except Exception:
                continue
            x0, x1 = ax.get_xlim()
            # Clamp to visible window (allows clicks in the row's y-label gutter)
            t = float(min(max(xdata, min(x0, x1)), max(x0, x1)))
            break

        if t is None and event.inaxes in self._chart_axes and event.xdata is not None:
            t = float(event.xdata)
        if t is None:
            return

        self.playing = False
        self.play_btn.setText("Play")
        self.seek_time(t)
        self.status.setText(f"Seek {t:.2f}s")

    def fill_occurrence_list(self, mtype: MotifType) -> None:
        self.occ_list.clear()
        for i, occ in enumerate(mtype.occurrences, start=1):
            if getattr(occ, "is_query", False):
                label = f"#{i}  {occ.start:.1f}s–{occ.end:.1f}s  ·  query"
            else:
                bits = [f"#{i}  {occ.start:.1f}s–{occ.end:.1f}s"]
                if occ.score is not None:
                    bits.append(f"{occ.score * 100:.0f}%")
                tempo = format_tempo_label(getattr(occ, "tempo_ratio", None))
                if tempo:
                    bits.append(tempo)
                label = "  ·  ".join(bits)
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, occ.id)
            self.occ_list.addItem(item)

    def select_motif_type(self, type_id: str, *, seek_first: bool = True) -> None:
        if not self.analysis:
            return
        mtype = next((m for m in self.analysis.motifTypes if m.id == type_id), None)
        if not mtype:
            return
        self.selected_type_id = type_id
        self.fill_occurrence_list(mtype)
        if mtype.occurrences:
            if seek_first or not self.selected_occ_id:
                self.goto_occurrence(mtype.occurrences[0], select_in_list=True)
            else:
                ids = {o.id for o in mtype.occurrences}
                if self.selected_occ_id not in ids:
                    self.goto_occurrence(mtype.occurrences[0], select_in_list=True)
                else:
                    for row in range(self.occ_list.count()):
                        if (
                            self.occ_list.item(row).data(Qt.ItemDataRole.UserRole)
                            == self.selected_occ_id
                        ):
                            self.occ_list.setCurrentRow(row)
                            break
        self.refresh_chart()

    def goto_occurrence(self, occ: MotifOccurrence, *, select_in_list: bool = True) -> None:
        self.selected_occ_id = occ.id
        if self.analysis:
            for i, a in enumerate(self.analysis.actors):
                if a.id == occ.actorId:
                    self.actor_combo.blockSignals(True)
                    self.actor_combo.setCurrentIndex(i)
                    self.actor_combo.blockSignals(False)
                    break
        self.playing = False
        self.play_btn.setText("Play")
        self.sync_overlay()
        self.seek_time(occ.start)
        if select_in_list:
            for row in range(self.occ_list.count()):
                if self.occ_list.item(row).data(Qt.ItemDataRole.UserRole) == occ.id:
                    self.occ_list.setCurrentRow(row)
                    break
        self.refresh_chart()
        self.status.setText(f"Occurrence {occ.start:.1f}s–{occ.end:.1f}s")

    def on_motif_clicked(self, item: QListWidgetItem) -> None:
        mid = item.data(Qt.ItemDataRole.UserRole)
        if mid:
            self.select_motif_type(str(mid), seek_first=True)

    def on_occ_clicked(self, item: QListWidgetItem) -> None:
        mtype = self.selected_motif_type()
        if not mtype:
            return
        oid = item.data(Qt.ItemDataRole.UserRole)
        occ = next((o for o in mtype.occurrences if o.id == oid), None)
        if occ:
            self.goto_occurrence(occ, select_in_list=False)

    def step_occurrence(self, delta: int) -> None:
        mtype = self.selected_motif_type()
        if not mtype or not mtype.occurrences:
            return
        occs = mtype.occurrences
        if self.selected_occ_id:
            try:
                idx = next(i for i, o in enumerate(occs) if o.id == self.selected_occ_id)
            except StopIteration:
                idx = 0
        else:
            idx = 0
        idx = (idx + delta) % len(occs)
        self.goto_occurrence(occs[idx], select_in_list=True)

    def on_motif_rename(self, item: QListWidgetItem) -> None:
        self.motif_list.setCurrentItem(item)
        self.rename_selected_motif()

    def rename_selected_motif(self) -> None:
        item = self.motif_list.currentItem()
        if not item or not self.analysis:
            return
        mid = item.data(Qt.ItemDataRole.UserRole)
        mtype = next((m for m in self.analysis.motifTypes if m.id == mid), None)
        if not mtype:
            return
        name, ok = QInputDialog.getText(self, "Rename", "Type name:", text=mtype.name)
        if not ok or not name.strip():
            return
        mtype.name = name.strip()
        n = len(mtype.occurrences)
        item.setText(f"{mtype.name}  ·  {n}×")
        self._dirty = True
        self.refresh_chart()
        self.status.setText("Renamed (unsaved)")

    def closeEvent(self, event) -> None:  # noqa: N802
        self.playing = False
        self.decode.shutdown()
        super().closeEvent(event)


def run_app() -> None:
    _configure_matplotlib()
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.show()
    app.exec()
