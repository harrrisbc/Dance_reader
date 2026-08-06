from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
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
    QVBoxLayout,
    QWidget,
)

from app.pipeline.motifs import find_similar_motif, segment_near_time
from app.pipeline.pose_track import (
    COCO_EDGES,
    DEFAULT_MAX_PEOPLE,
    DEFAULT_MODEL,
    DEFAULT_STRIDE,
    TrackBuffer,
)
from app.pipeline.run import analyze_video
from app.project_io import (
    DanceProject,
    buffers_from_analysis,
    load_project,
    save_project,
)
from app.schemas import AnalysisResult, Actor, MotifOccurrence, MotifType

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
ARTIFACTS = BACKEND / "artifacts"
PREVIEW_MAX_W = 960
ZOOM_MIN_WINDOW_S = 2.0
ROW_HEIGHT_IN = 0.72  # ~70px per metric row at default DPI

# Chart series: attr, label, color, default_on, kind, ylabel, ylim
# kind: area | line | level | travel | ribbon | pos
CHART_SERIES = (
    ("energy", "能量", "#4aa3ff", True, "area", "相對強度", (0.0, 1.0)),
    ("speed", "速度", "#f0a040", True, "line", "肩寬／秒", (0.0, 4.0)),
    ("reach", "伸展", "#78b4a0", True, "reach", "肩寬倍數", (1.0, 3.5)),
    ("level", "高低", "#c45c4a", False, "level", "站高比例", (0.0, 1.05)),
    ("travel", "走位", "#e8c56e", False, "area", "肩寬／秒", (0.0, 2.0)),
    ("directness", "直進", "#a67c52", False, "ribbon", "", (0.0, 1.0)),
    ("pos_x", "左右", "#8f7cc0", False, "pos", "畫面比例", (0.0, 1.0)),
    ("pos_y", "前後", "#c07fa8", False, "pos", "畫面比例", (0.0, 1.0)),
)


def _configure_matplotlib_cjk() -> None:
    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft JhengHei",
        "Noto Sans TC",
        "Microsoft YaHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False


_configure_matplotlib_cjk()


class AnalyzeWorker(QThread):
    progress = Signal(float, str)
    finished_ok = Signal(object, object)  # AnalysisResult, keyed buffers
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
            self.finished_ok.emit(result, keyed)
        except Exception as exc:
            self.finished_err.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Dance Analyser")
        self.resize(1100, 720)

        self.video_path: Path | None = None
        self.cap: cv2.VideoCapture | None = None
        self.fps = 30.0
        self.frame_count = 0
        self.playing = False
        self._play_pos = -1
        self.analysis: AnalysisResult | None = None
        self.track_buffers: dict[str, TrackBuffer] = {}
        self.worker: AnalyzeWorker | None = None
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

        self.video_label = QLabel("開啟本機影片開始")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumHeight(360)
        self.video_label.setStyleSheet("background:#111; color:#ccc;")

        self.status = QLabel("準備就緒")
        self.actor_combo = QComboBox()
        self.actor_combo.currentIndexChanged.connect(self.refresh_chart)

        self.max_people_spin = QSpinBox()
        self.max_people_spin.setRange(1, 4)
        self.max_people_spin.setValue(DEFAULT_MAX_PEOPLE)
        self.max_people_spin.setToolTip("分析時最多追蹤幾位舞者（丟失後會回收同一編號）")

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.sliderMoved.connect(self.on_seek)
        self.slider.sliderReleased.connect(self.on_seek_release)

        self.motif_list = QListWidget()
        self.motif_list.itemClicked.connect(self.on_motif_clicked)
        self.motif_list.itemDoubleClicked.connect(self.on_motif_rename)

        self.occ_list = QListWidget()
        self.occ_list.itemClicked.connect(self.on_occ_clicked)

        self.prev_occ_btn = QPushButton("上一處")
        self.prev_occ_btn.clicked.connect(lambda: self.step_occurrence(-1))
        self.next_occ_btn = QPushButton("下一處")
        self.next_occ_btn.clicked.connect(lambda: self.step_occurrence(1))

        self.fig = Figure(figsize=(5, 3.5), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setMinimumHeight(220)
        self.canvas.mpl_connect("button_press_event", self.on_chart_click)
        self.canvas.mpl_connect("scroll_event", self.on_chart_scroll)

        self.floor_fig = Figure(figsize=(5, 2.4), tight_layout=True)
        self.floor_canvas = FigureCanvasQTAgg(self.floor_fig)
        self.floor_canvas.setMinimumHeight(160)

        self.floor_toggle = QCheckBox("走位圖（畫面座標）")
        self.floor_toggle.setChecked(True)
        self.floor_toggle.setToolTip(
            "螢幕座標，非真實舞台座標。固定鏡頭的排練片最準；鏡頭平移／縮放會扭曲。"
        )
        self.floor_toggle.stateChanged.connect(self._on_floor_toggle)

        open_btn = QPushButton("開啟影片")
        open_btn.clicked.connect(self.open_video)
        open_project_btn = QPushButton("開啟專案")
        open_project_btn.clicked.connect(self.open_project)
        save_project_btn = QPushButton("儲存專案")
        save_project_btn.clicked.connect(self.save_project_dialog)
        self.play_btn = QPushButton("播放")
        self.play_btn.clicked.connect(self.toggle_play)
        analyze_btn = QPushButton("分析（Supervision）")
        analyze_btn.clicked.connect(self.start_analyze)
        self.find_similar_btn = QPushButton("搵相似")
        self.find_similar_btn.clicked.connect(self.find_similar)
        rename_btn = QPushButton("重新命名類型")
        rename_btn.clicked.connect(self.rename_selected_motif)

        zoom_in_btn = QPushButton("放大")
        zoom_in_btn.clicked.connect(lambda: self.zoom_timeline(0.7))
        zoom_out_btn = QPushButton("縮小")
        zoom_out_btn.clicked.connect(lambda: self.zoom_timeline(1.4))
        zoom_reset_btn = QPushButton("重設")
        zoom_reset_btn.clicked.connect(self.reset_timeline_zoom)

        top = QHBoxLayout()
        top.addWidget(open_btn)
        top.addWidget(open_project_btn)
        top.addWidget(save_project_btn)
        top.addWidget(self.play_btn)
        top.addWidget(analyze_btn)
        top.addWidget(self.find_similar_btn)
        top.addWidget(QLabel("舞者"))
        top.addWidget(self.actor_combo)
        top.addWidget(QLabel("最多舞者"))
        top.addWidget(self.max_people_spin)
        top.addWidget(rename_btn)
        top.addStretch(1)

        occ_nav = QHBoxLayout()
        occ_nav.addWidget(self.prev_occ_btn)
        occ_nav.addWidget(self.next_occ_btn)

        zoom_row = QHBoxLayout()
        zoom_row.addWidget(QLabel("時間軸"))
        zoom_row.addWidget(zoom_in_btn)
        zoom_row.addWidget(zoom_out_btn)
        zoom_row.addWidget(zoom_reset_btn)
        zoom_row.addStretch(1)

        series_row = QHBoxLayout()
        series_row.addWidget(QLabel("曲線"))
        self.series_checks: dict[str, QCheckBox] = {}
        for attr, label, _color, default_on, *_rest in CHART_SERIES:
            cb = QCheckBox(label)
            cb.setChecked(default_on)
            cb.stateChanged.connect(self.refresh_chart)
            self.series_checks[attr] = cb
            series_row.addWidget(cb)
        series_row.addStretch(1)

        right = QVBoxLayout()
        right.addWidget(QLabel("重複動作類型（雙擊改名）"))
        right.addWidget(self.motif_list, stretch=1)
        right.addWidget(QLabel("出現次數（點選跳轉）"))
        right.addWidget(self.occ_list, stretch=1)
        right.addLayout(occ_nav)
        right.addLayout(zoom_row)
        right.addLayout(series_row)
        right.addWidget(QLabel("點圖跳轉 · 滾輪縮放"))
        right.addWidget(self.canvas, stretch=3)
        right.addWidget(self.floor_toggle)
        right.addWidget(self.floor_canvas, stretch=2)

        body = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(self.video_label, stretch=1)
        left.addWidget(self.slider)
        body.addLayout(left, stretch=3)
        body.addLayout(right, stretch=2)

        root = QVBoxLayout()
        root.addLayout(top)
        root.addLayout(body, stretch=1)
        root.addWidget(self.status)

        w = QWidget()
        w.setLayout(root)
        self.setCentralWidget(w)

        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.on_tick)

    def open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "選擇影片",
            "",
            "Video (*.mp4 *.mov *.avi *.mkv *.webm);;All (*.*)",
        )
        if not path:
            return
        self.load_video(Path(path))

    def load_video(self, path: Path, *, clear_analysis: bool = True) -> None:
        if self.cap is not None:
            self.cap.release()
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
        self._play_pos = -1

        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            QMessageBox.warning(self, "錯誤", f"無法開啟：{path}")
            self.cap = None
            return
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._duration_s = self.frame_count / max(self.fps, 1e-6)
        self.view_start = 0.0
        self.view_end = max(self._duration_s, 1.0)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(0, self.frame_count - 1))
        self.slider.setValue(0)
        self.playing = False
        self.play_btn.setText("播放")
        self.timer.setInterval(max(1, int(round(1000.0 / max(self.fps, 1.0)))))
        if clear_analysis:
            self.status.setText(str(path.name))
        self.show_frame(0, force_seek=True)

    def current_time(self) -> float:
        if not self.cap:
            return 0.0
        idx = int(self.slider.value())
        return idx / max(self.fps, 1e-6)

    def seek_time(self, t: float) -> None:
        if self.cap is None or self.frame_count <= 0:
            return
        frame = int(round(t * self.fps))
        frame = max(0, min(frame, self.frame_count - 1))
        self.show_frame(frame, force_seek=True)

    def show_frame(self, frame_idx: int, *, force_seek: bool = False) -> None:
        if self.cap is None:
            return
        frame_idx = max(0, min(frame_idx, max(0, self.frame_count - 1)))

        # Sequential read while playing — avoid random seek every frame
        need_seek = force_seek or self._play_pos < 0 or frame_idx != self._play_pos + 1
        if need_seek:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.cap.read()
        if not ok:
            return
        self._play_pos = frame_idx

        t = frame_idx / max(self.fps, 1e-6)
        frame = self.draw_poses(frame, t)

        # Cap preview size before Qt scale
        h0, w0 = frame.shape[:2]
        if w0 > PREVIEW_MAX_W:
            scale = PREVIEW_MAX_W / w0
            frame = cv2.resize(
                frame,
                (PREVIEW_MAX_W, max(1, int(h0 * scale))),
                interpolation=cv2.INTER_AREA,
            )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        xform = (
            Qt.TransformationMode.FastTransformation
            if self.playing
            else Qt.TransformationMode.SmoothTransformation
        )
        pix = QPixmap.fromImage(img).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            xform,
        )
        self.video_label.setPixmap(pix)
        self.slider.blockSignals(True)
        self.slider.setValue(frame_idx)
        self.slider.blockSignals(False)
        self.update_playhead(t)

    def draw_poses(self, frame: np.ndarray, t: float) -> np.ndarray:
        if not self.analysis or not self.analysis.actors:
            return frame
        out = frame.copy()
        h, w = out.shape[:2]
        actor = self.selected_actor()
        actors = [actor] if actor else self.analysis.actors
        colors = [(80, 180, 255), (80, 220, 120), (220, 160, 80), (200, 100, 220)]
        for ai, act in enumerate(actors):
            if not act or not act.poses:
                continue
            pose = min(act.poses, key=lambda p: abs(p.time - t))
            if abs(pose.time - t) > 0.35:
                continue
            color = colors[ai % len(colors)]
            pts = []
            for kp in pose.keypoints:
                if len(kp) < 3 or kp[2] < 0.3:
                    pts.append(None)
                    continue
                x, y = int(kp[0] * w), int(kp[1] * h)
                pts.append((x, y))
                cv2.circle(out, (x, y), 3, color, -1)
            for a, b in COCO_EDGES:
                if a < len(pts) and b < len(pts) and pts[a] and pts[b]:
                    cv2.line(out, pts[a], pts[b], color, 2)
        return out

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

    def toggle_play(self) -> None:
        if self.cap is None:
            return
        self.playing = not self.playing
        self.play_btn.setText("暫停" if self.playing else "播放")
        if self.playing:
            self.timer.setInterval(max(1, int(round(1000.0 / max(self.fps, 1.0)))))
            self.timer.start()
        else:
            self.timer.stop()
            # Refresh with smooth scale when paused
            self.show_frame(self.slider.value(), force_seek=True)

    def on_tick(self) -> None:
        if self.cap is None or not self.playing:
            return
        idx = self._play_pos + 1 if self._play_pos >= 0 else int(self.slider.value()) + 1
        if idx >= self.frame_count - 1:
            self.playing = False
            self.play_btn.setText("播放")
            self.timer.stop()
            return
        self.show_frame(idx, force_seek=False)

    def on_seek(self, value: int) -> None:
        self.playing = False
        self.play_btn.setText("播放")
        self.timer.stop()
        self.show_frame(value, force_seek=True)

    def on_seek_release(self) -> None:
        self.show_frame(self.slider.value(), force_seek=True)

    def start_analyze(self) -> None:
        if self.video_path is None:
            QMessageBox.information(self, "提示", "請先開啟影片")
            return
        if self.worker and self.worker.isRunning():
            return
        n = int(self.max_people_spin.value())
        self.status.setText(f"分析中…（最多 {n} 位舞者）")
        self.worker = AnalyzeWorker(self.video_path, max_people=n)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished_ok.connect(self.on_analyze_done)
        self.worker.finished_err.connect(self.on_analyze_err)
        self.worker.start()

    def on_progress(self, p: float, msg: str) -> None:
        self.status.setText(f"{msg}  {int(p * 100)}%")

    def on_analyze_done(self, result: object, keyed: object) -> None:
        assert isinstance(result, AnalysisResult)
        self.analysis = result
        self.track_buffers = keyed if isinstance(keyed, dict) else {}
        self._dirty = True
        self._project_path = None
        self.apply_analysis_to_ui(seek_first_motif=True)
        self.status.setText(
            f"完成：{len(result.actors)} 位舞者，{len(result.motifTypes)} 種重複類型（未儲存）"
        )
        self.show_frame(self.slider.value(), force_seek=True)

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
            item = QListWidgetItem(f"{m.name}  ·  {len(m.occurrences)} 次")
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

    def save_project_dialog(self) -> None:
        if not self.analysis or self.video_path is None:
            QMessageBox.information(self, "提示", "請先開啟影片並完成分析，再儲存專案。")
            return
        suggested = self._project_path or self.video_path.with_suffix(".dance.json")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "儲存專案",
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
            )
            save_project(out, project)
            self._project_path = out
            self._dirty = False
            self.status.setText(f"已儲存：{out.name}")
        except Exception as exc:
            QMessageBox.critical(self, "儲存失敗", str(exc))

    def open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "開啟專案",
            "",
            "Dance project (*.dance.json);;JSON (*.json);;All (*.*)",
        )
        if not path:
            return
        try:
            project = load_project(Path(path))
        except Exception as exc:
            QMessageBox.critical(self, "開啟失敗", str(exc))
            return

        video = Path(project.video_path)
        if not video.is_file():
            QMessageBox.warning(
                self,
                "搵唔到影片",
                f"專案入面嘅影片路徑唔存在：\n{project.video_path}\n\n請重新揀選影片檔。",
            )
            picked, _ = QFileDialog.getOpenFileName(
                self,
                "揀選專案對應影片",
                "",
                "Video (*.mp4 *.mov *.avi *.mkv *.webm);;All (*.*)",
            )
            if not picked:
                return
            video = Path(picked)

        if project.max_people:
            self.max_people_spin.setValue(max(1, min(4, int(project.max_people))))

        self.analysis = project.analysis
        self.track_buffers = buffers_from_analysis(project.analysis)
        self._project_path = Path(path)
        self._dirty = False
        self.load_video(video, clear_analysis=False)
        self.apply_analysis_to_ui(seek_first_motif=False)
        self.show_frame(0, force_seek=True)
        self.status.setText(
            f"已載入專案：{Path(path).name}（{len(project.analysis.actors)} 舞者，"
            f"{len(project.analysis.motifTypes)} 類型）— 唔使再分析"
        )

    def find_similar(self) -> None:
        if not self.analysis or not self.track_buffers:
            QMessageBox.information(self, "提示", "請先完成分析")
            return
        actor = self.selected_actor()
        if actor is None:
            QMessageBox.information(self, "提示", "請選擇舞者")
            return

        query_time = self.current_time()
        mtype = self.selected_motif_type()
        if mtype and self.selected_occ_id:
            occ = next((o for o in mtype.occurrences if o.id == self.selected_occ_id), None)
            if occ:
                query_time = (occ.start + occ.end) / 2
                actor_id = occ.actorId
            else:
                actor_id = actor.id
        else:
            actor_id = actor.id

        query = segment_near_time(self.track_buffers, actor_id, query_time)
        if query is None:
            QMessageBox.warning(self, "搵相似", "而家呢一段動態唔夠，請跳去有動作嘅位置再試。")
            return

        self._sim_counter += 1
        n = self._sim_counter
        found = find_similar_motif(
            self.track_buffers,
            query,
            name=f"相似 {n}",
            type_id=f"type_sim_{n}",
            color="#78b4a0" if n % 2 else "#a67c52",
        )
        if found is None or not found.occurrences:
            QMessageBox.information(self, "搵相似", "搵唔到相似動作。")
            return

        remaining = [m for m in self.analysis.motifTypes if m.id != found.id]
        remaining.append(found)
        self.analysis.motifTypes = remaining

        self.motif_list.clear()
        for m in self.analysis.motifTypes:
            item = QListWidgetItem(f"{m.name}  ·  {len(m.occurrences)} 次")
            item.setData(Qt.ItemDataRole.UserRole, m.id)
            self.motif_list.addItem(item)
        for row in range(self.motif_list.count()):
            if self.motif_list.item(row).data(Qt.ItemDataRole.UserRole) == found.id:
                self.motif_list.setCurrentRow(row)
                break
        self.select_motif_type(found.id, seek_first=True)
        self._dirty = True
        self.status.setText(f"{found.name}：找到 {len(found.occurrences)} 處（未儲存）")

    def on_analyze_err(self, err: str) -> None:
        self.status.setText("分析失敗")
        QMessageBox.critical(self, "分析失敗", err)

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
            ax.axhline(1.5, color="#999999", linewidth=0.7, linestyle="--", alpha=0.7)
            ax.axhline(3.0, color="#999999", linewidth=0.7, linestyle="--", alpha=0.7)
            ax.text(
                self.view_end,
                1.5,
                " 收",
                va="bottom",
                ha="right",
                fontsize=7,
                color="#666666",
                clip_on=False,
            )
            ax.text(
                self.view_end,
                3.0,
                " 全開",
                va="bottom",
                ha="right",
                fontsize=7,
                color="#666666",
                clip_on=False,
            )
        elif kind == "level":
            ax.axhspan(0.0, 0.45, color="#c45c4a", alpha=0.08, zorder=0)
            ax.axhspan(0.45, 0.8, color="#e8c56e", alpha=0.08, zorder=0)
            ax.axhspan(0.8, 1.05, color="#78b4a0", alpha=0.08, zorder=0)
            ax.plot(times, ys, color=color, linewidth=1.3)
        elif kind == "ribbon":
            # Color by directness; gaps (NaN) where dancer is still
            ax.set_yticks([])
            for i in range(1, len(times)):
                a, b = ys[i - 1], ys[i]
                if a != a or b != b:  # NaN
                    continue
                # 1 = 直線 (cool), 0 = 迂迴 (warm)
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
        actor = self.selected_actor()
        enabled = self._enabled_series()
        n = len(enabled)

        if not actor or not actor.metrics:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "尚無分析資料", ha="center", va="center")
            ax.set_axis_off()
            self.canvas.draw_idle()
            self.refresh_floor_map()
            return

        if n == 0:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "請勾選至少一條曲線", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            self.canvas.draw_idle()
            self.refresh_floor_map()
            return

        # Resize figure height to ~70px per row
        self.fig.set_size_inches(5.0, max(2.0, n * ROW_HEIGHT_IN + 0.6), forward=False)
        axes = self.fig.subplots(n, 1, sharex=True, squeeze=False)
        axes = [axes[i, 0] for i in range(n)]
        self._chart_axes = axes

        times = [m.time for m in actor.metrics]
        t_now = self.current_time()

        for i, (attr, label, color, _on, kind, ylabel, ylim) in enumerate(enabled):
            ax = axes[i]
            ys = [float(getattr(m, attr, 0.0)) for m in actor.metrics]
            self._plot_series_row(
                ax, times, ys, kind=kind, color=color, ylabel=ylabel or label, ylim=ylim
            )
            ax.set_xlim(self.view_start, self.view_end)
            ax.tick_params(axis="y", labelsize=7)
            # Row title on the left
            ax.set_title(label, loc="left", fontsize=8, pad=2)
            self._draw_motif_spans(ax, actor.id)
            ph = ax.axvline(t_now, color="#222222", linewidth=1.4, zorder=5)
            self._playhead_lines.append(ph)
            if i < n - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("時間（秒）", fontsize=8)

        self.fig.tight_layout(h_pad=0.15)
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
            ax.text(0.5, 0.5, "尚無走位資料", ha="center", va="center")
            ax.set_axis_off()
            self.floor_canvas.draw_idle()
            return

        # Points inside current zoom window
        pts = [
            (m.pos_x, m.pos_y, m.time)
            for m in actor.metrics
            if self.view_start <= m.time <= self.view_end
        ]
        if len(pts) < 1:
            ax.text(0.5, 0.5, "此時間窗無走位點", ha="center", va="center")
            ax.set_axis_off()
            self.floor_canvas.draw_idle()
            return

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ts = [p[2] for p in pts]
        t0, t1 = ts[0], ts[-1]
        span = max(t1 - t0, 1e-6)

        # Time-colored segments: light → dark
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
            )

        # Playhead position
        t_now = self.current_time()
        best = min(pts, key=lambda p: abs(p[2] - t_now))
        self._floor_playhead = ax.scatter(
            [best[0]], [best[1]], c="#c45c4a", s=48, zorder=5, edgecolors="white", linewidths=0.6
        )

        # Aspect from analysis frame size (fallback 16:9)
        fw = int(self.analysis.frame_width) if self.analysis else 0
        fh = int(self.analysis.frame_height) if self.analysis else 0
        if fw <= 0 or fh <= 0:
            fw, fh = 16, 9
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.invert_yaxis()  # image y: 0 = top/far at top of plot
        ax.set_box_aspect(fh / float(fw))

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("走位圖（畫面座標）", fontsize=9)
        for spine in ax.spines.values():
            spine.set_color("#aaaaaa")
        self.floor_fig.tight_layout()
        self.floor_canvas.draw_idle()

    def update_playhead(self, t: float) -> None:
        if self._playhead_lines:
            for line in self._playhead_lines:
                line.set_xdata([t, t])
            self.canvas.draw_idle()
        # Move floor-map dot without full redraw when possible
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
        if event.inaxes is None or event.xdata is None:
            return
        if self.cap is None or self.frame_count <= 0:
            return
        if event.inaxes not in self._chart_axes:
            return
        self.playing = False
        self.play_btn.setText("播放")
        self.timer.stop()
        self.seek_time(float(event.xdata))
        self.status.setText(f"跳至 {float(event.xdata):.2f}s")

    def fill_occurrence_list(self, mtype: MotifType) -> None:
        self.occ_list.clear()
        for i, occ in enumerate(mtype.occurrences, start=1):
            item = QListWidgetItem(f"#{i}  {occ.start:.1f}s–{occ.end:.1f}s")
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
        self.play_btn.setText("播放")
        self.timer.stop()
        self.seek_time(occ.start)
        if select_in_list:
            for row in range(self.occ_list.count()):
                if self.occ_list.item(row).data(Qt.ItemDataRole.UserRole) == occ.id:
                    self.occ_list.setCurrentRow(row)
                    break
        self.refresh_chart()
        self.status.setText(f"出現處 {occ.start:.1f}s–{occ.end:.1f}s")

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
        name, ok = QInputDialog.getText(self, "重新命名", "類型名稱：", text=mtype.name)
        if not ok or not name.strip():
            return
        mtype.name = name.strip()
        n = len(mtype.occurrences)
        item.setText(f"{mtype.name}  ·  {n} 次")
        self._dirty = True
        self.refresh_chart()
        self.status.setText("已改名（未儲存）")

    def closeEvent(self, event) -> None:  # noqa: N802
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        super().closeEvent(event)


def run_app() -> None:
    _configure_matplotlib_cjk()
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.show()
    app.exec()
