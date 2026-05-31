"""
GUI - View and controls for the stereoscope application.

PyQt6-based interface with camera setup prompts for focus and shutter state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from compositor import (
    AlignmentMetrics,
    AnaglyphMethod,
    affine_to_metrics,
    apply_180_flip_to_transform,
    build_anaglyph,
    build_three_way_overlay,
    compose_affine,
    compute_alignment,
    compute_sharpness,
    invert_affine,
    normalize_rotation_deg,
)

logger = logging.getLogger(__name__)

try:
    from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
    from PyQt6.QtGui import QImage, QPixmap
    from PyQt6.QtWidgets import (
        QApplication,
        QButtonGroup,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMenu,
        QMenuBar,
        QMessageBox,
        QPushButton,
        QRadioButton,
        QVBoxLayout,
        QWidget,
    )

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


@dataclass
class SetupState:
    """User's focus and shutter selections."""

    left_in_focus: bool = False
    right_in_focus: bool = False
    top_in_focus: bool = False
    shutter_top: bool = True  # True = top-down camera, False = left+right eyepieces


# Legacy aliases for any external code referencing the old private names
_affine_to_metrics = affine_to_metrics
_compose_affine = compose_affine
_invert_affine = invert_affine
_normalize_rotation_deg = normalize_rotation_deg
_apply_180_flip_to_transform = apply_180_flip_to_transform


def build_anaglyph_overlap(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    M_right_to_left: np.ndarray,
    method: AnaglyphMethod = AnaglyphMethod.WIMMER,
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    """Thin wrapper around compositor.build_anaglyph for backward compatibility."""
    return build_anaglyph(left_bgr, right_bgr, M_right_to_left, method)


def cv2_to_qimage(frame: np.ndarray) -> QImage:
    """Convert OpenCV BGR frame to QImage for display."""
    if frame is None or frame.size == 0:
        return QImage()
    h, w, ch = frame.shape
    bytes_per_line = ch * w
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)


# Sharpness thresholds for stoplight (Laplacian variance)
SHARPNESS_RED = 100  # Below: poor focus
SHARPNESS_YELLOW = 300  # Red–Yellow: improving; Yellow–Green: good
SHARPNESS_GREEN = 500  # Above: excellent focus


class CaptureWorker(QThread):
    """Background thread that grabs frames from cameras."""

    frame_ready = pyqtSignal(str, object)  # camera_id, frame (numpy array)

    def __init__(
        self,
        left_cap: cv2.VideoCapture | None,
        right_cap: cv2.VideoCapture | None,
        top_cap: cv2.VideoCapture | None,
    ) -> None:
        super().__init__()
        self._caps = {
            "left": left_cap,
            "right": right_cap,
            "top": top_cap,
        }
        self._running = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        while self._running:
            for cam_id, cap in self._caps.items():
                if cap is not None and cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        self.frame_ready.emit(cam_id, frame)
            self.msleep(33)  # ~30 fps


class CameraSetupWindow(QMainWindow):
    """Window that prompts for focus and shutter state with live camera previews."""

    def __init__(self) -> None:
        super().__init__()
        self._pair = None
        self._worker: CaptureWorker | None = None
        self._state = SetupState()
        self._last_frame: dict[str, np.ndarray | None] = {
            "left": None,
            "right": None,
            "top": None,
        }
        self._init_frames_remaining = 180  # Show all feeds for ~2–3 sec on startup
        self._last_valid_align_text: dict[str, str] = {}
        self._last_valid_overall: str = "Overall: —"
        self._stereo_M_right_to_left: np.ndarray | None = None
        self._stereo_overlap_roi: tuple[int, int, int, int] | None = None
        self._M_top_to_left: np.ndarray | None = None
        self._transforms_locked: bool = False
        self._anaglyph_method: AnaglyphMethod = AnaglyphMethod.WIMMER
        self._video_recorder = None  # lazy import to avoid circular deps
        self._calibration = None  # CalibrationResult when loaded/computed
        self._setup_ui()
        self._init_cameras()
        self._align_timer = QTimer(self)
        self._align_timer.timeout.connect(self._update_alignment)
        self._align_timer.start(500)
        self._anaglyph_timer = QTimer(self)
        self._anaglyph_timer.timeout.connect(self._update_anaglyph_and_overlay)
        self._anaglyph_timer.start(100)

    def _setup_ui(self) -> None:
        self.setWindowTitle("Stereoscope - Camera Setup & Alignment")
        self.setMinimumSize(1280, 720)
        self.resize(1400, 800)
        # Standard window buttons (minimize, maximize, close) are default on QMainWindow
        self._setup_menu_bar()
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # === LEFT SIDEBAR: Camera previews and controls ===
        sidebar = QWidget()
        sidebar.setMaximumWidth(380)
        sidebar_layout = QVBoxLayout(sidebar)
        preview_group = QGroupBox("Cameras")
        preview_layout = QVBoxLayout(preview_group)
        # Top (compact)
        self._preview_top = QLabel()
        self._preview_top.setMinimumSize(340, 200)
        self._preview_top.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_top.setStyleSheet("background: #1a1a1a; color: #888;")
        self._preview_top.setText("Top\n(no feed)")
        preview_layout.addWidget(self._preview_top)
        self._focus_top = self._make_stoplight("Top")
        preview_layout.addLayout(self._focus_top)
        # Left | Right (compact)
        lr_row = QHBoxLayout()
        self._preview_left = QLabel()
        self._preview_left.setMinimumSize(165, 124)
        self._preview_left.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_left.setStyleSheet("background: #1a1a1a; color: #888;")
        self._preview_left.setText("Left\n(no feed)")
        self._preview_right = QLabel()
        self._preview_right.setMinimumSize(165, 124)
        self._preview_right.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_right.setStyleSheet("background: #1a1a1a; color: #888;")
        self._preview_right.setText("Right\n(no feed)")
        lr_row.addWidget(self._preview_left)
        lr_row.addWidget(self._preview_right)
        preview_layout.addLayout(lr_row)
        focus_row = QHBoxLayout()
        self._focus_left = self._make_stoplight("Left")
        self._focus_right = self._make_stoplight("Right")
        focus_row.addLayout(self._focus_left)
        focus_row.addLayout(self._focus_right)
        preview_layout.addLayout(focus_row)
        sidebar_layout.addWidget(preview_group)
        self._setup_ui_controls(sidebar_layout)
        main_layout.addWidget(sidebar)

        # === RIGHT: Metrics, three-way overlay, anaglyph ===
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        align_group = QGroupBox("Alignment Metrics (rotation, scale, translation)")
        align_group.setToolTip("Left↔Right updates when 'Eyepieces' slider position is selected.")
        align_inner = QVBoxLayout(align_group)
        self._align_labels: dict[str, QLabel] = {}
        for pair in ("Top↔Left", "Top↔Right", "Left↔Right"):
            lbl = QLabel("—")
            lbl.setWordWrap(True)
            lbl.setStyleSheet("font-family: monospace; padding: 4px;")
            self._align_labels[pair] = lbl
            align_inner.addWidget(QLabel(pair + ":"))
            align_inner.addWidget(lbl)
        self._align_overall = QLabel("Overall: —")
        self._align_overall.setStyleSheet("font-weight: bold; font-size: 12pt; padding: 8px;")
        align_inner.addWidget(self._align_overall)
        right_layout.addWidget(align_group)

        overlay_group = QGroupBox("Three-way overlay (all three aligned, grayscale)")
        overlay_group.setToolTip("Left as reference; right and top warped to align. Full extent of all three blended.")
        self._overlay_label = QLabel()
        self._overlay_label.setMinimumSize(420, 280)
        self._overlay_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._overlay_label.setStyleSheet("background: #1a1a1a; color: #666;")
        self._overlay_label.setText("Top / Left / Right")
        overlay_group_layout = QVBoxLayout(overlay_group)
        overlay_group_layout.addWidget(self._overlay_label)
        right_layout.addWidget(overlay_group)

        anaglyph_group = QGroupBox("Anaglyph (cyan/magenta)")
        anaglyph_group.setToolTip("Transform (rotation, scale, translation) is stored for 3D sample mode.")
        self._anaglyph_label = QLabel()
        self._anaglyph_label.setMinimumSize(480, 360)
        self._anaglyph_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._anaglyph_label.setStyleSheet("background: #0a0a0a; color: #666;")
        self._anaglyph_label.setText("Left + Right aligned")
        anaglyph_group_layout = QVBoxLayout(anaglyph_group)
        anaglyph_group_layout.addWidget(self._anaglyph_label)
        right_layout.addWidget(anaglyph_group, 1)
        main_layout.addWidget(right_panel, 1)

    def _make_stoplight(self, name: str) -> QHBoxLayout:
        """Create a stoplight indicator layout: label + colored circle."""
        row = QHBoxLayout()
        lbl = QLabel(f"{name}:")
        lbl.setMinimumWidth(45)
        row.addWidget(lbl)
        indicator = QLabel()
        indicator.setFixedSize(24, 24)
        indicator.setStyleSheet("background: #333; border-radius: 12px; border: 2px solid #555;")
        indicator.setObjectName(f"stoplight_{name.lower()}")
        row.addWidget(indicator)
        sharpness_lbl = QLabel("—")
        sharpness_lbl.setObjectName(f"sharpness_{name.lower()}")
        sharpness_lbl.setMinimumWidth(50)
        row.addWidget(sharpness_lbl)
        row.addStretch()
        setattr(self, f"_stoplight_{name.lower()}", indicator)
        setattr(self, f"_sharpness_lbl_{name.lower()}", sharpness_lbl)
        return row

    def _setup_menu_bar(self) -> None:
        """Add menu bar: File, Capture, Calibration."""
        menubar = QMenuBar(self)
        self.setMenuBar(menubar)
        file_menu = QMenu("&File", self)
        menubar.addMenu(file_menu)
        act_quit = file_menu.addAction("E&nd program")
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self._on_end_program)
        file_menu.addSeparator()
        act_save_overlay = file_menu.addAction("Save overlay as &JPG...")
        act_save_overlay.triggered.connect(self._save_overlay_jpg)
        act_save_anaglyph = file_menu.addAction("Save ana&glyph as JPG...")
        act_save_anaglyph.triggered.connect(self._save_anaglyph_jpg)

        # Capture menu
        capture_menu = QMenu("&Capture", self)
        menubar.addMenu(capture_menu)
        act_capture_still = capture_menu.addAction("Capture &Still Set (TIFF+JPG)")
        act_capture_still.setShortcut("Ctrl+S")
        act_capture_still.triggered.connect(self._capture_still_set)

        # Calibration menu
        calib_menu = QMenu("C&alibration", self)
        menubar.addMenu(calib_menu)
        act_start_calib = calib_menu.addAction("Start &Calibration Wizard...")
        act_start_calib.setShortcut("Ctrl+K")
        act_start_calib.triggered.connect(self._start_calibration_wizard)
        act_load_calib = calib_menu.addAction("&Load Calibration...")
        act_load_calib.triggered.connect(self._load_calibration_file)
        act_save_calib = calib_menu.addAction("&Save Calibration...")
        act_save_calib.triggered.connect(self._save_calibration_file)

    def _setup_ui_controls(self, layout: QVBoxLayout) -> None:
        """Add focus checkboxes, shutter radio, and continue button."""
        # Focus checkboxes
        focus_group = QGroupBox("Which cameras are in focus?")
        focus_layout = QHBoxLayout(focus_group)
        self._cb_left_focus = QCheckBox("Left")
        self._cb_right_focus = QCheckBox("Right")
        self._cb_top_focus = QCheckBox("Top")
        focus_layout.addWidget(self._cb_left_focus)
        focus_layout.addWidget(self._cb_right_focus)
        focus_layout.addWidget(self._cb_top_focus)
        focus_layout.addStretch()
        layout.addWidget(focus_group)

        # Shutter/slider radio: trinocular slider sends light to top-down OR both eyepieces
        shutter_group = QGroupBox("Trinocular slider position")
        shutter_layout = QVBoxLayout(shutter_group)
        self._shutter_group = QButtonGroup(self)
        self._rb_shutter_top = QRadioButton("Top (top-down camera)")
        self._rb_shutter_top.setChecked(True)
        self._rb_shutter_eyepieces = QRadioButton("Eyepieces (left + right cameras)")
        self._shutter_group.addButton(self._rb_shutter_top)
        self._shutter_group.addButton(self._rb_shutter_eyepieces)
        self._rb_shutter_top.toggled.connect(self._on_shutter_changed)
        self._rb_shutter_eyepieces.toggled.connect(self._on_shutter_changed)
        shutter_layout.addWidget(self._rb_shutter_top)
        shutter_layout.addWidget(self._rb_shutter_eyepieces)
        layout.addWidget(shutter_group)

        # Anaglyph method selector
        method_group = QGroupBox("Anaglyph Method")
        method_layout = QHBoxLayout(method_group)
        self._method_combo = QComboBox()
        for m in AnaglyphMethod:
            self._method_combo.addItem(m.value.replace("_", " ").title(), m)
        self._method_combo.setCurrentIndex(0)
        self._method_combo.currentIndexChanged.connect(self._on_method_changed)
        method_layout.addWidget(self._method_combo)
        layout.addWidget(method_group)

        # Lock / Unlock transforms (replaces Continue)
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self._btn_lock = QPushButton("Lock")
        self._btn_lock.setMinimumWidth(120)
        self._btn_lock.setToolTip("Freeze alignment transforms so you can change slides without changing alignments.")
        self._btn_lock.clicked.connect(self._on_lock_toggle)
        btn_layout.addWidget(self._btn_lock)
        layout.addLayout(btn_layout)

        # Capture controls
        capture_group = QGroupBox("Capture")
        capture_layout = QHBoxLayout(capture_group)
        self._btn_capture_still = QPushButton("Capture Still")
        self._btn_capture_still.setMinimumWidth(100)
        self._btn_capture_still.setToolTip("Capture full-resolution stills (TIFF+JPG) — Ctrl+S")
        self._btn_capture_still.clicked.connect(self._capture_still_set)
        capture_layout.addWidget(self._btn_capture_still)
        self._btn_record = QPushButton("Record")
        self._btn_record.setMinimumWidth(100)
        self._btn_record.setStyleSheet("color: red; font-weight: bold;")
        self._btn_record.setToolTip("Start/stop stereo video recording (left, right, and anaglyph MP4)")
        self._btn_record.clicked.connect(self._on_record_toggle)
        capture_layout.addWidget(self._btn_record)
        self._record_status = QLabel("")
        capture_layout.addWidget(self._record_status)
        layout.addWidget(capture_group)

        # Calibration
        calib_group = QGroupBox("Calibration")
        calib_layout = QHBoxLayout(calib_group)
        self._btn_calibrate = QPushButton("Calibrate...")
        self._btn_calibrate.setMinimumWidth(100)
        self._btn_calibrate.setToolTip("Start stereo calibration wizard — Ctrl+K")
        self._btn_calibrate.clicked.connect(self._start_calibration_wizard)
        calib_layout.addWidget(self._btn_calibrate)
        self._calib_status = QLabel("Not calibrated")
        calib_layout.addWidget(self._calib_status)
        layout.addWidget(calib_group)

    def _init_cameras(self) -> None:
        from camera_manager import CameraManager

        mgr = CameraManager()
        self._pair = mgr.get_stereo_pair(prefer_amscope=False, include_top=True)
        if not self._pair.left or not self._pair.right:
            QMessageBox.critical(
                self,
                "No Cameras",
                "Could not find stereo cameras. Connect both MD500L cameras and try again.",
            )
            return
        if not mgr.open_captures(self._pair):
            QMessageBox.critical(
                self,
                "Open Failed",
                "Could not open one or more cameras.",
            )
            return
        # Disable top options if no top camera
        if not self._pair.top or not self._pair.top_capture:
            self._preview_top.setText("Top\n(not connected)")
            self._cb_top_focus.setEnabled(False)
            self._cb_top_focus.setToolTip("Top camera not available")
        self._start_capture()

    def _start_capture(self) -> None:
        if self._pair is None:
            return
        self._worker = CaptureWorker(
            self._pair.left_capture,
            self._pair.right_capture,
            self._pair.top_capture,
        )
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.start()

    def _shutter_open(self, cam_id: str) -> bool:
        """True if this camera has the shutter open (should show live feed)."""
        if self._rb_shutter_top.isChecked():
            return cam_id == "top"
        return cam_id in ("left", "right")

    def _on_shutter_changed(self) -> None:
        """When shutter selection changes, show last frame for newly opened cameras."""
        self._refresh_static_feeds()

    def _both_cams_live_for_pair(self, pair_key: str) -> bool:
        """True if both cameras in this pair have shutter open (so alignment is meaningful)."""
        eyepieces_open = self._rb_shutter_eyepieces.isChecked()
        if pair_key == "Left↔Right":
            return eyepieces_open
        # Top↔Right and Top↔Left: never both live simultaneously
        # (slider sends light to top OR eyepieces, not both)
        return False

    def _update_alignment(self) -> None:
        """Compute and display alignment metrics. Keep last valid when static (persistent metrics)."""
        if getattr(self, "_transforms_locked", False):
            return
        if not hasattr(self, "_align_labels"):
            return
        scores = []
        last_valid = self._last_valid_align_text

        def set_pair(key: str, m: AlignmentMetrics, text_override: str | None = None) -> None:
            if not self._align_labels.get(key):
                return
            both_live = self._both_cams_live_for_pair(key)
            if text_override:
                if both_live or key == "Top↔Left":
                    self._align_labels[key].setText(text_override)
                    last_valid[key] = text_override
                    if m.valid:
                        scores.append(m.score)
                return
            if m.valid:
                t = f"rot={m.rotation_deg:.1f}° scale={m.scale:.3f} tx={m.tx:.0f} ty={m.ty:.0f}px  score={m.score:.0f}"
                if both_live or key == "Top↔Left":
                    self._align_labels[key].setText(t)
                    last_valid[key] = t
                    scores.append(m.score)
                else:
                    self._align_labels[key].setText(last_valid.get(key, "—") + " (use correct shutter to refresh)")
            else:
                if both_live:
                    self._align_labels[key].setText(last_valid.get(key, "— (insufficient features)"))
                else:
                    if key == "Top↔Right" or key == "Top↔Left":
                        self._align_labels[key].setText(
                            last_valid.get(key, "—") + " (top and eyepieces can't be live together)"
                        )
                    elif key == "Left↔Right":
                        self._align_labels[key].setText(
                            last_valid.get(key, "—") + " (select 'Eyepieces' slider position)"
                        )
                    else:
                        self._align_labels[key].setText(last_valid.get(key, "— (insufficient features)"))

        try:
            m_tr, M_tr = compute_alignment(self._last_frame.get("top"), self._last_frame.get("right"))
            if not m_tr.valid and M_tr is None:
                m_rt, M_rt = compute_alignment(self._last_frame.get("right"), self._last_frame.get("top"))
                if M_rt is not None:
                    M_tr = _invert_affine(M_rt)
                    m_tr = _affine_to_metrics(M_tr)
                    m_tr.valid = True
            m_lr, M_lr = compute_alignment(self._last_frame.get("right"), self._last_frame.get("left"))
            set_pair("Top↔Right", m_tr)
            set_pair("Left↔Right", m_lr)
            if m_lr.valid and M_lr is not None:
                self._stereo_M_right_to_left = M_lr.copy()
                left_img = self._last_frame.get("left")
                if left_img is not None:
                    anag, roi = build_anaglyph_overlap(
                        left_img, self._last_frame.get("right"), M_lr, self._anaglyph_method
                    )
                    if roi is not None:
                        self._stereo_overlap_roi = roi

            m_tl, M_tl = compute_alignment(self._last_frame.get("top"), self._last_frame.get("left"))
            if m_tl.valid and M_tl is not None:
                pass  # will apply consistency fix below
            elif M_tr is not None and M_lr is not None:
                M_tl = _compose_affine(M_tr, M_lr)
                m_tl = _affine_to_metrics(M_tl)
                m_tl.valid = True
            # If Top↔Right failed (e.g. top vs side view), derive from Top↔Left and Left↔Right
            if (not m_tr.valid) and M_tl is not None and M_lr is not None:
                M_tr = _compose_affine(_invert_affine(M_lr), M_tl)
                m_tr = _affine_to_metrics(M_tr)
                m_tr.valid = True
                t = (
                    f"rot={m_tr.rotation_deg:.1f}° scale={m_tr.scale:.3f} "
                    f"tx={m_tr.tx:.0f} ty={m_tr.ty:.0f}px  score={m_tr.score:.0f} (transitive)"
                )
                set_pair("Top↔Right", m_tr, t)
                scores.append(m_tr.score)
            if m_tl.valid and M_tl is not None:
                # Enforce consistency: r_tl ≈ r_tr + r_lr (mod 360). Correct ~180° ambiguity.
                r_tr = _normalize_rotation_deg(m_tr.rotation_deg)
                r_lr = _normalize_rotation_deg(m_lr.rotation_deg)
                r_tl = _normalize_rotation_deg(m_tl.rotation_deg)
                expected_tl = _normalize_rotation_deg(r_tr + r_lr)
                top_img = self._last_frame.get("top")
                right_img = self._last_frame.get("right")
                if top_img is not None and abs(r_tl - expected_tl) > 90:
                    # Top↔Left is ~180° off; apply 180° flip to M_tl (source = top)
                    h_t, w_t = top_img.shape[:2]
                    M_tl = _apply_180_flip_to_transform(M_tl, w_t, h_t)
                    m_tl = _affine_to_metrics(M_tl)
                    r_tl = _normalize_rotation_deg(m_tl.rotation_deg)
                if right_img is not None and M_lr is not None:
                    r_lr_new = _normalize_rotation_deg(m_lr.rotation_deg)
                    expected_lr = _normalize_rotation_deg(r_tl - r_tr)
                    if abs(r_lr_new - expected_lr) > 90:
                        h_r, w_r = right_img.shape[:2]
                        M_lr = _apply_180_flip_to_transform(M_lr, w_r, h_r)
                        m_lr = _affine_to_metrics(M_lr)
                        set_pair("Left↔Right", m_lr)
                        self._stereo_M_right_to_left = M_lr.copy()
                if M_tr is not None and top_img is not None:
                    r_tr_new = _normalize_rotation_deg(m_tr.rotation_deg)
                    expected_tr = _normalize_rotation_deg(r_tl - r_lr)
                    if abs(r_tr_new - expected_tr) > 90:
                        h_t, w_t = top_img.shape[:2]
                        M_tr = _apply_180_flip_to_transform(M_tr, w_t, h_t)
                        m_tr = _affine_to_metrics(M_tr)
                        set_pair("Top↔Right", m_tr)
                self._M_top_to_left = M_tl.copy()
                set_pair("Top↔Left", m_tl)
                scores.append(m_tl.score)
            else:
                set_pair("Top↔Left", m_tl)
        except Exception:
            for key in ("Top↔Left", "Top↔Right", "Left↔Right"):
                if self._align_labels.get(key) and key in last_valid:
                    self._align_labels[key].setText(last_valid[key])

        if scores:
            avg = sum(scores) / len(scores)
            overall = f"Overall: {avg:.0f}/100"
            self._align_overall.setText(overall)
            self._last_valid_overall = overall
        else:
            self._align_overall.setText(getattr(self, "_last_valid_overall", "Overall: —"))

    def _update_anaglyph_and_overlay(self) -> None:
        """Refresh anaglyph (using stored transform) and three-way overlay from last frames."""
        if not hasattr(self, "_overlay_label"):
            return
        try:
            overlay = build_three_way_overlay(
                self._last_frame.get("top"),
                self._last_frame.get("left"),
                self._last_frame.get("right"),
                getattr(self, "_stereo_M_right_to_left", None),
                getattr(self, "_M_top_to_left", None),
            )
            img = cv2_to_qimage(overlay)
            self._overlay_label.setPixmap(
                QPixmap.fromImage(img).scaled(
                    self._overlay_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                )
            )
            self._overlay_label.setStyleSheet("background: #000;")
        except Exception:
            pass
        try:
            M = getattr(self, "_stereo_M_right_to_left", None)
            left_img = self._last_frame.get("left")
            right_img = self._last_frame.get("right")
            if M is not None and left_img is not None and right_img is not None:
                anag, _ = build_anaglyph_overlap(left_img, right_img, M, self._anaglyph_method)
                if anag is not None:
                    img = cv2_to_qimage(anag)
                    self._anaglyph_label.setPixmap(
                        QPixmap.fromImage(img).scaled(
                            self._anaglyph_label.size(),
                            Qt.AspectRatioMode.KeepAspectRatio,
                        )
                    )
                    self._anaglyph_label.setStyleSheet("background: #000;")
        except Exception:
            pass

    def get_stereo_params(self) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
        """Return (M_right_to_left, overlap_roi) for use in 3D mode."""
        return (
            getattr(self, "_stereo_M_right_to_left", None),
            getattr(self, "_stereo_overlap_roi", None),
        )

    def _refresh_static_feeds(self) -> None:
        """Display last stored frame for cameras that just became 'open'."""
        for cam_id in ("left", "right", "top"):
            if self._shutter_open(cam_id) and self._last_frame.get(cam_id) is not None:
                self._display_frame(cam_id, self._last_frame[cam_id])

    def _display_frame(self, cam_id: str, frame: np.ndarray) -> None:
        """Update preview label for cam_id with frame. Handles left/right swap."""
        if frame is None or frame.size == 0:
            return
        # Left and right are switched: left camera -> right preview, right camera -> left preview
        display_map = {
            "left": self._preview_right,
            "right": self._preview_left,
            "top": self._preview_top,
        }
        label = display_map.get(cam_id)
        if not label:
            return
        img = cv2_to_qimage(frame)
        pix = QPixmap.fromImage(img)
        sizes = {"left": (165, 124), "right": (165, 124), "top": (340, 200)}
        w, h = sizes.get(cam_id, (320, 240))
        scaled = pix.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio)
        label.setPixmap(scaled)
        label.setStyleSheet("background: #000;")

    def _update_stoplight(self, cam_id: str, sharpness: float) -> None:
        """Set stoplight color from sharpness. Left/right reversed to match swapped previews."""
        if sharpness >= SHARPNESS_GREEN:
            color = "#22c55e"  # green
        elif sharpness >= SHARPNESS_YELLOW:
            color = "#eab308"  # yellow
        elif sharpness >= SHARPNESS_RED:
            color = "#ef4444"  # red
        else:
            color = "#6b7280"  # gray (poor/no signal)
        # Left camera feed is shown in right preview, right in left preview; match stoplights
        display_id = "right" if cam_id == "left" else ("left" if cam_id == "right" else cam_id)
        indicator = getattr(self, f"_stoplight_{display_id}", None)
        sharpness_lbl = getattr(self, f"_sharpness_lbl_{display_id}", None)
        if indicator:
            indicator.setStyleSheet(f"background: {color}; border-radius: 12px; border: 2px solid #333;")
        if sharpness_lbl:
            sharpness_lbl.setText(f"{int(sharpness)}")

    def _on_frame(self, cam_id: str, frame: np.ndarray) -> None:
        if frame is None or frame.size == 0:
            return
        # Apply rectification if calibration is loaded
        if self._calibration is not None:
            from calibration import apply_rectification

            if cam_id == "left" and self._calibration.map_l is not None:
                frame = apply_rectification(frame, self._calibration.map_l)
            elif cam_id == "right" and self._calibration.map_r is not None:
                frame = apply_rectification(frame, self._calibration.map_r)
        self._last_frame[cam_id] = frame.copy()
        sharpness = compute_sharpness(frame)
        self._update_stoplight(cam_id, sharpness)
        # On init show all feeds; after init only show cameras with shutter open
        if self._init_frames_remaining > 0:
            self._init_frames_remaining -= 1
            self._display_frame(cam_id, frame)
        elif self._shutter_open(cam_id):
            self._display_frame(cam_id, frame)
        # Feed frames to video recorder when recording
        if self._video_recorder is not None and self._video_recorder.is_recording and cam_id == "right":
            self._video_recorder.add_frame(
                self._last_frame.get("left"),
                self._last_frame.get("right"),
                self._stereo_M_right_to_left,
            )

    def _on_lock_toggle(self) -> None:
        """Lock or unlock alignment transforms. When locked, changing slides won't change alignments."""
        self._transforms_locked = not self._transforms_locked
        if self._transforms_locked:
            self._align_timer.stop()
            self._btn_lock.setText("Unlock")
            self._btn_lock.setToolTip("Resume live alignment updates.")
        else:
            self._align_timer.start(500)
            self._btn_lock.setText("Lock")
            self._btn_lock.setToolTip(
                "Freeze alignment transforms so you can change slides without changing alignments."
            )

    def _on_method_changed(self, index: int) -> None:
        """Update anaglyph compositing method from combo box selection."""
        method = self._method_combo.itemData(index)
        if method is not None:
            self._anaglyph_method = method

    def _on_record_toggle(self) -> None:
        """Start or stop video recording."""
        from video_recorder import StereoVideoRecorder

        if self._video_recorder is not None and self._video_recorder.is_recording:
            stats = self._video_recorder.stop()
            self._btn_record.setText("Record")
            self._btn_record.setStyleSheet("color: red; font-weight: bold;")
            self._record_status.setText(f"Saved: {stats.left_frames} frames, {stats.duration_sec:.1f}s")
            QMessageBox.information(
                self,
                "Recording Complete",
                f"Saved to: {stats.output_dir}\n"
                f"Duration: {stats.duration_sec:.1f}s\n"
                f"Left frames: {stats.left_frames}\n"
                f"Right frames: {stats.right_frames}\n"
                f"Dropped: {stats.dropped_frames}\n"
                f"Max drift: {stats.max_drift_ms:.1f}ms",
            )
            return

        left_frame = self._last_frame.get("left")
        if left_frame is None:
            QMessageBox.warning(self, "Cannot Record", "No camera frames available.")
            return
        h, w = left_frame.shape[:2]
        self._video_recorder = StereoVideoRecorder(
            output_dir="./captures",
            anaglyph_method=self._anaglyph_method,
        )
        session_dir = self._video_recorder.start(width=w, height=h, fps=15.0)
        self._btn_record.setText("Stop")
        self._btn_record.setStyleSheet("color: white; background: red; font-weight: bold;")
        self._record_status.setText(f"Recording to {session_dir.name}...")

    def _on_end_program(self) -> None:
        """Quit the application."""
        if self._video_recorder is not None and self._video_recorder.is_recording:
            self._video_recorder.stop()
        QApplication.quit()

    def _save_overlay_jpg(self) -> None:
        """Save current three-way overlay as JPG (file dialog)."""
        path, _ = QFileDialog.getSaveFileName(self, "Save overlay as JPG", "", "JPEG (*.jpg *.jpeg);;All files (*)")
        if not path:
            return
        if not path.lower().endswith((".jpg", ".jpeg")):
            path = path + ".jpg"
        try:
            left_img = self._last_frame.get("left")
            if left_img is None:
                QMessageBox.warning(self, "Save failed", "No left frame available.")
                return
            h_l, w_l = left_img.shape[:2]
            overlay = build_three_way_overlay(
                self._last_frame.get("top"),
                left_img,
                self._last_frame.get("right"),
                getattr(self, "_stereo_M_right_to_left", None),
                getattr(self, "_M_top_to_left", None),
                out_w=w_l,
                out_h=h_l,
            )
            if overlay.size > 0:
                cv2.imwrite(path, overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
                QMessageBox.information(self, "Saved", f"Overlay saved to:\n{path}")
            else:
                QMessageBox.warning(self, "Save failed", "No overlay image available (need frames and transforms).")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))

    def _save_anaglyph_jpg(self) -> None:
        """Save current anaglyph as JPG (file dialog)."""
        path, _ = QFileDialog.getSaveFileName(self, "Save anaglyph as JPG", "", "JPEG (*.jpg *.jpeg);;All files (*)")
        if not path:
            return
        if not path.lower().endswith((".jpg", ".jpeg")):
            path = path + ".jpg"
        M = getattr(self, "_stereo_M_right_to_left", None)
        left_img = self._last_frame.get("left")
        right_img = self._last_frame.get("right")
        if M is None or left_img is None or right_img is None:
            QMessageBox.warning(self, "Save failed", "No anaglyph available (need left/right frames and alignment).")
            return
        try:
            anag, _ = build_anaglyph_overlap(left_img, right_img, M, self._anaglyph_method)
            if anag is not None and anag.size > 0:
                cv2.imwrite(path, anag, [cv2.IMWRITE_JPEG_QUALITY, 95])
                QMessageBox.information(self, "Saved", f"Anaglyph saved to:\n{path}")
            else:
                QMessageBox.warning(self, "Save failed", "Could not build anaglyph image.")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))

    # ------------------------------------------------------------------
    # Still capture
    # ------------------------------------------------------------------

    def _capture_still_set(self) -> None:
        """Capture full-resolution stills of all available frames."""
        from pathlib import Path

        from still_capture import CaptureMetadata, capture_still_set

        left_img = self._last_frame.get("left")
        right_img = self._last_frame.get("right")
        M = getattr(self, "_stereo_M_right_to_left", None)
        anag = None
        if left_img is not None and right_img is not None and M is not None:
            anag, _ = build_anaglyph(left_img, right_img, M, self._anaglyph_method)

        if left_img is None and right_img is None:
            QMessageBox.warning(self, "Capture", "No camera frames available.")
            return

        output_dir = Path("./captures/stills")
        metadata = CaptureMetadata(
            anaglyph_method=self._anaglyph_method.value,
        )
        if M is not None:
            from compositor import affine_to_metrics as _atm

            m = _atm(M)
            metadata.alignment_rotation_deg = m.rotation_deg
            metadata.alignment_scale = m.scale
            metadata.alignment_tx = m.tx
            metadata.alignment_ty = m.ty
        if hasattr(self, "_calibration") and self._calibration is not None:
            metadata.calibration_rms = self._calibration.rms_error

        result = capture_still_set(
            left_frame=left_img,
            right_frame=right_img,
            anaglyph_frame=anag,
            output_dir=output_dir,
            method=self._anaglyph_method.value,
            metadata=metadata,
            formats=(".tiff", ".jpg"),
            top_frame=self._last_frame.get("top"),
        )
        if result is not None:
            QMessageBox.information(
                self,
                "Capture Complete",
                f"Saved {len(result.paths)} files to:\n{output_dir}\n\n"
                f"Formats: TIFF + JPG\nMetadata: {result.metadata_path}",
            )

    # ------------------------------------------------------------------
    # Calibration wizard
    # ------------------------------------------------------------------

    def _start_calibration_wizard(self) -> None:
        """Run the calibration wizard: capture checkerboard poses, then calibrate."""
        from calibration import CalibrationSession, stereo_calibrate

        left_img = self._last_frame.get("left")
        right_img = self._last_frame.get("right")
        if left_img is None or right_img is None:
            QMessageBox.warning(
                self,
                "Calibration",
                "Both left and right cameras must be active.\n"
                "Set the trinocular slider to 'Eyepieces' and wait for frames.",
            )
            return

        # Step 1: Explain the process
        reply = QMessageBox.question(
            self,
            "Stereo Calibration Wizard",
            "This wizard will guide you through stereo calibration.\n\n"
            "You will need a printed checkerboard pattern (9×6 inner corners).\n\n"
            "Steps:\n"
            "1. Place the checkerboard under the stereoscope\n"
            "2. Click 'Capture Pose' to capture (need ≥5 poses)\n"
            "3. Move/rotate the checkerboard between captures\n"
            "4. Click 'Calibrate' when you have enough poses\n\n"
            "Ready to start?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        session = CalibrationSession()
        h, w = left_img.shape[:2]

        # Step 2: Capture poses in a loop
        while True:
            reply = QMessageBox.question(
                self,
                f"Capture Pose ({session.num_poses} captured)",
                f"Poses captured: {session.num_poses}\n"
                f"(Need at least 5 for good calibration, 8-12 recommended)\n\n"
                "Position the checkerboard and click 'Yes' to capture,\n"
                "or 'No' to finish and calibrate.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                break

            left_now = self._last_frame.get("left")
            right_now = self._last_frame.get("right")
            if left_now is None or right_now is None:
                QMessageBox.warning(self, "Calibration", "Camera frames not available. Try again.")
                continue

            left_ok, right_ok = session.add_pose(left_now, right_now)
            if left_ok and right_ok:
                QMessageBox.information(
                    self,
                    "Pose Captured",
                    f"Checkerboard detected in both cameras.\n"
                    f"Total poses: {session.num_poses}\n\n"
                    "Move the checkerboard to a new position/angle for the next pose.",
                )
            else:
                msg_parts = []
                if not left_ok:
                    msg_parts.append("Left camera: checkerboard NOT detected")
                if not right_ok:
                    msg_parts.append("Right camera: checkerboard NOT detected")
                QMessageBox.warning(
                    self,
                    "Detection Failed",
                    "\n".join(msg_parts) + "\n\nMake sure the checkerboard is fully visible and well-lit.",
                )

        # Step 3: Calibrate
        if session.num_poses < 3:
            QMessageBox.warning(
                self,
                "Calibration",
                f"Only {session.num_poses} poses captured (need ≥3).\nCalibration cancelled.",
            )
            return

        result = stereo_calibrate(
            session.left_detections,
            session.right_detections,
            (w, h),
        )
        if result is None:
            QMessageBox.critical(
                self,
                "Calibration Failed",
                "Stereo calibration failed. This can happen if:\n"
                "- The checkerboard was not detected consistently\n"
                "- The poses were too similar (try more variety)\n"
                "- The images are too small or blurry",
            )
            return

        self._calibration = result
        self._calib_status.setText(f"RMS={result.rms_error:.3f} ({result.num_poses} poses)")
        QMessageBox.information(
            self,
            "Calibration Complete",
            f"Stereo calibration successful!\n\n"
            f"RMS reprojection error: {result.rms_error:.4f}\n"
            f"Poses used: {result.num_poses}\n"
            f"Baseline: {np.linalg.norm(result.T):.2f} units\n"
            f"Rectification maps: {'computed' if result.map_l is not None else 'not available'}\n\n"
            "Use Calibration → Save to persist this calibration.",
        )

    def _load_calibration_file(self) -> None:
        """Load a calibration from .npz file."""
        from calibration import load_calibration

        path, _ = QFileDialog.getOpenFileName(self, "Load Calibration", "", "NumPy archive (*.npz);;All files (*)")
        if not path:
            return
        from pathlib import Path as _Path

        result = load_calibration(_Path(path))
        if result is None:
            QMessageBox.critical(self, "Load Failed", f"Could not load calibration from:\n{path}")
            return
        self._calibration = result
        self._calib_status.setText(f"RMS={result.rms_error:.3f} ({result.num_poses} poses)")
        QMessageBox.information(
            self,
            "Calibration Loaded",
            f"Loaded calibration from:\n{path}\n\n"
            f"RMS error: {result.rms_error:.4f}\n"
            f"Image size: {result.image_size}\n"
            f"Poses: {result.num_poses}\n"
            f"Rectification: {'available' if result.map_l is not None else 'not available'}",
        )

    def _save_calibration_file(self) -> None:
        """Save current calibration to .npz file."""
        from calibration import save_calibration

        if not hasattr(self, "_calibration") or self._calibration is None:
            QMessageBox.warning(self, "Save", "No calibration data. Run the wizard first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Calibration", "stereo_calibration.npz", "NumPy archive (*.npz);;All files (*)"
        )
        if not path:
            return
        from pathlib import Path as _Path

        save_calibration(self._calibration, _Path(path))
        QMessageBox.information(self, "Saved", f"Calibration saved to:\n{path}")

    def get_state(self) -> SetupState | None:
        return self._state

    def closeEvent(self, event) -> None:
        if self._video_recorder is not None and self._video_recorder.is_recording:
            self._video_recorder.stop()
        if hasattr(self, "_align_timer"):
            self._align_timer.stop()
        if hasattr(self, "_anaglyph_timer"):
            self._anaglyph_timer.stop()
        if self._worker:
            self._worker.stop()
            self._worker.wait(2000)
        if self._pair:
            self._pair.release()
        event.accept()


def create_main_window() -> QMainWindow | None:
    """Create and return the main application window."""
    if not HAS_PYQT6:
        logger.error("PyQt6 not installed. Run: pip install PyQt6")
        return None
    return CameraSetupWindow()


def run_gui() -> int:
    """Launch the GUI application. Returns exit code."""
    if not HAS_PYQT6:
        print("PyQt6 not installed. Run: pip install PyQt6")
        return 1
    app = QApplication([])
    win = create_main_window()
    if win:
        win.show()
        return app.exec()
    return 1
