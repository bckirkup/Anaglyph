"""
GUI - View and controls for the stereoscope application.

PyQt6-based interface with camera setup prompts for focus and shutter state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal, QTimer
    from PyQt6.QtGui import QImage, QPixmap
    from PyQt6.QtWidgets import (
        QApplication,
        QButtonGroup,
        QCheckBox,
        QFileDialog,
        QFrame,
        QGridLayout,
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
    shutter_top: bool = True  # True = top+right open, False = left+right open


@dataclass
class AlignmentMetrics:
    """Rotation (deg), scale (ratio), translation (px) between two images."""

    rotation_deg: float = 0.0
    scale: float = 1.0
    tx: float = 0.0
    ty: float = 0.0
    score: float = 0.0  # 0-100, higher = better aligned
    valid: bool = False


def _affine_to_metrics(M: np.ndarray) -> AlignmentMetrics:
    """Decompose 2x3 similarity matrix into AlignmentMetrics. Rotation normalized to [-180,180]."""
    out = AlignmentMetrics()
    if M is None or M.size == 0:
        return out
    a, b, tx = M[0]
    c, d, ty = M[1]
    scale = np.sqrt(a * a + c * c)
    rotation_rad = np.arctan2(c, a)
    rotation_deg = _normalize_rotation_deg(np.degrees(rotation_rad))
    out.rotation_deg = float(rotation_deg)
    out.scale = float(scale)
    out.tx = float(tx)
    out.ty = float(ty)
    out.valid = True
    rot_penalty = min(abs(rotation_deg) * 2, 50)
    scale_penalty = min(abs(1.0 - scale) * 100, 30)
    trans_penalty = min(np.sqrt(tx * tx + ty * ty) * 0.1, 20)
    out.score = max(0, 100 - rot_penalty - scale_penalty - trans_penalty)
    return out


def _compose_affine(M1: np.ndarray, M2: np.ndarray) -> np.ndarray:
    """Compose two 2x3 affine matrices: result maps via M1 then M2 (M2 @ M1)."""
    R1, t1 = M1[:, :2], M1[:, 2:3]
    R2, t2 = M2[:, :2], M2[:, 2:3]
    R = R2 @ R1
    t = R2 @ t1 + t2
    return np.hstack([R, t])


def _invert_affine(M: np.ndarray) -> np.ndarray:
    """Invert a 2x3 similarity matrix."""
    R, t = M[:, :2], M[:, 2:3]
    R_inv = np.linalg.inv(R)
    t_inv = -R_inv @ t
    return np.hstack([R_inv, t_inv])


def _normalize_rotation_deg(deg: float) -> float:
    """Normalize angle to [-180, 180]."""
    d = deg % 360.0
    if d > 180:
        d -= 360
    elif d <= -180:
        d += 360
    return d


def _apply_180_flip_to_transform(M: np.ndarray, src_w: int, src_h: int) -> np.ndarray:
    """Apply 180° rotation around source image center. M maps src -> dst; result is M ∘ F with F = 180° around (src_w/2, src_h/2)."""
    R, t = M[:, :2], M[:, 2:3]
    cx, cy = src_w / 2.0, src_h / 2.0
    # F: p' = -p + [2*cx, 2*cy]. So F = [-I | 2*c]
    R_flip = -np.eye(2)
    t_flip = np.array([[2 * cx], [2 * cy]])
    R_new = R @ R_flip
    t_new = R @ t_flip + t
    return np.hstack([R_new, t_new])


def compute_alignment(img1: np.ndarray, img2: np.ndarray) -> tuple[AlignmentMetrics, Optional[np.ndarray]]:
    """
    Estimate similarity transform (rotation, scale, translation) from img1 to img2.
    Returns (metrics, 2x3 matrix in original image coords) or (invalid, None).
    """
    out = AlignmentMetrics()
    if img1 is None or img2 is None or img1.size == 0 or img2.size == 0:
        return out, None
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
    h1, w1 = gray1.shape
    h2, w2 = gray2.shape
    if min(h1, w1, h2, w2) < 32:
        return out, None
    target = 320
    r1, r2 = 1.0, 1.0
    if max(w1, h1) > target or max(w2, h2) > target:
        r1 = target / max(w1, h1)
        r2 = target / max(w2, h2)
        gray1 = cv2.resize(gray1, None, fx=r1, fy=r1)
        gray2 = cv2.resize(gray2, None, fx=r2, fy=r2)
    orb = cv2.ORB_create(nfeatures=800)
    kp1, d1 = orb.detectAndCompute(gray1, None)
    kp2, d2 = orb.detectAndCompute(gray2, None)
    if d1 is None or d2 is None or len(kp1) < 4 or len(kp2) < 4:
        return out, None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(d1, d2)
    if len(matches) < 4:
        return out, None
    matches = sorted(matches, key=lambda m: m.distance)[:80]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    try:
        M, inliers = cv2.estimateAffinePartial2D(pts1, pts2)
    except cv2.error:
        return out, None
    if M is None:
        return out, None
    # Scale M from resized coords to full image coords: p2_resized = M @ p1_resized, p1_resized = r1*p1_full, p2_full = p2_resized/r2 => M_full: R_full = (r1/r2)*R, t_full = t/r2
    if r1 != 1.0 or r2 != 1.0:
        scale = r1 / r2
        M = np.array([[M[0, 0] * scale, M[0, 1] * scale, M[0, 2] / r2],
                      [M[1, 0] * scale, M[1, 1] * scale, M[1, 2] / r2]], dtype=np.float64)
    out = _affine_to_metrics(M)
    return out, M


def build_anaglyph_overlap(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    M_right_to_left: np.ndarray,
) -> tuple[Optional[np.ndarray], Optional[tuple[int, int, int, int]]]:
    """
    Build red-cyan anaglyph for red/cyan glasses. Left eye sees red (left image),
    right eye sees cyan (right image). Crop to overlap so no big single-color regions.
    M_right_to_left: 2x3 affine mapping right image coords to left image coords.
    Returns (anaglyph_bgr, roi (x,y,w,h)) or (None, None).
    """
    if left_bgr is None or right_bgr is None or M_right_to_left is None:
        return None, None
    h_l, w_l = left_bgr.shape[:2]
    h_r, w_r = right_bgr.shape[:2]
    if min(h_l, w_l, h_r, w_r) < 8:
        return None, None
    right_warped = cv2.warpAffine(right_bgr, M_right_to_left, (w_l, h_l))
    left_g = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    right_g = cv2.cvtColor(right_warped, cv2.COLOR_BGR2GRAY)
    # Red-cyan encoding: R = left (red filter = left eye), G = B = right (cyan = right eye)
    out = np.zeros((h_l, w_l, 3), dtype=np.uint8)
    out[:, :, 0] = right_g   # B
    out[:, :, 1] = right_g   # G  -> cyan = right image for right eye
    out[:, :, 2] = left_g    # R  -> red = left image for left eye

    # Crop to overlap region so there are no large areas of only red or only cyan
    right_has_content = right_g > 16
    if not np.any(right_has_content):
        roi = (0, 0, w_l, h_l)
        return out, roi
    ys, xs = np.where(right_has_content)
    x_min, x_max = int(np.clip(xs.min(), 0, w_l - 1)), int(np.clip(xs.max() + 1, 1, w_l))
    y_min, y_max = int(np.clip(ys.min(), 0, h_l - 1)), int(np.clip(ys.max() + 1, 1, h_l))
    # Slight inset to avoid thin single-color edges
    margin = min(4, (x_max - x_min) // 8, (y_max - y_min) // 8)
    x_min = min(x_min + margin, x_max - 1)
    y_min = min(y_min + margin, y_max - 1)
    x_max = max(x_max - margin, x_min + 1)
    y_max = max(y_max - margin, y_min + 1)
    roi = (x_min, y_min, x_max - x_min, y_max - y_min)
    out = out[y_min:y_max, x_min:x_max]
    return out, roi


def build_three_way_overlay(
    top_bgr: Optional[np.ndarray],
    left_bgr: Optional[np.ndarray],
    right_bgr: Optional[np.ndarray],
    M_right_to_left: Optional[np.ndarray],
    M_top_to_left: Optional[np.ndarray],
    out_w: int = 420,
    out_h: int = 280,
) -> np.ndarray:
    """
    All three images entirely displayed and optimally overlaid in a common frame.
    Uses left as reference; warps right and top into left space (or resizes if no transform), then blends.
    """
    if left_bgr is None:
        out = np.ones((out_h, out_w), dtype=np.uint8) * 32
        return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    h_l, w_l = left_bgr.shape[:2]
    left_g = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    blend = left_g.astype(np.float32)
    count = np.ones(left_g.shape, dtype=np.float32)
    if right_bgr is not None:
        if M_right_to_left is not None:
            right_w = cv2.warpAffine(right_bgr, M_right_to_left, (w_l, h_l))
        else:
            right_w = cv2.resize(right_bgr, (w_l, h_l))
        right_g = cv2.cvtColor(right_w, cv2.COLOR_BGR2GRAY)
        blend += right_g.astype(np.float32)
        count += 1.0
    if top_bgr is not None:
        if M_top_to_left is not None:
            top_w = cv2.warpAffine(top_bgr, M_top_to_left, (w_l, h_l))
        else:
            top_w = cv2.resize(top_bgr, (w_l, h_l))
        top_g = cv2.cvtColor(top_w, cv2.COLOR_BGR2GRAY)
        blend += top_g.astype(np.float32)
        count += 1.0
    out = np.clip(blend / np.maximum(count, 1), 0, 255).astype(np.uint8)
    out = cv2.resize(out, (out_w, out_h))
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def compute_sharpness(frame: np.ndarray) -> float:
    """Laplacian variance - higher = sharper. Used for focus indicator."""
    if frame is None or frame.size == 0:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    return float(cv2.Laplacian(gray, cv2.CV_64F, ksize=3).var())


def cv2_to_qimage(frame: np.ndarray) -> QImage:
    """Convert OpenCV BGR frame to QImage for display."""
    if frame is None or frame.size == 0:
        return QImage()
    h, w, ch = frame.shape
    bytes_per_line = ch * w
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)


# Sharpness thresholds for stoplight (Laplacian variance)
SHARPNESS_RED = 100   # Below: poor focus
SHARPNESS_YELLOW = 300  # Red–Yellow: improving; Yellow–Green: good
SHARPNESS_GREEN = 500   # Above: excellent focus


class CaptureWorker(QThread):
    """Background thread that grabs frames from cameras."""

    frame_ready = pyqtSignal(str, object)  # camera_id, frame (numpy array)

    def __init__(
        self,
        left_cap: Optional[cv2.VideoCapture],
        right_cap: Optional[cv2.VideoCapture],
        top_cap: Optional[cv2.VideoCapture],
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
        self._worker: Optional[CaptureWorker] = None
        self._state = SetupState()
        self._last_frame: dict[str, Optional[np.ndarray]] = {
            "left": None,
            "right": None,
            "top": None,
        }
        self._init_frames_remaining = 180  # Show all feeds for ~2–3 sec on startup
        self._last_valid_align_text: dict[str, str] = {}
        self._last_valid_overall: str = "Overall: —"
        self._stereo_M_right_to_left: Optional[np.ndarray] = None
        self._stereo_overlap_roi: Optional[tuple[int, int, int, int]] = None
        self._M_top_to_left: Optional[np.ndarray] = None
        self._transforms_locked: bool = False
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
        align_group.setToolTip("Top↔Right updates when 'Top and Right' shutter is selected; Left↔Right when 'Left and Right'.")
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
        indicator.setStyleSheet(
            "background: #333; border-radius: 12px; border: 2px solid #555;"
        )
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
        """Add menu bar: File (End program, Save overlay/anaglyph as JPG)."""
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

        # Shutter radio: Top+Right open together, or Left+Right open together
        shutter_group = QGroupBox("Which has the shutter open?")
        shutter_layout = QVBoxLayout(shutter_group)
        self._shutter_group = QButtonGroup(self)
        self._rb_shutter_top_right = QRadioButton("Top and Right")
        self._rb_shutter_top_right.setChecked(True)
        self._rb_shutter_left_right = QRadioButton("Left and Right")
        self._shutter_group.addButton(self._rb_shutter_top_right)
        self._shutter_group.addButton(self._rb_shutter_left_right)
        self._rb_shutter_top_right.toggled.connect(self._on_shutter_changed)
        self._rb_shutter_left_right.toggled.connect(self._on_shutter_changed)
        shutter_layout.addWidget(self._rb_shutter_top_right)
        shutter_layout.addWidget(self._rb_shutter_left_right)
        layout.addWidget(shutter_group)

        # Lock / Unlock transforms (replaces Continue)
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self._btn_lock = QPushButton("Lock")
        self._btn_lock.setMinimumWidth(120)
        self._btn_lock.setToolTip("Freeze alignment transforms so you can change slides without changing alignments.")
        self._btn_lock.clicked.connect(self._on_lock_toggle)
        btn_layout.addWidget(self._btn_lock)
        layout.addLayout(btn_layout)

    def _init_cameras(self) -> None:
        from camera_manager import CameraManager, StereoPair

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
        if self._rb_shutter_top_right.isChecked():
            return cam_id in ("top", "right")
        return cam_id in ("left", "right")

    def _on_shutter_changed(self) -> None:
        """When shutter selection changes, show last frame for newly opened cameras."""
        self._refresh_static_feeds()

    def _both_cams_live_for_pair(self, pair_key: str) -> bool:
        """True if both cameras in this pair have shutter open (so alignment is meaningful)."""
        top_right_open = self._rb_shutter_top_right.isChecked()
        if pair_key == "Top↔Right":
            return top_right_open
        if pair_key == "Left↔Right":
            return not top_right_open
        # Top↔Left: never both live (we have Top+Right or Left+Right only)
        return False

    def _update_alignment(self) -> None:
        """Compute and display alignment metrics. Keep last valid when static (persistent metrics)."""
        if getattr(self, "_transforms_locked", False):
            return
        if not hasattr(self, "_align_labels"):
            return
        scores = []
        last_valid = self._last_valid_align_text
        top_right_open = self._rb_shutter_top_right.isChecked()

        def set_pair(key: str, m: AlignmentMetrics, text_override: Optional[str] = None) -> None:
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
                t = (
                    f"rot={m.rotation_deg:.1f}° scale={m.scale:.3f} "
                    f"tx={m.tx:.0f} ty={m.ty:.0f}px  score={m.score:.0f}"
                )
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
                    if key == "Top↔Right":
                        self._align_labels[key].setText(last_valid.get(key, "—") + " (use 'Top and Right' shutter)")
                    elif key == "Left↔Right":
                        self._align_labels[key].setText(last_valid.get(key, "—") + " (use 'Left and Right' shutter)")
                    else:
                        self._align_labels[key].setText(last_valid.get(key, "— (insufficient features)"))

        try:
            m_tr, M_tr = compute_alignment(
                self._last_frame.get("top"), self._last_frame.get("right")
            )
            if not m_tr.valid and M_tr is None:
                m_rt, M_rt = compute_alignment(
                    self._last_frame.get("right"), self._last_frame.get("top")
                )
                if M_rt is not None:
                    M_tr = _invert_affine(M_rt)
                    m_tr = _affine_to_metrics(M_tr)
                    m_tr.valid = True
            m_lr, M_lr = compute_alignment(
                self._last_frame.get("right"), self._last_frame.get("left")
            )
            set_pair("Top↔Right", m_tr)
            set_pair("Left↔Right", m_lr)
            if m_lr.valid and M_lr is not None:
                self._stereo_M_right_to_left = M_lr.copy()
                left_img = self._last_frame.get("left")
                if left_img is not None:
                    anag, roi = build_anaglyph_overlap(
                        left_img, self._last_frame.get("right"), M_lr
                    )
                    if roi is not None:
                        self._stereo_overlap_roi = roi

            m_tl, M_tl = compute_alignment(
                self._last_frame.get("top"), self._last_frame.get("left")
            )
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
                anag, _ = build_anaglyph_overlap(left_img, right_img, M)
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

    def get_stereo_params(self) -> tuple[Optional[np.ndarray], Optional[tuple[int, int, int, int]]]:
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
            indicator.setStyleSheet(
                f"background: {color}; border-radius: 12px; border: 2px solid #333;"
            )
        if sharpness_lbl:
            sharpness_lbl.setText(f"{int(sharpness)}")

    def _on_frame(self, cam_id: str, frame: np.ndarray) -> None:
        if frame is None or frame.size == 0:
            return
        self._last_frame[cam_id] = frame.copy()
        sharpness = compute_sharpness(frame)
        self._update_stoplight(cam_id, sharpness)
        # On init show all feeds; after init only show cameras with shutter open
        if self._init_frames_remaining > 0:
            self._init_frames_remaining -= 1
            self._display_frame(cam_id, frame)
        elif self._shutter_open(cam_id):
            self._display_frame(cam_id, frame)

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
            self._btn_lock.setToolTip("Freeze alignment transforms so you can change slides without changing alignments.")

    def _on_end_program(self) -> None:
        """Quit the application."""
        QApplication.quit()

    def _save_overlay_jpg(self) -> None:
        """Save current three-way overlay as JPG (file dialog)."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save overlay as JPG", "", "JPEG (*.jpg *.jpeg);;All files (*)"
        )
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
        path, _ = QFileDialog.getSaveFileName(
            self, "Save anaglyph as JPG", "", "JPEG (*.jpg *.jpeg);;All files (*)"
        )
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
            anag, _ = build_anaglyph_overlap(left_img, right_img, M)
            if anag is not None and anag.size > 0:
                cv2.imwrite(path, anag, [cv2.IMWRITE_JPEG_QUALITY, 95])
                QMessageBox.information(self, "Saved", f"Anaglyph saved to:\n{path}")
            else:
                QMessageBox.warning(self, "Save failed", "Could not build anaglyph image.")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))

    def get_state(self) -> Optional[SetupState]:
        return self._state

    def closeEvent(self, event) -> None:
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


def create_main_window() -> Optional[QMainWindow]:
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
