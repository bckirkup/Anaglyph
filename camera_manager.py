"""
Camera Manager - Hardware access for stereoscope dual-camera setup.

Discovers cameras via VID/PID to ensure consistent Left/Right assignment across restarts.
Uses cv2-enumerate-cameras for reliable identification on Windows.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2

# Optional: cv2-enumerate-cameras for VID/PID discovery (recommended on Windows)
try:
    from cv2_enumerate_cameras import enumerate_cameras
    HAS_ENUMERATE_CAMERAS = True
except ImportError:
    HAS_ENUMERATE_CAMERAS = False

logger = logging.getLogger(__name__)

# AmScope MD500L uses VID 0x0AC8 (USB imaging chip); MD1900 uses 0x0547
# Support both for flexibility across AmScope product line
AMSCOPE_VIDS = (0x0AC8, 0x0547)
CONFIG_FILENAME = "camera_config.json"


@dataclass
class CameraInfo:
    """Identifies a camera for Left/Right assignment."""

    index: int
    backend: int
    name: str
    vid: Optional[int] = None
    pid: Optional[int] = None
    path: Optional[str] = None
    key: str = ""  # Unique key: vid:pid:path or index for fallback

    def __post_init__(self) -> None:
        if not self.key:
            vid = self.vid if self.vid is not None else 0
            pid = self.pid if self.pid is not None else 0
            path = self.path or ""
            self.key = f"{vid:04X}:{pid:04X}:{path}"


@dataclass
class StereoPair:
    """Left and Right camera info for the stereoscope."""

    left: Optional[CameraInfo] = None
    right: Optional[CameraInfo] = None
    left_capture: Optional[cv2.VideoCapture] = field(default=None, repr=False)
    right_capture: Optional[cv2.VideoCapture] = field(default=None, repr=False)

    def release(self) -> None:
        """Release both camera captures."""
        for cap in (self.left_capture, self.right_capture):
            if cap is not None:
                cap.release()
        self.left_capture = None
        self.right_capture = None


class CameraManager:
    """
    Discovers and manages two cameras for stereoscopic capture.
    Persists Left/Right mapping by VID/PID/path so eyepieces don't swap on restart.
    """

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self.config_path = config_path or Path(__file__).parent / CONFIG_FILENAME
        self._config: dict = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load persisted Left/Right camera keys from config."""
        if self.config_path.exists():
            try:
                with open(self.config_path, encoding="utf-8") as f:
                    self._config = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Could not load camera config: %s", e)
                self._config = {}
        else:
            self._config = {}

    def _save_config(self) -> None:
        """Persist Left/Right camera keys to config."""
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self._config, f, indent=2)
        except OSError as e:
            logger.warning("Could not save camera config: %s", e)

    def discover_cameras(self) -> list[CameraInfo]:
        """
        Enumerate all available cameras with VID/PID when possible.
        Returns list of CameraInfo sorted for stable ordering.
        """
        cameras: list[CameraInfo] = []

        if HAS_ENUMERATE_CAMERAS:
            try:
                # Use MSMF on Windows for better compatibility
                backend = cv2.CAP_MSMF
                for info in enumerate_cameras(backend):
                    ci = CameraInfo(
                        index=info.index,
                        backend=info.backend,
                        name=info.name or f"Camera {info.index}",
                        vid=info.vid if hasattr(info, "vid") else None,
                        pid=info.pid if hasattr(info, "pid") else None,
                        path=info.path if hasattr(info, "path") else None,
                    )
                    cameras.append(ci)
            except Exception as e:
                logger.warning("cv2_enumerate_cameras failed: %s", e)
                cameras = self._fallback_discover()
        else:
            cameras = self._fallback_discover()

        # Sort by key for stable ordering
        cameras.sort(key=lambda c: c.key)
        return cameras

    def _fallback_discover(self) -> list[CameraInfo]:
        """Fallback when cv2-enumerate-cameras is not available."""
        cameras = []
        for i in range(10):
            cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
            if cap.isOpened():
                cameras.append(
                    CameraInfo(
                        index=i,
                        backend=cv2.CAP_MSMF,
                        name=f"Camera {i}",
                        key=f"fallback:{i}",
                    )
                )
                cap.release()
        return cameras

    def get_stereo_pair(
        self,
        left_key: Optional[str] = None,
        right_key: Optional[str] = None,
        prefer_amscope: bool = True,
    ) -> StereoPair:
        """
        Resolve Left and Right cameras from discovered list and persisted config.

        Args:
            left_key: Override persisted left camera key.
            right_key: Override persisted right camera key.
            prefer_amscope: If True, filter to AmScope VID when multiple cameras exist.

        Returns:
            StereoPair with left/right CameraInfo (and optionally opened captures).
        """
        all_cams = self.discover_cameras()

        # Filter to AmScope if requested and we have VID/PID
        if prefer_amscope and any(c.vid for c in all_cams):
            amscope = [c for c in all_cams if c.vid in AMSCOPE_VIDS]
            if len(amscope) >= 2:
                all_cams = amscope

        if len(all_cams) < 2:
            logger.warning(
                "Found %d camera(s); need 2 for stereoscope. Connect both AmScope MD500L cameras.",
                len(all_cams),
            )
            return StereoPair()

        # Resolve keys from config or args
        left_key = left_key or self._config.get("left_key")
        right_key = right_key or self._config.get("right_key")

        left_info = self._find_by_key(all_cams, left_key, 0)
        right_info = self._find_by_key(all_cams, right_key, 1)

        # Ensure we don't assign same camera to both
        if left_info and right_info and left_info.key == right_info.key:
            right_info = self._find_by_key(all_cams, None, 1)

        # Persist for next run
        if left_info and right_info:
            self._config["left_key"] = left_info.key
            self._config["right_key"] = right_info.key
            self._save_config()

        return StereoPair(left=left_info, right=right_info)

    def _find_by_key(
        self,
        cameras: list[CameraInfo],
        key: Optional[str],
        default_index: int,
    ) -> Optional[CameraInfo]:
        """Find camera by persisted key, or use default index."""
        if key:
            for c in cameras:
                if c.key == key:
                    return c
        if 0 <= default_index < len(cameras):
            return cameras[default_index]
        return None

    def open_captures(
        self,
        pair: StereoPair,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[float] = None,
    ) -> bool:
        """
        Open VideoCapture for both cameras. Call release() when done.

        Args:
            pair: StereoPair with left/right CameraInfo.
            width: Desired frame width (optional).
            height: Desired frame height (optional).
            fps: Desired FPS (optional).

        Returns:
            True if both captures opened successfully.
        """
        pair.release()

        if not pair.left or not pair.right:
            return False

        def open_one(info: CameraInfo) -> Optional[cv2.VideoCapture]:
            cap = cv2.VideoCapture(info.index, info.backend)
            if not cap.isOpened():
                logger.error("Failed to open camera: %s", info.name)
                return None
            if width is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height is not None:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if fps is not None:
                cap.set(cv2.CAP_PROP_FPS, fps)
            return cap

        pair.left_capture = open_one(pair.left)
        pair.right_capture = open_one(pair.right)

        ok = pair.left_capture is not None and pair.right_capture is not None
        if ok:
            logger.info(
                "Opened Left: %s, Right: %s",
                pair.left.name,
                pair.right.name,
            )
        return ok

    def swap_left_right(self, pair: StereoPair) -> None:
        """Swap Left and Right assignment and persist."""
        pair.left, pair.right = pair.right, pair.left
        pair.left_capture, pair.right_capture = pair.right_capture, pair.left_capture
        if pair.left and pair.right:
            self._config["left_key"] = pair.left.key
            self._config["right_key"] = pair.right.key
            self._save_config()


def verify_hardware_access() -> bool:
    """
    Quick verification that cameras are accessible.
    Call this at startup to confirm hardware before launching the full app.

    Returns:
        True if at least two cameras can be opened and read a frame.
    """
    mgr = CameraManager()
    pair = mgr.get_stereo_pair(prefer_amscope=False)

    if not pair.left or not pair.right:
        print("ERROR: Need 2 cameras. Found:", len(mgr.discover_cameras()))
        return False

    if not mgr.open_captures(pair):
        print("ERROR: Could not open both cameras.")
        return False

    try:
        ok_l, _ = pair.left_capture.read()
        ok_r, _ = pair.right_capture.read()
        if not ok_l or not ok_r:
            print("ERROR: Could not read a frame from both cameras.")
            return False
        print("OK: Both cameras accessible.")
        print("  Left:", pair.left.name, f"({pair.left.key})")
        print("  Right:", pair.right.name, f"({pair.right.key})")
        return True
    finally:
        pair.release()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = verify_hardware_access()
    exit(0 if success else 1)
