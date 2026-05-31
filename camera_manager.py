"""
Camera Manager - Hardware access for stereoscope dual-camera setup.

Discovers cameras via VID/PID to ensure consistent Left/Right assignment across restarts.
Uses cv2-enumerate-cameras for reliable identification on Windows.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2

# Optional: cv2-enumerate-cameras for VID/PID discovery (recommended on Windows)
try:
    from cv2_enumerate_cameras import enumerate_cameras

    HAS_ENUMERATE_CAMERAS = True
except ImportError:
    HAS_ENUMERATE_CAMERAS = False

logger = logging.getLogger(__name__)

# AmScope camera VIDs: MD500L uses 0x0AC8 (imaging chip); MD1900/MU503 use 0x0547
# MU503 (USB3.0 Camera, top-down) uses VID 0547, PIDs 3510/3511/4510/4511
AMSCOPE_VIDS = (0x0AC8, 0x0547)
MU503_PIDS = (0x3510, 0x3511, 0x4510, 0x4511)  # AmScope MU503 variants
CONFIG_FILENAME = "camera_config.json"


@dataclass
class CameraInfo:
    """Identifies a camera for Left/Right assignment."""

    index: int
    backend: int
    name: str
    vid: int | None = None
    pid: int | None = None
    path: str | None = None
    key: str = ""  # Unique key: vid:pid:path or index for fallback

    def __post_init__(self) -> None:
        if not self.key:
            vid = self.vid if self.vid is not None else 0
            pid = self.pid if self.pid is not None else 0
            path = self.path or ""
            self.key = f"{vid:04X}:{pid:04X}:{path}"

    def is_mu503(self) -> bool:
        """True if this camera is likely an AmScope MU503 (top-down USB3.0)."""
        return self.vid == 0x0547 and self.pid in MU503_PIDS

    def is_usb3_camera(self) -> bool:
        """True if name suggests USB3.0 Camera (common MU503 label)."""
        return self.name and "USB3.0" in self.name.upper()

    def is_mu503_by_name(self) -> bool:
        """True if name suggests AmScope MU503 (USB2.0 or USB3.0 mode)."""
        return self.name and "MU503" in self.name.upper()


@dataclass
class StereoPair:
    """Left and Right camera info for the stereoscope, plus optional Top (top-down)."""

    left: CameraInfo | None = None
    right: CameraInfo | None = None
    top: CameraInfo | None = None  # Optional: MU503 or other top-down camera
    left_capture: cv2.VideoCapture | None = field(default=None, repr=False)
    right_capture: cv2.VideoCapture | None = field(default=None, repr=False)
    top_capture: cv2.VideoCapture | None = field(default=None, repr=False)

    def release(self) -> None:
        """Release all camera captures."""
        for cap in (self.left_capture, self.right_capture, self.top_capture):
            if cap is not None:
                cap.release()
        self.left_capture = None
        self.right_capture = None
        self.top_capture = None


class CameraManager:
    """
    Discovers and manages two cameras for stereoscopic capture.
    Persists Left/Right mapping by VID/PID/path so eyepieces don't swap on restart.
    """

    def __init__(self, config_path: Path | None = None) -> None:
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

    def discover_cameras(self, include_dshow: bool = True) -> list[CameraInfo]:
        """
        Enumerate all available cameras with VID/PID when possible.
        Uses DSHOW in addition to MSMF so MU503 (often DSHOW-only) is discovered.

        Returns list of CameraInfo sorted for stable ordering, deduplicated by path.
        """
        cameras: list[CameraInfo] = []
        seen_paths: set[str] = set()

        def add_unique(info) -> None:
            path = (info.path or "").strip()
            vid = info.vid if hasattr(info, "vid") else None
            pid = info.pid if hasattr(info, "pid") else None
            # Dedupe by device instance (path up to #{GUID} - same physical camera)
            dedupe_key = path.split("#{")[0] if path else f"{vid}:{pid}:{id(info)}"
            if dedupe_key and dedupe_key in seen_paths:
                return
            if dedupe_key:
                seen_paths.add(dedupe_key)
            ci = CameraInfo(
                index=info.index,
                backend=info.backend,
                name=info.name or f"Camera {info.index}",
                vid=vid,
                pid=pid,
                path=path or None,
            )
            cameras.append(ci)

        if HAS_ENUMERATE_CAMERAS:
            try:
                # MSMF first (better for MD500L)
                for info in enumerate_cameras(cv2.CAP_MSMF):
                    add_unique(info)
                # DSHOW for MU503 and other cameras that don't appear in MSMF
                if include_dshow:
                    for info in enumerate_cameras(cv2.CAP_DSHOW):
                        add_unique(info)
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
        left_key: str | None = None,
        right_key: str | None = None,
        top_key: str | None = None,
        prefer_amscope: bool = True,
        include_top: bool = True,
    ) -> StereoPair:
        """
        Resolve Left, Right, and optional Top cameras from discovered list and persisted config.

        Args:
            left_key: Override persisted left camera key.
            right_key: Override persisted right camera key.
            top_key: Override persisted top camera key (MU503 / top-down).
            prefer_amscope: If True, filter to AmScope VID when multiple cameras exist.
            include_top: If True, assign third camera as top when 3+ cameras found.

        Returns:
            StereoPair with left/right CameraInfo (and optionally top).
        """
        all_cams = self.discover_cameras()

        # Filter to AmScope if requested (MD500L, MU503 "USB3.0 Camera", MU503(USB2.0))
        if prefer_amscope:
            amscope = [
                c
                for c in all_cams
                if c.vid in AMSCOPE_VIDS or (c.name and ("USB3.0" in c.name.upper() or "MU503" in c.name.upper()))
            ]
            if len(amscope) >= 2:
                all_cams = amscope

        if len(all_cams) < 2:
            logger.warning(
                "Found %d camera(s); need 2 for stereoscope. Connect both AmScope MD500L cameras.",
                len(all_cams),
            )
            return StereoPair()

        # Identify top-down camera (MU503) vs stereo pair (MD500L)
        top_candidates = [c for c in all_cams if c.is_mu503() or c.is_usb3_camera() or c.is_mu503_by_name()]
        stereo_candidates = [c for c in all_cams if c not in top_candidates]
        if len(stereo_candidates) < 2:
            stereo_candidates = all_cams  # Fallback if no clear MU503

        # Resolve keys from config or args
        left_key = left_key or self._config.get("left_key")
        right_key = right_key or self._config.get("right_key")
        top_key = top_key or self._config.get("top_key") if include_top else None

        # Left/Right from stereo pair (MD500L); Top from MU503 when present
        left_info = self._find_by_key(stereo_candidates, left_key, 0)
        right_info = self._find_by_key(stereo_candidates, right_key, 1)

        # Ensure we don't assign same camera to both
        if left_info and right_info and left_info.key == right_info.key:
            right_info = self._find_by_key(stereo_candidates, None, 1)

        # Top camera: MU503 or third camera when 3+ available
        top_info = None
        if include_top:
            if top_key:
                top_info = self._find_by_key(all_cams, top_key, 0)
            # Don't use a stereo camera as top when we have a dedicated MU503
            if top_info in stereo_candidates and top_candidates:
                top_info = None
            if not top_info and top_candidates:
                top_info = top_candidates[0]
            elif not top_info and len(all_cams) >= 3:
                remaining = [c for c in all_cams if c not in (left_info, right_info)]
                if remaining:
                    top_info = remaining[0]

        # Persist for next run
        if left_info and right_info:
            self._config["left_key"] = left_info.key
            self._config["right_key"] = right_info.key
            if top_info:
                self._config["top_key"] = top_info.key
            elif "top_key" in self._config and not top_info:
                self._config.pop("top_key", None)
            self._save_config()

        return StereoPair(left=left_info, right=right_info, top=top_info)

    def _find_by_key(
        self,
        cameras: list[CameraInfo],
        key: str | None,
        default_index: int,
    ) -> CameraInfo | None:
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
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
    ) -> bool:
        """
        Open VideoCapture for left, right, and optional top cameras. Call release() when done.

        Args:
            pair: StereoPair with left/right (and optionally top) CameraInfo.
            width: Desired frame width (optional).
            height: Desired frame height (optional).
            fps: Desired FPS (optional).

        Returns:
            True if at least both left and right captures opened successfully.
        """
        pair.release()

        if not pair.left or not pair.right:
            return False

        def open_one(info: CameraInfo, retries: int = 2) -> cv2.VideoCapture | None:
            for attempt in range(retries):
                cap = cv2.VideoCapture(info.index, info.backend)
                if cap.isOpened():
                    if width is not None:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    if height is not None:
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    if fps is not None:
                        cap.set(cv2.CAP_PROP_FPS, fps)
                    return cap
                cap.release()
                if attempt < retries - 1:
                    time.sleep(0.25)
            logger.error("Failed to open camera: %s (index=%s, backend=%s)", info.name, info.index, info.backend)
            return None

        # Open top first (often DSHOW/MU503), then left/right, so all get a chance to init
        if pair.top:
            pair.top_capture = open_one(pair.top)
            if pair.top_capture:
                time.sleep(0.2)
        pair.left_capture = open_one(pair.left)
        if pair.left_capture:
            time.sleep(0.2)
        pair.right_capture = open_one(pair.right)

        # Warmup: one read per camera so drivers settle
        for cap in (pair.top_capture, pair.left_capture, pair.right_capture):
            if cap is not None:
                cap.read()
                time.sleep(0.05)
        time.sleep(0.1)

        ok = pair.left_capture is not None and pair.right_capture is not None
        if ok:
            msg = f"Opened Left: {pair.left.name}, Right: {pair.right.name}"
            if pair.top_capture:
                msg += f", Top: {pair.top.name}"
            logger.info(msg)
        return ok

    def swap_left_right(self, pair: StereoPair) -> None:
        """Swap Left and Right assignment and persist."""
        pair.left, pair.right = pair.right, pair.left
        pair.left_capture, pair.right_capture = pair.right_capture, pair.left_capture
        if pair.left and pair.right:
            self._config["left_key"] = pair.left.key
            self._config["right_key"] = pair.right.key
            self._save_config()


def verify_hardware_access(include_top: bool = True) -> bool:
    """
    Quick verification that cameras are accessible.
    Call this at startup to confirm hardware before launching the full app.

    Args:
        include_top: If True, discover and test optional top-down (MU503) camera.

    Returns:
        True if at least two cameras (left + right) can be opened and read a frame.
    """
    mgr = CameraManager()
    pair = mgr.get_stereo_pair(prefer_amscope=False, include_top=include_top)

    if not pair.left or not pair.right:
        found = len(mgr.discover_cameras())
        print("ERROR: Need 2 cameras for stereo. Found:", found)
        return False

    if not mgr.open_captures(pair):
        print("ERROR: Could not open both stereo cameras.")
        return False

    try:
        ok_l, _ = pair.left_capture.read()
        ok_r, _ = pair.right_capture.read()
        if not ok_l or not ok_r:
            print("ERROR: Could not read a frame from both stereo cameras.")
            return False
        print("OK: Stereo cameras accessible.")
        print("  Left:", pair.left.name, f"({pair.left.key})")
        print("  Right:", pair.right.name, f"({pair.right.key})")
        if pair.top and pair.top_capture:
            ok_t, _ = pair.top_capture.read()
            if ok_t:
                print("  Top:", pair.top.name, f"({pair.top.key}) [MU503 / top-down]")
            else:
                print("  Top: detected but frame read failed")
        return True
    finally:
        pair.release()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = verify_hardware_access()
    exit(0 if success else 1)
