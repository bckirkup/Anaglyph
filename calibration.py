"""
Stereo Calibration - Math and optimization for stereoscope alignment.

Uses cv2.stereoCalibrate and cv2.initUndistortRectifyMap for optimal overlap.
Black-and-white target detection for calibration routine.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def detect_checkerboard(
    image: np.ndarray,
    pattern_size: tuple[int, int] = (9, 6),
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Detect checkerboard corners in a grayscale or BGR image.

    Args:
        image: Input frame (BGR or grayscale).
        pattern_size: Inner corners (cols, rows).

    Returns:
        (object_points, image_points) or None if not found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    ret, corners = cv2.findChessboardCorners(gray, pattern_size)
    if not ret:
        return None
    # Refine corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
    return objp, corners


@dataclass
class CalibrationResult:
    """Full stereo calibration output."""

    camera_matrix_l: np.ndarray
    dist_l: np.ndarray
    camera_matrix_r: np.ndarray
    dist_r: np.ndarray
    R: np.ndarray  # Rotation between cameras
    T: np.ndarray  # Translation between cameras
    E: np.ndarray  # Essential matrix
    F: np.ndarray  # Fundamental matrix
    rms_error: float  # Reprojection error from stereoCalibrate
    image_size: tuple[int, int]  # (width, height)
    # Rectification outputs
    R1: np.ndarray | None = None
    R2: np.ndarray | None = None
    P1: np.ndarray | None = None
    P2: np.ndarray | None = None
    Q: np.ndarray | None = None
    map_l: tuple[np.ndarray, np.ndarray] | None = None
    map_r: tuple[np.ndarray, np.ndarray] | None = None
    num_poses: int = 0


@dataclass
class CalibrationSession:
    """Accumulates checkerboard poses for calibration."""

    pattern_size: tuple[int, int] = (9, 6)
    left_detections: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    right_detections: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)

    @property
    def num_poses(self) -> int:
        return min(len(self.left_detections), len(self.right_detections))

    def add_pose(self, left_frame: np.ndarray, right_frame: np.ndarray) -> tuple[bool, bool]:
        """Try to detect checkerboard in both frames. Returns (left_ok, right_ok)."""
        left_result = detect_checkerboard(left_frame, self.pattern_size)
        right_result = detect_checkerboard(right_frame, self.pattern_size)
        left_ok = left_result is not None
        right_ok = right_result is not None
        if left_ok and right_ok:
            self.left_detections.append(left_result)
            self.right_detections.append(right_result)
            logger.info(
                "Pose %d captured (both cameras detected checkerboard)",
                self.num_poses,
            )
        else:
            if not left_ok:
                logger.debug("Left camera: checkerboard not found")
            if not right_ok:
                logger.debug("Right camera: checkerboard not found")
        return left_ok, right_ok

    def clear(self) -> None:
        self.left_detections.clear()
        self.right_detections.clear()


def stereo_calibrate(
    left_points: list[tuple[np.ndarray, np.ndarray]],
    right_points: list[tuple[np.ndarray, np.ndarray]],
    image_size: tuple[int, int],
) -> CalibrationResult | None:
    """
    Run cv2.stereoCalibrate and compute rectification maps.

    Args:
        left_points: List of (objp, corners) for left camera.
        right_points: List of (objp, corners) for right camera.
        image_size: (width, height) of images.

    Returns:
        CalibrationResult or None on failure.
    """
    if len(left_points) < 3 or len(right_points) < 3:
        logger.warning(
            "Need at least 3 poses for calibration, got %d left / %d right",
            len(left_points),
            len(right_points),
        )
        return None

    n = min(len(left_points), len(right_points))
    obj_points = [left_points[i][0] for i in range(n)]
    img_points_l = [left_points[i][1] for i in range(n)]
    img_points_r = [right_points[i][1] for i in range(n)]

    # Individual camera calibrations (initial estimates)
    flags_mono = cv2.CALIB_FIX_K3
    ret_l, cm_l, dist_l, _, _ = cv2.calibrateCamera(obj_points, img_points_l, image_size, None, None, flags=flags_mono)
    ret_r, cm_r, dist_r, _, _ = cv2.calibrateCamera(obj_points, img_points_r, image_size, None, None, flags=flags_mono)
    logger.info("Mono calibration RMS: left=%.3f right=%.3f", ret_l, ret_r)

    # Stereo calibration
    flags_stereo = (
        cv2.CALIB_FIX_INTRINSIC  # Use mono intrinsics as-is
    )
    try:
        rms, cm_l, dist_l, cm_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
            obj_points,
            img_points_l,
            img_points_r,
            cm_l,
            dist_l,
            cm_r,
            dist_r,
            image_size,
            flags=flags_stereo,
            criteria=(
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                100,
                1e-6,
            ),
        )
    except cv2.error as e:
        logger.error("stereoCalibrate failed: %s", e)
        return None

    logger.info("Stereo calibration RMS error: %.4f (using %d poses)", rms, n)

    # Rectification
    try:
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            cm_l,
            dist_l,
            cm_r,
            dist_r,
            image_size,
            R,
            T,
            alpha=0,  # Crop to valid pixels only
        )
    except cv2.error as e:
        logger.error("stereoRectify failed: %s", e)
        R1 = R2 = P1 = P2 = Q = None

    # Undistort+rectify maps
    map_l = map_r = None
    if R1 is not None:
        map_l = cv2.initUndistortRectifyMap(cm_l, dist_l, R1, P1, image_size, cv2.CV_16SC2)
        map_r = cv2.initUndistortRectifyMap(cm_r, dist_r, R2, P2, image_size, cv2.CV_16SC2)

    return CalibrationResult(
        camera_matrix_l=cm_l,
        dist_l=dist_l,
        camera_matrix_r=cm_r,
        dist_r=dist_r,
        R=R,
        T=T,
        E=E,
        F=F,
        rms_error=rms,
        image_size=image_size,
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        Q=Q,
        map_l=map_l,
        map_r=map_r,
        num_poses=n,
    )


def apply_rectification(
    frame: np.ndarray,
    remap: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Apply undistort+rectify maps to a frame."""
    return cv2.remap(frame, remap[0], remap[1], cv2.INTER_LINEAR)


def save_calibration(cal: CalibrationResult, path: Path) -> None:
    """Save calibration to .npz file."""
    path = Path(path)
    data = {
        "camera_matrix_l": cal.camera_matrix_l,
        "dist_l": cal.dist_l,
        "camera_matrix_r": cal.camera_matrix_r,
        "dist_r": cal.dist_r,
        "R": cal.R,
        "T": cal.T,
        "E": cal.E,
        "F": cal.F,
        "rms_error": np.array([cal.rms_error]),
        "image_size": np.array(cal.image_size),
        "num_poses": np.array([cal.num_poses]),
    }
    if cal.R1 is not None:
        data["R1"] = cal.R1
        data["R2"] = cal.R2
        data["P1"] = cal.P1
        data["P2"] = cal.P2
        data["Q"] = cal.Q
    if cal.map_l is not None:
        data["map_l_0"] = cal.map_l[0]
        data["map_l_1"] = cal.map_l[1]
        data["map_r_0"] = cal.map_r[0]
        data["map_r_1"] = cal.map_r[1]

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), **data)
    logger.info("Calibration saved to %s", path)

    # Also save a human-readable summary
    summary_path = path.with_suffix(".json")
    summary = {
        "rms_error": cal.rms_error,
        "image_size": list(cal.image_size),
        "num_poses": cal.num_poses,
        "baseline_mm": float(np.linalg.norm(cal.T)),
        "has_rectification": cal.R1 is not None,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Calibration summary saved to %s", summary_path)


def load_calibration(path: Path) -> CalibrationResult | None:
    """Load calibration from .npz file."""
    path = Path(path)
    if not path.exists():
        npz_path = path.with_suffix(".npz")
        if npz_path.exists():
            path = npz_path
        else:
            logger.warning("Calibration file not found: %s", path)
            return None

    try:
        data = np.load(str(path), allow_pickle=False)
    except Exception as e:
        logger.error("Failed to load calibration: %s", e)
        return None

    image_size = tuple(int(x) for x in data["image_size"])

    # Reconstruct maps if saved
    map_l = map_r = None
    if "map_l_0" in data:
        map_l = (data["map_l_0"], data["map_l_1"])
        map_r = (data["map_r_0"], data["map_r_1"])

    return CalibrationResult(
        camera_matrix_l=data["camera_matrix_l"],
        dist_l=data["dist_l"],
        camera_matrix_r=data["camera_matrix_r"],
        dist_r=data["dist_r"],
        R=data["R"],
        T=data["T"],
        E=data["E"],
        F=data["F"],
        rms_error=float(data["rms_error"][0]),
        image_size=image_size,
        R1=data.get("R1"),
        R2=data.get("R2"),
        P1=data.get("P1"),
        P2=data.get("P2"),
        Q=data.get("Q"),
        map_l=map_l,
        map_r=map_r,
        num_poses=int(data["num_poses"][0]),
    )
