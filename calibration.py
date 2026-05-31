"""
Stereo Calibration - Math and optimization for stereoscope alignment.

Uses cv2.stereoCalibrate and cv2.initUndistortRectifyMap for optimal overlap.
Black-and-white target detection for calibration routine.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    from pathlib import Path
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


def stereo_calibrate(
    left_points: list[tuple[np.ndarray, np.ndarray]],
    right_points: list[tuple[np.ndarray, np.ndarray]],
    image_size: tuple[int, int],
) -> dict | None:
    """
    Run cv2.stereoCalibrate and compute rectification maps.

    Args:
        left_points: List of (objp, corners) for left camera.
        right_points: List of (objp, corners) for right camera.
        image_size: (width, height) of images.

    Returns:
        Dict with camera_matrix_l, dist_l, camera_matrix_r, dist_r,
        R, T, E, F, and rectification maps, or None on failure.
    """
    # TODO: Implement full stereo calibration and initUndistortRectifyMap
    logger.info("Stereo calibration stub - to be implemented")
    return None


def load_calibration(path: Path) -> dict | None:
    """Load calibration from file."""
    # TODO: Load npz or yaml
    return None


def save_calibration(cal: dict, path: Path) -> None:
    """Save calibration to file."""
    # TODO: Save npz or yaml
    pass
