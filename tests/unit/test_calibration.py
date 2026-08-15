"""Tests for calibration module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from calibration import (
    CalibrationResult,
    CalibrationSession,
    detect_checkerboard,
    load_calibration,
    save_calibration,
    stereo_calibrate,
)


class TestDetectCheckerboard:
    def test_returns_none_on_blank_image(self) -> None:
        blank = np.zeros((480, 640), dtype=np.uint8)
        result = detect_checkerboard(blank)
        assert result is None

    def test_returns_none_on_noise(self) -> None:
        rng = np.random.default_rng(42)
        noise = rng.integers(0, 255, size=(480, 640), dtype=np.uint8)
        result = detect_checkerboard(noise)
        assert result is None

    def test_accepts_bgr_input(self) -> None:
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_checkerboard(bgr)
        assert result is None  # no checkerboard, but should not crash


class TestCalibrationSession:
    def test_empty_session(self) -> None:
        session = CalibrationSession()
        assert session.num_poses == 0

    def test_add_pose_rejects_blank(self) -> None:
        session = CalibrationSession()
        blank_l = np.zeros((480, 640, 3), dtype=np.uint8)
        blank_r = np.zeros((480, 640, 3), dtype=np.uint8)
        left_ok, right_ok = session.add_pose(blank_l, blank_r)
        assert not left_ok
        assert not right_ok
        assert session.num_poses == 0

    def test_clear_resets(self) -> None:
        session = CalibrationSession()
        # Manually add detections to test clear
        dummy_obj = np.zeros((54, 3), np.float32)
        dummy_corners = np.zeros((54, 1, 2), np.float32)
        session.left_detections.append((dummy_obj, dummy_corners))
        session.right_detections.append((dummy_obj, dummy_corners))
        assert session.num_poses == 1
        session.clear()
        assert session.num_poses == 0


class TestStereoCalibrate:
    def test_rejects_insufficient_poses(self) -> None:
        dummy_obj = np.zeros((54, 3), np.float32)
        dummy_corners = np.zeros((54, 1, 2), np.float32)
        left = [(dummy_obj, dummy_corners), (dummy_obj, dummy_corners)]
        right = [(dummy_obj, dummy_corners), (dummy_obj, dummy_corners)]
        result = stereo_calibrate(left, right, (640, 480))
        assert result is None

    def test_rejects_empty_lists(self) -> None:
        result = stereo_calibrate([], [], (640, 480))
        assert result is None


class TestSaveLoadCalibration:
    def test_round_trip(self, tmp_path: Path) -> None:
        """Save and reload a CalibrationResult; check values survive."""
        cal = CalibrationResult(
            camera_matrix_l=np.eye(3, dtype=np.float64),
            dist_l=np.zeros(5, dtype=np.float64),
            camera_matrix_r=np.eye(3, dtype=np.float64),
            dist_r=np.zeros(5, dtype=np.float64),
            R=np.eye(3, dtype=np.float64),
            T=np.array([[10.0], [0.0], [0.0]], dtype=np.float64),
            E=np.eye(3, dtype=np.float64),
            F=np.eye(3, dtype=np.float64),
            rms_error=0.42,
            image_size=(640, 480),
            num_poses=5,
        )
        path = tmp_path / "test_cal.npz"
        save_calibration(cal, path)
        assert path.exists()
        assert path.with_suffix(".json").exists()

        loaded = load_calibration(path)
        assert loaded is not None
        assert loaded.rms_error == cal.rms_error
        assert loaded.image_size == cal.image_size
        assert loaded.num_poses == cal.num_poses
        np.testing.assert_array_almost_equal(loaded.T, cal.T)

    def test_load_nonexistent(self, tmp_path: Path) -> None:
        result = load_calibration(tmp_path / "nope.npz")
        assert result is None

    def test_json_summary_content(self, tmp_path: Path) -> None:
        import json

        cal = CalibrationResult(
            camera_matrix_l=np.eye(3, dtype=np.float64),
            dist_l=np.zeros(5, dtype=np.float64),
            camera_matrix_r=np.eye(3, dtype=np.float64),
            dist_r=np.zeros(5, dtype=np.float64),
            R=np.eye(3, dtype=np.float64),
            T=np.array([[10.0], [0.0], [0.0]], dtype=np.float64),
            E=np.eye(3, dtype=np.float64),
            F=np.eye(3, dtype=np.float64),
            rms_error=0.5,
            image_size=(640, 480),
            num_poses=8,
        )
        path = tmp_path / "test_cal.npz"
        save_calibration(cal, path)
        summary = json.loads(path.with_suffix(".json").read_text())
        assert summary["rms_error"] == pytest.approx(0.5)
        assert summary["num_poses"] == 8
        assert summary["image_size"] == [640, 480]
        assert summary["baseline_mm"] == pytest.approx(10.0)
