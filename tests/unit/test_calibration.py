"""Tests for calibration module."""

from __future__ import annotations

import numpy as np

from calibration import detect_checkerboard


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
