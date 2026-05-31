"""Shared fixtures for Anaglyph tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def synthetic_left() -> np.ndarray:
    """640x480 BGR image with a white rectangle on the left half."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[100:380, 50:300] = 255
    return img


@pytest.fixture
def synthetic_right() -> np.ndarray:
    """640x480 BGR image with a white rectangle shifted ~20px right."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[100:380, 70:320] = 255
    return img


@pytest.fixture
def checkerboard_image() -> np.ndarray:
    """Synthetic 640x480 checkerboard (8x6 inner corners, 40px squares)."""
    img = np.zeros((480, 640), dtype=np.uint8)
    square = 40
    for r in range(480 // square):
        for c in range(640 // square):
            if (r + c) % 2 == 0:
                y0, y1 = r * square, (r + 1) * square
                x0, x1 = c * square, (c + 1) * square
                img[y0:y1, x0:x1] = 255
    return img


@pytest.fixture
def identity_affine() -> np.ndarray:
    """2x3 identity affine matrix."""
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


@pytest.fixture
def translation_affine() -> np.ndarray:
    """2x3 affine matrix with 10px horizontal translation."""
    return np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 0.0]], dtype=np.float64)
