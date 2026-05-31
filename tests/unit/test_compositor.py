"""Tests for anaglyph compositing and alignment functions in compositor.py."""

from __future__ import annotations

import numpy as np
import pytest

from compositor import (
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

# Aliases for backward-compat with existing test names
_affine_to_metrics = affine_to_metrics
_compose_affine = compose_affine
_invert_affine = invert_affine
_normalize_rotation_deg = normalize_rotation_deg
_apply_180_flip_to_transform = apply_180_flip_to_transform
build_anaglyph_overlap = build_anaglyph


class TestNormalizeRotation:
    def test_zero(self) -> None:
        assert _normalize_rotation_deg(0.0) == 0.0

    def test_positive_wrap(self) -> None:
        assert _normalize_rotation_deg(270.0) == pytest.approx(-90.0)

    def test_negative_wrap(self) -> None:
        assert _normalize_rotation_deg(-270.0) == pytest.approx(90.0)

    def test_180(self) -> None:
        result = _normalize_rotation_deg(180.0)
        assert result == pytest.approx(180.0) or result == pytest.approx(-180.0)

    def test_360(self) -> None:
        assert _normalize_rotation_deg(360.0) == pytest.approx(0.0)


class TestAffineToMetrics:
    def test_identity(self, identity_affine: np.ndarray) -> None:
        m = _affine_to_metrics(identity_affine)
        assert m.valid
        assert m.rotation_deg == pytest.approx(0.0, abs=0.01)
        assert m.scale == pytest.approx(1.0, abs=0.01)
        assert m.tx == pytest.approx(0.0, abs=0.01)
        assert m.ty == pytest.approx(0.0, abs=0.01)
        assert m.score == pytest.approx(100.0, abs=1.0)

    def test_translation(self, translation_affine: np.ndarray) -> None:
        m = _affine_to_metrics(translation_affine)
        assert m.valid
        assert m.tx == pytest.approx(10.0, abs=0.01)
        assert m.rotation_deg == pytest.approx(0.0, abs=0.01)

    def test_none_input(self) -> None:
        m = _affine_to_metrics(None)
        assert not m.valid

    def test_rotation_90(self) -> None:
        M = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        m = _affine_to_metrics(M)
        assert m.valid
        assert m.rotation_deg == pytest.approx(90.0, abs=0.1)

    def test_scale_2x(self) -> None:
        M = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float64)
        m = _affine_to_metrics(M)
        assert m.valid
        assert m.scale == pytest.approx(2.0, abs=0.01)


class TestComposeAffine:
    def test_identity_composition(self, identity_affine: np.ndarray) -> None:
        result = _compose_affine(identity_affine, identity_affine)
        np.testing.assert_allclose(result, identity_affine, atol=1e-10)

    def test_translation_accumulates(self, translation_affine: np.ndarray) -> None:
        result = _compose_affine(translation_affine, translation_affine)
        assert result[0, 2] == pytest.approx(20.0, abs=0.01)

    def test_compose_then_invert(self, translation_affine: np.ndarray) -> None:
        inv = _invert_affine(translation_affine)
        result = _compose_affine(translation_affine, inv)
        np.testing.assert_allclose(result[:, :2], np.eye(2), atol=1e-10)
        np.testing.assert_allclose(result[:, 2], 0.0, atol=1e-10)


class TestInvertAffine:
    def test_identity(self, identity_affine: np.ndarray) -> None:
        inv = _invert_affine(identity_affine)
        np.testing.assert_allclose(inv, identity_affine, atol=1e-10)

    def test_translation(self, translation_affine: np.ndarray) -> None:
        inv = _invert_affine(translation_affine)
        assert inv[0, 2] == pytest.approx(-10.0, abs=0.01)


class TestApply180Flip:
    def test_flip_identity(self, identity_affine: np.ndarray) -> None:
        flipped = _apply_180_flip_to_transform(identity_affine, 640, 480)
        m = _affine_to_metrics(flipped)
        assert abs(m.rotation_deg) == pytest.approx(180.0, abs=1.0)


class TestComputeAlignment:
    def test_identical_images(self) -> None:
        """Use a textured image so ORB can find enough keypoints."""
        rng = np.random.default_rng(42)
        img = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
        metrics, M = compute_alignment(img, img.copy())
        assert metrics.valid
        assert metrics.rotation_deg == pytest.approx(0.0, abs=5.0)
        assert metrics.scale == pytest.approx(1.0, abs=0.1)

    def test_none_input(self) -> None:
        metrics, M = compute_alignment(None, None)
        assert not metrics.valid
        assert M is None

    def test_empty_images(self) -> None:
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        metrics, M = compute_alignment(empty, empty)
        assert not metrics.valid

    def test_tiny_images(self) -> None:
        tiny = np.zeros((16, 16, 3), dtype=np.uint8)
        metrics, M = compute_alignment(tiny, tiny)
        assert not metrics.valid


class TestBuildAnaglyphOverlap:
    def test_basic_anaglyph(
        self,
        synthetic_left: np.ndarray,
        synthetic_right: np.ndarray,
        identity_affine: np.ndarray,
    ) -> None:
        anag, roi = build_anaglyph_overlap(synthetic_left, synthetic_right, identity_affine)
        assert anag is not None
        assert roi is not None
        assert anag.shape[2] == 3  # BGR
        assert anag.shape[0] > 0 and anag.shape[1] > 0

    def test_red_channel_is_left(
        self,
        synthetic_left: np.ndarray,
        synthetic_right: np.ndarray,
        identity_affine: np.ndarray,
    ) -> None:
        """Red channel (index 2 in BGR) should carry the left image."""
        anag, _ = build_anaglyph_overlap(synthetic_left, synthetic_right, identity_affine)
        assert anag is not None
        # In the overlap region, red channel should have some nonzero values from left
        assert anag[:, :, 2].max() > 0

    def test_none_inputs(self) -> None:
        anag, roi = build_anaglyph_overlap(None, None, None)
        assert anag is None
        assert roi is None

    def test_small_images(self, identity_affine: np.ndarray) -> None:
        tiny = np.zeros((4, 4, 3), dtype=np.uint8)
        anag, roi = build_anaglyph_overlap(tiny, tiny, identity_affine)
        assert anag is None


class TestBuildThreeWayOverlay:
    def test_left_only(self, synthetic_left: np.ndarray) -> None:
        overlay = build_three_way_overlay(None, synthetic_left, None, None, None)
        assert overlay.shape == (280, 420, 3)

    def test_all_three(
        self,
        synthetic_left: np.ndarray,
        synthetic_right: np.ndarray,
        identity_affine: np.ndarray,
    ) -> None:
        top = np.zeros((480, 640, 3), dtype=np.uint8)
        top[150:350, 100:500] = 128
        overlay = build_three_way_overlay(top, synthetic_left, synthetic_right, identity_affine, identity_affine)
        assert overlay.shape == (280, 420, 3)
        assert overlay.max() > 0

    def test_no_images(self) -> None:
        overlay = build_three_way_overlay(None, None, None, None, None)
        assert overlay.shape == (280, 420, 3)


class TestComputeSharpness:
    def test_sharp_image(self) -> None:
        """High-contrast edges should have high sharpness."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:50] = 255
        s = compute_sharpness(img)
        assert s > 0

    def test_blank_image(self) -> None:
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        s = compute_sharpness(img)
        assert s == pytest.approx(0.0, abs=1.0)

    def test_none_input(self) -> None:
        assert compute_sharpness(None) == 0.0

    def test_sharp_vs_blurry(self) -> None:
        """Sharp image should have higher sharpness than blurred version."""
        import cv2

        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[50:150, 50:150] = 255
        sharp = compute_sharpness(img)
        blurred = cv2.GaussianBlur(img, (21, 21), 5)
        blurry = compute_sharpness(blurred)
        assert sharp > blurry


class TestAnaglyphMethods:
    """Test all four anaglyph compositing methods."""

    @pytest.fixture()
    def _stereo_pair(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(42)
        left = rng.integers(50, 200, (120, 160, 3), dtype=np.uint8)
        right = rng.integers(50, 200, (120, 160, 3), dtype=np.uint8)
        M = np.eye(2, 3, dtype=np.float64)
        return left, right, M

    def test_wimmer_produces_output(self, _stereo_pair: tuple) -> None:
        left, right, M = _stereo_pair
        anag, roi = build_anaglyph(left, right, M, AnaglyphMethod.WIMMER)
        assert anag is not None
        assert anag.shape[2] == 3

    def test_dubois_produces_output(self, _stereo_pair: tuple) -> None:
        left, right, M = _stereo_pair
        anag, roi = build_anaglyph(left, right, M, AnaglyphMethod.DUBOIS)
        assert anag is not None
        assert anag.shape[2] == 3

    def test_half_color_produces_output(self, _stereo_pair: tuple) -> None:
        left, right, M = _stereo_pair
        anag, roi = build_anaglyph(left, right, M, AnaglyphMethod.HALF_COLOR)
        assert anag is not None
        assert anag.shape[2] == 3

    def test_gray_produces_output(self, _stereo_pair: tuple) -> None:
        left, right, M = _stereo_pair
        anag, roi = build_anaglyph(left, right, M, AnaglyphMethod.GRAY)
        assert anag is not None
        assert anag.shape[2] == 3

    def test_dubois_preserves_color(self, _stereo_pair: tuple) -> None:
        """Dubois should produce non-grayscale output (unlike Wimmer/gray)."""
        left, right, M = _stereo_pair
        anag, _ = build_anaglyph(left, right, M, AnaglyphMethod.DUBOIS)
        assert anag is not None
        # Check that channels differ (not all equal = has color)
        assert not np.array_equal(anag[:, :, 0], anag[:, :, 2])

    def test_half_color_red_from_left_luminance(self, _stereo_pair: tuple) -> None:
        """Half-color R channel should come from left image luminance (within ROI)."""
        import cv2

        left, right, M = _stereo_pair
        anag, roi = build_anaglyph(left, right, M, AnaglyphMethod.HALF_COLOR)
        assert anag is not None
        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        if roi is not None:
            x, y, w, h = roi
            expected = left_gray[y : y + h, x : x + w]
        else:
            expected = left_gray
        # R channel (index 2 in BGR) should match left luminance within the overlap ROI
        np.testing.assert_array_equal(anag[:, :, 2], expected)

    def test_gray_wimmer_same_structure(self, _stereo_pair: tuple) -> None:
        """Gray and Wimmer should have same structure: R=left_gray, G=B=right_gray."""
        left, right, M = _stereo_pair
        wimmer, _ = build_anaglyph(left, right, M, AnaglyphMethod.WIMMER)
        gray, _ = build_anaglyph(left, right, M, AnaglyphMethod.GRAY)
        assert wimmer is not None and gray is not None
        # Both use grayscale channels, so G==B in both cases
        np.testing.assert_array_equal(wimmer[:, :, 0], wimmer[:, :, 1])
        np.testing.assert_array_equal(gray[:, :, 0], gray[:, :, 1])

    def test_method_enum_values(self) -> None:
        assert AnaglyphMethod.WIMMER.value == "wimmer"
        assert AnaglyphMethod.DUBOIS.value == "dubois"
        assert AnaglyphMethod.HALF_COLOR.value == "half_color"
        assert AnaglyphMethod.GRAY.value == "gray"

    def test_none_inputs(self) -> None:
        M = np.eye(2, 3, dtype=np.float64)
        for method in AnaglyphMethod:
            anag, roi = build_anaglyph(None, None, M, method)
            assert anag is None
