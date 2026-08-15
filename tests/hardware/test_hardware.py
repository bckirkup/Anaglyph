"""
Human-in-the-loop hardware tests.

These tests require physical cameras and a human operator.
Run with: pytest tests/hardware/ -v --hardware
Skip in CI with: pytest -m "not hardware"

Each test prints instructions to the console and waits for operator input.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.hardware


def _prompt(msg: str) -> str:
    """Print instruction and wait for operator input."""
    print(f"\n{'=' * 60}")
    print("  OPERATOR ACTION REQUIRED")
    print(f"{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}")
    return input("  Press Enter when ready (or type 'skip' to skip): ").strip().lower()


def _confirm(msg: str) -> bool:
    """Ask operator yes/no question."""
    print(f"\n  {msg}")
    resp = input("  (y/n): ").strip().lower()
    return resp in ("y", "yes")


class TestCameraDetection:
    """HW-CAM: Camera detection and enumeration."""

    def test_single_camera_detected(self) -> None:
        """HW-CAM-01: At least one camera is accessible."""
        resp = _prompt("Connect at least one USB camera. Press Enter.")
        if resp == "skip":
            pytest.skip("Operator skipped")

        import cv2

        cap = cv2.VideoCapture(0)
        assert cap.isOpened(), "No camera found at index 0"
        ret, frame = cap.read()
        cap.release()
        assert ret, "Camera opened but could not read a frame"

    def test_stereo_pair_detected(self) -> None:
        """HW-CAM-02: Two cameras detected for stereo pair."""
        resp = _prompt("Connect both stereo cameras (left + right). Press Enter.")
        if resp == "skip":
            pytest.skip("Operator skipped")

        from camera_manager import CameraManager

        mgr = CameraManager()
        cams = mgr.discover_cameras()
        print(f"  Found {len(cams)} camera(s):")
        for c in cams:
            print(f"    - {c.name} (VID={c.vid:#06x}, PID={c.pid:#06x}, key={c.key})")
        assert len(cams) >= 2, f"Need >= 2 cameras, found {len(cams)}"

    def test_stereo_pair_opens(self) -> None:
        """HW-CAM-03: Stereo pair can be opened and frames read."""
        resp = _prompt("Ensure both stereo cameras are connected. Press Enter.")
        if resp == "skip":
            pytest.skip("Operator skipped")

        from camera_manager import CameraManager

        mgr = CameraManager()
        pair = mgr.get_stereo_pair(prefer_amscope=False)
        assert pair.left is not None, "Left camera not found"
        assert pair.right is not None, "Right camera not found"

        ok = mgr.open_captures(pair)
        assert ok, "Could not open stereo pair"
        try:
            ret_l, frame_l = pair.left_capture.read()
            ret_r, frame_r = pair.right_capture.read()
            assert ret_l, "Left camera: frame read failed"
            assert ret_r, "Right camera: frame read failed"
            print(f"  Left frame: {frame_l.shape}")
            print(f"  Right frame: {frame_r.shape}")
        finally:
            pair.release()


class TestFocusCalibration:
    """HW-FOCUS: Focus and sharpness validation."""

    def test_focus_sharpness(self) -> None:
        """HW-FOCUS-01: Sharpness metric responds to focus adjustment."""
        resp = _prompt(
            "Place a high-contrast target (printed grid, text) under the scope.\n"
            "  Adjust focus until sharp. Press Enter."
        )
        if resp == "skip":
            pytest.skip("Operator skipped")

        import cv2

        from gui import compute_sharpness

        cap = cv2.VideoCapture(0)
        assert cap.isOpened()
        ret, frame = cap.read()
        cap.release()
        assert ret

        sharpness = compute_sharpness(frame)
        print(f"  Sharpness (Laplacian variance): {sharpness:.1f}")
        assert sharpness > 50, f"Sharpness {sharpness:.1f} is very low — is the target in focus?"


class TestAlignment:
    """HW-ALIGN: Stereo alignment validation."""

    def test_alignment_computes(self) -> None:
        """HW-ALIGN-01: ORB alignment produces valid metrics from stereo pair."""
        resp = _prompt("Place a textured target under the scope with both cameras active.\n  Press Enter.")
        if resp == "skip":
            pytest.skip("Operator skipped")

        from camera_manager import CameraManager
        from gui import compute_alignment

        mgr = CameraManager()
        pair = mgr.get_stereo_pair(prefer_amscope=False)
        assert pair.left is not None
        assert pair.right is not None
        ok = mgr.open_captures(pair)
        assert ok
        try:
            _, left = pair.left_capture.read()
            _, right = pair.right_capture.read()
            metrics, M = compute_alignment(left, right)
            print(f"  Alignment: rot={metrics.rotation_deg:.1f}° scale={metrics.scale:.3f}")
            print(f"  Translation: tx={metrics.tx:.0f} ty={metrics.ty:.0f}")
            print(f"  Score: {metrics.score:.0f}/100")
            assert metrics.valid, "Alignment failed — not enough features?"
        finally:
            pair.release()


class TestAnaglyph:
    """HW-ANA: Anaglyph visual validation (requires 3D glasses)."""

    def test_anaglyph_looks_3d(self) -> None:
        """HW-ANA-01: Operator confirms anaglyph appears three-dimensional."""
        resp = _prompt(
            "The application will capture a stereo pair and display an anaglyph.\n"
            "  Put on your red/cyan 3D glasses. Press Enter to capture."
        )
        if resp == "skip":
            pytest.skip("Operator skipped")

        import cv2

        from camera_manager import CameraManager
        from gui import build_anaglyph_overlap, compute_alignment

        mgr = CameraManager()
        pair = mgr.get_stereo_pair(prefer_amscope=False)
        assert pair.left is not None
        assert pair.right is not None
        ok = mgr.open_captures(pair)
        assert ok
        try:
            _, left = pair.left_capture.read()
            _, right = pair.right_capture.read()
            metrics, M = compute_alignment(right, left)
            if M is not None:
                anag, _ = build_anaglyph_overlap(left, right, M)
                if anag is not None:
                    cv2.imshow("Anaglyph - Press any key", anag)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            confirmed = _confirm("Does the image appear three-dimensional through your glasses?")
            assert confirmed, "Operator reports anaglyph does not appear 3D"
        finally:
            pair.release()
