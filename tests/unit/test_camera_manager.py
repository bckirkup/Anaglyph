"""Tests for camera_manager data structures (no hardware required)."""

from __future__ import annotations

from camera_manager import CameraInfo, StereoPair


class TestCameraInfo:
    def test_key_generation(self) -> None:
        info = CameraInfo(index=0, backend=0, name="Test", vid=0x0AC8, pid=0x8370, path="/dev/video0")
        assert "0AC8" in info.key
        assert "8370" in info.key

    def test_key_fallback(self) -> None:
        info = CameraInfo(index=0, backend=0, name="Test")
        assert info.key  # should not be empty

    def test_is_mu503(self) -> None:
        info = CameraInfo(index=0, backend=0, name="USB3.0 Camera", vid=0x0547, pid=0x3510)
        assert info.is_mu503()

    def test_is_not_mu503(self) -> None:
        info = CameraInfo(index=0, backend=0, name="MD500L", vid=0x0AC8, pid=0x8370)
        assert not info.is_mu503()

    def test_is_usb3_camera(self) -> None:
        info = CameraInfo(index=0, backend=0, name="USB3.0 Camera")
        assert info.is_usb3_camera()

    def test_is_not_usb3(self) -> None:
        info = CameraInfo(index=0, backend=0, name="Webcam")
        assert not info.is_usb3_camera()


class TestStereoPair:
    def test_release_with_none(self) -> None:
        pair = StereoPair()
        pair.release()  # should not raise
        assert pair.left_capture is None
        assert pair.right_capture is None
        assert pair.top_capture is None

    def test_fields(self) -> None:
        left = CameraInfo(index=0, backend=0, name="Left")
        right = CameraInfo(index=1, backend=0, name="Right")
        pair = StereoPair(left=left, right=right)
        assert pair.left.name == "Left"
        assert pair.right.name == "Right"
        assert pair.top is None
