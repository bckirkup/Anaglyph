"""Tests for StereoVideoRecorder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from video_recorder import RecordingStats, StereoVideoRecorder


@pytest.fixture()
def tmp_capture_dir(tmp_path: Path) -> Path:
    d = tmp_path / "captures"
    d.mkdir()
    return d


class TestStereoVideoRecorder:
    def test_start_creates_session_dir(self, tmp_capture_dir: Path) -> None:
        rec = StereoVideoRecorder(output_dir=tmp_capture_dir)
        session_dir = rec.start(width=320, height=240, fps=10.0)
        assert session_dir.exists()
        rec.stop()

    def test_not_recording_initially(self, tmp_capture_dir: Path) -> None:
        rec = StereoVideoRecorder(output_dir=tmp_capture_dir)
        assert not rec.is_recording

    def test_is_recording_after_start(self, tmp_capture_dir: Path) -> None:
        rec = StereoVideoRecorder(output_dir=tmp_capture_dir)
        rec.start(width=320, height=240)
        assert rec.is_recording
        rec.stop()

    def test_double_start_raises(self, tmp_capture_dir: Path) -> None:
        rec = StereoVideoRecorder(output_dir=tmp_capture_dir)
        rec.start(width=320, height=240)
        with pytest.raises(RuntimeError):
            rec.start(width=320, height=240)
        rec.stop()

    def test_stop_returns_stats(self, tmp_capture_dir: Path) -> None:
        rec = StereoVideoRecorder(output_dir=tmp_capture_dir)
        rec.start(width=320, height=240, fps=10.0)
        left = np.zeros((240, 320, 3), dtype=np.uint8)
        right = np.zeros((240, 320, 3), dtype=np.uint8)
        for _ in range(5):
            rec.add_frame(left, right)
        stats = rec.stop()
        assert isinstance(stats, RecordingStats)
        assert stats.left_frames == 5
        assert stats.right_frames == 5
        assert stats.dropped_frames == 0
        assert stats.duration_sec > 0

    def test_none_frame_counted_as_drop(self, tmp_capture_dir: Path) -> None:
        rec = StereoVideoRecorder(output_dir=tmp_capture_dir)
        rec.start(width=320, height=240)
        rec.add_frame(None, None)
        left = np.zeros((240, 320, 3), dtype=np.uint8)
        rec.add_frame(left, None)
        stats = rec.stop()
        assert stats.dropped_frames == 2
        assert stats.left_frames == 0

    def test_add_frame_when_not_recording(self, tmp_capture_dir: Path) -> None:
        rec = StereoVideoRecorder(output_dir=tmp_capture_dir)
        left = np.zeros((240, 320, 3), dtype=np.uint8)
        rec.add_frame(left, left)  # should be no-op

    def test_stop_when_not_recording(self, tmp_capture_dir: Path) -> None:
        rec = StereoVideoRecorder(output_dir=tmp_capture_dir)
        stats = rec.stop()
        assert stats.left_frames == 0

    def test_anaglyph_recorded_with_transform(self, tmp_capture_dir: Path) -> None:
        rec = StereoVideoRecorder(output_dir=tmp_capture_dir, record_anaglyph=True)
        rec.start(width=160, height=120, fps=10.0)
        left = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
        right = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
        M = np.eye(2, 3, dtype=np.float64)
        for _ in range(3):
            rec.add_frame(left, right, M)
        stats = rec.stop()
        assert stats.left_frames == 3
        # Check that anaglyph file was created
        session_dir = Path(stats.output_dir)
        anaglyph_files = list(session_dir.glob("anaglyph*"))
        assert len(anaglyph_files) == 1

    def test_frame_resize(self, tmp_capture_dir: Path) -> None:
        """Frames of different size should be resized to recording dimensions."""
        rec = StereoVideoRecorder(output_dir=tmp_capture_dir)
        rec.start(width=320, height=240, fps=10.0)
        # Provide a larger frame
        left = np.zeros((480, 640, 3), dtype=np.uint8)
        right = np.zeros((480, 640, 3), dtype=np.uint8)
        rec.add_frame(left, right)
        stats = rec.stop()
        assert stats.left_frames == 1
