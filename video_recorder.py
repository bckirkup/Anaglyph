"""
Video Recorder — Synchronized stereo video recording.

Records left/right (and optionally top) camera streams to MP4 files,
plus a real-time anaglyph composite video. Detects frame-drop drift
and emits warnings.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from compositor import AnaglyphMethod, build_anaglyph

logger = logging.getLogger(__name__)

# Default codec preference order
_CODEC_CANDIDATES = [
    ("mp4v", ".mp4"),
    ("avc1", ".mp4"),
    ("XVID", ".avi"),
    ("MJPG", ".avi"),
]


@dataclass
class RecordingStats:
    """Statistics for a completed recording session."""

    duration_sec: float = 0.0
    left_frames: int = 0
    right_frames: int = 0
    dropped_frames: int = 0
    max_drift_ms: float = 0.0
    output_dir: str = ""


@dataclass
class _WriterBundle:
    """Internal: one cv2.VideoWriter + metadata."""

    writer: cv2.VideoWriter | None = None
    path: Path | None = None
    frame_count: int = 0


class StereoVideoRecorder:
    """
    Records synchronized stereo video from left/right camera streams.

    Usage:
        recorder = StereoVideoRecorder(output_dir="./captures")
        recorder.start(width=640, height=480, fps=15.0)
        # In your capture loop:
        recorder.add_frame(left_bgr, right_bgr, M_right_to_left)
        # When done:
        stats = recorder.stop()
    """

    def __init__(
        self,
        output_dir: str | Path = ".",
        prefix: str = "stereo",
        anaglyph_method: AnaglyphMethod = AnaglyphMethod.WIMMER,
        record_anaglyph: bool = True,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._prefix = prefix
        self._anaglyph_method = anaglyph_method
        self._record_anaglyph = record_anaglyph

        self._left_writer = _WriterBundle()
        self._right_writer = _WriterBundle()
        self._anaglyph_writer = _WriterBundle()

        self._recording = False
        self._start_time = 0.0
        self._fps = 15.0
        self._width = 640
        self._height = 480
        self._dropped = 0
        self._max_drift_ms = 0.0
        self._last_left_time = 0.0
        self._last_right_time = 0.0

    @property
    def is_recording(self) -> bool:
        return self._recording

    def start(self, width: int = 640, height: int = 480, fps: float = 15.0) -> Path:
        """
        Begin recording. Returns the output directory for this session.

        Raises RuntimeError if already recording.
        """
        if self._recording:
            raise RuntimeError("Already recording")

        self._width = width
        self._height = height
        self._fps = fps

        self._output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        session_dir = self._output_dir / f"{self._prefix}_{timestamp}"
        session_dir.mkdir(exist_ok=True)

        fourcc, ext = self._pick_codec()

        self._left_writer = self._open_writer(session_dir / f"left{ext}", fourcc, fps, width, height)
        self._right_writer = self._open_writer(session_dir / f"right{ext}", fourcc, fps, width, height)

        if self._record_anaglyph:
            self._anaglyph_writer = self._open_writer(session_dir / f"anaglyph{ext}", fourcc, fps, width, height)

        self._recording = True
        self._start_time = time.monotonic()
        self._dropped = 0
        self._max_drift_ms = 0.0
        self._last_left_time = 0.0
        self._last_right_time = 0.0

        logger.info("Recording started: %s", session_dir)
        return session_dir

    def add_frame(
        self,
        left_bgr: np.ndarray | None,
        right_bgr: np.ndarray | None,
        M_right_to_left: np.ndarray | None = None,
    ) -> None:
        """
        Add a synchronized frame pair to the recording.

        If either frame is None, the frame is counted as dropped.
        If M_right_to_left is provided and anaglyph recording is enabled,
        a composited anaglyph frame is also written.
        """
        if not self._recording:
            return

        now = time.monotonic()

        if left_bgr is None or right_bgr is None:
            self._dropped += 1
            return

        # Drift detection
        if self._last_left_time > 0 and self._last_right_time > 0:
            drift_ms = abs(self._last_left_time - self._last_right_time) * 1000
            self._max_drift_ms = max(self._max_drift_ms, drift_ms)
            frame_period_ms = 1000.0 / self._fps
            if drift_ms > frame_period_ms:
                logger.warning("Frame drift %.1f ms exceeds frame period %.1f ms", drift_ms, frame_period_ms)

        self._last_left_time = now
        self._last_right_time = now

        left_resized = self._ensure_size(left_bgr)
        right_resized = self._ensure_size(right_bgr)

        self._write_frame(self._left_writer, left_resized)
        self._write_frame(self._right_writer, right_resized)

        if self._record_anaglyph and M_right_to_left is not None:
            anaglyph, _ = build_anaglyph(left_resized, right_resized, M_right_to_left, self._anaglyph_method)
            if anaglyph is not None:
                anaglyph_resized = self._ensure_size(anaglyph)
                self._write_frame(self._anaglyph_writer, anaglyph_resized)

    def stop(self) -> RecordingStats:
        """Stop recording and return statistics."""
        if not self._recording:
            return RecordingStats()

        self._recording = False
        elapsed = time.monotonic() - self._start_time

        stats = RecordingStats(
            duration_sec=elapsed,
            left_frames=self._left_writer.frame_count,
            right_frames=self._right_writer.frame_count,
            dropped_frames=self._dropped,
            max_drift_ms=self._max_drift_ms,
            output_dir=str(self._left_writer.path.parent) if self._left_writer.path else "",
        )

        for bundle in (self._left_writer, self._right_writer, self._anaglyph_writer):
            if bundle.writer is not None:
                bundle.writer.release()
                bundle.writer = None

        logger.info(
            "Recording stopped: %.1fs, %d/%d frames (L/R), %d dropped, max drift %.1fms",
            stats.duration_sec,
            stats.left_frames,
            stats.right_frames,
            stats.dropped_frames,
            stats.max_drift_ms,
        )
        return stats

    def _ensure_size(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to recording dimensions if needed."""
        h, w = frame.shape[:2]
        if w != self._width or h != self._height:
            return cv2.resize(frame, (self._width, self._height))
        return frame

    def _pick_codec(self) -> tuple[int, str]:
        """Find a working fourcc codec."""
        for codec_str, ext in _CODEC_CANDIDATES:
            fourcc = cv2.VideoWriter_fourcc(*codec_str)
            test_path = self._output_dir / f"_codec_test{ext}"
            writer = cv2.VideoWriter(str(test_path), fourcc, 15.0, (320, 240))
            if writer.isOpened():
                writer.release()
                test_path.unlink(missing_ok=True)
                logger.debug("Using codec: %s (%s)", codec_str, ext)
                return fourcc, ext
            test_path.unlink(missing_ok=True)
        logger.warning("No working codec found; falling back to MJPG")
        return cv2.VideoWriter_fourcc(*"MJPG"), ".avi"

    def _open_writer(self, path: Path, fourcc: int, fps: float, width: int, height: int) -> _WriterBundle:
        """Open a VideoWriter and wrap it."""
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        if not writer.isOpened():
            logger.error("Failed to open video writer: %s", path)
            return _WriterBundle(path=path)
        return _WriterBundle(writer=writer, path=path)

    @staticmethod
    def _write_frame(bundle: _WriterBundle, frame: np.ndarray) -> None:
        """Write a single frame to a writer bundle."""
        if bundle.writer is not None:
            bundle.writer.write(frame)
            bundle.frame_count += 1
