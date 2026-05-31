"""Tests for still_capture module."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

from still_capture import (
    CaptureMetadata,
    capture_still_set,
    generate_filename,
    save_metadata,
    save_still,
)


class TestGenerateFilename:
    def test_default_jpg(self, tmp_path: Path) -> None:
        name = generate_filename(tmp_path)
        assert name.suffix == ".jpg"
        assert name.parent == tmp_path

    def test_with_subject_and_method(self, tmp_path: Path) -> None:
        name = generate_filename(tmp_path, subject="screwdriver", method="dubois")
        assert "screwdriver" in name.stem
        assert "dubois" in name.stem

    def test_tiff_extension(self, tmp_path: Path) -> None:
        name = generate_filename(tmp_path, suffix=".tiff")
        assert name.suffix == ".tiff"

    def test_unsafe_characters_sanitized(self, tmp_path: Path) -> None:
        name = generate_filename(tmp_path, subject="test sample/2")
        # Should not contain / in the filename
        assert "/" not in name.name


class TestSaveStill:
    def test_save_jpeg(self, tmp_path: Path) -> None:
        frame = np.full((100, 200, 3), 128, dtype=np.uint8)
        path = tmp_path / "test.jpg"
        result = save_still(frame, path)
        assert result is True
        assert path.exists()
        assert path.stat().st_size > 0

    def test_save_tiff(self, tmp_path: Path) -> None:
        frame = np.full((100, 200, 3), 128, dtype=np.uint8)
        path = tmp_path / "test.tiff"
        result = save_still(frame, path)
        assert result is True
        assert path.exists()

    def test_save_png(self, tmp_path: Path) -> None:
        frame = np.full((100, 200, 3), 128, dtype=np.uint8)
        path = tmp_path / "test.png"
        result = save_still(frame, path)
        assert result is True
        assert path.exists()

    def test_creates_parent_dir(self, tmp_path: Path) -> None:
        frame = np.full((100, 200, 3), 128, dtype=np.uint8)
        path = tmp_path / "subdir" / "nested" / "test.jpg"
        result = save_still(frame, path)
        assert result is True
        assert path.exists()


class TestCaptureMetadata:
    def test_to_dict(self) -> None:
        meta = CaptureMetadata(
            timestamp="2026-01-01T00:00:00",
            subject="test",
            anaglyph_method="wimmer",
        )
        d = meta.to_dict()
        assert d["timestamp"] == "2026-01-01T00:00:00"
        assert d["subject"] == "test"
        assert "extra" not in d  # empty extra omitted

    def test_to_dict_with_extra(self) -> None:
        meta = CaptureMetadata(extra={"magnification": "10x"})
        d = meta.to_dict()
        assert d["extra"]["magnification"] == "10x"


class TestSaveMetadata:
    def test_saves_json(self, tmp_path: Path) -> None:
        meta = CaptureMetadata(
            timestamp="2026-01-01T00:00:00",
            subject="sample",
        )
        path = tmp_path / "meta.json"
        save_metadata(meta, path)
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["subject"] == "sample"


class TestCaptureStillSet:
    def test_captures_all_frames(self, tmp_path: Path) -> None:
        left = np.full((100, 200, 3), 100, dtype=np.uint8)
        right = np.full((100, 200, 3), 150, dtype=np.uint8)
        anag = np.full((100, 200, 3), 128, dtype=np.uint8)
        result = capture_still_set(
            left,
            right,
            anag,
            tmp_path,
            subject="test",
            method="wimmer",
            formats=(".jpg",),
        )
        assert result is not None
        assert len(result.paths) == 3  # left, right, anaglyph
        assert result.metadata_path is not None
        assert result.metadata_path.exists()

    def test_captures_with_top(self, tmp_path: Path) -> None:
        left = np.full((100, 200, 3), 100, dtype=np.uint8)
        right = np.full((100, 200, 3), 150, dtype=np.uint8)
        anag = np.full((100, 200, 3), 128, dtype=np.uint8)
        top = np.full((100, 200, 3), 200, dtype=np.uint8)
        result = capture_still_set(
            left,
            right,
            anag,
            tmp_path,
            subject="test",
            method="wimmer",
            formats=(".jpg",),
            top_frame=top,
        )
        assert result is not None
        assert len(result.paths) == 4  # left, right, anaglyph, top

    def test_multiple_formats(self, tmp_path: Path) -> None:
        left = np.full((100, 200, 3), 100, dtype=np.uint8)
        result = capture_still_set(
            left,
            None,
            None,
            tmp_path,
            subject="test",
            method="wimmer",
            formats=(".jpg", ".tiff"),
        )
        assert result is not None
        assert len(result.paths) == 2  # left in both formats

    def test_none_frames_returns_none(self, tmp_path: Path) -> None:
        result = capture_still_set(None, None, None, tmp_path)
        assert result is None

    def test_auto_populates_metadata(self, tmp_path: Path) -> None:
        frame = np.full((100, 200, 3), 128, dtype=np.uint8)
        result = capture_still_set(
            frame,
            None,
            None,
            tmp_path,
            subject="auto",
            method="dubois",
            formats=(".jpg",),
        )
        assert result is not None
        assert result.metadata.subject == "auto"
        assert result.metadata.anaglyph_method == "dubois"
        assert result.metadata.timestamp != ""
