"""
Still Capture — Full-resolution still image capture with metadata.

Supports TIFF (lossless) and JPEG output with embedded metadata including
timestamp, camera info, anaglyph method, and alignment parameters.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CaptureMetadata:
    """Metadata for a captured still image."""

    timestamp: str = ""
    subject: str = ""
    anaglyph_method: str = ""
    left_camera: str = ""
    right_camera: str = ""
    top_camera: str = ""
    image_width: int = 0
    image_height: int = 0
    alignment_rotation_deg: float = 0.0
    alignment_scale: float = 1.0
    alignment_tx: float = 0.0
    alignment_ty: float = 0.0
    calibration_rms: float | None = None
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        # Remove empty extra
        if not d["extra"]:
            del d["extra"]
        return d


@dataclass
class CaptureResult:
    """Result of a still capture operation."""

    paths: list[Path]
    metadata: CaptureMetadata
    metadata_path: Path | None = None


def generate_filename(
    output_dir: Path,
    subject: str = "",
    method: str = "",
    suffix: str = ".jpg",
) -> Path:
    """Generate a timestamped filename for a capture."""
    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    parts = [ts]
    if subject:
        safe_subject = "".join(c if c.isalnum() or c in "-_" else "_" for c in subject)
        parts.append(safe_subject)
    if method:
        parts.append(method)
    name = "_".join(parts) + suffix
    return output_dir / name


def save_still(
    frame: np.ndarray,
    path: Path,
    jpeg_quality: int = 95,
) -> bool:
    """Save a single frame as JPEG or TIFF depending on extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    try:
        if ext in (".tif", ".tiff"):
            cv2.imwrite(str(path), frame)
        elif ext in (".jpg", ".jpeg"):
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        elif ext == ".png":
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        else:
            cv2.imwrite(str(path), frame)
        logger.info("Saved still: %s (%dx%d)", path, frame.shape[1], frame.shape[0])
        return True
    except Exception as e:
        logger.error("Failed to save %s: %s", path, e)
        return False


def save_metadata(metadata: CaptureMetadata, path: Path) -> None:
    """Save capture metadata as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata.to_dict(), indent=2))
    logger.info("Saved metadata: %s", path)


def capture_still_set(
    left_frame: np.ndarray | None,
    right_frame: np.ndarray | None,
    anaglyph_frame: np.ndarray | None,
    output_dir: Path,
    subject: str = "",
    method: str = "wimmer",
    metadata: CaptureMetadata | None = None,
    formats: tuple[str, ...] = (".tiff", ".jpg"),
    top_frame: np.ndarray | None = None,
) -> CaptureResult | None:
    """
    Capture a full set of still images: left, right, anaglyph, and optionally top.

    Saves each in all requested formats, plus a metadata JSON.

    Args:
        left_frame: Left camera frame (BGR).
        right_frame: Right camera frame (BGR).
        anaglyph_frame: Composited anaglyph frame (BGR).
        output_dir: Directory to save files in.
        subject: Subject name for filename.
        method: Anaglyph method name for filename.
        metadata: Pre-populated metadata (timestamp/cameras filled if empty).
        formats: File extensions to save (e.g., (".tiff", ".jpg")).
        top_frame: Optional top-down camera frame.

    Returns:
        CaptureResult or None if nothing to save.
    """
    if left_frame is None and right_frame is None and anaglyph_frame is None:
        logger.warning("No frames to capture")
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if metadata is None:
        metadata = CaptureMetadata()
    if not metadata.timestamp:
        metadata.timestamp = datetime.now(tz=timezone.utc).isoformat()
    if not metadata.subject:
        metadata.subject = subject
    if not metadata.anaglyph_method:
        metadata.anaglyph_method = method

    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    safe_subject = "".join(c if c.isalnum() or c in "-_" else "_" for c in subject) if subject else ""

    saved_paths: list[Path] = []

    frames = {
        "left": left_frame,
        "right": right_frame,
        "anaglyph": anaglyph_frame,
        "top": top_frame,
    }

    for label, frame in frames.items():
        if frame is None:
            continue
        if label == "anaglyph" and metadata.image_width == 0:
            metadata.image_width = frame.shape[1]
            metadata.image_height = frame.shape[0]
        for fmt in formats:
            parts = [ts]
            if safe_subject:
                parts.append(safe_subject)
            parts.append(label)
            if label == "anaglyph":
                parts.append(method)
            name = "_".join(parts) + fmt
            path = output_dir / name
            if save_still(frame, path):
                saved_paths.append(path)

    # Save metadata
    meta_parts = [ts]
    if safe_subject:
        meta_parts.append(safe_subject)
    meta_parts.append("metadata")
    meta_path = output_dir / ("_".join(meta_parts) + ".json")
    save_metadata(metadata, meta_path)

    return CaptureResult(
        paths=saved_paths,
        metadata=metadata,
        metadata_path=meta_path,
    )
