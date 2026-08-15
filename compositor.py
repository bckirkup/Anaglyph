"""
Compositor — Anaglyph compositing, alignment, and sharpness metrics.

Provides multiple anaglyph methods (Wimmer, Dubois, half-color, gray),
ORB-based alignment, affine transform utilities, and focus quality measurement.
"""

from __future__ import annotations

import enum
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Anaglyph method enum
# ---------------------------------------------------------------------------


class AnaglyphMethod(enum.Enum):
    """Available anaglyph compositing methods."""

    WIMMER = "wimmer"
    DUBOIS = "dubois"
    HALF_COLOR = "half_color"
    GRAY = "gray"


# ---------------------------------------------------------------------------
# Dubois optimized matrices (sRGB, for red-cyan glasses)
# Source: Eric Dubois, "A Projection Method for Producing Anaglyph
#         Stereo Images", IEEE ICASSP 2001.
# Each matrix maps [R, G, B] of one eye to [R, G, B] output contribution.
# ---------------------------------------------------------------------------

# Left eye contribution (seen through red filter)
_DUBOIS_LEFT = np.array(
    [
        [0.4561, 0.500484, 0.176381],
        [-0.0400822, -0.0378246, -0.0157589],
        [-0.0152161, -0.0205971, -0.00546856],
    ],
    dtype=np.float64,
)

# Right eye contribution (seen through cyan filter)
_DUBOIS_RIGHT = np.array(
    [
        [-0.0434706, -0.0879388, -0.00155529],
        [0.378476, 0.73364, -0.0184503],
        [-0.0721527, -0.112961, 1.2264],
    ],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# Alignment data structures
# ---------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402


@dataclass
class AlignmentMetrics:
    """Rotation (deg), scale (ratio), translation (px) between two images."""

    rotation_deg: float = 0.0
    scale: float = 1.0
    tx: float = 0.0
    ty: float = 0.0
    score: float = 0.0  # 0-100, higher = better aligned
    valid: bool = False


# ---------------------------------------------------------------------------
# Affine transform utilities
# ---------------------------------------------------------------------------


def affine_to_metrics(M: np.ndarray | None) -> AlignmentMetrics:
    """Decompose 2x3 similarity matrix into AlignmentMetrics."""
    out = AlignmentMetrics()
    if M is None or M.size == 0:
        return out
    a, _, tx = M[0]
    c, _, ty = M[1]
    scale = np.sqrt(a * a + c * c)
    rotation_rad = np.arctan2(c, a)
    rotation_deg = normalize_rotation_deg(np.degrees(rotation_rad))
    out.rotation_deg = float(rotation_deg)
    out.scale = float(scale)
    out.tx = float(tx)
    out.ty = float(ty)
    out.valid = True
    rot_penalty = min(abs(rotation_deg) * 2, 50)
    scale_penalty = min(abs(1.0 - scale) * 100, 30)
    trans_penalty = min(np.sqrt(tx * tx + ty * ty) * 0.1, 20)
    out.score = max(0, 100 - rot_penalty - scale_penalty - trans_penalty)
    return out


def compose_affine(M1: np.ndarray, M2: np.ndarray) -> np.ndarray:
    """Compose two 2x3 affine matrices: result maps via M1 then M2."""
    R1, t1 = M1[:, :2], M1[:, 2:3]
    R2, t2 = M2[:, :2], M2[:, 2:3]
    R = R2 @ R1
    t = R2 @ t1 + t2
    return np.hstack([R, t])


def invert_affine(M: np.ndarray) -> np.ndarray:
    """Invert a 2x3 similarity matrix."""
    R, t = M[:, :2], M[:, 2:3]
    R_inv = np.linalg.inv(R)
    t_inv = -R_inv @ t
    return np.hstack([R_inv, t_inv])


def normalize_rotation_deg(deg: float) -> float:
    """Normalize angle to [-180, 180]."""
    d = deg % 360.0
    if d > 180:
        d -= 360
    elif d <= -180:
        d += 360
    return d


def apply_180_flip_to_transform(M: np.ndarray, src_w: int, src_h: int) -> np.ndarray:
    """Apply 180deg rotation around source image center to an affine transform."""
    R, t = M[:, :2], M[:, 2:3]
    cx, cy = src_w / 2.0, src_h / 2.0
    R_flip = -np.eye(2)
    t_flip = np.array([[2 * cx], [2 * cy]])
    R_new = R @ R_flip
    t_new = R @ t_flip + t
    return np.hstack([R_new, t_new])


# ---------------------------------------------------------------------------
# Feature-based alignment
# ---------------------------------------------------------------------------


def compute_alignment(img1: np.ndarray, img2: np.ndarray) -> tuple[AlignmentMetrics, np.ndarray | None]:
    """
    Estimate similarity transform (rotation, scale, translation) from img1 to img2
    using ORB features + RANSAC.

    Returns (metrics, 2x3 affine matrix) or (invalid_metrics, None).
    """
    out = AlignmentMetrics()
    if img1 is None or img2 is None or img1.size == 0 or img2.size == 0:
        return out, None
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
    h1, w1 = gray1.shape
    h2, w2 = gray2.shape
    if min(h1, w1, h2, w2) < 32:
        return out, None
    target = 320
    r1, r2 = 1.0, 1.0
    if max(w1, h1) > target or max(w2, h2) > target:
        r1 = target / max(w1, h1)
        r2 = target / max(w2, h2)
        gray1 = cv2.resize(gray1, None, fx=r1, fy=r1)
        gray2 = cv2.resize(gray2, None, fx=r2, fy=r2)
    orb = cv2.ORB_create(nfeatures=800)
    kp1, d1 = orb.detectAndCompute(gray1, None)
    kp2, d2 = orb.detectAndCompute(gray2, None)
    if d1 is None or d2 is None or len(kp1) < 4 or len(kp2) < 4:
        return out, None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(d1, d2)
    if len(matches) < 4:
        return out, None
    matches = sorted(matches, key=lambda m: m.distance)[:80]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    try:
        M, _ = cv2.estimateAffinePartial2D(pts1, pts2)
    except cv2.error:
        return out, None
    if M is None:
        return out, None
    if r1 != 1.0 or r2 != 1.0:
        scale = r1 / r2
        M = np.array(
            [[M[0, 0] * scale, M[0, 1] * scale, M[0, 2] / r2], [M[1, 0] * scale, M[1, 1] * scale, M[1, 2] / r2]],
            dtype=np.float64,
        )
    out = affine_to_metrics(M)
    return out, M


# ---------------------------------------------------------------------------
# Sharpness metric
# ---------------------------------------------------------------------------


def compute_sharpness(frame: np.ndarray | None) -> float:
    """Laplacian variance — higher = sharper. Used for focus indicator."""
    if frame is None or frame.size == 0:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


# ---------------------------------------------------------------------------
# Anaglyph compositing
# ---------------------------------------------------------------------------


def strip_translation(M: np.ndarray) -> np.ndarray:
    """Remove translation from a 2x3 affine, keeping rotation+scale only.

    The resulting transform rotates/scales around the image center so the
    right-eye image is rotationally aligned but horizontal parallax (the
    stereo depth cue) is preserved.
    """
    R = M[:, :2].copy()
    # Rotation+scale around center: t' = center - R @ center
    # We don't know the image size here, so return a zero-translation
    # version; the caller should use center_rotation_affine() instead
    # for center-relative rotation.
    return np.hstack([R, np.zeros((2, 1))])


def center_rotation_affine(M: np.ndarray, w: int, h: int) -> np.ndarray:
    """Build a 2x3 affine that applies M's rotation+scale around image center.

    Translation is removed so that stereo parallax (horizontal disparity
    between left and right eyes) is preserved — this is the depth cue that
    red/cyan glasses decode.
    """
    R = M[:, :2].copy()
    cx, cy = w / 2.0, h / 2.0
    center = np.array([[cx], [cy]])
    # Rotate around center: new_point = R @ (point - center) + center
    t = center - R @ center
    return np.hstack([R, t])


def build_anaglyph(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    M_right_to_left: np.ndarray,
    method: AnaglyphMethod = AnaglyphMethod.WIMMER,
    parallax_offset_px: float = 0.0,
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    """
    Build anaglyph image for red/cyan glasses.

    The alignment transform is used for rotation+scale correction only;
    translation is stripped so that horizontal stereo parallax is preserved.
    A manual parallax_offset_px can be added to enhance or create the 3D
    depth effect when the natural stereo baseline is too small.

    Args:
        left_bgr: Left camera frame (BGR).
        right_bgr: Right camera frame (BGR).
        M_right_to_left: 2x3 affine mapping right coords to left coords.
        method: Compositing method.
        parallax_offset_px: Additional horizontal shift (pixels) applied to
            the right image.  Positive = right image shifts right (objects
            appear to recede), negative = shifts left (objects appear closer).

    Returns:
        (anaglyph_bgr, roi) or (None, None).
    """
    if left_bgr is None or right_bgr is None or M_right_to_left is None:
        return None, None
    h_l, w_l = left_bgr.shape[:2]
    h_r, w_r = right_bgr.shape[:2]
    if min(h_l, w_l, h_r, w_r) < 8:
        return None, None

    # Strip translation to preserve stereo parallax; only correct rotation+scale
    M_rot_only = center_rotation_affine(M_right_to_left, w_r, h_r)
    # Apply manual parallax offset (horizontal shift of right image)
    if parallax_offset_px != 0.0:
        M_rot_only[0, 2] += parallax_offset_px
    right_warped = cv2.warpAffine(right_bgr, M_rot_only, (w_l, h_l))

    if method == AnaglyphMethod.WIMMER:
        out = _anaglyph_wimmer(left_bgr, right_warped)
    elif method == AnaglyphMethod.DUBOIS:
        out = _anaglyph_dubois(left_bgr, right_warped)
    elif method == AnaglyphMethod.HALF_COLOR:
        out = _anaglyph_half_color(left_bgr, right_warped)
    elif method == AnaglyphMethod.GRAY:
        out = _anaglyph_gray(left_bgr, right_warped)
    else:
        out = _anaglyph_wimmer(left_bgr, right_warped)

    roi = _compute_overlap_roi(right_warped, w_l, h_l)
    if roi is not None:
        x, y, w, h = roi
        out = out[y : y + h, x : x + w]

    return out, roi


def _anaglyph_wimmer(left_bgr: np.ndarray, right_warped: np.ndarray) -> np.ndarray:
    """Wimmer method: R=left_gray, G=B=right_gray."""
    h, w = left_bgr.shape[:2]
    left_g = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    right_g = cv2.cvtColor(right_warped, cv2.COLOR_BGR2GRAY)
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:, :, 0] = right_g  # B (cyan component)
    out[:, :, 1] = right_g  # G (cyan component)
    out[:, :, 2] = left_g  # R (red component)
    return out


def _anaglyph_dubois(left_bgr: np.ndarray, right_warped: np.ndarray) -> np.ndarray:
    """Dubois optimized method: best color reproduction, least ghosting."""
    left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    right_rgb = cv2.cvtColor(right_warped, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0

    h, w = left_rgb.shape[:2]
    left_flat = left_rgb.reshape(-1, 3)
    right_flat = right_rgb.reshape(-1, 3)

    # Apply Dubois matrices: out_rgb = L @ left_rgb.T + R @ right_rgb.T
    result = (left_flat @ _DUBOIS_LEFT.T) + (right_flat @ _DUBOIS_RIGHT.T)
    result = np.clip(result, 0, 1)
    result = (result * 255).astype(np.uint8).reshape(h, w, 3)
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


def _anaglyph_half_color(left_bgr: np.ndarray, right_warped: np.ndarray) -> np.ndarray:
    """Half-color: left luminance -> red, right green+blue preserved."""
    left_g = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    out = np.zeros_like(left_bgr)
    out[:, :, 0] = right_warped[:, :, 0]  # B from right
    out[:, :, 1] = right_warped[:, :, 1]  # G from right
    out[:, :, 2] = left_g  # R from left (luminance)
    return out


def _anaglyph_gray(left_bgr: np.ndarray, right_warped: np.ndarray) -> np.ndarray:
    """Gray anaglyph: both eyes to grayscale first, pure depth, no color."""
    left_g = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    right_g = cv2.cvtColor(right_warped, cv2.COLOR_BGR2GRAY)
    out = np.zeros((*left_bgr.shape[:2], 3), dtype=np.uint8)
    out[:, :, 0] = right_g  # B
    out[:, :, 1] = right_g  # G
    out[:, :, 2] = left_g  # R
    return out


def _compute_overlap_roi(right_warped: np.ndarray, w_l: int, h_l: int) -> tuple[int, int, int, int] | None:
    """Compute overlap ROI from warped right image content region."""
    right_g = cv2.cvtColor(right_warped, cv2.COLOR_BGR2GRAY) if len(right_warped.shape) == 3 else right_warped
    right_has_content = right_g > 16
    if not np.any(right_has_content):
        return None
    ys, xs = np.where(right_has_content)
    x_min = int(np.clip(xs.min(), 0, w_l - 1))
    x_max = int(np.clip(xs.max() + 1, 1, w_l))
    y_min = int(np.clip(ys.min(), 0, h_l - 1))
    y_max = int(np.clip(ys.max() + 1, 1, h_l))
    margin = min(4, (x_max - x_min) // 8, (y_max - y_min) // 8)
    x_min = min(x_min + margin, x_max - 1)
    y_min = min(y_min + margin, y_max - 1)
    x_max = max(x_max - margin, x_min + 1)
    y_max = max(y_max - margin, y_min + 1)
    return (x_min, y_min, x_max - x_min, y_max - y_min)


# ---------------------------------------------------------------------------
# Three-way overlay
# ---------------------------------------------------------------------------


def build_three_way_overlay(
    top_bgr: np.ndarray | None,
    left_bgr: np.ndarray | None,
    right_bgr: np.ndarray | None,
    M_right_to_left: np.ndarray | None,
    M_top_to_left: np.ndarray | None,
    out_w: int = 420,
    out_h: int = 280,
) -> np.ndarray:
    """Blend all three images in left's reference frame (grayscale)."""
    if left_bgr is None:
        out = np.ones((out_h, out_w), dtype=np.uint8) * 32
        return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    h_l, w_l = left_bgr.shape[:2]
    left_g = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    blend = left_g.astype(np.float32)
    count = np.ones(left_g.shape, dtype=np.float32)
    if right_bgr is not None:
        if M_right_to_left is not None:
            right_w = cv2.warpAffine(right_bgr, M_right_to_left, (w_l, h_l))
        else:
            right_w = cv2.resize(right_bgr, (w_l, h_l))
        right_g = cv2.cvtColor(right_w, cv2.COLOR_BGR2GRAY)
        blend += right_g.astype(np.float32)
        count += 1.0
    if top_bgr is not None:
        if M_top_to_left is not None:
            top_w = cv2.warpAffine(top_bgr, M_top_to_left, (w_l, h_l))
        else:
            top_w = cv2.resize(top_bgr, (w_l, h_l))
        top_g = cv2.cvtColor(top_w, cv2.COLOR_BGR2GRAY)
        blend += top_g.astype(np.float32)
        count += 1.0
    out = np.clip(blend / np.maximum(count, 1), 0, 255).astype(np.uint8)
    out = cv2.resize(out, (out_w, out_h))
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
