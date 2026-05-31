# Architecture

## Module Overview

```
main.py ──┬── camera_manager.py  (hardware discovery, assignment, capture)
           ├── gui.py             (PyQt6 UI, alignment, compositing, display)
           └── calibration.py     (stereo calibration math — WIP)
```

## Data Flow

```
USB Cameras (MD500L × 2, MU503 × 1)
        │
   CameraManager
   ├─ discover_cameras()   →  enumerate USB devices by VID/PID
   ├─ get_stereo_pair()    →  assign Left / Right / Top from config
   └─ open_captures()      →  cv2.VideoCapture per camera
        │
   CaptureWorker (QThread)
   └─ grabs frames at ~30fps, emits frame_ready(cam_id, ndarray)
        │
   CameraSetupWindow (QMainWindow)
   ├─ _on_frame()           →  update preview + sharpness stoplight
   ├─ _update_alignment()   →  ORB + estimateAffinePartial2D
   │   ├─ compute_alignment()    →  AlignmentMetrics + 2x3 affine M
   │   ├─ _compose_affine()      →  chain Top→Right + Right→Left
   │   └─ _apply_180_flip()      →  resolve 180° ambiguity
   ├─ _update_anaglyph_and_overlay()
   │   ├─ build_anaglyph_overlap()   →  red=left, cyan=right, crop overlap
   │   └─ build_three_way_overlay()  →  grayscale blend in left's frame
   └─ save JPG (overlay / anaglyph)
```

## Key Design Decisions

### Camera Identity by VID/PID/Path

Cameras are identified by USB vendor/product ID and device path rather than
index. This ensures left/right assignment survives USB reconnections and
reboots. The mapping is persisted in `camera_config.json`.

### Alignment via ORB + Affine

Rather than full stereo calibration (which requires a known calibration
target), the app uses ORB feature matching with `estimateAffinePartial2D`
for real-time alignment. This handles rotation, scale, and translation
between any pair of views without a calibration step.

### Shutter Model

The trinocular head has a physical slider. The radio buttons tell the
software which cameras currently have an optical path (top-down OR
left+right eyepieces), so alignment is only computed for live pairs.

### Lock/Unlock

When examining a sample, the operator locks the current transforms.
New frames are composited using the frozen alignment so changing the
slide doesn't break the anaglyph.

### Parallax Preservation

The anaglyph compositor strips translation from the alignment transform
(`center_rotation_affine`), keeping only rotation and scale correction.
This preserves horizontal stereo disparity — the depth cue that
red/cyan glasses decode.

### Stereo Calibration

`calibration.py` provides full OpenCV stereo calibration:
- `CalibrationSession` — accumulates checkerboard poses from both cameras
- `stereo_calibrate()` — runs `cv2.stereoCalibrate` + `stereoRectify`
- `apply_rectification()` — warps frames through undistort+rectify maps
- Save/load to `.npz` files with JSON summary

When calibration is loaded, rectification is applied to each frame
before compositing, correcting lens distortion and aligning epipolar
geometry.

### Still Capture

`still_capture.py` saves full-resolution stills (TIFF + JPEG) with
JSON metadata (timestamp, cameras, anaglyph method, alignment params,
calibration RMS).

## Future Architecture (Planned)

### Z-Stack
- Guided focal sweep acquisition + Laplacian pyramid stacking

### Gallery
- `gallery/catalog.py` — SQLite capture catalog with tagging
- `gallery/export.py` — Static HTML gallery generator

### Cross-Platform
- `camera/enumerator.py` — Platform-specific USB topology parsing
  (Linux sysfs, macOS IOKit, Windows SetupAPI)
