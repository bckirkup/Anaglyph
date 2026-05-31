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

The trinocular head has a physical beam splitter. The "shutter" radio
buttons tell the software which cameras currently have an optical path
(top+right vs left+right), so alignment is only computed for live pairs.

### Lock/Unlock

When examining a sample, the operator locks the current transforms.
New frames are composited using the frozen alignment so changing the
slide doesn't break the anaglyph.

## Future Architecture (Planned)

### Phase 2: Capture Modes
- `capture/still.py` — Full-resolution still with EXIF-like metadata
- `capture/video.py` — Synchronized stereo video recording
- `capture/zstack.py` — Guided focal sweep + Laplacian pyramid stacking

### Phase 3: Gallery
- `gallery/catalog.py` — SQLite capture catalog with tagging
- `gallery/export.py` — Static HTML gallery generator

### Phase 4: Cross-Platform
- `camera/enumerator.py` — Platform-specific USB topology parsing
  (Linux sysfs, macOS IOKit, Windows SetupAPI)
