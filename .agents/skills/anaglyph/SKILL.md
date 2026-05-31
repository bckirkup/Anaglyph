---
name: anaglyph-stereo-microscopy
description: >
  Build, test, and operate the Anaglyph stereo microscopy application.
  Covers camera setup, calibration, capture, and anaglyph compositing
  for dual-camera AmScope stereoscopes.
---

## Overview

Anaglyph produces live red-cyan 3D anaglyphs from a trinocular stereoscope
with two AmScope MD500L cameras (left/right stereo pair) and an optional
AmScope MU503 (top-down USB3.0 camera). The GUI uses PyQt6 with OpenCV.

## Install

```bash
pip install -e ".[dev]"
pre-commit install
```

## Lint & Type Check

```bash
ruff check .
ruff format --check .
mypy main.py camera_manager.py calibration.py gui.py --ignore-missing-imports
```

## Run Unit Tests (no hardware)

```bash
pytest tests/ -v -m "not hardware" --tb=short
```

## Run Hardware Tests (requires cameras + human operator)

```bash
pytest tests/hardware/ -v --hardware
```

These tests print step-by-step instructions. The operator must have the
stereoscope connected and red/cyan 3D glasses available.

## Launch GUI

```bash
python main.py --gui
```

## Verify Camera Detection

```bash
python main.py --verify
```

## Key Architecture

- `main.py` — CLI entry point (`--verify`, `--gui`)
- `camera_manager.py` — USB camera discovery via VID/PID, left/right/top
  assignment persisted in `camera_config.json`
- `gui.py` — PyQt6 window: camera previews, focus stoplights, ORB-based
  alignment, three-way overlay, red-cyan anaglyph, lock/unlock transforms,
  save JPG
- `calibration.py` — Checkerboard detection (stereo calibration is WIP)

## Camera Hardware

- **MD500L** (stereo pair): VID `0x0AC8`, USB 2.0, accessed via MSMF/V4L2
- **MU503** (top-down): VID `0x0547`, USB 3.0, often requires DSHOW on Windows
- Both cameras are UVC-compliant; OpenCV `VideoCapture` works on all platforms
- Use MJPEG fourcc when running ≥2 cameras to avoid USB bandwidth limits
- **Trinocular slider**: physical slider on the head toggles optical path
  between eyepieces (left+right cameras) and top-down camera (MU503).
  The GUI "shutter" radio buttons must match the slider position.

## Illumination

- **Gooseneck** (epi/reflected): for opaque specimens; adjustable position
- **Substage** (transmitted): for transparent/translucent specimens; built-in
  or external light source below the stage
- Different illumination modes affect exposure matching between cameras

## Anaglyph Methods

The app currently uses Wimmer-method compositing:
- Red channel (BGR index 2) = left camera grayscale (for red-filter left eye)
- Green+Blue channels (BGR indices 0,1) = right camera grayscale (cyan right eye)

## Cross-Platform Notes

- **Windows**: MSMF + DSHOW backends; `cv2-enumerate-cameras` for VID/PID
- **Linux**: V4L2 backend; cameras at `/dev/videoN`; may need
  `sudo modprobe uvcvideo` and `usbfs_memory_mb` increase for multi-camera
- **macOS**: AVFoundation backend

## Test Accounts / Secrets

No authentication required. All camera access is local USB.
