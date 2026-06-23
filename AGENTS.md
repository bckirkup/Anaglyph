# AGENTS.md — AI Agent Guidelines for Anaglyph

## Repository Purpose
Real-time stereoscopic microscopy system for 3D anaglyph visualization.
Facilitates hardware alignment, calibration, and synchronized recording
for dual-camera AmScope stereoscope setups. Uses PyQt6 + OpenCV.

## Setup
```bash
pip install -e ".[dev]"
pre-commit install
```

## Validation Commands
Run these before committing:
```bash
ruff check .
ruff format --check .
mypy main.py camera_manager.py calibration.py gui.py --ignore-missing-imports
pytest tests/ -v -m "not hardware" --tb=short
```

## Architecture Rules
- **Local hardware only** — no external APIs or services; all camera access is USB
- **Cross-platform** — must work on Windows (MSMF/DSHOW), Linux (V4L2), macOS (AVFoundation)
- **VID/PID camera identity** — cameras are identified by hardware IDs, not port indices
- **Never modify tests to make them pass** — fix the implementation

## Key Files
| File | Purpose |
|------|---------|
| `main.py` | CLI entrypoint (`--verify`, `--gui`) |
| `camera_manager.py` | USB camera discovery via VID/PID, left/right/top assignment |
| `gui.py` | PyQt6 window: previews, stoplights, alignment, anaglyph compositing |
| `compositor.py` | Image alignment, warping, and anaglyph blending |
| `calibration.py` | Checkerboard-based stereo rectification |
| `video_recorder.py` | Synchronized multi-stream MP4 encoding |
| `still_capture.py` | TIFF/JPG/JSON capture and metadata serialization |
| `tests/` | Unit testing suite (no hardware required) |

## Camera Hardware
- **MD500L** (stereo pair): VID `0x0AC8`, USB 2.0, UVC-compliant
- **MU503** (top-down): VID `0x0547`, USB 3.0, often requires DSHOW on Windows
- Use MJPEG fourcc when running ≥2 cameras to avoid USB bandwidth limits
- **Trinocular slider**: physical slider toggles optical path between eyepieces and top camera

## Key Design Principles
1. **Wimmer-method compositing**: Red channel = left grayscale, Green+Blue = right grayscale
2. **Affine alignment**: M_right_to_left 2x3 matrix maps right camera space to left
3. **Lock transforms**: Freeze alignment matrices during slide changes
4. **Stoplight focus indicator**: Visual metric based on Laplacian variance

## Code Conventions
- Python 3.11+ with type hints
- OpenCV (cv2) for all image processing
- PyQt6 for GUI
- Ruff for linting and formatting
- Tests use pytest; hardware tests are marked and require physical cameras

## PR Requirements
- All ruff checks pass
- mypy passes on core modules
- All non-hardware tests pass
- New features include tests
- Cross-platform considerations documented if applicable
