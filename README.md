# Anaglyph Stereoscope

Live 3D red-cyan anaglyph compositing from dual-camera stereoscopes.

Manages two AmScope MD500L cameras (left/right stereo pair) and an optional
AmScope MU503 (top-down USB 3.0 camera) attached to a trinocular stereoscope.
Produces real-time anaglyph video for viewing with red/cyan 3D glasses.

## Features

- **Live anaglyph preview** — real-time red-cyan compositing with overlap cropping
- **ORB-based alignment** — automatic rotation, scale, and translation estimation
- **Focus stoplights** — Laplacian-variance sharpness indicator per camera
- **Three-way overlay** — grayscale blend of all cameras aligned to a common reference
- **Lock/unlock transforms** — freeze alignment while changing slides
- **Still capture** — save overlay and anaglyph as JPEG
- **Persistent camera mapping** — left/right assignment by VID/PID survives restarts

## Requirements

- Python 3.10+
- Two USB cameras (AmScope MD500L or any UVC-compliant pair)
- Optional: AmScope MU503 (top-down camera)
- Red/cyan 3D glasses for anaglyph viewing

## Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (choose your platform)
# Linux/macOS:
source venv/bin/activate
# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Install (choose one)
pip install -r requirements.txt          # runtime only
pip install -r requirements-dev.txt      # runtime + dev tools (lint, test, etc.)
pip install -e ".[dev]"                  # editable install with dev extras

# Install pre-commit hooks (if using dev setup)
pre-commit install
```

### Platform-Specific Notes

**Linux**: If multiple USB cameras fail with `ENOSPC`, increase the USB
buffer:

```bash
sudo sh -c 'echo 256 > /sys/module/uvcvideo/parameters/usbfs_memory_mb'
```

**Windows**: The `cv2-enumerate-cameras` package is recommended for
reliable VID/PID discovery. Install with:

```bash
pip install -e ".[enumerate]"
```

## Usage

### Verify camera detection

```bash
python main.py --verify
```

### Launch GUI

```bash
python main.py --gui
```

See [HOW_TO_USE.txt](HOW_TO_USE.txt) for detailed operating instructions.

## Project Structure

```
├── main.py               # CLI entry point (--verify, --gui)
├── camera_manager.py     # USB camera discovery, left/right/top assignment
├── gui.py                # PyQt6 GUI: previews, alignment, anaglyph
├── compositor.py         # Anaglyph compositing (Wimmer, Dubois, half-color, gray)
├── video_recorder.py     # Synchronized stereo MP4 recording
├── calibration.py        # Stereo calibration (checkerboard detection + WIP)
├── tests/
│   ├── unit/             # Automated tests (no hardware required)
│   └── hardware/         # Prompted human-in-the-loop tests
├── docs/
│   ├── architecture.md   # Module design and data flow
│   └── hardware_guide.md # Camera setup, wiring, USB topology
├── .github/workflows/
│   └── ci.yml            # Lint + test matrix (Python 3.11, 3.12)
├── pyproject.toml        # PEP 621 project metadata
├── requirements.txt      # Runtime dependencies
├── requirements-dev.txt  # Dev dependencies (includes requirements.txt)
└── HOW_TO_USE.txt        # Detailed operating instructions
```

## Development

### Lint & type check

```bash
ruff check .
ruff format --check .
mypy main.py camera_manager.py calibration.py gui.py --ignore-missing-imports
```

### Run tests

```bash
# Unit + integration tests (no cameras needed)
pytest tests/ -v -m "not hardware"

# Hardware tests (requires cameras + human operator)
pytest tests/hardware/ -v --hardware
```

## Roadmap

- [x] Dubois / half-color / gray anaglyph methods
- [x] Video recording (synchronized stereo MP4)
- [ ] Full stereo calibration (rectification maps, undistortion)
- [ ] Z-stack acquisition with guided focal sweep
- [ ] Focus stacking (Laplacian pyramid)
- [ ] Web gallery export
- [ ] Cross-platform camera enumerator (Linux sysfs, macOS IOKit)

## License

[GNU General Public License v3.0](LICENSE)
