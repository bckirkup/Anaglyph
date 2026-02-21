# Anaglyph Stereoscope

A modular Windows 11 Python application for aligning two AmScope MD500L cameras into a live 3D anaglyph (Cyan/Magenta) video stream.

## Project Structure

```
.
├── requirements.txt      # Dependencies
├── main.py               # Entry point (--verify, --gui)
├── camera_manager.py     # HW access, VID/PID discovery, Left/Right persistence
├── calibration.py        # Stereo calibration (stereoCalibrate, rectify maps)
├── gui.py                # PyQt6 GUI (Calibration / Live 3D modes)
├── camera_config.json    # Auto-generated: Left/Right camera mapping
└── README.md
```

## Setup

```powershell
cd "c:\Users\bckir\OneDrive\Documents\Anaglyph Program FEB2026"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Verify Hardware Access

With both AmScope MD500L cameras connected:

```powershell
python main.py --verify
```

Or run the camera manager directly:

```powershell
python camera_manager.py
```

This will:
1. Discover cameras via VID/PID (using `cv2-enumerate-cameras`)
2. Assign Left and Right based on persisted config (or first/second found)
3. Open both captures and read a test frame
4. Save the mapping to `camera_config.json` for consistent assignment on restart

## Next Steps

- **Calibration**: Implement full stereo calibration in `calibration.py`
- **Anaglyph Engine**: Cyan/Magenta overlay with rectification
- **GUI**: Mode toggle, still capture (JPG), video record (MP4)
