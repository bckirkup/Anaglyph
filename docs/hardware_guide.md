# Hardware Guide

## Bill of Materials

| Component | Model | Interface | Notes |
|---|---|---|---|
| Stereoscope | AmScope trinocular | — | White base, dual-knob zoom |
| Left/Right cameras | AmScope MD500L × 2 | USB 2.0 | VID `0x0AC8`, eyepiece adapters |
| Top camera (optional) | AmScope MU503 | USB 3.0 | VID `0x0547`, C-mount on trinocular port |
| Gooseneck illuminator | LED gooseneck | — | Epi/reflected illumination, adjustable intensity |
| Substage illuminator | Built-in or external | — | Transmitted illumination for transparent specimens |
| Trinocular slider | Built-in on head | — | Toggles optical path: eyepieces ↔ top-down camera |
| USB hub(s) | Any powered hub | USB 3.0 | See topology section |

## USB Topology

### Recommended Setup

```
Host PC
├── USB 3.0 Root Hub 1
│   └── MU503 (top camera, bandwidth-hungry)
└── USB 3.0 Root Hub 2
    ├── MD500L Left
    └── MD500L Right
```

**Key rule**: Put the USB 3.0 camera (MU503) on a **separate root hub**
from the USB 2.0 cameras (MD500L). Mixing USB 2.0 and 3.0 devices on the
same hub can cause bandwidth contention.

### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| "No space left on device" | USB bandwidth exceeded | Move camera to different root hub |
| Camera not detected | Driver issue | `sudo modprobe uvcvideo` (Linux) |
| Frames drop / stutter | Bandwidth contention | Use MJPEG: `cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))` |
| Left/right swapped | Camera assignment | Run calibration wizard or swap in config |

## Camera Settings

### MD500L (Stereo Pair)

- Sensor: 5MP CMOS
- USB VID: `0x0AC8`
- Default resolution: 2592 × 1944 (may auto-negotiate lower)
- Exposure: Auto (can be locked for stereo matching)

### MU503 (Top-Down)

- Sensor: 5MP APTINA CMOS
- USB VID: `0x0547`, PIDs: `0x3510`, `0x3511`, `0x4510`, `0x4511`
- Interface: USB 3.0 (may fall back to USB 2.0 mode)
- On Windows, often requires DSHOW backend; on Linux, V4L2

## Physical Setup

1. Mount MD500L cameras on both eyepiece tubes using C-mount adapters
2. Mount MU503 on the trinocular port (top)
3. Connect each camera to a USB port (see topology above)
4. Set up illumination:
   - **Gooseneck** (epi/reflected): for opaque specimens — position for even, shadow-free lighting
   - **Substage** (transmitted): for transparent/translucent specimens — adjust intensity for even field
5. Run `python main.py --verify` to confirm all cameras are detected

## Trinocular Slider

The trinocular head has a physical slider that directs light either to the
eyepieces or to the top-down camera port (MU503):

| Slider Position | Light Path | GUI Shutter Setting |
|---|---|---|
| **Eyepieces** | Left + Right eyepiece cameras receive light | "Left and Right" |
| **Top camera** | MU503 top-down camera receives light; one eyepiece may also receive light | "Top and Right" |

The GUI's "Which has the shutter open?" radio buttons correspond to the
physical slider position. Set them to match the slider so that alignment
is computed only for cameras that currently have an optical path.

## Focus Procedure

1. Place a high-contrast target on the stage (printed grid, resolution chart)
2. Adjust the coarse focus knob until the target is roughly in view
3. Use the fine focus knob for sharp focus
4. Watch the GUI's sharpness stoplights:
   - **Green** (≥500): Excellent focus
   - **Yellow** (300–500): Improving
   - **Red** (100–300): Poor focus
   - **Gray** (<100): Very poor or no signal
