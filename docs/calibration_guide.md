# Stereo Calibration Guide

## Overview

Stereo calibration computes the geometric relationship between the left and right cameras. This enables:
- **Lens distortion correction** — removes barrel/pincushion distortion
- **Stereo rectification** — aligns epipolar lines horizontally for better anaglyphs
- **Accurate depth estimation** — needed for future z-stack and depth-map features

## What You Need

1. **Printed checkerboard pattern** — 9×6 inner corners (10×7 squares)
   - Print on flat, rigid material (mount on cardboard or acrylic)
   - Standard size: letter/A4 paper, squares ~15-25mm
   - The pattern must be flat — warped paper will degrade calibration
2. **Both MD500L cameras active** — trinocular slider set to "Eyepieces"
3. **Good, even illumination** — use gooseneck light, minimize shadows

## Calibration Wizard (GUI)

### Step 1: Start the Wizard

- Menu: **Calibration → Start Calibration Wizard** (or `Ctrl+K`)
- Or click the **Calibrate...** button in the sidebar
- The wizard requires both left and right cameras to be producing frames

### Step 2: Capture Poses

The wizard will ask you to capture checkerboard poses:

1. Place the checkerboard under the stereoscope
2. Click **Yes** to capture the current pose
3. The software detects corners in both cameras simultaneously
4. If detection fails, you'll see which camera(s) missed — reposition and try again
5. **Move/tilt/rotate** the checkerboard between captures for diversity

**Recommendations:**
- Capture **8–12 poses** for good calibration
- Minimum is **5 poses** (3 absolute minimum, but quality suffers)
- Cover different positions: center, edges, corners of the field of view
- Tilt the board at various angles (15–30° from flat)
- Keep the board in focus and fully visible in both cameras

### Step 3: Calibrate

Click **No** when you have enough poses. The software will:

1. Run individual camera calibration (intrinsic parameters)
2. Run stereo calibration (extrinsic parameters: rotation + translation)
3. Compute rectification maps

### Step 4: Review Results

The wizard reports:
- **RMS reprojection error** — should be < 1.0 pixel for good calibration
  - < 0.5: excellent
  - 0.5–1.0: good
  - 1.0–2.0: acceptable
  - > 2.0: poor, consider re-calibrating with more/better poses
- **Baseline** — distance between cameras in checkerboard-square units
- **Rectification status** — whether undistort+rectify maps were computed

### Step 5: Save

Menu: **Calibration → Save Calibration**

Saves two files:
- `.npz` — full calibration data (camera matrices, distortion, rectification maps)
- `.json` — human-readable summary (RMS error, baseline, pose count)

## Loading Existing Calibration

Menu: **Calibration → Load Calibration**

Once loaded, rectification is automatically applied to incoming frames.

## Tips

- **Re-calibrate when**: you change the zoom level, swap cameras, or notice poor alignment
- **Consistent zoom**: calibration is specific to the current magnification setting
- **Clean checkerboard**: dust or marks on the pattern can confuse corner detection
- **Frame size**: calibration is tied to the image resolution — don't change resolution after calibrating

## How Rectification Works

After calibration, each incoming frame is warped through an undistort+rectify map:

```
raw frame → cv2.remap(frame, map1, map2) → rectified frame
```

This corrects lens distortion and aligns the epipolar geometry so that corresponding points in left and right images are on the same horizontal scan line. The anaglyph compositor then only needs to handle horizontal disparity (parallax = depth).

## Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| "Checkerboard not detected" | Board not in view, too small, or poor lighting | Reposition, zoom out, increase light |
| High RMS error (> 2.0) | Too few poses, poses too similar, or board not flat | Recalibrate with 10+ diverse poses |
| Rectified image looks warped | Wrong image size or zoom changed | Recalibrate at current settings |
| Calibration fails entirely | Degenerate poses (all same position) | Vary position, angle, and distance |
