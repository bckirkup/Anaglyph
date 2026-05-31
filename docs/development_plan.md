# Anaglyph — Updated Development Plan

_Revised 2026-05-31. Replaces the original greenfield plan with status-aware roadmap._

---

## 1. Current State (What Exists)

The codebase is a working stereoscope application (~2,600 lines of Python) with live anaglyph preview, camera management, and a PyQt6 GUI. Significant infrastructure and features are already in place.

### Completed (Phases 1–2 + bugfixes)

| Area | Status | Details |
|---|---|---|
| **CI/CD** | ✅ Done | GitHub Actions: ruff lint/format + pytest matrix (3.11, 3.12) |
| **Pre-commit** | ✅ Done | ruff lint + format hooks |
| **Packaging** | ✅ Done | `pyproject.toml` (hatchling), `requirements.txt`, `requirements-dev.txt` |
| **Camera discovery** | ✅ Done | VID/PID enumeration via `cv2-enumerate-cameras`, MSMF+DSHOW dual-backend, `alt_backends` fallback, persistent left/right/top assignment |
| **Camera open** | ✅ Done | Backend fallback (MSMF→DSHOW→CAP_ANY), MJPG fourcc for USB bandwidth |
| **Compositor** | ✅ Done | 4 anaglyph methods (Wimmer, Dubois, half-color, gray), ORB alignment, sharpness metric, three-way overlay |
| **Video recorder** | ✅ Done | `StereoVideoRecorder` with codec detection, frame-drop/drift monitoring |
| **GUI** | ✅ Done | Camera previews, focus stoplights, alignment metrics, anaglyph method dropdown, video record/stop, lock/unlock transforms, trinocular slider radio buttons, save JPG |
| **Unit tests** | ✅ Done | 61 tests (compositor, camera data structures, calibration, video recorder) |
| **HITL tests** | ✅ Done | 6 prompted hardware tests (`pytest --hardware`) |
| **Docs** | ✅ Done | `architecture.md`, `hardware_guide.md`, `README.md`, `SKILL.md` |
| **Shutter logic** | ✅ Fixed | Top OR eyepieces (not top+right) |

### Key Files

```
main.py               49 lines   CLI entry point
camera_manager.py     448 lines   USB discovery, assignment, capture
gui.py                805 lines   PyQt6 UI, previews, controls
compositor.py         374 lines   Anaglyph methods, alignment, sharpness
video_recorder.py     245 lines   Synchronized stereo MP4 recording
calibration.py         79 lines   Checkerboard detection (stereo calib is stub)
tests/                665 lines   61 unit + 6 HITL tests
```

### What's NOT Done Yet

| Area | Status | Gap |
|---|---|---|
| Stereo calibration | Stub only | `stereo_calibrate()`, `load/save_calibration()` are TODOs |
| Calibration wizard UI | Missing | No guided step-by-step flow |
| Still capture quality | Basic | JPG only, no metadata, no full-res capture |
| Z-stack acquisition | Missing | No focal sweep, no focus stacking |
| Gallery/catalog | Missing | No capture database, no web export |
| Color balance | Missing | No auto white-balance or exposure matching |
| Cross-platform enumerator | Missing | Only Windows MSMF/DSHOW; Linux/macOS not tested |
| CLI beyond --verify/--gui | Missing | No headless capture, no batch processing |
| Manual parallax adjustment | Missing | No arrow-key fine-tuning of alignment |

---

## 2. Hardware Profile

| Component | Details |
|---|---|
| **Microscope** | AmScope trinocular stereoscope (white base, dual-knob zoom) |
| **Stereo pair** | 2× AmScope MD500L — USB 2.0, VID `0x0AC8` |
| **Top-down camera** | AmScope MU503 — 5MP USB 3.0, VID `0x0547` |
| **Trinocular slider** | Physical slider: eyepieces (L+R) ↔ top-down (MU503) |
| **Illumination** | Gooseneck (epi/reflected) + substage (transmitted) |
| **Host** | Windows PC with multiple USB hubs, screen for cyan/magenta glasses |

---

## 3. Revised Phase Plan

### Phase 3 — Stereo Calibration & Still Capture _(next)_

**Goal**: Proper stereo calibration using checkerboard targets, full-resolution still capture with metadata, and a guided calibration wizard in the GUI.

| Task | Description | Effort |
|---|---|---|
| 3.1 Implement `stereo_calibrate()` | Complete the stub: collect checkerboard detections from both cameras, run `cv2.stereoCalibrate`, compute `initUndistortRectifyMap` | Medium |
| 3.2 `save_calibration()` / `load_calibration()` | Serialize calibration data (camera matrices, distortion, R, T, rectify maps) to `.npz` files | Small |
| 3.3 Calibration wizard panel | New QWidget panel in GUI: "Place checkerboard → capture N poses → calibrate → show reprojection error → save" | Large |
| 3.4 Apply rectification maps | When calibration is loaded, warp frames through undistort+rectify before compositing | Medium |
| 3.5 Enhanced still capture | Full-resolution capture (not just preview size), TIFF+JPEG output, embedded metadata (timestamp, cameras, magnification, alignment params, anaglyph method) | Medium |
| 3.6 Auto-naming | `YYYY-MM-DD_HHMMSS_{method}.tiff` pattern with configurable output directory | Small |
| 3.7 Unit tests | Tests for stereo calibrate, save/load round-trip, metadata embedding | Medium |
| 3.8 `docs/calibration_guide.md` | Step-by-step with expected screenshots for each wizard step | Small |

**Deliverables**: Working calibration wizard, rectified compositing, full-res still capture with metadata.

---

### Phase 4 — Color Balance & Exposure Matching

**Goal**: Automatic exposure and white balance matching between cameras for cleaner anaglyphs.

| Task | Description | Effort |
|---|---|---|
| 4.1 Histogram analysis | Compute per-channel histograms for L/R; quantify brightness/color difference | Small |
| 4.2 Auto-exposure matching | Iteratively adjust `CAP_PROP_EXPOSURE` on both cameras until mean intensity converges (ΔMean < 5%) | Medium |
| 4.3 White balance normalization | Gray-world assumption: per-channel gain adjustment so R/G/B ratios match between cameras | Medium |
| 4.4 GUI exposure/WB controls | Sliders or auto-button in sidebar; real-time histogram preview | Medium |
| 4.5 Integration with calibration wizard | Add exposure match + WB as calibration wizard steps | Small |
| 4.6 Unit tests | Tests for histogram analysis, gray-world normalization | Small |

**Deliverables**: Matched exposure and color between cameras; reduced ghosting and color fringing in anaglyphs.

---

### Phase 5 — Z-Stack Acquisition & Focus Stacking

**Goal**: Guided focal sweep acquisition with Laplacian-pyramid focus stacking for all-in-focus composites.

| Task | Description | Effort |
|---|---|---|
| 5.1 Focal sweep UI | New panel: "Slowly turn fine-focus knob clockwise" with real-time sharpness graph | Large |
| 5.2 Frame selection | Continuously capture frames during sweep; compute Laplacian variance per frame; select best N spanning focal range | Medium |
| 5.3 Stack alignment | Rigid registration of stack frames (no parallax shift, only Z-translation) | Medium |
| 5.4 Laplacian pyramid stacking | Per-level coefficient selection (max absolute value); reconstruct all-in-focus composite | Large |
| 5.5 Depth map (optional) | Generate depth map from focus-distance assignment | Medium |
| 5.6 Save stacked output | TIFF with metadata indicating source frames and focal range | Small |
| 5.7 Unit tests | Tests for frame selection, Laplacian pyramid merge, focus metric monotonicity | Medium |

**Deliverables**: All-in-focus composite from focal sweep; optional depth map.

---

### Phase 6 — Gallery & Catalog

**Goal**: Local capture catalog with tagging and static web gallery export.

| Task | Description | Effort |
|---|---|---|
| 6.1 SQLite catalog | `~/.anaglyph/catalog.db` with table: `captures(id, timestamp, subject, method, paths, metadata_json, tags)` | Medium |
| 6.2 GUI capture browser | New panel listing past captures with thumbnails, tags, metadata | Large |
| 6.3 Tagging system | User-assigned tags for art/science categorization | Small |
| 6.4 Static-site generator | Jinja2 templates → HTML + CSS + JS responsive gallery; click for full-res | Large |
| 6.5 Export CLI | `python main.py --export-gallery ./public` | Small |
| 6.6 Output modes | Anaglyph, side-by-side, wiggle-GIF export options | Medium |
| 6.7 Unit tests | CRUD operations on SQLite catalog, HTML gallery rendering | Medium |
| 6.8 `docs/gallery_workflow.md` | How to capture, curate, tag, and export | Small |

**Deliverables**: Searchable capture catalog, portable HTML gallery.

---

### Phase 7 — Cross-Platform & Polish

**Goal**: Reliable operation on Linux and macOS; packaging and release workflow.

| Task | Description | Effort |
|---|---|---|
| 7.1 Linux camera enumerator | Parse `/sys/class/video4linux/` for USB path topology → deterministic L/R mapping | Medium |
| 7.2 macOS camera enumerator | `system_profiler SPUSBDataType` → serial number mapping | Medium |
| 7.3 Platform-specific ENOSPC handling | Detect USB bandwidth errors, auto-downgrade resolution with warning | Small |
| 7.4 Manual parallax fine-tuning | Arrow-key adjustment of horizontal/vertical offset; stored in config | Medium |
| 7.5 CLI: headless capture | `python main.py --capture-still`, `--capture-video --duration 10` | Medium |
| 7.6 ToupTek SDK backend (optional) | Higher bit-depth, hardware triggers (if SDK obtained) | Large |
| 7.7 Pydantic config | Typed YAML/JSON config with validation, replacing raw `camera_config.json` | Medium |
| 7.8 PyInstaller packaging | One-click Windows installer; `.app` bundle for macOS | Medium |
| 7.9 Release workflow | `v*` tag → GitHub Actions → build wheel → GitHub Release | Small |
| 7.10 `CONTRIBUTING.md` | Dev setup, branching model, how to run HITL tests | Small |

**Deliverables**: Cross-platform support, packaged releases, complete documentation.

---

## 4. Revised Test Plan

### Unit Tests (automated, no hardware)

| Module | Tests | Status |
|---|---|---|
| `compositor` — Wimmer/Dubois/half-color/gray channel layout | 11 tests | ✅ Done |
| `compositor` — alignment, affine math, sharpness | 20 tests | ✅ Done |
| `calibration` — checkerboard detection | 3 tests | ✅ Done |
| `camera_manager` — data structures, key generation | 8 tests | ✅ Done |
| `video_recorder` — lifecycle, stats, transforms | 9 tests | ✅ Done |
| `calibration` — stereo calibrate, save/load round-trip | Planned (Phase 3) |
| `color_balance` — histogram analysis, gray-world | Planned (Phase 4) |
| `zstack` — focus metric monotonicity, Laplacian merge | Planned (Phase 5) |
| `gallery` — SQLite CRUD, HTML rendering | Planned (Phase 6) |
| `config` — Pydantic validation | Planned (Phase 7) |

### Hardware Tests (prompted HITL, NOT in CI)

| Test ID | Description | Status |
|---|---|---|
| HW-CAM-01 | Connect camera, verify opens | ✅ Done |
| HW-CAM-02 | Connect second camera, verify ≥2 detected | ✅ Done |
| HW-FOCUS-01 | Focus on target, verify Laplacian variance | ✅ Done |
| HW-ALIGN-01 | Alignment on centered target, verify reprojection | ✅ Done |
| HW-ANA-01 | Anaglyph 3D perception with glasses | ✅ Done |
| HW-ANA-02 | Parallax adjustment confirmation | ✅ Done |
| HW-EXPO-01 | Auto-exposure matching, histogram convergence | Planned (Phase 4) |
| HW-VID-01 | 5-second video recording, frame count | Planned (Phase 3) |
| HW-ZSTACK-01 | Focal sweep + stacking, sharpness gain | Planned (Phase 5) |
| HW-GALLERY-01 | Gallery page review in browser | Planned (Phase 6) |

### Calibration Evaluation Metrics

| Metric | Target | Measurement |
|---|---|---|
| Alignment reprojection error | < 2 px | ORB + affine RMSE |
| Exposure match | ΔMean < 5% | `abs(mean_L - mean_R) / mean_L` |
| Color balance | ΔChannel < 3% | Per-channel mean ratio |
| Frame sync drift | < 1 frame period | Timestamp difference between L/R captures |
| Preview FPS | ≥ 15 fps | Measured over 100-frame window |
| Z-stack sharpness gain | > 1.5× | Laplacian variance of stacked vs. best single frame |

---

## 5. Priority Recommendation

**Phase 3 (Stereo Calibration + Still Capture)** is the highest-impact next step:
- Calibration is the foundation for everything downstream (rectified compositing, color matching, z-stacking)
- Full-res still capture with metadata is the most immediately useful capture mode
- The calibration wizard turns the current "expert tool" into a guided workflow

Phases 4–7 can be reordered based on user priorities. Z-stack (Phase 5) and gallery (Phase 6) are independent and could be parallelized.

---

## 6. Effort Estimates

| Phase | Estimated Scope | Key Risk |
|---|---|---|
| Phase 3 | 6–8 tasks, ~800 lines new code | Calibration wizard UX complexity |
| Phase 4 | 5–6 tasks, ~400 lines new code | Camera exposure API varies by model |
| Phase 5 | 6–7 tasks, ~600 lines new code | Stack alignment quality with manual focus |
| Phase 6 | 7–8 tasks, ~700 lines new code | Gallery templating polish |
| Phase 7 | 8–10 tasks, ~500 lines new code | Cross-platform testing without hardware |
