"""
Anaglyph Stereoscope - Main entry point.

Modular app for aligning two AmScope MD500L cameras into a live 3D anaglyph stream.
"""

from __future__ import annotations

import argparse
import logging
import sys

from camera_manager import verify_hardware_access

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Anaglyph Stereoscope - Live 3D from dual AmScope MD500L cameras"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify camera hardware access and exit",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch GUI (Calibration / Live 3D modes)",
    )
    args = parser.parse_args()

    if args.verify:
        return 0 if verify_hardware_access() else 1

    if args.gui:
        from gui import run_gui
        return run_gui()

    # Default: verify hardware
    print("Run with --verify to test cameras, or --gui to launch the application.")
    return 0 if verify_hardware_access() else 1


if __name__ == "__main__":
    sys.exit(main())
