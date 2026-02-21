"""
GUI - View and controls for the stereoscope application.

PyQt6-based interface to toggle between 'Calibration' and 'Live 3D' modes.
Still capture (JPG) and video record (MP4) for the anaglyph view.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# PyQt6 imports - will raise if not installed
try:
    from PyQt6.QtWidgets import QApplication, QMainWindow
    from PyQt6.QtCore import Qt
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


def create_main_window() -> Optional["QMainWindow"]:
    """Create and return the main application window."""
    if not HAS_PYQT6:
        logger.error("PyQt6 not installed. Run: pip install PyQt6")
        return None
    # TODO: Implement full GUI with mode toggle, capture, record
    return QMainWindow()


def run_gui() -> int:
    """Launch the GUI application. Returns exit code."""
    if not HAS_PYQT6:
        print("PyQt6 not installed. Run: pip install PyQt6")
        return 1
    app = QApplication([])
    win = create_main_window()
    if win:
        win.show()
        return app.exec()
    return 1
