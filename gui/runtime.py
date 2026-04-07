"""Py2app bundle path setup; run before importing torch-dependent modules."""

import os
import sys


def configure_bundle_paths():
    """When frozen, chdir to bundle and extend sys.path (matches legacy app_gui behavior)."""
    if getattr(sys, "frozen", False):
        bundle_dir = sys._MEIPASS if hasattr(sys, "_MEIPASS") else os.path.dirname(sys.executable)
        os.chdir(bundle_dir)
        sys.path.insert(0, bundle_dir)
