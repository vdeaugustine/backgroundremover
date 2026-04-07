#!/usr/bin/env python3
"""
Background Remover - macOS Application
Entry point and compatibility re-exports for tests and py2app.

Tk and the full app load lazily so tests of pure helpers do not require _tkinter.
"""

from backgroundremover import bg

from gui.color_cleanup_core import apply_color_cleanup, finalize_processed_cutout
from gui.frame_export import (
    build_export_filename,
    crop_to_visible_bounds,
    dedupe_frame_items,
    resolve_output_prefix,
)
from gui.protection_mask import (
    apply_protection_brush,
    erase_protection_brush,
    make_empty_protection_mask,
)

import threading

__all__ = [
    "BackgroundRemoverApp",
    "apply_color_cleanup",
    "apply_protection_brush",
    "bg",
    "build_export_filename",
    "crop_to_visible_bounds",
    "dedupe_frame_items",
    "erase_protection_brush",
    "filedialog",
    "finalize_processed_cutout",
    "main",
    "make_empty_protection_mask",
    "resolve_output_prefix",
    "threading",
]


def __getattr__(name):
    if name == "BackgroundRemoverApp":
        from gui.app import BackgroundRemoverApp

        return BackgroundRemoverApp
    if name == "main":
        from gui.app import main

        return main
    if name == "filedialog":
        from tkinter import filedialog

        return filedialog
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    from gui.app import main as _main

    _main()
