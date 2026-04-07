"""Tk window title-bar icon (PNG); Dock icon comes from py2app .icns."""

import os
import sys

from PIL import Image, ImageTk


def _resolve_window_icon_png_path():
    """Path to app_window_icon.png in dev tree or inside a py2app bundle Resources folder."""
    if getattr(sys, "frozen", False):
        contents_dir = os.path.dirname(os.path.dirname(sys.executable))
        bundled = os.path.join(contents_dir, "Resources", "app_window_icon.png")
        if os.path.isfile(bundled):
            return bundled
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dev_path = os.path.join(repo_root, "app_window_icon.png")
    return dev_path if os.path.isfile(dev_path) else None


def apply_window_icon(root):
    """
    Set the Tk window icon where supported. The Dock and Finder icon for the distributed
    .app comes from BackgroundRemover.icns via py2app; this PNG keeps the title bar consistent.
    """
    path = _resolve_window_icon_png_path()
    if not path:
        return
    try:
        icon_image = Image.open(path)
        photo = ImageTk.PhotoImage(icon_image)
        root.iconphoto(True, photo)
        root._wm_icon_photo_ref = photo
    except Exception:
        pass
