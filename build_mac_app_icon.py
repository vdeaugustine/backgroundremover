#!/usr/bin/env python3
"""
Builds BackgroundRemover.icns and app_window_icon.png from backgroundRemoverAppIcon.jpeg.

Run from the repo root before packaging: python build_mac_app_icon.py

Requires macOS iconutil (Xcode CLT). The source JPEG uses a green-screen style background;
those pixels are made transparent for a clean Dock icon.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
SOURCE_JPEG = REPO_ROOT / "backgroundRemoverAppIcon.jpeg"
ICONSET_DIR = REPO_ROOT / "BackgroundRemover.iconset"
OUTPUT_ICNS = REPO_ROOT / "BackgroundRemover.icns"
OUTPUT_WINDOW_PNG = REPO_ROOT / "app_window_icon.png"

# macOS iconset filenames and output side length in pixels.
ICONSET_SIZES: list[tuple[str, int]] = [
    ("icon_16x16.png", 16),
    ("icon_16x16@2x.png", 32),
    ("icon_32x32.png", 32),
    ("icon_32x32@2x.png", 64),
    ("icon_128x128.png", 128),
    ("icon_128x128@2x.png", 256),
    ("icon_256x256.png", 256),
    ("icon_256x256@2x.png", 512),
    ("icon_512x512.png", 512),
    ("icon_512x512@2x.png", 1024),
]


def chroma_green_to_transparent(image: Image.Image) -> Image.Image:
    """Make dominant green-screen pixels transparent (RGB, not HSV, for speed)."""
    rgba = image.convert("RGBA")
    pixels = np.array(rgba, dtype=np.uint8)
    red = pixels[:, :, 0].astype(np.int16)
    green = pixels[:, :, 1].astype(np.int16)
    blue = pixels[:, :, 2].astype(np.int16)
    is_green_screen = (green > 160) & (green > red + 35) & (green > blue + 35)
    pixels[is_green_screen, 3] = 0
    return Image.fromarray(pixels)


def main() -> int:
    if sys.platform != "darwin":
        print("iconutil is macOS-only; skipping ICNS build.", file=sys.stderr)
        return 1
    if not SOURCE_JPEG.is_file():
        print(f"Missing source image: {SOURCE_JPEG}", file=sys.stderr)
        return 1

    base = chroma_green_to_transparent(Image.open(SOURCE_JPEG))

    if ICONSET_DIR.exists():
        shutil.rmtree(ICONSET_DIR)
    ICONSET_DIR.mkdir(parents=True)

    for filename, side in ICONSET_SIZES:
        resized = base.resize((side, side), Image.Resampling.LANCZOS)
        resized.save(ICONSET_DIR / filename, format="PNG")

    window_icon = base.resize((128, 128), Image.Resampling.LANCZOS)
    window_icon.save(OUTPUT_WINDOW_PNG, format="PNG")

    subprocess.run(
        ["iconutil", "-c", "icns", str(ICONSET_DIR), "-o", str(OUTPUT_ICNS)],
        check=True,
    )
    shutil.rmtree(ICONSET_DIR)
    print(f"Wrote {OUTPUT_ICNS}")
    print(f"Wrote {OUTPUT_WINDOW_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
