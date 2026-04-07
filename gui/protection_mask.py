"""Boolean protection masks for per-pixel color cleanup."""

import numpy as np


def make_empty_protection_mask(width, height):
    """Return a fresh boolean numpy array (H x W) with all pixels unprotected."""
    return np.zeros((height, width), dtype=bool)


def apply_protection_brush(protection_mask, cx, cy, radius):
    """Paint a circular brush stroke onto *protection_mask* in-place.

    Args:
        protection_mask: 2-D boolean numpy array (H x W).
        cx, cy: Centre of the brush stroke in mask coordinates (x=col, y=row).
        radius: Brush radius in pixels.
    """
    h, w = protection_mask.shape
    y_min = max(0, int(cy - radius))
    y_max = min(h - 1, int(cy + radius))
    x_min = max(0, int(cx - radius))
    x_max = min(w - 1, int(cx + radius))
    r2 = radius * radius
    for row in range(y_min, y_max + 1):
        for col in range(x_min, x_max + 1):
            if (row - cy) ** 2 + (col - cx) ** 2 <= r2:
                protection_mask[row, col] = True


def erase_protection_brush(protection_mask, cx, cy, radius):
    """Erase a circular brush stroke from *protection_mask* in-place."""
    h, w = protection_mask.shape
    y_min = max(0, int(cy - radius))
    y_max = min(h - 1, int(cy + radius))
    x_min = max(0, int(cx - radius))
    x_max = min(w - 1, int(cx + radius))
    r2 = radius * radius
    for row in range(y_min, y_max + 1):
        for col in range(x_min, x_max + 1):
            if (row - cy) ** 2 + (col - cx) ** 2 <= r2:
                protection_mask[row, col] = False
