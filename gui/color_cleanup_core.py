"""Color cleanup and post-process save pipeline shared by image and video paths."""

import os

import numpy as np
from PIL import Image

from gui.constants import MAX_COLOR_CLEANUP_WORKERS
from gui.frame_export import crop_to_visible_bounds


def apply_color_cleanup(
    image,
    cleanup_colors,
    threshold=0,
    exact_colors=None,
    protection_mask=None,
    protected_colors=None,
    protected_threshold=0,
):
    """Make pixels transparent where they match a cleanup color within tolerance (or exactly).

    Safety-first design: when a pixel matches both a cleanup color and a
    protected color, the pixel is KEPT (protection wins over cleanup).

    Args:
        image: PIL Image to clean up.
        cleanup_colors: List of RGB tuples to remove.
        threshold: Max per-channel distance for cleanup colour matching.
        exact_colors: List of RGB tuples to remove with exact match only.
        protection_mask: Optional boolean H x W numpy array (brush-based).
            Pixels marked True are never cleared, regardless of any colour
            match.  Pass None to skip.
        protected_colors: List of RGB tuples to protect from removal.
        protected_threshold: Max per-channel distance for protected colour
            matching.
    """
    rgba_image = image.convert("RGBA")
    if not cleanup_colors and not exact_colors:
        return rgba_image

    pixel_array = np.array(rgba_image, dtype=np.uint8)
    if pixel_array.size == 0:
        return rgba_image

    opaque_mask = pixel_array[:, :, 3] > 0
    if not np.any(opaque_mask):
        return rgba_image

    rgb_values = pixel_array[:, :, :3].astype(np.int16)
    matched_mask = np.zeros(opaque_mask.shape, dtype=bool)

    if cleanup_colors:
        threshold = max(0, int(threshold))
        for color in cleanup_colors:
            target = np.array([int(channel) for channel in color[:3]], dtype=np.int16)
            channel_difference = np.max(np.abs(rgb_values - target), axis=2)
            matched_mask |= channel_difference <= threshold

    if exact_colors:
        for color in exact_colors:
            target = np.array([int(channel) for channel in color[:3]], dtype=np.int16)
            channel_difference = np.max(np.abs(rgb_values - target), axis=2)
            matched_mask |= channel_difference == 0

    matched_mask &= opaque_mask

    if protected_colors:
        color_protect_mask = np.zeros(opaque_mask.shape, dtype=bool)
        pt = max(0, int(protected_threshold))
        for color in protected_colors:
            target = np.array([int(ch) for ch in color[:3]], dtype=np.int16)
            channel_diff = np.max(np.abs(rgb_values - target), axis=2)
            color_protect_mask |= channel_diff <= pt
        matched_mask &= ~color_protect_mask

    if protection_mask is not None:
        try:
            pm = np.asarray(protection_mask, dtype=bool)
            if pm.shape == matched_mask.shape:
                matched_mask &= ~pm
        except Exception:
            pass

    if not np.any(matched_mask):
        return rgba_image

    pixel_array[matched_mask, 3] = 0
    return Image.fromarray(pixel_array)


def resolve_color_cleanup_worker_count(frame_count):
    """Choose a bounded worker count for asynchronous color cleanup and saves."""
    available_cpus = os.cpu_count() or 1
    return max(1, min(MAX_COLOR_CLEANUP_WORKERS, available_cpus, int(frame_count or 1)))


def finalize_processed_cutout(
    cutout,
    destination,
    cleanup_colors,
    cleanup_threshold,
    exact_colors=None,
    auto_crop=True,
    target_size=None,
    crop_padding=0,
    protection_mask=None,
    protected_colors=None,
    protected_threshold=0,
):
    """Run the final cleanup pass, crop, resize and save one processed frame.

    Args:
        protection_mask: Optional boolean numpy array (H x W). Pixels that are
            True are never cleared by color cleanup.  Pass None to disable.
        protected_colors: List of RGB tuples to protect from removal.
        protected_threshold: Per-channel tolerance for protected colour matching.
    """
    finalized = apply_color_cleanup(
        cutout,
        cleanup_colors or [],
        cleanup_threshold,
        exact_colors or [],
        protection_mask=protection_mask,
        protected_colors=protected_colors or [],
        protected_threshold=protected_threshold,
    )
    if auto_crop:
        finalized = crop_to_visible_bounds(finalized, padding=crop_padding)
    if target_size and target_size[0] > 0 and target_size[1] > 0:
        target_width, target_height = target_size
        img_width, img_height = finalized.size
        left = (img_width - target_width) // 2
        top = (img_height - target_height) // 2
        right = (img_width + target_width) // 2
        bottom = (img_height + target_height) // 2
        finalized = finalized.crop((left, top, right, bottom))
    finalized.save(destination, "PNG")
    return destination
