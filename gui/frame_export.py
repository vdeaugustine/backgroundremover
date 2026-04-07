"""Frame deduplication, export naming, and alpha bounds cropping."""

import os

import numpy as np
from PIL import Image

from gui.constants import DUPLICATE_FRAME_MAX_MEAN_DIFFERENCE, DUPLICATE_FRAME_SIGNATURE_SIZE


def build_frame_similarity_signature(image, size=DUPLICATE_FRAME_SIGNATURE_SIZE):
    """Create a small normalized grayscale representation for frame comparison."""
    grayscale = image.convert("L").resize(size, Image.Resampling.BILINEAR)
    return np.asarray(grayscale, dtype=np.float32) / 255.0


def dedupe_frame_items(frame_items, threshold=DUPLICATE_FRAME_MAX_MEAN_DIFFERENCE):
    """Keep only frames that differ meaningfully from the most recent unique frame."""
    if not frame_items:
        return []

    unique_items = [frame_items[0]]
    previous_signature = frame_items[0]["compare_array"]

    for item in frame_items[1:]:
        current_signature = item["compare_array"]
        mean_difference = float(np.mean(np.abs(current_signature - previous_signature)))
        if mean_difference > threshold:
            unique_items.append(item)
            previous_signature = current_signature

    return unique_items


def resolve_output_prefix(prefix, fallback_name):
    """Resolve a user-provided output prefix with a simple safe fallback."""
    cleaned = (prefix or "").strip()
    if not cleaned:
        cleaned = fallback_name or "video"
    return cleaned.replace(os.sep, "_")


def build_export_filename(prefix, sequence_number, suffix=""):
    """Build sequential export filenames for frame batch operations."""
    return f"{prefix}_{sequence_number}{suffix}.png"


def crop_to_visible_bounds(image, padding=0):
    """Crop transparent outer bounds from an RGBA image, if any visible pixels exist."""
    bbox = image.getbbox()
    if bbox is None:
        return image

    if padding > 0:
        left, top, right, bottom = bbox
        left = max(0, left - padding)
        top = max(0, top - padding)
        right = min(image.width, right + padding)
        bottom = min(image.height, bottom + padding)
        bbox = (left, top, right, bottom)

    return image.crop(bbox)
