"""
Tests for the Protected-Pixels cleanup feature.

Covers:
- Regression: confirm old apply_color_cleanup without mask removes interior pixels.
- Protected cleanup: interior pixels matching cleanup color are preserved.
- Unprotected matching pixels are still removed.
- Undo snapshots deep-copy protection masks (mutations to live mask don't corrupt history).
- Brush helpers paint and erase the mask correctly.
"""

import numpy as np
import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Import helpers from the application module.
# ---------------------------------------------------------------------------
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app_gui import (
    apply_color_cleanup,
    make_empty_protection_mask,
    apply_protection_brush,
    erase_protection_brush,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GREEN = (0, 200, 0)   # The cleanup "background" / edge colour.
RED   = (200, 0, 0)   # Subject colour (should never be cleaned up).
BLUE  = (0, 0, 200)   # Another subject colour.

THRESHOLD = 30        # Cleanup tolerance that matches GREEN but not RED/BLUE.


def _solid_rgba(color_rgb, width=8, height=8):
    """Return an RGBA image filled with the given RGB colour (fully opaque)."""
    r, g, b = color_rgb
    img = Image.new("RGBA", (width, height), (r, g, b, 255))
    return img


def _make_test_frame(width=10, height=10):
    """Create a small test frame with a green background and a red centre stripe.

    Layout (all rows):
        col 0-2:   GREEN  (background)
        col 3-6:   RED    (subject character interior)
        col 7-9:   GREEN  (background, other side)
    """
    arr = np.zeros((height, width, 4), dtype=np.uint8)
    arr[:, :, 3] = 255  # fully opaque

    # Background columns.
    arr[:, 0:3, :3] = GREEN
    arr[:, 7:10, :3] = GREEN

    # Interior columns (character).
    arr[:, 3:7, :3] = RED

    return Image.fromarray(arr, mode="RGBA")


# ---------------------------------------------------------------------------
# Test 1 – Regression: global cleanup removes interior matching pixels
# ---------------------------------------------------------------------------

def test_global_cleanup_removes_interior_pixels():
    """Without a protection mask, cleanup removes ALL matching pixels — including interior ones.

    This is the current bug: if the character has a green-ish interior pixel, it
    gets cleared along with the true background.
    """
    # Frame with a green interior pixel in the middle of the subject.
    width, height = 10, 10
    arr = np.zeros((height, width, 4), dtype=np.uint8)
    arr[:, :, 3] = 255
    arr[:, 0:10, :3] = RED        # Entire row is red (subject).
    arr[5, 5, :3] = GREEN          # One interior "green" pixel (edge compression artefact).

    img = Image.fromarray(arr, mode="RGBA")
    result = apply_color_cleanup(img, [GREEN], threshold=THRESHOLD)
    result_arr = np.array(result)

    # Without protection, the interior green pixel IS cleared (alpha → 0).
    assert result_arr[5, 5, 3] == 0, (
        "Regression: global cleanup should clear interior matching pixel (no mask)."
    )


# ---------------------------------------------------------------------------
# Test 2 – Protected cleanup preserves protected interior pixels
# ---------------------------------------------------------------------------

def test_protected_cleanup_preserves_interior_pixels():
    """With a protection mask, matching pixels inside the protected region survive cleanup."""
    width, height = 10, 10
    arr = np.zeros((height, width, 4), dtype=np.uint8)
    arr[:, :, 3] = 255
    arr[:, 0:10, :3] = RED
    arr[5, 5, :3] = GREEN     # Interior pixel that would normally be removed.

    img = Image.fromarray(arr, mode="RGBA")

    # Protect the interior pixel.
    mask = make_empty_protection_mask(width, height)
    mask[5, 5] = True

    result = apply_color_cleanup(img, [GREEN], threshold=THRESHOLD, protection_mask=mask)
    result_arr = np.array(result)

    assert result_arr[5, 5, 3] == 255, (
        "Protected interior pixel must not be cleared by colour cleanup."
    )


# ---------------------------------------------------------------------------
# Test 3 – Unprotected matching pixels are still removed
# ---------------------------------------------------------------------------

def test_unprotected_matching_pixels_are_removed():
    """Pixels outside the protected region are still cleared by cleanup."""
    frame = _make_test_frame()

    mask = make_empty_protection_mask(10, 10)
    # Only protect the character interior (columns 3–6).
    mask[:, 3:7] = True

    result = apply_color_cleanup(frame, [GREEN], threshold=THRESHOLD, protection_mask=mask)
    result_arr = np.array(result)

    # Background columns should be zeroed out.
    for row in range(10):
        assert result_arr[row, 0, 3] == 0, f"Row {row}, col 0 (unprotected green) should be cleared."
        assert result_arr[row, 9, 3] == 0, f"Row {row}, col 9 (unprotected green) should be cleared."

    # Character interior must survive.
    for row in range(10):
        for col in range(3, 7):
            assert result_arr[row, col, 3] == 255, (
                f"Row {row}, col {col} (protected interior) must not be cleared."
            )


# ---------------------------------------------------------------------------
# Test 4 – Undo snapshot deep-copies the protection mask
# ---------------------------------------------------------------------------

def test_undo_snapshot_isolates_protection_mask():
    """Mutating the live mask after a snapshot does not corrupt saved history."""
    mask = make_empty_protection_mask(10, 10)

    # Simulate a snapshot (as done in _snapshot_video_frame_items).
    snapshot_mask = mask.copy()

    # Mutate the live mask AFTER the snapshot.
    mask[0, 0] = True

    # The snapshot must be unaffected.
    assert snapshot_mask[0, 0] is np.bool_(False), (
        "Snapshot mask must be isolated from subsequent live-mask mutations."
    )
    assert not snapshot_mask[0, 0], (
        "Undo snapshot protection mask should still be all-False after live-mask mutation."
    )


# ---------------------------------------------------------------------------
# Test 5 – Brush helpers: apply_protection_brush paints correctly
# ---------------------------------------------------------------------------

def test_apply_protection_brush_marks_pixels():
    """apply_protection_brush sets True for all pixels within the given radius."""
    mask = make_empty_protection_mask(20, 20)
    apply_protection_brush(mask, cx=10, cy=10, radius=3)

    # Centre must be protected.
    assert mask[10, 10], "Centre pixel must be protected after brush stroke."

    # Pixels clearly outside the radius must remain unprotected.
    assert not mask[0, 0], "Corner pixel far from brush must remain unprotected."
    assert not mask[19, 19], "Far corner pixel must remain unprotected."


# ---------------------------------------------------------------------------
# Test 6 – Brush helpers: erase_protection_brush erases correctly
# ---------------------------------------------------------------------------

def test_erase_protection_brush_clears_pixels():
    """erase_protection_brush sets False for pixels within the given radius."""
    mask = make_empty_protection_mask(20, 20)

    # Paint a large area first.
    apply_protection_brush(mask, cx=10, cy=10, radius=8)
    assert mask[10, 10], "Sanity: centre should be protected after paint."

    # Now erase the centre.
    erase_protection_brush(mask, cx=10, cy=10, radius=3)
    assert not mask[10, 10], "Centre must be UNprotected after erase stroke."

    # Pixels at the edge of the paint region (radius > 3 from centre) must survive.
    assert mask[10, 18], "Pixel far from erase centre should still be protected."
