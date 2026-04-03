import types
import unittest
from unittest import mock

import tkinter as tk
import numpy as np

from app_gui import BackgroundRemoverApp, dedupe_frame_items


class FrameSidebarScrollTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = BackgroundRemoverApp(self.root)
        self.root.update_idletasks()

    def tearDown(self):
        try:
            self.app.on_close()
        except Exception:
            self.root.destroy()

    def test_mousewheel_scrolls_frame_sidebar_when_hovered(self):
        self.app.frame_list_canvas.configure(height=200, scrollregion=(0, 0, 280, 4000))
        self.root.update_idletasks()

        self.app.frame_list_canvas.yview_moveto(0)
        event = types.SimpleNamespace(delta=-120, x_root=0, y_root=0)

        with mock.patch.object(self.app, "_event_is_over_frame_list", return_value=True):
            with mock.patch.object(self.app.frame_list_canvas, "yview_scroll") as yview_scroll:
                result = self.app._on_frame_list_mousewheel(event)

        self.assertEqual(result, "break")
        yview_scroll.assert_called_once_with(1, "units")

    def test_remove_duplicates_button_is_disabled_before_extraction_and_enabled_with_frames(self):
        self.assertTrue(self.app.remove_duplicates_btn._disabled)

        frame_items = [
            {
                "index": 0,
                "name": "Frame 1",
                "path": __file__,
                "size": (10, 10),
                "thumbnail": mock.Mock(),
                "compare_array": np.zeros((4, 4), dtype=np.float32),
            },
            {
                "index": 1,
                "name": "Frame 2",
                "path": __file__,
                "size": (10, 10),
                "thumbnail": mock.Mock(),
                "compare_array": np.ones((4, 4), dtype=np.float32),
            },
        ]

        with mock.patch.object(self.app, "_add_frame_thumbnail"), mock.patch.object(self.app, "_show_frame_preview"):
            self.app._rebuild_frame_list(frame_items)

        self.assertFalse(self.app.remove_duplicates_btn._disabled)


class FrameDeduplicationTests(unittest.TestCase):
    def _item(self, index, compare_array):
        return {
            "index": index,
            "name": f"Frame {index + 1}",
            "compare_array": np.array(compare_array, dtype=np.float32),
        }

    def test_exact_duplicates_collapse_to_one(self):
        items = [
            self._item(0, np.zeros((4, 4))),
            self._item(1, np.zeros((4, 4))),
            self._item(2, np.zeros((4, 4))),
        ]

        unique_items = dedupe_frame_items(items)

        self.assertEqual([item["index"] for item in unique_items], [0])

    def test_near_identical_frames_collapse(self):
        items = [
            self._item(0, np.zeros((4, 4))),
            self._item(1, np.full((4, 4), 0.01)),
            self._item(2, np.full((4, 4), 0.012)),
        ]

        unique_items = dedupe_frame_items(items)

        self.assertEqual([item["index"] for item in unique_items], [0])

    def test_distinct_frames_are_preserved_in_order(self):
        items = [
            self._item(0, np.zeros((4, 4))),
            self._item(1, np.full((4, 4), 0.05)),
            self._item(2, np.full((4, 4), 0.10)),
        ]

        unique_items = dedupe_frame_items(items)

        self.assertEqual([item["index"] for item in unique_items], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
