import types
import unittest
from unittest import mock
import os
import tempfile

import tkinter as tk
import numpy as np
from PIL import Image

from app_gui import (
    apply_color_cleanup,
    BackgroundRemoverApp,
    build_export_filename,
    crop_to_visible_bounds,
    dedupe_frame_items,
    resolve_output_prefix,
)


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
            self.app.all_extracted_frame_items = list(frame_items)
            self.app._rebuild_frame_list(frame_items)

        self.assertFalse(self.app.remove_duplicates_btn._disabled)

    def test_remove_duplicates_button_stays_enabled_when_original_extracted_set_is_larger(self):
        original_items = [
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
        visible_items = [original_items[0]]

        with mock.patch.object(self.app, "_add_frame_thumbnail"), mock.patch.object(self.app, "_show_frame_preview"):
            self.app.all_extracted_frame_items = list(original_items)
            self.app._rebuild_frame_list(visible_items)

        self.assertFalse(self.app.remove_duplicates_btn._disabled)

    def test_browse_video_populates_output_prefix_from_filename(self):
        with mock.patch("app_gui.filedialog.askopenfilename", return_value="/tmp/dog-run.mp4"):
            with mock.patch.object(self.app, "_reset_extracted_frames"):
                self.app.browse_video()

        self.assertEqual(self.app.video_output_prefix.get(), "dog-run")

    def test_image_auto_crop_toggle_defaults_on(self):
        self.assertTrue(self.app.auto_crop_output.get())

    def test_image_cleanup_controls_start_disabled_without_input(self):
        self.assertTrue(self.app.pick_image_cleanup_color_btn._disabled)
        self.assertFalse(self.app.add_image_cleanup_color_btn._disabled)

    def test_frame_results_area_uses_resizable_paned_layout(self):
        self.assertTrue(hasattr(self.app, "frame_results_paned"))
        self.assertEqual(str(self.app.frame_results_paned.cget("orient")), "horizontal")

    def test_frame_navigation_moves_selection_with_arrow_keys(self):
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

        def fake_show_preview(index):
            self.app.current_frame_index = index

        with mock.patch.object(self.app, "_add_frame_thumbnail"), mock.patch.object(
            self.app,
            "_show_frame_preview",
            side_effect=fake_show_preview,
        ) as show_preview:
            self.app.all_extracted_frame_items = list(frame_items)
            self.app._rebuild_frame_list(frame_items)

            self.assertEqual(self.app.current_frame_index, 0)
            show_preview.reset_mock()

            self.app.select_next_frame()
            self.assertEqual(self.app.current_frame_index, 1)
            show_preview.assert_called_once_with(1)
            show_preview.reset_mock()

            self.app.select_next_frame()
            self.assertEqual(self.app.current_frame_index, 1)
            show_preview.assert_not_called()

            self.app.select_previous_frame()
            self.assertEqual(self.app.current_frame_index, 0)
            show_preview.assert_called_once_with(0)


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

    def test_lower_threshold_keeps_more_frames_than_higher_threshold(self):
        items = [
            self._item(0, np.zeros((4, 4))),
            self._item(1, np.full((4, 4), 0.02)),
            self._item(2, np.full((4, 4), 0.04)),
        ]

        lower_threshold_unique = dedupe_frame_items(items, threshold=0.01)
        higher_threshold_unique = dedupe_frame_items(items, threshold=0.03)

        self.assertEqual([item["index"] for item in lower_threshold_unique], [0, 1, 2])
        self.assertEqual([item["index"] for item in higher_threshold_unique], [0, 2])


class VideoExportHelperTests(unittest.TestCase):
    def test_resolve_output_prefix_uses_custom_prefix(self):
        self.assertEqual(resolve_output_prefix(" dog ", "video"), "dog")

    def test_resolve_output_prefix_falls_back_to_video_name(self):
        self.assertEqual(resolve_output_prefix("", "dog-video"), "dog-video")

    def test_build_export_filename_uses_sequential_numbering(self):
        self.assertEqual(build_export_filename("dog", 1), "dog_1.png")
        self.assertEqual(build_export_filename("dog", 2, suffix="_no_bg"), "dog_2_no_bg.png")

    def test_crop_to_visible_bounds_removes_transparent_border(self):
        image = Image.new("RGBA", (10, 8), (0, 0, 0, 0))
        for x in range(3, 7):
            for y in range(2, 6):
                image.putpixel((x, y), (255, 0, 0, 255))

        cropped = crop_to_visible_bounds(image)

        self.assertEqual(cropped.size, (4, 4))

    def test_crop_to_visible_bounds_leaves_tight_image_unchanged(self):
        image = Image.new("RGBA", (4, 4), (255, 0, 0, 255))

        cropped = crop_to_visible_bounds(image)

        self.assertEqual(cropped.size, (4, 4))

    def test_crop_to_visible_bounds_keeps_fully_transparent_image(self):
        image = Image.new("RGBA", (5, 7), (0, 0, 0, 0))

        cropped = crop_to_visible_bounds(image)

        self.assertEqual(cropped.size, (5, 7))

    def test_apply_color_cleanup_removes_matching_pixels_with_threshold(self):
        image = Image.new("RGBA", (3, 1), (0, 0, 0, 0))
        image.putpixel((0, 0), (255, 0, 0, 255))
        image.putpixel((1, 0), (0, 255, 0, 255))
        image.putpixel((2, 0), (5, 245, 10, 255))

        cleaned = apply_color_cleanup(image, [(0, 255, 0)], threshold=15)

        self.assertEqual(cleaned.getpixel((0, 0)), (255, 0, 0, 255))
        self.assertEqual(cleaned.getpixel((1, 0))[3], 0)
        self.assertEqual(cleaned.getpixel((2, 0))[3], 0)


class VideoBackgroundCleanupPipelineTests(unittest.TestCase):
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

    def test_background_export_applies_selected_color_cleanup_before_save(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = os.path.join(temp_dir, "frame.png")
            Image.new("RGB", (4, 4), (255, 255, 255)).save(source_path, "PNG")

            selected_items = [{"path": source_path}]
            cutout = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
            cutout.putpixel((0, 0), (255, 0, 0, 255))
            cutout.putpixel((1, 0), (0, 255, 0, 255))
            cutout.putpixel((2, 0), (4, 248, 8, 255))
            cutout.putpixel((0, 1), (255, 0, 0, 255))
            cutout.putpixel((1, 1), (255, 0, 0, 255))
            cutout.putpixel((2, 1), (255, 0, 0, 255))

            saved_payload = {}

            with mock.patch.object(self.app, "_load_model", return_value=mock.sentinel.net), \
                 mock.patch.object(self.app, "_create_cutout_for_image", return_value=cutout.copy()), \
                 mock.patch.object(self.app.root, "after", side_effect=lambda _delay, callback: callback()), \
                 mock.patch.object(self.app, "_on_background_frames_saved", side_effect=lambda paths, target_dir: saved_payload.update({"paths": paths, "target_dir": target_dir})):
                self.app._remove_background_and_save_selected_frames_thread(
                    selected_items=selected_items,
                    target_dir=temp_dir,
                    output_prefix="avatar",
                    model_name="u2net",
                    alpha_matting=False,
                    cleanup_colors=[(0, 255, 0)],
                    cleanup_threshold=15,
                )

            with Image.open(saved_payload["paths"][0]) as saved_image_file:
                saved_image = saved_image_file.convert("RGBA")

            self.assertEqual(saved_image.getpixel((0, 0)), (255, 0, 0, 255))
            self.assertEqual(saved_image.getpixel((1, 0))[3], 0)
            self.assertEqual(saved_image.getpixel((2, 0))[3], 0)


class ImageBackgroundCleanupPipelineTests(unittest.TestCase):
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

    def test_image_export_applies_selected_color_cleanup_before_save(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = os.path.join(temp_dir, "input.png")
            output_path = os.path.join(temp_dir, "output.png")
            Image.new("RGB", (4, 4), (255, 255, 255)).save(source_path, "PNG")

            self.app.input_file.set(source_path)
            self.app.output_file.set(output_path)
            self.app.auto_crop_output.set(False)
            self.app.image_cleanup_colors = [(0, 255, 0)]
            self.app.image_cleanup_threshold.set(15)

            cutout = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
            cutout.putpixel((0, 0), (255, 0, 0, 255))
            cutout.putpixel((1, 0), (0, 255, 0, 255))
            cutout.putpixel((2, 0), (4, 248, 8, 255))

            with mock.patch.object(self.app, "_load_model", return_value=mock.sentinel.net), \
                 mock.patch.object(self.app, "_create_cutout_for_image", return_value=cutout.copy()), \
                 mock.patch.object(self.app.root, "after", side_effect=lambda _delay, callback: callback()), \
                 mock.patch.object(self.app, "_on_success"):
                self.app._process_thread()

            with Image.open(output_path) as saved_image_file:
                saved_image = saved_image_file.convert("RGBA")

            self.assertEqual(saved_image.getpixel((0, 0)), (255, 0, 0, 255))
            self.assertEqual(saved_image.getpixel((1, 0))[3], 0)
            self.assertEqual(saved_image.getpixel((2, 0))[3], 0)


class ImageSpriteKitTests(unittest.TestCase):
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

    def test_sprite_output_dir_defaults_to_downloads(self):
        self.assertTrue(self.app.sprite_output_dir.get().endswith("Downloads"))

    def test_sprite_kit_thread_uses_shared_export_pipeline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = os.path.join(temp_dir, "Items.png")
            Image.new("RGB", (6, 6), (255, 255, 255)).save(source_path, "PNG")

            self.app.input_file.set(source_path)

            saved_payload = {}
            export_result = {
                "sprite_count": 2,
                "manifest_path": os.path.join(temp_dir, "Items_sprite_manifest.json"),
                "sprites": [
                    {"filename": "Items_sprite_001.png"},
                    {"filename": "Items_sprite_002.png"},
                ],
            }

            with mock.patch("app_gui.bg.create_sprite_kit", return_value=export_result) as create_sprite_kit, \
                 mock.patch.object(self.app.root, "after", side_effect=lambda _delay, callback: callback()), \
                 mock.patch.object(self.app, "_on_sprite_kit_success", side_effect=lambda result, target_dir: saved_payload.update({"result": result, "target_dir": target_dir})):
                self.app._process_sprite_kit_thread(temp_dir)

            create_sprite_kit.assert_called_once()
            self.assertEqual(create_sprite_kit.call_args.kwargs["destination_dir"], temp_dir)
            self.assertEqual(create_sprite_kit.call_args.kwargs["prefix"], "Items")
            self.assertEqual(create_sprite_kit.call_args.kwargs["model_name"], self.app.model_choice.get())
            self.assertEqual(saved_payload["result"]["sprite_count"], 2)
            self.assertEqual(saved_payload["target_dir"], temp_dir)


if __name__ == "__main__":
    unittest.main()
