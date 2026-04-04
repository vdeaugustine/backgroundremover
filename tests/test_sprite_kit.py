import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from backgroundremover import bg


class SpriteKitHelperTests(unittest.TestCase):
    def _rgba_from_alpha_points(self, size, points):
        image = Image.new("RGBA", size, (0, 0, 0, 0))
        for x, y in points:
            image.putpixel((x, y), (255, 0, 0, 255))
        return image

    def test_harden_mask_thresholds_to_crisp_binary_alpha(self):
        mask = Image.new("L", (3, 1))
        mask.putpixel((0, 0), 40)
        mask.putpixel((1, 0), 129)
        mask.putpixel((2, 0), 255)

        hardened = bg.harden_sprite_mask(mask, threshold=128)

        self.assertEqual(list(hardened.getdata()), [0, 255, 255])

    def test_connected_objects_export_in_stable_top_to_bottom_then_left_to_right_order(self):
        alpha_points = {
            (7, 1), (8, 1), (7, 2), (8, 2),
            (1, 4), (2, 4), (1, 5), (2, 5),
            (6, 4), (7, 4), (6, 5), (7, 5),
        }
        cutout = self._rgba_from_alpha_points((10, 8), alpha_points)

        sprites = bg.split_sprite_cutout(cutout, merge_distance=0, min_sprite_area=1)

        self.assertEqual([sprite["bbox"] for sprite in sprites], [(7, 1, 9, 3), (1, 4, 3, 6), (6, 4, 8, 6)])

    def test_nearby_disconnected_islands_merge_into_one_sprite(self):
        alpha_points = {
            (1, 1), (2, 1), (1, 2), (2, 2),
            (5, 1), (6, 1), (5, 2), (6, 2),
        }
        cutout = self._rgba_from_alpha_points((8, 4), alpha_points)

        sprites = bg.split_sprite_cutout(cutout, merge_distance=3, min_sprite_area=1)

        self.assertEqual(len(sprites), 1)
        self.assertEqual(sprites[0]["bbox"], (1, 1, 7, 3))

    def test_tiny_noise_components_are_filtered_out(self):
        alpha_points = {
            (1, 1), (2, 1), (1, 2), (2, 2),
            (8, 8),
        }
        cutout = self._rgba_from_alpha_points((10, 10), alpha_points)

        sprites = bg.split_sprite_cutout(cutout, merge_distance=0, min_sprite_area=2)

        self.assertEqual(len(sprites), 1)
        self.assertEqual(sprites[0]["bbox"], (1, 1, 3, 3))

    def test_fully_transparent_cutout_returns_no_sprites(self):
        cutout = Image.new("RGBA", (6, 4), (0, 0, 0, 0))

        sprites = bg.split_sprite_cutout(cutout, merge_distance=2, min_sprite_area=1)

        self.assertEqual(sprites, [])

    def test_export_sprite_kit_writes_pngs_and_manifest(self):
        cutout = self._rgba_from_alpha_points(
            (8, 6),
            {(1, 1), (2, 1), (1, 2), (2, 2), (5, 3), (6, 3), (5, 4), (6, 4)},
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            result = bg.export_sprite_kit(
                cutout,
                destination_dir=temp_dir,
                prefix="items",
                model_name="u2net",
                min_sprite_area=1,
                merge_distance=0,
            )

            self.assertEqual(result["sprite_count"], 2)
            self.assertEqual(
                [sprite["filename"] for sprite in result["sprites"]],
                ["items_sprite_001.png", "items_sprite_002.png"],
            )

            manifest_path = os.path.join(temp_dir, "items_sprite_manifest.json")
            self.assertTrue(os.path.exists(manifest_path))

            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)

            self.assertEqual(manifest["source_size"], [8, 6])
            self.assertEqual(manifest["model"], "u2net")
            self.assertEqual(manifest["sprite_count"], 2)

    def test_create_sprite_kit_uses_existing_model_pipeline_and_exports_manifest(self):
        image = Image.new("RGB", (6, 6), (255, 255, 255))
        fake_cutout = self._rgba_from_alpha_points((6, 6), {(1, 1), (2, 1), (1, 2), (2, 2)})

        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(bg, "get_model", return_value=mock.sentinel.model), \
                 mock.patch.object(bg.detect, "predict", return_value=Image.new("L", (6, 6), 255)), \
                 mock.patch.object(bg, "naive_cutout", return_value=fake_cutout):
                result = bg.create_sprite_kit(
                    image,
                    destination_dir=temp_dir,
                    prefix="asset",
                    model_name="u2net",
                    alpha_matting=False,
                )

            self.assertEqual(result["sprite_count"], 1)
            self.assertEqual(result["sprites"][0]["bbox"], [1, 1, 3, 3])
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "asset_sprite_001.png")))


if __name__ == "__main__":
    unittest.main()
