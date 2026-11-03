"""Routine checks of published data; no new search or sensor model is assumed."""

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
from PIL import Image


DATA = Path(__file__).resolve().parents[1] / "data" / "radiomapseer-subset"


@unittest.skipUnless((DATA / "manifest.json").is_file(), "Optional RadioMapSeer subset not downloaded")
class RadioMapSeerDataTests(unittest.TestCase):
    def test_downloaded_files_match_manifest(self):
        manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["files"]), 727)
        for entry in manifest["files"]:
            with self.subTest(path=entry["path"]):
                content = (DATA / entry["path"]).read_bytes()
                self.assertEqual(len(content), entry["bytes"])
                self.assertEqual(hashlib.sha256(content).hexdigest(), entry["sha256"])

    def test_building_masks_have_published_shape_and_binary_encoding(self):
        for map_id in (1, 2, 3):
            with self.subTest(map_id=map_id):
                mask = np.asarray(Image.open(DATA / f"png/buildings_complete/{map_id}.png"))
                self.assertEqual(mask.shape, (256, 256))
                self.assertEqual(mask.dtype, np.uint8)
                self.assertEqual(set(np.unique(mask)), {0, 255})

    def test_all_candidate_gain_maps_are_present_and_well_formed(self):
        for map_id in (1, 2, 3):
            for variant in ("DPM", "IRT2"):
                for tx_id in range(80):
                    with self.subTest(map_id=map_id, variant=variant, tx_id=tx_id):
                        image = np.asarray(Image.open(DATA / f"gain/{variant}/{map_id}_{tx_id}.png"))
                        self.assertEqual(image.shape, (256, 256))
                        self.assertEqual(image.dtype, np.uint8)

    def test_json_positions_match_transmitter_markers_with_y_flip(self):
        for map_id in (1, 2, 3):
            positions = np.asarray(json.loads((DATA / f"antenna/{map_id}.json").read_text()))
            self.assertEqual(positions.shape, (80, 2))
            self.assertEqual(len(set(map(tuple, positions.tolist()))), 80)
            for tx_id, (x, y) in enumerate(positions):
                with self.subTest(map_id=map_id, tx_id=tx_id):
                    marker = np.asarray(Image.open(DATA / f"png/antennas/{map_id}_{tx_id}.png"))
                    self.assertEqual(marker.shape, (256, 256))
                    marked_cells = np.argwhere(marker > 0)
                    np.testing.assert_array_equal(marked_cells, [[255 - y, x]])


if __name__ == "__main__":
    unittest.main()
