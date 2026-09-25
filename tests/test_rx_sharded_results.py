from __future__ import annotations

import sys
import tempfile
import unittest
import json
from pathlib import Path

import numpy as np


GENERATOR_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GENERATOR_DIR))

from rx_sharded_results import (  # noqa: E402
    commit_rx_group,
    completed_rx_indices,
    desired_rx_signature,
    load_valid_rx_entries,
    rx_npy_path,
)
from render_rx_sharded_channel_maps import load_result_cube  # noqa: E402


class PerRxTensorTest(unittest.TestCase):
    def test_completed_prefix_survives_rx_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            block_dir = Path(temporary)
            shape = (2, 1, 2)
            for index in (0, 1):
                path = rx_npy_path(block_dir, index)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, np.arange(4, dtype=np.float32).reshape(shape) - index)

            physical = {"voxel": [2, 1, 2], "seed": 123}
            signatures = {
                index: desired_rx_signature(
                    rx_index=index, x_m=float(index), y_m=float(-index),
                    physical_signature=physical,
                )
                for index in (0, 1)
            }
            summaries = {
                str(index): [
                    {"tx_voxel_iz": iz, "height_m": 2.0 + iz * 2.0,
                     "value_count": 2, "rss_mean_dbm": -70.0,
                     "rss_min_dbm": -71.0, "rss_max_dbm": -69.0}
                    for iz in range(2)
                ]
                for index in (0, 1)
            }
            commit_rx_group(
                block_dir=block_dir, rx_signatures=signatures,
                expected_shape_zyx=shape, per_rx_height_summaries=summaries,
                target_rx_count=2,
            )
            expanded = dict(signatures)
            expanded[2] = desired_rx_signature(
                rx_index=2, x_m=2.0, y_m=-2.0, physical_signature=physical,
            )
            valid = load_valid_rx_entries(block_dir, expanded, shape)
            self.assertEqual(completed_rx_indices(valid), {0, 1})

    def test_preview_loader_reads_dense_npy_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            block_dir = Path(temporary)
            (block_dir / "metadata.json").write_text(json.dumps({
                "tx_voxel_shape_xyz": [128, 128, 2],
                "ground_rx_count": 2,
                "voxel_size_m": 2.0,
                "voxel_z_start_m": 2.0,
            }), encoding="utf-8")
            expected = np.arange(2 * 2 * 128 * 128, dtype=np.float32).reshape(2, 2, 128, 128)
            for index in range(2):
                path = rx_npy_path(block_dir, index)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, expected[index])
            actual = load_result_cube(block_dir, show_progress=False)
            np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
