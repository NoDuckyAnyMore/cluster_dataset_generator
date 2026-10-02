"""Protect old datasets and recompute all campuses when RX seeds change."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_cluster_5090 as cluster
from show_dataset_progress import load_skipped_regions


class DatasetBatchTest(unittest.TestCase):
    def test_no_external_skips_for_new_batch(self):
        with patch.object(cluster, "COMPLETED_REGIONS_SKIP_FILE", None):
            self.assertEqual(cluster.load_completed_region_skips({"pku_main"}), set())
        self.assertEqual(load_skipped_regions(None), set())

    def test_dataset_rejects_different_seed_before_preparation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "dataset_metadata.json"
            original = json.dumps({"simulation_defaults": {"random_seed_global": 1}})
            metadata.write_text(original)
            with (patch.object(cluster, "DATASET_ROOT", root),
                  patch.object(cluster, "acquire_file_lock", return_value=None),
                  patch.object(cluster, "run_checked") as run):
                with self.assertRaisesRegex(RuntimeError, "another seed"):
                    cluster.ensure_dataset_ready()
                run.assert_not_called()
            self.assertEqual(metadata.read_text(), original)


if __name__ == "__main__":
    unittest.main()
