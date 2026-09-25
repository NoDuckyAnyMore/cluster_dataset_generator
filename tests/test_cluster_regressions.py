from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


GENERATOR_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GENERATOR_DIR))

from campus_sionna_dataset import (  # noqa: E402
    GroundRxPoint,
    read_rx_positions_csv,
    rx_prefix_point_matches,
    write_rx_positions_csv,
)
import run_cluster_5090 as cluster_launcher  # noqa: E402
from rt_solver import should_persist_batch, summarize_drjit_kernel_history  # noqa: E402

estimate_osm_mesh_complexity = cluster_launcher.estimate_osm_mesh_complexity


class ClusterRegressionTest(unittest.TestCase):
    def test_mesh_estimate_ignores_large_nonbuilding_relation(self) -> None:
        xml = """<osm>
          <node id="1" lat="0" lon="0"/>
          <node id="2" lat="0" lon="1"/>
          <node id="3" lat="1" lon="1"/>
          <node id="4" lat="1" lon="0"/>
          <way id="10">
            <nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="1"/>
            <tag k="building" v="yes"/>
          </way>
          <way id="11"><nd ref="1"/><nd ref="2"/><tag k="highway" v="primary"/></way>
          <relation id="20">
            <member type="way" ref="11" role=""/>
            <member type="way" ref="11" role=""/>
            <tag k="type" v="route"/>
          </relation>
        </osm>"""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "osm_map.osm"
            path.write_text(xml, encoding="utf-8")
            self.assertEqual(estimate_osm_mesh_complexity(path), (1, 10))

    def test_rx_csv_is_six_decimal_and_old_three_decimal_is_compatible(self) -> None:
        desired = GroundRxPoint(0, 12.34549, -9.87649, 0.0, 0.0, 0.0)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rx_positions.csv"
            write_rx_positions_csv([desired], [5.0], path)
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["x_m"], "12.345490")
            self.assertEqual(row["y_m"], "-9.876490")

            # Simulate an existing snapshot written by the former .3f format.
            row["x_m"], row["y_m"] = "12.345", "-9.876"
            with path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=row.keys())
                writer.writeheader()
                writer.writerow(row)
            previous = read_rx_positions_csv(path)[0]
            self.assertTrue(rx_prefix_point_matches(previous, desired))

            changed = GroundRxPoint(0, 12.3461, -9.87649, 0.0, 0.0, 0.0)
            self.assertFalse(rx_prefix_point_matches(previous, changed))

    def test_drjit_kernel_history_summary_separates_cache_sources(self) -> None:
        records = [
            {"type": "jit", "hash": "a", "cache_hit": True, "execution_time": 1.5},
            {"type": "jit", "hash": "a", "cache_hit": False, "cache_disk": True,
             "execution_time": 2.0, "codegen_time": 0.25},
            {"type": "jit", "hash": "b", "cache_hit": False, "cache_disk": False,
             "uses_optix": True, "execution_time": 3.0, "backend_time": 0.5},
            {"type": "copy", "execution_time": 0.5},
        ]
        summary = summarize_drjit_kernel_history(records)
        self.assertEqual(summary["jit_kernels"], 3)
        self.assertEqual(summary["memory_hits"], 1)
        self.assertEqual(summary["disk_hits"], 1)
        self.assertEqual(summary["hard_misses"], 1)
        self.assertEqual(summary["unique_hashes"], 2)
        self.assertAlmostEqual(summary["execution_time_ms"], 7.0)
        self.assertAlmostEqual(summary["miss_backend_time_ms"], 0.5)
        self.assertAlmostEqual(summary["miss_compile_time_ms"], 0.5)
        self.assertAlmostEqual(summary["miss_execution_time_ms"], 3.0)
        self.assertAlmostEqual(summary["hit_execution_time_ms"], 3.5)

    def test_checkpoint_policy_commits_every_25_batches_and_height_end(self) -> None:
        committed = [
            batch for batch in range(1, 128)
            if should_persist_batch(
                batch_number=batch,
                batch_end=min(batch * 130, 16384),
                total_points=16384,
                interval_batches=25,
            )
        ]
        self.assertEqual(committed, [25, 50, 75, 100, 125, 127])
        self.assertTrue(should_persist_batch(1, 10, 10, 25))

    def test_dynamic_worker_calls_one_imported_generator_for_all_blocks(self) -> None:
        tasks = [
            cluster_launcher.BlockTask("a", "one", 0, 0, 0),
            cluster_launcher.BlockTask("b", "two", 0, 0, 0),
        ]
        completed: set[tuple[str, str]] = set()
        calls: list[object] = []

        class FakeGenerator:
            @staticmethod
            def process_aerial_tx_voxel_block(block, _root, _args) -> None:
                calls.append(block)
                completed.add(block)

        class FakeLock:
            def close(self) -> None:
                pass

        specs = {("a", "one"): ("a", "one"), ("b", "two"): ("b", "two")}
        with (
            patch.object(
                cluster_launcher,
                "block_is_complete",
                side_effect=lambda task: (task.region_slug, task.block_id) in completed,
            ),
            patch.object(cluster_launcher, "install_block_osm"),
            patch.object(cluster_launcher, "acquire_file_lock", return_value=FakeLock()),
            patch.object(cluster_launcher, "reclaim_drjit_block_memory") as reclaim,
        ):
            success = cluster_launcher.run_worker_pool_member(
                tasks, "test_0", FakeGenerator, object(), specs
            )
        self.assertTrue(success)
        self.assertEqual(calls, [("a", "one"), ("b", "two")])
        self.assertEqual(reclaim.call_count, 2)


if __name__ == "__main__":
    unittest.main()
