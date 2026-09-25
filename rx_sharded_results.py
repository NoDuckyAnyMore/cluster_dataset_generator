"""Utilities for extensible per-RX dense RSS tensors."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np


RESULTS_DIR_NAME = "sionna_results_by_rx"
WORK_DIR_NAME = ".work"
INDEX_FILENAME = "index.json"
TENSOR_SCHEMA = "rss_float32_zyx_v1"
INDEX_SCHEMA_VERSION = 2


def atomic_write_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def results_dir(block_dir: Path) -> Path:
    return block_dir / RESULTS_DIR_NAME


def rx_npy_path(block_dir: Path, rx_index: int) -> Path:
    return results_dir(block_dir) / f"rx_{rx_index:03d}.npy"


def rx_partial_path(block_dir: Path, rx_index: int) -> Path:
    return results_dir(block_dir) / WORK_DIR_NAME / f"rx_{rx_index:03d}.partial.npy"


def group_id(rx_indices: Iterable[int]) -> str:
    indices = sorted(int(value) for value in rx_indices)
    if not indices:
        raise ValueError("RX group cannot be empty")
    return f"rx_{indices[0]:03d}_{indices[-1]:03d}_n{len(indices):03d}"


def group_checkpoint_path(block_dir: Path, rx_indices: Iterable[int]) -> Path:
    return results_dir(block_dir) / WORK_DIR_NAME / f"{group_id(rx_indices)}.checkpoint.json"


def desired_rx_signature(
    *, rx_index: int, x_m: float, y_m: float, physical_signature: dict[str, object]
) -> dict[str, object]:
    return {
        "rx_index": int(rx_index),
        "rx_position_xy_m": [round(float(x_m), 6), round(float(y_m), 6)],
        "physical_signature": physical_signature,
    }


def load_results_index(block_dir: Path) -> dict[str, object]:
    path = results_dir(block_dir) / INDEX_FILENAME
    empty = {"schema_version": INDEX_SCHEMA_VERSION, "tensor_schema": TENSOR_SCHEMA, "rx": {}}
    if not path.is_file():
        return empty
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return empty
    if payload.get("schema_version") != INDEX_SCHEMA_VERSION or payload.get("tensor_schema") != TENSOR_SCHEMA:
        return empty
    return payload


def load_valid_rx_entries(
    block_dir: Path,
    desired_signatures: dict[int, dict[str, object]],
    expected_shape_zyx: tuple[int, int, int],
) -> dict[int, dict[str, object]]:
    """Return atomically indexed RX tensors whose signature and array match."""
    raw_entries = load_results_index(block_dir).get("rx", {})
    if not isinstance(raw_entries, dict):
        return {}
    valid: dict[int, dict[str, object]] = {}
    for key, entry in raw_entries.items():
        try:
            index = int(key)
            if index not in desired_signatures or not isinstance(entry, dict):
                continue
            if entry.get("signature") != desired_signatures[index]:
                continue
            path = rx_npy_path(block_dir, index)
            if entry.get("filename") != path.name or not path.is_file():
                continue
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.shape != expected_shape_zyx or array.dtype != np.dtype("float32"):
                continue
            if int(entry.get("size_bytes", -1)) != path.stat().st_size:
                continue
            valid[index] = entry
        except (OSError, ValueError, TypeError):
            continue
    return valid


def completed_rx_indices(entries: dict[int, dict[str, object]]) -> set[int]:
    return set(entries)


def commit_rx_group(
    *, block_dir: Path, rx_signatures: dict[int, dict[str, object]],
    expected_shape_zyx: tuple[int, int, int],
    per_rx_height_summaries: dict[str, object], target_rx_count: int,
) -> Path:
    """Atomically add completed NPY files to the block index."""
    payload = load_results_index(block_dir)
    entries = payload.setdefault("rx", {})
    if not isinstance(entries, dict):
        entries = {}
        payload["rx"] = entries
    for index, signature in sorted(rx_signatures.items()):
        path = rx_npy_path(block_dir, index)
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.shape != expected_shape_zyx or array.dtype != np.dtype("float32"):
            raise ValueError(f"Invalid completed tensor {path}: {array.shape}, {array.dtype}")
        entries[str(index)] = {
            "filename": path.name, "dtype": "float32",
            "shape_zyx": list(expected_shape_zyx), "value_count": int(array.size),
            "size_bytes": path.stat().st_size, "signature": signature,
            "height_summaries": per_rx_height_summaries[str(index)],
        }
    payload.update({
        "schema_version": INDEX_SCHEMA_VERSION, "tensor_schema": TENSOR_SCHEMA,
        "axes": ["tx_voxel_z", "tx_voxel_y", "tx_voxel_x"],
        "dtype": "float32", "target_rx_count": int(target_rx_count),
    })
    path = results_dir(block_dir) / INDEX_FILENAME
    atomic_write_json(payload, path)
    return path


def aggregate_height_summaries(
    entries: dict[int, dict[str, object]], heights_m: list[float]
) -> list[dict[str, object]]:
    accumulators = [
        {"count": 0, "sum": 0.0, "min": math.inf, "max": -math.inf}
        for _ in heights_m
    ]
    for entry in entries.values():
        for summary in entry["height_summaries"]:
            iz, count = int(summary["tx_voxel_iz"]), int(summary["value_count"])
            acc = accumulators[iz]
            acc["count"] += count
            acc["sum"] += float(summary["rss_mean_dbm"]) * count
            acc["min"] = min(acc["min"], float(summary["rss_min_dbm"]))
            acc["max"] = max(acc["max"], float(summary["rss_max_dbm"]))
    output: list[dict[str, object]] = []
    for iz, (height, acc) in enumerate(zip(heights_m, accumulators)):
        count = int(acc["count"])
        if count <= 0:
            raise ValueError(f"No completed tensor summary for height index {iz}")
        output.append({
            "height_m": float(height), "value_count": count,
            "sim_rx_power_mean_dbm": float(acc["sum"]) / count,
            "sim_rx_power_min_dbm": float(acc["min"]),
            "sim_rx_power_max_dbm": float(acc["max"]),
        })
    return output
