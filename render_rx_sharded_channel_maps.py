#!/usr/bin/env python3
"""Render dynamically sized per-RX NPY tensors as 128x128 PNG previews."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from render_voxel_channel_maps import (
    NO_PATH_THRESHOLD_DBM,
    OUTPUT_DIR_NAME,
    ProgressBar,
    atomic_write_json,
    choose_color_scale,
    compact_number,
    is_valid_128_png,
    local_now,
    log,
    source_signature,
)
from rx_sharded_results import RESULTS_DIR_NAME, rx_npy_path
from rt_osm_scene import lonlat_to_local_m


RENDERER_VERSION = 5
COMPLETION_MARKER = "render_complete.json"
IN_PROGRESS_MARKER = "render_in_progress.json"
DEFAULT_WORKERS = 12
MAX_WORKERS = 12


def block_dimensions(block_dir: Path) -> tuple[int, int, int, int, float, float]:
    metadata = json.loads((block_dir / "metadata.json").read_text(encoding="utf-8"))
    nx, ny, nz = (int(value) for value in metadata["tx_voxel_shape_xyz"])
    rx_count = int(metadata["ground_rx_count"])
    return (
        rx_count,
        nx,
        ny,
        nz,
        float(metadata["voxel_size_m"]),
        float(metadata["voxel_z_start_m"]),
    )


def shard_paths(block_dir: Path, rx_count: int) -> list[Path]:
    paths = [rx_npy_path(block_dir, index) for index in range(rx_count)]
    return paths if all(path.is_file() and path.stat().st_size > 0 for path in paths) else []


def image_path(output_dir: Path, rx_index: int, iz: int, z_start: float, voxel_size: float) -> Path:
    height = z_start + iz * voxel_size
    return output_dir / f"rx_{rx_index:03d}" / f"z_{iz:03d}_{compact_number(height)}m.png"


def source_state(block_dir: Path, paths: list[Path]) -> dict[str, object]:
    return {
        "renderer_version": RENDERER_VERSION,
        "source_shards": [
            {"filename": path.name, **source_signature(path)} for path in paths
        ],
        "source_osm": source_signature(block_dir / "osm_map.osm"),
    }


def all_expected_images_are_valid(
    output_dir: Path,
    rx_count: int,
    nz: int,
    z_start: float,
    voxel_size: float,
) -> bool:
    return all(
        is_valid_128_png(image_path(output_dir, rx_index, iz, z_start, voxel_size))
        for rx_index in range(rx_count)
        for iz in range(nz)
    )


def completed_render_is_current(block_dir: Path, output_dir: Path) -> bool:
    marker_path = output_dir / COMPLETION_MARKER
    if not marker_path.is_file() or not (block_dir / "osm_map.osm").is_file():
        return False
    try:
        rx_count, nx, ny, nz, voxel_size, z_start = block_dimensions(block_dir)
        paths = shard_paths(block_dir, rx_count)
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, TypeError):
        return False
    if not paths:
        return False
    return (
        marker.get("source_state") == source_state(block_dir, paths)
        and marker.get("cube_shape_rx_zyx") == [rx_count, nz, ny, nx]
        and marker.get("image_count") == rx_count * nz
        and all_expected_images_are_valid(output_dir, rx_count, nz, z_start, voxel_size)
    )


def load_result_cube(block_dir: Path, show_progress: bool = True) -> np.ndarray:
    rx_count, nx, ny, nz, _voxel_size, _z_start = block_dimensions(block_dir)
    if (nx, ny) != (128, 128):
        raise ValueError(f"Preview renderer requires 128x128 horizontal grid, got {nx}x{ny}")
    paths = shard_paths(block_dir, rx_count)
    if not paths:
        raise FileNotFoundError(f"Incomplete RX shard set under {block_dir / RESULTS_DIR_NAME}")
    cube = np.empty((rx_count, nz, ny, nx), dtype=np.float32)
    progress = ProgressBar(
        f"NPY {block_dir.parent.name}/{block_dir.name}", rx_count, "tensors",
    ) if show_progress else None
    loaded_count = 0
    try:
        for expected_rx, path in enumerate(paths):
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.shape != (nz, ny, nx) or array.dtype != np.dtype("float32"):
                raise ValueError(
                    f"Invalid tensor {path}: shape={array.shape}, dtype={array.dtype}; "
                    f"expected {(nz, ny, nx)} float32"
                )
            cube[expected_rx] = array
            loaded_count += 1
            if progress is not None:
                progress.update(loaded_count)
    finally:
        if progress is not None:
            progress.close(loaded_count)
    if not np.isfinite(cube).all():
        raise ValueError(f"Non-finite RSS values found under {block_dir / RESULTS_DIR_NAME}")
    return cube


def parse_osm_number(value: str | None) -> float | None:
    if not value:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value.replace(",", "."))
    if not match:
        return None
    number = float(match.group(0))
    return number if math.isfinite(number) and number > 0 else None


def load_building_height_grid(block_dir: Path, nx: int, ny: int) -> np.ndarray:
    """Rasterize roof height; a building is black only below its roof."""
    from matplotlib.path import Path as MatplotlibPath

    metadata = json.loads((block_dir / "metadata.json").read_text(encoding="utf-8"))
    center = metadata["block_center"]
    center_lat, center_lon = float(center["lat"]), float(center["lon"])
    root = ET.parse(block_dir / "osm_map.osm").getroot()
    nodes = {
        node.attrib["id"]: (float(node.attrib["lat"]), float(node.attrib["lon"]))
        for node in root.findall("node")
    }
    block_size = float(metadata.get("block_size_m", nx * metadata["voxel_size_m"]))
    xs = -block_size / 2.0 + block_size / nx * (np.arange(nx) + 0.5)
    ys = -block_size / 2.0 + block_size / ny * (np.arange(ny) + 0.5)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    query = np.column_stack((gx.ravel(), gy.ravel()))
    roof_height = np.zeros((ny, nx), dtype=np.float32)
    for way in root.findall("way"):
        tags = {tag.attrib.get("k", ""): tag.attrib.get("v", "") for tag in way.findall("tag")}
        if tags.get("building", "no") == "no":
            continue
        polygon_latlon = [
            nodes[nd.attrib["ref"]]
            for nd in way.findall("nd")
            if nd.attrib.get("ref") in nodes
        ]
        if len(polygon_latlon) < 3:
            continue
        polygon = np.asarray([
            lonlat_to_local_m(lon, lat, center_lon, center_lat)
            for lat, lon in polygon_latlon
        ])
        if not np.array_equal(polygon[0], polygon[-1]):
            polygon = np.vstack((polygon, polygon[0]))
        height = parse_osm_number(tags.get("height"))
        if height is None:
            levels = parse_osm_number(tags.get("building:levels")) or parse_osm_number(tags.get("levels"))
            height = levels * 3.2 if levels is not None else 12.0
        inside = MatplotlibPath(polygon, closed=True).contains_points(query, radius=1e-9)
        mask = inside.reshape((ny, nx))
        roof_height[mask] = np.maximum(roof_height[mask], float(height))
    return roof_height


def render_block(
    block_dir: Path,
    output_dir_name: str,
    requested_min: float | None,
    requested_max: float | None,
    overwrite: bool,
    show_progress: bool = True,
) -> tuple[str, int]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import image as matplotlib_image
    from matplotlib import pyplot as plt

    rx_count, nx, ny, nz, voxel_size, z_start = block_dimensions(block_dir)
    paths = shard_paths(block_dir, rx_count)
    if not paths:
        raise FileNotFoundError(f"Incomplete RX shard set under {block_dir / RESULTS_DIR_NAME}")
    output_dir = block_dir / output_dir_name
    if not overwrite and completed_render_is_current(block_dir, output_dir):
        return "complete_skip", 0

    current_state = source_state(block_dir, paths)
    in_progress_path = output_dir / IN_PROGRESS_MARKER
    previous_state = None
    if in_progress_path.is_file():
        try:
            previous_state = json.loads(in_progress_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    refresh_all = overwrite or previous_state != current_state
    atomic_write_json(current_state, in_progress_path)

    cube = load_result_cube(block_dir, show_progress=show_progress)
    building_height_grid = load_building_height_grid(block_dir, nx, ny)
    color_min, color_max, scale_method = choose_color_scale(cube, requested_min, requested_max)
    output_dir.mkdir(parents=True, exist_ok=True)
    color_map = plt.colormaps["turbo"].copy()
    no_path_rgba = np.asarray([209, 213, 219, 255], dtype=np.uint8)
    building_rgba = np.asarray([0, 0, 0, 255], dtype=np.uint8)

    written = 0
    processed = 0
    progress = ProgressBar(
        f"PNG {block_dir.parent.name}/{block_dir.name}", rx_count * nz, "images"
    ) if show_progress else None
    try:
        for rx_index in range(rx_count):
            (output_dir / f"rx_{rx_index:03d}").mkdir(parents=True, exist_ok=True)
            for iz in range(nz):
                destination = image_path(output_dir, rx_index, iz, z_start, voxel_size)
                if refresh_all or not is_valid_128_png(destination):
                    temporary = destination.with_suffix(".png.tmp")
                    layer = cube[rx_index, iz]
                    normalized = np.clip((layer - color_min) / (color_max - color_min), 0.0, 1.0)
                    rgba = color_map(normalized, bytes=True)
                    rgba[layer <= NO_PATH_THRESHOLD_DBM] = no_path_rgba
                    layer_height = z_start + iz * voxel_size
                    rgba[building_height_grid + 1e-6 >= layer_height] = building_rgba
                    matplotlib_image.imsave(temporary, rgba, origin="lower", format="png")
                    os.replace(temporary, destination)
                    written += 1
                processed += 1
                if progress is not None:
                    progress.update(processed)
    finally:
        if progress is not None:
            progress.close(processed)

    if not all_expected_images_are_valid(output_dir, rx_count, nz, z_start, voxel_size):
        raise RuntimeError("One or more preview PNG files are missing or invalid")
    marker = {
        "renderer_version": RENDERER_VERSION,
        "completed_local": local_now(),
        "source_state": current_state,
        "cube_shape_rx_zyx": [rx_count, nz, ny, nx],
        "image_shape_px": [ny, nx],
        "image_count": rx_count * nz,
        "file_layout": "rx_NNN/z_ZZZ_HEIGHTm.png",
        "colormap": "turbo",
        "color_min_dbm": color_min,
        "color_max_dbm": color_max,
        "color_scale_method": scale_method,
        "no_resolved_path_color": "#d1d5db",
        "building_mask_color": "#000000",
        "building_mask_rule": "black only where roof_height_m >= TX layer height",
    }
    atomic_write_json(marker, output_dir / COMPLETION_MARKER)
    in_progress_path.unlink(missing_ok=True)
    return "rendered", written


def render_block_worker(block_dir_text: str, output_dir_name: str, vmin, vmax, overwrite):
    block_dir = Path(block_dir_text)
    try:
        status, written = render_block(
            block_dir, output_dir_name, vmin, vmax, overwrite, show_progress=False
        )
        return {"block_dir": block_dir_text, "status": status, "written": written, "error": None}
    except Exception as exc:
        return {
            "block_dir": block_dir_text,
            "status": "failed",
            "written": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def manifest_block_dirs(dataset_root: Path) -> list[Path]:
    manifest = dataset_root / "blocks_manifest.csv"
    if not manifest.is_file():
        return []
    with manifest.open("r", newline="", encoding="utf-8-sig") as handle:
        return [
            dataset_root / row["region_slug"] / row["block_id"]
            for row in csv.DictReader(handle)
        ]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir-name", default=OUTPUT_DIR_NAME)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--vmin", type=float)
    parser.add_argument("--vmax", type=float)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        print(f"Dataset directory does not exist: {dataset_root}", file=sys.stderr)
        return 2
    if not 1 <= args.workers <= MAX_WORKERS:
        print(f"--workers must be between 1 and {MAX_WORKERS}", file=sys.stderr)
        return 2
    if (args.vmin is None) != (args.vmax is None):
        print("--vmin and --vmax must be supplied together", file=sys.stderr)
        return 2
    block_dirs = [
        path for path in manifest_block_dirs(dataset_root)
        if (path / RESULTS_DIR_NAME / "index.json").is_file()
    ]
    jobs = [
        path for path in block_dirs
        if args.overwrite or not completed_render_is_current(path, path / args.output_dir_name)
    ]
    log(f"Sharded preview jobs={len(jobs)}, already_complete={len(block_dirs)-len(jobs)}")
    failed = 0
    written = 0
    worker_count = min(args.workers, len(jobs)) if jobs else 0
    if worker_count == 1:
        results = [
            render_block_worker(str(path), args.output_dir_name, args.vmin, args.vmax, args.overwrite)
            for path in jobs
        ]
    elif worker_count > 1:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    render_block_worker,
                    str(path), args.output_dir_name, args.vmin, args.vmax, args.overwrite,
                )
                for path in jobs
            ]
            results = [future.result() for future in as_completed(futures)]
    else:
        results = []
    for result in results:
        written += int(result["written"])
        if result["status"] == "failed":
            failed += 1
            log(f"FAILED {result['block_dir']}: {result['error']}")
    log(f"Render complete: blocks={len(results)}, images_written={written}, failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
