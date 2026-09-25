#!/usr/bin/env python3
"""Render compact voxel RSS CSV files as exact 128 x 128 PNG previews.

The generator directory and dataset directory are expected to be siblings::

    ROOT/
      ubuntu_dataset_generator/
        render_voxel_channel_maps.py
      project985_39_main_voxel_256m_128x128x40_rx10_compact5_rand10to32/

Run this file without arguments on either server. Only finalized
``sionna_results.csv`` files are considered; an in-progress ``.csv.tmp`` file
is never read. Each block gets 400 images (10 ground RX x 40 aerial TX
heights), organized into one subdirectory per RX.

Building footprints are painted pure black. Cells for which Sionna resolved no
path remain gray. The color stretch is preview-only and is not a training-data
normalization contract.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import struct
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
from rt_osm_scene import lonlat_to_local_m


DATASET_NAME = "project985_39_main_voxel_256m_128x128x40_rx10_compact5_rand10to32"
GENERATOR_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = GENERATOR_DIR.parent / DATASET_NAME

GRID_NX = 128
GRID_NY = 128
GRID_NZ = 40
GROUND_RX_COUNT = 10
VOXEL_SIZE_M = 2.0
VOXEL_Z_START_M = 2.0
NO_PATH_THRESHOLD_DBM = -279.9
OUTPUT_DIR_NAME = "rss_maps_128px"
COMPLETION_MARKER = "render_complete.json"
IN_PROGRESS_MARKER = "render_in_progress.json"
RENDERER_VERSION = 2
EXPECTED_ROWS = GRID_NX * GRID_NY * GRID_NZ * GROUND_RX_COUNT
EXPECTED_IMAGES = GRID_NZ * GROUND_RX_COUNT
DEFAULT_WORKERS = 12
MAX_WORKERS = 12


def local_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{local_now()}] {message}", flush=True)


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "--:--"
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


class ProgressBar:
    """Small dependency-free terminal progress bar.

    Interactive terminals get an in-place bar. Redirected output gets one
    normal line at 5 percent intervals, which remains readable with nohup.
    Dynamic updates are intentionally not copied into the persistent log.
    """

    def __init__(self, label: str, total: int, unit: str) -> None:
        self.label = label
        self.total = max(int(total), 1)
        self.unit = unit
        self.started = time.perf_counter()
        self.last_drawn = 0.0
        self.last_value = -1
        self.interactive = bool(getattr(sys.stdout, "isatty", lambda: False)())
        self.next_noninteractive_percent = 0
        self.closed = False
        self.update(0, force=True)

    def update(self, value: int, force: bool = False) -> None:
        if self.closed:
            return
        value = min(max(int(value), 0), self.total)
        now = time.perf_counter()
        if not force and value < self.total and now - self.last_drawn < 0.20:
            return
        percent = 100.0 * value / self.total
        if not self.interactive and not force:
            integer_percent = int(percent)
            if integer_percent < self.next_noninteractive_percent and value < self.total:
                return
            self.next_noninteractive_percent = (integer_percent // 5 + 1) * 5
        elapsed = max(now - self.started, 1e-9)
        rate = value / elapsed
        eta = (self.total - value) / rate if rate > 0 else math.inf
        width = 30
        filled = min(width, int(width * value / self.total))
        bar = "=" * filled + ">" + "." * max(0, width - filled - 1)
        if value >= self.total:
            bar = "=" * width
        line = (
            f"{self.label} [{bar}] {percent:6.2f}% "
            f"{value:,}/{self.total:,} {self.unit} "
            f"{rate:,.0f}/s ETA {format_duration(eta)}"
        )
        if self.interactive:
            print("\r" + line.ljust(120), end="", flush=True)
        else:
            print(line, flush=True)
        self.last_drawn = now
        self.last_value = value

    def close(self, value: int | None = None) -> None:
        if self.closed:
            return
        final_value = self.last_value if value is None else value
        final_value = max(final_value, 0)
        if final_value != self.last_value:
            self.update(final_value, force=True)
        if self.interactive:
            print(flush=True)
        self.closed = True


def source_signature(csv_path: Path) -> dict[str, int]:
    stat = csv_path.stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def height_m(voxel_iz: int) -> float:
    return VOXEL_Z_START_M + voxel_iz * VOXEL_SIZE_M


def compact_number(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def image_path(output_dir: Path, rx_index: int, voxel_iz: int) -> Path:
    return (
        output_dir
        / f"rx_{rx_index:03d}"
        / f"z_{voxel_iz:03d}_{compact_number(height_m(voxel_iz))}m.png"
    )


def is_valid_128_png(path: Path) -> bool:
    """Check the PNG signature and IHDR dimensions without requiring Pillow."""
    try:
        with path.open("rb") as handle:
            header = handle.read(24)
        return (
            len(header) == 24
            and header[:8] == b"\x89PNG\r\n\x1a\n"
            and header[12:16] == b"IHDR"
            and struct.unpack(">II", header[16:24]) == (GRID_NX, GRID_NY)
        )
    except OSError:
        return False


def all_expected_images_are_valid(output_dir: Path) -> bool:
    return all(
        is_valid_128_png(image_path(output_dir, rx_index, voxel_iz))
        for rx_index in range(GROUND_RX_COUNT)
        for voxel_iz in range(GRID_NZ)
    )


def completed_render_is_current(csv_path: Path, output_dir: Path) -> bool:
    marker_path = output_dir / COMPLETION_MARKER
    osm_path = csv_path.parent / "osm_map.osm"
    if not marker_path.is_file() or not osm_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        marker.get("renderer_version") == RENDERER_VERSION
        and marker.get("source_csv") == source_signature(csv_path)
        and marker.get("source_osm") == source_signature(osm_path)
        and marker.get("image_shape_px") == [GRID_NY, GRID_NX]
        and marker.get("image_count") == EXPECTED_IMAGES
        and all_expected_images_are_valid(output_dir)
    )


def render_source_state(csv_path: Path) -> dict[str, object]:
    return {
        "renderer_version": RENDERER_VERSION,
        "source_csv": source_signature(csv_path),
        "source_osm": source_signature(csv_path.parent / "osm_map.osm"),
    }


def load_result_cube(csv_path: Path, show_progress: bool = True) -> np.ndarray:
    """Load and validate [ground_rx, z, y, x] RSS values from one block."""
    cube = np.full(
        (GROUND_RX_COUNT, GRID_NZ, GRID_NY, GRID_NX),
        np.nan,
        dtype=np.float32,
    )
    seen = np.zeros(cube.shape, dtype=np.bool_)
    duplicate_count = 0
    row_count = 0
    required = ("rx_index", "tx_voxel_ix", "tx_voxel_iy", "tx_voxel_iz")

    label = f"CSV {csv_path.parent.parent.name}/{csv_path.parent.name}"
    progress = ProgressBar(label, EXPECTED_ROWS, "rows") if show_progress else None
    try:
        with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise ValueError("CSV is empty") from exc
            column = {name: index for index, name in enumerate(header)}
            missing_columns = [name for name in required if name not in column]
            if missing_columns:
                raise ValueError(f"missing columns: {', '.join(missing_columns)}")
            rss_column = (
                "rss_dbm" if "rss_dbm" in column
                else "sim_rx_power_dbm" if "sim_rx_power_dbm" in column
                else None
            )
            if rss_column is None:
                raise ValueError("missing RSS column: expected rss_dbm or sim_rx_power_dbm")

            for csv_line, row in enumerate(reader, start=2):
                try:
                    rx_index = int(row[column["rx_index"]])
                    ix = int(row[column["tx_voxel_ix"]])
                    iy = int(row[column["tx_voxel_iy"]])
                    iz = int(row[column["tx_voxel_iz"]])
                    rss_dbm = float(row[column[rss_column]])
                except (IndexError, ValueError) as exc:
                    raise ValueError(f"invalid data at CSV line {csv_line}: {exc}") from exc

                if not (0 <= rx_index < GROUND_RX_COUNT):
                    raise ValueError(f"rx_index={rx_index} out of range at CSV line {csv_line}")
                if not (0 <= ix < GRID_NX and 0 <= iy < GRID_NY and 0 <= iz < GRID_NZ):
                    raise ValueError(
                        f"voxel index ({ix}, {iy}, {iz}) out of range at CSV line {csv_line}"
                    )
                if "height_m" in column:
                    csv_height = float(row[column["height_m"]])
                    expected_height = height_m(iz)
                    if not math.isclose(csv_height, expected_height, abs_tol=1e-6):
                        raise ValueError(
                            f"height {csv_height:g} does not match iz={iz} "
                            f"({expected_height:g} m) at CSV line {csv_line}"
                        )
                key = (rx_index, iz, iy, ix)
                if seen[key]:
                    duplicate_count += 1
                seen[key] = True
                cube[key] = rss_dbm
                row_count += 1
                if progress is not None and row_count % 10_000 == 0:
                    progress.update(row_count)
    finally:
        if progress is not None:
            progress.close(row_count)

    missing_count = int(seen.size - np.count_nonzero(seen))
    if row_count != EXPECTED_ROWS or duplicate_count or missing_count:
        raise ValueError(
            "incomplete or duplicated result cube: "
            f"rows={row_count:,}/{EXPECTED_ROWS:,}, "
            f"duplicates={duplicate_count:,}, missing_cells={missing_count:,}"
        )
    if not np.isfinite(cube).all():
        invalid_count = int(np.size(cube) - np.count_nonzero(np.isfinite(cube)))
        raise ValueError(f"result cube contains {invalid_count:,} non-finite RSS values")
    return cube


def load_building_mask(block_dir: Path) -> np.ndarray:
    """Rasterize the OSM building footprints onto the 128 x 128 voxel grid."""
    from matplotlib.path import Path as MatplotlibPath

    osm_path = block_dir / "osm_map.osm"
    metadata_path = block_dir / "metadata.json"
    if not osm_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("building preview requires osm_map.osm and metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    center = metadata.get("block_center", {})
    center_lat = float(center["lat"])
    center_lon = float(center["lon"])

    def local_xy(lat: float, lon: float) -> tuple[float, float]:
        return lonlat_to_local_m(lon, lat, center_lon, center_lat)

    root = ET.parse(osm_path).getroot()
    nodes = {
        node.attrib["id"]: (float(node.attrib["lat"]), float(node.attrib["lon"]))
        for node in root.findall("node")
    }
    block_size_m = float(metadata.get("block_size_m", GRID_NX * VOXEL_SIZE_M))
    voxel_size_x = block_size_m / GRID_NX
    voxel_size_y = block_size_m / GRID_NY
    x_coordinates = -block_size_m / 2.0 + voxel_size_x * (
        np.arange(GRID_NX, dtype=np.float64) + 0.5
    )
    y_coordinates = -block_size_m / 2.0 + voxel_size_y * (
        np.arange(GRID_NY, dtype=np.float64) + 0.5
    )
    grid_x, grid_y = np.meshgrid(x_coordinates, y_coordinates, indexing="xy")
    query_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    mask = np.zeros((GRID_NY, GRID_NX), dtype=np.bool_)
    for way in root.findall("way"):
        tags = {
            tag.attrib.get("k", ""): tag.attrib.get("v", "")
            for tag in way.findall("tag")
        }
        if tags.get("building", "no") == "no":
            continue
        lonlat = [
            nodes[nd.attrib["ref"]]
            for nd in way.findall("nd")
            if nd.attrib.get("ref") in nodes
        ]
        if len(lonlat) < 3:
            continue
        polygon = np.asarray([local_xy(lat, lon) for lat, lon in lonlat], dtype=np.float64)
        if not np.array_equal(polygon[0], polygon[-1]):
            polygon = np.vstack((polygon, polygon[0]))
        inside = MatplotlibPath(polygon, closed=True).contains_points(
            query_points, radius=1e-9
        )
        mask |= inside.reshape((GRID_NY, GRID_NX))
    return mask


def choose_color_scale(
    cube: np.ndarray,
    requested_min: float | None,
    requested_max: float | None,
) -> tuple[float, float, str]:
    if (requested_min is None) != (requested_max is None):
        raise ValueError("--vmin and --vmax must be supplied together")
    if requested_min is not None and requested_max is not None:
        if not requested_min < requested_max:
            raise ValueError("--vmin must be smaller than --vmax")
        return requested_min, requested_max, "command_line_fixed"

    covered = cube[cube > NO_PATH_THRESHOLD_DBM]
    values = covered if covered.size else cube[np.isfinite(cube)]
    if not values.size:
        raise ValueError("no finite RSS values are available for plotting")
    color_min, color_max = np.percentile(values, [1.0, 99.0]).astype(float)
    if math.isclose(color_min, color_max):
        color_min -= 0.5
        color_max += 0.5
    return color_min, color_max, "per_block_covered_cell_percentile_1_99"


def atomic_write_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def render_block(
    csv_path: Path,
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

    block_dir = csv_path.parent
    output_dir = block_dir / output_dir_name
    if not overwrite and completed_render_is_current(csv_path, output_dir):
        return "complete_skip", 0

    current_state = render_source_state(csv_path)
    in_progress_path = output_dir / IN_PROGRESS_MARKER
    previous_state: dict[str, object] | None = None
    if in_progress_path.is_file():
        try:
            previous_state = json.loads(in_progress_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous_state = None
    # Reuse valid partial PNGs only when they were created by this renderer
    # version from these exact CSV/OSM sources. Old-version previews must be
    # refreshed even if their dimensions happen to be correct.
    refresh_all_images = overwrite or previous_state != current_state
    atomic_write_json(current_state, in_progress_path)

    cube = load_result_cube(csv_path, show_progress=show_progress)
    building_mask = load_building_mask(block_dir)
    color_min, color_max, scale_method = choose_color_scale(
        cube, requested_min, requested_max
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    color_map = plt.colormaps["turbo"].copy()
    no_path_rgba = np.asarray([209, 213, 219, 255], dtype=np.uint8)
    building_rgba = np.asarray([0, 0, 0, 255], dtype=np.uint8)

    written = 0
    processed = 0
    label = f"PNG {csv_path.parent.parent.name}/{csv_path.parent.name}"
    progress = ProgressBar(label, EXPECTED_IMAGES, "images") if show_progress else None
    try:
        for rx_index in range(GROUND_RX_COUNT):
            (output_dir / f"rx_{rx_index:03d}").mkdir(parents=True, exist_ok=True)
            for iz in range(GRID_NZ):
                destination = image_path(output_dir, rx_index, iz)
                if refresh_all_images or not is_valid_128_png(destination):
                    temporary = destination.with_suffix(".png.tmp")
                    layer = cube[rx_index, iz]
                    normalized = np.clip(
                        (layer - color_min) / (color_max - color_min), 0.0, 1.0
                    )
                    rgba = color_map(normalized, bytes=True)
                    rgba[layer <= NO_PATH_THRESHOLD_DBM] = no_path_rgba
                    rgba[building_mask] = building_rgba
                    matplotlib_image.imsave(
                        temporary,
                        rgba,
                        origin="lower",
                        format="png",
                    )
                    os.replace(temporary, destination)
                    written += 1
                processed += 1
                if progress is not None:
                    progress.update(processed)
    finally:
        if progress is not None:
            progress.close(processed)

    if not all_expected_images_are_valid(output_dir):
        raise RuntimeError("one or more rendered PNG files are missing or not 128 x 128")

    marker = {
        "renderer_version": RENDERER_VERSION,
        "completed_local": local_now(),
        "source_filename": csv_path.name,
        "source_csv": source_signature(csv_path),
        "source_osm": source_signature(block_dir / "osm_map.osm"),
        "cube_shape_rx_zyx": [GROUND_RX_COUNT, GRID_NZ, GRID_NY, GRID_NX],
        "image_shape_px": [GRID_NY, GRID_NX],
        "image_count": EXPECTED_IMAGES,
        "file_layout": "rx_NNN/z_ZZZ_HEIGHTm.png",
        "pixel_orientation": "origin_lower; +x right; +y up",
        "rss_units": "dBm",
        "colormap": "turbo",
        "color_min_dbm": color_min,
        "color_max_dbm": color_max,
        "color_scale_method": scale_method,
        "no_resolved_path_rule": f"RSS <= {NO_PATH_THRESHOLD_DBM:g} dBm",
        "no_resolved_path_color": "#d1d5db",
        "building_mask_source": "OSM way[building] footprint at voxel-cell centers",
        "building_mask_color": "#000000",
        "building_masked_pixel_count": int(np.count_nonzero(building_mask)),
    }
    atomic_write_json(marker, output_dir / COMPLETION_MARKER)
    in_progress_path.unlink(missing_ok=True)
    return "rendered", written


def render_block_worker(
    csv_path_text: str,
    output_dir_name: str,
    requested_min: float | None,
    requested_max: float | None,
    overwrite: bool,
) -> dict[str, object]:
    """Process-pool entry point; keep child process output quiet."""
    csv_path = Path(csv_path_text)
    try:
        status, written = render_block(
            csv_path,
            output_dir_name,
            requested_min,
            requested_max,
            overwrite,
            show_progress=False,
        )
        return {
            "csv_path": csv_path_text,
            "status": status,
            "written": written,
            "error": None,
        }
    except Exception as exc:
        return {
            "csv_path": csv_path_text,
            "status": "failed",
            "written": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def manifest_block_dirs(dataset_root: Path) -> list[Path]:
    manifest = dataset_root / "blocks_manifest.csv"
    if manifest.is_file():
        with manifest.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        return [
            dataset_root / row["region_slug"] / row["block_id"]
            for row in rows
        ]

    # Fallback for a copied partial dataset without its root manifest.
    return sorted({path.parent for path in dataset_root.glob("*/*/sionna_results.csv")})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render each completed 10-RX x 40-height voxel RSS CSV into "
            "exact 128 x 128 PNG layers."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"dataset directory (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument(
        "--output-dir-name",
        default=OUTPUT_DIR_NAME,
        help=f"folder created inside every block (default: {OUTPUT_DIR_NAME})",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="optional fixed minimum RSS color scale in dBm",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="optional fixed maximum RSS color scale in dBm",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="render all PNG files again, including valid existing images",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"parallel block processes, 1-{MAX_WORKERS} (default: {DEFAULT_WORKERS})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        print(f"Dataset directory does not exist: {dataset_root}", file=sys.stderr)
        return 2
    if not args.output_dir_name or Path(args.output_dir_name).name != args.output_dir_name:
        print("--output-dir-name must be one directory name", file=sys.stderr)
        return 2
    if (args.vmin is None) != (args.vmax is None):
        print("--vmin and --vmax must be supplied together", file=sys.stderr)
        return 2
    if args.vmin is not None and not args.vmin < args.vmax:
        print("--vmin must be smaller than --vmax", file=sys.stderr)
        return 2
    if not 1 <= args.workers <= MAX_WORKERS:
        print(f"--workers must be between 1 and {MAX_WORKERS}", file=sys.stderr)
        return 2

    block_dirs = manifest_block_dirs(dataset_root)
    if not block_dirs:
        log(f"No block directories found under {dataset_root}")
        return 0

    log(f"Dataset root: {dataset_root}")
    log(
        f"Scanning {len(block_dirs)} blocks; each completed CSV produces "
        f"{EXPECTED_IMAGES} PNG files"
    )
    no_csv = 0
    complete_skip = 0
    rendered_blocks = 0
    written_images = 0
    failed = 0
    jobs: list[Path] = []
    for block_dir in block_dirs:
        relative = block_dir.relative_to(dataset_root)
        csv_path = block_dir / "sionna_results.csv"
        if not csv_path.is_file():
            no_csv += 1
            continue
        output_dir = block_dir / args.output_dir_name
        if not args.overwrite and completed_render_is_current(csv_path, output_dir):
            complete_skip += 1
            continue
        jobs.append(csv_path)

    worker_count = min(args.workers, len(jobs)) if jobs else 0
    log(
        f"READY: render_jobs={len(jobs)}, workers={worker_count}, "
        f"already_complete={complete_skip}, no_final_csv={no_csv}"
    )
    results: list[dict[str, object]] = []
    if jobs:
        if worker_count == 1:
            # Preserve detailed per-CSV and per-PNG progress in explicit
            # single-worker mode.
            for csv_path in jobs:
                try:
                    status, written = render_block(
                        csv_path,
                        args.output_dir_name,
                        args.vmin,
                        args.vmax,
                        args.overwrite,
                        show_progress=True,
                    )
                    results.append({
                        "csv_path": str(csv_path),
                        "status": status,
                        "written": written,
                        "error": None,
                    })
                except Exception as exc:
                    results.append({
                        "csv_path": str(csv_path),
                        "status": "failed",
                        "written": 0,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
        else:
            progress = ProgressBar("BLOCK RENDER", len(jobs), "blocks")
            completed_jobs = 0
            try:
                with ProcessPoolExecutor(max_workers=worker_count) as executor:
                    futures = [
                        executor.submit(
                            render_block_worker,
                            str(csv_path),
                            args.output_dir_name,
                            args.vmin,
                            args.vmax,
                            args.overwrite,
                        )
                        for csv_path in jobs
                    ]
                    for future in as_completed(futures):
                        try:
                            results.append(future.result())
                        except BaseException as exc:
                            results.append({
                                "csv_path": "<worker process>",
                                "status": "failed",
                                "written": 0,
                                "error": f"{type(exc).__name__}: {exc}",
                            })
                        completed_jobs += 1
                        progress.update(completed_jobs, force=True)
            finally:
                progress.close(completed_jobs)

    for result in results:
        csv_path = Path(str(result["csv_path"]))
        try:
            relative = csv_path.parent.relative_to(dataset_root)
        except ValueError:
            relative = csv_path.parent
        if result["status"] == "failed":
            failed += 1
            log(f"FAILED {relative}: {result['error']}")
        elif result["status"] == "complete_skip":
            complete_skip += 1
        else:
            rendered_blocks += 1
            written = int(result["written"])
            written_images += written
            log(f"RENDERED {relative}: wrote {written}/{EXPECTED_IMAGES} images")

    log(
        "SUMMARY: "
        f"blocks={len(block_dirs)}, rendered={rendered_blocks}, "
        f"already_complete={complete_skip}, no_final_csv={no_csv}, "
        f"failed={failed}, new_png_files={written_images}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
