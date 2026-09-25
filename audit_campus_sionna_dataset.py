"""Strict completion audit for the OSM campus Sionna RT dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from render_voxel_channel_maps import OUTPUT_DIR_NAME as RSS_PREVIEW_DIR_NAME
from render_rx_sharded_channel_maps import completed_render_is_current
from rx_sharded_results import (
    RESULTS_DIR_NAME,
    completed_rx_indices,
    desired_rx_signature,
    load_valid_rx_entries,
    rx_npy_path,
)

from campus_sionna_dataset import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_MESH_SPACING_M,
    DEFAULT_RX_COUNT,
    DEFAULT_RX_HEIGHT_M,
    DEFAULT_VOXEL_NX,
    DEFAULT_VOXEL_NY,
    DEFAULT_VOXEL_NZ,
    DEFAULT_VOXEL_SIZE_M,
    DEFAULT_VOXEL_Z_START_M,
    MAP_BLOCK_SIZE_M,
    MIN_BUILDING_CLEARANCE_M,
    REGIONS,
    TX_MESH_SIZE_M,
    BlockSpec,
    building_local_polygons,
    parse_block_buildings,
    point_to_polygon_distance,
    read_rx_positions_csv,
)


def audit_block(dataset_root: Path, block: BlockSpec, verify_results: bool) -> dict[str, object]:
    block_dir = dataset_root / block.region_slug / block.block_id
    errors: list[str] = []
    required_files = [
        "metadata.json",
        "osm_map.osm",
        "osm_buildings_2d.png",
        "osm_buildings_3d.png",
        "rx_positions.csv",
        "rx_positions_2d.png",
        "sionna_results.csv",
        "sionna_results_summary.csv",
    ]
    for name in required_files:
        path = block_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty {name}")
    scene_xml = block_dir / "scene_generated" / "osm_sustech_block.xml"
    if not scene_xml.is_file() or scene_xml.stat().st_size == 0:
        errors.append("missing scene XML")
    if errors:
        return {"region_slug": block.region_slug, "block_id": block.block_id, "status": "failed", "errors": errors}

    metadata = json.loads((block_dir / "metadata.json").read_text(encoding="utf-8"))
    grid_axis_count = len(np.arange(-TX_MESH_SIZE_M / 2.0, TX_MESH_SIZE_M / 2.0 + DEFAULT_MESH_SPACING_M * 0.25, DEFAULT_MESH_SPACING_M))
    tx_points_per_height = grid_axis_count * grid_axis_count
    expected_rows = DEFAULT_RX_COUNT * 5 * tx_points_per_height
    checks = {
        "metadata_complete": metadata.get("status") == "complete",
        "map_block_size_400m": math.isclose(float(metadata.get("map_block_size_m", -1)), MAP_BLOCK_SIZE_M),
        "tx_mesh_size_500m": math.isclose(float(metadata.get("tx_mesh_size_m", -1)), TX_MESH_SIZE_M),
        "ground_rx_count_20": metadata.get("ground_rx_count") == DEFAULT_RX_COUNT,
        "rx_height_0p5m": math.isclose(float(metadata.get("ground_rx_height_m", -1)), 0.5),
        "tx_mesh_spacing_expected": math.isclose(float(metadata.get("tx_mesh_spacing_m", -1)), DEFAULT_MESH_SPACING_M),
        "tx_points_per_height_expected": metadata.get("tx_mesh_points_per_height") == tx_points_per_height,
        "expected_result_rows_match": metadata.get("expected_result_rows") == expected_rows,
        "actual_result_rows_match": metadata.get("actual_result_rows") == expected_rows,
        "isotropic_tx": metadata.get("transmitter", {}).get("pattern") == "iso",
        "isotropic_rx": metadata.get("receiver", {}).get("pattern") == "iso",
    }
    maximum_height = float(metadata.get("maximum_building_height_m", 0.0))
    expected_heights = [max(10.0, maximum_height) + 5.0 * index for index in range(5)]
    actual_heights = [float(value) for value in metadata.get("simulation_heights_m", [])]
    checks["five_height_rule"] = len(actual_heights) == 5 and all(
        math.isclose(actual, expected, abs_tol=1e-6) for actual, expected in zip(actual_heights, expected_heights)
    )

    rx_points = read_rx_positions_csv(block_dir / "rx_positions.csv")
    checks["rx_csv_20_unique"] = len(rx_points) == DEFAULT_RX_COUNT and len({point.index for point in rx_points}) == DEFAULT_RX_COUNT
    osm_bytes = (block_dir / "osm_map.osm").read_bytes()
    checks["osm_sha256_matches_metadata"] = metadata.get("osm_sha256") == hashlib.sha256(osm_bytes).hexdigest()
    checks["osm_height_variant_recorded"] = bool(metadata.get("osm_height_variant"))
    buildings = parse_block_buildings(osm_bytes, block)
    polygons = building_local_polygons(block, buildings)
    actual_clearances = [
        min((point_to_polygon_distance(point.x_m, point.y_m, polygon) for polygon in polygons), default=math.inf)
        for point in rx_points
    ]
    checks["rx_outside_and_clearance_5m"] = all(value + 1e-6 >= MIN_BUILDING_CLEARANCE_M for value in actual_clearances)
    map_count = len(list((block_dir / "rss_maps_by_rx").glob("rx_*_rss_5_heights.png")))
    checks["twenty_per_rx_rss_figures"] = map_count == DEFAULT_RX_COUNT

    result_rows = 0
    duplicate_rows = 0
    finite_power_rows = 0
    if verify_results and not errors:
        height_lookup = {round(value, 6): index for index, value in enumerate(expected_heights)}
        seen = np.zeros((DEFAULT_RX_COUNT, 5, tx_points_per_height), dtype=bool)
        with (block_dir / "sionna_results.csv").open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            expected_columns = {"height_m", "tx_mesh_index", "tx_lat", "tx_lon", "rx_index", "rx_lat", "rx_lon", "sim_rx_power_dbm"}
            checks["result_columns"] = expected_columns.issubset(set(reader.fieldnames or []))
            for row in reader:
                result_rows += 1
                try:
                    rx_index = int(row["rx_index"])
                    tx_index = int(row["tx_mesh_index"])
                    height_index = height_lookup[round(float(row["height_m"]), 6)]
                    if seen[rx_index, height_index, tx_index]:
                        duplicate_rows += 1
                    else:
                        seen[rx_index, height_index, tx_index] = True
                    if math.isfinite(float(row["sim_rx_power_dbm"])):
                        finite_power_rows += 1
                except (KeyError, ValueError, IndexError):
                    errors.append(f"invalid result row at line {result_rows + 1}")
                    break
        checks["result_rows_expected"] = result_rows == expected_rows
        checks["all_result_combinations_once"] = bool(seen.all()) and duplicate_rows == 0
        checks["all_rss_finite"] = finite_power_rows == expected_rows

    errors.extend(name for name, passed in checks.items() if not passed)
    return {
        "region_slug": block.region_slug,
        "block_id": block.block_id,
        "status": "passed" if not errors else "failed",
        "checks": checks,
        "result_rows": result_rows if verify_results else None,
        "minimum_actual_building_clearance_m": None if all(math.isinf(value) for value in actual_clearances) else min(actual_clearances),
        "rss_map_file_count": map_count,
        "errors": errors,
    }


def audit_voxel_block(
    dataset_root: Path, block: BlockSpec, verify_results: bool
) -> dict[str, object]:
    block_dir = dataset_root / block.region_slug / block.block_id
    errors: list[str] = []
    required_files = [
        "metadata.json", "osm_map.osm", "osm_buildings_2d.png",
        "osm_buildings_3d.png", "tx_positions.csv", "tx_positions_2d.png",
        "sionna_results.csv", "sionna_results_summary.csv",
    ]
    for name in required_files:
        path = block_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty {name}")
    scene_xml = block_dir / "scene_generated" / "osm_sustech_block.xml"
    if not scene_xml.is_file() or scene_xml.stat().st_size == 0:
        errors.append("missing scene XML")
    if errors:
        return {
            "region_slug": block.region_slug, "block_id": block.block_id,
            "status": "failed", "errors": errors,
        }

    metadata = json.loads((block_dir / "metadata.json").read_text(encoding="utf-8"))
    voxel_count = DEFAULT_VOXEL_NX * DEFAULT_VOXEL_NY * DEFAULT_VOXEL_NZ
    expected_rows = DEFAULT_TX_COUNT * voxel_count
    checks = {
        "metadata_complete": metadata.get("status") == "complete",
        "schema_3": metadata.get("schema_version") == "3.0",
        "map_block_size_256m": math.isclose(float(metadata.get("map_block_size_m", -1)), MAP_BLOCK_SIZE_M),
        "ground_tx_count_10": metadata.get("ground_tx_count") == DEFAULT_TX_COUNT,
        "ground_tx_height_expected": math.isclose(float(metadata.get("ground_tx_height_m", -1)), DEFAULT_TX_HEIGHT_M),
        "voxel_shape_expected": metadata.get("voxel_shape_xyz") == [DEFAULT_VOXEL_NX, DEFAULT_VOXEL_NY, DEFAULT_VOXEL_NZ],
        "voxel_size_2m": math.isclose(float(metadata.get("voxel_size_m", -1)), DEFAULT_VOXEL_SIZE_M),
        "voxel_z_start_2m": math.isclose(float(metadata.get("voxel_z_start_m", -1)), DEFAULT_VOXEL_Z_START_M),
        "expected_result_rows_match": metadata.get("expected_result_rows") == expected_rows,
        "actual_result_rows_match": metadata.get("actual_result_rows") == expected_rows,
        "isotropic_tx": metadata.get("transmitter", {}).get("pattern") == "iso",
        "isotropic_rx": metadata.get("receiver", {}).get("pattern") == "iso",
    }

    with (block_dir / "tx_positions.csv").open("r", newline="", encoding="utf-8-sig") as file:
        tx_rows = list(csv.DictReader(file))
    checks["tx_csv_10_unique"] = (
        len(tx_rows) == DEFAULT_TX_COUNT
        and len({int(row["tx_index"]) for row in tx_rows}) == DEFAULT_TX_COUNT
    )
    osm_bytes = (block_dir / "osm_map.osm").read_bytes()
    checks["osm_sha256_matches_metadata"] = metadata.get("osm_sha256") == hashlib.sha256(osm_bytes).hexdigest()
    checks["osm_height_variant_recorded"] = bool(metadata.get("osm_height_variant"))
    buildings = parse_block_buildings(osm_bytes, block)
    polygons = building_local_polygons(block, buildings)
    actual_clearances = [
        min(
            (
                point_to_polygon_distance(float(row["x_m"]), float(row["y_m"]), polygon)
                for polygon in polygons
            ),
            default=math.inf,
        )
        for row in tx_rows
    ]
    checks["tx_outside_and_clearance_5m"] = all(
        value + 1e-6 >= MIN_BUILDING_CLEARANCE_M for value in actual_clearances
    )

    result_rows = 0
    duplicate_rows = 0
    finite_power_rows = 0
    if verify_results:
        seen = np.zeros((DEFAULT_TX_COUNT, voxel_count), dtype=bool)
        with (block_dir / "sionna_results.csv").open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            expected_columns = {
                "tx_index", "voxel_index", "voxel_ix", "voxel_iy", "voxel_iz",
                "rx_x_m", "rx_y_m", "rx_z_m", "sim_rx_power_dbm",
            }
            checks["result_columns"] = expected_columns.issubset(set(reader.fieldnames or []))
            for row in reader:
                result_rows += 1
                try:
                    tx_index = int(row["tx_index"])
                    voxel_index = int(row["voxel_index"])
                    if seen[tx_index, voxel_index]:
                        duplicate_rows += 1
                    else:
                        seen[tx_index, voxel_index] = True
                    if math.isfinite(float(row["sim_rx_power_dbm"])):
                        finite_power_rows += 1
                except (KeyError, ValueError, IndexError):
                    errors.append(f"invalid result row at line {result_rows + 1}")
                    break
        checks["result_rows_expected"] = result_rows == expected_rows
        checks["all_result_combinations_once"] = bool(seen.all()) and duplicate_rows == 0
        checks["all_rss_finite"] = finite_power_rows == expected_rows

    errors.extend(name for name, passed in checks.items() if not passed)
    return {
        "region_slug": block.region_slug,
        "block_id": block.block_id,
        "status": "passed" if not errors else "failed",
        "checks": checks,
        "result_rows": result_rows if verify_results else None,
        "minimum_actual_building_clearance_m": (
            None if all(math.isinf(value) for value in actual_clearances)
            else min(actual_clearances)
        ),
        "errors": errors,
    }


def audit_aerial_tx_voxel_block(
    dataset_root: Path, block: BlockSpec, verify_results: bool
) -> dict[str, object]:
    block_dir = dataset_root / block.region_slug / block.block_id
    errors: list[str] = []
    required_files = [
        "metadata.json", "osm_map.osm", "osm_buildings_2d.png",
        "osm_buildings_3d.png", "rx_positions.csv", "rx_positions_2d.png",
        "sionna_results_summary.csv",
    ]
    for name in required_files:
        path = block_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty {name}")
    scene_xml = block_dir / "scene_generated" / "osm_sustech_block.xml"
    if not scene_xml.is_file() or scene_xml.stat().st_size == 0:
        errors.append("missing scene XML")
    if errors:
        return {
            "region_slug": block.region_slug, "block_id": block.block_id,
            "status": "failed", "errors": errors,
        }

    metadata = json.loads((block_dir / "metadata.json").read_text(encoding="utf-8"))
    try:
        rx_count = int(metadata["ground_rx_count"])
        nx, ny, nz = (int(value) for value in metadata["tx_voxel_shape_xyz"])
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "region_slug": block.region_slug, "block_id": block.block_id,
            "status": "failed", "errors": [f"invalid dynamic dimensions: {exc}"],
        }
    tx_per_height = nx * ny
    tx_voxel_count = tx_per_height * nz
    expected_values = rx_count * tx_voxel_count
    checks = {
        "metadata_complete": metadata.get("status") == "complete",
        "schema_supported": metadata.get("schema_version") == "3.4",
        "per_rx_storage": metadata.get("results_storage") == "per_rx_npy_tensors_v1",
        "correct_layout": metadata.get("layout") == "3d_voxel_tx_to_fixed_ground_rx",
        "map_block_size_256m": math.isclose(float(metadata.get("map_block_size_m", -1)), MAP_BLOCK_SIZE_M),
        "ground_rx_count_positive": rx_count > 0,
        "ground_rx_height_expected": math.isclose(float(metadata.get("ground_rx_height_m", -1)), DEFAULT_RX_HEIGHT_M),
        "tx_voxel_shape_expected": [nx, ny, nz] == [DEFAULT_VOXEL_NX, DEFAULT_VOXEL_NY, DEFAULT_VOXEL_NZ],
        "voxel_size_2m": math.isclose(float(metadata.get("voxel_size_m", -1)), DEFAULT_VOXEL_SIZE_M),
        "voxel_z_start_2m": math.isclose(float(metadata.get("voxel_z_start_m", -1)), DEFAULT_VOXEL_Z_START_M),
        "expected_result_values_match": metadata.get("expected_result_values") == expected_values,
        "actual_result_values_match": metadata.get("actual_result_values") == expected_values,
        "isotropic_tx": metadata.get("transmitter", {}).get("pattern") == "iso",
        "isotropic_rx": metadata.get("receiver", {}).get("pattern") == "iso",
    }
    if metadata.get("layout") == "3d_voxel_tx_to_fixed_ground_rx" and "rss_preview" in metadata:
        checks["rss_preview_complete"] = (
            metadata.get("rss_preview", {}).get("status") in {"rendered", "complete_skip"}
            and completed_render_is_current(
                block_dir,
                block_dir / RSS_PREVIEW_DIR_NAME,
            )
        )
    rx_points = read_rx_positions_csv(block_dir / "rx_positions.csv")
    checks["rx_csv_count_and_indices"] = (
        len(rx_points) == rx_count
        and {point.index for point in rx_points} == set(range(rx_count))
    )
    physical_signature = metadata.get("physical_signature")
    desired_signatures = {
        point.index: desired_rx_signature(
            rx_index=point.index,
            x_m=point.x_m,
            y_m=point.y_m,
            physical_signature=physical_signature,
        )
        for point in rx_points
    } if isinstance(physical_signature, dict) else {}
    valid_entries = load_valid_rx_entries(
        block_dir, desired_signatures, (nz, ny, nx)
    ) if desired_signatures else []
    checks["all_rx_tensors_committed"] = (
        completed_rx_indices(valid_entries) == set(range(rx_count))
        and (block_dir / RESULTS_DIR_NAME / "index.json").is_file()
    )
    osm_bytes = (block_dir / "osm_map.osm").read_bytes()
    checks["osm_sha256_matches_metadata"] = metadata.get("osm_sha256") == hashlib.sha256(osm_bytes).hexdigest()
    checks["osm_height_variant_recorded"] = bool(metadata.get("osm_height_variant"))
    buildings = parse_block_buildings(osm_bytes, block)
    polygons = building_local_polygons(block, buildings)
    actual_clearances = [
        min(
            (point_to_polygon_distance(point.x_m, point.y_m, polygon) for polygon in polygons),
            default=math.inf,
        )
        for point in rx_points
    ]
    checks["rx_outside_and_clearance_5m"] = all(
        value + 1e-6 >= MIN_BUILDING_CLEARANCE_M for value in actual_clearances
    )

    result_values = 0
    if verify_results:
        for expected_rx in range(rx_count):
            tensor_path = rx_npy_path(block_dir, expected_rx)
            try:
                tensor = np.load(tensor_path, mmap_mode="r", allow_pickle=False)
                if tensor.shape != (nz, ny, nx) or tensor.dtype != np.dtype("float32"):
                    raise ValueError(f"shape={tensor.shape}, dtype={tensor.dtype}")
                if not np.isfinite(tensor).all():
                    raise ValueError("contains non-finite RSS values")
                result_values += int(tensor.size)
            except (OSError, ValueError) as exc:
                errors.append(f"invalid result tensor {tensor_path.name}: {exc}")
                break
        checks["result_values_expected"] = result_values == expected_values
        checks["all_rss_finite"] = result_values == expected_values

    errors.extend(name for name, passed in checks.items() if not passed)
    return {
        "region_slug": block.region_slug,
        "block_id": block.block_id,
        "status": "passed" if not errors else "failed",
        "checks": checks,
        "result_values": result_values if verify_results else None,
        "minimum_actual_building_clearance_m": (
            None if all(math.isinf(value) for value in actual_clearances)
            else min(actual_clearances)
        ),
        "errors": errors,
    }


def load_manifest(path: Path) -> list[BlockSpec]:
    valid_fields = {field.name for field in fields(BlockSpec)}
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    blocks: list[BlockSpec] = []
    for row in rows:
        values: dict[str, object] = {key: value for key, value in row.items() if key in valid_fields}
        for key in ("center_lat", "center_lon", "min_lat", "min_lon", "max_lat", "max_lon", "size_m"):
            values[key] = float(values[key])
        blocks.append(BlockSpec(**values))
    return blocks


def completed_block_profile(dataset_root: Path, block: BlockSpec) -> dict[str, object] | None:
    """Return the hardware-independent settings that must match after merging."""
    metadata_path = dataset_root / block.region_slug / block.block_id / "metadata.json"
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        return None
    return {
        "schema_version": metadata.get("schema_version"),
        "map_block_size_m": metadata.get("map_block_size_m"),
        "layout": metadata.get("layout"),
        "ground_rx_count": metadata.get("ground_rx_count"),
        "ground_rx_height_m": metadata.get("ground_rx_height_m"),
        "random_seed_global": metadata.get("random_seed_global"),
        "minimum_building_clearance_m": metadata.get("minimum_building_clearance_m"),
        "tx_voxel_shape_xyz": metadata.get("tx_voxel_shape_xyz"),
        "voxel_size_m": metadata.get("voxel_size_m"),
        "voxel_z_start_m": metadata.get("voxel_z_start_m"),
        "osm_height_variant": metadata.get("osm_height_variant"),
        "frequency_hz": metadata.get("frequency_hz"),
        "receiver": metadata.get("receiver"),
        "transmitter": metadata.get("transmitter"),
        "materials": metadata.get("materials"),
        "rt": metadata.get("rt"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--quick", action="store_true", help="Skip the million-row combination scan.")
    parser.add_argument(
        "--region",
        action="append",
        choices=[region.slug for region in REGIONS],
        help="Audit only one or more selected regions; may be repeated.",
    )
    parser.add_argument(
        "--block",
        action="append",
        help="Audit only one or more selected block IDs; may be repeated.",
    )
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve()
    blocks = load_manifest(dataset_root / "blocks_manifest.csv")
    selected_regions = set(args.region or [])
    if selected_regions:
        blocks = [block for block in blocks if block.region_slug in selected_regions]
    selected_blocks = set(args.block or [])
    if selected_blocks:
        blocks = [block for block in blocks if block.block_id in selected_blocks]
    records: list[dict[str, object]] = []
    for index, block in enumerate(blocks, start=1):
        print(f"[{index}/{len(blocks)}] audit {block.region_slug}/{block.block_id}")
        records.append(audit_aerial_tx_voxel_block(dataset_root, block, not args.quick))
    passed = sum(record["status"] == "passed" for record in records)
    profile_blocks: dict[str, list[str]] = {}
    for block in blocks:
        profile = completed_block_profile(dataset_root, block)
        if profile is None:
            continue
        key = json.dumps(profile, ensure_ascii=False, sort_keys=True)
        profile_blocks.setdefault(key, []).append(f"{block.region_slug}/{block.block_id}")
    configurations_match = len(profile_blocks) <= 1
    report = {
        "audited_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_root": str(dataset_root),
        "verify_all_result_values": not args.quick,
        "block_count": len(blocks),
        "passed_blocks": passed,
        "failed_blocks": len(blocks) - passed,
        "hardware_independent_configuration_profiles": [
            {"settings": json.loads(key), "block_count": len(value), "first_block": value[0]}
            for key, value in profile_blocks.items()
        ],
        "all_completed_block_configurations_match": configurations_match,
        "records": records,
    }
    (dataset_root / "AUDIT_REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 数据集验收报告",
        "",
        f"- 数据集：`{dataset_root}`",
        f"- 分块数：{len(blocks)}",
        f"- 通过：{passed}",
        f"- 失败：{len(blocks) - passed}",
        f"- 完成区块参数一致：{configurations_match}",
        f"- 参数配置种类：{len(profile_blocks)}",
        f"- 已逐行检查百万行组合：{not args.quick}",
        "",
    ]
    for record in records:
        errors = ", ".join(record.get("errors", [])) or "无"
        lines.append(f"- {record['region_slug']}/{record['block_id']}: {record['status']}；错误：{errors}")
    text = "\n".join(lines) + "\n"
    (dataset_root / "AUDIT_REPORT.md").write_text(text, encoding="utf-8")
    Path("CAMPUS_SIONNA_DATASET_AUDIT.md").write_text(text, encoding="utf-8")
    print(
        f"Audit complete: {passed}/{len(blocks)} blocks passed; "
        f"configuration_profiles={len(profile_blocks)}"
    )
    return 0 if passed == len(blocks) and configurations_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
