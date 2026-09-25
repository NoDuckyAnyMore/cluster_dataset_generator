"""Redraw every existing block's OSM 3D figure without running Sionna RT."""

from __future__ import annotations

from pathlib import Path

from campus_sionna_dataset import (
    DEFAULT_DATASET_ROOT,
    load_blocks_manifest,
    parse_block_buildings,
    plot_buildings_3d,
)


# Edit here only if the dataset was moved to a different directory.
DATASET_ROOT = DEFAULT_DATASET_ROOT


def main() -> int:
    dataset_root = Path(DATASET_ROOT).resolve()
    blocks = load_blocks_manifest(dataset_root / "blocks_manifest.csv")
    redrawn = 0
    missing = 0
    failed: list[str] = []

    print(f"Dataset root: {dataset_root}")
    print("Mode: redraw OSM 3D PNG only; Sionna RT and result CSV files are untouched.")
    for index, block in enumerate(blocks, start=1):
        block_key = f"{block.region_slug}/{block.block_id}"
        block_dir = dataset_root / block.region_slug / block.block_id
        osm_path = block_dir / "osm_map.osm"
        output_path = block_dir / "osm_buildings_3d.png"
        if not osm_path.is_file() or osm_path.stat().st_size == 0:
            missing += 1
            print(f"[{index}/{len(blocks)}] SKIP missing OSM: {block_key}")
            continue
        try:
            buildings = parse_block_buildings(osm_path.read_bytes(), block)
            plot_buildings_3d(block, buildings, output_path)
        except Exception as exc:  # Continue so one malformed block does not hide the rest.
            failed.append(f"{block_key}: {exc}")
            print(f"[{index}/{len(blocks)}] ERROR {block_key}: {exc}")
            continue
        redrawn += 1
        print(
            f"[{index}/{len(blocks)}] REDRAWN {block_key}: "
            f"{len(buildings)} buildings -> {output_path.name}"
        )

    print(f"REDRAW COMPLETE: {redrawn} written, {missing} missing OSM, {len(failed)} failed")
    if failed:
        print("Failures:")
        for message in failed:
            print(f"  {message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
