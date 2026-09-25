"""Create an OSM-only cache with deterministic synthetic building heights.

Only buildings that currently fall through to the generator's 12 m default
(no valid ``height``, ``building:levels``, or ``levels``) are modified. Every
eligible building receives ``12 m + U[-2, 20] m``, i.e. 10--32 m. A way
crossing block boundaries receives the same deterministic height everywhere.

Synthetic heights are positive and intentionally marked as synthetic. They are
useful for simulation diversity, not as measured or OSM ground truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import xml.etree.ElementTree as ET
from pathlib import Path

from campus_sionna_dataset import parse_meters


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = SCRIPT_DIR / "offline_osm_cache_shenzhen_256m"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "osm_randomized_height_shenzhen_256m_u10_32"
DEFAULT_SEED = 20260911
DEFAULT_RANDOMIZE_FRACTION = 1.0
ORIGINAL_DEFAULT_HEIGHT_M = 12.0
MIN_HEIGHT_OFFSET_M = -2.0
MAX_HEIGHT_OFFSET_M = 20.0


def stable_integer(seed: int, way_id: str, purpose: str) -> int:
    payload = f"{seed}|{way_id}|{purpose}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def building_tags(way: ET.Element) -> dict[str, str]:
    return {
        tag.attrib.get("k", ""): tag.attrib.get("v", "")
        for tag in way.findall("tag")
    }


def is_building(tags: dict[str, str]) -> bool:
    return tags.get("building", "no") != "no"


def uses_default_height(tags: dict[str, str]) -> bool:
    height = parse_meters(tags.get("height"))
    if height is not None and height > 0:
        return False
    levels = parse_meters(tags.get("building:levels")) or parse_meters(tags.get("levels"))
    return levels is None or levels <= 0


def synthetic_height_m(seed: int, way_id: str) -> float:
    """Return the deterministic 12 m + U[-2, 20] m synthetic height."""
    rng = random.Random(stable_integer(seed, way_id, "height"))
    value = ORIGINAL_DEFAULT_HEIGHT_M + rng.uniform(
        MIN_HEIGHT_OFFSET_M, MAX_HEIGHT_OFFSET_M
    )
    return round(value, 1)


def manifest_osm_paths(input_root: Path) -> list[tuple[str, str, Path]]:
    manifest = input_root / "blocks_manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest}")
    with manifest.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    paths = [
        (
            row["region_slug"],
            row["block_id"],
            input_root / row["region_slug"] / row["block_id"] / "osm_map.osm",
        )
        for row in rows
    ]
    missing = [str(path) for _, _, path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} input OSM files; first missing file: {missing[0]}"
        )
    return paths


def collect_default_way_ids(paths: list[tuple[str, str, Path]]) -> set[str]:
    default_ids: set[str] = set()
    for _, _, path in paths:
        root = ET.parse(path).getroot()
        for way in root.findall("way"):
            tags = building_tags(way)
            if is_building(tags) and uses_default_height(tags):
                default_ids.add(way.attrib["id"])
    return default_ids


def select_exact_fraction(
    way_ids: set[str], seed: int, fraction: float
) -> set[str]:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("randomize fraction must be between 0 and 1")
    count = round(len(way_ids) * fraction)
    ranked = sorted(
        way_ids,
        key=lambda way_id: (stable_integer(seed, way_id, "selection"), way_id),
    )
    return set(ranked[:count])


def set_or_append_tag(way: ET.Element, key: str, value: str) -> None:
    matching = [tag for tag in way.findall("tag") if tag.attrib.get("k") == key]
    if matching:
        matching[-1].set("v", value)
        for duplicate in matching[:-1]:
            way.remove(duplicate)
    else:
        ET.SubElement(way, "tag", {"k": key, "v": value})


def write_modified_osm(
    source: Path,
    target: Path,
    selected_ids: set[str],
    seed: int,
) -> tuple[int, int]:
    tree = ET.parse(source)
    changed_occurrences = 0
    eligible_occurrences = 0
    for way in tree.getroot().findall("way"):
        tags = building_tags(way)
        if not is_building(tags) or not uses_default_height(tags):
            continue
        eligible_occurrences += 1
        way_id = way.attrib["id"]
        if way_id not in selected_ids:
            continue
        height = synthetic_height_m(seed, way_id)
        set_or_append_tag(way, "height", f"{height:.1f}")
        set_or_append_tag(way, "rid:height_source", "synthetic_12m_plus_uniform_m2_20_v2")
        set_or_append_tag(way, "rid:original_default_height_m", f"{ORIGINAL_DEFAULT_HEIGHT_M:.1f}")
        set_or_append_tag(way, "rid:synthetic_height_seed", str(seed))
        changed_occurrences += 1

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    ET.indent(tree, space="  ")
    tree.write(temporary, encoding="utf-8", xml_declaration=True)
    temporary.replace(target)
    return eligible_occurrences, changed_occurrences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--fraction", type=float, default=DEFAULT_RANDOMIZE_FRACTION,
        help="Exact fraction of unique default-height building IDs to randomize.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if input_root == output_root:
        raise ValueError("output-root must differ from input-root")

    paths = manifest_osm_paths(input_root)
    default_ids = collect_default_way_ids(paths)
    selected_ids = select_exact_fraction(default_ids, args.seed, args.fraction)
    heights = [synthetic_height_m(args.seed, way_id) for way_id in selected_ids]
    print(f"Input OSM files: {len(paths)}")
    print(f"Unique default-height buildings: {len(default_ids)}")
    print(
        f"Selected unique buildings: {len(selected_ids)} "
        f"({100.0 * len(selected_ids) / max(len(default_ids), 1):.3f}%)"
    )
    print(
        f"Synthetic height range: {min(heights):.1f}--{max(heights):.1f} m; "
        f"mean={sum(heights) / len(heights):.2f} m"
    )
    print(f"Output root: {output_root}")

    eligible_occurrences = 0
    changed_occurrences = 0
    for index, (region, block, source) in enumerate(paths, start=1):
        target = output_root / region / block / "osm_map.osm"
        eligible, changed = write_modified_osm(
            source, target, selected_ids, args.seed
        )
        eligible_occurrences += eligible
        changed_occurrences += changed
        if index == 1 or index % 50 == 0 or index == len(paths):
            print(
                f"[{index}/{len(paths)}] {region}/{block}: "
                f"eligible={eligible_occurrences}, changed={changed_occurrences}",
                flush=True,
            )

    output_files = list(output_root.rglob("osm_map.osm"))
    other_files = [path for path in output_root.rglob("*") if path.is_file() and path.name != "osm_map.osm"]
    if len(output_files) != len(paths):
        raise RuntimeError(
            f"Output verification failed: expected {len(paths)} OSM files, found {len(output_files)}"
        )
    if other_files:
        raise RuntimeError(f"Output contains unexpected non-OSM file: {other_files[0]}")
    print(
        f"COMPLETE: wrote {len(output_files)} OSM-only block files; "
        f"changed {changed_occurrences}/{eligible_occurrences} default-height occurrences."
    )
    print("No satellite imagery was copied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
