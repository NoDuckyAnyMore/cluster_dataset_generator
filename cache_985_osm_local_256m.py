#!/usr/bin/env python3
"""Download and verify 256 m OSM blocks for all 39 Project-985 main campuses."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from campus_sionna_dataset import MAP_BLOCK_SIZE_M, PROJECT_985_REGIONS


SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_ROOT = SCRIPT_DIR / "offline_osm_cache_985_256m"
GENERATOR = SCRIPT_DIR / "campus_sionna_dataset.py"
REPORT_PATH = CACHE_ROOT / "OFFLINE_CACHE_REPORT.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_cache() -> dict[str, object]:
    manifest_path = CACHE_ROOT / "blocks_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    counts = Counter(row["region_slug"] for row in rows)
    expected_slugs = {region.slug for region in PROJECT_985_REGIONS}
    missing_regions = sorted(expected_slugs - set(counts))
    unexpected_regions = sorted(set(counts) - expected_slugs)
    missing_files: list[str] = []
    invalid_osm_files: list[str] = []
    wrong_block_sizes: list[str] = []
    for row in rows:
        relative = Path(row["region_slug"]) / row["block_id"] / "osm_map.osm"
        path = CACHE_ROOT / relative
        if not path.is_file():
            missing_files.append(relative.as_posix())
        else:
            try:
                if ET.parse(path).getroot().tag != "osm":
                    invalid_osm_files.append(relative.as_posix())
            except (OSError, ET.ParseError):
                invalid_osm_files.append(relative.as_posix())
        if abs(float(row["size_m"]) - MAP_BLOCK_SIZE_M) > 1e-9:
            wrong_block_sizes.append(f"{row['region_slug']}/{row['block_id']}")

    boundaries: list[dict[str, object]] = []
    for region in PROJECT_985_REGIONS:
        path = CACHE_ROOT / region.slug / "region_boundary.geojson"
        if not path.is_file():
            missing_files.append(f"{region.slug}/region_boundary.geojson")
            continue
        feature = json.loads(path.read_text(encoding="utf-8"))
        properties = feature.get("properties", {})
        boundaries.append({
            "region_slug": region.slug,
            "name_zh": region.name_zh,
            "block_count": counts.get(region.slug, 0),
            "boundary_source": properties.get("boundary_source"),
            "display_name": properties.get("display_name"),
            "osm_type": properties.get("osm_type"),
            "osm_id": properties.get("osm_id"),
        })
    cache_files = [
        path for path in CACHE_ROOT.rglob("*")
        if path.is_file() and path != REPORT_PATH
    ]
    report: dict[str, object] = {
        "schema_version": "2.0",
        "verified_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cache_root": str(CACHE_ROOT),
        "map_block_size_m": MAP_BLOCK_SIZE_M,
        "project_985_main_campus_count": len(PROJECT_985_REGIONS),
        "block_count": len(rows),
        "region_block_counts": dict(sorted(counts.items())),
        "file_count_excluding_report": len(cache_files),
        "cache_bytes_excluding_report": sum(path.stat().st_size for path in cache_files),
        "manifest_sha256": sha256_file(manifest_path),
        "missing_regions": missing_regions,
        "unexpected_regions": unexpected_regions,
        "missing_files": missing_files,
        "invalid_osm_files": invalid_osm_files,
        "wrong_block_sizes": wrong_block_sizes,
        "satellite_imagery": "skipped",
        "boundaries": boundaries,
    }
    report["valid"] = not any((
        missing_regions, unexpected_regions, missing_files,
        invalid_osm_files, wrong_block_sizes,
    ))
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> int:
    if len(PROJECT_985_REGIONS) != 39:
        raise RuntimeError(f"Expected 39 Project-985 campuses, found {len(PROJECT_985_REGIONS)}")
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Local cache directory: {CACHE_ROOT}", flush=True)
    print("Downloading OSM only; satellite imagery and Sionna RT are skipped.", flush=True)
    command = [
        sys.executable,
        str(GENERATOR),
        "--dataset-root", str(CACHE_ROOT),
        "--prefetch-only",
        "--log-name", "local_cache_download.log",
    ]
    for region in PROJECT_985_REGIONS:
        command.extend(["--region", region.slug])
    environment = os.environ.copy()
    environment["RID_PREFER_OSM_MAP_API"] = "1"
    environment["RID_OSM_DOWNLOAD_WORKERS"] = "4"
    environment["PYTHONUNBUFFERED"] = "1"
    subprocess.run(command, cwd=SCRIPT_DIR, env=environment, check=True)
    report = validate_cache()
    print(
        f"CACHE VERIFIED: valid={report['valid']}, campuses=39, "
        f"blocks={report['block_count']}, bytes={report['cache_bytes_excluding_report']}",
        flush=True,
    )
    print(f"Verification report: {REPORT_PATH}", flush=True)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
