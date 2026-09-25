"""Print cached campus boundary selections and approximate metric extents."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from campus_sionna_dataset import REGIONS, outer_rings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(__file__).resolve().parent / "offline_osm_cache_shenzhen_256m",
    )
    args = parser.parse_args()
    for region in REGIONS:
        path = args.cache_root / region.slug / "region_boundary.geojson"
        if not path.is_file():
            print(f"{region.slug:18} MISSING")
            continue
        feature = json.loads(path.read_text(encoding="utf-8"))
        rings = outer_rings(feature["geometry"])
        longitudes = [point[0] for ring in rings for point in ring]
        latitudes = [point[1] for ring in rings for point in ring]
        mean_latitude = sum(latitudes) / len(latitudes)
        width_m = (max(longitudes) - min(longitudes)) * 111_320.0 * math.cos(math.radians(mean_latitude))
        height_m = (max(latitudes) - min(latitudes)) * 110_540.0
        properties = feature.get("properties", {})
        print(
            f"{region.slug:18} {width_m:6.0f} x {height_m:6.0f} m  "
            f"{str(properties.get('boundary_source')):32} "
            f"{properties.get('display_name')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
