"""Inspect Nominatim candidates while curating campus boundaries."""

from __future__ import annotations

import argparse
import json
import urllib.parse

from campus_sionna_dataset import http_get


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="+")
    args = parser.parse_args()
    for search_text in args.query:
        query = urllib.parse.urlencode({
            "format": "jsonv2",
            "q": search_text,
            "polygon_geojson": 0,
            "addressdetails": 1,
            "countrycodes": "cn",
            "limit": 10,
        })
        payload = json.loads(http_get(f"https://nominatim.openstreetmap.org/search?{query}").decode("utf-8"))
        print(f"QUERY: {search_text}")
        for index, record in enumerate(payload, start=1):
            print(
                f"  {index}. {record.get('osm_type')} {record.get('osm_id')} "
                f"{record.get('class')}/{record.get('type')} {record.get('display_name')}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
