"""Build a resumable OSM/Sionna RT dataset for Chinese university campuses.

Dataset layout::

    DATASET_ROOT/
      dataset_metadata.json
      blocks_manifest.csv
      <region_slug>/
        region_boundary.geojson
        <center_lat>_<center_lon>/
          metadata.json
          osm_map.osm
          osm_buildings_2d.png
          osm_buildings_3d.png
          scene_generated/...
          rx_positions.csv
          sionna_results_by_rx/rx_000.npy ...

Every map/building block is exactly 256 m x 256 m. A 128 x 128 x 40 aerial
transmitter-voxel grid is observed by an extensible deterministic random ground
receivers. Voxels have 2 m edges and TX heights are 2, 4, ..., 80 m.
Both TX/RX arrays use Sionna's ideal ``iso`` pattern with zero extra gain.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import json
import math
import os
import platform
import re
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from rt_osm_scene import local_m_to_lonlat, lonlat_to_local_m, write_sionna_scene
from rt_solver import (
    ENABLE_TIMING_LOGS,
    Building,
    GroundTxPoint,
    GroundRxPoint,
    MeshTxPoint,
    RtExperimentConfig,
    VoxelRxPoint,
    run_sionna_rt_multi_rx,
    run_sionna_rt_multi_rx_npy,
    run_sionna_rt_voxel_rx,
    write_dict_csv,
)
from render_voxel_channel_maps import EXPECTED_IMAGES as RSS_PREVIEW_IMAGE_COUNT
from render_voxel_channel_maps import OUTPUT_DIR_NAME as RSS_PREVIEW_DIR_NAME
from render_rx_sharded_channel_maps import render_block as render_voxel_rss_preview
from rx_sharded_results import (
    RESULTS_DIR_NAME,
    aggregate_height_summaries,
    commit_rx_group,
    completed_rx_indices,
    desired_rx_signature,
    group_checkpoint_path,
    load_valid_rx_entries,
    results_dir as sharded_results_dir,
)


DATASET_NAME = "shenzhen_campus_sionna_voxel_256m_128x128x40_rx10_rand10to32"
DEFAULT_DATASET_ROOT = Path("/home/lasso/workspace/wxy") / DATASET_NAME
MAP_BLOCK_SIZE_M = 256.0
TX_MESH_SIZE_M = MAP_BLOCK_SIZE_M
# Compatibility alias for helper scripts that interpret this as map size.
BLOCK_SIZE_M = MAP_BLOCK_SIZE_M
DEFAULT_RX_COUNT = 10
DEFAULT_RX_SOLVER_GROUP_SIZE = 10
DEFAULT_RX_HEIGHT_M = 0.5
DEFAULT_RANDOM_SEED = 20260907
DEFAULT_OSM_HEIGHT_VARIANT = "unspecified"
MIN_BUILDING_CLEARANCE_M = 5.0
DEFAULT_VOXEL_SIZE_M = 2.0
DEFAULT_VOXEL_NX = 128
DEFAULT_VOXEL_NY = 128
DEFAULT_VOXEL_NZ = 40
DEFAULT_VOXEL_Z_START_M = 2.0
DEFAULT_TX_BATCH_SIZE = 10
# Kept for legacy helper functions that are no longer used by the voxel run.
DEFAULT_TX_COUNT = 10
DEFAULT_TX_HEIGHT_M = DEFAULT_RX_HEIGHT_M
DEFAULT_MESH_SPACING_M = DEFAULT_VOXEL_SIZE_M
SATELLITE_IMAGE_PIXELS = 1024
SATELLITE_CACHE_FILENAME = "satellite_imagery.png"
SATELLITE_RENDERED_FILENAME = "satellite_topdown.png"
# Satellite imagery is not required by Sionna or model training. Keep its
# helpers for optional future use, but skip every satellite network request in
# the default dataset pipeline so an imagery-service failure cannot block RT.
ENABLE_SATELLITE_IMAGERY = False
SATELLITE_EXPORT_ENDPOINT = (
    "https://services.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/export"
)
SATELLITE_SERVICE_URL = (
    "https://services.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer"
)
SATELLITE_ATTRIBUTION = "Esri World Imagery — © Esri and imagery providers"
USER_AGENT = "OSM Campus Sionna Dataset Builder/1.0 (research dataset)"
HTTP_MAX_ATTEMPTS = 6
HTTP_RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}
OVERPASS_ENDPOINTS = (
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass-api.de/api/interpreter",
)
CHINA_STANDARD_TIME = timezone(timedelta(hours=8), name="Asia/Shanghai")
_OPEN_LOG_FILES: list[object] = []
# The Ubuntu server may not have a route to api.openstreetmap.org. Prefer the
# bundled offline cache, then go directly to the alternative Overpass list.
_OSM_MAP_API_AVAILABLE: bool | None = (
    None if os.environ.get("RID_PREFER_OSM_MAP_API") == "1" else False
)


class TeeTextStream:
    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file
        self.log_line_start = True

    def write(self, text: str) -> int:
        result = self.terminal.write(text)
        for part in text.splitlines(keepends=True):
            if self.log_line_start and part not in ("\n", "\r\n"):
                self.log_file.write(f"[{local_now()}] ")
            self.log_file.write(part)
            self.log_line_start = part.endswith("\n") or part.endswith("\r")
        self.log_file.flush()
        return result

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.terminal, "isatty", lambda: False)())

    def __getattr__(self, name: str):
        return getattr(self.terminal, name)


def enable_persistent_logging(dataset_root: Path, argv: list[str], log_name: str = "generation.log") -> Path:
    if Path(log_name).name != log_name:
        raise ValueError("--log-name must be a plain filename")
    log_path = dataset_root / log_name
    log_file = log_path.open("a", encoding="utf-8", buffering=1)
    _OPEN_LOG_FILES.append(log_file)
    sys.stdout = TeeTextStream(sys.stdout, log_file)
    sys.stderr = TeeTextStream(sys.stderr, log_file)
    print("\n" + "=" * 88)
    print("RUN START")
    print("COMMAND " + " ".join(argv))
    return log_path


@dataclass(frozen=True)
class RegionSpec:
    slug: str
    name_zh: str
    name_en: str
    osm_type: str | None
    osm_id: int | None
    official_or_reference_url: str
    nominatim_query: str = ""
    is_985_main_campus: bool = False
    fallback_boundary_size_m: float = 1600.0

    @property
    def osm_key(self) -> str:
        if not self.osm_type or self.osm_id is None:
            raise ValueError(f"{self.slug} has no fixed OSM object ID")
        prefix = "R" if self.osm_type == "relation" else "W"
        return f"{prefix}{self.osm_id}"


SHENZHEN_REGIONS = [
    RegionSpec("sustech", "南方科技大学", "Southern University of Science and Technology", "way", 456349157, "https://sustech.edu.cn/zh/contact_us.html"),
    RegionSpec("cuhk_shenzhen", "香港中文大学（深圳）", "The Chinese University of Hong Kong, Shenzhen", "relation", 10129044, "https://www.cuhk.edu.cn/zh-hans/campus-life"),
    RegionSpec("hit_shenzhen", "哈尔滨工业大学（深圳）", "Harbin Institute of Technology, Shenzhen", "way", 319515452, "https://www.hitsz.edu.cn/"),
    RegionSpec("pku_shenzhen", "北京大学深圳研究生院", "Peking University Shenzhen Graduate School", "relation", 15626536, "https://www.pkusz.edu.cn/"),
    RegionSpec("tsinghua_shenzhen", "清华大学深圳国际研究生院", "Tsinghua Shenzhen International Graduate School", "relation", 15626537, "https://www.sigs.tsinghua.edu.cn/"),
    RegionSpec("szu_yuehai", "深圳大学（粤海校区）", "Shenzhen University, Yuehai Campus", "relation", 13900447, "https://www.szu.edu.cn/xxgk/xydt.htm"),
    RegionSpec("szu_lihu", "深圳大学（丽湖校区）", "Shenzhen University, Lihu Campus", "way", 480227299, "https://lihu.szu.edu.cn/"),
]

PROJECT_985_REFERENCE_URL = (
    "https://www.moe.gov.cn/srcsite/A22/s7065/200612/t20061206_128833.html"
)
PROJECT_985_REGIONS = [
    RegionSpec("pku_main", "北京大学", "Peking University", "way", 1330709889, PROJECT_985_REFERENCE_URL, "", True),
    RegionSpec("tsinghua_main", "清华大学", "Tsinghua University", None, None, PROJECT_985_REFERENCE_URL, "清华大学 海淀区 北京市 中国", True),
    RegionSpec("ruc", "中国人民大学", "Renmin University of China", None, None, PROJECT_985_REFERENCE_URL, "中国人民大学 中关村校区 北京市 中国", True),
    RegionSpec("buaa", "北京航空航天大学", "Beihang University", None, None, PROJECT_985_REFERENCE_URL, "北京航空航天大学 学院路校区 北京市 中国", True),
    RegionSpec("bit", "北京理工大学", "Beijing Institute of Technology", None, None, PROJECT_985_REFERENCE_URL, "北京理工大学 中关村校区 北京市 中国", True),
    RegionSpec("cau", "中国农业大学", "China Agricultural University", None, None, PROJECT_985_REFERENCE_URL, "中国农业大学 东校区 北京市 中国", True),
    RegionSpec("bnu", "北京师范大学", "Beijing Normal University", None, None, PROJECT_985_REFERENCE_URL, "北京师范大学 海淀校园 北京市 中国", True),
    RegionSpec("minzu", "中央民族大学", "Minzu University of China", None, None, PROJECT_985_REFERENCE_URL, "中央民族大学 海淀校区 北京市 中国", True),
    RegionSpec("nankai", "南开大学", "Nankai University", None, None, PROJECT_985_REFERENCE_URL, "南开大学 八里台校区 天津市 中国", True),
    RegionSpec("tju", "天津大学", "Tianjin University", None, None, PROJECT_985_REFERENCE_URL, "天津大学 卫津路校区 天津市 中国", True),
    RegionSpec("dlut", "大连理工大学", "Dalian University of Technology", None, None, PROJECT_985_REFERENCE_URL, "大连理工大学 凌水校区 大连市 中国", True),
    RegionSpec("neu", "东北大学", "Northeastern University", None, None, PROJECT_985_REFERENCE_URL, "东北大学 南湖校区 沈阳市 中国", True),
    RegionSpec("jlu", "吉林大学", "Jilin University", None, None, PROJECT_985_REFERENCE_URL, "吉林大学 前卫南区 长春市 中国", True),
    RegionSpec("hit_main", "哈尔滨工业大学", "Harbin Institute of Technology", None, None, PROJECT_985_REFERENCE_URL, "哈尔滨工业大学 一校区 哈尔滨市 中国", True),
    RegionSpec("fudan", "复旦大学", "Fudan University", None, None, PROJECT_985_REFERENCE_URL, "复旦大学 邯郸校区 上海市 中国", True),
    RegionSpec("tongji", "同济大学", "Tongji University", "relation", 18788116, PROJECT_985_REFERENCE_URL, "", True),
    RegionSpec("sjtu", "上海交通大学", "Shanghai Jiao Tong University", None, None, PROJECT_985_REFERENCE_URL, "上海交通大学 闵行校区 上海市 中国", True),
    RegionSpec("ecnu", "华东师范大学", "East China Normal University", None, None, PROJECT_985_REFERENCE_URL, "华东师范大学 闵行校区 上海市 中国", True),
    RegionSpec("nju", "南京大学", "Nanjing University", None, None, PROJECT_985_REFERENCE_URL, "南京大学 仙林校区 南京市 中国", True),
    RegionSpec("seu", "东南大学", "Southeast University", None, None, PROJECT_985_REFERENCE_URL, "东南大学 九龙湖校区 南京市 中国", True),
    RegionSpec("zju", "浙江大学", "Zhejiang University", None, None, PROJECT_985_REFERENCE_URL, "浙江大学 紫金港校区 杭州市 中国", True),
    RegionSpec("ustc", "中国科学技术大学", "University of Science and Technology of China", None, None, PROJECT_985_REFERENCE_URL, "中国科学技术大学 东校区 合肥市 中国", True),
    RegionSpec("xmu", "厦门大学", "Xiamen University", "way", 154986493, PROJECT_985_REFERENCE_URL, "", True),
    RegionSpec("sdu", "山东大学", "Shandong University", None, None, PROJECT_985_REFERENCE_URL, "山东大学 中心校区 济南市 中国", True),
    RegionSpec("ouc", "中国海洋大学", "Ocean University of China", None, None, PROJECT_985_REFERENCE_URL, "中国海洋大学 崂山校区 青岛市 中国", True),
    RegionSpec("whu", "武汉大学", "Wuhan University", "relation", 10717504, PROJECT_985_REFERENCE_URL, "", True),
    RegionSpec("hust", "华中科技大学", "Huazhong University of Science and Technology", None, None, PROJECT_985_REFERENCE_URL, "华中科技大学 主校区 武汉市 中国", True),
    RegionSpec("hnu", "湖南大学", "Hunan University", None, None, PROJECT_985_REFERENCE_URL, "湖南大学 岳麓区 长沙市 中国", True),
    RegionSpec("csu", "中南大学", "Central South University", None, None, PROJECT_985_REFERENCE_URL, "中南大学 新校区 长沙市 中国", True),
    RegionSpec("sysu", "中山大学", "Sun Yat-sen University", None, None, PROJECT_985_REFERENCE_URL, "中山大学 广州校区南校园 广州市 中国", True),
    RegionSpec("scut", "华南理工大学", "South China University of Technology", None, None, PROJECT_985_REFERENCE_URL, "华南理工大学 五山校区 广州市 中国", True),
    RegionSpec("scu", "四川大学", "Sichuan University", None, None, PROJECT_985_REFERENCE_URL, "四川大学 望江校区 成都市 中国", True),
    RegionSpec("cqu", "重庆大学", "Chongqing University", None, None, PROJECT_985_REFERENCE_URL, "重庆大学 A区 重庆市 中国", True),
    RegionSpec("uestc", "电子科技大学", "University of Electronic Science and Technology of China", None, None, PROJECT_985_REFERENCE_URL, "电子科技大学 清水河校区 成都市 中国", True),
    RegionSpec("xjtu", "西安交通大学", "Xi'an Jiaotong University", None, None, PROJECT_985_REFERENCE_URL, "西安交通大学 兴庆校区 西安市 中国", True),
    RegionSpec("nwpu", "西北工业大学", "Northwestern Polytechnical University", None, None, PROJECT_985_REFERENCE_URL, "西北工业大学 长安校区 西安市 中国", True),
    RegionSpec("nwafu", "西北农林科技大学", "Northwest A&F University", "way", 375646584, PROJECT_985_REFERENCE_URL, "", True),
    RegionSpec("lzu", "兰州大学", "Lanzhou University", "way", 464554875, PROJECT_985_REFERENCE_URL, "", True),
    RegionSpec("nudt", "国防科技大学", "National University of Defense Technology", None, None, PROJECT_985_REFERENCE_URL, "国防科技大学 开福区 长沙市 中国", True),
]

# Keep one CLI/catalogue so a prepared manifest from either cache can be read.
# Cache builders explicitly select their own catalogue and never mix datasets.
REGIONS = [*SHENZHEN_REGIONS, *PROJECT_985_REGIONS]


@dataclass
class BlockSpec:
    region_slug: str
    region_name_zh: str
    block_id: str
    center_lat: float
    center_lon: float
    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float
    size_m: float = MAP_BLOCK_SIZE_M


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def local_now() -> str:
    return datetime.now(CHINA_STANDARD_TIME).isoformat(timespec="seconds")


def http_get(url: str, timeout: int = 90, max_attempts: int = HTTP_MAX_ATTEMPTS) -> bytes:
    """Download bytes with bounded exponential backoff for transient failures."""
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in HTTP_RETRYABLE_STATUS_CODES:
                raise
            last_error = exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.IncompleteRead) as exc:
            last_error = exc

        if attempt < max_attempts:
            delay_seconds = min(30, 2 ** attempt)
            print(
                f"DOWNLOAD RETRY {attempt}/{max_attempts}: {last_error}; "
                f"waiting {delay_seconds} s"
            )
            time.sleep(delay_seconds)

    raise RuntimeError(f"Download failed after {max_attempts} attempts: {url}") from last_error


def fetch_region_boundary(region: RegionSpec, cache_path: Path) -> dict[str, object]:
    if cache_path.is_file():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    query_used = region.nominatim_query
    if region.osm_type and region.osm_id is not None:
        query = urllib.parse.urlencode({
            "format": "jsonv2",
            "osm_ids": region.osm_key,
            "polygon_geojson": 1,
        })
        payload = json.loads(http_get(
            f"https://nominatim.openstreetmap.org/lookup?{query}"
        ).decode("utf-8"))
        time.sleep(1.1)
        candidates = [record for record in payload if record.get("geojson", {}).get("type") in {"Polygon", "MultiPolygon"}]
        boundary_source = "fixed_osm_id"
    else:
        search_attempts = list(dict.fromkeys(filter(None, (
            region.nominatim_query,
            f"{region.name_zh} 中国",
            f"{region.name_en} China",
        ))))
        payload: list[dict[str, object]] = []
        candidates: list[dict[str, object]] = []
        first_non_polygon: dict[str, object] | None = None
        for search_text in search_attempts:
            query = urllib.parse.urlencode({
                "format": "jsonv2",
                "q": search_text,
                "polygon_geojson": 1,
                "addressdetails": 1,
                "countrycodes": "cn",
                "limit": 10,
            })
            payload = json.loads(http_get(
                f"https://nominatim.openstreetmap.org/search?{query}"
            ).decode("utf-8"))
            time.sleep(1.1)
            if payload and first_non_polygon is None:
                first_non_polygon = payload[0]
            candidates = [
                record for record in payload
                if record.get("geojson", {}).get("type") in {"Polygon", "MultiPolygon"}
            ]
            if candidates:
                query_used = search_text
                break
        if not candidates and first_non_polygon is not None:
            payload = [first_non_polygon]
        boundary_source = "nominatim_search_polygon"
    selected = candidates[0] if candidates else (payload[0] if payload else None)
    if selected is None:
        raise RuntimeError(
            f"Nominatim returned no result for {region.slug}: "
            f"{region.nominatim_query or region.osm_key}"
        )
    geometry = selected.get("geojson")
    if not isinstance(geometry, dict) or geometry.get("type") not in {"Polygon", "MultiPolygon"}:
        center_lat = float(selected["lat"])
        center_lon = float(selected["lon"])
        half = float(region.fallback_boundary_size_m) / 2.0
        corners = [
            local_m_to_lonlat(-half, -half, center_lon, center_lat),
            local_m_to_lonlat(half, -half, center_lon, center_lat),
            local_m_to_lonlat(half, half, center_lon, center_lat),
            local_m_to_lonlat(-half, half, center_lon, center_lat),
        ]
        geometry = {
            "type": "Polygon",
            "coordinates": [[[lon, lat] for lat, lon in corners + [corners[0]]]],
        }
        boundary_source = "nominatim_point_square_fallback"
    record = {
        "type": "Feature",
        "properties": {
            "region_slug": region.slug,
            "name_zh": region.name_zh,
            "name_en": region.name_en,
            "osm_type": selected.get("osm_type", region.osm_type),
            "osm_id": selected.get("osm_id", region.osm_id),
            "nominatim_query": region.nominatim_query,
            "nominatim_query_used": query_used,
            "boundary_source": boundary_source,
            "display_name": selected.get("display_name"),
            "license": selected.get("licence"),
            "downloaded_utc": utc_now(),
        },
        "geometry": geometry,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def outer_rings(geometry: dict[str, object]) -> list[list[tuple[float, float]]]:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if geometry_type == "Polygon":
        return [[(float(lon), float(lat)) for lon, lat in coordinates[0]]]  # type: ignore[index]
    if geometry_type == "MultiPolygon":
        return [[(float(lon), float(lat)) for lon, lat in polygon[0]] for polygon in coordinates]  # type: ignore[union-attr]
    raise ValueError(f"Unsupported campus boundary geometry: {geometry_type}")


def point_in_ring(x: float, y: float, ring: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if ((yi > y) != (yj > y)) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi:
            inside = not inside
        j = i
    return inside


def segments_intersect(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float], d: tuple[float, float]) -> bool:
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    return (o1 == 0 or o2 == 0 or o1 * o2 < 0) and (o3 == 0 or o4 == 0 or o3 * o4 < 0)


def rectangle_intersects_ring(rect: tuple[float, float, float, float], ring: list[tuple[float, float]]) -> bool:
    min_x, min_y, max_x, max_y = rect
    if any(min_x <= x <= max_x and min_y <= y <= max_y for x, y in ring):
        return True
    corners = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
    if any(point_in_ring(x, y, ring) for x, y in corners):
        return True
    edges = list(zip(corners, corners[1:] + corners[:1]))
    for a, b in zip(ring, ring[1:] + ring[:1]):
        if any(segments_intersect(a, b, c, d) for c, d in edges):
            return True
    return False


def generate_region_blocks(region: RegionSpec, feature: dict[str, object]) -> list[BlockSpec]:
    lonlat_rings = outer_rings(feature["geometry"])  # type: ignore[arg-type]
    all_lon = [point[0] for ring in lonlat_rings for point in ring]
    all_lat = [point[1] for ring in lonlat_rings for point in ring]
    anchor_lon = (min(all_lon) + max(all_lon)) / 2.0
    anchor_lat = (min(all_lat) + max(all_lat)) / 2.0
    rings = [
        [lonlat_to_local_m(lon, lat, anchor_lon, anchor_lat) for lon, lat in ring]
        for ring in lonlat_rings
    ]
    min_x = math.floor(min(x for ring in rings for x, _ in ring) / MAP_BLOCK_SIZE_M) * MAP_BLOCK_SIZE_M
    max_x = math.ceil(max(x for ring in rings for x, _ in ring) / MAP_BLOCK_SIZE_M) * MAP_BLOCK_SIZE_M
    min_y = math.floor(min(y for ring in rings for _, y in ring) / MAP_BLOCK_SIZE_M) * MAP_BLOCK_SIZE_M
    max_y = math.ceil(max(y for ring in rings for _, y in ring) / MAP_BLOCK_SIZE_M) * MAP_BLOCK_SIZE_M

    blocks: list[BlockSpec] = []
    for y0 in np.arange(min_y, max_y, MAP_BLOCK_SIZE_M):
        for x0 in np.arange(min_x, max_x, MAP_BLOCK_SIZE_M):
            rect = (float(x0), float(y0), float(x0 + MAP_BLOCK_SIZE_M), float(y0 + MAP_BLOCK_SIZE_M))
            if not any(rectangle_intersects_ring(rect, ring) for ring in rings):
                continue
            center_x, center_y = x0 + MAP_BLOCK_SIZE_M / 2.0, y0 + MAP_BLOCK_SIZE_M / 2.0
            center_lat, center_lon = local_m_to_lonlat(center_x, center_y, anchor_lon, anchor_lat)
            min_lat_block, min_lon_block = local_m_to_lonlat(-MAP_BLOCK_SIZE_M / 2.0, -MAP_BLOCK_SIZE_M / 2.0, center_lon, center_lat)
            max_lat_block, max_lon_block = local_m_to_lonlat(MAP_BLOCK_SIZE_M / 2.0, MAP_BLOCK_SIZE_M / 2.0, center_lon, center_lat)
            block_id = f"{center_lat:.7f}_{center_lon:.7f}"
            blocks.append(BlockSpec(
                region.slug, region.name_zh, block_id, float(center_lat), float(center_lon),
                float(min_lat_block), float(min_lon_block), float(max_lat_block), float(max_lon_block),
            ))
    return blocks


def download_block_osm(block: BlockSpec, path: Path) -> bytes:
    if path.is_file():
        cached = path.read_bytes()
        try:
            validate_osm_payload(cached)
            return cached
        except (ET.ParseError, ValueError) as exc:
            print(f"OSM CACHE INVALID {block.region_slug}/{block.block_id}: {exc}; downloading again")
    data = fetch_block_osm_data(block)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)
    time.sleep(0.25)
    return data


def fetch_block_osm_data(block: BlockSpec) -> bytes:
    """Fetch one block, falling back to building-focused Overpass XML."""
    global _OSM_MAP_API_AVAILABLE

    map_bbox = f"{block.min_lon:.10f},{block.min_lat:.10f},{block.max_lon:.10f},{block.max_lat:.10f}"
    last_error: BaseException | None = None
    if _OSM_MAP_API_AVAILABLE is not False:
        try:
            data = http_get(
                f"https://api.openstreetmap.org/api/0.6/map?bbox={map_bbox}",
                max_attempts=2,
            )
            validate_osm_payload(data)
            _OSM_MAP_API_AVAILABLE = True
            return data
        except (OSError, RuntimeError, ET.ParseError, ValueError) as exc:
            last_error = exc
            _OSM_MAP_API_AVAILABLE = False
            print(f"OSM MAP API UNAVAILABLE: {exc}")
            print("OSM FALLBACK: switching this run to Overpass API")

    overpass_bbox = f"{block.min_lat:.10f},{block.min_lon:.10f},{block.max_lat:.10f},{block.max_lon:.10f}"
    query = (
        "[out:xml][timeout:90];"
        f"(way[\"building\"]({overpass_bbox});relation[\"building\"]({overpass_bbox}););"
        "(._;>;);out body;"
    )
    for endpoint in OVERPASS_ENDPOINTS:
        url = f"{endpoint}?{urllib.parse.urlencode({'data': query})}"
        try:
            data = http_get(url, timeout=120, max_attempts=3)
            validate_osm_payload(data)
            print(f"OSM FALLBACK READY: {endpoint}")
            return data
        except (OSError, RuntimeError, ET.ParseError, ValueError) as exc:
            last_error = exc
            print(f"OVERPASS ENDPOINT FAILED: {endpoint}: {exc}")

    raise RuntimeError(
        f"All OSM download sources failed for {block.region_slug}/{block.block_id}"
    ) from last_error


def validate_osm_payload(data: bytes) -> None:
    if not data:
        raise ValueError("empty OSM response")
    root = ET.fromstring(data)
    if root.tag != "osm":
        raise ValueError(f"unexpected OSM root element: {root.tag}")


def has_valid_osm_cache(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        validate_osm_payload(path.read_bytes())
        return True
    except (OSError, ET.ParseError, ValueError):
        return False


def validate_png_payload(data: bytes) -> tuple[int, int]:
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise ValueError("response is not a valid PNG image")
    width, height = struct.unpack(">II", data[16:24])
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid PNG dimensions: {width}x{height}")
    return width, height


def has_valid_satellite_cache(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        width, height = validate_png_payload(path.read_bytes())
        return width == SATELLITE_IMAGE_PIXELS and height == SATELLITE_IMAGE_PIXELS
    except (OSError, ValueError):
        return False


def download_block_satellite(block: BlockSpec, path: Path) -> bytes:
    if path.is_file():
        cached = path.read_bytes()
        try:
            width, height = validate_png_payload(cached)
            if width == SATELLITE_IMAGE_PIXELS and height == SATELLITE_IMAGE_PIXELS:
                return cached
        except ValueError as exc:
            print(f"SATELLITE CACHE INVALID {block.region_slug}/{block.block_id}: {exc}; downloading again")
    query = urllib.parse.urlencode({
        "bbox": f"{block.min_lon:.10f},{block.min_lat:.10f},{block.max_lon:.10f},{block.max_lat:.10f}",
        "bboxSR": 4326,
        "imageSR": 3857,
        "size": f"{SATELLITE_IMAGE_PIXELS},{SATELLITE_IMAGE_PIXELS}",
        "dpi": 96,
        "format": "png",
        "transparent": "false",
        "f": "image",
    })
    data = http_get(f"{SATELLITE_EXPORT_ENDPOINT}?{query}", timeout=120)
    width, height = validate_png_payload(data)
    if width != SATELLITE_IMAGE_PIXELS or height != SATELLITE_IMAGE_PIXELS:
        raise ValueError(
            f"satellite service returned {width}x{height}; "
            f"expected {SATELLITE_IMAGE_PIXELS}x{SATELLITE_IMAGE_PIXELS}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)
    return data


def prefetch_block_assets(blocks: list[BlockSpec], dataset_root: Path) -> None:
    """Cache every selected block's required OSM file before simulation."""
    worker_count = max(1, int(os.environ.get("RID_OSM_DOWNLOAD_WORKERS", "1")))
    print(
        f"\nASSET PREFETCH START: {len(blocks)} blocks; "
        "simulation will start after required OSM files are cached; satellite=SKIP; "
        f"download_workers={worker_count}"
    )
    osm_cached_count = 0
    osm_downloaded_count = 0

    def fetch_one(index: int, block: BlockSpec) -> tuple[int, BlockSpec, bool]:
        block_dir = dataset_root / block.region_slug / block.block_id
        osm_path = block_dir / "osm_map.osm"
        osm_was_cached = has_valid_osm_cache(osm_path)
        download_block_osm(block, osm_path)
        return index, block, osm_was_cached

    if worker_count == 1:
        results = (fetch_one(index, block) for index, block in enumerate(blocks, start=1))
        for index, block, osm_was_cached in results:
            if osm_was_cached:
                osm_cached_count += 1
                state = "cached"
            else:
                osm_downloaded_count += 1
                state = "downloaded"
            print(
                f"ASSET READY [{index}/{len(blocks)}] "
                f"{block.region_slug}/{block.block_id}: OSM={state}; satellite=SKIP"
            )
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(fetch_one, index, block): (index, block)
                for index, block in enumerate(blocks, start=1)
            }
            completed_count = 0
            for future in as_completed(futures):
                index, block, osm_was_cached = future.result()
                completed_count += 1
                if osm_was_cached:
                    osm_cached_count += 1
                    state = "cached"
                else:
                    osm_downloaded_count += 1
                    state = "downloaded"
                print(
                    f"ASSET READY [{completed_count}/{len(blocks)}; manifest={index}] "
                    f"{block.region_slug}/{block.block_id}: OSM={state}; satellite=SKIP"
                )
    print(
        f"ASSET PREFETCH COMPLETE: {len(blocks)} blocks ready; "
        f"OSM={osm_cached_count} cached/{osm_downloaded_count} downloaded; satellite=SKIP"
    )


def parse_meters(value: str | None) -> float | None:
    if not value:
        return None
    match = re.search(r"[-+]?\d*\.?\d+", value.lower())
    if not match:
        return None
    meters = float(match.group(0))
    if "ft" in value.lower() or "feet" in value.lower():
        meters *= 0.3048
    return meters


def osm_building_height(tags: dict[str, str], default_level_height_m: float = 3.2, default_height_m: float = 12.0) -> float:
    height = parse_meters(tags.get("height"))
    if height is not None and height > 0:
        return height
    levels = parse_meters(tags.get("building:levels")) or parse_meters(tags.get("levels"))
    return levels * default_level_height_m if levels is not None and levels > 0 else default_height_m


def clip_polygon(points: list[tuple[float, float]], half_size: float) -> list[tuple[float, float]]:
    def clip_edge(vertices, inside, intersection):
        if not vertices:
            return []
        output = []
        previous = vertices[-1]
        previous_inside = inside(previous)
        for current in vertices:
            current_inside = inside(current)
            if current_inside != previous_inside:
                output.append(intersection(previous, current))
            if current_inside:
                output.append(current)
            previous, previous_inside = current, current_inside
        return output

    result = points
    result = clip_edge(result, lambda p: p[0] >= -half_size, lambda a, b: (-half_size, a[1] + (b[1] - a[1]) * (-half_size - a[0]) / (b[0] - a[0])))
    result = clip_edge(result, lambda p: p[0] <= half_size, lambda a, b: (half_size, a[1] + (b[1] - a[1]) * (half_size - a[0]) / (b[0] - a[0])))
    result = clip_edge(result, lambda p: p[1] >= -half_size, lambda a, b: (a[0] + (b[0] - a[0]) * (-half_size - a[1]) / (b[1] - a[1]), -half_size))
    result = clip_edge(result, lambda p: p[1] <= half_size, lambda a, b: (a[0] + (b[0] - a[0]) * (half_size - a[1]) / (b[1] - a[1]), half_size))
    return result


def parse_block_buildings(osm_bytes: bytes, block: BlockSpec) -> list[Building]:
    root = ET.fromstring(osm_bytes)
    nodes = {
        node.attrib["id"]: (float(node.attrib["lat"]), float(node.attrib["lon"]))
        for node in root.findall("node")
    }
    buildings: list[Building] = []
    for way in root.findall("way"):
        tags = {tag.attrib.get("k", ""): tag.attrib.get("v", "") for tag in way.findall("tag")}
        if tags.get("building", "no") == "no":
            continue
        lonlat = [nodes[nd.attrib["ref"]] for nd in way.findall("nd") if nd.attrib.get("ref") in nodes]
        if len(lonlat) < 3:
            continue
        if lonlat[0] == lonlat[-1]:
            lonlat = lonlat[:-1]
        local = [lonlat_to_local_m(lon, lat, block.center_lon, block.center_lat) for lat, lon in lonlat]
        clipped = clip_polygon(local, MAP_BLOCK_SIZE_M / 2.0)
        if len(clipped) < 3:
            continue
        clipped_lonlat = [local_m_to_lonlat(x, y, block.center_lon, block.center_lat) for x, y in clipped]
        buildings.append(Building(
            [lat for lat, _ in clipped_lonlat],
            [lon for _, lon in clipped_lonlat],
            osm_building_height(tags),
        ))
    return buildings


def plot_buildings_2d(block: BlockSpec, buildings: list[Building], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8), dpi=180, constrained_layout=True)
    for building in buildings:
        xy = np.asarray([lonlat_to_local_m(lon, lat, block.center_lon, block.center_lat) for lat, lon in zip(building.lat, building.lon)])
        ax.fill(xy[:, 0], xy[:, 1], facecolor="#d8dcdf", edgecolor="#4b5563", linewidth=0.7)
    half = MAP_BLOCK_SIZE_M / 2.0
    ax.set(xlim=(-half, half), ylim=(-half, half), aspect="equal", xlabel="East from block center (m)", ylabel="North from block center (m)")
    ax.set_title(f"OSM buildings 2D\n{block.region_slug} | {block.block_id}")
    ax.grid(True, alpha=0.25)
    fig.savefig(path)
    plt.close(fig)


def plot_satellite_topdown(block: BlockSpec, satellite_path: Path, output_path: Path) -> None:
    """Render a georeferenced, equal-axis satellite plan view with attribution."""
    import matplotlib.pyplot as plt

    image = plt.imread(satellite_path)
    half = MAP_BLOCK_SIZE_M / 2.0
    fig, ax = plt.subplots(figsize=(8, 8), dpi=180, constrained_layout=True)
    ax.imshow(image, extent=(-half, half, -half, half), origin="upper")
    ax.set(
        xlim=(-half, half),
        ylim=(-half, half),
        aspect="equal",
        xlabel="East from block center (m)",
        ylabel="North from block center (m)",
    )
    ax.set_title(f"Satellite top-down view\n{block.region_slug} | {block.block_id}")
    ax.text(
        0.995,
        0.008,
        SATELLITE_ATTRIBUTION,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 2},
    )
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    fig.savefig(temporary, format="png")
    plt.close(fig)
    temporary.replace(output_path)


def plot_buildings_3d(block: BlockSpec, buildings: list[Building], path: Path) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9, 8), dpi=180, constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    for building in buildings:
        xy = [lonlat_to_local_m(lon, lat, block.center_lon, block.center_lat) for lat, lon in zip(building.lat, building.lon)]
        height = building.height_m
        roof = [[(x, y, height) for x, y in xy]]
        walls = [[(x1, y1, 0.0), (x2, y2, 0.0), (x2, y2, height), (x1, y1, height)] for (x1, y1), (x2, y2) in zip(xy, xy[1:] + xy[:1])]
        ax.add_collection3d(Poly3DCollection(roof, facecolor="#d8a03b", edgecolor="#374151", linewidth=0.4, alpha=0.9))
        ax.add_collection3d(Poly3DCollection(walls, facecolor="#73a9ad", edgecolor="#374151", linewidth=0.35, alpha=0.88))
    half = MAP_BLOCK_SIZE_M / 2.0
    max_height = max([building.height_m for building in buildings] + [10.0])
    z_max = max_height * 1.15
    ax.set(
        xlim=(-half, half),
        ylim=(-half, half),
        zlim=(0, z_max),
        xlabel="East (m)",
        ylabel="North (m)",
        zlabel="Height (m)",
    )
    # The axes box follows the numerical data ranges, so one meter has the
    # same visual length on X, Y, and Z. Orthographic projection avoids
    # perspective foreshortening that would otherwise distort this ratio.
    ax.set_box_aspect((MAP_BLOCK_SIZE_M, MAP_BLOCK_SIZE_M, z_max))
    ax.set_proj_type("ortho")
    ax.set_title(f"OSM buildings 3D (equal XYZ scale)\n{block.region_slug} | {block.block_id}")
    ax.view_init(elev=32, azim=-58)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fig.savefig(temporary, format="png")
    plt.close(fig)
    temporary.replace(path)


def building_local_polygons(block: BlockSpec, buildings: list[Building]) -> list[list[tuple[float, float]]]:
    return [
        [lonlat_to_local_m(lon, lat, block.center_lon, block.center_lat) for lat, lon in zip(building.lat, building.lon)]
        for building in buildings
    ]


def point_to_segment_distance(x: float, y: float, a: tuple[float, float], b: tuple[float, float]) -> float:
    ab_x, ab_y = b[0] - a[0], b[1] - a[1]
    length_squared = ab_x * ab_x + ab_y * ab_y
    if length_squared <= 1e-20:
        return math.hypot(x - a[0], y - a[1])
    projection = max(0.0, min(1.0, ((x - a[0]) * ab_x + (y - a[1]) * ab_y) / length_squared))
    closest_x, closest_y = a[0] + projection * ab_x, a[1] + projection * ab_y
    return math.hypot(x - closest_x, y - closest_y)


def point_to_polygon_distance(x: float, y: float, polygon: list[tuple[float, float]]) -> float:
    if point_in_ring(x, y, polygon):
        return 0.0
    return min(point_to_segment_distance(x, y, a, b) for a, b in zip(polygon, polygon[1:] + polygon[:1]))


def block_random_seed(block: BlockSpec, global_seed: int) -> int:
    digest = hashlib.sha256(f"{global_seed}:{block.region_slug}:{block.block_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def random_rx_positions(
    block: BlockSpec,
    buildings: list[Building],
    count: int,
    global_seed: int,
    minimum_clearance_m: float = MIN_BUILDING_CLEARANCE_M,
) -> tuple[list[GroundRxPoint], int, list[float]]:
    """Generate deterministic ground RX sites with building clearance."""
    if count <= 0:
        raise ValueError("rx_count must be positive")
    polygons = building_local_polygons(block, buildings)
    seed = block_random_seed(block, global_seed)
    rng = np.random.default_rng(seed)
    half = MAP_BLOCK_SIZE_M / 2.0
    points: list[GroundRxPoint] = []
    clearances: list[float] = []
    max_attempts = max(100_000, count * 10_000)
    for _ in range(max_attempts):
        if len(points) >= count:
            break
        x, y = (float(value) for value in rng.uniform(-half, half, size=2))
        clearance = min((point_to_polygon_distance(x, y, polygon) for polygon in polygons), default=math.inf)
        if clearance + 1e-9 < minimum_clearance_m:
            continue
        lat, lon = local_m_to_lonlat(x, y, block.center_lon, block.center_lat)
        points.append(GroundRxPoint(len(points), x, y, lat, lon, math.hypot(x, y)))
        clearances.append(clearance)
    if len(points) != count:
        raise RuntimeError(
            f"Could only place {len(points)}/{count} RX sites in {block.region_slug}/{block.block_id} "
            f"with {minimum_clearance_m:.1f} m building clearance"
        )
    return points, seed, clearances


def voxel_tx_horizontal_grid(
    block: BlockSpec,
    voxel_size_m: float,
    nx: int,
    ny: int,
) -> list[MeshTxPoint]:
    """Return the XY cell centers shared by all aerial TX voxel layers."""
    if voxel_size_m <= 0 or min(nx, ny) <= 0:
        raise ValueError("voxel size and horizontal dimensions must be positive")
    if not math.isclose(nx * voxel_size_m, MAP_BLOCK_SIZE_M, abs_tol=1e-9):
        raise ValueError("voxel nx*size must equal the map size")
    if not math.isclose(ny * voxel_size_m, MAP_BLOCK_SIZE_M, abs_tol=1e-9):
        raise ValueError("voxel ny*size must equal the map size")
    half = MAP_BLOCK_SIZE_M / 2.0
    xs = -half + voxel_size_m * (np.arange(nx, dtype=float) + 0.5)
    ys = -half + voxel_size_m * (np.arange(ny, dtype=float) + 0.5)
    points: list[MeshTxPoint] = []
    for y_m in ys:
        for x_m in xs:
            lat, lon = local_m_to_lonlat(
                float(x_m), float(y_m), block.center_lon, block.center_lat
            )
            points.append(
                MeshTxPoint(
                    len(points), float(x_m), float(y_m), lat, lon,
                    math.hypot(float(x_m), float(y_m)),
                )
            )
    return points


def random_ground_tx_positions(
    block: BlockSpec,
    buildings: list[Building],
    count: int,
    height_m: float,
    global_seed: int,
    minimum_clearance_m: float,
) -> tuple[list[GroundTxPoint], int, list[float]]:
    """Generate deterministic ground TX sites outside building clearance zones."""
    if count <= 0:
        raise ValueError("tx_count must be positive")
    if height_m <= 0:
        raise ValueError("tx_height_m must be positive")
    polygons = building_local_polygons(block, buildings)
    seed = block_random_seed(block, global_seed)
    rng = np.random.default_rng(seed)
    half = MAP_BLOCK_SIZE_M / 2.0
    points: list[GroundTxPoint] = []
    clearances: list[float] = []
    max_attempts = max(100_000, count * 10_000)
    for _ in range(max_attempts):
        if len(points) >= count:
            break
        x, y = (float(value) for value in rng.uniform(-half, half, size=2))
        clearance = min(
            (point_to_polygon_distance(x, y, polygon) for polygon in polygons),
            default=math.inf,
        )
        if clearance + 1e-9 < minimum_clearance_m:
            continue
        lat, lon = local_m_to_lonlat(x, y, block.center_lon, block.center_lat)
        points.append(
            GroundTxPoint(
                len(points), x, y, height_m, lat, lon, math.hypot(x, y)
            )
        )
        clearances.append(clearance)
    if len(points) != count:
        raise RuntimeError(
            f"Could only place {len(points)}/{count} TX sites in "
            f"{block.region_slug}/{block.block_id} with "
            f"{minimum_clearance_m:.1f} m building clearance"
        )
    return points, seed, clearances


def voxel_rx_grid(
    block: BlockSpec,
    voxel_size_m: float,
    nx: int,
    ny: int,
    nz: int,
    z_start_m: float,
) -> list[VoxelRxPoint]:
    """Return cell-center RX positions for an exact regular voxel volume."""
    if voxel_size_m <= 0 or min(nx, ny, nz) <= 0 or z_start_m <= 0:
        raise ValueError("voxel size, dimensions, and starting height must be positive")
    expected_x = nx * voxel_size_m
    expected_y = ny * voxel_size_m
    if not math.isclose(expected_x, MAP_BLOCK_SIZE_M, abs_tol=1e-9):
        raise ValueError(f"voxel nx*size must equal {MAP_BLOCK_SIZE_M:g} m, got {expected_x:g}")
    if not math.isclose(expected_y, MAP_BLOCK_SIZE_M, abs_tol=1e-9):
        raise ValueError(f"voxel ny*size must equal {MAP_BLOCK_SIZE_M:g} m, got {expected_y:g}")
    half = MAP_BLOCK_SIZE_M / 2.0
    xs = -half + voxel_size_m * (np.arange(nx, dtype=float) + 0.5)
    ys = -half + voxel_size_m * (np.arange(ny, dtype=float) + 0.5)
    zs = z_start_m + voxel_size_m * np.arange(nz, dtype=float)
    points: list[VoxelRxPoint] = []
    for iz, z_m in enumerate(zs):
        for iy, y_m in enumerate(ys):
            for ix, x_m in enumerate(xs):
                lat, lon = local_m_to_lonlat(
                    float(x_m), float(y_m), block.center_lon, block.center_lat
                )
                points.append(
                    VoxelRxPoint(
                        len(points), ix, iy, iz,
                        float(x_m), float(y_m), float(z_m), lat, lon,
                    )
                )
    return points


def write_tx_positions_csv(
    points: list[GroundTxPoint], clearances: list[float], path: Path
) -> None:
    rows = [{
        "tx_index": point.index,
        "x_m": f"{point.x_m:.6f}",
        "y_m": f"{point.y_m:.6f}",
        "z_m": f"{point.z_m:.3f}",
        "lat": f"{point.lat:.10f}",
        "lon": f"{point.lon:.10f}",
        "radial_distance_m": f"{point.radial_distance_m:.6f}",
        "building_clearance_m": "inf" if math.isinf(clearance) else f"{clearance:.6f}",
    } for point, clearance in zip(points, clearances)]
    write_dict_csv(rows, path)


def plot_tx_positions_2d(
    block: BlockSpec,
    buildings: list[Building],
    points: list[GroundTxPoint],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8), dpi=180, constrained_layout=True)
    for polygon in building_local_polygons(block, buildings):
        xy = np.asarray(polygon)
        ax.fill(xy[:, 0], xy[:, 1], facecolor="#d8dcdf", edgecolor="#4b5563", linewidth=0.7)
    ax.scatter(
        [point.x_m for point in points], [point.y_m for point in points],
        s=35, c="#dc2626", marker="x", linewidths=1.8, label="Ground TX",
    )
    for point in points:
        ax.annotate(str(point.index + 1), (point.x_m, point.y_m), xytext=(3, 3), textcoords="offset points", fontsize=7)
    half = MAP_BLOCK_SIZE_M / 2.0
    ax.set(
        xlim=(-half, half), ylim=(-half, half), aspect="equal",
        xlabel="East from block center (m)", ylabel="North from block center (m)",
    )
    ax.set_title(
        f"Random ground TX sites (minimum {MIN_BUILDING_CLEARANCE_M:g} m building clearance)\n"
        f"{block.region_slug} | {block.block_id}"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(path)
    plt.close(fig)


def square_tx_mesh(block: BlockSpec, spacing_m: float, max_points: int = 0) -> list[MeshTxPoint]:
    if spacing_m <= 0:
        raise ValueError("mesh spacing must be positive")
    half = TX_MESH_SIZE_M / 2.0
    coordinates = np.arange(-half, half + spacing_m * 0.25, spacing_m)
    points: list[MeshTxPoint] = []
    for y in coordinates:
        for x in coordinates:
            lat, lon = local_m_to_lonlat(float(x), float(y), block.center_lon, block.center_lat)
            points.append(MeshTxPoint(len(points), float(x), float(y), lat, lon, math.hypot(float(x), float(y))))
    if max_points > 0 and len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points).round().astype(int)
        points = [points[int(index)] for index in np.unique(indices)]
        points = [MeshTxPoint(index, point.x_m, point.y_m, point.lat, point.lon, point.radial_distance_m) for index, point in enumerate(points)]
    return points


def plot_rx_positions_2d(
    block: BlockSpec,
    buildings: list[Building],
    points: list[GroundRxPoint],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8), dpi=180, constrained_layout=True)
    for polygon in building_local_polygons(block, buildings):
        xy = np.asarray(polygon)
        ax.fill(xy[:, 0], xy[:, 1], facecolor="#d8dcdf", edgecolor="#4b5563", linewidth=0.7)
    ax.scatter([point.x_m for point in points], [point.y_m for point in points], s=28, c="#2563eb", marker="x", linewidths=1.8, label="Ground RX")
    for point in points:
        ax.annotate(str(point.index + 1), (point.x_m, point.y_m), xytext=(3, 3), textcoords="offset points", fontsize=6)
    half = MAP_BLOCK_SIZE_M / 2.0
    ax.set(xlim=(-half, half), ylim=(-half, half), aspect="equal", xlabel="East from block center (m)", ylabel="North from block center (m)")
    ax.set_title(f"Random ground RX sites (minimum {MIN_BUILDING_CLEARANCE_M:g} m building clearance)\n{block.region_slug} | {block.block_id}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(path)
    plt.close(fig)


def make_block_config(block: BlockSpec, heights_m: list[float], block_dir: Path, args: argparse.Namespace) -> RtExperimentConfig:
    return RtExperimentConfig(
        run_tag="block",
        lat_rx=block.center_lat,
        lon_rx=block.center_lon,
        rx_height_m=args.rx_height,
        lat0=block.center_lat,
        lon0=block.center_lon,
        radius_m=MAP_BLOCK_SIZE_M / math.sqrt(2.0),
        selected_experiment_heights=heights_m,
        discard_path_start_seconds=[0.0] * len(heights_m),
        discard_path_end_seconds=[0.0] * len(heights_m),
        start_timestamp="",
        rid_log_dir=Path(),
        rid_log_pattern="",
        frequency_hz=2.437e9,
        tx_power_dbm=20.0,
        mesh_spacing_m=args.voxel_size,
        mesh_batch_size=args.tx_batch_size,
        mesh_max_points_per_height=0,
        max_depth=args.max_depth,
        samples_per_tx=args.samples_per_tx,
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,
        refraction=True,
        diffraction=True,
        default_level_height_m=3.2,
        default_building_height_m=12.0,
        override_building_height_m=None,
        ground_size_m=MAP_BLOCK_SIZE_M,
        add_building_roofs=True,
        output_dir=block_dir,
        scene_work_dir=block_dir / "scene_generated",
        rx_antenna_pattern="iso",
        rx_antenna_polarization="V",
        rx_orientation_deg=(0.0, 0.0, 0.0),
        rx_extra_gain_dbi=0.0,
        plan_view_marker_size=18,
        plan_view_rx_marker_size=90,
        plan_view_figsize=(10, 8),
        plan_view_grid_figsize=(16, 12),
        plan_view_dpi=160,
        plan_view_rssi_limits_dbm=None,
        plan_view_sim_rx_power_limits_dbm=None,
    )


def write_rx_positions_csv(points: list[GroundRxPoint], clearances: list[float], path: Path) -> None:
    rows = [{
        "rx_index": point.index,
        "x_m": f"{point.x_m:.6f}",
        "y_m": f"{point.y_m:.6f}",
        "lat": f"{point.lat:.10f}",
        "lon": f"{point.lon:.10f}",
        "distance_to_nearest_building_m": "inf" if math.isinf(clearance) else f"{clearance:.3f}",
    } for point, clearance in zip(points, clearances)]
    write_dict_csv(rows, path)


def read_rx_positions_csv(path: Path) -> list[GroundRxPoint]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    return [
        GroundRxPoint(
            index=int(row["rx_index"]),
            x_m=float(row["x_m"]),
            y_m=float(row["y_m"]),
            lat=float(row["lat"]),
            lon=float(row["lon"]),
            radial_distance_m=math.hypot(float(row["x_m"]), float(row["y_m"])),
        )
        for row in rows
    ]


def rx_prefix_point_matches(previous: GroundRxPoint, desired: GroundRxPoint) -> bool:
    """Compare deterministic RXs while accepting legacy .3f CSV rounding."""
    return previous.index == desired.index and (
        math.isclose(previous.x_m, desired.x_m, abs_tol=5.01e-4)
        and math.isclose(previous.y_m, desired.y_m, abs_tol=5.01e-4)
    )


def plot_rss_maps_by_rx(
    block: BlockSpec,
    buildings: list[Building],
    rx_points: list[GroundRxPoint],
    heights_m: list[float],
    tx_points: list[MeshTxPoint],
    results_csv: Path,
    output_dir: Path,
) -> list[Path]:
    """Create one five-height RSS comparison figure for every ground RX."""
    import matplotlib.pyplot as plt

    height_lookup = {round(float(height), 6): index for index, height in enumerate(heights_m)}
    values = np.full((len(rx_points), len(heights_m), len(tx_points)), np.nan, dtype=np.float32)
    with results_csv.open("r", newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            rx_index = int(row["rx_index"])
            height_index = height_lookup[round(float(row["height_m"]), 6)]
            tx_index = int(row["tx_mesh_index"])
            values[rx_index, height_index, tx_index] = float(row["sim_rx_power_dbm"])
    missing = int(np.isnan(values).sum())
    if missing:
        raise RuntimeError(f"Cannot plot incomplete result cube: {missing} RX/height/TX values are missing")

    x_values = sorted({point.x_m for point in tx_points})
    y_values = sorted({point.y_m for point in tx_points})
    expected_grid_points = len(x_values) * len(y_values)
    if expected_grid_points != len(tx_points):
        raise RuntimeError("TX positions do not form a complete square grid")
    value_cube = values.reshape(len(rx_points), len(heights_m), len(y_values), len(x_values))
    covered = value_cube[np.isfinite(value_cube) & (value_cube > -279.9)]
    if covered.size == 0:
        covered = value_cube[np.isfinite(value_cube)]
    color_min, color_max = (float(np.percentile(covered, 1.0)), float(np.percentile(covered, 99.0)))
    if math.isclose(color_min, color_max):
        color_min, color_max = float(finite.min()), float(finite.max() + 1e-6)

    polygons = building_local_polygons(block, buildings)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    color_map = plt.colormaps["turbo"].copy()
    color_map.set_bad("#d1d5db")
    for rx_point in rx_points:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10.5), dpi=150, constrained_layout=True)
        image_artist = None
        for height_index, (height, ax) in enumerate(zip(heights_m, axes.flat[:5])):
            image_artist = ax.imshow(
                np.ma.masked_less_equal(value_cube[rx_point.index, height_index], -279.9),
                origin="lower",
                extent=[min(x_values), max(x_values), min(y_values), max(y_values)],
                cmap=color_map,
                vmin=color_min,
                vmax=color_max,
                interpolation="nearest",
                aspect="equal",
            )
            for polygon in polygons:
                xy = np.asarray(polygon)
                ax.plot(xy[:, 0], xy[:, 1], color="white", linewidth=0.55, alpha=0.75)
            ax.scatter([rx_point.x_m], [rx_point.y_m], marker="x", s=62, linewidths=2.0, c="black")
            ax.set_title(f"TX height {float(height):g} m")
            ax.set_xlabel("East (m)")
            ax.set_ylabel("North (m)")
            ax.set_xlim(-TX_MESH_SIZE_M / 2.0, TX_MESH_SIZE_M / 2.0)
            ax.set_ylim(-TX_MESH_SIZE_M / 2.0, TX_MESH_SIZE_M / 2.0)
        info_ax = axes.flat[5]
        info_ax.axis("off")
        info_ax.text(
            0.04,
            0.96,
            f"Ground RX {rx_point.index + 1}\n"
            f"lat: {rx_point.lat:.10f}\nlon: {rx_point.lon:.10f}\n"
            f"local: ({rx_point.x_m:.3f}, {rx_point.y_m:.3f}) m\n"
            f"RX height: 0.5 m\nTX mesh: {len(x_values)} × {len(y_values)}\n"
            f"shared covered-cell scale: {color_min:.1f} to {color_max:.1f} dBm\n"
            "gray: no resolved path",
            ha="left",
            va="top",
            fontsize=12,
        )
        if image_artist is not None:
            fig.colorbar(image_artist, ax=list(axes.flat[:5]), label="Simulated RSS (dBm)", shrink=0.88)
        fig.suptitle(f"{block.region_slug} | {block.block_id} | RX {rx_point.index + 1}: five aerial TX planes", fontsize=15)
        output_path = output_dir / f"rx_{rx_point.index:03d}_rss_5_heights.png"
        temp_path = output_path.with_suffix(".png.tmp")
        fig.savefig(temp_path, format="png")
        plt.close(fig)
        temp_path.replace(output_path)
        written.append(output_path)
    return written


def process_block(block: BlockSpec, dataset_root: Path, args: argparse.Namespace) -> str:
    block_timing_started = time.perf_counter()
    phase_timings: dict[str, float] = {}

    def record_phase(name: str, phase_started: float) -> float:
        elapsed = time.perf_counter() - phase_started
        if not ENABLE_TIMING_LOGS:
            return elapsed
        phase_timings[name] = phase_timings.get(name, 0.0) + elapsed
        print(
            f"BLOCK PHASE TIMING {block.region_slug}/{block.block_id}: "
            f"{name}={elapsed:.3f}s",
            flush=True,
        )
        return elapsed

    block_dir = dataset_root / block.region_slug / block.block_id
    block_dir.mkdir(parents=True, exist_ok=True)
    phase_started = time.perf_counter()
    osm_path = block_dir / "osm_map.osm"
    osm_bytes = download_block_osm(block, osm_path)
    osm_sha256 = hashlib.sha256(osm_bytes).hexdigest()
    record_phase("osm_load_hash", phase_started)
    metadata_path = block_dir / "metadata.json"
    if args.resume and metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        desired_status = "prepared" if args.skip_rt else "complete"
        settings_match = (
            existing.get("ground_rx_count") == args.rx_count
            and existing.get("random_seed_global") == args.random_seed
            and existing.get("minimum_building_clearance_m") == MIN_BUILDING_CLEARANCE_M
            and existing.get("block_size_m") == MAP_BLOCK_SIZE_M
            and existing.get("tx_mesh_size_m") == TX_MESH_SIZE_M
            and existing.get("tx_mesh_spacing_m") == args.mesh_spacing
            and existing.get("tx_mesh_points_per_height") == len(square_tx_mesh(block, args.mesh_spacing, args.max_tx_points))
            and existing.get("osm_height_variant") == args.osm_height_variant
            and existing.get("osm_sha256") == osm_sha256
            and existing.get("rt", {}).get("samples_per_tx") == args.samples_per_tx
            and existing.get("rt", {}).get("max_depth") == args.max_depth
        )
        status_matches = existing.get("status") == desired_status or (args.skip_rt and existing.get("status") == "complete")
        if settings_match and status_matches:
            satellite_output = block_dir / SATELLITE_RENDERED_FILENAME
            if ENABLE_SATELLITE_IMAGERY and (
                not satellite_output.is_file() or satellite_output.stat().st_size == 0
            ):
                print(f"PLOTTING {block.region_slug}/{block.block_id}: creating satellite top-down figure")
                phase_started = time.perf_counter()
                satellite_path = block_dir / SATELLITE_CACHE_FILENAME
                download_block_satellite(block, satellite_path)
                plot_satellite_topdown(block, satellite_path, satellite_output)
                record_phase("resume_satellite_plot", phase_started)
                existing.setdefault("files", {})["satellite_topdown"] = SATELLITE_RENDERED_FILENAME
                existing["files"]["satellite_imagery_raw"] = SATELLITE_CACHE_FILENAME
                existing["satellite_imagery"] = {
                    "service": "Esri World Imagery",
                    "service_url": SATELLITE_SERVICE_URL,
                    "attribution": SATELLITE_ATTRIBUTION,
                    "raw_file": SATELLITE_CACHE_FILENAME,
                }
                metadata_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
            if existing.get("status") == "complete" and not args.skip_rt:
                maps_dir = block_dir / "rss_maps_by_rx"
                map_count = len(list(maps_dir.glob("rx_*_rss_5_heights.png"))) if maps_dir.is_dir() else 0
                if map_count != args.rx_count or existing.get("rss_plot_schema_version") != 2:
                    print(f"PLOTTING {block.region_slug}/{block.block_id}: creating {args.rx_count} per-RX RSS figures")
                    phase_started = time.perf_counter()
                    buildings = parse_block_buildings(osm_bytes, block)
                    rx_points = read_rx_positions_csv(block_dir / "rx_positions.csv")
                    heights_m = [float(value) for value in existing["simulation_heights_m"]]
                    tx_mesh_points = square_tx_mesh(block, args.mesh_spacing, args.max_tx_points)
                    files = plot_rss_maps_by_rx(
                        block,
                        buildings,
                        rx_points,
                        heights_m,
                        tx_mesh_points,
                        block_dir / "sionna_results.csv",
                        maps_dir,
                    )
                    record_phase("resume_rss_maps", phase_started)
                    existing.setdefault("files", {})["rss_maps_by_rx"] = "rss_maps_by_rx"
                    existing["rss_map_file_count"] = len(files)
                    existing["rss_plot_schema_version"] = 2
                    metadata_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
                (block_dir / "sionna_results.csv.checkpoint.json").unlink(missing_ok=True)
            if ENABLE_TIMING_LOGS:
                print(
                    f"SKIP {block.region_slug}/{block.block_id}: {existing.get('status')} "
                    f"(invocation_timing={time.perf_counter() - block_timing_started:.3f}s)"
                )
            else:
                print(f"SKIP {block.region_slug}/{block.block_id}: {existing.get('status')}")
            return "skipped"

    started = time.time()
    phase_started = time.perf_counter()
    buildings = parse_block_buildings(osm_bytes, block)
    maximum_building_height_m = max([building.height_m for building in buildings] + [0.0])
    height_min_m = max(10.0, maximum_building_height_m)
    heights_m = [height_min_m + 5.0 * index for index in range(5)]
    record_phase("parse_buildings_and_heights", phase_started)

    phase_started = time.perf_counter()
    plot_buildings_2d(block, buildings, block_dir / "osm_buildings_2d.png")
    record_phase("plot_buildings_2d", phase_started)
    phase_started = time.perf_counter()
    plot_buildings_3d(block, buildings, block_dir / "osm_buildings_3d.png")
    record_phase("plot_buildings_3d", phase_started)
    if ENABLE_SATELLITE_IMAGERY:
        phase_started = time.perf_counter()
        satellite_path = block_dir / SATELLITE_CACHE_FILENAME
        download_block_satellite(block, satellite_path)
        plot_satellite_topdown(block, satellite_path, block_dir / SATELLITE_RENDERED_FILENAME)
        record_phase("satellite_download_and_plot", phase_started)
    else:
        print(f"SATELLITE SKIP {block.region_slug}/{block.block_id}: disabled")
    phase_started = time.perf_counter()
    scene_dir = block_dir / "scene_generated"
    scene_xml = write_sionna_scene(make_block_config(block, heights_m, block_dir, args), buildings, scene_dir)
    record_phase("write_sionna_scene", phase_started)
    phase_started = time.perf_counter()
    rx_points, random_seed_derived, building_clearances = random_rx_positions(
        block,
        buildings,
        args.rx_count,
        args.random_seed,
    )
    tx_mesh_points = square_tx_mesh(block, args.mesh_spacing, args.max_tx_points)
    write_rx_positions_csv(rx_points, building_clearances, block_dir / "rx_positions.csv")
    plot_rx_positions_2d(block, buildings, rx_points, block_dir / "rx_positions_2d.png")
    record_phase("rx_tx_setup_and_plot", phase_started)

    metadata = {
        "schema_version": "2.1",
        "status": "prepared",
        "created_utc": utc_now(),
        "region_slug": block.region_slug,
        "region_name_zh": block.region_name_zh,
        "block_id": block.block_id,
        "block_center": {"lat": block.center_lat, "lon": block.center_lon},
        "block_bbox": {"min_lat": block.min_lat, "min_lon": block.min_lon, "max_lat": block.max_lat, "max_lon": block.max_lon},
        "block_size_m": MAP_BLOCK_SIZE_M,
        "map_block_size_m": MAP_BLOCK_SIZE_M,
        "tx_mesh_size_m": TX_MESH_SIZE_M,
        "building_count": len(buildings),
        "maximum_building_height_m": maximum_building_height_m,
        "height_min_m": height_min_m,
        "simulation_heights_m": heights_m,
        "rx_position_method": "deterministic uniform random rejection sampling in the 400 m map block",
        "ground_rx_count": len(rx_points),
        "ground_rx_height_m": 0.5,
        "random_seed_global": args.random_seed,
        "random_seed_derived": random_seed_derived,
        "osm_height_variant": args.osm_height_variant,
        "osm_sha256": osm_sha256,
        "minimum_building_clearance_m": MIN_BUILDING_CLEARANCE_M,
        "minimum_observed_building_clearance_m": None if all(math.isinf(value) for value in building_clearances) else min(building_clearances),
        "tx_mesh_spacing_m": args.mesh_spacing,
        "tx_mesh_points_per_height": len(tx_mesh_points),
        "expected_result_rows": len(rx_points) * len(tx_mesh_points) * len(heights_m),
        "receiver": {"count": len(rx_points), "height_m": 0.5, "pattern": "iso", "extra_gain_dbi": 0.0},
        "transmitter": {"mesh": "500 m square including boundaries", "points_per_height": len(tx_mesh_points), "pattern": "iso", "power_dbm": 20.0, "polarization": "V"},
        "frequency_hz": 2.437e9,
        "materials": {"building": "ITU concrete, thickness 0.3 m", "ground": "ITU concrete, thickness 0.3 m (shared source mesh)"},
        "rt": {"max_depth": args.max_depth, "samples_per_tx": args.samples_per_tx, "los": True, "specular_reflection": True, "diffuse_reflection": False, "refraction": True, "diffraction": True},
        "osm": {
            "retrieval": "preloaded randomized-height cache; OpenStreetMap API 0.6 with Overpass API fallback",
            "license": "ODbL",
            "raw_file": "osm_map.osm",
            "height_variant": args.osm_height_variant,
            "sha256": osm_sha256,
        },
        "satellite_imagery": {"enabled": False, "status": "skipped"},
        "files": {
            "buildings_2d": "osm_buildings_2d.png",
            "buildings_3d": "osm_buildings_3d.png",
            "rx_positions_2d": "rx_positions_2d.png",
            "rx_positions": "rx_positions.csv",
            "scene": str(scene_xml.relative_to(block_dir)),
        },
    }
    phase_started = time.perf_counter()
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    record_phase("write_prepared_metadata", phase_started)
    if args.skip_rt:
        if ENABLE_TIMING_LOGS:
            print(
                f"PREPARED {block.region_slug}/{block.block_id}: {len(buildings)} buildings, "
                f"heights={heights_m}; invocation_timing={time.perf_counter() - block_timing_started:.3f}s"
            )
        else:
            print(f"PREPARED {block.region_slug}/{block.block_id}: {len(buildings)} buildings, heights={heights_m}")
        return "prepared"

    config = make_block_config(block, heights_m, block_dir, args)
    phase_started = time.perf_counter()
    solve_summary = run_sionna_rt_multi_rx(
        config=config,
        scene_xml=scene_xml,
        out_csv=block_dir / "sionna_results.csv",
        tx_points=tx_mesh_points,
        rx_points=rx_points,
        heights_m=heights_m,
        mesh_batch_size=args.mesh_batch_size,
        samples_per_tx=args.samples_per_tx,
        max_depth=args.max_depth,
        region_slug=block.region_slug,
        block_id=block.block_id,
        environment_signature=f"{args.osm_height_variant}:{osm_sha256}",
        resume=args.resume,
    )
    record_phase("sionna_rt_total", phase_started)
    phase_started = time.perf_counter()
    write_dict_csv(solve_summary["height_summaries"], block_dir / "sionna_results_summary.csv")
    record_phase("write_rt_summary", phase_started)
    phase_started = time.perf_counter()
    rss_map_files = plot_rss_maps_by_rx(
        block,
        buildings,
        rx_points,
        heights_m,
        tx_mesh_points,
        block_dir / "sionna_results.csv",
        block_dir / "rss_maps_by_rx",
    )
    record_phase("plot_rss_maps_by_rx", phase_started)
    metadata["status"] = "complete"
    metadata["completed_utc"] = utc_now()
    metadata["actual_result_rows"] = int(solve_summary["row_count"])
    metadata["elapsed_seconds"] = round(time.time() - started, 3)
    metadata["files"]["results"] = "sionna_results.csv"
    metadata["files"]["results_summary"] = "sionna_results_summary.csv"
    metadata["files"]["rss_maps_by_rx"] = "rss_maps_by_rx"
    metadata["rss_map_file_count"] = len(rss_map_files)
    metadata["rss_plot_schema_version"] = 2
    phase_started = time.perf_counter()
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (block_dir / "sionna_results.csv.checkpoint.json").unlink(missing_ok=True)
    record_phase("finalize_metadata_and_checkpoint", phase_started)
    block_timing_seconds = time.perf_counter() - block_timing_started
    if ENABLE_TIMING_LOGS:
        print(
            f"BLOCK TIMING COMPLETE {block.region_slug}/{block.block_id}: "
            f"total={block_timing_seconds:.3f}s "
            + " ".join(f"{name}={seconds:.3f}s" for name, seconds in phase_timings.items()),
            flush=True,
        )
    print(f"COMPLETE {block.region_slug}/{block.block_id}: {metadata['actual_result_rows']} rows in {metadata['elapsed_seconds']} s")
    return "complete"


def process_voxel_block(block: BlockSpec, dataset_root: Path, args: argparse.Namespace) -> str:
    """Prepare and solve one 256 m block using the 3-D voxel contract."""
    started = time.time()
    block_dir = dataset_root / block.region_slug / block.block_id
    block_dir.mkdir(parents=True, exist_ok=True)
    osm_path = block_dir / "osm_map.osm"
    osm_bytes = download_block_osm(block, osm_path)
    osm_sha256 = hashlib.sha256(osm_bytes).hexdigest()
    metadata_path = block_dir / "metadata.json"
    voxel_shape = (args.voxel_nx, args.voxel_ny, args.voxel_nz)
    voxel_count = args.voxel_nx * args.voxel_ny * args.voxel_nz
    expected_rows = args.tx_count * voxel_count

    if args.resume and metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        settings_match = (
            existing.get("schema_version") == "3.0"
            and existing.get("block_size_m") == MAP_BLOCK_SIZE_M
            and existing.get("random_seed_global") == args.random_seed
            and existing.get("ground_tx_count") == args.tx_count
            and existing.get("ground_tx_height_m") == args.tx_height
            and existing.get("minimum_building_clearance_m") == args.tx_building_clearance
            and existing.get("voxel_shape_xyz") == list(voxel_shape)
            and existing.get("voxel_size_m") == args.voxel_size
            and existing.get("voxel_z_start_m") == args.voxel_z_start
            and existing.get("rx_batch_size") == args.rx_batch_size
            and existing.get("osm_height_variant") == args.osm_height_variant
            and existing.get("osm_sha256") == osm_sha256
            and existing.get("rt", {}).get("samples_per_tx") == args.samples_per_tx
            and existing.get("rt", {}).get("max_depth") == args.max_depth
        )
        desired_status = "prepared" if args.skip_rt else "complete"
        status_matches = (
            existing.get("status") == desired_status
            or (args.skip_rt and existing.get("status") == "complete")
        )
        results_ok = (
            args.skip_rt
            or (
                existing.get("actual_result_rows") == expected_rows
                and (block_dir / "sionna_results.csv").is_file()
            )
        )
        if settings_match and status_matches and results_ok:
            if existing.get("status") == "complete":
                (block_dir / "sionna_results.csv.checkpoint.json").unlink(missing_ok=True)
            print(f"SKIP {block.region_slug}/{block.block_id}: {existing.get('status')}")
            return "skipped"

    buildings = parse_block_buildings(osm_bytes, block)
    maximum_building_height_m = max([building.height_m for building in buildings] + [0.0])
    voxel_heights = [
        args.voxel_z_start + args.voxel_size * iz
        for iz in range(args.voxel_nz)
    ]
    plot_buildings_2d(block, buildings, block_dir / "osm_buildings_2d.png")
    plot_buildings_3d(block, buildings, block_dir / "osm_buildings_3d.png")
    print(f"SATELLITE SKIP {block.region_slug}/{block.block_id}: disabled")
    config = make_block_config(block, voxel_heights, block_dir, args)
    scene_xml = write_sionna_scene(config, buildings, block_dir / "scene_generated")
    tx_points, derived_seed, clearances = random_ground_tx_positions(
        block=block,
        buildings=buildings,
        count=args.tx_count,
        height_m=args.tx_height,
        global_seed=args.random_seed,
        minimum_clearance_m=args.tx_building_clearance,
    )
    write_tx_positions_csv(tx_points, clearances, block_dir / "tx_positions.csv")
    plot_tx_positions_2d(block, buildings, tx_points, block_dir / "tx_positions_2d.png")

    metadata = {
        "schema_version": "3.0",
        "layout": "fixed_ground_tx_to_3d_voxel_rx",
        "status": "prepared",
        "created_utc": utc_now(),
        "region_slug": block.region_slug,
        "region_name_zh": block.region_name_zh,
        "block_id": block.block_id,
        "block_center": {"lat": block.center_lat, "lon": block.center_lon},
        "block_bbox": {
            "min_lat": block.min_lat, "min_lon": block.min_lon,
            "max_lat": block.max_lat, "max_lon": block.max_lon,
        },
        "block_size_m": MAP_BLOCK_SIZE_M,
        "map_block_size_m": MAP_BLOCK_SIZE_M,
        "building_count": len(buildings),
        "maximum_building_height_m": maximum_building_height_m,
        "ground_tx_count": len(tx_points),
        "ground_tx_height_m": args.tx_height,
        "tx_position_method": "deterministic uniform random rejection sampling in the 256 m map block",
        "minimum_building_clearance_m": args.tx_building_clearance,
        "minimum_observed_building_clearance_m": (
            None if all(math.isinf(value) for value in clearances) else min(clearances)
        ),
        "random_seed_global": args.random_seed,
        "random_seed_derived": derived_seed,
        "voxel_shape_xyz": list(voxel_shape),
        "voxel_size_m": args.voxel_size,
        "voxel_count": voxel_count,
        "voxel_z_start_m": args.voxel_z_start,
        "voxel_z_values_m": voxel_heights,
        "voxel_horizontal_positions": "cell centers from -127 m to +127 m",
        "rx_batch_size": args.rx_batch_size,
        "expected_result_rows": expected_rows,
        "frequency_hz": 2.437e9,
        "transmitter": {
            "count": len(tx_points), "height_m": args.tx_height,
            "pattern": "iso", "power_dbm": 20.0, "polarization": "V",
        },
        "receiver": {
            "grid": f"{args.voxel_nx} x {args.voxel_ny} x {args.voxel_nz} voxel centers",
            "pattern": "iso", "polarization": "V",
        },
        "materials": {
            "building": "ITU concrete, thickness 0.3 m",
            "ground": "ITU concrete, thickness 0.3 m (shared source mesh)",
        },
        "rt": {
            "max_depth": args.max_depth,
            "samples_per_tx": args.samples_per_tx,
            "los": True, "specular_reflection": True,
            "diffuse_reflection": False, "refraction": True, "diffraction": True,
        },
        "osm_height_variant": args.osm_height_variant,
        "osm_sha256": osm_sha256,
        "osm": {
            "retrieval": "preloaded randomized-height cache",
            "license": "ODbL", "raw_file": "osm_map.osm",
            "height_variant": args.osm_height_variant, "sha256": osm_sha256,
        },
        "satellite_imagery": {"enabled": False, "status": "skipped"},
        "files": {
            "buildings_2d": "osm_buildings_2d.png",
            "buildings_3d": "osm_buildings_3d.png",
            "tx_positions_2d": "tx_positions_2d.png",
            "tx_positions": "tx_positions.csv",
            "scene": str(scene_xml.relative_to(block_dir)),
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.skip_rt:
        print(
            f"PREPARED {block.region_slug}/{block.block_id}: {len(buildings)} buildings, "
            f"TX={len(tx_points)}, voxels={voxel_count}"
        )
        return "prepared"

    voxel_points = voxel_rx_grid(
        block, args.voxel_size, args.voxel_nx, args.voxel_ny,
        args.voxel_nz, args.voxel_z_start,
    )
    solve_summary = run_sionna_rt_voxel_rx(
        config=config,
        scene_xml=scene_xml,
        out_csv=block_dir / "sionna_results.csv",
        tx_points=tx_points,
        voxel_points=voxel_points,
        rx_batch_size=args.rx_batch_size,
        samples_per_tx=args.samples_per_tx,
        max_depth=args.max_depth,
        region_slug=block.region_slug,
        block_id=block.block_id,
        environment_signature=f"{args.osm_height_variant}:{osm_sha256}",
        voxel_shape=voxel_shape,
        voxel_size_m=args.voxel_size,
        resume=args.resume,
    )
    write_dict_csv([solve_summary["summary"]], block_dir / "sionna_results_summary.csv")
    metadata["status"] = "complete"
    metadata["completed_utc"] = utc_now()
    metadata["actual_result_rows"] = int(solve_summary["row_count"])
    metadata["elapsed_seconds"] = round(time.time() - started, 3)
    metadata["files"]["results"] = "sionna_results.csv"
    metadata["files"]["results_summary"] = "sionna_results_summary.csv"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (block_dir / "sionna_results.csv.checkpoint.json").unlink(missing_ok=True)
    print(
        f"COMPLETE {block.region_slug}/{block.block_id}: "
        f"{solve_summary['row_count']} rows in {time.time() - started:.3f} s"
    )
    return "complete"


def _process_aerial_tx_voxel_block_monolithic(
    block: BlockSpec, dataset_root: Path, args: argparse.Namespace
) -> str:
    """Solve a 3-D aerial TX voxel grid against ten fixed ground receivers."""
    started = time.time()
    block_dir = dataset_root / block.region_slug / block.block_id
    block_dir.mkdir(parents=True, exist_ok=True)
    osm_path = block_dir / "osm_map.osm"
    osm_bytes = download_block_osm(block, osm_path)
    osm_sha256 = hashlib.sha256(osm_bytes).hexdigest()
    metadata_path = block_dir / "metadata.json"
    voxel_shape = (args.voxel_nx, args.voxel_ny, args.voxel_nz)
    tx_points_per_height = args.voxel_nx * args.voxel_ny
    tx_voxel_count = tx_points_per_height * args.voxel_nz
    expected_rows = args.rx_count * tx_voxel_count

    if args.resume and metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        settings_match = (
            existing.get("schema_version") in {"3.1", "3.2"}
            and existing.get("layout") == "3d_voxel_tx_to_fixed_ground_rx"
            and existing.get("block_size_m") == MAP_BLOCK_SIZE_M
            and existing.get("random_seed_global") == args.random_seed
            and existing.get("ground_rx_count") == args.rx_count
            and existing.get("ground_rx_height_m") == args.rx_height
            and existing.get("minimum_building_clearance_m") == args.rx_building_clearance
            and existing.get("tx_voxel_shape_xyz") == list(voxel_shape)
            and existing.get("voxel_size_m") == args.voxel_size
            and existing.get("voxel_z_start_m") == args.voxel_z_start
            and existing.get("osm_height_variant") == args.osm_height_variant
            and existing.get("osm_sha256") == osm_sha256
            and existing.get("rt", {}).get("samples_per_tx") == args.samples_per_tx
            and existing.get("rt", {}).get("max_depth") == args.max_depth
        )
        desired_status = "prepared" if args.skip_rt else "complete"
        status_matches = (
            existing.get("status") == desired_status
            or (args.skip_rt and existing.get("status") == "complete")
        )
        results_ok = (
            args.skip_rt
            or (
                existing.get("actual_result_rows") == expected_rows
                and (block_dir / "sionna_results.csv").is_file()
            )
        )
        if settings_match and status_matches and results_ok:
            if existing.get("status") == "complete" and args.render_preview_after_block:
                try:
                    preview_status, preview_written = render_voxel_rss_preview(
                        block_dir / "sionna_results.csv",
                        RSS_PREVIEW_DIR_NAME,
                        None,
                        None,
                        False,
                        show_progress=False,
                    )
                    existing.setdefault("files", {})["rss_preview_layers"] = RSS_PREVIEW_DIR_NAME
                    existing["rss_preview"] = {
                        "status": preview_status,
                        "image_count": RSS_PREVIEW_IMAGE_COUNT,
                        "new_images_written": preview_written,
                        "building_footprints": "black",
                        "no_resolved_path_cells": "gray",
                    }
                except Exception as exc:
                    existing["rss_preview"] = {
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    print(
                        f"PREVIEW WARNING {block.region_slug}/{block.block_id}: {exc}",
                        flush=True,
                    )
                metadata_path.write_text(
                    json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            if existing.get("status") == "complete":
                (block_dir / "sionna_results.csv.checkpoint.json").unlink(missing_ok=True)
            print(f"SKIP {block.region_slug}/{block.block_id}: {existing.get('status')}")
            return "skipped"

    buildings = parse_block_buildings(osm_bytes, block)
    maximum_building_height_m = max([building.height_m for building in buildings] + [0.0])
    tx_heights = [
        args.voxel_z_start + args.voxel_size * iz
        for iz in range(args.voxel_nz)
    ]
    plot_buildings_2d(block, buildings, block_dir / "osm_buildings_2d.png")
    plot_buildings_3d(block, buildings, block_dir / "osm_buildings_3d.png")
    print(f"SATELLITE SKIP {block.region_slug}/{block.block_id}: disabled")
    config = make_block_config(block, tx_heights, block_dir, args)
    scene_xml = write_sionna_scene(config, buildings, block_dir / "scene_generated")
    rx_points, derived_seed, clearances = random_rx_positions(
        block=block,
        buildings=buildings,
        count=args.rx_count,
        global_seed=args.random_seed,
        minimum_clearance_m=args.rx_building_clearance,
    )
    tx_xy_points = voxel_tx_horizontal_grid(
        block, args.voxel_size, args.voxel_nx, args.voxel_ny
    )
    write_rx_positions_csv(rx_points, clearances, block_dir / "rx_positions.csv")
    plot_rx_positions_2d(block, buildings, rx_points, block_dir / "rx_positions_2d.png")

    metadata = {
        "schema_version": "3.2",
        "layout": "3d_voxel_tx_to_fixed_ground_rx",
        "results_csv_schema": "compact_rss_5col_v1",
        "results_csv_columns": [
            "rx_index", "tx_voxel_ix", "tx_voxel_iy", "tx_voxel_iz", "rss_dbm",
        ],
        "status": "prepared",
        "created_utc": utc_now(),
        "region_slug": block.region_slug,
        "region_name_zh": block.region_name_zh,
        "block_id": block.block_id,
        "block_center": {"lat": block.center_lat, "lon": block.center_lon},
        "block_bbox": {
            "min_lat": block.min_lat, "min_lon": block.min_lon,
            "max_lat": block.max_lat, "max_lon": block.max_lon,
        },
        "block_size_m": MAP_BLOCK_SIZE_M,
        "map_block_size_m": MAP_BLOCK_SIZE_M,
        "building_count": len(buildings),
        "maximum_building_height_m": maximum_building_height_m,
        "ground_rx_count": len(rx_points),
        "ground_rx_height_m": args.rx_height,
        "rx_position_method": "deterministic uniform random rejection sampling in the 256 m map block",
        "minimum_building_clearance_m": args.rx_building_clearance,
        "minimum_observed_building_clearance_m": (
            None if all(math.isinf(value) for value in clearances) else min(clearances)
        ),
        "random_seed_global": args.random_seed,
        "random_seed_derived": derived_seed,
        "tx_voxel_shape_xyz": list(voxel_shape),
        "voxel_size_m": args.voxel_size,
        "tx_voxel_count": tx_voxel_count,
        "tx_points_per_height": tx_points_per_height,
        "voxel_z_start_m": args.voxel_z_start,
        "tx_heights_m": tx_heights,
        "tx_voxel_horizontal_positions": "cell centers from -127 m to +127 m",
        "tx_batch_size": args.tx_batch_size,
        "expected_result_rows": expected_rows,
        "frequency_hz": 2.437e9,
        "transmitter": {
            "grid": f"{args.voxel_nx} x {args.voxel_ny} x {args.voxel_nz} aerial voxel centers",
            "pattern": "iso", "power_dbm": 20.0, "polarization": "V",
        },
        "receiver": {
            "count": len(rx_points), "height_m": args.rx_height,
            "pattern": "iso", "polarization": "V",
        },
        "materials": {
            "building": "ITU concrete, thickness 0.3 m",
            "ground": "ITU concrete, thickness 0.3 m (shared source mesh)",
        },
        "rt": {
            "max_depth": args.max_depth,
            "samples_per_tx": args.samples_per_tx,
            "los": True, "specular_reflection": True,
            "diffuse_reflection": False, "refraction": True, "diffraction": True,
        },
        "osm_height_variant": args.osm_height_variant,
        "osm_sha256": osm_sha256,
        "osm": {
            "retrieval": "preloaded randomized-height cache",
            "license": "ODbL", "raw_file": "osm_map.osm",
            "height_variant": args.osm_height_variant, "sha256": osm_sha256,
        },
        "satellite_imagery": {"enabled": False, "status": "skipped"},
        "files": {
            "buildings_2d": "osm_buildings_2d.png",
            "buildings_3d": "osm_buildings_3d.png",
            "rx_positions_2d": "rx_positions_2d.png",
            "rx_positions": "rx_positions.csv",
            "scene": str(scene_xml.relative_to(block_dir)),
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.skip_rt:
        print(
            f"PREPARED {block.region_slug}/{block.block_id}: {len(buildings)} buildings, "
            f"ground RX={len(rx_points)}, aerial TX voxels={tx_voxel_count}"
        )
        return "prepared"

    solve_summary = run_sionna_rt_multi_rx(
        config=config,
        scene_xml=scene_xml,
        out_csv=block_dir / "sionna_results.csv",
        tx_points=tx_xy_points,
        rx_points=rx_points,
        heights_m=tx_heights,
        mesh_batch_size=args.tx_batch_size,
        samples_per_tx=args.samples_per_tx,
        max_depth=args.max_depth,
        region_slug=block.region_slug,
        block_id=block.block_id,
        environment_signature=f"{args.osm_height_variant}:{osm_sha256}",
        resume=args.resume,
    )
    write_dict_csv(
        solve_summary["height_summaries"], block_dir / "sionna_results_summary.csv"
    )
    if args.render_preview_after_block:
        try:
            preview_status, preview_written = render_voxel_rss_preview(
                block_dir / "sionna_results.csv",
                RSS_PREVIEW_DIR_NAME,
                None,
                None,
                False,
                show_progress=False,
            )
            metadata["files"]["rss_preview_layers"] = RSS_PREVIEW_DIR_NAME
            metadata["rss_preview"] = {
                "status": preview_status,
                "image_count": RSS_PREVIEW_IMAGE_COUNT,
                "new_images_written": preview_written,
                "building_footprints": "black",
                "no_resolved_path_cells": "gray",
            }
            print(
                f"PREVIEW READY {block.region_slug}/{block.block_id}: "
                f"{RSS_PREVIEW_IMAGE_COUNT} images ({preview_written} newly written)",
                flush=True,
            )
        except Exception as exc:
            metadata["rss_preview"] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(
                f"PREVIEW WARNING {block.region_slug}/{block.block_id}: {exc}",
                flush=True,
            )
    metadata["status"] = "complete"
    metadata["completed_utc"] = utc_now()
    metadata["actual_result_rows"] = int(solve_summary["row_count"])
    metadata["tx_batch_sizes_used"] = list(
        solve_summary.get("mesh_batch_sizes_used", [args.tx_batch_size])
    )
    metadata["tx_batch_size"] = metadata["tx_batch_sizes_used"][-1]
    metadata["elapsed_seconds"] = round(time.time() - started, 3)
    metadata["files"]["results"] = "sionna_results.csv"
    metadata["files"]["results_summary"] = "sionna_results_summary.csv"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (block_dir / "sionna_results.csv.checkpoint.json").unlink(missing_ok=True)
    print(
        f"COMPLETE {block.region_slug}/{block.block_id}: "
        f"{solve_summary['row_count']} rows in {time.time() - started:.3f} s"
    )
    return "complete"


def process_aerial_tx_voxel_block(
    block: BlockSpec, dataset_root: Path, args: argparse.Namespace
) -> str:
    """Solve extensible fixed ground RXs and store one dense NPY per RX."""
    started = time.time()
    block_dir = dataset_root / block.region_slug / block.block_id
    block_dir.mkdir(parents=True, exist_ok=True)
    osm_path = block_dir / "osm_map.osm"
    osm_bytes = download_block_osm(block, osm_path)
    osm_sha256 = hashlib.sha256(osm_bytes).hexdigest()
    metadata_path = block_dir / "metadata.json"
    voxel_shape = (args.voxel_nx, args.voxel_ny, args.voxel_nz)
    tx_points_per_height = args.voxel_nx * args.voxel_ny
    tx_voxel_count = tx_points_per_height * args.voxel_nz
    expected_values = args.rx_count * tx_voxel_count
    buildings = parse_block_buildings(osm_bytes, block)
    maximum_building_height_m = max([building.height_m for building in buildings] + [0.0])
    tx_heights = [
        args.voxel_z_start + args.voxel_size * iz
        for iz in range(args.voxel_nz)
    ]
    rx_points, derived_seed, clearances = random_rx_positions(
        block=block,
        buildings=buildings,
        count=args.rx_count,
        global_seed=args.random_seed,
        minimum_clearance_m=args.rx_building_clearance,
    )
    tx_xy_points = voxel_tx_horizontal_grid(
        block, args.voxel_size, args.voxel_nx, args.voxel_ny
    )

    physical_signature: dict[str, object] = {
        "schema_version": 1,
        "layout": "3d_voxel_tx_to_fixed_ground_rx",
        "region_slug": block.region_slug,
        "block_id": block.block_id,
        "environment_signature": f"{args.osm_height_variant}:{osm_sha256}",
        "random_seed_global": args.random_seed,
        "rx_height_m": args.rx_height,
        "rx_building_clearance_m": args.rx_building_clearance,
        "tx_voxel_shape_xyz": list(voxel_shape),
        "voxel_size_m": args.voxel_size,
        "voxel_z_start_m": args.voxel_z_start,
        "heights_m": [float(value) for value in tx_heights],
        "tx_count_per_height": len(tx_xy_points),
        "samples_per_tx": args.samples_per_tx,
        "max_depth": args.max_depth,
        "frequency_hz": 2.437e9,
    }
    desired_signatures = {
        point.index: desired_rx_signature(
            rx_index=point.index,
            x_m=point.x_m,
            y_m=point.y_m,
            physical_signature=physical_signature,
        )
        for point in rx_points
    }

    existing: dict[str, object] = {}
    if metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        old_count = int(existing.get("ground_rx_count", 0) or 0)
        if old_count and args.rx_count < old_count:
            raise RuntimeError(
                f"RX shrink is not allowed in-place for {block.region_slug}/{block.block_id}: "
                f"existing={old_count}, requested={args.rx_count}. Use a new dataset root."
            )
    existing_common_match = (
        existing.get("schema_version") == "3.4"
        and existing.get("results_storage") == "per_rx_npy_tensors_v1"
        and existing.get("physical_signature") == physical_signature
    )
    result_files_exist = sharded_results_dir(block_dir).is_dir() and any(
        sharded_results_dir(block_dir).glob("rx_*.npy")
    )
    if result_files_exist and not existing_common_match:
        raise RuntimeError(
            f"Existing RX shards for {block.region_slug}/{block.block_id} use different "
            "physical settings. Change DATASET_NAME instead of mixing them."
        )

    expected_shape_zyx = (args.voxel_nz, args.voxel_ny, args.voxel_nx)
    valid_entries = load_valid_rx_entries(
        block_dir, desired_signatures, expected_shape_zyx
    ) if existing_common_match else []
    completed_indices = completed_rx_indices(valid_entries)
    all_requested_complete = completed_indices == set(range(args.rx_count))
    if args.resume and existing_common_match and all_requested_complete:
        if args.render_preview_after_block and not args.skip_rt:
            try:
                preview_status, preview_written = render_voxel_rss_preview(
                    block_dir,
                    RSS_PREVIEW_DIR_NAME,
                    None,
                    None,
                    False,
                    show_progress=False,
                )
                existing.setdefault("files", {})["rss_preview_layers"] = RSS_PREVIEW_DIR_NAME
                existing["rss_preview"] = {
                    "status": preview_status,
                    "image_count": args.rx_count * args.voxel_nz,
                    "new_images_written": preview_written,
                    "building_footprints": "black",
                    "no_resolved_path_cells": "gray",
                }
            except Exception as exc:
                existing["rss_preview"] = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(f"PREVIEW WARNING {block.region_slug}/{block.block_id}: {exc}", flush=True)
            metadata_path.write_text(
                json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        print(f"SKIP {block.region_slug}/{block.block_id}: complete RX tensors={args.rx_count}")
        return "skipped"

    if existing_common_match and (block_dir / "rx_positions.csv").is_file():
        previous_points = read_rx_positions_csv(block_dir / "rx_positions.csv")
        for previous, desired in zip(previous_points, rx_points):
            # Older cluster snapshots serialized XY with only three decimals.
            # The helper accepts their maximum 0.5 mm rounding error; new
            # files are written with six decimals above.
            if not rx_prefix_point_matches(previous, desired):
                raise RuntimeError(
                    f"Deterministic RX prefix changed at index {desired.index} in "
                    f"{block.region_slug}/{block.block_id}"
                )

    plot_buildings_2d(block, buildings, block_dir / "osm_buildings_2d.png")
    plot_buildings_3d(block, buildings, block_dir / "osm_buildings_3d.png")
    print(f"SATELLITE SKIP {block.region_slug}/{block.block_id}: disabled")
    config = make_block_config(block, tx_heights, block_dir, args)
    scene_xml = write_sionna_scene(config, buildings, block_dir / "scene_generated")
    write_rx_positions_csv(rx_points, clearances, block_dir / "rx_positions.csv")
    plot_rx_positions_2d(block, buildings, rx_points, block_dir / "rx_positions_2d.png")

    metadata: dict[str, object] = {
        "schema_version": "3.4",
        "layout": "3d_voxel_tx_to_fixed_ground_rx",
        "results_storage": "per_rx_npy_tensors_v1",
        "results_tensor_schema": "rss_float32_zyx_v1",
        "results_tensor_dtype": "float32",
        "results_tensor_axes": ["tx_voxel_z", "tx_voxel_y", "tx_voxel_x"],
        "results_tensor_shape_zyx": list(expected_shape_zyx),
        "status": "prepared" if not completed_indices else "expanding",
        "created_utc": existing.get("created_utc", utc_now()),
        "updated_utc": utc_now(),
        "region_slug": block.region_slug,
        "region_name_zh": block.region_name_zh,
        "block_id": block.block_id,
        "block_center": {"lat": block.center_lat, "lon": block.center_lon},
        "block_bbox": {
            "min_lat": block.min_lat, "min_lon": block.min_lon,
            "max_lat": block.max_lat, "max_lon": block.max_lon,
        },
        "block_size_m": MAP_BLOCK_SIZE_M,
        "map_block_size_m": MAP_BLOCK_SIZE_M,
        "building_count": len(buildings),
        "maximum_building_height_m": maximum_building_height_m,
        "ground_rx_count": len(rx_points),
        "completed_rx_count_before_run": len(completed_indices),
        "rx_solver_group_size": args.rx_solver_group_size,
        "ground_rx_height_m": args.rx_height,
        "rx_position_method": "deterministic prefix-stable uniform random rejection sampling",
        "minimum_building_clearance_m": args.rx_building_clearance,
        "minimum_observed_building_clearance_m": (
            None if all(math.isinf(value) for value in clearances) else min(clearances)
        ),
        "random_seed_global": args.random_seed,
        "random_seed_derived": derived_seed,
        "tx_voxel_shape_xyz": list(voxel_shape),
        "voxel_size_m": args.voxel_size,
        "tx_voxel_count": tx_voxel_count,
        "tx_points_per_height": tx_points_per_height,
        "voxel_z_start_m": args.voxel_z_start,
        "tx_heights_m": tx_heights,
        "tx_voxel_horizontal_positions": "cell centers from -127 m to +127 m",
        "tx_batch_size": args.tx_batch_size,
        "expected_result_values": expected_values,
        "expected_values_per_rx_tensor": tx_voxel_count,
        "expected_bytes_per_rx_tensor_payload": tx_voxel_count * 4,
        "physical_signature": physical_signature,
        "frequency_hz": 2.437e9,
        "transmitter": {
            "grid": f"{args.voxel_nx} x {args.voxel_ny} x {args.voxel_nz} aerial voxel centers",
            "pattern": "iso", "power_dbm": 20.0, "polarization": "V",
        },
        "receiver": {
            "count": len(rx_points), "height_m": args.rx_height,
            "pattern": "iso", "polarization": "V",
        },
        "materials": {
            "building": "ITU concrete, thickness 0.3 m",
            "ground": "ITU concrete, thickness 0.3 m (shared source mesh)",
        },
        "rt": {
            "max_depth": args.max_depth,
            "samples_per_tx": args.samples_per_tx,
            "los": True, "specular_reflection": True,
            "diffuse_reflection": False, "refraction": True, "diffraction": True,
        },
        "osm_height_variant": args.osm_height_variant,
        "osm_sha256": osm_sha256,
        "osm": {
            "retrieval": "preloaded randomized-height cache",
            "license": "ODbL", "raw_file": "osm_map.osm",
            "height_variant": args.osm_height_variant, "sha256": osm_sha256,
        },
        "satellite_imagery": {"enabled": False, "status": "skipped"},
        "files": {
            "buildings_2d": "osm_buildings_2d.png",
            "buildings_3d": "osm_buildings_3d.png",
            "rx_positions_2d": "rx_positions_2d.png",
            "rx_positions": "rx_positions.csv",
            "scene": str(scene_xml.relative_to(block_dir)),
            "results": RESULTS_DIR_NAME,
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.skip_rt:
        print(
            f"PREPARED {block.region_slug}/{block.block_id}: {len(buildings)} buildings, "
            f"target RX={len(rx_points)}, completed RX tensors={len(completed_indices)}"
        )
        return "prepared"

    missing_points = [point for point in rx_points if point.index not in completed_indices]
    batch_sizes_used = list(existing.get("tx_batch_sizes_used", []))
    for group_start in range(0, len(missing_points), args.rx_solver_group_size):
        group_points = missing_points[group_start:group_start + args.rx_solver_group_size]
        group_indices = [point.index for point in group_points]
        print(
            f"RX EXPANSION {block.region_slug}/{block.block_id}: solving {group_indices}; "
            f"already_complete={len(completed_indices)}/{args.rx_count}",
            flush=True,
        )
        solve_summary = run_sionna_rt_multi_rx_npy(
            config=config,
            scene_xml=scene_xml,
            block_dir=block_dir,
            tx_points=tx_xy_points,
            rx_points=group_points,
            heights_m=tx_heights,
            tensor_shape_zyx=expected_shape_zyx,
            mesh_batch_size=args.tx_batch_size,
            samples_per_tx=args.samples_per_tx,
            max_depth=args.max_depth,
            region_slug=block.region_slug,
            block_id=block.block_id,
            environment_signature=f"{args.osm_height_variant}:{osm_sha256}",
            resume=args.resume,
        )
        commit_rx_group(
            block_dir=block_dir,
            rx_signatures={index: desired_signatures[index] for index in group_indices},
            expected_shape_zyx=expected_shape_zyx,
            per_rx_height_summaries=solve_summary["per_rx_height_summaries"],
            target_rx_count=args.rx_count,
        )
        for value in solve_summary.get("mesh_batch_sizes_used", [args.tx_batch_size]):
            if value not in batch_sizes_used:
                batch_sizes_used.append(value)
        group_checkpoint_path(block_dir, group_indices).unlink(missing_ok=True)
        completed_indices.update(group_indices)

    valid_entries = load_valid_rx_entries(block_dir, desired_signatures, expected_shape_zyx)
    completed_indices = completed_rx_indices(valid_entries)
    if completed_indices != set(range(args.rx_count)):
        raise RuntimeError(
            f"RX tensor commit incomplete for {block.region_slug}/{block.block_id}: "
            f"completed={sorted(completed_indices)}, expected=0..{args.rx_count - 1}"
        )
    results_index = sharded_results_dir(block_dir) / "index.json"
    height_summaries = aggregate_height_summaries(valid_entries, tx_heights)
    write_dict_csv(height_summaries, block_dir / "sionna_results_summary.csv")

    if args.render_preview_after_block:
        try:
            preview_status, preview_written = render_voxel_rss_preview(
                block_dir,
                RSS_PREVIEW_DIR_NAME,
                None,
                None,
                False,
                show_progress=False,
            )
            metadata["files"]["rss_preview_layers"] = RSS_PREVIEW_DIR_NAME
            metadata["rss_preview"] = {
                "status": preview_status,
                "image_count": args.rx_count * args.voxel_nz,
                "new_images_written": preview_written,
                "building_footprints": "black",
                "no_resolved_path_cells": "gray",
            }
            print(
                f"PREVIEW READY {block.region_slug}/{block.block_id}: "
                f"{args.rx_count * args.voxel_nz} images ({preview_written} newly written)",
                flush=True,
            )
        except Exception as exc:
            metadata["rss_preview"] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"PREVIEW WARNING {block.region_slug}/{block.block_id}: {exc}", flush=True)

    metadata["status"] = "complete"
    metadata["completed_utc"] = utc_now()
    metadata["completed_rx_count"] = len(completed_indices)
    metadata["actual_result_values"] = expected_values
    metadata["result_tensor_count"] = args.rx_count
    metadata["tx_batch_sizes_used"] = batch_sizes_used or [args.tx_batch_size]
    metadata["tx_batch_size"] = metadata["tx_batch_sizes_used"][-1]
    metadata["elapsed_seconds"] = round(time.time() - started, 3)
    metadata["files"]["results_index"] = str(results_index.relative_to(block_dir))
    metadata["files"]["results_summary"] = "sionna_results_summary.csv"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"COMPLETE {block.region_slug}/{block.block_id}: "
        f"{expected_values} values across {args.rx_count} RX tensors in "
        f"{time.time() - started:.3f} s"
    )
    return "complete"


def prepare_manifest(dataset_root: Path, selected_regions: set[str] | None = None) -> list[BlockSpec]:
    blocks: list[BlockSpec] = []
    for region in REGIONS:
        if selected_regions and region.slug not in selected_regions:
            continue
        region_dir = dataset_root / region.slug
        feature = fetch_region_boundary(region, region_dir / "region_boundary.geojson")
        region_blocks = generate_region_blocks(region, feature)
        blocks.extend(region_blocks)
        print(f"REGION {region.slug}: {len(region_blocks)} intersecting {MAP_BLOCK_SIZE_M:.0f} m blocks")

    dataset_root.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset_root / "blocks_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(blocks[0]).keys()) if blocks else ["region_slug", "block_id"])
        writer.writeheader()
        writer.writerows(asdict(block) for block in blocks)
    return blocks


def load_blocks_manifest(path: Path) -> list[BlockSpec]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing prepared manifest: {path}")
    blocks: list[BlockSpec] = []
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            blocks.append(BlockSpec(
                region_slug=row["region_slug"],
                region_name_zh=row["region_name_zh"],
                block_id=row["block_id"],
                center_lat=float(row["center_lat"]),
                center_lon=float(row["center_lon"]),
                min_lat=float(row["min_lat"]),
                min_lon=float(row["min_lon"]),
                max_lat=float(row["max_lat"]),
                max_lon=float(row["max_lon"]),
                size_m=float(row.get("size_m") or MAP_BLOCK_SIZE_M),
            ))
    return blocks


def write_dataset_metadata(dataset_root: Path, blocks: list[BlockSpec], args: argparse.Namespace) -> None:
    included_slugs = {block.region_slug for block in blocks}
    included_regions = [region for region in REGIONS if region.slug in included_slugs]
    counts = {
        region.slug: sum(block.region_slug == region.slug for block in blocks)
        for region in included_regions
    }
    metadata = {
        "schema_version": "3.4",
        "dataset_name": dataset_root.name,
        "generated_utc": utc_now(),
        "dataset_root": str(dataset_root),
        "description": "OSM campus geometry and Sionna RT 3-D aerial-TX voxel simulations",
        "osm_height_variant": args.osm_height_variant,
        "block_size_m": MAP_BLOCK_SIZE_M,
        "map_block_size_m": MAP_BLOCK_SIZE_M,
        "block_count": len(blocks),
        "region_block_counts": counts,
        "regions": [asdict(region) for region in included_regions],
        "results_storage": "per_rx_npy_tensors_v1",
        "results_directory": RESULTS_DIR_NAME,
        "results_tensor_schema": "rss_float32_zyx_v1",
        "results_tensor_dtype": "float32",
        "results_tensor_axes": ["tx_voxel_z", "tx_voxel_y", "tx_voxel_x"],
        "simulation_defaults": {
            "ground_rx_count": args.rx_count,
            "rx_solver_group_size": args.rx_solver_group_size,
            "ground_rx_height_m": args.rx_height,
            "rx_position_method": "deterministic uniform random rejection sampling",
            "random_seed_global": args.random_seed,
            "minimum_building_clearance_m": args.rx_building_clearance,
            "tx_voxel_shape_xyz": [args.voxel_nx, args.voxel_ny, args.voxel_nz],
            "voxel_size_m": args.voxel_size,
            "voxel_z_start_m": args.voxel_z_start,
            "tx_batch_size": args.tx_batch_size,
            "frequency_hz": 2.437e9,
            "tx_power_dbm": 20.0,
            "antenna_pattern": "iso",
            "max_depth": args.max_depth,
            "samples_per_tx": args.samples_per_tx,
            "render_preview_after_block": args.render_preview_after_block,
        },
        "satellite_imagery": {"enabled": False, "status": "skipped"},
        "software": {"python": sys.version, "platform": platform.platform()},
        "license_notes": {"osm": "OpenStreetMap data © contributors, ODbL", "generated_dataset": "Set project-specific license before distribution"},
    }
    (dataset_root / "dataset_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def write_report(dataset_root: Path, blocks: list[BlockSpec]) -> None:
    statuses: dict[str, int] = {}
    result_rows = 0
    for block in blocks:
        metadata_path = dataset_root / block.region_slug / block.block_id / "metadata.json"
        status = "missing"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            status = str(metadata.get("status", "unknown"))
            result_rows += int(metadata.get("actual_result_rows", 0))
        statuses[status] = statuses.get(status, 0) + 1
    lines = [
        "# OSM 校园 Sionna RT 数据集结果报告",
        "",
        f"- 数据集路径：`{dataset_root}`",
        f"- OSM/建筑分块：{MAP_BLOCK_SIZE_M:.0f} m × {MAP_BLOCK_SIZE_M:.0f} m",
        f"- 空中 TX 网格范围：{TX_MESH_SIZE_M:.0f} m × {TX_MESH_SIZE_M:.0f} m",
        f"- 清单分块数：{len(blocks)}",
        f"- 状态统计：`{json.dumps(statuses, ensure_ascii=False)}`",
        f"- 已生成 Sionna 逐点结果行数：{result_rows}",
        "",
        "## 区域",
        "",
    ]
    for region in REGIONS:
        lines.append(f"- {region.name_zh} (`{region.slug}`)：{sum(block.region_slug == region.slug for block in blocks)} 个分块")
    lines += [
        "",
        "## 单分块必需文件",
        "",
        "- `metadata.json`",
        "- `osm_map.osm`",
        "- `osm_buildings_2d.png`",
        "- `osm_buildings_3d.png`",
        "- `rx_positions_2d.png`",
        "- `rx_positions.csv`",
        "- `rss_maps_by_rx/`（每个 RX 一张五高度子图）",
        "- `scene_generated/`",
        "- `sionna_results.csv`（状态为 complete 时）",
        "",
        f"报告更新时间：{utc_now()}",
    ]
    lines += ["", "根目录运行日志：`generation.log`；未完成区块保留批次级 checkpoint。"]
    text = "\n".join(lines) + "\n"
    (dataset_root / "RESULTS_REPORT.md").write_text(text, encoding="utf-8")
    Path("CAMPUS_SIONNA_DATASET_RESULTS.md").write_text(text, encoding="utf-8")


def write_voxel_report(dataset_root: Path, blocks: list[BlockSpec]) -> None:
    statuses: dict[str, int] = {}
    result_rows = 0
    for block in blocks:
        metadata_path = dataset_root / block.region_slug / block.block_id / "metadata.json"
        status = "missing"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            status = str(metadata.get("status", "unknown"))
            result_rows += int(metadata.get("actual_result_rows", 0))
        statuses[status] = statuses.get(status, 0) + 1
    lines = [
        "# 深圳高校 Sionna RT 三维体素数据集状态",
        "",
        f"- 数据集目录：`{dataset_root}`",
        f"- 地图分块：{MAP_BLOCK_SIZE_M:.0f} m × {MAP_BLOCK_SIZE_M:.0f} m",
        "- 接收体素：128 × 128 × 40，体素边长 2 m",
        "- 体素高度：2–80 m",
        "- 每块随机地面 RX：10",
        f"- block 总数：{len(blocks)}",
        f"- 状态：`{json.dumps(statuses, ensure_ascii=False)}`",
        f"- 已生成结果行数：{result_rows}",
        "",
        "## 学校",
        "",
    ]
    for region in REGIONS:
        count = sum(block.region_slug == region.slug for block in blocks)
        lines.append(f"- {region.name_en} (`{region.slug}`)：{count} blocks")
    lines += [
        "",
        "## 单个 block 的主要文件",
        "",
        "- `metadata.json`",
        "- `osm_map.osm`",
        "- `osm_buildings_2d.png` / `osm_buildings_3d.png`",
        "- `rx_positions.csv` / `rx_positions_2d.png`",
        "- `scene_generated/`",
        "- `sionna_results.csv`（完成后）",
        "- `sionna_results.csv.checkpoint.json`（未完成时）",
        "",
        f"更新时间：{utc_now()}",
    ]
    text = "\n".join(lines) + "\n"
    (dataset_root / "RESULTS_REPORT.md").write_text(text, encoding="utf-8")
    Path("CAMPUS_SIONNA_DATASET_RESULTS.md").write_text(text, encoding="utf-8")


def write_sharded_voxel_report(dataset_root: Path, blocks: list[BlockSpec]) -> None:
    """Write a concise report for the extensible per-RX tensor layout."""
    statuses: dict[str, int] = {}
    result_values = 0
    completed_rx_tensors = 0
    for block in blocks:
        metadata_path = dataset_root / block.region_slug / block.block_id / "metadata.json"
        status = "missing"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            status = str(metadata.get("status", "unknown"))
            result_values += int(metadata.get("actual_result_values", 0) or 0)
            completed_rx_tensors += int(metadata.get("completed_rx_count", 0) or 0)
        statuses[status] = statuses.get(status, 0) + 1
    lines = [
        "# 高校 Sionna RT 三维体素数据集状态",
        "",
        f"- 数据集目录：`{dataset_root}`",
        f"- 地图分块：{MAP_BLOCK_SIZE_M:.0f} m × {MAP_BLOCK_SIZE_M:.0f} m",
        "- 空中 TX 体素：128 × 128 × 40，体素边长 2 m，高度 2–80 m",
        "- 地面 RX：确定性随机位置，可在原目录中向上扩容",
        f"- block 总数：{len(blocks)}",
        f"- 状态：`{json.dumps(statuses, ensure_ascii=False)}`",
        f"- 已提交 RX NPY 张量：{completed_rx_tensors}",
        f"- 已生成 RSS 数值：{result_values}",
        "",
        "## 单个 block 的主要文件",
        "",
        "- `metadata.json`",
        "- `osm_map.osm`",
        "- `osm_buildings_2d.png` / `osm_buildings_3d.png`",
        "- `rx_positions.csv` / `rx_positions_2d.png`",
        "- `scene_generated/`",
        "- `sionna_results_by_rx/rx_NNN.npy` (float32 `[Z,Y,X]`)",
        "- `sionna_results_by_rx/index.json`",
        "- `sionna_results_by_rx/.work/*.checkpoint.json` (incomplete only)",
        "- `rss_maps_128px/rx_NNN/*.png`（启用自动预览时）",
        "",
        f"更新时间：{utc_now()}",
    ]
    text = "\n".join(lines) + "\n"
    (dataset_root / "RESULTS_REPORT.md").write_text(text, encoding="utf-8")
    Path("CAMPUS_SIONNA_DATASET_RESULTS.md").write_text(text, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--region", action="append", choices=[region.slug for region in REGIONS], help="Limit to one or more regions.")
    parser.add_argument("--block", action="append", help="Limit to one or more block IDs (center_lat_center_lon).")
    parser.add_argument("--prepare-only", action="store_true", help="Only create region boundaries and block manifest.")
    parser.add_argument("--prefetch-only", action="store_true", help="Prepare the manifest and cache every selected OSM block, then exit before RT. Satellite imagery is skipped.")
    parser.add_argument("--worker-mode", action="store_true", help="Internal multi-GPU mode: read the prepared manifest and avoid shared root report writes.")
    parser.add_argument("--skip-prefetch", action="store_true", help="Internal worker option: trust the parent prefetch phase.")
    parser.add_argument("--log-name", default="generation.log", help="Log filename inside the dataset root.")
    parser.add_argument("--skip-rt", action="store_true", help="Generate OSM, figures, random ground RX sites, scene and metadata without RT results.")
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume checkpoints and skip compatible completed blocks (default).",
    )
    run_mode.add_argument(
        "--restart",
        dest="resume",
        action="store_false",
        help="Explicitly recompute selected blocks from the beginning.",
    )
    parser.add_argument("--max-blocks", type=int, default=0)
    parser.add_argument("--rx-count", type=int, default=DEFAULT_RX_COUNT)
    parser.add_argument(
        "--rx-solver-group-size",
        type=int,
        default=DEFAULT_RX_SOLVER_GROUP_SIZE,
        help=(
            "Number of missing RXs solved together before results are committed "
            "as one NPY tensor per RX. This affects speed/VRAM, not compatibility."
        ),
    )
    parser.add_argument("--rx-height", type=float, default=DEFAULT_RX_HEIGHT_M)
    parser.add_argument("--rx-building-clearance", type=float, default=MIN_BUILDING_CLEARANCE_M)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED, help="Global seed used to derive a stable seed for each block.")
    parser.add_argument(
        "--osm-height-variant",
        default=DEFAULT_OSM_HEIGHT_VARIANT,
        help="Label for the OSM building-height source; recorded in metadata and RT checkpoints.",
    )
    parser.add_argument("--voxel-size", type=float, default=DEFAULT_VOXEL_SIZE_M)
    parser.add_argument("--voxel-nx", type=int, default=DEFAULT_VOXEL_NX)
    parser.add_argument("--voxel-ny", type=int, default=DEFAULT_VOXEL_NY)
    parser.add_argument("--voxel-nz", type=int, default=DEFAULT_VOXEL_NZ)
    parser.add_argument("--voxel-z-start", type=float, default=DEFAULT_VOXEL_Z_START_M)
    parser.add_argument("--tx-batch-size", type=int, default=DEFAULT_TX_BATCH_SIZE)
    parser.add_argument("--samples-per-tx", type=int, default=10_000)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument(
        "--render-preview-after-block",
        action="store_true",
        help=(
            "After each completed RT block, render RX-count x NZ height-layer "
            "128x128 PNG previews with OSM building footprints in black."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.rx_count <= 0:
        raise SystemExit("--rx-count must be positive")
    if args.rx_solver_group_size <= 0:
        raise SystemExit("--rx-solver-group-size must be positive")
    if args.rx_height <= 0 or args.rx_building_clearance < 0:
        raise SystemExit("--rx-height must be positive and --rx-building-clearance non-negative")
    if min(args.voxel_nx, args.voxel_ny, args.voxel_nz, args.tx_batch_size) <= 0:
        raise SystemExit("voxel dimensions and --tx-batch-size must be positive")
    if not math.isclose(args.voxel_nx * args.voxel_size, MAP_BLOCK_SIZE_M, abs_tol=1e-9):
        raise SystemExit("--voxel-nx * --voxel-size must equal the 256 m map size")
    if not math.isclose(args.voxel_ny * args.voxel_size, MAP_BLOCK_SIZE_M, abs_tol=1e-9):
        raise SystemExit("--voxel-ny * --voxel-size must equal the 256 m map size")
    dataset_root = args.dataset_root.resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)
    enable_persistent_logging(
        dataset_root,
        [str(Path(__file__).name), *(argv if argv is not None else sys.argv[1:])],
        args.log_name,
    )
    selected_regions = set(args.region) if args.region else None
    if args.worker_mode:
        blocks = load_blocks_manifest(dataset_root / "blocks_manifest.csv")
    else:
        # A bundled/offline manifest is authoritative and must not be replaced
        # on the server. A brand-new local cache, however, contains exactly the
        # explicitly selected catalogue (e.g. 39 Project-985 main campuses).
        manifest_path = dataset_root / "blocks_manifest.csv"
        if manifest_path.is_file():
            blocks = load_blocks_manifest(manifest_path)
        else:
            blocks = prepare_manifest(dataset_root, selected_regions)
        write_dataset_metadata(dataset_root, blocks, args)
        requirements_source = Path(__file__).with_name("CAMPUS_SIONNA_DATASET_REQUIREMENTS.md")
        if requirements_source.is_file():
            (dataset_root / "REQUIREMENTS.md").write_text(
                requirements_source.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
    if args.prepare_only:
        write_sharded_voxel_report(dataset_root, blocks)
        print("RUN COMPLETE (prepare-only)")
        return 0

    selected_blocks = set(args.block or [])
    work = [
        block for block in blocks
        if (not selected_regions or block.region_slug in selected_regions)
        and (not selected_blocks or block.block_id in selected_blocks)
    ]
    if args.max_blocks > 0:
        work = work[:args.max_blocks]
    if not args.skip_prefetch:
        prefetch_block_assets(work, dataset_root)
    if args.prefetch_only:
        write_sharded_voxel_report(dataset_root, blocks)
        print("RUN COMPLETE (OSM prefetch-only; satellite=SKIP)")
        return 0
    for index, block in enumerate(work, start=1):
        print(f"\n[{index}/{len(work)}] {block.region_slug}/{block.block_id}")
        process_aerial_tx_voxel_block(block, dataset_root, args)
        if not args.worker_mode:
            write_sharded_voxel_report(dataset_root, blocks)
    if not args.worker_mode:
        write_sharded_voxel_report(dataset_root, blocks)
    print("RUN COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
