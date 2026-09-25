"""OSM building download, local geometry, and Mitsuba/Sionna scene generation."""

from __future__ import annotations

import math
import re
import shutil
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from rt_solver import Building, RtExperimentConfig


def download_osm_buildings(config: RtExperimentConfig) -> list[Building]:
    min_lon, min_lat, max_lon, max_lat = radius_to_bbox(config.lon0, config.lat0, config.radius_m)
    bbox = f"{min_lat:.15g},{min_lon:.15g},{max_lat:.15g},{max_lon:.15g}"
    overpass_query = (
        "[out:xml][timeout:90];"
        f"(way[\"building\"]({bbox});relation[\"building\"]({bbox}););"
        "(._;>;);out body;"
    )
    query = urllib.parse.urlencode({"data": overpass_query})
    url = f"https://maps.mail.ru/osm/tools/overpass/api/interpreter?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "Sionna RT OSM building converter"})
    with urllib.request.urlopen(request, timeout=60) as response:
        xml_bytes = response.read()

    root = ET.fromstring(xml_bytes)
    nodes: dict[str, tuple[float, float]] = {}
    for node in root.findall("node"):
        node_id = node.attrib.get("id")
        if node_id:
            nodes[node_id] = (float(node.attrib["lat"]), float(node.attrib["lon"]))

    buildings: list[Building] = []
    for way in root.findall("way"):
        tags = {tag.attrib.get("k", ""): tag.attrib.get("v", "") for tag in way.findall("tag")}
        if tags.get("building", "no") == "no":
            continue

        lats: list[float] = []
        lons: list[float] = []
        for nd in way.findall("nd"):
            ref = nd.attrib.get("ref")
            if ref in nodes:
                lat, lon = nodes[ref]
                lats.append(lat)
                lons.append(lon)

        if len(lats) < 3:
            continue
        if hypot_local(float(np.mean(lons)), float(np.mean(lats)), config.lon0, config.lat0) > config.radius_m * 1.25:
            continue
        if lats[0] == lats[-1] and lons[0] == lons[-1]:
            lats = lats[:-1]
            lons = lons[:-1]
        if len(lats) < 3:
            continue

        buildings.append(Building(lats, lons, building_height(tags, config)))

    return buildings


def write_sionna_scene(config: RtExperimentConfig, buildings: list[Building], scene_dir: Path) -> Path:
    mesh_dir = scene_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    ply_path = mesh_dir / "osm_buildings_and_ground.ply"
    xml_path = scene_dir / f"osm_sustech_{config.run_tag}.xml"

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    add_ground_mesh(vertices, faces, config.ground_size_m)
    for building in buildings:
        add_extruded_building_mesh(config, vertices, faces, building)

    write_ply(ply_path, vertices, faces)
    write_scene_xml(xml_path)
    return xml_path


def add_ground_mesh(vertices: list[tuple[float, float, float]], faces: list[tuple[int, int, int]], size_m: float) -> None:
    half = size_m / 2.0
    base = len(vertices)
    vertices.extend([
        (-half, -half, 0.0),
        (half, -half, 0.0),
        (half, half, 0.0),
        (-half, half, 0.0),
    ])
    faces.extend([(base, base + 1, base + 2), (base, base + 2, base + 3)])


def add_extruded_building_mesh(
    config: RtExperimentConfig,
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
    building: Building,
) -> None:
    xy = [lonlat_to_local_m(lon, lat, config.lon0, config.lat0) for lat, lon in zip(building.lat, building.lon)]
    if len(xy) < 3:
        return
    if polygon_area(xy) < 0:
        xy = list(reversed(xy))

    base = len(vertices)
    for x, y in xy:
        vertices.append((x, y, 0.0))
    for x, y in xy:
        vertices.append((x, y, building.height_m))

    n = len(xy)
    for i in range(n):
        j = (i + 1) % n
        b0, b1 = base + i, base + j
        t0, t1 = base + n + i, base + n + j
        faces.append((b0, b1, t1))
        faces.append((b0, t1, t0))

    if config.add_building_roofs:
        for i, j, k in triangulate_roof_polygon(xy):
            faces.append((base + n + i, base + n + j, base + n + k))


def write_ply(path: Path, vertices: list[tuple[float, float, float]], faces: list[tuple[int, int, int]]) -> None:
    with path.open("w", encoding="ascii", newline="\n") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(vertices)}\n")
        file.write("property float x\nproperty float y\nproperty float z\n")
        file.write(f"element face {len(faces)}\n")
        file.write("property list uchar int vertex_indices\n")
        file.write("end_header\n")
        for x, y, z in vertices:
            file.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
        for i, j, k in faces:
            file.write(f"3 {i} {j} {k}\n")


def write_scene_xml(path: Path) -> None:
    text = """<scene version="2.1.0">
    <bsdf type="itu-radio-material" id="building-material">
        <string name="type" value="concrete"/>
        <float name="thickness" value="0.3"/>
    </bsdf>

    <shape type="ply" id="osm-buildings-and-ground">
        <string name="filename" value="meshes/osm_buildings_and_ground.ply"/>
        <boolean name="face_normals" value="true"/>
        <ref id="building-material" name="bsdf"/>
    </shape>
</scene>
"""
    path.write_text(text, encoding="utf-8")


def copy_generated_scene(source_dir: Path, target_dir: Path) -> Path:
    try:
        shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
        return target_dir
    except PermissionError:
        from datetime import datetime

        fallback_dir = target_dir.parent / f"{target_dir.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copytree(source_dir, fallback_dir)
        print(f"Warning: could not overwrite {target_dir}; copied scene to {fallback_dir}")
        return fallback_dir


def building_height(tags: dict[str, str], config: RtExperimentConfig) -> float:
    if config.override_building_height_m is not None and config.override_building_height_m > 0:
        return float(config.override_building_height_m)

    height = parse_measurement_float(tags.get("height"))
    if height is not None and height > 0:
        return height
    levels = parse_measurement_float(tags.get("building:levels")) or parse_measurement_float(tags.get("levels"))
    if levels is not None and levels > 0:
        return levels * config.default_level_height_m
    return config.default_building_height_m


def triangulate_roof_polygon(xy: list[tuple[float, float]]) -> list[tuple[int, int, int]]:
    if len(xy) < 3:
        return []

    earclip_triangles = triangulate_roof_polygon_earclip(xy)
    if len(earclip_triangles) >= max(1, len(xy) - 2):
        return earclip_triangles

    try:
        from matplotlib.path import Path as MplPath
        from scipy.spatial import Delaunay
    except Exception:
        return earclip_triangles

    points = np.asarray(xy, dtype=float)
    if np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
        return []

    try:
        delaunay = Delaunay(points)
    except Exception:
        return earclip_triangles

    polygon_path = MplPath(np.vstack([points, points[0]]), closed=True)
    triangles: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for simplex in delaunay.simplices:
        tri_points = points[simplex]
        probes = np.vstack([
            tri_points.mean(axis=0),
            (tri_points[0] + tri_points[1]) * 0.5,
            (tri_points[1] + tri_points[2]) * 0.5,
            (tri_points[2] + tri_points[0]) * 0.5,
        ])
        if not all(polygon_path.contains_point(probe, radius=1e-8) for probe in probes):
            continue

        i, j, k = (int(v) for v in simplex)
        area = triangle_signed_area(points[i], points[j], points[k])
        if abs(area) < 1e-9:
            continue
        if area < 0:
            j, k = k, j
        key = tuple(sorted((i, j, k)))
        if key not in seen:
            triangles.append((i, j, k))
            seen.add(key)

    return triangles or earclip_triangles


def triangulate_roof_polygon_earclip(xy: list[tuple[float, float]]) -> list[tuple[int, int, int]]:
    points = np.asarray(xy, dtype=float)
    indices = list(range(len(points)))
    if polygon_area(xy) < 0:
        indices.reverse()

    triangles: list[tuple[int, int, int]] = []
    guard = 0
    while len(indices) > 3 and guard < len(points) * len(points):
        guard += 1
        clipped = False
        for pos, curr in enumerate(indices):
            prev = indices[pos - 1]
            nxt = indices[(pos + 1) % len(indices)]
            if triangle_signed_area(points[prev], points[curr], points[nxt]) <= 1e-9:
                continue
            if not is_internal_polygon_diagonal(points, prev, nxt, indices):
                continue
            tri = (points[prev], points[curr], points[nxt])
            if any(point_in_triangle(points[idx], tri) for idx in indices if idx not in (prev, curr, nxt)):
                continue
            triangles.append((prev, curr, nxt))
            del indices[pos]
            clipped = True
            break
        if not clipped:
            break

    if len(indices) == 3:
        i, j, k = indices
        if triangle_signed_area(points[i], points[j], points[k]) < 0:
            j, k = k, j
        triangles.append((i, j, k))
    return triangles


def is_internal_polygon_diagonal(points: np.ndarray, i: int, j: int, polygon_indices: list[int]) -> bool:
    a = points[i]
    b = points[j]
    for edge_pos, edge_start in enumerate(polygon_indices):
        edge_end = polygon_indices[(edge_pos + 1) % len(polygon_indices)]
        if i in (edge_start, edge_end) or j in (edge_start, edge_end):
            continue
        if segments_intersect(a, b, points[edge_start], points[edge_end]):
            return False
    midpoint = (a + b) * 0.5
    return point_in_polygon_2d(midpoint, points[polygon_indices])


def segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    eps = 1e-9
    o1 = orientation_2d(a, b, c)
    o2 = orientation_2d(a, b, d)
    o3 = orientation_2d(c, d, a)
    o4 = orientation_2d(c, d, b)
    if abs(o1) <= eps and on_segment_2d(a, c, b):
        return True
    if abs(o2) <= eps and on_segment_2d(a, d, b):
        return True
    if abs(o3) <= eps and on_segment_2d(c, a, d):
        return True
    if abs(o4) <= eps and on_segment_2d(c, b, d):
        return True
    return (o1 > eps) != (o2 > eps) and (o3 > eps) != (o4 > eps)


def orientation_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def on_segment_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    eps = 1e-9
    return (
        min(a[0], c[0]) - eps <= b[0] <= max(a[0], c[0]) + eps
        and min(a[1], c[1]) - eps <= b[1] <= max(a[1], c[1]) + eps
    )


def point_in_polygon_2d(point: np.ndarray, polygon: np.ndarray) -> bool:
    x, y = float(point[0]), float(point[1])
    inside = False
    n = len(polygon)
    for i in range(n):
        x0, y0 = float(polygon[i][0]), float(polygon[i][1])
        x1, y1 = float(polygon[(i + 1) % n][0]), float(polygon[(i + 1) % n][1])
        if abs(orientation_2d(np.asarray([x0, y0]), np.asarray([x1, y1]), point)) <= 1e-9 and on_segment_2d(np.asarray([x0, y0]), point, np.asarray([x1, y1])):
            return True
        crosses = (y0 > y) != (y1 > y)
        if crosses:
            x_cross = (x1 - x0) * (y - y0) / (y1 - y0) + x0
            if x < x_cross:
                inside = not inside
    return inside


def triangle_signed_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return 0.5 * float((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))


def point_in_triangle(point: np.ndarray, tri: tuple[np.ndarray, np.ndarray, np.ndarray]) -> bool:
    a, b, c = tri
    area = abs(triangle_signed_area(a, b, c))
    if area < 1e-12:
        return False
    a1 = abs(triangle_signed_area(point, b, c))
    a2 = abs(triangle_signed_area(a, point, c))
    a3 = abs(triangle_signed_area(a, b, point))
    return abs((a1 + a2 + a3) - area) <= 1e-8


def radius_to_bbox(lon: float, lat: float, radius_m: float) -> tuple[float, float, float, float]:
    meters_per_deg_lat, meters_per_deg_lon = meters_per_degree(lat)
    dlat = radius_m / meters_per_deg_lat
    dlon = radius_m / meters_per_deg_lon
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def lonlat_to_local_m(lon: float, lat: float, lon0: float, lat0: float) -> tuple[float, float]:
    meters_per_deg_lat, meters_per_deg_lon = meters_per_degree(lat0)
    return (lon - lon0) * meters_per_deg_lon, (lat - lat0) * meters_per_deg_lat


def local_m_to_lonlat(x_m: float, y_m: float, lon0: float, lat0: float) -> tuple[float, float]:
    meters_per_deg_lat, meters_per_deg_lon = meters_per_degree(lat0)
    return lat0 + y_m / meters_per_deg_lat, lon0 + x_m / meters_per_deg_lon


def meters_per_degree(lat: float) -> tuple[float, float]:
    meters_per_deg_lat = (
        111132.92
        - 559.82 * math.cos(math.radians(2 * lat))
        + 1.175 * math.cos(math.radians(4 * lat))
        - 0.0023 * math.cos(math.radians(6 * lat))
    )
    meters_per_deg_lon = (
        111412.84 * math.cos(math.radians(lat))
        - 93.5 * math.cos(math.radians(3 * lat))
        + 0.118 * math.cos(math.radians(5 * lat))
    )
    return meters_per_deg_lat, meters_per_deg_lon


def hypot_local(lon: float, lat: float, lon0: float, lat0: float) -> float:
    x, y = lonlat_to_local_m(lon, lat, lon0, lat0)
    return math.hypot(x, y)


def polygon_area(xy: list[tuple[float, float]]) -> float:
    area = 0.0
    for i, (x0, y0) in enumerate(xy):
        x1, y1 = xy[(i + 1) % len(xy)]
        area += x0 * y1 - x1 * y0
    return 0.5 * area


def parse_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_measurement_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    match = re.match(r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))", value)
    return None if not match else parse_float(match.group(1))
