from __future__ import annotations

import csv
import math
import re
import statistics
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS = {
    "shenzhen": SCRIPT_DIR / "osm_randomized_height_shenzhen_256m_u10_32",
    "985": SCRIPT_DIR / "osm_randomized_height_985_256m_u10_32",
}
OUTPUT_DIR = SCRIPT_DIR / "building_count_statistics_256m"
HALF_SIZE_M = 128.0
WORKERS = 12


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
    result = clip_edge(result, lambda p: p[0] >= -half_size,
                       lambda a, b: (-half_size, a[1] + (b[1] - a[1]) * (-half_size - a[0]) / (b[0] - a[0])))
    result = clip_edge(result, lambda p: p[0] <= half_size,
                       lambda a, b: (half_size, a[1] + (b[1] - a[1]) * (half_size - a[0]) / (b[0] - a[0])))
    result = clip_edge(result, lambda p: p[1] >= -half_size,
                       lambda a, b: (a[0] + (b[0] - a[0]) * (-half_size - a[1]) / (b[1] - a[1]), -half_size))
    return clip_edge(result, lambda p: p[1] <= half_size,
                     lambda a, b: (a[0] + (b[0] - a[0]) * (half_size - a[1]) / (b[1] - a[1]), half_size))


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


def osm_building_height(tags: dict[str, str]) -> float:
    """Match campus_sionna_dataset.osm_building_height exactly."""
    height = parse_meters(tags.get("height"))
    if height is not None and height > 0:
        return height
    levels = parse_meters(tags.get("building:levels")) or parse_meters(tags.get("levels"))
    return levels * 3.2 if levels is not None and levels > 0 else 12.0


def count_buildings(item: tuple[str, Path]) -> tuple[dict[str, object], list[dict[str, object]]]:
    dataset, osm_path = item
    block_id = osm_path.parent.name
    region = osm_path.parent.parent.name
    center_lat, center_lon = (float(value) for value in block_id.split("_", maxsplit=1))
    meters_per_deg_lat, meters_per_deg_lon = meters_per_degree(center_lat)

    root = ET.parse(osm_path).getroot()
    nodes = {
        node.attrib["id"]: (float(node.attrib["lat"]), float(node.attrib["lon"]))
        for node in root.findall("node")
    }
    building_records = []
    for way in root.findall("way"):
        tags = {tag.attrib.get("k", ""): tag.attrib.get("v", "") for tag in way.findall("tag")}
        if tags.get("building", "no") == "no":
            continue
        lonlat = [nodes[nd.attrib["ref"]] for nd in way.findall("nd") if nd.attrib.get("ref") in nodes]
        if len(lonlat) < 3:
            continue
        if lonlat[0] == lonlat[-1]:
            lonlat = lonlat[:-1]
        local = [
            ((lon - center_lon) * meters_per_deg_lon, (lat - center_lat) * meters_per_deg_lat)
            for lat, lon in lonlat
        ]
        if len(clip_polygon(local, HALF_SIZE_M)) >= 3:
            building_records.append({
                "dataset": dataset,
                "region_slug": region,
                "block_id": block_id,
                "osm_way_id": way.attrib.get("id", ""),
                "height_m": osm_building_height(tags),
            })

    block_record = {
        "dataset": dataset,
        "region_slug": region,
        "block_id": block_id,
        "building_count": len(building_records),
    }
    return block_record, building_records


def quantile(values: list[int], q: float) -> float:
    return float(np.quantile(np.asarray(values), q))


def summarize(dataset: str, region: str, values: list[int]) -> dict[str, object]:
    return {
        "dataset": dataset,
        "region_slug": region,
        "block_count": len(values),
        "empty_block_count": sum(value == 0 for value in values),
        "empty_block_pct": 100 * sum(value == 0 for value in values) / len(values),
        "min": min(values),
        "q1": quantile(values, 0.25),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "q3": quantile(values, 0.75),
        "max": max(values),
        "total_building_occurrences": sum(values),
    }


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
tasks = [
    (dataset, osm_path)
    for dataset, root in DATASETS.items()
    for osm_path in root.glob("*/*/osm_map.osm")
]
with ThreadPoolExecutor(max_workers=WORKERS) as executor:
    results = list(executor.map(count_buildings, tasks))
rows = [block_record for block_record, _ in results]
height_rows = [building for _, buildings in results for building in buildings]
rows.sort(key=lambda row: (str(row["dataset"]), str(row["region_slug"]), str(row["block_id"])))
height_rows.sort(key=lambda row: (
    str(row["dataset"]), str(row["region_slug"]), str(row["block_id"]), str(row["osm_way_id"])
))

with (OUTPUT_DIR / "block_building_counts.csv").open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["dataset", "region_slug", "block_id", "building_count"])
    writer.writeheader()
    writer.writerows(rows)

with (OUTPUT_DIR / "building_height_samples.csv").open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["dataset", "region_slug", "block_id", "osm_way_id", "height_m"],
    )
    writer.writeheader()
    writer.writerows(height_rows)

grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
dataset_values: dict[str, list[int]] = defaultdict(list)
for row in rows:
    value = int(row["building_count"])
    grouped[(str(row["dataset"]), str(row["region_slug"]))].append(value)
    dataset_values[str(row["dataset"])].append(value)

summary_rows = []
for dataset, values in sorted(dataset_values.items()):
    summary_rows.append(summarize(dataset, "ALL", values))
    for (group_dataset, region), region_values in sorted(grouped.items()):
        if group_dataset == dataset:
            summary_rows.append(summarize(dataset, region, region_values))

with (OUTPUT_DIR / "dataset_and_region_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
    writer.writeheader()
    writer.writerows(summary_rows)

# Building-count distribution: every integer count has its own width-1 bin.
distribution_rows = []
for dataset, values in sorted(dataset_values.items()):
    frequencies = defaultdict(int)
    for value in values:
        frequencies[value] += 1
    for building_count in range(0, max(values) + 1):
        count = frequencies[building_count]
        distribution_rows.append({
            "dataset": dataset,
            "building_count": building_count,
            "bin_left_edge": building_count - 0.5,
            "bin_right_edge": building_count + 0.5,
            "block_count": count,
            "percentage": 100 * count / len(values),
        })
for filename in ("distribution_bins.csv", "building_count_distribution_bin1.csv"):
    with (OUTPUT_DIR / filename).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(distribution_rows[0]))
        writer.writeheader()
        writer.writerows(distribution_rows)

# Height statistics use the exact height selected by the simulation generator.
dataset_height_values: dict[str, list[float]] = defaultdict(list)
region_height_values: dict[tuple[str, str], list[float]] = defaultdict(list)
for row in height_rows:
    value = float(row["height_m"])
    dataset = str(row["dataset"])
    region = str(row["region_slug"])
    dataset_height_values[dataset].append(value)
    region_height_values[(dataset, region)].append(value)

height_summary_rows = []
for dataset, values in sorted(dataset_height_values.items()):
    groups = [("ALL", values)] + [
        (region, region_values)
        for (group_dataset, region), region_values in sorted(region_height_values.items())
        if group_dataset == dataset
    ]
    for region, group_values in groups:
        height_summary_rows.append({
            "dataset": dataset,
            "region_slug": region,
            "building_sample_count": len(group_values),
            "min_height_m": min(group_values),
            "q1_height_m": quantile(group_values, 0.25),
            "median_height_m": statistics.median(group_values),
            "mean_height_m": statistics.fmean(group_values),
            "q3_height_m": quantile(group_values, 0.75),
            "max_height_m": max(group_values),
        })
with (OUTPUT_DIR / "building_height_dataset_and_region_summary.csv").open(
    "w", encoding="utf-8-sig", newline=""
) as handle:
    writer = csv.DictWriter(handle, fieldnames=list(height_summary_rows[0]))
    writer.writeheader()
    writer.writerows(height_summary_rows)

height_distribution_rows = []
for dataset, values in sorted(dataset_height_values.items()):
    first_edge = math.floor(min(values))
    final_left_edge = math.floor(max(values))
    edges = np.arange(first_edge, final_left_edge + 2, 1.0)
    frequencies, _ = np.histogram(values, bins=edges)
    for left, count in zip(edges[:-1], frequencies):
        height_distribution_rows.append({
            "dataset": dataset,
            "bin_left_m_inclusive": left,
            "bin_right_m_exclusive": left + 1.0,
            "building_sample_count": int(count),
            "percentage": 100 * int(count) / len(values),
        })
with (OUTPUT_DIR / "building_height_distribution_bin1m.csv").open(
    "w", encoding="utf-8-sig", newline=""
) as handle:
    writer = csv.DictWriter(handle, fieldnames=list(height_distribution_rows[0]))
    writer.writeheader()
    writer.writerows(height_distribution_rows)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=180)
colors = {"shenzhen": "#2878B5", "985": "#D1495B"}
for ax, dataset in zip(axes, ("shenzhen", "985")):
    values = dataset_values[dataset]
    maximum = max(values)
    bins = np.arange(-0.5, maximum + 1.5, 1.0)
    ax.hist(values, bins=bins, color=colors[dataset], edgecolor="white")
    ax.axvline(statistics.median(values), color="black", linestyle="--", linewidth=1.2,
               label=f"median={statistics.median(values):g}")
    ax.set_title(f"{dataset.upper()} ({len(values)} blocks)")
    ax.set_xlabel("Buildings per 256 m block")
    ax.set_ylabel("Number of blocks")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
fig.suptitle("Building-count Distribution by Block", fontsize=15)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUTPUT_DIR / "building_count_histograms.png", facecolor="white")
fig.savefig(OUTPUT_DIR / "building_count_histograms_bin1.png", facecolor="white")
plt.close(fig)

for filename, x_limits, title_suffix in (
    ("building_height_histograms_bin1m.png", None, "Full range"),
    ("building_height_histograms_bin1m_zoom_0_40m.png", (0, 40), "0-40 m zoom"),
):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), dpi=180)
    for ax, dataset in zip(axes, ("shenzhen", "985")):
        values = dataset_height_values[dataset]
        first_edge = math.floor(min(values))
        final_left_edge = math.floor(max(values))
        bins = np.arange(first_edge, final_left_edge + 2, 1.0)
        ax.hist(values, bins=bins, color=colors[dataset], edgecolor="white")
        ax.axvline(statistics.median(values), color="black", linestyle="--", linewidth=1.2,
                   label=f"median={statistics.median(values):.1f} m")
        ax.set_title(f"{dataset.upper()} (n={len(values):,})")
        ax.set_xlabel("Height (m); bin width = 1 m")
        ax.set_ylabel("Building samples")
        if x_limits is not None:
            ax.set_xlim(*x_limits)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    fig.suptitle(f"Building-height Distribution ({title_suffix})", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUTPUT_DIR / filename, facecolor="white")
    plt.close(fig)

lines = [
    "Building count and height statistics for current 256 m datasets",
    "Counting rule: identical to campus_sionna_dataset.parse_block_buildings",
    "Only OSM ways with building != no and polygon intersection with the 256 m block are counted.",
    "Histogram bin widths: building count = 1 building; building height = 1 m.",
    "",
]
for dataset in ("shenzhen", "985"):
    values = dataset_values[dataset]
    overall = summarize(dataset, "ALL", values)
    heights = dataset_height_values[dataset]
    core_height_count = sum(10 <= height <= 32 for height in heights)
    over_40_count = sum(height > 40 for height in heights)
    over_100_count = sum(height > 100 for height in heights)
    lines.extend([
        f"[{dataset}]",
        f"blocks={overall['block_count']}, buildings_occurrences={overall['total_building_occurrences']}",
        f"building_count: min={overall['min']}, q1={overall['q1']:.2f}, median={overall['median']:.2f}, "
        f"mean={overall['mean']:.2f}, q3={overall['q3']:.2f}, max={overall['max']}",
        f"empty={overall['empty_block_count']} ({overall['empty_block_pct']:.2f}%)",
        f"building_height_m: min={min(heights):.2f}, q1={quantile(heights, 0.25):.2f}, "
        f"median={statistics.median(heights):.2f}, mean={statistics.fmean(heights):.2f}, "
        f"q3={quantile(heights, 0.75):.2f}, max={max(heights):.2f}",
        f"height_10_to_32m={core_height_count} ({100 * core_height_count / len(heights):.2f}%), "
        f"height_over_40m={over_40_count} ({100 * over_40_count / len(heights):.2f}%), "
        f"height_over_100m={over_100_count}",
    ])
    ranked = sorted((row for row in rows if row["dataset"] == dataset),
                    key=lambda row: (-int(row["building_count"]), str(row["region_slug"]), str(row["block_id"])))
    lines.append("top5: " + "; ".join(
        f"{row['region_slug']}/{row['block_id']}={row['building_count']}" for row in ranked[:5]
    ))
    lines.append("")
(OUTPUT_DIR / "building_count_report.txt").write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))
print(f"Reports: {OUTPUT_DIR}")
