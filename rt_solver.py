"""Sionna RT data models, mesh solving, and solver result outputs."""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# Single switch for both per-height RT timing and block-level phase timing.
# The cluster launcher exports RID_ENABLE_TIMING; direct runs stay quiet unless
# the same environment variable is explicitly enabled.
ENABLE_TIMING_LOGS = os.environ.get("RID_ENABLE_TIMING", "0").strip().lower() in {
    "1", "true", "yes", "on",
}
ENABLE_DRJIT_KERNEL_HISTORY = os.environ.get(
    "RID_DRJIT_KERNEL_HISTORY", "0"
).strip().lower() in {"1", "true", "yes", "on"}
CHECKPOINT_INTERVAL_BATCHES = max(
    1, int(os.environ.get("RID_CHECKPOINT_INTERVAL_BATCHES", "25"))
)


def should_persist_batch(
    batch_number: int,
    batch_end: int,
    total_points: int,
    interval_batches: int = CHECKPOINT_INTERVAL_BATCHES,
) -> bool:
    """Commit a recovery boundary periodically and at every height end."""
    return batch_end == total_points or batch_number % max(1, interval_batches) == 0


def summarize_drjit_kernel_history(records: list[dict[str, object]]) -> dict[str, object]:
    """Reduce Dr.Jit's official kernel-history records to log-sized counters."""
    type_counts: dict[str, int] = {}
    jit_records: list[dict[str, object]] = []
    for record in records:
        kernel_type = record.get("type", "unknown")
        type_name = getattr(kernel_type, "name", str(kernel_type)).lower()
        type_counts[type_name] = type_counts.get(type_name, 0) + 1
        if "cache_hit" in record or "hash" in record:
            jit_records.append(record)

    def finite_sum(items: list[dict[str, object]], key: str) -> float:
        total = 0.0
        for record in items:
            try:
                value = float(record.get(key, 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                total += value
        return total

    memory_hit_records = [record for record in jit_records if bool(record.get("cache_hit", False))]
    disk_hit_records = [
        record for record in jit_records
        if not bool(record.get("cache_hit", False))
        and bool(record.get("cache_disk", False))
    ]
    miss_records = [
        record for record in jit_records
        if not bool(record.get("cache_hit", False))
        and not bool(record.get("cache_disk", False))
    ]
    hashes = {
        str(record["hash"])
        for record in jit_records
        if record.get("hash") not in (None, "")
    }
    miss_codegen_time_ms = finite_sum(miss_records, "codegen_time")
    miss_backend_time_ms = finite_sum(miss_records, "backend_time")
    return {
        "records": len(records),
        "jit_kernels": len(jit_records),
        "memory_hits": len(memory_hit_records),
        "disk_hits": len(disk_hit_records),
        "hard_misses": len(miss_records),
        "optix_kernels": sum(bool(record.get("uses_optix", False)) for record in jit_records),
        "unique_hashes": len(hashes),
        "execution_time_ms": finite_sum(records, "execution_time"),
        "codegen_time_ms": finite_sum(records, "codegen_time"),
        "backend_time_ms": finite_sum(records, "backend_time"),
        "miss_codegen_time_ms": miss_codegen_time_ms,
        "miss_backend_time_ms": miss_backend_time_ms,
        "miss_compile_time_ms": miss_codegen_time_ms + miss_backend_time_ms,
        "miss_execution_time_ms": finite_sum(miss_records, "execution_time"),
        "hit_execution_time_ms": finite_sum(
            [*memory_hit_records, *disk_hit_records], "execution_time"
        ),
        "type_counts": type_counts,
    }


def checkpoint_signatures_compatible(
    saved_signature: object,
    requested_signature: object,
) -> bool:
    """Compare RT checkpoints while treating batch size as a runtime detail.

    Checkpoints store absolute height and TX offsets, so changing the batch
    size does not alter which samples have already been confirmed. All
    physical, geometry, schema, and sampling parameters remain strict.
    """
    if not isinstance(saved_signature, dict) or not isinstance(requested_signature, dict):
        return False
    ignored_keys = {"mesh_batch_size"}
    saved = {key: value for key, value in saved_signature.items() if key not in ignored_keys}
    requested = {key: value for key, value in requested_signature.items() if key not in ignored_keys}
    return saved == requested


def checkpoint_batch_size_history(checkpoint: dict[str, object], current_batch_size: int) -> list[int]:
    """Return ordered, unique batch sizes used by one block."""
    history: list[int] = []
    raw_history = checkpoint.get("mesh_batch_sizes_used", [])
    if isinstance(raw_history, list):
        for value in raw_history:
            try:
                size = int(value)
            except (TypeError, ValueError):
                continue
            if size > 0 and size not in history:
                history.append(size)
    saved_signature = checkpoint.get("signature")
    if isinstance(saved_signature, dict):
        try:
            saved_size = int(saved_signature.get("mesh_batch_size", 0))
        except (TypeError, ValueError):
            saved_size = 0
        if saved_size > 0 and saved_size not in history:
            history.append(saved_size)
    if current_batch_size > 0 and current_batch_size not in history:
        history.append(current_batch_size)
    return history


@dataclass
class RtExperimentConfig:
    run_tag: str
    lat_rx: float
    lon_rx: float
    rx_height_m: float
    lat0: float
    lon0: float
    radius_m: float
    selected_experiment_heights: list[float]
    discard_path_start_seconds: list[float]
    discard_path_end_seconds: list[float]
    start_timestamp: str
    rid_log_dir: Path
    rid_log_pattern: str
    frequency_hz: float
    tx_power_dbm: float
    mesh_spacing_m: float
    mesh_batch_size: int
    mesh_max_points_per_height: int
    max_depth: int
    samples_per_tx: int
    los: bool
    specular_reflection: bool
    diffuse_reflection: bool
    refraction: bool
    diffraction: bool
    default_level_height_m: float
    default_building_height_m: float
    override_building_height_m: float | None
    ground_size_m: float
    add_building_roofs: bool
    output_dir: Path
    scene_work_dir: Path
    rx_antenna_pattern: str = "dipole"
    rx_antenna_polarization: str = "V"
    rx_orientation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rx_extra_gain_dbi: float = 0.0
    plan_view_marker_size: int = 28
    plan_view_rx_marker_size: int = 130
    plan_view_figsize: tuple[float, float] = (14, 11)
    plan_view_grid_figsize: tuple[float, float] = (18, 14)
    plan_view_dpi: int = 160
    plan_view_rssi_limits_dbm: tuple[float, float] | None = (-70.0, -45.0)
    plan_view_sim_rx_power_limits_dbm: tuple[float, float] | None = None


@dataclass
class Building:
    lat: list[float]
    lon: list[float]
    height_m: float


@dataclass
class RidSample:
    timestamp_s: int
    lat: float
    lon: float
    log_alt_m: float | None
    experiment_height_m: int
    measured_rssi_dbm: float | None
    source_file: str
    line_no: int


@dataclass
class TopdownPoint:
    height_m: int
    x_m: float
    y_m: float
    value_dbm: float
    lat: float
    lon: float


@dataclass
class MeshTxPoint:
    index: int
    x_m: float
    y_m: float
    lat: float
    lon: float
    radial_distance_m: float


@dataclass
class GroundRxPoint:
    index: int
    x_m: float
    y_m: float
    lat: float
    lon: float
    radial_distance_m: float


@dataclass
class GroundTxPoint:
    index: int
    x_m: float
    y_m: float
    z_m: float
    lat: float
    lon: float
    radial_distance_m: float


@dataclass
class VoxelRxPoint:
    index: int
    ix: int
    iy: int
    iz: int
    x_m: float
    y_m: float
    z_m: float
    lat: float
    lon: float


def run_sionna_rt_multi_rx(
    config: RtExperimentConfig,
    scene_xml: Path,
    out_csv: Path,
    tx_points: list[MeshTxPoint],
    rx_points: list[GroundRxPoint],
    heights_m: list[float],
    mesh_batch_size: int,
    samples_per_tx: int,
    max_depth: int,
    region_slug: str,
    block_id: str,
    environment_signature: str,
    resume: bool = True,
) -> dict[str, object]:
    """Solve aerial TX mesh points against multiple fixed ground receivers.

    Every completed batch is flushed to a temporary CSV and recorded in a
    checkpoint. A resumed run truncates the CSV to the last confirmed byte
    position and continues at the next batch. Only a fully completed CSV is
    atomically installed at ``out_csv``. The completed checkpoint remains
    until the caller finishes plots and metadata, so a post-processing
    interruption does not force the RT solver to run again.
    """
    if not tx_points or not rx_points or not heights_m:
        raise ValueError("TX points, RX points, and heights must all be non-empty")
    if mesh_batch_size <= 0:
        mesh_batch_size = len(tx_points)

    horizontal_axis_count = int(round(math.sqrt(len(tx_points))))
    if horizontal_axis_count * horizontal_axis_count != len(tx_points):
        raise ValueError("Aerial TX voxel grid must be a square horizontal grid")
    # The block directory, metadata.json, and rx_positions.csv already contain
    # every constant/derivable coordinate. Repeating them for 6,553,600 rows
    # made the raw dataset unnecessarily large. Keep only the four tensor
    # indices and the measured scalar required to reconstruct [RX,Z,Y,X].
    fieldnames = [
        "rx_index", "tx_voxel_ix", "tx_voxel_iy", "tx_voxel_iz", "rss_dbm",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    temp_csv = out_csv.with_suffix(out_csv.suffix + ".tmp")
    checkpoint_path = out_csv.with_suffix(out_csv.suffix + ".checkpoint.json")
    signature = {
        "schema_version": 4,
        "csv_schema": "compact_rss_5col_v1",
        "layout": "3d_voxel_tx_to_fixed_ground_rx",
        "region_slug": region_slug,
        "block_id": block_id,
        "environment_signature": environment_signature,
        "heights_m": [float(value) for value in heights_m],
        "tx_count": len(tx_points),
        "rx_positions_xy_m": [[round(point.x_m, 6), round(point.y_m, 6)] for point in rx_points],
        "mesh_batch_size": mesh_batch_size,
        "samples_per_tx": samples_per_tx,
        "max_depth": max_depth,
        "mesh_spacing_m": config.mesh_spacing_m,
    }

    state: dict[str, object] | None = None
    candidate: dict[str, object] | None = None
    if resume and checkpoint_path.is_file():
        try:
            loaded = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                candidate = loaded
        except (OSError, ValueError) as exc:
            print(f"IGNORING INVALID CHECKPOINT {checkpoint_path}: {exc}")

    expected_rows = len(rx_points) * len(tx_points) * len(heights_m)
    candidate_signature_matches = (
        candidate is not None
        and checkpoint_signatures_compatible(candidate.get("signature"), signature)
    )
    if (
        candidate_signature_matches
        and int(candidate.get("next_height_index", -1)) == len(heights_m)
        and int(candidate.get("total_rows", -1)) == expected_rows
        and out_csv.is_file()
        and out_csv.stat().st_size == int(candidate.get("csv_size_bytes", -1))
    ):
        print(
            f"RESUME POSTPROCESS {region_slug}/{block_id}: RT CSV already complete; "
            f"rows {candidate['total_rows']}"
        )
        return {
            "row_count": int(candidate["total_rows"]),
            "height_summaries": list(candidate["height_summaries"]),
            # RT was already complete, so the newly requested batch size was
            # not actually used during this post-processing-only resume.
            "mesh_batch_sizes_used": checkpoint_batch_size_history(candidate, 0),
        }

    if candidate_signature_matches and temp_csv.is_file():
        confirmed_size = int(candidate["csv_size_bytes"])
        if 0 <= confirmed_size <= temp_csv.stat().st_size:
            with temp_csv.open("r+b") as file:
                file.truncate(confirmed_size)
            state = candidate
            batch_sizes_used = checkpoint_batch_size_history(state, mesh_batch_size)
            saved_batch_size = state.get("signature", {}).get("mesh_batch_size")
            state["signature"] = signature
            state["mesh_batch_sizes_used"] = batch_sizes_used
            print(
                f"RESUME CHECKPOINT {region_slug}/{block_id}: height index "
                f"{state['next_height_index']}, TX offset {state['next_batch_start']}, "
                f"rows {state['total_rows']}; batch {saved_batch_size} -> {mesh_batch_size}"
            )
        else:
            print(f"IGNORING INVALID CHECKPOINT SIZE {checkpoint_path}")
    if state is None:
        with temp_csv.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            file.flush()
        state = {
            "signature": signature,
            "mesh_batch_sizes_used": [mesh_batch_size],
            "next_height_index": 0,
            "next_batch_start": 0,
            "total_rows": 0,
            "height_summaries": [],
            "current_height_count": 0,
            "current_height_sum": 0.0,
            "current_height_min": None,
            "current_height_max": None,
            "csv_size_bytes": temp_csv.stat().st_size,
        }
        write_json_checkpoint(state, checkpoint_path)

    rt_import_started = time.perf_counter()
    import sionna.rt as rt

    rt_import_seconds = time.perf_counter() - rt_import_started

    scene_load_started = time.perf_counter()
    scene = rt.load_scene(str(scene_xml), merge_shapes=True)
    scene_load_seconds = time.perf_counter() - scene_load_started
    scene_setup_started = time.perf_counter()
    scene.frequency = config.frequency_hz
    scene.tx_array = rt.PlanarArray(
        num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="iso", polarization="V",
    )
    scene.rx_array = rt.PlanarArray(
        num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="iso", polarization="V",
    )
    for point in rx_points:
        scene.add(rt.Receiver(
            name=f"rx_{point.index:03d}",
            position=[point.x_m, point.y_m, config.rx_height_m],
            orientation=[0.0, 0.0, 0.0],
            display_radius=1.0,
        ))

    solver = rt.PathSolver()
    scene_setup_seconds = time.perf_counter() - scene_setup_started
    timing_totals = {
        "tx_setup": 0.0,
        "solver_call": 0.0,
        "cir_numpy_sync": 0.0,
        "gain_reduce": 0.0,
        "row_build": 0.0,
        "csv_write": 0.0,
        "tx_cleanup": 0.0,
        "checkpoint": 0.0,
        "batch_total": 0.0,
    }
    timed_batch_count = 0
    rt_loop_started = time.perf_counter()
    if ENABLE_TIMING_LOGS:
        print(
            f"RT SETUP TIMING {region_slug}/{block_id}: "
            f"import_sionna_rt={rt_import_seconds:.3f}s "
            f"scene_load={scene_load_seconds:.3f}s "
            f"arrays_receivers_solver={scene_setup_seconds:.3f}s",
            flush=True,
        )
    try:
        with temp_csv.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            start_height_index = int(state["next_height_index"])
            for height_index in range(start_height_index, len(heights_m)):
                height = float(heights_m[height_index])
                height_timing_totals = {key: 0.0 for key in timing_totals}
                height_timed_batch_count = 0
                if height_index == start_height_index:
                    batch_start_offset = int(state["next_batch_start"])
                    height_count = int(state["current_height_count"])
                    height_sum = float(state["current_height_sum"])
                    height_min = math.inf if state["current_height_min"] is None else float(state["current_height_min"])
                    height_max = -math.inf if state["current_height_max"] is None else float(state["current_height_max"])
                else:
                    batch_start_offset = 0
                    height_count, height_sum = 0, 0.0
                    height_min, height_max = math.inf, -math.inf
                remaining_batch_count = math.ceil(
                    (len(tx_points) - batch_start_offset) / mesh_batch_size
                )
                print(
                    f"Solving {len(rx_points)} ground RX receivers against {len(tx_points)} aerial TX points "
                    f"at {height:g} m in {remaining_batch_count} remaining batches "
                    f"(start offset {batch_start_offset}, batch size {mesh_batch_size}) ..."
                )
                for batch_start in range(batch_start_offset, len(tx_points), mesh_batch_size):
                    batch_started = time.perf_counter()
                    batch_points = tx_points[batch_start:batch_start + mesh_batch_size]
                    tx_names: list[str] = []
                    height_name = f"{height:.3f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")
                    phase_started = time.perf_counter()
                    for point in batch_points:
                        name = f"tx_{height_name}_{point.index:05d}"
                        scene.add(rt.Transmitter(
                            name=name,
                            position=[point.x_m, point.y_m, height],
                            power_dbm=config.tx_power_dbm,
                            display_radius=0.8,
                        ))
                        tx_names.append(name)
                    tx_setup_seconds = time.perf_counter() - phase_started

                    phase_started = time.perf_counter()
                    paths = solver(
                        scene,
                        max_depth=max_depth,
                        samples_per_src=samples_per_tx,
                        los=config.los,
                        specular_reflection=config.specular_reflection,
                        diffuse_reflection=config.diffuse_reflection,
                        refraction=config.refraction,
                        diffraction=config.diffraction,
                    )
                    solver_call_seconds = time.perf_counter() - phase_started

                    # out_type="numpy" synchronizes pending GPU work and transfers
                    # the CIR to host memory, so this phase can contain work that
                    # was submitted asynchronously by the solver call above.
                    phase_started = time.perf_counter()
                    a, _tau = paths.cir(normalize_delays=False, out_type="numpy")
                    cir_numpy_seconds = time.perf_counter() - phase_started

                    phase_started = time.perf_counter()
                    gains = path_gain_matrix_from_cir(a, len(rx_points), len(batch_points))
                    del paths, a, _tau
                    gain_reduce_seconds = time.perf_counter() - phase_started

                    phase_started = time.perf_counter()
                    rows: list[dict[str, object]] = []
                    for rx_offset, rx_point in enumerate(rx_points):
                        for tx_offset, tx_point in enumerate(batch_points):
                            gain = float(gains[rx_offset, tx_offset])
                            safe_gain = max(gain, 1e-30)
                            rx_power_dbm = config.tx_power_dbm + 10.0 * math.log10(safe_gain)
                            rows.append({
                                "rx_index": rx_point.index,
                                "tx_voxel_ix": tx_point.index % horizontal_axis_count,
                                "tx_voxel_iy": tx_point.index // horizontal_axis_count,
                                "tx_voxel_iz": height_index,
                                "rss_dbm": f"{rx_power_dbm:.3f}",
                            })
                            height_count += 1
                            height_sum += rx_power_dbm
                            height_min = min(height_min, rx_power_dbm)
                            height_max = max(height_max, rx_power_dbm)
                    row_build_seconds = time.perf_counter() - phase_started

                    phase_started = time.perf_counter()
                    writer.writerows(rows)
                    written_row_count = len(rows)
                    del gains, rows
                    csv_write_seconds = time.perf_counter() - phase_started
                    state["total_rows"] = int(state["total_rows"]) + written_row_count

                    phase_started = time.perf_counter()
                    for name in tx_names:
                        scene.remove(name)
                    tx_cleanup_seconds = time.perf_counter() - phase_started

                    checkpoint_started = time.perf_counter()
                    batch_end = min(batch_start + mesh_batch_size, len(tx_points))
                    if batch_end == len(tx_points):
                        summaries = list(state["height_summaries"])
                        summaries.append({
                            "height_m": height,
                            "row_count": height_count,
                            "sim_rx_power_mean_dbm": height_sum / height_count,
                            "sim_rx_power_min_dbm": height_min,
                            "sim_rx_power_max_dbm": height_max,
                        })
                        state.update({
                            "next_height_index": height_index + 1,
                            "next_batch_start": 0,
                            "height_summaries": summaries,
                            "current_height_count": 0,
                            "current_height_sum": 0.0,
                            "current_height_min": None,
                            "current_height_max": None,
                        })
                    else:
                        state.update({
                            "next_height_index": height_index,
                            "next_batch_start": batch_end,
                            "current_height_count": height_count,
                            "current_height_sum": height_sum,
                            "current_height_min": height_min,
                            "current_height_max": height_max,
                        })
                    file.flush()
                    state["csv_size_bytes"] = temp_csv.stat().st_size
                    write_json_checkpoint(state, checkpoint_path)
                    checkpoint_seconds = time.perf_counter() - checkpoint_started
                    batch_total_seconds = time.perf_counter() - batch_started
                    batch_number = (batch_start - batch_start_offset) // mesh_batch_size + 1
                    batch_timings = {
                        "tx_setup": tx_setup_seconds,
                        "solver_call": solver_call_seconds,
                        "cir_numpy_sync": cir_numpy_seconds,
                        "gain_reduce": gain_reduce_seconds,
                        "row_build": row_build_seconds,
                        "csv_write": csv_write_seconds,
                        "tx_cleanup": tx_cleanup_seconds,
                        "checkpoint": checkpoint_seconds,
                        "batch_total": batch_total_seconds,
                    }
                    if ENABLE_TIMING_LOGS:
                        for key, value in batch_timings.items():
                            timing_totals[key] += value
                            height_timing_totals[key] += value
                        timed_batch_count += 1
                        height_timed_batch_count += 1
                    if ENABLE_TIMING_LOGS and batch_end == len(tx_points):
                        print(
                            f"HEIGHT TIMING {region_slug}/{block_id} height={height:g}m "
                            f"measured_batches={height_timed_batch_count}/{remaining_batch_count} "
                            + " ".join(
                                f"avg_{key}={value / height_timed_batch_count:.3f}s"
                                for key, value in height_timing_totals.items()
                            ),
                            flush=True,
                        )
                    if (
                        batch_number == 1
                        or batch_number == remaining_batch_count
                        or batch_number % 25 == 0
                    ):
                        print(
                            f"  batch {batch_number}/{remaining_batch_count}; "
                            f"cumulative rows={state['total_rows']}"
                        )
        if int(state["next_height_index"]) != len(heights_m):
            raise RuntimeError("Solver stopped before all height groups completed")
        temp_csv.replace(out_csv)
    except BaseException:
        print(f"CHECKPOINT RETAINED: {checkpoint_path}")
        if ENABLE_TIMING_LOGS and timed_batch_count:
            print(
                f"RT TIMING PARTIAL {region_slug}/{block_id}: batches={timed_batch_count} "
                f"loop_wall={time.perf_counter() - rt_loop_started:.3f}s "
                + " ".join(
                    f"avg_{key}={value / timed_batch_count:.3f}s"
                    for key, value in timing_totals.items()
                ),
                flush=True,
            )
        raise

    if ENABLE_TIMING_LOGS and timed_batch_count:
        print(
            f"RT TIMING COMPLETE {region_slug}/{block_id}: batches={timed_batch_count} "
            f"loop_wall={time.perf_counter() - rt_loop_started:.3f}s "
            + " ".join(
                f"avg_{key}={value / timed_batch_count:.3f}s"
                for key, value in timing_totals.items()
            ),
            flush=True,
        )
    return {
        "row_count": int(state["total_rows"]),
        "height_summaries": list(state["height_summaries"]),
        "mesh_batch_sizes_used": list(state.get("mesh_batch_sizes_used", [mesh_batch_size])),
    }


def run_sionna_rt_voxel_rx(
    config: RtExperimentConfig,
    scene_xml: Path,
    out_csv: Path,
    tx_points: list[GroundTxPoint],
    voxel_points: list[VoxelRxPoint],
    rx_batch_size: int,
    samples_per_tx: int,
    max_depth: int,
    region_slug: str,
    block_id: str,
    environment_signature: str,
    voxel_shape: tuple[int, int, int],
    voxel_size_m: float,
    resume: bool = True,
) -> dict[str, object]:
    """Solve fixed ground transmitters against a dense 3-D receiver voxel grid.

    The receiver grid is processed in batches and every confirmed batch is
    checkpointed. The large ``Paths`` and CIR objects are explicitly released
    before the next PathSolver call to avoid a two-batch GPU-memory peak.
    """
    if not tx_points or not voxel_points:
        raise ValueError("Ground TX points and voxel RX points must be non-empty")
    if rx_batch_size <= 0:
        rx_batch_size = len(voxel_points)

    fieldnames = [
        "region_slug", "block_id",
        "tx_index", "tx_x_m", "tx_y_m", "tx_z_m", "tx_lat", "tx_lon",
        "voxel_index", "voxel_ix", "voxel_iy", "voxel_iz",
        "rx_x_m", "rx_y_m", "rx_z_m", "rx_lat", "rx_lon",
        "voxel_size_m", "distance_3d_m", "path_gain_linear", "path_loss_db",
        "tx_power_dbm", "tx_antenna_pattern", "rx_antenna_pattern",
        "polarization", "sim_rx_power_dbm",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    temp_csv = out_csv.with_suffix(out_csv.suffix + ".tmp")
    checkpoint_path = out_csv.with_suffix(out_csv.suffix + ".checkpoint.json")
    signature = {
        "schema_version": 3,
        "layout": "fixed_ground_tx_to_3d_voxel_rx",
        "region_slug": region_slug,
        "block_id": block_id,
        "environment_signature": environment_signature,
        "voxel_shape_xyz": list(voxel_shape),
        "voxel_size_m": float(voxel_size_m),
        "voxel_count": len(voxel_points),
        "tx_positions_xyz_m": [
            [round(point.x_m, 6), round(point.y_m, 6), round(point.z_m, 6)]
            for point in tx_points
        ],
        "rx_batch_size": rx_batch_size,
        "samples_per_tx": samples_per_tx,
        "max_depth": max_depth,
    }
    expected_rows = len(tx_points) * len(voxel_points)
    state: dict[str, object] | None = None
    candidate: dict[str, object] | None = None
    if resume and checkpoint_path.is_file():
        try:
            loaded = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                candidate = loaded
        except (OSError, ValueError) as exc:
            print(f"IGNORING INVALID CHECKPOINT {checkpoint_path}: {exc}")

    if (
        candidate is not None
        and candidate.get("signature") == signature
        and int(candidate.get("next_voxel_start", -1)) == len(voxel_points)
        and int(candidate.get("total_rows", -1)) == expected_rows
        and out_csv.is_file()
        and out_csv.stat().st_size == int(candidate.get("csv_size_bytes", -1))
    ):
        print(
            f"RESUME POSTPROCESS {region_slug}/{block_id}: voxel RT CSV already complete; "
            f"rows {candidate['total_rows']}"
        )
        return {
            "row_count": int(candidate["total_rows"]),
            "summary": dict(candidate["summary"]),
        }

    if candidate is not None and candidate.get("signature") == signature and temp_csv.is_file():
        confirmed_size = int(candidate.get("csv_size_bytes", -1))
        if 0 <= confirmed_size <= temp_csv.stat().st_size:
            with temp_csv.open("r+b") as file:
                file.truncate(confirmed_size)
            state = candidate
            print(
                f"RESUME CHECKPOINT {region_slug}/{block_id}: voxel offset "
                f"{state['next_voxel_start']}, rows {state['total_rows']}"
            )
        else:
            print(f"IGNORING INVALID CHECKPOINT SIZE {checkpoint_path}")

    if state is None:
        with temp_csv.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            file.flush()
        state = {
            "signature": signature,
            "next_voxel_start": 0,
            "total_rows": 0,
            "power_count": 0,
            "power_sum": 0.0,
            "power_min": None,
            "power_max": None,
            "summary": {},
            "csv_size_bytes": temp_csv.stat().st_size,
        }
        write_json_checkpoint(state, checkpoint_path)

    import sionna.rt as rt

    scene = rt.load_scene(str(scene_xml), merge_shapes=True)
    scene.frequency = config.frequency_hz
    scene.tx_array = rt.PlanarArray(
        num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="iso", polarization="V",
    )
    scene.rx_array = rt.PlanarArray(
        num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="iso", polarization="V",
    )
    for point in tx_points:
        scene.add(rt.Transmitter(
            name=f"tx_{point.index:03d}",
            position=[point.x_m, point.y_m, point.z_m],
            power_dbm=config.tx_power_dbm,
            display_radius=0.8,
        ))
    solver = rt.PathSolver()
    batches = math.ceil(len(voxel_points) / rx_batch_size)
    start_offset = int(state["next_voxel_start"])
    power_count = int(state["power_count"])
    power_sum = float(state["power_sum"])
    power_min = math.inf if state["power_min"] is None else float(state["power_min"])
    power_max = -math.inf if state["power_max"] is None else float(state["power_max"])
    print(
        f"Solving {len(tx_points)} fixed ground TX against {len(voxel_points)} voxel RX "
        f"({voxel_shape[0]}x{voxel_shape[1]}x{voxel_shape[2]}) in {batches} batches "
        f"(start offset {start_offset}) ..."
    )

    try:
        with temp_csv.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            for batch_start in range(start_offset, len(voxel_points), rx_batch_size):
                batch_points = voxel_points[batch_start:batch_start + rx_batch_size]
                rx_names: list[str] = []
                for point in batch_points:
                    name = f"rx_voxel_{point.index:06d}"
                    scene.add(rt.Receiver(
                        name=name,
                        position=[point.x_m, point.y_m, point.z_m],
                        orientation=[0.0, 0.0, 0.0],
                        display_radius=0.25,
                    ))
                    rx_names.append(name)

                paths = solver(
                    scene,
                    max_depth=max_depth,
                    samples_per_src=samples_per_tx,
                    los=config.los,
                    specular_reflection=config.specular_reflection,
                    diffuse_reflection=config.diffuse_reflection,
                    refraction=config.refraction,
                    diffraction=config.diffraction,
                )
                a, tau = paths.cir(normalize_delays=False, out_type="numpy")
                gains = path_gain_matrix_from_cir(a, len(batch_points), len(tx_points))
                del paths, a, tau

                rows: list[dict[str, object]] = []
                for rx_offset, voxel in enumerate(batch_points):
                    for tx_offset, tx_point in enumerate(tx_points):
                        gain = float(gains[rx_offset, tx_offset])
                        safe_gain = max(gain, 1e-30)
                        rx_power_dbm = config.tx_power_dbm + 10.0 * math.log10(safe_gain)
                        distance_3d = math.sqrt(
                            (voxel.x_m - tx_point.x_m) ** 2
                            + (voxel.y_m - tx_point.y_m) ** 2
                            + (voxel.z_m - tx_point.z_m) ** 2
                        )
                        rows.append({
                            "region_slug": region_slug,
                            "block_id": block_id,
                            "tx_index": tx_point.index,
                            "tx_x_m": f"{tx_point.x_m:.3f}",
                            "tx_y_m": f"{tx_point.y_m:.3f}",
                            "tx_z_m": f"{tx_point.z_m:.3f}",
                            "tx_lat": f"{tx_point.lat:.10f}",
                            "tx_lon": f"{tx_point.lon:.10f}",
                            "voxel_index": voxel.index,
                            "voxel_ix": voxel.ix,
                            "voxel_iy": voxel.iy,
                            "voxel_iz": voxel.iz,
                            "rx_x_m": f"{voxel.x_m:.3f}",
                            "rx_y_m": f"{voxel.y_m:.3f}",
                            "rx_z_m": f"{voxel.z_m:.3f}",
                            "rx_lat": f"{voxel.lat:.10f}",
                            "rx_lon": f"{voxel.lon:.10f}",
                            "voxel_size_m": f"{voxel_size_m:.3f}",
                            "distance_3d_m": f"{distance_3d:.3f}",
                            "path_gain_linear": f"{gain:.12e}",
                            "path_loss_db": f"{-10.0 * math.log10(safe_gain):.3f}",
                            "tx_power_dbm": f"{config.tx_power_dbm:.1f}",
                            "tx_antenna_pattern": "iso",
                            "rx_antenna_pattern": "iso",
                            "polarization": "V",
                            "sim_rx_power_dbm": f"{rx_power_dbm:.3f}",
                        })
                        power_count += 1
                        power_sum += rx_power_dbm
                        power_min = min(power_min, rx_power_dbm)
                        power_max = max(power_max, rx_power_dbm)
                writer.writerows(rows)
                del gains, rows
                for name in rx_names:
                    scene.remove(name)

                batch_end = min(batch_start + rx_batch_size, len(voxel_points))
                state.update({
                    "next_voxel_start": batch_end,
                    "total_rows": int(state["total_rows"]) + len(batch_points) * len(tx_points),
                    "power_count": power_count,
                    "power_sum": power_sum,
                    "power_min": power_min,
                    "power_max": power_max,
                })
                if batch_end == len(voxel_points):
                    state["summary"] = {
                        "row_count": power_count,
                        "sim_rx_power_mean_dbm": power_sum / power_count,
                        "sim_rx_power_min_dbm": power_min,
                        "sim_rx_power_max_dbm": power_max,
                    }
                file.flush()
                state["csv_size_bytes"] = temp_csv.stat().st_size
                write_json_checkpoint(state, checkpoint_path)
                batch_number = batch_start // rx_batch_size + 1
                if batch_number == 1 or batch_number == batches or batch_number % 25 == 0:
                    print(
                        f"  voxel batch {batch_number}/{batches}; "
                        f"cumulative rows={state['total_rows']}"
                    )
        if int(state["next_voxel_start"]) != len(voxel_points):
            raise RuntimeError("Solver stopped before all voxel batches completed")
        temp_csv.replace(out_csv)
    except BaseException:
        print(f"CHECKPOINT RETAINED: {checkpoint_path}")
        raise

    return {"row_count": int(state["total_rows"]), "summary": dict(state["summary"])}


def write_json_checkpoint(state: dict[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def path_gain_matrix_from_cir(a: np.ndarray, n_rx: int, n_tx: int) -> np.ndarray:
    """Reduce CIR coefficients to one scalar path gain per RX/TX pair."""
    power = np.abs(np.asarray(a)) ** 2
    if power.ndim != 6:
        raise ValueError(f"Unexpected CIR coefficient shape {power.shape}; expected 6 dimensions")
    gains = power.sum(axis=(1, 3, 4, 5))
    gains = np.asarray(gains, dtype=float)
    if gains.shape != (n_rx, n_tx):
        raise ValueError(f"Unexpected gain matrix shape {gains.shape}; expected {(n_rx, n_tx)}")
    return gains


def run_sionna_rt_multi_rx_npy(
    config: RtExperimentConfig,
    scene_xml: Path,
    block_dir: Path,
    tx_points: list[MeshTxPoint],
    rx_points: list[GroundRxPoint],
    heights_m: list[float],
    tensor_shape_zyx: tuple[int, int, int],
    mesh_batch_size: int,
    samples_per_tx: int,
    max_depth: int,
    region_slug: str,
    block_id: str,
    environment_signature: str,
    resume: bool = True,
) -> dict[str, object]:
    """Directly write one resumable float32 ``[Z,Y,X]`` NPY per RX."""
    from rx_sharded_results import group_checkpoint_path, rx_npy_path, rx_partial_path

    nz, ny, nx = tensor_shape_zyx
    if not rx_points or len(heights_m) != nz or len(tx_points) != nx * ny:
        raise ValueError("RX points and tensor dimensions do not match the voxel grid")
    mesh_batch_size = mesh_batch_size if mesh_batch_size > 0 else len(tx_points)
    indices = [point.index for point in rx_points]
    checkpoint_path = group_checkpoint_path(block_dir, indices)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    signature = {
        "schema_version": 5,
        "tensor_schema": "rss_float32_zyx_v1",
        "layout": "3d_voxel_tx_to_fixed_ground_rx",
        "region_slug": region_slug,
        "block_id": block_id,
        "environment_signature": environment_signature,
        "tensor_shape_zyx": list(tensor_shape_zyx),
        "heights_m": [float(value) for value in heights_m],
        "rx_indices": indices,
        "rx_positions_xy_m": [[round(point.x_m, 6), round(point.y_m, 6)] for point in rx_points],
        "mesh_batch_size": mesh_batch_size,
        "samples_per_tx": samples_per_tx,
        "max_depth": max_depth,
        "mesh_spacing_m": config.mesh_spacing_m,
    }
    candidate = None
    if resume and checkpoint_path.is_file():
        try:
            loaded = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            candidate = loaded if isinstance(loaded, dict) else None
        except (OSError, ValueError) as exc:
            print(f"IGNORING INVALID CHECKPOINT {checkpoint_path}: {exc}")

    compatible = candidate is not None and checkpoint_signatures_compatible(
        candidate.get("signature"), signature
    )
    final_paths = {point.index: rx_npy_path(block_dir, point.index) for point in rx_points}
    partial_paths = {point.index: rx_partial_path(block_dir, point.index) for point in rx_points}
    if compatible and int(candidate.get("next_height_index", -1)) == nz:
        try:
            arrays = [np.load(final_paths[index], mmap_mode="r", allow_pickle=False) for index in indices]
            if all(array.shape == tensor_shape_zyx and array.dtype == np.dtype("float32") for array in arrays):
                print(f"RESUME POSTPROCESS {region_slug}/{block_id}: RX tensors already complete")
                return {
                    "value_count": len(indices) * nx * ny * nz,
                    "per_rx_height_summaries": candidate["per_rx_height_summaries"],
                    "mesh_batch_sizes_used": checkpoint_batch_size_history(candidate, 0),
                    "checkpoint_path": str(checkpoint_path),
                }
        except (OSError, ValueError, KeyError):
            pass

    state = None
    if compatible and all(path.is_file() for path in partial_paths.values()):
        try:
            arrays = {
                index: np.load(path, mmap_mode="r+", allow_pickle=False)
                for index, path in partial_paths.items()
            }
            if all(array.shape == tensor_shape_zyx and array.dtype == np.dtype("float32") for array in arrays.values()):
                state = candidate
                saved_size = state.get("signature", {}).get("mesh_batch_size")
                state["signature"] = signature
                state["mesh_batch_sizes_used"] = checkpoint_batch_size_history(state, mesh_batch_size)
                print(
                    f"RESUME CHECKPOINT {region_slug}/{block_id}: height index "
                    f"{state['next_height_index']}, TX offset {state['next_batch_start']}; "
                    f"batch {saved_size} -> {mesh_batch_size}", flush=True,
                )
            else:
                arrays = {}
        except (OSError, ValueError):
            arrays = {}
    else:
        arrays = {}
    if state is None:
        arrays = {}
        for index, path in partial_paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=tensor_shape_zyx)
            array[:] = np.nan
            array.flush()
            arrays[index] = array
        state = {
            "signature": signature,
            "mesh_batch_sizes_used": [mesh_batch_size],
            "next_height_index": 0,
            "next_batch_start": 0,
            "per_rx_height_summaries": {str(index): [] for index in indices},
            "current_height_stats": {
                str(index): {"count": 0, "sum": 0.0, "min": None, "max": None}
                for index in indices
            },
        }
        write_json_checkpoint(state, checkpoint_path)

    import sionna.rt as rt

    drjit_module = None
    drjit_previous_history_flag = None
    if ENABLE_DRJIT_KERNEL_HISTORY:
        try:
            import drjit as dr

            drjit_module = dr
            drjit_previous_history_flag = dr.flag(dr.JitFlag.KernelHistory)
            dr.set_flag(dr.JitFlag.KernelHistory, True)
            # Official API: retrieving the history also clears it. Discard
            # records made before this block/RX group to keep the scope exact.
            dr.kernel_history()
            print(
                "DRJIT OFFICIAL DIAGNOSTICS ENABLED: KernelHistory; "
                "summary_scope=one_height; raw_info_log=disabled",
                flush=True,
            )
        except BaseException as exc:
            drjit_module = None
            drjit_previous_history_flag = None
            print(
                f"DRJIT OFFICIAL DIAGNOSTICS UNAVAILABLE: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )

    scene = rt.load_scene(str(scene_xml), merge_shapes=True)
    scene.frequency = config.frequency_hz
    scene.tx_array = rt.PlanarArray(
        num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="iso", polarization="V",
    )
    scene.rx_array = rt.PlanarArray(
        num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="iso", polarization="V",
    )
    for point in rx_points:
        scene.add(rt.Receiver(
            name=f"rx_{point.index:03d}", position=[point.x_m, point.y_m, config.rx_height_m],
            orientation=[0.0, 0.0, 0.0], display_radius=1.0,
        ))
    solver = rt.PathSolver()
    print(
        f"CHECKPOINT POLICY {region_slug}/{block_id}: persist every "
        f"{CHECKPOINT_INTERVAL_BATCHES} batches and at each height end",
        flush=True,
    )
    timing_keys = (
        "tx_setup",
        "solver_call",
        "cir_numpy_sync",
        "gain_reduce",
        "tensor_write_stats",
        "npy_flush",
        "checkpoint",
        "tx_cleanup",
    )

    def print_height_timing(
        *,
        height: float,
        batch_count: int,
        totals: dict[str, float],
        elapsed: float,
        partial: bool = False,
    ) -> None:
        if not ENABLE_TIMING_LOGS or batch_count <= 0:
            return
        accounted = sum(totals[key] for key in timing_keys)
        other = max(0.0, elapsed - accounted)
        label = "HEIGHT TIMING PARTIAL" if partial else "HEIGHT TIMING"
        fields = []
        for key in timing_keys:
            seconds = totals[key]
            percentage = 100.0 * seconds / elapsed if elapsed > 0.0 else 0.0
            fields.append(
                f"{key}={seconds:.3f}s(avg={seconds / batch_count:.3f}s,{percentage:.1f}%)"
            )
        other_percentage = 100.0 * other / elapsed if elapsed > 0.0 else 0.0
        fields.append(f"other={other:.3f}s({other_percentage:.1f}%)")
        print(
            f"{label} {region_slug}/{block_id} height={height:g}m "
            f"batches={batch_count} total={elapsed:.3f}s "
            f"avg_batch={elapsed / batch_count:.3f}s "
            + " ".join(fields),
            flush=True,
        )

    def print_drjit_kernel_history(*, height: float, partial: bool = False) -> None:
        if drjit_module is None:
            return
        try:
            records = drjit_module.kernel_history()
            summary = summarize_drjit_kernel_history(records)
            type_counts = ",".join(
                f"{name}:{count}"
                for name, count in sorted(summary["type_counts"].items())
            ) or "none"
            label = "DRJIT KERNEL HISTORY PARTIAL" if partial else "DRJIT KERNEL HISTORY"
            print(
                f"{label} {region_slug}/{block_id} height={height:g}m "
                f"records={summary['records']} jit={summary['jit_kernels']} "
                f"memory_hits={summary['memory_hits']} disk_hits={summary['disk_hits']} "
                f"hard_misses={summary['hard_misses']} "
                f"optix={summary['optix_kernels']} unique_hashes={summary['unique_hashes']} "
                f"execution_time_ms={summary['execution_time_ms']:.3f} "
                f"codegen_time_ms={summary['codegen_time_ms']:.3f} "
                f"backend_time_ms={summary['backend_time_ms']:.3f} "
                f"miss_codegen_time_ms={summary['miss_codegen_time_ms']:.3f} "
                f"miss_backend_time_ms={summary['miss_backend_time_ms']:.3f} "
                f"miss_compile_time_ms={summary['miss_compile_time_ms']:.3f} "
                f"miss_execution_time_ms={summary['miss_execution_time_ms']:.3f} "
                f"hit_execution_time_ms={summary['hit_execution_time_ms']:.3f} "
                f"types={type_counts}",
                flush=True,
            )
        except BaseException as exc:
            print(
                f"DRJIT KERNEL HISTORY READ FAILED {region_slug}/{block_id}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )

    try:
        first_height = int(state["next_height_index"])
        for height_index in range(first_height, nz):
            height = float(heights_m[height_index])
            offset = int(state["next_batch_start"]) if height_index == first_height else 0
            stats = state["current_height_stats"] if height_index == first_height else {
                str(index): {"count": 0, "sum": 0.0, "min": None, "max": None}
                for index in indices
            }
            remaining = math.ceil((len(tx_points) - offset) / mesh_batch_size)
            print(
                f"Solving {len(rx_points)} ground RX receivers against {len(tx_points)} aerial TX points "
                f"at {height:g} m in {remaining} remaining batches "
                f"(start offset {offset}, batch size {mesh_batch_size}) ...", flush=True,
            )
            height_started = time.perf_counter()
            height_timing = {key: 0.0 for key in timing_keys}
            height_batch_count = 0
            for batch_start in range(offset, len(tx_points), mesh_batch_size):
                batch_points = tx_points[batch_start:batch_start + mesh_batch_size]
                tx_names: list[str] = []
                # Clear references before evaluating the next solver call. In
                # Python the right-hand side of ``paths = solver(...)`` is
                # evaluated while the previous ``paths`` is still alive. That
                # briefly kept two large Dr.Jit path buffers in VRAM and could
                # OOM even when one batch fitted comfortably.
                paths = a = _tau = gains = powers = values = None
                height_name = f"{height:.3f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")
                batch_timing = {key: 0.0 for key in timing_keys}
                try:
                    phase_started = time.perf_counter()
                    for point in batch_points:
                        name = f"tx_{height_name}_{point.index:05d}"
                        scene.add(rt.Transmitter(
                            name=name, position=[point.x_m, point.y_m, height],
                            power_dbm=config.tx_power_dbm, display_radius=0.8,
                        ))
                        tx_names.append(name)
                    batch_timing["tx_setup"] = time.perf_counter() - phase_started

                    phase_started = time.perf_counter()
                    paths = solver(
                        scene, max_depth=max_depth, samples_per_src=samples_per_tx,
                        los=config.los, specular_reflection=config.specular_reflection,
                        diffuse_reflection=config.diffuse_reflection,
                        refraction=config.refraction, diffraction=config.diffraction,
                    )
                    batch_timing["solver_call"] = time.perf_counter() - phase_started

                    # out_type="numpy" is the first explicit host result here.
                    # This bucket includes pending CIR work, synchronization,
                    # and the device-to-host transfer.
                    phase_started = time.perf_counter()
                    a, _tau = paths.cir(normalize_delays=False, out_type="numpy")
                    batch_timing["cir_numpy_sync"] = time.perf_counter() - phase_started

                    phase_started = time.perf_counter()
                    gains = path_gain_matrix_from_cir(a, len(rx_points), len(batch_points))
                    powers = (
                        config.tx_power_dbm
                        + 10.0 * np.log10(np.maximum(gains, 1e-30))
                    ).astype(np.float32, copy=False)
                    batch_timing["gain_reduce"] = time.perf_counter() - phase_started

                    phase_started = time.perf_counter()
                    batch_end = batch_start + len(batch_points)
                    for rx_offset, point in enumerate(rx_points):
                        values = powers[rx_offset]
                        arrays[point.index][height_index].reshape(-1)[batch_start:batch_end] = values
                        item = stats[str(point.index)]
                        item["count"] = int(item["count"]) + len(values)
                        item["sum"] = float(item["sum"]) + float(values.sum(dtype=np.float64))
                        value_min, value_max = float(values.min()), float(values.max())
                        item["min"] = value_min if item["min"] is None else min(float(item["min"]), value_min)
                        item["max"] = value_max if item["max"] is None else max(float(item["max"]), value_max)
                    batch_timing["tensor_write_stats"] = time.perf_counter() - phase_started

                    batch_number = (batch_start - offset) // mesh_batch_size + 1
                    persist_batch = should_persist_batch(
                        batch_number, batch_end, len(tx_points)
                    )
                    phase_started = time.perf_counter()
                    if persist_batch:
                        for array in arrays.values():
                            array.flush()
                    batch_timing["npy_flush"] = time.perf_counter() - phase_started

                    phase_started = time.perf_counter()
                    if batch_end == len(tx_points):
                        summaries = state["per_rx_height_summaries"]
                        for index in indices:
                            item = stats[str(index)]
                            count = int(item["count"])
                            summaries[str(index)].append({
                                "tx_voxel_iz": height_index, "height_m": height,
                                "value_count": count, "rss_mean_dbm": float(item["sum"]) / count,
                                "rss_min_dbm": float(item["min"]), "rss_max_dbm": float(item["max"]),
                            })
                        state.update({
                            "next_height_index": height_index + 1, "next_batch_start": 0,
                            "current_height_stats": {
                                str(index): {"count": 0, "sum": 0.0, "min": None, "max": None}
                                for index in indices
                            },
                        })
                    else:
                        state.update({
                            "next_height_index": height_index, "next_batch_start": batch_end,
                            "current_height_stats": stats,
                        })
                    if persist_batch:
                        write_json_checkpoint(state, checkpoint_path)
                    batch_timing["checkpoint"] = time.perf_counter() - phase_started
                    if (
                        batch_number == 1
                        or batch_number == remaining
                        or batch_number % 25 == 0
                    ):
                        cumulative_values = (
                            height_index * len(tx_points) + batch_end
                        ) * len(rx_points)
                        print(
                            f"  batch {batch_number}/{remaining}; "
                            f"height TX={batch_end}/{len(tx_points)}; "
                            f"cumulative RSS values={cumulative_values}",
                            flush=True,
                        )
                finally:
                    # Release both successful and partially-created batch
                    # objects before the next PathSolver invocation.
                    cleanup_started = time.perf_counter()
                    paths = a = _tau = gains = powers = values = None
                    for name in tx_names:
                        scene.remove(name)
                    batch_timing["tx_cleanup"] = time.perf_counter() - cleanup_started
                    if ENABLE_TIMING_LOGS:
                        for key in timing_keys:
                            height_timing[key] += batch_timing[key]
                        height_batch_count += 1
            height_elapsed = time.perf_counter() - height_started
            print_height_timing(
                height=height,
                batch_count=height_batch_count,
                totals=height_timing,
                elapsed=height_elapsed,
            )
            print_drjit_kernel_history(height=height)
            print(
                f"HEIGHT COMPLETE {region_slug}/{block_id}: height={height:g}m "
                f"elapsed={height_elapsed:.3f}s", flush=True,
            )
        if int(state["next_height_index"]) != nz:
            raise RuntimeError("Solver stopped before every height completed")
        for index, array in list(arrays.items()):
            array.flush()
            if not np.isfinite(array).all():
                raise ValueError(f"Non-finite RSS values remain in RX {index} tensor")
        del array
        arrays.clear()
        for index in indices:
            os.replace(partial_paths[index], final_paths[index])
    except BaseException:
        if "height_timing" in locals() and "height_started" in locals():
            print_height_timing(
                height=height,
                batch_count=height_batch_count,
                totals=height_timing,
                elapsed=time.perf_counter() - height_started,
                partial=True,
            )
        if "height" in locals():
            print_drjit_kernel_history(height=height, partial=True)
        print(f"CHECKPOINT RETAINED: {checkpoint_path}", flush=True)
        raise
    finally:
        if drjit_module is not None and drjit_previous_history_flag is not None:
            try:
                drjit_module.set_flag(
                    drjit_module.JitFlag.KernelHistory,
                    drjit_previous_history_flag,
                )
            except BaseException as exc:
                print(
                    f"DRJIT KERNEL HISTORY RESTORE FAILED: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
    return {
        "value_count": len(indices) * nx * ny * nz,
        "per_rx_height_summaries": state["per_rx_height_summaries"],
        "mesh_batch_sizes_used": checkpoint_batch_size_history(state, mesh_batch_size),
        "checkpoint_path": str(checkpoint_path),
    }


def run_sionna_rt(
    config: RtExperimentConfig,
    scene_xml: Path,
    buildings: list[Building],
    out_dir: Path,
    mesh_spacing_m: float,
    mesh_batch_size: int,
    max_grid_points_per_height: int,
    samples_per_tx: int,
    max_depth: int,
    mesh_points_override: list[MeshTxPoint] | None = None,
) -> list[dict[str, object]]:
    import sionna.rt as rt
    from rt_osm_scene import lonlat_to_local_m

    mesh_points = (
        list(mesh_points_override)
        if mesh_points_override is not None
        else generate_tx_mesh(config, mesh_spacing_m, max_grid_points_per_height)
    )
    print(
        f"Generated {len(mesh_points)} mesh transmitter points per height "
        f"inside radius {config.radius_m:.1f} m, spacing {mesh_spacing_m:.1f} m."
    )
    write_mesh_points_csv(mesh_points, out_dir / f"sionna_rt_mesh_points_{config.run_tag}.csv")

    scene = rt.load_scene(str(scene_xml), merge_shapes=True)
    scene.frequency = config.frequency_hz
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="iso", polarization="V")
    scene.rx_array = rt.PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern=config.rx_antenna_pattern,
        polarization=config.rx_antenna_polarization,
    )

    rx_x, rx_y = lonlat_to_local_m(config.lon_rx, config.lat_rx, config.lon0, config.lat0)
    rx_orientation_rad = [math.radians(value) for value in config.rx_orientation_deg]
    scene.add(rt.Receiver(
        name="rx",
        position=[rx_x, rx_y, config.rx_height_m],
        orientation=rx_orientation_rad,
        display_radius=1.5,
    ))

    solver = rt.PathSolver()
    results: list[dict[str, object]] = []
    if mesh_batch_size <= 0:
        mesh_batch_size = len(mesh_points)

    for height in config.selected_experiment_heights:
        print(f"Solving RT mesh for {height} m group, {len(mesh_points)} transmitters ...")
        height_name = f"{float(height):.3f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")
        for batch_start in range(0, len(mesh_points), mesh_batch_size):
            batch_points = mesh_points[batch_start:batch_start + mesh_batch_size]
            tx_names: list[str] = []
            for batch_idx, point in enumerate(batch_points):
                name = f"tx_{height_name}_{batch_start + batch_idx:04d}"
                scene.add(rt.Transmitter(
                    name=name,
                    position=[point.x_m, point.y_m, float(height)],
                    power_dbm=config.tx_power_dbm,
                    display_radius=1.0,
                ))
                tx_names.append(name)

            print(
                f"  batch {batch_start // mesh_batch_size + 1}/"
                f"{math.ceil(len(mesh_points) / mesh_batch_size)}: {len(batch_points)} mesh points"
            )
            paths = solver(
                scene,
                max_depth=max_depth,
                samples_per_src=samples_per_tx,
                los=config.los,
                specular_reflection=config.specular_reflection,
                diffuse_reflection=config.diffuse_reflection,
                refraction=config.refraction,
                diffraction=config.diffraction,
            )
            a, _tau = paths.cir(normalize_delays=False, out_type="numpy")
            gains = path_gains_from_cir(a, len(batch_points))

            for point, gain in zip(batch_points, gains):
                rx_power_dbm = config.tx_power_dbm + 10.0 * math.log10(max(gain, 1e-30)) + config.rx_extra_gain_dbi
                path_loss_db = -10.0 * math.log10(max(gain, 1e-30))
                distance_3d = math.sqrt((point.x_m - rx_x) ** 2 + (point.y_m - rx_y) ** 2 + (height - config.rx_height_m) ** 2)
                results.append({
                    "height_m": height,
                    "mesh_index": point.index,
                    "mesh_spacing_m": f"{mesh_spacing_m:.3f}",
                    "tx_x_m": f"{point.x_m:.3f}",
                    "tx_y_m": f"{point.y_m:.3f}",
                    "tx_radial_distance_m": f"{point.radial_distance_m:.3f}",
                    "tx_lat": f"{point.lat:.10f}",
                    "tx_lon": f"{point.lon:.10f}",
                    "tx_z_m": height,
                    "rx_lat": f"{config.lat_rx:.10f}",
                    "rx_lon": f"{config.lon_rx:.10f}",
                    "rx_z_m": config.rx_height_m,
                    "distance_3d_m": f"{distance_3d:.3f}",
                    "path_gain_linear": f"{gain:.12e}",
                    "path_loss_db": f"{path_loss_db:.3f}",
                    "tx_power_dbm": f"{config.tx_power_dbm:.1f}",
                    "rx_antenna_pattern": config.rx_antenna_pattern,
                    "rx_antenna_polarization": config.rx_antenna_polarization,
                    "rx_orientation_deg": ",".join(f"{value:.1f}" for value in config.rx_orientation_deg),
                    "rx_extra_gain_dbi": f"{config.rx_extra_gain_dbi:.1f}",
                    "sim_rx_power_dbm": f"{rx_power_dbm:.3f}",
                })

            for name in tx_names:
                scene.remove(name)

    csv_path = out_dir / f"sionna_rt_mesh_results_{config.run_tag}.csv"
    write_dict_csv(results, csv_path)
    print(f"Wrote RT mesh results: {csv_path}")
    write_rt_summary(config, results, out_dir / f"sionna_rt_mesh_summary_{config.run_tag}.csv")
    try_plot_results(config, results, out_dir)
    return results


def path_gains_from_cir(a: np.ndarray, n_tx: int) -> np.ndarray:
    a = np.asarray(a)
    power = np.abs(a) ** 2
    if power.ndim != 6:
        return power.reshape(-1)[:n_tx]
    gains = power.sum(axis=(1, 3, 4, 5))
    return gains[0, :n_tx]


def generate_tx_mesh(config: RtExperimentConfig, spacing_m: float, max_points: int) -> list[MeshTxPoint]:
    from rt_osm_scene import local_m_to_lonlat

    if spacing_m <= 0:
        raise ValueError("mesh spacing must be positive")

    n_steps = max(0, int(math.floor(config.radius_m / spacing_m)))
    coords = np.arange(-n_steps, n_steps + 1, dtype=float) * spacing_m
    points: list[MeshTxPoint] = []
    mesh_index = 0
    for y_m in coords:
        for x_m in coords:
            radial_distance_m = math.hypot(float(x_m), float(y_m))
            if radial_distance_m > config.radius_m + 1e-9:
                continue
            lat, lon = local_m_to_lonlat(float(x_m), float(y_m), config.lon0, config.lat0)
            points.append(MeshTxPoint(mesh_index, float(x_m), float(y_m), lat, lon, radial_distance_m))
            mesh_index += 1

    if max_points > 0 and len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points).round().astype(int)
        points = [points[int(i)] for i in np.unique(indices)]
        print(f"Limited mesh to {len(points)} points per height for this run.")

    return points


def write_mesh_points_csv(points: list[MeshTxPoint], path: Path) -> None:
    rows = [
        {
            "mesh_index": point.index,
            "x_m": f"{point.x_m:.3f}",
            "y_m": f"{point.y_m:.3f}",
            "radial_distance_m": f"{point.radial_distance_m:.3f}",
            "lat": f"{point.lat:.10f}",
            "lon": f"{point.lon:.10f}",
        }
        for point in points
    ]
    write_dict_csv(rows, path)


def write_rt_summary(config: RtExperimentConfig, rows: list[dict[str, object]], path: Path) -> None:
    summary: list[dict[str, object]] = []
    for height in config.selected_experiment_heights:
        vals = [
            float(row["sim_rx_power_dbm"])
            for row in rows
            if math.isclose(float(row["height_m"]), float(height), abs_tol=1e-6)
        ]
        if not vals:
            continue
        summary.append({
            "height_m": height,
            "n": len(vals),
            "sim_rx_power_mean_dbm": f"{np.mean(vals):.3f}",
            "sim_rx_power_median_dbm": f"{np.median(vals):.3f}",
            "sim_rx_power_min_dbm": f"{np.min(vals):.3f}",
            "sim_rx_power_max_dbm": f"{np.max(vals):.3f}",
        })
    write_dict_csv(summary, path)
    print(f"Wrote RT summary: {path}")


def try_plot_results(config: RtExperimentConfig, rows: list[dict[str, object]], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    if not rows:
        return
    fig, ax = plt.subplots(figsize=(10, 6), dpi=140)
    for height in config.selected_experiment_heights:
        vals = [
            float(row["sim_rx_power_dbm"])
            for row in rows
            if math.isclose(float(row["height_m"]), float(height), abs_tol=1e-6)
        ]
        if vals:
            ax.scatter([height] * len(vals), vals, s=16, alpha=0.6, label=f"Mesh sim {height} m")
    ax.set_xlabel("Experiment height (m)")
    ax.set_ylabel("Simulated RX power (dBm)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    path = out_dir / f"sionna_rt_mesh_rx_power_by_height_{config.run_tag}.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Wrote plot: {path}")


def write_dict_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
