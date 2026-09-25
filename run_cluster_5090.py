"""One-GPU worker for the 39-campus Sionna dataset Slurm job array.

Every array task is identical and dynamically claims unlocked blocks. Changing
the array size changes only the worker count; simulation settings stay here.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Cluster and dataset settings. The tensor schema uses a new dataset name so
# it cannot be accidentally mixed with earlier CSV blocks.
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_ROOT = Path(
    os.environ.get("RID_CLUSTER_ROOT", str(Path.home() / "vast" / "UAV_RM"))
).expanduser().resolve()
DATASET_NAME = "project985_39_main_voxel_256m_128x128x40_rxexpand_float32_rand10to32"
DATASET_ROOT = BASE_ROOT / DATASET_NAME

RANDOM_SEED = 20260921
RX_COUNT = 10
# Missing receivers are solved in groups to retain Sionna's multi-RX speedup.
# Changing this only changes VRAM/throughput; existing per-RX shards stay valid.
RX_SOLVER_GROUP_SIZE = 10
RX_HEIGHT_M = 0.5
RX_BUILDING_CLEARANCE_M = 5.0
VOXEL_SIZE_M = 2.0
VOXEL_NX, VOXEL_NY, VOXEL_NZ = 128, 128, 40
VOXEL_Z_START_M = 2.0
# Runtime-only setting: 5090 keeps the default 120, while submit_4090.slurm
# exports RID_TX_BATCH_SIZE=60. Checkpoints store absolute TX offsets and are
# deliberately compatible across batch-size changes and GPU models.
TX_BATCH_SIZE = int(os.environ.get("RID_TX_BATCH_SIZE", "120"))
SAMPLES_PER_TX = 10_000
MAX_DEPTH = 3
RENDER_PREVIEW_AFTER_BLOCK = False
# Print one detailed RT-stage timing summary after each completed height. This
# only reads clocks around existing work; it does not add GPU synchronizations.
ENABLE_STAGE_TIMING = True
# Record Dr.Jit's official per-kernel history and write a compact summary after
# every completed height. Raw Info logging is intentionally not used because
# thousands of kernels per block would make the worker log impractically large.
ENABLE_DRJIT_KERNEL_HISTORY = True
CHECKPOINT_INTERVAL_BATCHES = 25
ENABLE_GPU_MEMORY_LOG = True
# At a completed block boundary, release only unused Dr.Jit allocations. The
# compiled in-memory and on-disk kernel caches are deliberately preserved.
ENABLE_BLOCK_MEMORY_RECLAIM = True
OSM_HEIGHT_VARIANT = "synthetic_12m_plus_uniform_m2_20_v2_seed20260921_fraction1.0"
# Scheduling changes only which unfinished block is claimed first. It does not
# affect simulation results or checkpoint compatibility.
TASK_ORDER = os.environ.get("RID_TASK_ORDER", "simple_first").strip().lower()

CPU_THREADS_PER_WORKER = max(1, int(os.environ.get("RID_CPU_THREADS_PER_GPU", "8")))

GENERATOR = SCRIPT_DIR / "campus_sionna_dataset.py"
AUDITOR = SCRIPT_DIR / "audit_campus_sionna_dataset.py"
OPTIX_CHECKER = SCRIPT_DIR / "check_gpu_rt.py"
BASE_OFFLINE_OSM_CACHE = SCRIPT_DIR / "offline_osm_cache_985_256m"
RANDOMIZED_OSM_CACHE = SCRIPT_DIR / "osm_randomized_height_985_256m_u10_32"
COMPLETED_REGIONS_SKIP_FILE = SCRIPT_DIR / "completed_regions_skip.txt"


@dataclass(frozen=True)
class BlockTask:
    region_slug: str
    block_id: str
    estimated_buildings: int
    estimated_mesh_faces: int
    estimated_bytes: int


def configure_optix_library() -> Path | None:
    """Expose the driver-provided OptiX library to Dr.Jit."""
    configured = os.environ.get("DRJIT_LIBOPTIX_PATH")
    if configured:
        configured_path = Path(configured)
        if configured_path.is_file():
            print(f"OptiX library (configured): {configured_path}")
            return configured_path
        print(f"Warning: DRJIT_LIBOPTIX_PATH does not exist: {configured_path}")

    candidates = [
        Path("/usr/lib/x86_64-linux-gnu/libnvoptix.so.1"),
        Path("/lib/x86_64-linux-gnu/libnvoptix.so.1"),
        Path("/usr/lib64/libnvoptix.so.1"),
        Path("/usr/lib/aarch64-linux-gnu/libnvoptix.so.1"),
    ]
    for root in (
        Path("/usr/lib/x86_64-linux-gnu/nvidia"),
        Path("/usr/lib/aarch64-linux-gnu/nvidia"),
    ):
        if root.is_dir():
            candidates.extend(root.glob("*/libnvoptix.so.1"))

    for candidate in candidates:
        if candidate.is_file():
            resolved = candidate.resolve()
            os.environ["DRJIT_LIBOPTIX_PATH"] = str(resolved)
            print(f"OptiX library (auto-detected): {resolved}")
            return resolved
    print("Warning: libnvoptix.so.1 was not found in standard Ubuntu paths.")
    return None


def allocated_gpu_tokens() -> list[str]:
    """Return only the GPUs assigned to this Slurm job.

    Slurm may expose physical indices, MIG identifiers, or GPU UUIDs. Preserve
    the tokens verbatim when creating each one-GPU child environment.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible != "NoDevFiles":
        tokens = [token.strip() for token in visible.split(",") if token.strip()]
    else:
        if not os.environ.get("SLURM_JOB_ID") and os.environ.get("RID_ALLOW_DIRECT") != "1":
            raise RuntimeError(
                "This launcher must run inside a Slurm allocation. Submit "
                "submit_5090.slurm instead of running on the login node."
            )
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        tokens = [line.strip() for line in result.stdout.splitlines() if line.strip()]

    if not tokens:
        raise RuntimeError("No GPU was assigned to this job")
    if len(tokens) != 1:
        raise RuntimeError(
            f"Each array worker must receive exactly one GPU, got {tokens}. "
            "Keep --gpus=1 and change only the Slurm --array range."
        )
    return tokens


def acquire_file_lock(path: Path, *, blocking: bool):
    """Acquire a Linux advisory lock; closing the handle releases it."""
    try:
        import fcntl
    except ImportError:
        if os.environ.get("RID_ALLOW_DIRECT") == "1":
            return None
        raise RuntimeError("The cluster launcher requires Linux fcntl file locking")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        fcntl.flock(handle.fileno(), flags)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def worker_id() -> str:
    job_id = os.environ.get("SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID", "direct"))
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    return f"{job_id}_{task_id}"


def gpu_environment(token: str) -> dict[str, str]:
    temporary_root = Path(
        os.environ.get("SLURM_TMPDIR")
        or os.environ.get("TMPDIR")
        or "/tmp"
    ) / f"uav_rm_{worker_id()}"
    optix_cache = temporary_root / "optix_cache"
    cuda_cache = temporary_root / "cuda_cache"
    optix_cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    cuda_cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    environment = os.environ.copy()
    environment.update({
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": token,
        # Concurrent OptiX contexts must not share one SQLite disk-cache
        # database on the network-mounted home directory. API error 7012 is
        # otherwise possible when several array workers initialize together.
        "OPTIX_CACHE_PATH": str(optix_cache),
        "CUDA_CACHE_PATH": str(cuda_cache),
        "PYTHONUNBUFFERED": "1",
        "RID_ENABLE_TIMING": "1" if ENABLE_STAGE_TIMING else "0",
        "RID_DRJIT_KERNEL_HISTORY": "1" if ENABLE_DRJIT_KERNEL_HISTORY else "0",
        "RID_CHECKPOINT_INTERVAL_BATCHES": str(CHECKPOINT_INTERVAL_BATCHES),
        "OMP_NUM_THREADS": str(CPU_THREADS_PER_WORKER),
        "MKL_NUM_THREADS": str(CPU_THREADS_PER_WORKER),
        "OPENBLAS_NUM_THREADS": str(CPU_THREADS_PER_WORKER),
    })
    return environment


def log_gpu_memory(label: str) -> None:
    """Log this worker's VRAM plus its physical GPU totals via nvidia-smi."""
    if not ENABLE_GPU_MEMORY_LOG:
        return
    try:
        apps = subprocess.run(
            [
                "nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.splitlines()
        own_rows = []
        for row in csv.reader(apps):
            if len(row) >= 3 and row[0].strip() == str(os.getpid()):
                own_rows.append((row[1].strip(), float(row[2].strip())))
        if not own_rows:
            print(f"GPU MEMORY {label}: process entry unavailable", flush=True)
            return
        gpu_uuid = own_rows[0][0]
        process_used_mib = sum(row[1] for row in own_rows)
        devices = subprocess.run(
            [
                "nvidia-smi", "--query-gpu=uuid,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.splitlines()
        matching = None
        for row in csv.reader(devices):
            if len(row) >= 4 and row[0].strip() == gpu_uuid:
                matching = tuple(float(value.strip()) for value in row[1:4])
                break
        if matching is None:
            print(
                f"GPU MEMORY {label}: process_used_mib={process_used_mib:.0f} "
                f"gpu_uuid={gpu_uuid} device totals unavailable",
                flush=True,
            )
            return
        total_mib, used_mib, free_mib = matching
        print(
            f"GPU MEMORY {label}: process_used_mib={process_used_mib:.0f} "
            f"gpu_used_mib={used_mib:.0f} gpu_free_mib={free_mib:.0f} "
            f"gpu_total_mib={total_mib:.0f} gpu_uuid={gpu_uuid}",
            flush=True,
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        print(f"GPU MEMORY {label}: unavailable ({type(exc).__name__}: {exc})", flush=True)


def reclaim_drjit_block_memory(label: str) -> None:
    """Return unused block allocations to CUDA without clearing JIT kernels."""
    if not ENABLE_BLOCK_MEMORY_RECLAIM:
        log_gpu_memory(f"after_block={label} reclaim=disabled")
        return

    log_gpu_memory(f"before_reclaim after_block={label}")
    started = time.perf_counter()
    collected = gc.collect()

    import drjit as dr

    # Dr.Jit device frees are asynchronous. Finish this worker thread's queue
    # before releasing its now-unused allocation cache back to CUDA.
    dr.sync_thread()
    dr.flush_malloc_cache()
    elapsed = time.perf_counter() - started
    print(
        f"DRJIT MEMORY RECLAIM after_block={label}: elapsed={elapsed:.3f}s "
        f"python_gc_collected={collected} malloc_cache=flushed "
        "kernel_cache=preserved",
        flush=True,
    )
    log_gpu_memory(f"after_reclaim after_block={label}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_if_different(source: Path, target: Path) -> str:
    """Atomically install one bundled cache file."""
    if target.is_file() and target.stat().st_size > 0:
        if source.stat().st_size == target.stat().st_size and sha256_file(source) == sha256_file(target):
            return "unchanged"
        result = "replaced"
    else:
        result = "copied"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".seed_tmp")
    shutil.copy2(source, temporary)
    temporary.replace(target)
    return result


def install_offline_osm_cache() -> None:
    """Install only the shared manifest/boundaries; blocks copy OSM on claim."""
    manifest_source = BASE_OFFLINE_OSM_CACHE / "blocks_manifest.csv"
    if not manifest_source.is_file():
        raise FileNotFoundError(f"Missing offline manifest: {manifest_source}")
    if not RANDOMIZED_OSM_CACHE.is_dir():
        raise FileNotFoundError(f"Missing randomized OSM cache: {RANDOMIZED_OSM_CACHE}")

    support_sources = [manifest_source]
    support_sources.extend(BASE_OFFLINE_OSM_CACHE.rglob("region_boundary.geojson"))
    support_counts = {"copied": 0, "replaced": 0, "unchanged": 0}
    for source in support_sources:
        relative = source.relative_to(BASE_OFFLINE_OSM_CACHE)
        support_counts[copy_if_different(source, DATASET_ROOT / relative)] += 1
    print(f"Offline manifest/boundaries ready: {support_counts}")


def install_block_osm(task: BlockTask) -> None:
    relative = Path(task.region_slug) / task.block_id / "osm_map.osm"
    source = RANDOMIZED_OSM_CACHE / relative
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(f"Missing bundled randomized OSM: {source}")
    status = copy_if_different(source, DATASET_ROOT / relative)
    print(f"OSM {status}: {task.region_slug}/{task.block_id}", flush=True)


def ensure_dataset_ready() -> None:
    """Let the first arriving worker install assets; later workers wait/skip."""
    lock = acquire_file_lock(DATASET_ROOT / ".dataset_prepare.lock", blocking=True)
    try:
        manifest_source = BASE_OFFLINE_OSM_CACHE / "blocks_manifest.csv"
        signature = ":".join([
            sha256_file(manifest_source),
            RANDOMIZED_OSM_CACHE.name,
            OSM_HEIGHT_VARIANT,
        ])
        marker = DATASET_ROOT / ".dataset_assets_ready"
        manifest_target = DATASET_ROOT / "blocks_manifest.csv"
        if (
            marker.is_file()
            and manifest_target.is_file()
            and marker.read_text(encoding="utf-8").strip() == signature
        ):
            print("Dataset assets already prepared by another worker.", flush=True)
            return
        install_offline_osm_cache()
        run_checked([
            sys.executable,
            str(GENERATOR),
            *common_generator_args(),
            "--prepare-only",
            "--log-name", "generation_prepare.log",
        ])
        temporary = marker.with_name(f"{marker.name}.{os.getpid()}.tmp")
        temporary.write_text(signature + "\n", encoding="utf-8")
        temporary.replace(marker)
    finally:
        if lock is not None:
            lock.close()


def common_generator_args() -> list[str]:
    args = [
        "--dataset-root", str(DATASET_ROOT),
        "--random-seed", str(RANDOM_SEED),
        "--osm-height-variant", OSM_HEIGHT_VARIANT,
        "--rx-count", str(RX_COUNT),
        "--rx-solver-group-size", str(RX_SOLVER_GROUP_SIZE),
        "--rx-height", str(RX_HEIGHT_M),
        "--rx-building-clearance", str(RX_BUILDING_CLEARANCE_M),
        "--voxel-size", str(VOXEL_SIZE_M),
        "--voxel-nx", str(VOXEL_NX),
        "--voxel-ny", str(VOXEL_NY),
        "--voxel-nz", str(VOXEL_NZ),
        "--voxel-z-start", str(VOXEL_Z_START_M),
        "--tx-batch-size", str(TX_BATCH_SIZE),
        "--samples-per-tx", str(SAMPLES_PER_TX),
        "--max-depth", str(MAX_DEPTH),
    ]
    if RENDER_PREVIEW_AFTER_BLOCK:
        args.append("--render-preview-after-block")
    return args


def run_checked(command: list[str], environment: dict[str, str] | None = None) -> None:
    print("LAUNCH:", " ".join(command), flush=True)
    subprocess.run(command, cwd=SCRIPT_DIR, env=environment, check=True)


def verify_optix_on_gpus(tokens: list[str]) -> None:
    if not OPTIX_CHECKER.is_file():
        raise FileNotFoundError(f"Missing GPU checker: {OPTIX_CHECKER}")
    for slot, token in enumerate(tokens):
        print(f"Checking allocated GPU slot {slot} (token={token}) ...", flush=True)
        result = subprocess.run(
            [sys.executable, str(OPTIX_CHECKER)],
            cwd=SCRIPT_DIR,
            env=gpu_environment(token),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n", flush=True)
        if result.returncode != 0:
            raise RuntimeError(f"Allocated GPU slot {slot} failed the OptiX preflight")


def estimate_osm_mesh_complexity(path: Path) -> tuple[int, int]:
    """Estimate the generated mesh from building ways, ignoring large relations."""
    if not path.is_file():
        return 0, 0
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return 0, 0

    building_count = 0
    face_count = 0
    for way in root.findall("way"):
        tags = {
            tag.attrib.get("k", ""): tag.attrib.get("v", "")
            for tag in way.findall("tag")
        }
        if tags.get("building", "no") == "no":
            continue
        refs = [nd.attrib.get("ref") for nd in way.findall("nd") if nd.attrib.get("ref")]
        if len(refs) >= 2 and refs[0] == refs[-1]:
            refs.pop()
        if len(refs) < 3:
            continue
        building_count += 1
        # The generated mesh normally has 2*n wall triangles and n-2 roof
        # triangles for an n-vertex footprint.
        face_count += 3 * len(refs) - 2
    return building_count, face_count


def load_block_tasks() -> list[BlockTask]:
    manifest_path = DATASET_ROOT / "blocks_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing prepared manifest: {manifest_path}")
    tasks: list[BlockTask] = []
    seen_block_ids: set[str] = set()
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            block_id = row["block_id"]
            if block_id in seen_block_ids:
                raise RuntimeError(f"Block ID is not globally unique: {block_id}")
            seen_block_ids.add(block_id)
            osm_path = DATASET_ROOT / row["region_slug"] / block_id / "osm_map.osm"
            if not osm_path.is_file():
                osm_path = RANDOMIZED_OSM_CACHE / row["region_slug"] / block_id / "osm_map.osm"
            estimated_bytes = osm_path.stat().st_size if osm_path.is_file() else 0
            estimated_buildings, estimated_mesh_faces = estimate_osm_mesh_complexity(osm_path)
            tasks.append(BlockTask(
                row["region_slug"], block_id, estimated_buildings,
                estimated_mesh_faces, estimated_bytes,
            ))
    # Actual building geometry, rather than raw XML bytes, is the useful RT
    # complexity proxy. OSM route relations can contain thousands of members
    # while contributing no geometry to the Sionna scene.
    if TASK_ORDER == "simple_first":
        tasks.sort(key=lambda task: (
            task.estimated_mesh_faces, task.estimated_buildings,
            task.estimated_bytes, task.region_slug, task.block_id,
        ))
    elif TASK_ORDER == "complex_first":
        tasks.sort(key=lambda task: (
            -task.estimated_mesh_faces, -task.estimated_buildings,
            -task.estimated_bytes, task.region_slug, task.block_id,
        ))
    elif TASK_ORDER != "manifest":
        raise ValueError(
            f"Unsupported RID_TASK_ORDER={TASK_ORDER!r}; expected "
            "simple_first, complex_first, or manifest"
        )
    return tasks


def load_completed_region_skips(valid_region_slugs: set[str]) -> set[str]:
    """Load externally completed campuses that this cluster must not claim."""
    path = COMPLETED_REGIONS_SKIP_FILE
    if not path.is_file():
        print(f"Completed-region skip file not found; no campuses skipped: {path}")
        return set()

    skipped: set[str] = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        slug = raw_line.partition("#")[0].strip()
        if not slug:
            continue
        if any(character.isspace() for character in slug):
            raise RuntimeError(f"Invalid region slug at {path}:{line_number}: {slug!r}")
        if slug in skipped:
            raise RuntimeError(f"Duplicate region slug at {path}:{line_number}: {slug}")
        skipped.add(slug)

    unknown = skipped - valid_region_slugs
    if unknown:
        raise RuntimeError(
            f"Unknown region slug(s) in {path}: {', '.join(sorted(unknown))}"
        )
    return skipped


def block_is_complete(task: BlockTask) -> bool:
    block_dir = DATASET_ROOT / task.region_slug / task.block_id
    metadata_path = block_dir / "metadata.json"
    index_path = block_dir / "sionna_results_by_rx" / "index.json"
    if not metadata_path.is_file() or not index_path.is_file():
        return False
    try:
        import json

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        index = json.loads(index_path.read_text(encoding="utf-8"))
        physical = metadata.get("physical_signature", {})
        rx_entries = index.get("rx", {})
        tensors_complete = (
            index.get("schema_version") == 2
            and index.get("tensor_schema") == "rss_float32_zyx_v1"
            and isinstance(rx_entries, dict)
            and all(
                str(rx_index) in rx_entries
                and (index_path.parent / f"rx_{rx_index:03d}.npy").is_file()
                for rx_index in range(RX_COUNT)
            )
        )
        return (
            tensors_complete
            and
            metadata.get("status") == "complete"
            and metadata.get("schema_version") == "3.4"
            and metadata.get("results_storage") == "per_rx_npy_tensors_v1"
            and int(metadata.get("ground_rx_count", -1)) == RX_COUNT
            and int(metadata.get("completed_rx_count", -1)) == RX_COUNT
            and metadata.get("tx_voxel_shape_xyz") == [VOXEL_NX, VOXEL_NY, VOXEL_NZ]
            and float(metadata.get("voxel_size_m", -1)) == VOXEL_SIZE_M
            and int(physical.get("samples_per_tx", -1)) == SAMPLES_PER_TX
            and int(physical.get("max_depth", -1)) == MAX_DEPTH
            and int(physical.get("random_seed_global", -1)) == RANDOM_SEED
            and metadata.get("osm_height_variant") == OSM_HEIGHT_VARIANT
        )
    except (OSError, TypeError, ValueError):
        return False


def run_worker_pool_member(
    tasks: list[BlockTask],
    identity: str,
    generator_module: object,
    generator_args: object,
    block_specs: dict[tuple[str, str], object],
) -> bool:
    """Claim blocks while keeping one Python/Sionna process alive per GPU."""
    interrupted = False

    def handle_stop(signum: int, _frame: object) -> None:
        nonlocal interrupted
        if interrupted:
            return
        interrupted = True
        print(f"Received signal {signum}; preserving the active checkpoint...", flush=True)
        # Interrupt the in-process solver. Its existing exception path retains
        # the last fully written batch checkpoint before this reaches us.
        raise KeyboardInterrupt

    previous_sigterm = signal.signal(signal.SIGTERM, handle_stop)
    previous_sigint = signal.signal(signal.SIGINT, handle_stop)
    pending = list(tasks)
    completed_here = 0
    try:
        while pending and not interrupted:
            deferred: list[BlockTask] = []
            claimed_this_round = 0
            for task in pending:
                if interrupted:
                    break
                if block_is_complete(task):
                    continue
                block_dir = DATASET_ROOT / task.region_slug / task.block_id
                lock = acquire_file_lock(block_dir / ".worker.lock", blocking=False)
                if lock is None:
                    deferred.append(task)
                    continue
                try:
                    if block_is_complete(task):
                        continue
                    claimed_this_round += 1
                    print(
                        f"WORKER {identity} CLAIM {task.region_slug}/{task.block_id}",
                        flush=True,
                    )
                    install_block_osm(task)
                    key = (task.region_slug, task.block_id)
                    block_spec = block_specs.get(key)
                    if block_spec is None:
                        raise RuntimeError(
                            f"Prepared manifest has no generator block for {task.region_slug}/{task.block_id}"
                        )
                    generator_module.process_aerial_tx_voxel_block(
                        block_spec, DATASET_ROOT, generator_args
                    )
                    if not block_is_complete(task):
                        raise RuntimeError(
                            f"Generator returned without a complete block: "
                            f"{task.region_slug}/{task.block_id}"
                        )
                    completed_here += 1
                    print(
                        f"WORKER {identity} DONE {task.region_slug}/{task.block_id}; "
                        f"completed_by_this_worker={completed_here}",
                        flush=True,
                    )
                    reclaim_drjit_block_memory(
                        f"{task.region_slug}/{task.block_id} "
                        f"completed_by_this_worker={completed_here}"
                    )
                finally:
                    lock.close()
            pending = deferred
            if pending and claimed_this_round == 0 and not interrupted:
                print(
                    f"WORKER {identity} EXIT: the remaining {len(pending)} blocks are "
                    "owned by other workers; releasing this GPU instead of waiting",
                    flush=True,
                )
                return True
        return not interrupted
    except KeyboardInterrupt:
        return False
    except BaseException:
        print(f"WORKER {identity} FAILED inside the long-lived Sionna process", file=sys.stderr)
        traceback.print_exc()
        return False
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(signal.SIGINT, previous_sigint)


def try_finalize(tasks: list[BlockTask]) -> None:
    """The first worker seeing a complete dataset performs the final audit."""
    if any(not block_is_complete(task) for task in tasks):
        return
    lock = acquire_file_lock(DATASET_ROOT / ".dataset_finalize.lock", blocking=False)
    if lock is None:
        print("Another worker is finalizing the dataset.", flush=True)
        return
    try:
        marker = DATASET_ROOT / f".audit_complete_rx{RX_COUNT}"
        if marker.is_file():
            return
        run_checked([
            sys.executable,
            str(GENERATOR),
            *common_generator_args(),
            "--prepare-only",
            "--log-name", "generation_finalize.log",
        ])
        run_checked([
            sys.executable,
            str(AUDITOR),
            "--dataset-root", str(DATASET_ROOT),
            "--quick",
        ])
        marker.write_text(f"completed by {worker_id()}\n", encoding="utf-8")
        print("Dataset generation completed and quick-audited.", flush=True)
    finally:
        lock.close()


def main() -> int:
    if not os.environ.get("SLURM_JOB_ID") and os.environ.get("RID_ALLOW_DIRECT") != "1":
        raise RuntimeError(
            "Do not run this program on the login node. Submit it with: "
            "sbatch submit_5090.slurm"
        )
    tokens = allocated_gpu_tokens()
    token = tokens[0]
    identity = worker_id()
    # Apply the one-GPU and diagnostics environment before importing the
    # generator. rt_solver reads these switches at import time, and Sionna must
    # see the correct CUDA device before its first import/initialization.
    environment = gpu_environment(token)
    os.environ.update(environment)
    configured_optix = configure_optix_library()

    # Import exactly once. The worker then processes every dynamically claimed
    # block through this module, preserving Dr.Jit/Mitsuba/Sionna in-memory JIT
    # caches across block boundaries.
    import campus_sionna_dataset as generator_module

    generator_cli = [*common_generator_args(), "--worker-mode", "--skip-prefetch"]
    generator_args = generator_module.parse_args(generator_cli)
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    generator_module.enable_persistent_logging(
        DATASET_ROOT,
        [Path(__file__).name, *generator_cli],
        f"generation_worker_{identity}.log",
    )
    print(f"Slurm worker: {identity}")
    print(f"Python: {sys.executable}")
    print(f"Generator: {SCRIPT_DIR}")
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Allocated GPU token: {token}")
    print(f"Task order: {TASK_ORDER}")
    print(f"OptiX library: {configured_optix or '<not found>'}")
    print(f"OptiX cache path: {environment['OPTIX_CACHE_PATH']}")
    print(
        f"Seed={RANDOM_SEED}; RX/block={RX_COUNT}; RX solver group={RX_SOLVER_GROUP_SIZE}; "
        f"voxel={VOXEL_NX}x{VOXEL_NY}x{VOXEL_NZ}@{VOXEL_SIZE_M:g}m; "
        f"TX batch={TX_BATCH_SIZE}; samples/TX={SAMPLES_PER_TX}; depth={MAX_DEPTH}; "
        f"stage timing={ENABLE_STAGE_TIMING}; "
        f"Dr.Jit kernel history={ENABLE_DRJIT_KERNEL_HISTORY}; "
        f"checkpoint interval={CHECKPOINT_INTERVAL_BATCHES} batches; "
        f"GPU memory log={ENABLE_GPU_MEMORY_LOG}; "
        f"block memory reclaim={ENABLE_BLOCK_MEMORY_RECLAIM}; "
        "execution model=one long-lived Sionna process per worker"
    )

    verify_optix_on_gpus([token])
    ensure_dataset_ready()

    all_tasks = load_block_tasks()
    generator_blocks = generator_module.load_blocks_manifest(
        DATASET_ROOT / "blocks_manifest.csv"
    )
    block_specs = {
        (block.region_slug, block.block_id): block for block in generator_blocks
    }
    if len(block_specs) != len(generator_blocks):
        raise RuntimeError("Duplicate region/block keys in the prepared manifest")
    skipped_regions = load_completed_region_skips(
        {task.region_slug for task in all_tasks}
    )
    tasks = [task for task in all_tasks if task.region_slug not in skipped_regions]
    skipped_blocks = len(all_tasks) - len(tasks)
    if skipped_regions:
        print(
            f"External completed-region skip: {len(skipped_regions)} campuses, "
            f"{skipped_blocks} blocks; slugs={','.join(sorted(skipped_regions))}",
            flush=True,
        )
    print(
        f"Worker {identity} joins the dynamic pool for {len(tasks)} resumable blocks "
        f"({len(all_tasks)} manifest blocks total)",
        flush=True,
    )
    if not run_worker_pool_member(
        tasks, identity, generator_module, generator_args, block_specs
    ):
        print(
            "Worker stopped. Its completed blocks and active batch checkpoint are preserved.",
            file=sys.stderr,
        )
        return 1
    # Finalization must still validate the full manifest. Skipped campuses are
    # expected to be copied into DATASET_ROOT before the final audit can pass.
    try_finalize(all_tasks)
    print(f"Worker {identity} finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
