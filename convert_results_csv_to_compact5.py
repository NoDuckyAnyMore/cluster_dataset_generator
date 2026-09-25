#!/usr/bin/env python3
"""Safely replace legacy 27-column result CSVs with compact 5-column CSVs.

Only finalized ``sionna_results.csv`` files are scanned. Each legacy file is
written to a temporary compact file and fully verified for all 6,553,600
unique [RX,Z,Y,X] combinations before the legacy file is replaced. A backup
is retained until the verified compact file has been atomically installed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


DATASET_NAME = "shenzhen_campus_sionna_voxel_256m_128x128x40_rx10_rand10to32"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = SCRIPT_DIR.parent / DATASET_NAME
COMPACT_COLUMNS = (
    "rx_index", "tx_voxel_ix", "tx_voxel_iy", "tx_voxel_iz", "rss_dbm",
)
LEGACY_RSS_COLUMN = "sim_rx_power_dbm"
RX_COUNT, NX, NY, NZ = 10, 128, 128, 40
EXPECTED_ROWS = RX_COUNT * NX * NY * NZ
DEFAULT_WORKERS = 12
MAX_WORKERS = 12
TEMP_SUFFIX = ".compact5.tmp"
BACKUP_SUFFIX = ".legacy27.backup"


def local_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{local_now()}] {message}", flush=True)


def header(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        try:
            return next(csv.reader(handle))
        except StopIteration as exc:
            raise ValueError(f"empty CSV: {path}") from exc


def validate_compact(path: Path) -> int:
    """Perform a full second-pass validation of one compact temporary file."""
    seen = bytearray(EXPECTED_ROWS)
    row_count = 0
    duplicate_count = 0
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            actual_header = tuple(next(reader))
        except StopIteration as exc:
            raise ValueError("compact CSV is empty") from exc
        if actual_header != COMPACT_COLUMNS:
            raise ValueError(f"unexpected compact header: {actual_header}")
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(COMPACT_COLUMNS):
                raise ValueError(f"line {line_number} has {len(row)} rather than 5 fields")
            try:
                rx, ix, iy, iz = (int(row[index]) for index in range(4))
                rss = float(row[4])
            except ValueError as exc:
                raise ValueError(f"invalid numeric value at line {line_number}: {exc}") from exc
            if not (0 <= rx < RX_COUNT and 0 <= ix < NX and 0 <= iy < NY and 0 <= iz < NZ):
                raise ValueError(
                    f"out-of-range tensor index ({rx},{ix},{iy},{iz}) at line {line_number}"
                )
            if not math.isfinite(rss):
                raise ValueError(f"non-finite RSS at line {line_number}")
            flat_index = (((rx * NZ + iz) * NY + iy) * NX + ix)
            if seen[flat_index]:
                duplicate_count += 1
            else:
                seen[flat_index] = 1
            row_count += 1
    unique_count = seen.count(1)
    if row_count != EXPECTED_ROWS or unique_count != EXPECTED_ROWS or duplicate_count:
        raise ValueError(
            f"compact verification failed: rows={row_count:,}/{EXPECTED_ROWS:,}, "
            f"unique={unique_count:,}/{EXPECTED_ROWS:,}, duplicates={duplicate_count:,}"
        )
    return row_count


def update_metadata(block_dir: Path) -> None:
    path = block_dir / "metadata.json"
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["results_csv_schema"] = "compact_rss_5col_v1"
    payload["results_csv_columns"] = list(COMPACT_COLUMNS)
    payload["results_csv_compacted_local"] = local_now()
    temporary = path.with_suffix(".json.compact_tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def convert_one(block_dir_text: str) -> dict[str, object]:
    block_dir = Path(block_dir_text)
    source = block_dir / "sionna_results.csv"
    temporary = source.with_name(source.name + TEMP_SUFFIX)
    backup = source.with_name(source.name + BACKUP_SUFFIX)
    try:
        # Recover the only vulnerable rename window from a previously killed
        # process. The legacy file is never discarded during this recovery.
        if not source.is_file() and backup.is_file():
            os.replace(backup, source)
        if not source.is_file():
            return {"block": block_dir_text, "status": "missing", "error": None}

        source_header = header(source)
        if tuple(source_header) == COMPACT_COLUMNS:
            if backup.is_file():
                validate_compact(source)
                backup.unlink()
            temporary.unlink(missing_ok=True)
            update_metadata(block_dir)
            return {"block": block_dir_text, "status": "already_compact", "error": None}

        columns = {name: index for index, name in enumerate(source_header)}
        required = (*COMPACT_COLUMNS[:4], LEGACY_RSS_COLUMN)
        missing = [name for name in required if name not in columns]
        if missing:
            raise ValueError(f"legacy CSV lacks required columns: {missing}")
        if backup.exists():
            raise RuntimeError(f"refusing to overwrite unexpected backup: {backup}")

        input_size = source.stat().st_size
        row_count = 0
        with source.open("r", newline="", encoding="utf-8-sig") as input_handle, temporary.open(
            "w", newline="", encoding="utf-8-sig"
        ) as output_handle:
            reader = csv.reader(input_handle)
            next(reader)
            writer = csv.writer(output_handle, lineterminator="\n")
            writer.writerow(COMPACT_COLUMNS)
            for line_number, row in enumerate(reader, start=2):
                try:
                    writer.writerow((
                        row[columns["rx_index"]],
                        row[columns["tx_voxel_ix"]],
                        row[columns["tx_voxel_iy"]],
                        row[columns["tx_voxel_iz"]],
                        row[columns[LEGACY_RSS_COLUMN]],
                    ))
                except IndexError as exc:
                    raise ValueError(f"truncated legacy row at line {line_number}") from exc
                row_count += 1
            output_handle.flush()
            os.fsync(output_handle.fileno())
        if row_count != EXPECTED_ROWS:
            raise ValueError(f"legacy row count is {row_count:,}, expected {EXPECTED_ROWS:,}")

        verified_rows = validate_compact(temporary)
        output_size = temporary.stat().st_size

        # The verified new file now exists. Keep the old file as a backup until
        # the new file has been atomically installed at the canonical path.
        os.replace(source, backup)
        try:
            os.replace(temporary, source)
        except BaseException:
            if not source.exists() and backup.exists():
                os.replace(backup, source)
            raise
        # ``source`` is the same verified inode that was just named temporary.
        # Only now is removal of the legacy backup allowed.
        backup.unlink()
        update_metadata(block_dir)
        return {
            "block": block_dir_text,
            "status": "converted",
            "rows": verified_rows,
            "old_bytes": input_size,
            "new_bytes": output_size,
            "error": None,
        }
    except Exception as exc:
        return {
            "block": block_dir_text,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.dataset_root.expanduser().resolve()
    if not root.is_dir():
        print(f"Dataset directory does not exist: {root}", file=sys.stderr)
        return 2
    if not 1 <= args.workers <= MAX_WORKERS:
        print(f"--workers must be between 1 and {MAX_WORKERS}", file=sys.stderr)
        return 2

    block_dirs = {
        path.parent for path in root.glob("*/*/sionna_results.csv")
    } | {
        path.parent for path in root.glob(f"*/*/sionna_results.csv{BACKUP_SUFFIX}")
    }
    jobs = sorted(block_dirs)
    if not jobs:
        log(f"No finalized sionna_results.csv files found under {root}")
        return 0
    workers = min(args.workers, len(jobs))
    log(f"START: files={len(jobs)}, worker_processes={workers}, root={root}")
    converted = already = failed = 0
    old_bytes = new_bytes = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(convert_one, str(block)): block for block in jobs}
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            relative = Path(str(result["block"])).relative_to(root)
            status = str(result["status"])
            if status == "converted":
                converted += 1
                old_bytes += int(result["old_bytes"])
                new_bytes += int(result["new_bytes"])
                log(
                    f"[{completed}/{len(jobs)}] CONVERTED {relative}: "
                    f"{int(result['rows']):,} rows, "
                    f"{int(result['old_bytes']) / 1e9:.3f} -> "
                    f"{int(result['new_bytes']) / 1e9:.3f} GB"
                )
            elif status == "already_compact":
                already += 1
                log(f"[{completed}/{len(jobs)}] SKIP COMPACT {relative}")
            else:
                failed += 1
                log(f"[{completed}/{len(jobs)}] FAILED {relative}: {result['error']}")
    saved = old_bytes - new_bytes
    log(
        f"SUMMARY: converted={converted}, already_compact={already}, failed={failed}, "
        f"space_saved={saved / 1e9:.3f} GB"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
