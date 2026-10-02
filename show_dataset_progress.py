"""Report global progress without importing Sionna or occupying a GPU."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import run_cluster_5090 as cluster


def load_tasks(dataset_root: Path) -> list[cluster.BlockTask]:
    manifest = dataset_root / "blocks_manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest}")
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            cluster.BlockTask(row["region_slug"], row["block_id"], 0, 0, 0)
            for row in csv.DictReader(handle)
        ]


def load_skipped_regions(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    return {
        line.partition("#")[0].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.partition("#")[0].strip()
    }


def has_checkpoint(dataset_root: Path, task: cluster.BlockTask) -> bool:
    work = dataset_root / task.region_slug / task.block_id / "sionna_results_by_rx" / ".work"
    return work.is_dir() and any(work.glob("*.checkpoint.json"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=cluster.DATASET_ROOT)
    parser.add_argument("--skip-file", type=Path, default=cluster.COMPLETED_REGIONS_SKIP_FILE)
    parser.add_argument("--by-region", action="store_true", help="Also print per-campus progress.")
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    # Keep compatibility with an already-running launcher from an earlier
    # upload: block_is_complete() reads this module-level path.
    cluster.DATASET_ROOT = dataset_root
    tasks = load_tasks(dataset_root)
    skipped_regions = load_skipped_regions(
        args.skip_file.expanduser().resolve() if args.skip_file else None
    )
    skipped_tasks = [task for task in tasks if task.region_slug in skipped_regions]
    cluster_tasks = [task for task in tasks if task.region_slug not in skipped_regions]

    completed: dict[tuple[str, str], bool] = {
        (task.region_slug, task.block_id): cluster.block_is_complete(task)
        for task in cluster_tasks
    }
    cluster_done = sum(completed.values())
    cluster_remaining = len(cluster_tasks) - cluster_done
    resumable = sum(
        not completed[(task.region_slug, task.block_id)] and has_checkpoint(dataset_root, task)
        for task in cluster_tasks
    )
    task_done = len(skipped_tasks) + cluster_done
    percent = 100.0 * task_done / len(tasks) if tasks else 100.0

    print(f"Dataset              : {dataset_root}")
    print(f"Manifest blocks      : {len(tasks)}")
    print(f"External skip blocks : {len(skipped_tasks)} ({len(skipped_regions)} campuses)")
    print(f"Cluster complete     : {cluster_done}")
    print(f"Cluster remaining    : {cluster_remaining}")
    print(f"Resumable checkpoints: {resumable}")
    print(f"Task progress        : {task_done}/{len(tasks)} ({percent:.2f}%)")

    if args.by_region:
        totals = Counter(task.region_slug for task in tasks)
        done = Counter(
            task.region_slug
            for task in cluster_tasks
            if completed[(task.region_slug, task.block_id)]
        )
        print("\nPer-campus progress:")
        for region in sorted(totals):
            if region in skipped_regions:
                status = "external-skip"
            else:
                status = f"{done[region]}/{totals[region]}"
            print(f"  {region:<24} {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
