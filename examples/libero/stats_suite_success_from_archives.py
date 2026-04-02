"""Scan LIBERO frame archives (.npz) and report success rate per suite only.

Does not read frame data or produce MP4. Run after evaluation, e.g.:
python examples/libero/stats_suite_success_from_archives.py
"""

import dataclasses
import logging
import pathlib
from collections import defaultdict

import tqdm
import tyro


@dataclasses.dataclass
class Args:
    frame_archive_path: str = "data/libero/frame_archives"


def main(args: Args) -> None:
    archive_dir = pathlib.Path(args.frame_archive_path).resolve()

    archives = sorted(archive_dir.rglob("*.npz"))
    if not archives:
        logging.warning("No frame archives found under %s", archive_dir)
        return

    logging.info("Scanning %d archives under %s", len(archives), archive_dir)
    num_success = 0
    num_failure = 0
    suite_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"success": 0, "failure": 0})

    for archive_path in tqdm.tqdm(archives):
        try:
            rel = archive_path.relative_to(archive_dir)
        except ValueError:
            rel = pathlib.Path(archive_path.name)

        if len(rel.parts) >= 2 and rel.parts[1].startswith("libero_"):
            task_suite = rel.parts[1]
        elif len(rel.parts) >= 1 and rel.parts[0].startswith("libero_"):
            task_suite = rel.parts[0]
        else:
            task_suite = "unknown_suite"

        if archive_path.stem.endswith("_success"):
            num_success += 1
            suite_stats[task_suite]["success"] += 1
        elif archive_path.stem.endswith("_failure"):
            num_failure += 1
            suite_stats[task_suite]["failure"] += 1

    total = num_success + num_failure
    success_rate = (num_success / total * 100.0) if total > 0 else 0.0
    logging.info(
        "Overall: success=%d, failure=%d, total=%d, success_rate=%.2f%%",
        num_success,
        num_failure,
        total,
        success_rate,
    )
    logging.info("Success rate by task suite:")
    for suite_name in sorted(suite_stats):
        suite_success = suite_stats[suite_name]["success"]
        suite_failure = suite_stats[suite_name]["failure"]
        suite_total = suite_success + suite_failure
        suite_rate = (suite_success / suite_total * 100.0) if suite_total > 0 else 0.0
        logging.info(
            "  %s: success=%d, failure=%d, total=%d, success_rate=%.2f%%",
            suite_name,
            suite_success,
            suite_failure,
            suite_total,
            suite_rate,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)
