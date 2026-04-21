"""Convert saved LIBERO frame archives (.npz) into MP4 videos.

Run after evaluation, e.g.:
python examples/libero/postprocess_frame_archives.py
"""

import dataclasses
import logging
import pathlib
from collections import defaultdict

import imageio
import numpy as np
import tqdm
import tyro


@dataclasses.dataclass
class Args:
    frame_archive_path: str = "data/libero/frame_archives"
    video_out_path: str = "data/libero/videos"
    overwrite: bool = False
    default_fps: int = 10


def main(args: Args) -> None:
    archive_dir = pathlib.Path(args.frame_archive_path).resolve()
    video_dir = pathlib.Path(args.video_out_path).resolve()
    video_dir.mkdir(parents=True, exist_ok=True)

    # Recursively find all .npz under archive_dir (supports date/suite/task subdirs from main_local.py)
    archives = sorted(archive_dir.rglob("*.npz"))
    if not archives:
        logging.warning("No frame archives found under %s", archive_dir)
        return

    logging.info("Converting %d archives from %s", len(archives), archive_dir)
    num_success = 0
    num_failure = 0
    suite_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"success": 0, "failure": 0})
    for archive_path in tqdm.tqdm(archives):
        try:
            rel = archive_path.relative_to(archive_dir)
        except ValueError:
            rel = pathlib.Path(archive_path.name)

        # Supports both "date/suite/task/*.npz" and "suite/task/*.npz" layouts.
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

        # Mirror archive directory structure under video_out_path (e.g. 2025-03-19/libero_spatial/task_000_.../rollout_episode000_success.mp4)
        out_path = video_dir / rel.with_suffix(".mp4")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not args.overwrite:
            continue

        with np.load(archive_path) as data:
            # Backward/forward compatibility:
            # - eval archives usually store "frames"
            # - rollout archives from RL pipeline store "images"
            if "frames" in data:
                frames = data["frames"]
            elif "images" in data:
                frames = data["images"]
            else:
                available_keys = ", ".join(sorted(data.files))
                raise KeyError(
                    f"Neither 'frames' nor 'images' found in archive: {archive_path}. "
                    f"Available keys: [{available_keys}]"
                )
            fps = int(data["fps"]) if "fps" in data else args.default_fps

        imageio.mimwrite(out_path, [np.asarray(frame) for frame in frames], fps=fps)

    total = num_success + num_failure
    success_rate = (num_success / total * 100.0) if total > 0 else 0.0
    logging.info("Done. MP4 videos saved to %s", video_dir)
    logging.info(
        "Evaluation summary from archives: success=%d, failure=%d, total=%d, success_rate=%.2f%%",
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
