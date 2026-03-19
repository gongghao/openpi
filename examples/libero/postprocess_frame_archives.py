"""Convert saved LIBERO frame archives (.npz) into MP4 videos.

Run after evaluation, e.g.:
python examples/libero/postprocess_frame_archives.py
"""

import dataclasses
import logging
import pathlib

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
    for archive_path in tqdm.tqdm(archives):
        if archive_path.stem.endswith("_success"):
            num_success += 1
        elif archive_path.stem.endswith("_failure"):
            num_failure += 1

        # Mirror archive directory structure under video_out_path (e.g. 2025-03-19/libero_spatial/task_000_.../rollout_episode000_success.mp4)
        try:
            rel = archive_path.relative_to(archive_dir)
        except ValueError:
            rel = archive_path.name
        out_path = video_dir / rel.with_suffix(".mp4")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not args.overwrite:
            continue

        with np.load(archive_path) as data:
            frames = data["frames"]
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)
