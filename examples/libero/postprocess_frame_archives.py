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
    archive_dir = pathlib.Path(args.frame_archive_path)
    video_dir = pathlib.Path(args.video_out_path)
    video_dir.mkdir(parents=True, exist_ok=True)

    archives = sorted(archive_dir.glob("*.npz"))
    if not archives:
        logging.warning("No frame archives found in %s", archive_dir)
        return

    logging.info("Converting %d archives from %s", len(archives), archive_dir)
    for archive_path in tqdm.tqdm(archives):
        out_path = video_dir / f"{archive_path.stem}.mp4"
        if out_path.exists() and not args.overwrite:
            continue

        with np.load(archive_path) as data:
            frames = data["frames"]
            fps = int(data["fps"]) if "fps" in data else args.default_fps

        imageio.mimwrite(out_path, [np.asarray(frame) for frame in frames], fps=fps)

    logging.info("Done. MP4 videos saved to %s", video_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)
