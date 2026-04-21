#!/usr/bin/env python3
"""Quality-gated top-k filter over rollout ``.npz`` archives.

Given a source directory produced by ``collect_libero_rollouts.py``, rank
episodes by ``normalized_returns.mean()`` (success episodes end with a 0
reward and therefore score higher than failures), keep the top ``keep_ratio``
fraction, and copy them into ``dst_dir`` while preserving the
``<suite>/<task>/episode_*.npz`` layout expected by
``rollout_manifest.build_manifest``.

Optionally, the ``--min-success-floor`` flag guarantees that every
``success=True`` episode is kept regardless of the top-k cut, so that
valuable sparse positive data is never dropped.

Usage::

    python scripts/filter_rollouts.py \
        --src-dir data/libero/rwfm_iters/iter_0/rollouts \
        --dst-dir data/libero/rwfm_iters/iter_0/rollouts_topk \
        --keep-ratio 0.5
"""

from __future__ import annotations

import dataclasses
import logging
import math
import pathlib
import shutil

import numpy as np
import tqdm
import tyro


@dataclasses.dataclass
class Args:
    # Source directory (scanned recursively for ``episode_*.npz``).
    src_dir: str = ""
    # Destination directory. Created if it does not exist. Existing files are
    # kept and overwritten when ``--overwrite`` is set.
    dst_dir: str = ""
    # Fraction of episodes to keep, in ``(0, 1]``. ``1.0`` keeps everything.
    keep_ratio: float = 0.5
    # If True, every ``success=True`` episode is kept regardless of the
    # top-k cut. Failures are still ranked by ``normalized_returns.mean()``.
    min_success_floor: bool = True
    # When True, overwrite existing files at the destination.
    overwrite: bool = False


def _score_episode(path: pathlib.Path) -> tuple[float, bool]:
    """Return ``(normalized_return_mean, success)`` for the episode at ``path``."""
    with np.load(path, allow_pickle=True) as data:
        norm_ret = np.asarray(data["normalized_returns"], dtype=np.float32)
        success = bool(np.asarray(data["success"]).item()) if "success" in data.files else False
    if norm_ret.size == 0:
        return float("-inf"), success
    return float(norm_ret.mean()), success


def _copy_preserve_layout(src: pathlib.Path, dst_root: pathlib.Path, src_root: pathlib.Path, overwrite: bool) -> bool:
    rel = src.relative_to(src_root)
    dst = dst_root / rel
    if dst.exists() and not overwrite:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main(args: Args) -> None:
    if not args.src_dir:
        raise ValueError("--src-dir is required")
    if not args.dst_dir:
        raise ValueError("--dst-dir is required")
    if not (0.0 < args.keep_ratio <= 1.0):
        raise ValueError(f"--keep-ratio must be in (0, 1]; got {args.keep_ratio}")

    src_root = pathlib.Path(args.src_dir).resolve()
    dst_root = pathlib.Path(args.dst_dir).resolve()
    if not src_root.exists():
        raise FileNotFoundError(f"Source directory does not exist: {src_root}")

    all_paths = sorted(src_root.rglob("episode_*.npz"))
    n_total = len(all_paths)
    if n_total == 0:
        raise FileNotFoundError(f"No episode_*.npz archives under {src_root}")

    logging.info("Scanning %d episodes under %s", n_total, src_root)
    scored: list[tuple[pathlib.Path, float, bool]] = []
    for p in tqdm.tqdm(all_paths, desc="scoring"):
        score, success = _score_episode(p)
        scored.append((p, score, success))

    success_paths = [s for s in scored if s[2]]
    failure_paths = [s for s in scored if not s[2]]
    n_success = len(success_paths)
    n_failure = len(failure_paths)

    # Sort each bucket by score descending (ties stable on path).
    success_paths.sort(key=lambda x: (-x[1], str(x[0])))
    failure_paths.sort(key=lambda x: (-x[1], str(x[0])))

    target_keep = max(1, math.ceil(n_total * args.keep_ratio))

    if args.min_success_floor and n_success >= target_keep:
        # Floor dominates: take the top ``target_keep`` successes, no failures.
        kept = success_paths[:target_keep]
    elif args.min_success_floor:
        # Keep every success; fill remaining budget from best failures.
        remaining = max(0, target_keep - n_success)
        kept = list(success_paths) + failure_paths[:remaining]
    else:
        # Pure score-based top-k over the combined pool.
        pooled = sorted(scored, key=lambda x: (-x[1], str(x[0])))
        kept = pooled[:target_keep]

    kept_paths = [p for p, _, _ in kept]
    n_kept = len(kept_paths)
    n_kept_success = sum(1 for p, _, s in kept if s)

    logging.info(
        "Selected %d / %d episodes (keep_ratio=%.3f, min_success_floor=%s): %d success, %d failure",
        n_kept,
        n_total,
        args.keep_ratio,
        args.min_success_floor,
        n_kept_success,
        n_kept - n_kept_success,
    )

    dst_root.mkdir(parents=True, exist_ok=True)
    n_copied = 0
    n_skipped = 0
    for p in tqdm.tqdm(kept_paths, desc="copying"):
        if _copy_preserve_layout(p, dst_root, src_root, overwrite=args.overwrite):
            n_copied += 1
        else:
            n_skipped += 1

    logging.info(
        "Done: copied=%d skipped=%d (dst=%s); input success_rate=%.2f%%  selection success_rate=%.2f%%",
        n_copied,
        n_skipped,
        dst_root,
        100.0 * n_success / max(n_total, 1),
        100.0 * n_kept_success / max(n_kept, 1),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    tyro.cli(main)
