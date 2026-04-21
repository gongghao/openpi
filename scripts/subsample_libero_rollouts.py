#!/usr/bin/env python3
"""Subsample LIBERO rollout .npz archives: keep N episodes per task from a larger pool.

Mirrors the directory layout produced by `collect_libero_rollouts.py`:
  input_dir / <suite> / task_XXX_<desc> / episode_YYY_success|failure.npz

Example (50 episodes per task → 10 per task, deterministic first-10):
    python scripts/subsample_libero_rollouts.py \\
        --input-dir data/libero/rollouts/2026-04-07 \\
        --output-dir data/libero/rollouts/2026-04-07_k10 \\
        --episodes-per-task 10 \\
        --strategy first

Random 10 with fixed seed:
    python scripts/subsample_libero_rollouts.py \\
        --input-dir ... --output-dir ... --episodes-per-task 10 --strategy random --seed 42
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import random
import re
import shutil

import tyro

_EPISODE_RE = re.compile(r"^episode_(\d+)_(success|failure)\.npz$")


def _episode_index(name: str) -> int | None:
    m = _EPISODE_RE.match(name)
    return int(m.group(1)) if m else None


def _episode_outcome(name: str) -> str | None:
    m = _EPISODE_RE.match(name)
    return m.group(2) if m else None


def _collect_episode_paths(task_dir: pathlib.Path) -> dict[int, pathlib.Path]:
    """One path per episode index (success/failure are mutually exclusive per index)."""
    by_idx: dict[int, pathlib.Path] = {}
    for p in task_dir.iterdir():
        if not p.is_file() or p.suffix != ".npz":
            continue
        idx = _episode_index(p.name)
        if idx is None:
            continue
        if idx in by_idx:
            logging.warning("Duplicate episode index %s in %s — keeping %s", idx, task_dir, by_idx[idx])
            continue
        by_idx[idx] = p
    return by_idx


def _select_indices(available: list[int], k: int, strategy: str, rng: random.Random) -> list[int]:
    available = sorted(available)
    if len(available) <= k:
        return available
    if strategy == "first":
        return available[:k]
    if strategy == "random":
        return sorted(rng.sample(available, k))
    raise ValueError(f"Unknown strategy: {strategy}")


def _select_with_success_ratio(
    by_idx: dict[int, pathlib.Path],
    k: int,
    strategy: str,
    rng: random.Random,
    success_ratio: float,
) -> list[int]:
    """Select k indices with target success ratio (best effort)."""
    all_indices = sorted(by_idx.keys())
    if len(all_indices) <= k:
        return all_indices

    success = [i for i in all_indices if _episode_outcome(by_idx[i].name) == "success"]
    failure = [i for i in all_indices if _episode_outcome(by_idx[i].name) == "failure"]

    target_success = int(round(k * success_ratio))
    target_success = max(0, min(k, target_success))
    target_failure = k - target_success

    sel_success = _select_indices(success, min(target_success, len(success)), strategy, rng)
    sel_failure = _select_indices(failure, min(target_failure, len(failure)), strategy, rng)
    selected = sel_success + sel_failure

    # Backfill when one class is insufficient.
    if len(selected) < k:
        remaining_pool = [i for i in all_indices if i not in set(selected)]
        selected += _select_indices(remaining_pool, k - len(selected), strategy, rng)

    return sorted(selected)


@dataclasses.dataclass
class Args:
    """Subsample rollouts to fewer episodes per task directory."""

    input_dir: str = "/seu_share/home/linli/213221101/MyDatasets/libero_fewshot/rollouts/2026-04-16"
    output_dir: str = "/seu_share/home/linli/213221101/MyDatasets/libero_fewshot/rollouts/2026-04-16_k10"
    episodes_per_task: int = 10
    """Target number of episode files to keep per `task_*` folder."""

    strategy: str = "first"
    """`first`: lowest episode indices (e.g. 0..9 when 50 exist). `random`: sample without replacement."""

    seed: int = 0
    """Used when strategy is `random`."""

    success_ratio: float = 0.7
    """Target fraction of success trajectories per task (best effort when data is imbalanced)."""

    renumber: bool = False
    """If true, rename copied files to episode_000_*.npz ... episode_{K-1}_*.npz in order of selected indices."""

    dry_run: bool = False
    """Print actions without copying files."""


def main(args: Args) -> None:
    inp = pathlib.Path(args.input_dir).resolve()
    out = pathlib.Path(args.output_dir).resolve()
    if not inp.is_dir():
        raise FileNotFoundError(f"input_dir is not a directory: {inp}")
    if not (0.0 <= args.success_ratio <= 1.0):
        raise ValueError(f"success_ratio must be in [0,1], got {args.success_ratio}")

    rng = random.Random(args.seed)
    copied_files = 0
    skipped_tasks = 0

    # Walk: input_dir/<suite_name>/task_*/
    for suite_dir in sorted(inp.iterdir()):
        if not suite_dir.is_dir():
            continue
        rel_suite = suite_dir.relative_to(inp)
        for task_dir in sorted(suite_dir.glob("task_*")):
            if not task_dir.is_dir():
                continue
            by_idx = _collect_episode_paths(task_dir)
            if not by_idx:
                logging.warning("No episode_*.npz under %s, skipping", task_dir)
                skipped_tasks += 1
                continue

            selected = _select_with_success_ratio(
                by_idx,
                args.episodes_per_task,
                args.strategy,
                rng,
                args.success_ratio,
            )
            out_task = out / rel_suite / task_dir.name
            if not args.dry_run:
                out_task.mkdir(parents=True, exist_ok=True)

            succ_n = sum(1 for i in selected if _episode_outcome(by_idx[i].name) == "success")
            fail_n = len(selected) - succ_n
            logging.info(
                "task=%s selected=%d success=%d failure=%d target_success_ratio=%.2f",
                task_dir.name,
                len(selected),
                succ_n,
                fail_n,
                args.success_ratio,
            )

            for new_ord, ep_idx in enumerate(selected):
                src = by_idx[ep_idx]
                if args.renumber:
                    suffix = _EPISODE_RE.match(src.name)
                    assert suffix is not None
                    dst_name = f"episode_{new_ord:03d}_{suffix.group(2)}.npz"
                else:
                    dst_name = src.name
                dst = out_task / dst_name
                logging.info("%s -> %s", src.relative_to(inp), dst.relative_to(out) if not args.dry_run else dst)
                if not args.dry_run:
                    shutil.copy2(src, dst)
                copied_files += 1

    logging.info(
        "Done: %s %s episode files -> %s (skipped empty tasks=%s)",
        "would copy" if args.dry_run else "copied",
        copied_files,
        out,
        skipped_tasks,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    tyro.cli(main)
