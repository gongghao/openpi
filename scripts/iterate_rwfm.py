#!/usr/bin/env python3
"""B5 / B6 orchestrator: iterated offline RL with RWFM.

Runs the four RWFM phases in a loop so every iteration re-uses the previous
iteration's policy as a better behaviour policy::

    collect_libero_rollouts  ->  (replay_pool sync)
                             ->  train_value_function
                             ->  precompute_advantages
                             ->  train.py

Per-iteration layout under ``--base-dir``::

    iter_{k}/
      rollouts/           # policy rollouts collected with iter_{k-1} ckpt (or SFT for k=0)
      vf/                 # state_vf / encoder_features / metrics
      advantages.npz      # manifest-aligned per-frame advantages
      rwfm_ckpt/<config_name>/iter_{k}/...  # RWFM-finetuned policy

Default weight-loading mode is ``warm-weights`` (the plan's B5 pick): each
iteration loads the previous iter's RWFM weights while resetting the
optimizer / EMA / step counter.

Usage::

    python scripts/iterate_rwfm.py \
        --base-dir data/libero/rwfm_iters \
        --sft-checkpoint-dir /path/to/sft/4999/params \
        --num-iters 3

With a shared replay pool (B6)::

    python scripts/iterate_rwfm.py \
        --base-dir data/libero/rwfm_iters \
        --sft-checkpoint-dir /path/to/sft/4999/params \
        --replay-pool-dir data/libero/replay_pool \
        --replay-pool-size 4000 \
        --replay-pool-mode fifo \
        --num-iters 4

Replay-pool contract (B6):
    * After ``collect_libero_rollouts`` writes to ``iter_{k}/rollouts``, the
      orchestrator copies each ``episode_*.npz`` (preserving
      ``<suite>/<task>/`` subpaths) into ``--replay-pool-dir`` and then
      trims the pool to ``--replay-pool-size`` using ``fifo`` (oldest
      mtime first) or ``reservoir`` (deterministic uniform subsample).
    * When the pool is active it REPLACES ``iter_{k}/rollouts`` as the
      primary rollout directory fed into VF / advantage / training
      stages, so the manifest is a pure function of current pool contents.
    * Manually deleting files from the pool between iterations is safe:
      ``rollout_manifest`` recomputes ``manifest_sha`` from scratch, and
      both ``train_value_function.py`` and ``precompute_advantages.py``
      refuse to reuse any cached ``encoder_features.npz`` /
      ``advantages.npz`` whose stored ``manifest_sha`` disagrees. The
      VF feature extraction and advantage precomputation both run from
      scratch in that case.

The training stage is launched via ``python -c`` so each iteration runs in a
fresh Python process. This keeps JAX compilation caches and GPU allocations
isolated between stages.
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import random
import shlex
import shutil
import subprocess
import sys
import textwrap
from typing import Literal

import tyro

Stage = Literal["rollouts", "vf", "advantages", "train"]


@dataclasses.dataclass
class Args:
    # Output root. Per-iteration data lives under ``{base_dir}/iter_{k}``.
    base_dir: str = "data/libero/rwfm_iters"
    # Name of the RWFM TrainConfig (from ``openpi.training.config._CONFIGS``).
    train_config_name: str = "pi0_libero_rwfm"

    # Iteration bounds. ``start_iter`` is useful for resuming / debugging one
    # iteration at a time.
    start_iter: int = 0
    num_iters: int = 3

    # SFT checkpoint. Iteration 0 loads policy weights from here and
    # ``train_value_function.py`` always uses it for frozen encoder features.
    sft_config: str = "pi0_libero_fewshot"
    sft_checkpoint_dir: str = ""
    # ``params`` directory under the SFT checkpoint. When empty we append
    # ``params`` to ``sft_checkpoint_dir``. Used by ``collect_libero_rollouts``.
    sft_params_path: str = ""

    # Rollout collection
    task_suite_names: tuple[str, ...] = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    num_trials_per_task: int = 20

    # Value function training
    vf_num_steps: int = 5000
    vf_batch_size: int = 64
    vf_feature_batch_size: int = 64
    # When True, pass ``--resume`` to ``train_value_function.py`` so that it
    # restores params / optimiser state / RNG from
    # ``{vf_output_dir}/{state,action_cond}_vf_resume.msgpack`` (if present)
    # and continues training up to ``vf_num_steps`` instead of re-initialising
    # from scratch. Note: by default each iteration uses its own
    # ``iter_{k}/vf`` directory, so to actually warm-start the VF across
    # rounds either set ``vf_output_subdir`` below or copy the previous iter's
    # ``*_resume.msgpack`` into the new directory before this stage.
    vf_resume: bool = False
    # Optional checkpoint interval (in optimiser steps) for the VF trainer's
    # own resume file. ``0`` means only save once at the very end.
    vf_save_interval: int = 0
    # If set, VF artefacts for every iteration are written into
    # ``<iter_root_base>/<vf_output_subdir>`` **shared across iterations**
    # (rather than ``iter_{k}/vf``). Combined with ``vf_resume=True`` this
    # gives an across-iteration warm-start for the VF. Must not be empty if
    # enabled.
    vf_output_subdir: str = ""

    # Additional rollout directories unioned with each iteration's policy
    # rollouts (e.g. exported demo npz for B8 Scheme B, or a replay-pool
    # directory for B6). The manifest concatenation order is
    # ``iter_{k}/rollouts`` first, then these entries.
    extra_rollout_dirs: tuple[str, ...] = ()

    # B6: replay pool. When ``replay_pool_dir`` is set, each iteration's
    # rollouts are copied into this shared directory and a FIFO / reservoir
    # trim is applied; the pool (not the per-iter rollouts dir) becomes the
    # primary rollout source for VF / advantage / training stages. Deleting
    # files from the pool is safe: ``train_value_function`` and
    # ``precompute_advantages`` re-extract from scratch whenever the
    # manifest SHA changes (see ``rollout_manifest.py``).
    replay_pool_dir: str = ""
    replay_pool_size: int = 0  # 0 disables trimming (keeps everything).
    replay_pool_mode: Literal["fifo", "reservoir"] = "fifo"

    # RWFM training
    num_train_steps: int = 30_000
    batch_size: int = 32

    # Weight loading strategy between iterations (see plan §Phase 3 / B5):
    #   ``warm-weights`` -- iter_k loads iter_{k-1} RWFM params; optimizer / EMA / step reset.
    #   ``cold``         -- iter_k reloads SFT weights; ignores prior iterations.
    weight_mode: Literal["warm-weights", "cold"] = "warm-weights"

    # Stages to actually execute. Skip any for partial reruns.
    stages: tuple[Stage, ...] = ("rollouts", "vf", "advantages", "train")

    # Print commands without executing them.
    dry_run: bool = False

    # Python interpreter used for subprocess stages (defaults to the current one).
    python: str = sys.executable

    # Optional override for the rwfm config's batch size used by precompute.
    precompute_batch_size: int = 256


def _run(cmd: list[str], *, dry_run: bool, cwd: str | None = None) -> None:
    pretty = " ".join(shlex.quote(c) for c in cmd)
    logging.info(">>> %s", pretty)
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=cwd)


def _iter_dir(base: pathlib.Path, k: int) -> pathlib.Path:
    return base / f"iter_{k}"


def _rollout_dirs(
    iter_root: pathlib.Path,
    extra: tuple[str, ...],
    *,
    replay_pool_dir: str = "",
) -> list[str]:
    """Manifest-order concatenation of directories consumed by VF / advantages / train.

    When ``replay_pool_dir`` is set (B6 enabled), the pool REPLACES the
    per-iter ``rollouts`` dir as the primary source: this guarantees the
    manifest SHA is a deterministic function of the current pool contents,
    so any files added / removed between stages are detected by
    ``train_value_function`` and ``precompute_advantages``.
    """
    primary = replay_pool_dir if replay_pool_dir else str(iter_root / "rollouts")
    dirs = [primary]
    dirs.extend(d for d in extra if d)
    return dirs


def _update_replay_pool(
    *,
    source_dir: pathlib.Path,
    pool_dir: pathlib.Path,
    max_size: int,
    mode: Literal["fifo", "reservoir"],
    dry_run: bool,
    rng_seed: int,
) -> tuple[int, int, int]:
    """B6: copy new episodes into the pool and apply a FIFO / reservoir trim.

    Returns ``(added, removed, final_size)`` counts.

    The pool preserves the full ``suite/task_XXX_.../episode_*.npz`` subpath
    so the resulting manifest matches the format built by
    :func:`openpi.training.rollout_manifest.build_manifest`.

    Deleting files from ``pool_dir`` at any time is safe: the next
    ``train_value_function`` / ``precompute_advantages`` invocation will
    detect a ``manifest_sha`` mismatch against the cached
    ``encoder_features.npz`` / ``advantages.npz`` and re-extract from
    scratch (enforced in ``rollout_manifest.load_manifest_from_npz``).
    """
    if dry_run:
        logging.info("[dry-run] would sync %s -> %s (mode=%s size=%d)", source_dir, pool_dir, mode, max_size)
        return 0, 0, 0

    pool_dir.mkdir(parents=True, exist_ok=True)
    added = 0
    if source_dir.exists():
        for src_path in source_dir.rglob("episode_*.npz"):
            rel = src_path.relative_to(source_dir)
            dst = pool_dir / rel
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst)
            added += 1

    files = sorted(pool_dir.rglob("episode_*.npz"), key=lambda p: p.stat().st_mtime)
    removed = 0
    if max_size > 0 and len(files) > max_size:
        if mode == "fifo":
            to_remove = files[: len(files) - max_size]
        elif mode == "reservoir":
            rnd = random.Random(rng_seed)
            keep = set(rnd.sample(files, max_size))
            to_remove = [f for f in files if f not in keep]
        else:
            raise ValueError(f"Unknown replay_pool_mode: {mode!r}")
        for f in to_remove:
            try:
                f.unlink()
                removed += 1
            except OSError as e:
                logging.warning("Failed to evict %s: %s", f, e)

    final_files = list(pool_dir.rglob("episode_*.npz"))
    logging.info(
        "Replay pool '%s' updated: +%d -%d = %d episodes (mode=%s, cap=%s)",
        pool_dir,
        added,
        removed,
        len(final_files),
        mode,
        max_size or "inf",
    )
    return added, removed, len(final_files)


def _sft_params_path(args: Args) -> str:
    if args.sft_params_path:
        return args.sft_params_path
    if not args.sft_checkpoint_dir:
        raise ValueError("Set --sft-checkpoint-dir (and optionally --sft-params-path).")
    return str(pathlib.Path(args.sft_checkpoint_dir) / "params")


def _policy_weight_path(args: Args, iter_k: int, iter_root_prev: pathlib.Path | None) -> str:
    """Weights used to seed this iteration's rollouts + RWFM training."""
    if args.weight_mode == "cold" or iter_k == 0 or iter_root_prev is None:
        return _sft_params_path(args)
    # warm-weights: params directory of the previous iter's RWFM checkpoint.
    # train.py writes ``{ckpt_base}/{config_name}/{exp_name}/{step}/params``.
    step_dir = iter_root_prev / "rwfm_ckpt" / args.train_config_name / f"iter_{iter_k - 1}" / str(
        args.num_train_steps - 1
    )
    return str(step_dir / "params")


def _rollout_cli_args(rollout_dirs: list[str]) -> list[str]:
    """Split rollout_dirs into the ``--rollout-dir`` + ``--rollout-dirs`` pair.

    ``_resolve_rollout_dirs`` in both downstream scripts concatenates the two
    in order, so we preserve the manifest-canonical ordering by putting the
    first entry in ``--rollout-dir`` and the rest in ``--rollout-dirs``.
    """
    if not rollout_dirs:
        raise ValueError("rollout_dirs must contain at least one directory.")
    out = ["--rollout-dir", rollout_dirs[0]]
    if len(rollout_dirs) > 1:
        out += ["--rollout-dirs", *rollout_dirs[1:]]
    return out


def _stage_rollouts(args: Args, iter_root: pathlib.Path, policy_params_path: str) -> None:
    """Collect fresh policy rollouts for this iteration.

    ``collect_libero_rollouts.py`` writes under
    ``{output_dir}/<archive_date>/<suite>/task_*/episode_*.npz``; we fix the
    archive date to ``rollouts`` so the per-iteration manifest directory is
    always ``iter_root/rollouts``.
    """
    cmd = [
        args.python,
        "scripts/collect_libero_rollouts.py",
        "--policy.config",
        args.train_config_name,
        "--policy.dir",
        str(pathlib.Path(policy_params_path).parent),
        "--num-trials-per-task",
        str(args.num_trials_per_task),
        "--output-dir",
        str(iter_root),
        "--archive-date",
        "rollouts",
        "--task-suite-names",
        *args.task_suite_names,
    ]
    _run(cmd, dry_run=args.dry_run)


def _vf_dir(args: Args, iter_root: pathlib.Path) -> pathlib.Path:
    """Resolve the VF output directory.

    When ``--vf-output-subdir`` is set, a single directory is shared across
    all iterations (placed next to the ``iter_*`` folders). Combined with
    ``--vf-resume`` this gives a true across-iteration warm-start for the
    value function. Otherwise every iteration keeps its own ``iter_k/vf``.
    """
    if args.vf_output_subdir:
        return iter_root.parent / args.vf_output_subdir
    return iter_root / "vf"


def _stage_vf(args: Args, iter_root: pathlib.Path, rollout_dirs: list[str]) -> None:
    vf_dir = _vf_dir(args, iter_root)
    cmd = [
        args.python,
        "scripts/train_value_function.py",
        "--sft-config",
        args.sft_config,
        "--sft-checkpoint-dir",
        args.sft_checkpoint_dir,
        "--output-dir",
        str(vf_dir),
        "--num-steps",
        str(args.vf_num_steps),
        "--batch-size",
        str(args.vf_batch_size),
        "--feature-batch-size",
        str(args.vf_feature_batch_size),
        *_rollout_cli_args(rollout_dirs),
    ]
    if args.vf_resume:
        cmd.append("--resume")
    if args.vf_save_interval > 0:
        cmd.extend(["--save-interval", str(args.vf_save_interval)])
    _run(cmd, dry_run=args.dry_run)


def _stage_advantages(args: Args, iter_root: pathlib.Path, rollout_dirs: list[str]) -> None:
    vf_dir = _vf_dir(args, iter_root)
    cmd = [
        args.python,
        "scripts/precompute_advantages.py",
        "--vf-path",
        str(vf_dir / "state_vf.nnx.npz"),
        "--features-path",
        str(vf_dir / "encoder_features.npz"),
        "--output-path",
        str(iter_root / "advantages.npz"),
        "--batch-size",
        str(args.precompute_batch_size),
        *_rollout_cli_args(rollout_dirs),
    ]
    _run(cmd, dry_run=args.dry_run)


def _build_train_inline_script(
    args: Args,
    iter_k: int,
    iter_root: pathlib.Path,
    rollout_dir: str,
    extra_rollout_dirs: tuple[str, ...],
    weight_params_path: str,
    adv_path: str,
) -> str:
    """Inline Python that builds a ``TrainConfig`` and calls ``train.main``.

    We go through ``dataclasses.replace`` rather than tyro overrides so that
    nested fields (``data.rollout_dir``, ``weight_loader.params_path``) are
    changed by value regardless of which tyro version the user has installed.
    """
    config_name = args.train_config_name
    exp_name = f"iter_{iter_k}"
    ckpt_base = str(iter_root / "rwfm_ckpt")
    return textwrap.dedent(
        f"""
        import dataclasses
        import pathlib
        import runpy

        from openpi.training import config as _config
        from openpi.training import weight_loaders

        base = _config.get_config({config_name!r})
        data = dataclasses.replace(
            base.data,
            rollout_dir={rollout_dir!r},
            rollout_dirs={tuple(extra_rollout_dirs)!r},
        )
        weight_loader = weight_loaders.CheckpointWeightLoader(params_path={weight_params_path!r})
        cfg = dataclasses.replace(
            base,
            data=data,
            weight_loader=weight_loader,
            exp_name={exp_name!r},
            overwrite=True,
            resume=False,
            num_train_steps={int(args.num_train_steps)},
            batch_size={int(args.batch_size)},
            rwfm_advantages_path={adv_path!r},
            checkpoint_base_dir={ckpt_base!r},
        )

        ns = runpy.run_path(str(pathlib.Path("scripts/train.py")), run_name="__iter_rwfm__")
        ns["main"](cfg)
        """
    ).strip()


def _stage_train(
    args: Args,
    iter_k: int,
    iter_root: pathlib.Path,
    rollout_dirs: list[str],
    weight_params_path: str,
) -> None:
    adv_path = str(iter_root / "advantages.npz")
    code = _build_train_inline_script(
        args,
        iter_k,
        iter_root,
        rollout_dir=rollout_dirs[0],
        extra_rollout_dirs=tuple(rollout_dirs[1:]),
        weight_params_path=weight_params_path,
        adv_path=adv_path,
    )
    cmd = [args.python, "-c", code]
    pretty_args = {
        "rollout_dirs": rollout_dirs,
        "weight_params_path": weight_params_path,
        "advantages_path": adv_path,
        "exp_name": f"iter_{iter_k}",
        "checkpoint_base": str(iter_root / "rwfm_ckpt"),
    }
    logging.info(">>> train stage (inline): %s", pretty_args)
    if args.dry_run:
        return
    subprocess.run(cmd, check=True)


def main(args: Args) -> None:
    base_dir = pathlib.Path(args.base_dir).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    for k in range(args.start_iter, args.start_iter + args.num_iters):
        iter_root = _iter_dir(base_dir, k)
        iter_root.mkdir(parents=True, exist_ok=True)
        iter_root_prev = _iter_dir(base_dir, k - 1) if k > 0 else None
        policy_weights = _policy_weight_path(args, k, iter_root_prev)
        logging.info(
            "=== iter %d | iter_root=%s | policy_weights=%s ===", k, iter_root, policy_weights
        )

        if "rollouts" in args.stages:
            _stage_rollouts(args, iter_root, policy_weights)

        # B6: sync freshly collected rollouts into the shared replay pool
        # *before* downstream stages run, so the manifest they see already
        # reflects the trimmed pool.
        if args.replay_pool_dir:
            pool_dir = pathlib.Path(args.replay_pool_dir).resolve()
            _update_replay_pool(
                source_dir=iter_root / "rollouts",
                pool_dir=pool_dir,
                max_size=args.replay_pool_size,
                mode=args.replay_pool_mode,
                dry_run=args.dry_run,
                rng_seed=1_000_000 + k,
            )

        rollout_dirs = _rollout_dirs(
            iter_root,
            args.extra_rollout_dirs,
            replay_pool_dir=args.replay_pool_dir,
        )

        if "vf" in args.stages:
            _stage_vf(args, iter_root, rollout_dirs)
        if "advantages" in args.stages:
            _stage_advantages(args, iter_root, rollout_dirs)
        if "train" in args.stages:
            _stage_train(args, k, iter_root, rollout_dirs, policy_weights)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    tyro.cli(main)
