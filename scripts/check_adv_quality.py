#!/usr/bin/env python3
"""Sanity checks for precomputed advantages used by RWFM.

Checks:
1) Advantage distribution statistics.
2) Weight distribution statistics, where w = exp(A / beta).
3) VF quality via corr(V, G), where V = G - A.
4) Bucket monotonicity: higher A buckets should have higher success rates.
5) Alignment checks: frame counts and ordering assumptions.
"""

import dataclasses
import glob
import pathlib

import numpy as np
from scipy.stats import pearsonr, spearmanr
import tyro


@dataclasses.dataclass
class Args:
    rollout_dir: str = "data/libero/rollouts"
    advantages_path: str = "data/libero/advantages.npz"
    beta: float = 1.0
    num_bins: int = 10


def _load_rollout_paths(rollout_dir: str) -> list[str]:
    pattern = str(pathlib.Path(rollout_dir) / "**" / "episode_*.npz")
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        raise FileNotFoundError(f"No rollout episodes found under {rollout_dir}")
    return paths


def _frame_success_labels(paths: list[str]) -> tuple[np.ndarray, int]:
    labels = []
    total_frames = 0
    for p in paths:
        ep = np.load(p, allow_pickle=True)
        t = len(ep["states"])
        total_frames += t
        success = int(bool(ep["success"]))
        labels.extend([success] * t)
    return np.asarray(labels, dtype=np.int32), total_frames


def _print_stats(x: np.ndarray, name: str) -> None:
    print(
        f"[{name}] mean={x.mean():.6f} std={x.std():.6f} "
        f"p95={np.percentile(x, 95):.6f} p99={np.percentile(x, 99):.6f} "
        f"max={x.max():.6f} min={x.min():.6f}"
    )


def _bucket_success(advantages: np.ndarray, success: np.ndarray, num_bins: int) -> None:
    qs = np.quantile(advantages, np.linspace(0.0, 1.0, num_bins + 1))
    print(f"\n[Bucket Check] num_bins={num_bins}")
    prev_rate = -1.0
    monotonic_flags = []
    for i in range(num_bins):
        lo, hi = qs[i], qs[i + 1]
        if i == num_bins - 1:
            mask = (advantages >= lo) & (advantages <= hi)
        else:
            mask = (advantages >= lo) & (advantages < hi)
        n = int(mask.sum())
        rate = float(success[mask].mean()) if n > 0 else float("nan")
        print(f"  bin={i:02d} range=[{lo:.6f}, {hi:.6f}] n={n} success={rate:.6f}")
        if i > 0 and np.isfinite(rate):
            monotonic_flags.append(rate >= prev_rate - 1e-9)
        if np.isfinite(rate):
            prev_rate = rate
    if monotonic_flags:
        mono_ratio = np.mean(monotonic_flags)
        print(f"  monotonic_non_decreasing_ratio={mono_ratio:.3f} (1.0 is best)")


def main(args: Args) -> None:
    paths = _load_rollout_paths(args.rollout_dir)
    success_labels, rollout_frames = _frame_success_labels(paths)

    adv_npz = np.load(args.advantages_path, allow_pickle=True)
    advantages = adv_npz["advantages"].astype(np.float32)
    returns = adv_npz["returns"].astype(np.float32)
    saved_num_frames = int(adv_npz["num_frames"]) if "num_frames" in adv_npz.files else -1

    print("[Alignment Check]")
    print(f"  rollout_frames={rollout_frames}")
    print(f"  len(advantages)={len(advantages)}")
    print(f"  len(returns)={len(returns)}")
    print(f"  saved_num_frames={saved_num_frames}")

    if len(advantages) != rollout_frames or len(returns) != rollout_frames:
        raise ValueError(
            "Length mismatch: rollout frame count does not match advantages/returns length. "
            "Re-run precompute_advantages.py on the exact rollout directory used for training."
        )
    if saved_num_frames > 0 and saved_num_frames != rollout_frames:
        raise ValueError(
            f"num_frames mismatch: advantages file says {saved_num_frames}, but rollout has {rollout_frames}."
        )
    print("  length check PASSED")

    print("\n[Distribution Check]")
    _print_stats(advantages, "A")
    # Clip only for numeric stability in stats print.
    weights = np.exp(np.clip(advantages / args.beta, -20.0, 20.0))
    _print_stats(weights, f"w=exp(A/{args.beta})")

    print("\n[VF Quality Check]")
    values = returns - advantages
    sp = spearmanr(values, returns).correlation
    pr = pearsonr(values, returns)[0]
    print(f"  corr(V,G): Spearman={sp:.6f}, Pearson={pr:.6f}")

    _bucket_success(advantages, success_labels, args.num_bins)

    print("\n[Heuristic Warnings]")
    p99_w = float(np.percentile(weights, 99))
    max_w = float(np.max(weights))
    if p99_w > 10.0:
        print(f"  WARN: weight p99={p99_w:.3f} > 10, weighting may be too sharp.")
    if max_w > 50.0:
        print(f"  WARN: weight max={max_w:.3f} >> 50, consider larger beta or clipping.")
    if not (np.isfinite(sp) and sp > 0):
        print("  WARN: Spearman corr(V,G) is non-positive, VF ranking quality may be poor.")
    if not (np.isfinite(pr) and pr > 0):
        print("  WARN: Pearson corr(V,G) is non-positive, VF calibration may be poor.")


if __name__ == "__main__":
    tyro.cli(main)
