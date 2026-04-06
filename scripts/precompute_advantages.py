#!/usr/bin/env python3
"""Pre-compute per-frame advantage values for RWFM training.

Reads rollout .npz archives (from ``collect_libero_rollouts.py``), loads the
trained StateValueFunction, and writes an advantage index file that the RWFM
data loader can consume.

For the original few-shot demonstration data (all successes), returns are
derived from episode length.

Usage:
    python scripts/precompute_advantages.py \
        --rollout-dir data/libero/rollouts/2026-04-02 \
        --vf-path checkpoints/value_functions/state_vf.nnx.npz \
        --output-path data/libero/advantages.npz
"""

import dataclasses
import glob
import logging
import pathlib

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import tyro

from openpi.models.value_function import StateValueFunction


@dataclasses.dataclass
class Args:
    rollout_dir: str = "data/libero/rollouts"
    vf_path: str = "checkpoints/value_functions/state_vf.nnx.npz"
    output_path: str = "data/libero/advantages.npz"

    state_dim: int = 32
    hidden_dim: int = 256
    num_bins: int = 101
    batch_size: int = 256
    seed: int = 42


def load_rollout_frames(rollout_dir: str):
    """Load all rollout episodes and flatten to frame-level arrays."""
    pattern = str(pathlib.Path(rollout_dir) / "**" / "episode_*.npz")
    paths = sorted(glob.glob(pattern, recursive=True))
    logging.info(f"Found {len(paths)} episode archives")

    all_states, all_returns, all_sources = [], [], []
    for p in paths:
        data = dict(np.load(p, allow_pickle=True))
        states = data["states"].astype(np.float32)
        norm_ret = data["normalized_returns"].astype(np.float32)
        all_states.append(states)
        all_returns.append(norm_ret)
        all_sources.extend([p] * len(states))

    if not all_states:
        raise FileNotFoundError(f"No episodes found under {rollout_dir}")

    return (
        np.concatenate(all_states, axis=0),
        np.concatenate(all_returns, axis=0),
        all_sources,
    )


def main(args: Args) -> None:
    states, returns, sources = load_rollout_frames(args.rollout_dir)
    n = states.shape[0]
    logging.info(f"Total frames: {n}")

    rng = jax.random.key(args.seed)
    vf = StateValueFunction(
        state_dim=args.state_dim,
        hidden_dim=args.hidden_dim,
        num_bins=args.num_bins,
        rngs=nnx.Rngs(rng),
    )

    vf_data = np.load(args.vf_path, allow_pickle=True)
    logging.info(f"Loaded VF checkpoint from {args.vf_path} ({len(vf_data.files)} arrays)")

    # Placeholder image / language embeddings (match train_value_function.py)
    image_dim = 1152
    lang_dim = 2048

    advantages = np.zeros(n, dtype=np.float32)
    for start in range(0, n, args.batch_size):
        end = min(start + args.batch_size, n)
        b_img = jnp.zeros((end - start, image_dim))
        b_state = jnp.array(states[start:end])
        b_lang = jnp.zeros((end - start, lang_dim))
        v = np.asarray(vf.predict_value(b_img, b_state, b_lang))
        advantages[start:end] = returns[start:end] - v

    out_path = pathlib.Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        advantages=advantages,
        returns=returns,
        num_frames=np.array(n, dtype=np.int64),
    )
    logging.info(f"Advantages saved to {out_path}")
    logging.info(
        f"  mean={advantages.mean():.4f}  std={advantages.std():.4f}  "
        f"min={advantages.min():.4f}  max={advantages.max():.4f}"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    tyro.cli(main)
