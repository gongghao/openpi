#!/usr/bin/env python3
"""Pre-compute per-frame advantage values for RWFM training.

Reads rollout .npz archives (from ``collect_libero_rollouts.py``), loads the
trained StateValueFunction **and** the cached encoder features produced by
``train_value_function.py``, then writes an advantage index file that the
RWFM data loader can consume.

Usage:
    python scripts/precompute_advantages.py \
        --rollout-dir data/libero/rollouts/2026-04-02 \
        --vf-path checkpoints/value_functions/state_vf.nnx.npz \
        --features-path checkpoints/value_functions/encoder_features.npz \
        --output-path data/libero/advantages.npz
"""

import dataclasses
import glob
import logging
import pathlib
import re

import flax.nnx as nnx
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import tyro

from openpi.models.value_function import StateValueFunction


@dataclasses.dataclass
class Args:
    rollout_dir: str = "data/libero/rollouts"
    vf_path: str = "checkpoints/value_functions/state_vf.nnx.npz"
    features_path: str = "checkpoints/value_functions/encoder_features.npz"
    output_path: str = "data/libero/advantages.npz"

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


def _load_vf_params_from_npz(npz_path: str) -> dict:
    """Load flattened slash-key npz params and convert to nested pure dict."""
    data = np.load(npz_path, allow_pickle=True)
    def _clean_part(part: str) -> str:
        part = part.strip()
        m = re.fullmatch(r"\['(.+)'\]", part)
        if m:
            return m.group(1)
        m = re.fullmatch(r'DictKey\(key=[\'"](.+)[\'"]\)', part)
        if m:
            return m.group(1)
        m = re.fullmatch(r"GetAttrKey\(name=[\'\"](.+)[\'\"]\)", part)
        if m:
            return m.group(1)
        m = re.fullmatch(r"SequenceKey\(idx=(\d+)\)", part)
        if m:
            return m.group(1)
        return part

    flat = {tuple(_clean_part(p) for p in k.split("/") if p): data[k] for k in data.files}
    return traverse_util.unflatten_dict(flat)


def load_encoder_features(features_path: str, num_frames: int):
    """Load cached SigLIP / Gemma features produced by train_value_function.py."""
    data = np.load(features_path)
    img = data["image_features"]
    lang = data["lang_features"]
    if len(img) != num_frames:
        raise ValueError(
            f"Feature file has {len(img)} frames but rollout data has {num_frames}. "
            "Re-run train_value_function.py with the same rollout directory to refresh features."
        )
    logging.info(f"Loaded encoder features from {features_path}  image={img.shape}  lang={lang.shape}")
    return img, lang


def main(args: Args) -> None:
    states, returns, sources = load_rollout_frames(args.rollout_dir)
    n = states.shape[0]
    logging.info(f"Total frames: {n}")
    state_dim = int(states.shape[-1])
    logging.info(f"Detected state_dim={state_dim} from rollout data")

    img_feats, lang_feats = load_encoder_features(args.features_path, n)
    image_dim = int(img_feats.shape[-1])
    lang_dim = int(lang_feats.shape[-1])

    rng = jax.random.key(args.seed)
    vf = StateValueFunction(
        image_dim=image_dim,
        state_dim=state_dim,
        lang_dim=lang_dim,
        hidden_dim=args.hidden_dim,
        num_bins=args.num_bins,
        rngs=nnx.Rngs(rng),
    )

    vf_params = _load_vf_params_from_npz(args.vf_path)
    vf_state = nnx.state(vf, nnx.Param)
    vf_state.replace_by_pure_dict(vf_params)
    nnx.update(vf, vf_state)
    logging.info(f"Loaded VF checkpoint from {args.vf_path}")

    advantages = np.zeros(n, dtype=np.float32)
    for start in range(0, n, args.batch_size):
        end = min(start + args.batch_size, n)
        b_img = jnp.array(img_feats[start:end])
        b_state = jnp.array(states[start:end])
        b_lang = jnp.array(lang_feats[start:end])
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
