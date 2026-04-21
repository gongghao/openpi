#!/usr/bin/env python3
"""Pre-compute per-frame advantage values for RWFM training.

Reads rollout .npz archives (from ``collect_libero_rollouts.py``), loads the
trained StateValueFunction **and** the cached encoder features produced by
``train_value_function.py``, then writes an advantage index file that the
RWFM data loader can consume.

Usage (single directory):
    python scripts/precompute_advantages.py \
        --rollout-dir data/libero/rollouts/2026-04-02 \
        --vf-path checkpoints/value_functions/state_vf.nnx.npz \
        --features-path checkpoints/value_functions/encoder_features.npz \
        --output-path data/libero/advantages.npz

Usage (multiple directories — e.g. policy rollouts + demo npz union for B8):
    python scripts/precompute_advantages.py \
        --rollout-dirs data/libero/rollouts/iter0 data/libero/demos \
        --vf-path .../state_vf.nnx.npz \
        --features-path .../encoder_features.npz \
        --output-path .../advantages.npz

The emitted ``advantages.npz`` contains, in addition to per-frame values, the
manifest fields ``manifest_sha`` / ``episode_paths`` / ``ep_lengths`` /
``cum_offsets`` so that ``AdvantageInjectorDataset`` can refuse to run when
the rollout directory has diverged from the one that was scored.
"""

import dataclasses
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
from openpi.training.rollout_manifest import (
    RolloutManifest,
    build_manifest,
    load_manifest_from_npz,
    manifest_npz_fields,
)


@dataclasses.dataclass
class Args:
    # Single rollout directory. Use ``rollout_dirs`` instead when combining
    # multiple sources (e.g. policy rollouts + demo npz for B8).
    rollout_dir: str | None = None
    rollout_dirs: tuple[str, ...] = ()
    vf_path: str = "checkpoints/value_functions/state_vf.nnx.npz"
    features_path: str = "checkpoints/value_functions/encoder_features.npz"
    output_path: str = "data/libero/advantages.npz"

    hidden_dim: int = 256
    num_bins: int = 101
    batch_size: int = 256
    seed: int = 42


def _resolve_rollout_dirs(args: Args) -> tuple[str, ...]:
    dirs: list[str] = []
    if args.rollout_dir:
        dirs.append(args.rollout_dir)
    dirs.extend(d for d in args.rollout_dirs if d)
    if not dirs:
        raise ValueError("Specify at least one of --rollout-dir or --rollout-dirs.")
    return tuple(dirs)


def load_rollout_frames(manifest: RolloutManifest) -> tuple[np.ndarray, np.ndarray]:
    """Load all rollout episodes (in manifest order) and flatten to frame-level arrays."""
    all_states: list[np.ndarray] = []
    all_returns: list[np.ndarray] = []
    for p in manifest.paths:
        data = dict(np.load(p, allow_pickle=True))
        all_states.append(data["states"].astype(np.float32))
        all_returns.append(data["normalized_returns"].astype(np.float32))
    states = np.concatenate(all_states, axis=0)
    returns = np.concatenate(all_returns, axis=0)
    if states.shape[0] != manifest.num_frames:
        raise RuntimeError(
            f"Loaded {states.shape[0]} frames but manifest expects {manifest.num_frames}; "
            "episode files have changed on disk mid-run."
        )
    return states, returns


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


def load_encoder_features(features_path: str, manifest: RolloutManifest) -> tuple[np.ndarray, np.ndarray]:
    """Load cached SigLIP / Gemma features and verify they match the manifest."""
    data = np.load(features_path, allow_pickle=True)
    img = data["image_features"]
    lang = data["lang_features"]
    if len(img) != manifest.num_frames:
        raise ValueError(
            f"Feature file has {len(img)} frames but manifest has {manifest.num_frames}. "
            "Re-run train_value_function.py with the same --rollout-dir(s) to refresh features."
        )
    feat_manifest = load_manifest_from_npz(data)
    if feat_manifest is not None and feat_manifest.manifest_sha != manifest.manifest_sha:
        raise ValueError(
            f"manifest_sha mismatch between feature cache ({feat_manifest.manifest_sha[:12]}) "
            f"and rollout manifest ({manifest.manifest_sha[:12]}). "
            f"Delete {features_path} or re-run train_value_function.py."
        )
    logging.info(f"Loaded encoder features from {features_path}  image={img.shape}  lang={lang.shape}")
    return img, lang


def main(args: Args) -> None:
    rollout_dirs = _resolve_rollout_dirs(args)
    manifest = build_manifest(rollout_dirs)
    logging.info(
        "Built rollout manifest: %d episodes, %d frames, manifest_sha=%s",
        manifest.num_episodes,
        manifest.num_frames,
        manifest.manifest_sha[:12],
    )

    states, returns = load_rollout_frames(manifest)
    n = states.shape[0]
    state_dim = int(states.shape[-1])
    logging.info(f"Total frames: {n}, detected state_dim={state_dim}")

    img_feats, lang_feats = load_encoder_features(args.features_path, manifest)
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
        **manifest_npz_fields(manifest),
    )
    logging.info(f"Advantages saved to {out_path}")
    logging.info(
        f"  mean={advantages.mean():.4f}  std={advantages.std():.4f}  "
        f"min={advantages.min():.4f}  max={advantages.max():.4f}"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    tyro.cli(main)
