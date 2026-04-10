#!/usr/bin/env python3
"""Train StateValueFunction and ActionConditionedValueFunction on rollout data.

Reads the .npz episode archives produced by ``collect_libero_rollouts.py``,
extracts frozen SigLIP / Gemma features from the SFT checkpoint, and trains
both value function variants.

Usage:
    python scripts/train_value_function.py \
        --rollout-dir data/libero/rollouts/2026-04-02 \
        --sft-config pi0_libero_fewshot \
        --sft-checkpoint-dir /path/to/sft_checkpoint/4999 \
        --output-dir checkpoints/value_functions \
        --num-steps 5000
"""

import dataclasses
import glob
import logging
import pathlib

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.array_typing as at
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.models.value_function import (
    ActionConditionedValueFunction,
    StateValueFunction,
    action_conditioned_vf_loss,
    state_vf_loss,
)
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    rollout_dir: str = "/seu_share/home/linli/213221101/MyDatasets/data/libero/rollouts"
    sft_config: str = "pi0_libero_fewshot"
    sft_checkpoint_dir: str = "/seu_share/home/linli/213221101/checkpoints/pi0_libero_fewshot/physical-intelligence/libero/4999"
    output_dir: str = "checkpoints/value_functions"

    num_steps: int = 5000
    batch_size: int = 64
    lr: float = 3e-4
    seed: int = 42

    hidden_dim: int = 256
    num_bins: int = 101
    feature_batch_size: int = 8


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_rollout_episodes(rollout_dir: str) -> list[dict]:
    """Load all .npz episode archives under *rollout_dir*."""
    pattern = str(pathlib.Path(rollout_dir) / "**" / "episode_*.npz")
    paths = sorted(glob.glob(pattern, recursive=True))
    logging.info(f"Found {len(paths)} episode archives under {rollout_dir}")

    episodes = []
    for p in paths:
        data = dict(np.load(p, allow_pickle=True))
        episodes.append(data)
    return episodes


def flatten_episodes(episodes: list[dict]) -> dict[str, np.ndarray]:
    """Flatten episode-level archives into a frame-level dataset."""
    all_states, all_actions, all_returns = [], [], []
    for ep in episodes:
        all_states.append(ep["states"])
        all_actions.append(ep["actions"])
        all_returns.append(ep["normalized_returns"])
    return {
        "states": np.concatenate(all_states, axis=0).astype(np.float32),
        "actions": np.concatenate(all_actions, axis=0).astype(np.float32),
        "returns": np.concatenate(all_returns, axis=0).astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Frozen encoder feature extraction
# ---------------------------------------------------------------------------

def load_pi0_model(sft_config: str, sft_checkpoint_dir: str):
    """Load a frozen Pi0 model from an SFT checkpoint for feature extraction."""
    train_config = _config.get_config(sft_config)
    params = _model.restore_params(
        pathlib.Path(sft_checkpoint_dir) / "params",
        dtype=jnp.bfloat16,
    )
    model = train_config.model.load(params)
    return model, train_config


def extract_image_features(
    model, episodes: list[dict], batch_size: int = 8
) -> np.ndarray:
    """Run frozen SigLIP on every rollout frame and mean-pool patch tokens.

    Returns float32 array of shape ``[total_frames, image_dim]`` (typically 1152).
    """
    all_feats: list[np.ndarray] = []
    for ep in tqdm.tqdm(episodes, desc="SigLIP features"):
        images = ep["images"]  # [T, 224, 224, 3] uint8
        T = len(images)
        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            batch_np = images[start:end].astype(np.float32) / 127.5 - 1.0
            image_tokens, _ = model.PaliGemma.img(
                jnp.array(batch_np), train=False
            )  # [B, num_patches, D]
            feats = jnp.mean(image_tokens, axis=1)  # [B, D]
            all_feats.append(np.asarray(feats, dtype=np.float32))
    return np.concatenate(all_feats, axis=0)


def extract_language_features(
    model, episodes: list[dict], max_token_len: int = 48
) -> np.ndarray:
    """Run frozen Gemma embedding on each unique task description.

    Returns float32 array of shape ``[total_frames, lang_dim]`` (typically 2048).
    Language features are identical for all frames within the same episode.
    """
    tokenizer = PaligemmaTokenizer(max_len=max_token_len)
    desc_to_emb: dict[str, np.ndarray] = {}
    all_feats: list[np.ndarray] = []

    for ep in tqdm.tqdm(episodes, desc="Gemma features"):
        desc = str(ep["task_description"])
        T = len(ep["states"])
        if desc not in desc_to_emb:
            tokens, mask = tokenizer.tokenize(desc)
            tokens_jax = jnp.array(tokens, dtype=jnp.int32)[None, :]      # [1, L]
            mask_f = jnp.array(mask, dtype=jnp.float32)[None, :, None]    # [1, L, 1]
            embeddings = model.PaliGemma.llm(tokens_jax, method="embed")   # [1, L, D]
            lang_emb = (
                jnp.sum(embeddings * mask_f, axis=1)
                / jnp.maximum(jnp.sum(mask_f, axis=1), 1.0)
            )  # [1, D]
            desc_to_emb[desc] = np.asarray(lang_emb[0], dtype=np.float32)
        all_feats.append(np.tile(desc_to_emb[desc], (T, 1)))

    return np.concatenate(all_feats, axis=0)


def get_or_extract_features(
    args: Args, episodes: list[dict], num_frames: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (image_features, lang_features) arrays, extracting from the SFT
    checkpoint if a cached file does not exist yet.

    The features are saved to ``{output_dir}/encoder_features.npz`` for reuse
    by ``precompute_advantages.py``.
    """
    out = pathlib.Path(args.output_dir)
    cache_path = out / "encoder_features.npz"

    if cache_path.exists():
        logging.info(f"Loading cached encoder features from {cache_path}")
        data = np.load(cache_path)
        img_feats = data["image_features"]
        lang_feats = data["lang_features"]
        if len(img_feats) == num_frames:
            return img_feats, lang_feats
        logging.warning(
            f"Cached features length ({len(img_feats)}) != current frames ({num_frames}); re-extracting."
        )

    logging.info("Loading Pi0 model for encoder feature extraction …")
    model, train_config = load_pi0_model(args.sft_config, args.sft_checkpoint_dir)

    img_feats = extract_image_features(model, episodes, batch_size=args.feature_batch_size)
    lang_feats = extract_language_features(
        model, episodes, max_token_len=train_config.model.max_token_len
    )
    assert img_feats.shape[0] == num_frames
    assert lang_feats.shape[0] == num_frames

    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, image_features=img_feats, lang_features=lang_feats)
    logging.info(
        f"Encoder features saved to {cache_path}  "
        f"image={img_feats.shape}  lang={lang_feats.shape}"
    )

    del model
    return img_feats, lang_feats


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def train_state_vf(
    args: Args, dataset: dict, image_emb: np.ndarray, lang_emb: np.ndarray
) -> StateValueFunction:
    rng = jax.random.key(args.seed)
    n = dataset["states"].shape[0]

    vf = StateValueFunction(
        image_dim=image_emb.shape[-1],
        state_dim=dataset["states"].shape[-1],
        lang_dim=lang_emb.shape[-1],
        hidden_dim=args.hidden_dim,
        num_bins=args.num_bins,
        rngs=nnx.Rngs(rng),
    )
    tx = optax.adam(args.lr)
    opt_state = tx.init(nnx.state(vf, nnx.Param))

    pbar = tqdm.tqdm(range(args.num_steps), desc="StateVF")
    for step in pbar:
        rng, batch_rng = jax.random.split(rng)
        idx = jax.random.randint(batch_rng, (args.batch_size,), 0, n)
        idx_np = np.asarray(idx)
        b_img = jnp.array(image_emb[idx_np])
        b_state = jnp.array(dataset["states"][idx_np])
        b_lang = jnp.array(lang_emb[idx_np])
        b_ret = jnp.array(dataset["returns"][idx_np])

        loss, grads = nnx.value_and_grad(state_vf_loss)(vf, b_img, b_state, b_lang, b_ret)
        params = nnx.state(vf, nnx.Param)
        updates, opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        nnx.update(vf, new_params)

        if step % 200 == 0:
            pbar.set_postfix(loss=f"{float(loss):.4f}")

    return vf


def train_action_conditioned_vf(
    args: Args, dataset: dict, image_emb: np.ndarray
) -> ActionConditionedValueFunction:
    rng = jax.random.key(args.seed + 1)
    n = dataset["states"].shape[0]
    action_dim = dataset["actions"].shape[-1]

    vf = ActionConditionedValueFunction(
        image_dim=image_emb.shape[-1],
        state_dim=dataset["states"].shape[-1],
        action_dim=action_dim,
        action_horizon=1,
        hidden_dim=args.hidden_dim,
        rngs=nnx.Rngs(rng),
    )
    tx = optax.adam(args.lr)
    opt_state = tx.init(nnx.state(vf, nnx.Param))

    pbar = tqdm.tqdm(range(args.num_steps), desc="ActionCondVF")
    for step in pbar:
        rng, batch_rng, noise_rng, time_rng = jax.random.split(rng, 4)
        idx = jax.random.randint(batch_rng, (args.batch_size,), 0, n)
        idx_np = np.asarray(idx)

        b_img = jnp.array(image_emb[idx_np])
        b_state = jnp.array(dataset["states"][idx_np])
        b_actions = jnp.array(dataset["actions"][idx_np])[:, None, :]  # [B, 1, ad]
        b_ret = jnp.array(dataset["returns"][idx_np])

        noise = jax.random.normal(noise_rng, b_actions.shape)
        t = jax.random.uniform(time_rng, (args.batch_size,))
        noisy_actions = t[:, None, None] * noise + (1 - t[:, None, None]) * b_actions

        loss, grads = nnx.value_and_grad(action_conditioned_vf_loss)(
            vf, b_img, b_state, noisy_actions, t, b_ret,
        )
        params = nnx.state(vf, nnx.Param)
        updates, opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        nnx.update(vf, new_params)

        if step % 200 == 0:
            pbar.set_postfix(loss=f"{float(loss):.4f}")

    return vf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: Args) -> None:
    logging.info("Loading rollout data …")
    episodes = load_rollout_episodes(args.rollout_dir)
    if not episodes:
        raise FileNotFoundError(f"No episode archives found under {args.rollout_dir}")
    dataset = flatten_episodes(episodes)
    n = dataset["states"].shape[0]
    logging.info(f"Flattened dataset: {n} frames")

    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    image_emb, lang_emb = get_or_extract_features(args, episodes, n)
    logging.info(f"Feature shapes – image: {image_emb.shape}, lang: {lang_emb.shape}")

    logging.info("Training StateValueFunction …")
    svf = train_state_vf(args, dataset, image_emb, lang_emb)
    svf_path = out / "state_vf.nnx"
    svf_params = jax.tree.map(np.asarray, nnx.state(svf, nnx.Param).to_pure_dict())
    np.savez(svf_path, **{"/".join(k): v for k, v in jax.tree_util.tree_leaves_with_path(svf_params)})
    logging.info(f"StateVF saved to {svf_path}.npz")

    logging.info("Training ActionConditionedValueFunction …")
    acvf = train_action_conditioned_vf(args, dataset, image_emb)
    acvf_path = out / "action_cond_vf.nnx"
    acvf_params = jax.tree.map(np.asarray, nnx.state(acvf, nnx.Param).to_pure_dict())
    np.savez(acvf_path, **{"/".join(k): v for k, v in jax.tree_util.tree_leaves_with_path(acvf_params)})
    logging.info(f"ActionCondVF saved to {acvf_path}.npz")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    tyro.cli(main)
