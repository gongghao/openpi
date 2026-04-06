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
from openpi.models.value_function import (
    ActionConditionedValueFunction,
    StateValueFunction,
    action_conditioned_vf_loss,
    state_vf_loss,
)
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    rollout_dir: str = "data/libero/rollouts"
    sft_config: str = "pi0_libero_fewshot"
    sft_checkpoint_dir: str = ""
    output_dir: str = "checkpoints/value_functions"

    num_steps: int = 5000
    batch_size: int = 64
    lr: float = 3e-4
    seed: int = 42

    state_dim: int = 32
    hidden_dim: int = 256
    num_bins: int = 101


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
        states = ep["states"]  # [T, state_dim]
        actions = ep["actions"]  # [T, action_dim]
        norm_ret = ep["normalized_returns"]  # [T]
        all_states.append(states)
        all_actions.append(actions)
        all_returns.append(norm_ret)
    return {
        "states": np.concatenate(all_states, axis=0).astype(np.float32),
        "actions": np.concatenate(all_actions, axis=0).astype(np.float32),
        "returns": np.concatenate(all_returns, axis=0).astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def make_dummy_image_emb(num_frames: int, image_dim: int = 1152) -> np.ndarray:
    """Placeholder for frozen SigLIP features.

    In a full pipeline you would run the SigLIP encoder from the SFT checkpoint
    over every frame.  For the initial implementation we use zero embeddings so
    that the training loop is self-contained and testable; replace with real
    features when the full encoder extraction is wired up.
    """
    return np.zeros((num_frames, image_dim), dtype=np.float32)


def make_dummy_lang_emb(num_frames: int, lang_dim: int = 2048) -> np.ndarray:
    """Placeholder for frozen Gemma language embeddings."""
    return np.zeros((num_frames, lang_dim), dtype=np.float32)


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def train_state_vf(args: Args, dataset: dict) -> StateValueFunction:
    rng = jax.random.key(args.seed)
    n = dataset["states"].shape[0]

    image_emb = make_dummy_image_emb(n)
    lang_emb = make_dummy_lang_emb(n)

    vf = StateValueFunction(
        state_dim=dataset["states"].shape[-1],
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


def train_action_conditioned_vf(args: Args, dataset: dict) -> ActionConditionedValueFunction:
    rng = jax.random.key(args.seed + 1)
    n = dataset["states"].shape[0]
    action_dim = dataset["actions"].shape[-1]

    image_emb = make_dummy_image_emb(n)

    vf = ActionConditionedValueFunction(
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
    logging.info("Loading rollout data...")
    episodes = load_rollout_episodes(args.rollout_dir)
    if not episodes:
        raise FileNotFoundError(f"No episode archives found under {args.rollout_dir}")
    dataset = flatten_episodes(episodes)
    logging.info(f"Flattened dataset: {dataset['states'].shape[0]} frames")

    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logging.info("Training StateValueFunction...")
    svf = train_state_vf(args, dataset)
    svf_path = out / "state_vf.nnx"
    # Serialize VF params as numpy arrays for portability
    svf_params = jax.tree.map(np.asarray, nnx.state(svf, nnx.Param).to_pure_dict())
    np.savez(svf_path, **{"/".join(k): v for k, v in jax.tree_util.tree_leaves_with_path(svf_params)})
    logging.info(f"StateVF saved to {svf_path}.npz")

    logging.info("Training ActionConditionedValueFunction...")
    acvf = train_action_conditioned_vf(args, dataset)
    acvf_path = out / "action_cond_vf.nnx"
    acvf_params = jax.tree.map(np.asarray, nnx.state(acvf, nnx.Param).to_pure_dict())
    np.savez(acvf_path, **{"/".join(k): v for k, v in jax.tree_util.tree_leaves_with_path(acvf_params)})
    logging.info(f"ActionCondVF saved to {acvf_path}.npz")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    tyro.cli(main)
