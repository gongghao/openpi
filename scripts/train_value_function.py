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
        --num-steps 5000 \
        --log-interval 100

Writes ``state_vf_metrics.jsonl`` and ``action_cond_vf_metrics.jsonl`` under ``--output-dir``
(averaged loss every ``--log-interval`` steps, same idea as ``train.py``).

Checkpointing / resume
----------------------
At the end of training (and optionally every ``--save-interval`` steps) the
scripts persist a **full** resume checkpoint next to the metrics files:

* ``{output_dir}/state_vf_resume.msgpack``
* ``{output_dir}/action_cond_vf_resume.msgpack``

Each file contains ``{params, opt_state, step, rng, num_steps}`` serialised
via ``flax.serialization``. Passing ``--resume`` will load the matching file
(if present and compatible), restore the VF parameters / Adam state / RNG,
and continue training from ``step + 1`` up to the current ``--num-steps``.
If no compatible checkpoint is found the trainer silently falls back to a
fresh start.

The downstream params-only artefacts (``state_vf.nnx.npz``,
``action_cond_vf.nnx.npz``) are still written unconditionally so that
``precompute_advantages.py`` keeps working without changes.

Tip: to **increase** the total number of VF steps across rollout rounds,
run again with ``--resume`` and a larger ``--num-steps``. To **retrain from
scratch**, drop ``--resume`` (existing resume file is overwritten at the
end of training).
"""

import dataclasses
import json
import logging
import pathlib

import flax.nnx as nnx
import flax.serialization as fser
from flax import traverse_util
from flax.training import common_utils
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
from openpi.training.rollout_manifest import (
    RolloutManifest,
    build_manifest,
    load_manifest_from_npz,
    manifest_npz_fields,
)


@dataclasses.dataclass
class Args:
    # Single rollout directory. Use ``rollout_dirs`` instead when combining
    # multiple sources (e.g. policy rollouts + demo npz for B8 / B5).
    rollout_dir: str = "/seu_share/home/linli/213221101/MyDatasets/data/libero/rollouts"
    rollout_dirs: tuple[str, ...] = ()
    sft_config: str = "pi0_libero_fewshot"
    sft_checkpoint_dir: str = "/seu_share/home/linli/213221101/checkpoints/pi0_libero_fewshot/libero_fewshot_no_90/4999"
    output_dir: str = "checkpoints/value_functions"

    num_steps: int = 5000
    batch_size: int = 64
    lr: float = 3e-4
    seed: int = 42
    log_interval: int = 100

    hidden_dim: int = 256
    num_bins: int = 101
    feature_batch_size: int = 64

    # ------------------------------------------------------------------
    # Resume / checkpointing of the VF training itself.
    # ------------------------------------------------------------------
    # When True, try to load ``{output_dir}/{state|action_cond}_vf_resume.msgpack``
    # and continue training from the stored step until ``num_steps``. If the
    # checkpoint is missing or incompatible the training falls back to a
    # fresh start (a warning is logged).
    resume: bool = False
    # Periodic resume-save interval (in optimizer steps). ``0`` means only
    # save once at the very end of training (in addition to the final
    # params-only npz used by downstream scripts).
    save_interval: int = 0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _resolve_rollout_dirs(args: Args) -> tuple[str, ...]:
    dirs: list[str] = []
    if args.rollout_dir:
        dirs.append(args.rollout_dir)
    dirs.extend(d for d in getattr(args, "rollout_dirs", ()) if d)
    if not dirs:
        raise ValueError("Specify at least one of --rollout-dir or --rollout-dirs.")
    return tuple(dirs)


def load_rollout_episodes(manifest: RolloutManifest) -> list[dict]:
    """Load all .npz episode archives in the order defined by *manifest*."""
    episodes: list[dict] = []
    for p in manifest.paths:
        episodes.append(dict(np.load(p, allow_pickle=True)))
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
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    # Flatten once so we run a single fixed-shape JIT loop over all rollout frames.
    flat_images = np.concatenate([ep["images"] for ep in episodes], axis=0)  # [N, H, W, C], uint8
    num_frames = flat_images.shape[0]
    if num_frames == 0:
        return np.empty((0, 0), dtype=np.float32)

    @jax.jit
    def _encode_images(batch_u8: jax.Array) -> jax.Array:
        # Keep host->device transfer compact by sending uint8 and normalize on device.
        x = batch_u8.astype(jnp.bfloat16) / 127.5 - 1.0
        image_tokens, _ = model.PaliGemma.img(x, train=False)  # [B, num_patches, D]
        return jnp.mean(image_tokens, axis=1).astype(jnp.float32)  # [B, D]

    def _make_padded_batch(start: int) -> tuple[np.ndarray, int]:
        end = min(start + batch_size, num_frames)
        batch_np = flat_images[start:end]
        valid = end - start
        # Pad tail to fixed B so XLA keeps one compiled shape.
        if valid < batch_size:
            pad = np.repeat(batch_np[-1:], batch_size - valid, axis=0)
            batch_np = np.concatenate([batch_np, pad], axis=0)
        return batch_np, valid

    # Reduce host/device sync overhead by flushing multiple device batches at once.
    flush_every = 8
    pending_feats: list[jax.Array] = []
    pending_valids: list[int] = []
    all_feats: list[np.ndarray] = []

    def _flush_pending() -> None:
        nonlocal pending_feats, pending_valids
        if not pending_feats:
            return
        host_batches = jax.device_get(pending_feats)
        for host_arr, valid in zip(host_batches, pending_valids, strict=True):
            all_feats.append(np.asarray(host_arr[:valid], dtype=np.float32))
        pending_feats = []
        pending_valids = []

    pbar = tqdm.tqdm(range(0, num_frames, batch_size), desc="SigLIP features")

    # Method 5: simple double-buffer prefetch pipeline (next batch prepared + device_put ahead of compute).
    next_start = 0
    next_batch_np, next_valid = _make_padded_batch(next_start)
    next_batch_dev = jax.device_put(next_batch_np)

    for start in pbar:
        current_batch_dev = next_batch_dev
        current_valid = next_valid

        next_start = start + batch_size
        if next_start < num_frames:
            next_batch_np, next_valid = _make_padded_batch(next_start)
            next_batch_dev = jax.device_put(next_batch_np)

        feats_dev = _encode_images(current_batch_dev)
        pending_feats.append(feats_dev)
        pending_valids.append(current_valid)

        if len(pending_feats) >= flush_every:
            _flush_pending()

    _flush_pending()
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
    args: Args, episodes: list[dict], manifest: RolloutManifest
) -> tuple[np.ndarray, np.ndarray]:
    """Return (image_features, lang_features) arrays, extracting from the SFT
    checkpoint if a cached file does not exist yet.

    The features are saved to ``{output_dir}/encoder_features.npz`` together
    with the manifest fields so that ``precompute_advantages.py`` can refuse
    to reuse stale features after the rollout directory has changed.
    """
    out = pathlib.Path(args.output_dir)
    cache_path = out / "encoder_features.npz"
    num_frames = manifest.num_frames

    if cache_path.exists():
        logging.info(f"Loading cached encoder features from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        img_feats = data["image_features"]
        lang_feats = data["lang_features"]
        cached = load_manifest_from_npz(data)
        if cached is not None and cached.manifest_sha == manifest.manifest_sha:
            return img_feats, lang_feats
        if cached is None and len(img_feats) == num_frames:
            logging.warning(
                "Legacy feature cache without manifest_sha; length matches so reusing, "
                "but re-extract to upgrade."
            )
            return img_feats, lang_feats
        logging.warning(
            "Feature cache manifest_sha / length mismatch (cached=%s, current=%s); re-extracting.",
            cached.manifest_sha[:12] if cached else f"len={len(img_feats)}",
            manifest.manifest_sha[:12],
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
    np.savez_compressed(
        cache_path,
        image_features=img_feats,
        lang_features=lang_feats,
        **manifest_npz_fields(manifest),
    )
    logging.info(
        f"Encoder features saved to {cache_path}  "
        f"image={img_feats.shape}  lang={lang_feats.shape}  manifest_sha={manifest.manifest_sha[:12]}"
    )

    del model
    return img_feats, lang_feats


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def _write_metrics_row(metrics_path: pathlib.Path, step: int, reduced_info: dict) -> None:
    row = {"step": int(step)}
    for k, v in reduced_info.items():
        row[k] = float(v)
    with metrics_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def _flush_metrics_log(
    pbar,
    infos: list,
    metrics_path: pathlib.Path,
    step: int,
) -> None:
    if not infos:
        return
    stacked_infos = common_utils.stack_forest(infos)
    reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
    info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
    pbar.write(f"Step {step}: {info_str}")
    _write_metrics_row(metrics_path, step, reduced_info)


def _save_nnx_params_npz(path: pathlib.Path, params_pure: dict) -> None:
    """Save nested NNX param dict as slash-key npz."""
    flat = traverse_util.flatten_dict(params_pure, sep="/")
    np.savez(path, **{k: np.asarray(v) for k, v in flat.items()})


# ---------------------------------------------------------------------------
# Resume-checkpoint helpers (params + opt_state + step + rng)
# ---------------------------------------------------------------------------

def _resume_path(out: pathlib.Path, prefix: str) -> pathlib.Path:
    return out / f"{prefix}_resume.msgpack"


def _key_to_raw(key: jax.Array) -> np.ndarray:
    """Serialise a typed PRNG key as a host-side uint32 ndarray."""
    return np.asarray(jax.random.key_data(key))


def _raw_to_key(arr) -> jax.Array:
    return jax.random.wrap_key_data(jnp.asarray(arr))


def _make_resume_template(vf, opt_state, rng) -> dict:
    """Build the template pytree used by ``flax.serialization.from_bytes``."""
    return {
        "params": nnx.state(vf, nnx.Param).to_pure_dict(),
        "opt_state": opt_state,
        "step": 0,
        "rng": _key_to_raw(rng),
        "num_steps": 0,
    }


def _save_resume_state(
    path: pathlib.Path,
    vf,
    opt_state,
    step: int,
    rng,
    num_steps: int,
) -> None:
    """Serialise ``{params, opt_state, step, rng, num_steps}`` to *path* via msgpack.

    ``step`` is the **index of the last completed step** so that resuming
    continues from ``step + 1``.
    """
    state = {
        "params": nnx.state(vf, nnx.Param).to_pure_dict(),
        "opt_state": opt_state,
        "step": int(step),
        "rng": _key_to_raw(rng),
        "num_steps": int(num_steps),
    }
    # Use flax's high-level serializer so nested optax states (including tuples)
    # are converted via to_state_dict before msgpack encoding.
    buf = fser.to_bytes(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(buf)
    tmp.replace(path)


def _try_load_resume_state(
    path: pathlib.Path,
    vf,
    opt_state,
    rng,
) -> dict | None:
    """Attempt to load a resume checkpoint. Returns ``None`` on any error.

    The VF / optimiser state are **not** mutated here; callers must apply
    the returned ``params`` / ``opt_state`` themselves.
    """
    if not path.exists():
        return None
    template = _make_resume_template(vf, opt_state, rng)
    try:
        restored = fser.from_bytes(template, path.read_bytes())
    except Exception as e:  # noqa: BLE001
        logging.warning("Failed to load resume state from %s: %s", path, e)
        return None
    return restored


def _apply_resume_state(
    vf,
    loaded: dict,
) -> None:
    """Copy restored params into *vf* in-place."""
    vf_state = nnx.state(vf, nnx.Param)
    vf_state.replace_by_pure_dict(loaded["params"])
    nnx.update(vf, vf_state)


def train_state_vf(
    args: Args,
    dataset: dict,
    image_emb: np.ndarray,
    lang_emb: np.ndarray,
    metrics_path: pathlib.Path,
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

    out = pathlib.Path(args.output_dir)
    resume_file = _resume_path(out, "state_vf")
    start_step = 0
    if args.resume:
        loaded = _try_load_resume_state(resume_file, vf, opt_state, rng)
        if loaded is not None:
            _apply_resume_state(vf, loaded)
            opt_state = loaded["opt_state"]
            rng = _raw_to_key(loaded["rng"])
            start_step = int(loaded["step"]) + 1
            prev_total = int(loaded.get("num_steps", 0))
            logging.info(
                "Resumed StateVF from %s: last_step=%d, prev_num_steps=%d, resuming at %d",
                resume_file, int(loaded["step"]), prev_total, start_step,
            )
        else:
            logging.info("--resume requested but %s not usable; starting StateVF from scratch.", resume_file)

    if start_step >= args.num_steps:
        logging.info(
            "StateVF already trained for %d >= num_steps=%d steps; skipping loop.",
            start_step, args.num_steps,
        )
        return vf

    infos: list = []
    pbar = tqdm.tqdm(range(start_step, args.num_steps), desc="StateVF", initial=start_step, total=args.num_steps)
    last_step = args.num_steps - 1
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

        infos.append({"loss": jnp.asarray(loss)})
        if step % args.log_interval == 0:
            _flush_metrics_log(pbar, infos, metrics_path, step)
            infos = []
        if step % 200 == 0:
            pbar.set_postfix(loss=f"{float(loss):.4f}")
        if args.save_interval > 0 and step > start_step and step % args.save_interval == 0:
            _save_resume_state(resume_file, vf, opt_state, step, rng, args.num_steps)

    _flush_metrics_log(pbar, infos, metrics_path, last_step)
    _save_resume_state(resume_file, vf, opt_state, last_step, rng, args.num_steps)
    logging.info("StateVF resume checkpoint saved to %s (step=%d)", resume_file, last_step)

    return vf


def train_action_conditioned_vf(
    args: Args,
    dataset: dict,
    image_emb: np.ndarray,
    metrics_path: pathlib.Path,
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

    out = pathlib.Path(args.output_dir)
    resume_file = _resume_path(out, "action_cond_vf")
    start_step = 0
    if args.resume:
        loaded = _try_load_resume_state(resume_file, vf, opt_state, rng)
        if loaded is not None:
            _apply_resume_state(vf, loaded)
            opt_state = loaded["opt_state"]
            rng = _raw_to_key(loaded["rng"])
            start_step = int(loaded["step"]) + 1
            prev_total = int(loaded.get("num_steps", 0))
            logging.info(
                "Resumed ActionCondVF from %s: last_step=%d, prev_num_steps=%d, resuming at %d",
                resume_file, int(loaded["step"]), prev_total, start_step,
            )
        else:
            logging.info("--resume requested but %s not usable; starting ActionCondVF from scratch.", resume_file)

    if start_step >= args.num_steps:
        logging.info(
            "ActionCondVF already trained for %d >= num_steps=%d steps; skipping loop.",
            start_step, args.num_steps,
        )
        return vf

    infos: list = []
    pbar = tqdm.tqdm(
        range(start_step, args.num_steps), desc="ActionCondVF", initial=start_step, total=args.num_steps
    )
    last_step = args.num_steps - 1
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

        infos.append({"loss": jnp.asarray(loss)})
        if step % args.log_interval == 0:
            _flush_metrics_log(pbar, infos, metrics_path, step)
            infos = []
        if step % 200 == 0:
            pbar.set_postfix(loss=f"{float(loss):.4f}")
        if args.save_interval > 0 and step > start_step and step % args.save_interval == 0:
            _save_resume_state(resume_file, vf, opt_state, step, rng, args.num_steps)

    _flush_metrics_log(pbar, infos, metrics_path, last_step)
    _save_resume_state(resume_file, vf, opt_state, last_step, rng, args.num_steps)
    logging.info("ActionCondVF resume checkpoint saved to %s (step=%d)", resume_file, last_step)

    return vf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: Args) -> None:
    rollout_dirs = _resolve_rollout_dirs(args)
    manifest = build_manifest(rollout_dirs)
    logging.info(
        "Built rollout manifest: %d episodes, %d frames, manifest_sha=%s",
        manifest.num_episodes,
        manifest.num_frames,
        manifest.manifest_sha[:12],
    )

    episodes = load_rollout_episodes(manifest)
    dataset = flatten_episodes(episodes)
    n = dataset["states"].shape[0]
    if n != manifest.num_frames:
        raise RuntimeError(
            f"Manifest reports {manifest.num_frames} frames but flattened dataset has {n}."
        )
    logging.info(f"Flattened dataset: {n} frames")

    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    image_emb, lang_emb = get_or_extract_features(args, episodes, manifest)
    logging.info(f"Feature shapes – image: {image_emb.shape}, lang: {lang_emb.shape}")

    state_vf_metrics_path = out / "state_vf_metrics.jsonl"
    action_cond_vf_metrics_path = out / "action_cond_vf_metrics.jsonl"
    logging.info(f"Metrics (state VF): {state_vf_metrics_path}")
    logging.info(f"Metrics (action-cond VF): {action_cond_vf_metrics_path}")

    logging.info("Training StateValueFunction …")
    svf = train_state_vf(args, dataset, image_emb, lang_emb, state_vf_metrics_path)
    svf_path = out / "state_vf.nnx"
    svf_params = nnx.state(svf, nnx.Param).to_pure_dict()
    _save_nnx_params_npz(svf_path, svf_params)
    logging.info(f"StateVF saved to {svf_path}.npz")

    logging.info("Training ActionConditionedValueFunction …")
    acvf = train_action_conditioned_vf(args, dataset, image_emb, action_cond_vf_metrics_path)
    acvf_path = out / "action_cond_vf.nnx"
    acvf_params = nnx.state(acvf, nnx.Param).to_pure_dict()
    _save_nnx_params_npz(acvf_path, acvf_params)
    logging.info(f"ActionCondVF saved to {acvf_path}.npz")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    tyro.cli(main)
