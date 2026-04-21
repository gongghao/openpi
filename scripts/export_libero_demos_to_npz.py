#!/usr/bin/env python3
"""Export LIBERO demos (LeRobot) to per-episode ``.npz`` archives that match
the rollout schema emitted by ``collect_libero_rollouts.py``.

B8 Scheme B: the exported directory is passed alongside the policy-rollout
directory via ``--rollout-dirs`` to ``train_value_function.py`` and
``precompute_advantages.py``. The value function then learns over the union
of demos + rollouts and assigns a real advantage to every demo frame --- no
hand-picked constant, no dual Dataset wrapper.

Usage:
    python scripts/export_libero_demos_to_npz.py \
        --repo-id libero_fewshot_no_90 \
        --output-dir data/libero/demos \
        --suite-name demos

    # Limit to a subset of episode indices (fast sanity checks):
    python scripts/export_libero_demos_to_npz.py \
        --repo-id libero_fewshot_no_90 --episodes 0 1 2 \
        --output-dir data/libero/demos
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import re

import numpy as np
import tqdm
import tyro

# LIBERO's longest suite is libero_10 with 520 steps. We use it as the default
# ``max_steps`` so that demo normalized-returns live in the same numeric range
# as the longest-suite rollouts (``[-1, 0]``).
LIBERO_DEFAULT_MAX_STEPS = 520


@dataclasses.dataclass
class Args:
    repo_id: str = "libero_fewshot_no_90"
    output_dir: str = "data/libero/demos"
    # ``<output_dir>/<suite_name>/task_XXX_<slug>/episode_YYY_success.npz`` is
    # the emitted layout. The manifest treats this level as another "suite".
    suite_name: str = "demos"
    max_steps: int = LIBERO_DEFAULT_MAX_STEPS
    # Optional LeRobot dataset root (for offline datasets). Empty means default cache.
    root: str | None = None
    # Subset of episode indices. Empty tuple exports all episodes.
    episodes: tuple[int, ...] = ()
    # Skip episodes whose output .npz already exists (resume-friendly).
    resume: bool = True


def _slugify(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_") or "demo"


def _to_numpy(v) -> np.ndarray:
    if hasattr(v, "detach"):
        v = v.detach()
    if hasattr(v, "cpu"):
        v = v.cpu()
    if hasattr(v, "numpy"):
        v = v.numpy()
    return np.asarray(v)


def _to_uint8_hwc(arr) -> np.ndarray:
    x = _to_numpy(arr)
    while x.ndim == 4 and x.shape[0] == 1:
        x = x[0]
    if x.ndim == 3 and x.shape[0] in (1, 3, 4) and x.shape[-1] not in (1, 3, 4):
        x = np.transpose(x, (1, 2, 0))  # CHW -> HWC
    if x.dtype == np.uint8:
        return x
    if np.issubdtype(x.dtype, np.floating):
        hi = float(np.nanmax(x)) if x.size else 1.0
        x = np.clip(x, 0.0, 1.0) * 255.0 if hi <= 1.0 + 1e-5 else np.clip(x, 0.0, 255.0)
    return x.astype(np.uint8)


def _infer_keys(features: dict) -> tuple[str, str | None, str, str]:
    """Return ``(state_key, wrist_image_key, main_image_key, action_key)``."""
    visual_keys: list[str] = [
        k
        for k, v in features.items()
        if isinstance(v, dict) and str(v.get("dtype", "")).lower() in {"image", "video"}
    ]
    if not visual_keys:
        raise ValueError(f"No image/video features found; got {list(features.keys())}")
    main_image_key: str | None = None
    wrist_image_key: str | None = None
    for key in visual_keys:
        if "wrist" in key.lower() and wrist_image_key is None:
            wrist_image_key = key
        elif main_image_key is None:
            main_image_key = key
    if main_image_key is None:
        main_image_key = visual_keys[0]

    state_key = None
    for candidate in ("state", "observation.state"):
        if candidate in features:
            state_key = candidate
            break
    if state_key is None:
        for k in features:
            if "state" in k.lower() or "proprio" in k.lower():
                state_key = k
                break

    action_key = None
    for candidate in ("actions", "action"):
        if candidate in features:
            action_key = candidate
            break
    if action_key is None:
        for k in features:
            if "action" in k.lower():
                action_key = k
                break
    if action_key is None:
        raise ValueError(f"No action feature found; got {list(features.keys())}")

    return state_key or "state", wrist_image_key, main_image_key, action_key


def _resolve_task_text(meta, task_index: int, fallback_row: dict | None = None) -> str:
    """LeRobot v2/v3 compatible task text lookup (mirrors ``examples/libero/check_lerobot_dataset.py``)."""
    if fallback_row is not None:
        for k in ("task", "language_instruction", "instruction", "prompt"):
            v = fallback_row.get(k)
            if v is None:
                continue
            if isinstance(v, (bytes, bytearray)):
                return v.decode("utf-8", errors="replace")
            s = str(v).strip()
            if s:
                return s

    tasks = getattr(meta, "tasks", None)
    if tasks is None:
        return f"task_{task_index}"
    if isinstance(tasks, dict):
        if task_index in tasks:
            return str(tasks[task_index])
        if str(task_index) in tasks:
            return str(tasks[str(task_index)])
    if isinstance(tasks, (list, tuple)) and 0 <= int(task_index) < len(tasks):
        return str(tasks[int(task_index)])
    cols = getattr(tasks, "columns", None)
    if cols is not None and "task_index" in cols:
        try:
            sub = tasks[tasks["task_index"] == int(task_index)]
            if len(sub) >= 1:
                for col in ("task", "text", "instruction", "language_instruction"):
                    if col in sub.columns:
                        val = sub[col].iloc[0]
                        if val is not None and str(val).strip():
                            return str(val)
                return str(sub.index[0])
        except Exception:  # noqa: BLE001 -- best-effort lookup
            pass
    return f"task_{task_index}"


def _compute_rewards_returns(ep_len: int, max_steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror ``collect_libero_rollouts`` for success=True episodes."""
    rewards = np.full(ep_len, -1.0, dtype=np.float32)
    if ep_len > 0:
        rewards[-1] = 0.0
    returns = np.zeros_like(rewards)
    g = 0.0
    for t in reversed(range(ep_len)):
        g += float(rewards[t])
        returns[t] = g
    norm = np.clip(returns / max(max_steps, 1), -1.0, 0.0).astype(np.float32)
    return rewards, returns, norm


def _open_episode(lerobot_dataset, repo_id: str, root: str | None, episode: int):
    kwargs: dict = {"episodes": [int(episode)]}
    if root:
        kwargs["root"] = pathlib.Path(root)
    return lerobot_dataset.LeRobotDataset(repo_id, **kwargs)


def _load_metadata(lerobot_dataset, repo_id: str, root: str | None):
    if root:
        try:
            return lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=pathlib.Path(root))
        except TypeError:
            pass
    return lerobot_dataset.LeRobotDatasetMetadata(repo_id)


def main(args: Args) -> None:
    from lerobot.datasets import lerobot_dataset

    meta = _load_metadata(lerobot_dataset, args.repo_id, args.root)
    features = getattr(meta, "features", {}) or {}
    state_key, wrist_key, main_key, action_key = _infer_keys(features)
    logging.info(
        "Inferred keys: main=%s wrist=%s state=%s action=%s", main_key, wrist_key, state_key, action_key
    )

    total = int(getattr(meta, "total_episodes", 0) or 0)
    if args.episodes:
        episode_indices = list(args.episodes)
    elif total > 0:
        episode_indices = list(range(total))
    else:
        raise ValueError(
            f"Cannot determine total_episodes for {args.repo_id}; pass --episodes explicitly."
        )

    out_root = pathlib.Path(args.output_dir) / args.suite_name
    out_root.mkdir(parents=True, exist_ok=True)
    logging.info("Exporting %d episodes to %s", len(episode_indices), out_root)

    written = 0
    skipped = 0
    for ep in tqdm.tqdm(episode_indices, desc="episodes"):
        ds = _open_episode(lerobot_dataset, args.repo_id, args.root, ep)
        if len(ds) == 0:
            logging.warning("Episode %d is empty, skipping", ep)
            continue

        first_row = ds[0]
        task_index = 0
        if "task_index" in first_row and first_row["task_index"] is not None:
            task_index = int(_to_numpy(first_row["task_index"]).item())
        task_text = _resolve_task_text(meta, task_index, first_row)
        task_slug = _slugify(task_text)[:60]
        task_dir = out_root / f"task_{task_index:03d}_{task_slug}"
        task_dir.mkdir(parents=True, exist_ok=True)
        out_path = task_dir / f"episode_{ep:03d}_success.npz"
        if args.resume and out_path.exists():
            skipped += 1
            continue

        images, wrist_images, states, actions = [], [], [], []
        for i in range(len(ds)):
            row = ds[i]
            img = _to_uint8_hwc(row[main_key])
            images.append(img)
            if wrist_key is not None and wrist_key in row:
                wrist_images.append(_to_uint8_hwc(row[wrist_key]))
            else:
                wrist_images.append(img)
            state_val = _to_numpy(row[state_key]).astype(np.float32) if state_key in row else np.zeros(
                0, dtype=np.float32
            )
            states.append(state_val.reshape(-1))
            actions.append(_to_numpy(row[action_key]).astype(np.float32).reshape(-1))

        ep_len = len(images)
        rewards, returns, norm_returns = _compute_rewards_returns(ep_len, args.max_steps)

        np.savez_compressed(
            out_path,
            images=np.array(images, dtype=np.uint8),
            wrist_images=np.array(wrist_images, dtype=np.uint8),
            states=np.array(states, dtype=np.float32),
            actions=np.array(actions, dtype=np.float32),
            rewards=rewards,
            returns=returns,
            normalized_returns=norm_returns,
            success=np.array(True),
            task_description=np.array(task_text),
            max_steps=np.array(args.max_steps, dtype=np.int32),
        )
        written += 1

    logging.info("Done: wrote %d new episode files, skipped %d existing.", written, skipped)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    tyro.cli(main)
