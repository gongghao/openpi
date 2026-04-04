"""Few-shot dataset wrappers for episode-level per-task sampling.

Supports selecting K complete episodes per task from a LeRobot dataset,
following the methodology in piRL (arXiv:2510.25889v3) where few-shot SFT
provides minimal seed data before RL fine-tuning.
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("openpi")


def _unwrap_lerobot_dataset(dataset: Any) -> Any:
    """Walk through TransformedDataset wrappers to find the underlying LeRobotDataset."""
    current = dataset
    for _ in range(10):
        if hasattr(current, "hf_dataset"):
            return current
        if hasattr(current, "_dataset"):
            current = current._dataset
        else:
            break
    return None


def _build_episode_task_mapping_fast(lerobot_ds: Any) -> dict[int, set[int]]:
    """Build task_index -> {episode_indices} mapping using HuggingFace dataset columns.

    This avoids iterating through __getitem__ which would load images.
    """
    hf_ds = lerobot_ds.hf_dataset
    episode_indices = np.array(hf_ds["episode_index"])
    task_indices = np.array(hf_ds["task_index"])

    task_to_episodes: dict[int, set[int]] = {}
    unique_episodes = np.unique(episode_indices)
    for ep_idx in unique_episodes:
        first_frame = np.searchsorted(episode_indices, ep_idx)
        task_idx = int(task_indices[first_frame])
        task_to_episodes.setdefault(task_idx, set()).add(int(ep_idx))

    return task_to_episodes


def _build_episode_frame_mapping_fast(lerobot_ds: Any) -> dict[int, list[int]]:
    """Build episode_index -> [frame_indices] mapping using HuggingFace dataset columns."""
    hf_ds = lerobot_ds.hf_dataset
    episode_indices = np.array(hf_ds["episode_index"])

    episode_to_frames: dict[int, list[int]] = {}
    unique_episodes, counts = np.unique(episode_indices, return_counts=True)

    pos = 0
    for ep_idx, count in zip(unique_episodes, counts):
        episode_to_frames[int(ep_idx)] = list(range(pos, pos + count))
        pos += count

    return episode_to_frames


def _build_mappings_slow(dataset: Any) -> tuple[dict[int, set[int]], dict[int, list[int]]]:
    """Fallback: scan all frames via __getitem__ to build both mappings."""
    task_to_episodes: dict[int, set[int]] = {}
    episode_to_frames: dict[int, list[int]] = {}

    total = len(dataset)
    log_interval = max(total // 10, 1)
    for idx in range(total):
        sample = dataset[idx]
        ep_idx = int(sample.get("episode_index", 0))
        task_idx = int(sample.get("task_index", 0))

        task_to_episodes.setdefault(task_idx, set()).add(ep_idx)
        episode_to_frames.setdefault(ep_idx, []).append(idx)

        if idx % log_interval == 0:
            logger.info(f"  Scanning dataset: {idx}/{total} frames")

    return task_to_episodes, episode_to_frames


class FewShotEpisodeDataset:
    """Selects K complete episodes per task from a LeRobot-based dataset.

    This wrapper preserves all frames from selected episodes, maintaining temporal
    coherence needed for action chunk prediction in flow-based VLAs.

    Args:
        base_dataset: The original dataset (LeRobotDataset or TransformedDataset wrapping one).
        episodes_per_task: Number of episodes to sample per task.
        seed: Random seed for reproducible sampling.
    """

    def __init__(self, base_dataset: Any, episodes_per_task: int, seed: int = 42):
        self._base_dataset = base_dataset
        self._episodes_per_task = episodes_per_task
        self._seed = seed

        self._selected_indices = self._build_indices()

    def _build_indices(self) -> list[int]:
        lerobot_ds = _unwrap_lerobot_dataset(self._base_dataset)

        if lerobot_ds is not None:
            logger.info("FewShotEpisodeDataset: using fast metadata path")
            task_to_episodes = _build_episode_task_mapping_fast(lerobot_ds)
            episode_to_frames = _build_episode_frame_mapping_fast(lerobot_ds)
        else:
            logger.info("FewShotEpisodeDataset: falling back to full scan")
            task_to_episodes, episode_to_frames = _build_mappings_slow(self._base_dataset)

        rng = np.random.RandomState(self._seed)
        selected_episodes: set[int] = set()

        for task_idx in sorted(task_to_episodes.keys()):
            episodes = sorted(task_to_episodes[task_idx])
            num_available = len(episodes)
            num_to_select = min(self._episodes_per_task, num_available)

            if num_available < self._episodes_per_task:
                logger.warning(
                    f"Task {task_idx}: only {num_available} episodes available, "
                    f"requested {self._episodes_per_task}"
                )

            chosen = rng.choice(episodes, size=num_to_select, replace=False)
            selected_episodes.update(int(e) for e in chosen)

        selected_frames: list[int] = []
        for ep_idx in sorted(selected_episodes):
            if ep_idx in episode_to_frames:
                selected_frames.extend(episode_to_frames[ep_idx])

        num_tasks = len(task_to_episodes)
        num_episodes = len(selected_episodes)
        logger.info(
            f"FewShotEpisodeDataset: {num_episodes} episodes across {num_tasks} tasks, "
            f"{len(selected_frames)} total frames "
            f"(from {len(self._base_dataset)} original frames)"
        )

        return selected_frames

    @property
    def task_summary(self) -> dict[str, int]:
        """Return a summary mapping for logging: task_index -> selected episode count."""
        lerobot_ds = _unwrap_lerobot_dataset(self._base_dataset)
        if lerobot_ds is not None:
            hf_ds = lerobot_ds.hf_dataset
            episode_indices = np.array(hf_ds["episode_index"])
            task_indices = np.array(hf_ds["task_index"])

            task_episodes: dict[int, set[int]] = {}
            for frame_idx in self._selected_indices:
                ep = int(episode_indices[frame_idx])
                task = int(task_indices[frame_idx])
                task_episodes.setdefault(task, set()).add(ep)
            return {f"task_{k}": len(v) for k, v in sorted(task_episodes.items())}
        return {}

    def __getitem__(self, idx: int) -> Any:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range (size={len(self)})")
        return self._base_dataset[self._selected_indices[idx]]

    def __len__(self) -> int:
        return len(self._selected_indices)
