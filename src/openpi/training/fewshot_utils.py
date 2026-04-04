"""Few-shot integration utilities and preset configurations.

Provides a unified entry point for applying few-shot sampling to datasets,
plus preset configurations aligned with piRL (arXiv:2510.25889v3) experiments.
"""

import logging
from typing import Any

logger = logging.getLogger("openpi")


def apply_few_shot_sampling(
    dataset: Any,
    *,
    episodes_per_task: int | None = None,
    seed: int = 42,
) -> Any:
    """Apply episode-level per-task few-shot sampling to a dataset.

    Args:
        dataset: Original dataset (LeRobotDataset or TransformedDataset wrapping one).
        episodes_per_task: Number of complete episodes to keep per task.
        seed: Random seed for reproducible sampling.

    Returns:
        A FewShotEpisodeDataset wrapping the original dataset with only selected episodes.
    """
    from openpi.training.fewshot_dataset import FewShotEpisodeDataset

    if episodes_per_task is None:
        raise ValueError("episodes_per_task must be specified")

    logger.info(f"Applying episode-level few-shot: {episodes_per_task} episodes/task, seed={seed}")
    return FewShotEpisodeDataset(
        base_dataset=dataset,
        episodes_per_task=episodes_per_task,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Preset configurations aligned with piRL (arXiv:2510.25889v3) experiments
# ---------------------------------------------------------------------------

# pi0.5: 40 trajectories across 40 subtasks (1 per task) — Table 3
LIBERO_PI05_FEWSHOT = {
    "episodes_per_task": 1,
    "seed": 42,
}

# pi0: ~58 trajectories across 30 subtasks (Spatial+Object+Goal, ~2 per task) — Table 3
LIBERO_PI0_FEWSHOT = {
    "episodes_per_task": 2,
    "seed": 42,
}

# pi0 Long: 208 trajectories across 10 subtasks (~20 per task) — Appendix C.1
LIBERO_PI0_LONG_FEWSHOT = {
    "episodes_per_task": 20,
    "seed": 42,
}
