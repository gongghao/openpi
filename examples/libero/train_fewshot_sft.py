#!/usr/bin/env python3
"""Few-shot SFT training for LIBERO, following piRL (arXiv:2510.25889v3).

This script provides a convenient entry point for few-shot SFT training on LIBERO.
It wraps the standard training pipeline with few-shot specific defaults.

Usage (via registered configs — recommended):
    # pi0.5 with 1 episode/task (40 total, matching piRL pi0.5)
    python scripts/train.py pi05_libero_fewshot --exp_name fewshot_pi05

    # pi0 with 2 episodes/task (~58 total, matching piRL pi0)
    python scripts/train.py pi0_libero_fewshot --exp_name fewshot_pi0

Usage (via this script — for quick experiments):
    python examples/libero/train_fewshot_sft.py \\
        --config pi05_libero \\
        --episodes_per_task 1 \\
        --exp_name fewshot_pi05_1shot \\
        --num_train_steps 5000

The output checkpoint can be used as the SFT initialization for RL fine-tuning.
"""

import dataclasses
import logging
import sys

import tyro

import openpi.training.config as _config

logger = logging.getLogger("openpi.fewshot")


@dataclasses.dataclass
class FewShotArgs:
    """Few-shot SFT arguments."""

    # Base config name to start from (e.g. "pi0_libero", "pi05_libero").
    config: str = "pi05_libero"

    # Number of complete episodes to sample per task.
    episodes_per_task: int = 1

    # Random seed for episode sampling.
    few_shot_seed: int = 42

    # Experiment name for checkpoint directory.
    exp_name: str = "fewshot_sft"

    # Override number of training steps (fewer steps needed for few-shot).
    num_train_steps: int = 5_000

    # Override batch size if needed.
    batch_size: int | None = None


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = tyro.cli(FewShotArgs)

    base_config = _config.get_config(args.config)
    logger.info(f"Base config: {args.config}")
    logger.info(f"Few-shot: {args.episodes_per_task} episodes/task, seed={args.few_shot_seed}")

    data_factory = base_config.data
    if hasattr(data_factory, "base_config") and data_factory.base_config is not None:
        new_base_data = dataclasses.replace( 
            data_factory.base_config,
            few_shot_enabled=True,
            few_shot_episodes_per_task=args.episodes_per_task,
            few_shot_seed=args.few_shot_seed,
        )
    else:
        new_base_data = _config.DataConfig(
            prompt_from_task=True,
            few_shot_enabled=True,
            few_shot_episodes_per_task=args.episodes_per_task,
            few_shot_seed=args.few_shot_seed,
        )
    new_data_factory = dataclasses.replace(data_factory, base_config=new_base_data)

    overrides: dict = {
        "data": new_data_factory,
        "exp_name": args.exp_name,
        "num_train_steps": args.num_train_steps,
    }
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size

    train_config = dataclasses.replace(base_config, **overrides)

    logger.info(f"Training config: {train_config.name}")
    logger.info(f"Checkpoint dir: {train_config.checkpoint_dir}")
    logger.info(f"Train steps: {train_config.num_train_steps}")

    # Delegate to the standard training main function
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "scripts"))
    from train import main as train_main

    train_main(train_config)


if __name__ == "__main__":
    main()
