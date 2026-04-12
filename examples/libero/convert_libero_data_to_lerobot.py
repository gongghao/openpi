"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

Per RLDS suite, each task (``language_instruction``) keeps a configurable number of episodes:
spatial / goal / object use **2** episodes per task; ``libero_10_no_noops`` uses **21** per task
(see ``EPISODES_PER_TASK_BY_SUITE``). Once every task in a suite reaches its cap, scanning
that suite stops early. Use ``--episodes-per-task-override 0`` to disable caps for **all**
suites and convert everything.

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $HF_LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import itertools
import shutil

from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro

REPO_NAME = "libero_fewshot_no_90"  # Name of the output dataset, also used for the Hugging Face Hub
RAW_DATASET_NAMES = [
    "libero_spatial_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_10_no_noops",
]  # For simplicity we will combine multiple Libero datasets into one training dataset

# LIBERO 中 spatial / goal / object / libero_10 每个 suite 均为 10 个任务；用于「配额已满」时提前结束扫描以加速。
EXPECTED_TASKS_PER_SUITE: dict[str, int | None] = {
    "libero_spatial_no_noops": 10,
    "libero_goal_no_noops": 10,
    "libero_object_no_noops": 10,
    "libero_10_no_noops": 10,
}

# 每个 suite 内，每个 task 保留的 episode 数（按 language_instruction 区分 task）。
EPISODES_PER_TASK_BY_SUITE: dict[str, int] = {
    "libero_spatial_no_noops": 2,
    "libero_goal_no_noops": 2,
    "libero_object_no_noops": 2,
    "libero_10_no_noops": 21,
}


def _suite_fewshot_quota_reached(
    task_counts: dict[str, int],
    *,
    episodes_per_task: int,
    expected_num_tasks: int | None,
) -> bool:
    """当已知该 suite 有 expected_num_tasks 个任务且每个都已存满 episodes_per_task 时返回 True。"""
    if episodes_per_task <= 0 or expected_num_tasks is None:
        return False
    if len(task_counts) < expected_num_tasks:
        return False
    return all(v >= episodes_per_task for v in task_counts.values())


def main(
    data_dir: str,
    *,
    push_to_hub: bool = False,
    episodes_per_task_override: int | None = None,
):
    """Convert RLDS episodes to LeRobot.

    Args:
        data_dir: RLDS data directory for tensorflow_datasets.
        push_to_hub: If True, push the result to the Hugging Face Hub.
        episodes_per_task_override: If set, use this cap for **every** suite (0 = unlimited).
            If ``None``, use per-suite values in ``EPISODES_PER_TASK_BY_SUITE``.
    """
    print(f"[INFO] Start converting Libero RLDS -> LeRobot. data_dir={data_dir}")
    print(f"[INFO] Output repo_id={REPO_NAME}, push_to_hub={push_to_hub}")
    if episodes_per_task_override is None:
        print(f"[INFO] Per-suite episodes/task: {EPISODES_PER_TASK_BY_SUITE}")
    else:
        print(f"[INFO] episodes_per_task_override={episodes_per_task_override} (0 = unlimited, applies to all suites)")

    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        print(f"[INFO] Removing existing output directory: {output_path}")
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
    print("[INFO] LeRobot dataset writer initialized.")

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # Per-task cap is **per RLDS suite** (reset each raw_dataset_name); task key = language_instruction.
    total_episodes = 0
    total_frames = 0
    total_skipped = 0
    all_task_names: set[str] = set()

    try:
        import tensorflow as tf
    except ImportError:
        tf = None

    for raw_dataset_name in RAW_DATASET_NAMES:
        task_episode_counts: dict[str, int] = {}
        expected_tasks = EXPECTED_TASKS_PER_SUITE.get(raw_dataset_name)
        if episodes_per_task_override is not None:
            episodes_per_task = episodes_per_task_override
        else:
            episodes_per_task = EPISODES_PER_TASK_BY_SUITE.get(raw_dataset_name, 2)
        print(
            f"[INFO] Loading dataset split: {raw_dataset_name} "
            f"(episodes_per_task={episodes_per_task or '∞'} per task in this suite)"
        )
        raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
        if tf is not None:
            try:
                raw_dataset = raw_dataset.prefetch(tf.data.AUTOTUNE)
            except Exception:
                pass
        dataset_episodes = 0
        dataset_frames = 0
        skipped_in_split = 0
        early_stopped = False
        for episode in raw_dataset:
            steps_iter = episode["steps"].as_numpy_iterator()
            first_step = next(steps_iter, None)
            if first_step is None:
                continue
            task_name = first_step["language_instruction"].decode()
            if episodes_per_task > 0 and task_episode_counts.get(task_name, 0) >= episodes_per_task:
                skipped_in_split += 1
                total_skipped += 1
                continue

            episode_frames = 0
            for step in itertools.chain((first_step,), steps_iter):
                dataset.add_frame(
                    {
                        "image": step["observation"]["image"],
                        "wrist_image": step["observation"]["wrist_image"],
                        "state": step["observation"]["state"],
                        "actions": step["action"],
                        "task": step["language_instruction"].decode(),
                    }
                )
                episode_frames += 1
            dataset.save_episode()
            task_episode_counts[task_name] = task_episode_counts.get(task_name, 0) + 1
            all_task_names.add(task_name)
            dataset_episodes += 1
            total_episodes += 1
            dataset_frames += episode_frames
            total_frames += episode_frames
            print(
                f"[INFO] {raw_dataset_name}: saved episode {dataset_episodes} "
                f"({episode_frames} frames), task={task_name!r} "
                f"[{task_episode_counts[task_name]}/{episodes_per_task or '∞'} this task in suite]. "
                f"split_frames={dataset_frames}, total_frames={total_frames}"
            )
            if _suite_fewshot_quota_reached(
                task_episode_counts,
                episodes_per_task=episodes_per_task,
                expected_num_tasks=expected_tasks,
            ):
                print(
                    f"[INFO] Suite few-shot quota done: {len(task_episode_counts)} tasks each "
                    f">= {episodes_per_task} episodes — stopping early for {raw_dataset_name} "
                    f"(skip remaining episodes in this TFDS split)."
                )
                early_stopped = True
                break
        print(
            f"[INFO] Finished {raw_dataset_name}: "
            f"{dataset_episodes} episodes written, {skipped_in_split} skipped (per-task cap), "
            f"{dataset_frames} frames."
            + (" [early stop]" if early_stopped else "")
        )

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        print("[INFO] Pushing converted dataset to Hugging Face Hub...")
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        print("[INFO] Push to hub completed.")

    print(
        f"[INFO] Conversion done. episodes={total_episodes}, skipped={total_skipped}, "
        f"frames={total_frames}, distinct_tasks (global)={len(all_task_names)}"
    )


if __name__ == "__main__":
    tyro.cli(main)