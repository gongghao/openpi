#!/usr/bin/env python3
"""Collect rollout trajectories with reward labels from LIBERO using a trained policy.

Produces per-episode .npz archives with (observations, actions, rewards, returns,
success) that are consumed by the value-function training and advantage
pre-computation scripts.

Usage:
    python scripts/collect_libero_rollouts.py \
        --policy.config pi0_libero_fewshot \
        --policy.dir /path/to/sft_checkpoint/4999 \
        --task-suite-names libero_spatial libero_object libero_goal libero_10 \
        --num-trials-per-task 50 \
        --output-dir data/libero/rollouts
"""

import collections
import dataclasses
import logging
import math
import pathlib
from datetime import date

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

C_FAIL = 200  # large penalty for failed episodes


@dataclasses.dataclass
class Checkpoint:
    config: str = "pi0_libero_fewshot"
    dir: str = ""


@dataclasses.dataclass
class Args:
    policy: Checkpoint = dataclasses.field(default_factory=Checkpoint)
    resize_size: int = 224
    replan_steps: int = 5

    task_suite_names: tuple[str, ...] = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    num_steps_wait: int = 10
    num_trials_per_task: int = 50

    output_dir: str = "data/libero/rollouts"
    seed: int = 7


def _max_steps_for_suite(name: str) -> int:
    table = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    if name not in table:
        raise ValueError(f"Unknown task suite: {name}")
    return table[name]


def compute_rewards(episode_length: int, success: bool) -> np.ndarray:
    rewards = np.full(episode_length, -1.0, dtype=np.float32)
    if success:
        rewards[-1] = 0.0
    else:
        rewards[-1] = -float(C_FAIL)
    return rewards


def compute_returns(rewards: np.ndarray) -> np.ndarray:
    returns = np.zeros_like(rewards)
    g = 0.0
    for t in reversed(range(len(rewards))):
        g += rewards[t]
        returns[t] = g
    return returns


def normalize_returns(returns: np.ndarray, max_episode_length: int) -> np.ndarray:
    return returns / max(max_episode_length, 1)


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    quat = np.array(quat, copy=True)
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def main(args: Args) -> None:
    np.random.seed(args.seed)

    logging.info("Loading policy locally...")
    train_config = _config.get_config(args.policy.config)
    policy = _policy_config.create_trained_policy(train_config, args.policy.dir)
    logging.info("Policy loaded.")

    benchmark_dict = benchmark.get_benchmark_dict()
    archive_date = date.today().isoformat()
    base_output = pathlib.Path(args.output_dir) / archive_date

    grand_total, grand_success = 0, 0

    for suite_name in args.task_suite_names:
        max_steps = _max_steps_for_suite(suite_name)
        task_suite = benchmark_dict[suite_name]()
        num_tasks = task_suite.n_tasks
        logging.info(f"Suite: {suite_name} ({num_tasks} tasks)")

        suite_dir = base_output / suite_name
        suite_total, suite_success = 0, 0

        for task_id in tqdm.tqdm(range(num_tasks), desc=f"{suite_name} tasks"):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

            task_segment = task_description.replace(" ", "_").replace("/", "_")
            task_dir = suite_dir / f"task_{task_id:03d}_{task_segment}"
            task_dir.mkdir(parents=True, exist_ok=True)

            for ep_idx in tqdm.tqdm(range(args.num_trials_per_task), leave=False):
                env.reset()
                obs = env.set_init_state(initial_states[ep_idx])
                action_plan: collections.deque = collections.deque()

                ep_images, ep_wrist_images, ep_states, ep_actions = [], [], [], []
                t = 0
                done = False

                while t < max_steps + args.num_steps_wait:
                    if t < args.num_steps_wait:
                        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
                    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size))

                    state = np.concatenate((
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )).astype(np.float32)

                    ep_images.append(img)
                    ep_wrist_images.append(wrist_img)
                    ep_states.append(state)

                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": state,
                            "prompt": str(task_description),
                        }
                        action_chunk = policy.infer(element)["actions"]
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    ep_actions.append(np.asarray(action, dtype=np.float32))

                    obs, _, done, _ = env.step(action.tolist())
                    if done:
                        break
                    t += 1

                success = bool(done)
                ep_len = len(ep_actions)
                rewards = compute_rewards(ep_len, success)
                returns = compute_returns(rewards)
                norm_returns = normalize_returns(returns, max_steps)

                suffix = "success" if success else "failure"
                out_path = task_dir / f"episode_{ep_idx:03d}_{suffix}.npz"
                np.savez_compressed(
                    out_path,
                    images=np.array(ep_images, dtype=np.uint8),
                    wrist_images=np.array(ep_wrist_images, dtype=np.uint8),
                    states=np.array(ep_states, dtype=np.float32),
                    actions=np.array(ep_actions, dtype=np.float32),
                    rewards=rewards,
                    returns=returns,
                    normalized_returns=norm_returns,
                    success=np.array(success),
                    task_description=np.array(task_description),
                    max_steps=np.array(max_steps, dtype=np.int32),
                )

                suite_total += 1
                if success:
                    suite_success += 1

        logging.info(f"Suite '{suite_name}': {suite_success}/{suite_total} successes")
        grand_total += suite_total
        grand_success += suite_success

    logging.info(f"All suites: {grand_success}/{grand_total} ({grand_success/max(grand_total,1)*100:.1f}%)")
    logging.info(f"Data saved to {base_output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    tyro.cli(main)
