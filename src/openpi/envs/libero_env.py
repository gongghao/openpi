"""LIBERO environment adapter for openpi RL rollout."""

from __future__ import annotations

import dataclasses
import logging
import math
import pathlib
from typing import Any

import numpy as np

from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer
import openpi.training.rl_config as _rl_config

logger = logging.getLogger("openpi")


def _quat_to_axisangle(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float32).copy()
    quat[3] = float(np.clip(quat[3], -1.0, 1.0))
    denom = math.sqrt(max(1.0 - float(quat[3] * quat[3]), 0.0))
    if math.isclose(denom, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * (2.0 * math.acos(float(quat[3])) / denom)).astype(np.float32)


@dataclasses.dataclass
class LiberoStepResult:
    observation: _model.Observation[np.ndarray]
    rewards: np.ndarray  # [num_envs]
    dones: np.ndarray  # [num_envs], float32 mask includes truncation
    success: np.ndarray  # [num_envs], float32 mask


class LiberoVecEnv:
    """Simple synchronous vectorized LIBERO adapter.

    This wrapper aligns LIBERO raw observations/actions with openpi's
    `Observation` schema and action-chunk execution.
    """

    def __init__(
        self,
        config: _rl_config.LiberoEnvConfig,
        *,
        state_dim: int,
        policy_action_dim: int,
    ):
        self._cfg = config
        self._state_dim = state_dim
        self._policy_action_dim = policy_action_dim
        self._num_envs = config.num_envs
        self._rng = np.random.default_rng(config.seed)
        self._episode_steps = np.zeros((self._num_envs,), dtype=np.int32)

        self._task_suite, self._task_id = self._build_task_suite(config.suite, config.task)
        self._task = self._task_suite.get_task(self._task_id)
        self._task_prompt = str(getattr(self._task, "language", "do the task"))
        self._task_init_states = self._task_suite.get_task_init_states(self._task_id)
        self._env_action_dim = int(config.env_action_dim)
        self._prompt_tokenizer = _tokenizer.PaligemmaTokenizer(max_len=config.max_token_len)
        prompt_tokens, prompt_mask = self._prompt_tokenizer.tokenize(self._task_prompt)
        self._prompt_tokens = prompt_tokens.astype(np.int32)
        self._prompt_mask = prompt_mask.astype(np.bool_)
        self._envs = [self._make_single_env() for _ in range(self._num_envs)]
        self._last_raw_obs: list[dict[str, Any] | None] = [None] * self._num_envs

    def _build_task_suite(self, suite_name: str, task_name: str) -> tuple[Any, int]:
        try:
            import libero.libero.benchmark as benchmark
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "LIBERO is required for non-dry-run RL. Install libero/libero-rl dependencies first."
            ) from exc

        suite_cls = benchmark.get_benchmark(suite_name)
        suite = suite_cls()
        normalized_target = task_name.strip().lower()
        matched_task_id = None
        for task_id in range(suite.get_num_tasks()):
            task = suite.get_task(task_id)
            bddl_stem = str(getattr(task, "bddl_file", "")).replace(".bddl", "")
            candidates = {
                str(getattr(task, "name", "")).lower(),
                str(getattr(task, "problem_folder", "")).lower(),
                bddl_stem.lower(),
                str(getattr(task, "language", "")).lower(),
                f"{str(suite_name).lower()}_{bddl_stem.lower()}",
            }
            if normalized_target in candidates or any(
                normalized_target in cand or cand in normalized_target for cand in candidates if cand
            ):
                matched_task_id = task_id
                break
        if matched_task_id is None:
            raise ValueError(f"LIBERO task '{task_name}' not found in suite '{suite_name}'.")
        return suite, matched_task_id

    def _make_single_env(self) -> Any:
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        task_bddl = pathlib.Path(get_libero_path("bddl_files")) / self._task.problem_folder / self._task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=str(task_bddl),
            camera_heights=self._cfg.image_size,
            camera_widths=self._cfg.image_size,
        )
        return env

    def _sample_init_state(self) -> Any:
        idx = int(self._rng.integers(0, len(self._task_init_states)))
        return self._task_init_states[idx]

    def _extract_obs(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        base_img = np.asarray(raw_obs["agentview_image"])[::-1, ::-1].astype(np.uint8)
        wrist_img = np.asarray(raw_obs["robot0_eye_in_hand_image"])[::-1, ::-1].astype(np.uint8)
        if self._cfg.use_right_wrist:
            right_wrist_img = wrist_img
            right_wrist_mask = np.True_
        else:
            # Match RLinf's openpi Libero policy semantics: right wrist is zero-padded for pi0.
            right_wrist_img = np.zeros_like(base_img)
            right_wrist_mask = np.False_
        state = np.concatenate(
            [
                np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32),
                _quat_to_axisangle(np.asarray(raw_obs["robot0_eef_quat"], dtype=np.float32)),
                np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32),
            ],
            axis=0,
        )
        if state.shape[0] < self._state_dim:
            state = np.pad(state, (0, self._state_dim - state.shape[0]), mode="constant")
        elif state.shape[0] > self._state_dim:
            state = state[: self._state_dim]
        return {
            "image": {
                "base_0_rgb": base_img,
                "left_wrist_0_rgb": wrist_img,
                "right_wrist_0_rgb": right_wrist_img,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": right_wrist_mask,
            },
            "state": state.astype(np.float32),
        }

    def _raw_to_observation(self, raws: list[dict[str, Any]]) -> _model.Observation[np.ndarray]:
        parsed = [self._extract_obs(obs) for obs in raws]
        batch_images = {
            "base_0_rgb": np.stack([item["image"]["base_0_rgb"] for item in parsed], axis=0),
            "left_wrist_0_rgb": np.stack([item["image"]["left_wrist_0_rgb"] for item in parsed], axis=0),
            "right_wrist_0_rgb": np.stack([item["image"]["right_wrist_0_rgb"] for item in parsed], axis=0),
        }
        image_mask = {
            "base_0_rgb": np.asarray([item["image_mask"]["base_0_rgb"] for item in parsed], dtype=np.bool_),
            "left_wrist_0_rgb": np.asarray([item["image_mask"]["left_wrist_0_rgb"] for item in parsed], dtype=np.bool_),
            "right_wrist_0_rgb": np.asarray([item["image_mask"]["right_wrist_0_rgb"] for item in parsed], dtype=np.bool_),
        }
        batch_state = np.stack([item["state"] for item in parsed], axis=0)
        return _model.Observation.from_dict(
            {
                "image": batch_images,
                "image_mask": image_mask,
                "state": batch_state,
                "tokenized_prompt": np.broadcast_to(self._prompt_tokens[None, :], (self._num_envs, self._prompt_tokens.shape[0])),
                "tokenized_prompt_mask": np.broadcast_to(
                    self._prompt_mask[None, :], (self._num_envs, self._prompt_mask.shape[0])
                ),
            }
        )

    def reset(self) -> _model.Observation[np.ndarray]:
        self._episode_steps[:] = 0
        for env_id, env in enumerate(self._envs):
            env.seed(self._cfg.seed + env_id)
            env.reset()
            init_state = self._sample_init_state()
            raw_obs = env.set_init_state(init_state=init_state)
            if isinstance(raw_obs, tuple):
                raw_obs = raw_obs[0]
            self._last_raw_obs[env_id] = raw_obs
        raws = [obs for obs in self._last_raw_obs if obs is not None]
        if len(raws) != self._num_envs:
            raise RuntimeError("LIBERO reset failed to populate all vectorized observations.")
        return self._raw_to_observation(raws)

    def _project_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        if action.shape[0] < self._env_action_dim:
            action = np.pad(action, (0, self._env_action_dim - action.shape[0]), mode="constant")
        elif action.shape[0] > self._env_action_dim:
            action = action[: self._env_action_dim]
        return action

    def align_policy_action(self, action: np.ndarray) -> np.ndarray:
        """Clamp policy action to env-action dimensions and zero the padded tail."""
        action = np.asarray(action, dtype=np.float32)
        aligned = action.copy()
        if aligned.shape[-1] > self._env_action_dim:
            aligned[..., self._env_action_dim :] = 0.0
        return aligned

    def step_action_chunk(self, action_chunk: np.ndarray) -> LiberoStepResult:
        """Execute chunked actions.

        Args:
            action_chunk: [num_envs, chunk_size, policy_action_dim]
        """
        if action_chunk.shape[0] != self._num_envs:
            raise ValueError(f"Expected num_envs={self._num_envs}, got {action_chunk.shape[0]}.")
        chunk_size = int(action_chunk.shape[1])
        rewards = np.zeros((self._num_envs,), dtype=np.float32)
        dones = np.zeros((self._num_envs,), dtype=np.bool_)
        success = np.zeros((self._num_envs,), dtype=np.bool_)

        for chunk_idx in range(chunk_size):
            repeated = max(1, int(self._cfg.action_repeat))
            for _ in range(repeated):
                for env_id, env in enumerate(self._envs):
                    if dones[env_id]:
                        continue
                    env_action = self._project_action(action_chunk[env_id, chunk_idx])
                    raw_obs, reward, terminated, info = env.step(env_action)
                    self._last_raw_obs[env_id] = raw_obs
                    rewards[env_id] += float(reward)
                    self._episode_steps[env_id] += 1
                    truncated = self._episode_steps[env_id] >= self._cfg.max_episode_steps
                    done = bool(terminated) or bool(truncated)
                    dones[env_id] = done
                    success[env_id] = success[env_id] or bool(info.get("success", terminated))
                    if done:
                        env.reset()
                        raw_reset = env.set_init_state(init_state=self._sample_init_state())
                        if isinstance(raw_reset, tuple):
                            raw_reset = raw_reset[0]
                        self._last_raw_obs[env_id] = raw_reset
                        self._episode_steps[env_id] = 0

        raws = [obs for obs in self._last_raw_obs if obs is not None]
        if len(raws) != self._num_envs:
            raise RuntimeError("LIBERO step failed to populate all vectorized observations.")
        obs = self._raw_to_observation(raws)
        return LiberoStepResult(
            observation=obs,
            rewards=rewards,
            dones=dones.astype(np.float32),
            success=success.astype(np.float32),
        )
