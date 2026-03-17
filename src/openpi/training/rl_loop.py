"""JAX-native RL loop with PPO+GAE update path."""

import dataclasses
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax import struct
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.shared.array_typing as at
from openpi.models import model as _model
import openpi.training.checkpoints as _checkpoints
import openpi.training.gae as _gae
import openpi.training.ppo as _ppo
import openpi.training.rl_config as _rl_config

_OBS_DIM = 8
_SMOKE_SNAPSHOT_KEYS = (
    "rollout/mean_reward",
    "rollout/success_rate",
    "rollout/mean_advantage",
    "rollout/returns_mean",
    "ppo/policy_loss",
    "ppo/value_loss",
    "ppo/approx_kl",
    "ppo/clip_frac",
    "ppo/entropy",
    "ppo/grad_norm",
    "ppo/value_clip_ratio",
    "ppo/explained_variance",
    "ppo/epochs_ran",
    "ppo/early_stop",
)
_WANDB_SCHEMA_KEYS = (
    "train/iteration",
    "train/env_steps",
    *_SMOKE_SNAPSHOT_KEYS,
    "ppo/loss",
    "ppo/policy_grad_norm",
    "ppo/value_grad_norm",
    "ppo/ratio",
    "ppo/clipped_ratio",
    "ppo/dual_clipped_ratio",
    "ppo/value_loss_scaled",
    "rollout/steps",
)


@struct.dataclass
class RLTrainState:
    """RL train state with separate actor/critic optimizers."""

    iteration: jax.Array
    env_steps: jax.Array
    policy_params: nnx.State
    policy_model_def: nnx.GraphDef[_model.BaseModel] = struct.field(pytree_node=False)
    value_head_params: nnx.State
    value_head_def: nnx.GraphDef[nnx.Module] = struct.field(pytree_node=False)
    policy_opt_state: optax.OptState
    value_opt_state: optax.OptState
    policy_tx: optax.GradientTransformation = struct.field(pytree_node=False)
    value_tx: optax.GradientTransformation = struct.field(pytree_node=False)


class ValueHead(nnx.Module):
    """Minimal MLP critic head over pi0 value features."""

    def __init__(self, in_dim: int, hidden_dim: int, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(in_dim, hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.out = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(self, features: jax.Array) -> jax.Array:
        x = nnx.swish(self.fc1(features))
        x = nnx.swish(self.fc2(x))
        return self.out(x)


def init_logging() -> None:
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _rl_config.RLTrainConfig, *, resuming: bool, enabled: bool = True) -> None:
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    wandb.run.log_code(epath.Path(__file__).parent.parent)
    wandb.define_metric("train/iteration")
    for metric_key in _WANDB_SCHEMA_KEYS:
        if metric_key != "train/iteration":
            wandb.define_metric(metric_key, step_metric="train/iteration")


def _make_tx(config: _rl_config.PPOConfig) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate),
    )


def _init_train_state(rng: at.KeyArrayLike, config: _rl_config.RLTrainConfig) -> RLTrainState:
    policy_rng, value_rng, head_rng = jax.random.split(rng, 3)
    policy_tx = _make_tx(config.ppo)
    value_tx = _make_tx(config.ppo)

    policy_model = config.model.create(policy_rng)
    if config.policy_params_path:
        loaded = _model.restore_params(config.policy_params_path, restore_type=np.ndarray)
        policy_model = config.model.load(loaded, remove_extra_params=True)
    policy_params = nnx.state(policy_model)
    policy_model_def = nnx.graphdef(policy_model)

    fake_obs = config.model.fake_obs(batch_size=1)
    fake_features = _policy_value_features_from_state(policy_model_def, policy_params, fake_obs)
    critic_hidden_dim = max(128, int(fake_features.shape[-1]))
    value_head = ValueHead(int(fake_features.shape[-1]), critic_hidden_dim, rngs=nnx.Rngs(head_rng))
    value_head_params = nnx.state(value_head)
    value_head_def = nnx.graphdef(value_head)
    return RLTrainState(
        iteration=jnp.asarray(0, dtype=jnp.int32),
        env_steps=jnp.asarray(0, dtype=jnp.int32),
        policy_params=policy_params,
        policy_model_def=policy_model_def,
        value_head_params=value_head_params,
        value_head_def=value_head_def,
        policy_opt_state=policy_tx.init(policy_params.filter(nnx.Param)),
        value_opt_state=value_tx.init(value_head_params.filter(nnx.Param)),
        policy_tx=policy_tx,
        value_tx=value_tx,
    )


def _policy_sample_with_trace(
    state: RLTrainState,
    rng: at.KeyArrayLike,
    observation: _model.Observation[jax.Array],
    *,
    num_steps: int,
    stochastic: bool,
) -> dict[str, jax.Array]:
    model = nnx.merge(state.policy_model_def, state.policy_params)
    model.eval()
    sampled = model.sample_actions_with_trace(
        rng,
        observation,
        num_steps=num_steps,
        stochastic=stochastic,
    )
    return {
        "actions": sampled["actions"],
        "trace": sampled["trace"],
    }


def _policy_logprob(
    state: RLTrainState,
    trace: dict[str, jax.Array],
    *,
    samples: jax.Array,
    action_mask: jax.Array | None = None,
) -> jax.Array:
    model = nnx.merge(state.policy_model_def, state.policy_params)
    model.eval()
    return model.compute_flow_noise_logprob(
        trace,
        samples=samples,
        action_mask=action_mask,
        include_initial_prior=False,
    )["joint_logprob"]


def _policy_value_features_from_state(
    policy_model_def: nnx.GraphDef[_model.BaseModel],
    policy_params: nnx.State,
    observation: _model.Observation[jax.Array],
) -> jax.Array:
    model = nnx.merge(policy_model_def, policy_params)
    model.eval()
    return model.value_features(observation, train=False)


def _value_prediction_from_state(
    state: RLTrainState,
    observation: _model.Observation[jax.Array],
) -> jax.Array:
    features = _policy_value_features_from_state(state.policy_model_def, state.policy_params, observation)
    features = jax.lax.stop_gradient(features)
    value_head = nnx.merge(state.value_head_def, state.value_head_params)
    value_head.eval()
    return jnp.squeeze(value_head(features), axis=-1)


def _collect_rollout(
    rng: at.KeyArrayLike,
    state: RLTrainState,
    config: _rl_config.RLTrainConfig,
) -> tuple[dict[str, jax.Array], dict[str, jax.Array], _model.Observation[np.ndarray] | None]:
    horizon = config.rollout.rollout_horizon
    num_envs = config.env.num_envs
    action_dim = config.model.action_dim
    action_horizon = config.model.action_horizon
    action_mask = (jnp.arange(action_dim) < config.env.env_action_dim).astype(jnp.float32)

    if config.dry_run:
        obs_key, act_key, done_key = jax.random.split(rng, 3)
        state_seq = jax.random.normal(obs_key, (horizon + 1, num_envs, _OBS_DIM), dtype=jnp.float32)
        image_seq = jax.random.uniform(
            obs_key,
            (horizon + 1, num_envs, 224, 224, 3),
            minval=-1.0,
            maxval=1.0,
            dtype=jnp.float32,
        )
        prompt_tokens = jnp.zeros((horizon + 1, num_envs, config.model.max_token_len), dtype=jnp.int32)
        prompt_mask = jnp.ones((horizon + 1, num_envs, config.model.max_token_len), dtype=jnp.bool_)
        image_mask = jnp.ones((horizon + 1, num_envs), dtype=jnp.bool_)
        obs = _model.Observation(
            images={
                "base_0_rgb": image_seq[:-1],
                "left_wrist_0_rgb": image_seq[:-1],
                "right_wrist_0_rgb": image_seq[:-1],
            },
            image_masks={
                "base_0_rgb": image_mask[:-1],
                "left_wrist_0_rgb": image_mask[:-1],
                "right_wrist_0_rgb": image_mask[:-1],
            },
            state=state_seq[:-1],
            tokenized_prompt=prompt_tokens[:-1],
            tokenized_prompt_mask=prompt_mask[:-1],
        )
        next_obs_state = state_seq[1:]
        obs_flat = jax.tree.map(lambda x: x.reshape((-1, *x.shape[2:])), obs)
        sampled = _policy_sample_with_trace(
            state,
            act_key,
            obs_flat,
            num_steps=config.policy_denoise_steps,
            stochastic=True,
        )
        actions_flat = sampled["actions"]
        old_logprob_flat = _policy_logprob(
            state,
            sampled["trace"],
            samples=actions_flat[:, None, :, :],
            action_mask=action_mask,
        )

        actions = actions_flat.reshape(horizon, num_envs, action_horizon, action_dim)
        values = _value_prediction_from_state(state, obs_flat).reshape(horizon, num_envs)
        next_flat = _model.Observation(
            images={k: v[-1] for k, v in obs.images.items()},
            image_masks={k: v[-1] for k, v in obs.image_masks.items()},
            state=next_obs_state[-1],
            tokenized_prompt=obs.tokenized_prompt[-1],
            tokenized_prompt_mask=obs.tokenized_prompt_mask[-1],
        )
        last_values = _value_prediction_from_state(state, next_flat)

        reward_target = jnp.pad(obs.state, ((0, 0), (0, 0), (0, max(0, action_dim - _OBS_DIM))))[..., :action_dim]
        rewards = -jnp.mean(jnp.square(actions[:, :, 0, :] - reward_target), axis=-1)
        done_prob = jnp.asarray(1.0 / max(config.env.max_episode_steps, 1), dtype=jnp.float32)
        dones = jax.random.bernoulli(done_key, p=done_prob, shape=(horizon, num_envs)).astype(jnp.float32)
        success_rate = jnp.mean((rewards > -1.0).astype(jnp.float32))
        env_steps = config.steps_per_iteration
        final_obs = None
    else:
        rollout_rng = rng
        from openpi.envs.libero_env import LiberoVecEnv

        if not hasattr(_collect_rollout, "_libero_env"):
            env_cfg = dataclasses.replace(config.env, max_token_len=config.model.max_token_len)
            _collect_rollout._libero_env = LiberoVecEnv(  # type: ignore[attr-defined]
                env_cfg,
                state_dim=_OBS_DIM,
                policy_action_dim=action_dim,
            )
            _collect_rollout._libero_obs = _collect_rollout._libero_env.reset()  # type: ignore[attr-defined]

        env: LiberoVecEnv = _collect_rollout._libero_env  # type: ignore[attr-defined]
        current_obs: _model.Observation[np.ndarray] = _collect_rollout._libero_obs  # type: ignore[attr-defined]
        obs = _model.Observation(
            images={
                "base_0_rgb": jnp.zeros((horizon, num_envs, 224, 224, 3), dtype=jnp.float32),
                "left_wrist_0_rgb": jnp.zeros((horizon, num_envs, 224, 224, 3), dtype=jnp.float32),
                "right_wrist_0_rgb": jnp.zeros((horizon, num_envs, 224, 224, 3), dtype=jnp.float32),
            },
            image_masks={
                "base_0_rgb": jnp.zeros((horizon, num_envs), dtype=jnp.bool_),
                "left_wrist_0_rgb": jnp.zeros((horizon, num_envs), dtype=jnp.bool_),
                "right_wrist_0_rgb": jnp.zeros((horizon, num_envs), dtype=jnp.bool_),
            },
            state=jnp.zeros((horizon, num_envs, _OBS_DIM), dtype=jnp.float32),
            tokenized_prompt=jnp.zeros((horizon, num_envs, config.model.max_token_len), dtype=jnp.int32),
            tokenized_prompt_mask=jnp.zeros((horizon, num_envs, config.model.max_token_len), dtype=jnp.bool_),
        )
        next_obs = jnp.zeros((horizon, num_envs, _OBS_DIM), dtype=jnp.float32)
        actions = jnp.zeros((horizon, num_envs, action_horizon, action_dim), dtype=jnp.float32)
        old_logprob = jnp.zeros((horizon, num_envs), dtype=jnp.float32)
        rewards = jnp.zeros((horizon, num_envs), dtype=jnp.float32)
        dones = jnp.zeros((horizon, num_envs), dtype=jnp.float32)
        success = jnp.zeros((horizon, num_envs), dtype=jnp.float32)

        for t in range(horizon):
            rollout_rng, step_key = jax.random.split(rollout_rng)
            obs_t = jax.tree.map(jnp.asarray, current_obs)
            obs = _model.Observation(
                images={k: obs.images[k].at[t].set(obs_t.images[k]) for k in obs.images},
                image_masks={k: obs.image_masks[k].at[t].set(obs_t.image_masks[k]) for k in obs.image_masks},
                state=obs.state.at[t].set(obs_t.state),
                tokenized_prompt=obs.tokenized_prompt.at[t].set(obs_t.tokenized_prompt),
                tokenized_prompt_mask=obs.tokenized_prompt_mask.at[t].set(obs_t.tokenized_prompt_mask),
            )
            sampled = _policy_sample_with_trace(
                state,
                step_key,
                obs_t,
                num_steps=config.policy_denoise_steps,
                stochastic=True,
            )
            actions_t = jnp.asarray(env.align_policy_action(np.asarray(sampled["actions"], dtype=np.float32)))
            logprob_t = _policy_logprob(
                state,
                sampled["trace"],
                samples=actions_t[:, None, :, :],
                action_mask=action_mask,
            )
            if actions_t.shape[1] >= config.env.action_chunk_size:
                action_chunk = actions_t[:, : config.env.action_chunk_size, :]
            else:
                pad = jnp.repeat(actions_t[:, -1:, :], config.env.action_chunk_size - actions_t.shape[1], axis=1)
                action_chunk = jnp.concatenate([actions_t, pad], axis=1)
            env_out = env.step_action_chunk(np.asarray(action_chunk, dtype=np.float32))
            current_obs = env_out.observation
            next_obs_t = jnp.asarray(current_obs.state, dtype=jnp.float32)

            next_obs = next_obs.at[t].set(next_obs_t)
            actions = actions.at[t].set(actions_t)
            old_logprob = old_logprob.at[t].set(logprob_t)
            rewards = rewards.at[t].set(jnp.asarray(env_out.rewards, dtype=jnp.float32))
            dones = dones.at[t].set(jnp.asarray(env_out.dones, dtype=jnp.float32))
            success = success.at[t].set(jnp.asarray(env_out.success, dtype=jnp.float32))

        _collect_rollout._libero_obs = current_obs  # type: ignore[attr-defined]
        obs_flat = jax.tree.map(lambda x: x.reshape((-1, *x.shape[2:])), obs)
        values = _value_prediction_from_state(state, obs_flat).reshape(horizon, num_envs)
        final_obs_jax = jax.tree.map(jnp.asarray, current_obs)
        last_values = _value_prediction_from_state(state, final_obs_jax)
        old_logprob_flat = old_logprob.reshape(-1)
        success_rate = jnp.mean(success)
        env_steps = config.steps_per_iteration * config.env.action_chunk_size * max(1, config.env.action_repeat)
        final_obs = current_obs

    advantages, returns = _gae.compute_gae(
        rewards,
        dones,
        values,
        last_values,
        gamma=config.ppo.gamma,
        gae_lambda=config.ppo.gae_lambda,
    )
    done_prefix = jnp.concatenate([jnp.zeros((1, num_envs), dtype=jnp.float32), jnp.cumsum(dones, axis=0)[:-1]], axis=0)
    loss_mask = (done_prefix == 0).astype(jnp.float32)
    advantages = _gae.maybe_normalize_advantages(
        advantages,
        normalize=config.ppo.normalize_advantage,
        loss_mask=loss_mask,
    )

    rollout_batch = {
        "obs": obs,
        "actions": actions,
        "old_logprob": old_logprob_flat.reshape(horizon, num_envs, 1),
        "old_values": values[..., None],
        "advantages": advantages[..., None],
        "returns": returns[..., None],
        "loss_mask": loss_mask[..., None],
        "action_mask": jnp.broadcast_to(action_mask, (horizon, num_envs, action_dim)),
    }
    rollout_metrics = {
        "rollout/mean_reward": jnp.mean(rewards),
        "rollout/success_rate": success_rate,
        "rollout/mean_advantage": jnp.mean(advantages),
        "rollout/returns_mean": jnp.mean(returns),
        "rollout/steps": jnp.asarray(env_steps, dtype=jnp.int32),
    }
    return rollout_batch, rollout_metrics, final_obs


def _flatten_rollout_batch(rollout_batch: dict[str, jax.Array]) -> dict[str, jax.Array]:
    return {
        "obs": jax.tree.map(lambda x: x.reshape((-1, *x.shape[2:])), rollout_batch["obs"]),
        "actions": rollout_batch["actions"].reshape(-1, rollout_batch["actions"].shape[-2], rollout_batch["actions"].shape[-1]),
        "old_logprob": rollout_batch["old_logprob"].reshape(-1, 1),
        "old_values": rollout_batch["old_values"].reshape(-1, 1),
        "advantages": rollout_batch["advantages"].reshape(-1, 1),
        "returns": rollout_batch["returns"].reshape(-1, 1),
        "loss_mask": rollout_batch["loss_mask"].reshape(-1, 1),
        "action_mask": rollout_batch["action_mask"].reshape(-1, rollout_batch["action_mask"].shape[-1]),
    }


def _update_minibatch(
    rng: at.KeyArrayLike,
    state: RLTrainState,
    minibatch: dict[str, Any],
    config: _rl_config.RLTrainConfig,
) -> tuple[RLTrainState, dict[str, jax.Array]]:
    critic_features = jax.lax.stop_gradient(
        _policy_value_features_from_state(state.policy_model_def, state.policy_params, minibatch["obs"])
    )

    def actor_loss_fn(model: _model.BaseModel) -> tuple[jax.Array, dict[str, jax.Array]]:
        sampled = model.sample_actions_with_trace(
            rng,
            minibatch["obs"],
            num_steps=config.policy_denoise_steps,
            stochastic=False,
        )
        new_logprob = model.compute_flow_noise_logprob(
            sampled["trace"],
            samples=minibatch["actions"][:, None, :, :],
            action_mask=minibatch["action_mask"],
            include_initial_prior=False,
        )["joint_logprob"]
        entropy = jnp.asarray(0.0, dtype=jnp.float32)
        total_loss, actor_metrics = _ppo.ppo_actor_metrics(
            new_logprob=new_logprob,
            old_logprob=minibatch["old_logprob"].squeeze(-1),
            advantages=minibatch["advantages"].squeeze(-1),
            clip_ratio_low=config.ppo.clip_ratio_low,
            clip_ratio_high=config.ppo.clip_ratio_high,
            entropy=entropy,
            entropy_coef=config.ppo.entropy_coef,
            loss_mask=minibatch["loss_mask"].squeeze(-1),
            clip_ratio_c=config.ppo.clip_ratio_c,
        )
        return total_loss, actor_metrics

    def critic_loss_fn(value_head: ValueHead) -> tuple[jax.Array, dict[str, jax.Array]]:
        new_values = jnp.squeeze(value_head(critic_features), axis=-1)
        total_loss, value_metrics = _ppo.ppo_value_loss(
            new_values=new_values,
            old_values=minibatch["old_values"].squeeze(-1),
            returns=minibatch["returns"].squeeze(-1),
            value_coef=config.ppo.value_coef,
            value_clip_epsilon=config.ppo.value_clip_epsilon,
            huber_delta=config.ppo.huber_delta,
            loss_mask=minibatch["loss_mask"].squeeze(-1),
        )
        return total_loss, value_metrics

    policy_model = nnx.merge(state.policy_model_def, state.policy_params)
    policy_model.train()
    diff_state = nnx.DiffState(0, nnx.Param)
    (actor_total_loss, actor_metrics), actor_grads = nnx.value_and_grad(actor_loss_fn, argnums=diff_state, has_aux=True)(
        policy_model
    )
    actor_params = state.policy_params.filter(nnx.Param)
    actor_updates, new_policy_opt_state = state.policy_tx.update(
        actor_grads, state.policy_opt_state, actor_params
    )
    new_actor_params = optax.apply_updates(actor_params, actor_updates)
    nnx.update(policy_model, new_actor_params)
    new_policy_params = nnx.state(policy_model)
    actor_grad_norm = optax.global_norm(actor_grads)

    value_head = nnx.merge(state.value_head_def, state.value_head_params)
    value_head.train()
    value_diff_state = nnx.DiffState(0, nnx.Param)
    (critic_total_loss, critic_metrics), critic_grads = nnx.value_and_grad(
        critic_loss_fn, argnums=value_diff_state, has_aux=True
    )(value_head)
    value_head_params = state.value_head_params.filter(nnx.Param)
    critic_updates, new_value_opt_state = state.value_tx.update(critic_grads, state.value_opt_state, value_head_params)
    new_value_head_params = optax.apply_updates(value_head_params, critic_updates)
    nnx.update(value_head, new_value_head_params)
    updated_value_head_state = nnx.state(value_head)
    critic_grad_norm = optax.global_norm(critic_grads)

    new_state = dataclasses.replace(
        state,
        policy_params=new_policy_params,
        value_head_params=updated_value_head_state,
        policy_opt_state=new_policy_opt_state,
        value_opt_state=new_value_opt_state,
    )
    metrics = {
        **actor_metrics,
        **critic_metrics,
        "ppo/loss": actor_total_loss + critic_total_loss,
        "ppo/grad_norm": 0.5 * (actor_grad_norm + critic_grad_norm),
        "ppo/policy_grad_norm": actor_grad_norm,
        "ppo/value_grad_norm": critic_grad_norm,
    }
    return new_state, metrics


def _ppo_update(
    rng: at.KeyArrayLike,
    state: RLTrainState,
    rollout_batch: dict[str, jax.Array],
    rollout_metrics: dict[str, jax.Array],
    config: _rl_config.RLTrainConfig,
) -> tuple[RLTrainState, dict[str, jax.Array]]:
    flat_batch: dict[str, Any] = _flatten_rollout_batch(rollout_batch)
    batch_size = flat_batch["obs"].state.shape[0]
    minibatch_size = min(config.ppo.minibatch_size, batch_size)
    minibatch_metrics: list[dict[str, jax.Array]] = []
    epochs_ran = 0
    early_stop = False

    for epoch in range(config.ppo.update_epochs):
        epochs_ran = epoch + 1
        rng, perm_rng = jax.random.split(rng)
        permutation = jax.random.permutation(perm_rng, batch_size)
        epoch_kls = []
        for start in range(0, batch_size, minibatch_size):
            mb_idx = permutation[start : start + minibatch_size]
            minibatch = jax.tree.map(lambda x: x[mb_idx], flat_batch)
            rng, mb_rng = jax.random.split(rng)
            state, mb_metrics = _update_minibatch(mb_rng, state, minibatch, config)
            minibatch_metrics.append(mb_metrics)
            epoch_kls.append(mb_metrics["ppo/approx_kl"])
        mean_epoch_kl = jnp.mean(jnp.asarray(epoch_kls)) if epoch_kls else jnp.asarray(0.0, dtype=jnp.float32)
        if mean_epoch_kl > config.ppo.target_kl:
            early_stop = True
            break

    if minibatch_metrics:
        stacked = common_utils.stack_forest(minibatch_metrics)
        reduced = jax.tree.map(jnp.mean, stacked)
    else:
        reduced = {
            "ppo/policy_loss": jnp.asarray(0.0, dtype=jnp.float32),
            "ppo/value_loss": jnp.asarray(0.0, dtype=jnp.float32),
            "ppo/entropy": jnp.asarray(0.0, dtype=jnp.float32),
            "ppo/approx_kl": jnp.asarray(0.0, dtype=jnp.float32),
            "ppo/clip_frac": jnp.asarray(0.0, dtype=jnp.float32),
            "ppo/grad_norm": jnp.asarray(0.0, dtype=jnp.float32),
        }

    new_state = dataclasses.replace(
        state,
        iteration=state.iteration + 1,
        env_steps=state.env_steps + rollout_metrics["rollout/steps"],
    )
    update_metrics = {
        **reduced,
        "ppo/epochs_ran": jnp.asarray(epochs_ran, dtype=jnp.int32),
        "ppo/early_stop": jnp.asarray(int(early_stop), dtype=jnp.int32),
    }
    return new_state, update_metrics


def _build_smoke_snapshot(metrics: dict[str, Any]) -> dict[str, Any]:
    return {k: metrics.get(k, jnp.asarray(jnp.nan, dtype=jnp.float32)) for k in _SMOKE_SNAPSHOT_KEYS}


def _build_wandb_payload(metrics: dict[str, Any]) -> dict[str, Any]:
    return {k: metrics.get(k, jnp.asarray(jnp.nan, dtype=jnp.float32)) for k in _WANDB_SCHEMA_KEYS}


def _save_state(checkpoint_manager: Any, state: RLTrainState, step: int) -> None:
    checkpoint_manager.save(
        step,
        items={
            "train_state": state,
            "params": {
                "params": {
                    "policy": state.policy_params,
                    "value_head": state.value_head_params,
                }
            },
        },
    )


def _restore_state(checkpoint_manager: Any, state: RLTrainState) -> RLTrainState:
    restored = checkpoint_manager.restore(
        items={
            "train_state": state,
            "params": {
                "params": {
                    "policy": state.policy_params,
                    "value_head": state.value_head_params,
                }
            },
        }
    )
    return restored["train_state"]


def main(config: _rl_config.RLTrainConfig) -> None:
    init_logging()
    logging.info(f"Running on: {platform.node()}")
    if config.dry_run:
        logging.warning("RL loop is running in dry_run mode with synthetic rollout data.")

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))
    rng = jax.random.key(config.seed)
    init_rng, loop_rng = jax.random.split(rng)

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    train_state = _init_train_state(init_rng, config)
    if resuming:
        train_state = _restore_state(checkpoint_manager, train_state)

    start_iteration = int(train_state.iteration)
    pbar = tqdm.tqdm(
        range(start_iteration, config.rollout.total_iterations),
        initial=start_iteration,
        total=config.rollout.total_iterations,
        dynamic_ncols=True,
    )

    infos: list[dict[str, jax.Array]] = []
    for iteration in pbar:
        loop_rng, rollout_rng, update_rng = jax.random.split(loop_rng, 3)
        rollout_batch, rollout_metrics, _ = _collect_rollout(rollout_rng, train_state, config)
        train_state, ppo_metrics = _ppo_update(update_rng, train_state, rollout_batch, rollout_metrics, config)

        info = {
            **rollout_metrics,
            **ppo_metrics,
            "train/iteration": train_state.iteration,
            "train/env_steps": train_state.env_steps,
        }
        infos.append(info)

        if iteration % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            smoke_snapshot = _build_smoke_snapshot(reduced_info)
            info_str = ", ".join(f"{k}={float(v):.4f}" for k, v in smoke_snapshot.items())
            pbar.write(f"Iteration {iteration}: {info_str}")
            wandb_payload = _build_wandb_payload(reduced_info)
            wandb.log(wandb_payload, step=iteration)
            infos = []

        if (
            (iteration % config.save_interval == 0 and iteration > start_iteration)
            or iteration == config.rollout.total_iterations - 1
        ):
            _save_state(checkpoint_manager, train_state, iteration)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()
