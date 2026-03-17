"""PPO math helpers used by the RL loop."""

import jax
import jax.numpy as jnp

_LOG_2PI = jnp.log(2.0 * jnp.pi)


def policy_mean(
    policy_params: dict[str, jax.Array],
    obs: jax.Array,
) -> jax.Array:
    return obs @ policy_params["w"] + policy_params["b"]


def value_prediction(
    value_params: dict[str, jax.Array],
    obs: jax.Array,
) -> jax.Array:
    return jnp.squeeze(obs @ value_params["w"] + value_params["b"], axis=-1)


def gaussian_log_prob(
    actions: jax.Array,
    mean: jax.Array,
    log_std: jax.Array,
) -> jax.Array:
    centered = (actions - mean) / jnp.exp(log_std)
    log_prob_per_dim = -0.5 * (jnp.square(centered) + 2.0 * log_std + _LOG_2PI)
    return jnp.sum(log_prob_per_dim, axis=-1)


def gaussian_entropy(log_std: jax.Array) -> jax.Array:
    entropy_per_dim = log_std + 0.5 * (1.0 + _LOG_2PI)
    return jnp.sum(entropy_per_dim)


def _masked_mean(values: jax.Array, mask: jax.Array | None) -> jax.Array:
    if mask is None:
        return jnp.mean(values)
    mask_f = _align_mask(mask, values).astype(values.dtype)
    denom = jnp.maximum(jnp.sum(mask_f), jnp.asarray(1.0, dtype=values.dtype))
    return jnp.sum(values * mask_f) / denom


def _align_mask(mask: jax.Array, target: jax.Array) -> jax.Array:
    out = mask
    while out.ndim < target.ndim:
        out = out[..., None]
    if out.ndim > target.ndim:
        extra_dims = out.ndim - target.ndim
        squeeze_axes = tuple(i for i in range(out.ndim - 1, out.ndim - extra_dims - 1, -1) if out.shape[i] == 1)
        if squeeze_axes:
            out = jnp.squeeze(out, axis=squeeze_axes)
    return jnp.broadcast_to(out, target.shape)


def _broadcast_mask(mask: jax.Array | None, target: jax.Array) -> jax.Array | None:
    if mask is None:
        return None
    return _align_mask(mask, target)


def ppo_actor_metrics(
    *,
    new_logprob: jax.Array,
    old_logprob: jax.Array,
    advantages: jax.Array,
    clip_ratio_low: float,
    clip_ratio_high: float,
    entropy: jax.Array,
    entropy_coef: float,
    loss_mask: jax.Array | None = None,
    clip_ratio_c: float | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Computes PPO clipped actor objective and diagnostics."""

    log_ratio = new_logprob - old_logprob
    ratio = jnp.exp(log_ratio)
    clipped_ratio = jnp.clip(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high)
    policy_loss1 = -advantages * ratio
    policy_loss2 = -advantages * clipped_ratio
    policy_loss = jnp.maximum(policy_loss1, policy_loss2)
    dual_clipped_ratio = jnp.zeros_like(ratio)
    if clip_ratio_c is not None:
        if clip_ratio_c <= 1.0:
            raise ValueError(f"clip_ratio_c must be > 1.0, got {clip_ratio_c}")
        policy_loss3 = jnp.sign(advantages) * clip_ratio_c * advantages
        dual_mask = policy_loss3 < policy_loss
        dual_clipped_ratio = jnp.where(dual_mask, ratio, 0.0)
        policy_loss = jnp.minimum(policy_loss, policy_loss3)
    policy_loss = _masked_mean(policy_loss, loss_mask)
    entropy_bonus = entropy_coef * entropy
    total_loss = policy_loss - entropy_bonus

    approx_kl = _masked_mean(old_logprob - new_logprob, loss_mask)
    clip_mask = policy_loss1 < policy_loss2
    clip_frac = _masked_mean(clip_mask.astype(jnp.float32), _broadcast_mask(loss_mask, clip_mask))

    metrics = {
        "ppo/policy_loss": policy_loss,
        "ppo/entropy": entropy,
        "ppo/entropy_bonus": entropy_bonus,
        "ppo/approx_kl": approx_kl,
        "ppo/clip_frac": clip_frac,
        "ppo/ratio": _masked_mean(ratio, _broadcast_mask(loss_mask, ratio)),
        "ppo/clipped_ratio": _masked_mean(clipped_ratio, _broadcast_mask(loss_mask, clipped_ratio)),
        "ppo/dual_clipped_ratio": _masked_mean(
            dual_clipped_ratio, _broadcast_mask(loss_mask, dual_clipped_ratio)
        ),
    }
    return total_loss, metrics


def _huber_loss(error: jax.Array, delta: float) -> jax.Array:
    abs_error = jnp.abs(error)
    quadratic = jnp.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return 0.5 * jnp.square(quadratic) + delta * linear


def ppo_value_loss(
    *,
    new_values: jax.Array,
    old_values: jax.Array,
    returns: jax.Array,
    value_coef: float,
    value_clip_epsilon: float | None = None,
    huber_delta: float | None = None,
    loss_mask: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Computes PPO critic loss with optional value clipping."""

    if value_clip_epsilon is None:
        value_pred = new_values
    else:
        value_delta = new_values - old_values
        value_pred = old_values + jnp.clip(value_delta, -value_clip_epsilon, value_clip_epsilon)

    if huber_delta is None:
        raw_value_loss = jnp.square(new_values - returns)
        clipped_value_loss = jnp.square(value_pred - returns)
    else:
        raw_value_loss = _huber_loss(returns - new_values, huber_delta)
        clipped_value_loss = _huber_loss(returns - value_pred, huber_delta)

    value_loss = _masked_mean(jnp.maximum(raw_value_loss, clipped_value_loss), loss_mask)
    total_loss = value_coef * value_loss
    value_clip_ratio = (
        jnp.asarray(0.0, dtype=jnp.float32)
        if value_clip_epsilon is None
        else jnp.mean((jnp.abs(value_pred - old_values) > value_clip_epsilon).astype(jnp.float32))
    )

    if loss_mask is None:
        masked_returns = returns
        masked_values = new_values
    else:
        mask_bool = _align_mask(loss_mask, returns).astype(jnp.bool_)
        masked_returns = jnp.where(mask_bool, returns, jnp.nan)
        masked_values = jnp.where(mask_bool, new_values, jnp.nan)
    var_returns = jnp.nanvar(masked_returns)
    var_diff = jnp.nanvar(masked_returns - masked_values)
    explained_variance = jnp.where(var_returns > 0, 1.0 - (var_diff / (var_returns + 1e-8)), jnp.asarray(0.0))

    metrics = {
        "ppo/value_loss": value_loss,
        "ppo/value_loss_scaled": total_loss,
        "ppo/value_clip_ratio": value_clip_ratio,
        "ppo/explained_variance": explained_variance,
    }
    return total_loss, metrics
