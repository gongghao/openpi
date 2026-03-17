"""GAE utilities for PPO-style policy optimization."""

import jax
import jax.numpy as jnp


def _masked_mean(values: jax.Array, mask: jax.Array | None) -> jax.Array:
    if mask is None:
        return jnp.mean(values)
    mask_f = mask.astype(values.dtype)
    denom = jnp.maximum(jnp.sum(mask_f), jnp.asarray(1.0, dtype=values.dtype))
    return jnp.sum(values * mask_f) / denom


def _safe_normalize(values: jax.Array, mask: jax.Array | None, eps: float) -> jax.Array:
    mean = _masked_mean(values, mask)
    centered = values - mean
    var = _masked_mean(jnp.square(centered), mask)
    inv_std = jax.lax.rsqrt(jnp.maximum(var, eps))
    return centered * inv_std


def compute_gae(
    rewards: jax.Array,
    dones: jax.Array,
    values: jax.Array,
    last_values: jax.Array,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """Computes generalized advantage estimates and bootstrapped returns."""

    # Align with RLinf-style usage where dones can be [T + 1, N] and
    # transition at step t uses dones[t + 1].
    if dones.shape[0] == rewards.shape[0] + 1:
        done_next = dones[1:]
    elif dones.shape[0] == rewards.shape[0]:
        done_next = dones
    else:
        raise ValueError(
            f"dones first dim should be T or T+1, got rewards={rewards.shape}, dones={dones.shape}"
        )

    next_values = jnp.concatenate([values[1:], last_values[None, :]], axis=0)
    not_done = 1.0 - done_next.astype(values.dtype)
    deltas = rewards + gamma * next_values * not_done - values

    def scan_step(
        gae_next: jax.Array,
        inputs: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        delta_t, not_done_t = inputs
        gae_t = delta_t + gamma * gae_lambda * not_done_t * gae_next
        return gae_t, gae_t

    _, advantages_rev = jax.lax.scan(
        scan_step,
        jnp.zeros_like(last_values),
        (deltas[::-1], not_done[::-1]),
    )
    advantages = advantages_rev[::-1]
    returns = advantages + values
    return advantages, returns


def maybe_normalize_advantages(
    advantages: jax.Array,
    *,
    normalize: bool = True,
    loss_mask: jax.Array | None = None,
    eps: float = 1e-8,
) -> jax.Array:
    """Optionally normalizes advantages to zero mean and unit variance."""

    if not normalize:
        return advantages

    return _safe_normalize(advantages, loss_mask, eps)
