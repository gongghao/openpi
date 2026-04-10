"""Distributional value functions for RWFM advantage estimation and Classifier Guidance.

StateValueFunction — V(s): used to compute advantages A(s,a) = G_t - V(s_t).
ActionConditionedValueFunction — V(s, x_t, t): used at inference time to provide
    gradient guidance that steers the ODE sampling toward higher-value actions.

Both follow the RECAP-style distributional value function design: the output is a
categorical distribution over B bins spanning (-1, 0), trained with cross-entropy
against discretised normalised Monte-Carlo returns.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.shared import array_typing as at


def _posemb_sincos_1d(pos, dim, min_period=4e-3, max_period=4.0):
    """Sine-cosine positional embedding for scalar positions."""
    fraction = jnp.linspace(0.0, 1.0, dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid = pos[..., None] / period * 2 * jnp.pi
    return jnp.concatenate([jnp.sin(sinusoid), jnp.cos(sinusoid)], axis=-1)


class StateValueFunction(nnx.Module):
    """V(image_emb, state, lang_emb) -> distributional value over B bins."""

    def __init__(
        self,
        *,
        image_dim: int = 1152,
        state_dim: int = 32,
        lang_dim: int = 2048,
        hidden_dim: int = 256,
        num_bins: int = 101,
        rngs: nnx.Rngs,
    ):
        self.num_bins = num_bins
        self.image_proj = nnx.Linear(image_dim, hidden_dim, rngs=rngs)
        self.state_proj = nnx.Linear(state_dim, hidden_dim, rngs=rngs)
        self.lang_proj = nnx.Linear(lang_dim, hidden_dim, rngs=rngs)
        self.mlp1 = nnx.Linear(hidden_dim * 3, hidden_dim, rngs=rngs)
        self.mlp2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.head = nnx.Linear(hidden_dim, num_bins, rngs=rngs)

    def __call__(
        self,
        image_emb: at.Float[at.Array, "b d_img"],
        state: at.Float[at.Array, "b d_s"],
        lang_emb: at.Float[at.Array, "b d_l"],
    ) -> at.Float[at.Array, "b bins"]:
        h_img = nnx.relu(self.image_proj(image_emb))
        h_st = nnx.relu(self.state_proj(state))
        h_ln = nnx.relu(self.lang_proj(lang_emb))
        h = jnp.concatenate([h_img, h_st, h_ln], axis=-1)
        h = nnx.relu(self.mlp1(h))
        h = nnx.relu(self.mlp2(h))
        return self.head(h)

    def predict_value(self, image_emb, state, lang_emb) -> at.Float[at.Array, " b"]:
        logits = self(image_emb, state, lang_emb)
        probs = jax.nn.softmax(logits, axis=-1)
        # NNX modules must not hold raw Array attributes (breaks nnx.state(..., nnx.Param)).
        bin_values = jnp.linspace(-1.0, 0.0, self.num_bins)
        return jnp.sum(probs * bin_values, axis=-1)


class ActionConditionedValueFunction(nnx.Module):
    """V(image_emb, state, noisy_actions, timestep) -> scalar value per sample.

    Used for Classifier Guidance: the gradient d V / d x_t steers the ODE sampler.
    """

    def __init__(
        self,
        *,
        image_dim: int = 1152,
        state_dim: int = 32,
        action_dim: int = 32,
        action_horizon: int = 50,
        hidden_dim: int = 256,
        rngs: nnx.Rngs,
    ):
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        flat_action_dim = action_dim * action_horizon

        self.image_proj = nnx.Linear(image_dim, hidden_dim, rngs=rngs)
        self.state_proj = nnx.Linear(state_dim, hidden_dim, rngs=rngs)
        self.action_proj = nnx.Linear(flat_action_dim, hidden_dim, rngs=rngs)
        self.time_proj = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.mlp1 = nnx.Linear(hidden_dim * 4, hidden_dim, rngs=rngs)
        self.mlp2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.head = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(
        self,
        image_emb: at.Float[at.Array, "b d_img"],
        state: at.Float[at.Array, "b d_s"],
        noisy_actions: at.Float[at.Array, "b ah ad"],
        timestep: at.Float[at.Array, " b"],
    ) -> at.Float[at.Array, " b"]:
        h_img = nnx.relu(self.image_proj(image_emb))
        h_st = nnx.relu(self.state_proj(state))
        flat_a = noisy_actions.reshape(noisy_actions.shape[0], -1)
        h_a = nnx.relu(self.action_proj(flat_a))
        time_emb = _posemb_sincos_1d(timestep, self.time_proj.in_features)
        h_t = nnx.relu(self.time_proj(time_emb))
        h = jnp.concatenate([h_img, h_st, h_a, h_t], axis=-1)
        h = nnx.relu(self.mlp1(h))
        h = nnx.relu(self.mlp2(h))
        return self.head(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def state_vf_loss(
    vf: StateValueFunction,
    image_emb: at.Float[at.Array, "b d_img"],
    state: at.Float[at.Array, "b d_s"],
    lang_emb: at.Float[at.Array, "b d_l"],
    target_returns: at.Float[at.Array, " b"],
) -> at.Float[at.Array, ""]:
    """Cross-entropy loss for the distributional state value function."""
    logits = vf(image_emb, state, lang_emb)
    bin_idx = jnp.round((target_returns + 1.0) * (vf.num_bins - 1)).astype(jnp.int32)
    bin_idx = jnp.clip(bin_idx, 0, vf.num_bins - 1)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    loss = -log_probs[jnp.arange(bin_idx.shape[0]), bin_idx]
    return jnp.mean(loss)


def action_conditioned_vf_loss(
    vf: ActionConditionedValueFunction,
    image_emb: at.Float[at.Array, "b d_img"],
    state: at.Float[at.Array, "b d_s"],
    noisy_actions: at.Float[at.Array, "b ah ad"],
    timestep: at.Float[at.Array, " b"],
    target_returns: at.Float[at.Array, " b"],
) -> at.Float[at.Array, ""]:
    """MSE regression loss for the action-conditioned value function."""
    pred = vf(image_emb, state, noisy_actions, timestep)
    return jnp.mean(jnp.square(pred - target_returns))
