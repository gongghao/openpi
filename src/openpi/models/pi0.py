import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # Mixed noise modules: shared (global learnable) + task-specific (MLP from language embedding).
        self.use_mixed_noise = config.use_mixed_noise
        self.noise_mix_alpha = config.noise_mix_alpha
        self.noise_kl_weight = config.noise_kl_weight
        self.moe_num_experts = config.moe_num_experts
        self.moe_top_k = config.moe_top_k
        self.moe_balance_weight = config.moe_balance_weight
        if self.use_mixed_noise:
            self.shared_mu = nnx.Param(jnp.zeros((config.action_horizon, config.action_dim)))
            self.shared_log_sigma = nnx.Param(jnp.zeros((config.action_horizon, config.action_dim)))
            noise_hidden_dim = config.moe_hidden_dim or config.noise_head_hidden_dim or action_expert_config.width
            noise_out_dim = config.action_horizon * config.action_dim

            if self.moe_num_experts > 1:
                self.noise_router = nnx.Linear(paligemma_config.width, config.moe_num_experts, rngs=rngs)
                self.noise_expert_hidden = nnx.List([
                    nnx.Linear(paligemma_config.width, noise_hidden_dim, rngs=rngs)
                    for _ in range(config.moe_num_experts)
                ])
                self.noise_expert_mu = nnx.List([
                    nnx.Linear(noise_hidden_dim, noise_out_dim, rngs=rngs)
                    for _ in range(config.moe_num_experts)
                ])
                self.noise_expert_log_sigma = nnx.List([
                    nnx.Linear(noise_hidden_dim, noise_out_dim, rngs=rngs)
                    for _ in range(config.moe_num_experts)
                ])
            else:
                self.task_noise_hidden = nnx.Linear(paligemma_config.width, noise_hidden_dim, rngs=rngs)
                self.task_noise_mu_head = nnx.Linear(noise_hidden_dim, noise_out_dim, rngs=rngs)
                self.task_noise_log_sigma_head = nnx.Linear(noise_hidden_dim, noise_out_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond
    
    # ---- Mixed noise helpers ----

    def _get_task_emb(self, obs: _model.Observation) -> at.Float[at.Array, "b emb"]:
        """Masked mean-pool over tokenized prompt embeddings."""
        token_embs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")  # [B, L, D]
        mask = obs.tokenized_prompt_mask[..., None].astype(token_embs.dtype)   # [B, L, 1]
        return jnp.sum(token_embs * mask, axis=1) / jnp.maximum(jnp.sum(mask, axis=1), 1.0)

    def _predict_task_noise_params(
        self, task_emb: at.Float[at.Array, "b emb"]
    ) -> tuple[at.Float[at.Array, "b ah ad"], at.Float[at.Array, "b ah ad"]]:
        """Predict per-task noise mean and log-std from language embedding (single MLP fallback)."""
        h = nnx.swish(self.task_noise_hidden(task_emb))
        mu = self.task_noise_mu_head(h)
        log_sigma = self.task_noise_log_sigma_head(h)
        mu = mu.reshape(task_emb.shape[0], self.action_horizon, self.action_dim)
        log_sigma = log_sigma.reshape(task_emb.shape[0], self.action_horizon, self.action_dim)
        log_sigma = jnp.clip(log_sigma, -2.0, 2.0)
        return mu, log_sigma

    def _moe_top_k_route(
        self, task_emb: at.Float[at.Array, "b emb"]
    ) -> tuple[at.Float[at.Array, "b n"], at.Float[at.Array, "b n"]]:
        """Top-k routing: returns (sparse_weights [B, N], router_logits [B, N])."""
        logits = self.noise_router(task_emb)  # [B, N]
        n = self.moe_num_experts
        k = self.moe_top_k

        top_k_values, top_k_indices = jax.lax.top_k(logits, k)  # [B, k]
        top_k_weights = jax.nn.softmax(top_k_values, axis=-1)    # [B, k]

        # Scatter into sparse weight matrix [B, N]
        batch_size = task_emb.shape[0]
        sparse_weights = jnp.zeros((batch_size, n))
        batch_idx = jnp.arange(batch_size)[:, None]  # [B, 1]
        sparse_weights = sparse_weights.at[batch_idx, top_k_indices].set(top_k_weights)

        return sparse_weights, logits

    def _predict_task_noise_params_moe(
        self, task_emb: at.Float[at.Array, "b emb"]
    ) -> tuple[
        at.Float[at.Array, "b ah ad"],
        at.Float[at.Array, "b ah ad"],
        at.Float[at.Array, "b n"],
        at.Float[at.Array, "b n"],
        list,
    ]:
        """MoE version: returns (mu_task, log_sigma_task, sparse_weights, router_logits, per_expert_params)."""
        sparse_weights, logits = self._moe_top_k_route(task_emb)  # [B, N], [B, N]
        b = task_emb.shape[0]

        all_mu = []
        all_log_sigma = []
        for i in range(self.moe_num_experts):
            h_i = nnx.swish(self.noise_expert_hidden[i](task_emb))
            mu_i = self.noise_expert_mu[i](h_i).reshape(b, self.action_horizon, self.action_dim)
            ls_i = jnp.clip(
                self.noise_expert_log_sigma[i](h_i).reshape(b, self.action_horizon, self.action_dim),
                -2.0, 2.0,
            )
            all_mu.append(mu_i)
            all_log_sigma.append(ls_i)

        # Stack: [N, B, ah, ad] -> transpose to [B, N, ah, ad]
        stacked_mu = jnp.stack(all_mu, axis=0).transpose(1, 0, 2, 3)          # [B, N, ah, ad]
        stacked_log_sigma = jnp.stack(all_log_sigma, axis=0).transpose(1, 0, 2, 3)

        w = sparse_weights[:, :, None, None]  # [B, N, 1, 1]
        mu_task = jnp.sum(w * stacked_mu, axis=1)              # [B, ah, ad]
        log_sigma_task = jnp.sum(w * stacked_log_sigma, axis=1)  # [B, ah, ad]

        per_expert_params = list(zip(all_mu, all_log_sigma))

        return mu_task, log_sigma_task, sparse_weights, logits, per_expert_params

    def _compute_load_balance_loss(
        self,
        router_logits: at.Float[at.Array, "b n"],
    ) -> at.Float[at.Array, ""]:
        """Switch Transformer load balancing loss: L_bal = N * sum(f_i * p_i)."""
        n = self.moe_num_experts
        # f_i: fraction of samples where expert i is the top-1 choice
        top1 = jnp.argmax(router_logits, axis=-1)  # [B]
        f = jnp.mean(jax.nn.one_hot(top1, n), axis=0)  # [N]
        # p_i: average router probability for expert i
        p = jnp.mean(jax.nn.softmax(router_logits, axis=-1), axis=0)  # [N]
        return n * jnp.sum(f * p)

    def _diagonal_kl_to_standard(
        self, mu: at.Array, log_sigma: at.Array
    ) -> at.Float[at.Array, ""]:
        """KL(N(mu, sigma^2 I) || N(0, I)), averaged over all dimensions."""
        var = jnp.exp(2.0 * log_sigma)
        return 0.5 * jnp.mean(mu ** 2 + var - 1.0 - 2.0 * log_sigma)

    def _compute_mixed_noise(
        self, rng: at.KeyArrayLike, obs: _model.Observation, shape: tuple
    ) -> tuple[at.Array, at.Float[at.Array, ""]]:
        """Sample mixed noise and return (noise, kl_loss)."""
        rng_shared, rng_task = jax.random.split(rng)

        sigma_s = jnp.exp(jnp.clip(self.shared_log_sigma.value, -2.0, 2.0))
        eps_shared = self.shared_mu.value + sigma_s * jax.random.normal(rng_shared, shape)

        task_emb = self._get_task_emb(obs)

        if self.moe_num_experts > 1:
            mu_k, log_sigma_k, sparse_weights, router_logits, per_expert_params = (
                self._predict_task_noise_params_moe(task_emb)
            )
        else:
            mu_k, log_sigma_k = self._predict_task_noise_params(task_emb)

        sigma_k = jnp.exp(log_sigma_k)
        eps_task = mu_k + sigma_k * jax.random.normal(rng_task, shape)

        alpha = self.noise_mix_alpha
        noise = jnp.sqrt(1.0 - alpha ** 2) * eps_shared + alpha * eps_task

        kl_shared = self._diagonal_kl_to_standard(
            self.shared_mu.value, jnp.clip(self.shared_log_sigma.value, -2.0, 2.0)
        )

        if self.moe_num_experts > 1:
            # Weighted KL over activated experts
            kl_experts = []
            for i, (mu_i, ls_i) in enumerate(per_expert_params):
                kl_experts.append(self._diagonal_kl_to_standard(mu_i, ls_i))
            kl_per_expert = jnp.stack(kl_experts)  # [N]
            # Weight by average routing weight per expert across batch
            avg_weights = jnp.mean(sparse_weights, axis=0)  # [N]
            kl_task = jnp.sum(avg_weights * kl_per_expert)
            balance_loss = self._compute_load_balance_loss(router_logits)
            kl_loss = kl_shared + kl_task + self.moe_balance_weight * balance_loss
        else:
            kl_task = self._diagonal_kl_to_standard(mu_k, log_sigma_k)
            kl_loss = kl_shared + kl_task

        return noise, kl_loss

    # ---- End mixed noise helpers ----

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        advantages: at.Float[at.Array, " *b"] | None = None,
        rwfm_beta: float = 1.0,
        rwfm_noise_adaptive: bool = True,
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        if self.use_mixed_noise:
            noise, kl_loss = self._compute_mixed_noise(noise_rng, observation, actions.shape)
        else:
            noise = jax.random.normal(noise_rng, actions.shape)
            kl_loss = 0.0
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        if self.use_mixed_noise:
            u_t = jax.lax.stop_gradient(noise) - actions
        else:
            u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        fm_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)  # [*b, ah]

        # RWFM: reweight flow-matching loss by exponentiated advantage
        if advantages is not None:
            raw_w = jnp.exp(advantages / rwfm_beta)
            weights = raw_w / jnp.mean(raw_w)
            if rwfm_noise_adaptive:
                weights = 1.0 + (1.0 - time) * (weights - 1.0)
            fm_loss = fm_loss * weights[..., None]

        if self.use_mixed_noise:
            kl_per_element = self.noise_kl_weight * kl_loss / max(fm_loss.size, 1)
            return fm_loss + kl_per_element
        return fm_loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        value_fn=None,
        guidance_scale: float = 0.0,
        guidance_noise_adaptive: bool = True,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            if self.use_mixed_noise:
                noise, _ = self._compute_mixed_noise(
                    rng, observation, (batch_size, self.action_horizon, self.action_dim)
                )
            else:
                noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            # Classifier Guidance: steer ODE toward higher-value actions
            if value_fn is not None and guidance_scale > 0:
                def _value_for_grad(x_flat):
                    x = x_flat.reshape(batch_size, self.action_horizon, self.action_dim)
                    time_bc = jnp.broadcast_to(time, (batch_size,))
                    return jnp.sum(value_fn(x, time_bc))

                grad_v = jax.grad(_value_for_grad)(x_t.reshape(batch_size, -1))
                grad_v = grad_v.reshape(batch_size, self.action_horizon, self.action_dim)
                lam = guidance_scale * (1.0 - time) if guidance_noise_adaptive else guidance_scale
                v_t = v_t - lam * grad_v

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0