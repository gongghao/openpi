"""RL training config scaffold for JAX-native online fine-tuning."""

import dataclasses
import difflib
import pathlib

import tyro

import openpi.models.pi0_config as pi0_config


@dataclasses.dataclass(frozen=True)
class LiberoEnvConfig:
    """Environment config for LIBERO rollout."""

    suite: str = "libero_goal"
    task: str = "LIBERO_GOAL_OpenTheDoor"
    num_envs: int = 8
    max_episode_steps: int = 300
    action_repeat: int = 1
    action_chunk_size: int = 1
    env_action_dim: int = 7
    image_size: int = 224
    seed: int = 42
    use_right_wrist: bool = False
    max_token_len: int = 48


@dataclasses.dataclass(frozen=True)
class PPOConfig:
    """PPO hyperparameters used by the RL loop."""

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2
    clip_ratio_c: float | None = None
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    target_kl: float = 0.02
    update_epochs: int = 4
    minibatch_size: int = 256
    max_grad_norm: float = 1.0
    learning_rate: float = 3e-5
    normalize_advantage: bool = True
    value_clip_epsilon: float | None = None
    huber_delta: float | None = None


@dataclasses.dataclass(frozen=True)
class RolloutConfig:
    """Online rollout sampling settings."""

    rollout_horizon: int = 128
    total_iterations: int = 5_000
    warmup_iterations: int = 10
    eval_interval: int = 50


@dataclasses.dataclass(frozen=True)
class RLTrainConfig:
    """Top-level RL train config."""

    name: tyro.conf.Suppress[str]
    exp_name: str = tyro.MISSING
    project_name: str = "openpi-rl"
    checkpoint_base_dir: str = "./checkpoints_rl"
    seed: int = 42
    model: pi0_config.Pi0Config = dataclasses.field(
        default_factory=lambda: pi0_config.Pi0Config(
            action_dim=32,
            action_horizon=10,
            flow_noise_enabled=True,
        )
    )
    policy_params_path: str = "gs://openpi-assets/checkpoints/pi0_base/params"
    policy_denoise_steps: int = 1

    rollout: RolloutConfig = dataclasses.field(default_factory=RolloutConfig)
    ppo: PPOConfig = dataclasses.field(default_factory=PPOConfig)
    env: LiberoEnvConfig = dataclasses.field(default_factory=LiberoEnvConfig)

    # Keep this true in scaffold phase while env/model adapters are being built.
    dry_run: bool = True

    log_interval: int = 10
    save_interval: int = 100
    keep_period: int | None = 500
    overwrite: bool = False
    resume: bool = False
    wandb_enabled: bool = True

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def steps_per_iteration(self) -> int:
        return self.env.num_envs * self.rollout.rollout_horizon

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")
        if self.ppo.minibatch_size <= 0:
            raise ValueError("ppo.minibatch_size must be > 0.")
        if self.ppo.update_epochs <= 0:
            raise ValueError("ppo.update_epochs must be > 0.")
        if self.ppo.clip_ratio_low < 0 or self.ppo.clip_ratio_high < 0:
            raise ValueError("ppo.clip_ratio_low/high must be >= 0.")
        if self.ppo.clip_ratio_c is not None and self.ppo.clip_ratio_c <= 1.0:
            raise ValueError("ppo.clip_ratio_c must be > 1.0 when set.")
        if self.steps_per_iteration <= 0:
            raise ValueError("steps_per_iteration must be > 0.")
        if self.env.action_chunk_size <= 0:
            raise ValueError("env.action_chunk_size must be > 0.")
        if self.env.env_action_dim <= 0:
            raise ValueError("env.env_action_dim must be > 0.")
        if self.env.image_size <= 0:
            raise ValueError("env.image_size must be > 0.")
        if self.env.max_token_len <= 0:
            raise ValueError("env.max_token_len must be > 0.")
        if self.policy_denoise_steps <= 0:
            raise ValueError("policy_denoise_steps must be > 0.")


_CONFIGS = [
    RLTrainConfig(
        name="rl_debug",
        exp_name="debug",
        dry_run=True,
        model=pi0_config.Pi0Config(
            paligemma_variant="dummy",
            action_expert_variant="dummy",
            action_dim=8,
            action_horizon=2,
            flow_noise_enabled=True,
        ),
        policy_params_path="",
        rollout=RolloutConfig(
            rollout_horizon=16,
            total_iterations=5,
            warmup_iterations=1,
            eval_interval=2,
        ),
        env=LiberoEnvConfig(
            num_envs=2,
            max_episode_steps=50,
        ),
        wandb_enabled=False,
        overwrite=True,
    ),
    RLTrainConfig(
        name="pi0_libero_rl",
        exp_name="pi0_libero_rl",
        dry_run=False,
        model=pi0_config.Pi0Config(
            action_dim=32,
            action_horizon=10,
            flow_noise_enabled=True,
        ),
        policy_params_path="gs://openpi-assets/checkpoints/pi0_base/params",
        env=LiberoEnvConfig(
            action_chunk_size=10,
            env_action_dim=7,
        ),
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> RLTrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> RLTrainConfig:
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")
    return _CONFIGS_DICT[config_name]
