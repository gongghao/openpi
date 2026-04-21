from collections import OrderedDict
from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import pathlib
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
from openpi.training.rollout_manifest import RolloutManifest, build_manifest, load_manifest_from_npz
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def _resolve_rollout_dirs(data_config: _config.DataConfig) -> tuple[str, ...]:
    """Return the ordered tuple of rollout directories defined on ``data_config``.

    Supports the legacy single ``rollout_dir`` field plus (for B5 / B8) an
    optional ``rollout_dirs`` collection. The concatenation order is the canon
    used to build :class:`RolloutManifest` -- ``rollout_dir`` first (if set),
    then each entry in ``rollout_dirs``.
    """
    dirs: list[str] = []
    primary = getattr(data_config, "rollout_dir", None)
    if primary:
        dirs.append(str(primary))
    extra = getattr(data_config, "rollout_dirs", None) or ()
    for d in extra:
        if d:
            dirs.append(str(d))
    return tuple(dirs)


class RolloutNpzDataset:
    """Dataset that reads rollout episode .npz archives and yields per-frame
    samples in the same format as the LeRobot LIBERO dataset.

    Each sample is a dict with:
        ``image``, ``wrist_image``, ``state``, ``actions``, ``prompt``
    which matches the keys expected by the LIBERO ``RepackTransform``.

    The flattened frame order is defined by :class:`RolloutManifest` and must
    exactly match the manifest used by ``precompute_advantages.py`` so that
    ``advantages.npz`` is paired with the same dataset (enforced by
    :class:`AdvantageInjectorDataset`).
    """

    def __init__(
        self,
        rollout_dir: str | Sequence[str],
        action_horizon: int = 50,
        *,
        manifest: RolloutManifest | None = None,
        cache_size: int = 32,
    ):
        self._manifest = manifest if manifest is not None else build_manifest(rollout_dir)
        self._action_horizon = action_horizon
        # A2: LRU over materialized episode arrays. The collected rollouts are
        # compressed ``.npz`` files, so ``mmap_mode`` cannot avoid decompression;
        # we keep decompression cheap by holding only ``cache_size`` episodes in
        # RAM at a time. DataLoader workers each get their own cache via
        # pickling -- an empty one to start with.
        self._cache_size = max(1, int(cache_size))
        self._cache: OrderedDict[int, dict] = OrderedDict()

        logging.info(
            "RolloutNpzDataset: %d episodes, %d frames, action_horizon=%d, "
            "manifest_sha=%s, cache_size=%d",
            self._manifest.num_episodes,
            self._manifest.num_frames,
            action_horizon,
            self._manifest.manifest_sha[:12],
            self._cache_size,
        )

    @property
    def manifest(self) -> RolloutManifest:
        return self._manifest

    def _locate(self, idx: int) -> tuple[int, int]:
        cum = self._manifest.cum_offsets
        ep_idx = int(np.searchsorted(cum, idx, side="right") - 1)
        return ep_idx, idx - int(cum[ep_idx])

    def _load_episode(self, ep_idx: int) -> dict:
        path = self._manifest.paths[ep_idx]
        handle = np.load(path, mmap_mode="r", allow_pickle=True)
        try:
            return {k: handle[k] for k in handle.files}
        finally:
            handle.close()

    def _get_episode(self, ep_idx: int) -> dict:
        cached = self._cache.get(ep_idx)
        if cached is not None:
            self._cache.move_to_end(ep_idx)
            return cached
        ep = self._load_episode(ep_idx)
        self._cache[ep_idx] = ep
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return ep

    def __getitem__(self, index: SupportsIndex) -> dict:
        idx = index.__index__() if hasattr(index, "__index__") else int(index)
        ep_idx, t = self._locate(idx)
        ep = self._get_episode(ep_idx)

        T = len(ep["actions"])
        action_chunk = np.stack(
            [ep["actions"][min(t + dt, T - 1)] for dt in range(self._action_horizon)]
        ).astype(np.float32)

        return {
            "image": np.asarray(ep["images"][t]),
            "wrist_image": np.asarray(ep["wrist_images"][t]),
            "state": np.asarray(ep["states"][t], dtype=np.float32),
            "actions": action_chunk,
            "prompt": str(ep["task_description"]),
        }

    def __len__(self) -> int:
        return self._manifest.num_frames

    def __getstate__(self) -> dict:
        # Reset the LRU when the dataset is pickled to a DataLoader worker so
        # each process opens its own handles lazily.
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        return state


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    rollout_dirs = _resolve_rollout_dirs(data_config)
    if rollout_dirs:
        return RolloutNpzDataset(rollout_dirs, action_horizon=action_horizon)

    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    is_real_lerobot = data_config.repo_id not in (None, "fake")
    if is_real_lerobot and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
    if data_config.norm_stats is not None:
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
        advantages_path=getattr(config, "rwfm_advantages_path", None),
        per_enabled=getattr(config, "rwfm_per_enabled", False),
        per_alpha=getattr(config, "rwfm_per_alpha", 1.0),
        per_beta=getattr(config, "rwfm_per_beta", 1.0),
        per_eps=getattr(config, "rwfm_per_eps", 1e-6),
        rwfm_beta=getattr(config, "rwfm_beta", 1.0),
        rwfm_adv_clip=getattr(config, "rwfm_adv_clip", 3.0),
        rwfm_source_ratios=tuple(getattr(config, "rwfm_source_ratios", ()) or ()),
        rwfm_source_dirs=tuple(getattr(config, "rwfm_source_dirs", ()) or ()),
    )


def _build_source_weights(
    source_dirs: Sequence[str],
    source_ratios: Sequence[float],
    *,
    expected_num_frames: int,
) -> np.ndarray:
    """B9: per-frame sampling weights that realise the requested source ratios.

    Builds one :class:`RolloutManifest` per source to count frames, then
    assigns every frame in source ``i`` the weight
    ``ratios[i] / max(num_frames_i, 1)``. Summing over all frames in a
    source gives exactly ``ratios[i]``, so the
    :class:`torch.utils.data.WeightedRandomSampler` that consumes these
    weights draws each source with expected proportion ``ratios[i]``.

    ``expected_num_frames`` is the concatenated manifest length used by the
    wrapping :class:`RolloutNpzDataset`; we hard-fail on mismatch because a
    divergent total invalidates the per-frame index mapping that
    :class:`AdvantageInjectorDataset` relies on.
    """
    if len(source_dirs) != len(source_ratios):
        raise ValueError(
            f"rwfm_source_dirs ({len(source_dirs)}) and rwfm_source_ratios "
            f"({len(source_ratios)}) must have the same length."
        )
    if not source_dirs:
        raise ValueError("rwfm_source_dirs must contain at least one directory.")

    frames_per_source: list[int] = []
    for d in source_dirs:
        manifest = build_manifest(d)
        frames_per_source.append(int(manifest.num_frames))

    total_frames = int(sum(frames_per_source))
    if total_frames != expected_num_frames:
        raise ValueError(
            "rwfm_source_dirs frame counts do not match the concatenated "
            f"RolloutNpzDataset: sum={total_frames} vs dataset={expected_num_frames}. "
            "Ensure rwfm_source_dirs == [data.rollout_dir, *data.rollout_dirs]."
        )

    ratios = np.asarray(source_ratios, dtype=np.float64)
    if float(ratios.sum()) <= 0:
        raise ValueError(f"rwfm_source_ratios must have positive sum; got {source_ratios}")
    ratios = ratios / ratios.sum()

    weights = np.empty(total_frames, dtype=np.float64)
    offset = 0
    for n, r in zip(frames_per_source, ratios.tolist(), strict=True):
        per_frame = float(r) / max(n, 1)
        weights[offset : offset + n] = per_frame
        offset += n

    logging.info(
        "B9 source-weighted sampling: sources=%d total_frames=%d ratios=%s frames=%s",
        len(source_dirs),
        total_frames,
        tuple(round(float(r), 4) for r in ratios.tolist()),
        tuple(frames_per_source),
    )
    return weights


def _compute_per_priorities(
    advantages: np.ndarray,
    *,
    rwfm_beta: float,
    rwfm_adv_clip: float,
    per_alpha: float,
    per_eps: float,
) -> np.ndarray:
    """B7: compute unnormalized priorities for ``WeightedRandomSampler``.

    We approximate the per-batch RWFM weighting with a global variant:
    clip the raw advantage to the A4 range, divide by ``rwfm_beta``, and
    exponentiate. The ``per_alpha`` exponent controls sharpness --
    ``alpha=1`` gives plain RWFM-proportional priorities, ``alpha<1``
    smooths them toward uniform, ``alpha=0`` recovers uniform sampling.
    """
    adv = np.asarray(advantages, dtype=np.float64)
    if rwfm_adv_clip > 0.0:
        adv = np.clip(adv, -rwfm_adv_clip * rwfm_beta, rwfm_adv_clip * rwfm_beta)
    weights = np.exp(adv / max(rwfm_beta, 1e-8))
    weights = np.maximum(weights, 0.0) + float(per_eps)
    if per_alpha != 1.0:
        weights = np.power(weights, float(per_alpha))
    return weights


def _compute_per_importance_weights(
    priorities: np.ndarray,
    *,
    per_beta: float,
) -> np.ndarray:
    """B7: IS correction ``((1 / (N * p_i)) / max_is) ** per_beta``.

    Normalizing by ``max(is)`` keeps gradients from exploding on rare
    high-priority samples, mirroring the original PER paper.
    """
    p = np.asarray(priorities, dtype=np.float64)
    p = p / max(p.sum(), 1e-12)
    n = len(p)
    is_weights = 1.0 / (n * np.maximum(p, 1e-12))
    is_weights = is_weights / max(is_weights.max(), 1e-12)
    if per_beta != 1.0:
        is_weights = np.power(is_weights, float(per_beta))
    return is_weights.astype(np.float32)


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
    advantages_path: str | None = None,
    per_enabled: bool = False,
    per_alpha: float = 1.0,
    per_beta: float = 1.0,
    per_eps: float = 1e-6,
    rwfm_beta: float = 1.0,
    rwfm_adv_clip: float = 3.0,
    rwfm_source_ratios: Sequence[float] = (),
    rwfm_source_dirs: Sequence[str] = (),
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    raw_dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(raw_dataset, data_config, skip_norm_stats=skip_norm_stats)

    # ---- Inject precomputed advantages at the Dataset level (shuffle-safe) ----
    per_sampler: torch.utils.data.Sampler | None = None
    if advantages_path is not None:
        adv_data = np.load(advantages_path, allow_pickle=True)
        dataset_manifest = getattr(raw_dataset, "manifest", None)
        advantages_array = np.asarray(adv_data["advantages"], dtype=np.float32)

        importance_weights: np.ndarray | None = None
        if per_enabled:
            # B7: offline prioritized sampling.
            if framework == "pytorch" and torch.distributed.is_initialized():
                logging.warning(
                    "rwfm_per_enabled=True is not supported under torch.distributed; "
                    "falling back to uniform DistributedSampler and unit IS weights."
                )
            else:
                priorities = _compute_per_priorities(
                    advantages_array,
                    rwfm_beta=rwfm_beta,
                    rwfm_adv_clip=rwfm_adv_clip,
                    per_alpha=per_alpha,
                    per_eps=per_eps,
                )
                importance_weights = _compute_per_importance_weights(priorities, per_beta=per_beta)
                per_generator = torch.Generator()
                per_generator.manual_seed(int(seed))
                per_sampler = torch.utils.data.WeightedRandomSampler(
                    weights=torch.as_tensor(priorities, dtype=torch.double),
                    num_samples=len(dataset),
                    replacement=True,
                    generator=per_generator,
                )
                logging.info(
                    "B7 prioritized sampling: N=%d alpha=%.3f per_beta=%.3f min_p=%.3e max_p=%.3e",
                    len(priorities),
                    per_alpha,
                    per_beta,
                    float(priorities.min() / max(priorities.sum(), 1e-12)),
                    float(priorities.max() / max(priorities.sum(), 1e-12)),
                )

        dataset = AdvantageInjectorDataset(
            dataset,
            advantages_array,
            expected_manifest=dataset_manifest,
            advantages_manifest=load_manifest_from_npz(adv_data),
            advantages_path=advantages_path,
            importance_weights=importance_weights,
        )
        logging.info(
            "Injected advantages from %s (%d values, PER=%s)",
            advantages_path,
            len(advantages_array),
            "on" if importance_weights is not None else "off",
        )

    # B9: three-source weighted sampler. Built from per-source manifest
    # counts so it is robust to changes in ``data.rollout_dir`` /
    # ``data.rollout_dirs`` between iterations, and overrides B7 PER when
    # both are configured (we already pass advantages untouched).
    b9_sampler: torch.utils.data.Sampler | None = None
    if rwfm_source_ratios and rwfm_source_dirs:
        if per_sampler is not None:
            logging.warning(
                "Both rwfm_per_enabled (B7) and rwfm_source_ratios (B9) are active; "
                "B9 weighted sampler wins and B7 priorities are dropped."
            )
            per_sampler = None
        b9_weights = _build_source_weights(
            rwfm_source_dirs,
            rwfm_source_ratios,
            expected_num_frames=len(dataset),
        )
        b9_generator = torch.Generator()
        b9_generator.manual_seed(int(seed) + 1_000_003)
        b9_sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.as_tensor(b9_weights, dtype=torch.double),
            num_samples=len(dataset),
            replacement=True,
            generator=b9_generator,
        )

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler: torch.utils.data.Sampler | None = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
            if b9_sampler is not None:
                logging.warning(
                    "rwfm_source_ratios (B9) is not supported under torch.distributed; "
                    "falling back to uniform DistributedSampler."
                )
                b9_sampler = None
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    # B9 weighted sampler takes precedence over B7 PER (both already
    # reconciled above). Under DDP the distributed sampler wins.
    if b9_sampler is not None and sampler is None:
        sampler = b9_sampler
    elif per_sampler is not None and sampler is None:
        sampler = per_sampler

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class AdvantageInjectorDataset(Dataset):
    """Wraps a dataset and injects precomputed advantage values per sample.

    The advantage value travels with each sample through shuffling / batching,
    so the pairing is always correct regardless of data loader ordering.

    Strictness (A1): the advantage array must have exactly the same length as
    the wrapped dataset. When a :class:`RolloutManifest` is available on both
    sides, the ``manifest_sha`` must also match -- otherwise we refuse to run
    because the rollout directory has diverged from the one
    ``precompute_advantages.py`` scored.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        advantages: np.ndarray,
        *,
        expected_manifest: RolloutManifest | None = None,
        advantages_manifest: RolloutManifest | None = None,
        advantages_path: str | None = None,
        importance_weights: np.ndarray | None = None,
    ):
        self._dataset = base_dataset
        self._advantages = np.asarray(advantages, dtype=np.float32)
        self._importance_weights = (
            np.asarray(importance_weights, dtype=np.float32) if importance_weights is not None else None
        )

        ds_len = len(self._dataset)
        adv_len = len(self._advantages)
        if adv_len != ds_len:
            raise ValueError(
                f"Advantage array length ({adv_len}) != dataset length ({ds_len}). "
                f"Re-run scripts/precompute_advantages.py against the current rollout "
                f"directory so the two stay aligned"
                + (f" (advantages file: {advantages_path})" if advantages_path else "")
                + "."
            )
        if self._importance_weights is not None and len(self._importance_weights) != ds_len:
            raise ValueError(
                f"importance_weights length ({len(self._importance_weights)}) != dataset length ({ds_len})."
            )

        if expected_manifest is not None and advantages_manifest is not None:
            if expected_manifest.manifest_sha != advantages_manifest.manifest_sha:
                raise ValueError(
                    "manifest_sha mismatch between rollout dataset and advantages file: "
                    f"dataset={expected_manifest.manifest_sha[:12]} "
                    f"advantages={advantages_manifest.manifest_sha[:12]}"
                    + (f" (file: {advantages_path})" if advantages_path else "")
                    + ". The set or order of episode_*.npz files has changed; re-run "
                    "scripts/precompute_advantages.py."
                )

    def __getitem__(self, index: SupportsIndex):
        sample = self._dataset[index]
        idx = index.__index__() if hasattr(index, "__index__") else int(index)
        sample["advantages"] = self._advantages[idx]
        if self._importance_weights is not None:
            sample["importance_weights"] = self._importance_weights[idx]
        return sample

    def __len__(self) -> int:
        return len(self._dataset)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            obs = _model.Observation.from_dict(batch)
            actions = batch["actions"]
            adv = batch.get("advantages", None)
            if adv is not None:
                adv = jnp.asarray(adv)
            is_w = batch.get("importance_weights", None)
            if is_w is not None:
                is_w = jnp.asarray(is_w)
            yield obs, actions, adv, is_w
