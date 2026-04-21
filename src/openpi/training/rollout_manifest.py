"""Canonical manifest over rollout ``.npz`` episode files.

Both :class:`openpi.training.data_loader.RolloutNpzDataset` and the offline
scripts (``precompute_advantages.py``, ``train_value_function.py``) must agree
on which episodes are in the dataset and in which order their frames are
flattened. Without a shared contract the precomputed ``advantages.npz`` /
``encoder_features.npz`` can silently get out of sync with the dataset the
trainer actually iterates over.

The manifest is defined purely by the ordered list of episode paths:

* ``paths``        -- the ordered, flattened list of ``episode_*.npz`` files
* ``ep_lengths``   -- ``len(states)`` for each episode
* ``cum_offsets``  -- prefix sums (length ``E+1``) over ``ep_lengths``
* ``manifest_sha`` -- ``sha256("\\n".join(paths))``

Multiple rollout directories are supported (needed for B5 iteration / B8 demo
union): we concatenate each directory's
``sorted(glob("**/episode_*.npz", recursive=True))`` in the given order.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import glob as _glob
import hashlib
import pathlib

import numpy as np


@dataclasses.dataclass(frozen=True)
class RolloutManifest:
    """Immutable description of the flattened frame index of a rollout dataset."""

    rollout_dirs: tuple[str, ...]
    paths: tuple[str, ...]
    ep_lengths: np.ndarray  # int64 [E]
    cum_offsets: np.ndarray  # int64 [E+1]
    manifest_sha: str

    @property
    def num_episodes(self) -> int:
        return len(self.paths)

    @property
    def num_frames(self) -> int:
        return int(self.cum_offsets[-1])


def _list_episode_paths(rollout_dir: str) -> list[str]:
    pattern = str(pathlib.Path(rollout_dir) / "**" / "episode_*.npz")
    return sorted(_glob.glob(pattern, recursive=True))


def _sha256_of_paths(paths: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(paths).encode("utf-8")).hexdigest()


def _episode_length(path: str) -> int:
    with np.load(path, allow_pickle=True) as data:
        return int(data["states"].shape[0])


def _ep_lengths_and_offsets(paths: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    ep_lengths = np.array([_episode_length(p) for p in paths], dtype=np.int64)
    cum_offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(ep_lengths, dtype=np.int64)])
    return ep_lengths, cum_offsets


def build_manifest(rollout_dirs: str | Sequence[str]) -> RolloutManifest:
    """Build a manifest from one or more rollout directories.

    The path list is the concatenation of each directory's sorted glob, in the
    order given by ``rollout_dirs``.
    """
    dirs = (rollout_dirs,) if isinstance(rollout_dirs, str) else tuple(rollout_dirs)
    if not dirs:
        raise ValueError("rollout_dirs must contain at least one directory")

    all_paths: list[str] = []
    for d in dirs:
        dir_paths = _list_episode_paths(d)
        if not dir_paths:
            raise FileNotFoundError(f"No episode archives found under {d}")
        all_paths.extend(dir_paths)

    ep_lengths, cum_offsets = _ep_lengths_and_offsets(all_paths)
    return RolloutManifest(
        rollout_dirs=dirs,
        paths=tuple(all_paths),
        ep_lengths=ep_lengths,
        cum_offsets=cum_offsets,
        manifest_sha=_sha256_of_paths(all_paths),
    )


def manifest_from_paths(paths: Sequence[str]) -> RolloutManifest:
    """Build a manifest from an explicit, already-ordered list of episode paths."""
    paths_t = tuple(paths)
    ep_lengths, cum_offsets = _ep_lengths_and_offsets(paths_t)
    return RolloutManifest(
        rollout_dirs=(),
        paths=paths_t,
        ep_lengths=ep_lengths,
        cum_offsets=cum_offsets,
        manifest_sha=_sha256_of_paths(paths_t),
    )


def manifest_npz_fields(manifest: RolloutManifest) -> dict:
    """Fields suitable for ``np.savez_compressed(**fields)``."""
    return {
        "manifest_sha": np.array(manifest.manifest_sha),
        "episode_paths": np.array(list(manifest.paths)),
        "ep_lengths": manifest.ep_lengths,
        "cum_offsets": manifest.cum_offsets,
        "num_frames": np.array(manifest.num_frames, dtype=np.int64),
    }


def load_manifest_from_npz(data) -> RolloutManifest | None:
    """Try to extract a :class:`RolloutManifest` from an ``NpzFile``-like object.

    Returns ``None`` when the manifest fields are absent (old-format files).
    """
    files = getattr(data, "files", None)
    if files is None or "manifest_sha" not in files:
        return None
    paths = tuple(str(p) for p in np.asarray(data["episode_paths"]).tolist())
    return RolloutManifest(
        rollout_dirs=(),
        paths=paths,
        ep_lengths=np.asarray(data["ep_lengths"], dtype=np.int64),
        cum_offsets=np.asarray(data["cum_offsets"], dtype=np.int64),
        manifest_sha=str(np.asarray(data["manifest_sha"]).item()),
    )
