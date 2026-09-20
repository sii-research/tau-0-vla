from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, ClassVar, Mapping, Sequence

import numpy as np
import torch

from tau0_vla.data.modalities import ComponentSpec, Quat2Rot6D
from tau0_vla.data.modalities import Image as _Image
from tau0_vla.data.modalities import Prompt as _Prompt
from tau0_vla.data.modalities.base import normalize_field_description_map, serialize_field_description_map
from tau0_vla.data.modalities.image import SyntheticImage as _SyntheticImage
from tau0_vla.data.pipeline import (
    ComponentAssembler,
    DatasetPipeline,
    MappedDataset,
    Repack,
    _ActionPostprocess,
    _build_component_output_dims,
    _build_image_transforms,
    _concatenate_components,
    _pad_vector,
    _resolve_structure,
    load_norm_stats,
)
from tau0_vla.data.prompt_source import PromptSource
from tau0_vla.data.source import (
    LeRobotDataset,
    _attach_episode_indices,
    _load_dataset_fps,
    _load_field_descriptions,
    _require_backend,
)
from tau0_vla.data.stats import norm_stats_path_for_config


@dataclasses.dataclass(frozen=True)
class FinchResolvedRobotConfig:
    robot_config: "RobotConfig"
    field_descriptions: dict[str, dict[str, dict[str, Any]]]
    action_key_forward: str
    state_key_forward: str
    action_dim: int
    state_dim: int
    action_horizon: int
    cam_keys: tuple[str, ...]
    norm_stats_path: str | None
    synthetic_images: tuple[tuple[str, str], ...] = ()


@dataclasses.dataclass(frozen=True)
class _ResolvedSourceSpec:
    field_descriptions: dict[str, dict[str, dict[str, Any]]]
    root: str | None
    episodes: Sequence[int] | None
    delta_timestamps: dict[str, list[float]] | None
    tolerance_s: float
    download_videos: bool
    video_backend: str | None


@dataclasses.dataclass(frozen=True)
class _ResolvedRuntimeSpec:
    norm_stats: dict[str, Any]
    state_map: Mapping[str, str | list[str]]
    action_map: Mapping[str, str | list[str]]
    state_components: Sequence[ComponentSpec]
    action_components: Sequence[ComponentSpec]
    state_descriptions: Mapping[str, Any]
    action_descriptions: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class FrameFilter:
    """Declarative annotation-driven filter policy for training anchors.

    ``positive`` entries are intersected; ``negative`` entries are subtracted.

    Supported labels:
      - ``"l3"``: ``instruction_segments``
      - ``"l2"``: sub-task ``key_frame`` intervals (``Subtask frame``)
      - ``"error_frame"``: ``key_frame`` intervals marked ``error_frame=true``
        (valid in ``negative`` only)
    """

    positive: tuple[str, ...] = ("l3", "l2")
    negative: tuple[str, ...] = ("error_frame",)

    _ALIASES: ClassVar[dict[str, str]] = {
        "l3": "l3",
        "instruction_segments": "l3",
        "instruction_segment": "l3",
        "segment": "l3",
        "segments": "l3",
        "l2": "l2",
        "subtask_frame": "l2",
        "subtask_frames": "l2",
        "subtask": "l2",
        "key_frame": "l2",
        "key_frames": "l2",
        "error_frame": "error_frame",
        "error_frames": "error_frame",
    }
    _POSITIVE_ALLOWED: ClassVar[frozenset[str]] = frozenset({"l3", "l2"})
    _NEGATIVE_ALLOWED: ClassVar[frozenset[str]] = frozenset({"error_frame"})

    def __post_init__(self) -> None:
        pos = self._normalize_many(self.positive, allowed=self._POSITIVE_ALLOWED, role="positive")
        neg = self._normalize_many(self.negative, allowed=self._NEGATIVE_ALLOWED, role="negative")
        object.__setattr__(self, "positive", pos)
        object.__setattr__(self, "negative", neg)

    @classmethod
    def _normalize_many(
        cls,
        labels: Sequence[str],
        *,
        allowed: frozenset[str],
        role: str,
    ) -> tuple[str, ...]:
        out: list[str] = []
        seen: set[str] = set()
        for raw in labels:
            key = cls._normalize_one(raw, role=role)
            if key not in allowed:
                raise ValueError(f"FrameFilter.{role} label {raw!r} resolved to {key!r}, allowed={sorted(allowed)}")
            if key not in seen:
                out.append(key)
                seen.add(key)
        return tuple(out)

    @classmethod
    def _normalize_one(cls, raw: str, *, role: str) -> str:
        key = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
        normalized = cls._ALIASES.get(key)
        if normalized is None:
            raise ValueError(f"Unknown FrameFilter.{role} label: {raw!r}")
        return normalized

    def uses_instruction_segments(self) -> bool:
        return "l3" in self.positive


@dataclasses.dataclass(frozen=True)
class RobotConfig:
    # ``repo_id`` is normally a single LeRobot dataset root, but may also be a
    # ``list[str]`` — the training pipeline then uses lerobot's
    # ``MultiLeRobotDataset`` to concatenate the underlying datasets into one
    # logical source (mirrors openpi_baseline's ``LerobotArxBCDataConfig(repo_id=[...])``
    # usage and the downstream ``MultiDataConfigFactory`` / DataMix expecting
    # "one ARX child" rather than N per-repo children).
    repo_id: "str | Sequence[str]"
    state: Sequence[ComponentSpec] | None = None
    action: Sequence[ComponentSpec] | None = None
    action_horizon: int | None = None
    prompt_source: PromptSource | None = None
    source_kwargs: dict[str, Any] | None = None
    norm_stats_dir: str | None = None
    # Explicit norm_stats JSON file override. When set, load from this path
    # verbatim (no per-config fingerprint check, no auto-compute) — useful for
    # reusing a precomputed stats file across fine-tune configs that share the
    # upstream data distribution. Mutually exclusive with ``norm_stats_dir``.
    norm_stats_path: str | None = None
    state_padding_dim: int | None = None
    action_padding_dim: int | None = None
    # Per-camera image declarations and the prompt template. Both are consumed
    # by training (`build_pipeline`) and deploy (`resolve_images` /
    # `resolve_prompt`) through the same spec objects, so the two paths cannot
    # drift.
    images: Sequence[_Image] | None = None
    prompt: _Prompt | None = None
    # Multi-frame / temporal window declaration. Each entry value is one of:
    #   int N                — past-N-frames shorthand ``[-(N-1), ..., 0]`` (frames).
    #   list[float | int]    — explicit offsets in **seconds**. Int entries are cast to
    #                          float (they do NOT mean frame counts). Use
    #                          ``[i/fps for i in range(k)]`` for frame-spaced values.
    # Supported keys:
    #   "state"                           — maps to ``observation.state``
    #   "action"                          — maps to ``action`` (overrides ``action_horizon``)
    #   "image" / "images"                — applies to every camera in ``repack["images"]``
    #   "image.<cam>" / "images.<cam>"    — per-camera override (e.g. ``image.head``)
    #   "<cam>"                           — bare camera name also accepted as shortcut
    temporal_sequence: Mapping[str, Any] | None = None
    # Instruction-segment filter: when True, ``build_pipeline`` restricts
    # anchors to frames inside the ``instruction_segments`` intervals declared
    # in ``meta/info.json`` (+ head tail_buffer = action_horizon - 1 so every
    # anchor's action window stays inside its own segment).
    # Defaults to True: pairs with ``prompt_source=instruction_segments`` and
    # rules out unannotated / tail-reset frames that would otherwise inject
    # garbage prompts into training. Set False to train on every frame.
    filter_by_segments: bool = True
    # Preferred API: positive labels are intersected, negative labels are
    # removed. ``FrameFilter(positive=("l3","l2"),
    # negative=("error_frame",))`` means:
    # ``instruction_segments ∩ subtask key_frame - error_frame``. When unset,
    # ``filter_by_segments`` selects between the default policy and no filter.
    frame_filter: FrameFilter | None = None
    # When True, each batch returned by ``FinchDataLoader`` carries an extra
    # ``"extras"`` key with per-role concat ndarrays of state/action in all
    # three norm forms side-by-side: ``raw`` (pre-component-transform
    # projection, dataset-native coordinates — Quaternion not Rot6D, abs not
    # relative), ``mean_std_norm`` (mean/std-normalized post-transform),
    # ``q_norm`` (q01/q99-normalized post-transform). Default False — zero
    # behavioral change for existing configs. Not part of ``config_fingerprint``
    # since the content is derivable from norm_stats + components.
    return_all_norm_forms: bool = False

    robot_name: ClassVar[str]
    repack: ClassVar[dict[str, object]]
    quaternion_order: ClassVar[str] = "xyzw"

    def resolve_images(self, cam_keys: Sequence[str]) -> list[_Image]:
        """Resolve per-camera `Image` specs for inference / deploy.

        When ``self.images`` is set, its names must cover ``cam_keys``;
        otherwise every camera passes through at native resolution.

        Callers drive image preprocessing by iterating the returned list and
        calling ``img_spec.apply(raw_image, train=False)``.
        """
        if self.images is not None:
            declared = {img.name: img for img in self.images}
            missing = [k for k in cam_keys if k not in declared]
            if missing:
                raise ValueError(
                    f"`images` declaration is missing specs for cameras {missing}. Declared: {sorted(declared)}."
                )
            return [declared[k] for k in cam_keys]
        return [_Image(name=k) for k in cam_keys]

    def canonical_cam_keys(self) -> tuple[str, ...]:
        keys = [str(key) for key in self.repack.get("images", {}).keys()]
        if self.images is not None:
            for spec in self.images:
                name = str(spec.name)
                if name not in keys:
                    keys.append(name)
        return tuple(keys)

    def synthetic_image_modes(self, cam_keys: Sequence[str]) -> dict[str, str]:
        if self.images is None:
            return {}
        declared = {img.name: img for img in self.images}
        out: dict[str, str] = {}
        for key in cam_keys:
            spec = declared.get(key)
            if isinstance(spec, _SyntheticImage):
                out[str(key)] = str(spec.mode)
        return out

    def resolve_prompt(self) -> _Prompt:
        """Resolve the `Prompt` modality for deploy.

        Falls back to the identity prompt ``Prompt("{instruction}")`` when the
        config declares no template.
        """
        if self.prompt is not None:
            return self.prompt
        return _Prompt()

    @classmethod
    def supports_sdk_payload(cls) -> bool:
        return False

    @classmethod
    def observation_from_payload(cls, payload: Mapping[str, Any]) -> Any:
        raise NotImplementedError(f"Robot '{cls.robot_name}' does not define an SDK payload adapter")

    @classmethod
    def build_sdk_state_with_contract(
        cls,
        payload: Mapping[str, Any],
        *,
        norm_stats: Mapping[str, Any],
        state_key: str,
        use_quantile: bool,
        is_eef: bool,
        mode: str,
    ) -> np.ndarray:
        raise NotImplementedError(f"Robot '{cls.robot_name}' does not define SDK state conversion")

    @classmethod
    def recover_joint_actions(
        cls,
        actions: np.ndarray,
        *,
        payload: Mapping[str, Any],
        mode: str,
    ) -> np.ndarray:
        raise NotImplementedError(f"Robot '{cls.robot_name}' does not define EEF->joint conversion")

    @classmethod
    def recover_joint_actions_from_context(
        cls,
        actions: np.ndarray,
        *,
        joint_state: np.ndarray,
        waist: np.ndarray | None,
        head: np.ndarray | None,
        mode: str,
    ) -> np.ndarray:
        raise NotImplementedError(f"Robot '{cls.robot_name}' does not define context EEF->joint conversion")

    @classmethod
    def project_joint_actions(
        cls,
        actions: np.ndarray,
        *,
        payload: Mapping[str, Any],
        mode: str,
    ) -> np.ndarray:
        raise NotImplementedError(f"Robot '{cls.robot_name}' does not define joint->EEF conversion")

    @classmethod
    def project_joint_actions_from_context(
        cls,
        actions: np.ndarray,
        *,
        waist: np.ndarray | None,
        head: np.ndarray | None,
        mode: str,
    ) -> np.ndarray:
        raise NotImplementedError(f"Robot '{cls.robot_name}' does not define context joint->EEF conversion")

    @classmethod
    def present_eef_actions(
        cls,
        actions: np.ndarray,
        *,
        mode: str,
    ) -> np.ndarray:
        return np.asarray(actions, dtype=np.float32)

    @classmethod
    def native_dim_labels(cls, *, is_eef: bool, native_dim: int) -> list[str]:
        del native_dim
        if not is_eef:
            return [f"dim_{idx}" for idx in range(16)]
        return [f"dim_{idx}" for idx in range(14)]

    def resolved_norm_stats_dir(self) -> str:
        if self.norm_stats_dir:
            return self.norm_stats_dir
        repos = self._repo_id_list()
        if len(repos) > 1:
            raise ValueError(
                "Multi-repo configs (manifest or list[str] repo_id) must set "
                "``norm_stats_dir`` explicitly — there is no canonical per-repo "
                "``norm_stats/`` directory to fall back to."
            )
        return str(Path(repos[0]) / "norm_stats")

    def resolved_norm_stats_file(self) -> str:
        """Where the norm_stats JSON sits on disk — the explicit
        ``norm_stats_path`` override, or the dir-based ``{dir}/{storage_stem}.json``
        derivation otherwise."""
        if self.norm_stats_path is not None:
            return self.norm_stats_path
        return str(norm_stats_path_for_config(self.resolved_norm_stats_dir(), self))

    def _load_norm_stats(self, *, overwrite: bool = False):
        """Load norm_stats as a dict[str, NormStats]. Explicit
        ``norm_stats_path`` bypasses fingerprint check and auto-compute;
        the dir path keeps both. ``overwrite=True`` recomputes and
        overwrites when the JSON's fingerprint disagrees with the current
        config (dir-path only — the explicit-path branch has no fingerprint)."""
        from tau0_vla.data.pipeline import canonicalize_norm_stats
        from tau0_vla.data.stats import load_file

        if self.norm_stats_path is not None:
            if self.norm_stats_dir is not None:
                raise ValueError("``norm_stats_path`` and ``norm_stats_dir`` are mutually exclusive")
            return canonicalize_norm_stats(load_file(self.norm_stats_path))
        return load_norm_stats(self.resolved_norm_stats_dir(), self, overwrite=overwrite)

    def frame_filter_policy(self) -> FrameFilter | None:
        if self.frame_filter is not None:
            return self.frame_filter
        if not self.filter_by_segments:
            return None
        # ``FrameFilter``'s own defaults are the historical default policy:
        # ``instruction_segments ∩ subtask key_frame - error_frame``.
        return FrameFilter()

    def uses_instruction_segment_filter(self) -> bool:
        policy = self.frame_filter_policy()
        return policy is not None and policy.uses_instruction_segments()

    def _repo_id_list(self) -> list[str]:
        """Normalize ``self.repo_id`` to a list of repo paths.

        - ``str`` ending in ``.txt`` is treated as a manifest file: one
          repo_id per line, ``#`` comments + blank lines ignored. The N
          repos become a single ``MultiLeRobotDataset`` — i.e., one route,
          one norm_stats file, one fingerprint.
        - ``str`` otherwise: single repo, returned as ``[repo_id]``.
        - ``list[str]`` / ``tuple[str, ...]``: each entry may be either a
          direct repo path or a ``.txt`` manifest; all entries are flattened
          into one concatenated repo list, preserving order.

        Call sites that need ``LeRobotDataset`` vs ``MultiLeRobotDataset``
        key on ``len(self._repo_id_list()) > 1``.
        """
        if isinstance(self.repo_id, str):
            if self.repo_id.endswith(".txt"):
                return _read_repo_manifest(self.repo_id)
            return [self.repo_id]
        if isinstance(self.repo_id, (list, tuple)):
            repos: list[str] = []
            for item in self.repo_id:
                item_str = str(item)
                if item_str.endswith(".txt"):
                    repos.extend(_read_repo_manifest(item_str))
                else:
                    repos.append(item_str)
            return repos
        return [str(self.repo_id)]

    def _primary_repo_id(self) -> str:
        """First repo_id — used for dataset metadata (info.json, fps,
        field_descriptions, norm_stats filename fingerprint). All repos in a
        multi-repo config are assumed to share the same schema (``ARX(slow)``
        and ``ARX(base)`` both record ``observation.state`` as 14D, same keys)."""
        return self._repo_id_list()[0]

    def _sample_repo_id(self, raw: Mapping[str, Any]) -> str:
        """Return the source repo_id for one raw sample.

        In multi-repo configs (``.txt`` manifest or ``list[str]``),
        ``MultiLeRobotDataset.__getitem__`` injects ``raw["dataset_index"]``.
        We index ``_repo_id_list()`` with it so per-sample lookups (notably
        ``PromptSource.from_label(source="instruction_segments")`` which reads
        ``meta/info.json`` of the ACTUAL source repo) don't silently resolve
        against the first repo's metadata. Falls back to ``_primary_repo_id()``
        for single-repo configs, where ``dataset_index`` is absent.
        """
        idx_val = raw.get("dataset_index")
        if idx_val is None:
            return self._primary_repo_id()
        try:
            idx = int(idx_val.item() if hasattr(idx_val, "item") else idx_val)
        except (TypeError, ValueError):
            return self._primary_repo_id()
        repo_ids = self._repo_id_list()
        if not (0 <= idx < len(repo_ids)):
            raise IndexError(
                f"_sample_repo_id: dataset_index={idx} out of range for "
                f"{len(repo_ids)} repo_ids. This usually means MultiLeRobotDataset's "
                f"internal _datasets order disagrees with self.repo_id — "
                f"build_pipeline should have aligned them; check that "
                f"_align_self_to_raw_dataset ran."
            )
        return repo_ids[idx]

    def build_pipeline(
        self,
        *,
        disable_component_normalization: bool = False,
        overwrite_norm_stats: bool = False,
        physically_split: bool = False,
        physical_rank: int | None = None,
        physical_world_size: int | None = None,
        video_backend: str | None = None,
    ) -> DatasetPipeline:
        source = self._resolve_source_spec(video_backend=video_backend)
        runtime = self._resolve_runtime_spec(
            source.field_descriptions,
            disable_component_normalization=disable_component_normalization,
            overwrite_norm_stats=overwrite_norm_stats,
        )
        _require_backend()
        _install_bounded_decoder_cache()
        _warn_if_instruction_segments_without_filter(self)
        repo_ids = self._repo_id_list()
        if physically_split:
            from tau0_vla.data.pipeline import _apply_physical_sharding_to_repos

            repo_ids, _ = _apply_physical_sharding_to_repos(
                repo_ids,
                physical_rank=physical_rank,
                physical_world_size=physical_world_size,
            )
            # MultiLeRobotDataset.__getitem__ injects ``dataset_index`` relative
            # to the SLICED list it was built with. ``_sample_repo_id`` (called
            # per-sample by ``_inject_prompt`` to resolve which repo's
            # ``meta/info.json`` to consult for ``instruction_segments``) indexes
            # into ``self._repo_id_list()`` — so we must rebind ``self`` to a
            # copy carrying the sliced repo_id list, otherwise dataset_index→
            # repo_id resolves against the full manifest and we either crash on
            # KeyError or, worse, silently feed the wrong repo's instruction
            # into training.
            config_name = getattr(self, "_finch_config_name", None)
            self = dataclasses.replace(self, repo_id=list(repo_ids))
            if config_name is not None:
                object.__setattr__(self, "_finch_config_name", config_name)
        lerobot_repo_ids, lerobot_root, lerobot_episodes, local_repo_aliases = _lerobot_local_repo_args(
            repo_ids,
            source.episodes,
            source.root,
        )
        if len(repo_ids) == 1:
            raw_dataset = _build_dataset_with_retry(
                lambda: LeRobotDataset(
                    repo_id=lerobot_repo_ids[0],
                    root=lerobot_root,
                    episodes=lerobot_episodes,
                    delta_timestamps=source.delta_timestamps,
                    tolerance_s=source.tolerance_s,
                    download_videos=source.download_videos,
                    video_backend=source.video_backend,
                ),
                desc=str(lerobot_repo_ids[0]),
            )
        else:
            from lerobot.datasets import lerobot_dataset as _lr

            _patch_multirepo_stats_safely(_lr)
            raw_dataset = _build_dataset_with_retry(
                lambda: _lr.MultiLeRobotDataset(
                    repo_ids=lerobot_repo_ids,
                    root=lerobot_root,
                    episodes=lerobot_episodes,
                    delta_timestamps=source.delta_timestamps,
                    tolerances_s={rid: source.tolerance_s for rid in lerobot_repo_ids},
                    download_videos=source.download_videos,
                    video_backend=source.video_backend,
                ),
                desc=f"{len(lerobot_repo_ids)} repos",
            )
            if local_repo_aliases is not None:
                raw_dataset.repo_ids = list(repo_ids)
        # Anchor repo-list order on the raw dataset's authoritative ordering.
        # MultiLeRobotDataset.__getitem__ injects dataset_index relative to its
        # own _datasets list — when LeRobot reorders/dedupes vs the manifest,
        # silently reusing manifest order misroutes per-sample lookups (e.g.
        # instruction_segments → wrong info.json) and pairs the wrong meta with
        # the wrong child in _compute_filter_ranges. No-op for single-repo
        # (LeRobotDataset has no .repo_ids) and for already-aligned multi-repo
        # (incl. the physically_split=True path which rebound earlier).
        self, repo_ids = _align_self_to_raw_dataset(self, raw_dataset, repo_ids)
        field_descriptions_by_dataset = _field_descriptions_by_repo_order(
            repo_ids,
            fallback=source.field_descriptions,
        )
        _attach_episode_indices(raw_dataset)
        if not disable_component_normalization:
            filter_ranges = _compute_filter_ranges(raw_dataset, self, repo_ids)
            if filter_ranges is not None:
                raw_dataset = _RangeFilteredDataset(raw_dataset, *filter_ranges)
        dataset = MappedDataset(
            raw_dataset,
            transforms=(_RawSampleWrapper(self, field_descriptions_by_dataset),),
        )
        action_postprocess = _ActionPostprocess(
            state_components=runtime.state_components,
            action_components=runtime.action_components,
            state_component_dims=_build_component_output_dims(
                components=runtime.state_components,
                role="state",
                descriptions=runtime.state_descriptions,
                mapping=runtime.state_map,
            ),
            action_component_dims=_build_component_output_dims(
                components=runtime.action_components,
                role="action",
                descriptions=runtime.action_descriptions,
                mapping=runtime.action_map,
            ),
            norm_stats=runtime.norm_stats,
            state_padding_dim=self.state_padding_dim,
            action_padding_dim=self.action_padding_dim,
            disable_component_normalization=disable_component_normalization,
        )
        prompt_transform = (self._inject_prompt,) if self.prompt_source is not None else ()
        return DatasetPipeline(
            dataset=dataset,
            name=self.display_name(),
            transforms=(
                *prompt_transform,
                Repack(self._raw_repack_structure()),
                self._wrap_prompt,
                *(_build_image_transforms(images=self.images)),
                self._build_component_assembler(
                    field_descriptions=source.field_descriptions,
                    disable_component_normalization=disable_component_normalization,
                ),
            ),
            action_postprocess=action_postprocess,
        )

    def _resolve_source_spec(
        self,
        *,
        video_backend: str | None = None,
    ) -> _ResolvedSourceSpec:
        source_kwargs = dict(self.source_kwargs or {})
        allowed_source_keys = {
            "root",
            "episodes",
            "delta_timestamps",
            "tolerance_s",
            "download_videos",
            "video_backend",
        }
        unknown_source_keys = set(source_kwargs) - allowed_source_keys
        if unknown_source_keys:
            unknown = ", ".join(sorted(unknown_source_keys))
            raise ValueError(f"Unsupported source_kwargs for Finch/LeRobotDataset: {unknown}")
        return _ResolvedSourceSpec(
            field_descriptions=_load_field_descriptions(self._primary_repo_id())
            if Path(self._primary_repo_id()).exists()
            else {},
            root=source_kwargs.get("root"),
            episodes=source_kwargs.get("episodes"),
            delta_timestamps=self._resolve_source_delta_timestamps(source_kwargs),
            tolerance_s=float(source_kwargs.get("tolerance_s", 0.001)),
            download_videos=bool(source_kwargs.get("download_videos", True)),
            video_backend=video_backend if video_backend is not None else source_kwargs.get("video_backend", "pyav"),
        )

    def _resolve_runtime_spec(
        self,
        field_descriptions: Mapping[str, Any],
        *,
        disable_component_normalization: bool = False,
        overwrite_norm_stats: bool = False,
    ) -> _ResolvedRuntimeSpec:
        state_map, action_map, state_components, action_components = self._resolve_component_bundle()
        # Skip stats loading entirely when no declared component asks for
        # normalization. Mirrors baseline's behavior where ``stats_dir=None`` +
        # missing ``<assets_dir>/<repo>/norm_stats.json`` returns ``None`` and
        # the training pipeline runs in raw space.
        any_component_needs_stats = any(
            getattr(c, "requires_normalization", lambda: True)()
            for c in list(state_components) + list(action_components)
        )
        norm_stats = None
        if not disable_component_normalization and any_component_needs_stats:
            norm_stats = self._load_norm_stats(overwrite=overwrite_norm_stats)
        state_descriptions, action_descriptions = self._normalize_field_descriptions(field_descriptions)
        return _ResolvedRuntimeSpec(
            norm_stats=norm_stats,
            state_map=state_map,
            action_map=action_map,
            state_components=state_components,
            action_components=action_components,
            state_descriptions=state_descriptions,
            action_descriptions=action_descriptions,
        )

    def _resolve_source_delta_timestamps(
        self,
        source_kwargs: Mapping[str, Any],
    ) -> dict[str, list[float]] | None:
        configured = source_kwargs.get("delta_timestamps")
        if configured is not None:
            return configured
        fps = _load_dataset_fps(self._primary_repo_id())
        merged: dict[str, list[float]] = {}
        action_horizon = int(self.action_horizon or 1)
        if action_horizon > 1 and fps is not None:
            action_col = self.repack.get("action", {}).get("raw", "action")
            merged[action_col] = [t / fps for t in range(action_horizon)]
        # `temporal_sequence` entries override action_horizon-generated action offsets
        # when they provide an explicit "action" key; other keys (state / images)
        # augment the delta_timestamps dict.
        temporal = self._resolve_temporal_offsets(fps=fps)
        merged.update(temporal)
        return merged or None

    def _resolve_temporal_offsets(self, *, fps: float | None) -> dict[str, list[float]]:
        if not self.temporal_sequence:
            return {}
        if fps is None:
            raise RuntimeError(
                f"temporal_sequence requires dataset fps; `{self._primary_repo_id()}` did not expose one."
            )
        dt = 1.0 / float(fps)
        camera_map = dict(self.repack.get("images", {}) or {})
        out: dict[str, list[float]] = {}
        for user_key, value in self.temporal_sequence.items():
            offsets = _normalize_offsets(value, dt=dt, key=user_key)
            for lerobot_key in _resolve_temporal_key(user_key, camera_map):
                out[lerobot_key] = offsets
        return out

    def display_name(self) -> str:
        return Path(self._primary_repo_id()).name or self.robot_name

    def repack_observation(
        self,
        raw_sample: Mapping[str, Any],
        *,
        field_descriptions: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        repacked = self._repack_raw_sample(raw_sample)
        prompt = self._canonicalize_prompt(repacked.get("prompt"))
        images = {key: _canonicalize_image(value) for key, value in repacked.get("images", {}).items()}
        state_raw = np.asarray(repacked["_state_raw"], dtype=np.float32)
        action_raw = np.asarray(repacked["_action_raw"], dtype=np.float32)
        state_description_map, action_description_map = self._normalize_field_descriptions(field_descriptions)
        # Apply robot-family raw-tensor augmentation so indices declared in
        # ``_normalize_field_descriptions`` resolve — exactly what
        # ``ComponentAssembler`` does at training time. Keeping the two
        # callsites in lockstep is the whole point: a raw-sample helper
        # that diverges from the pipeline it's supposed to mirror is a bug.
        state_raw, _ = self.augment_raw_tensors(
            state_raw=state_raw,
            action_raw=action_raw,
            state_descriptions=state_description_map,
            action_descriptions=action_description_map,
        )
        state_map, _, state_components, _ = self._resolve_component_bundle()
        # Only project fields backing components declared on `self.state`. The
        # class-level `repack["state"]["semantic"]` dict advertises every field
        # the robot class COULD expose (e.g. eef_pose on G1), but a concrete
        # config may opt into just a subset (arm_joint + gripper), and the
        # backing dataset may not even carry the unused fields.
        state_values = {
            component.key: _project_field_value(
                descriptions=state_description_map,
                field_key=state_map.get(component.key, component.field_key),
                raw_values=state_raw,
            )
            for component in state_components
        }
        return {
            "prompt": prompt,
            "images": images,
            "state": state_values,
        }

    def repack_action_target(
        self,
        raw_sample: Mapping[str, Any],
        *,
        field_descriptions: Mapping[str, Any] | None = None,
        action_padding_dim: int | None = None,
    ) -> np.ndarray:
        repacked = self._repack_raw_sample(raw_sample)
        state_raw = np.asarray(repacked["_state_raw"], dtype=np.float32)
        action_raw = np.asarray(repacked["_action_raw"], dtype=np.float32)
        state_description_map, action_description_map = self._normalize_field_descriptions(field_descriptions)
        # See ``repack_observation`` — same reason: training's
        # ``ComponentAssembler`` runs ``augment_raw_tensors`` before projecting,
        # so this raw-sample helper must too or the two paths diverge on
        # adapter-added fields.
        _, action_raw = self.augment_raw_tensors(
            state_raw=state_raw,
            action_raw=action_raw,
            state_descriptions=state_description_map,
            action_descriptions=action_description_map,
        )
        _, action_map, _, action_components = self._resolve_component_bundle()
        action = _concatenate_components(
            [
                _project_field_value(
                    descriptions=action_description_map,
                    field_key=action_map.get(component.key, component.field_key),
                    raw_values=action_raw,
                )
                for component in action_components
            ]
        )
        return _pad_vector(action, action_padding_dim or self.action_padding_dim, label="action")

    def build_model_vectors(
        self,
        raw_sample: Mapping[str, Any],
        *,
        field_descriptions: Mapping[str, Any],
        state_padding_dim: int | None = None,
        action_padding_dim: int | None = None,
    ) -> dict[str, np.ndarray]:
        assembler = self._build_component_assembler(
            field_descriptions=field_descriptions,
            state_padding_dim=state_padding_dim,
            action_padding_dim=action_padding_dim,
        )
        repacked = self._repack_raw_sample(raw_sample)
        repacked["_field_descriptions"] = field_descriptions
        built = assembler(repacked)
        return {
            "state": np.asarray(built["state"], dtype=np.float32),
            "action": np.asarray(built["action"], dtype=np.float32),
        }

    def resolve_output_spec(
        self,
        *,
        field_descriptions: Mapping[str, Any],
    ) -> FinchResolvedRobotConfig:
        state_map, action_map, state_components, action_components = self._resolve_component_bundle()
        state_descriptions, action_descriptions = self._normalize_field_descriptions(field_descriptions)
        state_dim = sum(
            _build_component_output_dims(
                components=state_components,
                role="state",
                descriptions=state_descriptions,
                mapping=state_map,
            )
        )
        action_dim = sum(
            _build_component_output_dims(
                components=action_components,
                role="action",
                descriptions=action_descriptions,
                mapping=action_map,
            )
        )
        state_dim = self.state_padding_dim or state_dim
        action_dim = self.action_padding_dim or action_dim
        cam_keys = self.canonical_cam_keys()
        return FinchResolvedRobotConfig(
            robot_config=self,
            field_descriptions=dict(field_descriptions),
            action_key_forward=_default_action_key(action_components),
            state_key_forward=_default_state_key(state_components),
            action_dim=int(action_dim),
            state_dim=int(state_dim),
            action_horizon=int(self.action_horizon or 1),
            # ``cam_keys`` is the deploy-side camera identifier list, stored
            # in the same form the policy sees AFTER repack — i.e. the
            # canonical short names (``head`` / ``wrist_left`` /
            # ``wrist_right``), which are the ``repack["images"]`` dict
            # KEYS. The dict's VALUES are raw lerobot field names
            # (``observation.images.top_head`` etc.) consumed only during
            # dataset reading; they should never leak into the deploy
            # contract. Both ``canonical["images"]`` at train/eval time and
            # the SDK payload at serving time key cameras by the short
            # names, so ``cam_keys`` must too for ``_select_camera_frames``
            # (and anything else that validates against this list) to
            # round-trip correctly.
            cam_keys=cam_keys,
            norm_stats_path=self.resolved_norm_stats_file(),
            synthetic_images=tuple(self.synthetic_image_modes(cam_keys).items()),
        )

    def resolve_output_spec_for_repo(self, repo_id: str | Path) -> FinchResolvedRobotConfig:
        resolved_repo = str(Path(repo_id).resolve())
        config = self._with_repo_binding(resolved_repo)
        field_descriptions = _load_field_descriptions(resolved_repo)
        return config.resolve_output_spec(field_descriptions=field_descriptions)

    def _build_component_assembler(
        self,
        *,
        field_descriptions: Mapping[str, Any],
        state_padding_dim: int | None = None,
        action_padding_dim: int | None = None,
        disable_component_normalization: bool = False,
    ) -> ComponentAssembler:
        state_map, action_map, state_components, action_components = self._resolve_component_bundle()
        any_component_needs_stats = any(
            getattr(c, "requires_normalization", lambda: True)()
            for c in list(state_components) + list(action_components)
        )
        norm_stats = None
        if not disable_component_normalization and any_component_needs_stats:
            norm_stats = self._load_norm_stats()
        # ``return_all_norm_forms`` itself doesn't require norm_stats
        # (extras["raw"] is always available), but mean_std_norm / q_norm DO
        # need them. Force-load when the flag is on to keep the extras dict
        # useful even when the canonical pipeline disabled normalization.
        if self.return_all_norm_forms and norm_stats is None and any_component_needs_stats:
            norm_stats = self._load_norm_stats()
        return ComponentAssembler(
            state_components=state_components,
            action_components=action_components,
            norm_stats=norm_stats,
            state_field_map=state_map,
            action_field_map=action_map,
            state_padding_dim=state_padding_dim or self.state_padding_dim,
            action_padding_dim=action_padding_dim or self.action_padding_dim,
            state_active_indices=getattr(self, "_state_active_indices", None),
            action_active_indices=getattr(self, "_action_active_indices", None),
            disable_component_normalization=disable_component_normalization,
            state_has_temporal_axis=bool(self.temporal_sequence and "state" in self.temporal_sequence),
            augment_fn=self.augment_raw_tensors,
            return_all_norm_forms=self.return_all_norm_forms,
        )

    def augment_raw_tensors(
        self,
        *,
        state_raw: np.ndarray,
        action_raw: np.ndarray,
        state_descriptions: Mapping[str, Any],
        action_descriptions: Mapping[str, Any],
    ) -> "tuple[np.ndarray, np.ndarray]":
        """Extend raw state/action tensors with config-derived fields.

        Called with the flat raw tensors (``[D]`` or ``[B, D]`` for state,
        ``[H, D]`` or ``[B, H, D]`` for action) AFTER reading from disk and
        BEFORE the tensors are sliced into per-component values via
        ``field_descriptions``. Both the stats fast path
        (``_iter_hf_repo_module_windows``) and the row-wise pipeline
        (``ComponentAssembler.__call__``) invoke this hook.

        Default: identity (returns inputs unchanged, zero cost).

        Override in robot-family subclasses to synthesize computed fields
        (e.g. online FK for G1). The returned tensors must match the indices
        declared in the overriding config's ``_normalize_field_descriptions``
        for any newly-declared fields (the assembler slices by those indices).
        """
        return state_raw, action_raw

    def _repack_raw_sample(self, raw_sample: Mapping[str, Any]) -> dict[str, Any]:
        if self.prompt_source is not None:
            raw_sample = self._inject_prompt({"raw": dict(raw_sample)})["raw"]
        return _resolve_structure(self._raw_repack_structure(), raw_sample)

    def _raw_repack_structure(self) -> dict[str, Any]:
        return {
            "prompt": self.repack["prompt"],
            "images": self.repack["images"],
            "_state_raw": self.repack["state"]["raw"],
            "_action_raw": self.repack["action"]["raw"],
        }

    def _inject_prompt(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Pre-Repack: resolve ``prompt_source`` and overwrite ``raw[repack['prompt']]``
        with the RAW instruction string. Template / prefix / suffix wrapping is
        applied later by ``_wrap_prompt`` (training) or ``_canonicalize_prompt``
        (deploy) — both go through ``resolve_prompt().format(...)`` so the two
        paths cannot drift.
        """
        raw = sample["raw"]
        key = str(self.repack["prompt"])
        resolved = self.prompt_source.resolve(
            raw,
            repo_id=self._sample_repo_id(raw),
            default_parquet_key=key,
        )
        return {**sample, "raw": {**dict(raw), key: resolved}}

    def _wrap_prompt(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Post-Repack training transform: apply ``resolve_prompt().format(...)``
        so ``batch["prompt"]`` is what the VLM actually sees at deploy."""
        if "prompt" not in sample:
            return sample
        raw = sample["prompt"]
        if isinstance(raw, np.ndarray):
            raw = raw.item()
        return {**sample, "prompt": self.resolve_prompt().format(str(raw))}

    def _wrap_raw_sample(
        self,
        raw_sample: Mapping[str, Any],
        *,
        field_descriptions: Mapping[str, Any],
    ) -> dict[str, Any]:
        state_desc, action_desc = self._normalize_field_descriptions(field_descriptions)
        return {
            "meta": {
                "source_type": "finch",
                "dataset": self._primary_repo_id(),
                "field_descriptions": {
                    "state": serialize_field_description_map(state_desc),
                    "action": serialize_field_description_map(action_desc),
                },
            },
            "raw": dict(raw_sample),
        }

    def _normalize_field_descriptions(
        self,
        field_descriptions: Mapping[str, Any] | None,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        state_description_map = normalize_field_description_map((field_descriptions or {}).get("state"))
        if not state_description_map:
            raise ValueError("Dataset metadata must define state field_descriptions")
        action_description_map = normalize_field_description_map((field_descriptions or {}).get("action"))
        if not action_description_map:
            raise ValueError("Dataset metadata must define action field_descriptions")
        return state_description_map, action_description_map

    def _with_repo_binding(self, repo_id: str) -> "RobotConfig":
        original_repo = str(Path(self._primary_repo_id()).resolve())
        # When the caller pinned an explicit ``norm_stats_path`` we must
        # preserve it AND keep ``norm_stats_dir`` as None — the two are
        # mutually exclusive (see ``_load_norm_stats``). Rebasing to a
        # per-repo default directory here would silently shadow the path.
        if self.norm_stats_path is not None:
            if self._primary_repo_id() == repo_id:
                return self
            bound = dataclasses.replace(self, repo_id=repo_id)
            config_name = getattr(self, "_finch_config_name", None)
            if config_name is not None:
                try:
                    object.__setattr__(bound, "_finch_config_name", config_name)
                except Exception:
                    pass
            return bound
        if self.norm_stats_dir is None:
            norm_stats_dir = str(Path(repo_id) / "norm_stats")
        else:
            resolved_stats = Path(self.norm_stats_dir).resolve()
            try:
                relative_stats = resolved_stats.relative_to(Path(original_repo))
            except ValueError:
                relative_stats = None
            norm_stats_dir = (
                str((Path(repo_id) / relative_stats).resolve()) if relative_stats is not None else self.norm_stats_dir
            )
        if self._primary_repo_id() == repo_id and self.norm_stats_dir == norm_stats_dir:
            return self
        bound = dataclasses.replace(self, repo_id=repo_id, norm_stats_dir=norm_stats_dir)
        config_name = getattr(self, "_finch_config_name", None)
        if config_name is not None:
            try:
                object.__setattr__(bound, "_finch_config_name", config_name)
            except Exception:
                pass
        return bound

    def _resolve_component_bundle(
        self,
    ) -> tuple[
        Mapping[str, str | list[str]],
        Mapping[str, str | list[str]],
        Sequence[ComponentSpec],
        Sequence[ComponentSpec],
    ]:
        state_map = {
            key: [str(item) for item in value] if isinstance(value, list) else str(value)
            for key, value in self.repack["state"]["semantic"].items()
        }
        action_map = {
            key: [str(item) for item in value] if isinstance(value, list) else str(value)
            for key, value in self.repack["action"]["semantic"].items()
        }
        state_components = self._bind_components(self.state, state_map, role="state")
        action_components = self._bind_components(self.action, action_map, role="action")
        return state_map, action_map, state_components, action_components

    def _bind_components(
        self,
        components: Sequence[ComponentSpec] | None,
        mapping: Mapping[str, str | list[str]],
        *,
        role: str,
    ) -> tuple[ComponentSpec, ...]:
        if components is None:
            raise ValueError(f"Robot '{self.robot_name}' must explicitly specify {role} components")
        resolved: list[ComponentSpec] = []
        for component in components:
            try:
                field_key = component.field_key
                if field_key is None:
                    field_key = mapping[component.key]
                    if isinstance(field_key, list):
                        field_key = f"{role}/{component.key}"
                transforms = [
                    Quat2Rot6D(
                        quat_order=transform.quat_order if transform.quat_order is not None else self.quaternion_order
                    )
                    if isinstance(transform, Quat2Rot6D)
                    else transform
                    for transform in component.transforms
                ]
                resolved.append(
                    type(component)(
                        key=component.key,
                        normalize=component.normalize,
                        field_key=field_key,
                        transforms=transforms,
                    )
                )
            except KeyError as exc:
                raise ValueError(
                    f"Robot '{self.robot_name}' does not support {role} component '{component.key}'"
                ) from exc
        return tuple(resolved)

    def _canonicalize_prompt(self, prompt: Any) -> str:
        if prompt is None:
            raise ValueError(f"Robot '{self.robot_name}' requires a prompt/task for canonical observation repack")
        if isinstance(prompt, np.ndarray):
            prompt = prompt.item()
        raw = prompt if isinstance(prompt, str) else str(prompt)
        # Unify train / deploy: the declaration ``resolve_prompt()`` picks is
        # also what the training batch sees. Otherwise ``batch["prompt"]`` is
        # the raw instruction while the deploy runner hands the wrapped version
        # to the VLM — silent skew.
        return self.resolve_prompt().format(raw)


def _build_dataset_with_retry(construct, *, desc: str, attempts: int = 8, base_delay: float = 1.5):
    """Construct a (Multi)LeRobotDataset, retrying TRANSIENT distributed-FS read
    errors. Under heavy concurrent IO (many ranks at scale hammering the shared
    parquet store), a read can momentarily return a short/empty buffer ->
    ``pyarrow.lib.ArrowInvalid: Tried reading schema message, was null or length
    0`` (or OSError). The files are fine (verified offline); a retry succeeds.
    Persistent errors (real corruption / missing file) still raise after the last
    attempt — only the failure is delayed, never masked."""
    import time

    for attempt in range(1, attempts + 1):
        try:
            return construct()
        except Exception as exc:  # noqa: BLE001
            name = type(exc).__name__
            transient = ("ArrowInvalid" in name) or ("schema message" in str(exc)) or isinstance(exc, OSError)
            if not transient or attempt == attempts:
                raise
            delay = base_delay * attempt
            print(
                f"[tau0-vla] transient dataset-read error on {desc} "
                f"(attempt {attempt}/{attempts}): {name}: {str(exc)[:90]}; retrying in {delay:.1f}s",
                flush=True,
            )
            time.sleep(delay)


def _patch_multirepo_stats_safely(lerobot_dataset_module: Any) -> None:
    """Wrap lerobot's ``aggregate_stats`` so ``MultiLeRobotDataset.__init__``
    doesn't die on cross-embodiment manifests with per-repo shape diversity.

    ``MultiLeRobotDataset`` calls ``aggregate_stats`` over every sub-dataset's
    ``meta.stats`` to populate ``self.stats``. When repos in the same manifest
    have different ``observation.state`` / ``action`` dims (common in
    cross-embodiment or multi-version pretrain manifests — e.g. the AgiBot
    500h pretrain mixes a 32D-state format and a 94D-state format), the
    per-feature ``np.stack`` of mean arrays raises ``ValueError``.

    This pipeline does NOT consume ``MultiLeRobotDataset.stats`` — it computes
    its own norm_stats via the fast parquet path (``tau0_vla.data.stats``), so
    the lerobot aggregation is purely collateral. Catch the shape-mismatch
    ValueError and substitute an empty dict; warn once per process so the
    cause is visible in training logs. ``_assert_type_and_shape`` is called
    upstream and would still fire on genuinely malformed stats; we only
    soften the cross-repo ``np.stack`` failure.
    """
    import warnings

    if getattr(lerobot_dataset_module, "_tau0_multirepo_stats_patched", False):
        return
    original_aggregate = lerobot_dataset_module.aggregate_stats

    def _safe_aggregate_stats(stats_list):
        try:
            return original_aggregate(stats_list)
        except ValueError as exc:
            if "same shape" not in str(exc):
                raise
            warnings.warn(
                "tau0-vla: MultiLeRobotDataset.aggregate_stats skipped due to "
                f"heterogeneous per-repo shapes ({exc}). ``self.stats`` set to "
                "empty dict; tau-0-vla computes its own norm_stats over parquet, "
                "so training is unaffected.",
                stacklevel=2,
            )
            return {}

    lerobot_dataset_module.aggregate_stats = _safe_aggregate_stats
    lerobot_dataset_module._tau0_multirepo_stats_patched = True


def _warn_if_instruction_segments_without_filter(config: "RobotConfig") -> None:
    """Emit a yellow-highlighted stderr warning when a config combines
    ``prompt_source=instruction_segments`` without an ``l3``-based annotation
    filter.

    That combination is almost always a misconfiguration: any sample whose
    episode is stored as ``[]`` in ``info.json`` will hit a runtime ``KeyError``
    at ``_lookup_segment_instruction``. Turning on an annotation filter whose
    ``positive`` includes ``"l3"`` skips those frames at anchor-build time.
    The warning is printed once per ``build_pipeline`` call.
    """
    import sys

    if config.uses_instruction_segment_filter():
        return
    source = config.prompt_source
    if source is None or not source.uses_instruction_segments():
        return
    label = config.display_name()
    yellow = "\033[93m" if sys.stderr.isatty() else ""
    reset = "\033[0m" if sys.stderr.isatty() else ""
    print(
        f"{yellow}[tau0-vla] WARNING ({label!r}): prompt_source uses "
        f"'instruction_segments' but the frame filter does not include "
        f"'l3' — unannotated "
        f"episodes (info.json entry `[]`) will raise at runtime. Set "
        f"frame_filter=FrameFilter(positive=('l3', ...)) on "
        f"the RobotConfig to exclude them at "
        f"anchor-build time.{reset}",
        file=sys.stderr,
    )


def _install_bounded_decoder_cache(maxsize: int | None = None) -> None:
    """Replace lerobot's unbounded ``VideoDecoderCache`` with a bounded LRU.

    Lerobot's stock ``_default_decoder_cache`` is a plain ``dict`` that
    NEVER evicts — every distinct video path ever opened keeps a live
    ``torchcodec.VideoDecoder`` (+ its ``fsspec`` file handle, NVDEC
    ref-frame pool, pinned host buffer) for the lifetime of the worker.
    On large manifests (DaaS 2k: ~3k MP4 files; 5k: ~7k) and with
    ``persistent_workers=True`` this is an unbounded memory leak:
    host RAM and GPU VRAM climb monotonically until the worker is
    OOM-killed.

    The fix is bounded — keep at most ``maxsize`` decoders, evict the
    least-recently-used one on overflow (``OrderedDict.popitem(last=False)``
    after ``move_to_end`` on hit). The evicted decoder's file handle is
    closed before its Python reference is dropped, which releases the
    NVDEC context promptly (torchcodec's ``VideoDecoder.__del__`` waits
    on its internal decode queue drain, then frees GPU buffers).

    ``maxsize`` defaults to 32 — enough that a single training step's
    ~batch_size distinct MP4s fit without thrashing, but small enough
    that K × (NVDEC pool + pinned buffer) ≈ a few hundred MB per worker.
    Override via ``TAU0_DECODER_CACHE_SIZE`` env var if your per-
    step video diversity is unusual (e.g. long sequences sampling many
    episodes per batch).

    Concurrency model: the cache is safe for use within a single Python
    process (each torch DataLoader worker is a separate process with its
    own cache instance). Eager ``close()`` on eviction ASSUMES no other
    Python thread in the same process is still actively calling
    ``get_frames_at`` on the evicted decoder — if it were, closing the
    file handle out from under it would race. DataLoader workers are
    single-threaded Python, so this holds for the training use case.

    Idempotent: subsequent calls are no-ops once the patch is installed.
    """
    import multiprocessing
    import os

    if os.environ.get("TAU0_DISABLE_DECODER_CACHE", "").strip().lower() in ("1", "true", "yes"):
        return  # opt-out: fall back to lerobot's built-in _default_decoder_cache
    try:
        from lerobot.datasets import video_utils as _vu
    except ImportError:
        return  # lerobot not installed — nothing to patch
    if getattr(_vu, "_tau0_lru_decoder_cache_installed", False):
        return

    from collections import OrderedDict
    from threading import Lock

    # fsspec is a hard dependency of lerobot (`lerobot.datasets.video_utils`
    # imports it at module top). If the lerobot import above succeeded,
    # fsspec is guaranteed available — no defensive try/except needed.
    import fsspec

    resolved = maxsize
    if resolved is None:
        env = os.environ.get("TAU0_DECODER_CACHE_SIZE", "").strip()
        resolved = int(env) if env else 32

    class _LRUVideoDecoderCache:
        def __init__(self, maxsize: int):
            self.maxsize = int(maxsize)
            self._cache: "OrderedDict[str, tuple[Any, Any]]" = OrderedDict()
            self._lock = Lock()

        def get_decoder(self, video_path: str):
            from torchcodec.decoders import VideoDecoder

            path = str(video_path)
            with self._lock:
                hit = self._cache.get(path)
                if hit is not None:
                    self._cache.move_to_end(path)
                    return hit[0]
                # Evict LRU BEFORE allocating new decoder so peak stays at
                # maxsize+1 rather than maxsize*2 during the transient.
                if len(self._cache) >= self.maxsize:
                    _evicted_path, (evicted_dec, evicted_fh) = self._cache.popitem(last=False)
                    try:
                        evicted_fh.close()
                    except Exception:
                        pass
                    del evicted_dec
                file_handle = fsspec.open(path).__enter__()
                decoder = VideoDecoder(file_handle, seek_mode="approximate")
                self._cache[path] = (decoder, file_handle)
                return decoder

        def clear(self):
            with self._lock:
                for _dec, fh in self._cache.values():
                    try:
                        fh.close()
                    except Exception:
                        pass
                self._cache.clear()

        def size(self) -> int:
            with self._lock:
                return len(self._cache)

    _vu._default_decoder_cache = _LRUVideoDecoderCache(maxsize=resolved)
    _vu._tau0_lru_decoder_cache_installed = True
    # Only the main process prints the install banner. Each DataLoader worker
    # also runs this install (each has its own per-process cache), but 64
    # identical log lines swamp the training output — suppress everywhere
    # except ``MainProcess``.
    if multiprocessing.current_process().name == "MainProcess":
        print(
            f"[tau0-vla] installed bounded LRU decoder cache (maxsize={resolved}) "
            f"over lerobot.datasets.video_utils._default_decoder_cache",
            flush=True,
        )


def _project_field_value(
    *,
    descriptions: Mapping[str, Any],
    field_key: str | list[str],
    raw_values: np.ndarray,
) -> np.ndarray:
    if isinstance(field_key, list):
        return np.concatenate(
            [np.asarray(descriptions[key].project(raw_values), dtype=np.float32) for key in field_key],
            axis=-1,
        )
    return np.asarray(descriptions[field_key].project(raw_values), dtype=np.float32)


def _canonicalize_image(value: Any) -> np.ndarray:
    """Convert a lerobot raw image sample into canonical HWC uint8.

    Strict contract — no autodetect, no fallback:

    - torch.Tensor-like (duck-typed via ``detach/cpu/numpy``): rank-3 CHW,
      floating-point, values in ``[0, 1]`` (lerobot video-decode default).
    - ``np.ndarray``: rank-3 HWC, ``uint8`` (already canonical).

    Anything else — CHW uint8 numpy, float ``np.ndarray``, HWC float tensor,
    ambiguous-range float — raises. We don't guess the caller's intent.
    """
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        if value.ndim != 3 or value.shape[0] not in (1, 3):
            raise ValueError(f"tensor image must be CHW rank-3 (lerobot default); got shape={tuple(value.shape)}")
        arr = value.detach().cpu().numpy()
        if not np.issubdtype(arr.dtype, np.floating):
            raise TypeError(f"tensor image must be floating-point in [0, 1]; got dtype={arr.dtype}")
        arr = np.moveaxis(arr, 0, -1)
        return np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if isinstance(value, np.ndarray):
        if value.dtype != np.uint8:
            raise TypeError(f"ndarray image must be uint8; got dtype={value.dtype}")
        if value.ndim != 3 or value.shape[-1] not in (1, 3):
            raise ValueError(f"ndarray image must be HWC rank-3; got shape={value.shape}")
        return value
    raise TypeError(f"image must be torch.Tensor or np.ndarray; got {type(value).__name__}")


def _default_action_key(components: Sequence[ComponentSpec]) -> str:
    return "action_eef" if any(component.key == "eef_pose" for component in components) else "action"


def _default_state_key(components: Sequence[ComponentSpec]) -> str:
    return (
        "observation.state_eef" if any(component.key == "eef_pose" for component in components) else "observation.state"
    )


def _read_repo_manifest(path: str) -> list[str]:
    """Read a ``.txt`` manifest of repo_ids, one per line.

    Skips blank lines and ``#``-prefixed comment lines (after lstrip). Raises
    ``FileNotFoundError`` if the manifest itself is missing and ``ValueError``
    if the file is present but yields zero usable lines — either case is a
    configuration error worth failing early on.
    """
    manifest = Path(path)
    if not manifest.exists():
        raise FileNotFoundError(f"repo_id manifest not found: {manifest}")
    repos = [
        line.strip() for line in manifest.read_text().splitlines() if line.strip() and not line.strip().startswith("#")
    ]
    if not repos:
        raise ValueError(f"repo_id manifest {manifest} contains no repo_ids")
    return repos


def _normalize_offsets(value: Any, *, dt: float, key: str) -> list[float]:
    """Convert a user ``temporal_sequence`` value to an explicit list of seconds.

    Accepted forms:
      - ``int N``           → past-N-frames shorthand, yields ``[-(N-1), ..., 0]`` frames.
      - ``list[float|int]`` → offset list in **seconds**. Int elements are cast to float;
                              they are NOT interpreted as frame counts. Use
                              ``[i/fps for i in range(k)]`` if you want frame-spaced values.

    Negative = past, zero = current, positive = future. Irregular spacing OK.
    """
    if isinstance(value, bool):
        raise TypeError(f"temporal_sequence[{key!r}] must be int or list, got bool")
    if isinstance(value, int):
        if value < 1:
            raise ValueError(f"temporal_sequence[{key!r}] int form must be >= 1, got {value}")
        return [-(value - 1 - i) * dt for i in range(value)]
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"temporal_sequence[{key!r}] must be a non-empty list of offsets")
        return [float(v) for v in value]
    raise TypeError(f"temporal_sequence[{key!r}] must be int or list, got {type(value).__name__}")


def _resolve_temporal_key(user_key: str, camera_map: Mapping[str, str]) -> list[str]:
    """Map a user-facing key to LeRobot feature key(s).

    Accepted forms:
      ``state``, ``action``,
      ``image`` / ``images``               — shortcut for all cameras,
      ``image.<cam>`` / ``images.<cam>``   — per-camera,
      ``<cam>``                            — bare camera name.
    """
    # Normalize "image" ↔ "images" aliasing.
    if user_key == "image":
        user_key = "images"
    elif user_key.startswith("image."):
        user_key = "images" + user_key[len("image") :]

    if user_key == "state":
        return ["observation.state"]
    if user_key == "action":
        return ["action"]
    if user_key == "images":
        if not camera_map:
            raise ValueError("temporal_sequence['image(s)'] used but robot has no cameras")
        return list(camera_map.values())
    if user_key.startswith("images."):
        cam = user_key[len("images.") :]
    else:
        cam = user_key
    if cam in camera_map:
        return [camera_map[cam]]
    raise ValueError(
        f"Unknown temporal_sequence key {user_key!r}; expected one of "
        f"'state' / 'action' / 'image(s)' / 'image(s).<cam>' / '<cam>', "
        f"available cameras: {sorted(camera_map)}"
    )


class _RawSampleWrapper:
    """Module-level callable replacing `RobotConfig.build_pipeline`'s lambda;
    picklable so torch DataLoader multiprocessing workers can spawn with it."""

    __slots__ = ("_config", "_field_descriptions_by_dataset")

    def __init__(self, config: "RobotConfig", field_descriptions_by_dataset: Sequence[Mapping[str, Any]]):
        self._config = config
        self._field_descriptions_by_dataset = tuple(field_descriptions_by_dataset)

    def __call__(self, raw_sample: Mapping[str, Any]) -> dict[str, Any]:
        dataset_index = _raw_sample_dataset_index(raw_sample)
        if not (0 <= dataset_index < len(self._field_descriptions_by_dataset)):
            raise IndexError(
                f"raw sample dataset_index={dataset_index} out of range for "
                f"{len(self._field_descriptions_by_dataset)} field_descriptions entries"
            )
        field_descriptions = self._field_descriptions_by_dataset[dataset_index]
        return self._config._wrap_raw_sample(raw_sample, field_descriptions=field_descriptions)


def _raw_sample_dataset_index(raw_sample: Mapping[str, Any]) -> int:
    raw_index = raw_sample.get("dataset_index", 0)
    if hasattr(raw_index, "item"):
        raw_index = raw_index.item()
    return int(raw_index)


def _field_descriptions_by_repo_order(
    repo_ids: Sequence[str],
    *,
    fallback: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    """Load field descriptions in the same order as raw dataset ``dataset_index``.

    MultiLeRobotDataset can contain repos with compatible model-level outputs
    but different raw vector layouts. Runtime wrapping must therefore use the
    source repo's own field_descriptions instead of reusing the primary repo's
    schema for every child.
    """
    descriptions: list[Mapping[str, Any]] = []
    for repo_id in repo_ids:
        loaded = _load_field_descriptions(repo_id) if Path(repo_id).exists() else {}
        descriptions.append(loaded or fallback)
    if not descriptions:
        descriptions.append(fallback)
    return tuple(descriptions)


class _RangeFilteredDataset(torch.utils.data.Dataset):
    """Index-remapping wrapper that restricts ``inner`` to the global raw-frame
    ranges in ``ranges_arr``.

    Unlike the previous transparent-metadata wrapper (which existed only to
    feed openpi's now-removed ``FrameSampler.sample_subdataset``), this class
    actually remaps: ``len(self) == sum(e - s for s, e in ranges_arr)``, and
    ``self[i]`` maps ``i ∈ [0, len(self))`` to a raw frame index via
    ``map_selected_index``. Used for segment filtering.
    """

    def __init__(self, inner: Any, ranges_arr: "np.ndarray", cumsum: "np.ndarray"):
        from tau0_vla.data.sampler import map_selected_index  # noqa: PLC0415

        self._inner = inner
        self._dataset = inner  # some external tools descend via ``._dataset``
        self._ranges = ranges_arr
        self._cumsum = cumsum
        self._map_selected_index = map_selected_index
        # Copy through common lerobot metadata so downstream code that peeks
        # (e.g. ``.episode_data_index``, ``.meta``) keeps working on the wrapper.
        for attr in ("episode_data_index", "episodes", "meta", "features", "repo_id"):
            if hasattr(inner, attr):
                try:
                    setattr(self, attr, getattr(inner, attr))
                except Exception:
                    pass

    def __len__(self) -> int:
        return int(self._cumsum[-1]) if self._cumsum.size else 0

    def __getitem__(self, i: int) -> Any:
        raw_idx = self._map_selected_index(self._ranges, self._cumsum, int(i))
        return self._inner[int(raw_idx)]

    def __getattr__(self, name: str):  # noqa: D401 — passthrough to inner
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._inner, name)


def _align_self_to_raw_dataset(
    config: "RobotConfig",
    raw_dataset: Any,
    repo_ids: list[str],
) -> tuple["RobotConfig", list[str]]:
    """Rebind ``config.repo_id`` to whatever order ``raw_dataset`` actually used.

    ``MultiLeRobotDataset.__getitem__`` injects ``raw["dataset_index"]``
    relative to its own ``_datasets`` list. If LeRobot reordered or deduped
    vs the manifest, the manifest order is no longer the truth — per-sample
    repo_id lookups (``_sample_repo_id`` → ``_repo_id_list()``) and per-repo
    filter iteration (``_compute_filter_ranges``: ``zip(dataset_metas built
    from repo_ids, _per_repo_child_datasets(raw_dataset))``) must use
    ``raw_dataset.repo_ids`` instead, or instruction_segments lookups land
    on the wrong repo's ``meta/info.json``.

    Single-repo path (``LeRobotDataset``, no ``.repo_ids``) → no-op.
    """
    actual = getattr(raw_dataset, "repo_ids", None)
    if actual is None:
        return config, repo_ids
    actual_list = list(actual)
    if actual_list == list(repo_ids):
        return config, repo_ids
    config_name = getattr(config, "_finch_config_name", None)
    rebound = dataclasses.replace(config, repo_id=actual_list)
    if config_name is not None:
        object.__setattr__(rebound, "_finch_config_name", config_name)
    return rebound, actual_list


def _lerobot_local_repo_args(
    repo_ids: Sequence[str],
    selected_episodes: Any,
    source_root: str | None,
) -> tuple[list[str], str | None, Any, dict[str, str] | None]:
    """Return LeRobot-compatible repo/root args for local absolute datasets.

    LeRobotDataset accepts a local dataset as ``repo_id=<dataset name>,
    root=<dataset path>`` for a single repo. MultiLeRobotDataset accepts a
    shared ``root=<parent>`` and relative ``repo_ids=[<dataset names>]``.
    Passing absolute paths as repo ids makes LeRobot fall back to HuggingFace
    Hub validation after a local-cache miss.
    """
    if source_root is not None:
        return list(repo_ids), source_root, selected_episodes, None

    paths = [Path(str(repo_id)) for repo_id in repo_ids]
    if not paths or not all(path.is_absolute() and path.exists() for path in paths):
        return list(repo_ids), source_root, selected_episodes, None

    if len(paths) == 1:
        alias = paths[0].name
        return [alias], str(paths[0]), selected_episodes, {str(repo_ids[0]): alias}

    parents = {path.parent for path in paths}
    if len(parents) != 1:
        return list(repo_ids), source_root, selected_episodes, None

    aliases = [path.name for path in paths]
    alias_by_repo = {str(repo_id): alias for repo_id, alias in zip(repo_ids, aliases, strict=True)}
    if isinstance(selected_episodes, Mapping):
        episodes = {
            alias_by_repo[str(repo_id)]: selected_episodes[str(repo_id)]
            for repo_id in repo_ids
            if str(repo_id) in selected_episodes
        }
    else:
        episodes = selected_episodes
    return aliases, str(paths[0].parent), episodes, alias_by_repo


def _per_repo_child_datasets(raw_dataset: Any, repo_ids: Sequence[str]) -> list[Any]:
    """Unwrap ``raw_dataset`` into the per-repo leaf list the sampler helpers
    need (single-repo: ``[raw_dataset]``; multi-repo: ``MultiLeRobotDataset._datasets``)."""
    if len(repo_ids) == 1:
        return [raw_dataset]
    inner_list = getattr(raw_dataset, "_datasets", None)
    if inner_list is None or len(inner_list) != len(repo_ids):
        raise RuntimeError(
            "filter path: expected MultiLeRobotDataset with "
            f"{len(repo_ids)} children, got {type(raw_dataset).__name__} "
            f"with {'no' if inner_list is None else len(inner_list)} children"
        )
    return list(inner_list)


def _build_cumsum(ranges_arr: "np.ndarray") -> "np.ndarray":
    if ranges_arr.size == 0:
        return np.array([], dtype=np.int64)
    lengths = ranges_arr[:, 1] - ranges_arr[:, 0]
    return np.cumsum(lengths, dtype=np.int64)


def _compute_filter_ranges(
    raw_dataset: Any,
    config: "RobotConfig",
    repo_ids: Sequence[str],
) -> "tuple[np.ndarray, np.ndarray] | None":
    """Return ``(ranges_arr, cumsum)`` for the annotation filter, or ``None``
    when it is not active."""
    from tau0_vla.data.sampler import build_segment_index_ranges  # noqa: PLC0415

    frame_policy = config.frame_filter_policy()
    if frame_policy is None:
        return None

    datasets = _per_repo_child_datasets(raw_dataset, repo_ids)

    from tau0_vla.data.source import LeRobotDatasetMetadata  # noqa: PLC0415

    dataset_metas = [LeRobotDatasetMetadata(root=r, repo_id=Path(r).name) for r in repo_ids]
    ranges, _ = build_segment_index_ranges(
        dataset_metas=dataset_metas,
        datasets=datasets,
        tail_buffer=max(0, (config.action_horizon or 1) - 1),
        positive_labels=frame_policy.positive,
        negative_labels=frame_policy.negative,
    )
    return ranges, _build_cumsum(ranges)
