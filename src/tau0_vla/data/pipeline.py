from __future__ import annotations

import copy
import dataclasses
import os
import random
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from tau0_vla.data.config import get_config
from tau0_vla.data.modalities import ComponentSpec, WorkflowContext
from tau0_vla.data.modalities.base import normalize_field_description_map
from tau0_vla.data.modalities.image.resize_ops import resize_with_pad_numpy
from tau0_vla.data.sampler import map_selected_index, select_physical_shard
from tau0_vla.data.source import LeRobotDataset, LeRobotDatasetMetadata, _attach_episode_indices, _require_backend
from tau0_vla.data.stats import NormStats, load_dir_for_summary, normalize_array, unnormalize_array


@dataclasses.dataclass(frozen=True)
class DatasetPipeline:
    dataset: torch.utils.data.Dataset
    name: str | None = None
    transforms: Sequence | None = None
    action_postprocess: Any = None


class FinchDataLoader:
    def __init__(
        self,
        source: torch.utils.data.Dataset | Any,
        *,
        batch_size: int | None = None,
        sampler: torch.utils.data.Sampler | None = None,
        batch_sampler: torch.utils.data.Sampler | None = None,
        num_workers: int = 0,
        collate_fn=None,
        shuffle: bool = False,
        overwrite: bool = False,
        physically_split: bool = False,
        physical_rank: int | None = None,
        physical_world_size: int | None = None,
        video_backend: str | None = None,
    ):
        self.config_name: str | None = None
        self._source_config: Any | None = None
        self.video_backend = video_backend
        # Exposed as an attribute so downstream trainers (e.g. tau-0-vla's
        # accelerator.prepare decision at trainer/train.py:385-399) can read
        # it via ``getattr(loader, "physically_split", False)`` and skip the
        # DistributedSampler wrap that would otherwise double-shard.
        self.physically_split = bool(physically_split)
        self.physical_rank = physical_rank
        self.physical_world_size = physical_world_size
        if batch_sampler is None and sampler is None and hasattr(source, "build_pipeline"):
            # Single-route only: one Robot Config in, one pipeline out. The
            # multi-dataset mixing routes (``build_loader_spec`` /
            # ``build_pipelines``, both DataMix-only) are gone: the pipeline
            # ships a single training route.
            built = build_loader_from_pipelines(
                [
                    source.build_pipeline(
                        overwrite_norm_stats=overwrite,
                        physically_split=self.physically_split,
                        physical_rank=self.physical_rank,
                        physical_world_size=self.physical_world_size,
                        video_backend=self.video_backend,
                    )
                ],
                batch_size=batch_size or 1,
                num_workers=num_workers,
                shuffle=shuffle,
                collate_fn=collate_fn,
            )
            self._dataset = built.dataset
            self._loader = built._loader
            self._source_config = source
            return

        dataset = source
        self._dataset = dataset
        loader_kwargs = {
            "dataset": dataset,
            "num_workers": num_workers,
            "collate_fn": collate_fn or finch_collate,
        }
        if batch_sampler is not None:
            loader_kwargs["batch_sampler"] = batch_sampler
        else:
            loader_kwargs["batch_size"] = batch_size
            loader_kwargs["sampler"] = sampler
        self._loader = torch.utils.data.DataLoader(**loader_kwargs)

    def __iter__(self) -> Iterator[Any]:
        yield from self._loader

    def __len__(self) -> int:
        return len(self._loader)

    @property
    def dataset(self) -> torch.utils.data.Dataset:
        return self._dataset

    def save_data_spec(
        self,
        run_dir: str | Path,
        *,
        config_modules: tuple[str, ...] = (),
        vlm_model_type: str = "qwen2.5vl",
        vlm_aux_transforms: tuple[dict[str, Any], ...] = (),
        cam_keys: tuple[str, ...] | None = None,
        cam_view_template: str | None = None,
        cam_view_names: dict[str, str] | None = None,
        max_images_per_sample: int | None = None,
    ) -> Path:
        from tau0_vla.data.data_spec import save_data_spec

        if self._source_config is not None:
            return save_data_spec(
                self._source_config,
                run_dir,
                config_modules=config_modules,
                vlm_model_type=vlm_model_type,
                vlm_aux_transforms=vlm_aux_transforms,
                cam_keys=cam_keys,
                cam_view_template=cam_view_template,
                cam_view_names=cam_view_names,
                max_images_per_sample=max_images_per_sample,
            )
        if self.config_name is not None:
            return save_data_spec(
                self.config_name,
                run_dir,
                config_modules=config_modules,
                vlm_model_type=vlm_model_type,
                vlm_aux_transforms=vlm_aux_transforms,
                cam_keys=cam_keys,
                cam_view_template=cam_view_template,
                cam_view_names=cam_view_names,
                max_images_per_sample=max_images_per_sample,
            )
        raise TypeError("save_data_spec is only available when FinchDataLoader was created from a Finch config")

    @classmethod
    def from_config_name(
        cls,
        config_name: str,
        *,
        batch_size: int | None = None,
        num_workers: int = 0,
        shuffle: bool = False,
        collate_fn=None,
        overwrite: bool = False,
        physically_split: bool = False,
        physical_rank: int | None = None,
        physical_world_size: int | None = None,
        video_backend: str | None = None,
    ) -> "FinchDataLoader":
        config = get_config(config_name)
        loader = cls(
            config,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            collate_fn=collate_fn,
            overwrite=overwrite,
            physically_split=physically_split,
            physical_rank=physical_rank,
            physical_world_size=physical_world_size,
            video_backend=video_backend,
        )
        loader.config_name = config_name
        return loader


def _apply_physical_sharding_to_repos(
    repo_ids: "list[str]",
    *,
    dataset_metas: "list[Any] | None" = None,
    physical_rank: int | None = None,
    physical_world_size: int | None = None,
) -> "tuple[list[str], list[Any]]":
    """Slice ``repo_ids`` so each rank owns a disjoint subset, balanced by frame count.

    Preserves the legacy loader's physical-sharding contract: the same LPT
    bin-pack via :func:`select_physical_shard`, the same frame-count
    fallback (``meta.total_episodes * 100`` when ``total_frames`` is missing),
    same ``[PhysicalShard rank=...]`` log/error wording. The two pipeline
    entry paths (the RobotConfig build path and
    ``RobotConfig.build_pipeline(physically_split=True)``) both route through
    here so behavior stays bit-aligned regardless of which loader the user
    instantiates.

    ``dataset_metas`` is optional: a caller that already holds a metadata
    object per repo passes them in to avoid double-IO. The
    ``RobotConfig.build_pipeline`` path passes ``None``, and we open metas
    locally with :class:`LeRobotDatasetMetadata`.

    Returns ``(sliced_repo_ids, sliced_dataset_metas)``. When sharding is a
    no-op (``world_size <= 1`` or ``len(repo_ids) <= 1``) returns the inputs
    unchanged.
    """
    if physical_rank is not None or physical_world_size is not None:
        if physical_rank is None or physical_world_size is None:
            raise ValueError("physical_rank and physical_world_size must be provided together")
        world_size = int(physical_world_size)
        global_rank = int(physical_rank)
        if world_size <= 0:
            raise ValueError(f"physical_world_size must be positive, got {world_size}")
        if not (0 <= global_rank < world_size):
            raise ValueError(f"physical_rank must be in [0, {world_size}), got {global_rank}")
    elif torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        global_rank = torch.distributed.get_rank()
    else:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        global_rank = int(os.environ.get("RANK", "0"))

    total_datasets = len(repo_ids)
    if not (world_size > 1 and total_datasets > 1):
        # No-op: world_size==1 or single-repo. Match baseline's skip-print
        # gating (dataset_lerobot.py:749-755) — only rank 0 emits, otherwise
        # multi-rank runs duplicate the same line N times. Skip metadata
        # loading entirely; the LPT branch is the only consumer of metas.
        if global_rank == 0:
            print(
                f"[PhysicalShard rank={global_rank}] Skipped: world_size={world_size}, datasets={total_datasets}",
                flush=True,
            )
        return list(repo_ids), list(dataset_metas) if dataset_metas is not None else []

    if dataset_metas is None:
        dataset_metas = [LeRobotDatasetMetadata(root=r, repo_id=Path(r).name) for r in repo_ids]
    total_frames_list = [getattr(meta, "total_frames", meta.total_episodes * 100) for meta in dataset_metas]
    if total_datasets < world_size:
        # More ranks than repos in this leaf: a disjoint bin-pack would leave the
        # high ranks empty AND globally UNDER-WEIGHT this leaf (only
        # ``total_datasets`` ranks would ever sample it, so its effective mix
        # proportion would drop to w_L * total_datasets/world_size). Replicate the
        # whole leaf to EVERY rank instead — identical to the single-repo branch
        # above — so every rank samples it at its configured weight and the mix
        # proportion is preserved exactly. Cheap: these are small leaves by
        # definition (< world_size repos). The per-rank repo SET grows for these
        # small leaves only; the sampler logic and per-embodiment frequencies are
        # unchanged. Big leaves (>= world_size) still shard disjointly below.
        if global_rank == 0:
            print(
                f"[PhysicalShard rank={global_rank}/{world_size}] Replicated leaf "
                f"({total_datasets} datasets < world_size) to every rank "
                f"(preserves mix weight; would otherwise leave ranks empty).",
                flush=True,
            )
        return list(repo_ids), list(dataset_metas)
    shard = select_physical_shard(total_frames_list, world_size, global_rank, repo_ids, dataset_metas)
    if not shard["indices"]:
        raise ValueError(
            f"[PhysicalShard rank={global_rank}] No datasets assigned. "
            f"world_size={world_size} > total_datasets={total_datasets}. "
            f"Reduce world_size or disable physically_split."
        )
    sliced_repos = list(shard["aligned"][0])
    sliced_metas = list(shard["aligned"][1])
    total_frames_sum = sum(total_frames_list)
    print(
        f"[PhysicalShard rank={global_rank}/{world_size}] "
        f"Assigned {len(sliced_repos)}/{total_datasets} datasets, "
        f"{shard['sum']}/{total_frames_sum} frames "
        f"({shard['sum'] / total_frames_sum * 100:.1f}%)",
        flush=True,
    )
    return sliced_repos, sliced_metas


class MappedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset, transforms: Sequence | None = None):
        self._dataset = dataset
        self._transform = _compose_transforms(transforms or [])

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int | tuple[int, int]) -> dict[str, Any]:
        try:
            return self._transform(self._dataset[index])
        except _MappedDatasetFailure:
            # Inner MappedDataset already tagged the repo; don't double-wrap.
            raise
        except Exception as e:
            repo_id = _resolve_repo_for_index(self._dataset, index)
            where = f"repo={repo_id}" if repo_id else "repo=unresolved"
            raise _MappedDatasetFailure(
                f"MappedDataset[{index}] failed ({where}): {type(e).__name__}: {e}"
            ) from e


class _MappedDatasetFailure(RuntimeError):
    """Diagnostic re-raise from MappedDataset.__getitem__. Subclass so the
    outer MappedDataset wrapper (RobotConfig.build_pipeline stacks two of
    them) can pass the inner one through unchanged."""


def _resolve_repo_for_index(dataset: Any, index: Any) -> str | None:
    # Diagnostic only: map a failing MappedDataset index back to the source
    # repo path. Descends through nested MappedDataset layers and the
    # _RangeFilteredDataset(MultiLeRobotDataset) stack that
    # RobotConfig.build_pipeline sets up. Never in the hot path.
    try:
        idx = int(index) if not isinstance(index, tuple) else int(index[0])
    except Exception:
        return None
    inner = dataset
    for _ in range(8):
        if isinstance(inner, MappedDataset):
            inner = inner._dataset
            continue
        if hasattr(inner, "_ranges") and hasattr(inner, "_cumsum") and hasattr(inner, "_inner"):
            try:
                from tau0_vla.data.sampler import map_selected_index

                idx = int(map_selected_index(inner._ranges, inner._cumsum, idx))
            except Exception:
                # Without the remap, `idx` still indexes the *filtered* stream
                # while the repo walk below indexes the unfiltered one, so
                # continuing would name a plausible but wrong repo. A diagnostic
                # that lies is worse than one that abstains.
                return None
            inner = inner._inner
            continue
        break
    repo_ids = getattr(inner, "repo_ids", None)
    children = getattr(inner, "_datasets", None)
    if repo_ids and children:
        start = 0
        for repo, child in zip(repo_ids, children, strict=True):
            try:
                n = int(getattr(child, "num_frames", None) or len(child))
            except Exception:
                return str(repo)
            if idx < start + n:
                return str(repo)
            start += n
        return None
    single = getattr(inner, "repo_id", None)
    return str(single) if single else None


def finch_collate(batch: list[Any]) -> Any:
    if not batch:
        return batch

    elem = batch[0]
    if isinstance(elem, Mapping):
        keys = set().union(*(item.keys() for item in batch))
        result: dict[str, Any] = {}
        for key in keys:
            if all(key in item for item in batch):
                result[key] = finch_collate([item[key] for item in batch])
            else:
                result[key] = [item.get(key) for item in batch]
        return result

    if isinstance(elem, torch.Tensor):
        same_shape = all(isinstance(item, torch.Tensor) and item.shape == elem.shape for item in batch)
        return default_collate(batch) if same_shape else list(batch)

    if isinstance(elem, np.ndarray):
        same_shape = all(
            isinstance(item, np.ndarray) and item.shape == elem.shape and item.dtype == elem.dtype for item in batch
        )
        return default_collate(batch) if same_shape else list(batch)

    if isinstance(elem, (list, tuple)):
        try:
            return default_collate(batch)
        except Exception:
            return list(batch)

    try:
        return default_collate(batch)
    except Exception:
        return list(batch)


class Repack:
    def __init__(self, structure: Mapping[str, Any]):
        self.structure = dict(structure)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        raw = sample.get("raw", {})
        repacked = _resolve_structure(self.structure, raw)
        repacked["_field_descriptions"] = copy.deepcopy(sample.get("meta", {}).get("field_descriptions", {}))
        return repacked


class _CanonicalizeImages:
    """Normalize raw lerobot image tensors to canonical HWC uint8 ndarray.

    Always runs as the pipeline's first image-facing transform when the config
    declares no per-camera ``images`` specs. Downstream consumers can therefore
    rely on a single image format.
    """

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        from tau0_vla.data.robots.base import _canonicalize_image  # local: avoid import cycle

        updated = copy.deepcopy(sample)
        images = updated.get("images", {})
        updated["images"] = {key: _canonicalize_image(value) for key, value in images.items()}
        return updated


class _ImagePreprocessFromSpecs:
    """Per-camera training-time pipeline driven by ``RobotConfig.images=[Image(...)]``.

    Mirrors the deploy-side ``RobotConfig.resolve_images`` resolution: each
    camera's array is canonicalized to HWC uint8 then passed through that
    camera's ``Image.transforms`` chain (e.g. ``ColorJitter`` →
    ``ResizeWithPad``). Random transforms (``ColorJitter`` / ``Random*``)
    apply at train time and swap to ``eval_fallback()`` (typically
    ``Identity``) when ``train=False``, so train/eval pixels are byte-equal
    on the deterministic part of the chain.

    Cameras present in the dataset but absent from the spec list pass
    through canonicalized — same lenient policy ``_CanonicalizeImages``
    uses, useful for configs that only declare augmentation on a subset of
    cameras. Cameras present in the spec list but missing from the dataset
    are silently skipped (the assembler wouldn't see them anyway).

    This is the augmentation surface — it replaces the legacy yaml-side
    ``data_args.transforms: [{type: ColorJitterImages}]`` flow for image
    augmentations: per-camera, declarative, and deploy-symmetric. Selected
    whenever ``RobotConfig.images`` is non-None.
    """

    def __init__(self, *, images: "Sequence[Any]", train: bool = True):
        self._spec_by_name = {img.name: img for img in images}
        self._train = bool(train)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        updated = copy.deepcopy(sample)
        images = updated.get("images", {})
        if not isinstance(images, dict):
            return updated
        out: dict[str, Any] = {}
        from tau0_vla.data.modalities.image import SyntheticImage

        for cam_key, spec in self._spec_by_name.items():
            if cam_key in images:
                canonical = canonicalize_image_hwc(images[cam_key])
                if isinstance(spec, SyntheticImage):
                    out[cam_key] = np.asarray(spec.materialize())
                else:
                    out[cam_key] = np.asarray(spec.apply(canonical, train=self._train))
            elif isinstance(spec, SyntheticImage):
                out[cam_key] = np.asarray(spec.materialize())
        for cam_key, value in images.items():
            if cam_key in out:
                continue
            # Camera in dataset but not declared — pass through canonicalized.
            out[cam_key] = canonicalize_image_hwc(value)
        updated["images"] = out
        return updated


def resize_image_hwc(image: Any, target_size: tuple[int, int]) -> np.ndarray:
    """Aspect-preserving resize to ``target_size=(h, w)`` with black padding → HWC uint8 ndarray.

    Single entry point used by both training (``ResizeWithPad``, declared per
    camera on ``RobotConfig.images``) and deploy (``encode_payload``,
    which takes HWC uint8 / PIL images from a robot SDK). Both sides delegate
    to ``resize_with_pad_numpy`` (PIL bilinear + zero-pad), matching openpi's
    ``image_tools.resize_with_pad`` so the VLM processor sees byte-identical
    pixels at training and inference. uint8 inputs are padded with 0; float
    inputs are first canonicalized to uint8 [0, 255] then padded.

    Accepts:
      - ``torch.Tensor`` CHW with C ∈ {1, 3}, float (assumed [0, 1]) or uint8
      - ``np.ndarray`` HWC with C ∈ {1, 3}, uint8 or float (assumed [0, 1])
      - ``np.ndarray`` HW (grayscale, treated as single-channel)
      - ``PIL.Image.Image``
    """
    target_h, target_w = int(target_size[0]), int(target_size[1])
    canonical = canonicalize_image_hwc(image)
    return resize_with_pad_numpy(canonical, target_h, target_w)


def canonicalize_image_hwc(image: Any) -> np.ndarray:
    """HWC uint8 ndarray, no resize. Same input surface as ``resize_image_hwc``.

    Used by ``encode_payload`` when ``data_spec.target_size`` is None —
    just ensure the deploy payload matches training's post-``_CanonicalizeImages``
    format (HWC uint8). Fast path for already-canonical HWC uint8 ndarrays
    skips the float roundtrip.

    This is the pipeline's *internal* normalizer: inputs arrive from LeRobot in
    known layouts, so it is strict (raises on anything unexpected) and preserves
    single-channel images. Do not merge it with
    ``tau0_vla.vlm._image_utils.to_uint8_hwc``, which is a deliberately
    lenient adapter for ``PIL.Image.fromarray`` — see its docstring for the three
    places their behaviour diverges on purpose.
    """
    if isinstance(image, np.ndarray) and image.dtype == np.uint8:
        if image.ndim == 2:
            return image[..., None]
        if image.ndim == 3 and image.shape[-1] in (1, 3):
            return image
    chw_f = _to_chw_float01(image)
    arr = np.moveaxis(chw_f.detach().cpu().numpy(), 0, -1)
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def _to_chw_float01(image: Any) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        if image.ndim != 3 or image.shape[0] not in (1, 3):
            raise ValueError(f"CHW tensor must be rank-3 with C∈{{1,3}}; got shape={tuple(image.shape)}")
        if image.dtype.is_floating_point:
            return image.to(dtype=torch.float32)
        if image.dtype == torch.uint8:
            return image.to(dtype=torch.float32) / 255.0
        raise TypeError(f"Unsupported CHW tensor dtype: {image.dtype}")
    try:
        from PIL import Image as _PILImage  # noqa: PLC0415
    except ImportError:  # pragma: no cover — Pillow is a hard dep
        _PILImage = None  # type: ignore[assignment]
    if _PILImage is not None and isinstance(image, _PILImage.Image):
        arr = np.asarray(image)
    elif isinstance(image, np.ndarray):
        arr = image
    else:
        raise TypeError(f"Unsupported image type: {type(image).__name__}")
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3 or arr.shape[-1] not in (1, 3):
        raise ValueError(f"HWC image must be rank-3 with C∈{{1,3}}; got shape={arr.shape}")
    chw = np.ascontiguousarray(np.moveaxis(arr, -1, 0))
    tensor = torch.from_numpy(chw)
    if tensor.dtype == torch.uint8:
        return tensor.to(dtype=torch.float32) / 255.0
    return tensor.to(dtype=torch.float32)


def _concat_extras_per_form(
    extras_per_component: "dict[str, dict[str, np.ndarray | None]] | None",
    *,
    padding_dim: int | None,
    label: str,
) -> "dict[str, np.ndarray | None]":
    """Collapse per-component extras into one concatenated array per form.

    Input shape (assembler-internal): ``{component_key: {"raw": arr | None,
    "mean_std_norm": arr | None, "q_norm": arr | None}}``.

    Output shape (batch-facing): ``{"raw": ndarray, "mean_std_norm": ndarray,
    "q_norm": ndarray}`` — each is the per-component arrays concatenated on
    the last axis, then zero-padded to ``padding_dim`` on that same axis using
    the same op as the canonical ``state`` / ``action`` keys. Padding keeps the
    batch-facing dims fixed at ``padding_dim`` regardless of the config's native
    dims (e.g. G1 14D arm vs Franka 7D arm) — without it, ``finch_collate``
    would fall back to a ``list[ndarray]`` of mixed shapes.

    Components emitting ``None`` for a form are skipped during concat. If every
    component is None for a form, the output value is ``None`` (e.g. q_norm
    when norm_stats has no q01/q99) — no padding to do.

    A None input (return_all_norm_forms disabled) yields an empty dict so the
    caller can attach uniformly.
    """
    if extras_per_component is None:
        return {}
    forms = ("raw", "mean_std_norm", "q_norm")
    out: dict[str, np.ndarray | None] = {}
    for form in forms:
        parts = [sub[form] for sub in extras_per_component.values() if sub.get(form) is not None]
        if not parts:
            out[form] = None
            continue
        concatenated = np.concatenate([np.asarray(p) for p in parts], axis=-1)
        out[form] = _pad_vector(concatenated, padding_dim, label=f"{label}.{form}")
    return out


class ComponentAssembler:
    def __init__(
        self,
        *,
        state_components: Sequence[ComponentSpec],
        action_components: Sequence[ComponentSpec],
        norm_stats: dict[str, NormStats] | None = None,
        state_field_map: Mapping[str, str] | None = None,
        action_field_map: Mapping[str, str] | None = None,
        state_padding_dim: int | None = None,
        action_padding_dim: int | None = None,
        state_active_indices: Sequence[int] | None = None,
        action_active_indices: Sequence[int] | None = None,
        disable_component_normalization: bool = False,
        state_has_temporal_axis: bool = False,
        augment_fn: "Callable[..., tuple[np.ndarray, np.ndarray]] | None" = None,
        return_all_norm_forms: bool = False,
    ):
        self.state_components = tuple(state_components)
        self.action_components = tuple(action_components)
        self.norm_stats = norm_stats or {}
        self.state_field_map = dict(state_field_map or {})
        self.action_field_map = dict(action_field_map or {})
        self.state_padding_dim = state_padding_dim
        self.action_padding_dim = action_padding_dim
        self.state_active_indices = None if state_active_indices is None else tuple(int(i) for i in state_active_indices)
        self.action_active_indices = None if action_active_indices is None else tuple(int(i) for i in action_active_indices)
        self.disable_component_normalization = disable_component_normalization
        self.state_has_temporal_axis = state_has_temporal_axis
        self.augment_fn = augment_fn
        self.return_all_norm_forms = bool(return_all_norm_forms)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        updated = copy.deepcopy(sample)
        state_raw = np.asarray(updated.pop("_state_raw"), dtype=np.float32)
        action_raw = np.asarray(updated.pop("_action_raw"), dtype=np.float32)
        field_descriptions = updated.pop("_field_descriptions", {})
        state_descriptions = normalize_field_description_map(field_descriptions.get("state"))
        action_descriptions = normalize_field_description_map(field_descriptions.get("action"))

        if self.augment_fn is not None:
            state_raw, action_raw = self.augment_fn(
                state_raw=state_raw,
                action_raw=action_raw,
                state_descriptions=state_descriptions,
                action_descriptions=action_descriptions,
            )

        context = WorkflowContext(
            state=self._build_context_values(descriptions=state_descriptions, raw_values=state_raw, role="state"),
            action=self._build_context_values(descriptions=action_descriptions, raw_values=action_raw, role="action"),
            state_has_temporal_axis=self.state_has_temporal_axis,
        )
        state_components, state_extras_pc = self._assemble_components(
            components=self.state_components,
            context=context,
            role="state",
        )
        action_components, action_extras_pc = self._assemble_components(
            components=self.action_components,
            context=context,
            role="action",
        )

        state = _concatenate_components(state_components.values())
        action = _concatenate_components(action_components.values())
        result: dict[str, Any] = {
            "images": copy.deepcopy(updated.get("images", {})),
            "prompt": updated.get("prompt"),
            "state": _pad_vector(state, self.state_padding_dim, label="state"),
            "action": _pad_vector(action, self.action_padding_dim, label="action"),
            "state_mask": _active_indices_mask(
                state.shape[-1], self.state_padding_dim, self.state_active_indices, label="state_mask"
            ),
            "action_mask": _active_indices_mask(
                action.shape[-1], self.action_padding_dim, self.action_active_indices, label="action_mask"
            ),
        }
        if self.return_all_norm_forms:
            result["extras"] = {
                "state": _concat_extras_per_form(
                    state_extras_pc,
                    padding_dim=self.state_padding_dim,
                    label="state",
                ),
                "action": _concat_extras_per_form(
                    action_extras_pc,
                    padding_dim=self.action_padding_dim,
                    label="action",
                ),
            }
        return result

    def _build_context_values(
        self,
        *,
        descriptions: Mapping[str, Any],
        raw_values: np.ndarray,
        role: str,
    ) -> dict[str, np.ndarray]:
        values: dict[str, np.ndarray] = {}
        component_keys = {component.key for component in (*self.state_components, *self.action_components)}
        for key in component_keys:
            field_key = self._resolve_field_key(key=key, role=role)
            if field_key is None:
                continue
            if not _field_key_is_known(field_key, descriptions):
                # `_resolve_field_key` may synthesize a mirror (action<->state) for
                # components that live in only one role (e.g. chassis_velocity is
                # action-only). When the mirrored key isn't present in the dataset's
                # role-specific field_descriptions, skip rather than KeyError.
                continue
            values[key] = self._project_field_value(
                descriptions=descriptions, field_key=field_key, raw_values=raw_values
            )
        return values

    def _assemble_components(
        self,
        *,
        components: Sequence[ComponentSpec],
        context: WorkflowContext,
        role: str,
    ) -> "tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray | None]] | None]":
        assembled: dict[str, np.ndarray] = {}
        # Per-component ``{key: {"raw": ..., "mean_std_norm": ..., "q_norm": ...}}``
        # captured when ``return_all_norm_forms=True``; None otherwise so callers
        # can short-circuit cheaply. ``raw`` is the pre-component-transform
        # projection (post-``augment_raw_tensors`` but pre-``component.transform``)
        # — i.e. dataset-native coordinates (Quaternion / abs / ...). The two
        # normalized forms are computed on the SAME post-transform value the
        # canonical path uses, so their semantics mirror norm_stats; ``raw``
        # lives in a different coordinate system and may have a different last dim.
        extras: dict[str, dict[str, np.ndarray | None]] | None = {} if self.return_all_norm_forms else None
        stats_offset = 0
        context_values = context.state if role == "state" else context.action
        for component in components:
            value = np.asarray(component.transform(context, role=role, mode="pre"), dtype=np.float32)
            component_dim = int(value.shape[-1]) if value.ndim else 1
            stats_key = "state" if role == "state" else "action"
            stats_available = stats_key in self.norm_stats
            component_stats = (
                _slice_norm_stats(self.norm_stats[stats_key], stats_offset, component_dim) if stats_available else None
            )

            if extras is not None:
                raw_value = context_values.get(component.key)
                # ``raw_value`` may be missing when the component was synthesized
                # / mirrored across roles without a real field projection (the
                # ``_resolve_field_key`` fallback at L1004-1015). Emit None in
                # that case rather than crashing — the post-collate list-of-None
                # behavior (finch_collate L783-784) is the right multi-embodiment
                # signal for "this sample doesn't carry that component's raw".
                extras[component.key] = {
                    "raw": None if raw_value is None else np.asarray(raw_value, dtype=np.float32),
                    "mean_std_norm": (
                        normalize_array(value, component_stats, use_quantiles=False).astype(np.float32)
                        if component_stats is not None
                        else None
                    ),
                    # Legacy norm_stats predating the quantile compute have q01=None;
                    # gracefully degrade to None instead of raising — the user opted
                    # in to extras, not to a hard quantile contract.
                    "q_norm": (
                        normalize_array(value, component_stats, use_quantiles=True).astype(np.float32)
                        if component_stats is not None and component_stats.q01 is not None
                        else None
                    ),
                }

            if not self.disable_component_normalization and component.requires_normalization() and stats_available:
                value = normalize_array(value, component_stats, use_quantiles=component.use_quantile_norm()).astype(
                    np.float32
                )
            assembled[component.key] = value
            stats_offset += component_dim
        return assembled, extras

    def _resolve_field_key(self, *, key: str, role: str) -> str | list[str] | None:
        mapping = self.state_field_map if role == "state" else self.action_field_map
        if key in mapping:
            return mapping[key]

        prefix = f"{role}/"
        opposite_prefix = "action/" if role == "state" else "state/"
        for component in (*self.state_components, *self.action_components):
            if component.key != key or component.field_key is None:
                continue
            if component.field_key.startswith(prefix):
                return component.field_key
            if component.field_key.startswith(opposite_prefix):
                return prefix + component.field_key[len(opposite_prefix) :]
        return None

    def _project_field_value(
        self,
        *,
        descriptions: Mapping[str, Any],
        field_key: str | list[str],
        raw_values: np.ndarray,
    ) -> np.ndarray:
        if isinstance(field_key, list):
            arrays = []
            for key in field_key:
                if key not in descriptions:
                    raise KeyError(key)
                arrays.append(np.asarray(descriptions[key].project(raw_values), dtype=np.float32))
            return _concatenate_components(arrays)
        if field_key not in descriptions:
            raise KeyError(field_key)
        return np.asarray(descriptions[field_key].project(raw_values), dtype=np.float32)


class _ActionPostprocess:
    def __init__(
        self,
        *,
        state_components: Sequence[ComponentSpec],
        action_components: Sequence[ComponentSpec],
        state_component_dims: Sequence[int],
        action_component_dims: Sequence[int],
        norm_stats: dict[str, NormStats] | None = None,
        state_padding_dim: int | None = None,
        action_padding_dim: int | None = None,
        disable_component_normalization: bool = False,
        disable_state_component_normalization: bool | None = None,
        disable_action_component_normalization: bool | None = None,
    ):
        self.state_components = tuple(state_components)
        self.action_components = tuple(action_components)
        self.state_component_dims = tuple(int(dim) for dim in state_component_dims)
        self.action_component_dims = tuple(int(dim) for dim in action_component_dims)
        self.norm_stats = norm_stats or {}
        self.state_padding_dim = state_padding_dim
        self.action_padding_dim = action_padding_dim
        self.disable_component_normalization = disable_component_normalization
        self.disable_state_component_normalization = (
            disable_component_normalization
            if disable_state_component_normalization is None
            else disable_state_component_normalization
        )
        self.disable_action_component_normalization = (
            disable_component_normalization
            if disable_action_component_normalization is None
            else disable_action_component_normalization
        )

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        updated = copy.deepcopy(sample)
        state = self._trim_padding(
            np.asarray(updated["state"], dtype=np.float32), self.state_component_dims, self.state_padding_dim
        )
        actions = self._trim_padding(
            np.asarray(updated["actions"], dtype=np.float32),
            self.action_component_dims,
            self.action_padding_dim,
        )

        state_values = self._split_components(
            state,
            self.state_components,
            self.state_component_dims,
            stats_key="state",
            disable_component_normalization=self.disable_state_component_normalization,
        )
        action_values = self._split_components(
            actions,
            self.action_components,
            self.action_component_dims,
            stats_key="action",
            disable_component_normalization=self.disable_action_component_normalization,
        )
        context = WorkflowContext(state=state_values, action=action_values, state_already_transformed=True)

        restored_actions = [
            np.asarray(
                component.apply_transforms(action_values[component.key], context=context, mode="post"), dtype=np.float32
            )
            for component in self.action_components
        ]
        updated["actions"] = _concatenate_components(restored_actions)
        return updated

    def _split_components(
        self,
        values: np.ndarray,
        components: Sequence[ComponentSpec],
        component_dims: Sequence[int],
        *,
        stats_key: str,
        disable_component_normalization: bool,
    ) -> dict[str, np.ndarray]:
        split: dict[str, np.ndarray] = {}
        offset = 0
        for component, dim in zip(components, component_dims, strict=True):
            value = np.asarray(values[..., offset : offset + dim], dtype=np.float32)
            if not disable_component_normalization and component.requires_normalization():
                if stats_key not in self.norm_stats:
                    raise ValueError(
                        f"Component '{component.key}' (role={stats_key}) declares "
                        f"normalize={component.normalize!r} but no '{stats_key}' entry "
                        f"is available in norm_stats (have keys: {sorted(self.norm_stats)}). "
                        "Silently skipping un-normalization would leave outputs in "
                        "normalized space and silently inflate openloop MAE by ~5x. "
                        "Check that the checkpoint's finch_data_spec has the right "
                        "artifacts_dir (for a multi-route ckpt, the per-route subdir) or pass "
                        "disable_component_normalization=True if this is intentional."
                    )
                component_stats = _slice_norm_stats(self.norm_stats[stats_key], offset, dim)
                value = unnormalize_array(value, component_stats, use_quantiles=component.use_quantile_norm())
            split[component.key] = value.astype(np.float32)
            offset += dim
        return split

    @staticmethod
    def _trim_padding(values: np.ndarray, component_dims: Sequence[int], padding_dim: int | None) -> np.ndarray:
        del padding_dim
        expected_dim = int(sum(component_dims))
        if values.shape[-1] < expected_dim:
            raise ValueError(f"Expected at least {expected_dim} dims, got shape={values.shape}")
        return np.asarray(values[..., :expected_dim], dtype=np.float32)


def canonicalize_norm_stats(norm_stats: dict[str, NormStats]) -> dict[str, NormStats]:
    remapped: dict[str, NormStats] = {}
    aliases = {
        "inputs/state/robot": "state",
        "targets/action": "action",
    }
    for key, value in norm_stats.items():
        remapped[aliases.get(key, key)] = value
    return remapped


def load_norm_stats(
    norm_stats_dir: str | None,
    config: Any,
    *,
    overwrite: bool = False,
) -> dict[str, NormStats] | None:
    if norm_stats_dir is None:
        return None
    return canonicalize_norm_stats(load_dir_for_summary(norm_stats_dir, config, overwrite=overwrite))


def _build_image_transforms(
    *,
    images: "Sequence[Any] | None" = None,
) -> tuple:
    """Finchdata's image path always emits canonical HWC uint8 ndarray.

    - ``images=[Image(...)]``: per-camera transform chain (``ColorJitter`` /
      ``ResizeWithPad`` / ...) declared on the ``RobotConfig.images`` field.
      Train and deploy share the same Image spec, so pixels are byte-equal on
      the deterministic part of the chain.
    - ``images=None``: canonicalize only, keep native resolution.
    """
    if images is not None:
        return (_ImagePreprocessFromSpecs(images=images),)
    return (_CanonicalizeImages(),)


def _build_component_output_dims(
    *,
    components: Sequence[ComponentSpec],
    role: str,
    descriptions: Mapping[str, Any],
    mapping: Mapping[str, str | list[str]],
) -> tuple[int, ...]:
    dims: list[int] = []
    for component in components:
        field_key = mapping.get(component.key, component.field_key)
        if field_key is None:
            raise ValueError(f"Component '{component.key}' is missing field_key")
        raw_dim = _field_key_dim(field_key, descriptions)
        dummy_value = np.zeros(raw_dim, dtype=np.float32)
        dummy_context = WorkflowContext(
            state={component.key: dummy_value},
            action={component.key: dummy_value},
        )
        transformed = np.asarray(component.transform(dummy_context, role=role, mode="pre"), dtype=np.float32)
        dims.append(int(transformed.shape[-1]) if transformed.ndim else 1)
    return tuple(dims)


def _field_key_dim(field_key: str | list[str], descriptions: Mapping[str, Any]) -> int:
    if isinstance(field_key, list):
        return sum(_field_key_dim(key, descriptions) for key in field_key)
    description = descriptions[field_key]
    return int(getattr(description, "dimensions", len(getattr(description, "indices", ()))))


def _field_key_is_known(field_key: str | list[str], descriptions: Mapping[str, Any]) -> bool:
    if isinstance(field_key, list):
        return all(k in descriptions for k in field_key)
    return field_key in descriptions


def build_loader_from_pipelines(
    pipelines: Sequence[DatasetPipeline],
    *,
    batch_size: int = 1,
    num_workers: int = 0,
    shuffle: bool = False,
    collate_fn=None,
):
    if not pipelines:
        raise ValueError("At least one dataset pipeline is required")
    if len(pipelines) > 1:
        # Single-route only: the proportional batch sampler that
        # used to aggregate several pipelines is gone, so more than one here
        # would silently train on ``pipelines[0]`` alone.
        raise ValueError(
            f"Single-route training only: expected exactly one dataset pipeline, got {len(pipelines)}. "
            "Multi-dataset mixing (DataMix) was removed; implement your own sampler over this loader instead."
        )

    mapped_datasets = [MappedDataset(pipeline.dataset, transforms=pipeline.transforms) for pipeline in pipelines]
    return _build_loader_from_mapped_datasets(
        mapped_datasets,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )


def _unique_dataset_names(names: Sequence[str]) -> list[str]:
    counts: dict[str, int] = {}
    unique_names: list[str] = []
    for name in names:
        index = counts.get(name, 0)
        unique_name = name if index == 0 else f"{name}_{index}"
        counts[name] = index + 1
        unique_names.append(unique_name)
    return unique_names


# Public alias: stable for external callers that need to deduplicate a set of
# aggregated dataset names (openpi / tau-0-vla training data loaders).
unique_dataset_names = _unique_dataset_names


def assemble_state_array(
    state_payload: Mapping[str, np.ndarray],
    *,
    components: Sequence[ComponentSpec],
    norm_stats: "NormStats | None" = None,
    state_padding_dim: int | None = None,
) -> np.ndarray:
    """Flatten a per-component canonical state dict into a normalized + padded
    model-input state vector.

    The state-side counterpart to `restore_action` / `normalize_state`: takes
    an already-canonical ``{component_key: per_component_array}`` mapping
    (produced by `FinchDataLoader`'s canonical output), applies each
    component's pre-transform, per-component normalization, concatenates, and
    pads to ``state_padding_dim``.

    Used by training-time data-pipeline transforms that need to reproduce the
    training state-vector layout at openpi's ``_OpenPIFormat`` boundary.
    """
    context_state: dict[str, np.ndarray] = {}
    for component in components:
        if component.key not in state_payload:
            raise KeyError(f"Missing canonical Finch state component '{component.key}'")
        context_state[component.key] = np.asarray(state_payload[component.key], dtype=np.float32)

    context = WorkflowContext(state=context_state, action={})
    values: list[np.ndarray] = []
    offset = 0
    for component in components:
        value = np.asarray(component.transform(context, role="state", mode="pre"), dtype=np.float32)
        component_dim = int(value.shape[-1]) if value.ndim else 1
        if component.requires_normalization() and norm_stats is not None:
            component_stats = _slice_norm_stats(norm_stats, offset, component_dim)
            value = normalize_array(value, component_stats, use_quantiles=component.use_quantile_norm()).astype(
                np.float32
            )
        values.append(value)
        offset += component_dim
    state = _concatenate_components(values)
    return _pad_vector(state, state_padding_dim, label="state")


def _build_loader_from_mapped_datasets(
    mapped_datasets: Sequence[Dataset],
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    collate_fn,
):
    sampler = torch.utils.data.RandomSampler(mapped_datasets[0]) if shuffle else None
    return FinchDataLoader(
        mapped_datasets[0],
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def _compose_transforms(transforms: Sequence) -> Any:
    if not transforms:
        return _IdentityTransform()
    return _ComposedTransform(tuple(transforms))


class _IdentityTransform:
    """Picklable no-op transform (closures aren't picklable across workers)."""

    def __call__(self, sample: Any) -> Any:
        return sample


class _ComposedTransform:
    """Picklable chain of transforms — module-level class so torch DataLoader
    workers can spawn with it."""

    __slots__ = ("transforms",)

    def __init__(self, transforms: tuple):
        self.transforms = transforms

    def __call__(self, sample: Any) -> Any:
        value = sample
        for transform in self.transforms:
            value = transform(value)
        return value


def _resolve_structure(structure: Any, raw: Mapping[str, Any]) -> Any:
    if isinstance(structure, Mapping):
        return {key: _resolve_structure(value, raw) for key, value in structure.items()}
    if isinstance(structure, str):
        return _get_by_path(raw, structure)
    return copy.deepcopy(structure)


def _get_by_path(data: Mapping[str, Any], path: str) -> Any:
    if path in data:
        return data[path]
    value: Any = data
    for segment in path.split("/"):
        if not isinstance(value, Mapping) or segment not in value:
            raise KeyError(path)
        value = value[segment]
    return value


def _concatenate_components(values: Sequence[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(value, dtype=np.float32) for value in values]
    if not arrays:
        return np.asarray([], dtype=np.float32)
    if len(arrays) == 1:
        return arrays[0]
    return np.concatenate(arrays, axis=-1)


def _pad_vector(values: np.ndarray, padding_dim: int | None, *, label: str) -> np.ndarray:
    if padding_dim is None:
        return values
    current_dim = int(values.shape[-1]) if values.ndim else 0
    if current_dim > padding_dim:
        raise ValueError(f"{label} padding_dim={padding_dim} is smaller than output dim={current_dim}")
    if current_dim == padding_dim:
        return values
    pad_width = [(0, 0)] * values.ndim
    pad_width[-1] = (0, padding_dim - current_dim)
    return np.pad(values, pad_width, mode="constant", constant_values=0.0).astype(np.float32)


def _component_mask(active_dim: int, padding_dim: int | None, *, label: str) -> np.ndarray:
    output_dim = int(active_dim) if padding_dim is None else int(padding_dim)
    if active_dim > output_dim:
        raise ValueError(f"{label} padding_dim={output_dim} is smaller than active dim={active_dim}")
    mask = np.zeros(output_dim, dtype=np.float32)
    mask[:active_dim] = 1.0
    return mask


def _active_indices_mask(
    component_dim: int,
    padding_dim: int | None,
    active_indices: Sequence[int] | None,
    *,
    label: str,
) -> np.ndarray:
    if active_indices is None:
        return _component_mask(component_dim, padding_dim, label=label)
    output_dim = int(component_dim) if padding_dim is None else int(padding_dim)
    if component_dim > output_dim:
        raise ValueError(f"{label} padding_dim={output_dim} is smaller than component dim={component_dim}")
    indices = tuple(int(index) for index in active_indices)
    invalid = [index for index in indices if index < 0 or index >= output_dim]
    if invalid:
        raise ValueError(f"{label} contains indices outside [0, {output_dim - 1}]: {invalid}")
    if len(indices) != len(set(indices)):
        raise ValueError(f"{label} contains duplicate active indices: {indices}")
    mask = np.zeros(output_dim, dtype=np.float32)
    mask[list(indices)] = 1.0
    return mask


def _slice_norm_stats(stats: NormStats, offset: int, width: int) -> NormStats:
    end = offset + width
    return NormStats(
        mean=stats.mean[..., offset:end],
        std=stats.std[..., offset:end],
        q01=None if stats.q01 is None else stats.q01[..., offset:end],
        q99=None if stats.q99 is None else stats.q99[..., offset:end],
    )


__all__ = [
    "ComponentAssembler",
    "FinchDataLoader",
    "MappedDataset",
    "Repack",
    "assemble_state_array",
    "unique_dataset_names",
]
