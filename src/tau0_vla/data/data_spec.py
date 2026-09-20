from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any

import numpy as np

from tau0_vla.data.config import discover_config_modules, get_config
from tau0_vla.data.modalities.base import normalize_field_description_map, serialize_field_description_map
from tau0_vla.data.robot import resolve_finch_robot_configs
from tau0_vla.data.robots import RobotConfig, _get_robot_class
from tau0_vla.data.stats import NormStats, unnormalize_array

_logger = logging.getLogger(__name__)


FORMAT_VERSION = 2
# FROZEN. "finch" is the pre-rename project name; this string is the directory
# name inside every checkpoint ever written, and `load_data_spec` locates the
# data contract by it. Renaming it orphans every trained checkpoint, so it
# stays as-is and the README explains the name rather than changing it.
DATA_SPEC_DIR = "finch_data_spec"
DATA_SPEC_FILENAME = "spec.json"
NORM_STATS_FILENAME = "norm_stats.json"
FIELD_DESCRIPTIONS_FILENAME = "field_descriptions.json"
COMPONENTS_FILENAME = "components.json"


@dataclass(frozen=True)
class FinchDataSpec:
    format_version: int
    robot_name: str | None
    finch_config_name: str | None
    finch_config_names_by_root: dict[str, str]
    config_modules: tuple[str, ...]
    action_key_forward: str
    state_key_forward: str
    action_dim: int
    state_dim: int
    action_chunk_size: int
    cam_keys: tuple[str, ...]
    target_size: tuple[int, int] | None
    synthetic_images: tuple[tuple[str, str], ...]
    use_quantile: bool
    norm_stats_path: str | None
    vlm_model_type: str
    vlm_aux_transforms: tuple[dict[str, Any], ...]
    is_eef: bool
    eef_state_mode: str
    eef_action_mode: str
    action_space: str | None
    prompt_template: str
    robot_cls: type
    norm_stats: dict[str, Any]
    artifacts_dir: str | None = None
    cam_view_template: str | None = None
    cam_view_names: dict[str, str] | None = None
    max_images_per_sample: int | None = None
    # Component lists + per-component output dims — persisted under
    # ``finch_data_spec/<route>/components.json`` so deploy can rebuild the
    # pipeline (state assembler + action postprocessor) without calling
    # ``@register_config`` or field_description lookups. Empty for legacy
    # checkpoints that predate the components.json artifact.
    state_components: tuple[Any, ...] = ()
    action_components: tuple[Any, ...] = ()
    state_component_dims: tuple[int, ...] = ()
    action_component_dims: tuple[int, ...] = ()
    # Per-component → field_key mappings.  Persisted in components.json so
    # deploy can rebuild the ComponentAssembler without the live config.
    # Values are either a plain str (single field_key) or a list[str] (multi-
    # field composite like gripper → [left_effector, right_effector]).
    state_field_map: tuple[tuple[str, Any], ...] = ()
    action_field_map: tuple[tuple[str, Any], ...] = ()
    # Optional exact active slots for padded component routes. Empty preserves
    # the historical contiguous-prefix mask.
    state_active_indices: tuple[int, ...] = ()
    action_active_indices: tuple[int, ...] = ()
    # Unified 40D contract. When ``unified_registry_key`` is set, the route uses
    # the fixed-slot UnifiedAssembler scatter (NOT declared components), so
    # ``restore_action`` must un-scatter via ``restore_unified_action`` instead of
    # the ComponentAssembler inverse. ``unified_has_eef`` selects the active-slot
    # mask (EEF slots [0:18]) for the deploy inverse. None/False for legacy routes.
    unified_registry_key: str | None = None
    unified_has_eef: bool = True

    def format_instruction(self, task_description: str) -> str:
        # Delegate to ``Prompt.format`` so named placeholders (``{instruction}``
        # + extras like ``{object}``) work; a naive ``str.format(pos)`` raises
        # ``KeyError`` on named templates, which every real config uses.
        from tau0_vla.data.modalities import Prompt

        return Prompt(template=self.prompt_template).format(str(task_description))


def _data_spec_path(path: str | Path, *, route: str | None = None) -> Path:
    """Locate ``spec.json`` under ``<path>/finch_data_spec/<config_name>/``.

    Every config (single RobotConfig or DataMix child) lives in its own
    ``<config_name>/`` subdir. ``route=...`` picks one when >1 subdirs exist;
    when only 1 subdir is present it is auto-selected.
    """
    candidate = Path(path).resolve()
    if candidate.is_file() and candidate.name == DATA_SPEC_FILENAME:
        return candidate
    search_roots = [candidate] if candidate.is_dir() else [candidate.parent]
    search_roots.extend(search_roots[0].parents)
    for root in search_roots:
        spec_root = root / DATA_SPEC_DIR
        if not spec_root.is_dir():
            continue
        subdirs = sorted(sub for sub in spec_root.iterdir() if sub.is_dir() and (sub / DATA_SPEC_FILENAME).exists())
        if not subdirs:
            continue
        if route is not None:
            from tau0_vla.data.config import _normalize_config_name

            normalized_route = _normalize_config_name(route)
            for sub in subdirs:
                if sub.name == normalized_route:
                    return sub / DATA_SPEC_FILENAME
            available = ", ".join(s.name for s in subdirs)
            raise FileNotFoundError(
                f"route={route!r} (normalized {normalized_route!r}) not found under {spec_root}. Available: {available}"
            )
        if len(subdirs) == 1:
            return subdirs[0] / DATA_SPEC_FILENAME
        available = ", ".join(s.name for s in subdirs)
        raise FileNotFoundError(
            f"Multiple configs under {spec_root}; pass `route=<name>` to pick one. Available: {available}"
        )
    raise FileNotFoundError(f"Could not find {DATA_SPEC_DIR}/<config>/{DATA_SPEC_FILENAME} from: {candidate}")


def _persisted_payload(spec: FinchDataSpec) -> dict[str, Any]:
    payload = asdict(spec)
    payload.pop("robot_cls", None)
    payload.pop("norm_stats", None)
    payload.pop("artifacts_dir", None)
    # Components live in their own components.json — not in spec.json — so their
    # serialization is isolated and easy to evolve independently.
    payload.pop("state_components", None)
    payload.pop("action_components", None)
    payload.pop("state_component_dims", None)
    payload.pop("action_component_dims", None)
    payload.pop("state_field_map", None)
    payload.pop("action_field_map", None)
    return payload


def _write_data_spec(
    spec: FinchDataSpec,
    out_dir: str | Path,
    *,
    field_descriptions: dict[str, Any] | None = None,
    state_components: "list[Any] | None" = None,
    action_components: "list[Any] | None" = None,
    state_component_dims: "list[int] | None" = None,
    action_component_dims: "list[int] | None" = None,
    state_field_map: "dict[str, Any] | None" = None,
    action_field_map: "dict[str, Any] | None" = None,
) -> Path:
    """Write ``spec.json`` + ``norm_stats.json`` + ``field_descriptions.json``
    + ``components.json`` flat into ``out_dir``. The caller chooses ``out_dir``
    — typically ``<run_dir>/finch_data_spec/<config_name>/`` so DataMix and
    single-config checkpoints share the same per-config-subdir shape.
    """
    import shutil

    from tau0_vla.data.modalities.serialization import components_to_dict

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    payload = _persisted_payload(spec)
    payload.pop("norm_stats_path", None)
    output_path = out / DATA_SPEC_FILENAME
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if spec.norm_stats_path is None:
        import warnings
        warnings.warn(
            f"No norm_stats path resolved for '{spec.finch_config_name}'. "
            "Skipping norm_stats persistence — deploy will need stats provided separately.",
            stacklevel=2,
        )
    elif not os.path.isfile(spec.norm_stats_path):
        import warnings
        warnings.warn(
            f"norm_stats file not found at {spec.norm_stats_path!r}. "
            f"Skipping norm_stats persistence.",
            stacklevel=2,
        )
    else:
        shutil.copy2(spec.norm_stats_path, str(out / NORM_STATS_FILENAME))

    if field_descriptions is not None:
        fd_path = out / FIELD_DESCRIPTIONS_FILENAME
        fd_path.write_text(json.dumps(field_descriptions, indent=2), encoding="utf-8")

    if state_components is not None or action_components is not None:
        components_payload = {
            "state": components_to_dict(list(state_components or ())),
            "action": components_to_dict(list(action_components or ())),
            "state_dims": list(state_component_dims or ()),
            "action_dims": list(action_component_dims or ()),
        }
        if state_field_map:
            components_payload["state_field_map"] = {
                k: v if isinstance(v, list) else str(v) for k, v in state_field_map.items()
            }
        if action_field_map:
            components_payload["action_field_map"] = {
                k: v if isinstance(v, list) else str(v) for k, v in action_field_map.items()
            }
        (out / COMPONENTS_FILENAME).write_text(json.dumps(components_payload, indent=2), encoding="utf-8")

    return output_path


def _per_config_out_dir(run_dir: str | Path, config_name: str) -> Path:
    """Target directory for a single config's spec/stats/field_descriptions.

    The subdir name is the config's **normalized** name (underscores → hyphens,
    matching ``_normalize_config_name`` / ``_config_storage_stem`` used for
    training-time norm_stats file naming). Callers can therefore pass either
    the underscored form or the normalized form — the layout is the same.

    Single-config checkpoint → ``<run_dir>/finch_data_spec/<name>/`` (1 subdir).
    DataMix checkpoint → one such subdir per child (N subdirs).
    """
    from tau0_vla.data.config import _normalize_config_name

    return Path(run_dir) / DATA_SPEC_DIR / _normalize_config_name(config_name)


def _build_data_spec(
    *,
    resolved: list[Any],
    finch_config_name: str | None,
    finch_config_names_by_root: dict[str, str],
    config_modules: tuple[str, ...],
    vlm_model_type: str,
    vlm_aux_transforms: tuple[dict[str, Any], ...],
    cam_keys: tuple[str, ...] | None = None,
    cam_view_template: str | None = None,
    cam_view_names: dict[str, str] | None = None,
    max_images_per_sample: int | None = None,
) -> FinchDataSpec:
    exemplar = resolved[0]
    _validate_homogeneous_data_spec(resolved)
    target_size = _resolve_target_size(exemplar)
    action_key = exemplar.action_key_forward
    state_key = exemplar.state_key_forward
    is_eef = action_key == "action_eef"
    # Unified routes carry the registry key + EEF-active flag so deploy can run
    # the un-scatter inverse. ``has_eef`` mirrors UnifiedAssembler's
    # ``has_eef_action`` (declared native column or an adapter provider).
    rc = exemplar.robot_config
    state_active_indices = tuple(int(i) for i in (getattr(rc, "_state_active_indices", None) or ()))
    action_active_indices = tuple(int(i) for i in (getattr(rc, "_action_active_indices", None) or ()))
    unified_registry_key = getattr(rc, "_unified_registry_key", None)
    unified_has_eef = True
    if unified_registry_key is not None:
        eef_provider = rc._eef_provider() if hasattr(rc, "_eef_provider") else None
        unified_has_eef = (getattr(rc, "_eef_action_col", None) is not None) or (eef_provider is not None)
    return FinchDataSpec(
        format_version=FORMAT_VERSION,
        robot_name=exemplar.robot_config.robot_name,
        finch_config_name=finch_config_name,
        finch_config_names_by_root=finch_config_names_by_root,
        config_modules=config_modules,
        action_key_forward=action_key,
        state_key_forward=state_key,
        action_dim=int(exemplar.action_dim),
        state_dim=int(exemplar.state_dim),
        action_chunk_size=int(exemplar.action_horizon),
        cam_keys=tuple(str(value) for value in (cam_keys if cam_keys is not None else exemplar.cam_keys)),
        target_size=target_size,
        synthetic_images=tuple((str(k), str(v)) for k, v in exemplar.synthetic_images),
        use_quantile=False,
        norm_stats_path=_normalize_norm_stats_path(exemplar.norm_stats_path),
        vlm_model_type=vlm_model_type,
        vlm_aux_transforms=vlm_aux_transforms,
        is_eef=is_eef,
        eef_state_mode=_eef_mode_for_key(state_key, exemplar.state_dim),
        eef_action_mode=_eef_mode_for_key(action_key, exemplar.action_dim),
        action_space="end-effector" if is_eef else "joint",
        prompt_template=exemplar.robot_config.resolve_prompt().template,
        robot_cls=_get_robot_class(exemplar.robot_config.robot_name),
        norm_stats={},
        cam_view_template=cam_view_template,
        cam_view_names=cam_view_names,
        max_images_per_sample=max_images_per_sample,
        state_active_indices=state_active_indices,
        action_active_indices=action_active_indices,
        unified_registry_key=unified_registry_key,
        unified_has_eef=unified_has_eef,
    )


def save_data_spec(
    config_or_name: str | RobotConfig,
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
    """Persist a data spec for ``save_state``.

    Layout (uniform across single-config and DataMix):

        ``<run_dir>/finch_data_spec/<config_name>/spec.json``
        ``<run_dir>/finch_data_spec/<config_name>/norm_stats.json``
        ``<run_dir>/finch_data_spec/<config_name>/field_descriptions.json``

    Single RobotConfig → 1 ``<config_name>`` subdir.
    DataMix → N subdirs, one per child. No top-level ``spec.json`` with
    exemplar-leaked values — deploy side selects the right child via
    ``load_data_spec(ckpt, route=<config_name>)``.
    """
    if isinstance(config_or_name, str):
        config_name = config_or_name
        config = get_config(config_name)
    else:
        config_name = None
        config = config_or_name

    if _is_data_mix(config):
        return _save_data_mix_spec(
            config,
            config_name,
            run_dir,
            config_modules=config_modules,
            vlm_model_type=vlm_model_type,
            vlm_aux_transforms=vlm_aux_transforms,
            cam_keys=cam_keys,
            cam_view_template=cam_view_template,
            cam_view_names=cam_view_names,
            max_images_per_sample=max_images_per_sample,
        )
    if not isinstance(config, RobotConfig):
        raise TypeError(f"save_data_spec expected RobotConfig, got {type(config).__name__}")

    # Single-config: normalize the config name (hyphenated form) to keep
    # subdir names consistent with the per-config stats filename convention.
    effective_name = config_name or getattr(config, "_finch_config_name", None) or config.robot_name
    effective_name = str(effective_name)

    repo_id = str(Path(config._primary_repo_id()).resolve())
    resolved = config.resolve_output_spec_for_repo(repo_id)
    payload = _build_data_spec(
        resolved=[resolved],
        finch_config_name=effective_name,
        finch_config_names_by_root={repo_id: effective_name},
        config_modules=config_modules,
        vlm_model_type=vlm_model_type,
        vlm_aux_transforms=vlm_aux_transforms,
        cam_keys=cam_keys,
        cam_view_template=cam_view_template,
        cam_view_names=cam_view_names,
        max_images_per_sample=max_images_per_sample,
    )
    state_map, action_map, state_components, action_components = config._resolve_component_bundle()
    # The spec is a deployment-time contract — it describes what the policy
    # consumes, not what lives in the parquet. Route raw descriptions through
    # the robot's ``_normalize_field_descriptions`` so robot-side augmentations
    # (for example, compatibility fallbacks for old exports)
    # are present in the persisted spec. The normalizer returns
    # ``FieldDescription`` instances; we serialize back to raw dicts for the
    # JSON write path.
    state_desc, action_desc = config._normalize_field_descriptions(resolved.field_descriptions)
    augmented_field_descriptions = {
        "state": serialize_field_description_map(state_desc),
        "action": serialize_field_description_map(action_desc),
    }
    state_dims, action_dims = _resolve_component_dims(
        state_components=state_components,
        action_components=action_components,
        state_map=state_map,
        action_map=action_map,
        field_descriptions=augmented_field_descriptions,
    )
    out_dir = _per_config_out_dir(run_dir, effective_name)
    return _write_data_spec(
        payload,
        out_dir,
        field_descriptions=augmented_field_descriptions,
        state_components=list(state_components),
        action_components=list(action_components),
        state_component_dims=list(state_dims),
        action_component_dims=list(action_dims),
        state_field_map=dict(state_map),
        action_field_map=dict(action_map),
    )


def _save_data_mix_spec(
    mix: Any,
    mix_name: str | None,
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
    """Persist a multi-route data_spec for a ``DataMix`` checkpoint.

    Each child RobotConfig is resolved individually and written into its own
    ``<run_dir>/finch_data_spec/<child_name>/`` subdir with ``spec.json`` +
    ``norm_stats.json`` + ``field_descriptions.json``. Deploy-side
    ``load_data_spec(ckpt, route=<child_name>)`` resolves to the matching subdir.
    The mix itself carries no persisted data — the policy_manifest.json ``routes``
    list (one level up) enumerates available children.
    """
    resolved_list: list[Any] = []
    names_by_root: dict[str, str] = {}
    children: list[tuple[str, Any, Any]] = []
    # Flatten the whole tree to leaves (mirrors ``build_loader_spec`` which uses
    # ``_flatten()``): a DataMix may nest intermediate DataMix children (e.g. the
    # 4w mix groups robocoin/robomind/hy as weighted sub-mixes), and each leaf —
    # not each direct child — is what resolves to a RobotConfig.
    for child_name, _leaf_weight in mix._flatten():
        child = get_config(child_name)
        if not isinstance(child, RobotConfig):
            raise TypeError(f"DataMix leaf '{child_name}' must resolve to a RobotConfig, got {type(child).__name__}")
        repo_id = str(Path(child._primary_repo_id()).resolve())
        resolved = child.resolve_output_spec_for_repo(repo_id)
        resolved_list.append(resolved)
        names_by_root[repo_id] = child_name
        children.append((child_name, child, resolved))

    last_path: Path | None = None
    for child_name, child, resolved in children:
        payload = _build_data_spec(
            resolved=[resolved],
            finch_config_name=child_name,
            finch_config_names_by_root=names_by_root,
            config_modules=config_modules,
            vlm_model_type=vlm_model_type,
            vlm_aux_transforms=vlm_aux_transforms,
            cam_keys=cam_keys,
            cam_view_template=cam_view_template,
            cam_view_names=cam_view_names,
            max_images_per_sample=max_images_per_sample,
        )
        child_state_map, child_action_map, child_state_components, child_action_components = (
            child._resolve_component_bundle()
        )
        # Apply the robot's ``_normalize_field_descriptions`` hook so that
        # dim computation sees the same augmented descriptions pipeline code
        # uses at training time. Skipping this step can feed raw fields with the
        # wrong declared dimensions into transforms and crash at checkpoint
        # save. Mirror the single-config
        # path: normalizer returns ``FieldDescription`` instances, serialize
        # back to plain dicts for the JSON write path.
        child_state_desc, child_action_desc = child._normalize_field_descriptions(resolved.field_descriptions)
        normalized_field_descriptions = {
            "state": serialize_field_description_map(child_state_desc),
            "action": serialize_field_description_map(child_action_desc),
        }
        child_state_dims, child_action_dims = _resolve_component_dims(
            state_components=child_state_components,
            action_components=child_action_components,
            state_map=child_state_map,
            action_map=child_action_map,
            field_descriptions=normalized_field_descriptions,
        )
        out_dir = _per_config_out_dir(run_dir, child_name)
        last_path = _write_data_spec(
            payload,
            out_dir,
            field_descriptions=normalized_field_descriptions,
            state_components=list(child_state_components),
            action_components=list(child_action_components),
            state_component_dims=list(child_state_dims),
            action_component_dims=list(child_action_dims),
            state_field_map=dict(child_state_map),
            action_field_map=dict(child_action_map),
        )
    _cleanup_stale_inline_spec_dirs(run_dir, live={_per_config_out_dir(run_dir, name).name for name, _, _ in children})
    return last_path if last_path is not None else (Path(run_dir) / DATA_SPEC_DIR)


def _cleanup_stale_inline_spec_dirs(run_dir: str | Path, *, live: set[str]) -> None:
    """Remove inline-config spec dirs that this save did not (re)write.

    Inline leaves are content-named (``DataMix._inline_config_fingerprint``),
    so the just-written set is the complete live set and any other
    ``*inline?config*``-prefixed dir is garbage: either from the earlier
    ``id()``-named era (every rank x launch minted fresh names — run dirs
    accumulated 100k+ dirs and every checkpoint mirror copied them all), or
    from a config revision. Named (non-inline) config dirs are never touched.
    Best-effort: concurrent writers may race on the same stale dir, so
    individual failures are ignored. Deletion runs on a daemon thread —
    a run dir from the id()-named era can hold 100k+ stale dirs, and a
    synchronous rmtree would stall this rank (and, through the first NCCL
    collective, the whole job) for tens of minutes. Nothing reads these dirs,
    so training proceeds while they drain; if the process exits mid-delete
    the next launch simply resumes the cleanup.
    """
    import shutil
    import threading

    spec_root = Path(run_dir) / DATA_SPEC_DIR
    if not spec_root.is_dir():
        return
    stale = [
        entry
        for entry in spec_root.iterdir()
        if entry.is_dir()
        and entry.name.startswith(("-inline-config-", "_inline_config_"))
        and entry.name not in live
    ]
    if not stale:
        return
    logging.getLogger(__name__).warning(
        "finch_data_spec: removing %d stale inline-config spec dirs under %s "
        "in the background (left by earlier launches; the live mix has %d inline leaves)",
        len(stale),
        spec_root,
        len(live),
    )

    def _drain() -> None:
        for entry in stale:
            shutil.rmtree(entry, ignore_errors=True)

    threading.Thread(target=_drain, name="finch-spec-cleanup", daemon=True).start()


def _resolve_component_dims(
    *,
    state_components: "list[Any]",
    action_components: "list[Any]",
    state_map: "Any",
    action_map: "Any",
    field_descriptions: dict[str, Any],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Compute per-component output dims using the loaded field_descriptions.

    ``state_map`` / ``action_map`` carry the original per-component field-key
    mapping from the robot's ``repack["semantic"]`` dict — which may be a
    single field key or a list of field keys (e.g. gripper = left + right
    effector). ``_build_component_output_dims`` handles list-valued mappings
    by summing per-field dims, so the mapping must stay in its pre-bind
    form here (the bound form collapses the list into a synthetic composite
    key that is not present in the raw field_descriptions).
    """
    from tau0_vla.data.modalities.base import normalize_field_description_map
    from tau0_vla.data.pipeline import _build_component_output_dims

    state_desc = normalize_field_description_map(field_descriptions.get("state", {}))
    action_desc = normalize_field_description_map(field_descriptions.get("action", {}))
    state_dims = _build_component_output_dims(
        components=state_components,
        role="state",
        descriptions=state_desc,
        mapping=state_map,
    )
    action_dims = _build_component_output_dims(
        components=action_components,
        role="action",
        descriptions=action_desc,
        mapping=action_map,
    )
    return state_dims, action_dims


def _load_persisted_data_spec(
    path: str | Path,
    *,
    route: str | None = None,
    robot_type: str | None = None,
    image_size_override: int | None = None,
) -> FinchDataSpec:
    spec_path = _data_spec_path(path, route=route)
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    if int(payload.get("format_version", 0)) != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported Finch data spec format_version={payload.get('format_version')}, expected {FORMAT_VERSION}"
        )
    target_size = payload.get("target_size")
    if image_size_override is not None:
        target_size = [int(image_size_override), int(image_size_override)]
    robot_name = payload.get("robot_name")
    if robot_name is None:
        robot_name = _infer_robot_name(payload, robot_type=robot_type)
    # artifacts_dir = the directory holding spec.json (finch_data_spec/)
    artifacts_dir = str(spec_path.parent)
    # Pre-refactor checkpoints wrote a single `use_quantile` bool in spec.json
    # and omitted per-component `normalize` entries from components.json. Read
    # that bool here so we can backfill the per-component normalize with a
    # matching legacy default ("quantile" or "mean_std") before components_from_dict
    # enforces its required-kwarg contract. New checkpoints always write the
    # per-component value explicitly, so backfill is a no-op on modern artifacts.
    legacy_use_quantile = bool(payload.get("use_quantile", False))
    legacy_default_normalize = "quantile" if legacy_use_quantile else "mean_std"
    components_path = spec_path.parent / COMPONENTS_FILENAME
    if components_path.exists():
        from tau0_vla.data.modalities.serialization import components_from_dict

        components_payload = json.loads(components_path.read_text(encoding="utf-8"))

        def _backfill(entries: list[dict]) -> list[dict]:
            for entry in entries:
                entry.setdefault("normalize", legacy_default_normalize)
            return entries

        state_components = tuple(components_from_dict(_backfill(components_payload.get("state", []))))
        action_components = tuple(components_from_dict(_backfill(components_payload.get("action", []))))
        state_component_dims = tuple(int(v) for v in components_payload.get("state_dims", []))
        action_component_dims = tuple(int(v) for v in components_payload.get("action_dims", []))
        state_field_map = tuple((k, v) for k, v in (components_payload.get("state_field_map") or {}).items())
        action_field_map = tuple((k, v) for k, v in (components_payload.get("action_field_map") or {}).items())
    else:
        state_components = ()
        action_components = ()
        state_component_dims = ()
        action_component_dims = ()
        state_field_map = ()
        action_field_map = ()

    finch_config_name = payload.get("finch_config_name")
    config_modules = tuple(payload.get("config_modules") or ())
    cam_keys = tuple(str(k) for k in (payload.get("cam_keys") or ()))
    synthetic_images = tuple((str(k), str(v)) for k, v in (payload.get("synthetic_images") or ()))

    return FinchDataSpec(
        format_version=int(payload["format_version"]),
        robot_name=robot_name,
        finch_config_name=finch_config_name,
        finch_config_names_by_root=dict(payload.get("finch_config_names_by_root") or {}),
        config_modules=config_modules,
        action_key_forward=str(payload["action_key_forward"]),
        state_key_forward=str(payload["state_key_forward"]),
        action_dim=int(payload["action_dim"]),
        state_dim=int(payload["state_dim"]),
        action_chunk_size=int(payload["action_chunk_size"]),
        cam_keys=cam_keys,
        target_size=None if target_size is None else tuple(int(v) for v in target_size),
        synthetic_images=synthetic_images,
        use_quantile=bool(payload.get("use_quantile", False)),
        norm_stats_path=payload.get("norm_stats_path"),
        vlm_model_type=str(payload.get("vlm_model_type", "qwen2.5vl")),
        vlm_aux_transforms=tuple(payload.get("vlm_aux_transforms") or ()),
        is_eef=bool(payload.get("is_eef", False)),
        eef_state_mode=str(payload.get("eef_state_mode", "ee6d")),
        eef_action_mode=str(payload.get("eef_action_mode", "ee6d")),
        action_space=payload.get("action_space"),
        prompt_template=str(payload.get("prompt_template", "What action should the robot take to {}?")),
        robot_cls=_get_robot_class(robot_name),
        norm_stats={},
        artifacts_dir=artifacts_dir,
        cam_view_template=payload.get("cam_view_template"),
        cam_view_names=dict(payload.get("cam_view_names") or {}),
        max_images_per_sample=(
            None if payload.get("max_images_per_sample") is None else int(payload["max_images_per_sample"])
        ),
        state_components=state_components,
        action_components=action_components,
        state_component_dims=state_component_dims,
        action_component_dims=action_component_dims,
        state_field_map=state_field_map,
        action_field_map=action_field_map,
        state_active_indices=tuple(int(i) for i in (payload.get("state_active_indices") or ())),
        action_active_indices=tuple(int(i) for i in (payload.get("action_active_indices") or ())),
        unified_registry_key=payload.get("unified_registry_key"),
        unified_has_eef=bool(payload.get("unified_has_eef", True)),
    )


def resolve_norm_stats_payload(loaded: dict, embodiment: str = "") -> dict:
    if "norm_stats" in loaded:
        loaded = loaded["norm_stats"]
    if "per_embodiment" in loaded or "global" in loaded:
        per_emb = loaded.get("per_embodiment", {})
        if embodiment and embodiment in per_emb:
            return per_emb[embodiment]
        available = list(per_emb.keys())
        raise KeyError(f"Embodiment '{embodiment}' not found in norm stats. Available: {available}")
    return loaded


def load_norm_stats_artifact(
    norm_stats_path: str | None, run_dir: str, embodiment: str = "", artifacts_dir: str | None = None
) -> dict:
    """Resolve ``norm_stats.json``.

    Primary source is ``artifacts_dir`` (the per-config subdir picked by
    ``_data_spec_path``). ``norm_stats_path`` is an older escape-hatch kept
    for specs that still record an absolute path to an external norm_stats
    file; it's otherwise unused by the new per-config-subdir layout.
    """
    candidates = []
    if artifacts_dir:
        candidates.append(os.path.join(artifacts_dir, NORM_STATS_FILENAME))
    if norm_stats_path:
        candidates.append(norm_stats_path)
    for path in candidates:
        if path and os.path.isfile(path):
            with open(path) as handle:
                loaded = json.load(handle)
            return resolve_norm_stats_payload(loaded, embodiment)
    raise FileNotFoundError(f"No norm stats found for {run_dir}")


def _hydrate_data_spec(
    data_spec: FinchDataSpec,
    run_dir: str,
    *,
    embodiment: str = "",
) -> FinchDataSpec:
    resolved_embodiment = embodiment or data_spec.robot_name or ""
    try:
        norm_stats = load_norm_stats_artifact(
            data_spec.norm_stats_path,
            run_dir,
            embodiment=resolved_embodiment,
            artifacts_dir=data_spec.artifacts_dir,
        )
    except FileNotFoundError:
        norm_stats = {}
    # Unified-40D deploy (build_unified_state_encoder / _restore_unified_slots)
    # reads per-embodiment stats from the norm_stats_path FILE, not the in-memory
    # norm_stats dict. Persisted spec.json never records norm_stats_path (the
    # per-config-subdir layout keeps stats in artifacts_dir), so a checkpoint-loaded
    # spec leaves it None and the unified encoder/restorer raise. Resolve it to the
    # per-config artifact load_norm_stats_artifact just used — same file, so forward
    # and inverse stay pinned to one source.
    resolved_norm_stats_path = data_spec.norm_stats_path
    if data_spec.artifacts_dir:
        candidate = os.path.join(data_spec.artifacts_dir, NORM_STATS_FILENAME)
        if os.path.isfile(candidate):
            resolved_norm_stats_path = candidate
    return dataclass_replace(
        data_spec,
        norm_stats=norm_stats,
        norm_stats_path=resolved_norm_stats_path,
    )


def load_data_spec(
    path: str | Path,
    *,
    route: str | None = None,
    embodiment: str = "",
    robot_type: str | None = None,
    image_size_override: int | None = None,
) -> FinchDataSpec:
    """Load a persisted data spec. ``route`` picks one child under
    ``<path>/finch_data_spec/<route>/`` (required when >1 children exist,
    optional when only 1)."""
    data_spec = _load_persisted_data_spec(
        path,
        route=route,
        robot_type=robot_type,
        image_size_override=image_size_override,
    )
    run_dir = str(Path(path).resolve())
    return _hydrate_data_spec(
        data_spec,
        run_dir,
        embodiment=embodiment,
    )


def parse_norm_stats_payload(raw: dict[str, Any]) -> dict[str, NormStats]:
    """Convert raw JSON norm stats dict to dict[str, NormStats].

    Keys are remapped to canonical 'state'/'action' names to match
    what _ActionPostprocess expects.
    """
    _STATE_KEYS = {"state", "observation.state", "observation.state_eef", "inputs/state/robot"}
    _ACTION_KEYS = {"action", "actions", "action_eef", "targets/action"}

    result: dict[str, NormStats] = {}
    for key, s in raw.items():
        if not isinstance(s, dict) or "mean" not in s:
            continue
        canonical = key
        if key in _STATE_KEYS:
            canonical = "state"
        elif key in _ACTION_KEYS:
            canonical = "action"
        result[canonical] = NormStats(
            mean=np.asarray(s["mean"], dtype=np.float32),
            std=np.asarray(s["std"], dtype=np.float32),
            q01=None if s.get("q01") is None else np.asarray(s["q01"], dtype=np.float32),
            q99=None if s.get("q99") is None else np.asarray(s["q99"], dtype=np.float32),
        )
    return result


_postprocessor_cache: dict[tuple, Any] = {}


def action_decoder(data_spec: FinchDataSpec):
    """Return a pure ``(flat_actions, flat_state) -> dict[str, np.ndarray]``
    decoder — framework-agnostic.

    ``flat_actions``: flat normalized padded action vector from the model
        (shape ``(..., action_padding_dim)``).
    ``flat_state``: flat normalized padded state that was fed to the model
        (shape ``(..., state_padding_dim)``), needed as the reference for
        any action-side ``RelativeToState`` inverse transform.
    Returns: ``{component_key: array}`` with each component already
    unnormalized + inverse-transformed and split out of the padded vector.

    Policy repos wrap this into their own transform convention (e.g. openpi's
    ``Policy(output_transforms=[...])`` expects ``sample_dict -> sample_dict``
    shape) — the pipeline stays tensor-in / dict-out.
    """
    from tau0_vla.data.modalities.base import WorkflowContext

    postprocessor = _build_action_postprocessor(data_spec)
    action_components = tuple(postprocessor.action_components)
    pre_dims = tuple(postprocessor.action_component_dims)

    # Probe each component's *post-inverse* output dim with a zero dummy so we
    # slice ``restored_flat`` by what the postprocessor actually produces, not
    # by the model-space (pre-inverse) dims. They only diverge for transforms
    # that change the last axis (``Quat2Rot6D``: 9→7 on inverse); for all
    # other components ``post_dim == pre_dim`` and behaviour is unchanged.
    post_dims: list[int] = []
    for comp, pre_dim in zip(action_components, pre_dims, strict=True):
        dummy = np.zeros(int(pre_dim), dtype=np.float32)
        ctx = WorkflowContext(
            state={comp.key: dummy.copy()},
            action={comp.key: dummy.copy()},
            state_already_transformed=True,
        )
        probed = np.asarray(comp.apply_transforms(dummy, context=ctx, mode="post"))
        post_dims.append(int(probed.shape[-1]) if probed.ndim else 1)

    def decode(flat_actions, flat_state):
        restored_flat = postprocessor({"state": flat_state, "actions": flat_actions})["actions"]
        out: dict[str, Any] = {}
        offset = 0
        for comp, dim in zip(action_components, post_dims, strict=True):
            out[comp.key] = restored_flat[..., offset : offset + dim]
            offset += dim
        return out

    return decode


def _build_action_postprocessor(data_spec: FinchDataSpec):
    """Internal: build / cache the ``_ActionPostprocess`` behind ``action_decoder``.

    Fast path: when the data_spec carries components + dims (post-M2
    ``components.json`` artifact), construct directly with zero
    ``@register_config`` / field-description lookups. This is the deploy-time
    path — the checkpoint is self-contained.

    Legacy path: for checkpoints that predate ``components.json``, fall
    back to resolving via the config registry + dataset field
    descriptions. Same cache key so both paths share one cached object.
    """
    cache_key = (data_spec.finch_config_name or "", data_spec.artifacts_dir or "")
    if cache_key in _postprocessor_cache:
        return _postprocessor_cache[cache_key]

    from tau0_vla.data.pipeline import _ActionPostprocess

    if (
        data_spec.state_components
        and data_spec.action_components
        and data_spec.state_component_dims
        and data_spec.action_component_dims
    ):
        postprocessor = _ActionPostprocess(
            state_components=data_spec.state_components,
            action_components=data_spec.action_components,
            state_component_dims=data_spec.state_component_dims,
            action_component_dims=data_spec.action_component_dims,
            norm_stats=parse_norm_stats_payload(data_spec.norm_stats),
            state_padding_dim=data_spec.state_dim,
            action_padding_dim=data_spec.action_dim,
        )
        _postprocessor_cache[cache_key] = postprocessor
        return postprocessor

    config_name = data_spec.finch_config_name
    if config_name is None:
        raise ValueError("Cannot build action postprocessor without finch_config_name")

    # Ensure config modules are loaded
    if data_spec.config_modules:
        discover_config_modules(module_names=list(data_spec.config_modules))

    config = get_config(config_name)

    # Load field descriptions: prefer local artifact, fallback to repo
    field_descriptions = None
    if data_spec.artifacts_dir:
        fd_path = Path(data_spec.artifacts_dir) / FIELD_DESCRIPTIONS_FILENAME
        if fd_path.exists():
            field_descriptions = json.loads(fd_path.read_text(encoding="utf-8"))
    if field_descriptions is None:
        from tau0_vla.data.source import _load_field_descriptions

        field_descriptions = _load_field_descriptions(str(Path(config._primary_repo_id()).resolve()))

    # Apply robot-class fallback for datasets whose meta/info.json predates
    # the field_descriptions schema (e.g. old G1 hilserl / ARX exports).
    state_desc_fb, action_desc_fb = config._normalize_field_descriptions(field_descriptions)
    field_descriptions = {
        "state": serialize_field_description_map(state_desc_fb),
        "action": serialize_field_description_map(action_desc_fb),
    }

    # Get components
    state_map, action_map, state_components, action_components = config._resolve_component_bundle()
    state_descriptions, action_descriptions = config._normalize_field_descriptions(field_descriptions)

    # Compute component output dims
    from tau0_vla.data.pipeline import _build_component_output_dims

    state_component_dims = _build_component_output_dims(
        components=state_components,
        role="state",
        descriptions=state_descriptions,
        mapping=state_map,
    )
    action_component_dims = _build_component_output_dims(
        components=action_components,
        role="action",
        descriptions=action_descriptions,
        mapping=action_map,
    )

    norm_stats = parse_norm_stats_payload(data_spec.norm_stats)

    postprocessor = _ActionPostprocess(
        state_components=state_components,
        action_components=action_components,
        state_component_dims=state_component_dims,
        action_component_dims=action_component_dims,
        norm_stats=norm_stats,
        state_padding_dim=config.state_padding_dim,
        action_padding_dim=config.action_padding_dim,
    )
    _postprocessor_cache[cache_key] = postprocessor
    return postprocessor


def state_encoder(data_spec: FinchDataSpec):
    """Return a callable ``(named_state: dict) -> flat_normalized_padded``
    that turns a per-component state dict into the flat vector openpi's
    ``Policy`` expects on input, using components + norm_stats persisted
    in ``data_spec``. Self-contained — no ``@register_config`` lookup.
    """
    from tau0_vla.data.pipeline import assemble_state_array

    if not data_spec.state_components:
        raise ValueError(
            "data_spec.state_components is empty; run migrate_ckpt on legacy checkpoints to backfill components.json."
        )
    components = data_spec.state_components
    norm_stats = parse_norm_stats_payload(data_spec.norm_stats).get("state")
    state_padding_dim = data_spec.state_dim

    def encode(named_state):
        return assemble_state_array(
            named_state,
            components=components,
            norm_stats=norm_stats,
            state_padding_dim=state_padding_dim,
        )

    return encode


_state_encoder_cache: dict[tuple, Any] = {}


def build_state_encoder(data_spec: "FinchDataSpec"):
    """Build a callable ``(raw_state: np.ndarray) -> np.ndarray`` that replays
    the training-time state pipeline on a raw robot observation. Despite the
    historical ``normalize_state`` name, the pipeline does more than just
    normalize — it runs, in order:

      1. ``config.augment_raw_tensors``
      2. ``state_field_map`` per-component projection
      3. per-component transforms (e.g. ``Quat2Rot6D``)
      4. ``norm_stats`` normalization
      5. concat
      6. pad to ``state_padding_dim``

    Symmetric to `action_decoder` / `restore_action` for the state side.
    Cached by config-name + artifacts-dir, so the underlying
    ``ComponentAssembler`` is built once per deploy process.
    """
    cache_key = (data_spec.finch_config_name or "", data_spec.artifacts_dir or "")
    if cache_key in _state_encoder_cache:
        return _state_encoder_cache[cache_key]

    # Unified 40D routes: the ComponentAssembler pipeline below would produce a
    # component-layout state (wrong shape, wrong normalization) — delegate to
    # the UnifiedAssembler-backed encoder and return its normalized state.
    if data_spec.unified_registry_key is not None:
        bundle_encoder = build_unified_state_encoder(data_spec)

        def _encode_unified(raw_state: np.ndarray) -> np.ndarray:
            return bundle_encoder(raw_state)["state"]

        _state_encoder_cache[cache_key] = _encode_unified
        return _encode_unified

    config_name = data_spec.finch_config_name
    if config_name is None:
        raise ValueError("Cannot build state encoder without finch_config_name")

    if data_spec.config_modules:
        discover_config_modules(module_names=list(data_spec.config_modules))

    try:
        config = get_config(config_name)
    except KeyError:
        config = None
        _logger.warning(
            "build_state_encoder: config %r not found in registry, "
            "falling back to deploy-only mode (no augment_fn, "
            "field_descriptions from checkpoint artifacts).",
            config_name,
        )

    field_descriptions = None
    if data_spec.artifacts_dir:
        fd_path = Path(data_spec.artifacts_dir) / FIELD_DESCRIPTIONS_FILENAME
        if fd_path.exists():
            field_descriptions = json.loads(fd_path.read_text(encoding="utf-8"))
    if field_descriptions is None and config is not None:
        from tau0_vla.data.source import _load_field_descriptions

        field_descriptions = _load_field_descriptions(str(Path(config._primary_repo_id()).resolve()))
    if field_descriptions is None:
        raise ValueError(
            f"Cannot build state encoder: config {config_name!r} not in registry "
            f"and no field_descriptions.json found in artifacts_dir={data_spec.artifacts_dir!r}"
        )

    if config is not None:
        state_desc, action_desc = config._normalize_field_descriptions(field_descriptions)
        field_descriptions = {
            "state": serialize_field_description_map(state_desc),
            "action": serialize_field_description_map(action_desc),
        }
        state_map, action_map, state_components, action_components = config._resolve_component_bundle()
        augment_fn = config.augment_raw_tensors
        has_temporal_axis = bool(config.temporal_sequence and "state" in config.temporal_sequence)
    else:
        # Deploy-only fallback: use persisted components + field maps from
        # data_spec (originally saved in components.json). No augmentation,
        # no temporal axis.
        state_components = data_spec.state_components
        action_components = data_spec.action_components
        state_map = dict(data_spec.state_field_map)
        action_map = dict(data_spec.action_field_map)
        if not state_map and state_components:
            raise ValueError(
                f"Deploy-only mode for config {config_name!r}: checkpoint's "
                "components.json is missing state_field_map. Re-save the "
                "checkpoint with a newer pipeline to persist field maps."
            )
        augment_fn = None
        has_temporal_axis = False

    from tau0_vla.data.pipeline import ComponentAssembler

    # Mirror the TRAINING-time ComponentAssembler construction in
    # ``RobotConfig._build_component_assembler`` — every kwarg here must be
    # set in lockstep with the training side, otherwise deploy silently reads
    # state from a different shape than training wrote it. Historical bugs
    # caused by forgetting a kwarg here:
    #   - ``augment_fn`` missing → adapter-added fields can be read from the
    #     wrong raw-tensor positions at inference.
    #   - ``state_has_temporal_axis`` missing → shape mismatch for any config
    #     with ``temporal_sequence={"state": ...}``.
    assembler = ComponentAssembler(
        state_components=state_components,
        action_components=action_components,
        norm_stats=parse_norm_stats_payload(data_spec.norm_stats),
        state_field_map=state_map,
        action_field_map=action_map,
        state_padding_dim=data_spec.state_dim,
        action_padding_dim=data_spec.action_dim,
        state_active_indices=data_spec.state_active_indices or None,
        action_active_indices=data_spec.action_active_indices or None,
        state_has_temporal_axis=has_temporal_axis,
        augment_fn=augment_fn,
    )

    # Dummy action large enough to cover the full raw-action index range,
    # so the action-side half of the assembler can run without IndexError even
    # though we only care about the state output.
    action_fields = normalize_field_description_map(field_descriptions.get("action"))
    raw_action_dim = 0
    for desc in action_fields.values():
        indices = getattr(desc, "indices", None)
        if indices:
            raw_action_dim = max(raw_action_dim, max(indices) + 1)
    dummy_action = np.zeros(max(raw_action_dim, int(data_spec.action_dim)), dtype=np.float32)

    def _encode(raw_state: np.ndarray) -> np.ndarray:
        result = assembler(
            {
                "_state_raw": np.asarray(raw_state, dtype=np.float32),
                "_action_raw": dummy_action,
                "_field_descriptions": field_descriptions,
            }
        )
        return np.asarray(result["state"], dtype=np.float32).flatten()

    _state_encoder_cache[cache_key] = _encode
    return _encode


def encode_state(raw_state: np.ndarray, data_spec: "FinchDataSpec") -> np.ndarray:
    """Apply the training-time state pipeline to a raw robot state.

    Forward dual of `restore_action`: augment → project → component transforms
    (e.g. ``Quat2Rot6D``) → normalize → concat → pad. See
    ``build_state_encoder`` for the full ordered list. Caches the underlying
    pipeline per ``(finch_config_name, artifacts_dir)``.
    """
    return build_state_encoder(data_spec)(raw_state)


_unified_state_encoder_cache: dict[tuple, Any] = {}


def _build_unified_assembler(
    data_spec: "FinchDataSpec",
    *,
    caller: str = "build_unified_assembler",
    relative_action: bool = True,
    disable_normalization: bool = False,
    return_all_norm_forms: bool = False,
):
    """Construct the training-time ``UnifiedAssembler`` for a unified-40D route
    from its resolved config (EEF provider / format / single-arm) + the ckpt's
    per-embodiment norm stats. Shared by the deploy state encoder
    (``build_unified_state_encoder``) and the eval GT builder
    (``unified_native_gt_from_raw``) so scatter, prefer-EEF masking,
    relativization and normalization can never drift between them or from training.

    Returns ``(assembler, config)`` — ``config`` is ``None`` only in the
    registry-key-only fallback (config not registered). ``caller`` only tunes the
    error/warning text. ``relative_action`` / ``disable_normalization`` /
    ``return_all_norm_forms`` are forwarded to the assembler so a caller can ask
    for the absolute (un-relativized, un-normalized) scatter (GT) or the model's
    normalized-relative forms (deploy state).

    Column-EEF bodies raise ``NotImplementedError`` when their EEF feed is not
    present in the deploy payload; joint-only bodies work from raw vectors.
    """
    key = data_spec.unified_registry_key
    if key is None:
        raise ValueError(f"{caller} called on a non-unified data_spec")

    from tau0_vla.data.robots.unified import UnifiedAssembler
    from tau0_vla.data.stats import load_file_with_per_embodiment

    if data_spec.config_modules:
        discover_config_modules(module_names=list(data_spec.config_modules))
    config = None
    if data_spec.finch_config_name:
        try:
            config = get_config(data_spec.finch_config_name)
        except KeyError:
            _logger.warning(
                "%s: config %r not in registry — falling back to registry-key-only "
                "mode (no EEF provider; eef_format/is_single_arm from defaults).",
                caller, data_spec.finch_config_name,
            )

    eef_provider = config._eef_provider() if config is not None and hasattr(config, "_eef_provider") else None
    if config is not None:
        eef_state_col = getattr(config, "_eef_state_col", None)
        has_eef_state = eef_state_col is not None or eef_provider is not None
        has_eef_action = getattr(config, "_eef_action_col", None) is not None or eef_provider is not None
        eef_format = getattr(config, "_eef_format", "euler")
        is_single_arm = getattr(config, "_is_single_arm", False)
    else:
        eef_state_col = None
        has_eef_state = has_eef_action = bool(data_spec.unified_has_eef)
        eef_format = "euler"
        is_single_arm = False

    if has_eef_state and eef_provider is None:
        raise NotImplementedError(
            f"{caller}: this unified route sources EEF from a parquet column "
            f"({eef_state_col!r}), which the deploy/eval payload does not carry — "
            "joint-only bodies work from the raw state/action vectors alone. "
            "Wire an EEF feed into the payload before using a "
            "column-EEF body."
        )

    if data_spec.norm_stats_path is None:
        raise ValueError(
            f"{caller}: data_spec.norm_stats_path is required (per-embodiment "
            "stats keyed by the registry key)."
        )
    global_stats, per_emb = load_file_with_per_embodiment(data_spec.norm_stats_path)

    assembler = UnifiedAssembler(
        registry_key=key,
        has_eef_action=has_eef_action,
        has_eef_state=has_eef_state,
        eef_format=eef_format,
        is_single_arm=is_single_arm,
        norm_stats=global_stats,
        per_embodiment_stats=per_emb,
        eef_provider=eef_provider,
        relative_action=relative_action,
        disable_normalization=disable_normalization,
        return_all_norm_forms=return_all_norm_forms,
    )
    return assembler, config


def build_unified_state_encoder(data_spec: "FinchDataSpec"):
    """Deploy forward for unified-40D routes: raw native state → the exact
    scattered / EEF-prioritised / per-embodiment-normalized 40D state that the
    training-time ``UnifiedAssembler`` produced, plus the pieces the rest of the
    deploy stack needs. The returned callable maps ``raw_state`` to::

        {
          "state":       normalized masked 40D  (the model's state input),
          "state_abs":   absolute (pre-normalize) scattered 40D, mask-zeroed —
                         what ``restore_action``'s relative→absolute inverse needs,
          "state_mask":  [40] active-slot mask,
          "action_mask": [40] active-slot mask (``sample_action`` uses it to pin
                         inactive dims — Scheme-C inference consistency),
        }

    It constructs the real training ``UnifiedAssembler`` (from the resolved
    config's class attributes when the config registry is available) so the
    scatter, prefer-EEF masking and normalization cannot drift from training.
    Norm stats are read from ``data_spec.norm_stats_path`` — the same source
    ``_restore_unified_slots`` uses, so forward and inverse always share stats.

    Limitations (explicit, fail-loud): bodies that source EEF from a parquet
    COLUMN (e.g. yam/hy) cannot be encoded from the raw state vector alone —
    the robot would have to send the EEF feed as well; that wiring raises
    ``NotImplementedError`` until someone wires that input. Joint-only bodies
    work from the raw state alone.
    """
    cache_key = (data_spec.finch_config_name or "", data_spec.artifacts_dir or "")
    if cache_key in _unified_state_encoder_cache:
        return _unified_state_encoder_cache[cache_key]

    key = data_spec.unified_registry_key
    if key is None:
        raise ValueError("build_unified_state_encoder called on a non-unified data_spec")

    from tau0_vla.data.robots.unified import get_registry_entry

    # ``return_all_norm_forms=True`` so ``extras['state']['raw']`` gives back the
    # absolute (pre-normalize) scattered state the restore inverse needs.
    assembler, _config = _build_unified_assembler(
        data_spec, caller="build_unified_state_encoder", return_all_norm_forms=True,
    )
    entry = get_registry_entry(key)
    dummy_action = np.zeros(int(entry.get("action_dim") or data_spec.action_dim), dtype=np.float32)

    def _encode(raw_state: np.ndarray) -> dict[str, np.ndarray]:
        result = assembler(
            {
                "_state_raw": np.asarray(raw_state, dtype=np.float32).reshape(-1),
                "_action_raw": dummy_action,
                "prompt": "",
                "images": {},
            }
        )
        return {
            "state": np.asarray(result["state"], dtype=np.float32).reshape(-1),
            "state_abs": np.asarray(result["extras"]["state"]["raw"], dtype=np.float32).reshape(-1),
            "state_mask": np.asarray(result["state_mask"], dtype=np.float32).reshape(-1),
            "action_mask": np.asarray(result["action_mask"], dtype=np.float32).reshape(-1),
        }

    _unified_state_encoder_cache[cache_key] = _encode
    return _encode


# Legacy names. ``normalize_state`` / ``build_state_normalizer`` only covered
# step 4 of the pipeline by name, which misled callers into thinking the
# function was a simple normalize; the real work is the whole ordered list
# documented on ``build_state_encoder``. New code should use the ``encode_*``
# names. Aliases are kept for one release cycle so existing imports don't break.
normalize_state = encode_state
build_state_normalizer = build_state_encoder
_state_normalizer_cache = _state_encoder_cache


def detect_quat_order(data_spec: "FinchDataSpec") -> str:
    """Return the quaternion channel order (``"xyzw"`` or ``"wxyz"``) declared
    by this checkpoint's resolved Finch config, inspecting the ``Quat2Rot6D``
    transform attached to any ``EefPose`` action component. Defaults to
    ``"xyzw"`` (Finch's canonical convention) only when the config loads
    cleanly but has no ``EefPose`` + ``Quat2Rot6D`` — config lookup / module
    import failures propagate so they can't silently invert rotations.
    """
    from tau0_vla.data.modalities.transforms import Quat2Rot6D

    if data_spec.config_modules:
        discover_config_modules(module_names=list(data_spec.config_modules))
    if data_spec.finch_config_name is None:
        return "xyzw"
    config = get_config(data_spec.finch_config_name)
    _, _, _, action_components = config._resolve_component_bundle()
    for comp in action_components:
        for t in getattr(comp, "transforms", []):
            if isinstance(t, Quat2Rot6D):
                return t.resolved_quat_order
    return "xyzw"


# Canonical native-slot order for the unified deploy inverse — shared by
# ``restore_unified_action`` (dict), ``restore_action`` (flat concat) and
# ``action_slices`` (labels) so all three agree. Active slots only.
_UNIFIED_NATIVE_ORDER = (
    "left_eef", "right_eef", "left_gripper", "right_gripper",
    "waist", "chassis_velocity", "left_arm", "right_arm",
)


def _restore_unified_slots(
    action_inferred: np.ndarray,
    data_spec: FinchDataSpec,
    *,
    state: np.ndarray,
    action_stats: NormStats | None = None,
) -> "list[tuple[str, np.ndarray]]":
    """Un-scatter a normalized 40D unified action into ordered native slots.

    Inverse of the unified forward, in the legacy
    order ``unnormalize → RelativeToState⁻¹ → Quat2Rot6D⁻¹``:
      1. ``unnormalize_masked`` (per-embodiment ``registry_key`` stats, std_floor 0.2),
      2. ``unrelativize_unified_action`` (add the absolute scattered ``state`` back
         on the delta slots),
      3. EEF rot6d→quat per active arm.
    Returns ``[(name, array)]`` for the ACTIVE slots in ``_UNIFIED_NATIVE_ORDER``;
    EEF as xyz+quat (7), arms truncated to their active joint count.

    ``state`` MUST be the absolute (un-normalized) scattered 40D state.
    ``action_stats`` overrides the per-embodiment lookup (tests / round-trip).
    """
    from tau0_vla.data.robots.unified import (
        build_unified_mask,
        get_registry_entry,
        unrelativize_unified_action,
    )
    from tau0_vla.data.stats import unnormalize_masked

    key = data_spec.unified_registry_key
    if key is None:
        raise ValueError("_restore_unified_slots called on a non-unified data_spec")
    entry = get_registry_entry(key)
    mask = build_unified_mask(entry, has_eef=data_spec.unified_has_eef, role="action")

    if action_stats is None:
        from tau0_vla.data.stats import load_file_with_per_embodiment

        global_stats, per_emb = load_file_with_per_embodiment(data_spec.norm_stats_path)
        resolved = per_emb.get(key) or global_stats
        action_stats = resolved["action"]

    a = unnormalize_masked(np.asarray(action_inferred, dtype=np.float32), action_stats, mask=mask, std_floor=0.2)
    a = unrelativize_unified_action(a, np.asarray(state, dtype=np.float32), mask)
    # Shared tail with the eval GT path: gather the now-absolute scattered action
    # into active native slots (EEF rot6d→quat) in _UNIFIED_NATIVE_ORDER.
    return _gather_unified_native_slots(a, data_spec)


def _gather_unified_native_slots(
    action_absolute: np.ndarray, data_spec: FinchDataSpec,
) -> "list[tuple[str, np.ndarray]]":
    """Gather an ABSOLUTE scattered 40D action into its active native slots, in
    ``_UNIFIED_NATIVE_ORDER`` — EEF as 9D rot6d → 7D xyz+quat, arms truncated to
    the body's active joint count, grippers/waist/chassis as-is.

    Shared tail of the two unified action paths so they always land in the
    identical layout:
      - pred (``_restore_unified_slots``): after unnormalize + unrelativize;
      - GT   (``unified_native_gt_from_raw``): already absolute from the scatter.
    """
    from tau0_vla.data.modalities.transforms import Quat2Rot6D
    from tau0_vla.data.robots.unified import (
        CHASSIS_VELOCITY,
        LEFT_ARM,
        LEFT_EEF,
        LEFT_GRIPPER,
        RIGHT_ARM,
        RIGHT_EEF,
        RIGHT_GRIPPER,
        WAIST,
        build_unified_mask,
        get_registry_entry,
    )

    key = data_spec.unified_registry_key
    if key is None:
        raise ValueError("_gather_unified_native_slots called on a non-unified data_spec")
    mask = build_unified_mask(get_registry_entry(key), has_eef=data_spec.unified_has_eef, role="action")
    a = np.asarray(action_absolute, dtype=np.float32)

    blocks = {
        "left_eef": LEFT_EEF, "right_eef": RIGHT_EEF,
        "left_gripper": LEFT_GRIPPER, "right_gripper": RIGHT_GRIPPER,
        "waist": WAIST, "chassis_velocity": CHASSIS_VELOCITY,
        "left_arm": LEFT_ARM, "right_arm": RIGHT_ARM,
    }
    quat = Quat2Rot6D()  # xyzw, matching the scatter
    out: list[tuple[str, np.ndarray]] = []
    for name in _UNIFIED_NATIVE_ORDER:
        blk = blocks[name]
        if name in ("left_eef", "right_eef"):
            if bool(np.all(mask[blk] > 0)):
                out.append((name, quat.inverse(a[..., blk], None, component=None)))  # 9D rot6d → 7D xyz+quat
        else:
            n = int(round(float(np.sum(mask[blk]))))
            if n > 0:
                out.append((name, a[..., blk.start : blk.start + n]))
    return out


def unified_native_gt_from_raw(
    raw_sample: "dict[str, Any]", data_spec: FinchDataSpec,
) -> np.ndarray:
    """Eval GT for a unified-40D route: raw lerobot sample → the absolute native
    action chunk in the SAME compact ``_UNIFIED_NATIVE_ORDER`` layout that
    ``restore_action`` returns for pred (EEF xyz+quat, arms truncated to active
    joints, grippers/waist/chassis as-is).

    Why this exists: ``repack_action_target`` builds GT from the config's DECLARED
    components, whose order (and membership) differs from the unified scatter the
    model actually learned — e.g. the a2d joint route declares only ``ArmJoint``,
    so its native GT is ``[arm…]`` with grippers OMITTED and arms ordered first,
    which does not line up with the model's ``[gripper, arm…]`` scatter target.
    Comparing that against the un-scattered pred yields a pure layout mismatch.

    This instead runs the training ``UnifiedAssembler`` with relativization AND
    normalization OFF to recover the absolute scattered action, then applies the
    shared ``_gather_unified_native_slots`` — so GT and pred share one space by
    construction and openloop compares like-for-like.
    """
    assembler, config = _build_unified_assembler(
        data_spec, caller="unified_native_gt_from_raw",
        relative_action=False, disable_normalization=True,
    )
    if config is None:
        raise ValueError(
            "unified_native_gt_from_raw: finch_config_name did not resolve to a "
            "registered config, so the raw sample cannot be repacked for GT."
        )
    repacked = config._repack_raw_sample(raw_sample)
    result = assembler(repacked)
    action_abs = np.asarray(result["action"], dtype=np.float32)
    slots = _gather_unified_native_slots(action_abs, data_spec)
    return np.concatenate([arr for _, arr in slots], axis=-1).astype(np.float32)


def restore_unified_action(
    action_inferred: np.ndarray,
    data_spec: FinchDataSpec,
    *,
    state: np.ndarray,
) -> dict[str, np.ndarray]:
    """Deploy inverse for a unified-40D route → robot-agnostic semantic native dict
    (EEF as xyz+quat, arms truncated to active joints, grippers/waist/chassis as-is).
    Keyed by ``UNIFIED_LAYOUT`` slot names. ``state`` is the absolute scattered 40D
    state. See :func:`_restore_unified_slots`."""
    return {name: arr for name, arr in _restore_unified_slots(action_inferred, data_spec, state=state)}


def restore_action(
    action_inferred: np.ndarray,
    data_spec: FinchDataSpec,
    *,
    state: np.ndarray | None = None,
) -> np.ndarray:
    # Unified 40D routes use the fixed-slot scatter, not declared components — the
    # ComponentAssembler inverse below would truncate + skip un-normalization. Route
    # to the un-scatter inverse and return the flat native concat (canonical slot
    # order; ``restore_unified_action`` gives the same data as a dict).
    if data_spec.unified_registry_key is not None:
        if state is None:
            raise ValueError(
                "restore_action for a unified route requires the absolute scattered 40D `state` "
                "(needed for the relative→absolute action inverse)."
            )
        slots = _restore_unified_slots(action_inferred, data_spec, state=state)
        return np.concatenate([arr for _, arr in slots], axis=-1).astype(np.float32)
    # Full inverse pipeline: unnormalize → RelativeToState⁻¹ → Quat2Rot6D⁻¹.
    # Exceptions here used to be caught and silently downgraded to
    # ``_restore_action_array`` (unnormalize-only). That hid real bugs —
    # rot6d-valued outputs were sent to the robot as if they were quaternions,
    # etc. Let exceptions propagate so wrong-answer deployments fail loudly
    # instead of driving policies with half-restored actions.
    if state is not None and data_spec.finch_config_name is not None:
        # Use the raw ``_build_action_postprocessor`` rather than
        # ``action_decoder`` — the latter wraps the flat result into a
        # ``{component_key: array}`` dict for sample-shape consumers, but
        # ``restore_action`` is contractually flat-array in / flat-array out.
        postprocessor = _build_action_postprocessor(data_spec)
        result = postprocessor(
            {
                "state": np.asarray(state, dtype=np.float32),
                "actions": np.asarray(action_inferred, dtype=np.float32),
            }
        )
        return np.asarray(result["actions"], dtype=np.float32)
    # Fallback: unnormalize only (no inverse transforms). Reached only when the
    # caller explicitly cannot provide a state (e.g. policies whose actions
    # don't depend on state for inverse), or when finch_config_name is absent
    # (legacy ckpts). Never as an error-handling fallback.
    return _restore_action_array(
        action_inferred,
        data_spec.norm_stats,
        data_spec.action_key_forward,
        use_quantile=data_spec.use_quantile,
    )


_CANONICAL_COMPONENT_NAMES = {
    "ArmJoint": "arm_joint",
    "EefPose": "eef_pose",
    "Gripper": "gripper",
    "Waist": "waist",
    "ChassisVelocity": "chassis_velocity",
}


def action_slices(data_spec: FinchDataSpec) -> list[tuple[str, int, int]]:
    """Per-component ``(name, offset, native_dim)`` layout of
    ``restore_action``'s flat last axis, in declared order.

    ``name`` is canonical (``arm_joint`` / ``eef_pose`` / ``gripper`` /
    ``waist`` / ``chassis_velocity``) — stable across robots so the HTTP
    contract keys stay robot-agnostic. Unknown component types raise
    ``KeyError`` so new component classes can't silently ship with an
    unstable wire name. ``native_dim`` is pre-transform (EefPose = xyz+quat
    = 7 per arm; rot6d only applies in model space).

    Used by deploy-side tools that need to split/merge the flat action
    array (e.g. dict-shape JSON responses, per-component plotting).

    Field-descriptions source: if ``data_spec.artifacts_dir`` is set, reads
    the ckpt-saved ``field_descriptions.json`` there (training-frozen,
    matches the trained output shape); missing file under that dir raises
    (ckpt is broken). If ``artifacts_dir`` is None, scans the repo as the
    single source of truth — this is the expected path when callers stub a
    deploy-side data_spec without a ckpt (see ``openloop_with_server.py``).
    """
    # Unified 40D routes have no declared components — derive the layout from the
    # active registry slots, matching ``restore_action``'s flat concat order.
    if data_spec.unified_registry_key is not None:
        from tau0_vla.data.robots.unified import (
            CHASSIS_VELOCITY,
            LEFT_ARM,
            LEFT_EEF,
            LEFT_GRIPPER,
            RIGHT_ARM,
            RIGHT_EEF,
            RIGHT_GRIPPER,
            WAIST,
            build_unified_mask,
            get_registry_entry,
        )

        entry = get_registry_entry(data_spec.unified_registry_key)
        mask = build_unified_mask(entry, has_eef=data_spec.unified_has_eef, role="action")
        blocks = {
            "left_eef": LEFT_EEF, "right_eef": RIGHT_EEF,
            "left_gripper": LEFT_GRIPPER, "right_gripper": RIGHT_GRIPPER,
            "waist": WAIST, "chassis_velocity": CHASSIS_VELOCITY,
            "left_arm": LEFT_ARM, "right_arm": RIGHT_ARM,
        }
        out: list[tuple[str, int, int]] = []
        offset = 0
        for name in _UNIFIED_NATIVE_ORDER:
            blk = blocks[name]
            if name in ("left_eef", "right_eef"):
                native = 7 if bool(np.all(mask[blk] > 0)) else 0  # xyz+quat
            else:
                native = int(round(float(np.sum(mask[blk]))))
            if native > 0:
                out.append((name, offset, native))
                offset += native
        return out

    from tau0_vla.data.config import get_config

    config = get_config(data_spec.finch_config_name)
    artifacts_dir = getattr(data_spec, "artifacts_dir", None)
    if artifacts_dir is not None:
        fd_path = Path(artifacts_dir) / "field_descriptions.json"
        if not fd_path.exists():
            raise FileNotFoundError(
                f"action_slices: artifacts_dir={artifacts_dir!r} has no "
                f"field_descriptions.json; ckpt artifacts are incomplete."
            )
        field_descriptions = json.loads(fd_path.read_text(encoding="utf-8"))
    else:
        from tau0_vla.data.source import _load_field_descriptions

        field_descriptions = _load_field_descriptions(str(Path(config._primary_repo_id()).resolve()))
    _, action_map, _, action_components = config._resolve_component_bundle()
    _, action_desc = config._normalize_field_descriptions(field_descriptions)

    out: list[tuple[str, int, int]] = []
    offset = 0
    for comp in action_components:
        kind = type(comp).__name__
        if kind not in _CANONICAL_COMPONENT_NAMES:
            raise KeyError(
                f"action_slices: unknown action component {kind!r}; "
                f"extend _CANONICAL_COMPONENT_NAMES in tau0_vla/data/data_spec.py."
            )
        name = _CANONICAL_COMPONENT_NAMES[kind]
        fk = action_map.get(comp.key, comp.field_key)
        keys = fk if isinstance(fk, list) else [fk]
        native_dim = sum(int(action_desc[k].dimensions) for k in keys)
        out.append((name, offset, native_dim))
        offset += native_dim
    return out


def encode_payload(payload: "dict[str, Any]", data_spec: FinchDataSpec) -> dict[str, Any]:
    """Apply every forward transform ``data_spec`` declares for deploy use.

    Forward dual of ``restore_action``. Mirrors the deploy-applicable subset
    of the training-time ``RobotConfig.build_pipeline``:

      - ``prompt`` via ``data_spec.format_instruction`` (``Prompt`` template —
        matches training's ``_wrap_prompt``).
      - ``images`` via ``data_spec.target_size`` bilinear resize + HWC uint8
        canonicalize (matches the ``ResizeWithPad`` training configs declare
        on ``RobotConfig.images``).
      - ``state`` via ``encode_state`` (augment → project → transforms →
        normalize → concat → pad; matches the state half of training's
        ``ComponentAssembler``).

    The VLM tokenization step (``input_ids`` / ``pixel_values``) is NOT part of
    this function, by design: the data layer knows nothing about which VLM
    consumes its output. Policy classes call this, then run their own VLM
    processor on the returned dict.

    Single source of truth: adding a new declarative transform to
    ``FinchDataSpec`` means extending this function once; downstream policy
    repos get the fix for free.
    """
    for required in ("state", "prompt", "images"):
        if required not in payload:
            raise KeyError(f"encode_payload requires '{required}' in payload; got keys={sorted(payload)}")
    raw_state = np.asarray(payload["state"], dtype=np.float32).reshape(-1)
    encoded: dict[str, Any] = {
        "prompt": data_spec.format_instruction(str(payload["prompt"])),
        "images": _encode_images(payload["images"], data_spec),
        "meta": dict(payload.get("meta") or {}),
    }
    if data_spec.unified_registry_key is not None:
        # Unified 40D route: one assembler pass yields the model's normalized
        # state plus the absolute scattered state (restore inverse) and the
        # active-slot masks (sample_action's inactive-dim pinning).
        bundle = build_unified_state_encoder(data_spec)(raw_state)
        encoded["state"] = bundle["state"]
        encoded["state_abs"] = bundle["state_abs"]
        encoded["state_mask"] = bundle["state_mask"]
        encoded["action_mask"] = bundle["action_mask"]
    else:
        encoded["state"] = encode_state(raw_state, data_spec)
        if data_spec.state_active_indices:
            state_mask = np.zeros(data_spec.state_dim, dtype=np.float32)
            state_mask[list(data_spec.state_active_indices)] = 1.0
            encoded["state_mask"] = state_mask
        if data_spec.action_active_indices:
            action_mask = np.zeros(data_spec.action_dim, dtype=np.float32)
            action_mask[list(data_spec.action_active_indices)] = 1.0
            encoded["action_mask"] = action_mask
    return encoded


def _encode_images(images: "dict[str, Any]", data_spec: FinchDataSpec) -> dict[str, np.ndarray]:
    """Forward image transforms declared by ``data_spec``.

    Enforces ``data_spec.cam_keys`` membership — missing cameras raise so a
    miswired robot SDK can't silently degrade a VLM with wrong-named frames.
    Preserves ``cam_keys`` insertion order in the output dict (Python 3.7+
    ordered-dict semantics), so downstream consumers get the same ordering
    training did.
    """
    expected = tuple(data_spec.cam_keys)
    synthetic = {str(k): str(v) for k, v in data_spec.synthetic_images}
    if expected:
        missing = [k for k in expected if k not in images and k not in synthetic]
        if missing:
            raise KeyError(
                f"Observation is missing camera(s) {missing}; "
                f"data_spec.cam_keys declared {expected}. Rename the robot "
                f"SDK's image keys to match the training contract, or "
                f"re-train with keys matching your robot."
            )
        keys = expected
    else:
        keys = tuple(images.keys())

    from tau0_vla.data.pipeline import canonicalize_image_hwc, resize_image_hwc

    target_size = data_spec.target_size
    out: dict[str, np.ndarray] = {}
    for key in keys:
        if key in images:
            out[key] = canonicalize_image_hwc(images[key]) if target_size is None else resize_image_hwc(images[key], target_size)
            continue
        if synthetic.get(key) == "black":
            if target_size is None:
                raise ValueError(
                    f"Synthetic camera {key!r} requires target_size in FinchDataSpec; "
                    "declare a fixed image size or fixed-size Image transforms in training."
                )
            out[key] = np.zeros((int(target_size[0]), int(target_size[1]), 3), dtype=np.uint8)
            continue
        raise KeyError(f"Unsupported synthetic image mode for camera {key!r}: {synthetic.get(key)!r}")
    return out


def _resolve_target_size(exemplar: Any) -> tuple[int, int] | None:
    """Fixed (height, width) the deploy payload must resize to, or None.

    Derived from the per-camera ``Image`` transform chain — a
    ``ResizeWithPad(h, w)`` on every camera pins the size; a chain with no
    size-setting transform means "keep native resolution".
    """
    robot_config = exemplar.robot_config
    if getattr(robot_config, "images", None) is None:
        return None
    sizes = [spec.output_size for spec in robot_config.resolve_images(tuple(str(value) for value in exemplar.cam_keys))]
    fixed = [tuple(int(v) for v in size) for size in sizes if size is not None]
    if not fixed:
        return None
    first = fixed[0]
    if any(size != first for size in fixed[1:]):
        raise ValueError(f"Per-camera output sizes must match to persist FinchDataSpec target_size, got {fixed}")
    return first


def _resolve_finch_names(
    *,
    roots: list[str],
    finch_config_names: list[str] | None,
    shared_finch_config_name: str | None,
) -> dict[str, str]:
    if finch_config_names is not None:
        return {str(Path(root).resolve()): str(name) for root, name in zip(roots, finch_config_names, strict=True)}
    if shared_finch_config_name is not None:
        return {str(Path(root).resolve()): str(shared_finch_config_name) for root in roots}
    return {}


def _restore_action_array(
    pred: np.ndarray,
    norm_stats: dict,
    action_key: str,
    use_quantile: bool = False,
) -> np.ndarray:
    if not norm_stats:
        raise ValueError(
            "restore_action requires loaded norm stats, but data_spec.norm_stats is empty. "
            "Generate norm stats first, then load_data_spec from a run/checkpoint directory that points to them."
        )
    stats_key = action_key
    if stats_key not in norm_stats:
        stats_key = "action" if "action" in norm_stats else "actions"
    stats = norm_stats[stats_key]
    pred_dim = pred.shape[-1]
    ref_stat = stats.get("mean", stats.get("min"))
    if ref_stat is not None and len(ref_stat) != pred_dim:
        raise ValueError(
            f"restore_action: dimension mismatch — pred has dim={pred_dim} but "
            f"norm_stats['{stats_key}'] has dim={len(ref_stat)}. "
        )
    stats_obj = NormStats(
        mean=np.asarray(stats["mean"], dtype=np.float32)[..., :pred_dim],
        std=np.asarray(stats["std"], dtype=np.float32)[..., :pred_dim],
        q01=None if stats.get("q01") is None else np.asarray(stats["q01"], dtype=np.float32)[..., :pred_dim],
        q99=None if stats.get("q99") is None else np.asarray(stats["q99"], dtype=np.float32)[..., :pred_dim],
    )
    return unnormalize_array(np.asarray(pred, dtype=np.float32), stats_obj, use_quantiles=use_quantile)


def _normalize_norm_stats_path(path: str | None) -> str | None:
    if path is None:
        return None
    resolved = Path(path)
    if resolved.is_dir():
        return str((resolved / "norm_stats.json").resolve())
    return str(resolved.resolve())


def _eef_mode_for_key(key: str, dim: int) -> str:
    if "eef" not in key:
        return "joint"
    if dim == 20:
        return "ee6d"
    if dim == 14:
        return "euler"
    return "eef"


def _infer_robot_name(payload: dict[str, Any], *, robot_type: str | None) -> str | None:
    if robot_type is not None:
        return robot_type
    finch_names_by_root = dict(payload.get("finch_config_names_by_root") or {})
    config_modules = list(payload.get("config_modules") or ())
    if finch_names_by_root:
        first_root, first_name = next(iter(finch_names_by_root.items()))
        resolved = resolve_finch_robot_configs(
            repo_ids=[first_root],
            finch_config_names=[first_name],
            config_modules=config_modules,
        )
        return resolved[0].robot_config.robot_name
    return None


def _validate_homogeneous_data_spec(resolved: list[Any]) -> None:
    exemplar = resolved[0]
    for other in resolved[1:]:
        if (
            other.action_key_forward != exemplar.action_key_forward
            or other.state_key_forward != exemplar.state_key_forward
            or other.action_dim != exemplar.action_dim
            or other.state_dim != exemplar.state_dim
            or other.action_horizon != exemplar.action_horizon
            or tuple(other.cam_keys) != tuple(exemplar.cam_keys)
            or _resolve_target_size(other) != _resolve_target_size(exemplar)
        ):
            raise ValueError(
                "Resolved Finch configs do not share a single Finch data spec. "
                "tau-0-vla currently requires homogeneous forward keys, dims, horizon, cameras, and image size."
            )


def _is_data_mix(config: Any) -> bool:
    return hasattr(config, "configs") and hasattr(config, "proportions") and not isinstance(config, RobotConfig)


# Public aliases for the training-side data-spec write path. Policy-repo
# training code (`openpi` / `tau-0-vla`) uses these to persist a
# `FinchDataSpec` artifact alongside each checkpoint. The three-step split
# (``resolve_finch_names`` → ``build_data_spec`` → ``write_data_spec``) gives
# trainers control over resolving per-root config names before assembling the
# spec payload. Defined after the underscore originals so the references bind.
build_data_spec = _build_data_spec
resolve_finch_names = _resolve_finch_names
write_data_spec = _write_data_spec


__all__ = [
    "FinchDataSpec",
    "action_decoder",
    "build_data_spec",
    "build_state_normalizer",
    "detect_quat_order",
    "load_data_spec",
    "normalize_state",
    "resolve_finch_names",
    "restore_action",
    "restore_unified_action",
    "save_data_spec",
    "write_data_spec",
]
