"""Run-spec: the immutable, resolved snapshot of one training launch.

Written once by ``trainer.train.train()`` at launch time into
``output_dir/run_spec.json`` and mirrored into every ``checkpoint-N/`` by
``FinchDataSpecCallback``. Downstream tools (eval, deploy, resume) read it
to reconstruct the exact dataclasses the training run used — no more
re-parsing the yaml or re-passing ``--config_name`` by hand.

Principle: yaml is the *source*, run_spec is the *truth*. Any change to
yaml → new experiment dir.

Shape (``run_spec.json``)::

    {
        "spec_version": 1,
        "model_args": {...},  # vars(ModelArguments) at launch
        "data_args": {...},  # vars(DataArguments)  at launch
        "training_args": {...},  # TrainingArguments.to_dict() at launch
        "yaml_path": "configs/.../foo.yaml" | null,
        "yaml_sha256": "abcd..." | null,
        "git_hash": "<sha>" | "not_a_git_repo",
        "git_dirty": true | false,
        "start_time": "YYYY-MM-DD HH:MM:SS",
        "command": "python ... foo.yaml",
        "notes": "...",
    }
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

SPEC_FILENAME = "run_spec.json"
SPEC_VERSION = 1

# Legacy companion written by setup_experiment_directory since long before
# run_spec existed. Kept as a fallback source for load_run_spec so
# pre-refactor checkpoints keep loading via the same single entry point.
_LEGACY_YAML_FILENAME = "resolved_config_full.yaml"


@dataclass
class RunSpec:
    spec_version: int
    model_args: Dict[str, Any]
    data_args: Dict[str, Any]
    training_args: Dict[str, Any]
    yaml_path: Optional[str]
    yaml_sha256: Optional[str]
    git_hash: str
    git_dirty: bool
    start_time: str
    command: str
    notes: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Build / save
# ─────────────────────────────────────────────────────────────────────────────


def build_run_spec(
    *,
    model_args: Any,
    data_args: Any,
    training_args: Any,
    yaml_path: Optional[str],
    notes: str = "",
) -> RunSpec:
    """Snapshot the three HF dataclasses into a ``RunSpec``.

    Uses ``vars(...)`` for plain dataclasses and ``.to_dict()`` for HF
    ``TrainingArguments``. Runtime attributes that have been monkey-patched
    onto ``data_args`` (e.g. ``vlm_model_type``, ``data_seed``) before this
    call will be captured — that is intentional, those values are part of
    the resolved state downstream tools need.
    """
    model_dict = dict(vars(model_args))
    data_dict = dict(vars(data_args))
    training_dict = training_args.to_dict() if hasattr(training_args, "to_dict") else dict(vars(training_args))
    return RunSpec(
        spec_version=SPEC_VERSION,
        model_args=model_dict,
        data_args=data_dict,
        training_args=training_dict,
        yaml_path=str(yaml_path) if yaml_path else None,
        yaml_sha256=_sha256_of_file(yaml_path) if yaml_path else None,
        git_hash=_git_hash(),
        git_dirty=_git_dirty(),
        start_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        command=" ".join(sys.argv),
        notes=notes,
    )


def save_run_spec(output_dir: str, spec: RunSpec) -> Path:
    """Write ``<output_dir>/run_spec.json``. Safe to call on every rank; we
    only write on rank 0 via the caller's guard, but the write itself is
    idempotent enough that concurrent writes would not corrupt anything
    worse than a momentarily torn read."""
    path = Path(output_dir) / SPEC_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(asdict(spec), handle, indent=2, ensure_ascii=False, default=_json_default)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────


def load_run_spec(ckpt_or_output_dir: str) -> RunSpec:
    """Return the raw ``RunSpec`` read from disk.

    Resolution order:

    1. ``<dir>/run_spec.json``                 — canonical, written by train.py
       at launch (and mirrored into every ``checkpoint-N/`` by
       ``FinchDataSpecCallback``).
    2. ``<dir>/resolved_config_full.yaml``     — legacy, written by
       ``setup_experiment_directory`` since well before ``run_spec`` existed.
       Synthesized into a minimal ``RunSpec`` so pre-refactor checkpoints
       keep loading.
    3. Same search, one level up (parent ``output_dir``).

    Raises ``FileNotFoundError`` only when neither file is found in either
    location.
    """
    path = Path(ckpt_or_output_dir)
    # Accept a direct path to either file
    if path.is_file():
        if path.name == SPEC_FILENAME:
            return _load_json_spec(path)
        if path.name == _LEGACY_YAML_FILENAME:
            return _load_legacy_yaml_spec(path)

    for candidate_dir in (path, path.parent):
        json_path = candidate_dir / SPEC_FILENAME
        if json_path.exists():
            return _load_json_spec(json_path)
        yaml_path = candidate_dir / _LEGACY_YAML_FILENAME
        if yaml_path.exists():
            return _load_legacy_yaml_spec(yaml_path)

    raise FileNotFoundError(
        f"Could not find {SPEC_FILENAME} nor {_LEGACY_YAML_FILENAME} under "
        f"{ckpt_or_output_dir} or its parent. Checkpoints written by older "
        f"training code should at least have the yaml — re-save or point at "
        f"the output_dir directly."
    )


def load_resolved_args(
    ckpt_or_output_dir: str,
    *,
    inference: bool = False,
) -> Tuple[Any, Any, Any, RunSpec]:
    """Reconstruct ``(ModelArguments, DataArguments, TrainingArguments, RunSpec)``
    from a saved run_spec. Downstream callers (eval, deploy, resume) use
    this instead of re-parsing yaml/CLI.

    Unknown keys in the saved dicts (e.g. HF-internal derived fields in
    ``TrainingArguments.to_dict()``) are tolerated via
    ``allow_extra_keys=True`` on the HF parser and explicit filtering on
    the plain dataclasses.
    """
    from transformers import HfArgumentParser

    from tau0_vla.configs.data_config import DataArguments
    from tau0_vla.configs.model_config import ModelArguments
    from tau0_vla.configs.training_config import TrainingArguments

    spec = load_run_spec(ckpt_or_output_dir)

    def _filter_declared(raw: Dict[str, Any], cls) -> Dict[str, Any]:
        declared = set(cls.__dataclass_fields__.keys())
        return {k: v for k, v in raw.items() if k in declared}

    model_args = ModelArguments(**_filter_declared(spec.model_args, ModelArguments))
    data_args = DataArguments(**_filter_declared(spec.data_args, DataArguments))
    # Restore runtime attrs that train.py monkey-patches onto data_args so
    # downstream code sees the exact same shape it saw during training.
    for runtime_key in ("vlm_model_type", "data_seed"):
        if runtime_key in spec.data_args and not hasattr(data_args, runtime_key):
            setattr(data_args, runtime_key, spec.data_args[runtime_key])

    training_parser = HfArgumentParser(TrainingArguments)
    training_values = dict(spec.training_args)
    if inference:
        # Training snapshots can retain a CUDA/DeepSpeed launcher setting even
        # though deployment only needs the dataclass values used by the model.
        # Avoid importing the optional training-only package on inference paths.
        training_values["deepspeed"] = None
    (training_args,) = training_parser.parse_dict(training_values, allow_extra_keys=True)
    return model_args, data_args, training_args, spec


# ─────────────────────────────────────────────────────────────────────────────
# Internals
# ─────────────────────────────────────────────────────────────────────────────


def _load_json_spec(path: Path) -> RunSpec:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    # Forward-compat: drop unknown keys so a future version bump doesn't
    # crash old readers running against a new spec.
    valid = set(RunSpec.__dataclass_fields__.keys())
    return RunSpec(**{k: v for k, v in data.items() if k in valid})


def _load_legacy_yaml_spec(path: Path) -> RunSpec:
    """Synthesize a ``RunSpec`` from the legacy ``resolved_config_full.yaml``.

    That yaml was written by ``setup_experiment_directory`` long before
    run_spec existed; it carries the three HF dataclass dicts but no
    provenance (git_hash, yaml_sha, etc.). Provenance is left blank here;
    downstream code that needs it should re-train or resume once so
    ``train.py`` writes a real ``run_spec.json`` alongside the yaml.
    """
    import yaml  # local import; only legacy path needs it

    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    return RunSpec(
        spec_version=SPEC_VERSION,
        model_args=dict(cfg.get("model_args", {})),
        data_args=dict(cfg.get("data_args", {})),
        training_args=dict(cfg.get("training_args", {})),
        yaml_path=None,
        yaml_sha256=None,
        git_hash="legacy_yaml_no_provenance",
        git_dirty=False,
        start_time="",
        command="",
        notes=f"synthesized from legacy {path.name}",
    )


def _sha256_of_file(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    try:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except OSError:
        return None


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("ascii").strip()
    except Exception:
        return "not_a_git_repo"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode("ascii")
        return bool(out.strip())
    except Exception:
        return False


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "value"):  # enum-like (IntervalStrategy, SchedulerType, ...)
        return obj.value
    if hasattr(obj, "__fspath__"):
        return os.fspath(obj)
    return str(obj)


__all__ = [
    "RunSpec",
    "SPEC_FILENAME",
    "SPEC_VERSION",
    "build_run_spec",
    "save_run_spec",
    "load_run_spec",
    "load_resolved_args",
]
