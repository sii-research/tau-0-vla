"""Assorted helpers: rank-aware printing, YAML config loading, run-directory
setup, and JSON serialisation of NumPy scalars.
"""

import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch.distributed as dist
import yaml


def rank0_print(*args):
    """Print only on rank 0 under distributed training.

    Equivalent to a plain ``print`` outside a distributed run.

    Args:
        *args: Any positional arguments, forwarded to ``print``.

    Example:
        >>> rank0_print("Training started", "step=1")
        Training started step=1
    """
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)


def rank_print(*args):
    """Print on every process, prefixed with the current rank.

    For debugging situations where every process's state matters.

    Args:
        *args: Any positional arguments, forwarded to ``print``.

    Example:
        >>> rank_print("device ready")  # doctest: +SKIP
        Rank 0: device ready
    """
    if dist.is_initialized():
        print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)


def convert_numpy_to_json(obj: Any) -> Any:
    """Recursively convert NumPy arrays and nested structures to JSON-serializable form.

    Used to save dataset statistics — means, standard deviations and the like,
    normally held as ``np.ndarray`` — to a JSON file.

    Args:
        obj: The object to convert. Supported types:
            - ``np.ndarray``: converted to a nested Python list.
            - ``dict``: every value converted recursively.
            - ``list``: every element converted recursively.
            - anything else: returned unchanged.

    Returns:
        A JSON-serializable Python object (list, dict, or a primitive value).

    Example:
        >>> import numpy as np
        >>> convert_numpy_to_json({"mean": np.array([1.0, 2.0])})
        {'mean': [1.0, 2.0]}
        >>> convert_numpy_to_json([np.array([1, 2]), 3])
        [[1, 2], 3]
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_json(i) for i in obj]
    else:
        return obj


# Fields where transformers spells the "off" value as the *string* ``"none"``.
# ``TrainingArguments.__post_init__`` maps ``"none"``/``["none"]`` to ``[]``, but a
# real ``None`` falls through to ``[None]`` and the callback lookup then raises
# ``None is not supported, only azure_ml, ...``. Other fields — ``deepspeed`` is
# typed ``dict | str | None`` — genuinely want ``None``, so the exception is keyed
# to the field rather than to the token.
_STRING_NONE_KEYS = frozenset({"report_to"})


def _coerce_cli_value(val: str, key: str):
    """Coerce a CLI string into a native Python type (int / float / bool / None).

    ``key`` is required rather than defaulted: the ``"none"`` handling below is
    per-field, and a caller that omitted it would silently get the wrong one.
    """
    if val.lower() in ("true", "false"):
        return val.lower() == "true"
    if val.lower() == "none":
        return "none" if key in _STRING_NONE_KEYS else None
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def load_config_from_yaml(yaml_path: str, overrides: dict | None = None) -> dict:
    """Load the training config from a YAML file, filling in paths as it goes.

    ``output_dir`` is joined with ``run_name`` automatically, which keeps repeated
    experiments in separate directories.

    ``overrides`` replaces arbitrary fields after loading but before path
    derivation — convenient when a multi-node experiment needs to change only
    ``run_name`` or a similarly small set of fields.

    Key precedence within ``overrides``:
    - ``run_name`` overrides ``experiment.run_name``, and so also feeds
      ``output_dir`` derivation
    - ``experiment.*`` overrides the matching field in the ``experiment`` block
    - ``model_args.*`` overrides the matching field in ``model_args``
    - ``data_args.*`` overrides the matching field in ``data_args``
    - every other key overrides the matching field in ``training_args``

    Args:
        yaml_path: Path to the YAML config file.
        overrides: Optional overrides, mapping field name to new value. Values
            arrive as strings and are coerced here by ``_coerce_cli_value``;
            ``HfArgumentParser.parse_dict`` hands them to the dataclass as-is.

    Returns:
        The parsed config dict, with top-level keys ``experiment``,
        ``model_args``, ``data_args``, and ``training_args``.

    Raises:
        FileNotFoundError: When ``yaml_path`` does not exist.
        KeyError: When the YAML omits a required top-level key, such as
            ``experiment.run_name``.

    Example:
        >>> import os
        >>> os.environ["CLUSTER_NAME"] = "agibot"
        >>> cfg = load_config_from_yaml("configs/qwen3vl_2b_only_vlm.yaml")  # doctest: +SKIP
        >>> cfg["experiment"]["run_name"]
        'qwen3vl_2b_only_vlm'
        >>> cfg2 = load_config_from_yaml(  # doctest: +SKIP
        ...     "configs/qwen3vl_2b_only_vlm.yaml",
        ...     overrides={"run_name": "exp-2nodes"},
        ... )
        >>> cfg2["experiment"]["run_name"]  # doctest: +SKIP
        'exp-2nodes'
    """
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    # model_name_or_path must come from the YAML: there is no portable default
    # weight path to fall back to.
    if not config["model_args"].get("model_name_or_path"):
        raise KeyError(
            "model_args.model_name_or_path is required — set it to your VLM "
            "checkpoint directory (the shipped configs carry a "
            "<PATH_TO_YOUR_CHECKPOINT> placeholder)."
        )

    # Apply overrides BEFORE deriving output_dir, so a run_name override reaches it.
    _experiment_keys = {"run_name", "project_name", "cluster", "notes"}
    _model_keys = set(config.get("model_args", {}).keys())
    _data_keys = set(config.get("data_args", {}).keys())
    for key, val in (overrides or {}).items():
        # CLI overrides always arrive as strings; coerce against the type the YAML
        # already holds.
        if isinstance(val, str):
            val = _coerce_cli_value(val, key)
        if key in _experiment_keys:
            config["experiment"][key] = val
        elif key in _model_keys:
            config["model_args"][key] = val
        elif key in _data_keys:
            config["data_args"][key] = val
        else:
            config["training_args"][key] = val

    # Join the output path.
    run_name = config["experiment"]["run_name"]
    config["training_args"]["output_dir"] = os.path.join(config["training_args"]["output_dir"], run_name)
    config["training_args"]["run_name"] = run_name

    return config


def setup_experiment_directory(
    output_dir: str,
    cfg_path: Optional[str],
    training_args: Any,
    resolved_args_dict: Optional[dict] = None,
    notes: str = "",
):
    """Set up the experiment directory, archiving the config and recording metadata.

    Runs on the main process only (``process_index == 0``), so a distributed run
    does not write the same files many times over.

    It does four things:

    1. Creates ``output_dir`` if it does not exist.
    2. Copies the YAML config into ``output_dir``. If a file of that name is
       already there, the old one is archived under a timestamped suffix rather
       than overwritten, so earlier runs stay recoverable.
    3. Serializes the fully resolved argument dict to ``resolved_config_full.yaml``.
    4. Records experiment metadata — run_name, git hash, start time, user, command
       line — in ``metadata.yaml``.

    Args:
        output_dir: Root path for the experiment's outputs. Created if missing.
        cfg_path: Path to the original YAML config. ``None`` skips the config backup.
        training_args: The ``transformers.TrainingArguments``, read for
            ``process_index`` and ``run_name``.
        resolved_args_dict: The final argument dict after processing, holding the
            ``model_args``, ``data_args`` and ``training_args`` sub-dicts. ``None``
            skips saving it.
        notes: Free-text notes about this run, written into ``metadata.yaml``.

    Raises:
        PermissionError: When ``output_dir`` cannot be created.

    Example:
        >>> from unittest.mock import MagicMock
        >>> training_args = MagicMock(process_index=0, run_name="my_run")
        >>> setup_experiment_directory(  # doctest: +SKIP
        ...     output_dir="/tmp/my_experiment",
        ...     cfg_path="configs/my_config.yaml",
        ...     training_args=training_args,
        ...     notes="Initial baseline run",
        ... )
    """
    # Main process only.
    if training_args.process_index != 0:
        return

    exp_dir = Path(output_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Back up and archive the config file.
    if cfg_path and os.path.exists(cfg_path):
        cfg_file = Path(cfg_path)
        final_path = exp_dir / cfg_file.name

        # Rename any existing file out of the way rather than overwriting it.
        if final_path.exists():
            # Use the old file's mtime as the archive suffix.
            mtime = datetime.fromtimestamp(final_path.stat().st_mtime).strftime("%Y%m%d_%H%M%S")
            archive_path = exp_dir / f"{cfg_file.stem}_{mtime}{cfg_file.suffix}"
            final_path.rename(archive_path)
            print(f"[INFO] Archived previous config to: {archive_path.name}")

        # Copy in the current config under the original name.
        shutil.copyfile(cfg_path, final_path)
        print(f"[INFO] Latest config saved as: {final_path.name}")

    # 2. Save the fully resolved arguments.
    if resolved_args_dict:
        resolved_path = exp_dir / "resolved_config_full.yaml"
        # Archive any previous resolved_config the same way.
        if resolved_path.exists():
            mtime = datetime.fromtimestamp(resolved_path.stat().st_mtime).strftime("%Y%m%d_%H%M%S")
            resolved_path.rename(exp_dir / f"resolved_config_{mtime}.yaml")

        with open(resolved_path, "w", encoding="utf-8") as f:
            yaml.dump(resolved_args_dict, f, default_flow_style=False, allow_unicode=True)

    # 3. Record experiment metadata (git hash, time, user).
    metadata = {
        "run_name": getattr(training_args, "run_name", "unknown"),
        "notes": notes,
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "git_hash": get_git_hash(),
        "user": os.getenv("USER", "unknown"),
        "command": " ".join(sys.argv),
    }

    with open(exp_dir / "metadata.yaml", "w", encoding="utf-8") as f:
        yaml.dump(metadata, f, sort_keys=False, allow_unicode=True)


def get_git_hash() -> str:
    """Return the current repository's git commit hash.

    Recorded in the experiment metadata so a training result can be reproduced.

    Returns:
        The 40-character hexadecimal commit hash, or ``"not_a_git_repo"`` when the
        working directory is not a git repository or the git command fails.

    Example:
        >>> hash_val = get_git_hash()
        >>> isinstance(hash_val, str)
        True
        >>> len(hash_val) in (40, len("not_a_git_repo"))
        True
    """
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "not_a_git_repo"
