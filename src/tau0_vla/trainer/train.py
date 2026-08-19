"""Training entry point.

- :func:`action_forward` — one forward pass over an action batch.
- :func:`train` — the whole run, configured either from a YAML file or from
  command-line flags.

Usually launched through ``scripts/train.sh``, which sets up the torchrun
topology first. Directly::

    # from a YAML config
    python src/tau0_vla/trainer/train.py configs/my_task/train.yaml

    # or with flags
    python src/tau0_vla/trainer/train.py \
        --model_name_or_path /path/to/model \
        --output_dir /path/to/output
"""

import ast
import json
import logging
import os
import pathlib
import sys
from pathlib import Path

import torch
import transformers
from torch.utils.data import DataLoader, DistributedSampler

# TF32: let the H100/H200 Tensor Cores run matmul/conv at TF32 precision. The
# effect on bf16 training is negligible, the throughput gain is not. Kept
# alongside HF Trainer's tf32=true, which only covers some of the paths.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

project_root = Path(__file__).parent.parent.parent
repo_root = project_root.parent
sys.path.append(str(project_root))
sys.path.append(str(repo_root))

from tau0_vla.utils.logging import setup_py_logging


def _import_sibling_data(yaml_path: str) -> Path:
    """Load ``<yaml_dir>/data.py`` by file path so its ``@register_config``
    decorators populate the data pipeline's registry before the dataset
    pipeline asks ``get_config(config_name)`` for it.

    The same-directory requirement is load-bearing and easy to miss: move a YAML
    away from its ``data.py`` and its configs never register. Absence is not an
    error here — a run may instead declare ``data_args.config_modules`` — so the
    path is returned for :func:`_populate_finch_fields` to name if the config
    turns out to be unresolvable either way.
    """
    import importlib.util

    data_py = Path(yaml_path).resolve().parent / "data.py"
    if not data_py.exists():
        return data_py
    spec = importlib.util.spec_from_file_location(f"_data_{data_py.parent.name}", data_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return data_py


def _populate_finch_fields(config_dict: dict, sibling_data_py: Path | None = None) -> None:
    """Populate ``state_dim`` / ``action_dim`` / ``action_horizon`` from the
    matching ``tau0_vla.data`` ``@register_config`` referenced by
    ``data_args.config_name``.

    These fields are architecture-critical and must come from the finch data
    config. YAML should not pin them because stale values can silently build an
    action head that disagrees with the dataset payload.

    ``data_args.config_name`` (or ``config_names``) MUST be set explicitly;
    we deliberately do not fall back to ``experiment.run_name`` so that the
    run identifier and the data-config identifier can vary independently.
    """
    data_args = config_dict.setdefault("data_args", {})
    derived_field_names = ("state_dim", "action_dim", "action_horizon")
    pinned_fields = [name for name in derived_field_names if name in data_args]
    if pinned_fields:
        raise ValueError(
            "Do not set data_args.state_dim/action_dim/action_horizon in YAML. "
            "They are derived from data_args.config_name to keep the model head "
            f"consistent with the finch dataset. Remove these keys: {pinned_fields}"
        )

    config_name = data_args.get("config_name")
    if not config_name:
        if data_args.get("config_names"):
            return
        raise ValueError(
            "data_args.config_name is required (no fallback to experiment.run_name). "
            "Set it to the name of a @register_config in your data.py, e.g.\n"
            "  data_args:\n    config_name: pick_red_block_eef_franka\n"
            "or use data_args.config_names for a multi-config mix."
        )

    # No try/except around the import: ``tau0_vla.data`` is in-tree, so a failure
    # here means a broken install, not an optional dependency. Swallowing it
    # would leave state_dim/action_dim/action_horizon underived and build an
    # action head that disagrees with the data — the exact failure the pinned-
    # field check above raises loudly about.
    from tau0_vla.data.config import discover_config_modules, get_config, list_configs

    declared_modules = [str(m) for m in (data_args.get("config_modules") or [])]
    try:
        cfg = get_config(config_name)
    except KeyError:
        # A config declared through ``data_args.config_modules`` is not registered
        # yet — the discovery pass in ``train()`` runs much later, after argument
        # parsing. Import those modules here so derivation sees them. Discovery is
        # idempotent (it tracks what it has already imported), so the later call
        # becomes a no-op rather than a double import.
        if declared_modules:
            discover_config_modules(module_names=declared_modules)
        try:
            cfg = get_config(config_name)
        except KeyError:
            # Do NOT fall through. ``state_dim`` / ``action_dim`` /
            # ``action_horizon`` would keep their DataArguments defaults of 16,
            # and ModelBuilder would then *shrink* a 40D checkpoint's action and
            # state projections to 16D — logging it as though it were intended,
            # and discarding the pretrained weights outside the first 16 dims.
            # An unresolvable config name has to stop the run here.
            searched = [f"config_modules={declared_modules}" if declared_modules else "no config_modules declared"]
            if sibling_data_py is not None:
                searched.append(
                    f"sibling data.py at {sibling_data_py}"
                    f" ({'present' if sibling_data_py.exists() else 'ABSENT'})"
                )
            raise ValueError(
                f"data_args.config_name={config_name!r} is not a registered config.\n"
                f"  searched: {'; '.join(searched)}\n"
                f"  registered: {sorted(list_configs()) or '(none)'}\n"
                "\n"
                "A Robot Config reaches the registry one of two ways:\n"
                "  1. a ``data.py`` in the SAME DIRECTORY as this YAML, which is\n"
                "     imported by file path before the config name is resolved; or\n"
                "  2. ``data_args.config_modules``, a list of importable dotted\n"
                "     module paths, each calling @register_config at import time.\n"
                "\n"
                "This cannot be allowed to continue: state_dim / action_dim are "
                "derived from the config, and the fallback would build an action "
                "head that disagrees with the data."
            ) from None

    if cfg.state_padding_dim is not None:
        data_args.setdefault("state_dim", int(cfg.state_padding_dim))
    if cfg.action_padding_dim is not None:
        data_args.setdefault("action_dim", int(cfg.action_padding_dim))
    if cfg.action_horizon is not None:
        data_args.setdefault("action_horizon", int(cfg.action_horizon))


def action_forward(model, inputs, loss_scale=1.0):
    """Forward one action batch and return its scaled flow-matching loss.

    Passed to :class:`~tau0_vla.trainer.trainer.Tau0VLATrainer` as
    ``forward_fn``.

    Args:
        model: The model under training, already on device.
        inputs: Preprocessed batch, already on device.
        loss_scale: Scale applied to the loss before returning it.

    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
            - the scaled loss, with its graph intact for backward;
            - a detached dict of per-term losses for logging.
    """
    # --- DataParallel workaround for Qwen VL pixel_values ---
    # Qwen VL pixel_values has shape [num_patches, dim] (not batch-first).
    # DataParallel naively splits dim 0, corrupting the vision input.
    # When wrapped in DataParallel and pixel_values is present, unwrap and
    # call the underlying module directly on the primary device to avoid scatter.
    if isinstance(model, torch.nn.DataParallel) and "pixel_values" in inputs:
        model = model.module

    total_loss = 0
    outputs = {}  # for logging only — must not hold the graph

    inputs.pop("dataset_ids", None)
    inputs.pop("labels", None)
    model_output = model(**inputs)

    items = model_output.items() if isinstance(model_output, dict) else vars(model_output).items()
    for key, value in items:
        if not key.startswith("profile_") or value is None:
            continue
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            value = value.detach().float().item()
        outputs[key] = float(value)

    if hasattr(model_output, "vla_loss") and model_output.vla_loss is not None:
        outputs["vla_loss"] = model_output.vla_loss.detach()
        total_loss += model_output.vla_loss * loss_scale

    return total_loss, {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in outputs.items()}


def train():
    """Main entry point for the full tau-0-vla training run.

    Two ways to supply arguments:

    1. **YAML mode** (recommended): when the first command-line argument ends in
       ``.yaml``, the config is loaded through
       :func:`~tau0_vla.utils.utils.load_config_from_yaml`.
    2. **CLI mode** (for older scripts): ``transformers.HfArgumentParser`` parses
       the command line directly.

    **Training flow**:

    1. Parse the configuration, from YAML or the CLI.
    2. Set up the experiment directory, archiving the config and recording metadata.
    3. Build the model with
       :class:`~tau0_vla.models.model_builder.ModelBuilder`.
    4. Create the data module from the finch config named by
       ``data_args.config_name``, and write the data contract
       (``finch_data_spec/`` plus ``policy_manifest.json``) into ``output_dir``
       for openloop and deploy to read later.
    5. Construct :class:`~tau0_vla.trainer.trainer.Tau0VLATrainer` and train.
    6. Save the final model weights and the Processor.

    **Resuming from a checkpoint** is explicit and opt-in:

    - Pass ``--resume_from_checkpoint <path>`` (the standard HF argument) to
      resume from that path.
    - Set ``AUTO_RESUME=1`` and, if ``output_dir`` holds ``checkpoint-*``
      directories, the newest one is resumed from.
    - Otherwise, finding leftover ``checkpoint-*`` in ``output_dir`` is a **hard
      error**, rather than silently continuing from a state that may have
      diverged. Clear them or opt in explicitly.

    Raises:
        SystemExit: When argument parsing fails; raised by HfArgumentParser.
        RuntimeError: When model construction or data loading fails.

    Example:
        Launch training from a YAML::

            python src/tau0_vla/trainer/train.py configs/my_task/train.yaml

        Launch training from the CLI::

            python src/tau0_vla/trainer/train.py \\
                --model_name_or_path /path/to/model \\
                --output_dir /tmp/output \\
                --config_name my_task
    """
    setup_py_logging(only_main_process=False)

    from tau0_vla.configs.data_config import DataArguments
    from tau0_vla.configs.model_config import ModelArguments
    from tau0_vla.configs.training_config import TrainingArguments
    from tau0_vla.models.model_builder import ModelBuilder
    from tau0_vla.trainer.trainer import (
        FinchDataSpecCallback,
        SaveProcessorCallback,
        Tau0VLATrainer,
        safe_save_model_for_hf_trainer,
    )
    from tau0_vla.utils.run_spec import build_run_spec, save_run_spec
    from tau0_vla.utils.utils import convert_numpy_to_json, load_config_from_yaml, setup_experiment_directory
    from tau0_vla.vlm.dataset import make_supervised_finch_data_module

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    # --- Argument preparation ---
    # Two modes:
    #   (a) ``train.py configs/<family>/<variant>.yaml [--key value ...]``
    #   (b) ``train.py --flag value ...``                     (HF CLI)
    yaml_config_path = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".yaml") else None

    if yaml_config_path:
        overrides: dict = {}
        extra = sys.argv[2:]
        i = 0
        while i < len(extra):
            if extra[i].startswith("--") and i + 1 < len(extra) and not extra[i + 1].startswith("--"):
                overrides[extra[i][2:]] = extra[i + 1]
                i += 2
            else:
                i += 1

        config_dict = load_config_from_yaml(yaml_config_path, overrides=overrides)
        # Import sibling ``data.py`` so its @register_config decorators fire
        # before tau0_vla.data.get_config is queried. Auto-fill ``config_name`` +
        # padding dims on data_args if the yaml didn't pin them.
        sibling_data_py = _import_sibling_data(yaml_config_path)
        _populate_finch_fields(config_dict, sibling_data_py=sibling_data_py)

        os.environ["WANDB_PROJECT"] = config_dict["experiment"].get("project_name", "default_vla")
        model_args, data_args, training_args = parser.parse_dict(
            {
                **config_dict.get("model_args", {}),
                **config_dict.get("data_args", {}),
                **config_dict.get("training_args", {}),
                "run_name": os.environ.get("RUN_NAME", config_dict["experiment"]["run_name"]),
            }
        )
        notes = config_dict["experiment"].get("notes", "")
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        notes = "Launched via CLI"

    # --- Detect the GPU count and append it to run_name ---
    world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count() or 1))
    num_nodes = int(os.environ.get("NNODES", os.environ.get("SLURM_NNODES", 1)))
    gpus_per_node = world_size // max(num_nodes, 1)
    if num_nodes > 1:
        gpu_tag = f"_{num_nodes}x{gpus_per_node}gpu"
    else:
        gpu_tag = f"_{gpus_per_node}gpu"
    if hasattr(training_args, "run_name") and training_args.run_name:
        if not training_args.run_name.endswith(gpu_tag):
            training_args.run_name += gpu_tag
    logging.info(f"run_name: {training_args.run_name} (world_size={world_size}, nodes={num_nodes})")

    # Use the training seed as the default data seed so dataset shuffling/sampling
    # is reproducible and the effective value is captured in the resolved config.
    data_args.data_seed = training_args.seed

    # Every argument in one dict, for the backup.
    all_params = {
        "model_args": vars(model_args),
        "data_args": vars(data_args),
        "training_args": training_args.to_dict(),  # HF TrainingArguments brings its own to_dict
    }

    # Save the experiment configuration.
    setup_experiment_directory(
        output_dir=training_args.output_dir,
        cfg_path=yaml_config_path,
        training_args=training_args,
        resolved_args_dict=all_params,
        notes=notes,
    )

    # run_spec.json is the single canonical resolved snapshot of this launch;
    # eval / deploy / resume read it to reconstruct the exact dataclasses
    # without re-parsing the yaml (see tau0_vla.utils.run_spec). Rank-0
    # only; metadata-level I/O, failures must not abort training.
    if training_args.process_index == 0:
        try:
            spec = build_run_spec(
                model_args=model_args,
                data_args=data_args,
                training_args=training_args,
                yaml_path=yaml_config_path,
                notes=notes,
            )
            save_run_spec(training_args.output_dir, spec)
        except Exception as exc:
            logging.warning(f"save_run_spec failed: {exc}")

    os.makedirs(training_args.output_dir, exist_ok=True)

    # --- Model preparation ---
    mb = ModelBuilder(training_args, model_args, data_args, is_training=True)
    mb.build()
    model, processor = mb.model, mb.processor

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # --- Data preparation ---
    # Discover finch @register_config modules so get_config() can resolve
    # whatever config_name the yaml points at. Safe no-op if not set.
    if data_args.config_modules:
        from tau0_vla.data.config import discover_config_modules

        discover_config_modules(module_names=list(data_args.config_modules))

    # The output_dir data-contract artifacts (finch_data_spec/, dataset_stats,
    # policy_manifest) are identical across ranks — only global rank 0 writes
    # them. Without this guard every rank of every launch rewrote the spec
    # tree, and (with the old per-process inline-config names) never
    # overwrote: 294k stale dirs accumulated in one 768-rank run dir, turning
    # every checkpoint mirror into a ~70 min copy.
    is_artifact_writer = int(os.environ.get("RANK", "0")) == 0

    data_module = make_supervised_finch_data_module(processor, data_args)
    train_dataset = data_module["train_dataset"]

    # Finch dataset owns the training data contract: write data_spec
    # (``finch_data_spec/``) so deploy can reconstruct it later.
    # Failures here must be loud: a training run that proceeds with a
    # missing data_spec silently ships ckpts that deploy cannot open.
    if is_artifact_writer and hasattr(train_dataset, "save_data_spec"):
        train_dataset.save_data_spec(training_args.output_dir)

    if is_artifact_writer and hasattr(train_dataset, "stats"):
        with open(os.path.join(training_args.output_dir, "dataset_stats.json"), "w") as f:
            json.dump(convert_numpy_to_json(train_dataset.stats), f, indent=4)

    # Companion manifest ``policy_manifest.json``. ``FinchDataSpecCallback``
    # will mirror both into every ``checkpoint-N/`` at save time; deploy reads
    # the manifest through ``tau0_vla.data.load_checkpoint_spec(ckpt_dir)``.
    # Any failure here must propagate for the same reason: a ckpt shipped
    # without policy_manifest.json is useless to open-loop / deploy tools.
    if data_args.config_name and is_artifact_writer:
        from tau0_vla.data import save_policy_manifest

        manifest_policy_type = getattr(model_args, "vla_model_type", None) or "vlm"
        train_config_name = getattr(training_args, "run_name", None) or data_args.config_name or "tau0_vla"
        save_policy_manifest(
            Path(training_args.output_dir) / "policy_manifest.json",
            train_config_name=str(train_config_name),
            finch_config_name=data_args.config_name,
            policy_type=str(manifest_policy_type),
            config_modules=list(data_args.config_modules or []),
        )

    # --- DataLoader ---
    num_workers = getattr(training_args, "dataloader_num_workers", 8)
    ds_len = len(train_dataset)
    if ds_len <= 0:
        raise ValueError(
            f"Dataset resolved to zero samples before DataLoader construction. "
            f"config_name={getattr(data_args, 'config_name', None)!r}, "
            f"physically_split={getattr(train_dataset, 'physically_split', False)}"
        )
    _use_manual_sampler = bool(getattr(train_dataset, "use_manual_distributed_sampler", False))
    _manual_sampler = None
    if _use_manual_sampler:
        _manual_sampler = DistributedSampler(
            train_dataset,
            num_replicas=int(getattr(train_dataset, "logical_shard_world_size", 1)),
            rank=int(getattr(train_dataset, "logical_shard_rank", 0)),
            shuffle=True,
            drop_last=getattr(training_args, "dataloader_drop_last", True),
            seed=getattr(training_args, "seed", 42),
        )
    dl_kwargs = dict(
        dataset=train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        collate_fn=data_module["data_collator"],
        # In grouped physical split mode the dataset is split into physical
        # shards first; ranks sharing one shard are divided by this manual
        # DistributedSampler. Other modes keep the existing shuffle path.
        shuffle=not _use_manual_sampler,
        num_workers=num_workers,
        pin_memory=getattr(training_args, "dataloader_pin_memory", True),
        drop_last=getattr(training_args, "dataloader_drop_last", True),
    )
    if _manual_sampler is not None:
        dl_kwargs["sampler"] = _manual_sampler
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = getattr(training_args, "dataloader_persistent_workers", False)
        prefetch_factor = getattr(training_args, "dataloader_prefetch_factor", None)
        if prefetch_factor is not None:
            dl_kwargs["prefetch_factor"] = prefetch_factor
    train_dataloader = DataLoader(**dl_kwargs)
    logging.info(
        f"Dataset: size={ds_len}, batch_size={training_args.per_device_train_batch_size}, "
        f"physically_split={getattr(train_dataset, 'physically_split', False)}, "
        f"manual_sampler={_use_manual_sampler}, "
        f"physical_rank={getattr(train_dataset, 'physical_shard_rank', 'n/a')}/"
        f"{getattr(train_dataset, 'physical_shard_world_size', 'n/a')}, "
        f"logical_rank={getattr(train_dataset, 'logical_shard_rank', 'n/a')}/"
        f"{getattr(train_dataset, 'logical_shard_world_size', 'n/a')}"
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "processing_class": processor.tokenizer,
        "train_dataset": train_dataset,
        "train_dataloader": train_dataloader,
        "data_collator": data_module["data_collator"],
        "forward_fn": action_forward,
        "log_metrics_sync_across_ranks": getattr(training_args, "log_metrics_sync_across_ranks", True),
    }

    trainer = Tau0VLATrainer(
        **trainer_kwargs,
        callbacks=[SaveProcessorCallback(processor), FinchDataSpecCallback()],
    )

    # --- Start training ---
    # Resume must be opt-in: auto-resuming whenever output_dir contains
    # checkpoint-* silently re-loads stale/diverged state from prior runs
    # (previous incident: a debug run with save_steps=50 left diverged
    # checkpoints behind, and a subsequent full run transparently resumed
    # from loss≈36 instead of training from scratch).
    explicit_resume = getattr(training_args, "resume_from_checkpoint", None)
    auto_resume_env = os.environ.get("AUTO_RESUME", "").strip().lower() in ("1", "true", "yes")
    existing_ckpts = sorted(
        pathlib.Path(training_args.output_dir).glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )

    if explicit_resume:
        logging.info(f"Resuming from checkpoint: {explicit_resume}")
        trainer.train(resume_from_checkpoint=explicit_resume)
    elif auto_resume_env and existing_ckpts:
        latest = existing_ckpts[-1]
        logging.info(f"AUTO_RESUME=1 with {len(existing_ckpts)} checkpoint(s); resuming from {latest.name}")
        trainer.train(resume_from_checkpoint=True)
    elif existing_ckpts:
        recent = [c.name for c in existing_ckpts[-5:]]
        raise RuntimeError(
            f"Found {len(existing_ckpts)} stale checkpoint(s) under {training_args.output_dir}; "
            f"latest: {recent}. Auto-resume is disabled to prevent silently loading "
            "possibly-diverged state from a prior run. Pick one:\n"
            f"  1) rm -rf {training_args.output_dir}/checkpoint-*    # start fresh\n"
            "  2) AUTO_RESUME=1 <launch cmd>                         # resume from latest\n"
            "  3) --resume_from_checkpoint <path>                    # resume from a specific ckpt\n"
            "  4) change run_name / output_dir                       # new experiment dir"
        )
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    processor.save_pretrained(training_args.output_dir)


def _maybe_wait_for_debugger() -> None:
    """Env-gated debugpy hook: ``VLA_DEBUGPY_RANKS=0`` (comma list) makes the
    matching torchrun rank(s) listen on port ``VLA_DEBUGPY_PORT``(default
    5678)+rank and BLOCK until a debugger (VSCode "attach") connects. Unset =
    zero overhead. Non-listed ranks run free; while a listed rank sits at a
    breakpoint the others wait at the next collective (ddp_timeout budget).
    Listens on ``VLA_DEBUGPY_HOST`` (default ``127.0.0.1``); set it only when
    remote attach is genuinely required, since debugpy has no authentication."""
    ranks = os.environ.get("VLA_DEBUGPY_RANKS")
    if not ranks:
        return
    rank = int(os.environ.get("RANK", "0"))
    if rank not in {int(r) for r in ranks.split(",") if r.strip()}:
        return
    import debugpy

    port = int(os.environ.get("VLA_DEBUGPY_PORT", "5678")) + rank
    # debugpy has no authentication: whoever connects first executes arbitrary
    # code in the training process. Default to localhost so enabling the
    # debugger on a shared cluster does not expose RCE to every network peer;
    # VLA_DEBUGPY_HOST is the explicit escape hatch for remote attach.
    host = os.environ.get("VLA_DEBUGPY_HOST", "127.0.0.1")
    debugpy.listen((host, port))
    print(f"[debugpy] rank {rank} listening on {host}:{port} — waiting for attach ...", flush=True)
    debugpy.wait_for_client()


if __name__ == "__main__":
    _maybe_wait_for_debugger()
    train()
