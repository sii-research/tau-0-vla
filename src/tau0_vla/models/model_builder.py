"""Model construction: loading, parameter freezing, LoRA, and the FAST vocabulary
extension behind one call — ``ModelBuilder.build()``.

Two mutually exclusive paths:

- **VLA** (``vla_model_type="tau0_vla"``) — Qwen3.5 plus a Qwen3.5 action
  expert trained by flow matching. This is what the release ships.
- **VLM** (``vla_model_type=None``) — the bare Qwen-VL backbone, no action head.
"""

import os
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import transformers
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoConfig, AutoProcessor, Qwen3VLConfig

from tau0_vla.models.vision_language_action_models.tau0_vla import Tau0VLAConfig, Tau0VLAModel
from tau0_vla.models.vision_language_models.qwen_vl.qwen_vl import QwenVLInterfacewrapper
from tau0_vla.utils.utils import rank0_print
from tau0_vla.vlm.qwenvl_utils import update_qwenvl_processor_pixels


def _resize_linear(old: nn.Linear, new_in: int, new_out: int) -> nn.Linear:
    """Create a new Linear layer sized (new_in->new_out), warm-started from *old*."""
    new = nn.Linear(
        new_in,
        new_out,
        bias=old.bias is not None,
        device=old.weight.device,
        dtype=old.weight.dtype,
    )
    copy_in = min(old.in_features, new_in)
    copy_out = min(old.out_features, new_out)
    with torch.no_grad():
        new.weight.zero_()
        if new.bias is not None:
            new.bias.zero_()
        new.weight[:copy_out, :copy_in].copy_(old.weight[:copy_out, :copy_in])
        if new.bias is not None and old.bias is not None:
            new.bias[:copy_out].copy_(old.bias[:copy_out])
    return new


def _resize_action_state_projections(
    flow_matching,
    new_max_action_dim: int,
    new_max_state_dim: int,
) -> None:
    """Resize state_proj / action_in_proj / action_out_proj in-place.

    Copies overlapping weights so that the first N dims (e.g. the original
    G1 20-D layout) remain warm-started, and the new dims start at zero.
    """
    proj_width = flow_matching.state_proj.out_features
    flow_matching.state_proj = _resize_linear(
        flow_matching.state_proj,
        new_max_state_dim,
        proj_width,
    )
    flow_matching.action_in_proj = _resize_linear(
        flow_matching.action_in_proj,
        new_max_action_dim,
        proj_width,
    )
    flow_matching.action_out_proj = _resize_linear(
        flow_matching.action_out_proj,
        proj_width,
        new_max_action_dim,
    )


class ModelBuilder:
    """Builds the model and its Processor.

    Two mutually exclusive load paths:

    1. **VLA** (``vla_model_type="tau0_vla"``) — :meth:`load_tau_vla_model`. The
       older spelling ``"tau_vla"`` is also accepted, because that is what the
       released checkpoint records in its ``config.json``.
    2. **VLM** (``vla_model_type=None``) — :meth:`load_qwen_vlm_model`, the bare
       Qwen-VL backbone.

    ``build()`` enforces this order, and it is not arbitrary — the FAST tokens
    must be added before LoRA wraps the embedding layer, or the new rows are
    left outside the adapter:

    1. load the model and Processor
    2. replace attention (variable-length / packed sequences)
    3. add the FAST special tokens
    4. freeze parameters / wrap in LoRA
    5. configure gradient checkpointing
    6. update the Processor's pixel range

    Args:
        training_args: :class:`~tau0_vla.configs.training_config.TrainingArguments`.
        model_args: :class:`~tau0_vla.configs.model_config.ModelArguments`.
        data_args: :class:`~tau0_vla.configs.data_config.DataArguments`.
        is_training: When ``False`` (inference), skips freezing and gradient
            checkpointing and leaves the model in ``eval()``.

    Attributes:
        model: The built model; ``None`` until :meth:`build` runs.
        processor: The built Processor; ``None`` until :meth:`build` runs.
        modules_to_save: Module names LoRA keeps in full precision.

    Example:
        >>> from tau0_vla.models.model_builder import ModelBuilder
        >>> mb = ModelBuilder(training_args, model_args, data_args, is_training=True)
        >>> mb.build()
        >>> model, processor = mb.model, mb.processor
    """

    def __init__(self, training_args, model_args, data_args, is_training=False):
        """Initialize the ModelBuilder.

        Args:
            training_args: Training arguments, carrying the LoRA, gradient
                checkpointing, and learning-rate settings.
            model_args: Model arguments, carrying the model path, type, and the
                parameter-freezing flags.
            data_args: Data arguments, used to configure the Processor's pixel
                range and the action dimension.
            is_training: When ``True``, applies parameter freezing, LoRA, and
                gradient checkpointing. When ``False`` (inference), freezes every
                parameter and switches the model to ``eval()`` mode.
        """
        self.is_training = is_training
        self.training_args = training_args
        self.model_args = model_args
        self.data_args = data_args
        self.model = None
        self.processor = None
        self.modules_to_save = None

    def load_qwen_vlm_model(self):
        """Load a standard Qwen-VL model, without an action head.

        :class:`~tau0_vla.models.vision_language_models.qwen_vl.QwenVLInterfacewrapper`
        loads Qwen3-VL, Qwen2.5-VL, Qwen2-VL and friends behind one interface, and
        updates the Processor's pixel range (``max_pixels``, ``min_pixels``) to
        match.

        Sets ``self.model`` and ``self.processor`` once loading completes.

        Raises:
            ValueError: When ``model_args.vlm_model_type`` names an unsupported type.
            OSError: When ``model_args.model_name_or_path`` does not exist.
        """
        vlm_wrapper = QwenVLInterfacewrapper(
            model_args=self.model_args,
            training_args=self.training_args,
            data_args=self.data_args,
        )
        self.model = vlm_wrapper.model
        self.processor = vlm_wrapper.processor

    def load_tau_vla_model(self):
        """Load Tau VLA model (Qwen3.5 + Qwen3.5 action expert with flow matching)."""
        backbone_path = self.model_args.model_name_or_path
        attn_impl = getattr(self.training_args, "attn_implementation", "eager")

        rank0_print(f"Loading Tau VLA model from {backbone_path} ...")

        cfg = AutoConfig.from_pretrained(backbone_path, trust_remote_code=True)

        # Training-related overrides (shared by both paths)
        tune_vision = getattr(self.model_args, "tune_mm_vision", False)
        tune_mlp = getattr(self.model_args, "tune_mm_mlp", False)
        tune_llm = getattr(self.model_args, "tune_mm_llm", False)
        freeze_vision_encoder = not tune_vision
        train_expert_only = not (tune_vision or tune_mlp or tune_llm)
        train_state_proj = getattr(self.model_args, "train_state_proj", False)
        loss_type = getattr(self.model_args, "loss_type", "fm")
        num_steps = getattr(self.model_args, "num_steps", 10)

        if getattr(cfg, "model_type", None) == "tau_vla":
            # SFT from a trained VLA checkpoint — from_pretrained loads all weights
            rank0_print("Detected Tau0VLA checkpoint, loading with from_pretrained ...")

            # Fix FAST token vocab_size mismatch: training's resize_token_embeddings
            # does not propagate to Tau0VLAConfig.qwenvl_config, so the saved
            # config.json may have a stale text_config.vocab_size.  Detect and
            # patch before from_pretrained to avoid embed_tokens shape errors.
            _tokenizer = transformers.AutoTokenizer.from_pretrained(
                backbone_path,
                trust_remote_code=True,
            )
            _real_vocab = len(_tokenizer)
            del _tokenizer
            _cfg_vocab = cfg.vocab_size
            qwenvl_cfg = getattr(cfg, "qwenvl_config", None)
            if isinstance(qwenvl_cfg, dict) and "text_config" in qwenvl_cfg:
                _cfg_vocab = qwenvl_cfg["text_config"].get("vocab_size", _cfg_vocab)
            if _cfg_vocab != _real_vocab:
                rank0_print(
                    f"[FIX] config.json vocab_size={_cfg_vocab} != tokenizer vocab_size={_real_vocab}, patching config."
                )
                cfg.vocab_size = _real_vocab
                if isinstance(qwenvl_cfg, dict):
                    qwenvl_cfg["vocab_size"] = _real_vocab
                    if "text_config" in qwenvl_cfg:
                        qwenvl_cfg["text_config"]["vocab_size"] = _real_vocab

            new_max_action = int(self.data_args.action_dim)
            new_max_state = int(self.data_args.state_dim)
            old_max_action = getattr(cfg, "max_action_dim", new_max_action)
            old_max_state = getattr(cfg, "max_state_dim", new_max_state)
            need_resize = (old_max_action != new_max_action or old_max_state != new_max_state)

            new_action_steps = int(self.data_args.action_horizon)
            saved_action_steps = getattr(cfg, "n_action_steps", None)

            arch_overrides = {
                "max_action_dim": new_max_action,
                "max_state_dim": new_max_state,
                "adanorm_time": getattr(self.model_args, "adanorm_time", None),
            }
            if need_resize:
                arch_overrides.pop("max_action_dim")
                arch_overrides.pop("max_state_dim")
            Tau0VLAModel.validate_arch_compat(cfg, arch_overrides)

            if saved_action_steps != new_action_steps:
                rank0_print(
                    f"Adapting action horizon: {saved_action_steps}->{new_action_steps}. "
                    "Reusing the checkpoint's shared action-token weights."
                )
            cfg.n_action_steps = new_action_steps

            cfg.freeze_vision_encoder = freeze_vision_encoder
            cfg.train_expert_only = train_expert_only
            cfg.train_state_proj = train_state_proj
            cfg.attention_implementation = attn_impl
            cfg.tau_vla_prefix_flash_backend = getattr(self.model_args, "tau_vla_prefix_flash_backend", False)
            cfg.use_action_mask_loss = getattr(self.model_args, "use_action_mask_loss", True)
            cfg.vla_inactive_input_zero = getattr(self.model_args, "vla_inactive_input_zero", False)
            cfg.loss_type = loss_type
            cfg.num_steps = num_steps
            self.model = Tau0VLAModel.from_pretrained(backbone_path, config=cfg)

            if need_resize:
                rank0_print(
                    f"Resizing action/state projections: "
                    f"action {old_max_action}->{new_max_action}, "
                    f"state {old_max_state}->{new_max_state}"
                )
                _resize_action_state_projections(
                    self.model.flow_matching,
                    new_max_action,
                    new_max_state,
                )
                self.model.config.max_action_dim = new_max_action
                self.model.config.max_state_dim = new_max_state
                self.model.config.action_dim = new_max_action
        else:
            # First build from VLM base model — construct Tau0VLA, then load VLM weights
            config = Tau0VLAConfig(
                qwenvl_config=cfg.to_dict(),
                max_action_dim=int(self.data_args.action_dim),
                max_state_dim=int(self.data_args.state_dim),
                action_dim=int(self.data_args.action_dim),
                n_action_steps=int(self.data_args.action_horizon),
                attention_implementation=attn_impl,
                freeze_vision_encoder=freeze_vision_encoder,
                train_expert_only=train_expert_only,
                train_state_proj=train_state_proj,
                use_lm_head=getattr(self.model_args, "use_lm_head", False),
                adanorm_time=getattr(self.model_args, "adanorm_time", False),
                vlm_causal=getattr(self.model_args, "vlm_causal", True),
                tau_vla_prefix_flash_backend=getattr(self.model_args, "tau_vla_prefix_flash_backend", False),
                zero_state_emb=getattr(self.model_args, "zero_state_emb", False),
                use_action_mask_loss=getattr(self.model_args, "use_action_mask_loss", True),
                vla_inactive_input_zero=getattr(self.model_args, "vla_inactive_input_zero", False),
                loss_type=loss_type,
                num_steps=num_steps,
            )
            self.model = Tau0VLAModel.from_vlm_pretrained(backbone_path, config)

        processor = AutoProcessor.from_pretrained(backbone_path, trust_remote_code=True)
        self.processor = update_qwenvl_processor_pixels(processor, self.data_args)

        # Tau0VLA assumes right-padding everywhere: HF FA2 (VLM-only path) builds
        # cu_seqlens under that assumption, joint-attention prefix masks and
        # RoPE position_ids are computed under that assumption. Batch padding is
        # performed by the collator (torch.nn.utils.rnn.pad_sequence, default
        # padding_side='right'), so tokenizer.padding_side is irrelevant here;
        # the collator itself asserts right-padding on its first batch.
        expected_side = getattr(self.data_args, "padding_side", "right")
        if expected_side != "right":
            raise ValueError(f"Tau0VLA requires data_args.padding_side='right', got '{expected_side}'.")

    def prepare_lora_model(self):
        """Wrap the model with PEFT LoRA instead of finetuning every parameter.

        Freezes the whole model, then applies low-rank adaptation to the attention
        linear layers (``q_proj``, ``k_proj``, ``v_proj``, ``o_proj``). Modules
        named in ``modules_to_save`` train at full precision.

        The LoRA hyperparameters (r, alpha, dropout) come from ``training_args``.

        Raises:
            ImportError: When ``peft`` is not installed.

        Example:
            ``build()`` calls this automatically when
            ``training_args.lora_enable=True``. To call it directly:

            >>> mb.prepare_lora_model()  # doctest: +SKIP
            trainable params: 12,582,912 || all params: 2,121,900,032 || trainable%: 0.5930
        """

        rank0_print("LoRA enabled")

        self.model.requires_grad_(False)

        lora_config = LoraConfig(
            r=self.training_args.lora_r or 64,
            lora_alpha=self.training_args.lora_alpha or 128,
            lora_dropout=self.training_args.lora_dropout or 0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ],  # Qwen's attention linear layers
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=getattr(self, "modules_to_save", None),  # modules to train at full precision
        )
        # get_peft_model returns a new, wrapped model.
        self.model = get_peft_model(self.model, lora_config)
        # Report the LoRA configuration.
        self.model.print_trainable_parameters()

    def update_configs(self):
        if self.processor:
            self.processor.tokenizer.model_max_length = self.training_args.model_max_length
        self.data_args.vlm_model_type = self.model_args.vlm_model_type

    def set_training_parameters(self):
        """Set each module's trainable state from the ``tune_*`` flags in model_args.

        Four independent controls:

        - ``tune_mm_vision``: the vision encoder (the ``visual`` module).
        - ``tune_mm_mlp``: the vision-language projector (``visual.merger``).
        - ``tune_mm_llm``: the language-model layers (``language_model`` or
          ``model``) and ``lm_head``.
        - ``tune_vla_dit``: the DiT action head and its projections.

        Note: this must be called *before* LoRA wrapping, so that ``self.model`` is
        always the raw model rather than a PEFT wrapper.

        Raises:
            AttributeError: When the model lacks an expected submodule. Skipped
                silently, by design.
        """
        # a PeftModel. We must NOT use PreTrainedModel.base_model here because some
        # some wrappers set base_model_prefix = "vlm", which
        # makes .base_model return the *inner* VLM submodule rather than the top-level
        # model — breaking all subsequent parameter control.
        base_model = self.model

        # qwenvl
        def set_training_parameters_for_qwenvl(model_args, model):
            # Vision encoder
            if hasattr(model, "visual"):
                model.visual.requires_grad_(model_args.tune_mm_vision)

            # Vision-language projector (the merger)
            if hasattr(model, "visual") and hasattr(model.visual, "merger"):
                model.visual.merger.requires_grad_(model_args.tune_mm_mlp)

            # Language model
            # Standard HuggingFace Qwen3VL uses attribute "language_model";
            # Xiaomi's reimplementation uses "model". Support both.
            text_model = getattr(model, "language_model", None) or getattr(model, "model", None)
            if text_model is not None:
                text_model.requires_grad_(model_args.tune_mm_llm)

            if hasattr(model, "lm_head"):
                model.lm_head.requires_grad_(model_args.tune_mm_llm)

        # Pick the target model.
        vla_type = getattr(self.model_args, "vla_model_type", None)

        if self.model_args.vlm_model_type in ["qwen3vl", "qwen2.5vl", "qwen2vl", "qwen3vl_moe"]:
            if vla_type is None:
                # Tau0VLA handles VLM freezing internally (see below); the plain
                # VLM path configures the backbone directly.
                target_model = getattr(base_model, "qwen", base_model)
                set_training_parameters_for_qwenvl(self.model_args, target_model)

        if vla_type in ("tau_vla", "tau0_vla"):
            # Tau VLA: VLM freeze is handled internally by Qwen35vlWithExpertModel
            tune_dit = getattr(self.model_args, "tune_vla_dit", False)
            fm = getattr(base_model, "flow_matching", base_model)
            for module_name in [
                "state_proj",
                "action_in_proj",
                "action_out_proj",
                "action_time_mlp_in",
                "action_time_mlp_out",
            ]:
                module = getattr(fm, module_name, None)
                if module is not None:
                    module.requires_grad_(tune_dit)
            # Expert model
            expert = getattr(fm.qwenvl_with_expert, "qwen_expert", None)
            if expert is not None:
                expert.requires_grad_(tune_dit)

    def print_trainable_params(self):
        # Only unwrap PeftModel when LoRA is actually enabled; otherwise
        # PreTrainedModel.base_model is a property that returns the sub-module
        # indicated by base_model_prefix, which is wrong for wrapped VLMs.
        target_model = self.model
        if self.training_args.lora_enable and hasattr(self.model, "base_model"):
            target_model = self.model.base_model
        # qwen sub module
        vlm_backbone = getattr(target_model, "qwen", target_model)
        # Try to print trainable parameters from common attribute locations
        if hasattr(vlm_backbone, "visual"):
            try:
                vlm_backbone.visual.print_trainable_parameters()
            except Exception:
                pass
        if hasattr(vlm_backbone, "model"):
            try:
                vlm_backbone.model.print_trainable_parameters()
            except Exception:
                pass

        # total model
        total_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in self.model.parameters())
        total_names = [name for name, p in self.model.named_parameters()]
        trainable_names = [name for name, p in self.model.named_parameters() if p.requires_grad]
        trainable_params = sum(
            p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in self.model.parameters() if p.requires_grad
        )
        frozen_params = sum(
            p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in self.model.parameters() if not p.requires_grad
        )
        print(f"Total parameters: {total_params:,}")
        # print(f"Total parameter names: {total_names}")
        print(f"Total trainable parameters: {trainable_params:,}")
        print(f"Total frozen parameters: {frozen_params:,}")
        print(f"Trainable parameter names: {trainable_names}")

    def add_special_tokens_for_fast(self, fast_vocab_size: int = 2048):
        """Add the FAST discrete-action tokens to the tokenizer and embedding layer.

        FAST tokens are named ``<|fast_0|>`` through ``<|fast_{N-1}|>``. Each new
        token's embedding is initialized to the mean of the existing token
        embeddings, avoiding the instability random initialization would bring.

        This must be called *before* LoRA wrapping, because resizing the embedding
        layer gets considerably harder once the model is wrapped.

        **This must stay idempotent — do not "simplify" it into an
        unconditional append.** FAST training is not shipped, and an
        ``is_training`` gate keeps inference from ever reaching this method, so
        it reads like dead code. It is not: a warm-started finetune does run it,
        against a released checkpoint whose vocabulary *already* contains the
        2048 ``<|fast_i|>`` tokens. ``add_tokens`` recognises them, reports 0
        added, and the length holds at 250125. Appending unconditionally instead
        would resize embeddings to 252173 and silently misalign every weight in
        a finetune warm-started from a released checkpoint — no error, just a
        model that has quietly lost its pretrained embeddings.

        Args:
            fast_vocab_size: How many FAST tokens to add. Defaults to 2048.

        Example:
            >>> mb.add_special_tokens_for_fast(fast_vocab_size=2048)  # doctest: +SKIP
            Add 2048 special tokens.
            Vocabulary size: 151936 --> 153984
        """
        print(f"Add {fast_vocab_size} special tokens.")
        fast_tokens = [f"<|fast_{i}|>" for i in range(fast_vocab_size)]
        special_tokens_dict = {"additional_special_tokens": fast_tokens}

        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=self.processor.tokenizer,
            model=self.model,
        )

    def _setup_gradient_checkpointing(self):
        """Set up gradient checkpointing and input-require-grads together."""
        if not self.training_args.gradient_checkpointing:
            return

        rank0_print("Enabling Gradient Checkpointing...")
        # 1. Enable input-require-grads (must run AFTER freezing / LoRA wrapping)
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

        # 2. Enable checkpointing
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        elif hasattr(self.model, "config"):
            # Some models control this through config instead.
            self.model.config.use_cache = False  # checkpointing generally needs the cache off

    def build(self):
        """Main entry point: build the model end to end.

        In training mode the build steps run in this strict order:

        1. **Load the model** — ``vla_model_type`` decides which load method runs.
        2. **Replace attention** — swap in the Qwen-VL attention implementation
           that supports variable-length / packed sequences
           (``replace_qwen_vl_attention_class``).
        3. **Add FAST tokens** — extend the vocabulary and embeddings before LoRA
           wrapping.
        4. **Configure parameters** — ``lora_enable`` decides between LoRA wrapping
           and direct freeze/unfreeze.
        5. **Gradient checkpointing** — enabled once the parameters are configured.
        6. **Update config** — sync the Processor's ``model_max_length``.

        Inference mode (``is_training=False``) skips steps 2-6, freezing the model
        and switching it to eval instead.

        Returns:
            None: The results are reached through ``self.model`` and
            ``self.processor``.

        Raises:
            NotImplementedError: When ``vla_model_type`` names an unimplemented type.
            RuntimeError: When model loading fails; the underlying exception is
                passed through.

        Example:
            >>> mb = ModelBuilder(training_args, model_args, data_args, is_training=True)
            >>> mb.build()
            >>> model = mb.model  # ready to train
            >>> processor = mb.processor  # Processor with its pixel range configured
        """
        if getattr(self.model_args, "vla_model_type", None):
            # ``tau0_vla`` is the pre-rename spelling recorded in released
            # checkpoints' config.json; accept it as an alias so those load.
            if self.model_args.vla_model_type in ("tau_vla", "tau0_vla"):
                self.load_tau_vla_model()
            else:
                raise NotImplementedError(f"VLA model type {self.model_args.vla_model_type} not implemented.")
        else:
            self.load_qwen_vlm_model()

        if self.is_training:
            # 2. Replace Attention (Architecture Change) - must precede checkpointing
            #
            # ``replace_qwen_vl_attention_class()`` swaps in the variable-length
            # (packed-sequence) attention path. It is deliberately NOT called:
            # the shipped configs do not pack sequences, and the substitution
            # was never measured against this checkpoint. Enable it only
            # together with a packing data path, and verify numerics first —
            # see ``models/vision_language_models/qwen_vl/qwen_vl_patches.py``.

            # 3. Add Special Tokens - must precede LoRA
            if self.model_args.add_fast_special_tokens:
                self.add_special_tokens_for_fast(fast_vocab_size=self.model_args.fast_vocab_size)

            # 4. Configure Parameters (Freeze / LoRA) - must precede checkpointing
            if self.training_args.lora_enable:
                self.prepare_lora_model()
            else:
                self.set_training_parameters()

            # 5. Setup Gradient Checkpointing - must follow parameter configuration
            self._setup_gradient_checkpointing()

            # 6. Update Configs
            self.update_configs()

        else:
            # Inference mode
            self.model.requires_grad_(False)
            if hasattr(self.model, "eval"):
                self.model.eval()

        self.print_trainable_params()


def _sync_model_config_vocab_size(model: transformers.PreTrainedModel, vocab_size: int):
    """Keep saved config.json consistent after tokenizer/embedding resize."""

    def _set_vocab(obj):
        if obj is None:
            return
        if isinstance(obj, dict):
            obj["vocab_size"] = vocab_size
            text_cfg = obj.get("text_config")
            if isinstance(text_cfg, dict):
                text_cfg["vocab_size"] = vocab_size
            return

        if hasattr(obj, "vocab_size"):
            obj.vocab_size = vocab_size
        text_cfg = getattr(obj, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, "vocab_size"):
            text_cfg.vocab_size = vocab_size

    cfg = getattr(model, "config", None)
    _set_vocab(cfg)
    _set_vocab(getattr(cfg, "qwenvl_config", None))


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Extend the tokenizer's vocabulary and resize the model's embeddings to match.

    New tokens' input and output embeddings are initialized to the mean of the
    existing ones, avoiding the training instability random initialization brings,
    while leaving the existing tokens' embeddings untouched.

    Note: this implementation does not pad the embedding size to a multiple of 64,
    which can slightly reduce matmul efficiency on some hardware.

    Args:
        special_tokens_dict: The special tokens to add, in the form
            ``{"additional_special_tokens": ["<|fast_0|>", ...]}``.
        tokenizer: The tokenizer, modified in place as the vocabulary grows.
        model: The model; ``resize_token_embeddings`` is called on it to extend the
            embedding layer.

    Example:
        >>> smart_tokenizer_and_embedding_resize(  # doctest: +SKIP
        ...     special_tokens_dict={"additional_special_tokens": ["<|fast_0|>"]},
        ...     tokenizer=processor.tokenizer,
        ...     model=model,
        ... )
        Vocabulary size: 151936 --> 151937
    """
    old_length = len(tokenizer)
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    _sync_model_config_vocab_size(model, len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

    rank0_print(f"Vocabulary size: {old_length} --> {len(tokenizer)}")
    # print(
    #     f"New tokens: {[tokenizer.convert_tokens_to_ids(token) for token in special_tokens_dict['additional_special_tokens']]}"
    # )
