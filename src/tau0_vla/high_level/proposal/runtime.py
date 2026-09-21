"""Qwen3.5 inference shared by the CLI, evaluator and HTTP service."""
import json
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from .data import load_images, sha256
from .format import FORMAT_TAGS, decode, messages, parse_output, update_memory


def load_model(model_path, adapter=None, device="cuda:0", attention="eager", adapter_trainable=False):
    model_path = str(Path(model_path).resolve())
    if adapter:
        config = json.loads((Path(adapter) / "adapter_config.json").read_text())
        record_path = Path(adapter) / "training_record.json"
        record = json.loads(record_path.read_text()) if record_path.is_file() else {}
        expected = record.get("base_weights")
        if expected:
            actual = {p.name: sha256(p) for p in sorted(Path(model_path).glob("*.safetensors"))}
            if actual != expected:
                raise ValueError("Adapter was trained against different base weights")
        elif config.get("base_model_name_or_path") != model_path:
            raise ValueError("Adapter base path differs and has no weight hash binding")
    model, info = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation=attention,
        output_loading_info=True,
    )
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Incomplete checkpoint load: {info}")
    processor = AutoProcessor.from_pretrained(adapter or model_path)
    vocab = processor.tokenizer.get_vocab()
    if not all(tag in vocab for tag in FORMAT_TAGS):
        raise ValueError("Use the released proposal tokenizer containing all six format tags")
    if len(processor.tokenizer) > model.get_input_embeddings().weight.shape[0]:
        raise ValueError("Tokenizer exceeds checkpoint embedding size")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=adapter_trainable)
    for key in ("temperature", "top_p", "top_k"):
        setattr(model.generation_config, key, None)
    return model.to(device), processor


def encode(processor, prompt, generation=True):
    return processor.apply_chat_template(
        prompt, tokenize=True, return_dict=True, return_tensors="pt",
        add_generation_prompt=generation, enable_thinking=False,
    )


class Proposal:
    def __init__(self, model_path, adapter=None, device="cuda:0", max_new_tokens=512, attention="eager"):
        self.model, self.processor = load_model(model_path, adapter, device, attention)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    @torch.inference_mode()
    def predict(self, row, images=None):
        images = load_images(row) if images is None else images
        inputs = encode(self.processor, messages(row, images)).to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        text = decode(self.processor, output[0, inputs.input_ids.shape[1]:])
        parsed = parse_output(text, row.get("task_type", "full_qa"))
        return {"prediction": text, **parsed, "next_memory": update_memory(row.get("memory", ""), parsed)}
