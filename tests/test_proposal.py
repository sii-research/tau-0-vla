import base64
import io
import json

import pytest
from PIL import Image
from fastapi.testclient import TestClient

from tau0_vla.high_level.proposal.format import (
    answer_text, decode, messages, ordered_images, parse_output, prompt_text, update_memory,
)
from tau0_vla.high_level.proposal.evaluate import metrics, read_results
from tau0_vla.high_level.proposal.serve import create_app


@pytest.mark.parametrize("closing", ["/", ""])
def test_legacy_and_current_memory_contract(closing):
    text = "".join(f"<|{name}|>{value}<|{closing}{name}|>" for name, value in (
        ("think", "The cup is on the table."), ("memory", ""), ("subtask", "Pick up the cup.")))
    parsed = parse_output(text)
    assert parsed["format_valid"]
    assert update_memory("previous", parsed) == "(empty)"
    assert update_memory("previous", parse_output(text[:-5])) == "previous"


def test_missing_think_opener_and_subtask_only():
    parsed = parse_output("Ready<|/think|><|memory|>cup placed<|/memory|><|subtask|>(done)<|/subtask|>")
    assert parsed["think"] == "Ready"
    assert update_memory("old", parsed) == "cup placed"
    assert parse_output("Pick up the cup.", "subtask_only")["format_valid"]
    assert not parse_output("<|subtask|>truncated", "subtask_only")["format_valid"]


def test_camera_identity_and_prompt_equivalence():
    cameras = {"right": 3, "head": 1, "left": 2}
    assert ordered_images(cameras) == [1, 2, 3]
    with pytest.raises(ValueError):
        ordered_images({**cameras, "hand_left": 4})
    row = {"instruction": "Serve tea", "memory": "", "task_type": "full_qa"}
    explicit = messages(row, [1, 2, 3])
    raw = messages({"question": "<image>\n" * 3 + prompt_text("Serve tea")}, [1, 2, 3])
    assert explicit == raw
    assert "(empty)" in explicit[0]["content"][-1]["text"]
    assert "Current memory" not in prompt_text("Serve tea", mode="subtask_only")


def test_http_and_cli_use_same_images_and_outputs():
    class Policy:
        def predict(self, row, images):
            assert [image.getpixel((0, 0))[0] for image in images] == [10, 20, 30]
            assert [image.size for image in images] == [(640, 480)] * 3
            return {"prediction": "invalid", **parse_output("invalid"), "next_memory": row["memory"]}
    def picture(value):
        buffer = io.BytesIO()
        Image.new("RGB", (2, 2), (value, 0, 0)).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    client = TestClient(create_app(Policy()))
    payload = {"instruction": "Serve tea", "memory": "keep",
               "images": {"right": picture(30), "head": picture(10), "left": picture(20)}}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["memory_out"] is None
    assert response.json()["next_memory"] == "keep"
    payload["images"]["left"] = "invalid-base64"
    assert client.post("/predict", json=payload).status_code == 422


def test_decode_never_strips_business_tokens():
    class Processor:
        tokenizer = type("Tokenizer", (), {"eos_token": "<|im_end|>", "bos_token": None, "pad_token": None})()
        def decode(self, ids, **kwargs):
            assert kwargs["skip_special_tokens"] is False
            return "<|subtask|>pick<|/subtask|><|im_end|>"
    assert decode(Processor(), []) == "<|subtask|>pick<|/subtask|>"


def test_http_preserves_explicit_dataset_question():
    question = "Current phase: Close the microwave door.\nCurrent memory: ready"
    class Policy:
        def predict(self, row, images):
            assert messages(row, images)[0]["content"][-1]["text"] == question
            return {"subtask": "Close the door.", "format_valid": True}
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    payload = {"question": "<image>\n" * 3 + question, "memory": "ready",
               "images": {name: encoded for name in ("head", "left", "right")}}
    client = TestClient(create_app(Policy()))
    assert client.post("/predict", json=payload).status_code == 200
    payload["question"] = "<image>\n" + question
    assert client.post("/predict", json=payload).status_code == 422
    del payload["question"]
    assert client.post("/predict", json=payload).status_code == 422


def test_resume_rejects_duplicates_and_wrong_ground_truth(tmp_path):
    row = {"id": "a", "task_type": "subtask_only", "answer": "pick"}
    result = {"id": "a", "ground_truth": answer_text(row)}
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps(result) + "\n")
    assert set(read_results(path, [row])) == {"a"}
    path.write_text((json.dumps(result) + "\n") * 2)
    with pytest.raises(ValueError, match="duplicate"):
        read_results(path, [row])
    path.write_text(json.dumps({**result, "ground_truth": "different"}) + "\n")
    with pytest.raises(ValueError, match="Ground truth"):
        read_results(path, [row])


def test_metrics_use_field_denominators():
    report = metrics([{"format_valid": True, "subtask_exact_match": True, "memory_exact_match": False},
                      {"format_valid": True, "subtask_exact_match": False, "memory_exact_match": None}])
    assert report["subtask_exact_match"]["rate"] == 0.5
    assert report["memory_exact_match"]["denominator"] == 1
    assert report["think_exact_match"]["rate"] is None


def test_adapter_rejects_changed_base_weights_before_loading(tmp_path):
    from tau0_vla.high_level.proposal.runtime import load_model
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    base.mkdir()
    adapter.mkdir()
    (base / "model.safetensors").write_bytes(b"different weights")
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "training_record.json").write_text(json.dumps({
        "base_weights": {"model.safetensors": "expected original checksum"},
    }))
    with pytest.raises(ValueError, match="different base weights"):
        load_model(base, adapter, device="cpu")
