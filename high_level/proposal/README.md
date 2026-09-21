# Proposal

Proposal is the high-level planner in τ₀-VLA. It uses the task instruction,
head and wrist camera views, and task memory to propose the robot's next
subtask. The subtask can guide a low-level policy or be passed to the
[world model](../world_model/README.md) to generate a goal image.

For best results, we recommend **fine-tuning the released model on your own
data before use**, covering your robot, camera setup, and target tasks.
See the [fine-tuning guide](#fine-tune-on-your-data) below.

The released model is a full Qwen3.5-9B checkpoint with two output modes:

- `full_qa`: scene reasoning (`think`), updated `memory`, and the next `subtask`.
- `subtask_only`: the next subtask as plain text.

For sequential execution, send fresh camera observations and the returned
memory after each action so the planner can follow the task's progress.

## Install

From the repository root, use Python 3.12 and a CUDA GPU:

```bash
python -m venv .venv-proposal
source .venv-proposal/bin/activate
pip install -r high_level/proposal/requirements.txt
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python scripts/make_high_level_example.py
```

Generate the example once; skip the last command if that directory already exists.

Download the [released weights](https://huggingface.co/sii-research/tau-0-vla-proposal),
including the tokenizer and processor:

```bash
hf download sii-research/tau-0-vla-proposal --local-dir weights/proposal
```

## Inference

```bash
python -m tau0_vla.high_level.proposal.infer \
  --model weights/proposal \
  --input outputs/high_level_example/proposal.jsonl \
  --output outputs/proposals.jsonl
```

For your own observations, create a JSONL file with one record per observation.
Image paths are relative to that file:

```json
{"id":"sample_001","task_type":"full_qa","instruction":"Serve tea","memory":"(empty)","images":{"head":"images/head.jpg","left":"images/left.jpg","right":"images/right.jpg"}}
```

Pass it with `--input your_inputs.jsonl`. Inference needs no reference answers.
Each output contains `prediction`, `subtask`, `think`, `memory_out`,
`format_valid`, and `next_memory`. Pass `next_memory` with the next
observation to continue a task. An invalid parse preserves the input memory;
an explicitly empty memory resets it to `(empty)`.

Open the generated JSONL and inspect the proposed subtask alongside the images
and task instruction. Check that the object and arm match the scene and that
the action is a sensible next step. `format_valid` indicates whether the output
was parsed successfully; judge the action itself by reading the result.

The [bundled example](../examples/README.md) shows an open microwave with the
left gripper near the door. Its saved prediction is:

> Close the microwave door on the table with the left arm.

## HTTP service

```bash
python -m tau0_vla.high_level.proposal.serve --model weights/proposal --port 10089
```

`POST /predict` accepts `instruction` (or a complete `question`), `task_type`,
`memory`, and `images`. Images are base64 PNG/JPEG strings named `head`,
`left`, and `right`. To serve the bundled example:

```python
import base64, json, urllib.request
from pathlib import Path

root = Path("outputs/high_level_example")
payload = json.loads((root / "proposal.jsonl").read_text().splitlines()[0])
payload["images"] = {
    view: base64.b64encode((root / path).read_bytes()).decode()
    for view, path in payload["images"].items()
}
request = urllib.request.Request("http://127.0.0.1:10089/predict",
    data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
print(json.load(urllib.request.urlopen(request)))
```

CLI, service and SFT share camera order **head → left → right**,
480×640 image preprocessing, prompts, and tag parsing. Generation is greedy
with `enable_thinking=False`; custom think/memory/subtask tags remain in the decoded text.

## Fine-tune on your data

Write a JSONL file, one record per observation. Image paths are relative to
the JSONL file. A `full_qa` record is:

```json
{"id":"sample_001","task_type":"full_qa","instruction":"Serve tea","memory":"(empty)","images":{"head":"images/head.jpg","left":"images/left.jpg","right":"images/right.jpg"},"answer":{"think":"The cup is on the table.","memory":"(empty)","subtask":"Pick up the cup with the right arm."}}
```

For `subtask_only`, set `answer` to the subtask string. To reproduce an
existing dataset prompt verbatim, supply `question` instead of `instruction`;
the optional three leading `<image>\n` markers are supported. Keep the top-level
`memory` field as well so an invalid prediction can preserve the previous memory.
IDs must be unique. Missing images or invalid targets stop the run.

Start from the released full checkpoint and train a small LoRA using
assistant-only supervised loss:

```bash
python -m tau0_vla.high_level.proposal.finetune \
  --model weights/proposal \
  --input outputs/high_level_example/proposal_sft.jsonl \
  --output outputs/proposal-ft --steps 20
```

Replace `--input` with your dataset to adapt the policy. The saved directory
contains the adapter, processor/tokenizer and training record. Reload it
alongside the original full weights:

```bash
python -m tau0_vla.high_level.proposal.infer \
  --model weights/proposal --adapter outputs/proposal-ft \
  --input outputs/high_level_example/proposal.jsonl \
  --output outputs/proposals-ft.jsonl
```

To continue this adapter, pass `--adapter outputs/proposal-ft` to
`finetune` with a new output directory. Serving accepts the same adapter option.
