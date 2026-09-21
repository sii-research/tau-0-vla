# High-level components

| Component | Input | Output | Weights |
|---|---|---|---|
| [Proposal](proposal/README.md) | Three camera images, task, memory | Next subtask and updated memory | [Qwen3.5-9B full checkpoint](https://huggingface.co/sii-research/tau-0-vla-proposal) |
| [World model](world_model/README.md) | Head-camera image and subtask | Goal image | [Step1X-Edit-v1p2 robotics LoRA](https://huggingface.co/sii-research/tau-0-vla-world-model) |

Use separate Python environments for the two components.

Prepare the bundled [high-level examples](examples/README.md):

```bash
python scripts/make_high_level_example.py
```

Run the proposal example in its environment:

```bash
python -m tau0_vla.high_level.proposal.infer \
  --model weights/proposal \
  --input outputs/high_level_example/proposal.jsonl \
  --output outputs/proposals.jsonl
```

Switch to the world-model environment to run its example:

```bash
python -m tau0_world_model.infer \
  --model weights/step1x-base \
  --lora weights/world_model/robotics-lora.safetensors \
  --input outputs/high_level_example/world_model.jsonl --output outputs/goals
```

The examples use separate validation observations: closing a microwave for
Proposal, and opening a microwave for the world model. Inspect the predicted
subtask and goal image directly. Fine-tuning examples are in separate
`proposal_sft.jsonl` and `world_model_sft.jsonl` files.

To connect the two components on your own observation, use
`scripts/proposal_to_world_model.py` to pair a selected proposal with its
head-camera image, then pass the resulting JSONL to world-model inference.
