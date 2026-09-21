# World model

Generate a goal image from a head-camera observation and a subtask instruction.
The released robotics LoRA is a rank-64 adapter for
[Step1X-Edit-v1p2](https://huggingface.co/stepfun-ai/Step1X-Edit-v1p2).

For best results, we recommend **fine-tuning the released LoRA on your own
start/target image pairs before use**, covering your camera view, objects,
and target subtasks. See the
[fine-tuning guide](#continue-fine-tuning-the-released-lora) below.

## Install and prepare weights

Use a separate Python 3.12 environment from proposal. From the repository root:

```bash
python -m venv .venv-world-model
source .venv-world-model/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
  -c high_level/world_model/constraints.txt -e high_level/world_model
hf download stepfun-ai/Step1X-Edit-v1p2 --local-dir weights/step1x-diffusers
python -m tau0_world_model.convert \
  --input weights/step1x-diffusers --output weights/step1x-base
```

Download the [released robotics LoRA](https://huggingface.co/sii-research/tau-0-vla-world-model):

```bash
hf download sii-research/tau-0-vla-world-model robotics-lora.safetensors \
  --local-dir weights/world_model
```

The converted base directory contains `step1x-edit-v1p2.safetensors`,
`vae.safetensors` and `Qwen2.5-VL-7B-Instruct/`.
Conversion copies the text encoder and processing assets so it can be moved
independently of the downloaded snapshot.

## Single image and batch inference

Generate the example once; skip the first command if you already prepared it.

```bash
python scripts/make_high_level_example.py
python -m tau0_world_model.infer \
  --model weights/step1x-base \
  --lora weights/world_model/robotics-lora.safetensors \
  --image outputs/high_level_example/world_model/input.jpg \
  --instruction "Open the microwave door on the table with the right arm." \
  --output outputs/world-single
```

Batch input is JSONL, with image paths relative to the manifest:

```json
{"id":"example_001","image":"images/start.png","instruction":"Pick up the cup with the right arm."}
```

```bash
python -m tau0_world_model.infer \
  --model weights/step1x-base \
  --lora weights/world_model/robotics-lora.safetensors \
  --input your_inputs.jsonl --output outputs/world-batch
```

Each request writes a PNG and an entry in `results.json`. Default inference
uses 28 denoising steps, CFG 6, size level 512 and seed 42. For multiple GPUs,
launch independent workers with `--shard 0 --num-shards N` through
`--shard N-1 --num-shards N`, selecting a device and distinct output directory
for each. Omit `--lora` for a base-model comparison.

See the [bundled example](../examples/README.md) for the input observation and
the generated image showing the open microwave door.

## Continue fine-tuning the released LoRA

Prepare start/target image pairs from your demonstrations. Each record describes
one completed subtask; keep evaluation episodes separate:

```json
{"id":"pair_001","image":"images/start.png","target":"images/goal.png","instruction":"Pick up the cup with the right arm."}
```

The same prompt prefix is added automatically during training and inference:
`In a robotic workspace, a dual-arm robot: `.

```bash
CUDA_VISIBLE_DEVICES=0 python -m tau0_world_model.finetune \
  --model weights/step1x-base \
  --lora weights/world_model/robotics-lora.safetensors \
  --input outputs/high_level_example/world_model_sft.jsonl \
  --output outputs/world-ft --steps 20
```

Use your pairs in `--input` for real adaptation. The wrapper copies your
images into its output directory, prepares the training metadata, loads the
published adapter, and saves `weights/robotics.safetensors`. It uses batch 1,
BF16, gradient checkpointing and latents/text embeddings cached beside the
copied data in the output directory.

Reload the trained adapter with the same base model:

```bash
python -m tau0_world_model.infer \
  --model weights/step1x-base --lora outputs/world-ft/weights/robotics.safetensors \
  --input outputs/high_level_example/world_model.jsonl \
  --output outputs/world-ft-reloaded
```

Code adapted from [Step1X-Edit](https://github.com/stepfun-ai/Step1X-Edit) and its
Kohya-based LoRA trainer. See [NOTICE.md](NOTICE.md) for source attribution.
