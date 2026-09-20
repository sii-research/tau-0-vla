# LIBERO post-training and evaluation

`train.yaml` fine-tunes the pretrained Tau0VLA checkpoint on LIBERO while
preserving the checkpoint's unified 40D state/action interface. It selects the
`libero_eef_robot_prompt_ft` data route defined in `data.py`.

## Native LIBERO contract

```text
state  = [eef_xyz(3), eef_axis_angle(3), gripper_qpos(2)]  # 8D
action = [delta_xyz(3), delta_axis_angle(3), gripper(1)]   # 7D
action_horizon = 10
```

The action values already represent EEF deltas, so the action route uses
`abs2relative=False` and does not subtract the current state a second time.

## Checkpoint-aligned 40D representation

The native 3D axis-angle rotation is converted to a 6D rotation
representation. Consequently, each EEF pose changes from
`xyz(3) + axis-angle(3) = 6D` to `xyz(3) + rot6d(6) = 9D`.

Both state and action then use the following model-facing layout:

| 40D slots | Meaning | Active |
| --- | --- | --- |
| `0:3` | EEF xyz or delta xyz | yes |
| `3:9` | EEF rot6d or delta rot6d | yes |
| `9:18` | reserved/padding | no, always zero |
| `18` | left gripper | yes |
| `19:40` | reserved/padding | no, always zero |

The conversion and padding pipeline is:

1. `AxisAngle2Rot6D` converts the 6D EEF pose to 9D.
2. `PadToDim(9, 18)` right-pads the EEF component so that the following
   gripper component is placed at slot `18`.
3. `state_padding_dim=40` and `action_padding_dim=40` pad the assembled
   19D vectors to the checkpoint's 40D input/output dimensions.

For state input, LIBERO's two opposing finger joints are reduced to one
opening value:

```text
gripper = 0.5 * (qpos[0] - qpos[1])
```

The active state/action indices are therefore `0:9` and `18`. In particular,
`use_action_mask_loss: true` excludes all inactive action dimensions from the
flow-matching loss, while `vla_inactive_input_zero: true` keeps those inactive
action dimensions at zero in the flow input. `zero_state_emb: false` keeps
state conditioning enabled. During deployment, predicted rot6d EEF rotations
are converted back to native 3D axis-angle commands before they are returned
to the LIBERO simulator.

## Training

Set the dataset path (or a text manifest containing one dataset path per line)
and launch training:

```bash
export TAU0_LIBERO_DATA=/path/to/libero
bash scripts/train.sh configs/libero/train.yaml \
  --model_name_or_path sii-research/tau-0-vla
```

The selected robot-aware prompt is:

```text
You are controlling a robot.
Robot type: Panda
Control mode: end-effector
Whole-body control: disabled
Task: {instruction}
```

`LiberoRobot` interprets the eight state values as six EEF values followed by
two gripper values, including when loading exports with older field metadata.

## Checkpoint

The [τ₀-VLA LIBERO checkpoint](https://huggingface.co/sii-research/tau-0-vla-libero)
is post-trained for **60,000 steps** from
[`sii-research/tau-0-vla`](https://huggingface.co/sii-research/tau-0-vla).

```bash
hf download sii-research/tau-0-vla-libero \
  --local-dir checkpoints/tau-0-vla-libero
```

Its complete inference export has the following structure:

```text
tau-0-vla-libero/
├── model.safetensors
├── config.json
├── run_spec.json
├── policy_manifest.json
├── processor_config.json
├── tokenizer.json
├── tokenizer_config.json
├── chat_template.jinja
└── finch_data_spec/libero-eef-robot-prompt-ft/
    ├── spec.json
    ├── components.json
    ├── field_descriptions.json
    └── norm_stats.json
```

For this export, `model.safetensors` has SHA-256
`e03870720cbddbd0f3be44ee929a5d23efb9bf9532f0ac1c1bf676224aacc8ec`.
The export includes the weights, normalization statistics, transforms, prompt,
and camera labels needed for inference. When exporting another checkpoint,
include the same artifacts and resolve symlinks to make the directory portable.

## Separate model and simulation environments

Use the repository's [installation instructions](../../README.md#installation)
for the model server and install its serving extras there. An example server
environment uses Python 3.12.3, PyTorch 2.7.1+cu128, Transformers 5.5.4,
NumPy 2.3.5, and an RTX 4090. Run commands from the repository root:

```bash
# In the model environment, after scripts/setup.sh:
pip install -e '.[serve]'
python -m deploy.libero_server \
  --model checkpoints/tau-0-vla-libero \
  --host 127.0.0.1 --port 8000 \
  --seed 7 --infer-mode eager --warmup-steps 1
```

The commands here use `eager`. The server also supports `optim`, its
default optimized inference mode.

The client uses a separate Python 3.10 environment with LIBERO, robosuite
1.4.0, MuJoCo 3.2.3, and NumPy 1.24.4. It does not import the model or LeRobot.
A minimal simulation setup is:

```bash
conda create -n tau0-libero-sim python=3.10 -y
conda activate tau0-libero-sim
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install numpy==1.24.4 robosuite==1.4.0 mujoco==3.2.3 bddl==1.0.1 \
  gym==0.25.2 opencv-python==4.6.0.66 easydict==1.9 cloudpickle==2.1.0 \
  requests==2.34.2 tyro==1.0.16 imageio==2.37.4 imageio-ffmpeg==0.6.0 \
  pillow==12.3.0 pyyaml==6.0.3 tqdm==4.70.0

git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /path/to/LIBERO
git -C /path/to/LIBERO checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
pip install --no-deps -e /path/to/LIBERO
export PYTHONPATH=/path/to/LIBERO:${PYTHONPATH:-}
export MUJOCO_GL=egl
# The official LIBERO initial-state files contain NumPy arrays, not just tensors.
# Limit this setting to the simulator with trusted official benchmark assets.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
python -c 'from libero.libero import benchmark; print(benchmark.get_benchmark_dict().keys())'
```

On first import, LIBERO asks where to store its paths; accept the defaults or
configure `LIBERO_CONFIG_PATH/config.yaml` for your checkout. Evaluation needs
the repository's BDDL files, assets, and initial-state files; demonstration
training datasets are not needed for these rollouts. Follow the
[upstream setup guide](https://github.com/Lifelong-Robot-Learning/LIBERO#installtion)
for benchmark assets. On headless Linux, install the system OpenGL/EGL loader
libraries (Ubuntu: `libgl1 libglx0 libglvnd0 libegl1 libopengl0`) and expose the
NVIDIA EGL driver. `libGL.so.1` or EGL import errors indicate a rendering
setup problem before any model evaluation.

## Evaluation commands and protocol

In the simulation environment, return to the Tau0VLA repository root and
evaluate all four suites with 50 rollouts per task:

```bash
for suite in libero_spatial libero_object libero_goal libero_10; do
  python -m deploy.libero.main \
    --args.host 127.0.0.1 --args.port 8000 \
    --args.task-suite-name "$suite" \
    --args.seed 7 --args.replan-steps 8 \
    --args.episode-start 0 --args.num-trials-per-task 50 \
    --args.video-out-path "outputs/libero_eval/$suite" || exit 1
done
```

The four suites contain 40 tasks, totaling 2,000 episodes. The CLI uses the
`--args.` prefix shown above; `python -m deploy.libero.main --help` lists all
options. `--args.episode-start` selects the first initial-state index.

| Display name | CLI suite | Tasks | Maximum action steps |
| --- | --- | ---: | ---: |
| Spatial | `libero_spatial` | 10 | 220 |
| Object | `libero_object` | 10 | 280 |
| Goal | `libero_goal` | 10 | 300 |
| Long | `libero_10` | 10 | 520 |

`libero_90` is also supported (400 steps), but is excluded from the four-suite
average. All runs use 10 settling steps, 256×256 simulator renders, a 180°
rotation for both cameras, and PIL bilinear resizing to 224×224. The checkpoint
predicts 10 actions; the client executes 8 before replanning. Native gripper
commands are passed through without an extra sign flip. The simulator and
policy RNGs are reset to seed 7 at each episode, independently of prior
episodes. Use one client per server process;
concurrent clients would share its policy RNG.

Each output directory contains:

- `run.json`: arguments, client code revision/hash, and server/weight metadata;
- `episodes.jsonl`: task, initial-state index, success, exception, step count,
  and video filename for every attempted episode;
- `results.json`: running per-task counts and exception totals;
- `results.txt`: final aggregate success rate after normal completion;
- one MP4 per episode with captured frames, including unsuccessful rollouts.

Existing episode records are protected from overwrite. Setup/RPC/action/video
errors stop the run with a nonzero exit code and leave the completed episode
records available for inspection.

## Evaluation results

Success rates (%) over 50 rollouts per task:

| Spatial | Goal | Object | Long (`libero_10`) | Average |
| ---: | ---: | ---: | ---: | ---: |
| 97.40 | 98.20 | 98.80 | 95.00 | 97.35 |
