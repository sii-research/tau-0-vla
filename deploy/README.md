# Deployment

The `deploy` package loads post-trained τ₀-VLA checkpoints for local inference,
HTTP serving, and open-loop evaluation. Public v1 HTTP serving supports
joint-control checkpoints only.

| entry point | purpose |
|---|---|
| `policy.py` | `Tau0VLAPolicy.from_checkpoint(...).infer(payload)` |
| `server.py` | HTTP policy server |
| `openloop.py` | local checkpoint evaluation |
| `openloop_with_server.py` | evaluation through a running server |
| `check_parity.py` | compare local and server inference |

## Checkpoint contract

A post-trained serving checkpoint carries a Data Spec for each route. It records
the robot name, registry key, config modules, cameras, prompt, dimensions, and
normalization contract. Installed adapter/registry code resolves those IDs. The
matching route artifacts live under `finch_data_spec/<route>/`.

Deployment uses that contract to reconstruct the training path:

```text
SDK or canonical payload
    -> adapter instruction, camera, and state mapping
    -> checkpoint Data Spec transforms and 40D assembly
    -> model
    -> unnormalize and undo relative action
    -> semantic action slices
    -> adapter action keys or flat SDK order
```

`--adapter <dotted.package>` is an explicit override. Normally the adapter is
resolved from the checkpoint and must remain registered under the same
`robot_name`.

## Policy server

```bash
python3 -m deploy.server --model /path/to/checkpoint
```

For a multi-route checkpoint:

```bash
python3 -m deploy.server --model /path/to/checkpoint --route YOUR_ROUTE
```

The server exposes:

- `POST /act` — canonical `{prompt, images, state, meta}` input and a
  name-keyed semantic action dictionary;
- `POST /act_lerobot_bytes` — embodiment SDK payload and a flat positional
  action chunk;
- `GET /health` — health check.

Both POST request bodies are `.npz` bundles (`deploy/wire.py`): numpy arrays
travel as named entries, all other leaves in a JSON envelope, always decoded
with `allow_pickle=False`. FastAPI serializes
responses as JSON: `/act` returns a semantic dictionary and the legacy-named
`/act_lerobot_bytes` returns a nested numeric list, not raw bytes.

The wire format never unpickles request bodies — pickled payloads are remote
code execution for anyone who can reach the port. Bodies are also capped
(`MAX_BODY_BYTES`) so a hostile or broken client cannot exhaust memory. The
server binds to `127.0.0.1` by default; expose it beyond localhost only inside
a trusted, isolated network.

For `/act`, send raw task text in `payload["prompt"]`; `encode_payload` applies
the saved prompt template. Sending already-templated text wraps it twice.
Dictionary insertion order in the response is irrelevant. The flat endpoint is
positional and must match the SDK exactly.

## Adapter input mapping

`adapters/<robot>/deploy_io.py` owns live payload conversion:

1. parse the instruction, images, and SDK state into the adapter's observation;
2. load state fields from the checkpoint's `field_descriptions.json`;
3. scatter each SDK state channel into those declared indices;
4. map SDK camera keys to the canonical names stored in the Data Spec;
5. produce `{prompt, images, state, meta}` for the common policy.

`_STATE_CHANNELS` keys must equal checkpoint field-description names. Every
active checkpoint state field needs exactly one accessor, and the constructed
state must be checked for uncovered indices. The template scatter assumes a
field's indices are contiguous; use indexed assignment instead when they are
not.

For a unified route, keep the three mapping layers separate:

1. `repack.raw` selects the dataset vector;
2. registry groups select native indices and scatter them into 40D;
3. checkpoint field descriptions plus `_STATE_CHANNELS` rebuild that same flat
   vector from a live SDK payload.

Every state index referenced by an active registry group must be filled exactly
once.

Every camera name and left/right view must match training. Missing images fail
loudly; also check that the instruction is non-empty.

EEF columns and providers remain training/data-pipeline features. The public v1
server rejects routes with EEF action slices and does not accept or return an
EEF control contract. The public G1 adapter also never derives EEF state from
joints.

## Action restoration and SDK order

For unified routes, model output is first restored to absolute semantic values.
The compact flat order is:

```text
left_eef, right_eef, left_gripper, right_gripper,
waist, chassis_velocity, left_arm, right_arm
```

This is the data-level restoration order. Public v1 serving uses only its
joint-control branches; EEF slices are not a public server output.

Only active slices are included. For `g1_a2d_joint_unified` this becomes:

```text
[left_gripper, right_gripper, left_arm x7, right_arm x7]
```

The A2D SDK instead consumes:

```text
[left_arm x7, right_arm x7, left_gripper, right_gripper]
```

`build_sdk_action_perm` maps compact semantic offsets back to native positions
using registry `action_groups`. It must reject:

- an active semantic slice with no SDK-native group;
- duplicate or uncovered native positions;
- EEF output sent to a joint-only SDK.

If the SDK vector contains uncontrolled or pass-through columns, a permutation
alone is insufficient; implement an explicit fill/preserve policy in that
adapter. Returning `None` is valid only when restored component order already
equals SDK order.

The bundled `g1_agibot_36` and `g1_daas_36` registry layouts contain native
action gaps. For example, `g1_agibot_36` controls native columns `0`, `1`, and
`16:30`, while a 36D SDK vector also contains columns `2:16` and `30:36`. The
policy does not define values for those gaps, so a permutation cannot construct
the complete 36D command safely.

In public v1:

- use canonical `/act` for `g1_agibot_36` and `g1_daas_36`;
- use `/act_lerobot_bytes` only for a layout with a complete, contiguous action
  mapping, such as `g1_a2d_joint_unified`.

Before hardware use, log and inspect semantic slices, permutation, final width,
and a known action vector. Compare one identical observation through the
dataset encoder and SDK payload encoder.

## Open-loop evaluation

```bash
python3 deploy/openloop.py --ckpt /path/to/checkpoint --no-plot
```

Select another route/config or save plots when needed:

```bash
python3 deploy/openloop.py \
    --ckpt /path/to/checkpoint \
    --route YOUR_ROUTE \
    --config YOUR_CONFIG_NAME \
    --out-dir /path/to/output
```

Evaluate through HTTP:

```bash
python3 deploy/openloop_with_server.py \
    --server-url http://127.0.0.1:10088 \
    --config-module your_package.your_config_module \
    --no-plot
```

For an external config module, pass `--config-module` so the same
`@register_config` runs on the evaluation side.
