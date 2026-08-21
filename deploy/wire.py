"""Safe wire serialization for the inference server.

``/act`` and ``/act_lerobot_bytes`` transport ``{prompt, images, state, meta}``
payloads. Deserializing a request body with ``pickle.loads`` is unauthenticated
remote code execution for anyone who can reach the port, so payloads travel as
``.npz`` loaded with ``allow_pickle=False``: numpy arrays become named entries
and every other leaf rides in a JSON envelope.
"""
from __future__ import annotations

import json
from io import BytesIO
from typing import Any, Dict

import numpy as np

# Refuse unbounded request bodies before decoding. Batched camera frames are
# megabytes, not hundreds of megabytes; anything larger is a broken or hostile
# client and must not be allocated.
MAX_BODY_BYTES = 256 * 1024 * 1024

_JSON_ENTRY = "__json__"
_ARR_PREFIX = "__arr"
_ND_MARKER = "__nd__"


def pack_payload(payload: Dict[str, Any]) -> bytes:
    """Encode an inference payload as ``.npz`` bytes (no pickle)."""
    arrays: list[np.ndarray] = []

    def encode(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            arrays.append(np.asarray(value))
            return {_ND_MARKER: len(arrays) - 1}
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): encode(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(v) for v in value]
        return value

    tree = encode(payload)
    buf = BytesIO()
    np.savez(
        buf,
        **{_JSON_ENTRY: json.dumps(tree)},
        **{f"{_ARR_PREFIX}{i}": a for i, a in enumerate(arrays)},
    )
    return buf.getvalue()


def unpack_payload(data: bytes) -> Dict[str, Any]:
    """Decode a ``pack_payload`` body. Never unpickles anything."""
    if len(data) > MAX_BODY_BYTES:
        raise ValueError(
            f"payload body {len(data)} bytes exceeds {MAX_BODY_BYTES} limit"
        )
    with np.load(BytesIO(data), allow_pickle=False) as npz:
        tree = json.loads(str(npz[_JSON_ENTRY]))
        arrays = [
            npz[f"{_ARR_PREFIX}{i}"]
            for i in range(sum(1 for k in npz.files if k.startswith(_ARR_PREFIX)))
        ]

    def decode(node: Any) -> Any:
        if isinstance(node, dict):
            if set(node) == {_ND_MARKER}:
                return arrays[node[_ND_MARKER]]
            return {k: decode(v) for k, v in node.items()}
        if isinstance(node, list):
            return [decode(v) for v in node]
        return node

    return decode(tree)
