"""Server-backed openloop eval.

Client POSTs canonical ``{prompt, images, state}`` to ``deploy/server.py``'s
``/act``, receives per-component action dict
(``{"arm_joint": [[...], ...], "gripper": [[...]]}``), merges to the flat
ndarray ``run_eval`` expects. No ``--ckpt`` needed — route comes from
``GET /health``, data contract is rebuilt locally via
``get_config(route)`` (requires tau-0-vla's ``configs/`` registered).

Usage::

    python -m deploy.server --model <ckpt> [--route <leaf>] --port 10088

    python deploy/openloop_with_server.py \\
        --server-url http://127.0.0.1:10088 \\
        --out-dir openloop_plots/<run> --max-inferences 30
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
del _sys, _Path

import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
from urllib.request import Request, urlopen

import numpy as np

from deploy._bootstrap import ensure_configs_registered, resolve_deploy_io
from deploy.openloop import run_eval
from deploy.wire import pack_payload
from tau0_vla.data import action_slices
from tau0_vla.data.config import discover_config_modules, get_config, list_configs

logger = logging.getLogger(__name__)


class PolicyClient:
    """HTTP client for ``deploy/server.py``'s ``POST /act``. ``infer()``
    returns the server's raw dict response unchanged. Wire encoding lives in
    ``deploy/wire.py`` (stdlib + numpy) — keep both files together when
    dropping the client into an end-side script."""

    def __init__(self, *, server_url: str, timeout: float = 120.0):
        self._url = server_url.rstrip("/") + "/act"
        self._timeout = timeout

    def infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        req = Request(self._url, data=pack_payload(payload), method="POST",
                      headers={"Content-Type": "application/octet-stream"})
        with urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))


def _merge_action_chunk(action_dict: Dict[str, Any], slices, side_to_canonical) -> np.ndarray:
    """Inverse of the server's split: per-component dict → flat
    ``[chunk, native_dim]`` float32 ndarray, concatenated in ``slices``
    order. Inline: client-side half of this file's wire protocol.

    Unified routes get one more step. The server splits into per-side slots
    (``left_arm``/``right_arm``/…) and then runs
    ``deploy_io.canonicalize_action_dict``, which merges each left/right pair
    into one canonical key (``arm_joint``, ``gripper``, ``eef_pose``) with left
    first. ``slices`` still names the per-side slots, so we undo that merge
    here — taking the leading ``native_dim`` columns for a left slot and the
    trailing ones for its right partner. ``side_to_canonical`` is the adapter's
    own merge table, so client and server agree by construction."""
    parts = []
    for name, _o, d in slices:
        if name in action_dict:
            part = np.asarray(action_dict[name], dtype=np.float32)
        else:
            part = _unmerge_side(action_dict, name, d, side_to_canonical)
        if part.ndim != 2 or part.shape[-1] != d:
            raise ValueError(f"'{name}' shape {part.shape} incompatible with native_dim={d}")
        parts.append(part)
    return np.concatenate(parts, axis=-1)


def _unmerge_side(
    action_dict: Dict[str, Any], name: str, native_dim: int, side_to_canonical
) -> np.ndarray:
    """Recover one side of a canonical left+right key the server merged."""
    for left_key, right_key, canonical in side_to_canonical:
        if name not in (left_key, right_key) or canonical not in action_dict:
            continue
        merged = np.asarray(action_dict[canonical], dtype=np.float32)
        if merged.shape[-1] != 2 * native_dim:
            raise ValueError(
                f"'{canonical}' width {merged.shape[-1]} is not twice the "
                f"{name!r} native_dim={native_dim}; wire format changed."
            )
        return merged[..., :native_dim] if name == left_key else merged[..., native_dim:]
    if not side_to_canonical:
        raise KeyError(
            f"response missing '{name}' slice; got {sorted(action_dict)}. This "
            "adapter declares no UNIFIED_SIDE_TO_CANONICAL, so the client "
            "expects the server to return per-side keys unmerged — if its "
            "canonicalize_action_dict does merge them, it needs the table too."
        )
    raise KeyError(f"response missing '{name}' slice; got {sorted(action_dict)}")


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    ensure_configs_registered()

    with urlopen(args.server_url.rstrip("/") + "/health", timeout=args.http_timeout) as r:
        route = json.loads(r.read().decode("utf-8"))["route"]
    if args.route and args.route != route:
        raise ValueError(
            f"client --route {args.route!r} != server route {route!r}; "
            f"drop --route or restart server with --route {args.route!r}."
        )
    logger.info("server alive (%s), route=%s", args.server_url, route)

    # Reconstruct the bits run_eval reads from policy.data_spec. artifacts_dir=None
    # makes action_slices scan the repo instead of expecting a ckpt-saved
    # field_descriptions.json — server owns the forward pass, ckpt metadata
    # isn't needed on the client.
    # This is the one deploy entry point with no local checkpoint to read, so it
    # cannot recover `config_modules` from a manifest the way the others do — it
    # only learns the route name from the server's /health. If the route's config
    # is registered by module import rather than by a `configs/*/data.py`, name
    # the module here.
    if args.config_module:
        discover_config_modules(module_names=list(args.config_module))
    try:
        config = get_config(route)
    except KeyError:
        raise SystemExit(
            f"the server is serving route {route!r}, but no config by that name is "
            f"registered in this process.\n"
            f"  registered: {sorted(list_configs()) or '(none)'}\n"
            f"\n"
            f"This client rebuilds the eval dataset locally to score against ground "
            f"truth, so it needs the same Robot Config the server trained on. If it "
            f"lives outside configs/*/data.py, pass --config-module "
            f"<dotted.module.path> (repeatable)."
        ) from None
    _, _, _, action_components = config._resolve_component_bundle()
    # This client has no checkpoint, so it stubs the data_spec the eval helpers
    # expect. Unified 40D routes declare no components: ``action_slices`` derives
    # the layout from the registry, and ``unified_native_gt_from_raw`` rebuilds
    # the training UnifiedAssembler to score against. Both read fields off the
    # spec, so a stub missing any of them raises AttributeError on every unified
    # route. Derivation mirrors ``data_spec._build_data_spec``.
    unified_registry_key = getattr(config, "_unified_registry_key", None)
    unified_has_eef = True
    if unified_registry_key is not None:
        eef_provider = config._eef_provider() if hasattr(config, "_eef_provider") else None
        unified_has_eef = (getattr(config, "_eef_action_col", None) is not None) or (eef_provider is not None)
    data_spec = SimpleNamespace(
        finch_config_name=route,
        action_chunk_size=int(config.action_horizon),
        action_components=tuple(action_components),
        artifacts_dir=None,
        unified_registry_key=unified_registry_key,
        unified_has_eef=unified_has_eef,
        config_modules=tuple(args.config_module or ()),
        norm_stats_path=getattr(config, "norm_stats_path", None),
        action_dim=int(config.action_padding_dim or 0),
    )
    client = PolicyClient(server_url=args.server_url, timeout=args.http_timeout)
    slices = action_slices(data_spec)
    # The server merges left/right slots using its adapter's table; undo it with
    # the same one. `robot_cls` names the adapter, so the route answers this —
    # `--adapter` is only for a config registered outside the adapters package.
    deploy_io = resolve_deploy_io(
        SimpleNamespace(robot_cls=type(config)), override=args.adapter
    )
    logger.info("adapter deploy_io: %s", deploy_io.__name__)
    # The table is only needed when the adapter's `canonicalize_action_dict`
    # actually merges left/right pairs. An adapter whose version is the identity
    # (the skeleton's is) declares no table, and then there is nothing to undo.
    side_to_canonical = getattr(deploy_io, "UNIFIED_SIDE_TO_CANONICAL", ())
    policy = SimpleNamespace(
        data_spec=data_spec,
        infer=lambda payload: {
            "actions": _merge_action_chunk(client.infer(payload), slices, side_to_canonical)
        },
    )

    run_eval(
        policy,
        start_idx=args.start_idx, max_inferences=args.max_inferences,
        eval_action_dim=args.eval_action_dim, config_override=args.config,
        out_dir=Path(args.out_dir) if args.out_dir is not None else None,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server-url", required=True, help="e.g. http://127.0.0.1:10088")
    p.add_argument("--out-dir", type=Path, default=None, help="Required unless --no-plot.")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--route", default=None, help="Assert server is serving this route.")
    p.add_argument("--config", default=None, help="Override eval dataset config.")
    p.add_argument("--config-module", action="append", default=[],
                   help="Dotted module path whose import registers the route's config. "
                        "Repeatable. Only needed when the config is not under configs/*/data.py.")
    p.add_argument("--adapter", default=None,
                   help="Dotted adapter package whose deploy_io matches the server's. Normally "
                        "unnecessary — it is derived from the route's Robot Config.")
    p.add_argument("--max-inferences", type=int, default=20)
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument("--eval-action-dim", type=int, default=None)
    p.add_argument("--http-timeout", type=float, default=120.0)
    args = p.parse_args()
    if not args.no_plot and args.out_dir is None:
        p.error("--out-dir is required unless --no-plot")
    return args


if __name__ == "__main__":
    main()
