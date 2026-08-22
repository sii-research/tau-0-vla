"""Simulator-only HTTP server for LIBERO EEF-control checkpoints.

Request body for ``POST /act_libero`` is a pickled dictionary:

.. code-block:: python

    {
        "observation/image": np.ndarray[H, W, 3],
        "observation/wrist_image": np.ndarray[H, W, 3],
        "observation/state": np.ndarray[8],  # xyz + axis-angle + gripper(2)
        "prompt": str,
    }

The response is a JSON list with shape ``[action_horizon, 7]``:
delta xyz, delta axis-angle, and one gripper command.

This endpoint is deliberately separate from :mod:`deploy.server`. It does not
relax the public server's joint-only hardware safety contract.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
del _sys, _Path

import argparse
import logging
import pickle
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request

from deploy._bootstrap import (
    discover_checkpoint_config_modules,
    ensure_configs_registered,
    ensure_policy_manifest,
    resolve_deploy_io,
)
from deploy.policy import Tau0VLAPolicy
from deploy.warmup import _configure_inference_mode, _run_dummy_input_warmup
from tau0_vla.data import action_slices

logger = logging.getLogger(__name__)


def _validate_libero_checkpoint(policy: Tau0VLAPolicy) -> None:
    spec = policy.data_spec
    if spec.robot_name != "libero":
        raise ValueError(f"deploy.libero_server requires robot_name='libero', got {spec.robot_name!r}")
    if spec.unified_registry_key is not None:
        raise ValueError("LIBERO must use its native component route, not unified 40D")
    if not spec.is_eef:
        raise ValueError("LIBERO checkpoint must declare EEF actions")

    slices = action_slices(spec)
    expected = [("eef_pose", 0, 6), ("gripper", 6, 1)]
    if slices != expected:
        raise ValueError(f"LIBERO action contract must be 6D delta EEF + 1D gripper; checkpoint declares {slices}")


def _validate_action_chunk(actions: Any, *, horizon: int) -> np.ndarray:
    array = np.asarray(actions, dtype=np.float32)
    expected = (int(horizon), 7)
    if array.shape != expected:
        raise ValueError(f"LIBERO policy must return action shape {expected}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("LIBERO policy returned NaN or infinity")
    return array


def build_app(policy: Tau0VLAPolicy) -> FastAPI:
    _validate_libero_checkpoint(policy)
    deploy_io = resolve_deploy_io(policy.data_spec)
    state_fd = deploy_io.load_state_field_descriptions(policy.data_spec.artifacts_dir)
    adapt = deploy_io.build_payload_adapter(
        cam_keys=policy.data_spec.cam_keys,
        state_fd=state_fd,
        state_dim=deploy_io.state_dim_from_field_descriptions(state_fd),
    )

    app = FastAPI(title="tau-0-vla LIBERO server")

    @app.post("/act_libero")
    async def act_libero(request: Request):
        try:
            payload = pickle.loads(await request.body())
            if not isinstance(payload, dict):
                raise TypeError(f"request payload must be a dict, got {type(payload).__name__}")
            canonical = adapt(payload)
        except (KeyError, TypeError, ValueError, pickle.UnpicklingError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        actions = policy.infer(canonical)["actions"]
        validated = _validate_action_chunk(
            actions,
            horizon=policy.data_spec.action_chunk_size,
        )
        return validated.tolist()

    @app.post("/reset_episode")
    async def reset_episode():
        # Tau0VLA currently has no recurrent episode state. Keep this endpoint
        # for compatibility with the LIBERO evaluator and future stateful
        # policies.
        return {"status": "ok"}

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "route": policy.data_spec.finch_config_name,
            "state_contract": "eef_xyz(3)+eef_axis_angle(3)+gripper(2)",
            "action_contract": "delta_xyz(3)+delta_axis_angle(3)+gripper(1)",
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--route", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--infer-mode", default="optim", choices=["optim", "eager"])
    parser.add_argument("--max-prefix-len", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    ensure_configs_registered()
    ensure_policy_manifest(args.model, verbose=False)
    discover_checkpoint_config_modules(args.model)

    policy = Tau0VLAPolicy.from_checkpoint(
        args.model,
        route=args.route,
        device=args.device,
    )
    _validate_libero_checkpoint(policy)
    _configure_inference_mode(policy, args.infer_mode, args.max_prefix_len)
    _run_dummy_input_warmup(policy, warmup_steps=args.warmup_steps)

    # Pickle is unsafe for untrusted callers. The default localhost bind is
    # intentional; expose this only inside a trusted, isolated network.
    import uvicorn

    uvicorn.run(build_app(policy), host=args.host, port=args.port)


if __name__ == "__main__":
    main()


__all__ = ["build_app"]
