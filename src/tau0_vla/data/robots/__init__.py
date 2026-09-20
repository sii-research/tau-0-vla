from tau0_vla.data.robots.base import FinchResolvedRobotConfig, FrameFilter, RobotConfig
from tau0_vla.data.robots.unified import UNIFIED_DIM, UNIFIED_LAYOUT, UnifiedAssembler

# ─────────────────────────────────────────────────────────────────────────────
# Adapter boundary
# ─────────────────────────────────────────────────────────────────────────────
#
# The robot-name lookup below is the general pipeline's only knowledge of any
# specific embodiment, and it is a *serialization contract* rather than a
# pipeline internal: a Data Spec records a robot name, and this module has to
# turn that name back into a class at inference time. That makes this file part
# of the adapter boundary rather than the pipeline interior, which stays
# embodiment-free — ``tests/test_adapter_boundary.py`` enforces it.
#
# Adding an embodiment means adding its adapter package to ``_ADAPTER_MODULES``
# and its robot names to ``_ROBOT_CLASS_PATHS``, and nowhere else. Each adapter
# owns its own ``*_UNIFIED_CLASSES`` registry, aggregated here into the single
# ``UNIFIED_ROBOT_CLASSES`` lookup.
#
# Adapters are imported LAZILY (inside the accessors below, and via PEP 562
# module ``__getattr__``) rather than at the top of this file, because the
# dependency runs both ways: an adapter's layout module subclasses
# ``RobotConfig`` / ``_UnifiedMixin`` from this package. A top-level adapter
# import works when the pipeline is imported first but raises "partially
# initialized module" when a caller imports the adapter first — which the
# deploy path does. Deferring makes both orders work.

_ADAPTER_MODULES = (
    "tau0_vla.adapters.g1",
    "tau0_vla.adapters.libero",
)

# Robot name (as recorded in a Data Spec) -> ``(module, attr)`` of its
# RobotConfig class. These keys are part of the serialized contract.
#
# Every class an adapter can produce needs an entry, unified routes included:
# ``save_data_spec`` resolves ``robot_config.robot_name`` through this table at
# the end of training, so a missing name aborts the run after the pipeline has
# already built — long after the point where a typo would be cheap to notice.
_ROBOT_CLASS_PATHS = {
    "g1_agibot": ("tau0_vla.adapters.g1", "G1Agibot"),
    "g1_agibot_unified": ("tau0_vla.adapters.g1", "G1AgibotUnified"),
    "g1_daas_unified": ("tau0_vla.adapters.g1", "G1DaasUnified"),
    "g1_a2d_joint_unified": ("tau0_vla.adapters.g1", "G1A2dJointUnified"),
    "libero": ("tau0_vla.adapters.libero", "LiberoRobot"),
}


def _get_robot_class(robot_name: str | None) -> type[RobotConfig]:
    if not robot_name:
        raise KeyError("robot_name is required")
    try:
        module_name, attr = _ROBOT_CLASS_PATHS[robot_name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown robot '{robot_name}'. Available: {sorted(_ROBOT_CLASS_PATHS)}"
        ) from exc
    import importlib

    return getattr(importlib.import_module(module_name), attr)


def _load_unified_robot_classes() -> dict[str, type]:
    """Aggregate every adapter's ``*_UNIFIED_CLASSES`` registry."""
    import importlib

    merged: dict[str, type] = {}
    for module_name in _ADAPTER_MODULES:
        module = importlib.import_module(module_name)
        for attr in dir(module):
            if attr.endswith("_UNIFIED_CLASSES"):
                merged.update(getattr(module, attr))
    return merged


def __getattr__(name: str):
    """Resolve adapter-backed module attributes on first access (PEP 562)."""
    if name == "UNIFIED_ROBOT_CLASSES":
        value = _load_unified_robot_classes()
        globals()[name] = value  # cache; later lookups skip __getattr__
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | {"UNIFIED_ROBOT_CLASSES"})


__all__ = [
    "FinchResolvedRobotConfig",
    "FrameFilter",
    "RobotConfig",
    "UNIFIED_DIM",
    "UNIFIED_LAYOUT",
    "UNIFIED_ROBOT_CLASSES",
    "UnifiedAssembler",
]
