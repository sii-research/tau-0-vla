"""LIBERO single-arm simulation adapter.

The observation contract is intentionally explicit:

* state ``[0:6]``: end-effector pose ``xyz + axis-angle``
* state ``[6:8]``: two gripper joint positions
* action ``[0:6]``: delta end-effector pose ``xyz + axis-angle``
* action ``[6]``: scalar gripper command
"""

from tau0_vla.adapters.libero.layout import LiberoObservation, LiberoRobot

__all__ = ["LiberoObservation", "LiberoRobot"]
