from typing import List, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R

import Sofa
import Sofa.Core


class CatheterKeyboardController(Sofa.Core.Controller):
    """Keyboard control for insertion, axial rotation, and cable pulling.

    Joint indices:
        0 — insertion (translation along direction)
        1 — axial rotation (degrees)
        2 — cable pull (displacement, mm)

    Default key bindings:
        u/j → joint 0 ±
        k/h → joint 1 ±
        i/y → joint 2 ±
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.base_mo = kwargs.pop("base_mechanical_object", None)
        self.base_actuator = kwargs.pop("base_actuator", None)
        
        self.joint_pos = np.zeros(3)
        self.joint_rate = kwargs.pop("joint_rate", np.zeros(3))
        self.joint_upper_limits = kwargs.pop("joint_upper_limits", np.zeros(3))
        self.joint_lower_limits = kwargs.pop("joint_lower_limits", np.zeros(3))
        self.base_home_position = np.asarray(
            kwargs.pop("base_position", np.zeros(3))
        )
        base_home_orientation = np.asarray(
            kwargs.pop("base_orientation", [0.0, 0.0, 0.0, 1.0])
        )
        self._base_home_orientation = R.from_quat(base_home_orientation)
        # prefab_rotation_offset aligns the Cosserat rod axis with the controller's
        # local-Z convention.  It must be composited into every pose write so the
        # base_mo retains the correct orientation — but it is NOT applied to the
        # direction vector, which is already expressed in the semantic local frame.
        prefab_rot = np.asarray(
            kwargs.pop("prefab_rotation_offset", [0.0, 0.0, 0.0, 1.0])
        )
        self._prefab_rot_offset = R.from_quat(prefab_rot)
        local_direction = self._normalize(kwargs.pop("direction", [0.0, 0.0, 1.0]))
        self.direction = local_direction @ self._base_home_orientation.as_matrix().T
        self.cable_constraint = kwargs.pop("cable_constraint", None)
        self.listening = True
        self._cable_data = None
        if self.cable_constraint is not None:
            self._cable_data = self.cable_constraint.findData("value")
            current = self._cable_data.value if self._cable_data is not None else 0.0
            if isinstance(current, (list, tuple)):
                self.joint_pos[2] = float(current[0])
            else:
                self.joint_pos[2] = float(current)

        self.pressed_keys = set()
        self.keys = ['u', 'j', 'k', 'h', 'i', 'y']
        key_joint_idx = [0, 0, 1, 1, 2, 2]
        directions = [1, -1, 1, -1, 1, -1]
        self.key_bindings = {
            k: (j, d)
            for k, j, d in zip(self.keys, key_joint_idx, directions)
        }

    def onAnimateBeginEvent(self, _event) -> None:
        if self.base_mo is None or len(self.base_mo.position.value) == 0:
            return
        dt = float(self.base_mo.getContext().dt.value)
        for key in self.pressed_keys:
            if key in self.keys:
                joint, dir = self.key_bindings[key]
                self.joint_pos[joint] += self.joint_rate[joint] * dir * dt
        np.clip(self.joint_pos, self.joint_lower_limits, self.joint_upper_limits,
                out=self.joint_pos)

        self._apply_base_pose()
        if self._cable_data is not None:
            # SOFA's CableConstraint applies ~100× more effective torque than
            # the physical Rucker model predicts (discrete cable mechanism).
            # Scale down so the user-facing tension is in physical Newtons.
            _SOFA_CABLE_SCALE = 0.01
            self._cable_data.value = [self.joint_pos[2] * _SOFA_CABLE_SCALE]

    def onKeypressedEvent(self, event) -> None:
        key = event.get("key")
        if not isinstance(key, str):
            return
        self.pressed_keys.add(key.lower())

    def onKeyreleasedEvent(self, event) -> None:
        key = event.get("key")
        if not isinstance(key, str):
            return
        self.pressed_keys.discard(key.lower())

    def _apply_base_pose(self) -> None:
        if self.base_actuator is not None:
            dt = float(self.base_mo.getContext().dt.value)
            self.base_actuator.set_command(
                float(self.joint_pos[0]), float(self.joint_pos[1]), dt)
            return

        translation = self.direction * self.joint_pos[0]
        rotation = R.from_rotvec(self.joint_pos[1] * self.direction, degrees=True)
        # Compose: user rotation × semantic home × prefab offset.
        # The prefab offset must always be the innermost rotation so the Cosserat
        # rod axis stays aligned regardless of insertion or axial-rotation joint.
        base_orientation = (rotation * self._base_home_orientation * self._prefab_rot_offset).as_quat()
        target_pos = (self.base_home_position + translation).tolist()
        target_ori = base_orientation.tolist()
        # Legacy spring-deadzone mode: update the compliant force target.
        if hasattr(self.base_mo, "rest_position") and len(self.base_mo.rest_position.value) > 0:
            with self.base_mo.rest_position.writeable() as rest:
                rest[0][0:3] = target_pos
                rest[0][3:7] = target_ori

    @staticmethod
    def _normalize(vec: Sequence[float]) -> List[float]:
        arr = np.asarray(vec, dtype=float)
        norm_sq = float(np.dot(arr, arr))
        if norm_sq <= 0.0:
            raise ValueError("direction vector must be non-zero")
        return arr * (norm_sq ** -0.5)
