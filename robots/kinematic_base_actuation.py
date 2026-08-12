"""Rigid backlash (play) driver for a position-controlled catheter base.

The motor command is converted to an ideal, history-dependent transmission
output.  Unlike :mod:`robots.base_actuation_forcefield`, this class does not
turn tracking error into an elastic force.  Outside the backlash gap the motor
and catheter handle are rigidly engaged.

The SOFA implementation uses a kinematic ``Rigid3d`` target that is *outside*
the ODE solver and a ``BilateralLagrangianConstraint`` between that target and
the dynamic Cosserat base.  At the beginning of step ``k`` the target contains
the old pose ``b_k`` and the constant velocity

    v_{k+1} = (b_{k+1} - b_k) / dt.

The target's SOFA ``free_position`` is also set explicitly to ``b_{k+1}``.
This is essential when a held command changes: an unsolved MechanicalObject's
free state otherwise lags its newly written velocity by one animation cycle.
The constraint reaction is propagated through the full coupled compliance,
rather than being applied as a post-solve pose overwrite.

The command is zero-order held over an animation interval, but the *state* is
integrated with SOFA's first-order convention: position is piecewise linear,
velocity is piecewise constant, and acceleration is the discrete velocity
jump.  ``bddot`` is recorded only as a diagnostic; it is not differentiated and
injected as a separate force.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R


def play_update(previous: float, command: float, half_width: float) -> float:
    """One step of the scalar rate-independent play operator.

    The previous output is projected onto the admissible interval centered on
    the new motor command.  Hence the output sticks inside the gap and follows
    the active flank exactly outside it.
    """
    if half_width < 0.0:
        raise ValueError("backlash half-width must be non-negative")
    return float(np.clip(previous, command - half_width, command + half_width))


@dataclass
class PlayedBaseState:
    insertion: float = 0.0
    rotation: float = 0.0  # unwrapped radians
    insertion_rate: float = 0.0
    rotation_rate: float = 0.0


class KinematicPlayBaseActuator:
    """Drive a dynamic SOFA base through a rigid-play kinematic target."""

    mode = "kinematic_play"

    def __init__(
        self,
        *,
        target_mo,
        constraint=None,
        insertion_axis,
        home_position,
        home_orientation,
        insertion_deadzone: float,
        rotation_deadzone: float,
        friction_plateau=(0.0, 0.0),
        friction_speed=(1.0, 1.0),
    ) -> None:
        self.target_mo = target_mo
        self.constraint = constraint
        axis = np.asarray(insertion_axis, dtype=float)
        self.axis = axis / np.linalg.norm(axis)
        self.home_position = np.asarray(home_position, dtype=float)
        self.home_rotation = R.from_quat(
            np.asarray(home_orientation, dtype=float))
        self.insertion_deadzone = float(insertion_deadzone)
        self.rotation_deadzone = float(rotation_deadzone)
        self.friction_plateau = np.asarray(friction_plateau, dtype=float)
        self.friction_speed = np.asarray(friction_speed, dtype=float)
        self.state = PlayedBaseState()
        # Controllers and trajectory generators may store rotation in
        # [-pi, pi).  Keep a separate continuous motor coordinate so crossing
        # the branch cut cannot look like a nearly 2*pi reversal to the play law.
        self._rotation_command = 0.0
        self.last = {
            "command": np.zeros(2),
            "output": np.zeros(2),
            "velocity": np.zeros(2),
            "acceleration": np.zeros(2),
            "branch": np.zeros(2, dtype=np.int8),
            "friction_reaction": np.zeros(2),
        }
        self._write_target(self.state, self.state)

    def _pose(self, insertion: float, rotation: float) -> np.ndarray:
        position = self.home_position + self.axis * insertion
        orientation = (
            R.from_rotvec(self.axis * rotation) * self.home_rotation
        ).as_quat()
        return np.concatenate([position, orientation])

    def _write_target(
        self, current: PlayedBaseState, following: PlayedBaseState,
    ) -> None:
        """Write current pose and the constant velocity for the next step."""
        pose = self._pose(current.insertion, current.rotation)
        free_pose = self._pose(following.insertion, following.rotation)
        velocity = np.concatenate([
            self.axis * following.insertion_rate,
            self.axis * following.rotation_rate,
        ])
        with self.target_mo.position.writeable() as value:
            value[0][:] = pose
        if (hasattr(self.target_mo, "rest_position")
                and len(self.target_mo.rest_position.value) > 0):
            with self.target_mo.rest_position.writeable() as value:
                value[0][:] = pose
        with self.target_mo.velocity.writeable() as value:
            value[0][:] = velocity
        # BilateralLagrangianConstraint is evaluated on free_position/free_velocity.
        # Write both explicitly so a velocity change at AnimateBegin affects this
        # same interval rather than appearing one cycle late.
        self.target_mo.free_position.value = [free_pose.tolist()]
        self.target_mo.free_velocity.value = [velocity.tolist()]

    def reset_from_pose(self, pose) -> None:
        """Synchronize play memory with an externally restored base pose."""
        pose = np.asarray(pose, dtype=float)
        insertion = float(self.axis @ (pose[:3] - self.home_position))
        relative = R.from_quat(pose[3:7]) * self.home_rotation.inv()
        rotation = float(self.axis @ relative.as_rotvec())
        self.state = PlayedBaseState(insertion=insertion, rotation=rotation)
        self._rotation_command = rotation
        self._write_target(self.state, self.state)
        self.last = {
            "command": np.array([insertion, rotation]),
            "output": np.array([insertion, rotation]),
            "velocity": np.zeros(2),
            "acceleration": np.zeros(2),
            "branch": np.zeros(2, dtype=np.int8),
            "friction_reaction": np.zeros(2),
        }

    def set_command(
        self, insertion: float, rotation_degrees: float, dt: float,
    ) -> None:
        """Apply one zero-order-held motor-position sample.

        The public command may be wrapped to ``[-180, 180)`` by a trajectory
        generator.  It is unwrapped against the preceding motor sample before
        entering the scalar play law.  The play state is therefore continuous
        through ``+/-pi``; only its quaternion representation is periodic.
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        rotation_raw = float(np.deg2rad(rotation_degrees))
        rotation_increment = float(np.arctan2(
            np.sin(rotation_raw - self._rotation_command),
            np.cos(rotation_raw - self._rotation_command),
        ))
        self._rotation_command += rotation_increment
        command = np.array([insertion, self._rotation_command], dtype=float)
        old = self.state
        output = np.array([
            play_update(old.insertion, command[0], self.insertion_deadzone),
            play_update(old.rotation, command[1], self.rotation_deadzone),
        ])
        velocity = (output - np.array([old.insertion, old.rotation])) / dt
        old_velocity = np.array([old.insertion_rate, old.rotation_rate])
        following = PlayedBaseState(
            insertion=float(output[0]),
            rotation=float(output[1]),
            insertion_rate=float(velocity[0]),
            rotation_rate=float(velocity[1]),
        )

        # The kinematic target itself is not integrated by an ODE solver.  Its
        # current and free states explicitly define the two ends of this step.
        self._write_target(old, following)
        self.state = following

        error = command - output
        branch = np.zeros(2, dtype=np.int8)
        branch[error > np.array([
            self.insertion_deadzone, self.rotation_deadzone]) - 1e-12] = 1
        branch[error < -np.array([
            self.insertion_deadzone, self.rotation_deadzone]) + 1e-12] = -1
        self.last = {
            "command": command,
            "output": output,
            "velocity": velocity,
            "acceleration": (velocity - old_velocity) / dt,
            "branch": branch,
            # With an ideal position source, transmission friction changes the
            # required motor force/torque but not the prescribed base motion.
            # It is therefore diagnostic actuator load, not a rod force field.
            "friction_reaction": -self.friction_plateau * np.tanh(
                velocity / self.friction_speed),
        }

    def metadata(self) -> dict:
        return {
            "mode": self.mode,
            "law": "b[k+1]=clip(b[k], r[k+1]-delta, r[k+1]+delta)",
            "time_discretization": (
                "ZOH command; piecewise-linear pose; backward-difference "
                "velocity and acceleration"),
            "delta": [self.insertion_deadzone, self.rotation_deadzone],
            "friction_plateau": self.friction_plateau.tolist(),
            "friction_speed": self.friction_speed.tolist(),
            "dof_order": ["insertion_m", "rotation_rad"],
            "insertion_axis": self.axis.tolist(),
            "home_position": self.home_position.tolist(),
            "home_orientation_quat": self.home_rotation.as_quat().tolist(),
        }

    def constraint_reaction(self) -> np.ndarray:
        """Return the six-axis actuator wrench applied to the dynamic base.

        ``BilateralLagrangianConstraint`` contributes six consecutive scalar
        multipliers.  Its base Jacobian is ``-I``, hence the generalized force
        applied to the base is the negative of those multipliers.
        """
        if self.constraint is None:
            return np.zeros(6)
        root = self.target_mo.getContext().getRoot()
        solver = root.getObject("ConstraintSolver")
        if solver is None:
            return np.zeros(6)
        values = np.asarray(solver.constraintForces.value, dtype=float).reshape(-1)
        start = int(self.constraint.constraintIndex.value)
        if start < 0 or start + 6 > len(values):
            return np.zeros(6)
        return -values[start:start + 6].copy()
