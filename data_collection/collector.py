"""DataCollectorController — SOFA Controller that drives the catheter
with an InputGenerator and records per-timestep state via SofaReader.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

import Sofa
import Sofa.Core

from .generators.base import InputGenerator
from .schema import TrajectoryRecord, write_hdf5


class DataCollectorController(Sofa.Core.Controller):
    """Automated data collection controller.

    Replaces ``CatheterKeyboardController`` for scripted trajectory execution.
    Reads per-timestep state via a ``SofaReader`` and buffers it for HDF5
    export.

    Parameters (passed as kwargs)
    ----------
    generator : InputGenerator
        Produces joint commands at each timestep.
    reader : object
        A ``SofaReader`` instance (from ``state_estimation.sofa.bridge.reader``).
    base_mechanical_object : SOFA MechanicalObject
        The Rigid3d base DOF node (``robot.base_mo``).
    cable_constraint : SOFA object or None
        First cable constraint for writing cable commands.
    direction, base_position, base_orientation, prefab_rotation_offset :
        Same semantics as ``CatheterKeyboardController``.
    output_path : str
        Where to write the HDF5 file.
    warmup_steps : int
        Steps to skip before recording (let transients settle).
    metadata : dict or None
        Extra metadata to store in HDF5.
    """

    _SOFA_CABLE_SCALE = 0.01

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._generator: InputGenerator = kwargs.pop("generator")
        self._reader = kwargs.pop("reader")
        self._base_mo = kwargs.pop("base_mechanical_object")
        self._cable_constraint = kwargs.pop("cable_constraint", None)
        self._output_path: str = kwargs.pop("output_path", "trajectory.h5")
        self._warmup_steps: int = kwargs.pop("warmup_steps", 50)
        self._metadata: dict = kwargs.pop("metadata", {})

        init_bp = kwargs.pop("initial_base_pose", None)
        self._initial_base_pose = (
            np.asarray(init_bp, dtype=float) if init_bp is not None else None
        )
        init_sc = kwargs.pop("initial_strain_coords", None)
        self._initial_strain_coords = (
            np.asarray(init_sc, dtype=float) if init_sc is not None else None
        )
        self._strain_mo = kwargs.pop("strain_mechanical_object", None)
        # Insertion offset: back-calculated from initial base_pose so the
        # generator's joint commands are relative to the init configuration.
        self._insertion_offset = float(kwargs.pop("insertion_offset", 0.0))

        self._base_home_position = np.asarray(
            kwargs.pop("base_position", np.zeros(3)), dtype=float,
        )
        base_orient_quat = np.asarray(
            kwargs.pop("base_orientation", [0.0, 0.0, 0.0, 1.0]), dtype=float,
        )
        self._base_home_orientation = R.from_quat(base_orient_quat)
        prefab_rot_quat = np.asarray(
            kwargs.pop("prefab_rotation_offset", [0.0, 0.0, 0.0, 1.0]), dtype=float,
        )
        self._prefab_rot_offset = R.from_quat(prefab_rot_quat)
        direction = np.asarray(kwargs.pop("direction", [0.0, 0.0, 1.0]), dtype=float)
        direction = direction / np.linalg.norm(direction)
        self._direction = direction @ self._base_home_orientation.as_matrix().T

        self._cable_data = None
        if self._cable_constraint is not None:
            self._cable_data = self._cable_constraint.findData("value")

        # Random tip force (smooth band-limited perturbation)
        self._tip_force_field = kwargs.pop("tip_force_field", None)
        tip_force_max = kwargs.pop("tip_force_max", 0.0)
        tip_force_seed = kwargs.pop("tip_force_seed", 42)
        self._tip_force_data = None
        if self._tip_force_field is not None and tip_force_max > 0:
            self._tip_force_data = self._tip_force_field.findData("forces")
            n_harmonics = 5
            freq_range = (0.05, 0.5)
            rng = np.random.RandomState(tip_force_seed)
            self._tip_amplitudes = rng.uniform(0, tip_force_max, (3, n_harmonics))
            self._tip_frequencies = rng.uniform(freq_range[0], freq_range[1], (3, n_harmonics))
            self._tip_phases = np.zeros((3, n_harmonics))  # start at zero force
            # Normalize so peak per axis ≈ tip_force_max
            for ax in range(3):
                total = self._tip_amplitudes[ax].sum()
                if total > 1e-12:
                    self._tip_amplitudes[ax] *= tip_force_max / total

        self._record = TrajectoryRecord()
        self._step_count = 0
        self._done = False
        self._t_start: Optional[float] = None
        self._joint_cmd = np.zeros(3)
        self._t = 0.0
        self._generator_episode: list[int] = []
        self._generator_episode_time: list[float] = []

    def onAnimateBeginEvent(self, _event) -> None:
        if self._done:
            return
        if self._base_mo is None or len(self._base_mo.position.value) == 0:
            return

        # On first step, apply initial state and set rest positions so the
        # solver naturally maintains the configuration.
        if self._step_count == 0:
            self._apply_initial_state()
            return  # let solver run one step with correct state before generator

        t = self._t + self._t_start # t at the end of step

        self._joint_cmd = self._generator.step(t)
        self._joint_cmd = np.clip(self._joint_cmd, self._generator.joint_lower,
                                  self._generator.joint_upper)
        self._apply_joint_commands(self._joint_cmd)
        self._update_tip_force(t)

    def onAnimateEndEvent(self, _event) -> None:
        if self._step_count == 0:
            self._t_start = float(self._base_mo.getContext().time.value)
            self._t = 0.0
        else:
            self._t = float(self._base_mo.getContext().time.value) - self._t_start

        sofa_gt = self._reader.read()

        tip_force = np.zeros(3)
        if self._tip_force_data is not None:
            forces = np.array(self._tip_force_data.value).flat
            tip_force = np.array(forces[:3])

        self._record.append(
            t=self._t,
            frame_poses=sofa_gt.frame_poses,
            strain_coords=sofa_gt.strain_coords,
            joint_commands=self._joint_cmd,
            contact_force_body=sofa_gt.contact_force_body,
            tip_force=tip_force,
            frame_velocity=sofa_gt.frame_velocity,
            strain_velocity=sofa_gt.strain_velocity,
        )
        episode_index = getattr(self._generator, "episode_index", None)
        episode_time = getattr(self._generator, "episode_time", None)
        if callable(episode_index):
            self._generator_episode.append(int(episode_index(self._t)))
            self._generator_episode_time.append(
                float(episode_time(self._t)) if callable(episode_time) else 0.0)

        self._step_count += 1

        if self._generator.is_done(self._t):
            self._finish()
            return

    def _apply_initial_state(self) -> None:
        """Write saved base_pose and strain_coords to SOFA MOs.

        Sets both ``position`` and ``rest_position`` on the strain MO so
        the beam stiffness forces pull toward the init shape (not straight).
        Called once on the first step after ``Sofa.Simulation.init()``.
        """
        if self._initial_base_pose is not None:
            bp = self._initial_base_pose
            with self._base_mo.position.writeable() as pos:
                pos[0][:] = bp
            if (hasattr(self._base_mo, "rest_position")
                    and len(self._base_mo.rest_position.value) > 0):
                with self._base_mo.rest_position.writeable() as rest:
                    rest[0][:] = bp

        if self._initial_strain_coords is not None and self._strain_mo is not None:
            sc = self._initial_strain_coords
            with self._strain_mo.position.writeable() as strain_pos:
                n = min(len(sc), len(strain_pos))
                for i in range(n):
                    strain_pos[i][:] = sc[i]
            # Set rest_position so elastic forces maintain the bent shape
            if hasattr(self._strain_mo, "rest_position"):
                with self._strain_mo.rest_position.writeable() as rest:
                    n = min(len(sc), len(rest))
                    for i in range(n):
                        rest[i][:] = sc[i]

    def _update_tip_force(self, t: float) -> None:
        """Set the tip ConstantForceField to a smooth random force at time *t*."""
        if self._tip_force_data is None:
            return
        force = np.zeros(3)
        for ax in range(3):
            force[ax] = np.sum(
                self._tip_amplitudes[ax]
                * np.sin(2 * np.pi * self._tip_frequencies[ax] * t
                         + self._tip_phases[ax])
            )
        self._tip_force_data.value = [[
            float(force[0]), float(force[1]), float(force[2]),
            0.0, 0.0, 0.0,
        ]]

    def _apply_joint_commands(self, joint_cmd: np.ndarray) -> None:
        """Apply [insertion, rotation, cable] to SOFA objects."""
        insertion = joint_cmd[0] + self._insertion_offset
        rotation_deg = joint_cmd[1]
        cable_val = joint_cmd[2]

        translation = self._direction * insertion
        rotation = R.from_rotvec(rotation_deg * self._direction, degrees=True)
        base_orientation = (
            rotation * self._base_home_orientation * self._prefab_rot_offset
        ).as_quat()
        target_pos = (self._base_home_position + translation).tolist()
        target_ori = base_orientation.tolist()

        # Only update rest_position (spring target), not position directly.
        # The RestShapeSpringsForceField pulls the base toward the target,
        # letting base motion propagate through rod dynamics instead of
        # being applied as a rigid transform.
        if (hasattr(self._base_mo, "rest_position")
                and len(self._base_mo.rest_position.value) > 0):
            with self._base_mo.rest_position.writeable() as rest:
                rest[0][0:3] = target_pos
                rest[0][3:7] = target_ori

        if self._cable_data is not None:
            self._cable_data.value = [cable_val * self._SOFA_CABLE_SCALE]

    def _finish(self) -> None:
        self._done = True
        # Zero tip force
        if self._tip_force_data is not None:
            self._tip_force_data.value = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        n = len(self._record.timestamps)
        print(f"[DataCollector] Trajectory complete: {n} timesteps recorded.")
        os.makedirs(os.path.dirname(os.path.abspath(self._output_path)), exist_ok=True)
        meta = {
            "generator": self._generator.name,
            "n_timesteps": n,
            **self._metadata,
        }
        episode_names = getattr(self._generator, "episode_names", None)
        if episode_names is not None:
            meta["generator_episode_names"] = list(episode_names)
            meta["generator_episode_durations"] = list(getattr(
                self._generator, "episode_durations", []))
        extra = None
        if self._generator_episode:
            extra = {
                "generator_episode": self._generator_episode,
                "generator_episode_time": self._generator_episode_time,
            }
        write_hdf5(self._output_path, self._record, metadata=meta, extra=extra)
        print(f"[DataCollector] Saved to {self._output_path}")
