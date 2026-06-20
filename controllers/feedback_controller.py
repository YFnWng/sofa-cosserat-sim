"""FeedbackController — SOFA bridge that orchestrates an observer and
a controller from the ``control/`` repo each simulation timestep.

Handles:
  - Reading SOFA state via SofaReader
  - Running the observer (EKF/UKF/GRU) for state estimation
  - Calling the controller with raw tip xyz measurement
  - Applying commands to SOFA (base rest_position + cable constraint)
  - Recording trajectories to HDF5 (same schema as DataCollectorController)
  - AdapJ motor babbling initialization phase

Design: The controller receives the *raw measurement* (tip xyz from SOFA)
for computing joint commands. The observation model is only used by the
observer for filtering / by MPPI for internal rollouts — never to
reconstruct the controller input.
"""
from __future__ import annotations

import os
import time as _time

import numpy as np
from scipy.spatial.transform import Rotation as R

import Sofa
import Sofa.Core

from data_collection.schema import TrajectoryRecord, write_hdf5


class FeedbackController(Sofa.Core.Controller):
    """SOFA controller that bridges observer + control algorithm.

    Parameters (passed as kwargs)
    ----------
    controller : control.base.BaseController
        Control algorithm instance (PID, MPPI, AdapJ, OpenLoop).
    observer : BaseObserver or None
        State estimator (EKF/UKF/GRU). Updated each step for filtering;
        its latent state is logged but not used as controller input.
    observation_model : callable(state) -> obs or None
        Maps observer state [z, zdot] to observation vector. Used only
        by MPPI for internal rollouts, not for controller input.
    reader : SofaReader
        Reads simulation state each timestep.
    reference : control.reference.ReferenceTrajectory
        Target trajectory.
    base_mechanical_object : SOFA MechanicalObject
        Rigid3d base DOF node (robot.base_mo).
    cable_constraint : SOFA object or None
        Cable constraint for writing cable commands.
    direction, base_position, base_orientation, prefab_rotation_offset :
        Same semantics as DataCollectorController.
    initial_base_pose : (7,) array or None
    initial_strain_coords : (n_sec, 3) array or None
    strain_mechanical_object : SOFA MO or None
    insertion_offset : float
    output_path : str
        HDF5 output path.
    record : bool
        If False, skip all H5 recording.
    warmup_steps : int
        Steps before engaging the controller.
    control_rate : int
        Run controller every N simulation steps. Hold last command between.
    metadata : dict
        Extra metadata for HDF5.
    babble_steps : int
        Motor babbling steps for AdapJ initialization (0 = skip).
    babble_amplitude : (n_joints,) array
        Per-joint babble range for AdapJ.
    duration : float
        Maximum simulation duration in seconds (0 = unlimited).
    """

    _SOFA_CABLE_SCALE = 0.01

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._controller = kwargs.pop("controller")
        self._world_model = kwargs.pop("world_model", None)
        self._observer = self._world_model.observer if self._world_model else None
        self._observation_model = self._world_model.observation if self._world_model else None
        self._sensor_suite = kwargs.pop("sensor_suite", None)
        self._robot_iface = kwargs.pop("robot_interface", None)
        self._reader = kwargs.pop("reader")
        self._reference = kwargs.pop("reference")

        # Control mode: what quantity to control
        # mode: 'position' (3D) or 'pose' (7D quat)
        # sensor: which sensor type ('mri_coils' or 'em_coils')
        # sensor_index: index into that sensor's frame_indices (-1 = last)
        self._control_mode = kwargs.pop("control_mode", "position")
        self._control_sensor = kwargs.pop("control_sensor", "mri_coils")
        self._control_sensor_idx = kwargs.pop("control_sensor_index", -1)

        self._base_mo = kwargs.pop("base_mechanical_object")
        self._cable_constraint = kwargs.pop("cable_constraint", None)
        self._output_path: str = kwargs.pop("output_path", "feedback.h5")
        self._record_enabled: bool = kwargs.pop("record", True)
        self._warmup_steps: int = kwargs.pop("warmup_steps", 50)
        self._control_rate: int = kwargs.pop("control_rate", 1)
        self._metadata: dict = kwargs.pop("metadata", {})
        self._duration: float = kwargs.pop("duration", 0.0)

        # Initial state
        init_bp = kwargs.pop("initial_base_pose", None)
        self._initial_base_pose = (
            np.asarray(init_bp, dtype=float) if init_bp is not None else None)
        init_sc = kwargs.pop("initial_strain_coords", None)
        self._initial_strain_coords = (
            np.asarray(init_sc, dtype=float) if init_sc is not None else None)
        self._strain_mo = kwargs.pop("strain_mechanical_object", None)
        self._insertion_offset = float(kwargs.pop("insertion_offset", 0.0))

        # Base kinematics
        self._base_home_position = np.asarray(
            kwargs.pop("base_position", np.zeros(3)), dtype=float)
        base_orient_quat = np.asarray(
            kwargs.pop("base_orientation", [0.0, 0.0, 0.0, 1.0]), dtype=float)
        self._base_home_orientation = R.from_quat(base_orient_quat)
        prefab_rot_quat = np.asarray(
            kwargs.pop("prefab_rotation_offset", [0.0, 0.0, 0.0, 1.0]),
            dtype=float)
        self._prefab_rot_offset = R.from_quat(prefab_rot_quat)
        direction = np.asarray(
            kwargs.pop("direction", [0.0, 0.0, 1.0]), dtype=float)
        direction = direction / np.linalg.norm(direction)
        self._direction = direction @ self._base_home_orientation.as_matrix().T

        self._cable_data = None
        if self._cable_constraint is not None:
            self._cable_data = self._cable_constraint.findData("value")

        # Sensor config from robot interface (for resolving frame indices)
        self._sensor_cfg = (self._robot_iface.sensor_config
                            if self._robot_iface is not None else {})

        # AdapJ babbling config
        self._babble_steps = int(kwargs.pop("babble_steps", 0))
        babble_rate = kwargs.pop("babble_max_rate", None)
        self._babble_max_rate = (
            np.asarray(babble_rate, dtype=float)
            if babble_rate is not None else None)

        # State
        self._record = TrajectoryRecord()
        self._control_records = {
            "reference": [],
            "compute_time": [],
            "update_time": [],
            "latent_state": [],
        }
        self._step = -1  # incremented to 0 on first call
        self._done = False
        self._joint_cmd = np.zeros(3)
        self._last_compute_time = 0.0
        self._last_update_time = 0.0
        self._in_control_phase = False
        self._last_sofa_gt = None  # cached from previous onAnimateEnd
        self._last_controlled = None  # controlled quantity from last onAnimateEnd
        self._last_readings = None    # sensor readings from last onAnimateEnd
        self._display_writer = None   # set by scene builder (shape + sensors)
        self._shape_source = "observer"  # "observer" or "rollout", set by scene
        self._last_tracking_error = None  # cached for PlotController to read
        self._target_sphere_mo = None     # set by scene builder for path viz
        self._target_sphere_offsets = None
        self._diag_pos_sphere_mo = None   # +rot weighted tip prediction
        self._diag_neg_sphere_mo = None   # -rot weighted tip prediction
        self._diag_sphere_offsets = None

        # Autoregressive latent rollout state (for debugging dynamics model)
        self._rollout_state = None  # initialized after observer reset

        # Babbling state
        self._babbling = self._babble_steps > 0
        self._babble_states = []
        self._babble_actuations = []
        self._babble_count = 0
        self._babble_velocity = np.zeros(3)  # momentum for smooth random walk

    def _encode_cmd_for_model(self, raw_cmd: np.ndarray) -> np.ndarray:
        """Encode raw joint command for the dynamics/observer model.

        Raw: [insertion_m, rotation_deg, cable_N]
        Encoded: [base_enc, tendon] in robot order (cos/sin for angular joints)
        """
        return self._robot_iface.encode_command(raw_cmd)

    # ------------------------------------------------------------------
    # SOFA callbacks
    # ------------------------------------------------------------------

    def onAnimateBeginEvent(self, _event) -> None:
        if self._done:
            return
        if self._base_mo is None or len(self._base_mo.position.value) == 0:
            return

        self._step += 1
        self._in_control_phase = False

        # Step 0: apply initial state, record in onAnimateEnd with t=0
        if self._step == 0:
            self._apply_initial_state()
            return
        
        dt = float(self._base_mo.getContext().dt.value)
        t = (self._step - 1) * dt

        # Check duration
        if self._duration > 0 and t > self._duration:
            self._finish()
            return

        # Warmup: let transients settle
        if self._step <= self._warmup_steps:
            return

        # Use controlled quantity cached from previous onAnimateEnd
        controlled = self._last_controlled
        if controlled is None:
            return  # first control step; onAnimateEnd hasn't run yet

        # Tracking error — updated every sim step for plotting
        ref = self._reference.at(t)
        self._last_tracking_error = controlled - ref

        # --- Rate gate: control and babbling run at same cadence as observer ---
        if self._step % self._control_rate != 0:
            # Between control ticks: hold last command
            self._apply_joint_commands(self._joint_cmd)
            return

        # --- Babbling phase (AdapJ) ---
        if self._babbling:
            self._run_babble_step(controlled, t)
            return

        self._in_control_phase = True

        # Update integral error correction for model-bias compensation
        control_dt = dt * self._control_rate
        if (self._last_tracking_error is not None
                and hasattr(self._controller, 'update_tracking_error')):
            self._controller.update_tracking_error(
                self._last_tracking_error, control_dt)

        # Model-based controllers need the full latent state from the observer;
        # model-free controllers operate on raw sensor measurements.
        if self._controller.needs_latent_state and self._observer is not None:
            ctrl_state = self._observer.state.copy()
        else:
            ctrl_state = controlled

        # Build reference sequence over the horizon for trajectory-aware cost
        horizon = getattr(self._controller, 'horizon', 0)
        if horizon > 0:
            ref_seq = np.stack([self._reference.at(t + h * control_dt)
                                for h in range(horizon)])
        else:
            ref_seq = ref

        t0 = _time.perf_counter()
        raw_cmd = self._controller.compute(ctrl_state, ref_seq, self._joint_cmd, t)
        self._joint_cmd = self._controller.rate_limit(raw_cmd)
        self._last_compute_time = _time.perf_counter() - t0

        self._apply_joint_commands(self._joint_cmd)

    def onAnimateEndEvent(self, _event) -> None:
        if self._done:
            return

        sofa_gt = self._reader.read()
        self._last_sofa_gt = sofa_gt

        dt = float(self._base_mo.getContext().dt.value)
        t = self._step * dt

        # Sensor readings (if sensor suite configured)
        if self._sensor_suite is not None:
            self._last_readings = self._sensor_suite.observe(sofa_gt, t, dt)
        else:
            self._last_readings = None

        # Extract controlled quantity from sensor readings or raw SOFA state
        self._last_controlled = self._extract_controlled(sofa_gt)

        # Observer update — runs at control rate (matching trained dynamics dt).
        # Uses absolute step cadence throughout (warmup and control).
        if (self._observer is not None
                and self._step > 0
                and self._step % self._control_rate == 0):
            u_enc = self._encode_cmd_for_model(self._joint_cmd)
            if self._last_readings is not None:
                self._observer.step(u_enc, self._last_readings)
            else:
                y = self._build_observation(sofa_gt)
                self._observer.step(u_enc, y)

        # AdapJ online update
        self._last_update_time = 0.0
        if hasattr(self._controller, "update") and not self._babbling:
            if self._step > self._warmup_steps:
                t0 = _time.perf_counter()
                self._controller.update(self._last_controlled, self._joint_cmd)
                self._last_update_time = _time.perf_counter() - t0

        # Update display (sensor markers + ghost catheter shape)
        if self._display_writer is not None:
            # Sensor markers
            sensor_arr = None
            if self._last_readings is not None:
                sensor_pts = [np.asarray(p[:3], dtype=float)
                              for p in self._last_readings.positions.values()]
                if sensor_pts:
                    sensor_arr = np.array(sensor_pts, dtype=float)

            # Ghost shape: autoregressive rollout or observer estimate
            shape_positions = None
            if (self._observation_model is not None
                    and self._step > 0
                    and self._step % self._control_rate == 0):
                if self._shape_source == "rollout":
                    self._step_rollout()
                    state = self._rollout_state
                else:
                    state = (self._observer.state
                             if self._observer is not None else None)
                if state is not None:
                    shape_positions = self._decode_shape(state)

            self._display_writer.update_positions(shape_positions, sensor_arr)

        # Update moving target sphere along reference path
        if self._target_sphere_mo is not None:
            ref_pos = self._reference.at(t)
            self._target_sphere_mo.position.value = (
                self._target_sphere_offsets + ref_pos)

        # Update diagnostic spheres: MPPI-weighted +rot/-rot tip predictions
        if self._diag_pos_sphere_mo is not None:
            diag_pos = getattr(self._controller, '_diag_tip_pos_rot', None)
            diag_neg = getattr(self._controller, '_diag_tip_neg_rot', None)
            if diag_pos is not None:
                self._diag_pos_sphere_mo.position.value = (
                    self._diag_sphere_offsets + diag_pos)
            if diag_neg is not None:
                self._diag_neg_sphere_mo.position.value = (
                    self._diag_sphere_offsets + diag_neg)

        # --- Recording ---
        if not self._record_enabled:
            return

        self._record.append(
            t=t,
            frame_poses=sofa_gt.frame_poses,
            strain_coords=sofa_gt.strain_coords,
            joint_commands=self._joint_cmd,
            contact_force_body=sofa_gt.contact_force_body,
            tip_force=np.zeros(3),
            frame_velocity=sofa_gt.frame_velocity,
            strain_velocity=sofa_gt.strain_velocity,
        )

        if self._in_control_phase:
            ref = self._reference.at(t)
            self._control_records["reference"].append(ref.copy())
            self._control_records["compute_time"].append(self._last_compute_time)
            self._control_records["update_time"].append(self._last_update_time)

            if self._observer is not None:
                self._control_records["latent_state"].append(
                    self._observer.state.copy())

    # ------------------------------------------------------------------
    # Babbling (AdapJ initialization)
    # ------------------------------------------------------------------

    def _run_babble_step(self, controlled: np.ndarray, t: float) -> None:
        """Execute one motor babbling step via smooth random walk with momentum.

        Uses a velocity state with random acceleration and damping:
            vel = momentum * vel + (1 - momentum) * uniform(-max_rate, max_rate)
            cmd = cmd + vel
        This produces smooth, sinusoid-like trajectories matching the
        continuous command profiles seen in training data.
        """
        self._babble_count += 1

        # Record controlled quantity (position or pose)
        self._babble_states.append(controlled.copy())

        # Smooth random walk with momentum
        if self._babble_max_rate is not None:
            max_rate = self._babble_max_rate
        else:
            max_rate = (self._controller.joint_upper -
                        self._controller.joint_lower) * 0.1
        momentum = 0.9
        accel = np.random.uniform(-max_rate, max_rate)
        self._babble_velocity = (momentum * self._babble_velocity
                                 + (1 - momentum) * accel)
        if self._controller.max_rate is not None:
            self._babble_velocity = np.clip(
                self._babble_velocity,
                -self._controller.max_rate, self._controller.max_rate)
        cmd = self._joint_cmd + self._babble_velocity
        cmd = np.clip(cmd, self._controller.joint_lower,
                      self._controller.joint_upper)
        self._babble_actuations.append(cmd.copy())
        self._joint_cmd = cmd
        self._apply_joint_commands(cmd)

        # Check if babbling is done
        if self._babble_count >= self._babble_steps:
            self._babbling = False

            states = np.array(self._babble_states)
            actuations = np.array(self._babble_actuations)
            self._controller.initialize(states, actuations)
            # Seed rate limiter so first control command doesn't jump
            self._controller._prev_cmd = self._joint_cmd.copy()
            print(f"[FeedbackController] AdapJ initialized with "
                  f"{len(states)} babbling samples")

    # ------------------------------------------------------------------
    # Estimated shape rendering
    # ------------------------------------------------------------------

    def _decode_shape(self, state: np.ndarray) -> np.ndarray:
        """Decode latent state → FK → node positions (n_nodes, 3)."""
        obs = self._observation_model
        d = obs._d
        z = state[:d]
        q_flat = obs._encoder.decode(z.reshape(1, -1)).flatten()
        q = q_flat.reshape(obs._n_sec, 3)
        base = obs._base_frame_from_state(state)
        T_all, _, _ = self._robot_iface.forward_kinematics_with_jacobian(q, base)
        return np.array([T.translation() for T in T_all], dtype=float)

    def _step_rollout(self) -> None:
        """Advance autoregressive latent rollout by one control step.

        Uses the dynamics model to propagate the rollout state with the
        current joint command. Initial state set by scene builder.
        """
        dynamics = self._world_model.dynamics if self._world_model else None
        if dynamics is None or self._rollout_state is None:
            return
        u_enc = self._encode_cmd_for_model(self._joint_cmd)
        self._rollout_state = dynamics.predict(
            self._rollout_state.astype(np.float32), u_enc.astype(np.float32))

    # ------------------------------------------------------------------
    # State reading
    # ------------------------------------------------------------------

    def _build_observation(self, sofa_gt) -> np.ndarray:
        """Build observation vector for the observer from SOFA ground truth.

        Constructs [xi, eta] from frame poses and velocities,
        matching the format expected by the latent observer.
        """
        from cr_common.utils import SE3_to_R9, world_velocity_to_body
        from cr_common.utils import pose_from_vec7

        tip_pose = sofa_gt.frame_poses[-1]
        T_tip = pose_from_vec7(tip_pose)
        xi = SE3_to_R9(T_tip)

        if sofa_gt.frame_velocity is not None and len(sofa_gt.frame_velocity) > 0:
            tip_vel_world = sofa_gt.frame_velocity[-1]
            eta = world_velocity_to_body(tip_pose, tip_vel_world)
        else:
            eta = np.zeros(6)

        return np.concatenate([xi, eta]).astype(np.float32)

    def _extract_controlled(self, sofa_gt) -> np.ndarray:
        """Extract controlled quantity from sensor readings or raw SOFA state.

        Uses self._control_mode ('position' or 'pose'),
        self._control_sensor ('mri_coils' or 'em_coils'),
        self._control_sensor_idx (index into sensor's frame_indices array).
        """
        readings = self._last_readings

        if readings is not None:
            # Determine which frame index to use from sensor config
            sensor_block = self._sensor_cfg.get(self._control_sensor, {})
            frame_list = sensor_block.get("frame_indices", [])
            if not frame_list:
                raise ValueError(
                    f"Sensor '{self._control_sensor}' has no frame_indices "
                    f"configured but sensor_suite is active")

            idx = self._control_sensor_idx
            if idx < 0:
                idx = len(frame_list) + idx
            frame_idx = frame_list[idx]

            if self._control_mode == "position":
                pos = readings.positions.get(frame_idx)
                if pos is None:
                    raise ValueError(
                        f"No position reading at frame {frame_idx} from "
                        f"sensor '{self._control_sensor}'")
                return pos[:3].copy()
            elif self._control_mode == "pose":
                pose = readings.poses.get(frame_idx)
                if pose is None:
                    raise ValueError(
                        f"No pose reading at frame {frame_idx} from "
                        f"sensor '{self._control_sensor}'")
                return pose.copy()
            else:
                raise ValueError(
                    f"Unknown control mode: '{self._control_mode}'")

        # No sensor suite: fall back to tip position/pose from raw SOFA
        tip_frame = sofa_gt.frame_poses[-1]
        if self._control_mode == "pose":
            return tip_frame.copy()
        return tip_frame[:3].copy()

    # ------------------------------------------------------------------
    # Joint command application (same as DataCollectorController)
    # ------------------------------------------------------------------

    def _apply_initial_state(self) -> None:
        """Write saved base_pose and strain_coords to SOFA MOs."""
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
            if hasattr(self._strain_mo, "rest_position"):
                with self._strain_mo.rest_position.writeable() as rest:
                    n = min(len(sc), len(rest))
                    for i in range(n):
                        rest[i][:] = sc[i]

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

        if (hasattr(self._base_mo, "rest_position")
                and len(self._base_mo.rest_position.value) > 0):
            with self._base_mo.rest_position.writeable() as rest:
                rest[0][0:3] = target_pos
                rest[0][3:7] = target_ori

        if self._cable_data is not None:
            self._cable_data.value = [cable_val * self._SOFA_CABLE_SCALE]

    # ------------------------------------------------------------------
    # Finish and save
    # ------------------------------------------------------------------

    def _finish(self) -> None:
        self._done = True

        if hasattr(self._controller, 'save_actuation_diagnostic'):
            self._controller.save_actuation_diagnostic()

        if not self._record_enabled:
            print("[FeedbackController] Done (recording disabled).")
            return

        n = len(self._record.timestamps)
        print(f"[FeedbackController] Done: {n} timesteps recorded.")
        os.makedirs(os.path.dirname(os.path.abspath(self._output_path)),
                    exist_ok=True)
        meta = {
            "controller": type(self._controller).__name__,
            "n_timesteps": n,
            **self._metadata,
        }

        write_hdf5(self._output_path, self._record, metadata=meta,
                    extra=self._control_records)
        print(f"[FeedbackController] Saved to {self._output_path}")

    @property
    def done(self) -> bool:
        return self._done
