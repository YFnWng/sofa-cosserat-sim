"""Data collection scene — drives the catheter with scripted trajectories
and records per-timestep state to HDF5.

Supports multi-scene batch collection: the YAML ``scenes`` list defines
multiple environment configurations.  Each scene is run as a separate
SOFA session with its own trajectory and output file.

Three execution modes:

1. **GUI** (runSofa) — runs one scene with visual debugging + plotter:
       runSofa -g qt simulation/scenes/collect_data.py

2. **Headless collection** — runs ALL scenes sequentially:
       python simulation/scenes/collect_data.py

3. **Manual init** — opens each scene in GUI for manual positioning,
   then saves the rod state back to the YAML:
       python simulation/scenes/collect_data.py --init

Scene YAML format
-----------------
Each scene is a dict with ``objects`` and optional ``initial_config``::

    scenes:
      - objects:
          - {type: RingModel, position: [0, 0, 0.01], scale: 0.01}
        initial_config:
          base_pose: [x, y, z, qx, qy, qz, qw]   # Rigid3d
          strain_coords: [[ky, kz, tau], ...]       # per-section

The initial_config stores the rod's physical state (pose + strain) at
zero actuation.  All training trajectories start from zero control
inputs (zero rotation, zero cable tension).

For backward compatibility, a scene can also be a plain list of objects
(legacy format), in which case ``initial_config`` is absent and the
trajectory starts from the default rod configuration.

Environment variables
---------------------
COLLECT_GENERATOR : str
    Generator type: ``sweep`` (default), ``sinusoidal``, ``proximal_id``, or
    ``ssm_constraint``, or ``rich_proximal``.
COLLECT_SEED : int
    Reproducible generator seed (default: 0).
COLLECT_OUTPUT : str
    Output HDF5 path (default: auto-generated per scene).
COLLECT_WARMUP : int
    Warmup steps before recording (default: 0).
COLLECT_DURATION : float
    Maximum simulation duration in seconds per scene (default: unlimited).
COLLECT_SCENE_IDX : int
    Scene index to run in GUI mode (default: 0).
COLLECT_SCENES : str
    Path to a YAML file whose ``scenes`` list overrides the one in the
    main config.  Useful with ``generate_environments.py``.
COLLECT_DEBUG : str
    When "1", record diagnostic terms in matrix recorder (solver b/x,
    forces, Jacobians, b-reconstruction analysis). Default "0".
COLLECT_VERBOSE : str
    When "1", print verbose diagnostics during recording. Default "0".
COLLECT_COMPONENTS : str
    When "1", record component-resolved proximal/material/base forces and their
    full-coordinate implicit RHS terms. This implies debug recording and should
    be used with ``COLLECT_MATRICES=1``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import yaml

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR = os.path.normpath(os.path.join(_THIS_DIR, ".."))
_WORKSPACE = os.path.normpath(os.path.join(_SIM_DIR, ".."))

if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)
if _SIM_DIR in sys.path:
    sys.path.remove(_SIM_DIR)
sys.path.insert(0, _SIM_DIR)

from utils.sofa_env import ensure_sofa_paths
ensure_sofa_paths()

import Sofa
import Sofa.Core
import Sofa.Simulation
from Sofa import SofaLinearSolver  # noqa: F401 — registers .A()/.b()/.x() on solvers

from utils.scene import add_required_plugins, add_scene_utilities
from utils.message_handler import SofaMessageHandler
from objects.fixed_rigid_body import add_environment
from robots.catheter import CatheterRobot

from data_collection.generators import (
    SweepGenerator, SinusoidalGenerator, ProximalIdentificationGenerator,
    SSMConstraintGenerator, RichProximalGenerator)
from data_collection.collector import DataCollectorController
from data_collection.matrix_recorder import MatrixRecorderController

from utils.sofa_reader import SofaReader
from utils.plotter import DiagnosticPlotter
from controllers.plot_controller import PlotController

_ROBOT_CONFIG_PATH = os.path.join(_SIM_DIR, "configs", "catheter_ablation.yaml")
_SCENES_CONFIG_PATH = os.path.join(_SIM_DIR, "configs", "generated_scenes.yaml")

_GENERATORS = {
    "sweep": SweepGenerator,
    "sinusoidal": SinusoidalGenerator,
    "proximal_id": ProximalIdentificationGenerator,
    "ssm_constraint": SSMConstraintGenerator,
    "rich_proximal": RichProximalGenerator,
}


# ── Scene format helpers ──────────────────────────────────────────────────

def _normalize_scene(entry):
    """Normalize a scene entry to ``{"objects": [...], "initial_config": ...}``.

    Supports both the new dict format and legacy plain-list format.
    """
    if isinstance(entry, dict) and "objects" in entry:
        return entry
    # Legacy: scene is a plain list of objects
    if isinstance(entry, list):
        return {"objects": entry}
    raise ValueError(f"Invalid scene entry: {entry!r}")


def _get_scene_objects(scene_dict):
    """Extract the objects list from a normalized scene dict."""
    return scene_dict["objects"]


def _get_initial_config(scene_dict):
    """Extract initial_config or None from a normalized scene dict."""
    return scene_dict.get("initial_config")


def _obj_name(entry):
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        return entry.get("type", entry.get("name", "custom"))
    return "custom"


def _scene_tag(scene_objects):
    if not scene_objects:
        return "free_space"
    return "_".join(_obj_name(e) for e in scene_objects)


def _load_scenes(scenes_path=None):
    """Load scenes list from *scenes_path* or ``COLLECT_SCENES`` env var.

    Falls back to ``_SCENES_CONFIG_PATH`` if neither is provided.
    Also reads ``robot`` field to override the robot config path.
    """
    global _ROBOT_CONFIG_PATH
    path = scenes_path or os.environ.get("COLLECT_SCENES", "") or _SCENES_CONFIG_PATH
    with open(path) as f:
        data = yaml.safe_load(f)
    raw = data.get("scenes", [])
    if not raw:
        raise ValueError(f"No scenes found in {path}")

    # Override robot config if specified in the scene yaml
    robot_cfg = data.get("robot")
    if robot_cfg:
        # Resolve relative to the scene yaml's directory
        scenes_dir = os.path.dirname(os.path.abspath(path))
        robot_path = os.path.join(scenes_dir, robot_cfg)
        if os.path.exists(robot_path):
            _ROBOT_CONFIG_PATH = robot_path
            print(f"  Robot config: {_ROBOT_CONFIG_PATH}")

    return [_normalize_scene(s) for s in raw]


# ── _InitSaveController ──────────────────────────────────────────────────

class _InitSaveController(Sofa.Core.Controller):
    """Keyboard controller augmented with:

    - **p**: snapshot base_pose + strain_coords, save (persist) to YAML
    - **0**: gradually ramp rotation and cable to zero using joint rates

    The saved ``initial_config`` contains the physical rod state, not
    control inputs — all trajectories start from zero actuations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._keyboard_ctrl = kwargs.pop("keyboard_controller")
        self._base_mo = kwargs.pop("base_mechanical_object")
        self._strain_mo = kwargs.pop("strain_mechanical_object")
        self._scene_idx = kwargs.pop("scene_idx")
        self._yaml_path = kwargs.pop("yaml_path")
        self._saved = False
        self._zeroing = False
        self.listening = True

    def onKeypressedEvent(self, event):
        key = event.get("key")
        if not isinstance(key, str):
            return
        if key.lower() == "p":
            self._save_config()
        elif key == "0":
            self._start_zeroing()

    def _start_zeroing(self):
        """Begin gradual ramp of rotation and cable joints to zero."""
        kbd = self._keyboard_ctrl
        if self._zeroing:
            print("[init] Already zeroing...")
            return
        needs_zero = abs(kbd.joint_pos[1]) > 1e-8 or np.any(np.abs(kbd.joint_pos[2:]) > 1e-8)
        if not needs_zero:
            print("[init] Rotation and cable already at zero.")
            return
        self._zeroing = True
        print(f"[init] Zeroing rotation and cable from {kbd.joint_pos[1:]}")

    def onAnimateBeginEvent(self, _event):
        if not self._zeroing:
            return
        kbd = self._keyboard_ctrl
        dt = float(self._base_mo.getContext().dt.value)
        joint_rate = np.array(kbd.joint_rate, dtype=float)
        done = True
        # Ramp rotation (joint 1) toward zero
        if abs(kbd.joint_pos[1]) > 1e-8:
            step = joint_rate[1] * dt
            if abs(kbd.joint_pos[1]) <= step:
                kbd.joint_pos[1] = 0.0
            else:
                kbd.joint_pos[1] -= np.sign(kbd.joint_pos[1]) * step
                done = False
        # Ramp cable joints (joint 2+) toward zero
        for j in range(2, len(kbd.joint_pos)):
            if abs(kbd.joint_pos[j]) > 1e-8:
                rate = joint_rate[j] if j < len(joint_rate) else joint_rate[-1]
                step = rate * dt
                if abs(kbd.joint_pos[j]) <= step:
                    kbd.joint_pos[j] = 0.0
                else:
                    kbd.joint_pos[j] -= np.sign(kbd.joint_pos[j]) * step
                    done = False
        if done:
            self._zeroing = False
            print(f"[init] Zeroing complete: joint_pos = {kbd.joint_pos}")

    def _save_config(self):
        # Read physical state from SOFA
        base_pose = np.array(self._base_mo.position.value[0], dtype=float)
        strain_coords = np.array(self._strain_mo.position.value, dtype=float)

        initial_config = {
            "base_pose": [round(float(v), 8) for v in base_pose],
            "strain_coords": [[round(float(v), 8) for v in row]
                              for row in strain_coords],
        }

        with open(self._yaml_path) as f:
            data = yaml.safe_load(f) or {}
        scenes = data.get("scenes", [])
        if self._scene_idx < len(scenes):
            scene = scenes[self._scene_idx]
            if isinstance(scene, list):
                scenes[self._scene_idx] = {
                    "objects": scene,
                    "initial_config": initial_config,
                }
            else:
                scene["initial_config"] = initial_config

        with open(self._yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=None, sort_keys=False)

        print(f"\n[init] Scene {self._scene_idx}: saved initial_config")
        print(f"  base_pose = {initial_config['base_pose']}")
        print(f"  strain_coords: {len(strain_coords)} sections")
        print(f"[init] Written to {self._yaml_path}")
        print(f"[init] Close the window to proceed to the next scene.")
        self._saved = True


# ── Unified scene builder ───────────────────────────────────────────��────

def createScene(root: Sofa.Core.Node, headless: bool = False,
                scene_dict=None, scene_index: int = -1,
                init_mode: bool = False,
                init_scene_idx: int = 0,
                init_yaml_path: str = "",
                output_dir: str = "") -> Sofa.Core.Node:
    """Build a data-collection or manual-init scene.

    Parameters
    ----------
    root : Sofa.Core.Node
    headless : bool
        If True, skip the diagnostic plotter.
    scene_dict : dict or None
        Normalized scene dict with ``objects`` and optional ``initial_config``.
        If None, loads from config.
    init_mode : bool
        If True, add keyboard controller + init-save controller instead
        of the automated data collector.
    init_scene_idx : int
        Scene index in the YAML (used by _InitSaveController to write back).
    init_yaml_path : str
        Path to the YAML file (used by _InitSaveController to write back).
    """
    # (warmup removed — initial state applied once, solver maintains it)

    # Detect init mode (set by _run_init_mode via env vars on each subprocess)
    _init_idx_str = os.environ.get("_INIT_SCENE_IDX")
    if _init_idx_str is not None and not init_mode:
        yaml_path = os.environ.get("_INIT_YAML_PATH", "")
        scene_idx = int(_init_idx_str)
        scenes = _load_scenes(yaml_path or None)
        scene_dict = scenes[min(scene_idx, len(scenes) - 1)]
        init_mode = True
        init_scene_idx = scene_idx
        init_yaml_path = yaml_path

    with open(_ROBOT_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    # Read scene index from file to bypass runSofa's module caching.
    # Change the index by editing /tmp/collect_scene_idx.txt (just a number).
    _scene_idx_file = "/tmp/collect_scene_idx.txt"
    _default_idx = 0
    if os.path.isfile(_scene_idx_file):
        try:
            with open(_scene_idx_file) as _f:
                _default_idx = int(_f.read().strip())
        except (ValueError, OSError):
            pass
    if scene_dict is None:
        scenes = _load_scenes()
        scene_index = int(os.environ.get("COLLECT_SCENE_IDX", str(_default_idx)))
        scene_dict = scenes[min(scene_index, len(scenes) - 1)]

    scene_objects = _get_scene_objects(scene_dict)
    initial_config = _get_initial_config(scene_dict)
    tag = _scene_tag(scene_objects)

    # Per-scene toggles (default: enabled for backward compat)
    enable_tip_force = scene_dict.get("enable_tip_force", True)
    # enable_control: true/false (all joints), or list of bools per joint
    # e.g. [false, true, true] = no insertion, yes rotation, yes cable
    enable_control = scene_dict.get("enable_control", True)

    root.gravity = [0.0, 0.0, 0.0]
    root.dt = 0.01

    add_required_plugins(root)
    add_scene_utilities(root)

    cable_mode = cfg.get("actuation", {}).get("cable_mode", "force")

    add_environment(root, scene_objects)
    robot = CatheterRobot(root, config_path=_ROBOT_CONFIG_PATH, cable_mode=cable_mode)
    strain_mo = robot._prefab.cosseratCoordinate.cosseratCoordinateMO

    # NOTE: Do NOT write initial_config here — Sofa.Simulation.init() applies
    # the MO's translation/rotation params on top of position, which would
    # double-transform the saved state.  The controller applies it on step 1
    # (after init).

    # Tip force field for random perturbation during data collection
    rod_cfg = cfg.get("rod", {})
    n_frames = int(rod_cfg.get("n_frames", 32)) + 1  # geometry appends tip
    tip_frame_idx = n_frames - 1
    tip_force_field = robot._prefab.cosseratFrame.addObject(
        "ConstantForceField",
        name="tipForce",
        indices=[tip_frame_idx],
        forces=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        showArrowSize=0.01,
    )

    reader = SofaReader(
        prefab=robot._prefab,
        base_mo=robot.base_mo,
        cable_constraints=robot.cable_constraints,
        prefab_rotation_offset=robot.prefab_rotation_offset,
        cable_mode=cable_mode,
    )

    dt = float(root.dt.value)

    # ── Mode-specific controller ─────────────────────────────────────
    if init_mode:
        from controllers.keyboard_controller import CatheterKeyboardController

        kbd = CatheterKeyboardController(
            name="CatheterKeyboardController",
            base_mechanical_object=robot.base_mo,
            base_actuator=robot.base_actuator,
            direction=robot.insertion_direction,
            joint_rate=robot.joint_rate,
            joint_upper_limits=robot.joint_upper_limits,
            joint_lower_limits=robot.joint_lower_limits,
            base_position=robot.base_position,
            base_orientation=robot.base_orientation,
            prefab_rotation_offset=robot.prefab_rotation_offset,
            cable_constraint=robot.cable_constraint,
        )
        root.addObject(kbd)

        # Pre-load existing initial_config into keyboard controller
        if initial_config and "base_pose" in initial_config:
            from scipy.spatial.transform import Rotation as Rot
            saved_pose = np.array(initial_config["base_pose"], dtype=float)
            base_home = np.array(robot.base_position, dtype=float)
            offset = saved_pose[:3] - base_home
            home_rot = Rot.from_quat(np.array(robot.base_orientation))
            local_dir = np.array(robot.insertion_direction, dtype=float)
            world_dir = local_dir @ home_rot.as_matrix().T
            world_dir = world_dir / np.linalg.norm(world_dir)
            kbd.joint_pos[0] = float(np.dot(offset, world_dir))

        root.addObject(
            _InitSaveController(
                name="InitSaveController",
                keyboard_controller=kbd,
                base_mechanical_object=robot.base_mo,
                strain_mechanical_object=strain_mo,
                scene_idx=init_scene_idx,
                yaml_path=init_yaml_path,
            )
        )
        title = f"Manual Init — Scene {init_scene_idx}: {tag}"

    else:
        gen_name = os.environ.get("COLLECT_GENERATOR", "sweep")
        output_path = os.environ.get("COLLECT_OUTPUT", "")
        if not output_path:
            data_dir = output_dir or os.getcwd()
            ts = time.strftime("%Y%m%d_%H%M%S")
            idx_str = f"{scene_index:03d}" if scene_index >= 0 else "x"
            output_path = os.path.join(data_dir, f"{gen_name}_{idx_str}_{ts}.h5")

        joint_lower = np.array(robot.joint_lower_limits, dtype=float)
        joint_upper = np.array(robot.joint_upper_limits, dtype=float)

        gen_cls = _GENERATORS.get(gen_name)
        if gen_cls is None:
            raise ValueError(f"Unknown generator: {gen_name!r}. "
                             f"Available: {list(_GENERATORS.keys())}")
        gen_kwargs = {}
        import inspect
        if "seed" in inspect.signature(gen_cls.__init__).parameters:
            gen_kwargs["seed"] = int(os.environ.get("COLLECT_SEED", "0"))
        duration = float(os.environ.get("COLLECT_DURATION", "0"))
        if duration > 0:
            if "duration" in inspect.signature(gen_cls.__init__).parameters:
                gen_kwargs["duration"] = duration
        if hasattr(robot, "joint_rate"):
            import inspect
            if "joint_max_speeds" in inspect.signature(gen_cls.__init__).parameters:
                gen_kwargs["joint_max_speeds"] = np.array(robot.joint_rate, dtype=float)
        speed_factor = float(os.environ.get("COLLECT_SPEED_FACTOR", "0"))
        if speed_factor > 0:
            import inspect
            if "speed_factor" in inspect.signature(gen_cls.__init__).parameters:
                gen_kwargs["speed_factor"] = speed_factor
        generator = gen_cls(joint_lower, joint_upper, dt, **gen_kwargs)
        if isinstance(enable_control, list):
            # Per-joint mask: e.g. [false, true, true]
            _inner = generator
            _mask = np.array(enable_control, dtype=bool)
            class _MaskedGenerator:
                name = _inner.name
                joint_lower = _inner.joint_lower
                joint_upper = _inner.joint_upper
                def step(self, t):
                    cmd = _inner.step(t)
                    cmd[~_mask] = 0.0
                    return cmd
                def is_done(self, t):
                    return _inner.is_done(t)
            generator = _MaskedGenerator()
        elif not enable_control:
            # All joints disabled
            _inner = generator
            class _ZeroGenerator:
                name = "zero"
                joint_lower = _inner.joint_lower
                joint_upper = _inner.joint_upper
                def step(self, t):
                    return np.zeros_like(_inner.joint_lower)
                def is_done(self, t):
                    return _inner.is_done(t)
            generator = _ZeroGenerator()

        metadata = {
            "schema_version": 1,
            "scene_index": scene_index,
            "task": tag,
        }
        # Legacy fields kept for backward compatibility with loader
        metadata["scene_objects"] = scene_objects

        # Extract initial physical state + back-calculate insertion offset
        init_base_pose = None
        init_strain_coords = None
        insertion_offset = 0.0
        if initial_config:
            if "base_pose" in initial_config:
                init_base_pose = np.array(initial_config["base_pose"], dtype=float)
                # Back-calculate insertion from saved base_pose
                from scipy.spatial.transform import Rotation as Rot
                base_home = np.array(robot.base_position, dtype=float)
                offset = init_base_pose[:3] - base_home
                home_rot = Rot.from_quat(np.array(robot.base_orientation))
                local_dir = np.array(robot.insertion_direction, dtype=float)
                world_dir = local_dir @ home_rot.as_matrix().T
                world_dir = world_dir / np.linalg.norm(world_dir)
                insertion_offset = float(np.dot(offset, world_dir))
            if "strain_coords" in initial_config:
                init_strain_coords = np.array(initial_config["strain_coords"], dtype=float)

        tip_force_cfg = cfg.get("tip_force", {})
        tip_force_max = float(tip_force_cfg.get("max_force", 0.5))
        # The identification suite requires named force components to sum to
        # the complete free RHS. An unlabelled random tip force would violate
        # that reconstruction and confound passive ring-down episodes.
        if (not enable_tip_force
                or gen_name in {"proximal_id", "ssm_constraint",
                                "rich_proximal"}):
            tip_force_max = 0.0

        controller = DataCollectorController(
            name="DataCollectorController",
            generator=generator,
            reader=reader,
            base_mechanical_object=robot.base_mo,
            base_actuator=robot.base_actuator,
            cable_constraint=robot.cable_constraint,
            direction=robot.insertion_direction,
            base_position=robot.base_position,
            base_orientation=robot.base_orientation,
            prefab_rotation_offset=robot.prefab_rotation_offset,
            output_path=output_path,
            metadata=metadata,
            initial_base_pose=init_base_pose,
            initial_strain_coords=init_strain_coords,
            strain_mechanical_object=strain_mo,
            insertion_offset=insertion_offset,
            tip_force_field=tip_force_field,
            tip_force_max=tip_force_max,
        )
        root.addObject(controller)

        # Matrix recorder (enabled via COLLECT_MATRICES=1)
        record_matrices = os.environ.get("COLLECT_MATRICES", "0") == "1"
        if record_matrices:
            solver_node_obj = root.getChild("CatheterSimulation")
            A_interval = int(os.environ.get("COLLECT_A_INTERVAL", "1"))
            verbose = os.environ.get("COLLECT_VERBOSE", "0") == "1"
            debug = os.environ.get("COLLECT_DEBUG", "0") == "1"
            record_components = (
                os.environ.get("COLLECT_COMPONENTS", "0") == "1")
            matrix_recorder = MatrixRecorderController(
                name="MatrixRecorder",
                robot=robot,
                solver_node=solver_node_obj,
                A_interval=A_interval,
                verbose=verbose,
                debug=debug,
                record_components=record_components,
                distal_start_section=int(rod_cfg.get(
                    "distal_start_section", 25)),
            )
            root.addObject(matrix_recorder)

        title = f"Data Collection — {tag}"

    # ── Matrix analysis (GUI modes only, runs once after init) ────────
    if not headless:
        from utils.matrix_analysis import analyze_damping

        class _MatrixAnalysisController(Sofa.Core.Controller):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._robot = kwargs.pop("robot")
                self._solver_node = kwargs.pop("solver_node")
                self._step = 0

            def onAnimateEndEvent(self, _event):
                # Run after the first solve completes so A() is populated
                self._step += 1
                if self._step != 1:
                    return
                try:
                    analyze_damping(self._robot, self._solver_node)
                except Exception as e:
                    import traceback
                    print(f"[matrix_analysis] Failed: {e}")
                    traceback.print_exc()

        solver_node_obj = root.getChild("CatheterSimulation")
        root.addObject(
            _MatrixAnalysisController(
                name="MatrixAnalysis",
                robot=robot,
                solver_node=solver_node_obj,
            )
        )

    # ── Plotter (GUI modes only) ──────────────────────────────────────
    if not headless:
        rod_cfg = cfg.get("rod", {})
        n_nodes = int(rod_cfg.get("n_frames", 33))
        n_sections = int(rod_cfg.get("n_sections", 32))
        n_cables = len(cfg.get("actuation", {}).get("cable_locations", [[0, 0]]))

        plotter = DiagnosticPlotter(
            n_nodes=n_nodes,
            n_sections=n_sections,
            rod_length=float(rod_cfg.get("length", 0.16)),
            panels=("base_translation", "base_rotation",
                    "tendon_force", "contact_force", "tip_load"),
            window_seconds=10.0,
            dt=dt,
            n_cables=n_cables,
            title=title,
            size=(1300, 600),
        )
        root.addObject(
            PlotController(
                name="PlotController",
                plotter=plotter,
                reader=reader,
                n_nodes=n_nodes,
                base_home_position=robot.base_position,
                tip_force_field=tip_force_field,
                data_collector=controller if not init_mode else None,
            )
        )

    return root


# ---------------------------------------------------------------------------
# Manual init mode — opens runSofa per scene
# ---------------------------------------------------------------------------

def _run_init_mode(yaml_path, scene_indices=None):
    """Open each scene in runSofa GUI for manual init.

    Always loops through all scenes (or the specified indices).  If a
    scene already has ``initial_config``, it is pre-loaded so the user
    can review and optionally override by pressing Enter.
    """
    import subprocess

    with open(yaml_path) as f:
        data = yaml.safe_load(f) or {}
    scenes = [_normalize_scene(s) for s in data.get("scenes", [])]

    if scene_indices is None:
        scene_indices = list(range(len(scenes)))

    print(f"[init] {len(scene_indices)} scene(s) to initialize: {scene_indices}")
    print(f"[init] Controls: u/j insertion, k/h rotation, i/y cable")
    print(f"[init]           0 = zero rotation & cable")
    print(f"[init]           p = save (persist) config")
    print(f"[init] Close the window to proceed to the next scene.\n")

    for idx in scene_indices:
        scene = scenes[idx]
        objects = _get_scene_objects(scene)
        tag = _scene_tag(objects)
        existing = _get_initial_config(scene)
        status = "has config" if existing else "no config"
        print(f"[init] Opening scene {idx}: {tag} ({status})")

        env = os.environ.copy()
        env["_INIT_SCENE_IDX"] = str(idx)
        env["_INIT_YAML_PATH"] = os.path.abspath(yaml_path)

        script_path = os.path.abspath(__file__)
        cmd = ["runSofa", "-g", "qt", script_path]
        subprocess.run(cmd, env=env)

        # Reload to check if config was saved/updated
        with open(yaml_path) as f:
            data = yaml.safe_load(f) or {}
        scenes = [_normalize_scene(s) for s in data.get("scenes", [])]
        updated = _get_initial_config(scenes[idx])
        if updated:
            print(f"[init] Scene {idx}: config saved")
        else:
            print(f"[init] Scene {idx}: no config (skipped)")
        print()

    # Summary
    n_configured = sum(
        1 for s in scenes if _get_initial_config(s) is not None
    )
    print(f"[init] Done. {n_configured}/{len(scenes)} scenes have initial_config.")


# ---------------------------------------------------------------------------
# True headless entry point — runs ALL scenes sequentially
# ---------------------------------------------------------------------------

def _run_one_scene(scene_dict, scene_idx, total_scenes, output_dir=""):
    """Run one scene headless and return (n_steps, wall_time)."""
    max_duration = float(os.environ.get("COLLECT_DURATION", "0"))
    scene_objects = _get_scene_objects(scene_dict)
    initial_config = _get_initial_config(scene_dict)

    def _obj_label(entry):
        if isinstance(entry, str):
            return entry
        if isinstance(entry, dict):
            return entry.get("type", "custom")
        return "custom"
    label = ", ".join(_obj_label(e) for e in scene_objects)

    root = Sofa.Core.Node("root")
    createScene(root, headless=True, scene_dict=scene_dict, scene_index=scene_idx,
                output_dir=output_dir)
    Sofa.Simulation.init(root)

    dt = root.dt.value
    controller = root.getObject("DataCollectorController")
    gen = controller._generator
    total_sim_time = getattr(gen, "_total_time", None) or getattr(gen, "_duration", None)
    time_str = f", total_sim_time={total_sim_time:.1f}s" if total_sim_time else ""
    init_str = ""
    if initial_config:
        init_str = ", has initial_config"

    print(f"\n[collect_data] Scene {scene_idx + 1}/{total_scenes}: "
          f"[{label}]{init_str}{time_str}")

    step = 0
    t0 = time.time()

    while not controller._done:
        Sofa.Simulation.animate(root, dt)
        step += 1
        if step % 500 == 0:
            elapsed = time.time() - t0
            print(f"  step {step}, sim_time={step * dt:.2f}s, "
                  f"wall_time={elapsed:.1f}s", flush=True)
        if max_duration > 0 and step * dt > max_duration:
            print(f"  Reached duration={max_duration:.1f}s, forcing save.")
            controller._finish()
            break

    elapsed = time.time() - t0
    print(f"  Done: {step} steps in {elapsed:.1f}s ({step / elapsed:.0f} steps/s)")

    # Save matrix data if recorded
    matrix_recorder = root.getObject("MatrixRecorder")
    if matrix_recorder is not None:
        import h5py
        import json as _json
        mat_data = matrix_recorder.get_data()
        mat_meta = matrix_recorder.get_metadata()
        h5_path = controller._output_path
        print(f"  Appending matrix data to {h5_path}")
        with h5py.File(h5_path, "a") as f:
            grp = f.create_group("matrices")
            for name, arr in mat_data.items():
                if arr is None:
                    continue
                if np.ndim(arr) == 0:
                    grp.create_dataset(name, data=arr)
                elif hasattr(arr, 'size') and arr.size > 0:
                    grp.create_dataset(name, data=arr, compression="gzip",
                                       compression_opts=4)
                else:
                    grp.create_dataset(name, data=arr)
            grp.attrs["metadata"] = _json.dumps(mat_meta)

    Sofa.Simulation.unload(root)
    return step, elapsed


def _write_collection_meta(output_dir, gen_name, gen_cls, gen_kwargs, dt,
                           cable_mode):
    """Copy configs to output_dir and write collection_meta.json."""
    import shutil
    import json as _json

    os.makedirs(output_dir, exist_ok=True)

    # Copy rod and scene configs into the data folder (always overwrite)
    scenes_path = os.environ.get("COLLECT_SCENES", "") or _SCENES_CONFIG_PATH
    rod_dst = os.path.join(output_dir, os.path.basename(_ROBOT_CONFIG_PATH))
    scene_dst = os.path.join(output_dir, os.path.basename(scenes_path))
    shutil.copy2(_ROBOT_CONFIG_PATH, rod_dst)
    shutil.copy2(scenes_path, scene_dst)

    # Extract generator params from its __init__ signature defaults + kwargs
    import inspect
    sig = inspect.signature(gen_cls.__init__)
    gen_params = {}
    for name, param in sig.parameters.items():
        if name in ("self", "joint_lower_limits", "joint_upper_limits", "dt"):
            continue
        if name in gen_kwargs:
            gen_params[name] = gen_kwargs[name]
        elif param.default is not inspect.Parameter.empty:
            val = param.default
            if isinstance(val, tuple):
                val = list(val)
            gen_params[name] = val

    meta = {
        "schema_version": 1,
        "rod_config": os.path.basename(_ROBOT_CONFIG_PATH),
        "scene_config": os.path.basename(scenes_path),
        "generator": gen_name,
        "generator_params": gen_params,
        "simulation": {
            "dt": dt,
            "cable_mode": cable_mode,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    meta_path = os.path.join(output_dir, "collection_meta.json")
    with open(meta_path, "w") as f:
        _json.dump(meta, f, indent=2)
    print(f"[collect_data] Wrote {meta_path}")


def run_headless(output_dir="", scene_indices=None):
    """Run scenes from the scenes YAML config sequentially.

    Parameters
    ----------
    scene_indices : list[int] or None
        If provided, only run scenes at these indices. Default: all.
    """
    scenes = _load_scenes()
    if scene_indices is not None:
        selected = [(i, scenes[i]) for i in scene_indices if i < len(scenes)]
    else:
        selected = list(enumerate(scenes))
    print(f"[collect_data] {len(selected)} scene(s) to collect "
          f"(of {len(scenes)} total)")

    # Write collection-level metadata (configs + generator info)
    data_dir = output_dir or os.getcwd()
    gen_name = os.environ.get("COLLECT_GENERATOR", "sweep")
    gen_cls = _GENERATORS.get(gen_name)
    gen_kwargs = {}
    if gen_cls is not None:
        import inspect
        sig = inspect.signature(gen_cls.__init__)
        if "seed" in sig.parameters:
            gen_kwargs["seed"] = int(os.environ.get("COLLECT_SEED", "0"))
        duration = float(os.environ.get("COLLECT_DURATION", "0"))
        if duration > 0 and "duration" in sig.parameters:
            gen_kwargs["duration"] = duration
        speed_factor = float(os.environ.get("COLLECT_SPEED_FACTOR", "0"))
        if speed_factor > 0 and "speed_factor" in sig.parameters:
            gen_kwargs["speed_factor"] = speed_factor

    with open(_ROBOT_CONFIG_PATH) as f:
        rod_cfg = yaml.safe_load(f)
    cable_mode = rod_cfg.get("actuation", {}).get("cable_mode", "force")

    if gen_cls is not None:
        _write_collection_meta(data_dir, gen_name, gen_cls, gen_kwargs,
                               dt=0.01, cable_mode=cable_mode)

    msg_handler = SofaMessageHandler(print_info=False)
    total_steps = 0
    total_time = 0.0

    with msg_handler:
        for i, scene_dict in selected:
            steps, elapsed = _run_one_scene(scene_dict, i, len(scenes),
                                            output_dir=output_dir)
            total_steps += steps
            total_time += elapsed

    print(f"\n[collect_data] All done: {total_steps} total steps "
          f"in {total_time:.1f}s across {len(selected)} scene(s)")
    print(f"[collect_data] {msg_handler.summary()}")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data collection / manual init")
    parser.add_argument("--init", action="store_true",
                        help="Manual init mode: open each scene in GUI "
                             "to set initial robot configuration")
    parser.add_argument("--scenes", type=str, default=None,
                        help="Path to scenes YAML file (overrides COLLECT_SCENES)")
    parser.add_argument("--scene-idx", type=int, nargs="+", default=None,
                        help="Specific scene indices to init (default: all)")
    parser.add_argument("--output-dir", type=str, default="",
                        help="Output directory for HDF5 files (default: cwd)")
    parser.add_argument("--debug", action="store_true",
                        help="Record diagnostic terms in matrix recorder "
                             "(solver b/x, forces, Jacobians, b-reconstruction)")
    args = parser.parse_args()

    if args.scenes:
        os.environ["COLLECT_SCENES"] = args.scenes
    if args.debug:
        os.environ["COLLECT_DEBUG"] = "1"

    if args.init:
        yaml_path = args.scenes or os.environ.get("COLLECT_SCENES", "") or _SCENES_CONFIG_PATH
        _run_init_mode(yaml_path, scene_indices=args.scene_idx)
    else:
        run_headless(output_dir=args.output_dir, scene_indices=args.scene_idx)
