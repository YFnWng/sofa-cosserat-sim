"""Feedback control scene — drives the catheter with a closed-loop
controller and records per-timestep state to HDF5.

Supports multiple scenes (defined in a YAML ``scenes`` list), each run
as a separate SOFA session.

Execution modes:

1. **GUI** (runSofa) — runs one scene with visual debugging:
       runSofa -g qt simulation/scenes/feedback_control.py

2. **Headless** — runs ALL scenes sequentially:
       python simulation/scenes/feedback_control.py --config path/to/config.yaml

YAML config format
------------------
See ``simulation/configs/feedback_control.yaml`` for the full schema.
Key sections: robot, scenes, observer, controller, reference, output, simulation.
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

from utils.scene import add_required_plugins, add_scene_utilities
from utils.message_handler import SofaMessageHandler
from objects.fixed_rigid_body import add_environment
from robots.catheter import CatheterRobot
from utils.sofa_reader import SofaReader

_DEFAULT_CONFIG_PATH = os.path.join(_SIM_DIR, "configs", "feedback_control.yaml")


# ── Config loading ───────────────────────────────────────────────────

def _load_config(config_path: str = None) -> dict:
    """Load the feedback control YAML config."""
    path = config_path or os.environ.get(
        "FEEDBACK_CONFIG", _DEFAULT_CONFIG_PATH)
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_robot_config(config: dict, config_dir: str) -> str:
    """Resolve robot config path from the YAML."""
    robot_cfg = config.get("robot", "catheter_ablation.yaml")
    robot_path = os.path.join(config_dir, robot_cfg)
    if not os.path.exists(robot_path):
        robot_path = os.path.join(_SIM_DIR, "configs", robot_cfg)
    return robot_path


def _normalize_scene(entry):
    """Normalize a scene entry to dict with ``objects`` key."""
    if isinstance(entry, dict) and "objects" in entry:
        return entry
    if isinstance(entry, list):
        return {"objects": entry}
    raise ValueError(f"Invalid scene entry: {entry!r}")



# ── Scene builder ────────────────────────────────────────────────────

def createScene(root: Sofa.Core.Node, headless: bool = False,
                config: dict = None, scene_dict: dict = None,
                scene_index: int = 0,
                output_dir: str = "") -> Sofa.Core.Node:
    """Build a feedback control scene.

    Parameters
    ----------
    root : Sofa.Core.Node
    headless : bool
    config : dict
        Full YAML config. Loaded from default if None.
    scene_dict : dict
        Specific scene to run (overrides config scenes list).
    scene_index : int
        Index into the scenes list.
    output_dir : str
        Output directory for H5 files.
    """
    if config is None:
        config = _load_config()

    config_dir = os.path.dirname(os.path.abspath(
        os.environ.get("FEEDBACK_CONFIG", _DEFAULT_CONFIG_PATH)))
    robot_config_path = _resolve_robot_config(config, config_dir)

    with open(robot_config_path) as f:
        rod_cfg = yaml.safe_load(f)

    # Scene selection
    if scene_dict is None:
        scenes = [_normalize_scene(s) for s in config.get("scenes", [{}])]
        scene_dict = scenes[min(scene_index, len(scenes) - 1)]

    scene_objects = scene_dict.get("objects", [])
    initial_config = scene_dict.get("initial_config")

    # Simulation parameters
    sim_cfg = config.get("simulation", {})
    duration = float(sim_cfg.get("duration", 30.0))
    sim_dt = float(sim_cfg.get("dt", 0.01))
    control_rate = int(sim_cfg.get("control_rate", 1))

    root.gravity = [0.0, 0.0, 0.0]
    root.dt = sim_dt

    add_required_plugins(root)
    add_scene_utilities(root)

    cable_mode = rod_cfg.get("actuation", {}).get("cable_mode", "force")

    add_environment(root, scene_objects)
    robot = CatheterRobot(root, config_path=robot_config_path,
                          cable_mode=cable_mode)
    strain_mo = robot._prefab.cosseratCoordinate.cosseratCoordinateMO

    reader = SofaReader(
        prefab=robot._prefab,
        base_mo=robot.base_mo,
        cable_constraints=robot.cable_constraints,
        prefab_rotation_offset=robot.prefab_rotation_offset,
        cable_mode=cable_mode,
    )

    # Build robot interface and sensor suite
    from cr_common.robot_interface import RobotInterface
    robot_iface = RobotInterface.from_yaml(robot_config_path)
    sensor_suite = None
    sensor_cfg = robot_iface.sensor_config
    if sensor_cfg:
        n_nodes = int(rod_cfg.get("rod", {}).get("n_frames", 32)) + 1
        sensor_suite = robot_iface.build_sensor_suite(n_nodes)

    # Control mode
    ctrl_space_cfg = config.get("control", {})
    control_mode = ctrl_space_cfg.get("mode", "position")
    control_sensor = ctrl_space_cfg.get("sensor", "mri_coils")
    control_sensor_index = int(ctrl_space_cfg.get("sensor_index", -1))

    # Prevent BLAS threading conflicts between SOFA and PyTorch/MKL
    import torch
    torch.set_num_threads(1)

    # Build world model (observer + dynamics + observation)
    from cr_meta_lnn.networks.world_model import build_model
    model_cfg = config.get("model", config.get("observer", {"observer": "none"}))
    # Resolve model_dir relative to workspace root, not CWD
    if "model_dir" in model_cfg and not os.path.isabs(model_cfg["model_dir"]):
        model_cfg = dict(model_cfg)
        model_cfg["model_dir"] = os.path.normpath(
            os.path.join(_WORKSPACE, model_cfg["model_dir"]))
    model_dt = sim_dt * control_rate
    world_model = build_model(model_cfg, dt=model_dt, robot=robot_iface)
    # Reset observer with physically meaningful initial state:
    # z_rest = latent encoding of straight rod (q=0), not z=0
    if world_model.has_observer:
        encoder = world_model.observation._encoder
        z_rest = encoder.encode(np.zeros((1, encoder.n_strain))).flatten()
        init_base = robot_iface.encode_base_state(
            np.zeros(robot_iface.base_state_raw_dim))
        world_model.reset(z0=z_rest, base_state=init_base)

    # Build reference
    from control.factory import build_reference, build_controller
    ref_cfg = config.get("reference", {"type": "fixed",
                                        "target": [0, 0, 0.12]})
    reference = build_reference(ref_cfg)

    # Build controller
    ctrl_cfg = config.get("controller", {"type": "pid"})
    controller = build_controller(
        ctrl_cfg,
        world_model=world_model,
        control_mode=control_mode,
        control_sensor=control_sensor,
        control_sensor_index=control_sensor_index,
        robot=robot_iface,
        dt=model_dt)

    # Output path
    out_cfg = config.get("output", {})
    output_path = out_cfg.get("path", "")
    if not output_path:
        data_dir = output_dir or out_cfg.get("dir", os.getcwd())
        os.makedirs(data_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        ctrl_type = ctrl_cfg.get("type", "ctrl")
        output_path = os.path.join(
            data_dir, f"feedback_{ctrl_type}_{scene_index:03d}_{ts}.h5")

    warmup_steps = int(out_cfg.get("warmup_steps", 50))
    record_enabled = bool(out_cfg.get("record", True))

    # Extract initial state
    init_base_pose = None
    init_strain_coords = None
    insertion_offset = 0.0
    if initial_config:
        if "base_pose" in initial_config:
            init_base_pose = np.array(initial_config["base_pose"], dtype=float)
            from scipy.spatial.transform import Rotation as Rot
            base_home = np.array(robot.base_position, dtype=float)
            offset = init_base_pose[:3] - base_home
            home_rot = Rot.from_quat(np.array(robot.base_orientation))
            local_dir = np.array(robot.insertion_direction, dtype=float)
            world_dir = local_dir @ home_rot.as_matrix().T
            world_dir = world_dir / np.linalg.norm(world_dir)
            insertion_offset = float(np.dot(offset, world_dir))
        if "strain_coords" in initial_config:
            init_strain_coords = np.array(
                initial_config["strain_coords"], dtype=float)

    # AdapJ babbling config
    babble_steps = 0
    babble_max_rate = None
    if ctrl_cfg.get("type") == "adapj":
        params = ctrl_cfg.get("params", {})
        babble_steps = int(params.get("babble_steps", 100))
        rate = params.get("babble_max_rate")
        if rate is not None:
            babble_max_rate = np.array(rate, dtype=float)

    metadata = {
        "schema_version": 1,
        "scene_index": scene_index,
        "controller_type": ctrl_cfg.get("type", "unknown"),
        "scene_objects": scene_objects,
    }

    from controllers.feedback_controller import FeedbackController

    fb_controller = FeedbackController(
        name="FeedbackController",
        controller=controller,
        world_model=world_model,
        sensor_suite=sensor_suite,
        robot_interface=robot_iface,
        reader=reader,
        reference=reference,
        control_mode=control_mode,
        control_sensor=control_sensor,
        control_sensor_index=control_sensor_index,
        base_mechanical_object=robot.base_mo,
        cable_constraint=robot.cable_constraint,
        direction=robot.insertion_direction,
        base_position=robot.base_position,
        base_orientation=robot.base_orientation,
        prefab_rotation_offset=robot.prefab_rotation_offset,
        initial_base_pose=init_base_pose,
        initial_strain_coords=init_strain_coords,
        strain_mechanical_object=strain_mo,
        insertion_offset=insertion_offset,
        output_path=output_path,
        warmup_steps=warmup_steps,
        record=record_enabled,
        control_rate=control_rate,
        metadata=metadata,
        babble_steps=babble_steps,
        babble_max_rate=babble_max_rate,
        duration=duration,
        sensor_config=sensor_cfg,
    )
    root.addObject(fb_controller)

    # Initialize autoregressive rollout state from ground-truth rest config
    if world_model.has_observer:
        dynamics = world_model.dynamics
        d = dynamics.d
        init_state = np.zeros(dynamics.state_dim, dtype=np.float64)
        init_state[:d] = z_rest
        if dynamics._actuation_model is not None:
            init_state[2*d:2*d + len(init_base)] = init_base
        fb_controller._rollout_state = init_state

    # ── Target + sensor visualization (GUI only) ───────────────────
    if not headless:
        ref_type = ref_cfg.get("type", "fixed")
        if ref_type == "fixed":
            target = np.array(ref_cfg["target"], dtype=float)
            _add_target_marker(root, target)
        elif ref_type in ("circle", "rectangle", "waypoints", "file"):
            sphere_mo, sphere_offsets = _add_path_marker(root, reference)
            if sphere_mo is not None:
                fb_controller._target_sphere_mo = sphere_mo
                fb_controller._target_sphere_offsets = sphere_offsets

            # Diagnostic spheres: +rot (green) and -rot (blue) weighted predictions
            diag_pos_mo, diag_neg_mo, diag_offsets = _add_diag_spheres(root)
            fb_controller._diag_pos_sphere_mo = diag_pos_mo
            fb_controller._diag_neg_sphere_mo = diag_neg_mo
            fb_controller._diag_sphere_offsets = diag_offsets

        # Ghost catheter + sensor markers (single SofaWriter)
        vis_cfg = config.get("visualization", {})
        show_est = vis_cfg.get("show_estimated_shape", False)
        n_sens = (sensor_suite.n_position_sensors
                  if sensor_suite is not None else 0)
        if show_est or n_sens > 0:
            from utils.sofa_writer import add_estimation_display
            n_nodes_est = robot_iface.n_sections + 1
            writer = add_estimation_display(
                root,
                n_nodes=n_nodes_est if show_est else 0,
                ds=robot_iface.ds,
                n_sensors=n_sens,
                base_position=robot.base_position,
                base_orientation=robot.base_orientation,
            )
            fb_controller._display_writer = writer
            fb_controller._shape_source = vis_cfg.get("shape_source", "observer")

        # Diagnostic plotter
        from utils.plotter import DiagnosticPlotter
        from controllers.plot_controller import PlotController

        n_frames_cfg = int(rod_cfg.get("rod", {}).get("n_frames", 32)) + 1
        n_sections = int(rod_cfg.get("rod", {}).get("n_sections", 32))
        n_cables = len(rod_cfg.get("actuation", {}).get(
            "cable_locations", [[0, 0]]))

        plotter = DiagnosticPlotter(
            n_nodes=n_frames_cfg,
            n_sections=n_sections,
            rod_length=float(rod_cfg.get("rod", {}).get("length", 0.16)),
            panels=("base_translation", "base_rotation",
                    "tendon_force", "contact_force", "tracking_error"),
            window_seconds=10.0,
            dt=sim_dt,
            n_cables=n_cables,
            title=f"Feedback Control — {ctrl_cfg.get('type', 'ctrl')}",
            size=(1300, 600),
        )
        root.addObject(
            PlotController(
                name="PlotController",
                plotter=plotter,
                reader=reader,
                n_nodes=n_frames_cfg,
                base_home_position=robot.base_position,
                feedback_controller=fb_controller,
            )
        )

    return root


def _make_sphere_mesh(center, radius, n_lat=8, n_lon=12):
    """Generate a UV sphere mesh (vertices + triangles)."""
    verts = []
    # Top pole
    verts.append([center[0], center[1], center[2] + radius])
    for i in range(1, n_lat):
        theta = np.pi * i / n_lat
        for j in range(n_lon):
            phi = 2.0 * np.pi * j / n_lon
            x = center[0] + radius * np.sin(theta) * np.cos(phi)
            y = center[1] + radius * np.sin(theta) * np.sin(phi)
            z = center[2] + radius * np.cos(theta)
            verts.append([x, y, z])
    # Bottom pole
    verts.append([center[0], center[1], center[2] - radius])

    tris = []
    # Top cap
    for j in range(n_lon):
        tris.append([0, 1 + j, 1 + (j + 1) % n_lon])
    # Middle bands
    for i in range(n_lat - 2):
        for j in range(n_lon):
            curr = 1 + i * n_lon + j
            next_j = 1 + i * n_lon + (j + 1) % n_lon
            below = curr + n_lon
            below_next = next_j + n_lon
            tris.append([curr, below, next_j])
            tris.append([next_j, below, below_next])
    # Bottom cap
    bottom = len(verts) - 1
    base = 1 + (n_lat - 2) * n_lon
    for j in range(n_lon):
        tris.append([bottom, base + (j + 1) % n_lon, base + j])

    return verts, tris


def _add_target_marker(root, position: np.ndarray):
    """Add a visual sphere at the target position."""
    verts, tris = _make_sphere_mesh(position, radius=0.001)
    target_node = root.addChild("TargetMarker")
    target_node.addObject(
        "OglModel",
        name="targetSphere",
        position=verts,
        triangles=tris,
        color=[1.0, 0.2, 0.2, 0.8],
    )


def _add_path_marker(root, reference):
    """Render a reference trajectory as a polyline with a moving target sphere.

    Returns (ogl_model, vertex_offsets) for the moving sphere so the
    feedback controller can update its position each timestep.
    """
    targets = reference.targets  # (N, 3)
    N = len(targets)
    if N < 2:
        return None, None

    edges = [[i, i + 1] for i in range(N - 1)]
    path_node = root.addChild("PathMarker")
    path_node.addObject(
        "OglModel",
        name="pathLine",
        position=targets.tolist(),
        edges=edges,
        color=[1.0, 0.3, 0.3, 0.9],
        lineWidth=2.0,
    )

    # Moving target sphere — MO + IdentityMapping so SOFA re-renders
    origin = np.zeros(3)
    verts, tris = _make_sphere_mesh(origin, radius=0.001)
    offsets = np.array(verts, dtype=float)
    init_verts = (offsets + targets[0]).tolist()
    sphere_node = path_node.addChild("MovingTarget")
    sphere_mo = sphere_node.addObject(
        "MechanicalObject", name="TargetMO", template="Vec3d",
        position=init_verts,
    )
    vis = sphere_node.addChild("Visual")
    vis.addObject(
        "OglModel", name="targetSphere",
        triangles=tris,
        color=[1.0, 0.2, 0.2, 0.9],
    )
    vis.addObject(
        "IdentityMapping",
        input="@../TargetMO", output="@targetSphere",
    )
    return sphere_mo, offsets


def _add_diag_spheres(root):
    """Add two diagnostic spheres for MPPI +rot/-rot weighted tip predictions."""
    origin = np.zeros(3)
    verts, tris = _make_sphere_mesh(origin, radius=0.0008)
    offsets = np.array(verts, dtype=float)
    init_verts = offsets.tolist()

    diag_node = root.addChild("DiagSpheres")

    # +rot sphere (green)
    pos_node = diag_node.addChild("PosRot")
    pos_mo = pos_node.addObject(
        "MechanicalObject", name="PosRotMO", template="Vec3d",
        position=init_verts,
    )
    pos_vis = pos_node.addChild("Visual")
    pos_vis.addObject(
        "OglModel", name="posRotSphere", triangles=tris,
        color=[0.2, 1.0, 0.2, 0.9],
    )
    pos_vis.addObject(
        "IdentityMapping", input="@../PosRotMO", output="@posRotSphere",
    )

    # -rot sphere (blue)
    neg_node = diag_node.addChild("NegRot")
    neg_mo = neg_node.addObject(
        "MechanicalObject", name="NegRotMO", template="Vec3d",
        position=init_verts,
    )
    neg_vis = neg_node.addChild("Visual")
    neg_vis.addObject(
        "OglModel", name="negRotSphere", triangles=tris,
        color=[0.2, 0.2, 1.0, 0.9],
    )
    neg_vis.addObject(
        "IdentityMapping", input="@../NegRotMO", output="@negRotSphere",
    )

    return pos_mo, neg_mo, offsets


# ── Headless runner ──────────────────────────────────────────────────

def _run_one_scene(config: dict, scene_dict: dict, scene_idx: int,
                   total_scenes: int, output_dir: str = ""):
    """Run one scene headless. Returns (n_steps, wall_time)."""
    sim_cfg = config.get("simulation", {})
    duration = float(sim_cfg.get("duration", 30.0))
    dt = float(sim_cfg.get("dt", 0.01))

    root = Sofa.Core.Node("root")
    createScene(root, headless=True, config=config,
                scene_dict=scene_dict, scene_index=scene_idx,
                output_dir=output_dir)
    Sofa.Simulation.init(root)

    ctrl_type = config.get("controller", {}).get("type", "ctrl")
    print(f"\n[feedback_control] Scene {scene_idx + 1}/{total_scenes}: "
          f"controller={ctrl_type}, duration={duration:.1f}s")

    fb = root.getObject("FeedbackController")
    step = 0
    t0 = time.time()

    while not fb.done:
        Sofa.Simulation.animate(root, dt)
        step += 1
        if step % 500 == 0:
            elapsed = time.time() - t0
            print(f"  step {step}, sim_time={step * dt:.2f}s, "
                  f"wall_time={elapsed:.1f}s", flush=True)

    elapsed = time.time() - t0
    print(f"  Done: {step} steps in {elapsed:.1f}s "
          f"({step / elapsed:.0f} steps/s)")

    Sofa.Simulation.unload(root)
    return step, elapsed


def run_headless(config_path: str = None, output_dir: str = "",
                 scene_indices=None):
    """Run all scenes from config sequentially."""
    config = _load_config(config_path)
    scenes = [_normalize_scene(s) for s in config.get("scenes", [{}])]

    if scene_indices is not None:
        selected = [(i, scenes[i]) for i in scene_indices if i < len(scenes)]
    else:
        selected = list(enumerate(scenes))

    print(f"[feedback_control] {len(selected)} scene(s) to run")

    msg_handler = SofaMessageHandler(print_info=False)
    total_steps = 0
    total_time = 0.0

    with msg_handler:
        for i, scene_dict in selected:
            steps, elapsed = _run_one_scene(
                config, scene_dict, i, len(scenes),
                output_dir=output_dir)
            total_steps += steps
            total_time += elapsed

    print(f"\n[feedback_control] All done: {total_steps} total steps "
          f"in {total_time:.1f}s across {len(selected)} scene(s)")


# ── Entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feedback control with closed-loop controller")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to feedback control YAML config")
    parser.add_argument("--scene-idx", type=int, nargs="+", default=None,
                        help="Specific scene indices to run (default: all)")
    parser.add_argument("--output-dir", type=str, default="",
                        help="Output directory for H5 files")
    args = parser.parse_args()

    if args.config:
        os.environ["FEEDBACK_CONFIG"] = args.config

    run_headless(config_path=args.config,
                 output_dir=args.output_dir,
                 scene_indices=args.scene_idx)
