import os
import sys
import yaml

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR = os.path.normpath(os.path.join(_THIS_DIR, ".."))
_WORKSPACE = os.path.normpath(os.path.join(_SIM_DIR, ".."))

# _SIM_DIR must be first so `utils.scene` resolves to simulation/utils/,
# not state_estimation/utils.py.
if _SIM_DIR in sys.path:
    sys.path.remove(_SIM_DIR)
sys.path.insert(0, _SIM_DIR)
if _WORKSPACE not in sys.path:
    sys.path.append(_WORKSPACE)

import Sofa
import Sofa.Core

from utils.scene import add_required_plugins, add_scene_utilities
from objects.fixed_rigid_body import PipelineModel, HeartInsideModel
from robots.catheter import CatheterRobot
from controllers.keyboard_controller import CatheterKeyboardController
from controllers.plot_controller import PlotController
from utils.plotter import DiagnosticPlotter
from utils.sofa_reader import SofaReader

_CONFIG_PATH = os.path.join(_SIM_DIR, "configs", "catheter_ablation.yaml")


def createScene(root: Sofa.Core.Node) -> Sofa.Core.Node:
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    root.gravity = [0.0, 0.0, 0.0]
    root.dt = 0.01

    add_required_plugins(root)
    add_scene_utilities(root)

    cable_mode = cfg.get("actuation", {}).get("cable_mode", "force")
    n_cables = len(cfg.get("actuation", {}).get("cable_locations", [[0, 0]]))
    rod_cfg = cfg.get("rod", {})

    env = HeartInsideModel(root)
    robot = CatheterRobot(root, config_path=_CONFIG_PATH, cable_mode=cable_mode)

    contact_listener = root.addObject(
        "ContactListener",
        name="contactStats",
        collisionModel1=env.triangle_collision_model_path,
        collisionModel2=robot.point_collision_model_path,
    )

    root.addObject(
        CatheterKeyboardController(
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
    )

    # -- SofaReader + Diagnostic plotter --
    constraint_solver = root.getObject("ConstraintSolver")
    n_nodes = int(rod_cfg.get("n_frames", 33))
    n_sections = int(rod_cfg.get("n_sections", 32))

    reader = SofaReader(
        prefab=robot._prefab,
        base_mo=robot.base_mo,
        cable_constraints=robot.cable_constraints,
        prefab_rotation_offset=robot.prefab_rotation_offset,
        cable_mode=cable_mode,
        constraint_solver=constraint_solver,
        contact_listener=contact_listener,
    )

    plotter = DiagnosticPlotter(
        n_nodes=n_nodes,
        n_sections=n_sections,
        rod_length=float(rod_cfg.get("length", 0.16)),
        panels=("base_translation", "base_rotation",
                "tendon_force", "contact_force"),
        window_seconds=10.0,
        dt=float(root.dt.value),
        n_cables=n_cables,
        title="Simulation — Diagnostics",
        size=(1300, 600),
    )

    root.addObject(
        PlotController(
            name="PlotController",
            plotter=plotter,
            reader=reader,
            n_nodes=n_nodes,
            base_home_position=robot.base_position,
        )
    )

    return root
