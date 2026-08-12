from typing import Dict, Iterable, List, Sequence

import Sofa
import Sofa.Core

from cosserat.usefulFunctions import pluginList as _COSSERAT_PLUGIN_LIST  # type: ignore


BACKGROUND_COLOR = [1.0, 1.0, 1.0, 1.0]
DISPLAY_FLAGS = (
    "showVisualModels showBehaviorModels hideCollisionModels hideBoundingCollisionModels "
    "showForceFields hideInteractionForceFields hideWireframe showMechanicalMappings"
)

_BASE_PLUGINS = [
    "Sofa.Component.AnimationLoop",
    "Sofa.Component.Collision.Detection.Algorithm",
    "Sofa.Component.Collision.Detection.Intersection",
    "Sofa.Component.Collision.Geometry",
    "Sofa.Component.Collision.Response.Contact",
    "Sofa.Component.Constraint.Lagrangian.Correction",
    "Sofa.Component.Constraint.Lagrangian.Model",
    "Sofa.Component.Constraint.Lagrangian.Solver",
    "Sofa.Component.Constraint.Projective",
    "Sofa.Component.Controller",
    "Sofa.Component.Engine.Select",
    "Sofa.Component.IO.Mesh",
    "Sofa.Component.LinearSolver.Direct",
    "Sofa.Component.LinearSolver.Iterative",
    "Sofa.Component.Mass",
    "Sofa.Component.Mapping.Linear",
    "Sofa.Component.Mapping.NonLinear",
    "Sofa.Component.MechanicalLoad",
    "Sofa.Component.ODESolver.Backward",
    "Sofa.Component.StateContainer",
    "Sofa.Component.Topology.Container.Constant",
    "Sofa.Component.Topology.Container.Dynamic",
    "Sofa.Component.Topology.Mapping",
    "Sofa.Component.Visual",
    "Sofa.GL.Component.Rendering3D",
    "Cosserat",
    "SoftRobots",
]


def add_required_plugins(root: Sofa.Core.Node) -> None:
    config = root.addChild("Config")
    for plugin in _unique(_BASE_PLUGINS + list(_COSSERAT_PLUGIN_LIST)):
        config.addObject(
            "RequiredPlugin",
            name=f"plugin_{plugin.replace('.', '_')}",
            pluginName=plugin,
            printLog=False,
        )


def add_scene_utilities(root: Sofa.Core.Node) -> None:
    root.addObject("VisualStyle", displayFlags=DISPLAY_FLAGS)
    root.addObject("BackgroundSetting", color=BACKGROUND_COLOR)
    root.addObject("OglSceneFrame", style="Arrows", alignment="TopRight")
    root.addObject("DefaultVisualManagerLoop")
    root.addObject("FreeMotionAnimationLoop")
    root.addObject("BlockGaussSeidelConstraintSolver", name="ConstraintSolver",
                   tolerance=1e-9, maxIterations=500,
                   computeConstraintForces=True)
    root.addObject("CollisionPipeline")
    root.addObject("BruteForceBroadPhase")
    root.addObject("BVHNarrowPhase")
    root.addObject(
        "RuleBasedContactManager",
        name="ContactManager",
        responseParams="mu=0.0",
        response="FrictionContactConstraint",
    )
    root.addObject(
        "LocalMinDistance",
        name="Proximity",
        alarmDistance=4.0e-3,    # m (was 4 mm)
        contactDistance=1.5e-3,  # m (was 1.5 mm)
        angleCone=0.05,
    )


def add_constant_forces(
    cosserat_frame_node,
    forces: Dict[int, Sequence[float]],
) -> List:
    """Apply ``ConstantForceField`` at given frame indices on a Rigid3d frame MO.

    Parameters
    ----------
    cosserat_frame_node:
        The SOFA node containing the ``FramesMO`` Rigid3d MechanicalObject
        (typically ``robot._prefab.cosseratFrame``).
    forces:
        ``{frame_index: [fx, fy, fz]}`` in world frame (N).  Torque
        components are set to zero.

    Returns
    -------
    list
        The created ``ConstantForceField`` SOFA objects.
    """
    ffs = []
    for idx, f in forces.items():
        ff = cosserat_frame_node.addObject(
            "ConstantForceField",
            name=f"ExtForce_{idx}",
            template="Rigid3d",
            indices=[idx],
            forces=[[float(f[0]), float(f[1]), float(f[2]), 0.0, 0.0, 0.0]],
            showArrowSize=0.01,
        )
        ffs.append(ff)
    return ffs


def _unique(values: Iterable[str]) -> List[str]:
    seen: set = set()
    ordered: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered
