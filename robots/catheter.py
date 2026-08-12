from __future__ import annotations

import os
import sys
import yaml
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

# Make simulation root importable regardless of how runSofa sets sys.path
_SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)

import Sofa
import Sofa.Core

from cosserat.CosseratBase import CosseratBase  # type: ignore
from actuators.cable import PullingCable  # type: ignore
from useful.params import (  # type: ignore
    BeamGeometryParameters,
    BeamPhysicsParametersNoInertia,
    Parameters,
)

from utils.cable_utils import compute_cable_points
from robots.base_actuation_forcefield import BaseActuationForceField
from robots.kinematic_base_actuation import KinematicPlayBaseActuator

_DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs", "catheter_ablation.yaml",
)


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class CatheterRobot:
    """Tendon-driven Cosserat rod catheter with one or more actuation cables.

    Parameters
    ----------
    root:
        The SOFA root node.
    config_path:
        Path to the catheter YAML config file.  Defaults to
        ``simulation/configs/catheter_ablation.yaml`` next to this file.
    cable_mode:
        ``"displacement"`` (default) or ``"force"`` — passed to
        ``PullingCable(valueType=...)``.

    After construction the following attributes are available for wiring up
    controllers and contact listeners:

    Attributes
    ----------
    base_mo : Sofa MechanicalObject
        Rigid3d base DOF node (RigidBaseMO).
    cable_constraints : list
        CableConstraint objects for each actuation cable (one per
        ``cable_locations`` entry in the YAML config).
    cable_constraint : Sofa object or None
        First cable constraint (backward-compat alias for single-cable scenes).
    base_position : list[float]
        Home position of the catheter base [x, y, z] (mm).
    base_orientation : list[float]
        Home orientation of the catheter base as quaternion [qx,qy,qz,qw].
    prefab_rotation_offset : list[float]
        Quaternion [qx,qy,qz,qw] of the rotation applied to align the Cosserat
        prefab local-X rod axis with the scene local-Z convention.
    insertion_direction : list[float]
        Unit vector along which the catheter is inserted (local frame).
    joint_rate : list[float]
        Rates for [insertion (mm/s), rotation (deg/s), cable (mm/s)].
    joint_upper_limits : list[float]
        Upper bounds for [insertion, rotation, cable].
    joint_lower_limits : list[float]
        Lower bounds for [insertion, rotation, cable].
    point_collision_model_path : str
        SOFA link path to the catheter PointCollisionModel.
    """

    def __init__(
        self,
        root: Sofa.Core.Node,
        config_path: str = _DEFAULT_CONFIG,
        cable_mode: str = "displacement",
    ) -> None:
        cfg = _load_config(config_path)
        rod_cfg = cfg.get("rod", {})
        act_cfg = cfg.get("actuation", {})

        (self._prefab, self._collision, self.cable_constraints,
         self.base_actuator) = _build(
            root, rod_cfg, act_cfg, cable_mode=cable_mode
        )
        self.base_mo = self._prefab.rigidBaseNode.RigidBaseMO  # type: ignore[attr-defined]

        self.base_position: List[float] = list(rod_cfg["base_position"])
        base_home_orientation = R.from_euler(
            "xyz",
            rod_cfg.get("base_orientation_euler_xyz_deg", [0.0, 0.0, 0.0]),
            degrees=True,
        )
        self.base_orientation: List[float] = base_home_orientation.as_quat().tolist()
        prefab_rotation = R.from_euler(
            "xyz",
            rod_cfg.get("prefab_rotation_euler_xyz_deg", [0.0, -90.0, 0.0]),
            degrees=True,
        )
        self.prefab_rotation_offset: List[float] = prefab_rotation.as_quat().tolist()
        self.insertion_direction: List[float] = list(
            act_cfg.get("insertion_direction", [0.0, 0.0, 1.0])
        )

        # Load mode-specific actuation limits from the matching sub-dict.
        mode_cfg = act_cfg.get(cable_mode, {})
        self.joint_rate: List[float] = [
            float(act_cfg.get("insertion_speed", 30.0e-3)),
            float(act_cfg.get("rotation_speed", 30.0)),
            float(mode_cfg.get("pull_increment", 3.0e-3)),
        ]
        self.joint_upper_limits: List[float] = [
            float(act_cfg.get("max_travel", 160.0e-3)),
            float(act_cfg.get("max_rotation", 180.0)),
            float(mode_cfg.get("pull_max", 30.0e-3)),
        ]
        self.joint_lower_limits: List[float] = [
            0.0,
            -float(act_cfg.get("max_rotation", 180.0)),
            float(mode_cfg.get("pull_min", 0.0)),
        ]

        # Joint semantics: name and type for each control DOF.
        # Types: "linear" (m), "angle_deg" (degrees), "force" (N), "displacement" (m)
        self.joint_names: List[str] = ["insertion", "rotation"]
        self.joint_types: List[str] = ["linear", "angle_deg"]
        for i in range(len(self.cable_constraints)):
            self.joint_names.append(f"cable_{i}")
            self.joint_types.append(cable_mode)  # "force" or "displacement"

    @property
    def cable_constraint(self):
        """First cable constraint — backward-compat alias for single-cable scenes."""
        return self.cable_constraints[0] if self.cable_constraints else None

    @property
    def point_collision_model_path(self) -> str:
        if self._collision is None:
            return ""
        return self._collision.PointCollisionModel.getLinkPath()


# ---------------------------------------------------------------------------
# Internal construction helpers
# ---------------------------------------------------------------------------

def _apply_variable_stiffness(prefab: CosseratBase, rod_cfg: dict) -> None:
    """Set per-section Young's modulus and Poisson's ratio if configured.

    The YAML config supports a ``stiffness_sections`` block::

        rod:
          stiffness_sections:
            node_indices:   [0, 16]        # section boundaries (ascending)
            young_modulus:  [2.0e9, 1.0e9] # E for each segment
            poisson_ratio:  [0.38]         # single value → uniform

    ``node_indices`` lists the section indices where properties change.
    Each segment spans from ``node_indices[i]`` to ``node_indices[i+1]-1``
    (the last segment runs to the end of the rod).

    A list of length 1 means that property is uniform across all sections.
    If ``stiffness_sections`` is absent, nothing is changed (uniform E/nu
    from the top-level ``young_modulus`` / ``poisson_ratio``).
    """
    ss = rod_cfg.get("stiffness_sections")
    if ss is None:
        return

    n_sections = int(rod_cfg.get("n_sections", 32))
    node_indices = ss.get("node_indices", [0])
    E_list = ss.get("young_modulus", [float(rod_cfg.get("young_modulus", 2.0e9))])
    nu_list = ss.get("poisson_ratio", [float(rod_cfg.get("poisson_ratio", 0.38))])

    # Expand single-element lists to uniform
    if len(E_list) == 1:
        E_list = E_list * len(node_indices)
    if len(nu_list) == 1:
        nu_list = nu_list * len(node_indices)

    if len(E_list) != len(node_indices) or len(nu_list) != len(node_indices):
        raise ValueError(
            f"stiffness_sections: young_modulus ({len(E_list)}) and "
            f"poisson_ratio ({len(nu_list)}) must match "
            f"node_indices ({len(node_indices)}) or have length 1."
        )

    # Build per-section arrays
    E_per_section = np.zeros(n_sections)
    nu_per_section = np.zeros(n_sections)
    for i, start_idx in enumerate(node_indices):
        end_idx = node_indices[i + 1] if i + 1 < len(node_indices) else n_sections
        E_per_section[start_idx:end_idx] = E_list[i]
        nu_per_section[start_idx:end_idx] = nu_list[i]

    # Access the BeamHookeLawForceField created by CosseratBase
    beam_ff = prefab.cosseratCoordinate.BeamHookeLawForceField  # type: ignore[attr-defined]
    beam_ff.findData("variantSections").value = True
    beam_ff.findData("youngModulusList").value = E_per_section.tolist()
    beam_ff.findData("poissonRatioList").value = nu_per_section.tolist()


def _strain_rayleigh_coefficients(beam_ff, beta: float):
    """Per-section DiagonalVelocityDamping coefs reproducing Rayleigh β·K on the rod.

    The beam force is ``f -= (K_section · strain) · L`` with the DIAGONAL section
    stiffness ``K_section = diag(K0, K1, K2)`` (strain order
    ``[torsion, bend_y, bend_z]``).  Rayleigh stiffness damping is ``C = β·K``, so
    the equivalent explicit velocity-damping coefficient for section ``i`` is
    ``c_i = β · diag(K_section_i) · L_i``.

    ``K_section`` is a private C++ member of BeamHookeLawForceField (rebuilt in
    ``reinit()``) — not exposed as Data or via any Python accessor — so we cannot
    read it directly.  Instead we mirror ``reinit()`` exactly, reading the beam's
    own Data for every branch (cross-section, variant/uniform sections, inertia
    params) so the result matches whatever stiffness the beam actually uses.

    Returns a list of 3-vectors (one per section), or ``None`` if β ≤ 0.
    """
    if beta <= 0:
        return None
    d = lambda n: beam_ff.findData(n).value
    lengths = np.asarray(d("length"), dtype=float)
    n_sec = len(lengths)
    if n_sec == 0:
        return None

    # Cross-section second moments / polar moment (mirrors reinit()).
    shape = d("crossSectionShape")
    shape = shape.getSelectedItem() if hasattr(shape, "getSelectedItem") else str(shape)
    if shape == "rectangular":
        Ly, Lz = float(d("lengthY")), float(d("lengthZ"))
        Iy = Ly * Lz**3 / 12.0
        Iz = Lz * Ly**3 / 12.0
    else:  # circular (default)
        r, r_in = float(d("radius")), float(d("innerRadius"))
        Iy = Iz = np.pi * (r**4 - r_in**4) / 4.0
    J = Iy + Iz

    def K_diag(E, nu):
        """diag(G·J, E·Iy, E·Iz) for one section."""
        G = E / (2.0 * (1.0 + nu))
        return np.array([G * J, E * Iy, E * Iz])

    if bool(d("variantSections")):
        E_list = np.asarray(d("youngModulusList"), dtype=float)
        nu_list = np.asarray(d("poissonRatioList"), dtype=float)
        K_per = [K_diag(E_list[i], nu_list[i]) for i in range(n_sec)]
    elif bool(d("useInertiaParams")):
        # Stiffness given directly as inertia params — read from Data, no recompute.
        K_uniform = np.array([float(d("GI")), float(d("EI")), float(d("EI"))])
        K_per = [K_uniform] * n_sec
    else:
        K_uniform = K_diag(float(d("youngModulus")), float(d("poissonRatio")))
        K_per = [K_uniform] * n_sec

    return [(beta * K_per[i] * lengths[i]).tolist() for i in range(n_sec)]


def _build(
    root: Sofa.Core.Node,
    rod_cfg: dict,
    act_cfg: dict,
    cable_mode: str = "displacement",
) -> Tuple[CosseratBase, Sofa.Core.Node, List[Optional[Sofa.Core.Object]]]:
    solver_node = root.addChild("CatheterSimulation")
    rayleigh_stiffness = float(rod_cfg.get("rayleigh_stiffness", rod_cfg.get("rayleigh", 0.05)))
    rayleigh_mass = float(rod_cfg.get("rayleigh_mass", 1e-3))
    # --- Rod damping & base decoupling -------------------------------------
    # The rod needs stiffness-proportional damping (β·K): it damps the high-
    # frequency strain modes (largely discretisation artefacts) for stable
    # implicit integration while leaving the slow physical bending mode lively
    # (ζ_i = (β/2)·ω_i grows with frequency).  A uniform velocity damping (c·I)
    # would instead over-damp the bending mode, so it is NOT a substitute.
    #
    # Physically, β·K is the discrete form of Kelvin–Voigt INTERNAL/material
    # damping (σ = E·ε + η·ε̇ ⇒ C = (η/E)·K, β = η/E = a material relaxation
    # time): dissipation that scales with each section's elastic stiffness.  It
    # is genuinely realistic for a polymer rod, BUT the current β value is a
    # NUMERICAL-STABILITY choice, not calibrated to the catheter's measured loss
    # tangent (a physical η/E would be much smaller).  External drag (fluid /
    # tissue resistance) is a *separate*, velocity-proportional mechanism (≈ a
    # uniform/mass-proportional term) not modelled here.
    # TODO(fidelity): if simulated rod transients must match a real catheter,
    # calibrate this into (i) a small internal β·K from material loss + (ii) a
    # uniform external-drag term sized to the medium (air vs blood/tissue),
    # rather than the single numerical β below.
    #
    # IMPLEMENTATION: rather than the solver's GLOBAL rayleighStiffness (which
    # would also damp the soft base actuator and bury its friction — β·Kb is
    # huge), we set the solver β to 0 and apply the rod's β·K damping as an
    # EXPLICIT, strain-local DiagonalVelocityDampingForceField with coefficients
    # = β·diag(K_section)·L (computed below).  That is mathematically identical
    # to solver Rayleigh on the rod but leaves the base Rayleigh-free, so the
    # base actuation's deadzone/friction is visible.
    base_act_enabled = bool(rod_cfg.get("base_actuation", {}).get("enabled", False))
    solver_rayleigh = 0.0 if base_act_enabled else rayleigh_stiffness
    import os
    # NOTE: compare to "1" — a bare os.environ.get() is truthy for ANY value
    # (including "0"), so COLLECT_DIAGNOSTIC_SOLVER=0 would otherwise still enable it.
    solver_type = ("DiagnosticEulerImplicitSolver"
                   if os.environ.get("COLLECT_DIAGNOSTIC_SOLVER", "0") == "1"
                   else "EulerImplicitSolver")
    solver_node.addObject(
        solver_type,
        rayleighStiffness=solver_rayleigh,
        rayleighMass=rayleigh_mass,
    )
    solver_node.addObject(
        "SparseLDLSolver",
        name="solver",
        template="CompressedRowSparseMatrixd",
    )
    solver_node.addObject("GenericConstraintCorrection")

    params = _build_params(rod_cfg)

    base_pos = rod_cfg.get("base_position", [0.0, 0.0, -160.0e-3])
    base_orient = R.from_euler(
        "xyz",
        rod_cfg.get("base_orientation_euler_xyz_deg", [0.0, 0.0, 0.0]),
        degrees=True,
    )
    prefab_rot = R.from_euler(
        "xyz",
        rod_cfg.get("prefab_rotation_euler_xyz_deg", [0.0, -90.0, 0.0]),
        degrees=True,
    )
    # CosseratBase rotation= expects Euler angles in degrees (3 elements),
    # not a quaternion.  Passing a 4-element quaternion corrupts the heap.
    prefab_base_rotation = (base_orient * prefab_rot).as_euler("xyz", degrees=True).tolist()
    prefab = CosseratBase(
        parent=solver_node,
        params=params,
        name="catheter",
        translation=base_pos,
        rotation=prefab_base_rotation,
    )

    _apply_variable_stiffness(prefab, rod_cfg)

    # Force-field rayleighStiffness (β_ff): kept 0.  (β_ff would enter the system
    # matrix A but NOT the RHS b — bFactor=0 in the RHS assembly — so it provides
    # no real energy dissipation and the rod rings; not a usable damping source.)
    beam_ff = prefab.cosseratCoordinate.BeamHookeLawForceField  # type: ignore[attr-defined]
    beam_ff.findData("rayleighStiffness").value = float(
        rod_cfg.get("rayleigh_stiffness_ff", 0.0))

    base_act_cfg = rod_cfg.get("base_actuation", {})
    base_actuator = None
    if base_act_cfg.get("enabled", False):
        # Frame conventions mirror the controllers (see collector.py):
        #   u (world insertion axis) = direction_local @ base_orient.T
        #   q_home_full              = base_orient * prefab_rot
        direction_local = np.asarray(
            act_cfg.get("insertion_direction", [0.0, 0.0, 1.0]), dtype=float)
        u_world = direction_local @ base_orient.as_matrix().T
        q_home_full = (base_orient * prefab_rot).as_quat().tolist()
        base_mode = base_act_cfg.get("mode", "spring_deadzone")
        if base_mode == "spring_deadzone":
            # Backward-compatible compliant penalty actuator.
            prefab.rigidBaseNode.addObject(  # type: ignore[attr-defined]
                BaseActuationForceField(
                    name="BaseAttachment",
                    mo=prefab.rigidBaseNode.RigidBaseMO,  # type: ignore[attr-defined]
                    insertion_axis=u_world.tolist(),
                    home_position=list(base_pos),
                    home_orientation=q_home_full,
                    insertion=base_act_cfg["insertion"],
                    rotation=base_act_cfg["rotation"],
                    non_actuated=base_act_cfg["non_actuated"],
                )
            )
        elif base_mode == "kinematic_play":
            # The target has no ODE solver: its position and velocity are known
            # inputs.  The bilateral constraint corrects the *dynamic* base via
            # the full coupled compliance, retaining base/strain inertial coupling.
            target_node = root.addChild("BaseCommandTarget")
            target_mo = target_node.addObject(
                "MechanicalObject",
                name="BaseCommandMO",
                template="Rigid3d",
                position=[[*base_pos, *q_home_full]],
                rest_position=[[*base_pos, *q_home_full]],
            )
            base_constraint = prefab.rigidBaseNode.addObject(  # type: ignore[attr-defined]
                "BilateralLagrangianConstraint",
                name="BaseAttachment",
                template="Rigid3d",
                object1="@RigidBaseMO",
                object2=target_mo.getLinkPath(),
                first_point=[0],
                second_point=[0],
                keepOrientationDifference=False,
            )
            base_actuator = KinematicPlayBaseActuator(
                target_mo=target_mo,
                constraint=base_constraint,
                insertion_axis=u_world,
                home_position=base_pos,
                home_orientation=q_home_full,
                insertion_deadzone=float(base_act_cfg["insertion"]["delta"]),
                rotation_deadzone=float(base_act_cfg["rotation"]["delta"]),
                friction_plateau=[
                    float(base_act_cfg["insertion"].get("F0", 0.0)),
                    float(base_act_cfg["rotation"].get("F0", 0.0)),
                ],
                friction_speed=[
                    float(base_act_cfg["insertion"].get("v0", 1.0)),
                    float(base_act_cfg["rotation"].get("v0", 1.0)),
                ],
            )
        else:
            raise ValueError(
                f"unknown rod.base_actuation.mode={base_mode!r}; expected "
                "'spring_deadzone' or 'kinematic_play'")
    else:
        # Fallback: legacy stiff spring (A/B regression baseline).
        prefab.rigidBaseNode.addObject(  # type: ignore[attr-defined]
            "RestShapeSpringsForceField",
            name="BaseAttachment",
            stiffness=1e10,
            angularStiffness=1e10,
            external_points=0,
            points=0,
            template="Rigid3d",
        )
    # Rod damping (strain-local, leaves the base untouched).
    if base_act_enabled:
        # Stiffness-proportional Rayleigh-equivalent: c_i = β·diag(K_section_i)·L_i,
        # matching the beam exactly (force is K_section·strain·L) so this reproduces
        # solver Rayleigh on the rod while the solver β stays 0 for the base.  K is
        # diagonal [G·J, E·Iy, E·Iz] so DiagonalVelocityDampingForceField (per-DOF,
        # per-component) gives an EXACT match; a scalar UniformVelocityDamping (c·I)
        # would over-damp the bending mode (see header note on damping physics).
        # implicit=… is unconditional for DiagonalVelocityDamping (always builds
        # the df/dv matrix term), unlike UniformVelocityDamping.
        coefs = _strain_rayleigh_coefficients(beam_ff, rayleigh_stiffness)
        if coefs is not None:
            prefab.cosseratCoordinate.addObject(  # type: ignore[attr-defined]
                "DiagonalVelocityDampingForceField",
                name="StrainDamping",
                dampingCoefficient=coefs,
            )
    else:
        # Legacy path: solver Rayleigh damps the rod; optional extra uniform damping.
        rod_damping = float(rod_cfg.get("strain_damping", 0.0))
        if rod_damping > 0:
            prefab.cosseratCoordinate.addObject(  # type: ignore[attr-defined]
                "UniformVelocityDampingForceField",
                name="StrainDamping",
                dampingCoefficient=rod_damping,
                implicit=True,
            )

    # Scale visual aids to ~10% of rod length for visibility
    _vis_scale = params.beam_geo_params.beam_length * 0.005
    prefab.cosseratFrame.FramesMO.showObject = True  # type: ignore[attr-defined]
    prefab.cosseratFrame.FramesMO.showObjectScale = _vis_scale
    mass_obj = prefab.cosseratFrame.getObject("UniformMass")
    if mass_obj is not None:
        mass_obj.showAxisSizeFactor = _vis_scale  # type: ignore[attr-defined]
    prefab.cosseratCoordinate.cosseratCoordinateMO.showObject = False  # type: ignore[attr-defined]
    prefab.rigidBaseNode.RigidBaseMO.showObject = True  # type: ignore[attr-defined]
    prefab.rigidBaseNode.RigidBaseMO.showObjectScale = _vis_scale  # type: ignore[attr-defined]

    collision = prefab.addCollisionModel()
    if hasattr(collision, "CollisionDOFs"):
        collision.CollisionDOFs.showObject = False  # type: ignore[attr-defined]

    cable_constraints = _add_cables(prefab, act_cfg, cable_mode=cable_mode)
    return prefab, collision, cable_constraints, base_actuator


def _add_cables(
    prefab: CosseratBase,
    act_cfg: dict,
    cable_mode: str = "displacement",
) -> List[Optional[Sofa.Core.Object]]:
    """Create one PullingCable per entry in ``cable_locations`` and return their constraints."""
    cable_locations = act_cfg.get("cable_locations", [[1.4e-3, 0.0]])
    cable_point_count = int(act_cfg.get("cable_point_count", 16))
    constraints = []
    for i, loc in enumerate(cable_locations):
        cc = _add_single_cable(prefab, loc, cable_point_count, cable_mode, index=i)
        constraints.append(cc)
    return constraints


def _add_single_cable(
    prefab: CosseratBase,
    cable_location: list,
    cable_point_count: int,
    cable_mode: str = "displacement",
    index: int = 0,
) -> Optional[Sofa.Core.Object]:
    """Add one PullingCable at *cable_location* [x, y] in the cross-section."""
    frame_states = np.asarray(prefab.frames3D, dtype=float)
    cable_points = compute_cable_points(frame_states, cable_point_count, cable_location)
    if cable_points.shape[0] < 2:
        return None

    frame_node = prefab.rigidBaseNode.cosseratInSofaFrameNode  # type: ignore[attr-defined]
    # Each cable lives in its own uniquely-named child node; object names inside
    # each attachment node are kept consistent with the original single-cable setup
    # so SOFA component lookups (SkinningMapping, PullingCable) work correctly.
    attachment = frame_node.addChild(f"CableAttachment_{index}")
    guide_positions = cable_points.tolist()
    attachment.addObject(
        "MechanicalObject",
        name="CableGuideMO",
        template="Vec3d",
        position=guide_positions,
        showObject=True,
        showIndices=False,
        showObjectScale=0.001,
        showColor=[0.2, 0.85, 0.3, 1.0],   # green
    )
    # Draw cable as a connected line
    cable_edges = [[k, k + 1] for k in range(len(guide_positions) - 1)]
    attachment.addObject(
        "EdgeSetTopologyContainer",
        name="CableEdges",
        edges=cable_edges,
    )
    cable_visual = attachment.addChild("CableVisual")
    cable_visual.addObject(
        "OglModel", name="CableLine",
        color=[0.2, 0.85, 0.3, 1.0],
        edges=cable_edges,
    )
    cable_visual.addObject(
        "IdentityMapping",
        input="@../CableGuideMO", output="@CableLine",
    )
    attachment.addObject("SkinningMapping", nbRef="1", name="CableSkinning")

    cable = PullingCable(
        attachedTo=attachment,
        name="ActuationCable",
        cableGeometry=guide_positions,
        valueType=cable_mode,
    )

    cable_mo = getattr(cable, "MechanicalObject", None)
    if cable_mo is not None:
        cable_mo.showObject = False
        cable_mo.showIndices = False
        cable_mo.showObjectScale = 0.001
        cable_mo.showColor = [0.2, 0.85, 0.3, 1.0]

    return getattr(cable, "CableConstraint", None)


def _build_params(rod_cfg: dict) -> Parameters:
    length = float(rod_cfg.get("length", 160.0e-3))
    geometry = BeamGeometryParameters(
        beam_length=length,
        nb_section=int(rod_cfg.get("n_sections", 32)),
        nb_frames=int(rod_cfg.get("n_frames", 33)),
        build_collision_model=1,
    )
    physics = BeamPhysicsParametersNoInertia(
        beam_mass=float(rod_cfg.get("mass", 0.04)),
        young_modulus=float(rod_cfg.get("young_modulus", 8.0e5)),
        poisson_ratio=float(rod_cfg.get("poisson_ratio", 0.38)),
        beam_radius=float(rod_cfg.get("beam_radius", 1.45e-3)),
        beam_length=length,
    )
    params = Parameters(beam_geo_params=geometry, beam_physics_params=physics)
    params.simu_params.rayleigh_stiffness = float(rod_cfg.get("rayleigh_stiffness", rod_cfg.get("rayleigh", 0.05)))
    params.simu_params.rayleigh_mass = float(rod_cfg.get("rayleigh_mass", 1e-3))
    return params
