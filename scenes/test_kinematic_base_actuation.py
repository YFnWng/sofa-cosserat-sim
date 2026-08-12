"""Headless checks for the rigid-play, position-controlled base mode.

Run with::

    source /home/chen-lab/Yifan/cr-venv/bin/activate
    python simulation/scenes/test_kinematic_base_actuation.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

_SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)

import Sofa
import Sofa.Core
import Sofa.Simulation
import SofaRuntime

from robots.kinematic_base_actuation import (
    KinematicPlayBaseActuator,
    play_update,
)


def test_play() -> None:
    value = 0.0
    values = []
    for command in [0.0, 0.5, 1.5, 2.0, 0.5, -0.5]:
        value = play_update(value, command, 1.0)
        values.append(value)
    expected = [0.0, 0.0, 0.5, 1.0, 1.0, 0.5]
    np.testing.assert_allclose(values, expected)


def build_scene():
    SofaRuntime.importPlugin("Sofa.Component")
    root = Sofa.Core.Node("root")
    root.dt = 0.01
    root.gravity = [0.0, 0.0, 0.0]
    root.addObject("FreeMotionAnimationLoop")
    root.addObject(
        "BlockGaussSeidelConstraintSolver",
        tolerance=1e-12,
        maxIterations=1000,
        computeConstraintForces=True,
    )

    # No ODE solver on the target: its old pose and step velocity are inputs.
    target = root.addChild("target")
    target_mo = target.addObject(
        "MechanicalObject", name="targetMO", template="Rigid3d",
        position=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
    )

    dynamics = root.addChild("dynamics")
    dynamics.addObject("EulerImplicitSolver")
    dynamics.addObject("SparseLDLSolver", template="CompressedRowSparseMatrixd")
    dynamics.addObject("GenericConstraintCorrection")
    base = dynamics.addChild("base")
    base_mo = base.addObject(
        "MechanicalObject", name="baseMO", template="Rigid3d",
        position=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
    )
    base.addObject("UniformMass", totalMass=1.0)
    base.addObject(
        "BilateralLagrangianConstraint", template="Rigid3d",
        object1="@baseMO", object2=target_mo.getLinkPath(),
        first_point=[0], second_point=[0],
    )
    driver = KinematicPlayBaseActuator(
        target_mo=target_mo,
        insertion_axis=[0.0, 0.0, 1.0],
        home_position=[0.0, 0.0, 0.0],
        home_orientation=[0.0, 0.0, 0.0, 1.0],
        insertion_deadzone=1e-3,
        rotation_deadzone=0.1,
    )
    Sofa.Simulation.init(root)
    return root, base_mo, driver


def test_constraint_tracking() -> None:
    root, base, driver = build_scene()
    commands = [(0.0, 0.0), (0.005, 0.0), (0.010, 20.0), (0.003, 20.0)]
    previous_velocity = np.zeros(2)
    for insertion, rotation_deg in commands:
        driver.set_command(insertion, rotation_deg, 0.01)
        expected = driver.last["output"].copy()
        expected_acceleration = (
            driver.last["velocity"] - previous_velocity) / 0.01
        np.testing.assert_allclose(
            driver.last["acceleration"], expected_acceleration)
        previous_velocity = driver.last["velocity"].copy()

        Sofa.Simulation.animate(root, 0.01)
        pose = np.asarray(base.position.value)[0]
        actual_insertion = pose[2]
        actual_rotation = R.from_quat(pose[3:7]).as_rotvec()[2]
        np.testing.assert_allclose(
            [actual_insertion, actual_rotation], expected, atol=2e-9)
        np.testing.assert_allclose(
            np.asarray(base.velocity.value)[0][[2, 5]],
            driver.last["velocity"], atol=2e-9)


def test_pi_crossing() -> None:
    """A wrapped +pi crossing must retain positive angular direction."""
    root, base, driver = build_scene()
    driver.rotation_deadzone = np.deg2rad(10.0)
    outputs = []
    rates = []
    for rotation_deg in [160.0, 170.0, 179.0, -179.0, -170.0, -160.0]:
        driver.set_command(0.0, rotation_deg, 0.01)
        outputs.append(float(driver.last["output"][1]))
        rates.append(float(driver.last["velocity"][1]))
        Sofa.Simulation.animate(root, 0.01)
        np.testing.assert_allclose(
            np.asarray(base.velocity.value)[0, 5], rates[-1], atol=2e-9)
    increments = np.diff(outputs)
    assert np.all(increments >= -1e-12), (outputs, rates)
    assert max(np.abs(increments)) < np.deg2rad(20.0), outputs


if __name__ == "__main__":
    test_play()
    test_constraint_tracking()
    test_pi_crossing()
    print("kinematic base actuation: PASS")
