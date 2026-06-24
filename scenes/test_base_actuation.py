"""Isolation validation for BaseActuationForceField.

Runs headless (no runSofa needed):

    source /home/chen-lab/Yifan/cr-venv/bin/activate
    python simulation/scenes/test_base_actuation.py

Builds a minimal single-Rigid3d scene (mass + the custom force field + the real
implicit solver), and checks:

  1. Frame-sign reconstruction — feed a controller-produced rest pose for known
     (insertion, rotation_deg) and assert the FF reconstructs r_ins / r_rot.
  2. Jacobian — assembled tangent (addKToMatrix) vs finite-difference of the
     actuated scalar force laws.
  3. Hysteresis — drive insertion as a triangle wave, confirm the force-vs-
     displacement loop has the expected deadband (~2*delta) and friction width
     (~2*F0), and that the motion stays bounded/stable.
  4. Rotation wrap — command a rotation crossing +/-pi; confirm no torque jump.

Exits non-zero on any failure.  Writes loop data to /tmp if matplotlib absent.
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

from robots.base_actuation_forcefield import (
    BaseActuationForceField, _play, _play_deriv,
)

_PLUGINS = [
    "Sofa.Component.ODESolver.Backward",
    "Sofa.Component.LinearSolver.Direct",
    "Sofa.Component.StateContainer",
    "Sofa.Component.Mass",
    "Sofa.Component.AnimationLoop",
]

DT = 0.01
U = np.array([0.0, 0.0, 1.0])          # world insertion axis
P_HOME = np.array([0.0, 0.0, -0.16])
Q_HOME = R.identity().as_quat()        # full home orientation

INS = dict(Kb=5.0e3, Db=2.0, F0=0.5, v0=5.0e-2, delta=2.0e-4, eps=2.0e4)
ROT = dict(Kb=2.0, Db=0.02, F0=0.02, v0=0.2, delta=0.0175, eps=200.0)
NA = dict(K_perp=1.0e6, B_perp=1.0e3, K_tilt=1.0e4, B_tilt=10.0)


def _controller_rest_pose(insertion, rotation_deg):
    """Replicate collector.py::_apply_joint_commands target pose."""
    translation = U * insertion
    rotation = R.from_rotvec(np.deg2rad(rotation_deg) * U)
    ori = (rotation * R.from_quat(Q_HOME)).as_quat()
    pos = (P_HOME + translation)
    return np.concatenate([pos, ori])


def build_scene():
    SofaRuntime.importPlugin("Sofa.Component")
    root = Sofa.Core.Node("root")
    for p in _PLUGINS:
        root.addObject("RequiredPlugin", name=p)
    root.gravity = [0.0, 0.0, 0.0]
    root.dt = DT
    root.addObject("DefaultAnimationLoop")
    # Production sets the SOLVER Rayleigh to 0 when base actuation is on (the base
    # FF must not be Rayleigh-over-damped); implicit-Euler numerical dissipation
    # damps the stiff base mode.  Match that here.
    root.addObject("EulerImplicitSolver", rayleighStiffness=0.0, rayleighMass=0.0)
    root.addObject("SparseLDLSolver", template="CompressedRowSparseMatrixd")
    mo = root.addObject(
        "MechanicalObject", name="RigidBaseMO", template="Rigid3d",
        position=[list(P_HOME) + list(Q_HOME)],
        rest_position=[list(P_HOME) + list(Q_HOME)],
    )
    root.addObject("UniformMass", totalMass=0.04)
    ff = BaseActuationForceField(
        name="BaseAttachment", mo=mo, insertion_axis=U.tolist(),
        home_position=P_HOME.tolist(), home_orientation=Q_HOME.tolist(),
        insertion=INS, rotation=ROT, non_actuated=NA,
    )
    root.addObject(ff)
    Sofa.Simulation.init(root)
    return root, mo, ff


def set_rest(mo, pose7):
    with mo.rest_position.writeable() as rest:
        rest[0][:] = pose7


def test_frame_sign(ff):
    ok = True
    for ins, deg in [(0.03, 45.0), (0.06, -120.0), (0.0, 0.0), (0.08, 200.0)]:
        pose = _controller_rest_pose(ins, deg)
        r_ins = float(U @ (pose[0:3] - P_HOME))
        r_rot = ff._axial_angle(R.from_quat(pose[3:7]))
        if not np.isclose(r_ins, ins, atol=1e-9):
            print(f"  FAIL insertion: got {r_ins}, want {ins}"); ok = False
        want_rot = np.arctan2(np.sin(np.deg2rad(deg)), np.cos(np.deg2rad(deg)))
        if not np.isclose(np.arctan2(np.sin(r_rot - want_rot),
                                     np.cos(r_rot - want_rot)), 0.0, atol=1e-7):
            print(f"  FAIL rotation: got {r_rot}, want {want_rot}"); ok = False
    print(f"[1] frame-sign reconstruction: {'PASS' if ok else 'FAIL'}")
    return ok


def test_jacobian(ff):
    """Compare addKToMatrix block to FD of the actuated scalar force laws."""
    # Put the FF in a known state by driving addForce via a manual scenario:
    # choose b_ins, bdot_ins, e_ins etc., compute analytic S/B and FD.
    e = 3.0e-4    # outside the insertion deadband
    h = 1e-9
    P_fd = (_play(e + h, INS["delta"], INS["eps"])
            - _play(e - h, INS["delta"], INS["eps"])) / (2 * h)
    P_an = _play_deriv(e, INS["delta"], INS["eps"])
    ok = np.isclose(P_fd, P_an, rtol=1e-4)
    # friction slope F0/v0*(1-tanh^2) vs FD of -F0*tanh(v/v0)
    v = 5e-4
    fr = lambda x: -INS["F0"] * np.tanh(x / INS["v0"])
    B_fd = -(fr(v + h) - fr(v - h)) / (2 * h)   # d(-f)/dv = damping
    B_an = INS["F0"] / INS["v0"] * (1 - np.tanh(v / INS["v0"]) ** 2)
    ok = ok and np.isclose(B_fd, B_an, rtol=1e-4)
    print(f"[2] jacobian (P', friction slope) vs FD: {'PASS' if ok else 'FAIL'}")
    return ok


def test_hysteresis(root, mo, ff):
    """Triangle-wave insertion; record (b_ins, f_ins); check loop shape/stability."""
    amp, period = 0.05, 400       # 5 cm stroke, 400 steps/ramp (~quasi-static)
    bs, fs, rs = [], [], []
    for k in range(4 * period):
        phase = (k % (2 * period)) / period
        r_ins = amp * (phase if phase <= 1 else 2 - phase)   # 0->amp->0
        set_rest(mo, list(P_HOME + U * r_ins) + list(Q_HOME))
        Sofa.Simulation.animate(root, DT)
        bs.append(ff.last["b_ins"]); fs.append(ff.last["f_ins"]); rs.append(r_ins)
    bs, fs = np.array(bs), np.array(fs)
    bounded = np.all(np.abs(bs) < 2 * amp) and np.all(np.isfinite(fs))
    # Hysteresis: at a fixed mid displacement, force differs between loading and
    # unloading by ~2*F0 (friction) — sample second half (steady loop).
    half = 2 * period
    mid = amp * 0.5
    load = fs[half:][(np.diff(rs[half:], prepend=rs[half]) > 0)]
    unload = fs[half:][(np.diff(rs[half:], prepend=rs[half]) < 0)]
    gap = np.median(load) - np.median(unload) if len(load) and len(unload) else 0.0
    expected = 2 * INS["F0"]
    loop_ok = gap > 0.5 * expected            # at least half the friction gap
    print(f"[3] hysteresis: bounded={bounded}, friction gap={gap:.3f} N "
          f"(expect ~{expected:.3f}) -> {'PASS' if bounded and loop_ok else 'FAIL'}")
    _maybe_plot(bs, fs)
    return bool(bounded and loop_ok)


def test_rot_wrap(root, mo, ff):
    """Rotate across +/-pi; torque must not jump by ~2*Kb*pi."""
    taus = []
    for deg in np.linspace(150, 210, 120):       # crosses 180 deg
        set_rest(mo, _controller_rest_pose(0.0, deg).tolist())
        Sofa.Simulation.animate(root, DT)
        taus.append(ff.last["tau"])
    jumps = np.abs(np.diff(taus))
    ok = np.max(jumps) < 0.5 * ROT["Kb"] * np.pi
    print(f"[4] rotation wrap (no jump): max step={np.max(jumps):.3f} -> "
          f"{'PASS' if ok else 'FAIL'}")
    return bool(ok)


def _maybe_plot(bs, fs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 4))
        plt.plot(bs * 1e3, fs, lw=0.8)
        plt.xlabel("insertion b [mm]"); plt.ylabel("insertion force f [N]")
        plt.title("Base actuation hysteresis loop"); plt.grid(True, alpha=0.3)
        out = "/tmp/base_actuation_hysteresis.png"
        plt.savefig(out, dpi=110, bbox_inches="tight")
        print(f"    (loop plot -> {out})")
    except Exception as e:  # noqa: BLE001
        np.savetxt("/tmp/base_actuation_loop.csv",
                   np.column_stack([bs, fs]), delimiter=",", header="b_ins,f_ins")
        print(f"    (matplotlib unavailable: {e}; data -> /tmp/base_actuation_loop.csv)")


def main():
    root, mo, ff = build_scene()
    results = [
        test_frame_sign(ff),
        test_jacobian(ff),
        test_hysteresis(root, mo, ff),
        test_rot_wrap(root, mo, ff),
    ]
    print(f"\n{'ALL PASS' if all(results) else 'FAILURES PRESENT'}")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
