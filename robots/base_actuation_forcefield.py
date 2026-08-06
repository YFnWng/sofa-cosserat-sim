"""Custom SOFA force field for realistic catheter base actuation.

Replaces the unrealistically stiff ``RestShapeSpringsForceField`` (K=1e10) that
previously drove the catheter base.  The two *actuated* base DOFs — insertion
(translation along the world insertion axis ``u``) and axial rotation (about
``u``) — are driven by a soft, structured law that models a real
insertion/rotation drive with backlash (a smooth deadzone) and friction::

    f = Kb * P(r - b)  -  Db * b_dot  -  F0 * tanh(b_dot / v0)

    P(e)  = softplus(e - delta) - softplus(-e - delta)      # smooth play/deadzone
    P'(e) = sigmoid(eps*(e - delta)) + sigmoid(eps*(-e - delta))

where ``b`` is the actual DOF value, ``r`` the commanded rest value (read from
the MechanicalObject's ``rest_position`` — exactly what the controllers already
write), ``delta`` the half-deadzone (backlash), ``eps`` the softplus sharpness,
and ``F0``/``v0`` the friction plateau/smoothing speed.  Rotation error is
atan2-wrapped.

The four *non-actuated* DOFs (lateral translation perpendicular to ``u`` and
tilt) are held rigid by a moderately stiff linear spring on the orthogonal
complement, modelling a rigid actuator carriage.

This mirrors the structured base force in the learned LNN model
(``cr_meta_lnn/networks/lagrangian.py`` PotentialModule/DampingModule), so
simulation and model share one functional form.

Integration hooks: ``addForce`` (the force), ``addKToMatrix`` (the assembled
tangent the direct ``SparseLDLSolver`` and ``GenericConstraintCorrection``
consume — the load-bearing path), and ``addDForce`` (the matrix-free tangent).

Frame conventions are sourced identically to the controllers
(``simulation/data_collection/collector.py``):
    u (world insertion axis) = direction_local @ base_home_orientation.T
    q_home_full              = base_home_orientation * prefab_rotation_offset
    target pose written to rest_position is
        pos = home_position + u * insertion
        ori = from_rotvec(rotation * u) * q_home_full
so insertion = u . (pos - home_position) and the axial angle is
    u . (R_current * q_home_full^{-1}).as_rotvec().
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

import Sofa
import Sofa.Core


# --------------------------------------------------------------------------
# Numerically-stable scalar helpers (numpy)
# --------------------------------------------------------------------------
def _softplus(z: np.ndarray, eps: float) -> np.ndarray:
    """(1/eps) * log(1 + exp(eps*z)), stable for large |eps*z|."""
    ez = eps * z
    return np.logaddexp(0.0, ez) / eps


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def _play(e: np.ndarray, delta: float, eps: float) -> np.ndarray:
    """Smooth deadzone / play operator P(e)."""
    return _softplus(e - delta, eps) - _softplus(-e - delta, eps)


def _play_deriv(e: np.ndarray, delta: float, eps: float) -> np.ndarray:
    """P'(e) -> ~1 outside the +/-delta band, ~0 inside."""
    return _sigmoid(eps * (e - delta)) + _sigmoid(eps * (-e - delta))


def _wrap(angle: float) -> float:
    """atan2 wrap to (-pi, pi]."""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


class BaseActuationForceField(Sofa.Core.ForceFieldRigid3d):
    """Structured base actuation (deadzone + friction) for a Rigid3d base body.

    Construction kwargs:
        mo               : the Rigid3d MechanicalObject (RigidBaseMO) — python
                           ref, used to read ``rest_position`` each step.
        insertion_axis   : world unit vector ``u`` (len 3).
        home_position    : base home position ``p_home`` (len 3).
        home_orientation : ``q_home_full`` quaternion [qx,qy,qz,qw].
        insertion        : dict with Kb, Db, F0, v0, delta, eps.
        rotation         : dict with Kb, Db, F0, v0, delta, eps.
        non_actuated     : dict with K_perp, B_perp, K_tilt, B_tilt.
    """

    def __init__(self, *args, **kwargs):
        mo = kwargs.pop("mo")
        u = np.asarray(kwargs.pop("insertion_axis"), dtype=float)
        p_home = np.asarray(kwargs.pop("home_position"), dtype=float)
        q_home = np.asarray(kwargs.pop("home_orientation"), dtype=float)
        ins = dict(kwargs.pop("insertion"))
        rot = dict(kwargs.pop("rotation"))
        na = dict(kwargs.pop("non_actuated"))
        # NOTE: do NOT set instance attributes before the base __init__ — doing so
        # corrupts the pybind-bound object and segfaults SOFA.  Assign after.
        # The base sees no Rayleigh damping because the scene runs the solver with
        # rayleighStiffness=0 (the rod's stiffness-proportional damping is supplied
        # by an explicit DiagonalVelocityDampingForceField on the strain instead;
        # see catheter.py).  So this FF needs no Rayleigh-cancellation.
        Sofa.Core.ForceFieldRigid3d.__init__(self, *args, **kwargs)

        self._mo = mo
        self._u = u / np.linalg.norm(u)
        self._p_home = p_home
        self._R_home = R.from_quat(q_home)
        self._R_home_inv = self._R_home.inv()

        # Per-DOF structured-law params (index 0 = insertion, 1 = rotation).
        self._Kb = np.array([ins["Kb"], rot["Kb"]], dtype=float)
        self._Db = np.array([ins["Db"], rot["Db"]], dtype=float)
        self._F0 = np.array([ins["F0"], rot["F0"]], dtype=float)
        self._v0 = np.array([ins["v0"], rot["v0"]], dtype=float)
        self._delta = np.array([ins["delta"], rot["delta"]], dtype=float)
        self._eps = np.array([ins["eps"], rot["eps"]], dtype=float)

        # Rigid-carriage springs on the orthogonal complement.
        self._K_perp = float(na["K_perp"])
        self._B_perp = float(na["B_perp"])
        self._K_tilt = float(na["K_tilt"])
        self._B_tilt = float(na["B_tilt"])

        # Cached projectors (constant — u is fixed in world frame).
        self._uuT = np.outer(self._u, self._u)         # axial projector
        self._perp = np.eye(3) - self._uuT             # orthogonal complement

        # Damping derivatives needed by the tangent (filled in addForce).
        self._B_ins = self._Db[0]
        self._B_rot = self._Db[1]
        # Stiffness tangents Kb * P'(e) (filled in addForce).
        self._S_ins = self._Kb[0]
        self._S_rot = self._Kb[1]

        # No Rayleigh damping from this FF (damping is explicit in the law).
        if hasattr(self, "rayleighStiffness"):
            self.rayleighStiffness = 0.0

        # Diagnostics from the last addForce (read by validation tooling).
        self.last = {}

    # -- geometry helpers ---------------------------------------------------
    def _rest_pose(self):
        rest = np.asarray(self._mo.rest_position.value)[0]
        return rest[0:3], R.from_quat(rest[3:7])

    def _scalars(self, pos0, vel0):
        """Reconstruct actuated scalars from the rigid pose & velocity.

        Returns (b_ins, bdot_ins, b_rot, bdot_rot).
        """
        p = pos0[0:3]
        Rq = R.from_quat(pos0[3:7])
        v = vel0[0:3]
        w = vel0[3:6]
        b_ins = float(self._u @ (p - self._p_home))
        bdot_ins = float(self._u @ v)
        b_rot = float(self._u @ (Rq * self._R_home_inv).as_rotvec())
        bdot_rot = float(self._u @ w)
        return b_ins, bdot_ins, b_rot, bdot_rot

    def _axial_angle(self, Rq):
        return float(self._u @ (Rq * self._R_home_inv).as_rotvec())

    def component_diagnostics(self, pos=None, vel=None):
        """Evaluate named base-force components without mutating SOFA state.

        This is the single source of truth used by both :meth:`addForce` and
        the component-resolved matrix recorder.  Forces use SOFA's Rigid3d
        derivative ordering ``[linear, angular]``.  The returned stiffness and
        damping matrices are *positive* tangents, i.e. the force Jacobians are
        ``df/dx=-K`` and ``df/dv=-D``.

        ``insertion`` and ``rotation`` contain only the two actuated channels.
        ``carriage`` contains the four stiff non-actuated constraints.  Keeping
        these terms separate is important: their sum is observable at the base,
        but they have different physical meanings after proximal condensation.
        """
        pos_v = np.asarray(self._mo.position.value if pos is None else pos)
        vel_v = np.asarray(self._mo.velocity.value if vel is None else vel)
        p = pos_v[0, 0:3]
        Rq = R.from_quat(pos_v[0, 3:7])
        v = vel_v[0, 0:3]
        w = vel_v[0, 3:6]

        r_p, r_R = self._rest_pose()

        b_ins = float(self._u @ (p - self._p_home))
        r_ins = float(self._u @ (r_p - self._p_home))
        bdot_ins = float(self._u @ v)
        e_ins = r_ins - b_ins
        f_ins = (self._Kb[0] * _play(e_ins, self._delta[0], self._eps[0])
                 - self._Db[0] * bdot_ins
                 - self._F0[0] * np.tanh(bdot_ins / self._v0[0]))

        b_rot = self._axial_angle(Rq)
        r_rot = self._axial_angle(r_R)
        e_rot = _wrap(r_rot - b_rot)
        bdot_rot = float(self._u @ w)
        tau = (self._Kb[1] * _play(e_rot, self._delta[1], self._eps[1])
               - self._Db[1] * bdot_rot
               - self._F0[1] * np.tanh(bdot_rot / self._v0[1]))

        e_perp = self._perp @ (r_p - p)
        v_perp = self._perp @ v
        carriage_linear = self._K_perp * e_perp - self._B_perp * v_perp
        e_R = (r_R * Rq.inv()).as_rotvec()
        e_tilt = self._perp @ e_R
        w_perp = self._perp @ w
        carriage_angular = self._K_tilt * e_tilt - self._B_tilt * w_perp

        insertion_force = np.concatenate([f_ins * self._u, np.zeros(3)])
        rotation_force = np.concatenate([np.zeros(3), tau * self._u])
        carriage_force = np.concatenate([carriage_linear, carriage_angular])

        S_ins = self._Kb[0] * _play_deriv(
            e_ins, self._delta[0], self._eps[0])
        S_rot = self._Kb[1] * _play_deriv(
            e_rot, self._delta[1], self._eps[1])
        sech2_ins = 1.0 - np.tanh(bdot_ins / self._v0[0]) ** 2
        sech2_rot = 1.0 - np.tanh(bdot_rot / self._v0[1]) ** 2
        B_ins = self._Db[0] + self._F0[0] / self._v0[0] * sech2_ins
        B_rot = self._Db[1] + self._F0[1] / self._v0[1] * sech2_rot

        def block6(linear, angular):
            out = np.zeros((6, 6), dtype=float)
            out[:3, :3] = linear
            out[3:, 3:] = angular
            return out

        insertion_K = block6(S_ins * self._uuT, np.zeros((3, 3)))
        insertion_D = block6(B_ins * self._uuT, np.zeros((3, 3)))
        rotation_K = block6(np.zeros((3, 3)), S_rot * self._uuT)
        rotation_D = block6(np.zeros((3, 3)), B_rot * self._uuT)
        carriage_K = block6(self._K_perp * self._perp,
                            self._K_tilt * self._perp)
        carriage_D = block6(self._B_perp * self._perp,
                            self._B_tilt * self._perp)

        return {
            "insertion_force": insertion_force,
            "rotation_force": rotation_force,
            "carriage_force": carriage_force,
            "total_force": insertion_force + rotation_force + carriage_force,
            "insertion_stiffness": insertion_K,
            "rotation_stiffness": rotation_K,
            "carriage_stiffness": carriage_K,
            "insertion_damping": insertion_D,
            "rotation_damping": rotation_D,
            "carriage_damping": carriage_D,
            "b_ins": b_ins, "r_ins": r_ins, "bdot_ins": bdot_ins,
            "f_ins": float(f_ins), "b_rot": b_rot, "r_rot": r_rot,
            "bdot_rot": bdot_rot, "tau": float(tau),
        }

    # -- SOFA hooks ---------------------------------------------------------
    def addForce(self, m, forces, pos, vel):
        components = self.component_diagnostics(pos.value, vel.value)

        with forces.writeable() as f:
            f[0][0:6] += components["total_force"]

        # ---- cache tangents for addDForce / addKToMatrix -----------------
        self._S_ins = float(
            self._u @ components["insertion_stiffness"][:3, :3] @ self._u)
        self._S_rot = float(
            self._u @ components["rotation_stiffness"][3:, 3:] @ self._u)
        self._B_ins = float(
            self._u @ components["insertion_damping"][:3, :3] @ self._u)
        self._B_rot = float(
            self._u @ components["rotation_damping"][3:, 3:] @ self._u)

        self.last = {key: components[key] for key in (
            "b_ins", "r_ins", "bdot_ins", "f_ins",
            "b_rot", "r_rot", "bdot_rot", "tau")}

    def _tangent_blocks(self):
        """Return (Jx_trans, Jx_ang, Jv_trans, Jv_ang) 3x3 force Jacobians.

        df/dx and df/dv (negative-definite for restoring/dissipative terms).
        """
        Jx_trans = -self._S_ins * self._uuT - self._K_perp * self._perp
        Jx_ang = -self._S_rot * self._uuT - self._K_tilt * self._perp
        Jv_trans = -self._B_ins * self._uuT - self._B_perp * self._perp
        Jv_ang = -self._B_rot * self._uuT - self._B_tilt * self._perp
        return Jx_trans, Jx_ang, Jv_trans, Jv_ang

    def addDForce(self, m, dforce, dx):
        kF = m["kFactor"]
        bF = m["bFactor"]
        Jx_t, Jx_a, Jv_t, Jv_a = self._tangent_blocks()
        J_t = kF * Jx_t + bF * Jv_t
        J_a = kF * Jx_a + bF * Jv_a
        dx0 = np.asarray(dx.value)[0]
        with dforce.writeable() as df:
            df[0][0:3] += J_t @ dx0[0:3]
            df[0][3:6] += J_a @ dx0[3:6]

    def addKToMatrix(self, mparams, nNodes, nDofs):
        kF = mparams["kFactor"]
        bF = mparams["bFactor"]
        Jx_t, Jx_a, Jv_t, Jv_a = self._tangent_blocks()
        J_t = kF * Jx_t + bF * Jv_t
        J_a = kF * Jx_a + bF * Jv_a
        triplets = []
        # Base body is index 0; Rigid3d deriv has nDofs == 6 (3 trans + 3 ang).
        for i in range(3):
            for j in range(3):
                triplets.append([i, j, J_t[i, j]])
                triplets.append([3 + i, 3 + j, J_a[i, j]])
        return np.asarray(triplets, dtype=float)
