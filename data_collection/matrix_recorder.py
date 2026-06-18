"""Record SOFA solver matrices alongside trajectory data.

Runs as a separate Controller alongside DataCollectorController.
Records the full system matrix A (subsampled), full RHS b, and full
solution x at each step, plus the strain stiffness matrix K (once).

Also records pre-solve state for both strain and base DOFs so offline
analysis can reconstruct b exactly using the formulas from
simulation/docs/sofa_matrix_assembly.md.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import Sofa.Core


class MatrixRecorderController(Sofa.Core.Controller):
    """Record solver matrices during simulation.

    Parameters (kwargs)
    -------------------
    robot : CatheterRobot
    solver_node : Sofa.Core.Node
    A_interval : int
        Record full A matrix every N steps (default 100).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._robot = kwargs.pop("robot")
        self._solver_node = kwargs.pop("solver_node")
        self._A_interval = int(kwargs.pop("A_interval", 1))
        self._verbose = bool(kwargs.pop("verbose", False))
        self._debug = bool(kwargs.pop("debug", False))

        self._step = 0
        self._initialized = False

        # Extracted once
        self._K: Optional[np.ndarray] = None
        self._strain_offset: int = 0
        self._n_strain: int = 0
        self._n_base: int = 6
        self._n_full: int = 0
        self._alpha: float = 0.0
        self._beta: float = 0.0
        self._beta_ff: float = 0.0
        self._dt: float = 0.01
        self._base_mo = None
        self._strain_mo = None
        self._frame_mo = None
        self._frame_mass = None
        self._mapping = None

        # Pre-solve state (recorded in onAnimateBeginEvent)
        self._q_pre_list: list = []
        self._v_pre_list: list = []
        self._v_base_pre_list: list = []
        self._q_base_pre_list: list = []
        self._rest_base_pre_list: list = []
        self._v_frame_pre_list: list = []

        # Force vectors from computeForce (recorded in onAnimateEndEvent)
        self._f_strain_list: list = []
        self._f_base_list: list = []
        # Force vectors recorded BEFORE solver (in onAnimateBeginEvent)
        self._f_strain_begin_list: list = []
        # Post-solve position (for verifying which q the force corresponds to)
        self._q_post_list: list = []

        # Per-step buffers (full b and x vectors)
        self._b_full_list: list = []
        self._x_full_list: list = []
        # Per-interval buffers (full system matrix)
        self._A_full_list: list = []
        self._A_steps: list = []

        # Jacobian (recorded once at init, then at A_interval)
        self._J_list: list = []
        self._J_steps: list = []

        # Diagnostic data (stored at A_interval steps)
        self._diag_list: list = []
        self._diag_steps: list = []

        # Cable constraint data (H and λ)
        self._cable_constraints: list = []
        self._cable_force_list: list = []    # λ per step (n_cables,)
        self._H_strain_list: list = []       # H projected to strain DOFs
        self._H_base_list: list = []         # H projected to base DOFs

        # C++ diagnostic solver data (if DiagnosticEulerImplicitSolver is used)
        self._cpp_diag_solver = None
        self._cpp_diag_data: dict = {
            "b_after_force": [], "b_after_addMBKv": [],
            "b_after_scale": [], "b_after_project": [],
            "x_solution": [],
            "Mv_only": [], "Kv_only": [],
            "MBKv_full": [], "geom_stiffness": [],
            "b_linsys": [], "x_linsys": [],
            "base_pos_pre": [], "base_rest_pre": [],
            "base_force_total": [], "base_force_spring": [],
            "base_force_mapped": [],
        }

    def _init_matrices(self):
        """Extract constant matrices and locate strain DOF block."""
        from utils.matrix_analysis import extract_matrices
        M, K, C, alpha, beta, info = extract_matrices(
            self._robot, self._solver_node)

        self._K = K
        self._n_strain = info["n_strain"]
        self._strain_offset = info["strain_offset"]
        self._alpha = alpha
        self._beta = beta
        self._dt = info["dt"]

        prefab = self._robot._prefab
        beam_ff = prefab.cosseratCoordinate.getObject("BeamHookeLawForceField")
        self._beta_ff = float(beam_ff.rayleighStiffness.value)

        self._base_mo = prefab.rigidBaseNode.getObject("RigidBaseMO")
        self._n_base = self._strain_offset
        self._n_full = self._n_base + self._n_strain

        # Extract base stiffness from RestShapeSpringsForceField
        base_spring = prefab.rigidBaseNode.getObject("BaseAttachment")
        if base_spring is not None:
            k_trans = float(base_spring.stiffness.value[0])
            k_ang = float(base_spring.angularStiffness.value[0])
            self._K_base = np.diag([k_trans] * 3 + [k_ang] * 3)
            print(f"[MatrixRecorder] K_base: stiffness={k_trans:.2e}, "
                  f"angularStiffness={k_ang:.2e}")
        else:
            self._K_base = np.zeros((self._n_base, self._n_base))
            print("[MatrixRecorder] No BaseAttachment found, K_base = 0")

        self._strain_mo = prefab.cosseratCoordinate.cosseratCoordinateMO

        from scipy.spatial.transform import Rotation as Rot
        prefab_rot_quat = self._robot.prefab_rotation_offset
        R = Rot.from_quat(prefab_rot_quat).as_matrix()
        self._strain_rot = R

        self._initialized = True

        # Detect DiagnosticEulerImplicitSolver
        for obj in self._solver_node.objects:
            if obj.getClassName() == "DiagnosticEulerImplicitSolver":
                self._cpp_diag_solver = obj
                print(f"[MatrixRecorder] Found DiagnosticEulerImplicitSolver")
                break

        # Find CableConstraint objects in the scene graph
        self._cable_constraints = []
        self._find_cable_constraints(self._solver_node)
        if self._cable_constraints:
            names = [c.getName() for c in self._cable_constraints]
            print(f"[MatrixRecorder] Found {len(self._cable_constraints)} cable constraints: {names}")

        # Record rest position (q0) once
        self._q0 = np.array(self._strain_mo.rest_position.value).flatten()

        # NaN placeholder for step 1's pre-solve state (already consumed by solver)
        if self._debug:
            q = np.array(self._strain_mo.position.value).flatten()
            self._q_pre_list.append(np.full(q.shape, np.nan))
            self._v_pre_list.append(np.full(q.shape, np.nan))
            v_base = np.array(self._base_mo.velocity.value).flatten()[:self._n_base]
            self._v_base_pre_list.append(np.full(v_base.shape, np.nan))
            self._q_base_pre_list.append(np.full(v_base.shape, np.nan))
            self._rest_base_pre_list.append(np.full(v_base.shape, np.nan))
            self._f_strain_begin_list.append(np.full(q.shape, np.nan))

        # Locate frame MO, mass, and mapping
        frame_node = prefab.rigidBaseNode.getChild("cosseratInSofaFrameNode")
        if frame_node is None:
            for child in prefab.rigidBaseNode.children:
                for obj in child.objects:
                    if obj.getClassName() == "UniformMass":
                        frame_node = child
                        break
                if frame_node is not None:
                    break
        if frame_node is not None:
            self._frame_mo = frame_node.getObject("FramesMO")
            try:
                self._frame_mass = frame_node.getObject("frameMass")
                if self._frame_mass is None:
                    for obj in frame_node.objects:
                        if "Mass" in obj.getClassName():
                            self._frame_mass = obj
                            break
            except Exception:
                pass
            for obj in frame_node.objects:
                if "Mapping" in obj.getClassName() or "mapping" in obj.getClassName():
                    self._mapping = obj
                    break

        if self._verbose:
            print(f"\n[MatrixRecorder] === FULL SCENE GRAPH ===")
            self._enumerate_node(self._solver_node, depth=0)
            print(f"[MatrixRecorder] === END SCENE GRAPH ===\n")

        print(f"[MatrixRecorder] Initialized: n_strain={self._n_strain}, "
              f"n_base={self._n_base}, n_full={self._n_full}, "
              f"offset={self._strain_offset}, "
              f"alpha={alpha}, beta_solver={beta}, beta_ff={self._beta_ff}")
        print(f"[MatrixRecorder] q0 rest[:6] = {self._q0[:6]}")
        print(f"[MatrixRecorder] ||q0|| = {np.linalg.norm(self._q0):.6e}")
        if self._verbose:
            if self._frame_mo is not None:
                n_frames = len(self._frame_mo.position.value)
                print(f"[MatrixRecorder] FramesMO: {n_frames} frames")
            if self._frame_mass is not None:
                print(f"[MatrixRecorder] FrameMass: class={self._frame_mass.getClassName()}, "
                      f"totalMass={self._frame_mass.totalMass.value}")
            if self._mapping is not None:
                print(f"[MatrixRecorder] Mapping: class={self._mapping.getClassName()}, "
                      f"name={self._mapping.getName()}")
                try:
                    J = self._mapping.getJ()
                    if hasattr(J, 'toarray'):
                        J_dense = J.toarray()
                    elif hasattr(J, 'shape'):
                        J_dense = np.array(J)
                    else:
                        J_dense = None
                    if J_dense is not None:
                        print(f"[MatrixRecorder] Jacobian J: shape={J_dense.shape}, "
                              f"||J||_F={np.linalg.norm(J_dense):.4e}")
                except Exception as e:
                    print(f"[MatrixRecorder] getJ() failed: {e}")
                    try:
                        Js = self._mapping.getJs()
                        print(f"[MatrixRecorder] getJs() returned: type={type(Js)}")
                    except Exception as e2:
                        print(f"[MatrixRecorder] getJs() also failed: {e2}")

    def _enumerate_node(self, node, depth=0):
        """Print full scene graph from this node."""
        indent = "  " * depth
        print(f"[MatrixRecorder] {indent}[{node.getName()}]")
        for obj in node.objects:
            cls = obj.getClassName()
            name = obj.getName()
            extra = ""
            if "MechanicalObject" in cls:
                try:
                    pos = obj.position.value
                    extra = f" ({len(pos)} DOFs, template={obj.getTemplateName()})"
                except Exception:
                    pass
            elif "Mass" in cls:
                try:
                    extra = f" (totalMass={obj.totalMass.value})"
                except Exception:
                    pass
            elif "ForceField" in cls or "forcefield" in cls.lower():
                try:
                    rs = getattr(obj, 'rayleighStiffness', None)
                    if rs is not None:
                        extra = f" (rayleighStiffness={rs.value})"
                except Exception:
                    pass
            elif "Mapping" in cls or "mapping" in cls.lower():
                try:
                    extra = f" (template={obj.getTemplateName()})"
                except Exception:
                    pass
            elif "Constraint" in cls:
                extra = f" (type={cls})"
            print(f"[MatrixRecorder] {indent}  - {cls} ({name}){extra}")
        for child in node.children:
            self._enumerate_node(child, depth + 1)

    def _find_cable_constraints(self, node):
        """Recursively find CableConstraint objects in the scene graph."""
        for obj in node.objects:
            if "CableConstraint" in obj.getClassName():
                self._cable_constraints.append(obj)
        for child in node.children:
            self._find_cable_constraints(child)

    def _record_cable_data(self):
        """Record constraint Jacobian H and cable forces λ.

        When debug=False, only H_strain is recorded (sufficient for LNN
        training).  cable_lambda and H_base are debug-only.
        """
        ns = self._n_strain
        nb = self._n_base

        if self._debug:
            forces = []
            for cc in self._cable_constraints:
                try:
                    forces.append(float(cc.d_force.value))
                except Exception:
                    forces.append(0.0)
            self._cable_force_list.append(np.array(forces))

        # H_strain: always recorded (needed for tendon mapping verification)
        try:
            H_strain_raw = self._strain_mo.constraint.value
            if hasattr(H_strain_raw, 'toarray'):
                H_strain = H_strain_raw.toarray()
            elif hasattr(H_strain_raw, '__array__'):
                H_strain = np.array(H_strain_raw)
            else:
                H_strain = None

            if H_strain is not None:
                if H_strain.ndim == 2:
                    if H_strain.shape[1] == ns:
                        pass
                    elif H_strain.shape[1] == ns // 3:
                        H_strain = H_strain.reshape(H_strain.shape[0], -1)
                self._H_strain_list.append(H_strain.copy())
                if self._step <= 1:
                    print(f"[MatrixRecorder] H_strain: shape={H_strain.shape}, "
                          f"||H||={np.linalg.norm(H_strain):.4e}, "
                          f"nnz={np.count_nonzero(H_strain)}")
            else:
                self._H_strain_list.append(np.zeros((len(self._cable_constraints), ns)))
                if self._step <= 1:
                    print(f"[MatrixRecorder] H_strain: could not convert "
                          f"(type={type(H_strain_raw)})")
        except Exception as e:
            self._H_strain_list.append(np.zeros((len(self._cable_constraints), ns)))
            if self._step <= 1:
                print(f"[MatrixRecorder] H_strain read failed: {e}")

        if not self._debug:
            return

        try:
            H_base_raw = self._base_mo.constraint.value
            if hasattr(H_base_raw, 'toarray'):
                H_base = H_base_raw.toarray()
            elif hasattr(H_base_raw, '__array__'):
                H_base = np.array(H_base_raw)
            else:
                H_base = None

            if H_base is not None:
                if H_base.ndim == 2 and H_base.shape[1] != nb:
                    H_base = H_base.reshape(H_base.shape[0], -1)[:, :nb]
                self._H_base_list.append(H_base.copy())
                if self._step <= 1:
                    print(f"[MatrixRecorder] H_base: shape={H_base.shape}, "
                          f"||H||={np.linalg.norm(H_base):.4e}")
            else:
                self._H_base_list.append(np.zeros((len(self._cable_constraints), nb)))
        except Exception as e:
            self._H_base_list.append(np.zeros((len(self._cable_constraints), nb)))
            if self._step <= 1:
                print(f"[MatrixRecorder] H_base read failed: {e}")

    def onAnimateBeginEvent(self, _event):
        """Record pre-solve state (q, v) before the solver modifies them.

        Only recorded when debug=True — the same data is available in
        the H5 via DataCollectorController (strain_coords, strain_velocity,
        frame_velocity).
        """
        if not self._initialized or not self._debug:
            return
        try:
            v_base = np.array(self._base_mo.velocity.value).flatten()[:self._n_base]
            self._v_base_pre_list.append(v_base.copy())

            q_base = np.array(self._base_mo.position.value).flatten()[:self._n_base]
            self._q_base_pre_list.append(q_base.copy())

            rest_base = np.array(self._base_mo.rest_position.value).flatten()[:self._n_base]
            self._rest_base_pre_list.append(rest_base.copy())

            q = np.array(self._strain_mo.position.value).flatten()
            v = np.array(self._strain_mo.velocity.value).flatten()
            self._q_pre_list.append(q.copy())
            self._v_pre_list.append(v.copy())

            if self._frame_mo is not None:
                v_frame = np.array(self._frame_mo.velocity.value).flatten()
                self._v_frame_pre_list.append(v_frame.copy())

            f_begin = np.array(self._strain_mo.force.value).flatten().copy()
            self._f_strain_begin_list.append(f_begin)
        except Exception:
            pass

    def onAnimateEndEvent(self, _event):
        if self._step == 0:
            try:
                self._init_matrices()
            except Exception as e:
                print(f"[MatrixRecorder] Init failed: {e}")
                self._step += 1
                return

        if not self._initialized:
            self._step += 1
            return

        try:
            ldl = self._solver_node.getObject("solver")
            nf = self._n_full

            if self._debug:
                b_raw = ldl.b()
                b_full = np.array(b_raw.flatten())[:nf].copy()
                x_raw = ldl.x()
                x_full = np.array(x_raw.flatten())[:nf].copy()
                self._b_full_list.append(b_full)
                self._x_full_list.append(x_full)

                f_strain = np.array(self._strain_mo.force.value).flatten().copy()
                self._f_strain_list.append(f_strain)
                f_base_raw = self._base_mo.force.value
                f_base = np.array(f_base_raw).flatten()[:self._n_base].copy()
                self._f_base_list.append(f_base)

                q_post = np.array(self._strain_mo.position.value).flatten().copy()
                self._q_post_list.append(q_post)

            if self._step % self._A_interval == 0 or self._step == 0:
                A = ldl.A().toarray()[:nf, :nf].copy()
                self._A_full_list.append(A)
                self._A_steps.append(self._step)

                if self._debug:
                    b_full = self._b_full_list[-1] if self._b_full_list else np.array(ldl.b().flatten())[:nf].copy()
                    x_full = self._x_full_list[-1] if self._x_full_list else np.array(ldl.x().flatten())[:nf].copy()
                    f_strain = self._f_strain_list[-1] if self._f_strain_list else np.array(self._strain_mo.force.value).flatten().copy()
                    self._diagnose_b_reconstruction(A, b_full, x_full, f_strain)

                    if self._mapping is not None:
                        try:
                            J_raw = self._mapping.getJ()
                            if hasattr(J_raw, 'toarray'):
                                J_dense = J_raw.toarray()
                            else:
                                J_dense = np.array(J_raw)
                            self._J_list.append(J_dense.copy())
                            self._J_steps.append(self._step)
                            if self._step == 0:
                                print(f"[MatrixRecorder] J shape={J_dense.shape}, "
                                      f"nnz={np.count_nonzero(J_dense)}")
                        except Exception as e:
                            if self._step == 0:
                                print(f"[MatrixRecorder] getJ() at step 0 failed: {e}")

            # Record cable constraint data (H and λ).
            # Debug: every step.  Non-debug: same cadence as A_full_samples.
            if self._cable_constraints:
                if self._debug:
                    self._record_cable_data()
                elif self._step % self._A_interval == 0 or self._step == 0:
                    self._record_cable_data()

            # Read C++ diagnostic solver data (debug only)
            if self._debug and self._cpp_diag_solver is not None:
                try:
                    ds = self._cpp_diag_solver
                    for key in self._cpp_diag_data:
                        val = np.array(ds.getData(key).value).flatten()
                        if len(val) > 0:
                            self._cpp_diag_data[key].append(val.copy())
                except Exception as e:
                    if self._step <= 1:
                        print(f"[MatrixRecorder] C++ diag read failed: {e}")

        except Exception as e:
            if self._step <= 4:
                import traceback
                print(f"[MatrixRecorder] Recording failed at step {self._step}: {e}")
                traceback.print_exc()

        self._step += 1

    def _diagnose_b_reconstruction(self, A_full, b_full, x_full, f_strain):
        """Store diagnostic vectors and print brief summary."""
        off = self._strain_offset
        ns = self._n_strain
        nb = self._n_base
        dt = self._dt
        K = self._K
        norm = np.linalg.norm

        if not self._q_pre_list or np.any(np.isnan(self._q_pre_list[-1])):
            return
        q_pre = self._q_pre_list[-1]
        v_pre = self._v_pre_list[-1]
        v_base = self._v_base_pre_list[-1]

        mo = self._strain_mo
        q_pos = np.array(mo.position.value).flatten()
        v_vel = np.array(mo.velocity.value).flatten()

        def _read_mo_field(mo_obj, name):
            try:
                d = mo_obj.getData(name)
                if d is not None:
                    return np.array(d.value).flatten()
            except Exception:
                pass
            return None

        # Strain MO extra fields
        q_free = _read_mo_field(mo, "free_position")
        v_free = _read_mo_field(mo, "free_velocity")
        derivX = _read_mo_field(mo, "derivX")
        solution = _read_mo_field(mo, "solution")
        rhs_mo = _read_mo_field(mo, "RHS")
        dforce = _read_mo_field(mo, "dforce(V_DERIV)")
        lam = _read_mo_field(mo, "lambda")
        constraint_dx = _read_mo_field(mo, "constraint_dx")

        # Base MO fields
        bmo = self._base_mo
        b_q_pos = np.array(bmo.position.value).flatten()[:nb]
        b_v_vel = np.array(bmo.velocity.value).flatten()[:nb]
        b_q_free = _read_mo_field(bmo, "free_position")
        b_v_free = _read_mo_field(bmo, "free_velocity")
        b_derivX = _read_mo_field(bmo, "derivX")
        b_solution = _read_mo_field(bmo, "solution")
        b_rhs_mo = _read_mo_field(bmo, "RHS")
        b_force = _read_mo_field(bmo, "force")
        b_dforce = _read_mo_field(bmo, "dforce(V_DERIV)")
        b_lam = _read_mo_field(bmo, "lambda")
        b_constraint_dx = _read_mo_field(bmo, "constraint_dx")
        b_rest = _read_mo_field(bmo, "rest_position")

        if self._verbose and not self._diag_list:
            print(f"\n[MatrixRecorder] Base MO data fields:")
            try:
                for item in bmo.getDataFields():
                    try:
                        name = item.getName() if hasattr(item, 'getName') else str(item)
                        val = item.value if hasattr(item, 'value') else None
                        if val is not None and hasattr(val, '__len__') and len(val) > 0:
                            sz = len(val) if not hasattr(val[0], '__len__') else len(val) * len(val[0])
                            print(f"    {name}: {sz} components")
                    except Exception:
                        pass
            except Exception as e:
                print(f"    ERROR: {e}")

        f_begin = self._f_strain_begin_list[-1] if self._f_strain_begin_list else None
        if f_begin is not None and np.any(np.isnan(f_begin)):
            f_begin = None

        # Assembled system vectors
        b_s = b_full[off:off+ns]
        x_s = x_full[off:off+ns]
        x_b = x_full[:nb]

        # Extract M_ss, M_sb from A
        A_ss = A_full[off:off+ns, off:off+ns]
        A_sb = A_full[off:off+ns, :nb]
        coeff_M = 1.0 + dt * self._alpha
        coeff_K_A = dt * (dt + self._beta + self._beta_ff)
        M_ss = (A_ss - coeff_K_A * K) / coeff_M
        M_ss = 0.5 * (M_ss + M_ss.T)
        M_sb = A_sb / coeff_M

        alpha = self._alpha
        beta = self._beta
        coeff_K_b = dt + beta
        beta_ff = self._beta_ff

        # Computed terms
        Mv = M_ss @ v_pre
        Kv = K @ v_pre
        Msb_vb = M_sb @ v_base
        Kq = K @ q_pre

        # All candidate b reconstructions
        addMBKv_noFF = -alpha * Mv - coeff_K_b * Kv - alpha * Msb_vb
        addMBKv_withFF = -alpha * Mv - (coeff_K_b + beta_ff) * Kv - alpha * Msb_vb

        # Direct from A (no M extraction): addMBKv = rA*(A*v) + rK*(K*v)
        # where A*v = A_ss*v_s + A_sb*v_b (full coupling included)
        rA = -alpha / coeff_M
        rK = alpha * coeff_K_A / coeff_M - coeff_K_b
        Av_full = A_ss @ v_pre + A_sb @ v_base
        addMBKv_fromA = rA * Av_full + rK * Kv

        # Also test: b_s = A*x exactly, so b/dt = A*x/dt.
        # x = -dv. If b = -h*(f + addMBKv), then b = -h*f - h*addMBKv.
        # → h*addMBKv = -b - h*f = -b + h*K*q
        # → addMBKv_exact = -(b/h) - f = -(b_s/dt) + K*q
        addMBKv_exact = -b_s / dt + Kq  # this is what addMBKv MUST be (from b directly)

        combos = {}
        for sign_label, f_sign in [("-f", -1), ("+f", +1)]:
            for add_label, addv in [
                ("addMBKv", addMBKv_noFF),
                ("addMBKv_ff", addMBKv_withFF),
                ("-addMBKv", -addMBKv_noFF),
                ("-addMBKv_ff", -addMBKv_withFF),
            ]:
                key = f"{sign_label} {add_label}"
                b_try = dt * (f_sign * f_strain + addv)
                r = norm(b_s - b_try) / (norm(b_s) + 1e-30)
                combos[key] = r

        # Also add the from-A reconstruction
        b_fromA = dt * (-f_strain - addMBKv_fromA)
        combos["-f -addMBKv(fromA)"] = norm(b_s - b_fromA) / (norm(b_s) + 1e-30)

        # Best combo residual vector
        best_key = min(combos, key=combos.get)
        b_neg_f_neg_add = dt * (-f_strain - addMBKv_noFF)
        resid_nfna = b_s - b_neg_f_neg_add
        b_neg_f_pos_add = dt * (-f_strain + addMBKv_noFF)
        resid_nfpa = b_s - b_neg_f_pos_add

        # addMBKv comparison: exact (from b) vs M-based vs A-based
        addMBKv_gap_M = addMBKv_exact - addMBKv_noFF    # gap in our M extraction
        addMBKv_gap_A = addMBKv_exact - addMBKv_fromA   # gap bypassing M

        # Store diagnostic data
        diag = {
            "b_s": b_s.copy(),
            "x_s": x_s.copy(),
            "x_b": x_b.copy(),
            "q_pre": q_pre.copy(),
            "v_pre": v_pre.copy(),
            "v_base": v_base.copy(),
            "q_pos": q_pos.copy(),
            "v_vel": v_vel.copy(),
            "f_strain": f_strain.copy(),
            "Mv": Mv.copy(),
            "Kv": Kv.copy(),
            "Kq": Kq.copy(),
            "Msb_vb": Msb_vb.copy(),
            "resid_neg_f_neg_addMBKv": resid_nfna.copy(),
            "resid_neg_f_pos_addMBKv": resid_nfpa.copy(),
            "addMBKv_exact": addMBKv_exact.copy(),
            "addMBKv_gap_M": addMBKv_gap_M.copy(),
            "addMBKv_gap_A": addMBKv_gap_A.copy(),
            "combo_residuals": combos,
        }
        if q_free is not None:
            diag["q_free"] = q_free.copy()
        if v_free is not None:
            diag["v_free"] = v_free.copy()
        if derivX is not None:
            diag["derivX"] = derivX.copy()
        if solution is not None:
            diag["solution_mo"] = solution.copy()
        if rhs_mo is not None:
            diag["rhs_mo"] = rhs_mo.copy()
        if dforce is not None:
            diag["dforce"] = dforce.copy()
        if lam is not None:
            diag["lambda"] = lam.copy()
        if constraint_dx is not None:
            diag["constraint_dx"] = constraint_dx.copy()
        if f_begin is not None:
            diag["f_begin"] = f_begin.copy()

        # Base MO fields (truncated to nb)
        diag["base_q_pos"] = b_q_pos[:nb].copy()
        diag["base_v_vel"] = b_v_vel[:nb].copy()
        for key, val in [
            ("base_q_free", b_q_free), ("base_v_free", b_v_free),
            ("base_derivX", b_derivX), ("base_solution", b_solution),
            ("base_rhs_mo", b_rhs_mo), ("base_force", b_force),
            ("base_dforce", b_dforce), ("base_lambda", b_lam),
            ("base_constraint_dx", b_constraint_dx), ("base_rest", b_rest),
        ]:
            if val is not None:
                diag[key] = val[:nb].copy() if len(val) >= nb else val.copy()

        self._diag_list.append(diag)
        self._diag_steps.append(self._step)

        if self._verbose:
            print(f"\n[MatrixRecorder] Diagnostic step {self._step}:")
            print(f"  ||b_s||={norm(b_s):.4e} ||x_s||={norm(x_s):.4e}")
            addMBKv_exact_norm = norm(addMBKv_exact)
            if addMBKv_exact_norm > 1e-30:
                print(f"  addMBKv gap: M-based={norm(addMBKv_gap_M)/addMBKv_exact_norm*100:.2f}%, "
                      f"A-based={norm(addMBKv_gap_A)/addMBKv_exact_norm*100:.2f}%")
            if v_free is not None:
                dv = v_free[:ns] - v_pre
                print(f"  x_s vs dv_free: ||x_s||={norm(x_s):.4e}, ||dv||={norm(dv):.4e}, "
                      f"||x_s-dv||={norm(x_s-dv):.4e}, ||x_s+dv||={norm(x_s+dv):.4e}")
            if solution is not None:
                print(f"  solver.x vs MO.solution: ||diff||={norm(x_s - solution[:ns]):.4e}")
            if rhs_mo is not None:
                print(f"  solver.b vs MO.RHS: ||diff||={norm(b_s - rhs_mo[:ns]):.4e}")
            if derivX is not None:
                print(f"  solver.x vs MO.derivX: ||diff||={norm(x_s - derivX[:ns]):.4e}")
            b_b = b_full[:nb]
            print(f"  ||b_base||={norm(b_b):.4e} ||x_base||={norm(x_b):.4e}")
            if b_rhs_mo is not None:
                print(f"  solver.b_base vs base.RHS: ||diff||={norm(b_b - b_rhs_mo[:nb]):.4e}")
            if b_solution is not None:
                print(f"  solver.x_base vs base.solution: ||diff||={norm(x_b - b_solution[:nb]):.4e}")
            if b_v_free is not None:
                dv_base = b_v_free[:nb] - v_base
                print(f"  x_base vs dv_base_free: ||x_b||={norm(x_b):.4e}, ||dv||={norm(dv_base):.4e}, "
                      f"||x_b-dv||={norm(x_b-dv_base):.4e}, ||x_b+dv||={norm(x_b+dv_base):.4e}")
            top3 = sorted(combos.items(), key=lambda kv: kv[1])[:3]
            for key, r in top3:
                print(f"  {key:25s}: {r*100:.1f}%")
            print(f"  Best: {best_key} ({combos[best_key]*100:.1f}%)")

    def get_data(self) -> dict:
        """Return recorded matrix data as numpy arrays.

        When debug=False (default), returns only what LNN training needs:
            A_full_samples, A_full_steps, H_strain, K_strain, K_base,
            alpha, beta, beta_ff.

        When debug=True, also returns per-step pre-solve state, solver
        vectors, forces, diagnostics, Jacobians, etc.
        """
        data = {
            "K_strain": self._K,
            "K_base": self._K_base,
            "A_full_samples": np.stack(self._A_full_list) if self._A_full_list else np.array([]),
            "A_full_steps": np.array(self._A_steps, dtype=np.int64),
            "alpha": np.float64(self._alpha),
            "beta": np.float64(self._beta),
            "beta_ff": np.float64(self._beta_ff),
        }
        if self._H_strain_list:
            data["H_strain"] = np.stack(self._H_strain_list)

        if not self._debug:
            return data

        # --- Debug-only fields below ---
        data["q0_rest"] = self._q0 if self._q0 is not None else np.array([])

        if self._q_pre_list:
            data["q_pre_solve"] = np.stack(self._q_pre_list)
            data["v_pre_solve"] = np.stack(self._v_pre_list)
            data["v_base_pre_solve"] = np.stack(self._v_base_pre_list)
        if self._q_base_pre_list:
            data["q_base_pre_solve"] = np.stack(self._q_base_pre_list)
        if self._rest_base_pre_list:
            data["rest_base_pre_solve"] = np.stack(self._rest_base_pre_list)
        if self._b_full_list:
            data["solver_b_full"] = np.stack(self._b_full_list)
            data["solver_x_full"] = np.stack(self._x_full_list)
        if self._f_strain_list:
            data["f_strain"] = np.stack(self._f_strain_list)
            data["f_base"] = np.stack(self._f_base_list)
        if self._q_post_list:
            data["q_post_solve"] = np.stack(self._q_post_list)
        if self._f_strain_begin_list:
            data["f_strain_begin"] = np.stack(self._f_strain_begin_list)
        if self._v_frame_pre_list:
            data["v_frame_pre_solve"] = np.stack(self._v_frame_pre_list)
        if self._J_list:
            data["J_samples"] = np.stack(self._J_list)
            data["J_steps"] = np.array(self._J_steps, dtype=np.int64)
        if self._diag_list:
            data["diag_steps"] = np.array(self._diag_steps, dtype=np.int64)
            vec_keys = [
                "b_s", "x_s", "x_b", "q_pre", "v_pre", "v_base",
                "q_pos", "v_vel", "f_strain", "Mv", "Kv", "Kq", "Msb_vb",
                "resid_neg_f_neg_addMBKv", "resid_neg_f_pos_addMBKv",
                "addMBKv_exact", "addMBKv_gap_M", "addMBKv_gap_A",
                "base_q_pos", "base_v_vel",
            ]
            for k in vec_keys:
                data[f"diag_{k}"] = np.stack([d[k] for d in self._diag_list])
            opt_keys = [
                "q_free", "v_free", "derivX", "solution_mo", "rhs_mo",
                "dforce", "lambda", "constraint_dx", "f_begin",
                "base_q_free", "base_v_free", "base_derivX",
                "base_solution", "base_rhs_mo", "base_force",
                "base_dforce", "base_lambda", "base_constraint_dx",
                "base_rest",
            ]
            for k in opt_keys:
                vals = [d[k] for d in self._diag_list if k in d]
                if len(vals) == len(self._diag_list):
                    data[f"diag_{k}"] = np.stack(vals)
            combo_keys = list(self._diag_list[0]["combo_residuals"].keys())
            combo_arr = np.zeros((len(self._diag_list), len(combo_keys)))
            for i, d in enumerate(self._diag_list):
                for j, ck in enumerate(combo_keys):
                    combo_arr[i, j] = d["combo_residuals"].get(ck, np.nan)
            data["diag_combo_residuals"] = combo_arr
            data["diag_combo_names"] = np.array(combo_keys, dtype="S")

        if self._cable_force_list:
            data["cable_lambda"] = np.stack(self._cable_force_list)
        if self._H_base_list:
            data["H_base"] = np.stack(self._H_base_list)

        # C++ diagnostic solver data
        for key, vals in self._cpp_diag_data.items():
            if vals:
                data[f"cpp_{key}"] = np.stack(vals)

        return data

    def get_metadata(self) -> dict:
        """Return recording metadata.

        Non-debug data fields (always recorded):
        - Constants: K_strain, K_base, alpha, beta, beta_ff.
        - Per-step: H_strain.
        - Per-A_interval: A_full_samples, A_full_steps.

        Debug-only data fields (recorded when debug=True):
        - Per-step: q_pre_solve, v_pre_solve, v_base_pre_solve,
          cable_lambda, H_base, solver_b_full, solver_x_full,
          f_strain, f_base, q_post_solve, v_frame_pre_solve,
          f_strain_begin, cpp_* diagnostic vectors.
        - Per-A_interval: J_samples, diag_* fields.
        """
        meta = {
            "n_strain": self._n_strain,
            "n_base": self._n_base,
            "n_full": self._n_full,
            "strain_offset": self._strain_offset,
            "alpha": self._alpha,
            "beta": self._beta,
            "beta_ff": self._beta_ff,
            "dt": self._dt,
            "A_interval": self._A_interval,
            "strain_rotation": self._strain_rot.tolist(),
        }
        if self._cable_constraints:
            meta["n_cables"] = len(self._cable_constraints)
            meta["cable_names"] = [c.getName() for c in self._cable_constraints]
        if self._frame_mass is not None:
            try:
                meta["frame_total_mass"] = float(self._frame_mass.totalMass.value)
            except Exception:
                pass
        if self._frame_mo is not None:
            try:
                meta["n_frames"] = len(self._frame_mo.position.value)
            except Exception:
                pass
        return meta
