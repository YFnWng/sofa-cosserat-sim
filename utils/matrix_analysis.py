"""Extract system matrices from SOFA and analyze damping ratios.

The Cosserat rod has two DOF spaces:
- Strain coordinates q (cosseratCoordinate): n_sections * 3 DOFs
- Frame positions (cosseratFrame): n_frames * 6 DOFs (or 3 for Vec3d)

The EOM Mq'' + Cq' + Kq = f is in the strain coordinate space.
K comes directly from BeamHookeLawForceField on cosseratCoordinate.
M in strain space is obtained from the global system matrix A assembled
by the solver, which projects frame-space mass through the Cosserat mapping.

Usage (called from a SOFA controller after init):

    from utils.matrix_analysis import analyze_damping
    analyze_damping(robot, solver_node)
"""
from __future__ import annotations

import numpy as np


def _strain_damping_matrix(prefab, n_strain):
    """Explicit strain velocity-damping matrix C_strain (n_strain × n_strain).

    This is where the rod's stiffness-proportional (β·K) damping now lives when
    base actuation is enabled: a ``DiagonalVelocityDampingForceField`` on the
    strain with per-section coefficients ``β·diag(K_section)·L``.  Returns 0 if
    absent, and handles the legacy scalar ``UniformVelocityDampingForceField``.

    It contributes ``dt·C_strain`` to the assembled system matrix A (via
    ``addBToMatrix`` with bFactor=-dt), so the matrix extraction must account for
    it both when back-solving M and when reporting the damping C.
    """
    sd = prefab.cosseratCoordinate.getObject("StrainDamping")
    if sd is None:
        return np.zeros((n_strain, n_strain))
    coef_val = np.asarray(sd.findData("dampingCoefficient").value, dtype=float)
    if coef_val.ndim == 2:  # DiagonalVelocityDamping: per-DOF [.,.,.] coefficients
        ncomp = coef_val.shape[1]
        n_blocks = n_strain // ncomp
        diag = np.concatenate(
            [coef_val[min(i, len(coef_val) - 1)] for i in range(n_blocks)])
        return np.diag(diag[:n_strain])
    # UniformVelocityDamping: scalar coefficient
    c = float(np.ravel(coef_val)[0]) if coef_val.size else 0.0
    return c * np.eye(n_strain)


def extract_matrices(robot, solver_node):
    """Extract M, K, C matrices in strain coordinate space.

    Uses the global system matrix A from the SparseLDL solver, which
    already includes the mapped mass contribution in strain DOFs.

    For implicit Euler with step h:
        A = M + h * C + h^2 * K
    where C = alpha*M + beta*K (Rayleigh damping).

    So: A = M(1 + h*alpha) + K(h*beta + h^2)
    Given A and K, we can solve for M.

    Parameters
    ----------
    robot : CatheterRobot
    solver_node : Sofa.Core.Node

    Returns
    -------
    M : (n, n) effective mass in strain coordinates
    K : (n, n) stiffness matrix in strain coordinates
    C : (n, n) Rayleigh damping matrix
    alpha : float — Rayleigh mass coefficient
    beta : float — Rayleigh stiffness coefficient
    info : dict — diagnostic info (DOF counts, etc.)
    """
    prefab = robot._prefab

    # Rayleigh damping coefficients
    euler_solver = solver_node.getObject("DiagnosticEulerImplicitSolver")
    if euler_solver is None:
        euler_solver = solver_node.getObject("EulerImplicitSolver")
    alpha = float(euler_solver.rayleighMass.value)
    beta = float(euler_solver.rayleighStiffness.value)

    # Stiffness matrix from BeamHookeLawForceField (strain DOF space)
    beam_ff = prefab.cosseratCoordinate.getObject("BeamHookeLawForceField")
    K_sparse = beam_ff.assembleKMatrix()
    K = -K_sparse.toarray()  # SOFA convention: assembleKMatrix returns -K
    n_strain = K.shape[0]

    # Global system matrix from the SparseLDL solver.
    # IMPORTANT: ``from Sofa import SofaLinearSolver`` must be called
    # BEFORE the solver object is created (i.e. before createScene) so
    # that SOFA registers the Python type with .A()/.b()/.x() methods.
    # The caller (e.g. collect_data.py) is responsible for this import.
    ldl_solver = solver_node.getObject("solver")
    A_sparse = ldl_solver.A()
    A_full = A_sparse.toarray()
    n_global = A_full.shape[0]

    # The global system matrix includes all DOFs in assembly order.
    # For the Cosserat rod: [base rigid (6 or 7), strain coords (n_strain), ...]
    # Find the strain block by checking which offset gives a block
    # whose diagonal matches K's diagonal structure.
    base_mo = prefab.rigidBaseNode.getObject("RigidBaseMO")
    # Rigid3d uses 6 DOFs in tangent space (even though stored as 7 with quat)
    n_base_nodes = len(base_mo.position.value)

    # Try common offsets for the strain block within the global matrix
    best_offset = None
    best_match = -1.0
    for try_offset in [6 * n_base_nodes, 7 * n_base_nodes, 0, 6, 7]:
        end = try_offset + n_strain
        if end > n_global:
            continue
        A_block = A_full[try_offset:end, try_offset:end]
        # Compare diagonal pattern with K
        if np.linalg.norm(np.diag(K)) > 1e-12:
            corr = abs(np.corrcoef(np.diag(A_block), np.diag(K))[0, 1])
        else:
            corr = 0.0
        if corr > best_match:
            best_match = corr
            best_offset = try_offset

    if best_offset is None:
        raise RuntimeError(
            f"Cannot locate strain DOF block in global matrix "
            f"(n_global={n_global}, n_strain={n_strain})")

    offset = best_offset
    A_strain = A_full[offset:offset + n_strain, offset:offset + n_strain]

    # For implicit Euler with step h and Rayleigh damping:
    #   A = M_eff * (1 + h*alpha) + K * h*(h + beta_solver + beta_ff)
    # where beta_ff is the per-ForceField rayleighStiffness (enters via
    # kFactorIncludingRayleighDamping in addKToMatrix).
    # Solve for M_eff:
    dt = float(solver_node.getRoot().dt.value)
    beta_ff = float(beam_ff.rayleighStiffness.value)
    coeff_M = 1.0 + dt * alpha
    coeff_K = dt * (dt + beta + beta_ff)

    # Explicit strain damping (DiagonalVelocityDamping): carries the rod's β·K
    # stiffness-proportional damping when the solver β is 0 (base-actuation mode).
    # It adds dt·C_strain to A, so subtract it before back-solving M.
    C_strain = _strain_damping_matrix(prefab, n_strain)
    M = (A_strain - coeff_K * K - dt * C_strain) / coeff_M

    # Symmetrize
    M = 0.5 * (M + M.T)
    K = 0.5 * (K + K.T)
    C_strain = 0.5 * (C_strain + C_strain.T)

    # Total rod damping = Rayleigh (αM + βK from the solver) + explicit strain
    # damping.  With base actuation β_solver=0 and C_strain ≈ β·K, so C matches
    # the legacy αM + βK exactly — the stiffness-proportional damping just moved
    # from the solver into C_strain.
    C = alpha * M + beta * K + C_strain

    # Effective stiffness-proportional damping coefficient carried by C_strain
    # (β_strain such that C_strain ≈ β_strain·K), for consumers that want a scalar.
    kdiag = np.diag(K)
    nz = np.abs(kdiag) > 1e-12
    beta_strain = (float(np.median(np.diag(C_strain)[nz] / kdiag[nz]))
                   if nz.any() else 0.0)

    info = {
        "n_strain": n_strain,
        "n_global": n_global,
        "strain_offset": offset,
        "dt": dt,
        "C_strain": C_strain,        # explicit strain damping matrix
        "beta_strain": beta_strain,  # effective β carried by C_strain (≈ rayleigh_stiffness)
        "beta_eff": beta + beta_strain,  # total effective stiffness-damping β (rod)
    }

    return M, K, C, alpha, beta, info


def compute_modal_analysis(M, K):
    """Compute natural frequencies via generalized eigenvalue problem.

    Parameters
    ----------
    M : (n, n) mass matrix
    K : (n, n) stiffness matrix

    Returns
    -------
    frequencies : (n,) natural frequencies in Hz
    omega : (n,) natural angular frequencies in rad/s
    """
    from scipy.linalg import eigh

    K_sym = 0.5 * (K + K.T)
    M_sym = 0.5 * (M + M.T)

    # Regularize M if near-singular
    M_reg = M_sym + 1e-12 * np.eye(M_sym.shape[0])

    eigenvalues, _ = eigh(K_sym, M_reg)

    omega_sq = np.maximum(eigenvalues, 0.0)
    omega = np.sqrt(omega_sq)
    frequencies = omega / (2 * np.pi)

    return frequencies, omega


def damping_ratios_rayleigh(omega, alpha, beta):
    """Compute per-mode damping ratio for Rayleigh damping.

    zeta_i = alpha / (2 * omega_i) + beta * omega_i / 2

    Returns
    -------
    zeta : (n,) damping ratios (>1 = overdamped)
    """
    zeta = np.zeros_like(omega)
    nonzero = omega > 1e-10
    zeta[nonzero] = alpha / (2 * omega[nonzero]) + beta * omega[nonzero] / 2
    zeta[~nonzero] = np.inf
    return zeta


def analyze_damping(robot, solver_node, print_results=True):
    """Full analysis: extract matrices, compute modes, report damping.

    Returns dict with: M, K, C, alpha, beta, frequencies, omega, zeta, info
    """
    M, K, C, alpha, beta, info = extract_matrices(robot, solver_node)
    frequencies, omega = compute_modal_analysis(M, K)
    zeta = damping_ratios_rayleigh(omega, alpha, beta)

    if print_results:
        print(f"\n=== System Matrix Analysis ===")
        print(f"  Strain DOFs: {info['n_strain']}, "
              f"Global DOFs: {info['n_global']}, "
              f"Strain block offset: {info['strain_offset']}")
        print(f"  dt = {info['dt']}")
        print(f"  Rayleigh: alpha_M={alpha}, beta_K={beta}")
        print(f"  ||M||_F = {np.linalg.norm(M, 'fro'):.4e}")
        print(f"  ||K||_F = {np.linalg.norm(K, 'fro'):.4e}")
        print(f"  ||C||_F = {np.linalg.norm(C, 'fro'):.4e}")

        # Check M is positive (semi-)definite
        M_eigvals = np.linalg.eigvalsh(M)
        print(f"  M eigenvalues: [{M_eigvals.min():.4e}, {M_eigvals.max():.4e}]")
        if M_eigvals.min() < -1e-6:
            print(f"  WARNING: M has negative eigenvalues — "
                  f"strain block extraction may be incorrect")

        order = np.argsort(frequencies)
        freq_sorted = frequencies[order]
        zeta_sorted = zeta[order]

        active = freq_sorted > 0.01
        n_active = np.sum(active)
        n_overdamped = np.sum(zeta_sorted[active] >= 1.0)
        n_underdamped = np.sum(zeta_sorted[active] < 1.0)

        print(f"\n  Active modes (f > 0.01 Hz): {n_active}")
        print(f"  Overdamped (zeta >= 1): {n_overdamped}")
        print(f"  Underdamped (zeta < 1): {n_underdamped}")

        if n_active > 0:
            zeta_active = zeta_sorted[active]
            print(f"  Damping ratio range: [{zeta_active.min():.4f}, "
                  f"{zeta_active.max():.4f}]")
            print(f"  Median damping ratio: {np.median(zeta_active):.4f}")

        n_show = min(10, n_active)
        if n_show > 0:
            print(f"\n  Lowest-frequency modes:")
            print(f"  {'Mode':>6} {'Freq (Hz)':>12} {'Zeta':>10} {'Status':>12}")
            shown = 0
            for i in order:
                if frequencies[i] < 0.01:
                    continue
                status = "overdamped" if zeta[i] >= 1.0 else "UNDERDAMPED"
                print(f"  {i:>6} {frequencies[i]:>12.3f} {zeta[i]:>10.4f} "
                      f"{status:>12}")
                shown += 1
                if shown >= n_show:
                    break

            print(f"\n  Highest-frequency modes:")
            shown = 0
            for i in reversed(order):
                if frequencies[i] < 0.01:
                    continue
                status = "overdamped" if zeta[i] >= 1.0 else "UNDERDAMPED"
                print(f"  {i:>6} {frequencies[i]:>12.3f} {zeta[i]:>10.4f} "
                      f"{status:>12}")
                shown += 1
                if shown >= n_show:
                    break

        if n_underdamped == 0 and n_active > 0:
            print(f"\n  CONCLUSION: ALL modes overdamped — position-only "
                  f"state is Markovian")
        elif n_active > 0 and n_underdamped < n_active * 0.1:
            print(f"\n  CONCLUSION: Nearly all modes overdamped "
                  f"({n_underdamped}/{n_active} underdamped) — "
                  f"approximately Markovian")
        elif n_active > 0:
            print(f"\n  CONCLUSION: {n_underdamped}/{n_active} modes "
                  f"underdamped — velocity state may be needed")

    return {
        "M": M, "K": K, "C": C,
        "alpha": alpha, "beta": beta,
        "frequencies": frequencies, "omega": omega, "zeta": zeta,
        "info": info,
    }
