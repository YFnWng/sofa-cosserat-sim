# SOFA System Matrix (A) and RHS (b) Assembly — Complete Component Trace

Reference for offline analysis of recorded solver matrices from the Cosserat beam catheter simulation.

## Scene Graph Structure (DOF hierarchy)

```
root
├── FreeMotionAnimationLoop                     [animation loop]
├── ProjectedGaussSeidelConstraintSolver        [constraint solver]
└── solverNode
    ├── EulerImplicitSolver                     [ODE solver]
    ├── SparseLDLSolver (name="solver")         [linear solver]
    ├── GenericConstraintCorrection
    └── CosseratBase (parent DOFs)
        ├── rigidBaseNode
        │   ├── MechanicalObject "RigidBaseMO"  [Rigid3d, 1 DOF = 6 components]
        │   └── RestShapeSpringsForceField      [ForceField on base]
        ├── cosseratCoordinate
        │   ├── MechanicalObject "cosseratCoordinateMO"  [Vec3d, 32 DOFs = 96 components]
        │   └── BeamHookeLawForceField          [ForceField on strain]
        └── cosseratFrame (MAPPED child of both base + strain)
            ├── MechanicalObject "FramesMO"     [Rigid3d, 33 DOFs]
            ├── UniformMass                     [Mass (IS-A ForceField) on frames]
            ├── DiscreteCosseratMapping         [Mapping: (strain, base) → frames]
            ├── ConstantForceField "tipForce"   [ForceField on frames, zero in scene 1]
            └── CableAttachment_0, _1, ...      [child nodes for cables]
                ├── MechanicalObject "CableGuideMO"  [Vec3d]
                ├── SkinningMapping             [Mapping: frames → cable guide]
                └── CableConstraint             [CONSTRAINT, not a force field]
```

Key inheritance: `UniformMass` → `Mass<DataTypes>` → `ForceField<DataTypes>` + `BaseMass`.
Because Mass IS-A ForceField, it is visited by `fwdForceField` in both `MechanicalComputeForceVisitor` and `MechanicalAddMBKdxVisitor`.

## Phase 1: FreeMotionAnimationLoop Pre-Solve Setup

| Step | Source file | What it does |
|---|---|---|
| `freePosition = position` | `FreeMotionAnimationLoop.cpp:190` | Copy current state to free vectors |
| `freeVelocity = velocity` | `FreeMotionAnimationLoop.cpp:191` | " |
| `addToTotalForces(lambda)` | `FreeMotionAnimationLoop.cpp:197-200` | Registers lambda in `AccumulationVecId` proxy (for geometric stiffness in mappings). Does NOT inject lambda into force vector f. |
| `lambda *= 1/dt` | `FreeMotionAnimationLoop.cpp:247` | Scale lambda for geometric stiffness computation |
| `MechanicalComputeGeometricStiffness` | `FreeMotionAnimationLoop.cpp:253-255` | Calls `buildGeometricStiffnessMatrix` on mappings — **NO-OP for Cosserat** (`applyDJT` is empty, `buildGeometricStiffnessMatrix` not implemented) |

## Phase 2: RHS Vector b Assembly (EulerImplicitSolver::solve)

### Step 2a: `computeForce(f)` — MechanicalComputeForceVisitor

Source: `EulerImplicitSolver.cpp:129` → `MechanicalOperations.cpp:251-260`

Visitor impl: `MechanicalComputeForceVisitor.cpp`

Visitor walks the graph top-down (forward), then bottom-up (backward through mappings).

| Pass | Method called | Component | What it does | Contributes to f? |
|---|---|---|---|---|
| fwd | `accumulateForce` | MechanicalObject (base) | Adds `externalForce` vector to f. Typically empty unless set by a controller. | Only if externalForce set |
| fwd | `accumulateForce` | MechanicalObject (strain) | Same — adds externalForce to f. | Only if externalForce set |
| fwd | `ff->addForce` | **RestShapeSpringsForceField** (base) | `f_base -= k * (x_base - x_rest)`. Spring pulling base to commanded pose. | **YES — on base DOFs** |
| fwd | `ff->addForce` | **BeamHookeLawForceField** (strain) | `f_strain -= K_section * (q - q0) * length`. Elastic restoring force. Rest position q0 read from `restPosition` state vector (defaults to initial position, typically zero for straight beam). | **YES — on strain DOFs** |
| fwd (mapped) | `accumulateForce` | MechanicalObject (frames) | Adds externalForce on frames to f. | Only if externalForce set |
| fwd (mapped) | `ff->addForce` | **UniformMass** (frames) | `f_frame += mass * gravity`. With gravity=[0,0,0], contributes **ZERO**. | **NO (gravity=0)** |
| fwd (mapped) | `ff->addForce` | **ConstantForceField** (frames) | `f_frame += forces[tipIdx]`. Scene 1 has forces=zero. | **NO (force=0 in scene 1)** |
| **bwd** | `map->applyJT` | **DiscreteCosseratMapping** | `f_parent += J^T * f_frame`. Maps frame forces back to (base, strain). Since f_frame=0 in scene 1, contributes **ZERO**. | **NO (f_frame=0 in scene 1)** |

Note: `computeForce` backward pass does NOT call `applyDJT` (only `applyJT`). Geometric stiffness is only in `addMBKv`/`addMBKdx`.

**Result of computeForce for scene 1:** `f = [f_base; f_strain] = [-k*(x_base - x_rest); -K*(q - q0)]`

### Step 2b: `addMBKv(b, M(-α), B(0), K(dt+β_solver))` — MechanicalAddMBKdxVisitor

Source: `EulerImplicitSolver.cpp:150-152` → `MechanicalOperations.cpp:310-329`

This call first sets `dx = velocity` (line 317: `mparams.setDx(mparams.v())`), then dispatches `MechanicalAddMBKdxVisitor`.

Parameters on `mparams`: `mFactor = -α`, `bFactor = 0`, `kFactor = (dt + β_solver)`

Visitor impl: `MechanicalAddMBKdxVisitor.cpp`

#### Factor computation formulas (from MechanicalParams.h)

- `kFactorIncludingRayleighDamping(β_ff) = kFactor + bFactor * β_ff`
- `mFactorIncludingRayleighDamping(rayleighMass) = mFactor - bFactor * rayleighMass`

Since `bFactor = 0` for the RHS:
- **All β_ff terms vanish** — force field rayleighStiffness does NOT affect b
- **All rayleighMass-on-mass terms vanish** — mass component rayleighMass does NOT affect b

| Pass | Method called | Component | Internal factor | What it adds to b |
|---|---|---|---|---|
| fwd | `ff->addMBKdx` → `addDForce` | **RestShapeSpringsForceField** (base) | `kFactIncl = (dt+β_solver) + 0 = (dt+β_solver)` | `b_base -= k_spring * v_base * (dt+β_solver)` |
| fwd | `ff->addMBKdx` → `addDForce` | **BeamHookeLawForceField** (strain) | `kFactIncl = (dt+β_solver) + 0*β_ff = (dt+β_solver)` | `b_strain -= K * v_strain * (dt+β_solver)` |
| fwd (mapped) | `ff->addMBKdx` = `Mass::addMBKdx` | **UniformMass** (frames) | Calls BOTH `ForceField::addMBKdx` (→ `addDForce`, no-op for mass) AND `addMDx` with `mFactIncl = -α - 0 = -α` | `b_frame += (-α) * M_frame * v_frame` |
| fwd (mapped) | `ff->addMBKdx` → `addDForce` | **ConstantForceField** (frames) | Derivative of constant = 0. addDForce is no-op. | **ZERO** |
| **bwd** | `map->applyJT` | **DiscreteCosseratMapping** | Maps b_frame back to (base, strain) parent DOFs. | `b_parent += J^T * b_frame` |
| **bwd** | `map->applyDJT` | **DiscreteCosseratMapping** | Geometric stiffness. `applyDJT` is **EMPTY** for Cosserat (`override {}`). | **ZERO** |

#### Expanding the backward mapping (applyJT)

The DiscreteCosseratMapping has two inputs: strain and base. The Jacobian maps:
```
v_frame = J_strain * v_strain + J_base * v_base     (forward)
```

The mass contribution at frame level:
```
b_frame = -α * M_frame * v_frame
        = -α * M_frame * (J_strain * v_strain + J_base * v_base)
```

After `applyJT`, this is distributed to BOTH parent DOF spaces:
```
b_strain += J_strain^T * b_frame
          = -α * (J_strain^T M_frame J_strain) * v_strain    ← M_ss * v_strain
            -α * (J_strain^T M_frame J_base)   * v_base      ← M_sb * v_base

b_base   += J_base^T * b_frame
          = -α * (J_base^T M_frame J_strain) * v_strain      ← M_bs * v_strain
            -α * (J_base^T M_frame J_base)   * v_base        ← M_bb * v_base
```

### Step 2c: `b *= dt`

Source: `EulerImplicitSolver.cpp:155`

### Step 2d: `projectResponse(b)`

Source: `EulerImplicitSolver.cpp:160`

Projects b to the constrained space (FixedProjectiveConstraint, etc).

### Final b formula (full system)

**Physical RHS** (what the C++ source computes, matching `EulerImplicitSolver.cpp:145-155`):

```
b_strain = dt * [ -K*(q - q0)                          ← f_elastic (BeamHookeLaw)
                  - (dt + β_solver) * K * v_strain      ← stiffness-velocity (addMBKv, BeamHookeLaw)
                  - α * M_ss * v_strain                 ← mass-velocity, strain-strain (mapped mass)
                  - α * M_sb * v_base ]                 ← mass-velocity, strain-base COUPLING (mapped mass)

b_base   = dt * [ -K_base * (x_base - x_rest)          ← f_spring (RestShapeSpringsForceField, nonlinear)
                  - (dt + β_solver) * K_base * v_base   ← stiffness-velocity (addMBKv, RestShapeSprings)
                  - α * M_bb * v_base                   ← mass-velocity, base-base (mapped mass)
                  - α * M_bs * v_strain ]               ← mass-velocity, base-strain COUPLING (mapped mass)
```

Note: `K_base` enters `b_base` via the linearized `addDForce`, which is diagonal: `df = -k · dv`.
The *force* `f_spring` is nonlinear (quaternion axis-angle), but the velocity-dependent term
uses the linearized tangent `K_base = diag(k, k, k, k_a, k_a, k_a)`.

**WARNING: Python `solver.b()` / `solver.x()` are NOT the ODE RHS / solution.**

With `FreeMotionAnimationLoop`, the `SparseLDLSolver` is reused by
`GenericConstraintCorrection` after the free-motion ODE solve. By the time
Python reads the linear system vectors in `onAnimateEndEvent`, they contain
constraint correction artifacts, not the ODE vectors:

| Vector | Expected (ODE solve) | Actual (Python reads) |
|---|---|---|
| `solver.b()` | `b_physical` with `‖b‖ ~ O(1e4)` (RestShapeSprings) | Constraint correction residue, `‖b‖ ~ O(1e-15)` |
| `solver.x()` | `dv_free` (velocity increment) | Constraint correction `dv`, `‖x‖ ~ O(1e-15)` |

Evidence (C++ DiagnosticEulerImplicitSolver vs Python):
- `||b_true_base|| ~ 8.5e+04` vs `||b_python_base|| ~ 1.0e-15`
- `||x_true_strain|| ~ 1.4e+00` vs `||x_python_strain|| ~ 1.0e-15`

**To access the true ODE RHS/solution from Python, use `DiagnosticEulerImplicitSolver`**
(see section below), which snapshots the vectors inside `solve()` before
the constraint solver can overwrite them.

The system matrix `solver.A()` is NOT affected — it is rebuilt each step
by `setSystemMBKMatrix` and is correct when read from Python.

### Physical b formula (parameters)

See "Final b formula (full system)" above for both strain and base blocks.

Parameters:
- `K` = BeamHookeLawForceField stiffness (block-diagonal, 32 × 3×3 blocks)
- `K_base` = RestShapeSpringsForceField linearized stiffness = `diag(k, k, k, k_a, k_a, k_a)`
- `q0` = rest position (zero for straight beam)
- `x_rest` = commanded base rest position (updated each step by controller)
- `α` = rayleighMass on solver (0.1)
- `β_solver` = rayleighStiffness on solver (0.7)
- `M_ss = J_strain^T * M_frame * J_strain` (configuration-dependent mapped mass)
- `M_sb = J_strain^T * M_frame * J_base` (base-strain inertial coupling)
- `v_strain`, `v_base` = pre-solve velocities (current state at start of timestep)

Note: `β_ff` (rayleighStiffness on BeamHookeLawForceField) does NOT appear in b because `bFactor = 0`.

## Phase 3: System Matrix A Assembly (setSystemMBKMatrix)

Source: `EulerImplicitSolver.cpp:167-171`

Parameters: `mFact = 1 + dt*α`, `bFact = -dt`, `kFact = -dt*(dt + β_solver)`

Assembly: `MatrixLinearSystem.inl:141-218`

### Local contributions (built on each component's DOF space)

| Contribution type | Component | Factor computation | Matrix contribution |
|---|---|---|---|
| STIFFNESS | **RestShapeSpringsForceField** (base) | `kFactIncl = kFact + bFact*0 = -dt(dt+β_solver)` | Adds to A_bb diagonal |
| STIFFNESS | **BeamHookeLawForceField** (strain) | `kFactIncl = kFact + bFact*β_ff = -dt(dt+β_solver) + (-dt)*β_ff = -dt(dt+β_solver+β_ff)` | Adds `-K * kFactIncl = +dt(dt+β_solver+β_ff)*K` to A_ss |
| MASS | **UniformMass** (frames) | `mFactIncl = mFact - bFact*rayleighMass_on_mass = (1+dt*α) - (-dt)*0 = (1+dt*α)` | Adds `(1+dt*α) * M_frame` to frame-level matrix |
| DAMPING | (no explicit damping components in this scene) | — | — |
| GEOMETRIC_STIFFNESS | **DiscreteCosseratMapping** | `buildGeometricStiffnessMatrix` not implemented | **ZERO** |
| STIFFNESS | **ConstantForceField** (frames) | addKToMatrix is no-op (derivative of constant = 0) | **ZERO** |

### Mapped matrix projection

Source: `MatrixLinearSystem.inl:209-211` → `projectMappedMatrices`

Frame-level matrices are projected to top-level (base+strain) DOFs via:
```
A_projected += J^T * A_frame_local * J
```

### Final A block structure

```
A = [ A_bb    A_bs  ]
    [ A_sb    A_ss  ]

A_ss = (1 + dt*α) * M_ss + dt*(dt + β_solver + β_ff) * K
A_bs = (1 + dt*α) * M_bs                                      (no K cross-coupling)
A_sb = (1 + dt*α) * M_sb = A_bs^T                             (symmetric)
A_bb = (1 + dt*α) * M_bb + RestShapeSprings_terms
```

## Critical Asymmetry: β_ff in A but not in b

| Parameter | In A (setSystemMBKMatrix) | In b (addMBKv) |
|---|---|---|
| `β_solver` (solver rayleighStiffness) | YES: `kFactIncl = -dt(dt + β_solver + β_ff)` | YES: `kFactIncl = (dt + β_solver)` |
| `β_ff` (force field rayleighStiffness) | YES: via `bFact * β_ff` term | **NO**: `bFact = 0`, so β_ff drops out |

This means:
- Effective stiffness damping in A: `β_solver + β_ff` (= 0.7 + 0.7 = 1.4)
- Effective stiffness damping in b: `β_solver` only (= 0.7)

## Phase 4: Constraint Solve (AFTER free motion)

Source: `FreeMotionAnimationLoop.cpp:261-282`

The constraint solver (`ProjectedGaussSeidelConstraintSolver`) solves for Lagrange multipliers (cable tensions, contact forces). These modify velocity and position AFTER the free motion solve. Cable forces (`CableConstraint`) do NOT appear in b or A — they act through the constraint correction step.

## Base Frame Dynamics

### Base DOF Representation

- **Position**: `Rigid3d = (p ∈ R³, q ∈ S³)` — 7 stored values, 6 DOFs
- **Velocity**: `RigidDeriv = (v, ω) ∈ R⁶` — linear + angular, **global frame**, no angle wrapping
- **Integration**: exponential map `q_new = exp(ω·dt/2) · q_old` — singularity-free
  - Source: `RigidCoord.h:108-131`, `operator+=` with `Quat::axisToQuat`
- **Quaternion double-cover**: `q ≡ -q`; SOFA handles via `if (dq[3] < 0) dq *= -1` in `quatToAxis()` (`Quat.inl:249-276`)

### Base Motion: Position-Controlled Rigid Play (current default)

The current `kinematic_play` mode does **not** use a base spring.  For each
actuated scalar (insertion and unwrapped rotation), it first updates a
rate-independent backlash state

```
b[k+1] = clip(b[k], r[k+1] - delta, r[k+1] + delta).
```

At `AnimateBegin`, the motor sample is zero-order held.  The kinematic target
stores `b[k]`, the end-of-step free pose stores `b[k+1]`, and its velocity is
`(b[k+1]-b[k])/dt`.  A `BilateralLagrangianConstraint` attaches the dynamic
Cosserat base to this target.  Consequently:

- `K_base = 0`: there is no transmission compliance in the free-motion matrix;
- the constraint correction uses the full coupled compliance and therefore
  preserves base/strain inertial coupling;
- `bdot` is the step velocity and `bddot=(bdot[k+1]-bdot[k])/dt` is a recorded
  discrete diagnostic, not a separately injected force;
- the first six constraint multipliers give the actuator reaction;
- tanh friction is recorded as additional motor load and does not change the
  motion of an ideal position source.

Implementation: `robots/kinematic_base_actuation.py` and the
`kinematic_play` branch in `robots/catheter.py`.

### Legacy Base Motion: Rest/Penalty Spring

- In `spring_deadzone` or the older fallback, the controller updates
  `rest_position` each step: `q_rest = R_user(θ) · R_home · R_prefab`
  - Source: `controllers/keyboard_controller.py:101-117`, `_apply_base_pose()`
  - `θ` = rotation joint command (±180°), insertion = z-translation
- **Nonlinear force** (addForce, `RestShapeSpringsForceField.inl:396-423`):
  - `dq = q · q_rest⁻¹; normalize(dq); if dq[3] < 0: dq *= -1` (shorter path)
  - Axis-angle: `angle = 2·acos(dq[3])`, `axis = dq[0:3] / sin(angle/2)`
  - `f_rot -= angularStiffness · angle · axis`
  - Translation: `f_trans -= stiffness · (p - p_rest)`
- **Linearized tangent** (addDForce/addKToMatrix, `.inl:441-588`):
  - **Diagonal**: `K_base = diag(k, k, k, k_a, k_a, k_a)`, no cross-coupling between rotation axes
  - addDForce: `df -= k · dv` per component
  - addKToMatrix: diagonal entries only
- With `k = k_a = 1e10`: `A_bb ≈ 7.1e7·I`, `dv_base` rms ≈ 2.8e-4

### Coupling: Base → Strain

- **No direct stiffness coupling**: `K_bs = K_sb = 0`. In kinematic-play mode
  `K_base` is also zero.
- **Mass coupling via Cosserat mapping**: `M_sb = J_strain^T · M_frame · J_base`
  - `||M_sb|| / ||M_ss|| = 31×` — large coupling through mapped mass
  - Enters strain RHS: `b_s += h·(-α)·M_sb·v_base`
- **Cable constraint**: `H_base ≈ 0` (max 5.3e-14) — cable only affects strain DOFs

### Cosserat Mapping Velocity Chain

Source: `DiscreteCosseratMapping.inl:200-298` (applyJ), `BaseCosseratMapping.inl:527-593` (buildProjector, buildAdjoint)

- **Projector**: `P(T) = [[0,R],[R,0]]` — swaps between SOFA `[v,ω]` and Cosserat `[ω,v]` conventions
- **SE(3) adjoint**: `Ad_g = [[R,0],[[p]×R, R]]` — propagation through sections
- **Velocity chain**:
  - `η_0 = P(T_base⁻¹) · v_base` (base velocity in body frame, convention-swapped)
  - `η_i = Ad_{exp(-ξ_i·L_i)} · (η_{i-1} + T_tan,i · ξ̇_i)` (propagation through each section)
  - `v_frame_i = P(T_i) · η_i` (back to SOFA convention for frame DOFs)

### Verified Quantitative Summary

| Finding | Value |
|---|---|
| Geometric stiffness | Exactly 0 (`applyDJT` empty) |
| M_ss variation | 4.6% over trajectory |
| `\|\|M_sb\|\| / \|\|M_ss\|\|` | 31× |
| `\|\|Mv\|\| / \|\|f_elastic\|\|` | 0.5% (mass-proportional damping) |
| `\|\|Kv\|\| / \|\|f_elastic\|\|` | 2.1% (stiffness-proportional damping) |
| dv_base rms | 2.8e-4 (vs dv_strain ~1.4) |
| β_ff asymmetry | In A but NOT in b RHS |
| Cable linearity | B_eff residual 0.53% (nearly linear in λ) |

## Source File Reference

| File | Role |
|---|---|
| `SOFA/.../EulerImplicitSolver.cpp` | Orchestrates b and A assembly |
| `SOFA/.../MechanicalOperations.cpp` | `computeForce`, `addMBKv`, `setSystemMBKMatrix` dispatch |
| `SOFA/.../MechanicalComputeForceVisitor.cpp` | Graph walk for f: `accumulateForce` → `addForce` → `applyJT` |
| `SOFA/.../MechanicalAddMBKdxVisitor.cpp` | Graph walk for MBKv: `addMBKdx` → `applyJT` + `applyDJT` |
| `SOFA/.../MatrixLinearSystem.inl` | System matrix assembly: local + mapped projection |
| `SOFA/.../BaseForceField.cpp` | Default `addMBKdx` → routes to `addDForce`; `addMBKToMatrix` → routes to `addKToMatrix` |
| `SOFA/.../Mass.inl:90-98` | `Mass::addMBKdx` → `ForceField::addMBKdx` (K*dx) + `addMDx` (M*dx) |
| `SOFA/.../MechanicalParams.h:62` | `kFactorIncludingRayleighDamping = kFactor + bFactor * β_ff` |
| `SOFA/.../MechanicalParams.h:64` | `mFactorIncludingRayleighDamping = mFactor - bFactor * rayleighMass` |
| `SOFA/.../MechanicalObject.inl:1308` | `accumulateForce` → adds externalForce vector to f |
| `SOFA/.../FreeMotionAnimationLoop.cpp` | Pre-solve setup; `addToTotalForces(lambda)` — proxy only, NOT injected into f |
| `Cosserat/.../BeamHookeLawForceField.inl` | `addForce`: `f -= K*(q-q0)*L`. `addDForce`: `f -= K*dx*kFact*L` |
| `Cosserat/.../DiscreteCosseratMapping.h:111-113` | `applyDJT` is **empty** (`override {}`) — no geometric stiffness |
| `SOFA/.../UniformMass.inl:472` | `addForce`: `f += mass * gravity` (zero when gravity=0) |
| `SOFA/.../RestShapeSpringsForceField.inl:333` | `addForce`: `f -= k*(x-x_rest)` on base DOFs |
| `SOFA/.../ConstantForceField.inl:337,417` | `addForce`: adds constant force. `addDForce`/`addKToMatrix`: no-op |
| `SOFA/.../AccumulationVecId.inl` | Proxy that sums contributing vectors; used by `readTotalForces()` in mappings |

## DiagnosticEulerImplicitSolver — C++ Ground Truth Intercepts

A subclass of `EulerImplicitSolver` in the Cosserat plugin that snapshots
every stage of the RHS assembly and decomposes `addMBKv` into M-only, K-only,
and geometric stiffness contributions. All reads use the linear system's
`setRHS`/`getSystemRHSBaseVector` dispatch (same mechanism as Python's
`solver.b()`), avoiding the `copyToBaseVector` crash on unallocated
MechanicalStates.

Source: `sofa_plugins/Cosserat/src/Cosserat/solver/DiagnosticEulerImplicitSolver.{h,cpp}`

### Data fields (all accessible from Python via `.value`)

| Data field | When snapshotted | Contents |
|---|---|---|
| `b_after_force` | After `computeForce(f)`, `b = f` | Force vector f (elastic + RestShapeSprings) |
| `b_after_addMBKv` | After `addMBKv(b, M(-α), B(0), K(h+β))` | f + velocity-dependent terms (pre-scaling) |
| `b_after_scale` | After `b *= h` | h-scaled RHS (pre-projection) |
| `b_after_project` | After `projectResponse(b)` | Final projected RHS (= what goes into linear solve) |
| `x_solution` | After `solveSystem` + `dispatchSystemSolution` | dv_free (velocity increment from ODE solve) |
| `b_linsys` | From `getSystemRHSBaseVector` after `setRHS(b)` | Linear system RHS (should = `b_after_project`) |
| `x_linsys` | From `getSystemSolutionBaseVector` after solve | Linear system solution (should = `x_solution`) |

### Decomposed addMBKv fields

Each is the **contribution** (output minus f), not the accumulated result:

| Data field | addMBKv factors | Physical meaning |
|---|---|---|
| `Mv_only` | `M(-α), B(0), K(0)` | `-α * M * v` (mass-proportional damping) |
| `Kv_only` | `M(0), B(0), K(h+β)` | `(h+β) * K * v` (stiffness-proportional damping) |
| `MBKv_full` | `M(-α), B(0), K(h+β)` | Full addMBKv contribution |
| `geom_stiffness` | `MBKv_full - Mv_only - Kv_only` | Nonlinear remainder (geometric stiffness) |

### Control fields

| Data field | Default | Purpose |
|---|---|---|
| `recordInterval` | 1 | Record every N steps |
| `enableDecomposition` | true | Enable the 3 extra addMBKv calls |
| `stepCount` | (output) | Current simulation step |

### Verified properties

All verified to machine precision (`< 1e-15` relative error):

1. `b_after_project == b_linsys` (dispatch round-trip identity)
2. `x_solution == x_linsys` (dispatch round-trip identity)
3. `b_after_scale == dt * b_after_addMBKv` (h-scaling consistency)
4. `MBKv_full == Mv_only + Kv_only` (geometric stiffness = 0 for Cosserat rod)
5. `A * x_linsys == b_linsys` (linear system solved correctly)
6. Simulation trajectory identical with/without diagnostic solver

### Key findings

- **Geometric stiffness = 0**: `applyDJT` is empty for `DiscreteCosseratMapping`,
  and `addMBKv` is perfectly linear. The previously observed 3.5% Python gap
  was caused by reading constraint correction artifacts, not a missing physics term.
- **M_ss is configuration-dependent**: 8–13% variation over trajectory (from
  Jacobian changes as the rod bends). The StructuredNetwork should condition
  the effective mass on the current state z.
- **β_ff does NOT appear in b**: `bFactor = 0` in the RHS addMBKv call, so
  per-force-field rayleighStiffness only enters the system matrix A.

## Correspondence: SOFA Decomposition ↔ StructuredNetwork

The `StructuredNetwork` models: `y = M_eff⁻¹ @ f` where
`f = -K_learned @ (z-z0) - C_learned @ zdot + ctrl + base_terms + residual`.

| SOFA term | Source | StructuredNetwork term |
|---|---|---|
| `f_elastic = -K*q` (from `computeForce`) | `b_after_force` strain block | `-K_learned @ (z-z0)` |
| `-α*M*v` (Mv_only) | `Mv_only` strain block | Part of `-C_learned @ zdot` |
| `(h+β)*K*v` (Kv_only) | `Kv_only` strain block | Part of `-C_learned @ zdot` |
| `geom_stiffness` | `geom_stiffness` (= 0 for Cosserat) | Not needed (absorbed by residual) |
| `f_cable` (cable constraint correction) | `dv_actual - dv_free` | `ctrl` term |
| `M_sb * v_base` (in Mv) | Off-diagonal of mapped mass | `base_terms` |
| `M_ss` (from A decomposition) | `(A_ss - coeff_K*K) / coeff_M` | `M_eff` (should be state-conditioned) |

### Relative magnitudes (typical, freespace trajectory)

```
||f_elastic||     : 1.00   (dominant)
||Kv (stiff damp)||: ~0.05  (5% of elastic)
||Mv (mass damp)|| : ~0.01  (1% of elastic)
||geom_stiffness|| : 0      (exactly zero)
||dv_constraint||  : ~O(dv_free) when tendons active
```
