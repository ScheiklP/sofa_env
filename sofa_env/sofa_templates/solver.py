import Sofa.Core
from enum import Enum
from typing import Optional, Tuple

SOLVER_PLUGIN_LIST = [
    "Sofa.Component.LinearSolver.Direct",  # <- [AsyncSparseLDLSolver, BTDLinearSolver, SparseLDLSolver]
    "Sofa.Component.ODESolver.Backward",  # <- [EulerImplicitSolver]
    "Sofa.Component.ODESolver.Forward",  # [EulerExplicitSolver]
    "Sofa.Component.LinearSolver.Iterative",  # <- [CGLinearSolver]
]


class OdeSolverType(Enum):
    """Describes the numerical method to find the approximate solution for ordinary differential equations.

    Explicit solvers require small time steps, but are fast.
    Implicit solvers are slow, but much more stable.

    Warning:
        The listed solvers are not all considered in the Enum.
        The list is mostly for reference of existing solvers.

    Note:
        Rayleigh mass and Rayleigh stiffnes. From the [SOFA Documentation](www.sofa-framework.org/community/doc/simulation-principles/system-resolution/integration-scheme/):
        The Rayleigh damping is a numerical damping. This damping has therefore no physical meaning and must not be mixed up with physical damping.
        This numerical damping is usually used to stabilize or ease convergence of the simulation. However, it has to be used carefully.

    Explicit Solvers:
        - EulerExplicitSolver
        - CentralDifferenceSolver
        - RungeKutta2Solver

    Implicit Solvers:
        - EulerImplicitSolver
        - NewmarkImplicitSolver
        - VariationalSymplecticSolver

    TLDR:
        Use the EulerImplicitSolver.
    """

    EXPLICITEULER = "EulerExplicitSolver"
    IMPLICITEULER = "EulerImplicitSolver"


class LinearSolverType(Enum):
    """Describes the numerical methods that solves the matrix system Ax=b that is built by the OdeSolver.

    Direct solvers find exact solutions, but may be slow for large systems.
    Iterative solvers converge to an approximate solution and require additional settings.

    Warning:
        The listed solvers are not all considered in the Enum.
        The list is mostly for reference of existing solvers.

    Direct Solvers:
        - SparseLDLSolver
        - AsyncSparseLDLSolver
        - SparseLUSolver
        - CholeskySolver
        - SVDLinearSolver
        - BTDLinearSolver

    Iterative Solvers:
        - CGLinearSolver
        - ShewchukPCGLinearSolver
        - MinResLinearSolver

    TLDR:
        Use the CGLinearSolver.
    """

    CG = "CGLinearSolver"
    SPARSELDL = "SparseLDLSolver"
    ASYNCSPARSELDL = "AsyncSparseLDLSolver"
    BTD = "BTDLinearSolver"


LINEAR_SOLVER_DEFAULT_KWARGS = {
    LinearSolverType.CG: {
        "iterations": 25,
        "threshold": 1e-9,
        "tolerance": 1e-9,
    },
    LinearSolverType.SPARSELDL: {"template": "CompressedRowSparseMatrixMat3x3d"},
    LinearSolverType.ASYNCSPARSELDL: {},
    LinearSolverType.BTD: {"template": "BTDMatrix6d"},
}


class ConstraintCorrectionType(Enum):
    """SOFA names of the different types of constraint correction.

    Notes:
        UNCOUPLED is recommended for rigid objects.
        PRECOMPUTED is recommended for deformable objects. This will create a file on the first creation of the scene. Computation may take a few minutes.
        LINEAR is the most accurate but also computationally expensive.
        GENERIC is similar to LINEAR, but computes a global Matrix instead of a local per object matrix.

    Warning:
        LINEAR and GENERIC require the objects to have DIRECT linear solvers. See documentation of ``LinearSolverType``.
    """

    UNCOUPLED = "UncoupledConstraintCorrection"
    LINEAR = "LinearSolverConstraintCorrection"
    PRECOMPUTED = "PrecomputedConstraintCorrection"
    GENERIC = "GenericConstraintCorrection"


def add_solver(
    attached_to: Sofa.Core.Node,
    ode_solver_type: OdeSolverType = OdeSolverType.IMPLICITEULER,
    ode_solver_rayleigh_stiffness: float = 0.1,
    ode_solver_rayleigh_mass: float = 0.1,
    linear_solver_type: LinearSolverType = LinearSolverType.CG,
    linear_solver_kwargs: Optional[dict] = None,
) -> Tuple[Sofa.Core.Object, Sofa.Core.Object]:
    """Adds a time integration scheme and a linear solver to a node.

    Args:
        ode_solver_type (OdeSolverType): See documentation of ``OdeSolverType``.
        ode_solver_rayleigh_stiffness (float): See documentation of ``OdeSolverType``.
        ode_solver_rayleigh_mass (float): See documentation of ``OdeSolverType``.
        linear_solver_type (LinearSolverType): See documentation of ``LinearSolverType``.
        linear_solver_kwargs (Optional[dict]): Additional keyword arguments to the LinearSolverType. If ``None``, read from ``LINEAR_SOLVER_DEFAULT_KWARGS``.

    Returns:
        ode_solver (Sofa.Core.Object):
        linear_solver (Sofa.Core.Object):
    """

    # Integration scheme
    if ode_solver_type == OdeSolverType.EXPLICITEULER:
        ode_solver = attached_to.addObject(
            ode_solver_type.value,
        )
    else:
        ode_solver = attached_to.addObject(
            ode_solver_type.value,
            rayleighMass=ode_solver_rayleigh_mass,
            rayleighStiffness=ode_solver_rayleigh_stiffness,
        )

    # Linear solver
    if linear_solver_kwargs is None:
        linear_solver_kwargs = LINEAR_SOLVER_DEFAULT_KWARGS[linear_solver_type]

    linear_solver = attached_to.addObject(linear_solver_type.value, **linear_solver_kwargs)

    return ode_solver, linear_solver
