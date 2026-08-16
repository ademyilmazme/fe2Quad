"""Assembly of the global equations, boundary conditions and solution.

Fortran routines reproduced here
--------------------------------
=========================  ==========================================
``fe2quad.f``              Python
=========================  ==========================================
bandwidth loop in ``MAIN``   :func:`half_bandwidth`
assembly loop in ``MAIN``    :func:`assemble_global_stiffness`
penalty block in ``MAIN``    :func:`apply_penalty_boundary_conditions`
``band``                     :func:`solve_system`
whole ``MAIN`` driver        :func:`solve`
=========================  ==========================================

Storage and solver
------------------
The Fortran program stores the global stiffness matrix in the banded form
``bigk(nq, nbw)`` -- only the diagonal and the ``nbw - 1`` super-diagonals of
the symmetric matrix -- and factorises it with a hand-written banded Gauss
elimination (``band``).  This port assembles the full symmetric ``(nq, nq)``
matrix and calls :func:`numpy.linalg.solve` (LAPACK ``gesv``).  The *governing
finite element equations are unchanged*; only the storage scheme and the
factorisation code differ.  The half bandwidth is still computed and reported
so it can be compared with the Fortran output.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .element import constitutive_matrix, element_stiffness
from .errors import InputError, SolverError
from .model import Model, Result
from .post import element_stresses, principal_stress_table

#: Multiplier applied to ``K[0, 0]`` to build the penalty term (Fortran ``cnst``).
DEFAULT_PENALTY_FACTOR = 100_000.0

#: A matrix whose condition number exceeds this is singular to working precision.
#: The penalty method legitimately raises the condition number by roughly
#: ``DEFAULT_PENALTY_FACTOR``, which leaves ten orders of magnitude of margin.
SINGULARITY_THRESHOLD = 1.0 / np.finfo(float).eps

PENALTY = "penalty"
ELIMINATION = "elimination"


def half_bandwidth(model: Model) -> int:
    """Return the half bandwidth ``nbw`` of the banded Fortran storage scheme.

    For every element the largest node-number span is converted into a DOF
    span::

        nbw = max over elements of 2 * (max_node - min_node + 1)

    The value is informational in this port (see the module docstring) but is
    reported so it can be checked against the Fortran output.
    """
    nbw = 0
    for element in range(model.num_elements):
        nodes = model.element_node_numbers(element)
        span = 2 * (int(nodes.max()) - int(nodes.min()) + 1)
        nbw = max(nbw, span)
    return nbw


def assemble_global_stiffness(model: Model) -> np.ndarray:
    """Assemble and return the full ``(num_dof, num_dof)`` global stiffness matrix.

    The Fortran code scatters each element matrix into banded storage, keeping
    only entries with column >= row; assembling the full symmetric matrix here
    is equivalent.
    """
    k = np.zeros((model.num_dof, model.num_dof), dtype=float)
    for element in range(model.num_elements):
        c = constitutive_matrix(model.material_for_element(element), model.analysis_type)
        ke = element_stiffness(
            model.element_coordinates(element),
            c,
            model.thickness,
            element_number=element + 1,
        )
        dofs = model.element_dofs(element)
        k[np.ix_(dofs, dofs)] += ke
    return k


def penalty_constant(k: np.ndarray, factor: float = DEFAULT_PENALTY_FACTOR) -> float:
    """Return the penalty term ``cnst = K[0, 0] * factor`` (Fortran ``cnst``).

    The original program derives the "relatively large number" from the very
    first diagonal entry of the assembled stiffness matrix, *before* any
    boundary condition has been applied.
    """
    return float(k[0, 0]) * factor


def apply_penalty_boundary_conditions(
    k: np.ndarray,
    f: np.ndarray,
    model: Model,
    factor: float = DEFAULT_PENALTY_FACTOR,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Apply specified displacements by the penalty (large-number) method.

    For every specified displacement ``u`` at DOF ``n``::

        K[n, n] += cnst
        f[n]    += cnst * u

    Returns modified copies of ``k`` and ``f`` plus the penalty constant.
    Repeated DOFs accumulate, exactly as the Fortran loop does.
    """
    cnst = penalty_constant(k, factor)
    k_mod = k.copy()
    f_mod = f.copy()
    for bc in model.prescribed_displacements:
        k_mod[bc.index, bc.index] += cnst
        f_mod[bc.index] += cnst * bc.value
    return k_mod, f_mod, cnst


def _prescribed_map(model: Model) -> Dict[int, float]:
    """Return ``{zero-based dof: value}``, rejecting contradictory duplicates."""
    prescribed: Dict[int, float] = {}
    for bc in model.prescribed_displacements:
        previous = prescribed.get(bc.index)
        if previous is not None and previous != bc.value:
            raise InputError(
                f"DOF {bc.dof} has two different specified displacements "
                f"({previous} and {bc.value}); direct elimination cannot satisfy both"
            )
        prescribed[bc.index] = bc.value
    return prescribed


def solve_system(k: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Solve ``K d = f`` and return the displacement vector (replaces ``band``).

    An under-restrained structure gives a *numerically* singular matrix rather
    than an exactly singular one, and LAPACK happily returns meaningless numbers
    for it with a perfectly small backward error.  The condition number is
    therefore checked before solving; models of the size this program is meant
    for make that check negligible next to the solve itself.

    Raises
    ------
    SolverError
        If the matrix is singular to working precision, which for a well-formed
        model means the structure is insufficiently restrained.
    """
    if not np.all(np.isfinite(k)) or not np.all(np.isfinite(f)):
        raise SolverError("the global equations contain non-finite values (inf or nan)")

    condition = np.linalg.cond(k)
    if not np.isfinite(condition) or condition > SINGULARITY_THRESHOLD:
        raise SolverError(
            f"the global stiffness matrix is singular to working precision "
            f"(condition number {condition:.3g}); check that the model is adequately "
            "restrained against rigid body motion"
        )
    try:
        return np.linalg.solve(k, f)
    except np.linalg.LinAlgError as exc:
        raise SolverError(
            f"the global stiffness matrix is singular ({exc}); "
            "check that the model is adequately restrained against rigid body motion"
        ) from exc


def _solve_by_elimination(
    k: np.ndarray, f: np.ndarray, model: Model
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve with direct elimination of the constrained degrees of freedom.

    Returns the displacement vector and the reactions, the latter ordered like
    :attr:`Model.prescribed_displacements`.  Reactions are recovered from the
    unmodified equations as ``R = K d - f``, which is the same quantity the
    penalty method reports as ``cnst * (u - d)``.
    """
    prescribed = _prescribed_map(model)
    fixed = np.array(sorted(prescribed), dtype=int)
    free = np.setdiff1d(np.arange(model.num_dof), fixed)

    d = np.zeros(model.num_dof, dtype=float)
    d[fixed] = [prescribed[i] for i in fixed]
    if free.size:
        rhs = f[free] - k[np.ix_(free, fixed)] @ d[fixed]
        d[free] = solve_system(k[np.ix_(free, free)], rhs)

    residual = k @ d - f
    reactions = np.array([residual[bc.index] for bc in model.prescribed_displacements])
    return d, reactions


def solve(
    model: Model,
    boundary_method: str = PENALTY,
    penalty_factor: float = DEFAULT_PENALTY_FACTOR,
) -> Result:
    """Run a complete linear static analysis and return a :class:`Result`.

    Parameters
    ----------
    model:
        The finite element model, normally from
        :func:`fe2quad.fileio.read_fe2quad_input`.
    boundary_method:
        ``"penalty"`` (default) reproduces the Fortran large-number treatment
        and is the reference/compatibility mode.  ``"elimination"`` removes the
        constrained equations instead; it satisfies the constraints exactly but
        does **not** reproduce the small penalty displacements that the Fortran
        program prints at restrained degrees of freedom.
    penalty_factor:
        Multiplier used to build ``cnst`` from ``K[0, 0]``; only used by the
        penalty method.
    """
    if boundary_method not in (PENALTY, ELIMINATION):
        raise ValueError(f"boundary_method must be {PENALTY!r} or {ELIMINATION!r}, got {boundary_method!r}")

    k = assemble_global_stiffness(model)
    f = model.load_vector()

    if boundary_method == PENALTY:
        k_mod, f_mod, cnst = apply_penalty_boundary_conditions(k, f, model, penalty_factor)
        displacements = solve_system(k_mod, f_mod)
        # The Fortran program recovers reactions from the penalty term itself.
        reactions = np.array(
            [cnst * (bc.value - displacements[bc.index]) for bc in model.prescribed_displacements]
        )
    else:
        cnst = None
        displacements, reactions = _solve_by_elimination(k, f, model)

    stresses = element_stresses(model, displacements)
    return Result(
        displacements=displacements,
        reactions=reactions,
        element_stresses=stresses,
        principal_stresses=principal_stress_table(stresses),
        half_bandwidth=half_bandwidth(model),
        penalty=cnst,
        boundary_method=boundary_method,
    )
