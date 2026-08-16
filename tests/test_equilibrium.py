"""Static equilibrium checks.

Whatever the solver does internally, the applied loads and the computed
reactions must balance in both global directions.  The DOF convention is the
Fortran one: **odd DOF numbers are x, even DOF numbers are y**.
"""

from __future__ import annotations

import numpy as np
import pytest

from fe2quad import dof_direction, solve


def resultant(dof_values) -> dict:
    """Sum ``(dof, value)`` pairs into ``{"X": ..., "Y": ...}`` using the DOF convention."""
    totals = {"X": 0.0, "Y": 0.0}
    for dof, value in dof_values:
        totals[dof_direction(dof)] += float(value)
    return totals


def test_dof_direction_convention():
    """Odd DOF -> x, even DOF -> y, as used throughout the Fortran program."""
    assert [dof_direction(dof) for dof in (1, 2, 3, 4, 11, 12)] == ["X", "Y", "X", "Y", "X", "Y"]


def test_applied_loads_and_reactions_balance(example_model, example_result):
    """sum(applied forces) + sum(reactions) = 0 in both x and y."""
    applied = resultant((load.dof, load.value) for load in example_model.loads)
    reactions = resultant(
        (bc.dof, reaction)
        for bc, reaction in zip(example_model.prescribed_displacements, example_result.reactions)
    )

    # The reference case is loaded in x only: 125 + 250 + 125 = 500.
    assert applied["X"] == pytest.approx(500.0)
    assert applied["Y"] == pytest.approx(0.0)

    scale = max(abs(applied["X"]), abs(applied["Y"]), 1.0)
    assert applied["X"] + reactions["X"] == pytest.approx(0.0, abs=1e-9 * scale)
    assert applied["Y"] + reactions["Y"] == pytest.approx(0.0, abs=1e-9 * scale)


@pytest.mark.parametrize("boundary_method", ["penalty", "elimination"])
def test_equilibrium_holds_for_both_boundary_methods(example_model, boundary_method):
    result = solve(example_model, boundary_method=boundary_method)
    applied = resultant((load.dof, load.value) for load in example_model.loads)
    reactions = resultant(
        (bc.dof, reaction)
        for bc, reaction in zip(example_model.prescribed_displacements, result.reactions)
    )
    assert applied["X"] + reactions["X"] == pytest.approx(0.0, abs=1e-6)
    assert applied["Y"] + reactions["Y"] == pytest.approx(0.0, abs=1e-6)


def test_moment_equilibrium(example_model, example_result):
    """Moments of the applied loads and reactions about the origin also cancel.

    Force components are placed at the node that owns each DOF, so the moment
    about the origin is ``x Fy - y Fx``.
    """

    def moment(dof_values) -> float:
        total = 0.0
        for dof, value in dof_values:
            node = (dof + 1) // 2 - 1  # zero-based node index owning this DOF
            x, y = example_model.coordinates[node]
            total += x * value if dof_direction(dof) == "Y" else -y * value
        return total

    applied = moment((load.dof, load.value) for load in example_model.loads)
    reactions = moment(
        (bc.dof, reaction)
        for bc, reaction in zip(example_model.prescribed_displacements, example_result.reactions)
    )
    assert applied + reactions == pytest.approx(0.0, abs=1e-6 * max(abs(applied), 1.0))


def test_internal_forces_balance_the_external_load(example_model, example_result):
    """K d must reproduce the applied loads at every unrestrained DOF.

    This is the residual check the Fortran program never performed: away from
    the supports the assembled equations must be satisfied exactly.
    """
    from fe2quad import assemble_global_stiffness

    k = assemble_global_stiffness(example_model)
    f = example_model.load_vector()
    residual = k @ example_result.displacements - f

    restrained = {bc.index for bc in example_model.prescribed_displacements}
    free = [dof for dof in range(example_model.num_dof) if dof not in restrained]
    np.testing.assert_allclose(residual[free], 0.0, atol=1e-9 * np.abs(f).max())
