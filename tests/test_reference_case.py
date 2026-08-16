"""Regression test of the Python port against the original Fortran results.

``doc/ex-out.txt`` is the trusted reference: it is the output the original
single-precision ``fe2quad.f`` produced for ``doc/ex-data.txt``.  Every number
compared here is parsed out of that file rather than transcribed by hand.

Tolerances
----------
The reference was computed in single precision (about 7 significant digits) and
printed with 4, so a relative tolerance below ~5e-4 is not meaningful.  The
comparisons use ``rtol = 1e-3`` together with an absolute tolerance of ``1e-6``
times the largest value in the field being compared -- see
:func:`tests.fortran_output.noise_floor`.  That absolute floor is required
because three of the printed values are pure single-precision noise on
quantities that are exactly zero by symmetry (node 3 and node 4 y-displacement,
and the reaction at DOF 6, printed as ``-0.3000E-04``).

A second, far tighter comparison runs against
``tests/data/fe2quad-double-precision.txt``, produced by compiling the *same*
Fortran source with ``gfortran -fdefault-real-8`` (REAL promoted to double) and
wider output formats.  That file removes the single-precision noise and lets
the port be checked to ~1e-9 relative.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import (
    DOUBLE_PRECISION_OUTPUT,
    EXAMPLE_INPUT,
    EXAMPLE_OUTPUT,
    PLANE_STRAIN_OUTPUT,
)
from fe2quad import AnalysisType, parse_fe2quad_input, solve
from fortran_output import noise_floor, read_fortran_output

#: Tolerance for the single-precision reference in doc/ex-out.txt.  The observed
#: worst case is 3.8e-4 on the displacements, which is the quantisation of the
#: 4-digit printed values rather than a difference in the computation.
SINGLE_PRECISION_RTOL = 1e-3

#: Tolerance for the double-precision rerun.  Observed worst cases are 2.5e-15
#: (displacements), 4.1e-16 (reactions) and 1.7e-15 (stresses).
DOUBLE_PRECISION_RTOL = 1e-12

#: The principal stress *angle* is the one quantity that cannot reach that
#: level: ``prstrs`` converts radians to degrees with the 10-digit literal
#: 57.29577951, which differs from 180/pi by 5.4e-11 relative.  The Python port
#: uses math.degrees, so the angles agree only to about 1e-10.
ANGLE_RTOL = 1e-9


@pytest.fixture(scope="module")
def reference():
    """Parsed contents of ``doc/ex-out.txt``."""
    return read_fortran_output(EXAMPLE_OUTPUT)


@pytest.fixture(scope="module")
def double_precision_reference():
    """Parsed output of the same program built with REAL promoted to double."""
    return read_fortran_output(DOUBLE_PRECISION_OUTPUT)


# --------------------------------------------------------------------------
# The input echo: the Python parser must read the same model the Fortran read
# --------------------------------------------------------------------------

def test_model_matches_the_fortran_input_echo(example_model, reference):
    assert example_model.analysis_type.label == reference.analysis_type_label
    assert example_model.num_elements == reference.counts["NE"]
    assert example_model.num_nodes == reference.counts["NN"]
    assert len(example_model.prescribed_displacements) == reference.counts["ND"]
    assert len(example_model.loads) == reference.counts["NL"]
    assert len(example_model.materials) == reference.counts["NM"]

    np.testing.assert_allclose(example_model.coordinates, reference.coordinates, rtol=1e-6)
    assert example_model.thickness == pytest.approx(reference.thickness)

    materials = np.array([[m.youngs_modulus, m.poissons_ratio] for m in example_model.materials])
    np.testing.assert_allclose(materials, reference.materials, rtol=1e-6)

    assert [(bc.dof, bc.value) for bc in example_model.prescribed_displacements] == reference.prescribed
    assert [(load.dof, load.value) for load in example_model.loads] == reference.loads

    # Column 0 is the element number, columns 1..4 the nodes, column 5 the material.
    for element in range(example_model.num_elements):
        row = reference.connectivity[element]
        assert row[0] == element + 1
        np.testing.assert_array_equal(example_model.element_node_numbers(element), row[1:5])
        assert example_model.material_number(element) == row[5]


def test_half_bandwidth_matches(example_result, reference):
    """The banded storage is gone, but the reported half bandwidth must still match."""
    assert example_result.half_bandwidth == reference.half_bandwidth == 8


# --------------------------------------------------------------------------
# The results
# --------------------------------------------------------------------------

def test_displacements_match_the_fortran_output(example_result, reference):
    expected = reference.displacement_vector
    np.testing.assert_allclose(
        example_result.displacements,
        expected,
        rtol=SINGLE_PRECISION_RTOL,
        atol=noise_floor(expected),
    )


def test_reactions_match_the_fortran_output(example_model, example_result, reference):
    dofs = [bc.dof for bc in example_model.prescribed_displacements]
    expected = reference.reactions_for(dofs)
    np.testing.assert_allclose(
        example_result.reactions,
        expected,
        rtol=SINGLE_PRECISION_RTOL,
        atol=noise_floor(expected),
    )


def test_element_stresses_match_the_fortran_output(example_result, reference):
    expected = reference.stresses[:, 0:3]  # SX, SY, TXY
    np.testing.assert_allclose(
        example_result.element_stresses,
        expected,
        rtol=SINGLE_PRECISION_RTOL,
        atol=noise_floor(expected),
    )


def test_principal_stresses_and_angle_match_the_fortran_output(example_result, reference):
    expected = reference.stresses[:, 3:6]  # S1, S2, ANGLE
    np.testing.assert_allclose(
        example_result.principal_stresses,
        expected,
        rtol=SINGLE_PRECISION_RTOL,
        atol=noise_floor(expected),
    )


# --------------------------------------------------------------------------
# Spot checks on the individual values quoted in the reference output
# --------------------------------------------------------------------------

def test_quoted_nodal_displacements(example_result):
    """The six node displacements printed in ex-out.txt.

    Node 3 and node 4 have zero y-displacement by symmetry; the reference
    prints single-precision noise (0.3640E-13 and 0.2033E-08) for them, so they
    are checked against zero with the noise floor instead.
    """
    ux = example_result.nodal_displacements[:, 0]
    uy = example_result.nodal_displacements[:, 1]

    np.testing.assert_allclose(ux, [0.1094e-6, 0.9330e-2, 0.3878e-6, 0.1259e-1, 0.1094e-6, 0.9330e-2],
                               rtol=1e-3)
    np.testing.assert_allclose(uy[[0, 1, 4, 5]], [0.6311e-7, 0.1533e-2, -0.6311e-7, -0.1533e-2], rtol=1e-3)
    assert uy[2] == pytest.approx(0.0, abs=1e-12)
    assert uy[3] == pytest.approx(0.0, abs=1e-12)


def test_quoted_reactions(example_model, example_result):
    """The reactions printed in ex-out.txt, keyed by DOF number."""
    reactions = dict(zip((bc.dof for bc in example_model.prescribed_displacements), example_result.reactions))

    assert reactions[1] == pytest.approx(-0.9018e2, rel=1e-3)
    assert reactions[2] == pytest.approx(-0.5201e2, rel=1e-3)
    assert reactions[5] == pytest.approx(-0.3196e3, rel=1e-3)
    assert reactions[9] == pytest.approx(-0.9018e2, rel=1e-3)
    assert reactions[10] == pytest.approx(0.5201e2, rel=1e-3)
    # DOF 6 is zero by symmetry; ex-out.txt prints -0.3000E-04 of single-precision noise.
    assert reactions[6] == pytest.approx(0.0, abs=1e-4)


def test_quoted_element_stresses(example_result):
    """The two stress rows printed in ex-out.txt."""
    np.testing.assert_allclose(
        example_result.element_stresses,
        [[0.3333e2, 0.6935e1, 0.3691e1], [0.3333e2, 0.6935e1, -0.3691e1]],
        rtol=1e-3,
    )
    np.testing.assert_allclose(
        example_result.principal_stresses,
        [[0.3384e2, 0.6428e1, 0.7811e1], [0.3384e2, 0.6428e1, -0.7811e1]],
        rtol=1e-3,
    )


def test_symmetry_of_the_reference_model(example_result):
    """The model and loading are symmetric about y = 10, so the solution must be.

    Nodes 1 and 5 (and 2 and 6) are mirror images: equal x-displacement and
    opposite y-displacement.  This is what makes the near-zero reference values
    identifiable as noise rather than physics.
    """
    u = example_result.nodal_displacements
    assert u[0, 0] == pytest.approx(u[4, 0], rel=1e-12)
    assert u[0, 1] == pytest.approx(-u[4, 1], rel=1e-12)
    assert u[1, 0] == pytest.approx(u[5, 0], rel=1e-12)
    assert u[1, 1] == pytest.approx(-u[5, 1], rel=1e-12)


# --------------------------------------------------------------------------
# High-precision comparison against the same program built in double precision
# --------------------------------------------------------------------------

def test_displacements_match_the_double_precision_rerun(example_result, double_precision_reference):
    expected = double_precision_reference.displacement_vector
    np.testing.assert_allclose(
        example_result.displacements,
        expected,
        rtol=DOUBLE_PRECISION_RTOL,
        atol=noise_floor(expected, DOUBLE_PRECISION_RTOL),
    )


def test_reactions_match_the_double_precision_rerun(example_model, example_result, double_precision_reference):
    dofs = [bc.dof for bc in example_model.prescribed_displacements]
    expected = double_precision_reference.reactions_for(dofs)
    np.testing.assert_allclose(
        example_result.reactions,
        expected,
        rtol=DOUBLE_PRECISION_RTOL,
        atol=noise_floor(expected, DOUBLE_PRECISION_RTOL),
    )


def test_stresses_match_the_double_precision_rerun(example_result, double_precision_reference):
    """SX, SY, TXY, S1 and S2 must agree to near machine precision."""
    expected = double_precision_reference.stresses[:, 0:5]
    computed = np.hstack((example_result.element_stresses, example_result.principal_stresses[:, 0:2]))
    np.testing.assert_allclose(
        computed,
        expected,
        rtol=DOUBLE_PRECISION_RTOL,
        atol=noise_floor(expected, DOUBLE_PRECISION_RTOL),
    )


def test_principal_angle_matches_the_double_precision_rerun(example_result, double_precision_reference):
    """The angle is limited by the truncated degrees constant in ``prstrs``."""
    expected = double_precision_reference.stresses[:, 5]
    np.testing.assert_allclose(
        example_result.principal_stresses[:, 2],
        expected,
        rtol=ANGLE_RTOL,
        atol=noise_floor(expected, ANGLE_RTOL),
    )


# --------------------------------------------------------------------------
# Plane strain: the branch of ``cmat`` the reference case never reaches
# --------------------------------------------------------------------------

def test_plane_strain_matches_the_fortran_program():
    """Re-analyse the reference model as plane strain and compare with Fortran.

    ``doc/ex-data.txt`` selects plane stress, so the plane strain branch of
    ``cmat`` is never exercised by the shipped reference case.  This runs the
    same model with ``lc = 2`` against the output of the original program built
    in double precision with that same change.
    """
    with open(EXAMPLE_INPUT, "r") as handle:
        lines = handle.read().splitlines()
    lines[0] = lines[0].replace("1", "2", 1)  # lc: plane stress -> plane strain
    model = parse_fe2quad_input("\n".join(lines) + "\n")
    assert model.analysis_type is AnalysisType.PLANE_STRAIN

    result = solve(model)
    expected = read_fortran_output(PLANE_STRAIN_OUTPUT)

    np.testing.assert_allclose(
        result.displacements,
        expected.displacement_vector,
        rtol=DOUBLE_PRECISION_RTOL,
        atol=noise_floor(expected.displacement_vector, DOUBLE_PRECISION_RTOL),
    )
    dofs = [bc.dof for bc in model.prescribed_displacements]
    np.testing.assert_allclose(
        result.reactions,
        expected.reactions_for(dofs),
        rtol=DOUBLE_PRECISION_RTOL,
        atol=noise_floor(expected.reactions_for(dofs), DOUBLE_PRECISION_RTOL),
    )
    np.testing.assert_allclose(
        np.hstack((result.element_stresses, result.principal_stresses[:, 0:2])),
        expected.stresses[:, 0:5],
        rtol=DOUBLE_PRECISION_RTOL,
    )
    np.testing.assert_allclose(result.principal_stresses[:, 2], expected.stresses[:, 5], rtol=ANGLE_RTOL)

    # Plane strain restrains the out-of-plane strain, so SY rises above the
    # plane stress value of 6.9347 while SX is fixed by equilibrium.
    assert result.element_stresses[0, 1] > 9.0


# --------------------------------------------------------------------------
# The reference case must also solve correctly with the alternative BC method
# --------------------------------------------------------------------------

def test_elimination_reproduces_the_penalty_solution(example_model, example_result):
    """Direct elimination and the penalty method agree on everything that matters.

    They differ only in the tiny displacements the penalty method leaves at the
    restrained degrees of freedom, which elimination sets to exactly zero.
    """
    eliminated = solve(example_model, boundary_method="elimination")
    free = [dof for dof in range(example_model.num_dof)
            if dof not in {bc.index for bc in example_model.prescribed_displacements}]

    # Node 4 has zero y-displacement by symmetry, so the free DOFs need the same
    # absolute floor as the comparisons against the Fortran output.
    np.testing.assert_allclose(
        eliminated.displacements[free],
        example_result.displacements[free],
        rtol=1e-4,
        atol=noise_floor(example_result.displacements, 1e-12),
    )
    np.testing.assert_allclose(eliminated.reactions, example_result.reactions, rtol=1e-4, atol=1e-9)
    np.testing.assert_allclose(
        eliminated.element_stresses, example_result.element_stresses, rtol=1e-4
    )
    for bc in example_model.prescribed_displacements:
        assert eliminated.displacements[bc.index] == pytest.approx(bc.value, abs=1e-30)
