"""Tests for assembly, bandwidth, boundary conditions and the linear solve."""

from __future__ import annotations

import numpy as np
import pytest

from fe2quad import (
    AnalysisType,
    Material,
    Model,
    ModelError,
    SolverError,
    apply_penalty_boundary_conditions,
    assemble_global_stiffness,
    half_bandwidth,
    parse_fe2quad_input,
    solve,
    solve_system,
)
from fe2quad.solver import DEFAULT_PENALTY_FACTOR

#: One 10x10 element on rollers along its left edge, pulled to the right by 1000.
#: The left edge is free to contract vertically, so the exact answer is the
#: uniform state SX = 1000/(10*1) = 100, SY = TXY = 0.
SINGLE_ELEMENT = """1,1,4,3,2,1
1,0,0
2,10,0
3,10,10
4,0,10
1,0
2,0
7,0
3,500
5,500
1,30000,0.3
1
1,1,2,3,4,1
"""

#: The same element with every corner displacement prescribed to a uniform
#: strain state: u = 0.005 x, v = -nu * 0.005 y.  Free lateral contraction means
#: the exact answer is SX = E * 0.005 = 150, SY = TXY = 0.
PATCH_TEST = """1,1,4,8,0,1
1,0,0
2,10,0
3,10,10
4,0,10
1,0
2,0
3,0.05
4,0
5,0.05
6,-0.015
7,0
8,-0.015
1,30000,0.3
1
1,1,2,3,4,1
"""

#: The same element with no restraints at all.
UNRESTRAINED = """1,1,4,0,2,1
1,0,0
2,10,0
3,10,10
4,0,10
3,500
5,500
1,30000,0.3
1
1,1,2,3,4,1
"""


@pytest.fixture(scope="module")
def single_element_model():
    return parse_fe2quad_input(SINGLE_ELEMENT)


def test_half_bandwidth_of_the_reference_model(example_model):
    """nbw = max over elements of 2*(max node - min node + 1) = 2*4 = 8."""
    assert half_bandwidth(example_model) == 8


def test_half_bandwidth_grows_with_node_number_spread(single_element_model):
    assert half_bandwidth(single_element_model) == 8


def test_global_stiffness_is_symmetric_and_singular_before_restraints(example_model):
    """Without boundary conditions the structure still has its rigid body modes."""
    k = assemble_global_stiffness(example_model)

    assert k.shape == (example_model.num_dof, example_model.num_dof)
    np.testing.assert_allclose(k, k.T, rtol=1e-12, atol=1e-8)
    assert np.linalg.matrix_rank(k, tol=1e-6 * np.abs(k).max()) == example_model.num_dof - 3


def test_global_stiffness_rows_sum_to_zero(example_model):
    """A rigid translation generates no internal force."""
    k = assemble_global_stiffness(example_model)
    scale = np.abs(k).max()
    translation_x = np.zeros(example_model.num_dof)
    translation_x[0::2] = 1.0
    translation_y = np.zeros(example_model.num_dof)
    translation_y[1::2] = 1.0

    np.testing.assert_allclose(k @ translation_x, 0.0, atol=1e-9 * scale)
    np.testing.assert_allclose(k @ translation_y, 0.0, atol=1e-9 * scale)


def test_penalty_modification_touches_only_the_restrained_diagonal(example_model):
    """K[n,n] += cnst and f[n] += cnst*u, nothing else changes."""
    k = assemble_global_stiffness(example_model)
    f = example_model.load_vector()
    k_mod, f_mod, cnst = apply_penalty_boundary_conditions(k, f, example_model)

    assert cnst == pytest.approx(k[0, 0] * DEFAULT_PENALTY_FACTOR)
    difference = k_mod - k
    restrained = [bc.index for bc in example_model.prescribed_displacements]
    np.testing.assert_allclose(np.diag(difference)[restrained], cnst, rtol=1e-12)
    # Everything off the restrained diagonal is untouched.
    expected = np.zeros_like(k)
    expected[restrained, restrained] = cnst
    np.testing.assert_allclose(difference, expected, rtol=1e-12, atol=0.0)
    # All prescribed values are zero here, so the load vector is unchanged.
    np.testing.assert_allclose(f_mod, f)


def test_penalty_does_not_modify_its_inputs(example_model):
    """The helper returns copies; the caller's matrices stay intact."""
    k = assemble_global_stiffness(example_model)
    f = example_model.load_vector()
    k_before, f_before = k.copy(), f.copy()

    apply_penalty_boundary_conditions(k, f, example_model)

    np.testing.assert_array_equal(k, k_before)
    np.testing.assert_array_equal(f, f_before)


def test_non_zero_prescribed_displacements_reproduce_a_uniform_strain_state():
    """Displacement patch test: prescribe u = 0.005x, v = -nu*0.005y at every node.

    The element must return exactly the stress state that generated the field,
    SX = E * 0.005 = 150 with SY = TXY = 0, which exercises non-zero specified
    displacements in both boundary condition treatments.
    """
    model = parse_fe2quad_input(PATCH_TEST)

    penalty = solve(model)
    elimination = solve(model, boundary_method="elimination")

    # The penalty method enforces the value approximately, elimination exactly.
    assert penalty.displacements[2] == pytest.approx(0.05, rel=1e-4)
    assert elimination.displacements[2] == 0.05

    np.testing.assert_allclose(elimination.element_stresses[0], [150.0, 0.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(penalty.element_stresses[0], [150.0, 0.0, 0.0], rtol=1e-4, atol=1e-3)


def test_single_element_uniaxial_response(single_element_model):
    """A 10x10 element pulled by 1000 in total carries SX = 1000/(10*1) = 100."""
    result = solve(single_element_model)

    assert result.element_stresses[0, 0] == pytest.approx(100.0, rel=1e-4)
    assert result.element_stresses[0, 1] == pytest.approx(0.0, abs=1e-4)
    assert result.element_stresses[0, 2] == pytest.approx(0.0, abs=1e-4)
    # Free lateral contraction: eps_y = -nu * eps_x, so the state stays uniaxial.
    assert result.principal_stresses[0, 0] == pytest.approx(100.0, rel=1e-4)
    assert result.principal_stresses[0, 2] == pytest.approx(0.0, abs=1e-4)
    assert sum(result.reactions) == pytest.approx(-1000.0, rel=1e-4)

    # u = eps_x * x with eps_x = 100/30000, so the loaded edge moves by 10*eps_x.
    np.testing.assert_allclose(result.displacements[2], 10.0 * 100.0 / 30000.0, rtol=1e-4)


def test_solve_system_solves_a_small_system():
    k = np.array([[4.0, 1.0], [1.0, 3.0]])
    f = np.array([1.0, 2.0])
    np.testing.assert_allclose(solve_system(k, f), np.linalg.solve(k, f))


def test_singular_system_raises_solver_error():
    with pytest.raises(SolverError, match="singular"):
        solve_system(np.zeros((2, 2)), np.ones(2))


def test_unrestrained_model_raises_solver_error():
    """With no boundary conditions the penalty term is never added."""
    model = parse_fe2quad_input(UNRESTRAINED)
    with pytest.raises(SolverError, match="restrained"):
        solve(model)


def test_unknown_boundary_method_is_rejected(example_model):
    with pytest.raises(ValueError, match="boundary_method"):
        solve(example_model, boundary_method="magic")


def test_result_exposes_the_documented_api(example_model, example_result):
    assert example_result.displacements.shape == (example_model.num_dof,)
    assert example_result.reactions.shape == (len(example_model.prescribed_displacements),)
    assert example_result.element_stresses.shape == (example_model.num_elements, 3)
    assert example_result.principal_stresses.shape == (example_model.num_elements, 3)
    assert example_result.nodal_displacements.shape == (example_model.num_nodes, 2)
    assert example_result.boundary_method == "penalty"
    assert example_result.penalty == pytest.approx(
        assemble_global_stiffness(example_model)[0, 0] * DEFAULT_PENALTY_FACTOR
    )


def test_model_can_be_built_programmatically_without_a_file():
    """The API does not depend on the file format; a plain analysis code works."""
    model = Model(
        analysis_type=1,
        coordinates=np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        connectivity=np.array([[0, 1, 2, 3]]),
        element_material=np.array([0]),
        materials=(Material(1000.0, 0.0),),
        thickness=1.0,
        prescribed_displacements=(),
        loads=(),
    )
    assert model.analysis_type is AnalysisType.PLANE_STRESS  # coerced from the int
    k = assemble_global_stiffness(model)
    assert k.shape == (8, 8)
    np.testing.assert_allclose(k, k.T, rtol=1e-12, atol=1e-10)


def test_invalid_analysis_type_is_rejected():
    with pytest.raises(ModelError, match="analysis_type"):
        Model(
            analysis_type=7,
            coordinates=np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
            connectivity=np.array([[0, 1, 2, 3]]),
            element_material=np.array([0]),
            materials=(Material(1000.0, 0.0),),
            thickness=1.0,
            prescribed_displacements=(),
            loads=(),
        )
