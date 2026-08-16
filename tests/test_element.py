"""Tests for the Q4 element: integration rule, Jacobian, B matrix and stiffness.

Fortran routines under test: ``integ``, ``bmat`` and ``elstif``.
"""

from __future__ import annotations

import numpy as np
import pytest

from fe2quad import (
    AnalysisType,
    ElementGeometryError,
    Material,
    constitutive_matrix,
    element_stiffness,
    gauss_points_2x2,
    jacobian,
    shape_function_derivatives,
    strain_displacement_matrix,
)

MATERIAL = Material(youngs_modulus=30000.0, poissons_ratio=0.3)
C_PLANE_STRESS = constitutive_matrix(MATERIAL, AnalysisType.PLANE_STRESS)

WIDTH = 2.0
HEIGHT = 3.0
#: A rectangle with corners numbered counter-clockwise from the origin.
RECTANGLE = np.array([[0.0, 0.0], [WIDTH, 0.0], [WIDTH, HEIGHT], [0.0, HEIGHT]])
#: A general (non-rectangular) but valid convex quadrilateral.
SKEWED = np.array([[0.0, 0.0], [2.5, 0.3], [2.1, 3.4], [-0.2, 2.8]])


def rigid_body_modes(coordinates: np.ndarray) -> np.ndarray:
    """Return the three rigid body displacement vectors for an element.

    Translation in x, translation in y and an infinitesimal rotation
    ``u = -y, v = x``, each expressed in the element DOF ordering
    ``[u1, v1, u2, v2, u3, v3, u4, v4]``.
    """
    x, y = coordinates[:, 0], coordinates[:, 1]
    modes = np.zeros((3, 8))
    modes[0, 0::2] = 1.0
    modes[1, 1::2] = 1.0
    modes[2, 0::2] = -y
    modes[2, 1::2] = x
    return modes


# --------------------------------------------------------------------------
# Gaussian integration (Fortran ``integ``)
# --------------------------------------------------------------------------

def test_gauss_points_are_the_four_combinations_of_one_over_sqrt_three():
    points, weights = gauss_points_2x2()

    assert points.shape == (4, 2)
    assert weights.shape == (4,)
    c = 1.0 / np.sqrt(3.0)
    np.testing.assert_allclose(np.abs(points), c, rtol=1e-15, atol=0.0)
    np.testing.assert_allclose(weights, 1.0, rtol=0.0, atol=0.0)
    # All four sign combinations appear exactly once, in the Fortran order.
    np.testing.assert_allclose(points, np.array([[-c, -c], [c, -c], [c, c], [-c, c]]), rtol=1e-15)


def test_gauss_rule_integrates_cubic_polynomials_exactly():
    """A 2x2 rule is exact for polynomials up to degree 3 in each variable."""
    points, weights = gauss_points_2x2()
    # integral of (1 + s + t + s*t + s^2 + t^2 + s^2 t^2) over [-1,1]^2 = 4 + 4/3 + 4/3 + 4/9
    integrand = lambda s, t: 1.0 + s + t + s * t + s**2 + t**2 + (s**2) * (t**2)  # noqa: E731
    numeric = sum(w * integrand(s, t) for (s, t), w in zip(points, weights))
    assert numeric == pytest.approx(4.0 + 4.0 / 3.0 + 4.0 / 3.0 + 4.0 / 9.0, rel=1e-14)


def test_shape_function_derivatives_sum_to_zero():
    """The shape functions form a partition of unity, so their derivatives cancel."""
    for s, t in [(0.0, 0.0), (0.3, -0.7), (1.0, 1.0)]:
        dn = shape_function_derivatives(s, t)
        assert dn.shape == (2, 4)
        np.testing.assert_allclose(dn.sum(axis=1), 0.0, atol=1e-15)


# --------------------------------------------------------------------------
# Jacobian
# --------------------------------------------------------------------------

def test_jacobian_of_a_rectangle_is_constant_and_diagonal():
    """For a rectangle J = diag(a/2, b/2) and det(J) = area/4 everywhere."""
    for s, t in [(0.0, 0.0), (-0.5, 0.5), (1.0, -1.0)]:
        jac, det_j = jacobian(RECTANGLE, s, t)
        np.testing.assert_allclose(jac, np.diag([WIDTH / 2.0, HEIGHT / 2.0]), atol=1e-15)
        assert det_j == pytest.approx(WIDTH * HEIGHT / 4.0, rel=1e-14)


def test_jacobian_determinant_integrates_to_the_element_area():
    """sum(det(J) * w) over the Gauss points is the area of the quadrilateral."""
    points, weights = gauss_points_2x2()
    area = sum(w * jacobian(SKEWED, s, t)[1] for (s, t), w in zip(points, weights))
    # Shoelace formula for the same polygon.
    x, y = SKEWED[:, 0], SKEWED[:, 1]
    shoelace = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    assert area == pytest.approx(shoelace, rel=1e-13)


def test_inverted_element_raises():
    """Clockwise node numbering gives det(J) < 0."""
    inverted = RECTANGLE[::-1]
    with pytest.raises(ElementGeometryError, match="non-positive Jacobian"):
        jacobian(inverted, 0.0, 0.0)


def test_degenerate_element_raises():
    """Collinear nodes give det(J) = 0."""
    collinear = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    with pytest.raises(ElementGeometryError):
        jacobian(collinear, 0.0, 0.0)

    coincident = np.zeros((4, 2))
    with pytest.raises(ElementGeometryError):
        jacobian(coincident, 0.0, 0.0)


def test_geometry_error_identifies_the_element_and_integration_point():
    """The message must say which element and which Gauss point failed."""
    with pytest.raises(ElementGeometryError) as excinfo:
        strain_displacement_matrix(RECTANGLE[::-1], 0.0, 0.0, element_number=7, gauss_point=3)
    message = str(excinfo.value)
    assert "element 7" in message
    assert "integration point 3" in message


def test_element_stiffness_reports_the_failing_element():
    with pytest.raises(ElementGeometryError, match="element 42"):
        element_stiffness(RECTANGLE[::-1], C_PLANE_STRESS, 1.0, element_number=42)


# --------------------------------------------------------------------------
# Strain-displacement matrix (Fortran ``bmat``)
# --------------------------------------------------------------------------

def test_b_matrix_shape_and_sparsity():
    b, det_j = strain_displacement_matrix(RECTANGLE, 0.0, 0.0)

    assert b.shape == (3, 8)
    assert det_j > 0.0
    # eps_x depends only on the u DOFs (even columns), eps_y only on the v DOFs.
    np.testing.assert_allclose(b[0, 1::2], 0.0, atol=1e-15)
    np.testing.assert_allclose(b[1, 0::2], 0.0, atol=1e-15)
    # The shear row combines both, and each column sums to zero over the nodes.
    np.testing.assert_allclose(b.sum(axis=1), 0.0, atol=1e-14)


@pytest.mark.parametrize("coordinates", [RECTANGLE, SKEWED])
@pytest.mark.parametrize("s, t", [(0.0, 0.0), (-0.4, 0.6), (0.8, -0.2)])
def test_b_matrix_annihilates_rigid_body_motion(coordinates, s, t):
    """Translation and infinitesimal rotation produce no strain."""
    b, _ = strain_displacement_matrix(coordinates, s, t)
    for mode in rigid_body_modes(coordinates):
        np.testing.assert_allclose(b @ mode, np.zeros(3), atol=1e-14)


@pytest.mark.parametrize("coordinates", [RECTANGLE, SKEWED])
def test_b_matrix_reproduces_uniform_strain_states(coordinates):
    """A linear displacement field must give exactly the strain that generated it."""
    x, y = coordinates[:, 0], coordinates[:, 1]
    eps_x, eps_y, gamma = 1.5e-3, -4.0e-4, 7.0e-4

    d = np.zeros(8)
    d[0::2] = eps_x * x + gamma * y  # u = eps_x x + gamma y
    d[1::2] = eps_y * y  # v = eps_y y

    for s, t in [(0.0, 0.0), (0.5, -0.5), (-1.0, 1.0)]:
        b, _ = strain_displacement_matrix(coordinates, s, t)
        np.testing.assert_allclose(b @ d, [eps_x, eps_y, gamma], rtol=1e-12, atol=1e-15)


def test_b_matrix_of_a_rectangle_at_the_centroid_has_the_analytical_values():
    """At (0,0) the derivatives of N are +/-1/(2a) and +/-1/(2b) for an a x b rectangle."""
    b, _ = strain_displacement_matrix(RECTANGLE, 0.0, 0.0)
    dx = 1.0 / (2.0 * WIDTH)
    dy = 1.0 / (2.0 * HEIGHT)
    expected_dn_dx = np.array([-dx, dx, dx, -dx])
    expected_dn_dy = np.array([-dy, -dy, dy, dy])

    np.testing.assert_allclose(b[0, 0::2], expected_dn_dx, rtol=1e-14)
    np.testing.assert_allclose(b[1, 1::2], expected_dn_dy, rtol=1e-14)
    np.testing.assert_allclose(b[2, 0::2], expected_dn_dy, rtol=1e-14)
    np.testing.assert_allclose(b[2, 1::2], expected_dn_dx, rtol=1e-14)


# --------------------------------------------------------------------------
# Element stiffness matrix (Fortran ``elstif``)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("coordinates", [RECTANGLE, SKEWED])
def test_element_stiffness_shape_symmetry_and_finiteness(coordinates):
    ke = element_stiffness(coordinates, C_PLANE_STRESS, thickness=1.0)

    assert ke.shape == (8, 8)
    np.testing.assert_allclose(ke, ke.T, rtol=1e-12, atol=1e-9)
    assert np.all(np.isfinite(ke))
    assert np.all(np.diag(ke) > 0.0)


@pytest.mark.parametrize("coordinates", [RECTANGLE, SKEWED])
def test_element_stiffness_is_positive_semidefinite_with_three_zero_modes(coordinates):
    """A fully integrated Q4 has rank 5: 8 DOFs minus 3 rigid body modes."""
    ke = element_stiffness(coordinates, C_PLANE_STRESS, thickness=1.0)
    eigenvalues = np.linalg.eigvalsh(ke)

    scale = np.abs(ke).max()
    assert np.all(eigenvalues > -1e-9 * scale)
    assert np.linalg.matrix_rank(ke, tol=1e-8 * scale) == 5
    for mode in rigid_body_modes(coordinates):
        np.testing.assert_allclose(ke @ mode, np.zeros(8), atol=1e-8 * scale)


def test_element_stiffness_scales_linearly_with_thickness():
    ke = element_stiffness(RECTANGLE, C_PLANE_STRESS, thickness=1.0)
    ke_thick = element_stiffness(RECTANGLE, C_PLANE_STRESS, thickness=2.5)
    np.testing.assert_allclose(ke_thick, 2.5 * ke, rtol=1e-13)


def test_two_by_two_rule_is_exact_for_a_rectangle():
    """Compare the 2x2 stiffness with a 3x3 rule; for a rectangle they must agree.

    This checks the whole ``elstif`` chain (B, det(J), weighting) against an
    independent, higher-order integration of the same integrand.
    """
    nodes = np.array([1.0, 0.0, -1.0]) * np.sqrt(3.0 / 5.0)
    node_weights = np.array([5.0, 8.0, 5.0]) / 9.0

    ke_3x3 = np.zeros((8, 8))
    for s, ws in zip(nodes, node_weights):
        for t, wt in zip(nodes, node_weights):
            b, det_j = strain_displacement_matrix(RECTANGLE, s, t)
            ke_3x3 += b.T @ C_PLANE_STRESS @ b * det_j * ws * wt

    ke_2x2 = element_stiffness(RECTANGLE, C_PLANE_STRESS, thickness=1.0)
    np.testing.assert_allclose(ke_2x2, ke_3x3, rtol=1e-11, atol=1e-9)


def test_uniaxial_patch_gives_the_expected_nodal_forces():
    """Impose a uniform strain state and check Ke d against the exact stress resultant.

    A rectangle of width ``a`` and height ``b`` stretched by ``eps_x`` carries a
    uniform stress ``C[0,0] eps_x``; the total force transmitted across the two
    right-hand nodes must equal ``C[0,0] eps_x * b * thickness``.
    """
    thickness = 1.7
    eps_x = 2.0e-3
    ke = element_stiffness(RECTANGLE, C_PLANE_STRESS, thickness)

    d = np.zeros(8)
    d[0::2] = eps_x * RECTANGLE[:, 0]  # u = eps_x * x, v = 0 (restrained laterally)
    forces = ke @ d

    expected_total = C_PLANE_STRESS[0, 0] * eps_x * HEIGHT * thickness
    right_hand_nodes = [1, 2]  # local nodes at x = WIDTH
    total = sum(forces[2 * node] for node in right_hand_nodes)
    assert total == pytest.approx(expected_total, rel=1e-12)
    np.testing.assert_allclose(forces.sum(), 0.0, atol=1e-9)
