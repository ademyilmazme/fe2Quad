"""Tests for the principal stress calculation (Fortran ``prstrs``).

The angle convention under test is the one printed by the Fortran program as
``ANGLE SX->S1``: degrees, counter-clockwise positive, from the x axis to the
direction of the algebraically larger principal stress ``S1``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fe2quad import principal_stresses, principal_stress_table


def normal_stress_on_plane(sx: float, sy: float, txy: float, angle_degrees: float) -> float:
    """Normal stress on the plane whose outward normal is at ``angle`` from x.

    ``sigma_n = sx cos^2 a + sy sin^2 a + 2 txy sin a cos a``
    """
    a = math.radians(angle_degrees)
    return sx * math.cos(a) ** 2 + sy * math.sin(a) ** 2 + 2.0 * txy * math.sin(a) * math.cos(a)


STRESS_STATES = [
    (100.0, 0.0, 0.0),  # uniaxial tension along x
    (0.0, 100.0, 0.0),  # uniaxial tension along y
    (-40.0, -10.0, 0.0),  # biaxial compression, no shear
    (33.333333, 6.934673, 3.690852),  # reference case, element 1
    (33.333333, 6.934673, -3.690852),  # reference case, element 2
    (10.0, 10.0, 5.0),  # equal normal stresses with shear
    (10.0, 10.0, -5.0),
    (0.0, 0.0, 25.0),  # pure shear
    (0.0, 10.0, 5.0),  # sy > sx with shear: the "c > sx" branch
    (0.0, 10.0, -5.0),
    (-30.0, 60.0, -20.0),
]


@pytest.mark.parametrize("sx, sy, txy", STRESS_STATES)
def test_principal_values_are_the_eigenvalues_of_the_stress_tensor(sx, sy, txy):
    """S1 >= S2 and both are eigenvalues of [[sx, txy], [txy, sy]]."""
    s1, s2, _ = principal_stresses(sx, sy, txy)
    expected = np.linalg.eigvalsh(np.array([[sx, txy], [txy, sy]]))

    assert s1 >= s2
    np.testing.assert_allclose([s2, s1], expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("sx, sy, txy", STRESS_STATES)
def test_invariants_are_preserved(sx, sy, txy):
    """Trace and determinant of the stress tensor are invariant."""
    s1, s2, _ = principal_stresses(sx, sy, txy)
    assert s1 + s2 == pytest.approx(sx + sy, rel=1e-12, abs=1e-12)
    assert s1 * s2 == pytest.approx(sx * sy - txy**2, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("sx, sy, txy", STRESS_STATES)
def test_angle_points_at_the_s1_direction(sx, sy, txy):
    """Rotating to ``angle`` must expose S1, and to ``angle + 90`` must expose S2.

    This is the decisive test of the angle convention: it is independent of how
    the angle is computed and pins down both its sign and its reference axis.
    """
    s1, s2, angle = principal_stresses(sx, sy, txy)

    assert normal_stress_on_plane(sx, sy, txy, angle) == pytest.approx(s1, rel=1e-10, abs=1e-10)
    assert normal_stress_on_plane(sx, sy, txy, angle + 90.0) == pytest.approx(s2, rel=1e-10, abs=1e-10)
    assert -90.0 <= angle <= 90.0


def test_zero_shear_with_sx_greater_than_sy():
    """No shear: the axes are already principal, so the angle is zero."""
    s1, s2, angle = principal_stresses(100.0, 25.0, 0.0)
    assert (s1, s2, angle) == (100.0, 25.0, 0.0)


def test_zero_shear_with_sy_greater_than_sx_swaps_and_reports_ninety_degrees():
    """S1 must be the larger value, so the roles swap and the angle becomes 90."""
    s1, s2, angle = principal_stresses(25.0, 100.0, 0.0)
    assert (s1, s2, angle) == (100.0, 25.0, 90.0)


def test_equal_normal_stresses_without_shear_reports_ninety_degrees():
    """The Fortran ``s1 <= s2`` test is not strict, so an equal state reports 90."""
    s1, s2, angle = principal_stresses(50.0, 50.0, 0.0)
    assert (s1, s2, angle) == (50.0, 50.0, 90.0)


def test_uniaxial_tension_along_x():
    s1, s2, angle = principal_stresses(100.0, 0.0, 0.0)
    assert (s1, s2, angle) == (100.0, 0.0, 0.0)


def test_positive_and_negative_shear_are_mirror_images():
    """Flipping the sign of TXY mirrors the principal direction, values unchanged."""
    s1_pos, s2_pos, angle_pos = principal_stresses(33.333333, 6.934673, 3.690852)
    s1_neg, s2_neg, angle_neg = principal_stresses(33.333333, 6.934673, -3.690852)

    assert s1_pos == pytest.approx(s1_neg, rel=1e-14)
    assert s2_pos == pytest.approx(s2_neg, rel=1e-14)
    assert angle_pos == pytest.approx(-angle_neg, rel=1e-14)
    assert angle_pos > 0.0


def test_equal_normal_stresses_with_shear_give_forty_five_degrees():
    """For sx == sy the principal directions are the 45 degree diagonals."""
    s1, s2, angle = principal_stresses(10.0, 10.0, 5.0)
    assert (s1, s2) == pytest.approx((15.0, 5.0))
    assert angle == pytest.approx(45.0)

    s1, s2, angle = principal_stresses(10.0, 10.0, -5.0)
    assert (s1, s2) == pytest.approx((15.0, 5.0))
    assert angle == pytest.approx(-45.0)


def test_pure_shear():
    """Pure shear: S1 = -S2 = |txy| at 45 degrees."""
    s1, s2, angle = principal_stresses(0.0, 0.0, 25.0)
    assert (s1, s2) == pytest.approx((25.0, -25.0))
    assert angle == pytest.approx(45.0)


def test_sy_larger_than_sx_with_shear_uses_the_complementary_branch():
    """sx = 0, sy = 10, txy = 5: tan(2a) = 2 txy/(sx - sy) = -1 so a = 67.5 degrees."""
    s1, s2, angle = principal_stresses(0.0, 10.0, 5.0)
    assert s1 == pytest.approx(5.0 + math.sqrt(50.0))
    assert s2 == pytest.approx(5.0 - math.sqrt(50.0))
    assert angle == pytest.approx(67.5)

    s1, s2, angle = principal_stresses(0.0, 10.0, -5.0)
    assert angle == pytest.approx(-67.5)


def test_reference_case_element_one_matches_the_fortran_output():
    """The values printed in ex-out.txt for element 1."""
    s1, s2, angle = principal_stresses(0.3333e2, 0.6935e1, 0.3691e1)
    assert s1 == pytest.approx(0.3384e2, rel=1e-3)
    assert s2 == pytest.approx(0.6428e1, rel=1e-3)
    assert angle == pytest.approx(0.7811e1, rel=1e-3)


def test_principal_stress_table_applies_row_wise():
    stresses = np.array([[100.0, 0.0, 0.0], [10.0, 10.0, 5.0]])
    table = principal_stress_table(stresses)

    assert table.shape == (2, 3)
    np.testing.assert_allclose(table[0], [100.0, 0.0, 0.0])
    np.testing.assert_allclose(table[1], [15.0, 5.0, 45.0])
