"""Tests for the constitutive matrix (Fortran ``cmat``)."""

from __future__ import annotations

import numpy as np
import pytest

from fe2quad import AnalysisType, Material, constitutive_matrix

E = 30000.0
NU = 0.3
STEEL = Material(youngs_modulus=E, poissons_ratio=NU)


def test_plane_stress_matches_analytical_matrix():
    """Plane stress: c1 = E/(1-nu^2), c2 = nu c1, c3 = G."""
    c1 = E / (1.0 - NU**2)
    c2 = NU * c1
    c3 = E / (2.0 * (1.0 + NU))
    expected = np.array([[c1, c2, 0.0], [c2, c1, 0.0], [0.0, 0.0, c3]])

    c = constitutive_matrix(STEEL, AnalysisType.PLANE_STRESS)

    np.testing.assert_allclose(c, expected, rtol=1e-14, atol=0.0)
    # The numbers used by the reference case, spelled out.
    np.testing.assert_allclose(c[0, 0], 30000.0 / 0.91, rtol=1e-14)
    np.testing.assert_allclose(c[0, 1], 0.3 * 30000.0 / 0.91, rtol=1e-14)
    np.testing.assert_allclose(c[2, 2], 15000.0 / 1.3, rtol=1e-14)


def test_plane_strain_matches_analytical_matrix():
    """Plane strain: c1 = E(1-nu)/((1+nu)(1-2nu)), c2 = E nu/((1+nu)(1-2nu))."""
    factor = E / ((1.0 + NU) * (1.0 - 2.0 * NU))
    c1 = factor * (1.0 - NU)
    c2 = factor * NU
    c3 = E / (2.0 * (1.0 + NU))
    expected = np.array([[c1, c2, 0.0], [c2, c1, 0.0], [0.0, 0.0, c3]])

    c = constitutive_matrix(STEEL, AnalysisType.PLANE_STRAIN)

    np.testing.assert_allclose(c, expected, rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(c[0, 0], 30000.0 * 0.7 / 0.52, rtol=1e-14)
    np.testing.assert_allclose(c[0, 1], 30000.0 * 0.3 / 0.52, rtol=1e-14)


@pytest.mark.parametrize("analysis_type", list(AnalysisType))
def test_matrix_is_symmetric_and_decoupled(analysis_type):
    """Normal and shear responses are uncoupled for an isotropic material."""
    c = constitutive_matrix(STEEL, analysis_type)

    assert c.shape == (3, 3)
    np.testing.assert_allclose(c, c.T, rtol=0.0, atol=0.0)
    assert c[0, 2] == 0.0 and c[1, 2] == 0.0
    assert c[2, 0] == 0.0 and c[2, 1] == 0.0
    assert np.all(np.linalg.eigvalsh(c) > 0.0)


@pytest.mark.parametrize("analysis_type", list(AnalysisType))
def test_shear_modulus_is_the_same_for_both_analysis_types(analysis_type):
    """c3 = G = E/(2(1+nu)) regardless of plane stress or plane strain."""
    c = constitutive_matrix(STEEL, analysis_type)
    np.testing.assert_allclose(c[2, 2], E / (2.0 * (1.0 + NU)), rtol=1e-14)


def test_plane_strain_is_stiffer_than_plane_stress():
    """Restraining the out-of-plane strain raises the normal stiffness terms."""
    stress = constitutive_matrix(STEEL, AnalysisType.PLANE_STRESS)
    strain = constitutive_matrix(STEEL, AnalysisType.PLANE_STRAIN)
    assert strain[0, 0] > stress[0, 0]
    assert strain[0, 1] > stress[0, 1]


def test_plane_strain_equals_plane_stress_with_transformed_constants():
    """The classical substitution E -> E/(1-nu^2), nu -> nu/(1-nu)."""
    transformed = Material(
        youngs_modulus=E / (1.0 - NU**2),
        poissons_ratio=NU / (1.0 - NU),
    )
    np.testing.assert_allclose(
        constitutive_matrix(transformed, AnalysisType.PLANE_STRESS),
        constitutive_matrix(STEEL, AnalysisType.PLANE_STRAIN),
        rtol=1e-13,
        atol=0.0,
    )


def test_zero_poisson_ratio_gives_identical_plane_stress_and_plane_strain():
    """With nu = 0 the two formulations coincide."""
    material = Material(youngs_modulus=E, poissons_ratio=0.0)
    np.testing.assert_allclose(
        constitutive_matrix(material, AnalysisType.PLANE_STRESS),
        constitutive_matrix(material, AnalysisType.PLANE_STRAIN),
        rtol=1e-14,
        atol=0.0,
    )


def test_reference_case_material_matches_the_model_file(example_model):
    """The material read from ex-data.txt is E = 30000, nu = 0.3."""
    material = example_model.material_for_element(0)
    assert material.youngs_modulus == pytest.approx(30000.0)
    assert material.poissons_ratio == pytest.approx(0.3)
