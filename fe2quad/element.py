"""Formulation of the 4-node bilinear quadrilateral (Q4) element.

Fortran routines reproduced here
-------------------------------
=====================  =====================================
``fe2quad.f``          Python
=====================  =====================================
``integ``              :func:`gauss_points_2x2`
``cmat``               :func:`constitutive_matrix`
``bmat``               :func:`strain_displacement_matrix`
``elstif``             :func:`element_stiffness`
=====================  =====================================

Parent element
--------------
The element is mapped from a square parent element in natural coordinates
``(s, t) in [-1, 1]^2``.  Local node ``i`` sits at corner ``(s_i, t_i)``::

    node 1 -> (-1, -1)      node 3 -> (+1, +1)
    node 2 -> (+1, -1)      node 4 -> (-1, +1)

    N_i(s, t) = (1 + s_i s)(1 + t_i t) / 4

Strains follow the engineering convention ``[eps_x, eps_y, gamma_xy]`` and the
element displacement vector is ``[u1, v1, u2, v2, u3, v3, u4, v4]``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .errors import ElementGeometryError
from .model import (
    DOF_PER_ELEMENT,
    AnalysisType,
    Material,
    NODES_PER_ELEMENT,
)

# Number of strain components in 2-D: eps_x, eps_y, gamma_xy.
NUM_STRAIN_COMPONENTS = 3


def gauss_points_2x2() -> Tuple[np.ndarray, np.ndarray]:
    """Return the 2x2 Gauss-Legendre points and weights (Fortran ``integ``).

    Returns
    -------
    points:
        ``(4, 2)`` array of ``(s, t)`` coordinates, all combinations of
        ``+/-1/sqrt(3)``, in the original Fortran order (counter-clockwise
        starting at the lower-left corner).
    weights:
        ``(4,)`` array of unit weights.
    """
    c = 1.0 / np.sqrt(3.0)
    points = np.array([[-c, -c], [c, -c], [c, c], [-c, c]], dtype=float)
    weights = np.ones(4, dtype=float)
    return points, weights


def constitutive_matrix(material: Material, analysis_type: AnalysisType) -> np.ndarray:
    """Return the ``(3, 3)`` constitutive matrix ``C`` (Fortran ``cmat``).

    ``C`` relates stresses to strains, ``[SX, SY, TXY] = C [eps_x, eps_y, gamma_xy]``::

        plane stress:  c1 = E/(1 - nu^2)                c2 = c1 * nu
        plane strain:  c1 = E (1 - nu)/((1 + nu)(1 - 2 nu))
                       c2 = E nu     /((1 + nu)(1 - 2 nu))
        both:          c3 = E/(2 (1 + nu)) = G

        C = [[c1, c2,  0],
             [c2, c1,  0],
             [ 0,  0, c3]]
    """
    e = material.youngs_modulus
    nu = material.poissons_ratio
    if AnalysisType(analysis_type) is AnalysisType.PLANE_STRESS:
        c1 = e / (1.0 - nu * nu)
        c2 = c1 * nu
    else:
        cc = e / ((1.0 + nu) * (1.0 - 2.0 * nu))
        c1 = cc * (1.0 - nu)
        c2 = cc * nu
    c3 = 0.5 * e / (1.0 + nu)
    return np.array([[c1, c2, 0.0], [c2, c1, 0.0], [0.0, 0.0, c3]], dtype=float)


def shape_function_derivatives(s: float, t: float) -> np.ndarray:
    """Return ``(2, 4)`` derivatives of the shape functions in natural coordinates.

    Row 0 is ``dN_i/ds`` and row 1 is ``dN_i/dt``; these are the two rows the
    Fortran ``bmat`` routine spreads over its ``g(4, 8)`` array.
    """
    dn_ds = np.array([-(1.0 - t), (1.0 - t), (1.0 + t), -(1.0 + t)]) / 4.0
    dn_dt = np.array([-(1.0 - s), -(1.0 + s), (1.0 + s), (1.0 - s)]) / 4.0
    return np.vstack((dn_ds, dn_dt))


def jacobian(
    element_coordinates: np.ndarray,
    s: float,
    t: float,
    *,
    element_number: Optional[int] = None,
    gauss_point: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """Return the ``(2, 2)`` Jacobian matrix and its determinant at ``(s, t)``.

    ``J = [[dx/ds, dy/ds], [dx/dt, dy/dt]]``, computed as ``dN @ X``.  This is
    algebraically identical to the four explicit ``tj11..tj22`` expressions in
    the Fortran ``bmat`` routine, for example::

        tj11 = ((1 - t)(x2 - x1) + (1 + t)(x3 - x4))/4 == sum_i dN_i/ds * x_i

    Raises
    ------
    ElementGeometryError
        If ``det(J) <= 0``.  The Fortran program silently divided by ``dj`` and
        produced meaningless results for inverted or degenerate elements.
    """
    if element_coordinates.shape != (NODES_PER_ELEMENT, 2):
        raise ValueError(f"element_coordinates must have shape (4, 2), got {element_coordinates.shape}")
    jac = shape_function_derivatives(s, t) @ element_coordinates
    det_j = float(jac[0, 0] * jac[1, 1] - jac[0, 1] * jac[1, 0])
    if det_j <= 0.0:
        raise ElementGeometryError(
            f"non-positive Jacobian determinant det(J) = {det_j:.6g} "
            f"{_location(element_number, gauss_point)} at natural coordinates (s, t) = ({s:.6g}, {t:.6g}); "
            "the element is inverted (nodes must be numbered counter-clockwise) or degenerate. "
            f"Nodal coordinates: {np.array2string(element_coordinates, precision=6)}"
        )
    return jac, det_j


def _location(element_number: Optional[int], gauss_point: Optional[int]) -> str:
    """Format the ``element / integration point`` part of a geometry error message."""
    where = "in element" if element_number is None else f"in element {element_number}"
    if gauss_point is not None:
        where += f", integration point {gauss_point}"
    return where


def strain_displacement_matrix(
    element_coordinates: np.ndarray,
    s: float,
    t: float,
    *,
    element_number: Optional[int] = None,
    gauss_point: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """Return the ``(3, 8)`` strain-displacement matrix ``B`` and ``det(J)``.

    Reproduces the Fortran ``bmat`` routine as the product ``B = A @ G``:

    * ``G`` (4x8) maps the element displacements to the natural-coordinate
      derivatives ``[du/ds, du/dt, dv/ds, dv/dt]``.
    * ``A`` (3x4) maps those derivatives to the strains using the inverse
      Jacobian ``J^-1 = adj(J)/det(J)``.

    ``B`` therefore satisfies ``[eps_x, eps_y, gamma_xy] = B d``.
    """
    jac, det_j = jacobian(
        element_coordinates, s, t, element_number=element_number, gauss_point=gauss_point
    )
    # Inverse of a 2x2 matrix from its adjugate; det_j is already known non-zero.
    jac_inv = np.array([[jac[1, 1], -jac[0, 1]], [-jac[1, 0], jac[0, 0]]]) / det_j

    # a[0] : eps_x   = du/dx = jac_inv[0] . [du/ds, du/dt]
    # a[1] : eps_y   = dv/dy = jac_inv[1] . [dv/ds, dv/dt]
    # a[2] : gamma   = du/dy + dv/dx
    a = np.zeros((NUM_STRAIN_COMPONENTS, 4), dtype=float)
    a[0, 0:2] = jac_inv[0]
    a[1, 2:4] = jac_inv[1]
    a[2, 0:2] = jac_inv[1]
    a[2, 2:4] = jac_inv[0]

    dn = shape_function_derivatives(s, t)
    g = np.zeros((4, DOF_PER_ELEMENT), dtype=float)
    g[0, 0::2] = dn[0]  # du/ds from the u degrees of freedom (even columns)
    g[1, 0::2] = dn[1]  # du/dt
    g[2, 1::2] = dn[0]  # dv/ds from the v degrees of freedom (odd columns)
    g[3, 1::2] = dn[1]  # dv/dt

    return a @ g, det_j


def element_stiffness(
    element_coordinates: np.ndarray,
    constitutive: np.ndarray,
    thickness: float,
    *,
    element_number: Optional[int] = None,
) -> np.ndarray:
    """Return the ``(8, 8)`` element stiffness matrix (Fortran ``elstif``).

    Evaluates ``Ke = sum_ip B^T C B det(J) th w_ip`` over the four 2x2 Gauss
    points.  Note that the *signed* determinant is used, exactly as in the
    Fortran code; :func:`jacobian` guarantees it is positive.
    """
    points, weights = gauss_points_2x2()
    ke = np.zeros((DOF_PER_ELEMENT, DOF_PER_ELEMENT), dtype=float)
    for ip, ((s, t), weight) in enumerate(zip(points, weights), start=1):
        b, det_j = strain_displacement_matrix(
            element_coordinates, s, t, element_number=element_number, gauss_point=ip
        )
        ke += b.T @ (constitutive @ b) * det_j * thickness * weight
    return ke
