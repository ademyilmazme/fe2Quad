"""Stress recovery and principal stress calculation.

Fortran routines reproduced here
--------------------------------
=====================  ==============================================
``fe2quad.f``          Python
=====================  ==============================================
``prstrs``             :func:`principal_stresses`
stress loop in ``MAIN``  :func:`element_stresses`, :func:`principal_stress_table`
=====================  ==============================================
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .element import constitutive_matrix, strain_displacement_matrix
from .model import Model

# The Fortran program recovers stresses at the element centroid only, which is
# the point (s, t) = (0, 0) of the parent element.
CENTROID_S = 0.0
CENTROID_T = 0.0


def element_stresses(model: Model, displacements: np.ndarray) -> np.ndarray:
    """Return the ``(num_elements, 3)`` centroidal stresses ``[SX, SY, TXY]``.

    For every element the Fortran program rebuilds ``C`` and the ``cb = C B``
    matrix at ``s = t = 0``, gathers the element displacement vector ``d`` from
    the global solution, and evaluates ``[SX, SY, TXY] = cb d``.
    """
    stresses = np.zeros((model.num_elements, 3), dtype=float)
    for element in range(model.num_elements):
        c = constitutive_matrix(model.material_for_element(element), model.analysis_type)
        b, _ = strain_displacement_matrix(
            model.element_coordinates(element),
            CENTROID_S,
            CENTROID_T,
            element_number=element + 1,
        )
        d = displacements[model.element_dofs(element)]
        stresses[element] = c @ b @ d
    return stresses


def principal_stresses(sx: float, sy: float, txy: float) -> Tuple[float, float, float]:
    """Return ``(S1, S2, angle)`` for a 2-D stress state (Fortran ``prstrs``).

    ``S1 >= S2`` and ``angle`` is measured in **degrees**, counter-clockwise
    positive, from the x axis to the direction of ``S1`` -- the quantity the
    Fortran program labels ``ANGLE SX->S1``.

    The branch structure of the original routine is preserved exactly, including
    the exact comparison ``txy == 0`` and the two different arctangent forms:

    * ``c <= sx`` (the S1 direction is within +/-45 degrees of the x axis)::

          angle = atan(txy / (sx - S2))

    * otherwise the complementary angle is computed from ``atan(txy/(S1 - sx))``
      and folded into the +/-90 degree range according to the sign of ``txy``.
    """
    if txy == 0.0:
        # Pure normal stress state: the principal directions are the axes.
        s1, s2, angle = sx, sy, 0.0
        if s1 <= s2:
            s1, s2, angle = sy, sx, 90.0
        return s1, s2, angle

    centre = 0.5 * (sx + sy)
    radius = math.sqrt(0.25 * (sx - sy) ** 2 + txy**2)
    s1 = centre + radius
    s2 = centre - radius
    if centre <= sx:
        angle = math.degrees(math.atan(txy / (sx - s2)))
    else:
        angle = math.degrees(math.atan(txy / (s1 - sx)))
        if txy > 0.0:
            angle = 90.0 - angle
        elif txy < 0.0:
            angle = -90.0 - angle
    return s1, s2, angle


def principal_stress_table(stresses: np.ndarray) -> np.ndarray:
    """Apply :func:`principal_stresses` row-wise to an ``(n, 3)`` stress array.

    Returns an ``(n, 3)`` array of ``[S1, S2, ANGLE]``.  The loop is deliberate:
    ``prstrs`` branches on the sign and magnitude of individual components, and
    a vectorised rewrite would obscure the angle convention being reproduced.
    """
    table = np.zeros((stresses.shape[0], 3), dtype=float)
    for row, (sx, sy, txy) in enumerate(stresses):
        table[row] = principal_stresses(float(sx), float(sy), float(txy))
    return table
