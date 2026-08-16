"""Exceptions raised by the FE2Quad solver.

The original Fortran program performed almost no validation: bad input simply
produced garbage numbers.  The Python port keeps the same numerics but reports
problems through these exceptions instead.
"""

from __future__ import annotations


class FE2QuadError(Exception):
    """Base class for every error raised by this package."""


class InputError(FE2QuadError):
    """The input file could not be parsed, or describes an inconsistent model."""


class ModelError(FE2QuadError):
    """A :class:`~fe2quad.model.Model` is structurally inconsistent."""


class ElementGeometryError(FE2QuadError):
    """An element has a non-positive Jacobian determinant.

    ``det(J) <= 0`` means the element is inverted (nodes numbered clockwise) or
    degenerate (zero area, coincident or collinear nodes).  Either way the
    isoparametric mapping is not invertible and the element stiffness matrix is
    meaningless.
    """


class SolverError(FE2QuadError):
    """The global system of equations could not be solved."""
