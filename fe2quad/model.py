"""Data structures describing an FE2Quad finite element model and its results.

Numbering conventions (identical to the original Fortran program ``fe2quad.f``)
------------------------------------------------------------------------------
* Nodes, elements and materials are numbered from 1 in the *input file*.
* Every node carries two degrees of freedom.  For node ``i`` (1-based)::

      x-displacement -> DOF 2*i - 1   (odd  DOF numbers are the x direction)
      y-displacement -> DOF 2*i       (even DOF numbers are the y direction)

* Element nodes are listed counter-clockwise; the local order (1, 2, 3, 4) maps
  to the parent element corners (-1,-1), (+1,-1), (+1,+1), (-1,+1).
* The element degree-of-freedom vector is
  ``[u1, v1, u2, v2, u3, v3, u4, v4]``.

Internally the arrays stored on :class:`Model` use **zero-based** indices, which
is why the reader converts them on the way in and the report writer converts
them back on the way out.  Fields holding a DOF *number* keep the 1-based value
from the input file and expose a ``.index`` property for the zero-based version;
this keeps the file semantics visible at the call site.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple

import numpy as np

from .errors import ModelError

# Local degree-of-freedom counts for the 4-node quadrilateral.
NODES_PER_ELEMENT = 4
DOF_PER_NODE = 2
DOF_PER_ELEMENT = NODES_PER_ELEMENT * DOF_PER_NODE  # 8


class AnalysisType(IntEnum):
    """Analysis type read from the first field of the input file (``lc``)."""

    PLANE_STRESS = 1
    PLANE_STRAIN = 2

    @property
    def label(self) -> str:
        """Human readable name, as printed by the Fortran program."""
        return "Plane Stress Analysis" if self is AnalysisType.PLANE_STRESS else "Plane Strain Analysis"


def x_dof(node_number: int) -> int:
    """Return the 1-based DOF number of the x-displacement of ``node_number``."""
    return 2 * node_number - 1


def y_dof(node_number: int) -> int:
    """Return the 1-based DOF number of the y-displacement of ``node_number``."""
    return 2 * node_number


def dof_node(dof_number: int) -> int:
    """Return the 1-based node number owning the 1-based ``dof_number``."""
    return (dof_number + 1) // 2


def dof_direction(dof_number: int) -> str:
    """Return ``"X"`` for odd DOF numbers and ``"Y"`` for even ones."""
    return "X" if dof_number % 2 == 1 else "Y"


@dataclass(frozen=True)
class Material:
    """Isotropic linear-elastic material (Fortran array ``pm``)."""

    youngs_modulus: float
    poissons_ratio: float


@dataclass(frozen=True)
class PrescribedDisplacement:
    """A specified nodal displacement (Fortran arrays ``nu`` and ``u``)."""

    dof: int
    """1-based global DOF number, exactly as written in the input file."""

    value: float

    @property
    def index(self) -> int:
        """Zero-based index of :attr:`dof` into the global vectors."""
        return self.dof - 1


@dataclass(frozen=True)
class NodalLoad:
    """A specified nodal load (assembled into the Fortran vector ``f``)."""

    dof: int
    """1-based global DOF number, exactly as written in the input file."""

    value: float

    @property
    def index(self) -> int:
        """Zero-based index of :attr:`dof` into the global vectors."""
        return self.dof - 1


@dataclass(frozen=True)
class Model:
    """A complete FE2Quad model.

    Attributes
    ----------
    analysis_type:
        Plane stress or plane strain.
    coordinates:
        ``(num_nodes, 2)`` array of nodal x/y coordinates, ordered by node
        number (row 0 is node 1).  Fortran array ``x``.
    connectivity:
        ``(num_elements, 4)`` array of **zero-based** node indices, ordered by
        element number.  Fortran array ``noc`` minus one.
    element_material:
        ``(num_elements,)`` array of **zero-based** indices into
        :attr:`materials`.  Fortran array ``mat`` minus one.
    materials:
        Material properties, ordered by material number.
    thickness:
        Constant element thickness ``th``; the model carries a single value,
        as in the original program.
    prescribed_displacements, loads:
        Boundary conditions and applied nodal loads, in input-file order.
    """

    analysis_type: AnalysisType
    coordinates: np.ndarray
    connectivity: np.ndarray
    element_material: np.ndarray
    materials: Tuple[Material, ...]
    thickness: float
    prescribed_displacements: Tuple[PrescribedDisplacement, ...]
    loads: Tuple[NodalLoad, ...]

    def __post_init__(self) -> None:
        # Accept a plain integer analysis code (1 or 2) and normalise it, so that
        # identity comparisons against AnalysisType are always valid downstream.
        if not isinstance(self.analysis_type, AnalysisType):
            try:
                object.__setattr__(self, "analysis_type", AnalysisType(self.analysis_type))
            except ValueError:
                raise ModelError(
                    f"analysis_type must be 1 (plane stress) or 2 (plane strain), got {self.analysis_type!r}"
                ) from None
        if self.coordinates.ndim != 2 or self.coordinates.shape[1] != 2:
            raise ModelError(f"coordinates must have shape (num_nodes, 2), got {self.coordinates.shape}")
        if self.connectivity.ndim != 2 or self.connectivity.shape[1] != NODES_PER_ELEMENT:
            raise ModelError(f"connectivity must have shape (num_elements, 4), got {self.connectivity.shape}")
        if self.element_material.shape != (self.connectivity.shape[0],):
            raise ModelError(
                "element_material must have one entry per element, got "
                f"{self.element_material.shape} for {self.connectivity.shape[0]} elements"
            )
        if self.connectivity.size and (
            self.connectivity.min() < 0 or self.connectivity.max() >= self.num_nodes
        ):
            raise ModelError("connectivity refers to nodes outside the model")
        if self.element_material.size and (
            self.element_material.min() < 0 or self.element_material.max() >= len(self.materials)
        ):
            raise ModelError("element_material refers to materials outside the model")
        for bc in self.prescribed_displacements:
            self._check_dof(bc.dof, "specified displacement")
        for load in self.loads:
            self._check_dof(load.dof, "specified load")

    def _check_dof(self, dof: int, what: str) -> None:
        if not 1 <= dof <= self.num_dof:
            raise ModelError(f"{what} refers to DOF {dof}, outside the valid range 1..{self.num_dof}")

    @property
    def num_nodes(self) -> int:
        """Fortran ``nn``."""
        return int(self.coordinates.shape[0])

    @property
    def num_elements(self) -> int:
        """Fortran ``ne``."""
        return int(self.connectivity.shape[0])

    @property
    def num_dof(self) -> int:
        """Total number of degrees of freedom, Fortran ``nq = 2*nn``."""
        return DOF_PER_NODE * self.num_nodes

    def element_coordinates(self, element: int) -> np.ndarray:
        """Return the ``(4, 2)`` nodal coordinates of a zero-based ``element``."""
        return self.coordinates[self.connectivity[element]]

    def element_dofs(self, element: int) -> np.ndarray:
        """Return the 8 **zero-based** global DOF indices of a zero-based ``element``.

        The ordering matches the element vector ``[u1, v1, u2, v2, u3, v3, u4, v4]``:
        node ``n`` (zero-based) owns global DOF indices ``2*n`` (x) and ``2*n + 1`` (y).
        """
        nodes = self.connectivity[element]
        dofs = np.empty(DOF_PER_ELEMENT, dtype=int)
        dofs[0::2] = DOF_PER_NODE * nodes
        dofs[1::2] = DOF_PER_NODE * nodes + 1
        return dofs

    def element_node_numbers(self, element: int) -> np.ndarray:
        """Return the 1-based node numbers of a zero-based ``element`` (for reports)."""
        return self.connectivity[element] + 1

    def material_for_element(self, element: int) -> Material:
        """Return the :class:`Material` assigned to a zero-based ``element``."""
        return self.materials[int(self.element_material[element])]

    def material_number(self, element: int) -> int:
        """Return the 1-based material number of a zero-based ``element`` (for reports)."""
        return int(self.element_material[element]) + 1

    def load_vector(self) -> np.ndarray:
        """Build the global load vector ``f``.

        Loads are *assigned* rather than accumulated, reproducing the Fortran
        statement ``READ (linp, *) n, f(n)``: if the same DOF appears twice, the
        last entry wins.
        """
        f = np.zeros(self.num_dof, dtype=float)
        for load in self.loads:
            f[load.index] = load.value
        return f


@dataclass(frozen=True)
class Result:
    """Results of a linear static analysis.

    Attributes
    ----------
    displacements:
        ``(num_dof,)`` global displacement vector, indexed by ``dof - 1``.
    reactions:
        ``(num_prescribed,)`` reaction forces, in the same order as
        :attr:`Model.prescribed_displacements`.
    element_stresses:
        ``(num_elements, 3)`` centroidal stresses ``[SX, SY, TXY]``.
    principal_stresses:
        ``(num_elements, 3)`` values ``[S1, S2, ANGLE]``; ``ANGLE`` is measured
        in degrees from the x axis to the S1 direction.
    half_bandwidth:
        Half bandwidth of the banded Fortran storage scheme.  Reported for
        comparison only -- the Python solver does not use banded storage.
    penalty:
        The large number added to the diagonal by the penalty boundary
        condition treatment (Fortran ``cnst``), or ``None`` for direct
        elimination.
    boundary_method:
        ``"penalty"`` (Fortran-compatible) or ``"elimination"``.
    """

    displacements: np.ndarray
    reactions: np.ndarray
    element_stresses: np.ndarray
    principal_stresses: np.ndarray
    half_bandwidth: int
    penalty: Optional[float]
    boundary_method: str

    @property
    def nodal_displacements(self) -> np.ndarray:
        """``(num_nodes, 2)`` view of :attr:`displacements` as ``[ux, uy]`` per node."""
        return self.displacements.reshape(-1, DOF_PER_NODE)
