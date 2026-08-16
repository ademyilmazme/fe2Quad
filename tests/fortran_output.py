"""Parser for the text output produced by the original ``fe2quad.f`` program.

The reference results are read straight out of ``doc/ex-out.txt`` rather than
being copied into the tests by hand, so the regression test always compares
against the file the Fortran program actually wrote.

The same parser also reads ``tests/data/fe2quad-double-precision.txt``, which
has an identical layout with more printed digits.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

#: Fortran writes small negative values as ``-.3691E+01`` (no leading zero).
_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?"


def _to_float(token: str) -> float:
    return float(token.replace("D", "E").replace("d", "e"))


@dataclass(frozen=True)
class FortranOutput:
    """Everything the Fortran program printed, in machine-readable form.

    Attributes
    ----------
    analysis_type_label:
        ``"Plane Stress Analysis"`` or ``"Plane Strain Analysis"``.
    counts:
        ``{"NE": .., "NN": .., "ND": .., "NL": .., "NM": ..}``.
    coordinates:
        ``(nn, 2)`` echoed nodal coordinates.
    prescribed:
        ``[(dof, value), ...]`` echoed specified displacements.
    loads:
        ``[(dof, value), ...]`` echoed specified loads.
    materials:
        ``(nm, 2)`` echoed ``[E, nu]``.
    thickness, half_bandwidth:
        Echoed scalars.
    connectivity:
        ``(ne, 6)`` echoed ``[element, n1, n2, n3, n4, material]``.
    displacements:
        ``(nn, 2)`` nodal displacements ``[ux, uy]``.
    reactions:
        ``{dof: reaction}``, keyed by the 1-based DOF number.
    stresses:
        ``(ne, 6)`` values ``[SX, SY, TXY, S1, S2, ANGLE]``.
    """

    analysis_type_label: str
    counts: Dict[str, int]
    coordinates: np.ndarray
    prescribed: List[Tuple[int, float]]
    loads: List[Tuple[int, float]]
    materials: np.ndarray
    thickness: float
    half_bandwidth: int
    connectivity: np.ndarray
    displacements: np.ndarray
    reactions: Dict[int, float]
    stresses: np.ndarray

    @property
    def displacement_vector(self) -> np.ndarray:
        """Displacements flattened to the global DOF ordering (DOF 1 first)."""
        return self.displacements.reshape(-1)

    def reactions_for(self, dofs) -> np.ndarray:
        """Reactions for the given 1-based DOF numbers, in the order supplied."""
        return np.array([self.reactions[int(dof)] for dof in dofs])


def _section(lines: List[str], heading: str) -> int:
    """Return the index of the line following the one containing ``heading``."""
    for index, line in enumerate(lines):
        if heading in line:
            return index + 1
    raise AssertionError(f"heading {heading!r} not found in the Fortran output")


def _rows(lines: List[str], start: int, count: int, fields: int) -> List[List[str]]:
    """Collect ``count`` data rows of ``fields`` numbers each, skipping blanks."""
    rows: List[List[str]] = []
    index = start
    while len(rows) < count:
        assert index < len(lines), "the Fortran output ended before the expected rows"
        tokens = re.findall(_NUMBER, lines[index])
        if len(tokens) >= fields:
            rows.append(tokens[:fields])
        index += 1
    return rows


def parse_fortran_output(text: str) -> FortranOutput:
    """Parse the report written by ``fe2quad.f`` into a :class:`FortranOutput`."""
    lines = text.splitlines()

    label = "Plane Stress Analysis" if "Plane Stress Analysis" in text else "Plane Strain Analysis"

    counts_row = _rows(lines, _section(lines, "NE    NN    ND    NL    NM"), 1, 5)[0]
    counts = dict(zip(("NE", "NN", "ND", "NL", "NM"), (int(value) for value in counts_row)))

    start = _section(lines, "X-Coord.")
    coordinates = np.array(
        [[_to_float(row[1]), _to_float(row[2])] for row in _rows(lines, start, counts["NN"], 3)]
    )

    start = _section(lines, "Specified disp.")
    prescribed = [
        (int(row[0]), _to_float(row[1])) for row in _rows(lines, start, counts["ND"], 2)
    ]

    start = _section(lines, "Specified Load")
    loads = [(int(row[0]), _to_float(row[1])) for row in _rows(lines, start, counts["NL"], 2)]

    start = _section(lines, "Material#")
    materials = np.array(
        [[_to_float(row[1]), _to_float(row[2])] for row in _rows(lines, start, counts["NM"], 3)]
    )

    thickness = _to_float(re.findall(_NUMBER, lines[_section(lines, "Thickness =") - 1])[0])

    bandwidth_line = lines[_section(lines, "Half bandwidth is") - 1]
    half_bandwidth = int(re.findall(r"\d+", bandwidth_line)[0])

    start = _section(lines, "-------Nodes-------")
    connectivity = np.array(
        [[int(value) for value in row] for row in _rows(lines, start, counts["NE"], 6)]
    )

    start = _section(lines, "X-Displ")
    displacements = np.array(
        [[_to_float(row[1]), _to_float(row[2])] for row in _rows(lines, start, counts["NN"], 3)]
    )

    start = _section(lines, "Reaction")
    reactions = {
        int(row[0]): _to_float(row[1]) for row in _rows(lines, start, counts["ND"], 2)
    }

    start = _section(lines, "ANGLE SX->S1")
    stresses = np.array(
        [[_to_float(value) for value in row[1:]] for row in _rows(lines, start, counts["NE"], 7)]
    )

    return FortranOutput(
        analysis_type_label=label,
        counts=counts,
        coordinates=coordinates,
        prescribed=prescribed,
        loads=loads,
        materials=materials,
        thickness=thickness,
        half_bandwidth=half_bandwidth,
        connectivity=connectivity,
        displacements=displacements,
        reactions=reactions,
        stresses=stresses,
    )


def read_fortran_output(path: str) -> FortranOutput:
    """Read and parse a Fortran output file."""
    with open(path, "r") as handle:
        return parse_fortran_output(handle.read())


def noise_floor(reference: np.ndarray, relative: float = 1e-6) -> float:
    """Return an absolute tolerance scaled to the magnitude of a result field.

    The Fortran program computed in single precision (about 7 significant
    digits) and printed 4.  Quantities that are exactly zero by symmetry are
    therefore printed as small non-zero numbers -- ``ex-out.txt`` reports a
    reaction of ``-0.3000E-04`` where the exact answer is 0.  A relative
    tolerance cannot cover those, so results are compared with an absolute
    tolerance proportional to the largest value in the same field.
    """
    return relative * float(np.max(np.abs(reference)))
