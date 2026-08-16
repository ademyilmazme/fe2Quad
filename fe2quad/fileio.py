"""Reading FE2Quad input files and writing the analysis report.

Fortran routines reproduced here
--------------------------------
=========================  ==================================
``fe2quad.f``              Python
=========================  ==================================
``input`` (reading half)   :func:`read_fe2quad_input`
``input`` (echo half) and
the ``WRITE`` statements   :func:`format_report`, :func:`write_report`
=========================  ==================================

The module is named ``fileio`` rather than ``io`` so that it cannot be confused
with the standard library module of that name.

Input file format
-----------------
The file is free-format: values are separated by commas and/or blanks, exactly
as Fortran list-directed input (``READ (unit, *)``) expects.  The record
structure matters because each Fortran ``READ`` statement begins on a new
record: a single ``READ`` may span as many lines as it needs, but any values
left over on its last line are discarded.  :class:`_RecordReader` reproduces
that behaviour.

The groups, in order::

    1.  lc, ne, nn, nd, nl, nm          one READ
        lc = 1 plane stress, 2 plane strain
        ne = elements, nn = nodes, nd = specified displacements,
        nl = specified loads, nm = materials
    2.  nn * (node, x, y)               one READ
    3.  nd * (dof, value)               one READ
    4.  nl records of (dof, value)      one READ *per load*
    5.  nm * (material, E, nu)          one READ
    6.  th                              one READ
    7.  ne * (element, n1, n2, n3, n4, material)   one READ

Node, element and material numbers are the indices the data is stored at, so
the file may list them out of order; they are 1-based and are converted to
0-based indices when the :class:`~fe2quad.model.Model` is built.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

from .errors import InputError
from .model import (
    AnalysisType,
    Material,
    Model,
    NodalLoad,
    NODES_PER_ELEMENT,
    PrescribedDisplacement,
    Result,
    dof_direction,
    dof_node,
)


def _tokenize(line: str) -> List[str]:
    """Split one record into list-directed values (blank and comma separated)."""
    return [token for token in line.replace(",", " ").split() if token]


class _RecordReader:
    """Consume a Fortran list-directed input file record by record.

    Each call to :meth:`read` models one Fortran ``READ`` statement: it starts
    at the next unread record, pulls in further records until the requested
    number of values is available, and discards anything left over on the last
    record it touched.
    """

    def __init__(self, text: str, source: str) -> None:
        self._lines = text.splitlines()
        self._line_index = 0
        self._source = source

    def read(self, count: int, what: str) -> List[str]:
        """Return ``count`` raw value tokens, starting a new record."""
        tokens: List[str] = []
        first_line = self._line_index + 1
        while len(tokens) < count:
            if self._line_index >= len(self._lines):
                raise InputError(
                    f"{self._source}: unexpected end of file while reading {what}; "
                    f"expected {count} value(s) starting at line {first_line}, found {len(tokens)}"
                )
            tokens.extend(_tokenize(self._lines[self._line_index]))
            self._line_index += 1
        return tokens[:count]  # surplus values on the final record are ignored

    @property
    def line(self) -> int:
        """1-based number of the last record consumed (for error messages)."""
        return self._line_index


def _to_int(token: str, what: str, source: str, line: int) -> int:
    try:
        return int(token)
    except ValueError:
        raise InputError(f"{source}, line {line}: expected an integer for {what}, got {token!r}") from None


def _to_float(token: str, what: str, source: str, line: int) -> float:
    # Fortran may write exponents with D instead of E.
    try:
        return float(token.replace("D", "E").replace("d", "e"))
    except ValueError:
        raise InputError(f"{source}, line {line}: expected a number for {what}, got {token!r}") from None


def _check_number(value: int, limit: int, what: str, source: str, line: int) -> int:
    """Validate a 1-based number from the file and return its 0-based index."""
    if not 1 <= value <= limit:
        raise InputError(f"{source}, line {line}: {what} {value} is outside the valid range 1..{limit}")
    return value - 1


def parse_fe2quad_input(text: str, source: str = "<string>") -> Model:
    """Parse the contents of an FE2Quad input file into a :class:`Model`."""
    reader = _RecordReader(text, source)

    header = reader.read(6, "the header line (lc, ne, nn, nd, nl, nm)")
    lc, ne, nn, nd, nl, nm = (
        _to_int(token, name, source, reader.line)
        for token, name in zip(header, ("lc", "ne", "nn", "nd", "nl", "nm"))
    )

    try:
        analysis_type = AnalysisType(lc)
    except ValueError:
        raise InputError(
            f"{source}, line {reader.line}: analysis type lc = {lc} is not supported; "
            "use 1 for plane stress or 2 for plane strain"
        ) from None
    for name, value in (("ne", ne), ("nn", nn)):
        if value < 1:
            raise InputError(f"{source}, line {reader.line}: {name} must be at least 1, got {value}")
    for name, value in (("nd", nd), ("nl", nl), ("nm", nm)):
        if value < 0:
            raise InputError(f"{source}, line {reader.line}: {name} must not be negative, got {value}")
    if nm < 1:
        raise InputError(f"{source}, line {reader.line}: nm must be at least 1, got {nm}")

    num_dof = 2 * nn

    # --- nodal coordinates ------------------------------------------------
    tokens = reader.read(3 * nn, "nodal coordinates")
    coordinates = np.full((nn, 2), np.nan, dtype=float)
    seen_nodes: Dict[int, int] = {}
    for triple in range(nn):
        number, x_token, y_token = tokens[3 * triple : 3 * triple + 3]
        node = _to_int(number, "a node number", source, reader.line)
        index = _check_number(node, nn, "node", source, reader.line)
        if node in seen_nodes:
            raise InputError(f"{source}: node {node} is defined more than once")
        seen_nodes[node] = index
        coordinates[index, 0] = _to_float(x_token, f"the x coordinate of node {node}", source, reader.line)
        coordinates[index, 1] = _to_float(y_token, f"the y coordinate of node {node}", source, reader.line)
    _require_complete(seen_nodes, nn, "node", source)

    # --- specified displacements -----------------------------------------
    tokens = reader.read(2 * nd, "specified displacements")
    prescribed: List[PrescribedDisplacement] = []
    for pair in range(nd):
        dof_token, value_token = tokens[2 * pair : 2 * pair + 2]
        dof = _to_int(dof_token, "a DOF number", source, reader.line)
        _check_number(dof, num_dof, "DOF", source, reader.line)
        prescribed.append(
            PrescribedDisplacement(
                dof=dof,
                value=_to_float(value_token, f"the displacement specified at DOF {dof}", source, reader.line),
            )
        )

    # --- specified loads: one Fortran READ per load -----------------------
    loads: List[NodalLoad] = []
    for _ in range(nl):
        dof_token, value_token = reader.read(2, "a specified load")
        dof = _to_int(dof_token, "a DOF number", source, reader.line)
        _check_number(dof, num_dof, "DOF", source, reader.line)
        loads.append(
            NodalLoad(
                dof=dof,
                value=_to_float(value_token, f"the load specified at DOF {dof}", source, reader.line),
            )
        )

    # --- materials --------------------------------------------------------
    tokens = reader.read(3 * nm, "material properties")
    moduli = np.full(nm, np.nan, dtype=float)
    ratios = np.full(nm, np.nan, dtype=float)
    seen_materials: Dict[int, int] = {}
    for triple in range(nm):
        number, e_token, nu_token = tokens[3 * triple : 3 * triple + 3]
        material = _to_int(number, "a material number", source, reader.line)
        index = _check_number(material, nm, "material", source, reader.line)
        if material in seen_materials:
            raise InputError(f"{source}: material {material} is defined more than once")
        seen_materials[material] = index
        moduli[index] = _to_float(e_token, f"Young's modulus of material {material}", source, reader.line)
        ratios[index] = _to_float(nu_token, f"Poisson's ratio of material {material}", source, reader.line)
    _require_complete(seen_materials, nm, "material", source)
    materials = tuple(Material(float(e), float(nu)) for e, nu in zip(moduli, ratios))
    for number, material in enumerate(materials, start=1):
        _check_material(material, number, source)

    # --- thickness --------------------------------------------------------
    (thickness_token,) = reader.read(1, "the thickness")
    thickness = _to_float(thickness_token, "the thickness", source, reader.line)
    if thickness <= 0.0:
        raise InputError(f"{source}, line {reader.line}: thickness must be positive, got {thickness}")

    # --- element connectivity --------------------------------------------
    tokens = reader.read(6 * ne, "element connectivity")
    connectivity = np.full((ne, NODES_PER_ELEMENT), -1, dtype=int)
    element_material = np.full(ne, -1, dtype=int)
    seen_elements: Dict[int, int] = {}
    for row in range(ne):
        fields = tokens[6 * row : 6 * row + 6]
        element = _to_int(fields[0], "an element number", source, reader.line)
        index = _check_number(element, ne, "element", source, reader.line)
        if element in seen_elements:
            raise InputError(f"{source}: element {element} is defined more than once")
        seen_elements[element] = index
        for corner in range(NODES_PER_ELEMENT):
            node = _to_int(fields[1 + corner], f"node {corner + 1} of element {element}", source, reader.line)
            connectivity[index, corner] = _check_number(node, nn, "node", source, reader.line)
        material = _to_int(fields[5], f"the material number of element {element}", source, reader.line)
        element_material[index] = _check_number(material, nm, "material", source, reader.line)
    _require_complete(seen_elements, ne, "element", source)

    return Model(
        analysis_type=analysis_type,
        coordinates=coordinates,
        connectivity=connectivity,
        element_material=element_material,
        materials=materials,
        thickness=thickness,
        prescribed_displacements=tuple(prescribed),
        loads=tuple(loads),
    )


def _require_complete(seen: Dict[int, int], expected: int, what: str, source: str) -> None:
    missing = sorted(set(range(1, expected + 1)) - set(seen))
    if missing:
        raise InputError(
            f"{source}: {what} data is missing for "
            + ", ".join(f"{what} {number}" for number in missing[:10])
            + (" ..." if len(missing) > 10 else "")
        )


def _check_material(material: Material, number: int, source: str) -> None:
    if material.youngs_modulus <= 0.0:
        raise InputError(
            f"{source}: Young's modulus of material {number} must be positive, "
            f"got {material.youngs_modulus}"
        )
    if not -1.0 < material.poissons_ratio < 0.5:
        raise InputError(
            f"{source}: Poisson's ratio of material {number} must lie between -1 and 0.5, "
            f"got {material.poissons_ratio}"
        )


def read_fe2quad_input(path: str) -> Model:
    """Read an FE2Quad input file (the format of ``ex-data.txt``) into a :class:`Model`."""
    try:
        with open(path, "r") as handle:
            text = handle.read()
    except FileNotFoundError:
        hint = ""
        alternative = os.path.join("doc", os.path.basename(path))
        if os.path.isfile(alternative):
            hint = f" (did you mean {alternative}?)"
        raise InputError(f"input file not found: {path}{hint}") from None
    except OSError as exc:
        raise InputError(f"could not read input file {path}: {exc}") from None
    return parse_fe2quad_input(text, source=path)


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------

def format_fortran_e(value: float, digits: int = 4) -> str:
    """Format a number the way Fortran's ``E`` edit descriptor does.

    Produces a normalised mantissa in ``[0.1, 1)`` such as ``0.1094E-06`` or
    ``-0.3691E+01``, which lines up with the reference output of the original
    program.  ``digits`` is the number of significant digits in the mantissa.
    """
    if digits < 1:
        raise ValueError("digits must be at least 1")
    if not np.isfinite(value):
        return str(value)
    if value == 0.0:
        return f"0.{'0' * digits}E+00"
    mantissa, exponent = f"{abs(value):.{digits - 1}E}".split("E")
    sign = "-" if value < 0.0 else ""
    return f"{sign}0.{mantissa.replace('.', '')}E{int(exponent) + 1:+03d}"


#: Width of the stress columns; wide enough for the longest heading.
_STRESS_COLUMN_WIDTH = 14


def _column(value: float, digits: int, width: int = 15) -> str:
    return format_fortran_e(value, digits).rjust(width)


def format_report(model: Model, result: Result, digits: int = 4) -> str:
    """Render the analysis as text, following the layout of the Fortran output.

    The numerical values are those of the original program; the layout adds
    section headings and the direction/node columns that make the DOF numbering
    explicit.
    """
    lines: List[str] = []
    add = lines.append

    add("=" * 72)
    add(" FE2Quad -- 2-D stress analysis using 4-node quadrilateral elements")
    add(" Python port of fe2quad.f (M. Asghar Bhatti, University of Iowa)")
    add("=" * 72)
    add("")
    add("ANALYSIS TYPE")
    add(f"  {model.analysis_type.label}")
    add("")

    add("MODEL SUMMARY")
    add("      NE    NN    ND    NL    NM")
    add(
        f"  {model.num_elements:6d}{model.num_nodes:6d}"
        f"{len(model.prescribed_displacements):6d}{len(model.loads):6d}{len(model.materials):6d}"
    )
    add(f"  Degrees of freedom (NQ) : {model.num_dof}")
    add(f"  Half bandwidth (NBW)    : {result.half_bandwidth}")
    add(f"  Thickness               : {model.thickness:g}")
    add(f"  Boundary conditions     : {result.boundary_method}")
    if result.penalty is not None:
        add(f"  Penalty constant (CNST) : {format_fortran_e(result.penalty, digits)}")
    add("")

    add("NODE COORDINATES")
    add(" Node       X-Coord.       Y-Coord.")
    for node in range(model.num_nodes):
        x, y = model.coordinates[node]
        add(f" {node + 1:4d}  {_column(x, digits)}{_column(y, digits)}")
    add("")

    add("BOUNDARY CONDITIONS (specified displacements)")
    add("  DOF#  Dir  Node   Specified disp.")
    for bc in model.prescribed_displacements:
        add(f" {bc.dof:5d}   {dof_direction(bc.dof)} {dof_node(bc.dof):5d}  {_column(bc.value, digits)}")
    add("")

    add("LOADS")
    add("  DOF#  Dir  Node   Specified load")
    for load in model.loads:
        add(f" {load.dof:5d}   {dof_direction(load.dof)} {dof_node(load.dof):5d}  {_column(load.value, digits)}")
    add("")

    add("MATERIALS")
    add(" Material#              E             Nu")
    for number, material in enumerate(model.materials, start=1):
        add(
            f" {number:9d}  {_column(material.youngs_modulus, digits)}"
            f"{_column(material.poissons_ratio, digits)}"
        )
    add("")

    add("ELEMENT CONNECTIVITY")
    add(" Element#  -------Nodes-------   Material#")
    for element in range(model.num_elements):
        nodes = "".join(f"{node:6d}" for node in model.element_node_numbers(element))
        add(f" {element + 1:8d}{nodes}{model.material_number(element):11d}")
    add("")

    add("NODAL DISPLACEMENTS")
    add(" NODE#        X-Displ        Y-Displ")
    for node, (ux, uy) in enumerate(result.nodal_displacements, start=1):
        add(f" {node:5d}  {_column(ux, digits)}{_column(uy, digits)}")
    add("")

    add("REACTIONS")
    add("  DOF#  Dir  Node       Reaction")
    for bc, reaction in zip(model.prescribed_displacements, result.reactions):
        add(f" {bc.dof:5d}   {dof_direction(bc.dof)} {dof_node(bc.dof):5d}  {_column(reaction, digits)}")
    add("")

    add("ELEMENT STRESSES (at the element centroid, s = t = 0)")
    stress_columns = ("SX", "SY", "TXY", "S1", "S2", "ANGLE SX->S1")
    add(" ELEM#" + "".join(name.rjust(_STRESS_COLUMN_WIDTH) for name in stress_columns))
    for element in range(model.num_elements):
        values = list(result.element_stresses[element]) + list(result.principal_stresses[element])
        add(
            f" {element + 1:5d}"
            + "".join(_column(value, digits, width=_STRESS_COLUMN_WIDTH) for value in values)
        )
    add("")
    return "\n".join(lines)


def write_report(path: str, model: Model, result: Result, digits: int = 4) -> None:
    """Write :func:`format_report` to ``path``."""
    try:
        with open(path, "w") as handle:
            handle.write(format_report(model, result, digits))
    except OSError as exc:
        raise InputError(f"could not write output file {path}: {exc}") from None
