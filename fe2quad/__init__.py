"""FE2Quad -- 2-D stress analysis using 4-node quadrilateral elements.

A Python/NumPy port of the teaching program ``fe2quad.f`` by M. Asghar Bhatti
(University of Iowa).  The finite element formulation, the numbering
conventions and the output quantities are those of the original program; only
the matrix storage, the linear solver and the input validation differ.  See
``README.md`` for the full routine-by-routine mapping.

Typical use::

    from fe2quad import read_fe2quad_input, solve

    model = read_fe2quad_input("doc/ex-data.txt")
    result = solve(model)
    result.displacements       # (num_dof,)   ordered by DOF number
    result.reactions           # one per specified displacement
    result.element_stresses    # (num_elements, 3) -> SX, SY, TXY
    result.principal_stresses  # (num_elements, 3) -> S1, S2, ANGLE
"""

from .element import (
    constitutive_matrix,
    element_stiffness,
    gauss_points_2x2,
    jacobian,
    shape_function_derivatives,
    strain_displacement_matrix,
)
from .errors import (
    ElementGeometryError,
    FE2QuadError,
    InputError,
    ModelError,
    SolverError,
)
from .fileio import (
    format_report,
    parse_fe2quad_input,
    read_fe2quad_input,
    write_report,
)
from .model import (
    AnalysisType,
    Material,
    Model,
    NodalLoad,
    PrescribedDisplacement,
    Result,
    dof_direction,
    dof_node,
    x_dof,
    y_dof,
)
from .post import element_stresses, principal_stress_table, principal_stresses
from .solver import (
    apply_penalty_boundary_conditions,
    assemble_global_stiffness,
    half_bandwidth,
    solve,
    solve_system,
)

__version__ = "1.0.0"

__all__ = [
    "AnalysisType",
    "ElementGeometryError",
    "FE2QuadError",
    "InputError",
    "Material",
    "Model",
    "ModelError",
    "NodalLoad",
    "PrescribedDisplacement",
    "Result",
    "SolverError",
    "apply_penalty_boundary_conditions",
    "assemble_global_stiffness",
    "constitutive_matrix",
    "dof_direction",
    "dof_node",
    "element_stiffness",
    "element_stresses",
    "format_report",
    "gauss_points_2x2",
    "half_bandwidth",
    "jacobian",
    "parse_fe2quad_input",
    "principal_stress_table",
    "principal_stresses",
    "read_fe2quad_input",
    "shape_function_derivatives",
    "solve",
    "solve_system",
    "strain_displacement_matrix",
    "write_report",
    "x_dof",
    "y_dof",
]
