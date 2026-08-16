"""Tests for the input file reader (Fortran ``input``).

The reader must accept ``doc/ex-data.txt`` unchanged and reproduce the
list-directed record semantics of the original ``READ`` statements.
"""

from __future__ import annotations

import numpy as np
import pytest

from fe2quad import AnalysisType, InputError, parse_fe2quad_input, read_fe2quad_input

#: A minimal, valid one-element model used to exercise the parser.
MINIMAL = """1,1,4,3,1,1
1,0,0
2,2,0
3,2,3
4,0,3
1,0
2,0
4,0
5,1000
1,30000,0.3
0.5
1,1,2,3,4,1
"""


def test_reads_the_reference_input_file(example_model):
    """doc/ex-data.txt must be readable without any modification."""
    assert example_model.analysis_type is AnalysisType.PLANE_STRESS
    assert example_model.num_nodes == 6
    assert example_model.num_elements == 2
    assert example_model.thickness == pytest.approx(1.0)

    np.testing.assert_allclose(
        example_model.coordinates,
        [[0, 0], [10, 5], [0, 10], [10, 10], [0, 20], [10, 15]],
    )
    # Connectivity is stored zero-based; the file lists 1,2,4,3 and 3,4,6,5.
    np.testing.assert_array_equal(example_model.connectivity, [[0, 1, 3, 2], [2, 3, 5, 4]])
    np.testing.assert_array_equal(example_model.element_material, [0, 0])

    assert [(bc.dof, bc.value) for bc in example_model.prescribed_displacements] == [
        (1, 0.0), (2, 0.0), (5, 0.0), (6, 0.0), (9, 0.0), (10, 0.0)
    ]
    assert [(load.dof, load.value) for load in example_model.loads] == [
        (3, 125.0), (7, 250.0), (11, 125.0)
    ]


def test_dof_and_node_numbering_round_trip(example_model):
    """1-based file numbering is preserved on the way out."""
    np.testing.assert_array_equal(example_model.element_node_numbers(0), [1, 2, 4, 3])
    np.testing.assert_array_equal(example_model.element_node_numbers(1), [3, 4, 6, 5])
    assert example_model.material_number(0) == 1
    # Element 0 owns nodes 1,2,4,3 -> DOFs 1,2 3,4 7,8 5,6 (zero-based here).
    np.testing.assert_array_equal(example_model.element_dofs(0), [0, 1, 2, 3, 6, 7, 4, 5])


def test_load_vector_places_loads_at_the_right_dofs(example_model):
    f = example_model.load_vector()
    assert f.shape == (12,)
    assert f[2] == 125.0 and f[6] == 250.0 and f[10] == 125.0
    assert np.count_nonzero(f) == 3


def test_minimal_model_parses():
    model = parse_fe2quad_input(MINIMAL)
    assert model.num_nodes == 4
    assert model.num_elements == 1
    assert model.thickness == pytest.approx(0.5)
    assert model.materials[0].youngs_modulus == pytest.approx(30000.0)


def test_values_may_be_separated_by_blanks_or_commas():
    """Fortran list-directed input accepts either separator."""
    blanks = MINIMAL.replace(",", " ")
    np.testing.assert_allclose(
        parse_fe2quad_input(blanks).coordinates, parse_fe2quad_input(MINIMAL).coordinates
    )


def test_a_single_read_may_span_several_records():
    """The four nodes may be written on one line or on four."""
    one_line = MINIMAL.replace("1,0,0\n2,2,0\n3,2,3\n4,0,3\n", "1,0,0, 2,2,0, 3,2,3, 4,0,3\n")
    np.testing.assert_allclose(
        parse_fe2quad_input(one_line).coordinates, parse_fe2quad_input(MINIMAL).coordinates
    )


def test_surplus_values_on_the_last_record_of_a_read_are_ignored():
    """Fortran discards whatever is left on the record when the list is satisfied."""
    with_trailing = MINIMAL.replace("1,1,4,3,1,1\n", "1,1,4,3,1,1, 999, 888\n")
    model = parse_fe2quad_input(with_trailing)
    assert model.num_nodes == 4


def test_nodes_may_be_listed_out_of_order():
    """The node number in the file is the storage index, not the position."""
    shuffled = MINIMAL.replace("1,0,0\n2,2,0\n3,2,3\n4,0,3\n", "3,2,3\n1,0,0\n4,0,3\n2,2,0\n")
    np.testing.assert_allclose(
        parse_fe2quad_input(shuffled).coordinates, parse_fe2quad_input(MINIMAL).coordinates
    )


def test_plane_strain_analysis_type():
    model = parse_fe2quad_input(MINIMAL.replace("1,1,4,3,1,1", "2,1,4,3,1,1", 1))
    assert model.analysis_type is AnalysisType.PLANE_STRAIN
    assert model.analysis_type.label == "Plane Strain Analysis"


@pytest.mark.parametrize(
    "replacement, message",
    [
        ("3,1,4,3,1,1", "analysis type"),
        ("1,1,4,3,1,0", "nm must be at least 1"),
        ("1,0,4,3,1,1", "ne must be at least 1"),
    ],
)
def test_invalid_header_is_rejected(replacement, message):
    with pytest.raises(InputError, match=message):
        parse_fe2quad_input(MINIMAL.replace("1,1,4,3,1,1", replacement, 1))


def test_out_of_range_node_number_is_rejected():
    with pytest.raises(InputError, match="node 9 is outside the valid range"):
        parse_fe2quad_input(MINIMAL.replace("4,0,3\n", "9,0,3\n", 1))


def test_duplicate_node_is_rejected():
    with pytest.raises(InputError, match="defined more than once"):
        parse_fe2quad_input(MINIMAL.replace("4,0,3\n", "3,0,3\n", 1))


def test_out_of_range_dof_is_rejected():
    """The model has 4 nodes, so DOF 99 does not exist."""
    with pytest.raises(InputError, match="DOF 99 is outside the valid range"):
        parse_fe2quad_input(MINIMAL.replace("5,1000", "99,1000", 1))


def test_unknown_material_reference_is_rejected():
    with pytest.raises(InputError, match="material 7 is outside the valid range"):
        parse_fe2quad_input(MINIMAL.replace("1,1,2,3,4,1\n", "1,1,2,3,4,7\n", 1))


def test_truncated_file_is_rejected():
    with pytest.raises(InputError, match="unexpected end of file"):
        parse_fe2quad_input("1,1,4,3,1,1\n1,0,0\n")


def test_non_numeric_field_is_rejected():
    with pytest.raises(InputError, match="expected a number"):
        parse_fe2quad_input(MINIMAL.replace("2,2,0", "2,two,0", 1))


def test_non_positive_thickness_is_rejected():
    with pytest.raises(InputError, match="thickness must be positive"):
        parse_fe2quad_input(MINIMAL.replace("\n0.5\n", "\n0\n", 1))


def test_invalid_poisson_ratio_is_rejected():
    with pytest.raises(InputError, match="Poisson's ratio"):
        parse_fe2quad_input(MINIMAL.replace("1,30000,0.3", "1,30000,0.6", 1))


def test_missing_file_reports_a_clear_error():
    with pytest.raises(InputError, match="input file not found"):
        read_fe2quad_input("no-such-file.txt")
