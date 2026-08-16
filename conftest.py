"""Pytest configuration: make the repository importable and share the reference case."""

from __future__ import annotations

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DOC_DIR = os.path.join(REPO_ROOT, "doc")
TEST_DATA_DIR = os.path.join(REPO_ROOT, "tests", "data")

#: The reference input case shipped with the original Fortran program.
EXAMPLE_INPUT = os.path.join(DOC_DIR, "ex-data.txt")
#: The output the original single-precision Fortran program produced for it.
EXAMPLE_OUTPUT = os.path.join(DOC_DIR, "ex-out.txt")
#: The same run with REAL promoted to double precision and more printed digits.
DOUBLE_PRECISION_OUTPUT = os.path.join(TEST_DATA_DIR, "fe2quad-double-precision.txt")
#: The same model analysed as plane strain (lc = 2), also in double precision.
PLANE_STRAIN_OUTPUT = os.path.join(TEST_DATA_DIR, "fe2quad-plane-strain-double-precision.txt")

from fe2quad import read_fe2quad_input, solve  # noqa: E402  (needs REPO_ROOT on sys.path)


@pytest.fixture(scope="session")
def example_model():
    """The model described by ``doc/ex-data.txt``."""
    return read_fe2quad_input(EXAMPLE_INPUT)


@pytest.fixture(scope="session")
def example_result(example_model):
    """Analysis of the reference case with the Fortran-compatible penalty method."""
    return solve(example_model)
