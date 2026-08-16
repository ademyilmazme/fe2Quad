"""Command line interface for the FE2Quad solver."""

from __future__ import annotations

import argparse
from typing import List, Optional

from .errors import FE2QuadError
from .fileio import format_report, read_fe2quad_input, write_report
from .solver import ELIMINATION, PENALTY, solve


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for ``run_fe2quad.py``."""
    parser = argparse.ArgumentParser(
        prog="run_fe2quad.py",
        description="2-D stress analysis using 4-node quadrilateral elements "
        "(Python port of the fe2quad.f solver).",
    )
    parser.add_argument("input", help="FE2Quad input file, for example doc/ex-data.txt")
    parser.add_argument(
        "-o",
        "--output",
        help="write the report to this file as well as to the terminal",
    )
    parser.add_argument(
        "--boundary-method",
        choices=(PENALTY, ELIMINATION),
        default=PENALTY,
        help="treatment of specified displacements; 'penalty' (default) reproduces the "
        "large-number method of the Fortran program, 'elimination' removes the "
        "constrained equations instead",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=4,
        help="significant digits printed for each value (default: 4, as in the Fortran output)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Run an analysis from the command line.  Returns a process exit status."""
    args = build_parser().parse_args(argv)
    try:
        model = read_fe2quad_input(args.input)
        result = solve(model, boundary_method=args.boundary_method)
        report = format_report(model, result, digits=args.digits)
        print(report)
        if args.output:
            write_report(args.output, model, result, digits=args.digits)
            print(f"Report written to {args.output}")
    except FE2QuadError as exc:
        print(f"error: {exc}")
        return 1
    return 0
