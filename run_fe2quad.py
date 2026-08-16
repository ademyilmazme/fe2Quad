#!/usr/bin/env python3
"""Entry point for the FE2Quad solver.

Usage::

    python run_fe2quad.py doc/ex-data.txt
    python run_fe2quad.py doc/ex-data.txt --output python-out.txt
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fe2quad.cli import main  # noqa: E402  (import after the path is set up)

if __name__ == "__main__":
    raise SystemExit(main())
