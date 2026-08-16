# FE2Quad

2-D linear elastic stress analysis with 4-node bilinear quadrilateral (Q4)
elements — a Python/NumPy port of the teaching program `fe2quad.f` by
M. Asghar Bhatti (University of Iowa).

The port is **behaviourally equivalent** to the Fortran original: same element
formulation, same numbering conventions, same boundary condition treatment,
same stress recovery and same principal stress angle convention. It reads the
original input files unchanged and reproduces `doc/ex-out.txt`.

## Quick start

```bash
pip install -r requirements.txt
python run_fe2quad.py doc/ex-data.txt
```

Write the report to a file as well as the terminal:

```bash
python run_fe2quad.py doc/ex-data.txt --output python-out.txt
```

Other options: `--boundary-method {penalty,elimination}` and `--digits N`
(default 4, matching the Fortran output).

## Python API

```python
from fe2quad import read_fe2quad_input, solve

model = read_fe2quad_input("doc/ex-data.txt")
result = solve(model)

result.displacements        # (num_dof,)          global vector, index = DOF - 1
result.nodal_displacements  # (num_nodes, 2)      [ux, uy] per node
result.reactions            # (num_prescribed,)   one per specified displacement
result.element_stresses     # (num_elements, 3)   SX, SY, TXY at the centroid
result.principal_stresses   # (num_elements, 3)   S1, S2, ANGLE (degrees)
result.half_bandwidth       # the Fortran NBW, reported for comparison
result.penalty              # the Fortran CNST
```

`Model` can also be built directly in code — nothing in the solver depends on
the file format.

## Fortran → Python mapping

| `fe2quad.f`            | Python                                                     |
| ---------------------- | ---------------------------------------------------------- |
| `input`                | `fileio.read_fe2quad_input`, `fileio.parse_fe2quad_input`   |
| `integ`                | `element.gauss_points_2x2`                                  |
| `cmat`                 | `element.constitutive_matrix`                               |
| `bmat`                 | `element.strain_displacement_matrix` (+ `element.jacobian`) |
| `elstif`               | `element.element_stiffness`                                 |
| `band`                 | `solver.solve_system`                                       |
| `prstrs`               | `post.principal_stresses`                                   |
| bandwidth loop in main | `solver.half_bandwidth`                                     |
| assembly loop in main  | `solver.assemble_global_stiffness`                          |
| penalty block in main  | `solver.apply_penalty_boundary_conditions`                  |
| stress loop in main    | `post.element_stresses`                                     |
| `WRITE` statements     | `fileio.format_report`, `fileio.write_report`               |

`bmat` computed `C·B` as a single `cb` array; here `strain_displacement_matrix`
returns `B` and callers form `C @ B`, which keeps the two matrices separate
without changing any equation.

## Conventions (unchanged from the Fortran)

* **Nodes, elements, materials** are numbered from 1 in the input file. The
  number is the *storage index*, so entries may be listed out of order.
* **Degrees of freedom**: node `i` owns DOF `2i-1` (x) and `2i` (y), so **odd
  DOF numbers are x and even DOF numbers are y**.
* **Element nodes** are listed counter-clockwise; local nodes 1–4 map to the
  parent element corners `(-1,-1)`, `(+1,-1)`, `(+1,+1)`, `(-1,+1)`. The
  element displacement vector is `[u1, v1, u2, v2, u3, v3, u4, v4]`.
* **Strains** are `[eps_x, eps_y, gamma_xy]` (engineering shear).
* **Stresses** are recovered at the element centroid only, `s = t = 0`.
* **Principal angle** is `ANGLE SX->S1`: degrees, counter-clockwise positive,
  from the x axis to the S1 direction, in the range ±90°.
* Internally arrays are zero-based; conversion happens in the reader and the
  report writer. Fields that hold a *DOF number* keep the 1-based file value and
  expose a `.index` property for the zero-based one.

## Input file format

Free format, comma and/or blank separated, exactly as Fortran list-directed
input. Each Fortran `READ` starts a new record: one `READ` may span several
lines, but surplus values on its last line are discarded. The reader reproduces
that.

```text
lc, ne, nn, nd, nl, nm      lc: 1 = plane stress, 2 = plane strain
                            ne/nn/nd/nl/nm: elements, nodes, specified
                            displacements, loads, materials
nn  x (node, x, y)          one READ
nd  x (dof, value)          one READ    -- specified displacements
nl  x (dof, value)          one READ per load
nm  x (material, E, nu)     one READ
th                          thickness
ne  x (element, n1, n2, n3, n4, material)   one READ
```

## Deliberate differences from the Fortran

These change how the program behaves when something is *wrong*, or how the
numbers are stored — never the finite element equations.

1. **Double precision.** The Fortran used single-precision `REAL`. Three values
   in `doc/ex-out.txt` are therefore round-off noise on quantities that are
   exactly zero by symmetry: the y-displacements of nodes 3 and 4, and the
   reaction at DOF 6 (printed as `-0.3000E-04`). This port returns ~0 for them.
   See "Verification" below.
2. **Matrix storage and linear solver.** The Fortran stored the upper band
   `bigk(nq, nbw)` and factorised it with a hand-written banded elimination
   (`band`). This port assembles the full symmetric matrix and calls
   `numpy.linalg.solve`. The half bandwidth is still computed and reported so it
   can be compared. The governing equations are unchanged.
3. **Validation.** The original read its input with no checks and divided by
   `dj` without looking at it. This port raises `InputError` for malformed or
   inconsistent input, `ElementGeometryError` when `det(J) <= 0` (naming the
   element and integration point), and `SolverError` when the assembled system
   is singular to working precision — which is what an under-restrained model
   produces, and which LAPACK otherwise reports as a perfectly ordinary answer.
   An analysis type other than 1 or 2 is rejected; the Fortran silently treated
   anything that was not 1 as plane strain.
4. **Optional direct elimination.** `solve(model, boundary_method="elimination")`
   removes the constrained equations instead of penalising them. The penalty
   method remains the default and is the reference/compatibility mode.
5. **`math.degrees` instead of the literal `57.29577951`.** The Fortran constant
   is truncated at 10 digits, so the printed angles agree to about 1e-10
   relative rather than to machine precision.
6. **Report layout.** The report adds section headings and direction/node
   columns; the numbers are the same. `fileio.format_fortran_e` reproduces the
   Fortran `E` edit descriptor (`0.1094E-06`) so output can be compared by eye.
7. **Module named `fileio`, not `io`,** to avoid shadowing the standard library.

## Tests

```bash
python -m pytest -v
```

| File                            | Covers                                                       |
| ------------------------------- | ------------------------------------------------------------ |
| `test_reference_case.py`        | Regression against `doc/ex-out.txt` (the primary test)        |
| `test_material.py`              | `cmat`: plane stress and plane strain vs analytical matrices  |
| `test_element.py`               | `integ`, `bmat`, `elstif`, Jacobian validation                |
| `test_principal_stress.py`      | `prstrs`: values, invariants and the angle convention         |
| `test_equilibrium.py`           | Force and moment equilibrium of loads against reactions       |
| `test_solver.py`                | Bandwidth, assembly, both BC methods, error handling          |
| `test_input_parsing.py`         | `input`: record semantics, numbering, malformed input         |

## Verification

The regression test compares against two references.

**`doc/ex-out.txt`** — the trusted original output, single precision, 4 printed
digits. Values are parsed from the file rather than transcribed. Worst observed
disagreement:

| Quantity           | Max relative error |
| ------------------ | ------------------ |
| Displacements      | 3.8e-4             |
| Reactions          | 1.2e-4             |
| Stresses SX SY TXY | 1.0e-4             |
| Principal S1, S2   | 5.6e-5             |
| Angle SX->S1       | 1.8e-5             |

These are at the level of the reference's own 4-digit print quantisation
(±5e-4), so `rtol = 1e-3` is as tight as this file can support. Because three
printed values are noise on exact zeros, comparisons also use an absolute floor
of `1e-6 × max|reference|` for the field (`tests/fortran_output.noise_floor`).

**`tests/data/fe2quad-double-precision.txt`** — the *same Fortran source*
compiled with `gfortran -std=legacy -fdefault-real-8` (REAL promoted to double)
and wider output formats, which removes the single-precision noise and allows a
much tighter check:

| Quantity           | Max relative error | Tolerance used |
| ------------------ | ------------------ | -------------- |
| Displacements      | 2.5e-15            | 1e-12          |
| Reactions          | 4.1e-16            | 1e-12          |
| Stresses           | 1.7e-15            | 1e-12          |
| Principal S1, S2   | 0 (bit-identical)  | 1e-12          |
| Angle SX->S1       | 5.4e-11            | 1e-9           |

**`tests/data/fe2quad-plane-strain-double-precision.txt`** — `doc/ex-data.txt`
selects plane stress, so the shipped reference case never reaches the plane
strain branch of `cmat`. The same model was therefore re-run with `lc = 2`
through the same double-precision build, and the port is checked against it to
the same tolerances (worst case 1.2e-15).

To regenerate those files:

```bash
sed -e "s/'(I5,2E15.4)'/'(I5,2E26.16)'/" -e "s/'(I5,E15.4)'/'(I5,E26.16)'/" \
    -e "s/'(I4,6(1X,E10.4))'/'(I4,6(1X,E26.16))'/" doc/fe2quad.f > fe2quad_dp.f
gfortran -std=legacy -fdefault-real-8 -o fe2quad_dp fe2quad_dp.f
printf "'ex-data.txt'\n'fe2quad-double-precision.txt'\n" | ./fe2quad_dp
sed '1s/^1,/2,/' ex-data.txt > ex-data-ps.txt
printf "'ex-data-ps.txt'\n'fe2quad-plane-strain-double-precision.txt'\n" | ./fe2quad_dp
```

The suite was also checked for sensitivity by injecting faults: a 0.1 % error in
the shear modulus fails 8 tests, a flipped principal angle branch fails 4, and a
10x change to the penalty factor fails 6 — including the primary regression
against `doc/ex-out.txt` in the first and third cases.
