C     ***                 fe2quad                      ***
C     ***      2-d stress analysis using 4-node        ***
C     ***            quadrilateral elements            ***
C     ***       University of Iowa, Iowa City, IA      ***
C     ***        Instructor: M. Asghar Bhatti          ***

C     Problem size controlling parameters
      PARAMETER (maxnodes = 100, maxelem = 100, maxmat = 10)
      PARAMETER (maxdof = 2*maxnodes, maxband = 50, maxbdc = 50)
C     maxnodes = maximum number of nodes in a model.
C     maxelem = maximum number of elements in a model.
C     maxmat = maximum number of materials used in a model.
C     maxdof = maximum number of degrees of freedom = 2*maxnodes.
C     maxband = maximum bandwidth for the global stiffness matrix.
C     maxbdc = maximum number of specified boundary conditions.

C     Key data storage arrays
      REAL x(maxnodes, 2), pm(maxmat, 2), u(maxbdc)
      INTEGER nu(maxbdc), mat(maxelem), noc(maxelem, 4)
      REAL bigk(maxdof, maxband), f(maxdof)
C     x(maxnodes, 2):     x and y coordinates of the nodes
C     pm(maxmat, 2): material props [young's modulus, poisson's ratio]
C     u(maxbdc): specified boundary conditions
C     nu(maxbdc):  dof with the specified boundary condition
C     mat(maxelem): specified material number for each element
C     noc(maxelem, 4):    nodal connectivity for each element
C     bigk(maxdof, maxband): global stiffness matrix
C     f(maxdof):          global load vector

C     The following arrays are used during element stiffness formulation
      REAL c(3, 3), b(3, 8), cb(3, 8), elemk(8, 8), d(8)
      REAL xni(4, 2), wni(4)
C     c(3, 3):    constitutive matrix
C     b(3, 8):    strain-displacement matrix
C     cb(3, 8):   product of matrices c and b
C     elemk(8, 8):   element stiffness matrix
C     d(8):       element displacement vector
C     xni(4, 2):  (s,t) coordinates of integration points
C     assumes 2x2 integration. thus there are 4 points.
C     wni(4):     weights for integration points

C     The following array provides storage for element stresses
      REAL str(3)
C     str(3):  stresses (sx, sy, and txy) at element center.

C     Filenames for input data and program output.
      CHARACTER*16 inpfil, outfil
      DATA linp/10/, lout/11/

      WRITE (*, *) ' Enter the name of the input file'
      READ (*, *) inpfil
      WRITE (*, *) ' Enter the name of the output file'
      READ (*, *) outfil
      OPEN (UNIT = linp, FILE = inpfil, STATUS = 'OLD')
      OPEN (UNIT = lout, FILE = outfil, STATUS = 'UNKNOWN')

C     Get model parameters from the first line of input data file
      READ (linp, *) lc, ne, nn, nd, nl, nm
C     lc: analysis type
C     lc = 1 Plane stress analysis
C     lc = 2 Plane strain analysis
C     ne: number of elements in the model
C     nn: number of nodes in the model
C     nd: number of specified displacements
C     nl: number of specified loads
C     nm: number of materials in the model

      IF ( lc .EQ. 1 ) THEN
         WRITE (lout, *) 'Plane Stress Analysis'
      ELSE
         WRITE (lout, *) 'Plane Strain Analysis'
      END IF
      WRITE (lout, *) '---------------------'
      WRITE (lout, *) '    NE    NN    ND    NL    NM'
      WRITE (lout, '(1x, 6i6)') ne, nn, nd, nl, nm

C     Total number of degrees of freedom = nq
      nq = 2*nn

C     *************** MODEL DATA INPUT ***************
C     Initialize global force vector
      DO i = 1, nq
         f(i) = 0.
      END DO

      CALL input(ne, nn, nd, nl, nm, linp, lout, th, x, pm, u, nu, mat,
     & noc, f, maxnodes, maxelem, maxmat, maxbdc, maxdof)

      CLOSE (linp)

C     *************** FORM GLOBAL EQUATIONS ***************
C     calculate bandwidth nbw from element connectivity
      nbw = 0
      DO i = 1, ne
         cmin = nn + 1
         cmax = 0
         DO j = 1, 4
            IF ( cmin .GT. noc(i, j) ) cmin = noc(i, j)
            IF ( cmax .LT. noc(i, j) ) cmax = noc(i, j)
         END DO
         cc = 2*(cmax - cmin + 1)
         IF ( nbw .LT. cc ) nbw = cc
      END DO
      WRITE (lout, *)
      WRITE (lout, *) '  The Half bandwidth is ', nbw

C     initialize global stiffness matrix to all zeros
      DO i = 1, nq
         DO j = 1, nbw
            bigk(i, j) = 0
         END DO
      END DO

C     Gaussian integration points (s,t for parent element)
      CALL integ(xni, wni)

C     ---------------------------------------------------------
C     Loop over all elements. Form the element stiffness matrix
C     and assemble into the global stiffness matrix.

      DO n = 1, ne
C        form constitutive matrix c
         CALL cmat(n, lc, mat, pm, c, maxmat, maxelem)
C        form element stiffness matrix se
         CALL elstif(x, n, noc, xni, wni, c, elemk, th,
     &   maxnodes, maxelem)

C        assemble into the global matrix in banded form
         DO ii = 1, 4
            nrt = 2*(noc(n, ii) - 1)
            DO it = 1, 2
               nr = nrt + it
               i = 2*(ii - 1) + it
               DO jj = 1, 4
                  nct = 2*(noc(n, jj) - 1)
                  DO jt = 1, 2
                     j = 2*(jj - 1) + jt
                     nc = nct + jt - nr + 1
                     IF ( nc .GT. 0 ) bigk(nr, nc) = bigk(nr, nc) +
     &                 elemk(i, j)
                  END DO
               END DO
            END DO
         END DO
      END DO
C     ---------------------------------------------------------

C     modify to account for boundary conditions (approximate treatment)
C     generating arelatively large number
      cnst = bigk(1, 1)*100000
      DO i = 1, nd
         n = nu(i)
         bigk(n, 1) = bigk(n, 1) + cnst
         f(n) = f(n) + cnst*u(i)
      END DO

C     *************** SOLVE GLOBAL EQUATIONS ***************
C     bigk = global stiffness matrix in banded form
C     f = global load vector
      CALL band(bigk, f, maxdof, nbw, nq)
C     The band subroutine puts displacements in place of loads
C     Thus now f = nodal displacement vector

      WRITE (lout, *)
      WRITE (lout, *) 'NODE#      X-Displ        Y-Displ'
      WRITE (lout, '(I5,2E15.4)') (i, f(2*i - 1), f(2*i), i = 1, nn)

C     *************** SOLVE FOR SECONDARY QUANTITIES ***************
C     Compute reactions
      WRITE (lout, *)
      WRITE (lout, *) '  DOF#    Reaction'
      DO i = 1, nd
         n = nu(i)
         r = cnst*(u(i) - f(n))
         WRITE (lout, '(I5,E15.4)') n, r
      END DO

C     Compute stresses at the element centroid
C     (0,0 in the parent element)

      s = 0
      t = 0
      WRITE (lout, *)
      WRITE (lout, *) 
     & 'ELEM#    SX         SY         TXY        S1          S2   ANGLE
     & SX->S1'

C     ---------------------------------------------------------
C     Loop over all elements.
C     Compute and print stresses for each element

      DO n = 1, ne
C        Form constitutive matrix c.
C        This was formed in stiffness matrix calculations
C        but was not saved.
         CALL cmat(n, lc, mat, pm, c, maxmat, maxelem)
C        Form cb matrix
         CALL bmat(x, n, noc, s, t, b, cb, c, dj, maxnodes, maxelem)
C        Extract displacements at the nodes of the current element
         DO i = 1, 4
            in = 2*(noc(n, i) - 1)
            ii = 2*(i - 1)
            DO j = 1, 2
               d(ii + j) = f(in + j)
            END DO
         END DO
C        Calculate normal and shear stresses
         DO i = 1, 3
            str(i) = 0
            DO k = 1, 8
               str(i) = str(i) + cb(i, k)*d(k)
            END DO
         END DO
C        Calculate principal stresses
         CALL prstrs(str, s1, s2, ang)
         WRITE (lout, '(I4,6(1X,E10.4))') n, str(1), str(2), str(3),
     &    s1, s2, ang
      END DO
      CLOSE (lout)
      END
C     _____________________________________________________________________
      SUBROUTINE band(a, b, maxdof, nbw, n)

C     Solution of banded system of equations

      DIMENSION a(maxdof, nbw), b(maxdof)
      n1 = n - 1
C     forward elimination
      DO k = 1, n1
         nk = n - k + 1
         IF ( nk .GT. nbw ) nk = nbw
         DO i = 2, nk
            c1 = a(k, i)/a(k, 1)
            i1 = k + i - 1
            DO j = i, nk
               j1 = j - i + 1
               a(i1, j1) = a(i1, j1) - c1*a(k, j)
            END DO
            b(i1) = b(i1) - c1*b(k)
         END DO
      END DO
C     back substitution
      b(n) = b(n)/a(n, 1)
      DO kk = 1, n1
         k = n - kk
         c1 = 1.0/a(k, 1)
         b(k) = c1*b(k)
         nk = n - k + 1
         IF ( nk .GT. nbw ) nk = nbw
         DO j = 2, nk
            b(k) = b(k) - c1*a(k, j)*b(k + j - 1)
         END DO
      END DO
      RETURN
      END

C     __________________________________________________________________
      SUBROUTINE bmat(x, n, noc, s, t, b, cb, c, dj, maxnodes, 
     & maxelem)

C     Form c*b matrix

      REAL x(maxnodes, 2)
      INTEGER noc(maxelem, 4)

      DIMENSION a(3, 4), g(4, 8)
      DIMENSION b(3, 8), cb(3, 8), c(3, 3)

      n1 = noc(n, 1)
      n2 = noc(n, 2)
      n3 = noc(n, 3)
      n4 = noc(n, 4)
      x1 = x(n1, 1)
      y1 = x(n1, 2)
      x2 = x(n2, 1)
      y2 = x(n2, 2)
      x3 = x(n3, 1)
      y3 = x(n3, 2)
      x4 = x(n4, 1)
      y4 = x(n4, 2)

C     Form jacobian matrix
      tj11 = ((1 - t)*(x2 - x1) + (1 + t)*(x3 - x4))/4
      tj12 = ((1 - t)*(y2 - y1) + (1 + t)*(y3 - y4))/4
      tj21 = ((1 - s)*(x4 - x1) + (1 + s)*(x3 - x2))/4
      tj22 = ((1 - s)*(y4 - y1) + (1 + s)*(y3 - y2))/4

C     determinant of the jacobian
      dj = tj11*tj22 - tj12*tj21

C     a(3,4) matrix relates strains to local derivatives of u, v
      a(1, 1) = tj22/dj
      a(2, 1) = 0
      a(3, 1) =  - tj21/dj
      a(1, 2) =  - tj12/dj
      a(2, 2) = 0
      a(3, 2) = tj11/dj
      a(1, 3) = 0
      a(2, 3) =  - tj21/dj
      a(3, 3) = tj22/dj
      a(1, 4) = 0
      a(2, 4) = tj11/dj
      a(3, 4) =  - tj12/dj

C     g(4,8) matrix relates to local derivatives of u , v
C     to local nodal displacements d
      DO i = 1, 4
         DO j = 1, 8
            g(i, j) = 0
         END DO
      END DO
      g(1, 1) =  - (1 - t)/4
      g(2, 1) =  - (1 - s)/4
      g(3, 2) =  - (1 - t)/4
      g(4, 2) =  - (1 - s)/4
      g(1, 3) = (1 - t)/4
      g(2, 3) =  - (1 + s)/4
      g(3, 4) = (1 - t)/4
      g(4, 4) =  - (1 + s)/4
      g(1, 5) = (1 + t)/4
      g(2, 5) = (1 + s)/4
      g(3, 6) = (1 + t)/4
      g(4, 6) = (1 + s)/4
      g(1, 7) =  - (1 + t)/4
      g(2, 7) = (1 - s)/4
      g(3, 8) =  - (1 + t)/4
      g(4, 8) = (1 - s)/4

C     b(3,8) matrix relates strains to d
      DO i = 1, 3
         DO j = 1, 8
            b(i, j) = 0
            DO k = 1, 4
               b(i, j) = b(i, j) + a(i, k)*g(k, j)
            END DO
         END DO
      END DO

C     cb(3,8) matrix relates stresses to d
      DO i = 1, 3
         DO j = 1, 8
            cb(i, j) = 0
            DO k = 1, 3
               cb(i, j) = cb(i, j) + c(i, k)*b(k, j)
            END DO
         END DO
      END DO
      RETURN
      END

C     __________________________________________________________________
      SUBROUTINE cmat(n, lc, mat, pm, c, maxmat, maxelem)

C     Form constitutive matrix c relating stresses to strains

      REAL pm(maxmat, 2)
      INTEGER mat(maxelem)
      REAL c(3, 3)

      matn = mat(n)
      e = pm(matn, 1)
      pnu = pm(matn, 2)
      IF ( lc .EQ. 1 ) THEN
C        Plane stress problem
         c1 = e/(1 - pnu*pnu)
         c2 = c1*pnu
      ELSE
C        Plane strain problem
         cc = e/((1 + pnu)*(1 - 2*pnu))
         c1 = cc*(1 - pnu)
         c2 = cc*pnu
      END IF
      c3 = .5*e/(1 + pnu)
      c(1, 1) = c1
      c(1, 2) = c2
      c(1, 3) = 0
      c(2, 1) = c2
      c(2, 2) = c1
      c(2, 3) = 0
      c(3, 1) = 0
      c(3, 2) = 0
      c(3, 3) = c3
      RETURN
      END

C     __________________________________________________________________
      SUBROUTINE elstif(x, n, noc, xni, wni, c, elemk, th, maxnodes,
     & maxelem)

C     Form element stiffness

      REAL x(maxnodes, 2)
      INTEGER noc(maxelem, 4)
      DIMENSION xni(4, 2), wni(4), c(3, 3)
      DIMENSION elemk(8, 8), b(3, 8), cb(3, 8)

C     Initialize element stiffness matrix to zero
      DO i = 1, 8
         DO j = 1, 8
            elemk(i, j) = 0.
         END DO
      END DO

C     Loop over integration points
      DO ip = 1, 4
C        get cb, b, and jacobian at integration point ip
         s = xni(ip, 1)
         t = xni(ip, 2)
         CALL bmat(x, n, noc, s, t, b, cb, c, dj, maxnodes, maxelem)
C        form se
         DO i = 1, 8
            DO j = 1, 8
               DO k = 1, 3
                  elemk(i, j) = elemk(i, j) + b(k, i)*cb(k, j)*
     &             dj*th*wni(ip)
               END DO
            END DO
         END DO
      END DO
      RETURN
      END

C     __________________________________________________________________
      SUBROUTINE input(ne, nn, nd, nl, nm, linp, lout, th, x, pm, u,
     & nu, mat, noc, f, maxnodes, maxelem, maxmat, maxbdc, maxdof)

C     read and echo problem input data

      REAL x(maxnodes, 2), pm(maxmat, 2), u(maxbdc), f(maxdof)
      INTEGER nu(maxbdc), mat(maxelem), noc(maxelem, 4)


C     nodal coordinates
      READ (linp, *) (n, (x(n, j), j = 1, 2), i = 1, nn)
      WRITE (lout, *)
      WRITE (lout, *) ' Node     X-Coord.         Y-Coord.'
      WRITE (lout, '((i4, 2(5x, e11.4)))') (i, x(i, 1), x(i, 2), i = 1,
     & nn)

C     specified displacements
      READ (linp, *) (nu(i), u(i), i = 1, nd)
      WRITE (lout, *)
      WRITE (lout, *) ' DOF#  Specified disp.'
      WRITE (lout, '(i4, 5x, e10.4)') (nu(i), u(i), i = 1, nd)

C     specified loads
      WRITE (lout, *)
      WRITE (lout, *) ' DOF#  Specified Load'
      DO i = 1, nl
         READ (linp, *) n, f(n)
         WRITE (lout, '(i4, 5x, e10.4)') n, f(n)
      END DO
C     material properties
      READ (linp, *) (k, (pm(k, j), j = 1, 2), i = 1, nm)
      WRITE (lout, *)
      WRITE (lout, *) ' Material#   E               Nu'
      WRITE (lout, '(i4, 2(5x, e11.4))') (i, pm(i, 1), pm(i, 2), i = 1,
     & nm)
      READ (linp, *) th
      WRITE (lout, *) '  Thickness = ', th

C     element connectivity
      READ (linp, *) (k, (noc(k, j), j = 1, 4), mat(k), i = 1, ne)
      WRITE (lout, *)
      WRITE (lout, *) ' Element#  -------Nodes-------   Material#'
      WRITE (lout, '(1x, 6i6)') (i, (noc(i, j), j = 1, 4), mat(i), i =
     & 1, ne)

      RETURN
      END

C     __________________________________________________________________
      SUBROUTINE integ(xni, wni)
C     Gaussian integration points and weights
C     2x2 integration

      DIMENSION xni(4, 2), wni(4)

      c = 1.0/sqrt(3.0)
      xni(1, 1) =  - c
      xni(1, 2) =  - c
      xni(2, 1) = c
      xni(2, 2) =  - c
      xni(3, 1) = c
      xni(3, 2) = c
      xni(4, 1) =  - c
      xni(4, 2) = c
      wni(1) = 1
      wni(2) = 1
      wni(3) = 1
      wni(4) = 1
      RETURN
      END

C     __________________________________________________________________
      SUBROUTINE prstrs(str, s1, s2, ang)

C     Given normal stresses compute principal stresses

      REAL str(3)
      DATA degres/57.29577951/
C     degres = 180/pi:  to convert radians to degrees

      IF ( str(3) .EQ. 0 ) THEN
         s1 = str(1)
         s2 = str(2)
         ang = 0
         IF ( s1 .LE. s2 ) THEN
            s1 = str(2)
            s2 = str(1)
            ang = 90
         END IF
      ELSE
         c = .5*(str(1) + str(2))
         r = sqrt(.25*(str(1) - str(2))**2 + (str(3))**2)
         s1 = c + r
         s2 = c - r
         IF ( c .LE. str(1) ) THEN
            ang = degres*atan(str(3)/(str(1) - s2))
         ELSE
            ang = degres*atan(str(3)/(s1 - str(1)))
            IF ( str(3) .GT. 0 ) ang = 90 - ang
            IF ( str(3) .LT. 0 ) ang =  - 90 - ang
         END IF
      END IF
      RETURN
      END
