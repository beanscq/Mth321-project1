"""Authoritative Robertson chemical kinetics model.

Reaction network (labels are MNEMONIC only)::

    A     -> B        k1 = 0.04
    B + B -> B + C    k2 = 3e7
    B + C -> A + C    k3 = 1e4

The ODE system below is AUTHORITATIVE and fixes the mass-action coefficient
convention.  For the schematic ``B + B -> B + C`` step the *net* stoichiometry
is one B consumed and one C produced, so its contribution to the vector field is

    f_B  += -k2 * y2**2
    f_C  += +k2 * y2**2

In particular no additional factor of two is applied to the displayed
``3e7 * y2**2`` term.

Vector field::

    y1' = -k1*y1 + k3*y2*y3
    y2' =  k1*y1 - k3*y2*y3 - k2*y2**2
    y3' =  k2*y2**2                       with k2 = 3e7  (i.e. 3e7*y2**2)

Invariants::

    (1,1,1) . f(y) == 0        =>  y1 + y2 + y3 == 1  for all exact trajectories
                                  =>  J always has the structural eigenvalue 0

Chemical reading
----------------
* ``y1, y2, y3`` are normalised concentrations of A, B, C.
* B is a short-lived intermediate: A feeds B slowly (k1 = 0.04), B is consumed
  very fast by the quadratic step (k2 = 3e7) and converted to C.
* k1 -> k2 spans ~1e8, so fast and slow chemical time scales coexist.  That is
  the physical origin of the stiffness.
* Concentrations must stay non-negative: a negative concentration is not a
  chemically admissible numerical result.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Authoritative parameters -- DO NOT CHANGE
# ---------------------------------------------------------------------------
K1: float = 0.04
K2: float = 3.0e7
K3: float = 1.0e4

Y0: np.ndarray = np.array([1.0, 0.0, 0.0])
T_SPAN: tuple[float, float] = (0.0, 40.0)
TOTAL_CONCENTRATION: float = 1.0

# component indices
I_A, I_B, I_C = 0, 1, 2

# analytical orientation values published in the project brief (t = 40).
# Used ONLY as a regression target, never as a substitute for our own reference.
BRIEF_Y_AT_T40: np.ndarray = np.array([0.7158271, 9.1855e-6, 0.2841637])

# projection operators used for the reduced system, see reduced_jacobian()
PROJ_S: np.ndarray = np.array([[1.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0]])          # 2x3 : selects y1, y2
PROJ_T: np.ndarray = np.array([[1.0, 0.0],
                               [0.0, 1.0],
                               [-1.0, -1.0]])              # 3x2 : dy/dz with y3 = 1-y1-y2

LEFT_NULL_VECTOR: np.ndarray = np.ones(3) / np.sqrt(3.0)   # (1,1,1)/sqrt(3)


def _asfloat_or_complex(y: np.ndarray) -> np.ndarray:
    """Cast to float64, but PRESERVE complex128.

    Complex preservation is what makes the complex-step derivative check in
    ``tests/test_model.py`` exact to machine precision.  A naive
    ``np.asarray(y, dtype=float)`` would silently discard the imaginary part.
    """
    y = np.asarray(y)
    return y.astype(np.complex128 if y.dtype.kind == "c" else np.float64, copy=False)


# ---------------------------------------------------------------------------
# vector field
# ---------------------------------------------------------------------------
def rates(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mass-action reaction rates ``(r1, r2, r3)``.

    ``r1 = k1*y1`` (A -> B), ``r2 = k2*y2**2`` (B+B -> B+C, net -1 B/+1 C),
    ``r3 = k3*y2*y3`` (B+C -> A+C).
    """
    y = _asfloat_or_complex(y)
    r1 = K1 * y[..., I_A]
    r2 = K2 * y[..., I_B] ** 2
    r3 = K3 * y[..., I_B] * y[..., I_C]
    return r1, r2, r3


def f(t: float, y: np.ndarray) -> np.ndarray:
    """Authoritative right-hand side.  ``y`` may carry a leading batch axis."""
    y = _asfloat_or_complex(y)
    r1, r2, r3 = rates(y)
    out = np.empty_like(y)
    out[..., I_A] = -r1 + r3
    out[..., I_B] = r1 - r2 - r3
    out[..., I_C] = r2
    return out


# ---------------------------------------------------------------------------
# Jacobians
# ---------------------------------------------------------------------------
def jacobian(t: float, y: np.ndarray) -> np.ndarray:
    """Analytic 3x3 Jacobian J(y) of the authoritative vector field.

    ::

        J = [[ -0.04,             1e4*y3,          1e4*y2],
             [  0.04, -1e4*y3 - 6e7*y2,          -1e4*y2],
             [     0,            6e7*y2,               0]]
    """
    y = _asfloat_or_complex(y)
    y2, y3 = y[I_B], y[I_C]
    return np.array([
        [-K1,                            K3 * y3,   K3 * y2],
        [K1,  -K3 * y3 - 2.0 * K2 * y2,            -K3 * y2],
        [0.0,                2.0 * K2 * y2,             0.0],
    ], dtype=y.dtype)


def reduced_jacobian(t: float, y: np.ndarray) -> np.ndarray:
    """2x2 Jacobian of the reduced system obtained by eliminating y3 = 1-y1-y2.

    With ``z = (y1, y2)`` and ``g(z) = S f(y(z))`` the chain rule gives::

        B_red = S * J * T

    This is a genuine projection, **not** the upper-left 2x2 block of J.  The
    two differ: the upper-left block misses the ``6e11 * y2**2`` term of the
    determinant, which is a 3.3x error at t = 40.

    The two eigenvalues of ``B_red`` are exactly the two NON-ZERO eigenvalues
    of J, because the structural eigenvalue is 0.
    """
    return PROJ_S @ jacobian(t, y) @ PROJ_T


def reduced_jacobian_closed_form(t: float, y: np.ndarray) -> np.ndarray:
    """Closed form of :func:`reduced_jacobian`, used as an independent check.

    ::

        B_red = [[ -k1 - k3*y2 ,    k3*y3 - k3*y2            ],
                 [  k1 + k3*y2 , -k3*y3 + k3*y2 - 2*k2*y2    ]]

        tr(B_red)  = -k1 - k3*y3 - 2*k2*y2
        det(B_red) = 2*k2*y2 * (k1 + k3*y2)  =  2.4e6*y2 + 6e11*y2**2
    """
    y = np.asarray(y, dtype=float)
    y2, y3 = y[I_B], y[I_C]
    a = K3 * y2
    return np.array([
        [-K1 - a,          K3 * y3 - a],
        [K1 + a,  -K3 * y3 + a - 2.0 * K2 * y2],
    ])


def reduced_jacobian_invariants(y: np.ndarray):
    """Exact ``(trace, det)`` of the reduced Jacobian, vectorised over a batch.

    ``tr = -k1 - k3*y3 - 2*k2*y2`` and ``det = 2*k2*y2*(k1 + k3*y2)``.
    """
    y = np.asarray(y, dtype=float)
    y2, y3 = y[..., I_B], y[..., I_C]
    tr = -K1 - K3 * y3 - 2.0 * K2 * y2
    det = (2.0 * K2 * y2) * (K1 + K3 * y2)
    return tr, det


def reduced_eigenvalues_analytic(y: np.ndarray) -> np.ndarray:
    """The two non-zero eigenvalues of J via the quadratic formula.

    Solving ``lam**2 - tr*lam + det = 0`` with the exact invariants above.
    Vectorised: for ``y`` of shape ``(3,)`` returns shape ``(2,)``; for ``y`` of
    shape ``(n, 3)`` returns ``(n, 2)``.  Always complex, so that the narrow
    conjugate-pair window near t ~ 1.7e-8 is represented faithfully.

    This is preferred over ``np.linalg.eig`` because the two eigenvalues form a
    near-double root during the induction layer, where ``eig`` loses accuracy.
    """
    tr, det = reduced_jacobian_invariants(y)
    disc = tr * tr - 4.0 * det
    root = np.sqrt(np.asarray(disc, dtype=complex))
    return np.stack([(tr + root) / 2.0, (tr - root) / 2.0], axis=-1)


def eigenvalues_3x3_numeric(t: float, y: np.ndarray) -> np.ndarray:
    """All three eigenvalues of J from a general eigensolver (cross-check only).

    Returns complex values sorted by decreasing ``|Re lam|`` so that the
    structural zero lands last.
    """
    lam = np.linalg.eigvals(jacobian(t, y))
    order = np.argsort(-np.abs(lam.real))
    return lam[order]


# ---------------------------------------------------------------------------
# invariants / diagnostics
# ---------------------------------------------------------------------------
def invariant_sum(y: np.ndarray) -> np.ndarray:
    """``y1 + y2 + y3``."""
    return np.asarray(y, dtype=float).sum(axis=-1)


def invariant_defect(y: np.ndarray) -> np.ndarray:
    """``|(y1 + y2 + y3) - 1|`` -- the linear-invariant defect."""
    return np.abs(invariant_sum(y) - TOTAL_CONCENTRATION)


def principal_minors_2x2(t: float, y: np.ndarray) -> tuple[float, float, float, float]:
    """Sum of the three principal 2x2 minors of J, plus the three minors.

    The certificate is ``m2 == det(B_red)``; together with ``tr(J) == tr(B_red)``
    this proves that the eigenvalues of ``B_red`` are exactly the non-zero
    eigenvalues of J.
    """
    J = jacobian(t, y)
    m12 = float(np.linalg.det(J[np.ix_([0, 1], [0, 1])]))
    m13 = float(np.linalg.det(J[np.ix_([0, 2], [0, 2])]))
    m23 = float(np.linalg.det(J[np.ix_([1, 2], [1, 2])]))
    return m12 + m13 + m23, m12, m13, m23
