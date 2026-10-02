"""Absolute stability regions of the three methods.

For a scalar test equation ``y' = lam*y`` a one-step method gives
``y_{n+1} = R(z) y_n`` with ``z = h*lam``.  The absolute stability region is
``{z : |R(z)| <= 1}``.

* Explicit Euler    ``R(z) = 1 + z``              -> closed disk |1+z| <= 1,
                                                    real-axis interval [-2, 0]
* Classical RK4     ``R(z) = sum_{k=0..4} z^k/k!``-> bounded region,
                                                    real-axis interval (-2.785, 0)
* Implicit Euler    ``R(z) = 1/(1-z)``            -> |1-z| >= 1, i.e. the whole
                                                    left half plane (A-stable),
                                                    and R(inf) = 0 (L-stable)

Because the two non-zero Robertson eigenvalues are (except for a numerically
negligible window, see ``eigen.py``) real and negative, the relevant slice of
each region is its real-axis interval.  The overlay of the actual spectra
``h * lam(t)`` makes the explicit step-size restriction visible geometrically.
"""

from __future__ import annotations

import numpy as np

# R(x) = 1  <=>  x * (1 + x/2 + x^2/6 + x^3/24) = 0, so the non-trivial real
# axis stability limits are the roots of this CUBIC (highest order first).
RK4_R_MINUS_ONE_POLY = np.array([1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0])


def R_explicit_euler(z: np.ndarray) -> np.ndarray:
    return 1.0 + z


def R_rk4(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=complex)
    return 1.0 + z + z ** 2 / 2.0 + z ** 3 / 6.0 + z ** 4 / 24.0


def R_implicit_euler(z: np.ndarray) -> np.ndarray:
    """``1/(1-z)``.  The pole at ``z = 1`` is capped so that the level-set
    computation stays finite; ``|R|`` is enormous there and the point is
    correctly classified as unstable."""
    z = np.asarray(z, dtype=complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = 1.0 / (1.0 - z)
    return np.where(np.isfinite(R), R, 1e30 + 0j)


STABILITY_FUNCTIONS = {
    "Explicit Euler": R_explicit_euler,
    "RK4": R_rk4,
    "Implicit Euler": R_implicit_euler,
}

METHOD_ORDER = {"Explicit Euler": 1, "RK4": 4, "Implicit Euler": 1}
METHOD_COLORS = {"Explicit Euler": "#185FA5", "RK4": "#27500A", "Implicit Euler": "#854F0B"}


DEFAULT_RE_RANGE = (-6.0, 2.0)
DEFAULT_IM_RANGE = (-4.5, 4.5)


def stability_grid(re_range=DEFAULT_RE_RANGE,
                   im_range=DEFAULT_IM_RANGE,
                   n: int = 801):
    """Return ``(Re, Im, Z)`` where ``Z`` is the complex h*lam mesh."""
    re = np.linspace(re_range[0], re_range[1], int(n))
    im = np.linspace(im_range[0], im_range[1], int(n))
    RE, IM = np.meshgrid(re, im)
    return re, im, RE + 1j * IM


def stability_margin_grid(method: str, re_range=DEFAULT_RE_RANGE,
                          im_range=DEFAULT_IM_RANGE, n: int = 801):
    """``|R(z)| - 1`` on the mesh; values <= 0 mark the stable set."""
    _, _, Z = stability_grid(re_range, im_range, n)
    return np.abs(STABILITY_FUNCTIONS[method](Z)) - 1.0


def real_axis_stability_interval(method: str) -> tuple[float, float]:
    """Interval ``[a, 0]`` of the negative real axis contained in the region.

    ``a`` is computed from the method's stability polynomial, not hard-coded.
    """
    if method == "Explicit Euler":
        return -2.0, 0.0
    if method == "Implicit Euler":
        return -np.inf, 0.0
    if method == "RK4":
        roots = np.roots(RK4_R_MINUS_ONE_POLY)   # x^3 + 4x^2 + 12x + 24 = 0
        real_neg = [r.real for r in roots if abs(r.imag) < 1e-12 and r.real < 0]
        if not real_neg:
            raise RuntimeError("could not locate the RK4 real-axis stability limit")
        return float(min(real_neg)), 0.0
    raise KeyError(method)


def max_stable_real_step(method: str) -> float:
    """Largest ``h`` allowed by ``h * lam`` on the negative real axis."""
    a, _ = real_axis_stability_interval(method)
    return float(np.inf if np.isneginf(a) else abs(a))


def overlay_spectra(h: float, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``h * lam(t)`` for the two non-zero eigenvalues, as a flat complex array."""
    from ..model import robertson as R

    lam = R.reduced_eigenvalues_analytic(np.asarray(y, dtype=float))
    return (h * lam).ravel()


def summary_table(methods=("Explicit Euler", "RK4", "Implicit Euler")) -> list[dict]:
    rows = []
    for m in methods:
        a, _ = real_axis_stability_interval(m)
        rows.append({
            "method": m,
            "order": METHOD_ORDER[m],
            "real_axis_interval": "-inf < x <= 0" if np.isneginf(a) else f"[{a:.4f}, 0]",
            "A_stable": bool(m == "Implicit Euler"),
            "max_stable_real_h_lambda": max_stable_real_step(m),
            "h_limit_for_lambda_max_3p39e3": (
                "unbounded" if np.isneginf(a) else f"{abs(a) / 3.39278e3:.3e}"
            ),
        })
    return rows
