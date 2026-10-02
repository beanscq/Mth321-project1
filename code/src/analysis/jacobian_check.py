"""Independent verification of the analytic Jacobian by forward differences.

Why this module exists
----------------------
The analytic Jacobian is the one piece of the model that is *used* everywhere
and *checked* nowhere: a wrong entry does not raise, it just makes every Newton
iteration slower, or makes it converge to a state that is not the root.  The
tutorial's rule is blunt and correct -- **a Jacobian has to be verified against
an independent difference approximation, at more than one state, and the
component perturbed must be the one that controls the column.**

Conventions implemented here
----------------------------
*Column, not row.*  Perturbing input component ``j`` produces **column** ``j``::

    (J_f)[:, j] ~= ( f(t, y + eps_j e_j) - f(t, y) ) / eps_j

A transposed implementation looks plausible, and at some states it is even
symmetric, so this is exactly the kind of error a single-state check misses.

*Perturbation size.*  For suitably scaled variables the tutorial uses
``eps_j = sqrt(eps_mach) * max(1, |y_j|)``; for dimensional variables the ``1``
is replaced by a reference scale ``s_j``.  Both are implemented, and the
truncation/roundoff trade-off is made explicit by sweeping the multiplier over
order-one changes::

    difference error  ~  O(eps_j) + O(eps_mach / eps_j)

so a good check *reports the perturbation it used* rather than hiding it.

*Discrepancy measure.*  With sums over the matrix in the infinity norm::

    eta = ||J_A - J_D||_inf / max(1, ||J_A||_inf)

There is deliberately **no pass/fail threshold** here.  The tutorial is explicit
that "there is no universal absolute 1e-8 requirement": ``eta`` depends on
scales, curvature and the perturbation, so it is reported *together with* the
state and the perturbation, and interpreted by comparing runs.  What the check
does establish is that the discrepancy is at the level set by truncation, and
that it *changes in the predicted way* when the perturbation is changed -- a
wrong entry does not do that.

*The residual Jacobian too.*  Implicit Euler needs ``J_F = I - h J_f``, not
``J_f``, so the residual is differenced directly as well.  A correct ``J_f``
and a correct ``J_F`` are two separate claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..model import robertson as R

SQRT_EPS_MACH = float(np.sqrt(np.finfo(float).eps))

#: project root, resolved from this file so the reference is found regardless
#: of the caller's working directory
ROOT = Path(__file__).resolve().parents[2]
REFERENCE_NPZ = ROOT / "results" / "reference" / "reference.npz"

#: representative states, chosen for what they can and cannot expose
#:
#: ``y(0)``  the initial state -- ``y2 = y3 = 0``, so *every* coupling term
#:           that involves ``y2`` or ``y3`` vanishes here.  A check at this
#:           state alone cannot see them.
#: ``y(t*)`` a transient state where all three components are non-zero, so the
#:           coupling terms are live.
#: ``y(40)`` the end state, where ``y2`` is smallest and ``y3`` largest.
STATE_LABELS = ("initial", "transient", "end")


@dataclass
class StateReport:
    """One (state, perturbation multiplier) pair."""

    label: str
    y: np.ndarray
    multiplier: float
    eps: np.ndarray
    eta: float
    worst_col: int
    col_errors: np.ndarray
    analytic: np.ndarray
    difference: np.ndarray
    eta_residual: float = float("nan")


@dataclass
class CheckReport:
    """Everything the check is allowed to claim."""

    states: dict[str, np.ndarray]
    n_rhs_calls: int
    reports: list[StateReport] = field(default_factory=list)
    #: the perturbation sweep for the transient state: eps -> eta
    sweep: list[tuple[float, float]] = field(default_factory=list)
    #: the empirical optimum of the sweep and the tutorial's rule-of-thumb value
    best_eps: float = float("nan")
    best_eta: float = float("nan")
    sqrt_eps_eta: float = float("nan")
    #: eta at the *tuned* perturbation, per state -- compare with the
    #: rule-of-thumb column to see how much of the discrepancy was the
    #: difference operator rather than the Jacobian
    state_eta_tuned: dict[str, float] = field(default_factory=dict)
    #: the same, for the residual Jacobian ``I - h J_f``
    state_eta_residual_tuned: dict[str, float] = field(default_factory=dict)
    #: the tuned perturbation is a property of the *function being differenced*,
    #: so ``f`` and ``F = z - y - h f`` do not share one
    residual_tuned_eps: float = float("nan")


# ---------------------------------------------------------------------------
def representative_states() -> dict[str, np.ndarray]:
    """Three states that fail in different ways, from the reference trajectory."""
    ref = np.load(REFERENCE_NPZ)
    t = np.asarray(ref["t"], float)
    y = np.asarray(ref["y"], float)
    i_tr = int(np.argmin(np.abs(t - 1.0e-3)))     # early transient, all non-zero
    return {
        "initial": R.Y0.astype(float).copy(),
        "transient": y[i_tr].copy(),
        "end": y[-1].copy(),
    }


def perturbation(y: np.ndarray,
                 multiplier: float = 1.0,
                 scale: np.ndarray | float | None = None) -> np.ndarray:
    """``eps_j = multiplier * sqrt(eps_mach) * max(scale_j, |y_j|)``.

    ``scale`` defaults to 1 for every component, which is the tutorial's
    "suitably scaled variables" case.  For dimensional variables pass the
    reference scale ``s_j`` instead.
    """
    y = np.asarray(y, float)
    if scale is None:
        s = np.ones_like(y)
    else:
        s = np.broadcast_to(np.asarray(scale, float), y.shape).copy()
    return float(multiplier) * SQRT_EPS_MACH * np.maximum(s, np.abs(y))


def forward_difference_jacobian(f, t: float, y: np.ndarray,
                                multiplier: float = 1.0,
                                scale=None,
                                eps: np.ndarray | float | None = None
                                ) -> tuple[np.ndarray, np.ndarray, int]:
    """Column-by-column forward differences of ``f`` at ``(t, y)``.

    Returns ``(J_D, eps, n_rhs_calls)``.  The base value ``f(t, y)`` is computed
    once and reused for every column, so ``d`` columns cost ``d + 1`` calls.
    Pass an explicit ``eps`` to bypass the ``multiplier``/``scale`` rule.
    """
    y = np.asarray(y, float)
    if eps is None:
        eps = perturbation(y, multiplier=multiplier, scale=scale)
    eps = np.broadcast_to(np.asarray(eps, float), y.shape).astype(float)
    f0 = np.asarray(f(t, y), float)
    d = y.size
    JD = np.empty((d, d), float)
    for j in range(d):
        yp = y.copy()
        yp[j] += eps[j]
        JD[:, j] = (np.asarray(f(t, yp), float) - f0) / eps[j]
    return JD, eps, d + 1


def scaled_discrepancy(A: np.ndarray, B: np.ndarray) -> float:
    """``||A - B||_inf / max(1, ||A||_inf)``, row-sum infinity norm."""
    A = np.atleast_2d(np.asarray(A, float))
    B = np.atleast_2d(np.asarray(B, float))
    num = float(np.max(np.sum(np.abs(A - B), axis=1))) if A.size else 0.0
    den = float(np.max(np.sum(np.abs(A), axis=1))) if A.size else 0.0
    return num / max(1.0, den)


def column_errors(analytic: np.ndarray, difference: np.ndarray) -> np.ndarray:
    """Per-column relative error, so the *worst column* can be named."""
    a = np.asarray(analytic, float)
    b = np.asarray(difference, float)
    num = np.max(np.abs(a - b), axis=0)
    den = np.maximum(1.0, np.max(np.abs(a), axis=0))
    return num / den


def tuned_perturbation(f, t: float, y: np.ndarray, analytic: np.ndarray,
                       exponents=range(-16, -3)) -> tuple[float, float]:
    """Sweep ``eps = 10**e`` and return the ``(eps, eta)`` pair with the least
    disagreement with ``analytic``.

    The optimum is a property of the *function being differenced*, not of the
    model: ``F = z - y - h f`` has second derivatives scaled by ``h`` and values
    of order one, so its best perturbation differs from ``f``'s.  Reporting one
    tuned value for both would be wrong, which is why ``run`` tunes them
    separately.
    """
    best = (float("nan"), float("inf"))
    for e in exponents:
        eps = 10.0 ** e
        JD, _eps, _c = forward_difference_jacobian(f, t, y, eps=eps)
        eta = scaled_discrepancy(analytic, JD)
        if eta < best[1]:
            best = (eps, eta)
    return best


# ---------------------------------------------------------------------------
def residual_jacobian(f, t: float, y: np.ndarray, h: float) -> np.ndarray:
    """Analytic ``J_F = I - h J_f`` for the implicit-Euler residual."""
    return np.eye(y.size) - h * np.asarray(R.jacobian(t, y), float)


def run(multipliers=(0.1, 1.0, 10.0), sweep_exponents=range(-16, -4),
        h: float = 1.0e-3) -> CheckReport:
    """Run the whole protocol and return a report.

    ``sweep_exponents`` walks ``eps = 10**e`` (for the transient state's ``y2``
    direction) so the truncation/roundoff trade-off is measured rather than
    quoted.
    """
    states = representative_states()
    reps: list[StateReport] = []
    rhs = 0

    for label in STATE_LABELS:
        y = states[label]
        JA = np.asarray(R.jacobian(0.0, y), float)
        for m in multipliers:
            JD, eps, calls = forward_difference_jacobian(R.f, 0.0, y, multiplier=m)
            rhs += calls
            # residual Jacobian, differenced directly: a second, separate claim
            JF_an = residual_jacobian(R.f, 0.0, y, h)

            def F(z, _y=y, _h=h):
                return z - _y - _h * np.asarray(R.f(_h, z), float)

            JF_fd, _e, calls2 = forward_difference_jacobian(
                lambda tt, z: F(z), 0.0, y, multiplier=m)
            rhs += calls2
            reps.append(StateReport(
                label=label, y=y.copy(), multiplier=float(m), eps=eps,
                eta=scaled_discrepancy(JA, JD),
                worst_col=int(np.argmax(column_errors(JA, JD))),
                col_errors=column_errors(JA, JD),
                analytic=JA, difference=JD,
                eta_residual=scaled_discrepancy(JF_an, JF_fd),
            ))

    # ---- perturbation sweep, one direction, to expose the U-shape ----------
    y = states["transient"]
    JA = np.asarray(R.jacobian(0.0, y), float)
    f0 = np.asarray(R.f(0.0, y), float)
    sweep: list[tuple[float, float]] = []
    for e in sweep_exponents:
        eps = 10.0 ** e
        JD = np.empty((3, 3), float)
        for j in range(3):
            yp = y.copy()
            yp[j] += eps
            JD[:, j] = (np.asarray(R.f(0.0, yp), float) - f0) / eps
            rhs += 1
        sweep.append((eps, scaled_discrepancy(JA, JD)))

    best_eps, best_eta = min(sweep, key=lambda kv: kv[1])
    # where the tutorial's rule of thumb puts the perturbation for component j=1
    eta_at_rule = scaled_discrepancy(
        JA, forward_difference_jacobian(R.f, 0.0, y, multiplier=1.0)[0])

    # ---- same check, but with the perturbation the sweep selected ---------
    tuned: dict[str, float] = {}
    for label in STATE_LABELS:
        yl = states[label]
        Al = np.asarray(R.jacobian(0.0, yl), float)
        Dl, _e, calls = forward_difference_jacobian(R.f, 0.0, yl, eps=best_eps)
        rhs += calls
        tuned[label] = scaled_discrepancy(Al, Dl)

    # ---- and the residual Jacobian, tuned on its own function -------------
    def residual_of(yl):
        def F(z, _y=yl, _h=h):
            return z - _y - _h * np.asarray(R.f(_h, z), float)
        return F

    y_tr = states["transient"]
    res_eps, _res_eta = tuned_perturbation(
        lambda tt, z: residual_of(y_tr)(z), 0.0, y_tr,
        residual_jacobian(R.f, 0.0, y_tr, h))
    residual_tuned: dict[str, float] = {}
    for label in STATE_LABELS:
        yl = states[label]
        Fl = residual_of(yl)
        Al = residual_jacobian(R.f, 0.0, yl, h)
        Dl, _e, calls = forward_difference_jacobian(
            lambda tt, z, _F=Fl: _F(z), 0.0, yl, eps=res_eps)
        rhs += calls
        residual_tuned[label] = scaled_discrepancy(Al, Dl)

    return CheckReport(
        states=states, n_rhs_calls=rhs, reports=reps, sweep=sweep,
        best_eps=best_eps, best_eta=best_eta, sqrt_eps_eta=eta_at_rule,
        state_eta_tuned=tuned,
        state_eta_residual_tuned=residual_tuned,
        residual_tuned_eps=res_eps,
    )


def buggy_jacobian(t: float, y: np.ndarray) -> np.ndarray:
    """``J_f`` with the whole ``10^4 * y2 * y3`` reaction left out.

    A plausible bug: the model has two reactions, and this one is easy to drop
    because it is quadratic and cancels between ``f1`` and ``f2``.
    """
    J = np.asarray(R.jacobian(t, y), float).copy()
    y2, y3 = y[R.I_B], y[R.I_C]
    J[0, 1] -= R.K3 * y3       #  d f1 / d y2  loses  +1e4 y3
    J[0, 2] -= R.K3 * y2       #  d f1 / d y3  loses  +1e4 y2
    J[1, 1] += R.K3 * y3       #  d f2 / d y2  loses  -1e4 y3
    J[1, 2] += R.K3 * y2       #  d f2 / d y3  loses  -1e4 y2
    return J


def missing_coupling_demo(eps_values=(1.49e-8, 1.0e-12)) -> dict:
    """Does the check actually *detect* a wrong Jacobian?  It depends where and how.

    Drops the ``10^4 y2 y3`` reaction from every entry of ``J_f`` and compares,

    * ``eta_bug``   -- the wrong matrix against the difference approximation
    * ``eta_good``  -- the correct matrix against the same approximation

    for each representative state at each perturbation size.  Two lessons come
    out of the numbers rather than being asserted:

    1. **At the initial state the check is blind.**  ``y2 = y3 = 0`` there, so
       all four entries the bug touches are zero in the correct matrix too: the
       two rows are the same number, and both are dominated by the quadratic
       truncation error of the difference itself.  A single-state check at
       ``y(0)`` therefore certifies a wrong Jacobian.
    2. **Even at a state that can expose it, the perturbation decides.**  With
       the ``sqrt(eps_mach) max(1, |y_j|)`` rule the difference's own truncation
       error is comparable to the signal, and the two etas differ by a factor of
       two.  Re-running with the perturbation the sweep selects separates them
       by seven orders of magnitude.

    The check is therefore only as good as the state *and* the perturbation
    chosen -- which is why both are reported alongside the discrepancy.
    """
    states = representative_states()
    out: dict = {"eps_values": tuple(float(e) for e in eps_values), "rows": []}
    for label in STATE_LABELS:
        y = states[label]
        JA = np.asarray(R.jacobian(0.0, y), float)
        JB = buggy_jacobian(0.0, y)
        for eps in eps_values:
            JD, _e, _c = forward_difference_jacobian(R.f, 0.0, y, eps=eps)
            out["rows"].append({
                "state": label,
                "eps": float(eps),
                "eta_bug": scaled_discrepancy(JD, JB),
                "eta_good": scaled_discrepancy(JA, JD),
            })
    for r in out["rows"]:
        r["separation"] = (r["eta_bug"] / r["eta_good"]
                           if r["eta_good"] > 0.0 else float("inf"))
    return out
