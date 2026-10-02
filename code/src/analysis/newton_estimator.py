"""The Newton tolerance as a first-class knob: when does it change the *method*?

The claim being measured
------------------------
Implicit Euler is solved by Newton, and the solver stops on a residual
tolerance.  A loose tolerance is not merely "less accurate": past a computable
threshold it changes *which method is being run*, and the step-doubling error
estimator does not notice.

The mechanism is exact and easy to state.  ``ImplicitEuler`` starts from the
explicit-Euler predictor ``y_pred = y_n + h f(y_n)``, and Newton tests the
residual *before* taking any update.  Expanding,

    G(y_pred) = y_pred - y_n - h f(y_pred)
              = h f(y_n) - h f(y_n + h f(y_n))
              = -h^2 J_f(y_n) f(y_n) + O(h^3)

so the predictor is accepted -- Newton performs **zero updates**, and the step
is literally an explicit-Euler step -- whenever

    h  <=  sqrt( tol / ||J_f f||_inf ) .

That is a prediction, not an observation: ``predictor_threshold`` computes it
from the state alone, and the tests check the observed transition against it.

Why the estimator cannot see it
------------------------------
Both Euler methods are first order, so ``E = ||y_f - y_c||`` has the same
``h^2`` local-error structure in both, and the h-scaling exponent stays at 2.
The number barely moves.  So a controller at ``tol`` keeps accepting steps that
explicit Euler cannot stably take: for this problem the explicit stability bound
is ``h < 5.4e-4``, while ``tol = 1e-6`` makes every step implicit-in-name-only
up to ``h = 1.1e-2`` -- twenty times the stability bound.

Neither cheap diagnostic catches this.  The invariant defect is *exactly*
restored by one Newton update for any tolerance (it is a left null vector
argument), and the local error estimate is blind as above.  Only a measured
trajectory error sees it, which is why the tolerance is reported next to every
result rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..integrators.base import StepStats
from ..integrators.implicit_euler import ImplicitEuler
from ..model import robertson as R

#: project root -- resolved from this file so the reference can be found no
#: matter which directory the caller happens to be in.
ROOT = Path(__file__).resolve().parents[2]

#: the reference trajectory every state in this study is drawn from
REFERENCE_NPZ = ROOT / "results" / "reference" / "reference.npz"

#: tolerances swept; the interesting transition happens between 1e-10 and 1e-6
NTOLS = (1.0e-14, 1.0e-12, 1.0e-10, 1.0e-8, 1.0e-7, 1.0e-6, 1.0e-4)

#: step sizes swept at a fixed state; all of them sit above the explicit
#: stability bound of this problem (5.4e-4), which is the point
HS = (1.0e-4, 2.0e-4, 5.0e-4, 1.0e-3, 2.0e-3, 5.0e-3, 1.0e-2)


@dataclass
class StepRow:
    ntol: float
    h: float
    iters_max: int          # largest Newton update count over the three solves
    zero_updates: int       # solves that took NO update at all
    n_solves: int
    err_estimate: float     # E = ||y_f - y_c||, the step-doubling signal
    #: updates taken by each of the three solves, in order: the coarse step
    #: first, then the two half steps.  This is what matters for the estimate:
    #: ``E`` is a difference between the coarse and the fine side, so if the two
    #: sides were solved to different algebraic accuracy the mismatch lands
    #: straight in the estimate.
    iters_each: tuple[int, int, int] = (0, 0, 0)

    @property
    def is_explicit(self) -> bool:
        """Every solve in the trial was the explicit-Euler predictor."""
        return self.zero_updates == self.n_solves

    @property
    def sides_unequal(self) -> bool:
        """The coarse side and the fine side were solved to different accuracy."""
        return self.iters_each[0] != max(self.iters_each[1], self.iters_each[2])


@dataclass
class EstimatorStudy:
    t0: float
    y0: np.ndarray
    jf_norm: float
    rows: list[StepRow] = field(default_factory=list)
    #: ntol -> exponent of E ~ h^p, fitted over the sweep
    slope_by_ntol: dict[float, float] = field(default_factory=dict)
    #: ntol -> the h below which the predictor is accepted, from the formula
    threshold_by_ntol: dict[float, float] = field(default_factory=dict)
    #: ntol -> number of swept steps that were, in fact, explicit
    explicit_steps_by_ntol: dict[float, int] = field(default_factory=dict)
    #: the residual bound identity, checked rather than asserted
    bound_check_max_rel_err: float = float("nan")


# ---------------------------------------------------------------------------
def state_at(t0: float = 1.0) -> tuple[float, np.ndarray, np.ndarray]:
    """A smooth state from the reference trajectory, with its derivative."""
    ref = np.load(REFERENCE_NPZ)
    t = np.asarray(ref["t"], float)
    y = np.asarray(ref["y"], float)
    i = int(np.argmin(np.abs(t - t0)))
    return float(t[i]), y[i].copy(), np.asarray(R.f(t[i], y[i]), float)


def jf_norm(t: float, y: np.ndarray) -> float:
    """``||J_f f||_inf`` at a state -- the constant in the threshold formula."""
    return float(np.max(np.abs(np.asarray(R.jacobian(t, y), float)
                               @ np.asarray(R.f(t, y), float))))


def predictor_residual(t: float, y: np.ndarray, h: float) -> float:
    """``||G(y_pred)||_inf``: what Newton tests first."""
    f = np.asarray(R.f(t, y), float)
    yp = y + h * f
    return float(np.max(np.abs(yp - y - h * np.asarray(R.f(t + h, yp), float))))


def predictor_threshold(t: float, y: np.ndarray, ntol: float) -> float:
    """The ``h`` below which the predictor already satisfies the tolerance.

    ``h^2 ||J_f f||_inf <= tol``, so ``h <= sqrt(tol / ||J_f f||_inf)``.
    """
    jn = jf_norm(t, y)
    return float(np.sqrt(ntol / jn)) if jn > 0.0 else float("inf")


def step_doubling_err(t0: float, y0: np.ndarray, h: float, ntol: float,
                      maxiter: int = 50):
    """One trial: ``E = ||y_f - y_c||`` plus the Newton bookkeeping behind it.

    Returns ``(E, iters_max, zero_updates, n_solves, iters_each)``.  No
    Richardson factor is applied: for a first-order method ``y_f - y_c`` *is* the
    leading local error of ``y_f``, which is the tutorial's derivation and is
    confirmed by the fitted exponent of 2.0.
    """
    base = ImplicitEuler(newton_tol=ntol, maxiter=maxiter)
    st = StepStats()
    f = np.asarray(base.rhs(t0, np.asarray(y0, float)), float)

    counts = []
    for _ in range(3):
        before = st.newton_iters
        if not counts:
            yc, _ = base._step(t0, y0, f, h, st)
        elif len(counts) == 1:
            ym, fm = base._step(t0, y0, f, 0.5 * h, st)
        else:
            yf, _ = base._step(t0 + 0.5 * h, ym, fm, 0.5 * h, st)
        counts.append(int(st.newton_iters - before))

    e = float(np.linalg.norm(np.asarray(yf, float) - np.asarray(yc, float)))
    return (e, int(st.newton_iters_max_per_step), int(st.newton_zero_update_steps),
            3, (counts[0], counts[1], counts[2]))


def run(t0: float = 1.0, ntols=NTOLS, hs=HS) -> EstimatorStudy:
    """Sweep the tolerance against the step size at one fixed state."""
    t0, y0, _f0 = state_at(t0)
    jn = jf_norm(t0, y0)

    rows: list[StepRow] = []
    for ntol in ntols:
        for h in hs:
            e, imax, zero, ns, each = step_doubling_err(t0, y0, h, ntol)
            rows.append(StepRow(ntol=float(ntol), h=float(h), iters_max=imax,
                                zero_updates=zero, n_solves=ns,
                                err_estimate=e, iters_each=each))

    slope: dict[float, float] = {}
    explicit: dict[float, int] = {}
    for ntol in ntols:
        sub = [r for r in rows if r.ntol == ntol]
        # only steps whose estimate is above the roundoff floor can be fitted
        hs_f = np.array([r.h for r in sub])
        es_f = np.array([r.err_estimate for r in sub])
        good = es_f > 0.0
        slope[float(ntol)] = (float(np.polyfit(np.log(hs_f[good]), np.log(es_f[good]), 1)[0])
                              if good.sum() >= 2 else float("nan"))
        explicit[float(ntol)] = int(sum(r.is_explicit for r in sub))

    # the residual-bound identity, measured rather than assumed
    worst = 0.0
    for h in hs:
        exact = predictor_residual(t0, y0, h)
        approx = h * h * jn
        worst = max(worst, abs(exact - approx) / exact)

    return EstimatorStudy(
        t0=t0, y0=y0, jf_norm=jn, rows=rows,
        slope_by_ntol=slope,
        threshold_by_ntol={float(n): predictor_threshold(t0, y0, n) for n in ntols},
        explicit_steps_by_ntol=explicit,
        bound_check_max_rel_err=float(worst),
    )
