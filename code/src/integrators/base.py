"""Common integrator protocol: step statistics, a uniform-step driver, and
cubic-Hermite dense output.

Design notes
------------
* Every method is a subclass of :class:`Integrator` and only implements
  ``_step``.  The base class owns the step loop, divergence detection, cost
  bookkeeping and the storage of derivatives at the step nodes.

* Dense output uses the **cubic Hermite** interpolant built from ``(y, f)`` at
  both ends of the containing step.  The interpolation order has to be at least
  the method order: plain linear interpolation carries ``O(h^2)`` error and
  would cap the observed convergence order of RK4 at 2, silently destroying the
  convergence study.  Cubic Hermite has ``O(h^4)`` error.

* ``f`` at every step node is stored, and every method obtains it *for free*
  from an evaluation it already performs:

      Explicit Euler   the derivative at the new node is literally the next
                       step's first evaluation, so nothing is wasted;
      RK4              the value returned for the node is the next step's
                       ``k1``; the total stays at the textbook 4 evaluations
                       per step;
      Implicit Euler   the last Newton residual evaluation already computes
                       ``f`` at the accepted iterate.

  Consequently dense output adds **zero** right-hand-side evaluations and the
  counters below are the true number of evaluations performed.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from ..model import robertson as R

T_END_DEFAULT = 40.0

# A run is declared DIVERGED once the state leaves this ball.  The exact
# solution lives in the simplex [0,1]^3, so |y| <= 1 always; 1e3 is a very
# generous bound that no accurate solution can approach.
DIVERGENCE_BOUND = 1.0e3

# Cost model, in units of ONE right-hand-side evaluation.
# Anything not listed here is NOT counted (see README, "Cost model").
DEFAULT_COST_WEIGHTS = {"rhs": 1.0, "jac": 1.0, "lu": 1.0}


# ---------------------------------------------------------------------------
@dataclass
class StepStats:
    """Everything we are allowed to say about the cost of one run."""

    method: str = ""
    order: int = 0
    h_requested: float = float("nan")
    h_used: float = float("nan")

    rhs_calls: int = 0
    jac_calls: int = 0
    lu_solves: int = 0

    newton_iters: int = 0
    newton_iters_max_per_step: int = 0
    newton_zero_update_steps: int = 0
    newton_converged_steps: int = 0
    newton_failed_steps: int = 0
    max_newton_residual: float = 0.0
    sum_newton_residual: float = 0.0

    #: damping (backtracking) accounting.  ``newton_backtracks`` counts
    #: halvings of alpha; ``newton_damping_trials`` counts the residual
    #: evaluations those halvings cost.  Both are zero for a full-step solve,
    #: so a damped run's higher ``rhs_calls`` is visible rather than implied.
    newton_backtracks: int = 0
    newton_damping_trials: int = 0

    n_steps: int = 0
    n_accepted: int = 0
    n_rejected: int = 0

    aborted: bool = False
    abort_reason: str = ""
    abort_t: float = float("nan")
    abort_max_abs_y: float = float("nan")

    wall_time_s: float = 0.0

    def cost(self, weights: dict | None = None) -> float:
        """Weighted cost in RHS-equivalent units."""
        w = dict(DEFAULT_COST_WEIGHTS if weights is None else weights)
        return (float(w.get("rhs", 0.0)) * self.rhs_calls
                + float(w.get("jac", 0.0)) * self.jac_calls
                + float(w.get("lu", 0.0)) * self.lu_solves)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["cost_units"] = self.cost()
        return d


# ---------------------------------------------------------------------------
def hermite_dense(tn: np.ndarray, yn: np.ndarray, fn: np.ndarray,
                  tq) -> np.ndarray:
    """Cubic Hermite interpolation of ``(y, f)`` sampled at the nodes ``tn``.

    Query times outside ``[tn[0], tn[-1]]`` return ``nan`` -- a run that
    diverged before ``t = 40`` must not be silently extrapolated.
    """
    tn = np.asarray(tn, dtype=float)
    yn = np.asarray(yn, dtype=float)
    fn = np.asarray(fn, dtype=float)
    tq = np.atleast_1d(np.asarray(tq, dtype=float))
    out = np.full((tq.size, yn.shape[1]), np.nan)

    if tn.size < 2:            # a run that died inside its very first step
        return out

    inside = (tq >= tn[0]) & (tq <= tn[-1])
    if not np.any(inside):
        return out
    q = tq[inside]
    i = np.clip(np.searchsorted(tn, q, side="right") - 1, 0, tn.size - 2)

    t0, t1 = tn[i], tn[i + 1]
    h = (t1 - t0)[:, None]
    th = ((q - t0) / (t1 - t0))[:, None]

    h00 = 1.0 - 3.0 * th ** 2 + 2.0 * th ** 3
    h10 = th - 2.0 * th ** 2 + th ** 3
    h01 = 3.0 * th ** 2 - 2.0 * th ** 3
    h11 = th ** 3 - th ** 2

    out[inside] = (h00 * yn[i] + h10 * h * fn[i]
                   + h01 * yn[i + 1] + h11 * h * fn[i + 1])
    return out


@dataclass
class Trajectory:
    """A computed solution: nodes, derivatives at the nodes, and statistics."""

    t: np.ndarray
    y: np.ndarray
    f: np.ndarray
    stats: StepStats
    method: str = ""
    order: int = 0
    t_end: float = T_END_DEFAULT

    @property
    def t_covered(self) -> float:
        return float(self.t[-1])

    @property
    def covers_full_span(self) -> bool:
        return self.t_covered >= self.t_end - 1e-12 * max(self.t_end, 1.0)

    def dense(self, tq) -> np.ndarray:
        return hermite_dense(self.t, self.y, self.f, tq)

    def endpoint(self) -> np.ndarray:
        return self.y[-1].copy()

    def min_component(self) -> float:
        return float(np.min(self.y))


# ---------------------------------------------------------------------------
class Integrator:
    """Base class.  Subclasses implement :meth:`_step` only."""

    name = "?"
    order = 0

    def __init__(self, rhs=None, jac=None, **options):
        self.rhs = R.f if rhs is None else rhs
        self.jac = R.jacobian if jac is None else jac
        self.options = dict(options)

    # -- to be overridden ---------------------------------------------------
    def _step(self, t, y, f_cur, h, stats) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    # -- driver -------------------------------------------------------------
    def integrate(self,
                  h: float,
                  t_end: float = T_END_DEFAULT,
                  t0: float = 0.0,
                  y0=None,
                  divergence_bound: float = DIVERGENCE_BOUND) -> Trajectory:
        """Uniform-step integration on ``[t0, t_end]``.

        The requested step ``h`` is rounded *down* so that an integer number of
        steps lands exactly on ``t_end``: ``nsteps = ceil(span/h)`` and
        ``h_used = span/nsteps``.  Both numbers are recorded, because the
        convergence study has to plot the step size it actually used.
        """
        if y0 is None:
            y0 = R.Y0
        y = np.array(y0, dtype=float).reshape(-1)

        span = float(t_end - t0)
        if span <= 0.0:
            raise ValueError("t_end must exceed t0")
        if h <= 0.0:
            raise ValueError("h must be positive")
        nsteps = max(1, int(np.ceil(span / h)))
        h_used = span / nsteps

        st = StepStats(method=self.name, order=self.order,
                       h_requested=float(h), h_used=float(h_used))

        clock = time.perf_counter()
        f_cur = self.rhs(t0, y)
        st.rhs_calls += 1

        ts = [float(t0)]
        ys = [y.copy()]
        fs = [np.asarray(f_cur, dtype=float).copy()]

        for n in range(nsteps):
            # node times are formed as t0 + n*h_used rather than accumulated,
            # and the final node is pinned to t_end, so the node array is
            # strictly increasing and can never contain a duplicated time (a
            # duplicate would make the Hermite interpolant divide by zero)
            t = t0 + n * h_used
            t_next = t_end if n == nsteps - 1 else t0 + (n + 1) * h_used
            hh = t_next - t
            y_new, f_new = self._step(t, y, f_cur, hh, st)

            finite = bool(np.all(np.isfinite(y_new)))
            worst = float(np.max(np.abs(y_new))) if finite else np.inf

            if (not finite) or worst > divergence_bound:
                st.aborted = True
                st.abort_reason = ("non-finite state" if not finite
                                   else f"max|y| exceeded {divergence_bound:g}")
                st.abort_t = float(t_next)
                st.abort_max_abs_y = float(worst)
                break

            y = np.asarray(y_new, dtype=float)
            f_cur = np.asarray(f_new, dtype=float)
            st.n_steps += 1
            st.n_accepted += 1
            ts.append(float(t_next))
            ys.append(y.copy())
            fs.append(f_cur.copy())

        st.wall_time_s = time.perf_counter() - clock
        return Trajectory(t=np.asarray(ts, dtype=float),
                          y=np.asarray(ys, dtype=float),
                          f=np.asarray(fs, dtype=float),
                          stats=st, method=self.name, order=self.order,
                          t_end=float(t_end))
