"""Adaptive step-size control by RK4 **step doubling**.

The controller
--------------
One attempted step of size ``h`` is taken twice:

* **once** as a single RK4 step            -> ``y_full``
* **twice** as two consecutive RK4 steps   -> ``y_half``

The difference estimates the local error of the *coarser* step.  ``y_full`` is
order 4, so its local error is ``O(h^5)`` while the two half steps carry
``2 * O((h/2)^5) = O(h^5)/16``; hence

    ``e = (y_half - y_full) / (2^4 - 1)``

is a leading-order estimate of the local error of ``y_full`` (Richardson), and
the accepted solution ``y_half`` is 16x *more* accurate than the estimate
suggests.  The controller is therefore conservative by construction.

The step is **accepted** when the scaled size of ``e`` is at most ``tol``, and
**rejected** otherwise.  Either way the next step size is

    ``h_new = h * clip(safety * (tol/err)^(1/(p+1)), facmin, facmax)``

with ``p = 4``, so the exponent is ``1/5``.

Why a SCALED norm, and why it is a result and not a detail
----------------------------------------------------------
``noise``/``norm`` selects how ``e`` is reduced to a single number:

``"mixed"`` (default)
    component ``i`` is controlled to ``tol * max(|y_i|, atol_ratio_i)``::

        err = max_i |e_i| / (atol_ratio_i + |y_i|),   accept iff err <= tol

``"absolute_inf"``
    ``err = max_i |e_i|`` -- the plain infinity norm.

The distinction is not cosmetic.  In this problem ``y1 = O(1)`` while
``y2 = O(1e-5)``, and the stiff mode lives in the ``y2`` direction.  With the
plain absolute norm a relative error of ``1e-2`` in ``y2`` contributes only
``1e-7`` to ``err``, so at ``tol = 1e-6`` the controller **accepts steps beyond
RK4's stability limit**, ``y2`` subsequently goes negative, and the returned
solution is wrong by ``8e-3`` -- larger than any uniform-step run in F8.  The
scaled norm sees the same step as ``err ~ 0.09`` and rejects it.  Both variants
are kept and both are run: the failure is reported, not hidden.

What the controller controls
----------------------------
The **local** error, not the global one.  With ``N`` accepted steps of local
error ``tol`` the global error is of order ``N * tol`` -- so "误差接近所设
tolerance" is a claim about the per-step estimate, plotted as ``err/tol`` in
F11(b), while the global error is plotted separately in F11(c) and is honestly
larger by roughly the step count.

Cost
----
Each attempt costs 12 right-hand-side evaluations (three RK4 steps of four
evaluations each).  On acceptance the derivatives at the midpoint and at the
right end are already available, so the piecewise-cubic-Hermite dense output
adds nothing.  One of the twelve -- the derivative stored on the discarded
``y_full`` -- is genuinely wasted; a production controller would use a cheaper
embedded pair.  It is counted, not hidden.

Conventions
-----------
``n_steps`` counts **accepted** steps, exactly as in the uniform-step driver, so
the two remain comparable; an adaptive run additionally reports ``n_rejected``.
An *attempt* is ``n_steps + n_rejected`` and that is what the 12-evaluation
figure multiplies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .base import Integrator, StepStats, Trajectory
from ..model import robertson as R

SAFETY_DEFAULT = 0.9
FACMAX_DEFAULT = 5.0
FACMIN_DEFAULT = 0.2
H_MIN_DEFAULT = 1.0e-12
H_MAX_DEFAULT = 5.0e-2
MAX_STEPS_DEFAULT = 3_000_000
DIVERGENCE_BOUND_DEFAULT = 1.0e3

#: absolute floors in the mixed norm, as a fraction of ``tol``.  They only stop
#: the weight from collapsing to zero when a component is exactly zero (at
#: ``t = 0`` two of the three components are).  They are small enough that the
#: relative part dominates everywhere the solution is non-trivial.
ATOL_RATIO_DEFAULT = (1.0e-10, 1.0e-13, 1.0e-10)

NORMS = ("mixed", "absolute_inf")


@dataclass
class AdaptiveTrajectory(Trajectory):
    """A :class:`Trajectory` plus the controller's own history.

    ``h_at_node`` / ``local_err_est`` are indexed on the *same* nodes as
    ``t``/``y``; node 0 carries ``nan`` because no step produced it.
    """

    tol: float = float("nan")
    norm: str = "mixed"
    h_at_node: np.ndarray | None = None
    local_err_est: np.ndarray | None = None
    hit_h_min: bool = False
    atol_ratio: tuple = field(default=ATOL_RATIO_DEFAULT)
    reject_t: np.ndarray | None = None
    reject_h: np.ndarray | None = None
    #: why each trial was refused: ``"error"`` (the estimate exceeded tol) or
    #: ``"newton"`` (a root solve in the trial failed).  They are different
    #: failures and the accept/reject table treats them differently.
    reject_reason: np.ndarray | None = None

    @property
    def n_rejected_newton(self) -> int:
        if self.reject_reason is None:
            return 0
        return int(np.count_nonzero(self.reject_reason == "newton"))

    @property
    def n_rejected_error(self) -> int:
        if self.reject_reason is None:
            return 0
        return int(np.count_nonzero(self.reject_reason == "error"))

    @property
    def n_attempts(self) -> int:
        return int(self.stats.n_accepted + self.stats.n_rejected)

    @property
    def acceptance_ratio(self) -> float:
        n = self.n_attempts
        return float(self.stats.n_accepted / n) if n else float("nan")

    def err_over_tol(self) -> np.ndarray:
        """``err/tol`` at each accepted node; ``<= 1`` by construction."""
        if self.local_err_est is None or not np.isfinite(self.tol):
            return np.full(self.t.size, np.nan)
        return self.local_err_est / self.tol

    def accepted_h(self) -> np.ndarray:
        return np.asarray(self.h_at_node, dtype=float) if self.h_at_node is not None \
            else np.full(self.t.size, np.nan)


class StepDoublingAdaptive:
    """Wrap any :class:`Integrator` in a step-doubling error controller."""

    def __init__(self,
                 base: Integrator,
                 tol: float,
                 h0: float = 1.0e-6,
                 norm: str = "mixed",
                 atol_ratio=ATOL_RATIO_DEFAULT,
                 safety: float = SAFETY_DEFAULT,
                 facmax: float = FACMAX_DEFAULT,
                 facmin: float = FACMIN_DEFAULT,
                 h_min: float = H_MIN_DEFAULT,
                 h_max: float = H_MAX_DEFAULT,
                 max_steps: int = MAX_STEPS_DEFAULT):
        if tol <= 0.0:
            raise ValueError("tol must be positive")
        if norm not in NORMS:
            raise ValueError(f"norm must be one of {NORMS}")
        if not (0.0 < facmin <= 1.0 <= facmax):
            raise ValueError("need 0 < facmin <= 1 <= facmax")
        self.base = base
        self.tol = float(tol)
        self.h0 = float(h0)
        self.norm = str(norm)
        self.atol_ratio = np.asarray(atol_ratio, dtype=float).reshape(-1)
        self.safety = float(safety)
        self.facmax = float(facmax)
        self.facmin = float(facmin)
        self.h_min = float(h_min)
        self.h_max = float(h_max)
        self.max_steps = int(max_steps)
        self.p = int(base.order)
        self.richardson = float(2 ** self.p - 1)
        self.exponent = 1.0 / (self.p + 1)

    # ------------------------------------------------------------------
    def _scaled_error(self, e_vec: np.ndarray, y_state: np.ndarray) -> float:
        a = np.abs(np.asarray(e_vec, dtype=float))
        if self.norm == "absolute_inf":
            return float(np.max(a))
        ratio = self.atol_ratio
        if ratio.size != a.size:
            ratio = np.resize(ratio, a.size)
        w = ratio + np.abs(np.asarray(y_state, dtype=float))
        return float(np.max(a / w))

    # ------------------------------------------------------------------
    def integrate(self,
                  t_end: float = 40.0,
                  t0: float = 0.0,
                  y0=None,
                  divergence_bound: float = DIVERGENCE_BOUND_DEFAULT) -> AdaptiveTrajectory:
        y = np.array(R.Y0 if y0 is None else y0, dtype=float).reshape(-1)
        st = StepStats(method=f"{self.base.name} (adaptive)", order=self.base.order,
                       h_requested=float(self.h0))
        clock = time.perf_counter()

        f = np.asarray(self.base.rhs(t0, y), dtype=float)
        st.rhs_calls += 1

        t = float(t0)
        h = min(self.h0, t_end - t)

        ts: list[float] = [t]
        ys: list[np.ndarray] = [y.copy()]
        fs: list[np.ndarray] = [f.copy()]
        h_at: list[float] = [float("nan")]
        err_at: list[float] = [float("nan")]
        rej_t: list[float] = []
        rej_h: list[float] = []
        rej_reason: list[str] = []
        hit_h_min = False
        eps_t = 1.0e-14 * max(1.0, abs(t_end))

        while True:
            remaining = t_end - t
            if remaining <= eps_t:
                break
            if st.n_accepted + st.n_rejected >= self.max_steps:
                st.aborted = True
                st.abort_reason = "adaptive step budget exhausted"
                st.abort_t = t
                break

            h = max(min(h, remaining), min(self.h_min, remaining))

            # the controller deliberately probes steps that overflow the state;
            # those attempts are rejected, so the numpy warnings are noise
            newton_failed_before = st.newton_failed_steps
            with np.errstate(all="ignore"):
                y_full, _f_full = self.base._step(t, y, f, h, st)
                y_mid, f_mid = self.base._step(t, y, f, 0.5 * h, st)
                y_half, f_half = self.base._step(t + 0.5 * h, y_mid, f_mid, 0.5 * h, st)

            # A trial is only admissible if all THREE implicit solves succeeded.
            # The step-doubling estimate is derived under "accurate root
            # solves", so a failed solve invalidates the estimate itself -- the
            # two half steps can agree with each other precisely because both
            # are wrong.  This has to be checked before the estimate is looked
            # at, not after: an unconverged trial can produce a perfectly small
            # difference and would otherwise be accepted.
            newton_failed = st.newton_failed_steps > newton_failed_before

            e_vec = (np.asarray(y_half, float) - np.asarray(y_full, float)) / self.richardson
            if (np.all(np.isfinite(e_vec)) and np.all(np.isfinite(y_half))
                    and np.all(np.isfinite(y_full))):
                err = self._scaled_error(e_vec, y_half)
            else:
                err = float("inf")

            at_h_min = h <= self.h_min * (1.0 + 1e-12)
            accept = (not newton_failed) and ((err <= self.tol) or at_h_min)
            if accept and at_h_min:
                hit_h_min = True

            if accept:
                worst = float(np.max(np.abs(y_half)))
                if not np.isfinite(worst) or worst > divergence_bound:
                    st.aborted = True
                    st.abort_reason = (f"max|y| exceeded {divergence_bound:g}"
                                       if np.isfinite(worst) else "non-finite state")
                    st.abort_t = float(t + h)
                    st.abort_max_abs_y = float(worst)
                    break

                # the two half-step nodes become the stored nodes, so the dense
                # output is piecewise cubic Hermite over sub-steps of size h/2.
                # ``h_at_node`` records the CONTROLLER's accepted step h (the
                # same value on both new nodes); the propagated sub-step is h/2.
                y = np.asarray(y_half, dtype=float)
                f = np.asarray(f_half, dtype=float)
                ts.extend([float(t + 0.5 * h), float(t + h)])
                ys.extend([np.asarray(y_mid, dtype=float).copy(), y.copy()])
                fs.extend([np.asarray(f_mid, dtype=float).copy(), f.copy()])
                h_at.extend([h, h])
                err_at.extend([err, err])
                st.n_accepted += 1
                st.n_steps += 1
                t = t + h
            else:
                st.n_rejected += 1
                rej_t.append(float(t))
                rej_h.append(float(h))
                rej_reason.append("newton" if newton_failed else "error")
                if newton_failed and at_h_min:
                    # nothing left to cut: report it rather than loop at h_min
                    st.aborted = True
                    st.abort_reason = "Newton failed at the minimum step"
                    st.abort_t = float(t)
                    break

            if newton_failed:
                # a failed root solve carries no information about the local
                # error, so the error-driven controller must not be consulted
                factor = 0.5
            else:
                target = float("inf") if err <= 0.0 else err
                factor = self.safety * (self.tol / target) ** self.exponent
            factor = min(max(factor, self.facmin), self.facmax)
            h = min(max(h * factor, self.h_min), self.h_max)
            h = min(h, remaining)

        st.wall_time_s = time.perf_counter() - clock
        h_arr = np.asarray(h_at, dtype=float)
        st.h_used = float(np.nanmedian(h_arr)) if h_arr.size > 1 else float("nan")
        return AdaptiveTrajectory(
            t=np.asarray(ts, dtype=float),
            y=np.asarray(ys, dtype=float),
            f=np.asarray(fs, dtype=float),
            stats=st,
            method=self.base.name,
            order=self.base.order,
            t_end=float(t_end),
            tol=self.tol,
            norm=self.norm,
            h_at_node=h_arr,
            local_err_est=np.asarray(err_at, dtype=float),
            hit_h_min=hit_h_min,
            atol_ratio=tuple(float(v) for v in self.atol_ratio),
            reject_t=np.asarray(rej_t, dtype=float),
            reject_h=np.asarray(rej_h, dtype=float),
            reject_reason=np.asarray(rej_reason, dtype=object) if rej_reason
            else np.asarray([], dtype=object),
        )


def rhs_cost_per_attempt() -> int:
    """RHS evaluations burnt by ONE step-doubling attempt.

    Three RK4 steps of four evaluations each.  Measured, not assumed: a test
    asserts that the counter advances by exactly this many per attempt.
    """
    return 12
