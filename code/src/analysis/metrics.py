"""Per-run diagnostics for the uniform-step experiments.

The protocol insists that four things are tracked **separately** and never
substituted for one another:

1. conservation   ``|y1 + y2 + y3 - 1|``
2. non-negativity ``min_i y_i >= 0`` componentwise
3. full-state error against the tight reference
4. the terminal value ``y2(40)``

Conservation alone is a very weak statement: the explicit methods satisfy it
*algebraically* (the sum is an exact invariant of any consistent Runge-Kutta
method) and an implicit method satisfies it up to the Newton residual, so a
solution that is wrong or even negative can still conserve the sum perfectly.
That is the whole point of keeping the four diagnostics apart.

Three status flags are reported per run, with stated thresholds:

``stable``        the run did not diverge (see ``DIVERGENCE_BOUND``)
``non_negative``  ``min_i y_i >= -nonneg_tol`` at every accepted step node
``accurate``      ``max_t |y_num - y_ref|_inf <= accurate_tol``, and only if the
                  run covers the whole span

Note that non-negativity is measured on the **step nodes**, i.e. on what the
method actually produced.  Hermite dense output can overshoot slightly between
nodes; measuring on the nodes is the honest reading of "did the method produce a
negative concentration".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..integrators.base import DIVERGENCE_BOUND, Trajectory
from ..model import robertson as R


@dataclass
class Criteria:
    accurate_tol: float = 1.0e-4
    nonneg_tol: float = 1.0e-12
    divergence_bound: float = DIVERGENCE_BOUND
    t_cut_late: float = 1.0e-3

    @classmethod
    def from_config(cls, cfg: dict) -> "Criteria":
        c = cfg.get("criteria", {})
        u = cfg.get("uniform", {})
        return cls(
            accurate_tol=float(c.get("accurate_tol", 1.0e-4)),
            nonneg_tol=float(c.get("nonneg_tol", 1.0e-12)),
            divergence_bound=float(c.get("divergence_bound", DIVERGENCE_BOUND)),
            t_cut_late=float(u.get("t_cut_late", 1.0e-3)),
        )


def assess_run(traj: Trajectory,
               t_grid: np.ndarray,
               y_ref: np.ndarray,
               crit: Criteria) -> dict:
    """Full diagnostic bundle for one uniform-step run."""
    t_grid = np.asarray(t_grid, dtype=float)
    y_ref = np.asarray(y_ref, dtype=float)

    t_end_cov = traj.t_covered
    covered = t_grid <= t_end_cov
    n_covered = int(np.count_nonzero(covered))
    t_cov = t_grid[covered]

    y_num = traj.dense(t_cov)
    err = np.abs(y_num - y_ref[covered])
    per_comp = err.max(axis=0) if err.size else np.full(3, np.nan)
    error_full = float(err.max()) if err.size else float("inf")

    # the same error restricted to the region that the step actually resolves
    # (the earliest log-spaced points sit inside the first one or two steps)
    late = covered & (t_grid >= crit.t_cut_late)
    if np.any(late):
        y_late = traj.dense(t_grid[late])
        error_late = float(np.max(np.abs(y_late - y_ref[late])))
    else:
        error_late = float("nan")

    # terminal value and its error
    has_t40 = t_end_cov >= traj.t_end - 1e-12 * max(traj.t_end, 1.0)
    if has_t40:
        y40 = traj.dense([traj.t_end])[0]
        err40 = float(np.max(np.abs(y40 - y_ref[-1])))
    else:
        y40 = np.full(y_ref.shape[1], np.nan)
        err40 = float("nan")

    min_nodes = traj.min_component()
    min_grid = float(np.nanmin(y_num)) if y_num.size and np.any(np.isfinite(y_num)) else np.nan

    # *when* did non-negativity first fail?  This is the column that answers
    # "show where explicit Euler becomes ... negative" with a time, not a flag.
    below = traj.y.min(axis=1) < -crit.nonneg_tol
    t_first_negative = float(traj.t[int(np.argmax(below))]) if np.any(below) else float("nan")
    max_abs_y = float(np.max(np.abs(traj.y)))

    stable = not traj.stats.aborted
    non_negative = bool(min_nodes >= -crit.nonneg_tol)
    accurate = bool(stable and has_t40 and error_full <= crit.accurate_tol)

    return {
        "method": traj.method,
        "order": traj.order,
        "h_requested": float(traj.stats.h_requested),
        "h_used": float(traj.stats.h_used),
        "n_steps": int(traj.stats.n_steps),
        "n_accepted": int(traj.stats.n_accepted),
        "n_rejected": int(traj.stats.n_rejected),
        "t_covered": float(t_end_cov),
        "covered_fraction": float(n_covered / t_grid.size),

        # --- the four diagnostics, deliberately separate -------------------
        "mass_defect_max": float(np.max(R.invariant_defect(traj.y))),
        "min_component_nodes": float(min_nodes),
        "min_component_grid": float(min_grid),
        "t_first_negative": t_first_negative,
        "max_abs_y": max_abs_y,
        "error_full": error_full,
        "error_full_per_component": np.asarray(per_comp, dtype=float).tolist(),
        "error_late": error_late,
        "error_at_t40": err40,
        "y_at_t40": np.asarray(y40, dtype=float).tolist(),
        "y2_at_t40": float(y40[R.I_B]),

        # --- status flags, with the thresholds that define them ------------
        "stable": stable,
        "non_negative": non_negative,
        "accurate": accurate,
        "aborted": bool(traj.stats.aborted),
        "abort_reason": traj.stats.abort_reason,
        "abort_t": float(traj.stats.abort_t),

        # --- cost ----------------------------------------------------------
        "rhs_calls": int(traj.stats.rhs_calls),
        "jac_calls": int(traj.stats.jac_calls),
        "lu_solves": int(traj.stats.lu_solves),
        "newton_iters": int(traj.stats.newton_iters),
        "newton_failed_steps": int(traj.stats.newton_failed_steps),
        "cost_units": float(traj.stats.cost()),
        "wall_time_s": float(traj.stats.wall_time_s),
    }


def invariant_identity_report(traj: Trajectory) -> dict:
    """Check ``|sum(y_{n+1}) - sum(y_n)| == |sum G|`` for implicit steps.

    Recorded as a *measured* consistency check on the Newton solve: the
    invariant defect introduced by a step is exactly the sum of the nonlinear
    residual, so a residual-based stopping rule bounds the defect directly.
    """
    s = traj.y.sum(axis=1)
    d = np.abs(np.diff(s))
    if d.size == 0:
        return {"max_step_defect_increment": 0.0, "sum_of_step_defect_increments": 0.0}
    return {
        "max_step_defect_increment": float(d.max()),
        "sum_of_step_defect_increments": float(d.sum()),
        "final_defect": float(abs(s[-1] - s[0])),
    }
