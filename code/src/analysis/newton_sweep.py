"""Newton stopping-tolerance sweep for implicit Euler.

What has to be demonstrated
---------------------------
Newton does **not** have to be driven to machine precision; the algebraic error
only has to be negligible *relative to the time-discretisation error*.  The
report has to show the transition and pick a tolerance on the right side of it.

The mechanism, and why the bound is ``O(h^2)``
----------------------------------------------
Implicit Euler starts from the explicit predictor ``y_pred = y_n + h f(t_n, y_n)``.
Its residual is

    ``G(y_pred) = h [ f(t_n, y_n) - f(t_{n+1}, y_pred) ] = -h^2 (f_t + J f) + O(h^3)``

so when the stopping tolerance exceeds roughly ``h^2 |f_t + J f|`` the very
first residual test already passes and **zero Newton updates are performed**:
the method silently degenerates into explicit Euler with a lagged right-hand
side.  That is visible in the table as ``newton_iters ≈ 0`` and it shows up as
a large jump in the solution error, not as a large invariant defect.

The invariant defect cannot see this at all: every Newton iterate preserves
``y1 + y2 + y3`` exactly (see ``diagnostics.py`` for the proof), so the defect
stays at roundoff even for the degenerate runs.  The sweep therefore reports
**four** columns and the recommendation is based on the *solution error*, the
only one of them that responds.

Recommendation rule
-------------------
Let ``e_plateau`` be the error at the tightest tolerance (pure discretisation).
The largest tolerance whose error satisfies ``e(tol) <= max_inflation * e_plateau``
is reported as the recommended stopping tolerance, together with the observed
``h^2`` scaling that predicts it.
"""

from __future__ import annotations

import numpy as np

from ..integrators import make
from ..model import robertson as R
from . import metrics as MET


def sweep(cfg: dict,
          t_grid: np.ndarray,
          y_ref: np.ndarray,
          crit: MET.Criteria | None = None,
          h: float | None = None,
          tol_levels=None,
          maxiter: int | None = None) -> list[dict]:
    """Run implicit Euler at a fixed ``h`` for every Newton tolerance."""
    t_grid = np.asarray(t_grid, dtype=float)
    y_ref = np.asarray(y_ref, dtype=float)
    crit = crit or MET.Criteria.from_config(cfg)

    scfg = cfg.get("newton_sweep", {})
    h = float(scfg.get("h", 1.0e-3) if h is None else h)
    tol_levels = list(scfg.get("tol_levels", [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14])
                      if tol_levels is None else tol_levels)
    maxiter = int(scfg.get("maxiter", 50) if maxiter is None else maxiter)
    t_end = float(cfg["uniform"]["t_end"])

    rows = []
    for tol in tol_levels:
        ie = make("implicit_euler", newton_tol=float(tol), maxiter=maxiter)
        traj = ie.integrate(h, t_end=t_end, divergence_bound=crit.divergence_bound)
        r = MET.assess_run(traj, t_grid, y_ref, crit)
        n_steps = max(int(r["n_steps"]), 1)
        rows.append({
            "newton_tol": float(tol),
            "h_used": r["h_used"],
            "n_steps": r["n_steps"],
            # --- the four quantities that must not be confused -------------
            "error_full": r["error_full"],
            "error_late": r["error_late"],
            "error_at_t40": r["error_at_t40"],
            "y2_at_t40": r["y2_at_t40"],
            "mass_defect_max": r["mass_defect_max"],
            "min_component": r["min_component_nodes"],
            "t_first_negative": r["t_first_negative"],
            # --- the algebraic solve itself --------------------------------
            "newton_iters": r["newton_iters"],
            "newton_iters_per_step": r["newton_iters"] / n_steps,
            "newton_iters_per_step_max": float(traj.stats.newton_iters_max_per_step),
            "fraction_steps_without_newton":
                float(traj.stats.newton_zero_update_steps) / n_steps,
            "newton_failed_steps": r["newton_failed_steps"],
            "rhs_calls": r["rhs_calls"],
            "jac_calls": r["jac_calls"],
            "lu_solves": r["lu_solves"],
            "cost_units": r["cost_units"],
            # --- the theoretical envelope ----------------------------------
            "invariant_bound_3tol": 3.0 * float(tol),
            # --- status ----------------------------------------------------
            "stable": r["stable"],
            "accurate": r["accurate"],
            "non_negative": r["non_negative"],
        })
    return rows


def analyse(rows: list[dict], max_inflation: float = 1.01) -> dict:
    """Plateau error, inflation ratios, the recommended tolerance, and the
    measured ``h^2`` scaling of the predictor residual.

    ``e_plateau`` is the error at the **tightest** tolerance, where the algebraic
    error is as small as it can be and the error is pure time-discretisation
    error.  Inflation is measured against that, not against the minimum over the
    sweep: the error is not monotone in the tolerance (algebraic and
    discretisation error can partially cancel, which is why ``tol = 1e-8`` is
    marginally *better* than ``tol = 1e-14`` below), so "the minimum" would make
    the rule report a spurious failure at the tightest setting.
    """
    if not rows:
        return {}
    rs = sorted(rows, key=lambda r: r["newton_tol"])
    e_plateau = float(rs[0]["error_full"])
    h = float(np.median([r["h_used"] for r in rs]))

    for r in rs:
        r["error_inflation"] = (r["error_full"] / e_plateau) if e_plateau > 0.0 else np.nan
        r["exceeds_inflation_budget"] = bool(r["error_inflation"] > max_inflation)
        r["newton_is_free"] = bool(r["newton_iters_per_step"] < 0.5)

    within = [r for r in rs if r["error_inflation"] <= max_inflation]
    recommended = max(within, key=lambda r: r["newton_tol"]) if within else rs[0]
    first_bad = min((r for r in rs if r["error_inflation"] > max_inflation),
                    key=lambda r: r["newton_tol"], default=None)
    # the loose end of the sweep: where the predictor is accepted as-is
    worst = max(rs, key=lambda r: r["newton_tol"])

    return {
        "h": h,
        "error_plateau": e_plateau,
        "error_plateau_tol": float(rs[0]["newton_tol"]),
        "max_inflation": float(max_inflation),
        "recommended_newton_tol": float(recommended["newton_tol"]),
        "recommended_inflation": float(recommended["error_inflation"]),
        "recommended_newton_iters_per_step": float(recommended["newton_iters_per_step"]),
        "first_tolerance_that_breaks_it": (None if first_bad is None
                                           else float(first_bad["newton_tol"])),
        "first_breaking_inflation": (None if first_bad is None
                                     else float(first_bad["error_inflation"])),
        "loose_end": {
            "tol": float(worst["newton_tol"]),
            "error_full": float(worst["error_full"]),
            "error_inflation": float(worst["error_inflation"]),
            "min_component": float(worst["min_component"]),
            "mass_defect_max": float(worst["mass_defect_max"]),
            "newton_iters_per_step": float(worst["newton_iters_per_step"]),
            "fraction_steps_without_newton": float(worst["fraction_steps_without_newton"]),
            "non_negative": bool(worst["non_negative"]),
        },
        # the predictor's residual is O(h^2): the stopping tolerance has to be
        # pushed below that scale before a single Newton update is even needed
        "predictor_residual_scaling_h2": float(h * h),
    }
