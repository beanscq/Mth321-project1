"""Adaptive step-size experiments (RK4 step doubling).

Produces, for each requested tolerance, a run of
:class:`~src.integrators.adaptive.StepDoublingAdaptive` plus the diagnostics the
figure needs.  Two families of runs are made:

``mixed`` norm (the controller used for the results)
    Keeps every component under control.  Signature: the median accepted step
    sits at the frozen-Jacobian bound for *every* tolerance loose enough that
    accuracy would have permitted a larger step, i.e. ``h(t)`` plateaus and the
    rejections cluster on the plateau instead of spreading over the trajectory.

``absolute_inf`` norm (the control experiment)
    The plain infinity norm.  ``y1 = O(1)`` dominates it, the stiff mode in the
    ``y2`` direction is invisible, the controller accepts steps beyond RK4's
    stability limit, ``y2`` goes negative and the returned solution is wrong.
    It is reported, not hidden.

The two are separated in the output so the figure can put them side by side.
"""

from __future__ import annotations

import numpy as np

from ..integrators import make
from ..integrators.adaptive import ATOL_RATIO_DEFAULT, StepDoublingAdaptive
from ..model import robertson as R
from . import metrics as MET
from . import step_sweep as SW


def _controller(cfg: dict, tol: float, norm: str):
    acfg = cfg.get("adaptive", {})
    base = make(str(acfg.get("base", "rk4")))
    return StepDoublingAdaptive(
        base, tol=float(tol), norm=norm,
        h0=float(acfg.get("h0", 1.0e-6)),
        atol_ratio=acfg.get("atol_ratio", ATOL_RATIO_DEFAULT),
        safety=float(acfg.get("safety", 0.9)),
        facmax=float(acfg.get("facmax", 5.0)),
        facmin=float(acfg.get("facmin", 0.2)),
        h_min=float(acfg.get("h_min", 1.0e-12)),
        h_max=float(acfg.get("h_max", 5.0e-2)),
        max_steps=int(acfg.get("max_steps", 3_000_000)),
    )


def run_levels(cfg: dict,
               t_grid: np.ndarray,
               y_ref: np.ndarray,
               crit: MET.Criteria | None = None,
               keep_trajectories: bool = True) -> dict:
    """Run the controller at every configured tolerance, both norms."""
    t_grid = np.asarray(t_grid, dtype=float)
    y_ref = np.asarray(y_ref, dtype=float)
    crit = crit or MET.Criteria.from_config(cfg)
    acfg = cfg.get("adaptive", {})
    tols = list(acfg.get("tol_levels", [1e-4, 1e-6, 1e-8, 1e-10]))
    control_tols = list(acfg.get("control_tol_levels", [1e-4, 1e-6]))
    t_end = float(cfg["uniform"]["t_end"])

    frozen = SW.frozen_jacobian_bound(t_grid, y_ref)
    h_frozen = float(frozen["per_method"]["rk4"]["h_bound"])

    out = {"mixed": [], "absolute_inf": [], "frozen_rk4_bound": h_frozen}
    for norm, levels in (("mixed", tols), ("absolute_inf", control_tols)):
        for tol in levels:
            ctl = _controller(cfg, tol, norm)
            traj = ctl.integrate(t_end=t_end, divergence_bound=crit.divergence_bound)
            r = MET.assess_run(traj, t_grid, y_ref, crit)

            h = traj.accepted_h()
            hf = h[np.isfinite(h)]
            ratio = traj.err_over_tol()
            row = {
                "norm": norm,
                "tol": float(tol),
                "n_accepted": int(traj.stats.n_accepted),
                "n_rejected": int(traj.stats.n_rejected),
                "n_attempts": int(traj.n_attempts),
                "acceptance_ratio": float(traj.acceptance_ratio),
                "rhs_calls": int(traj.stats.rhs_calls),
                "cost_units": float(traj.stats.cost()),
                "wall_time_s": float(traj.stats.wall_time_s),
                "h_median_controller": float(np.median(hf)) if hf.size else np.nan,
                "h_min_controller": float(hf.min()) if hf.size else np.nan,
                "h_max_controller": float(hf.max()) if hf.size else np.nan,
                # the accepted solution is propagated with TWO half-steps, so
                # the step that the stability bound actually constrains is h/2
                "h_substep_median": (0.5 * float(np.median(hf))) if hf.size else np.nan,
                "h_substep_median_over_frozen":
                    (0.5 * float(np.median(hf)) / h_frozen) if hf.size else np.nan,
                "err_over_tol_median": float(np.nanmedian(ratio)),
                "err_over_tol_p95": float(np.nanpercentile(ratio, 95)) if np.any(np.isfinite(ratio)) else np.nan,
                "hits_h_min": bool(traj.hit_h_min),
                "error_full": r["error_full"],
                "error_late": r["error_late"],
                "error_at_t40": r["error_at_t40"],
                "y2_at_t40": r["y2_at_t40"],
                "mass_defect_max": r["mass_defect_max"],
                "min_component": r["min_component_nodes"],
                "t_first_negative": r["t_first_negative"],
                "t_covered": r["t_covered"],
                "stable": r["stable"],
                "non_negative": r["non_negative"],
                "accurate": r["accurate"],
                "aborted": r["aborted"],
                "abort_reason": r["abort_reason"],
                # the local-error-control point: the controller tracks the
                # tolerance even in the regime where the GLOBAL error does not
                "tracks_tolerance": bool(np.nanmedian(ratio) > 1e-3
                                         and np.nanpercentile(
                                             ratio[np.isfinite(ratio)], 95) <= 1.0)
                if np.any(np.isfinite(ratio)) else False,
                "error_over_tol_global": (r["error_full"] / float(tol)
                                          if np.isfinite(r["error_full"]) else np.nan),
                "rejection": rejection_summary(traj),
            }
            row["trajectory"] = traj if keep_trajectories else None
            out[norm].append(row)
    return out


def rejection_summary(traj) -> dict:
    """Where the controller had to back off, and how much.

    Rejections are recorded exactly by the driver: each entry is the ``t`` from
    which the rejected attempt was launched and the step size that was tried.
    """
    rt = np.asarray(getattr(traj, "reject_t", np.empty(0)), dtype=float)
    rh = np.asarray(getattr(traj, "reject_h", np.empty(0)), dtype=float)
    if rt.size == 0:
        return {"n_rejected": 0, "t_lo": np.nan, "t_hi": np.nan,
                "t_median": np.nan, "h_median": np.nan}
    return {
        "n_rejected": int(rt.size),
        "t_lo": float(rt.min()),
        "t_hi": float(rt.max()),
        "t_median": float(np.median(rt)),
        "h_median": float(np.median(rh)),
    }
