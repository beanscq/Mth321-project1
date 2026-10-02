"""Tight-tolerance reference solution.

Protocol: run Radau (primary) with a tight tolerance, repeat it with both
tolerances reduced by a factor of 100, and keep only the digits that are
*unchanged* by that repeat.  BDF and LSODA are run as independent oracles.

SciPy is used ONLY here, as a high-accuracy oracle.  The methods under study
(Explicit Euler, RK4, Implicit Euler) are implemented from scratch in
``src/integrators``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
from scipy.integrate import solve_ivp

from ..model import robertson as R


@dataclass
class ReferenceRun:
    method: str
    rtol: float
    atol: np.ndarray
    t: np.ndarray
    y: np.ndarray                       # shape (n, 3)
    nfev: int
    njev: int
    nlu: int
    nsteps: int
    wall_time_s: float
    success: bool
    message: str
    extra: dict = field(default_factory=dict)

    def meta(self) -> dict:
        d = asdict(self)
        d["atol"] = np.asarray(self.atol).tolist()
        return d


def solve(method: str,
          rtol: float,
          atol,
          t_eval: np.ndarray,
          y0: np.ndarray | None = None) -> ReferenceRun:
    """Integrate the authoritative Robertson problem with SciPy."""
    import time

    y0 = R.Y0 if y0 is None else np.asarray(y0, dtype=float)
    atol_arr = np.broadcast_to(np.asarray(atol, dtype=float), (3,)).copy()

    t0 = time.perf_counter()
    sol = solve_ivp(
        fun=R.f,
        t_span=R.T_SPAN,
        y0=y0.copy(),
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol_arr,
        jac=R.jacobian if method in ("Radau", "BDF") else None,
        dense_output=False,
    )
    wall = time.perf_counter() - t0

    return ReferenceRun(
        method=method,
        rtol=float(rtol),
        atol=atol_arr,
        t=np.asarray(sol.t, dtype=float),
        y=np.asarray(sol.y, dtype=float).T,
        nfev=int(sol.nfev),
        njev=int(getattr(sol, "njev", 0) or 0),
        nlu=int(getattr(sol, "nlu", 0) or 0),
        nsteps=int(getattr(sol, "nsteps", 0) or 0),
        wall_time_s=float(wall),
        success=bool(sol.success),
        message=str(sol.message),
    )


def zoom_reference(t_lo: float = 1.0e-8,
                   t_hi: float = 3.0e-7,
                   n: int = 600,
                   rtol: float = 1.0e-12,
                   atol=(1.0e-16, 1.0e-22, 1.0e-16)):
    """Dense early-time reference, used ONLY to resolve the eigenvalue collision.

    Returns ``(t, y)``.  This is deliberately **not** part of the comparison
    grid: it exists because the conjugate-pair window near ``t ~ 1.7e-8`` spans
    only ~5% in t, while consecutive comparison-grid points are ~12% apart, so
    the fixed grid steps straight over it.
    """
    t_eval = np.logspace(np.log10(t_lo), np.log10(t_hi), int(n))
    # np.logspace may land a few ulp outside the span
    t_eval = np.clip(t_eval, t_lo, t_hi)
    t_eval[0], t_eval[-1] = float(t_lo), float(t_hi)
    atol_arr = np.broadcast_to(np.asarray(atol, dtype=float), (3,)).copy()
    sol = solve_ivp(
        fun=R.f, t_span=(0.0, float(t_hi)), y0=R.Y0.copy(),
        method="Radau", t_eval=t_eval, rtol=rtol, atol=atol_arr, jac=R.jacobian,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return np.asarray(sol.t, dtype=float), np.asarray(sol.y, dtype=float).T


# ---------------------------------------------------------------------------
# agreement analysis
# ---------------------------------------------------------------------------
def matching_digits(a: np.ndarray, b: np.ndarray, floor_rel: float = 1e-16) -> dict:
    """Agreement between two approximations of the same solution profile.

    Two numbers are reported, and both matter:

    * ``max_abs_diff`` -- the largest absolute difference.  **This is the number
      that limits how accurately the reference can measure anyone else's error.**
    * ``per_component`` digits -- significant digits of the solution *profile*,
      defined as ``digits_i = -log10( max_t |a_i(t)-b_i(t)| / scale_i )`` with
      ``scale_i`` the component's own magnitude over the trajectory.

    The scale deliberately uses the trajectory magnitude, **not** the
    elementwise magnitude.  An elementwise scale is dominated by the moments
    when a component passes through (or sits at) zero -- ``y2(0) = y3(0) = 0``
    exactly -- and would report a meaningless few digits even though two runs
    agree to 2e-12 everywhere.  That failure mode was observed during
    development; see the AI transparency log.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = np.abs(a - b)
    if diff.ndim == 1:
        diff = diff[:, None]
    per_comp_maxdiff = diff.max(axis=0)
    scale = np.maximum(np.abs(a).max(axis=0), np.abs(b).max(axis=0))
    scale = np.maximum(np.asarray(scale, dtype=float), 1e-300)
    rel = np.maximum(per_comp_maxdiff / scale, floor_rel)
    digits = -np.log10(rel)
    return {
        "min_digits": float(digits.min()),
        "per_component": np.asarray(digits, dtype=float).tolist(),
        "max_abs_diff": float(diff.max()),
        "per_component_max_abs_diff": np.asarray(per_comp_maxdiff, dtype=float).tolist(),
    }


def build_reference_set(t_eval: np.ndarray,
                        cfg: dict) -> dict:
    """Run the full reference protocol and return a serialisable bundle."""
    ref_cfg = cfg["reference"]
    tight = ref_cfg["tight"]
    tighter = ref_cfg["tighter"]

    runs: dict[str, ReferenceRun] = {}
    runs["primary_tight"] = solve(ref_cfg["primary_method"], tight["rtol"], tight["atol"], t_eval)
    runs["primary_tighter"] = solve(ref_cfg["primary_method"], tighter["rtol"], tighter["atol"], t_eval)

    for m in ref_cfg.get("cross_check_methods", []):
        runs[f"cross_{m}"] = solve(m, tighter["rtol"], tighter["atol"], t_eval)

    # digits confirmed by the tolerance repeat
    repeat = matching_digits(runs["primary_tight"].y, runs["primary_tighter"].y)

    cross_digits = {}
    for key, run in runs.items():
        if key.startswith("cross_"):
            cross_digits[key] = matching_digits(run.y, runs["primary_tighter"].y)

    y_ref = runs["primary_tighter"].y

    summary = {
        "t_eval": np.asarray(t_eval, dtype=float),
        "y_reference": y_ref,
        "confirmed_digits_vs_tolerance_repeat": repeat["min_digits"],
        "confirmed_digits_per_component": repeat["per_component"],
        "max_abs_diff_vs_tolerance_repeat": repeat["max_abs_diff"],
        "max_abs_diff_per_component_vs_tolerance_repeat": repeat["per_component_max_abs_diff"],
        "cross_check_digits": cross_digits,
        "runs": {k: v.meta() for k, v in runs.items()},
    }
    return {"summary": summary, "runs": runs}


def characterise_reference(summary: dict) -> dict:
    """Extract the numbers the report is allowed to quote."""
    t = summary["t_eval"]
    y = summary["y_reference"]

    def at(t_target):
        i = int(np.argmin(np.abs(t - t_target)))
        return float(t[i]), y[i]

    _, y40 = at(40.0)
    i_peak = int(np.argmax(y[:, R.I_B]))

    per_comp_abs = np.asarray(summary["max_abs_diff_per_component_vs_tolerance_repeat"], dtype=float)
    scale40 = np.maximum(np.abs(y40), 1e-300)
    quotable = -np.log10(np.maximum(per_comp_abs / scale40, 1e-16))

    out = {
        "y_at_t40": y40.tolist(),
        "y2_peak_value": float(y[i_peak, R.I_B]),
        "y2_peak_time": float(t[i_peak]),
        "y2_at_t40": float(y40[R.I_B]),
        "max_invariant_defect": float(np.max(R.invariant_defect(y))),
        "min_component_over_all_times": float(y.min()),
        "brief_orientation_y_at_t40": R.BRIEF_Y_AT_T40.tolist(),
        "abs_diff_vs_brief_at_t40": np.abs(y40 - R.BRIEF_Y_AT_T40).tolist(),
        "confirmed_digits": summary["confirmed_digits_vs_tolerance_repeat"],
        "confirmed_digits_per_component": summary["confirmed_digits_per_component"],
        "max_abs_diff": summary["max_abs_diff_vs_tolerance_repeat"],
        "max_abs_diff_per_component": per_comp_abs.tolist(),
        "quotable_digits_at_t40": np.asarray(quotable, dtype=float).tolist(),
    }
    return out
