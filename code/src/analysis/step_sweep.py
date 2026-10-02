"""Frozen-Jacobian step-size bound vs an empirical step sweep.

The theory
----------
Along a frozen spectrum the explicit-Euler recursion ``y_{n+1} = (1 + h lam) y_n``
is absolutely stable iff ``|1 + h lam| <= 1``; for real ``lam < 0`` that means
``h <= -2/lam``.  Taking the most restrictive eigenvalue along the trajectory
gives the *local frozen-Jacobian estimate*

    h < 2 / max_t |lam|_max(t)  =  2 / 3392.79  =  5.895e-4        (Euler)
    h < 2.7853 / 3392.79        =  8.209e-4                       (RK4)

for the real-axis limits ``-2`` and ``-2.7853`` respectively.

Repeating the estimate over the trajectory is essential: ``|lam|_max(t)`` is
NOT maximal at ``t = 0``.  Near ``t = 0`` the fast eigenvalue is
``-(k3 y3 + 2 k2 y2)`` with ``y2, y3 -> 0``, so it is small; the bound is
tightest at ``t = 40`` where ``k3 y3 = 2841.6`` and ``2 k2 y2 = 551.1``.

What the estimate is NOT
------------------------
It is not a nonlinear stability theorem.  ``J`` varies along the trajectory, and
``J`` is strongly non-normal (the linear invariant makes the third eigenvalue
exactly zero), so eigenvalue real parts do not control transient growth.  The
only honest way to use the number is to *test* it: run the same method at a
range of step sizes and locate, separately, where it becomes unstable, where it
loses non-negativity, and where it loses accuracy, then report whether those
boundaries line up with 5.895e-4.
"""

from __future__ import annotations

import numpy as np

from ..integrators import make
from ..model import robertson as R
from . import metrics as MET
from . import stability_region as SR


def h_grid(h_min: float, h_max: float, n: int) -> np.ndarray:
    return np.logspace(np.log10(float(h_min)), np.log10(float(h_max)), int(n))


def abs_lambda_max_curve(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``|Re lam|_max(t)`` over the two non-zero (reduced) eigenvalues."""
    lam = R.reduced_eigenvalues_analytic(np.asarray(y, dtype=float))
    return np.abs(lam.real).max(axis=1)


def frozen_jacobian_bound(t: np.ndarray,
                          y: np.ndarray,
                          methods=("explicit_euler", "rk4", "implicit_euler")) -> dict:
    """The frozen-Jacobian step-size estimate, per method."""
    t = np.asarray(t, dtype=float)
    curve = abs_lambda_max_curve(t, y)
    i = int(np.argmax(curve))
    lam_max = float(curve[i])

    per_method = {}
    for key in methods:
        name = {"explicit_euler": "Explicit Euler",
                "rk4": "RK4",
                "implicit_euler": "Implicit Euler"}[key]
        limit = SR.max_stable_real_step(name)        # |a| from [a, 0]
        per_method[key] = {
            "method": name,
            "real_axis_limit": float(limit) if np.isfinite(limit) else float("inf"),
            "h_bound": float(limit / lam_max) if np.isfinite(limit) else float("inf"),
            "note": ("not restricted by the spectrum (A-/L-stable)"
                     if not np.isfinite(limit) else ""),
        }

    return {
        "abs_lambda_max_global": lam_max,
        "t_of_abs_lambda_max": float(t[i]),
        "where": "max over the whole trajectory, not at t = 0",
        "per_method": per_method,
    }


def run_sweep(method_keys, hs, t_grid, y_ref, cfg, crit=None,
              y0=None, t_end: float = 40.0) -> list[dict]:
    """Integrate every (method, h) pair and collect the diagnostics."""
    crit = crit or MET.Criteria.from_config(cfg)
    rows = []
    for key in method_keys:
        for h in hs:
            integ = make(key)
            traj = integ.integrate(float(h), t_end=t_end, y0=y0,
                                   divergence_bound=crit.divergence_bound)
            row = MET.assess_run(traj, t_grid, y_ref, crit)
            row["method_key"] = key
            row["trajectory"] = traj          # stripped before serialisation
            rows.append(row)
    return rows


def boundary_summary(rows: list[dict]) -> list[dict]:
    """Largest step size at which each of the three properties still holds.

    This is where the three boundaries are kept *separate*: they are computed
    from three different columns and are allowed to disagree.
    """
    by_method: dict[str, list[dict]] = {}
    for r in rows:
        by_method.setdefault(r["method_key"], []).append(r)

    out = []
    for key, rs in by_method.items():
        rs = sorted(rs, key=lambda r: r["h_used"])
        entry = {
            "method_key": key,
            "method": rs[0]["method"],
            "n_points": len(rs),
            # the largest step actually tested: if a "boundary" equals this, no
            # boundary was found *within the swept range* and that is all it means
            "h_max_swept": float(max(r["h_used"] for r in rs)),
        }
        for flag, label in (("stable", "stable"), ("non_negative", "non_negative"),
                            ("accurate", "accurate")):
            ok = [r for r in rs if r[flag]]
            entry[f"h_max_{label}"] = float(max((r["h_used"] for r in ok), default=np.nan))
            entry[f"n_{label}"] = len(ok)
        # where the first divergence happened
        bad = [r for r in rs if r["aborted"]]
        entry["h_min_diverged"] = float(min((r["h_used"] for r in bad), default=np.nan))
        entry["t_first_divergence"] = float(min((r["abort_t"] for r in bad), default=np.nan))
        out.append(entry)
    return out


def strip_trajectories(rows: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if k != "trajectory"} for r in rows]
