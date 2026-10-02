"""Observed convergence order from a sequence of uniformly refined runs.

Two things are reported for every fit and both matter:

* the **per-pair** observed orders ``p = log(e_coarse/e_fine) / log(h_coarse/h_fine)``
  so a reader can see whether the run is in the asymptotic regime at all;
* a **least-squares slope** on ``log e`` vs ``log h`` together with its standard
  error and ``R^2``, restricted to an explicitly stated window.

A level is excluded from the fit when

* the run aborted (it never reached ``t = 40``), or
* the error has fallen to the level of the reference's own uncertainty, where
  the measured error is a floor set by the oracle and not by the method.

The last exclusion is the one students most often skip: a "convergence order"
fitted through points sitting on the reference's error floor is meaningless.
"""

from __future__ import annotations

import numpy as np


def observed_order(rows: list[dict],
                   error_key: str = "error_late",
                   error_floor: float = 1.0e-11,
                   require_stable: bool = True,
                   h_key: str = "h_used",
                   method_key: str | None = None) -> dict:
    """Fit ``log e = p log h + c`` over the admissible levels."""
    pts = []
    for r in rows:
        e = r.get(error_key, np.nan)
        h = r.get(h_key, np.nan)
        if not np.isfinite(e) or not np.isfinite(h) or h <= 0.0:
            continue
        if e <= error_floor:
            continue
        if require_stable and not r.get("stable", True):
            continue
        pts.append((float(h), float(e), r))
    pts.sort(key=lambda p: -p[0])                      # coarsest first

    pairs = []
    for (h1, e1, _), (h2, e2, _) in zip(pts, pts[1:]):
        pairs.append({
            "h_coarse": h1, "h_fine": h2,
            "error_coarse": e1, "error_fine": e2,
            "ratio_h": h1 / h2, "ratio_e": e1 / e2,
            "order": float(np.log(e1 / e2) / np.log(h1 / h2)),
        })

    out = {
        "method_key": method_key,
        "error_key": error_key,
        "n_levels_used": len(pts),
        "ordered_when_filtered": bool(len(pts) >= 2 and
                                      all(p["ratio_h"] > 1.0 for p in pairs)),
        "h_window": [min((p[0] for p in pts), default=np.nan),
                     max((p[0] for p in pts), default=np.nan)],
        "pairs": pairs,
    }
    if len(pts) < 2 or not out["ordered_when_filtered"]:
        out.update({"slope": float("nan"), "slope_stderr": float("nan"),
                    "r2": float("nan"), "n_points": len(pts)})
        return out

    x = np.log(np.array([p[0] for p in pts]))
    y = np.log(np.array([p[1] for p in pts]))
    A = np.vstack([x, np.ones_like(x)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - (slope * x + intercept)
    dof = max(len(x) - 2, 1)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    # standard error must blow up when the h values are not spread out; guard
    # against a degenerate (duplicate-h) design matrix
    sxx = float(np.sum((x - x.mean()) ** 2))
    stderr = float(np.sqrt(ss_res / dof / sxx)) if sxx > 0.0 else float("nan")

    out.update({
        "slope": float(slope),
        "slope_stderr": stderr,
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan"),
        "n_points": len(pts),
        "fit": [[float(xx), float(yy)] for xx, yy in zip(x, y)],
    })
    return out


def summarise(rows_by_method: dict[str, list[dict]],
              error_key: str = "error_late",
              error_floor: float = 1.0e-11) -> dict:
    return {k: observed_order(v, error_key=error_key, error_floor=error_floor)
            for k, v in rows_by_method.items()}
