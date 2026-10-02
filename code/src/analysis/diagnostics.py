"""P3 diagnostics: the four tracks, kept apart, and the terminal value ``y2(40)``.

The protocol is unusually explicit that conservation, componentwise
non-negativity, full-state error and the terminal value have to be **tracked
separately and never substituted for one another**, because conservation is
nearly worthless as a correctness test here.  That is not a stylistic
preference; it is a theorem about this system:

    ``(1,1,1)^T f(y) == 0``  =>  ``(1,1,1)^T J(y) == 0``
                             =>  ``(1,1,1)^T (I - h J)^{-1} = (1,1,1)^T``

so for the implicit-Euler residual ``G(z) = z - y_n - h f(t_{n+1}, z)``::

    sum G(z) = sum(z) - sum(y_n) - h * (1,1,1)^T f(t_{n+1}, z)
             = sum(z) - sum(y_n)

Whence, for one Newton update ``z -> z + dz`` with ``(I - hJ) dz = -G``::

    sum(z + dz) = sum(z) - sum(G) = sum(y_n)

**Every Newton iterate preserves the sum exactly**, whatever the stopping
tolerance -- including the explicit-Euler predictor, and including the
zero-iteration case where the predictor is returned as the answer.  The measured
invariant defect is therefore at roundoff level even for runs whose error is
``1e2`` and whose concentrations have gone negative.  F9(a) plots exactly that,
and it is the sharpest available answer to "conservation alone is weak".

The terminal-value table quotes **only digits confirmed by our own tightened
reference** (see ``reference.py``); the brief's orientation value is compared
against it rather than the other way round.
"""

from __future__ import annotations

import numpy as np

from ..integrators import make
from ..model import robertson as R
from . import metrics as MET

#: never claim more digits than double precision allows
DIGIT_CAP = 15.5


def digits_vs_reference(value: float,
                        reference: float,
                        ref_digits: float | None = None) -> float:
    """Significant digits of ``value`` that agree with ``reference``.

    Capped at ``ref_digits`` when supplied: we may never claim more digits than
    the reference itself has confirmed.
    """
    v = float(value)
    r = float(reference)
    scale = max(abs(v), abs(r), 1e-300)
    d = -np.log10(max(abs(v - r) / scale, 1e-16))
    d = float(min(d, DIGIT_CAP))
    if ref_digits is not None and np.isfinite(ref_digits):
        d = float(min(d, float(ref_digits)))
    return d


def confirm_value(value: float, digits: float) -> float:
    """Round ``value`` to the ``digits`` significant digits we can confirm."""
    d = int(max(0, np.floor(float(digits))))
    if not np.isfinite(value) or value == 0.0:
        return float(value)
    return float(f"{float(value):.{d}g}")


# ---------------------------------------------------------------------------
# four-track table
# ---------------------------------------------------------------------------
FOUR_TRACK_COLUMNS = ["method", "h_used", "track1_mass_defect_max",
                      "track2_min_component", "track2_t_first_negative",
                      "track3_error_full", "track4_y2_at_t40",
                      "track4_error_at_t40", "stable", "non_negative",
                      "accurate"]


def four_track_rows(sweep_rows: list[dict]) -> list[dict]:
    """One row per (method, h) with the four diagnostics in four columns."""
    out = []
    for r in sorted(sweep_rows, key=lambda r: (r["method"], r["h_used"])):
        out.append({
            "method": r["method"],
            "h_used": r["h_used"],
            "track1_mass_defect_max": r["mass_defect_max"],
            "track2_min_component": r["min_component_nodes"],
            "track2_t_first_negative": r["t_first_negative"],
            "track3_error_full": r["error_full"],
            "track4_y2_at_t40": r["y2_at_t40"],
            "track4_error_at_t40": r["error_at_t40"],
            "stable": r["stable"],
            "non_negative": r["non_negative"],
            "accurate": r["accurate"],
        })
    return out


def first_failing_track(sweep_rows: list[dict]) -> list[dict]:
    """For each method, which diagnostic dies first as ``h`` grows.

    The three thresholds are different, so the three boundaries are allowed to
    disagree; this function makes the disagreement explicit instead of averaging
    it away.
    """
    by_method: dict[str, list[dict]] = {}
    for r in sweep_rows:
        by_method.setdefault(r["method"], []).append(r)

    out = []
    for name, rs in by_method.items():
        rs = sorted(rs, key=lambda r: r["h_used"])
        # the largest h at which each track is still "fine"
        ok_mass = [r for r in rs if np.isfinite(r["mass_defect_max"])
                   and r["mass_defect_max"] < 1.0e-10]
        ok_nonneg = [r for r in rs if r["non_negative"]]
        ok_acc = [r for r in rs if r["accurate"]]
        h_nonneg = max((r["h_used"] for r in ok_nonneg), default=np.nan)
        h_acc = max((r["h_used"] for r in ok_acc), default=np.nan)
        if np.isfinite(h_nonneg) and np.isfinite(h_acc):
            if h_acc < h_nonneg:
                first = "full-state error (accuracy) fails first"
            elif h_nonneg < h_acc:
                first = "non-negativity fails first"
            else:
                first = "accuracy and non-negativity fail together"
        else:
            first = "boundary not reached inside the swept range"
        out.append({
            "method": name,
            "h_max_track1_mass_ok": max((r["h_used"] for r in ok_mass), default=np.nan),
            "h_max_track2_non_negative": h_nonneg,
            "h_max_track3_accurate": h_acc,
            "h_max_track4_reaches_t40":
                max((r["h_used"] for r in rs if r["stable"]), default=np.nan),
            "first_track_to_fail": first,
        })
    return out


# ---------------------------------------------------------------------------
# terminal value y2(40) from the implicit method
# ---------------------------------------------------------------------------
ENDPOINT_COLUMNS = ["method", "h_used", "newton_tol", "y2_at_t40",
                    "abs_diff_vs_reference", "digits_vs_reference",
                    "error_at_t40", "error_full", "mass_defect_max",
                    "min_component", "newton_iters", "newton_failed_steps",
                    "cost_units"]


def endpoint_rows(cfg: dict,
                  t_grid: np.ndarray,
                  y_ref: np.ndarray,
                  ref_char: dict,
                  crit: MET.Criteria | None = None) -> tuple[list[dict], dict]:
    """Implicit Euler at several ``(h, Newton tol)`` settings.

    Returns ``(rows, meta)`` where ``meta`` carries the reference value, the
    digits it confirms, and the comparison against the brief's orientation
    value.
    """
    t_grid = np.asarray(t_grid, dtype=float)
    y_ref = np.asarray(y_ref, dtype=float)
    crit = crit or MET.Criteria.from_config(cfg)

    ecfg = cfg.get("endpoint", {})
    h_list = list(ecfg.get("step_sizes", [1.0e-3, 2.5e-4, 6.25e-5]))
    tol_list = list(ecfg.get("newton_tols", [1.0e-8, 1.0e-12]))
    t_end = float(cfg["uniform"]["t_end"])

    y2_ref = float(y_ref[-1, R.I_B])
    # digits the reference itself vouches for, at t = 40, for y2
    ref_digits_y2 = float(np.asarray(ref_char["quotable_digits_at_t40"])[R.I_B])

    rows = []
    for h in h_list:
        for tol in tol_list:
            ie = make("implicit_euler", newton_tol=float(tol),
                      maxiter=int(cfg["methods"]["implicit_euler"]["newton_maxiter"]))
            traj = ie.integrate(float(h), t_end=t_end,
                                divergence_bound=crit.divergence_bound)
            r = MET.assess_run(traj, t_grid, y_ref, crit)
            v = float(r["y2_at_t40"])
            rows.append({
                "method": f"Implicit Euler (tol={tol:g})",
                "h_used": r["h_used"],
                "newton_tol": float(tol),
                "y2_at_t40": v,
                "abs_diff_vs_reference": abs(v - y2_ref),
                "digits_vs_reference": digits_vs_reference(v, y2_ref, ref_digits_y2),
                "error_at_t40": r["error_at_t40"],
                "error_full": r["error_full"],
                "mass_defect_max": r["mass_defect_max"],
                "min_component": r["min_component_nodes"],
                "newton_iters": r["newton_iters"],
                "newton_failed_steps": r["newton_failed_steps"],
                "cost_units": r["cost_units"],
            })

    brief = R.BRIEF_Y_AT_T40
    diff_brief = np.abs(y_ref[-1] - brief)
    quotable = np.asarray(ref_char["quotable_digits_at_t40"], dtype=float)
    digits_brief = np.array([
        digits_vs_reference(float(y_ref[-1, i]), float(brief[i]), float(quotable[i]))
        for i in range(3)
    ])
    meta = {
        "reference_y_at_t40": y_ref[-1].tolist(),
        "reference_y2_at_t40": y2_ref,
        "reference_quotable_digits_at_t40": quotable.tolist(),
        "reference_max_abs_diff": float(ref_char["max_abs_diff"]),
        "reference_confirmed_digits": float(ref_char["confirmed_digits"]),
        "brief_y_at_t40": brief.tolist(),
        "brief_abs_diff_vs_our_reference": diff_brief.tolist(),
        "brief_digits_confirmed_by_our_reference": digits_brief.tolist(),
        "brief_y2_digits_confirmed": float(digits_brief[R.I_B]),
        "brief_y2_confirmable_value": confirm_value(
            float(brief[R.I_B]), float(digits_brief[R.I_B])),
        "our_y2_confirmable_value": confirm_value(y2_ref, ref_digits_y2),
    }
    return rows, meta
