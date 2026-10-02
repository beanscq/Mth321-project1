"""Table writers.  Every table is emitted as CSV (machine readable) and as
Markdown (paste-ready for the report)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ..model import robertson as R

SUMMARY_TIMES = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 40.0]


def _fmt(v):
    if isinstance(v, complex):
        return f"{v.real:.6e}{'+' if v.imag >= 0 else '-'}{abs(v.imag):.2e}i"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.6e}"
    return str(v)


def write_table(rows: list[dict], stem: str, outdir: Path, columns=None) -> list[Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return []
    columns = columns or list(rows[0].keys())

    csv_path = outdir / f"{stem}.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in columns})

    md_path = outdir / f"{stem}.md"
    with md_path.open("w") as fh:
        fh.write("| " + " | ".join(columns) + " |\n")
        fh.write("|" + "|".join(["---"] * len(columns)) + "|\n")
        for r in rows:
            fh.write("| " + " | ".join(_fmt(r.get(k)) for k in columns) + " |\n")

    return [csv_path, md_path]


def spectral_summary_rows(t, y) -> list[dict]:
    """Compact spectral fingerprint at a set of reporting times.

    ``t = 0`` is skipped on purpose: that state is exactly degenerate and the
    stiffness ratio is undefined there.
    """
    from ..analysis import eigen as E

    t = np.asarray(t, float)
    y = np.asarray(y, float)
    rows = []
    for tt in SUMMARY_TIMES:
        i = int(np.argmin(np.abs(t - tt)))
        if t[i] <= 0.0:
            continue
        fp = E.fingerprint(t[i], y[i])
        rows.append({
            "t": fp["t"],
            "y2": fp["y"][R.I_B],
            "lambda_fast": fp["lambda_1_fast"],
            "lambda_slow": fp["lambda_2_slow"],
            "abs_lambda_max": fp["abs_lambda_max"],
            "stiffness_ratio_S": fp["stiffness_ratio"],
            "det_reduced": fp["det_reduced"],
            "det_upper_left_block": fp["det_upper_left_block"],
            "complex_pair": fp["is_complex_pair"],
        })
    return rows


def check_value_rows(t, y) -> list[dict]:
    """Our own values next to the brief's orientation values."""
    from ..analysis import eigen as E

    rows = []
    for r in E.check_value_comparison(np.asarray(t, float), np.asarray(y, float)):
        rows.append({
            "t": r["t_target"],
            "t_used": r["t_used"],
            "y2": r["y2"],
            "S_ours": r["S_ours"],
            "S_brief": r["S_brief"],
            "ours_over_brief": r["ratio_ours_over_brief"],
            "abs_lambda_max": r["abs_lambda_max"],
        })
    return rows


# ---------------------------------------------------------------------------
# P2 tables
# ---------------------------------------------------------------------------
SWEEP_COLUMNS = ["method", "h_used", "error_late", "error_full",
                 "mass_defect_max",
                 "min_component_nodes", "t_first_negative", "max_abs_y",
                 "stable", "non_negative", "accurate", "n_steps",
                 "rhs_calls", "cost_units"]


def step_sweep_rows(sweep_rows: list[dict]) -> list[dict]:
    """One row per (method, step size) of the empirical sweep."""
    out = []
    for r in sorted(sweep_rows, key=lambda r: (r["method"], r["h_used"])):
        out.append({k: r.get(k) for k in SWEEP_COLUMNS})
    return out


def sweep_boundary_rows(boundaries: list[dict], frozen: dict) -> list[dict]:
    """Empirical boundaries next to the frozen-Jacobian estimate."""
    rows = []
    for b in boundaries:
        key = b["method_key"]
        fb = frozen["per_method"][key]
        hb = fb["h_bound"]
        emp = b["h_max_stable"]
        swept = b["h_max_swept"]
        rows.append({
            "method": b["method"],
            "h_max_stable_empirical": emp,
            "h_max_non_negative_empirical": b["h_max_non_negative"],
            "h_max_accurate_empirical": b["h_max_accurate"],
            "h_min_diverged_empirical": b["h_min_diverged"],
            "frozen_jacobian_bound": hb,
            "empirical_over_frozen": (emp / hb) if np.isfinite(hb) and np.isfinite(emp) else np.nan,
            "boundary_found_within_swept_range": bool(
                np.isfinite(emp) and emp < swept * (1.0 - 1e-9)),
            "h_max_swept": swept,
            "t_of_first_divergence": b["t_first_divergence"],
        })
    return rows


CONVERGENCE_COLUMNS = ["method", "h_used", "n_steps", "error_full", "error_late",
                       "error_at_t40", "y2_at_t40", "mass_defect_max",
                       "min_component_nodes", "t_first_negative",
                       "stable", "non_negative", "accurate"]


def convergence_rows(rows_by_method: dict[str, list[dict]]) -> list[dict]:
    out = []
    for key, rows in rows_by_method.items():
        for r in sorted(rows, key=lambda r: r["h_used"]):
            out.append({k: r.get(k) for k in CONVERGENCE_COLUMNS})
    return out


def observed_order_rows(orders_late: dict, orders_full: dict,
                        nominal: dict[str, int]) -> list[dict]:
    rows = []
    for key, o in orders_late.items():
        pairs = o["pairs"]
        finest = pairs[-1]["order"] if pairs else np.nan
        of = orders_full.get(key, {})
        rows.append({
            "method": o.get("method_key", key),
            "nominal_order": nominal.get(key, np.nan),
            "error_key": o["error_key"],
            "fitted_slope": o["slope"],
            "slope_stderr": o["slope_stderr"],
            "r2": o["r2"],
            "n_levels_used": o["n_levels_used"],
            "h_window_lo": o["h_window"][0],
            "h_window_hi": o["h_window"][1],
            "finest_pair_order": finest,
            "fitted_slope_full_grid": of.get("slope", np.nan),
        })
    return rows


COST_COLUMNS = ["method", "h_used", "error_late", "n_steps", "rhs_calls",
                "jac_calls", "lu_solves", "newton_iters", "cost_units",
                "wall_time_s"]


def cost_rows(rows_by_method: dict[str, list[dict]]) -> list[dict]:
    out = []
    for key, rows in rows_by_method.items():
        for r in sorted(rows, key=lambda r: r["h_used"]):
            out.append({k: r.get(k) for k in COST_COLUMNS})
    return out


# ---------------------------------------------------------------------------
# P3 tables
# ---------------------------------------------------------------------------
NEWTON_SWEEP_COLUMNS = ["newton_tol", "h_used", "n_steps", "error_full",
                        "error_inflation", "error_at_t40", "mass_defect_max",
                        "invariant_bound_3tol", "min_component",
                        "newton_iters_per_step", "newton_iters_per_step_max",
                        "fraction_steps_without_newton", "newton_failed_steps",
                        "cost_units", "non_negative"]


def newton_sweep_rows(rows: list[dict]) -> list[dict]:
    """One row per Newton stopping tolerance (implicit Euler at fixed ``h``)."""
    return [{k: r.get(k) for k in NEWTON_SWEEP_COLUMNS}
            for r in sorted(rows, key=lambda r: r["newton_tol"])]


ADAPTIVE_COLUMNS = ["norm", "tol", "n_accepted", "n_rejected",
                    "acceptance_ratio", "h_median_controller",
                    "h_substep_median", "h_substep_median_over_frozen",
                    "err_over_tol_median", "err_over_tol_p95", "error_full",
                    "error_at_t40", "error_over_tol_global", "mass_defect_max",
                    "min_component", "rhs_calls", "cost_units", "wall_time_s",
                    "stable", "accurate"]


def adaptive_rows(levels: dict) -> list[dict]:
    """One row per (norm, tolerance) of the adaptive controller."""
    out = []
    for norm in ("mixed", "absolute_inf"):
        for r in sorted(levels.get(norm, []), key=lambda r: r["tol"]):
            out.append({k: r.get(k) for k in ADAPTIVE_COLUMNS})
    return out


def four_track_boundary_rows(rows: list[dict]) -> list[dict]:
    """Where each of the four tracks stops being satisfied."""
    return rows


# ---------------------------------------------------------------------------
# P4 tables
# ---------------------------------------------------------------------------
def cost_at_error(series, target: float):
    """Cost required to reach ``target`` error, by log-log interpolation.

    ``series`` is an iterable of ``(cost, error)`` pairs, already restricted to
    converging runs.  Returns ``None`` when ``target`` lies outside the measured
    range, so that an extrapolation is never silently reported as a measurement.
    """
    pts = sorted((np.log(float(e)), np.log(float(c))) for c, e in series
                 if np.isfinite(c) and np.isfinite(e) and e > 0)
    if len(pts) < 2 or not (pts[0][0] <= np.log(target) <= pts[-1][0]):
        return None
    for i in range(len(pts) - 1):
        if pts[i][0] <= np.log(target) <= pts[i + 1][0]:
            w = (np.log(target) - pts[i][0]) / (pts[i + 1][0] - pts[i][0])
            return float(np.exp(pts[i][1] + w * (pts[i + 1][1] - pts[i][1])))
    return None


MATCHED_ACCURACY_COLUMNS = ["target_error", "method", "status", "cost_units",
                            "error_used", "wall_time_s", "cost_ratio_vs_cheapest",
                            "cheapest_method"]


def matched_accuracy_rows(rows_by_method: dict, targets) -> list[dict]:
    """One row per (target error, method): the cost to reach that error.

    This is the P4 question in table form: at *equal accuracy*, which method is
    cheapest?  Non-converging runs are excluded before interpolating.

    ``status`` records why a number is what it is, so that a missing entry can
    never be mistaken for "not studied":

    ``interpolated``                cost read off the error-cost curve
    ``already_better_than_target``  the method is MORE accurate than the target
                                    at its cheapest stable step, so it never
                                    crosses the target at all; the cost of that
                                    cheapest stable step is reported instead
    ``not_reached_in_sweep``        the target is beyond the swept range for
                                    this method - reported as a gap, not filled
    """
    out = []
    for target in targets:
        per: dict[str, tuple] = {}
        gaps: list[str] = []
        for key, rs in rows_by_method.items():
            acc = sorted([r for r in rs if r.get("accurate")],
                         key=lambda r: float(r["error_late"]))
            c = cost_at_error([(float(r["cost_units"]), float(r["error_late"]))
                               for r in acc], float(target))
            if c is not None:
                near = min(acc, key=lambda r: abs(float(r["error_late"]) - target))
                per[key] = ("interpolated", c, float(target), "interpolated",
                            float(near["wall_time_s"]) * c / float(near["cost_units"]))
                continue
            best_err = float(acc[0]["error_late"]) if acc else np.inf
            if best_err < float(target):
                knee = min(acc, key=lambda r: float(r["cost_units"]))
                per[key] = ("already_better_than_target", float(knee["cost_units"]),
                            float(knee["error_late"]), "already_better_than_target",
                            float(knee["wall_time_s"]))
            else:
                gaps.append(key)

        if not per:
            continue
        cheapest = min(per, key=lambda k: per[k][1])
        for key in sorted(per, key=lambda k: per[k][1]):
            status, c, err, _, wall = per[key]
            out.append({
                "target_error": float(target),
                "method": key,
                "status": status,
                "cost_units": c,
                "error_used": err,
                "cost_ratio_vs_cheapest": c / per[cheapest][1],
                "cheapest_method": cheapest,
                "wall_time_s": wall,
            })
        for key in sorted(gaps):
            out.append({
                "target_error": float(target),
                "method": key,
                "status": "not_reached_in_sweep",
                "cost_units": None,
                "error_used": None,
                "cost_ratio_vs_cheapest": None,
                "cheapest_method": cheapest,
                "wall_time_s": None,
            })
    return out
