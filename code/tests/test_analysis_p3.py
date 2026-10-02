"""Tests for the P3 diagnostics: four tracks, endpoint values, Newton tolerance."""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis import diagnostics as DIAG
from src.analysis import metrics as MET
from src.analysis import newton_sweep as NS
from src.analysis import reference as REF
from src.analysis import step_sweep as SW
from src.integrators import make


# ---------------------------------------------------------------------------
# digit bookkeeping
# ---------------------------------------------------------------------------
def test_digits_are_capped_by_the_reference_precision():
    exact = 1.2345678901234567
    near = exact * (1.0 + 1e-13)
    assert DIAG.digits_vs_reference(near, exact) > 12.0
    # the same agreement may not be claimed once the reference itself is
    # only good to 8 digits
    assert DIAG.digits_vs_reference(near, exact, ref_digits=8.0) == pytest.approx(8.0)


def test_confirm_value_rounds_to_the_confirmed_digits():
    assert DIAG.confirm_value(9.185534764557785e-06, 5.0) == pytest.approx(9.1855e-06)
    assert DIAG.confirm_value(0.7158270687, 7.0) == pytest.approx(0.7158271)


# ---------------------------------------------------------------------------
# the structural result: conservation is blind to the Newton tolerance
# ---------------------------------------------------------------------------
def test_the_invariant_defect_is_independent_of_the_newton_tolerance():
    """Every Newton iterate preserves the sum exactly, for ANY stopping tolerance.

    So a loose tolerance leaves the invariant spotless while destroying the
    solution -- which is precisely why the four diagnostics are kept apart.
    """
    t = np.array([0.0, 1.0, 40.0])
    y_ref = REF.solve("Radau", 1e-12, [1e-16, 1e-22, 1e-16], t).y
    crit = MET.Criteria()

    tight = MET.assess_run(make("implicit_euler", newton_tol=1e-12).integrate(1e-3),
                           t, y_ref, crit)
    loose = MET.assess_run(make("implicit_euler", newton_tol=1e-2).integrate(1e-3),
                           t, y_ref, crit)

    assert loose["mass_defect_max"] < 1e-10        # conservation: no complaint
    assert tight["mass_defect_max"] < 1e-10
    assert loose["error_full"] > 1e4 * tight["error_full"]   # accuracy: ruined
    assert loose["min_component_nodes"] < 0.0                # positivity: ruined


def test_loose_newton_steps_are_literally_explicit_euler_steps():
    """The predictor residual is O(h^2); above that scale Newton is skipped."""
    tr = make("implicit_euler", newton_tol=1e-2, maxiter=50).integrate(1e-3)
    frac = tr.stats.newton_zero_update_steps / tr.stats.n_steps
    assert frac > 0.3                       # most steps never update
    tight = make("implicit_euler", newton_tol=1e-12, maxiter=50).integrate(1e-3)
    assert tight.stats.newton_zero_update_steps == 0


# ---------------------------------------------------------------------------
# the Newton sweep rule
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sweep_rows():
    t = np.array([0.0, 1.0, 40.0])
    y_ref = REF.solve("Radau", 1e-12, [1e-16, 1e-22, 1e-16], t).y
    cfg = {"uniform": {"t_end": 40.0},
           "methods": {"implicit_euler": {"newton_maxiter": 50}},
           "newton_sweep": {"h": 1.0e-3,
                            "tol_levels": [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12],
                            "maxiter": 50}}
    rows = NS.sweep(cfg, t, y_ref, MET.Criteria())
    return rows, NS.analyse(rows, max_inflation=1.01)


def test_recommended_tolerance_is_below_the_predictor_residual_scale(sweep_rows):
    rows, stats = sweep_rows
    assert stats["recommended_newton_tol"] <= stats["predictor_residual_scaling_h2"]
    assert stats["first_tolerance_that_breaks_it"] is not None
    assert stats["recommended_newton_tol"] < stats["first_tolerance_that_breaks_it"]


def test_the_tightest_tolerance_defines_the_plateau(sweep_rows):
    rows, stats = sweep_rows
    tightest = min(rows, key=lambda r: r["newton_tol"])
    assert stats["error_plateau"] == pytest.approx(tightest["error_full"], rel=1e-12)
    assert stats["error_plateau_tol"] == tightest["newton_tol"]


def test_inflation_is_computed_against_the_plateau(sweep_rows):
    rows, stats = sweep_rows
    for r in rows:
        assert r["error_inflation"] == pytest.approx(
            r["error_full"] / stats["error_plateau"], rel=1e-12)
    assert stats["loose_end"]["tol"] == max(r["newton_tol"] for r in rows)
    assert stats["loose_end"]["error_inflation"] > 1.0


def test_the_invariant_bound_column_is_the_theoretical_envelope(sweep_rows):
    rows, _ = sweep_rows
    for r in rows:
        assert r["invariant_bound_3tol"] == pytest.approx(3.0 * r["newton_tol"])
        # ...and it is satisfied, which is exactly why it proves nothing
        assert r["mass_defect_max"] <= r["invariant_bound_3tol"]


# ---------------------------------------------------------------------------
# four-track boundaries
# ---------------------------------------------------------------------------
def test_the_four_tracks_can_disagree():
    """RK4 loses ACCURACY before it loses stability.

    At ``h = 8.309e-4`` -- just inside the frozen-Jacobian bound -- the run is
    stable and stays non-negative, yet its error at ``t = 40`` is ``1.2e-4``,
    two and a half orders above the accuracy target, while every earlier time is
    right to ``4e-7``.  The amplification per step is ``1 + O(eps)`` and there
    are fifty thousand steps.  That is the concrete form of "not blowing up is
    not an accuracy claim".
    """
    t = np.array([0.0, 1.0, 40.0])
    y_ref = REF.solve("Radau", 1e-12, [1e-16, 1e-22, 1e-16], t).y
    cfg = {"uniform": {"t_end": 40.0, "t_cut_late": 1e-3},
           "criteria": {"accurate_tol": 1e-4, "nonneg_tol": 1e-12,
                        "divergence_bound": 1e3}}
    crit = MET.Criteria.from_config(cfg)
    hs = SW.h_grid(1.5e-4, 3.0e-3, 22)
    hs = hs[(hs >= 7.0e-4) & (hs <= 8.5e-4)]
    assert hs.size == 2, hs
    rows = sorted(SW.run_sweep(("rk4",), hs, t, y_ref, cfg, crit),
                  key=lambda r: r["h_used"])
    fine, coarse = rows

    assert fine["stable"] and fine["non_negative"] and fine["accurate"]
    assert coarse["stable"] and coarse["non_negative"]
    assert not coarse["accurate"]
    assert coarse["error_at_t40"] > crit.accurate_tol > fine["error_at_t40"]

    # and the invariant defect is spotless for both of them
    assert fine["mass_defect_max"] < 1e-13
    assert coarse["mass_defect_max"] < 1e-13


def test_first_failing_track_names_the_diagnostic_not_a_number():
    rows = [
        {"method_key": "m", "method": "M", "h_used": 1e-3, "mass_defect_max": 1e-15,
         "non_negative": True, "accurate": True, "stable": True, "aborted": False,
         "abort_t": np.nan},
        {"method_key": "m", "method": "M", "h_used": 2e-3, "mass_defect_max": 1e-15,
         "non_negative": True, "accurate": False, "stable": True, "aborted": False,
         "abort_t": np.nan},
        {"method_key": "m", "method": "M", "h_used": 4e-3, "mass_defect_max": 1e-15,
         "non_negative": False, "accurate": False, "stable": False, "aborted": True,
         "abort_t": 1.0},
    ]
    out = DIAG.first_failing_track(rows)
    assert out[0]["h_max_track2_non_negative"] == pytest.approx(2e-3)
    assert out[0]["h_max_track3_accurate"] == pytest.approx(1e-3)
    assert "accuracy" in out[0]["first_track_to_fail"]


# ---------------------------------------------------------------------------
# endpoint table
# ---------------------------------------------------------------------------
def test_endpoint_table_is_capped_by_the_reference_digits_and_lands_close():
    t = np.array([0.0, 1.0, 40.0])
    y_ref = REF.solve("Radau", 1e-12, [1e-16, 1e-22, 1e-16], t).y
    ref_char = {"quotable_digits_at_t40": [11.5, 10.2, 11.1],
                "max_abs_diff": 2.1e-12,
                "confirmed_digits": 10.8}
    cfg = {"uniform": {"t_end": 40.0, "t_cut_late": 1e-3},
           "criteria": {"accurate_tol": 1e-4, "nonneg_tol": 1e-12,
                        "divergence_bound": 1e3},
           "methods": {"implicit_euler": {"newton_maxiter": 50}},
           "endpoint": {"step_sizes": [6.25e-5], "newton_tols": [1e-12]}}
    rows, meta = DIAG.endpoint_rows(cfg, t, y_ref, ref_char)

    r = rows[0]
    assert r["digits_vs_reference"] <= ref_char["quotable_digits_at_t40"][1] + 1e-9
    assert r["abs_diff_vs_reference"] < 1e-10
    assert meta["brief_y2_digits_confirmed"] == pytest.approx(5.42, abs=0.1)
    assert meta["brief_y2_confirmable_value"] == pytest.approx(9.1855e-06)
    # the brief's orientation value matches ours to the 5 digits we can confirm
    assert meta["our_y2_confirmable_value"] == pytest.approx(9.1855e-06, rel=2e-5)


def test_endpoint_rows_keep_conservation_separate_from_accuracy():
    t = np.array([0.0, 1.0, 40.0])
    y_ref = REF.solve("Radau", 1e-12, [1e-16, 1e-22, 1e-16], t).y
    ref_char = {"quotable_digits_at_t40": [11.5, 10.2, 11.1],
                "max_abs_diff": 2.1e-12, "confirmed_digits": 10.8}
    cfg = {"uniform": {"t_end": 40.0, "t_cut_late": 1e-3},
           "criteria": {"accurate_tol": 1e-4, "nonneg_tol": 1e-12,
                        "divergence_bound": 1e3},
           "methods": {"implicit_euler": {"newton_maxiter": 50}},
           "endpoint": {"step_sizes": [1.0e-3], "newton_tols": [1e-12]}}
    rows, _ = DIAG.endpoint_rows(cfg, t, y_ref, ref_char)
    assert rows[0]["mass_defect_max"] < 1e-13
    assert rows[0]["min_component"] >= 0.0
