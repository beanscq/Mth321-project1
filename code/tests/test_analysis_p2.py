"""Verification of the P2 analysis layer: run diagnostics, the empirical sweep,
the frozen-Jacobian estimate and the convergence-order fit.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis import convergence as CONV
from src.analysis import metrics as MET
from src.analysis import reference as REF
from src.analysis import step_sweep as SW
from src.analysis import stability_region as SR
from src.grid import output_times
from src.integrators import ExplicitEuler, ImplicitEuler


@pytest.fixture(scope="module")
def small_ref():
    t = output_times(n_log=60)
    run = REF.solve("Radau", 1.0e-12, [1.0e-16, 1.0e-22, 1.0e-16], t)
    assert run.success
    return t, run.y


# ---------------------------------------------------------------------------
# step grid
# ---------------------------------------------------------------------------
def test_h_grid_is_logarithmically_spaced_and_increasing():
    h = SW.h_grid(1.0e-4, 1.0e-2, 9)
    assert h[0] == pytest.approx(1.0e-4) and h[-1] == pytest.approx(1.0e-2)
    assert np.all(np.diff(h) > 0.0)
    assert np.allclose(np.diff(np.log(h)), np.diff(np.log(h))[0], rtol=1e-12)


# ---------------------------------------------------------------------------
# frozen-Jacobian estimate
# ---------------------------------------------------------------------------
def test_frozen_bound_reproduces_the_brief_numbers(small_ref):
    t, y = small_ref
    fb = SW.frozen_jacobian_bound(t, y)
    assert fb["abs_lambda_max_global"] == pytest.approx(3.39279e3, rel=2e-4)
    # and it is attained at the END of the trajectory, not at t = 0
    assert fb["t_of_abs_lambda_max"] > 1.0
    assert fb["per_method"]["explicit_euler"]["h_bound"] == pytest.approx(5.895e-4, rel=2e-3)
    assert fb["per_method"]["rk4"]["h_bound"] == pytest.approx(8.209e-4, rel=2e-3)
    assert not np.isfinite(fb["per_method"]["implicit_euler"]["h_bound"])


def test_frozen_bound_uses_the_largest_absolute_eigenvalue_not_the_stiffness_ratio(small_ref):
    t, y = small_ref
    curve = SW.abs_lambda_max_curve(t, y)
    # |lam|max reaches ~2190 at the y2 peak and 3392.8 at t = 40
    i_peak = int(np.argmax(y[:, 1]))
    assert 1500.0 < curve[i_peak] < 2600.0
    assert curve[-1] == pytest.approx(3.39279e3, rel=2e-4)


# ---------------------------------------------------------------------------
# assess_run
# ---------------------------------------------------------------------------
def test_assess_run_separates_the_four_diagnostics(small_ref):
    t, y = small_ref
    crit = MET.Criteria()

    ie = MET.assess_run(ImplicitEuler().integrate(1.0e-2, t_end=40.0), t, y, crit)
    assert ie["stable"] and ie["non_negative"] and ie["accurate"]
    assert np.isnan(ie["t_first_negative"])
    assert ie["min_component_nodes"] >= 0.0

    ee = MET.assess_run(ExplicitEuler().integrate(1.0e-2, t_end=40.0), t, y, crit)
    assert not ee["stable"]
    assert not ee["non_negative"]
    assert ee["t_first_negative"] > 0.0
    assert ee["abort_t"] > 0.0
    # THE point of the exercise: the badly wrong, negative explicit solution
    # still satisfies the linear invariant to roundoff
    assert ee["mass_defect_max"] < 1.0e-12
    assert ie["mass_defect_max"] < 1.0e-12


def test_assess_run_reports_full_coverage_for_a_complete_run(small_ref):
    t, y = small_ref
    r = MET.assess_run(ImplicitEuler().integrate(1.0e-2, t_end=40.0), t, y,
                       MET.Criteria())
    assert r["covered_fraction"] == pytest.approx(1.0)
    assert r["error_at_t40"] < 1.0e-2
    assert r["y2_at_t40"] > 0.0


# ---------------------------------------------------------------------------
# empirical sweep
# ---------------------------------------------------------------------------
def test_sweep_finds_the_stability_boundary_where_the_frozen_bound_says(small_ref):
    t, y = small_ref
    fb = SW.frozen_jacobian_bound(t, y)
    hb = fb["per_method"]["explicit_euler"]["h_bound"]
    hs = np.array([0.6 * hb, 1.4 * hb])           # bracket the prediction
    rows = SW.run_sweep(("explicit_euler",), hs, t, y, {})
    lo, hi = sorted(rows, key=lambda r: r["h_used"])
    assert lo["stable"] and lo["non_negative"]
    assert not hi["stable"] and not hi["non_negative"]
    assert lo["h_used"] < hb < hi["h_used"]

    summary = SW.boundary_summary(rows)
    assert summary[0]["h_max_stable"] == pytest.approx(lo["h_used"])
    assert summary[0]["h_min_diverged"] == pytest.approx(hi["h_used"])


# ---------------------------------------------------------------------------
# convergence-order fit
# ---------------------------------------------------------------------------
def test_observed_order_recovers_a_synthetic_slope():
    h = np.array([1.0e-2, 5.0e-3, 2.5e-3, 1.25e-3])
    p, c = 4.0, 1.0e-3
    rows = [{"h_used": float(hh), "error_late": float(c * hh ** p),
             "stable": True} for hh in h]
    fit = CONV.observed_order(rows, error_floor=0.0)
    assert fit["slope"] == pytest.approx(4.0, abs=1e-6)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-8)
    assert fit["n_levels_used"] == 4


def test_observed_order_drops_unstable_and_floor_limited_levels():
    """Only two levels survive the filters, and they are a clean order-1 pair."""
    rows = [
        {"h_used": 1e-2, "error_late": 1e-4, "stable": True},
        {"h_used": 5e-3, "error_late": 5e-5, "stable": True},
        {"h_used": 2.5e-3, "error_late": 1e-6, "stable": False},   # diverged
        {"h_used": 1.25e-3, "error_late": 1e-14, "stable": True},  # error floor
    ]
    fit = CONV.observed_order(rows, error_floor=1e-11)
    assert fit["n_levels_used"] == 2
    assert fit["slope"] == pytest.approx(1.0, abs=1e-6)


def test_observed_order_of_a_single_level_is_nan_not_a_crash():
    fit = CONV.observed_order([{"h_used": 1e-2, "error_late": 1e-4, "stable": True}],
                              error_floor=0.0)
    assert np.isnan(fit["slope"])
    assert fit["n_levels_used"] == 1


def test_real_convergence_orders_are_close_to_nominal(small_ref):
    """A compact end-to-end version of the report's headline experiment."""
    t, y = small_ref
    crit = MET.Criteria()
    hs = [1.0e-3, 5.0e-4, 2.5e-4]
    nominal = {"explicit_euler": 1.0, "rk4": 4.0, "implicit_euler": 1.0}
    from src.integrators import make
    for key, p in nominal.items():
        rows = []
        for h in hs:
            tr = make(key).integrate(h, t_end=40.0)
            row = MET.assess_run(tr, t, y, crit)
            row["method_key"] = key
            rows.append(row)
        fit = CONV.observed_order(rows, error_key="error_late", error_floor=1e-11)
        assert fit["n_levels_used"] >= 2, (key, fit)
        assert fit["slope"] == pytest.approx(p, abs=0.25), (key, fit)
