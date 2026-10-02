"""Tests for the adaptive step-size controller (RK4 step doubling)."""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis import metrics as MET
from src.analysis import reference as REF
from src.analysis import step_sweep as SW
from src.grid import output_times
from src.integrators import RK4
from src.integrators.adaptive import StepDoublingAdaptive, rhs_cost_per_attempt
from src.integrators.implicit_euler import ImplicitEuler


# ---------------------------------------------------------------------------
# the mechanism
# ---------------------------------------------------------------------------
def test_one_attempt_costs_exactly_three_rk4_steps():
    """Cost bookkeeping must be measured, never assumed."""
    ctl = StepDoublingAdaptive(RK4(), tol=1e-6, h0=1e-3)
    tr = ctl.integrate(t_end=1.0)
    assert tr.stats.rhs_calls == rhs_cost_per_attempt() * tr.n_attempts + 1


def test_the_error_estimate_never_exceeds_the_tolerance_on_an_accepted_step():
    ctl = StepDoublingAdaptive(RK4(), tol=1e-8, h0=1e-6)
    tr = ctl.integrate(t_end=4.0)
    ratio = tr.err_over_tol()[np.isfinite(tr.err_over_tol())]
    assert ratio.size > 100
    assert ratio.max() <= 1.0 + 1e-12


def test_the_controller_actually_changes_the_step_size():
    """A constant h would mean the adaptation is not happening at all."""
    ctl = StepDoublingAdaptive(RK4(), tol=1e-8, h0=1e-6)
    tr = ctl.integrate(t_end=40.0)
    h = tr.accepted_h()
    h = h[np.isfinite(h)]
    assert h.max() / h.min() > 100.0
    assert tr.stats.n_accepted > 20


def test_a_deliberately_unstable_first_step_is_rejected():
    """h0 well above the frozen-Jacobian bound must be rejected, not accepted.

    This is how the controller discovers the stability restriction even though
    the error estimate knows nothing about eigenvalues.
    """
    ctl = StepDoublingAdaptive(RK4(), tol=1e-10, h0=1.0e-1)
    tr = ctl.integrate(t_end=40.0)
    assert tr.stats.n_rejected >= 1
    assert tr.stats.n_accepted > 0
    assert tr.dense([40.0])[0].min() >= -1e-12


def test_controller_reproduces_the_reference_on_a_full_span():
    t = output_times(n_log=60)
    y_ref = REF.solve("Radau", 1e-12, [1e-16, 1e-22, 1e-16], t).y
    ctl = StepDoublingAdaptive(RK4(), tol=1e-8, h0=1e-6)
    tr = ctl.integrate(t_end=40.0)
    crit = MET.Criteria(accurate_tol=1e-4)
    row = MET.assess_run(tr, t, y_ref, crit)
    assert row["stable"] and row["non_negative"] and row["accurate"]
    assert row["error_full"] < 1e-6


# ---------------------------------------------------------------------------
# the finding: an absolute infinity norm cannot see the stiff mode
# ---------------------------------------------------------------------------
def test_the_frozen_bound_explains_the_plateau():
    """The propagated sub-step (h/2) sits on the frozen-Jacobian bound for every
    tolerance loose enough that accuracy would have allowed a larger step."""
    t = output_times(n_log=60)
    y_ref = REF.solve("Radau", 1e-12, [1e-16, 1e-22, 1e-16], t).y
    h_frozen = SW.frozen_jacobian_bound(t, y_ref)["per_method"]["rk4"]["h_bound"]

    medians = []
    for tol in (1e-4, 1e-6, 1e-8):
        ctl = StepDoublingAdaptive(RK4(), tol=tol, h0=1e-6, norm="mixed")
        tr = ctl.integrate(t_end=40.0)
        h = tr.accepted_h()
        medians.append(0.5 * float(np.nanmedian(h)) / h_frozen)
    # a plateau: the three ratios agree to within a few percent
    assert max(medians) / min(medians) < 1.1, medians
    assert all(0.9 < m < 1.3 for m in medians), medians
    # ...and it is NOT the accuracy requirement that sets them: the median
    # err/tol is far below 1 there
    ctl = StepDoublingAdaptive(RK4(), tol=1e-4, h0=1e-6, norm="mixed")
    tr = ctl.integrate(t_end=40.0)
    assert float(np.nanmedian(tr.err_over_tol())) < 0.5


def test_the_absolute_infinity_norm_loses_the_stiff_mode():
    """Regression guard for the failure the scaled norm exists to prevent.

    With the plain absolute norm at tol = 1e-6 the controller accepts steps the
    stiff mode cannot tolerate: y2 goes negative and the error is orders of
    magnitude worse than the same run with the componentwise-scaled norm.
    """
    t = output_times(n_log=60)
    y_ref = REF.solve("Radau", 1e-12, [1e-16, 1e-22, 1e-16], t).y
    crit = MET.Criteria()

    bad = StepDoublingAdaptive(RK4(), tol=1e-6, h0=1e-6, norm="absolute_inf")
    good = StepDoublingAdaptive(RK4(), tol=1e-6, h0=1e-6, norm="mixed")
    rb = MET.assess_run(bad.integrate(t_end=40.0), t, y_ref, crit)
    rg = MET.assess_run(good.integrate(t_end=40.0), t, y_ref, crit)

    assert rb["min_component_nodes"] < 0.0            # y2 went negative
    assert rg["min_component_nodes"] >= 0.0           # the scaled norm held it
    assert rb["error_full"] > 1e3 * rg["error_full"]

    # ...and conservation reported nothing wrong either time
    assert rb["mass_defect_max"] < 1e-12
    assert rg["mass_defect_max"] < 1e-12


def test_mixed_norm_weight_is_relative_where_components_are_small():
    ctl = StepDoublingAdaptive(RK4(), tol=1e-6, norm="mixed")
    # an error of 1e-8 on y2, whose magnitude is 1e-5, is a 1e-3 relative error
    e = np.array([0.0, 1e-8, 0.0])
    y = np.array([0.7, 1e-5, 0.3])
    assert ctl._scaled_error(e, y) == pytest.approx(1e-3, rel=1e-6)
    # the absolute norm would score it as 1e-8 and wave it through
    ctl_abs = StepDoublingAdaptive(RK4(), tol=1e-6, norm="absolute_inf")
    assert ctl_abs._scaled_error(e, y) == pytest.approx(1e-8)


def test_integrate_rejects_a_non_positive_tolerance():
    with pytest.raises(ValueError):
        StepDoublingAdaptive(RK4(), tol=0.0)
    with pytest.raises(ValueError):
        StepDoublingAdaptive(RK4(), tol=1e-6, norm="not-a-norm")


def test_rejections_are_recorded_with_their_launch_time():
    ctl = StepDoublingAdaptive(RK4(), tol=1e-10, h0=1e-1)
    tr = ctl.integrate(t_end=10.0)
    assert tr.reject_t.size == tr.stats.n_rejected
    assert tr.reject_h.size == tr.stats.n_rejected
    if tr.reject_t.size:
        assert np.all(tr.reject_t >= 0.0)


def test_dense_output_is_nan_outside_the_covered_span():
    ctl = StepDoublingAdaptive(RK4(), tol=1e-6, h0=1e-3)
    tr = ctl.integrate(t_end=1.0)
    assert np.isnan(tr.dense([-1.0])).all()
    assert np.isnan(tr.dense([5.0])).all()
    assert np.all(np.isfinite(tr.dense([0.5])))


def test_adaptive_run_has_no_newton_bookkeeping():
    ctl = StepDoublingAdaptive(RK4(), tol=1e-6, h0=1e-3)
    tr = ctl.integrate(t_end=1.0)
    assert tr.stats.jac_calls == 0
    assert tr.stats.newton_iters == 0
    assert tr.stats.lu_solves == 0


# ---------------------------------------------------------------------------
# a failed root solve invalidates the trial, not just the step
# ---------------------------------------------------------------------------
def test_a_trial_with_a_failed_newton_solve_is_rejected():
    """The tutorial's accept/reject table: any Newton failure -> reject.

    The estimate is derived on the assumption that the three implicit solves are
    accurate.  A trial where one of them failed has no valid estimate at all, so
    the fix is to look at the solver, not at ``err`` -- which is what makes this
    a different failure from ``err > tol``.
    """
    base = ImplicitEuler(newton_tol=1.0e-12, maxiter=1)     # budget too small
    ctl = StepDoublingAdaptive(base, tol=1e-6, h0=1.0)
    tr = ctl.integrate(t_end=40.0)

    assert tr.stats.n_rejected > 0
    assert tr.n_rejected_newton > 0
    assert tr.n_rejected_newton + tr.n_rejected_error == tr.stats.n_rejected
    # the two rejection kinds are recorded separately and add up
    assert set(tr.reject_reason.tolist()) <= {"newton", "error"}
    # exactly three implicit solves per attempt, each either converged or not
    attempts = tr.stats.n_accepted + tr.stats.n_rejected
    assert (tr.stats.newton_converged_steps + tr.stats.newton_failed_steps
            == 3 * attempts)
    assert tr.stats.n_accepted > 0


def test_a_run_whose_solves_never_converge_aborts_instead_of_looping():
    """At ``h_min`` with Newton still failing there is nothing left to cut.

    The controller must say so rather than spin: no accepted node is stored, and
    the failure is reported with a reason.
    """
    base = ImplicitEuler(newton_tol=0.0, maxiter=0)        # never "converges"
    ctl = StepDoublingAdaptive(base, tol=1e-6, h0=1e-3, h_min=1e-12)
    tr = ctl.integrate(t_end=40.0)

    assert tr.stats.aborted
    assert "Newton" in tr.stats.abort_reason
    assert tr.stats.n_accepted == 0
    assert tr.t.size == 1                      # only the initial node
    assert tr.n_rejected_newton == tr.stats.n_rejected > 0


def test_rejections_record_their_reason_alongside_the_step():
    base = ImplicitEuler(newton_tol=1.0e-12, maxiter=1)
    tr = StepDoublingAdaptive(base, tol=1e-6, h0=1.0).integrate(t_end=1.0)
    assert tr.reject_t.size == tr.reject_h.size == tr.reject_reason.size
    assert tr.stats.n_rejected == tr.reject_t.size

