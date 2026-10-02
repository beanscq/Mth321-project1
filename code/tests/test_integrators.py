"""Integrator verification.

The tests are ordered by how much they prove:

1. order of convergence on a scalar problem, where the answer is known exactly;
2. algebraic preservation of the linear invariant;
3. exactness of the cubic-Hermite dense output on cubics;
4. divergence detection, and the contrast between explicit and implicit
   behaviour on the stiff span;
5. "stable" is not "accurate" and "conserving" is not "correct".
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from src.integrators import ExplicitEuler, ImplicitEuler, RK4, make
from src.integrators.base import hermite_dense
from src.model import robertson as R

TIGHT_ATOL = np.array([1.0e-16, 1.0e-22, 1.0e-16])


def _scalar_problem():
    rhs = lambda t, y: -y                       # noqa: E731
    jac = lambda t, y: np.array([[-1.0]])       # noqa: E731
    return rhs, jac


def _tight_reference(t_end: float, n: int = 41):
    t_eval = np.linspace(0.0, t_end, int(n))
    sol = solve_ivp(R.f, (0.0, float(t_end)), R.Y0, method="Radau",
                    t_eval=t_eval, rtol=1.0e-12, atol=TIGHT_ATOL,
                    jac=R.jacobian)
    assert sol.success, sol.message
    return np.asarray(sol.t, float), np.asarray(sol.y, float).T


# ---------------------------------------------------------------------------
# 1. convergence order on a scalar problem
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cls", [ExplicitEuler, RK4, ImplicitEuler])
def test_observed_order_matches_the_method_order(cls):
    rhs, jac = _scalar_problem()
    hs = [1.0e-1, 5.0e-2, 2.5e-2, 1.25e-2]
    used, errs = [], []
    for h in hs:
        tr = cls(rhs=rhs, jac=jac).integrate(h, t_end=1.0, t0=0.0, y0=[1.0])
        used.append(tr.stats.h_used)
        errs.append(abs(tr.y[-1, 0] - np.exp(-1.0)))
    slope = float(np.polyfit(np.log(used), np.log(errs), 1)[0])
    assert slope == pytest.approx(cls.order, abs=0.3), (cls.__name__, slope, errs)


def test_newton_on_a_linear_problem_converges_in_one_iteration():
    rhs, jac = _scalar_problem()
    tr = ImplicitEuler(rhs=rhs, jac=jac).integrate(1.0e-1, t_end=1.0, y0=[1.0])
    assert tr.stats.newton_failed_steps == 0
    # 10 steps, linear problem: one Newton update each
    assert tr.stats.newton_iters == tr.stats.n_steps


# ---------------------------------------------------------------------------
# 2. algebraic conservation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cls", [ExplicitEuler, RK4, ImplicitEuler])
def test_runge_kutta_methods_preserve_the_linear_invariant(cls):
    tr = cls().integrate(1.0e-3, t_end=1.0)
    assert np.max(R.invariant_defect(tr.y)) < 1e-12


def test_rk4_preserves_the_invariant_exactly_not_merely_approximately():
    """A consistent RK method conserves the sum *algebraically*: the defect is
    pure roundoff and does not grow like an O(h^p) error term."""
    tr = RK4().integrate(2.5e-4, t_end=4.0)
    assert not tr.stats.aborted
    d = R.invariant_defect(tr.y)
    assert d.max() < 1e-13
    assert d.max() < 1e-11 * np.max(np.abs(tr.y - tr.y[0]))


# ---------------------------------------------------------------------------
# 3. dense output
# ---------------------------------------------------------------------------
def test_hermite_dense_output_is_exact_for_cubic_polynomials():
    rng = np.random.default_rng(0)
    c = rng.normal(size=(4, 3))

    def p(t):
        t = np.asarray(t, float)[..., None]
        return c[0] + c[1] * t + c[2] * t ** 2 + c[3] * t ** 3

    def dp(t):
        t = np.asarray(t, float)[..., None]
        return c[1] + 2.0 * c[2] * t + 3.0 * c[3] * t ** 2

    tn = np.linspace(0.0, 2.0, 9)
    tq = np.sort(rng.uniform(0.05, 1.95, 40))
    got = hermite_dense(tn, p(tn), dp(tn), tq)
    assert np.allclose(got, p(tq), rtol=0.0, atol=1.0e-12)


def test_dense_output_is_nan_outside_the_covered_span():
    tr = ExplicitEuler().integrate(2.5e-4, t_end=1.0)
    assert tr.covers_full_span
    assert np.isnan(tr.dense([-1.0])).all()
    assert np.isnan(tr.dense([5.0])).all()
    assert np.all(np.isfinite(tr.dense([0.5])))


def test_dense_output_at_the_nodes_reproduces_the_stored_states():
    tr = RK4().integrate(2.5e-4, t_end=1.0)
    assert np.allclose(tr.dense(tr.t), tr.y, rtol=0.0, atol=1.0e-13)


# ---------------------------------------------------------------------------
# 4. divergence detection and the explicit / implicit contrast
# ---------------------------------------------------------------------------
def test_explicit_euler_diverges_on_the_stiff_span():
    tr = ExplicitEuler().integrate(1.0e-2, t_end=40.0)
    assert tr.stats.aborted
    assert tr.t_covered < 40.0
    assert tr.stats.abort_max_abs_y > 1.0e3


def test_implicit_euler_takes_the_same_step_without_diverging():
    tr = ImplicitEuler().integrate(1.0e-2, t_end=40.0)
    assert not tr.stats.aborted
    assert tr.covers_full_span
    assert tr.min_component() >= -1.0e-12


@pytest.mark.parametrize("h", [3.0e-3, 7.0e-4, 1.0e-2])
def test_the_step_size_lands_exactly_on_the_end_time(h):
    tr = ImplicitEuler().integrate(h, t_end=40.0)
    assert tr.t[-1] == pytest.approx(40.0, rel=1.0e-13)
    assert tr.stats.h_used * tr.stats.n_steps == pytest.approx(40.0, rel=1.0e-12)
    assert tr.stats.h_used <= h


# ---------------------------------------------------------------------------
# 5. "stable" is not "accurate"; "conserving" is not "correct"
# ---------------------------------------------------------------------------
def test_implicit_euler_survives_a_large_step_yet_is_not_accurate():
    t_ref, y_ref = _tight_reference(4.0, 81)
    tr = ImplicitEuler().integrate(1.0e-1, t_end=4.0)
    assert not tr.stats.aborted                    # it did not blow up ...
    assert tr.min_component() >= -1.0e-12
    assert np.max(R.invariant_defect(tr.y)) < 1.0e-12   # ... and it "conserves"
    err = float(np.max(np.abs(tr.dense(t_ref) - y_ref)))
    assert err > 1.0e-4                            # ... but it is not accurate


def test_explicit_euler_is_stable_but_much_less_accurate_than_rk4():
    t_ref, y_ref = _tight_reference(1.0, 41)
    h = 2.5e-4
    ee = ExplicitEuler().integrate(h, t_end=1.0)
    rk = RK4().integrate(h, t_end=1.0)
    assert not ee.stats.aborted and not rk.stats.aborted
    e_ee = float(np.max(np.abs(ee.dense(t_ref) - y_ref)))
    e_rk = float(np.max(np.abs(rk.dense(t_ref) - y_ref)))
    assert e_ee > 1.0e3 * e_rk


def test_cost_counters_are_consistent_with_the_expected_per_step_counts():
    ee = ExplicitEuler().integrate(2.5e-4, t_end=1.0)
    assert ee.covers_full_span
    assert ee.stats.rhs_calls == ee.stats.n_steps + 1
    assert ee.stats.jac_calls == 0

    rk = RK4().integrate(2.5e-4, t_end=1.0)
    assert rk.covers_full_span
    assert rk.stats.rhs_calls == 4 * rk.stats.n_steps + 1

    ie = ImplicitEuler().integrate(1.0e-2, t_end=1.0)
    assert ie.stats.jac_calls == ie.stats.lu_solves == ie.stats.newton_iters
    assert ie.stats.rhs_calls > ie.stats.n_steps


def test_make_registry_returns_the_right_types():
    assert isinstance(make("explicit_euler"), ExplicitEuler)
    assert isinstance(make("rk4"), RK4)
    assert isinstance(make("implicit_euler"), ImplicitEuler)
    with pytest.raises(KeyError):
        make("adams_bashforth")
