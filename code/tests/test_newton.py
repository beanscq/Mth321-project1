"""Newton solver verification: quadratic convergence, residual control, and the
exact link between the nonlinear residual and the linear invariant.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.integrators import newton as NEW
from src.model import robertson as R


def _quadratic_residual(z):
    return np.array([z[0] ** 2 - 2.0, z[1] ** 2 - 3.0])


def _quadratic_jacobian(z):
    return np.diag([2.0 * z[0], 2.0 * z[1]])


def _wrap(G, J):
    return lambda z: (G(z), np.zeros_like(z)), J


def test_newton_converges_to_the_root():
    G, J = _wrap(_quadratic_residual, _quadratic_jacobian)
    res = NEW.solve(G, np.array([1.0, 1.0]), 1.0e-14, J, maxiter=30)
    assert res.converged
    assert np.allclose(res.x, np.sqrt([2.0, 3.0]), rtol=0, atol=1e-12)


def test_newton_convergence_is_quadratic():
    """Between the first and the last usable iterates the number of correct
    digits should roughly double each step."""
    G, J = _wrap(_quadratic_residual, _quadratic_jacobian)
    res = NEW.solve(G, np.array([1.0, 1.0]), 1.0e-15, J, maxiter=30)
    h = [r for r in res.residual_history if r > 0.0 and r != 1.0]
    assert len(h) >= 3
    ratios = [-np.log10(h[k + 1]) / -np.log10(h[k]) for k in range(len(h) - 1)]
    assert max(ratios) > 1.7          # clearly super-linear, approaching 2
    assert len(res.residual_history) <= 8     # ...and only a handful of steps


def test_residual_stopping_rule_is_respected():
    G, J = _wrap(_quadratic_residual, _quadratic_jacobian)
    tol = 1.0e-6
    res = NEW.solve(G, np.array([1.0, 1.0]), tol, J, maxiter=30)
    assert res.converged
    assert res.residual <= tol


def test_loose_tolerance_stops_earlier():
    G, J = _wrap(_quadratic_residual, _quadratic_jacobian)
    loose = NEW.solve(G, np.array([1.0, 1.0]), 1.0e-3, J, maxiter=30)
    tight = NEW.solve(G, np.array([1.0, 1.0]), 1.0e-14, J, maxiter=30)
    assert loose.iters < tight.iters
    assert loose.residual > tight.residual


def test_unreachable_tolerance_is_reported_not_hidden():
    G, J = _wrap(_quadratic_residual, _quadratic_jacobian)
    res = NEW.solve(G, np.array([1.0, 1.0]), 0.0, J, maxiter=4)
    assert not res.converged
    assert res.iters == 4


def test_singular_jacobian_does_not_raise():
    G = lambda z: (np.array([1.0, 1.0]), np.zeros(2))
    J = lambda z: np.zeros((2, 2))
    res = NEW.solve(G, np.array([0.0, 0.0]), 1.0e-10, J, maxiter=3)
    assert not res.converged


def test_residual_sum_equals_the_linear_invariant_defect_of_one_step():
    """``(1,1,1)^T G(z) = (1,1,1)^T (z - y_n)`` because ``(1,1,1)^T f == 0``.

    This identity is what makes a *residual*-based stopping rule the right one:
    it bounds the invariant defect introduced by the step.
    """
    y_n = np.array([0.9999, 1.0e-5, 1.0e-5])
    h = 1.0e-3
    t_next = h

    def residual(z):
        return z - y_n - h * R.f(t_next, z)

    for z in (y_n + 1e-7, np.array([1.0, 0.0, 0.0]), y_n + h * R.f(0.0, y_n)):
        G = residual(z)
        lhs = float(np.sum(G))
        rhs = float(np.sum(z) - np.sum(y_n))
        assert abs(lhs - rhs) <= 1e-13 * max(1.0, abs(rhs))
        # and therefore |invariant defect| <= 3 * ||G||_inf
        assert abs(rhs) <= 3.0 * np.max(np.abs(G)) + 1e-13


def test_one_newton_step_restores_the_linear_invariant_exactly():
    """The Newton map is invariant-preserving for THIS problem.

    ``(1,1,1)^T`` is a left null vector of ``J``, hence
    ``(1,1,1)^T (I - hJ)^{-1} = (1,1,1)^T``, hence

        sum(z + dz) = sum(z) - sum(G(z)) = sum(y_n)

    exactly, for ANY Newton tolerance.  The practical consequence is important:
    for the Robertson system the conservation diagnostic is *insensitive* to the
    Newton stopping tolerance, so it cannot be used to justify a tolerance -- the
    solution error has to be measured instead.
    """
    y_n = np.array([0.9999, 1.0e-5, 1.0e-5])
    h = 1.0e-3
    t_next = h

    def residual(z):
        return z - y_n - h * R.f(t_next, z)

    z = y_n + h * R.f(0.0, y_n)                   # predictor, sum equals sum(y_n)
    # deliberately stop far from the root
    G = residual(z)
    J = np.eye(3) - h * R.jacobian(t_next, z)
    dz = np.linalg.solve(J, -G)
    z_next = z + dz
    assert abs(float(np.sum(z_next) - np.sum(y_n))) < 1e-15
    # yet the residual of z_next is NOT negligible at that point
    r_after = float(np.max(np.abs(residual(z_next))))
    assert r_after > 1e-12


# ---------------------------------------------------------------------------
# Backtracking (damped) Newton, on the Robertson first step
# ---------------------------------------------------------------------------
def _first_step(h: float, damping: bool, armijo_c=None, norm: str = "inf",
                tol: float = 1.0e-12, maxiter: int = 200):
    """One implicit-Euler step from ``y(0) = (1,0,0)`` with a Newton solve.

    The guess is the old state, which is the tutorial's convention for its own
    experiment, so the iteration counts here are directly comparable with its
    Appendix C table.
    """
    y = R.Y0.astype(float)
    tn = h

    def res(z):
        fz = R.f(tn, z)
        return z - y - h * fz, fz

    return NEW.solve(res, y.copy(), tol, lambda z: np.eye(3) - h * R.jacobian(tn, z),
                     maxiter=maxiter, damping=damping, armijo_c=armijo_c, norm=norm)


def test_damping_reaches_the_same_root_with_fewer_directions():
    """Damping changes the *cost*, not the answer.

    At ``h = 40`` the undamped iteration needs 21 directions while the damped
    one needs 5, and the two returned states agree to 1e-11.  That is the
    Robertson counterpart of the tutorial's "full Newton takes 21 updates and
    damping takes 6 ... both reach the same implicit solution".
    """
    full = _first_step(40.0, False)
    damped = _first_step(40.0, True)

    assert full.converged and damped.converged
    assert np.allclose(full.x, damped.x, rtol=0.0, atol=1e-11)
    assert full.iters > damped.iters
    assert damped.iters <= 10
    assert full.iters > 10
    assert damped.n_backtracks > 0
    assert all(0.0 < a <= 1.0 for a in damped.alpha_history)
    assert len(damped.alpha_history) == damped.iters


def test_full_newton_cost_grows_with_h_while_damped_stays_flat():
    """The tutorial's "<= 10 updates" is a *cost* criterion, and it is h-dependent.

    Undamped Newton needs more and more directions as the step grows (3, 5, 12,
    21 for h = 1e-4 .. 40) while the damped count stays bounded.  Both still
    converge and agree: what changes is how much work the solve costs.
    """
    hs = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 40.0]
    full_iters, damped_iters = [], []
    for h in hs:
        f = _first_step(h, False)
        d = _first_step(h, True)
        assert f.converged and d.converged
        assert np.allclose(f.x, d.x, rtol=0.0, atol=1e-11)   # same root, always
        full_iters.append(f.iters)
        damped_iters.append(d.iters)

    assert full_iters == sorted(full_iters)          # monotonically harder
    assert full_iters[-1] >= 20 and full_iters[0] <= 5
    assert max(damped_iters) <= 8                    # bounded throughout
    assert max(damped_iters) < full_iters[-1] / 2

    # the h above which the cost criterion fails, which is the number the
    # tutorial's Appendix C reports for each variant
    def first_failing(seq):
        for h, it in zip(hs, seq):
            if it > 10:
                return h
        return float("inf")
    assert first_failing(damped_iters) == float("inf")
    assert first_failing(full_iters) < 1.0


def test_damped_solve_reports_failure_when_no_decrease_exists():
    """A step that cannot be damped must be reported, not returned.

    The caller needs this signal: the tutorial's accept/reject table turns any
    Newton failure into "keep (t_n, y_n) and reduce h".
    """
    G = lambda z: (np.array([1.0, 1.0]), np.zeros(2))   # residual is constant
    J = lambda z: np.eye(2)
    res = NEW.solve(G, np.array([0.0, 0.0]), 1.0e-10, J, maxiter=5, damping=True)
    assert not res.converged
    assert res.failure == NEW.FAIL_DAMPING
    assert np.allclose(res.x, 0.0)          # the failed update was not taken
    assert res.rejected_trials                # and the rejected trials are visible
    assert len(res.alpha_history) == 0        # nothing was accepted


def test_backtracking_trials_are_counted_as_extra_rhs_calls():
    """Damping buys fewer directions with more residual evaluations.

    Each halving costs one extra evaluation of the residual, so a cost-honest
    table has to charge for it.  "Damping is cheaper" therefore has to be
    measured across *all* the work, not just the iteration count.
    """
    full = _first_step(40.0, False)
    damped = _first_step(40.0, True)

    assert damped.n_trials == len(damped.rejected_trials) + damped.iters
    assert damped.rhs_calls == len(damped.residual_history) + damped.n_trials
    assert full.rhs_calls == len(full.residual_history)     # no damping, no extras
    assert damped.rhs_calls > 0 and damped.n_trials > 0


def test_armijo_constant_is_available_and_stricter():
    """``armijo_c`` replaces ``||G_new|| < ||G_old||`` by a sufficient decrease.

    The implemented form is the exact consequence of
    ``phi = ||G||^2 / 2`` and ``grad phi^T d = -||G||^2``:

        ||G_new||^2 <= (1 - 2 c alpha) ||G||^2
    """
    strict = _first_step(40.0, True)
    armijo = _first_step(40.0, True, armijo_c=0.25)

    assert strict.converged and armijo.converged
    assert np.allclose(strict.x, armijo.x, rtol=0.0, atol=1e-11)
    # the sufficient-decrease test can only ever accept steps that are no larger
    assert armijo.n_backtracks >= strict.n_backtracks
    assert armijo.mean_alpha <= strict.mean_alpha + 1e-12

    with pytest.raises(ValueError):
        _first_step(40.0, True, armijo_c=0.6)          # c must lie in (0, 1/2)


def test_scalar_exercise_reproduces_the_tutorial_numbers():
    """Tutorial 2, individual exercise and Appendix A.

    Implicit Euler for ``y' = -y^2``, ``y_n = 1``, ``h = 0.5``:

        F(z) = z - 1 + z^2/2,   F'(z) = 1 + z,
        z(1) = 0.75,  z(2) = 0.75 - 0.03125/1.75 = 0.732142857...

    The point of the exercise is that the *two* error sources are separate:
    the remaining root-solve error is ~9.2e-5, while the time-discretisation
    error is ~6.5e-2 -- three orders of magnitude apart.
    """
    F = lambda z: (np.array([z[0] - 1.0 + 0.5 * z[0] ** 2]), np.zeros(1))
    J = lambda z: np.array([[1.0 + z[0]]])

    # reproduce the two published iterates by hand
    z = 1.0
    r = z - 1.0 + 0.5 * z * z
    z = z - r / (1.0 + z)                             # update 1
    assert abs(z - 0.75) < 1e-15
    r = z - 1.0 + 0.5 * z * z
    z2 = z - r / (1.0 + z)                            # update 2
    assert abs(z2 - 0.7321428571428571) < 1e-15

    z_star = -1.0 + np.sqrt(3.0)                      # root of the discrete equation
    y_exact = 1.0 / 1.5                               # y(0.5) = 1/(1+t)

    algebraic = abs(z2 - z_star)
    discretisation = abs(z_star - y_exact)
    assert abs(algebraic - 9.20e-5) < 5e-7            # remaining root-solve error
    assert abs(discretisation - 0.0653841) < 5e-7     # time-discretisation error
    assert discretisation / algebraic > 500.0         # and they are NOT the same size

    # the solver reaches the same root, and a tighter tolerance only shrinks the
    # root-solve error: it leaves the discretisation error exactly where it was
    res = NEW.solve(F, np.array([1.0]), 1.0e-16, J, maxiter=60)
    assert res.converged
    assert abs(float(res.x[0]) - z_star) < algebraic
    assert abs(float(res.x[0]) - y_exact) > 0.05
    assert res.residual <= 1.0e-16



