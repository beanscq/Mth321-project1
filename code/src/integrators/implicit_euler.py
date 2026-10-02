"""Implicit (backward) Euler with a Newton solve.

    y_{n+1} = y_n + h f(t_{n+1}, y_{n+1})

The implicit equation is solved as ``G(z) = z - y_n - h f(t_{n+1}, z) = 0`` by
Newton iteration with the analytic Jacobian ``I - h J(t_{n+1}, z)``.  Order 1.

* A-stable and L-stable: ``R(z) = 1/(1-z)`` maps the entire left half plane
  into the unit disk, and ``R(inf) = 0``.  The step size is therefore *not*
  restricted by stiffness -- which is a robustness statement, **not** an
  accuracy statement.  The local truncation error is still ``O(h)``, so
  "implicit Euler did not blow up" must never be presented as accuracy.

* The predictor is an explicit Euler step, which reuses the derivative already
  stored at ``t_n`` and therefore costs no extra evaluation.

* The Newton stopping tolerance is a first-class knob: see ``newton.py`` and the
  tolerance sweep.
"""

from __future__ import annotations

import numpy as np

from . import newton as NEW
from .base import Integrator

NEWTON_TOL_DEFAULT = 1.0e-12
NEWTON_MAXITER_DEFAULT = 50


class ImplicitEuler(Integrator):
    name = "Implicit Euler"
    order = 1

    def __init__(self, rhs=None, jac=None,
                 newton_tol: float = NEWTON_TOL_DEFAULT,
                 maxiter: int = NEWTON_MAXITER_DEFAULT,
                 damping: bool = False,
                 newton_norm: str = "inf",
                 **options):
        super().__init__(rhs=rhs, jac=jac, **options)
        self.newton_tol = float(newton_tol)
        self.maxiter = int(maxiter)
        self.damping = bool(damping)
        self.newton_norm = str(newton_norm)

    def _step(self, t, y, f_cur, h, stats):
        tn = t + h
        y_pred = y + h * f_cur            # explicit-Euler predictor, free of cost

        def residual_fn(z):
            fz = self.rhs(tn, z)
            return z - y - h * fz, fz

        def jac_fn(z):
            return np.eye(z.size) - h * self.jac(tn, z)

        res = NEW.solve(residual_fn, y_pred, self.newton_tol, jac_fn,
                        maxiter=self.maxiter,
                        damping=self.damping, norm=self.newton_norm)

        stats.rhs_calls += res.rhs_calls
        stats.jac_calls += res.jac_calls
        stats.lu_solves += res.lu_solves
        stats.newton_iters += res.iters
        stats.newton_iters_max_per_step = max(stats.newton_iters_max_per_step, res.iters)
        stats.newton_backtracks += res.n_backtracks
        stats.newton_damping_trials += res.n_trials
        if res.iters == 0:
            # the predictor already satisfied the stopping test: this step is
            # literally an explicit-Euler step
            stats.newton_zero_update_steps += 1
        stats.max_newton_residual = max(stats.max_newton_residual, res.residual)
        stats.sum_newton_residual += res.residual
        if res.converged:
            stats.newton_converged_steps += 1
        else:
            stats.newton_failed_steps += 1
            # keep the dense output honest: the stored derivative must belong to
            # the state we actually return
            res.f_used = self.rhs(tn, res.x)
            stats.rhs_calls += 1

        return res.x, res.f_used
