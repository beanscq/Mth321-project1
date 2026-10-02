"""Explicit (forward) Euler.

    y_{n+1} = y_n + h f(t_n, y_n)

One right-hand-side evaluation per step.  Order 1.  Absolutely stable only
inside the disk ``|1 + h*lam| <= 1``, i.e. on the negative real axis
``-2 <= h*lam <= 0``.  With ``|lam|_max(t=40) = 3392.79`` that allows
``h < 5.895e-4`` -- but the bound is a *frozen, linear, local* estimate; the
empirical sweep in ``analysis/step_sweep.py`` tests it.
"""

from __future__ import annotations

import numpy as np

from .base import Integrator


class ExplicitEuler(Integrator):
    name = "Explicit Euler"
    order = 1

    def _step(self, t, y, f_cur, h, stats):
        y_new = y + h * f_cur
        # The derivative at the new node is exactly what the NEXT step needs as
        # its first evaluation, and it is also the right-hand derivative for the
        # Hermite interpolant.  Nothing is evaluated twice.
        f_new = self.rhs(t + h, y_new)
        stats.rhs_calls += 1
        return y_new, f_new
