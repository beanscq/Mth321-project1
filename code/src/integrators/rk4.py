"""Classical fourth-order Runge-Kutta.

    k1 = f(t, y)
    k2 = f(t + h/2, y + h/2 k1)
    k3 = f(t + h/2, y + h/2 k2)
    k4 = f(t + h,   y + h   k3)
    y_{n+1} = y_n + h/6 (k1 + 2 k2 + 2 k3 + k4)

Four evaluations per step, order 4.

Stiffness is *not* cured by a higher order.  The real-axis stability interval of
RK4 is ``[-2.7853, 0]`` -- only 1.39x wider than explicit Euler's ``[-2, 0]`` --
so with ``|lam|_max = 3392.79`` RK4 still needs ``h < 8.209e-4``.  RK4 is the
method used for the adaptive step-size controller, precisely because its order
makes step-doubling error estimates meaningful.
"""

from __future__ import annotations

import numpy as np

from .base import Integrator


class RK4(Integrator):
    name = "RK4"
    order = 4

    def _step(self, t, y, f_cur, h, stats):
        k1 = f_cur                                   # = f(t, y), stored at the node
        k2 = self.rhs(t + 0.5 * h, y + (0.5 * h) * k1)
        k3 = self.rhs(t + 0.5 * h, y + (0.5 * h) * k2)
        k4 = self.rhs(t + h, y + h * k3)
        stats.rhs_calls += 3
        y_new = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        # node derivative for dense output / next step (the 4th evaluation)
        f_new = self.rhs(t + h, y_new)
        stats.rhs_calls += 1
        return y_new, f_new
