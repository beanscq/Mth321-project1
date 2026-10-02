"""The three methods under study, plus the cost bookkeeping that goes with them.

SciPy is deliberately absent from this package: it is used only in
``analysis/reference.py`` as a high-accuracy oracle.
"""

from __future__ import annotations

from . import newton
from .base import DEFAULT_COST_WEIGHTS, StepStats, Trajectory, Integrator
from .explicit_euler import ExplicitEuler
from .implicit_euler import ImplicitEuler
from .rk4 import RK4

#: canonical keys -> (human label, class, order)
METHODS = {
    "explicit_euler": ("Explicit Euler", ExplicitEuler, 1),
    "rk4": ("RK4", RK4, 4),
    "implicit_euler": ("Implicit Euler", ImplicitEuler, 1),
}

ORDER = {k: v[2] for k, v in METHODS.items()}
LABEL = {k: v[0] for k, v in METHODS.items()}


def make(key: str, **kwargs) -> Integrator:
    """Instantiate a method by its canonical key."""
    if key not in METHODS:
        raise KeyError(f"unknown method {key!r}; known: {sorted(METHODS)}")
    return METHODS[key][1](**kwargs)


def all_keys() -> list[str]:
    return list(METHODS)


__all__ = [
    "METHODS", "ORDER", "LABEL", "make", "all_keys",
    "ExplicitEuler", "RK4", "ImplicitEuler",
    "Integrator", "Trajectory", "StepStats",
    "DEFAULT_COST_WEIGHTS", "newton",
]
