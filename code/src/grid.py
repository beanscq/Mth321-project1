"""The common comparison grid.

Protocol requirement: include ``t = 0`` and at least 200 logarithmically spaced
output times from ``1e-8`` to ``40``.  Solvers may take their own internal
steps, but every method and the reference are interpolated onto this *same*
grid before any comparison is made.
"""

from __future__ import annotations

import numpy as np

T_MIN_DEFAULT = 1.0e-8
T_MAX_DEFAULT = 40.0
N_LOG_DEFAULT = 200


def output_times(n_log: int = N_LOG_DEFAULT,
                 t_min: float = T_MIN_DEFAULT,
                 t_max: float = T_MAX_DEFAULT,
                 include_t0: bool = True) -> np.ndarray:
    """Return the comparison grid (strictly increasing)."""
    if n_log < 2:
        raise ValueError("n_log must be >= 2")
    if not 0.0 < t_min < t_max:
        raise ValueError("need 0 < t_min < t_max")
    t = np.logspace(np.log10(t_min), np.log10(t_max), int(n_log))
    if include_t0:
        t = np.concatenate(([0.0], t))
    return t


def positive_times(t: np.ndarray) -> np.ndarray:
    """Mask out ``t = 0`` -- used for every log-scale plot and for the
    stiffness ratio (the initial state is exactly degenerate)."""
    t = np.asarray(t, dtype=float)
    return t[t > 0.0]


def describe(t: np.ndarray) -> dict:
    """Small metadata dict stored next to every artefact."""
    t = np.asarray(t, dtype=float)
    return {
        "n_points": int(t.size),
        "n_positive_points": int(np.count_nonzero(t > 0.0)),
        "t_first_positive": float(t[t > 0.0].min()) if np.any(t > 0.0) else None,
        "t_last": float(t.max()),
        "log10_ratio_between_points": float(
            (np.log10(t[-1]) - np.log10(t[t > 0.0].min())) / max(t[t > 0.0].size - 1, 1)
        ),
    }
