import numpy as np
from scipy.integrate import solve_ivp


def make_output_times(n_log_points=200, t_final=40.0):
    """Return the common output grid required for the Robertson problem."""
    t_log = np.logspace(-8, np.log10(t_final), n_log_points)
    return np.concatenate(([0.0], t_log))


def solve_reference(rhs, y0, t_eval, method="Radau", rtol=1e-12, atol=1e-14):
    """Compute a tight-tolerance reference solution on the given output grid."""
    sol = solve_ivp(
        rhs,
        (float(t_eval[0]), float(t_eval[-1])),
        np.asarray(y0, dtype=float),
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )

    if not sol.success:
        raise RuntimeError(f"Reference solve failed: {sol.message}")

    return sol.t, sol.y.T


def compare_references(rhs, y0, t_eval):
    """Compare two tight-tolerance reference solves as a self-consistency check."""
    _, y_ref = solve_reference(rhs, y0, t_eval, rtol=1e-12, atol=1e-14)
    _, y_tighter = solve_reference(rhs, y0, t_eval, rtol=1e-13, atol=1e-15)

    return np.max(np.linalg.norm(y_ref - y_tighter, axis=1))