import numpy as np


def robertson_rhs(t, y):
    """Return the right-hand side of the Robertson chemical kinetics problem."""
    y1, y2, y3 = y

    return np.array([
        -0.04 * y1 + 1.0e4 * y2 * y3,
        0.04 * y1 - 1.0e4 * y2 * y3 - 3.0e7 * y2 * y2,
        3.0e7 * y2 * y2,
    ], dtype=float)


def robertson_jacobian(t, y):
    """Return the Jacobian matrix of the Robertson right-hand side."""
    y1, y2, y3 = y

    return np.array([
        [-0.04, 1.0e4 * y3, 1.0e4 * y2],
        [0.04, -1.0e4 * y3 - 6.0e7 * y2, -1.0e4 * y2],
        [0.0, 6.0e7 * y2, 0.0],
    ], dtype=float)