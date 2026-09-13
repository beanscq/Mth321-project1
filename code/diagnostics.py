import numpy as np


#质量守恒误差
def mass_error(y):
    """Return |y1 + y2 + y3 - 1| at each output time."""
    y = np.asarray(y, dtype=float)
    return np.abs(np.sum(y, axis=1) - 1.0)


#最小浓度
def min_component(y):
    """Return the smallest concentration at each output time."""
    y = np.asarray(y, dtype=float)
    return np.min(y, axis=1)


def is_nonnegative(y, tol=1e-12):
    """Check whether all concentrations are non-negative up to a small tolerance."""
    y = np.asarray(y, dtype=float)
    return bool(np.all(y >= -tol))


def full_state_error(y_num, y_ref):
    """Return the Euclidean full-state error at each output time."""
    y_num = np.asarray(y_num, dtype=float)
    y_ref = np.asarray(y_ref, dtype=float)

    if y_num.shape != y_ref.shape:
        raise ValueError(f"Shape mismatch: y_num has {y_num.shape}, y_ref has {y_ref.shape}")

    return np.linalg.norm(y_num - y_ref, axis=1)


def final_y2_error(y_num, y_ref):
    """Return the absolute error in y2 at the final output time."""
    y_num = np.asarray(y_num, dtype=float)
    y_ref = np.asarray(y_ref, dtype=float)
    return float(abs(y_num[-1, 1] - y_ref[-1, 1]))