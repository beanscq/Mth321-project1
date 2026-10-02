"""Jacobian spectrum, structural zero eigenvalue and stiffness ratio.

The authoritative vector field satisfies ``(1,1,1) . f(y) == 0`` for every y.
Differentiating along a trajectory gives ``(1,1,1)^T J(y) == 0``, so
``(1,1,1)/sqrt(3)`` is a LEFT null vector of J and J always has a zero
eigenvalue.  A stiffness ratio built from all three eigenvalues is therefore
infinite or undefined and carries no information.

We define, as required,

    S(t) = max_{lam_j != 0} |Re lam_j| / min_{lam_j != 0} |Re lam_j|

with an explicit numerical threshold for "zero", and we cross-check the
classification two independent ways:

1. magnitude test  |Re lam| <= zero_rtol * max_j |Re lam_j|
2. alignment test  |cos| between the left eigenvector for that eigenvalue and
   the exact left null vector (1,1,1)/sqrt(3)

The ratio is not reported at ``t = 0``: there ``y2 = y3 = 0`` and the reduced
Jacobian is ``[[-0.04, 0], [0.04, 0]]`` whose spectrum is ``{0, -0.04}`` -- i.e.
of the two "non-zero" eigenvalues one is exactly zero, so the ratio is
undefined (both in the full 3x3 picture, where the zero eigenvalue is a double
root, and in the reduced picture).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..model import robertson as R

# check values published in the project brief.  These are ORIENTATION values
# for debugging only -- every number in the report comes from our own run.
BRIEF_STIFFNESS_RATIO = {1e-4: 3.0e3, 1e-2: 5.4e3, 40.0: 1.58e5}
BRIEF_ABS_LAMBDA_MAX_AT_T40 = 3.39e3


@dataclass
class SpectralAnalysis:
    t: np.ndarray
    lam_full: np.ndarray            # (n, 3) complex, sorted by decreasing |Re|
    lam_reduced: np.ndarray         # (n, 2) complex, analytic
    is_zero: np.ndarray             # (n, 3) bool, magnitude test
    zero_alignment: np.ndarray      # (n,) |cos| left-eigenvector test
    stiffness_ratio: np.ndarray     # (n,) nan at t = 0
    lam_fast: np.ndarray            # (n,) complex
    lam_slow: np.ndarray            # (n,) complex
    discriminant: np.ndarray        # (n,) disc of the reduced characteristic poly
    n_nonzero: np.ndarray           # (n,) int

    @property
    def has_structural_zero(self) -> bool:
        return bool(np.all(self.is_zero[:, -1]))


def _classify_zero(abs_re: np.ndarray, zero_rtol: float) -> np.ndarray:
    scale = float(np.max(abs_re))
    if scale == 0.0:
        return np.ones_like(abs_re, dtype=bool)
    return abs_re <= zero_rtol * scale


def _left_eigenvector_alignment(t: float, y: np.ndarray) -> float:
    """|cos| between the exact left null vector and the computed one."""
    J = R.jacobian(t, y)
    w, V = np.linalg.eig(J.T)                 # left eigenvectors of J
    k = int(np.argmin(np.abs(w)))
    v = np.asarray(V[:, k], dtype=complex)
    nv = np.linalg.norm(v)
    if nv == 0.0:
        return 0.0
    return float(np.abs(np.vdot(R.LEFT_NULL_VECTOR.astype(complex), v)) / nv)


def analyse_trajectory(t: np.ndarray,
                       y: np.ndarray,
                       zero_rtol: float = 1e-8) -> SpectralAnalysis:
    """Full spectral diagnostic along a sampled trajectory."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = t.size

    lam_full = np.zeros((n, 3), dtype=complex)
    is_zero = np.zeros((n, 3), dtype=bool)
    align = np.zeros(n)
    S = np.full(n, np.nan)
    nz = np.zeros(n, dtype=int)
    lam_fast = np.zeros(n, dtype=complex)
    lam_slow = np.zeros(n, dtype=complex)

    for i in range(n):
        J = R.jacobian(t[i], y[i])
        lam = np.linalg.eigvals(J)
        lam = lam[np.argsort(-np.abs(lam.real))]
        lam_full[i] = lam
        is_zero[i] = _classify_zero(np.abs(lam.real), zero_rtol)
        align[i] = _left_eigenvector_alignment(t[i], y[i])

        nz[i] = int(np.count_nonzero(~is_zero[i]))
        if nz[i] >= 2:
            mags = np.abs(lam.real[~is_zero[i]])
            S[i] = float(mags.max() / mags.min())

    lam_red = R.reduced_eigenvalues_analytic(y)
    # sort by decreasing |Re|: [:, 0] is fast, [:, 1] is slow
    order = np.argsort(-np.abs(lam_red.real), axis=-1)
    lam_red_sorted = np.take_along_axis(lam_red, order, axis=-1)
    lam_fast = lam_red_sorted[:, 0]
    lam_slow = lam_red_sorted[:, 1]

    tr_red, det_red = R.reduced_jacobian_invariants(y)
    disc = tr_red ** 2 - 4.0 * det_red

    # t = 0 is the exactly degenerate state: never report a ratio there
    S[t <= 0.0] = np.nan

    return SpectralAnalysis(
        t=t, lam_full=lam_full, lam_reduced=lam_red_sorted, is_zero=is_zero,
        zero_alignment=align, stiffness_ratio=S, lam_fast=lam_fast,
        lam_slow=lam_slow, discriminant=disc, n_nonzero=nz,
    )


def analyse_from_series(t: np.ndarray, y: np.ndarray, zero_rtol: float = 1e-8) -> SpectralAnalysis:
    """Convenience wrapper (kept for symmetry with future method-based calls)."""
    return analyse_trajectory(t, y, zero_rtol=zero_rtol)


# ---------------------------------------------------------------------------
# pointwise fingerprint (report table)
# ---------------------------------------------------------------------------
def fingerprint(t_val: float, y_val: np.ndarray) -> dict:
    """Everything we want to quote about one state."""
    lam_full = R.eigenvalues_3x3_numeric(t_val, y_val)
    lam_red = R.reduced_eigenvalues_analytic(y_val)
    lam_red = lam_red[np.argsort(-np.abs(lam_red.real))]
    tr, det = R.reduced_jacobian_invariants(y_val)
    disc = float(tr ** 2 - 4.0 * det)
    abs_re = np.abs(lam_red.real)
    S = float(abs_re.max() / abs_re.min()) if abs_re.min() > 0 else np.nan

    m2, m12, m13, m23 = R.principal_minors_2x2(t_val, y_val)

    return {
        "t": float(t_val),
        "y": np.asarray(y_val, dtype=float).tolist(),
        "lambda_1_fast": complex(lam_red[0]),
        "lambda_2_slow": complex(lam_red[1]),
        "lambda_3_structural": complex(lam_full[-1]),
        "abs_lambda_max": float(abs_re.max()),
        "abs_lambda_min_nonzero": float(abs_re.min()),
        "stiffness_ratio": S,
        "tr_reduced": float(tr),
        "det_reduced": float(det),
        "det_upper_left_block": float(np.linalg.det(R.jacobian(t_val, y_val)[np.ix_([0, 1], [0, 1])])),
        "m2_principal_minors": float(m2),
        "discriminant": disc,
        "is_complex_pair": bool(disc < 0.0),
    }


def check_value_comparison(t: np.ndarray, y: np.ndarray) -> list[dict]:
    """Compare our own stiffness ratios with the brief's orientation values."""
    rows = []
    for t_target, brief in BRIEF_STIFFNESS_RATIO.items():
        i = int(np.argmin(np.abs(t - t_target)))
        fp = fingerprint(t[i], y[i])
        rows.append({
            "t_target": t_target,
            "t_used": fp["t"],
            "y2": fp["y"][R.I_B],
            "S_ours": fp["stiffness_ratio"],
            "S_brief": brief,
            "ratio_ours_over_brief": fp["stiffness_ratio"] / brief,
            "abs_lambda_max": fp["abs_lambda_max"],
        })
    return rows


# ---------------------------------------------------------------------------
# the eigenvalue-collision / conjugate-pair window
# ---------------------------------------------------------------------------
def discriminant_of(y: np.ndarray) -> np.ndarray:
    """Discriminant ``tr^2 - 4 det`` of the reduced characteristic polynomial."""
    tr, det = R.reduced_jacobian_invariants(np.asarray(y, dtype=float))
    return tr * tr - 4.0 * det


def complex_pair_window(t: np.ndarray, y: np.ndarray) -> dict:
    """Locate the (very narrow) interval where the two eigenvalues are a
    complex-conjugate pair.

    With ``y3 = 0`` the discriminant is the quadratic ``0.99933*q^2 - 0.08*q +
    0.0016`` in ``q = 6e7*y2``, whose minimum is slightly negative.  The window
    must be located on a *dense* early-time trajectory: the fixed comparison
    grid is too coarse to see it.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    disc = discriminant_of(y)
    mask = disc < 0.0

    out = {
        "n_negative_discriminant": int(np.count_nonzero(mask)),
        "disc_min": float(disc.min()),
        "disc_min_t": float(t[int(np.argmin(disc))]),
    }
    if np.any(mask):
        lam = R.reduced_eigenvalues_analytic(y[mask])
        out.update({
            "t_lo": float(t[mask].min()),
            "t_hi": float(t[mask].max()),
            "y2_at_window": [float(y[mask][:, R.I_B].min()), float(y[mask][:, R.I_B].max())],
            "max_abs_im_over_abs_re": float(
                np.max(np.abs(lam.imag) / np.maximum(np.abs(lam.real), 1e-300))
            ),
        })
    return out


# ---------------------------------------------------------------------------
# consistency certificate: m2(J) == det(B_red)
# ---------------------------------------------------------------------------
def consistency_check(t: np.ndarray, y: np.ndarray) -> dict:
    """Verify that the reduced Jacobian really carries the non-zero spectrum."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    worst_abs = 0.0
    worst_rel = 0.0
    worst_eig = 0.0
    for i in range(t.size):
        m2, *_ = R.principal_minors_2x2(t[i], y[i])
        det_red = float(np.linalg.det(R.reduced_jacobian(t[i], y[i])))
        det_closed = float(np.linalg.det(R.reduced_jacobian_closed_form(t[i], y[i])))
        scale = max(abs(m2), abs(det_red), 1e-300)
        worst_abs = max(worst_abs, abs(m2 - det_red))
        worst_rel = max(worst_rel, abs(m2 - det_red) / scale)
        worst_rel = max(worst_rel, abs(det_red - det_closed) / max(abs(det_red), 1e-300))

        lam_full = R.eigenvalues_3x3_numeric(t[i], y[i])
        nz_full = lam_full[np.abs(lam_full.real) > 1e-8 * np.max(np.abs(lam_full.real))]
        lam_red = R.reduced_eigenvalues_analytic(y[i])
        if nz_full.size == 2:
            worst_eig = max(worst_eig, float(np.max(np.abs(np.sort_complex(nz_full) - np.sort_complex(lam_red)))))

    return {
        "max_abs_diff_m2_vs_det_reduced": float(worst_abs),
        "max_rel_diff_m2_vs_det_reduced": float(worst_rel),
        "max_abs_diff_nonzero_eigenvalues_3x3_vs_reduced": float(worst_eig),
        "passes": bool(worst_rel < 1e-8 and worst_eig < 1e-8),
    }
