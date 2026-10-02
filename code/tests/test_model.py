"""Verification of the authoritative model.

These tests are the contract of the whole project: if the vector field, the
Jacobians or the eigenvalue formulas are wrong, every later number is wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.model import robertson as R

STATES = [
    np.array([1.0, 0.0, 0.0]),
    np.array([1.0 - 1e-5, 1e-5, 0.0]),
    np.array([0.7, 1e-6, 0.3]),
    np.array([0.7158271, 9.1855e-6, 0.2841637]),   # brief orientation state at t = 40
    np.array([0.9999999, 3.0e-9, 7.0e-8]),
]


# ---------------------------------------------------------------------------
# chemistry -> ODE
# ---------------------------------------------------------------------------
def test_y0_is_normalised_and_non_negative():
    assert R.Y0.shape == (3,)
    assert R.Y0.sum() == 1.0
    assert np.all(R.Y0 >= 0.0)


@pytest.mark.parametrize("y", STATES)
def test_invariant_identity_holds_exactly(y):
    """(1,1,1) . f(y) == 0 algebraically, not just approximately."""
    val = float(np.ones(3) @ R.f(0.0, y))
    assert abs(val) <= 1e-15 * max(1.0, float(np.abs(R.f(0.0, y)).max()))


def test_two_b_step_carries_no_extra_factor_of_two():
    """The schematic B+B -> B+C contributes -k2*y2^2 to y2' and +k2*y2^2 to y3'."""
    y = np.array([1.0, 1.0e-5, 0.0])
    r1, r2, r3 = R.rates(y)
    assert r2 == pytest.approx(R.K2 * y[1] ** 2, rel=1e-15)

    rhs = R.f(0.0, y)
    assert rhs[R.I_C] == pytest.approx(R.K2 * y[1] ** 2, rel=1e-15)
    assert rhs[R.I_B] == pytest.approx(r1 - r2 - r3, rel=1e-15)
    # explicit guard: no 2*k2 anywhere
    assert abs(rhs[R.I_B] - (r1 - 2.0 * r2 - r3)) > 1e-12


def test_rate_constants_match_the_brief():
    assert R.K1 == 0.04
    assert R.K2 == 3.0e7
    assert R.K3 == 1.0e4


# ---------------------------------------------------------------------------
# Jacobian
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(6))
def test_jacobian_matches_complex_step_derivative(seed):
    """Complex-step differentiation is exact to machine precision.

    ``eps = 1e-8`` is safe because the complex step has no subtractive
    cancellation; the truncation error is O(eps**2) ~ 1e-16.
    """
    rng = np.random.default_rng(seed)
    y = rng.random(3)
    y /= y.sum()

    J = R.jacobian(0.0, y)
    eps = 1e-6                 # no subtractive cancellation => large eps is fine
    J_cs = np.zeros((3, 3))
    for j in range(3):
        e = np.zeros(3, dtype=complex)
        e[j] = 1j * eps
        J_cs[:, j] = R.f(0.0, y.astype(complex) + e).imag / eps

    rel = np.abs(J - J_cs) / np.maximum(np.abs(J), 1e-9)
    assert rel.max() < 1e-7, (rel, J, J_cs)


@pytest.mark.parametrize("y", STATES)
def test_left_null_vector_annihilates_the_jacobian(y):
    """(1,1,1)^T J == 0  <=>  the structural eigenvalue 0 exists."""
    val = np.ones(3) @ R.jacobian(0.0, y)
    assert np.allclose(val, 0.0, atol=1e-14)


# ---------------------------------------------------------------------------
# reduced Jacobian : projection, closed form, determinant certificate
# ---------------------------------------------------------------------------
NON_DEGENERATE = [s for s in STATES if s[R.I_B] > 0.0 and s[R.I_C] > 0.0]


@pytest.mark.parametrize("y", STATES)
def test_reduced_jacobian_matches_its_closed_form(y):
    B = R.reduced_jacobian(0.0, y)
    B_closed = R.reduced_jacobian_closed_form(0.0, y)
    assert np.allclose(B, B_closed, rtol=1e-12, atol=1e-18)


@pytest.mark.parametrize("y", NON_DEGENERATE)
def test_reduced_jacobian_is_the_projection_not_the_upper_left_block(y):
    """Away from the degenerate state the two 2x2 matrices genuinely differ."""
    B = R.reduced_jacobian(0.0, y)
    upper_left = R.jacobian(0.0, y)[np.ix_([0, 1], [0, 1])]
    assert not np.allclose(B, upper_left)


def test_projection_and_upper_left_block_coincide_only_at_the_degenerate_state():
    """At y = (1,0,0) every correction term vanishes, so the two agree there.

    This is exactly why the mistake is so easy to make: the difference is
    invisible at t = 0 and only shows up once y2 and y3 become non-zero.
    """
    y = np.array([1.0, 0.0, 0.0])
    B = R.reduced_jacobian(0.0, y)
    upper_left = R.jacobian(0.0, y)[np.ix_([0, 1], [0, 1])]
    assert np.allclose(B, upper_left)


@pytest.mark.parametrize("y", STATES)
def test_det_reduced_equals_sum_of_principal_2x2_minors(y):
    """Certificate: tr(J) == tr(B_red) and m2(J) == det(B_red)."""
    m2, m12, m13, m23 = R.principal_minors_2x2(0.0, y)
    B = R.reduced_jacobian(0.0, y)
    assert float(np.trace(B)) == pytest.approx(float(np.trace(R.jacobian(0.0, y))), rel=1e-12)
    assert float(np.linalg.det(B)) == pytest.approx(m2, rel=1e-10, abs=1e-18)


def test_reduced_determinant_closed_form():
    """det = 2*k2*y2*(k1 + k3*y2) -- the second term is the one students miss."""
    y = np.array([0.7158271, 9.1855e-6, 0.2841637])
    y2 = y[R.I_B]
    expected = 2.0 * R.K2 * y2 * (R.K1 + R.K3 * y2)          # = 2.4e6*y2 + 6e11*y2^2
    naive = 2.0 * R.K2 * y2 * R.K1                            # = 2.4e6*y2  (WRONG)
    det = float(np.linalg.det(R.reduced_jacobian(0.0, y)))
    assert det == pytest.approx(expected, rel=1e-12)
    assert det / naive == pytest.approx(3.3, rel=0.05)


@pytest.mark.parametrize("y", STATES)
def test_reduced_eigenvalues_are_the_non_zero_eigenvalues_of_J(y):
    lam_red = np.sort_complex(R.reduced_eigenvalues_analytic(y))
    lam_full = R.eigenvalues_3x3_numeric(0.0, y)
    scale = float(np.abs(lam_full.real).max())
    nz = np.sort_complex(lam_full[np.abs(lam_full.real) > 1e-8 * scale])
    if nz.size == 2:
        assert np.allclose(lam_red, nz, rtol=1e-8, atol=1e-14 * max(scale, 1.0))


# ---------------------------------------------------------------------------
# spectral fingerprint against the values published in the brief
# ---------------------------------------------------------------------------
def test_stiffness_ratio_and_lambda_max_at_t40_match_the_brief():
    """Regression against the brief's orientation values.

    ``|lambda|max = 3.39e3`` and ``S(40) = 1.58e5``.  The ratio only comes out
    right if the reduced Jacobian is the true projection.
    """
    y = R.BRIEF_Y_AT_T40
    lam = R.reduced_eigenvalues_analytic(y)
    abs_re = np.abs(lam.real)

    assert abs_re.max() == pytest.approx(3.39e3, rel=0.01)
    assert abs_re.max() / abs_re.min() == pytest.approx(1.58e5, rel=0.02)

    # the naive upper-left block would give 5.2e5 instead
    J = R.jacobian(0.0, y)
    lam_naive = np.linalg.eigvals(J[np.ix_([0, 1], [0, 1])])
    naive_ratio = np.abs(lam_naive.real).max() / np.abs(lam_naive.real).min()
    assert naive_ratio == pytest.approx(5.2e5, rel=0.05)
