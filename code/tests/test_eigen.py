"""Spectral / stiffness verification."""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis import eigen as E
from src.analysis import reference as REF
from src.analysis import stability_region as SR
from src.grid import output_times
from src.model import robertson as R


@pytest.fixture(scope="module")
def ref():
    t = output_times(n_log=200, include_t0=True)
    run = REF.solve("Radau", 1.0e-10, [1.0e-14, 1.0e-20, 1.0e-14], t)
    assert run.success
    return np.asarray(run.t), np.asarray(run.y)


# ---------------------------------------------------------------------------
# structural zero eigenvalue
# ---------------------------------------------------------------------------
def test_exactly_one_eigenvalue_is_classified_as_structural_zero(ref):
    t, y = ref
    spec = E.analyse_trajectory(t, y)
    counts = spec.n_nonzero
    # t = 0 is degenerate (double zero eigenvalue); every later state has exactly one
    assert np.all(counts[t > 0.0] == 2)
    assert counts[0] == 1
    assert spec.has_structural_zero


def test_structural_zero_sits_at_machine_zero_and_aligns_with_left_null_vector(ref):
    t, y = ref
    spec = E.analyse_trajectory(t, y)
    lam0 = spec.lam_full[:, -1]
    fast = np.abs(spec.lam_full[:, 0].real)
    # the zero eigenvalue must be negligible relative to the largest eigenvalue
    assert np.max(np.abs(lam0.real) / np.maximum(fast, 1e-30)) < 1e-12
    # and the corresponding left eigenvector must be (1,1,1)/sqrt(3)
    assert np.min(spec.zero_alignment[t > 0.0]) > 1.0 - 1e-7


def test_stiffness_ratio_is_undefined_at_the_degenerate_initial_state(ref):
    t, y = ref
    spec = E.analyse_trajectory(t, y)
    assert np.isnan(spec.stiffness_ratio[0])
    assert np.all(np.isfinite(spec.stiffness_ratio[t > 0.0]))


# ---------------------------------------------------------------------------
# stiffness ratio values
# ---------------------------------------------------------------------------
def test_stiffness_ratio_matches_the_brief_check_values(ref):
    t, y = ref
    for row in E.check_value_comparison(t, y):
        assert row["ratio_ours_over_brief"] == pytest.approx(1.0, rel=0.10), row


def test_lambda_max_at_t40_is_about_3p39e3(ref):
    t, y = ref
    i = int(np.argmin(np.abs(t - 40.0)))
    assert E.fingerprint(t[i], y[i])["abs_lambda_max"] == pytest.approx(3.39e3, rel=0.01)


def test_stiffness_ratio_grows_from_benign_to_stiff(ref):
    """The system is NOT stiff at the start; stiffness builds up with y2 and y3."""
    t, y = ref
    spec = E.analyse_trajectory(t, y)
    S_first = spec.stiffness_ratio[t > 0.0][0]
    S_last = spec.stiffness_ratio[-1]
    assert 1.0 < S_first < 10.0
    assert S_last > 1e5
    assert S_last / S_first > 1e4


# ---------------------------------------------------------------------------
# the narrow conjugate-pair window (analytic prediction)
# ---------------------------------------------------------------------------
def test_narrow_complex_pair_window_exists_and_is_negligible():
    """disc < 0 on a tiny interval of y2 around 6.6e-10 (~5% wide in y2).

    With y3 = 0 the discriminant is 0.99933*q^2 - 0.08*q + 0.0016 with
    q = 6e7*y2; its minimum is slightly negative.
    """
    y2 = np.linspace(6.0e-10, 7.5e-10, 20001)
    y = np.stack([1.0 - y2, y2, np.zeros_like(y2)], axis=-1)
    disc = R.reduced_jacobian_invariants(y)[0] ** 2 - 4.0 * R.reduced_jacobian_invariants(y)[1]

    negative = disc < 0.0
    assert negative.any()

    q = 6.0e7 * y2
    expected_roots = np.sort(np.roots([1.0 - 1.0 / 1500.0, -0.08, 0.0016]))
    lo, hi = float(expected_roots[0]) / 6.0e7, float(expected_roots[1]) / 6.0e7
    assert y2[negative].min() == pytest.approx(lo, rel=0.02)
    assert y2[negative].max() == pytest.approx(hi, rel=0.02)

    lam = R.reduced_eigenvalues_analytic(y[negative])
    ratio = np.abs(lam.imag).max() / np.abs(lam.real).max()
    assert ratio < 0.02      # never more than 2% of the real part


def test_complex_window_never_binds_the_explicit_euler_bound(ref):
    """The complex window must never be the constraint that sets h.

    At each state take the eigenvalue with the largest |Re| (the binding one) and
    solve |1 + h*lam| <= 1 exactly for a possibly complex lam::

        (1 - r)^2 + d^2 <= 1  with  r = h|Re lam|, d = h|Im lam|
        =>  h <= (1 + sqrt(1 - (|Im|/|Re|)^2)) / |Re|

    The global bound is the minimum over the trajectory.  It must be attained on
    a *real* eigenvalue and must reproduce 2/3.39278e3 = 5.9e-4.
    """
    t, y = ref
    lam = R.reduced_eigenvalues_analytic(y)
    idx = np.argmax(np.abs(lam.real), axis=-1)
    re = np.take_along_axis(lam.real, idx[:, None], axis=-1)[:, 0]
    im = np.take_along_axis(lam.imag, idx[:, None], axis=-1)[:, 0]

    delta = np.abs(im / re)
    h_bound = (1.0 + np.sqrt(1.0 - delta ** 2)) / np.abs(re)
    pos = t > 0
    h_global = float(h_bound[pos].min())
    i_bind = int(np.where(pos)[0][int(np.argmin(h_bound[pos]))])

    assert delta[i_bind] == pytest.approx(0.0, abs=1e-12)     # binding lam is real
    lam_max = float(np.abs(lam.real).max())
    assert lam_max == pytest.approx(3.39e3, rel=0.01)
    assert h_global == pytest.approx(2.0 / lam_max, rel=1e-12)
    assert 5.85e-4 < h_global < 5.95e-4

    # where the pair is complex the bound is orders of magnitude looser
    assert h_bound[pos].max() > 10.0
    assert np.count_nonzero(delta[pos] > 0) < 5


# ---------------------------------------------------------------------------
# stability-region algebra
# ---------------------------------------------------------------------------
def test_stability_region_real_axis_intervals():
    assert SR.real_axis_stability_interval("Explicit Euler") == (-2.0, 0.0)
    a_rk4, _ = SR.real_axis_stability_interval("RK4")
    assert a_rk4 == pytest.approx(-2.7853, rel=1e-3)
    a_ie, _ = SR.real_axis_stability_interval("Implicit Euler")
    assert np.isneginf(a_ie)


def test_implicit_euler_is_a_stable_on_the_whole_left_half_plane():
    re = np.linspace(-1e4, -1e-6, 500)
    z = re + 0j
    assert np.all(np.abs(SR.R_implicit_euler(z)) <= 1.0 + 1e-14)


def test_explicit_euler_bound_from_lambda_max():
    """h < 2/|lambda|max = 5.9e-4 from the most restrictive frozen Jacobian."""
    h = SR.max_stable_real_step("Explicit Euler") / 3.39278e3
    assert h == pytest.approx(5.9e-4, rel=0.02)


def test_rk4_does_not_escape_the_stiffness_restriction():
    """RK4 allows only 1.39x the explicit-Euler step -- higher order is no cure."""
    h_rk4 = SR.max_stable_real_step("RK4") / 3.39278e3
    h_ee = SR.max_stable_real_step("Explicit Euler") / 3.39278e3
    assert h_rk4 == pytest.approx(8.2e-4, rel=0.03)
    assert 1.3 < h_rk4 / h_ee < 1.5
