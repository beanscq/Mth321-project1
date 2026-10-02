"""Reference-solution verification.

The reference is the measuring stick for everything else, so it gets its own
tests: tolerance repeat, independent methods, invariant, published orientation.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis import reference as REF
from src.grid import output_times
from src.model import robertson as R

T_EVAL = output_times(n_log=120, include_t0=True)


@pytest.fixture(scope="module")
def refs():
    tight = REF.solve("Radau", 1.0e-10, [1.0e-14, 1.0e-20, 1.0e-14], T_EVAL)
    tighter = REF.solve("Radau", 1.0e-12, [1.0e-16, 1.0e-22, 1.0e-16], T_EVAL)
    bdf = REF.solve("BDF", 1.0e-12, [1.0e-16, 1.0e-22, 1.0e-16], T_EVAL)
    lsoda = REF.solve("LSODA", 1.0e-12, [1.0e-16, 1.0e-22, 1.0e-16], T_EVAL)
    for r in (tight, tighter, bdf, lsoda):
        assert r.success, r.message
    return dict(tight=tight, tighter=tighter, bdf=bdf, lsoda=lsoda)


def test_tolerance_repeat_confirms_at_least_9_digits(refs):
    rep = REF.matching_digits(refs["tight"].y, refs["tighter"].y)
    assert rep["min_digits"] >= 9.0, rep
    # the absolute agreement is what limits every later error measurement
    assert rep["max_abs_diff"] < 1e-11, rep


def test_independent_methods_agree(refs):
    for key in ("bdf", "lsoda"):
        rep = REF.matching_digits(refs[key].y, refs["tighter"].y)
        assert rep["min_digits"] >= 7.0, (key, rep)
        assert rep["max_abs_diff"] < 1e-10, (key, rep)


def test_elementwise_scaling_of_the_digit_metric_is_a_trap():
    """Regression for a real bug: with an elementwise scale, y2(0)=y3(0)=0 makes
    the digit metric collapse even though the absolute agreement is 2e-12."""
    a = np.array([[1.0, 1e-20, 1e-20], [1.0, 1e-5, 1e-5]])
    b = a + 2e-12
    rep = REF.matching_digits(a, b)
    assert rep["min_digits"] >= 6.0        # profile-scale definition behaves
    assert rep["max_abs_diff"] < 1e-11


def test_reference_preserves_the_linear_invariant(refs):
    y = refs["tighter"].y
    assert np.max(R.invariant_defect(y)) < 1e-11


def test_reference_is_non_negative(refs):
    y = refs["tighter"].y
    assert y.min() >= -1e-15


def test_reference_y2_decays_monotonically_after_its_peak(refs):
    """Sanity check from the brief: no spurious oscillation after the peak."""
    t = refs["tighter"].t
    y = refs["tighter"].y
    i_peak = int(np.argmax(y[:, R.I_B]))
    tail = y[i_peak:, R.I_B]
    assert np.all(np.diff(tail) <= 1e-18)


def test_reference_matches_the_published_orientation_value_at_t40(refs):
    t = refs["tighter"].t
    y = refs["tighter"].y
    i = int(np.argmin(np.abs(t - 40.0)))
    assert np.allclose(y[i], R.BRIEF_Y_AT_T40, rtol=0, atol=5e-7)


def test_first_output_time_is_zero_and_grid_shape():
    assert T_EVAL[0] == 0.0
    assert T_EVAL.size == 121
    assert T_EVAL[1] == pytest.approx(1e-8, rel=1e-12)
    assert T_EVAL[-1] == pytest.approx(40.0, rel=1e-12)
