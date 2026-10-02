"""Newton tolerance vs the step-doubling estimate.

The claim under test is not "a loose tolerance is inaccurate" -- it is sharper:
past a *computable* threshold the tolerance silently changes the method, and the
error estimate that is supposed to police the step cannot tell.
"""

from __future__ import annotations

import json
import numpy as np
import pytest

from src.analysis import newton_estimator as NE
from src.model import robertson as R


@pytest.fixture(scope="module")
def study():
    return NE.run()


def _explicit_bound() -> float:
    """The measured explicit-Euler stability bound, read from the results file."""
    with open("results/summary.json") as fh:
        s = json.load(fh)
    return float(s["frozen_jacobian_bound"]["per_method"]["explicit_euler"]["h_bound"])


def test_predictor_residual_is_h_squared_times_jacobian_f(study):
    """``G(y_pred) = -h^2 J_f f + O(h^3)`` -- the identity the threshold rests on.

    Everything downstream is a consequence of this, so it is checked directly
    rather than assumed, and the agreement improves as ``h`` shrinks.
    """
    t0, y0, _f = NE.state_at()
    errs = []
    for h in (1e-3, 1e-2, 5e-2):
        exact = NE.predictor_residual(t0, y0, h)
        approx = h * h * NE.jf_norm(t0, y0)
        errs.append(abs(exact - approx) / exact)
    assert errs[0] < errs[1] < errs[2]          # O(h^3) relative to h^2
    assert max(errs) < 1e-2
    assert study.bound_check_max_rel_err < 5e-3


def test_the_predicted_threshold_matches_the_observed_transition(study):
    """``h <= sqrt(tol / ||J_f f||)`` predicts exactly which steps go explicit.

    This is a *prediction* from the state alone, tested against the measured
    Newton bookkeeping across the whole (tolerance, step) grid.
    """
    for ntol in NE.NTOLS:
        hstar = study.threshold_by_ntol[ntol]
        predicted = int(sum(1 for h in NE.HS if h <= hstar))
        assert study.explicit_steps_by_ntol[ntol] == predicted, ntol
        # and the classification agrees row by row
        for r in study.rows:
            if r.ntol == ntol:
                assert r.is_explicit == (r.h <= hstar)


def test_the_method_changes_while_the_estimate_does_not(study):
    """The headline, stated precisely: the flip to explicit Euler is invisible.

    At ``h = 1e-3`` the trial goes from ``0/7`` explicit steps at ``tol = 1e-14``
    to ``7/7`` at ``tol = 1e-6`` -- implicit Euler becomes explicit Euler, which
    at that step is running 1.7x above its own stability bound -- and the error
    estimate that gates every step changes by 0.08 %.
    """
    at = [r for r in study.rows if r.h == 1e-3]
    by_tol = {r.ntol: r.err_estimate for r in at}

    tight, loose = by_tol[1.0e-14], by_tol[1.0e-6]
    assert abs(loose - tight) / tight < 1e-3          # the estimate barely moves
    assert study.explicit_steps_by_ntol[1.0e-14] == 0
    assert study.explicit_steps_by_ntol[1.0e-6] == len(NE.HS)

    # at the largest swept step, likewise: fully implicit vs fully explicit
    big = {r.ntol: r.err_estimate for r in study.rows if r.h == 1e-2}
    assert abs(big[1e-6] - big[1e-14]) / big[1e-14] < 0.01


def test_the_estimate_depends_only_on_how_the_updates_are_distributed(study):
    """The mechanism, exactly: it is a *mismatch* between the two sides that shows.

    ``E = ||y_f - y_c||`` is a difference between a coarse solve and two half
    solves.  Grouping the whole grid by the triple of update counts shows two
    facts:

    * rows that share a triple agree to the last digit -- spread exactly 1.0,
      whatever the tolerance and whatever the count;
    * every visible jump (3.00x) is a row where the coarse solve and the half
      solves were corrected to *different* accuracy, i.e. ``(1, 0, 0)``.

    That is the tutorial's "Newton errors must be small enough not to dominate
    the difference", as a measurement rather than a warning.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for r in study.rows:
        groups[(r.h, r.iters_each)].append(r.err_estimate)
    for key, es in groups.items():
        es = np.array(es)
        assert es.max() / es.min() == pytest.approx(1.0, abs=1e-12), key

    outliers = []
    for h in NE.HS:
        rows = [r for r in study.rows if r.h == h]
        es = np.array([r.err_estimate for r in rows])
        if es.max() / es.min() > 1.5:
            outliers += [r for r in rows if r.err_estimate == es.max()]
    assert outliers, "expected the mismatch rows to stand out"
    for r in outliers:
        assert r.sides_unequal
        assert r.iters_each[0] > max(r.iters_each[1], r.iters_each[2])
    # the jump is a clean factor of three, not a drift
    for h in NE.HS:
        es = np.array([r.err_estimate for r in study.rows if r.h == h])
        spread = float(es.max() / es.min())
        assert spread == pytest.approx(1.0, abs=0.02) or \
            spread == pytest.approx(3.0, rel=0.02), (h, spread)


def test_a_fully_explicit_trial_agrees_with_a_fully_implicit_one(study):
    """Why the flip is invisible: ``(0,0,0)`` and ``(1,1,1)`` give nearly the same E.

    Both are *consistent* trials -- every solve treated the same way -- so the
    estimator returns the same number to within a percent.  First order is first
    order.  A controller therefore cannot use E to tell a stable implicit step
    from an unstable explicit one, which is the whole point of the table.
    """
    for h in NE.HS:
        rows = [r for r in study.rows if r.h == h]
        zeros = [r.err_estimate for r in rows if r.iters_each == (0, 0, 0)]
        ones = [r.err_estimate for r in rows if r.iters_each == (1, 1, 1)]
        if zeros and ones:
            assert abs(zeros[0] - ones[0]) / ones[0] < 1e-2, h

    # the extremes of the whole grid at one step size: the tightest tolerance
    # corrects all three solves, the loosest corrects none of them
    at = {r.ntol: r for r in study.rows if r.h == 1e-2}
    tight, loose = at[1e-14], at[1e-6]
    assert min(tight.iters_each) >= 1
    assert loose.iters_each == (0, 0, 0)
    assert abs(loose.err_estimate - tight.err_estimate) / tight.err_estimate < 0.01


def test_the_estimate_keeps_its_h_squared_scaling_in_both_regimes(study):
    """``E ~ h^2`` is the estimator's assumption, and it survives the switch.

    That is exactly why the switch is invisible: first order is first order, so
    the signal keeps the shape the controller expects.
    """
    for ntol in (1.0e-14, 1.0e-6):
        assert study.slope_by_ntol[ntol] == pytest.approx(2.0, abs=0.05)
    # both regimes agree on the exponent to within a few percent
    assert abs(study.slope_by_ntol[1e-14] - study.slope_by_ntol[1e-6]) < 0.05


def test_the_threshold_is_monotone_and_state_determined(study):
    ts = [study.threshold_by_ntol[n] for n in NE.NTOLS]
    assert ts == sorted(ts)
    # an order of magnitude in tolerance, half an order in the threshold
    assert study.threshold_by_ntol[1e-6] / study.threshold_by_ntol[1e-8] == \
        pytest.approx(np.sqrt(100.0), rel=1e-9)


def test_a_loose_tolerance_licenses_steps_explicit_euler_cannot_stablely_take(study):
    """The consequence, against the measured stability bound.

    With ``tol = 1e-6`` every swept step up to ``1e-2`` is an explicit step, and
    ``1e-2`` is seventeen times the explicit-Euler stability bound.  The
    controller sees a healthy estimate the whole way.
    """
    bound = _explicit_bound()
    assert bound < 1e-3, bound
    hstar = study.threshold_by_ntol[1.0e-6]
    assert hstar > 10.0 * bound
    # a step inside the explicit-only regime, and above the stability bound
    h = 1e-3
    assert bound < h < hstar
    row = [r for r in study.rows if r.ntol == 1.0e-6 and r.h == h][0]
    assert row.is_explicit
    assert row.err_estimate < 1e-8           # yet the estimate looks excellent


def test_a_tight_tolerance_really_solves_the_nonlinear_equation(study):
    """The other side of the ledger: a tight tolerance does take updates.

    If this failed, ``zero updates`` would mean something else entirely.
    """
    tight = [r for r in study.rows if r.ntol == 1e-14]
    assert all(r.zero_updates == 0 for r in tight)
    assert all(r.iters_max >= 1 for r in tight)
    # at the largest step the undamped solve needs several directions
    big = [r for r in tight if r.h == 1e-2][0]
    assert big.iters_max >= 2
