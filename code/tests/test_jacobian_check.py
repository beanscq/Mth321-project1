"""Verification of the analytic Jacobian against forward differences.

The point of these tests is not "the Jacobian is right" -- it is *how much the
check can be trusted*, which is what the perturbation and the state control.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis import jacobian_check as JC
from src.model import robertson as R


@pytest.fixture(scope="module")
def report():
    return JC.run()


def test_column_convention_is_the_one_implemented():
    """Perturbing component j must produce column j, not row j.

    ``J`` is symmetric at ``y(0)``, so the initial state cannot tell the two
    apart -- which is exactly why the convention has to be checked where the
    matrix is *not* symmetric.
    """
    y = JC.representative_states()["transient"]
    J = np.asarray(R.jacobian(0.0, y), float)
    assert not np.allclose(J, J.T), "need an asymmetric state for this test"

    JD, eps, calls = JC.forward_difference_jacobian(R.f, 0.0, y, eps=1e-12)
    assert calls == y.size + 1                 # the base value is reused

    # the difference reproduces J, not its transpose
    assert JC.scaled_discrepancy(J, JD) < 1e-6
    assert JC.scaled_discrepancy(J.T, JD) > 1e-2


def test_transposed_jacobian_is_a_detectable_bug():
    """A transposed residual Jacobian is caught, at every state."""
    y = JC.representative_states()["transient"]
    h = 1e-3
    JF = JC.residual_jacobian(R.f, 0.0, y, h)

    def F(z):
        return z - y - h * np.asarray(R.f(h, z), float)

    eps, _eta = JC.tuned_perturbation(lambda t, z: F(z), 0.0, y, JF)
    JF_fd, _e, _c = JC.forward_difference_jacobian(lambda t, z: F(z), 0.0, y, eps=eps)
    assert JC.scaled_discrepancy(JF, JF_fd) < 1e-5
    assert JC.scaled_discrepancy(JF.T, JF_fd) > 1e-2


def test_residual_jacobian_is_not_the_plain_jacobian(report):
    """``I - h J_f`` and ``J_f`` are different claims; both are checked.

    Two consequences show up in the numbers.  The residual Jacobian is better
    conditioned to difference, because ``h`` scales the curvature away; and it
    needs a *different* tuned perturbation, because the trade-off involves the
    size of the values being subtracted, which differ between ``f`` and ``F``.
    """
    y = JC.representative_states()["initial"]
    h = 1e-3
    JF = JC.residual_jacobian(R.f, 0.0, y, h)

    def F(z):
        return z - y - h * np.asarray(R.f(h, z), float)

    JF_fd, _e, _c = JC.forward_difference_jacobian(lambda t, z: F(z), 0.0, y)
    Jf = np.asarray(R.jacobian(0.0, y), float)

    eta_res = JC.scaled_discrepancy(JF, JF_fd)
    eta_plain = JC.scaled_discrepancy(Jf, JF_fd)
    assert eta_res < 1e-3
    assert eta_plain / eta_res > 100.0
    # the two functions do not share one optimal perturbation
    assert report.residual_tuned_eps != pytest.approx(report.best_eps, rel=1e-6)


def test_perturbation_follows_the_scaling_rule():
    y = np.array([1.0, 1.0e-5, 0.0])
    base = JC.perturbation(y, multiplier=1.0)
    # max(1, |y_j|): the two small components get the floor, not their own size
    assert np.allclose(base, JC.SQRT_EPS_MACH)
    # order-one changes to the multiplier move it proportionally
    assert np.allclose(JC.perturbation(y, multiplier=0.1), 0.1 * base)
    assert np.allclose(JC.perturbation(y, multiplier=10.0), 10.0 * base)
    # an explicit reference scale replaces the floor, for dimensional variables
    scaled = JC.perturbation(y, scale=np.array([1.0, 1.0e-5, 1.0e-5]))
    assert scaled[1] == pytest.approx(JC.SQRT_EPS_MACH * 1.0e-5)


def test_rule_of_thumb_perturbation_is_truncation_dominated(report):
    """At ``y(0)`` the correct Jacobian scores eta ~ 0.45 under the sqrt(eps) rule.

    The bulk of that is the difference operator, not the Jacobian: shrinking eps
    by 10 shrinks the discrepancy by 10, which is the truncation signature.  Any
    "eta must be below 1e-8" gate would fail a *correct* Jacobian here -- the
    tutorial's "no universal absolute requirement", measured.
    """
    rows = {(r.label, r.multiplier): r for r in report.reports}
    e1 = rows[("initial", 0.1)].eta
    e10 = rows[("initial", 10.0)].eta
    assert 1e-2 < e1 < 1.0
    assert e10 / e1 == pytest.approx(100.0, rel=0.05)     # linear in eps

    # the sweep finds a perturbation that removes almost all of it
    assert report.state_eta_tuned["initial"] < e1 / 1000.0
    assert report.best_eps < JC.SQRT_EPS_MACH / 1000.0


def test_tuned_perturbation_verifies_the_jacobian_at_every_state(report):
    """With a tuned perturbation the check is decisive -- where the state allows it.

    The initial state stays the worst of the three even when tuned: its Jacobian
    has ``||J||_inf = 0.04``, which the ``max(1, ...)`` floor in the discrepancy
    denominator turns into an absolute error measure.  That is a property of the
    *measure*, and saying so is part of using it honestly.
    """
    assert report.state_eta_tuned["transient"] < 1e-7
    assert report.state_eta_tuned["end"] < 1e-7
    assert report.state_eta_tuned["initial"] < 1e-4
    # worst of the three, and by a wide margin
    assert (report.state_eta_tuned["initial"]
            > 100.0 * report.state_eta_tuned["transient"])
    # the residual Jacobian verifies to the same order once tuned on its own
    for eta in report.state_eta_residual_tuned.values():
        assert eta < 1e-4


def test_perturbation_sweep_has_an_interior_minimum(report):
    """Roundoff on one side, truncation on the other -- the U is the evidence.

    A monotone sweep would mean the check is not measuring what it claims.
    """
    etas = [e for _eps, e in report.sweep]
    i = int(np.argmin(etas))
    assert 0 < i < len(etas) - 1
    assert etas[0] > etas[i] * 10.0        # small eps: cancellation
    assert etas[-1] > etas[i] * 10.0       # large eps: truncation
    assert report.best_eta < 1e-7


def test_missing_coupling_is_invisible_at_the_initial_state():
    """The tutorial's warning, as a measurement.

    Dropping the whole ``10^4 y2 y3`` reaction leaves ``J_f(y(0))`` *unchanged*,
    because ``y2 = y3 = 0`` kills all four entries the bug touches.  The
    separation between the buggy and correct matrices is exactly 1 there -- the
    check is blind -- and enormous at a state where the coupling is live.
    """
    demo = JC.missing_coupling_demo()
    rows = {(r["state"], r["eps"]): r for r in demo["rows"]}

    assert rows[("initial", 1.49e-8)]["separation"] == pytest.approx(1.0)
    assert rows[("initial", 1.0e-12)]["separation"] == pytest.approx(1.0)
    # ... and at the initial state the buggy matrix IS the correct matrix
    y0 = R.Y0.astype(float)
    assert np.array_equal(JC.buggy_jacobian(0.0, y0), np.asarray(R.jacobian(0.0, y0)))

    # a live-coupling state exposes it, and the perturbation decides how clearly
    assert rows[("end", 1.49e-8)]["separation"] > 1e3
    assert rows[("end", 1.0e-12)]["separation"] > 1e6
    assert rows[("transient", 1.0e-12)]["separation"] > 1e3
    # with the rule-of-thumb perturbation the same bug is nearly invisible
    assert rows[("transient", 1.49e-8)]["separation"] < 5.0
