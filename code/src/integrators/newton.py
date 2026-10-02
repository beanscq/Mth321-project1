"""Newton iteration for the nonlinear systems produced by implicit methods.

Residual and stopping rule
--------------------------
The convergence test is on the **nonlinear residual**

    G(z) = z - y_n - h f(t_{n+1}, z)

and the residual is the right thing to test, because for this problem it
controls the only invariant that can be violated:

    (1,1,1)^T G(z) = (z1+z2+z3) - (y_n1+y_n2+y_n3) - h * (1,1,1)^T f(t,z)
                   = (z1+z2+z3) - (y_n1+y_n2+y_n3)          since (1,1,1)^T f == 0

so ``|invariant defect introduced by the step| = |sum G| <= 3 * ||G||_inf``.
A stopping tolerance on the residual therefore translates *directly* into a
bound on the invariant defect, which a tolerance on the Newton correction
``||dz||`` does not.

``norm`` selects how the residual vector is reduced to one number (``"inf"``,
the default, or ``"2"``).  The infinity norm is the default because of the
identity above; the 2-norm is what the tutorial's tables are written in, so
``norm="2"`` reproduces them.  The same norm is used for the stopping test and
for the damping decrease test -- comparing one norm and testing another would
make "the residual went down" meaningless.

Damping (backtracking)
----------------------
With ``damping=True`` the direction ``JF(z) d = -F(z)`` is still computed once,
but the update is ``z + alpha d`` with ``alpha`` chosen by backtracking:
start at ``alpha = 1``, and halve until the trial residual is finite and
strictly smaller than the current one.  Passing ``armijo_c`` replaces that by a
*sufficient decrease* condition on the merit function ``phi = ||G||^2 / 2``.
Because ``grad phi^T d = G^T (JF d) = -||G||^2``, the condition
``phi(z + alpha d) <= phi(z) - c * alpha * ||G||^2`` is, in terms of the norm,

    ||G(z + alpha d)||^2 <= (1 - 2 c alpha) * ||G(z)||^2 ,

and that is the form implemented (so ``c`` must lie in ``(0, 1/2)``, matching
the tutorial's convention).  If the backtracking budget or ``alpha_min`` is
reached the solve is reported as **failed**, not silently returned: the caller
is then expected to reduce the time step and retry, exactly as the tutorial's
accept/reject table requires.

Cost is counted, not assumed.  Each backtracking trial is one extra evaluation
of the residual, hence one extra evaluation of ``f``.  Damping therefore trades
right-hand-side evaluations for a smaller iteration count -- and whether that
trade is worth it is a measurement, reported in the tables, not a claim.

Newton is deliberately **not** driven to machine precision by default: the
algebraic error only has to be negligible relative to the time-discretisation
error, which is what the tolerance sweep demonstrates.  That statement has a
sharp edge, though: the error *estimator* of a step-doubling controller also
measures a difference of two solves, so if the Newton tolerance is loose enough
its algebraic error contaminates the estimate.  The interaction experiment
measures where that happens.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

NORMS = ("inf", "2")

FAIL_NONE = None
FAIL_NONFINITE = "non_finite_residual"
FAIL_SINGULAR = "singular_linear_solve"
FAIL_STAGNATION = "stagnation"
FAIL_DAMPING = "damping_exhausted"
FAIL_MAXITER = "iteration_limit"


@dataclass
class NewtonResult:
    x: np.ndarray
    converged: bool
    iters: int                       # number of Newton *directions* computed
    residual: float                  # final residual, in the working norm
    residual_history: list[float] = field(default_factory=list)
    rhs_calls: int = 0
    jac_calls: int = 0
    lu_solves: int = 0
    stagnated: bool = False
    f_used: np.ndarray | None = None  # f(t, x) at the returned iterate
    damping: bool = False
    norm: str = "inf"
    alpha_history: list[float] = field(default_factory=list)
    n_backtracks: int = 0            # total halvings across all updates
    n_trials: int = 0                # residual evaluations spent on backtracking
    failure: str | None = None
    #: (alpha, residual) for every trial that was *rejected* by the damping test
    rejected_trials: list[tuple[float, float]] = field(default_factory=list)

    @property
    def mean_alpha(self) -> float:
        a = self.alpha_history
        return float(np.mean(a)) if a else float("nan")

    @property
    def min_alpha(self) -> float:
        a = self.alpha_history
        return float(np.min(a)) if a else float("nan")


def _reduce(v: np.ndarray, norm: str) -> float:
    a = np.abs(np.asarray(v, dtype=float))
    if a.size == 0:
        return 0.0
    if norm == "2":
        return float(np.sqrt(np.dot(a, a)))
    return float(np.max(a))


def solve(residual_fn,
          x0: np.ndarray,
          tol: float,
          jac_fn,
          maxiter: int = 50,
          damping: bool = False,
          norm: str = "inf",
          max_backtracks: int = 32,
          alpha_min: float = 1.0e-6,
          armijo_c: float | None = None) -> NewtonResult:
    """Solve ``G(x) = 0`` by Newton's method.

    ``residual_fn(x) -> (G, f)`` returns the residual and the vector field at
    ``x``; returning ``f`` lets the caller reuse the last evaluation for dense
    output at no extra cost.

    ``tol`` is an absolute tolerance on the residual in the chosen ``norm``.

    ``damping=True`` enables backtracking with the strict-decrease test
    ``||G_trial|| < ||G||``; passing ``armijo_c`` replaces it by the stronger
    sufficient-decrease condition.  A damped solve that cannot find a
    decreasing trial returns ``converged=False`` with
    ``failure=FAIL_DAMPING`` so the outer solver can cut ``h``.
    """
    if norm not in NORMS:
        raise ValueError(f"norm must be one of {NORMS}")
    if damping and not (0.0 < alpha_min <= 1.0):
        raise ValueError("need 0 < alpha_min <= 1")
    if armijo_c is not None and not (0.0 < armijo_c < 0.5):
        raise ValueError("armijo_c must lie strictly between 0 and 1/2")

    x = np.array(x0, dtype=float).reshape(-1)
    hist: list[float] = []
    alpha_hist: list[float] = []
    rejected: list[tuple[float, float]] = []
    converged = False
    stagnated = False
    failure: str | None = FAIL_NONE
    jac_calls = 0
    lu_solves = 0
    n_backtracks = 0
    n_trials = 0
    rhs_calls = 0
    f_used = None

    for it in range(maxiter + 1):
        G, f_used = residual_fn(x)
        rhs_calls += 1
        G = np.asarray(G, dtype=float)
        r = _reduce(G, norm)
        if not np.isfinite(r):
            hist.append(r)
            failure = FAIL_NONFINITE
            break
        hist.append(r)
        if r <= tol:
            converged = True
            break
        if it == maxiter:
            failure = FAIL_MAXITER
            break                       # budget exhausted; not converged

        Jn = np.asarray(jac_fn(x), dtype=float)
        jac_calls += 1
        try:
            dx = np.linalg.solve(Jn, -G)
        except np.linalg.LinAlgError:
            failure = FAIL_SINGULAR
            break
        lu_solves += 1
        if not np.all(np.isfinite(dx)):
            failure = FAIL_SINGULAR
            break
        step_norm = float(np.max(np.abs(dx)))
        if step_norm == 0.0:
            # exact root of the linear model but residual above tol: the
            # iteration cannot make progress, which happens when tol is
            # pushed below the rounding level of G.
            stagnated = True
            failure = FAIL_STAGNATION
            break

        if not damping:
            alpha_hist.append(1.0)
            x = x + dx
            continue

        # ---- backtracking ------------------------------------------------
        alpha = 1.0
        accepted = False
        for _ in range(max_backtracks + 1):
            xt = x + alpha * dx
            Gt, _ft = residual_fn(xt)
            n_trials += 1
            rhs_calls += 1
            rt = _reduce(np.asarray(Gt, dtype=float), norm)
            if np.isfinite(rt):
                if armijo_c is None:
                    good = rt < r
                else:
                    # sufficient decrease on phi = ||G||^2 / 2, written out in
                    # terms of the norm: ||G_t||^2 <= (1 - 2 c alpha) ||G||^2
                    good = rt * rt <= (1.0 - 2.0 * armijo_c * alpha) * r * r
                if good:
                    accepted = True
                    break
            rejected.append((float(alpha), float(rt)))
            if alpha <= alpha_min:
                break
            alpha *= 0.5
            n_backtracks += 1

        if not accepted:
            # deliberately leave x untouched: a failed update must not be
            # propagated, and the caller has to cut h and retry
            failure = FAIL_DAMPING
            break
        alpha_hist.append(float(alpha))
        x = np.asarray(xt, dtype=float)

    return NewtonResult(
        x=x,
        converged=converged,
        iters=jac_calls,
        residual=float(hist[-1]) if hist else float("inf"),
        residual_history=[float(v) for v in hist],
        rhs_calls=rhs_calls,
        jac_calls=jac_calls,
        lu_solves=lu_solves,
        stagnated=stagnated,
        f_used=None if f_used is None else np.asarray(f_used, dtype=float),
        damping=bool(damping),
        norm=str(norm),
        alpha_history=alpha_hist,
        n_backtracks=n_backtracks,
        n_trials=n_trials,
        failure=failure,
        rejected_trials=rejected,
    )
