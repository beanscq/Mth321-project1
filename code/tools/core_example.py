"""Six measurements that the core modules can make, printed as they happen.

    python tools/core_example.py                 # from the project root
    python robertson_core/example.py             # from the folder copy

Not part of the project proper -- ``make_bundle.py --core-dir`` copies this file
into ``robertson_core/example.py`` so the folder demonstrates itself.  It exists
because a folder of modules is not evidence that the modules run, and every
number below is recomputed here rather than quoted from a table.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
# works both in tools/ (project root one level up) and at the folder root
ROOT = HERE if (HERE / "src").is_dir() else HERE.parent
sys.path.insert(0, str(ROOT))

from src.analysis import eigen as EIG                     # noqa: E402
from src.analysis import jacobian_check as JC             # noqa: E402
from src.analysis import newton_estimator as NE           # noqa: E402
from src.integrators import make                          # noqa: E402
from src.integrators import newton as NEW                 # noqa: E402
from src.model import robertson as R                      # noqa: E402

y0 = R.Y0
print(f"core tree : {ROOT}")
print(f"numpy     : {np.__version__}")

# Robertson's initial state, as a plain array -- the modules all accept either.
y0 = np.asarray(y0, float)
print()


# --- 1 ---------------------------------------------------------------------
print("1. one implicit-Euler step at h = 1e-2")
tr = make("implicit_euler", h=1.0e-2, newton_tol=1.0e-12).integrate(
    1.0e-2, t_end=1.0e-2)
y1 = tr.endpoint()
print(f"   y(1e-2)      = {np.array2string(y1, precision=12)}")
print(f"   sum(y) - 1   = {abs(y1.sum() - 1.0):.3e}   (conserved by construction)")
print(f"   rhs/jac/lu   = {tr.stats.rhs_calls}/{tr.stats.jac_calls}/"
      f"{tr.stats.lu_solves}")


# --- 2 ---------------------------------------------------------------------
# Backtracking is a guard against a bad initial guess.  Whether it fires depends
# entirely on what that guess is, and the driver's guess is free.
def one_step_solve(h: float, guess: np.ndarray, damping: bool):
    """``G(z) = z - y0 - h f(h, z) = 0`` from ``guess``, as in ``ImplicitEuler._step``."""
    def residual_fn(z):
        fz = R.f(h, z)
        return z - y0 - h * fz, fz

    def jac_fn(z):
        return np.eye(y0.size) - h * R.jacobian(h, z)

    return NEW.solve(residual_fn, guess, 1.0e-12, jac_fn, maxiter=50,
                     damping=damping)


print("\n2. damping, and the warm start that makes it redundant")
print("   (a) guess = the old state  (T16's protocol)")
for h in (1.0e-2, 1.0, 4.0e1):
    full = one_step_solve(h, y0, False)
    damp = one_step_solve(h, y0, True)
    print(f"       h={h:8.0e}  full {full.iters:3} updates / {full.rhs_calls:3} rhs"
          f"   |   damped {damp.iters:3} updates / {damp.rhs_calls:3} rhs"
          f"   ({damp.n_backtracks} backtracks)")

print("   (b) guess = the free explicit-Euler predictor  (the driver's protocol)")
for damp in (False, True):
    t = make("implicit_euler", h=1.0e-2, newton_tol=1.0e-12,
             damping=damp).integrate(1.0e-2, t_end=40.0)
    st = t.stats
    print(f"       damping={str(damp):5} {st.n_steps:5} steps  "
          f"{st.newton_iters:6} updates  max/step={st.newton_iters_max_per_step}  "
          f"backtracks={st.newton_backtracks}  rhs={st.rhs_calls}")
print("   -> both protocols return the same root.  Damping is what saves the")
print("      cold start; with the predictor it never fires and costs one extra")
print("      residual per direction.  The guard is insurance, not a saving.")


# --- 3 ---------------------------------------------------------------------
print("\n3. stiffness: same h, one method dies and the other does not")
for key in ("explicit_euler", "implicit_euler"):
    t = make(key, h=1.0e-2).integrate(1.0e-2, t_end=1.0)
    tail = f"aborted at t={t.stats.abort_t:.4f} ({t.stats.abort_reason})" \
        if t.stats.aborted else f"reached t={t.t_covered:.4f}  y2={t.endpoint()[1]:.4e}"
    print(f"   {key:15} {tail}")


# --- 4 ---------------------------------------------------------------------
print("\n4. spectrum at t = 1 (the ratio needs a non-degenerate state)")
t1 = make("implicit_euler", h=1.0e-3).integrate(1.0e-3, t_end=1.0)
fp = EIG.fingerprint(1.0, t1.endpoint())
print(f"   lambda_fast  = {fp['lambda_1_fast']:.6e}")
print(f"   lambda_slow  = {fp['lambda_2_slow']:.6e}")
print(f"   structural   = {fp['lambda_3_structural']:.3e}")
print(f"   stiffness S  = {fp['stiffness_ratio']:.4e}")
print(f"   frozen bound = 2/|Re lambda_fast| = "
      f"{2.0 / fp['abs_lambda_max']:.4e}   (an estimate, not the measured bound)")


# --- 5 ---------------------------------------------------------------------
print("\n5. the analytic Jacobian against column-wise differences")
states = JC.representative_states()
for name in ("initial", "transient", "end"):
    y = states[name]
    Jd, eps, calls = JC.forward_difference_jacobian(R.f, 0.0, y, multiplier=1.0)
    eta = JC.scaled_discrepancy(R.jacobian(0.0, y), Jd)
    tuned_eps, tuned_eta = JC.tuned_perturbation(
        R.f, 0.0, y, R.jacobian(0.0, y))
    print(f"   {name:9} eta(rule)={eta:.3e}   eta(best eps={tuned_eps:.0e})="
          f"{tuned_eta:.3e}   ({calls} rhs calls)")

bug = JC.buggy_jacobian(0.0, states["initial"])
diff = R.jacobian(0.0, states["initial"]) - bug
print(f"   a Jacobian missing the whole 1e4*y2*y3 reaction differs from the "
      f"correct one by {abs(diff).max():.3e}")
print("   -> at the initial state that reaction touches only zero entries, so a")
print("      single-state check would hand the wrong matrix a pass")


# --- 6 ---------------------------------------------------------------------
print("\n6. the Newton tolerance decides which method actually runs")
t_star, y_star, _f_star = NE.state_at(1.0)
jf = NE.jf_norm(t_star, y_star)
print(f"   state     : t={t_star}  y2={y_star[1]:.6e}")
print(f"   ||J_f f|| = {jf:.6e}")
for ntol in (1.0e-14, 1.0e-10, 1.0e-8, 1.0e-6):
    h_star = NE.predictor_threshold(t_star, y_star, ntol)
    E, iters, zero, n_solves, each = NE.step_doubling_err(
        t_star, y_star, 1.0e-2, ntol)
    tag = "explicit" if zero else "Newton-corrected"
    print(f"   tol={ntol:.0e}  h*={h_star:.3e}  E(h=1e-2)={E:.3e}  "
          f"updates={each}  -> {tag}")

print("\n(6) is the point of the last study: E tracks the size of the local error,")
print("    not the name of the method that made it.  Above h* the predictor is")
print("    already inside tolerance, Newton takes no update, and the step is an")
print("    explicit-Euler step wearing an implicit-Euler label.")
