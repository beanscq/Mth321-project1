# Robertson problem — Numerical Analysis of ODEs and PDEs, Project 1 (Topic ③)

> 中文版见 **[README.zh.md](README.zh.md)**。两版逐节对应；本文件为准。

Stiff chemical-kinetics benchmark. Everything here is built to be **verifiable**, not
just plausible: the chemistry and the numerics have to support each other.

**Research question.**
*How do the chemically induced fast and slow time scales in the Robertson reaction
network affect the stability, accuracy, positivity and cost of explicit and implicit
ODE solvers?*

---

## Status

| Stage | Content | State |
|---|---|---|
| **P0** | scaffolding, authoritative model, unit tests, tight reference | **done** |
| **P1** | Jacobian spectrum, structural zero, stiffness ratio, absolute stability regions | **done** |
| **P2** | Explicit Euler / RK4 / Implicit Euler + Newton; frozen-Jacobian bound vs empirical step sweep; uniform-step convergence orders | **done** |
| **P3** | four-track diagnostics (conservation / non-negativity / full-state error / y2(40)); terminal `y2(40)` from the implicit method; Newton tolerance sweep; RK4 step-doubling adaptive controller | **done** |
| P4 | matched-accuracy cost model (F12, T15) | **done** |
| P4 | beamer slides (`slides/`) | **done** |
| **T2** | Tutorial 2 pass: damped Newton, independent Jacobian verification, Newton tolerance vs the step-doubling estimator (T16–T18, F14–F15) | **done** |
| P4 | LaTeX report, final pass over the AI transparency log | pending |

`run_all.py --list` prints the implemented stages.

---

## The model (authoritative — do not change)

```
y1' = -0.04*y1 + 1e4*y2*y3
y2' =  0.04*y1 - 1e4*y2*y3 - 3e7*y2^2
y3' =  3e7*y2^2
y(0) = [1, 0, 0],   t in [0, 40]
```

Reaction labels are mnemonic only: `r1 = k1*y1` (A→B), `r2 = k2*y2^2` (2B→B+C, **net**
−1 B / +1 C), `r3 = k3*y2*y3` (B+C→A+C). Following the brief, **no extra factor of two**
is applied to the displayed `3e7*y2^2` term. The invariant `(1,1,1)·f ≡ 0` is asserted
in the test suite.

---

## Running it

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

python run_all.py                 # all stages -> results/   (about 6 minutes)
python run_all.py --stages reference eigen
python -m pytest tests -q         # 148 verification tests
```

The pipeline is single-threaded. Measured on the machine this was developed on, the last full run
took 6 min 7 s, and the three experiments dominate: `sweep` 178 s, `integrators` 109 s,
`diagnostics` 50 s. Everything else is small — `newtonsweep` 11 s, `adaptive` 9 s, `figures` 8 s,
and the three Tutorial-2 studies together 1.5 s, which is why they are run inside `run_all.py`
rather than kept as a separate script. Set `OMP_NUM_THREADS=1` before running if the machine is
otherwise busy; the linear algebra here is 3×3 and gains nothing from threads. Every table and
figure under `results/` is produced by this command — no number in the report is typed by hand.

---

## Layout

```
config/default.yaml      single source of truth for every numeric setting
run_all.py               one command -> every table and figure
src/model/robertson.py   f, analytic J, reduced J, invariants
src/grid.py              the common comparison grid (t=0 plus 200 log points)
src/integrators/base.py       uniform-step driver, StepStats cost counters, cubic-Hermite dense output
src/integrators/explicit_euler.py / rk4.py / implicit_euler.py
src/integrators/newton.py     Newton with a residual-based stopping rule
src/integrators/adaptive.py   RK4 step doubling + the mixed/absolute error norms
src/analysis/reference.py  tight reference + zoom reference
src/analysis/eigen.py      spectrum, structural zero, stiffness ratio, consistency certificate
src/analysis/stability_region.py  R(z) for the three methods
src/analysis/metrics.py    per-run diagnostics and the status flags
src/analysis/step_sweep.py frozen-Jacobian bound + empirical step sweep
src/analysis/convergence.py observed-order fitting with a stated window
src/analysis/diagnostics.py    four tracks kept apart, digit bookkeeping, endpoint y2(40)
src/analysis/newton_sweep.py   Newton stopping-tolerance sweep and its recommendation rule
src/analysis/adaptive_study.py adaptive runs at several tolerances, both error norms
src/analysis/jacobian_check.py forward-difference verification of `J_f` and `J_F = I - h J_f`
src/analysis/newton_estimator.py Newton tolerance against the step-doubling estimate (T18)
src/reporting/figures.py   F2, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13
src/reporting/tables.py    CSV + Markdown tables
tools/render_p4_cost.py  post-processing only: rebuilds F12 + T15 from T7 ∩ T9 (no 10-minute rerun)
tools/render_newton_studies.py  renders T16–T18 and F14–F15; called by `run_all.py`
tools/core_example.py    six live measurements; ships as `robertson_core/example.py`
tools/make_bundle.py     packages the project: zip + one big .md + `robertson_core/`
tests/                   148 tests; all green
results/{reference,raw,figures,tables}
slides/robertson_slides.tex   beamer deck; every number copied from results/
appendix/ai_transparency_log.md
```

Handing the code to someone else — three shapes, one command each:

```sh
python tools/make_bundle.py            # robertson_code_bundle.zip + robertson_all_code.md
python tools/make_bundle.py --core     # robertson_core_code.md    (numerics only, one file)
python tools/make_bundle.py --core-dir # robertson_core/           (numerics only, runnable tree)
```

`robertson_core/` carries the reference solution so `src/analysis/jacobian_check.py` and
`newton_estimator.py` can be called, and `example.py`, which prints six live measurements —
a folder of modules is not evidence that the modules run.

---

## Conventions that must not be broken

**1. One comparison grid.** Include `t = 0` plus ≥200 logarithmically spaced output
times in `[1e-8, 40]` (201 points in total). Solvers choose their own internal steps;
every method **and** the reference are interpolated onto this grid before comparison.

**2. Interpolation order ≥ method order.** The dense-output scheme is **cubic Hermite**
(`config: interpolation.scheme`). Plain linear interpolation has `O(h^2)` error and
would cap RK4's observed convergence order at 2, silently destroying the convergence
study. This is a hard requirement, not a stylistic choice.

**3. The reduced Jacobian is a projection, not a submatrix.** Eliminating
`y3 = 1 - y1 - y2` gives `B_red = S·J·T`, i.e.

```
B_red = [[ -k1 - k3*y2 ,    k3*y3 - k3*y2          ],
         [  k1 + k3*y2 , -k3*y3 + k3*y2 - 2*k2*y2  ]]

tr(B_red)  = -k1 - k3*y3 - 2*k2*y2                  (= tr J)
det(B_red) = 2*k2*y2*(k1 + k3*y2) = 2.4e6*y2 + 6e11*y2^2
```

Taking the upper-left 2×2 block of `J` instead drops the `6e11*y2^2` term: at `t = 40`
that is a 3.3× error in the determinant and turns `S(40) = 1.58e5` into `5.2e5`.
The certificate `m2(J) == det(B_red)` (sum of principal 2×2 minors) is asserted in
`tests/test_model.py` and re-checked on the whole trajectory by `eigen.consistency_check`.

**4. Structural zero eigenvalue.** `(1,1,1)^T f ≡ 0` ⇒ `(1,1,1)^T J ≡ 0`. The stiffness
ratio uses **only the two non-zero eigenvalues**, with two independent classifications:
a magnitude threshold (`|Re λ| ≤ 1e-8 · max|Re λ|`) and alignment with the exact left
null vector `(1,1,1)/√3`.

**5. t = 0 is degenerate.** There `y2 = y3 = 0` and `B_red = [[-0.04, 0], [0.04, 0]]`,
so of the two "non-zero" eigenvalues one is exactly zero; the full Jacobian has a
*double* zero eigenvalue. No stiffness ratio is reported at `t = 0`; plots start from
the first positive logarithmic time.

**6. Dense output is free, and it is not allowed to be the bottleneck.** `f` is stored
at every step node, and each method obtains it from an evaluation it already performs
(Explicit Euler and RK4: the value at the new node is the next step's first evaluation;
Implicit Euler: the final Newton residual evaluation). So the Hermite interpolant adds
**zero** right-hand-side evaluations, and F8 needs no caveat of the form "the order is
capped by an implementation artefact". This is checked: RK4's fitted order on the **full**
grid (4.23) is the same as on `t ≥ 1e-3` (4.27).

**7. Status flags have stated thresholds and never substitute for one another.**
`config: criteria`. A run is *stable* if it does not diverge
(`max_i |y_i| ≤ 1e3`, or a non-finite state); *non-negative* if `min_i y_i ≥ -1e-12`
at every accepted step node; *accurate* if `max_t |y_h - y_ref|_∞ ≤ 1e-4` **and** the run
covers `[0, 40]`. Non-negativity is measured on the **step nodes**, i.e. on what the
method actually produced, not on the interpolant.

**8. Cost is counted, not estimated.** Per run we record right-hand-side evaluations,
Jacobian evaluations, linear solves, the Newton iteration count, accepted/rejected steps
and wall time. The default weighted cost uses `{rhs: 1, jac: 1, lu: 1}` with one RHS
evaluation as the unit; `config: cost.not_counted` lists explicitly what is **excluded**
(interpreter overhead, the free dense output, assembling the analytic Jacobian, I/O,
parallelism).

**9. Three things that are part of a result, not settings.** A Jacobian discrepancy is
reported *with the state and the perturbation* it was measured at, because it depends on
both and has no universal threshold. A Newton tolerance is reported *with every result*,
because past `h* = √(tol/‖J_f f‖_∞)` it changes which method is being run and no cheap
monitor sees it. A rejected step is reported *separately* and never counted in any
trajectory statistic — it did not happen. Damped Newton reports backtracking trials in
their own column, so it is charged for the residual evaluations a backtrack costs.

**10. Three questions stay separate.** Stability of the time-stepper, convergence of
Newton, and accuracy of the time discretisation are three claims with three different
tests. A successful solve is not evidence about the accuracy of the step it produced, and
neither is a conservation defect (see convention 4 and Entry 10).

---

## P0/P1 results

### Reference

Radau, `rtol = 1e-12`, `atol = [1e-16, 1e-22, 1e-16]`, with the tolerance-repeat
protocol (both tolerances also run at ×100 looser). Confirmed by the repeat:
**10.8 significant digits**, maximum absolute disagreement **2.1e-12**. BDF and LSODA
run as independent oracles agree to 10.7 and 11.0 digits. Linear invariant defect
≤ 2.4e-15.

| quantity | our reference | brief orientation value |
|---|---|---|
| `y1(40)` | 0.71582706872 | 0.7158271 |
| `y2(40)` | 9.1855347646e-06 | 9.1855e-06 |
| `y3(40)` | 0.28416374575 | 0.2841637 |
| `y2` peak | 3.6487e-05 at `t = 4.4201e-03` | — |

### Stiffness ratio — the brief's check values reproduced

| `t` | our `S(t)` | brief check value | ratio |
|---|---|---|---|
| `1e-4` | 3.0097e+03 | 3.0e+03 | 1.003 |
| `1e-2` | 5.4221e+03 | 5.4e+03 | 1.004 |
| `40` | 1.5840e+05 | 1.58e+05 | 1.003 |

`|λ|max(40) = 3392.788` (brief: ≈3.39e3). Reproducing the ratios — not just `|λ|max` —
is what confirms the reduced-Jacobian convention above, because `|λ|max ≈ tr(J)` is the
same under both conventions while the determinant is not.

**`S(t)` is not constant and does not start large.** It rises from `1.67` at `t = 1e-8`
to `1.58e5` at `t = 40` — about five decades. The system is benign at the start and
becomes stiff as `y2` and `y3` build up. The slow eigenvalue is **not** `k1 = 0.04`:
it is `0.0214` at `t = 40`, i.e. the slow mode is slower than `1/k1`.

### Eigenvalue collision (F5c)

The two non-zero eigenvalues are real and negative **except** in a very narrow window
`t ∈ [1.630e-08, 1.705e-08]` where the discriminant is slightly negative
(`min disc = -1.07e-6`) and they form a conjugate pair with
`|Im λ| / |Re λ| ≤ 1.29e-2`. Consequence for the numerics: none (the explicit-Euler
bound there is ~50, versus 5.9e-4 at `t = 40`). Consequence for the computation:
this is a near-double root, so eigenvalues are obtained from the **quadratic formula
with the exact trace/determinant**, not from `np.linalg.eig`, which is visibly noisy
there. The fixed 200-point grid steps straight over the window (0 states land in it);
`reference.zoom_reference()` exists purely to resolve it, and panel F5c says so.

### Absolute stability regions (F4)

| method | order | stable real interval | A-stable | implied `h` for `|λ|max = 3.39e3` |
|---|---|---|---|---|
| Explicit Euler | 1 | `[-2, 0]` | no | 5.895e-04 |
| RK4 | 4 | `[-2.7853, 0]` | no | 8.209e-04 |
| Implicit Euler | 1 | `-inf < x ≤ 0` | yes (L-stable) | unbounded |

RK4 buys only a factor **1.39** over explicit Euler. Higher order does not cure
stiffness, and the matched-accuracy cost comparison must be read with that in mind.

---

## P2 results

### The frozen-Jacobian estimate, tested rather than asserted

`|λ|max(t)` is **not** maximal at `t = 0`. Early on the fast eigenvalue is
`-(k3·y3 + 2·k2·y2)` with `y2, y3 → 0`, so it is small; the maximum is reached at the
**end** of the trajectory, where `k3·y3 = 2841.6` and `2·k2·y2 = 551.1` give
`|λ|max = 3392.79`. Hence `h < 2/3392.79 = 5.895e-4` (Euler) and
`h < 2.7853/3392.79 = 8.209e-4` (RK4).

The empirical sweep (22 step sizes in `[1.5e-4, 3e-3]`) brackets those numbers:

| method | largest stable `h` | first diverging `h` | frozen bound | ratio |
|---|---|---|---|---|
| Explicit Euler | 5.4159e-04 | 6.2464e-04 | 5.8949e-04 | 0.919 |
| RK4 | 8.3088e-04 | 9.5827e-04 | 8.2095e-04 | 1.012 |
| Implicit Euler | none in `[1.5e-4, 3e-3]` | — | ∞ | — |

For both explicit methods the frozen bound falls between the last stable step and the
first diverging one. RK4 survives 1.2% *beyond* its prediction; explicit Euler fails 8%
*short* of it. Both are within the one-grid-step resolution of the sweep, which is all a
local frozen-Jacobian estimate can be asked for — it is a diagnostic, not a theorem.

**The three boundaries do not coincide, and the discrepancy is the interesting part.**

* Explicit Euler: stability, non-negativity and accuracy all fail at the same `h`
  (5.4159e-04). Here instability *is* a positivity failure — once `|1+hλ| > 1` the mode
  alternates in sign and drives `y2, y3` negative. First divergence at `t = 9.8e-3`.
* RK4: **accuracy fails first**, at `h = 7.2041e-04`, well before stability
  (`8.3088e-04`). There is a window in which RK4 is stable and non-negative but has a
  full-state error above `1e-4`. "It did not blow up" is not an accuracy claim.
* Implicit Euler: no boundary within the swept range at all. At `h = 3e-3` it is still
  stable, non-negative and accurate (`error < 1e-4`) — the stability restriction is
  genuinely gone, not merely postponed.

### Uniform-step convergence

Five successively halved steps `h ∈ {1e-3, 5e-4, 2.5e-4, 1.25e-4, 6.25e-5}` (=four
halvings), every method, every run. Levels at `h = 1e-3` are above the explicit bound and
are reported as unstable rather than discarded.

| method | nominal order | **finest-pair** observed order | fitted slope (`t ≥ 1e-3`) | fitted slope (full grid) |
|---|---|---|---|---|
| Explicit Euler | 1 | **0.998** | 1.079 ± 0.033 | 1.146 |
| RK4 | 4 | **4.023** | 4.268 ± 0.074 | 4.231 |
| Implicit Euler | 1 | **1.0000** | 0.99996 ± 0.00001 | 0.99996 |

Notes that must survive into the report:

* The fitted slope of RK4 is **4.27**, not 4.00, because the coarsest admissible level
  (`h = 5e-4`) is close to the stability boundary where the error constant is amplified.
  The **finest pair** gives 4.023 and the per-pair orders fall monotonically
  4.49 → 4.21 → 4.00. Reporting only the least-squares slope would overstate the method;
  reporting only the finest pair would hide the pre-asymptotic region. Both are in
  `T8_observed_order.md` and panel F8(c).
* The full-grid slopes agree with the `t ≥ 1e-3` slopes, which is the evidence that the
  **cubic-Hermite dense output is not limiting the observed order** — the failure mode the
  project brief warns about would have shown up as RK4 stalling at order 2.
* Implicit Euler's fitted slope is `1.0000 ± 0.00001` with `R² = 1.000000`: its error is
  the textbook `C·h` with no contamination, because the Newton solve is exact to
  `1e-12` and its residual contributes nothing at this scale.

### Two things that change how the cost comparison must be read (P4)

1. **Newton is often free.** The explicit-Euler *predictor* residual is
   `h²·|f_t + J f|`, so for small `h` it is already below the Newton tolerance and no
   update is made. Measured iterations per step for implicit Euler:

   | `h` | residual evaluations / step | Newton updates / step |
   |---|---|---|
   | 1e-3 | 2.00 | 1.00 |
   | 2.5e-4 | 2.00 | 1.00 |
   | 1.25e-4 | 1.91 | 0.91 |
   | 6.25e-5 | 1.42 | 0.42 |

   At `h = 6.25e-5` most steps satisfy the implicit equation to `1e-12` immediately. That
   is a legitimate implicit step (the tolerance was met) and it is *not* a bug — but it
   means implicit Euler's per-step cost is not a constant multiple of a fixed
   `(rhs + jac + lu)` triple, and the cost model must use the measured counters.

2. **Conservation is completely insensitive to the Newton tolerance.** `(1,1,1)^T` is a
   left null vector of `J`, hence `(1,1,1)^T (I - hJ)^{-1} = (1,1,1)^T`, hence one Newton
   update restores the sum **exactly** for any stopping tolerance. The invariant defect is
   `~1e-14` for every run in every table here — including the divergent, negative explicit
   runs. This is the sharpest possible illustration of "conservation alone is weak", and
   it means the Newton tolerance sweep in P3 must be judged on the **solution error**, not
   on the invariant defect.

### A warning for the P4 cost comparison (not yet a conclusion)

Reading `T9_cost.md` at matched accuracy is already uncomfortable for the implicit method:

| method | `h` | error (`t ≥ 1e-3`) | weighted cost | RHS / Jacobian / LU |
|---|---|---|---|---|
| Explicit Euler | 5.0e-4 | 4.481e-06 | 8.00e+04 | 80001 / 0 / 0 |
| Implicit Euler | 1.0e-3 | 7.355e-06 | 1.60e+05 | 80016 / 40015 / 40015 |

Explicit Euler is **more accurate and cheaper** than implicit Euler at the same step count
here, because the implicit method pays a Jacobian and a linear solve per step while its
step-size advantage does not yet bite — explicit Euler is accuracy-limited, not
stability-limited, over this range, and its accuracy already passes `1e-4` at `h = 5e-4`.
Implicit Euler only becomes the cheaper choice once the target error is loose enough that
it can raise `h` past roughly `1e-2`. So the P4 report must **not** claim the usual
"implicit is more efficient"; it has to state the accuracy level at which each method wins,
and it must show the sensitivity of that crossover to the cost weights
(`config: cost.sensitivity_weights`) and to whether the Jacobian is analytic or
finite-differenced.

---

## P3 results

### 1. The four diagnostics disagree, and conservation is the useless one (F9, T13, T14)

This is not a stylistic separation. It is a theorem about this system. `(1,1,1)^T` is a
**left null vector** of `J`, so `(1,1,1)^T (I - hJ)^{-1} = (1,1,1)^T`, and for the implicit
residual `G(z) = z - y_n - h f(t_{n+1}, z)`:

```
sum G(z) = sum(z) - sum(y_n)        (because (1,1,1)^T f ≡ 0)
=>  for one Newton update,  sum(z + dz) = sum(z) - sum(G | z) = sum(y_n)
```

**Every Newton iterate preserves the sum exactly**, for *any* stopping tolerance —
including the explicit-Euler predictor, and including the zero-update case. Measured on
the `h`-sweep (T13):

| `h` | method | mass defect | `min_i y_i` | error | verdict |
|---|---|---|---|---|---|
| 1.5e-4 | all three | 9.5e-15 … 2.2e-14 | 0.0 | 4.6e-10 … 1.2e-6 | all four tracks pass |
| 8.31e-4 | RK4 | 6.2e-15 | 0.0 | **1.24e-4** | conservation, stability, positivity pass; **accuracy fails** |
| 1.27e-3 | Explicit Euler | 2.4e-15 | **−42.9** | **2.3e-2** | only conservation passes |

(These three rows are transcribed from `T13_four_track.md`. An earlier version of this table
carried `h = 1.5e-3`, `min_i y_i = −13.7`, `4.8e-2` and an RK4 mass defect of `2e-14`; none of
those appears in any sweep table — the sweep grid has no `1.5e-3` point, and the RK4 defect at
`8.3088e-4` is `6.2e-15`. The same bad row reached a draft slide frame and was caught there;
see Entry 11 of the transparency log, and Entry 13 for the README occurrence.)

The middle row is the whole point: a run can be stable, non-negative and still be wrong,
and no invariant defect will ever say so. The four tracks are computed from four different
columns and are allowed to disagree (`T14_four_track_boundaries.md` records where each one
stops being satisfied and *which one fails first*).

### 2. Terminal `y2(40)` from the implicit method (T10)

Only digits confirmed by our own tightened reference are quoted (the reference is good to
10.76 digits overall; `y2` to **10.16**, with a maximum absolute disagreement of 2.09e-12).
`y2(40)` is reported at two step sizes × two Newton tolerances:

| `h` | Newton tol | `y2(40)` | digits vs reference |
|---|---|---|---|
| 1.0e-3 | 1e-8 | 9.1840386117e-06 | 3.79 |
| 1.0e-3 | 1e-12 | 9.1856705286e-06 | 4.83 |
| 2.5e-4 | 1e-12 | 9.1855687065e-06 | 5.43 |
| 6.25e-5 | 1e-12 | **9.1855387584e-06** | **6.36** |

The convergence is first order, as it must be, so the digits come from the step size, not
from the Newton tolerance — once the tolerance is tight enough (see below) it stops
mattering. Our reference confirms the brief's orientation value
`y2 = 9.1855e-06` to **5.42 significant digits**, so `9.1855` may be quoted and the next
digit may not. (For completeness: `y1 = 0.7158271` is confirmed to 7.36 digits and
`y3 = 0.2841637` to 6.79.)

### 3. Newton need not be exact — and the safe tolerance is set by `h²`, not by intuition (F10, T11)

The explicit-Euler predictor has residual

```
G(y_pred) = h [ f(t_n, y_n) - f(t_{n+1}, y_pred) ] = -h² (f_t + J f) + O(h³)
```

so once the stopping tolerance exceeds roughly `h² |f_t + J f|` the **first** residual test
passes and the step is taken with **zero Newton updates** — implicit Euler silently becomes
explicit Euler with a lagged right-hand side. At `h = 10⁻³` that scale is `h² = 10⁻⁶`, and
the sweep puts the break exactly there:

| Newton tol | error (full grid) | inflation vs plateau | Newton updates / step | steps skipping Newton |
|---|---|---|---|---|
| 1e-2 | 1.44e+02 | 19602817 | 1.18 | 60% |
| 1e-4 | 3.59e+02 | 48855880 | 0.77 | 60% |
| **1e-6** | 1.96e-05 | **2.67** | 0.30 | 70% |
| **1e-8** | 6.44e-06 | **0.88** | 0.31 | 69% |
| 1e-10 | 7.35e-06 | 1.00 | 0.82 | 18% |
| 1e-12 | 7.35e-06 | 1.00 | 1.00 | 0% |
| 1e-14 | 7.35e-06 | 1.00 | 1.00 | 0% |

* **Plateau** `= 7.355e-06` at the tightest tolerance: pure time-discretisation error.
* **Recommended stopping tolerance `1e-8`**, the largest that keeps the algebraic error
  negligible (inflation 0.88 — the error is actually *smaller* than the plateau there,
  because algebraic and discretisation error partially cancel; inflation is measured
  against the tightest tolerance for exactly this reason).
* **First tolerance that breaks it: `1e-6`**, inflation 2.67.
* The **invariant defect is useless as a monitor**: at `tol = 1e-2` the error is 144, `min_i y_i = −143.8`, and the mass defect is 1.9e-12. Panel F10(a) plots the defect against
  the theoretical envelope `3·tol`: the bound is satisfied everywhere and carries no
  information whatsoever.
* Newton is **cheap**: one update per step at tight tolerances, and the update is free at
  loose ones. The cost model must use measured counters, not a fixed multiple.

### 4. Adaptive step size: the controller works, and one obvious error norm does not (F11, T12)

RK4 step doubling, `err = |y_half − y_full| / (2⁴ − 1)`, `safety = 0.9`, `facmax = 5`,
`facmin = 0.2`, componentwise-scaled acceptance `max_i |e_i| / (atol_i + |y_i|) ≤ tol`.

**(a) The step size really moves** (F11a): it collapses through the induction layer
(`h ≈ 5e-7` at `t = 1e-6`) and then plateaus. The plateau is the story:

| tol | median controller `h` | median sub-step `h/2` ÷ frozen bound | median `err/tol` | accepted / rejected |
|---|---|---|---|---|
| 1e-4 | 1.754e-03 | **1.07** | 0.18 | 20745 / 8127 |
| 1e-6 | 1.763e-03 | **1.07** | 0.19 | 20772 / 9241 |
| 1e-8 | 1.742e-03 | **1.06** | 0.31 | 22053 / 8317 |
| 1e-10 | 1.363e-03 | 0.83 | 0.59 | 31259 / 1 |

For every tolerance down to `1e-8` the **propagated sub-step `h/2` sits 6–7% above the
frozen-Jacobian bound 8.209e-4** — the controller is *stability*-limited, not
accuracy-limited, which is exactly what `err/tol ≈ 0.2 ≪ 1` says. The rejections cluster on
that plateau (F11a, grey crosses) and vanish (`1` rejection) as soon as accuracy takes over
at `tol = 1e-10`. This closes the loop with F7: the frozen bound is a local, linear
estimate, yet the *effective* step of a correctly implemented controller lands on it from
below to within 7%.

**(b) What is controlled is the local error, not the global one** (F11b/c). `err/tol` stays
below 1 at every accepted step with a roughly constant median (0.18 → 0.59). The *global*
error at `t = 40` is larger by roughly the step count:

| tol | 1e-4 | 1e-6 | 1e-8 | 1e-10 |
|---|---|---|---|---|
| global error | 3.23e-07 | 9.93e-09 | 1.99e-12 | 3.29e-14 |

roughly proportional to `tol` with `N ≈ 2e4` steps, so `global ≈ 0.03 · N · tol`. The
`tol = 1e-10` run is at the reference error floor — tightening further buys nothing.
Panel F11(c) states this honestly instead of claiming "error ≈ tolerance".

**(c) The failure that a plain `L∞` norm cannot see** (F11c, F11b of the control runs).
`y1 = O(1)` while `y2 = O(1e-5)`. With the plain absolute norm a relative error of `1e-2` in
`y2` contributes only `1e-7` to `err`, so at `tol = 1e-6` the controller **accepts steps
beyond RK4's stability limit**:

| norm | tol | reached | `min_i y_i` | global error | mass defect |
|---|---|---|---|---|---|
| absolute `L∞` | 1e-4 | **t = 1.9 only** | **−903** | 2.17e-03 | 1.1e-13 |
| absolute `L∞` | 1e-6 | 40 | **−4.77e-05** | **7.83e-03** | 7.8e-15 |
| scaled | 1e-6 | 40 | 0.0 | **9.93e-09** | 5.8e-15 |

Same controller, same tolerance, `10⁶` difference in accuracy — and the invariant defect
reports `1e-13` or below in all three rows, i.e. nothing wrong, every time. Both variants are
kept in the repository and both are run and tabulated; the failure is reported, not hidden.
(The three defect cells were corrected from `T12_adaptive.md` in a later verification pass; the
values originally written here were `1.9e-15`, `2.2e-14` and `2.1e-15`. Every other cell of this
table already matched. See Entry 14 of the transparency log.)

---

## Tutorial-2 pass — three checks the first draft did not make

Tutorial 2 separates three questions that the first draft of this work ran together:

1. **Is the time-stepper stable?** That is a property of `R(z)` on the spectrum of `J` — the
   A-stability / L-stability discussion. Implicit Euler is both; explicit Euler is neither.
2. **Does Newton converge?** That is a property of the *solve*, not of the trajectory, and it has
   its own acceptance test.
3. **Is the time discretisation accurate?** That is `O(h^p)`, and it is what the error estimator
   is supposed to police.

The first draft had (1) and (3) and treated (2) as a knob inside (3). Three concrete checks were
added; each produces a table and (two of them) a figure. Two of the three changed a *statement*,
not just a number.

### 5. Damped Newton: essential for a cold start, never needed by the driver (T16)

T16 starts Newton from the **old state**. That is a cold start, and it is a bad one: at `h = 40` the
first direction is a wild over-correction, so undamped Newton needs 21 updates — past the tutorial's
"at most 10 updates" allowance.

| `h` | full Newton directions | damped directions | backtracks | trials | full / damped RHS | same root |
|---|---|---|---|---|---|---|
| 1.0e-4 | 3 | 3 | 0 | 3 | 4 / 7 | yes |
| 1.0e-2 | 8 | 5 | 4 | 9 | 9 / 15 | yes |
| 1.0e-1 | 12 | 5 | 7 | 12 | 13 / 18 | yes |
| 1.0e+0 | 15 | 6 | 11 | 17 | 16 / 24 | yes |
| 4.0e+1 | 21 | 5 | 15 | 20 | 22 / 26 | yes |

Backtracking by halving `α` from 1, accepting only when `‖G‖` decreases, keeps the update count in
`[3, 6]` over four orders of magnitude of `h`; the full count grows `3 → 21`. The tutorial's
criterion is met by the damped variant at every swept step and is violated by the undamped variant
from `h = 1e-1` upward. The two variants return the **same state** at every step to `1e-11`, so
damping changes the cost and not the answer — which is the point. The two cost columns are
reported separately (`directions` vs `trials`) because a backtrack costs extra residual
evaluations; a table counting only converged updates would make damping look free.

**The driver never needs it, and an earlier draft of this section said otherwise.** It claimed the
predictor was "a poor initial guess for large `h`". `ImplicitEuler._step` warm-starts Newton from
the free explicit-Euler predictor `y + h f(t_n, y)`, which is a *good* guess, and the claim was
never measured on that path. Measured, over `[0, 40]`:

| `h` | steps | variant | total updates | max per step | backtracks | RHS | `y2(40)` |
|---|---|---|---|---|---|---|---|
| 1e-2 | 4 000 | full step | 4 014 | 7 | 0 | 8 015 | 9.186892e-06 |
| 1e-2 | 4 000 | damped | 4 014 | 7 | 0 | 12 029 | 9.186892e-06 |
| 1e-3 | 40 000 | full step | 40 015 | 4 | 0 | 80 016 | 9.185671e-06 |
| 1e-3 | 40 000 | damped | 40 015 | 4 | 0 | 120 031 | 9.185671e-06 |

Zero backtracks in 44 000 steps, identical update totals, identical terminal value, and a factor
`1.50` in RHS for a guard that never fires. **The free predictor, not the damping, is what keeps
the driver's Newton monotone.** Damping is the right safeguard against a cold start — and the cold
start is exactly what the tutorial's rule is about. Entry 18 of the transparency log records the
correction.

The last column is the one to carry forward: a **successful solve says nothing about the accuracy
of the step it produced.**

### 6. The analytic Jacobian, verified independently (T17, F15)

`J` is used everywhere in this project and was checked nowhere before this pass. The check is
column-by-column forward differences —

```
(J)[:, j]  ≈  ( f(t, y + eps_j e_j) - f(t, y) ) / eps_j ,
eta = ||J_A - J_D||_inf / max(1, ||J_A||_inf)
```

— and the residual Jacobian `J_F = I - h J_f` is differenced as well, since that is what implicit
Euler actually solves with.

| state | `η` at `√ε_mach · max(1,|y_j|)` | `η` at the tuned `ε` | `η(J_F)` at its own tuned `ε` |
|---|---|---|---|
| initial | 4.470e-01 | 3.048e-05 | 3.006e-07 |
| transient | 2.503e-04 | 1.935e-08 | 1.077e-07 |
| end | 1.318e-04 | 9.250e-09 | 8.712e-08 |

Three things are established by measurement rather than asserted:

* **There is no universal threshold.** At `y(0)` the *correct* Jacobian scores `4.47e-01` under the
  rule-of-thumb perturbation; a gate at `1e-8` would reject it. The perturbation sweep has its
  minimum at `ε = 1e-12`, four orders of magnitude below the rule of thumb, and rises on **both**
  sides — cancellation to the left, truncation to the right. `η` is therefore always reported
  together with the state and the perturbation.
* **One state is not a check.** Deleting the whole `10⁴ y₂y₃` reaction from `J_f` and re-running
  gives a separation of **exactly 1.000** at `y(0)`: there `y₂ = y₃ = 0`, so the four entries the
  bug touches are zero in the correct matrix too and the two matrices are identical. The same bug
  separates by `1.2e4×` at the transient state and `9.1e7×` at the end state (tuned `ε`).
* **The check is cheap.** The whole protocol costs 132 right-hand-side evaluations.

### 7. A loose Newton tolerance does not lose accuracy — it replaces the method (T18, F14)

This is the finding that changed a statement rather than a number. Newton tests the residual before
taking any update, and the predictor's residual is

```
G(y_pred) = -h² J_f f + O(h³)
```

so the predictor is **accepted with zero Newton updates** — the step is literally an explicit-Euler
step — whenever `h ≤ h* = √(tol / ‖J_f f‖_∞)`. At `t = 1.0227`, `‖J_f f‖_∞ = 8.2324e-03`, so
`h* = 1.1e-02` at `tol = 1e-6`, while explicit Euler's measured stability bound on this problem is
`h < 5.895e-4`. A controller at `tol = 1e-6` would therefore accept steps **19× past the stability
limit of the method it is silently running**.

The estimate that is supposed to police the step does not see it:

| `h` | `E` at `tol = 1e-14` (updates `1,1,1`) | `E` at `tol = 1e-6` (updates `0,0,0`) | ratio |
|---|---|---|---|
| 1e-03 | 2.908e-09 | 2.910e-09 | 1.0007 |
| 5e-03 | 7.247e-08 | 7.274e-08 | 1.0037 |
| 1e-02 | 2.887e-07 | 2.909e-07 | 1.0076 |

Both Euler methods are first order, so `E ~ h²` either way — the fitted `h`-scaling exponent stays
inside `[1.89, 2.10]` across all seven tolerances, which is the width of the fit. The estimator
answers "how large is the local error of the method I ran" and answers it correctly; it cannot
answer "which method did I run". A first-order estimate cannot distinguish two first-order methods.

**And it still carries the solver's own error.** `E = ‖y_f − y_c‖` is a difference between two
sides, so if the coarse side and the half sides were solved to different algebraic accuracy the
mismatch lands directly in the estimate. Read the second grid of T18 downwards: wherever the update
triple is `(1,0,0)` the value is **exactly 3.00×** the neighbouring consistent rows, and wherever
the triple is consistent the values agree to the last printed digit. That is the tutorial's "the
Newton errors must be small enough not to dominate the difference", stated as a measurement.

Both cheap monitors miss the switch. Conservation is insensitive by construction (`(1,1,1)ᵀ` is a
left null vector of `J_f`, so one update restores the invariant exactly at any tolerance — and at a
loose tolerance there are no updates at all, so the predictor already conserves it). The local
error estimate is blind as above. Only a measured trajectory error against the reference sees it,
which is why the Newton tolerance is reported beside every result in this project.

The substitution threshold is a ladder in `tol`, and it moves slowly because `h* ∝ √tol`:

| `tol` | `h*` | `h*` ÷ explicit-Euler stability bound |
|---|---|---|
| 1e-14 | 1.10e-06 | 0.002 |
| 1e-12 | 1.10e-05 | 0.02 |
| 1e-10 | 1.10e-04 | 0.19 |
| **1e-08** | 1.10e-03 | **1.87** |
| 1e-07 | 3.49e-03 | 5.91 |
| **1e-06** | 1.10e-02 | **18.70** |
| 1e-04 | 1.10e-01 | 186.97 |

So tightening the tolerance does not remove the substitution, it only moves `h*` down — and buying
one decimal of safety costs **two** decimals of tolerance. There is no tolerance that makes
implicit Euler safe at `h ≫ 1e-3`; what keeps the step an implicit one is the *step size* staying
below `h*`. That is the accurate form of the section-3 recommendation: `tol = 1e-8` is the largest
tolerance that keeps the algebraic error negligible, and at `h = 1e-3` — the step the P3 sweep
uses — it is already running explicit Euler at a step slightly above the explicit stability bound.
`tol = 1e-6` does the same thing nineteen times over. Reporting the tolerance next to every result,
rather than assuming a tight one, is therefore not a formality.

---

## Figures

| file | axes | data | question it answers |
|---|---|---|---|
| `F2_reference_solution` | `t` (log) vs concentration | reference | chemical picture; B is short-lived; invariant holds at roundoff |
| `F4_stability_regions` | `Re(hλ)` vs `Im(hλ)` | `|R(z)|-1` + actual spectra | why explicit methods are step-limited |
| `F5_jacobian_spectrum` | `t` (log) vs spectrum | reference | structural zero; the eigenvalue collision |
| `F6_stiffness_ratio` | `t` (log) vs `S(t)` | reference | how stiffness grows; check values reproduced |
| `F7_frozen_bound_vs_sweep` | `h` vs error, `min_i y_i`, trajectories | 22-step sweep | is the local estimate trustworthy, and where do the three boundaries fall? |
| `F8_convergence` | `h` vs error (log-log), observed order | 5 halvings × 3 methods | observed order, its asymptotic window and its limits |
| `F9_four_track_diagnostics` | `h` vs mass defect / `min_i y_i` / error / `y2(40)` | 22-step sweep | do the four diagnostics agree? (no) |
| `F10_newton_tolerance_sweep` | Newton tol vs defect / error / iterations | 7 tolerances at `h=1e-3` | when is the algebraic solve free, and what breaks first? |
| `F11_adaptive_stepping` | `t` vs `h(t)`, `err/tol`, error vs tol | 4 scaled + 2 absolute runs | does the controller track the tolerance, and which norm lies? |
| `F12_work_precision` | cost vs error, `h` vs error | T7 ∩ T9, T9 keys | at *equal accuracy*, is the implicit method cheaper? (no) |
| `F13_two_clocks` | `t` (log) vs concentration, and vs `1/|Re λ|` | reference | why the problem is hard, stated *before* any method is introduced |
| `F14_newton_tolerance_switch` | `h` vs `E`, and update triple / `h*` vs `tol` | T18's two 7×7 grids | where does the Newton tolerance stop being an accuracy knob and start being a different method? |
| `F15_jacobian_check` | `ε` vs `η`, and wrong-vs-correct separation | T17's 3 states × 2 perturbations | is the analytic Jacobian right, and would this check have said so if it were not? |

Reserved for later stages: F1 (reaction network), F3 (time scales).

---

## Slides (beamer)

`slides/robertson_slides.tex` builds a 27-page 16:9 beamer deck (title + 26 frames,
fourteen of them appendix). Metropolis theme, Chinese body text with English method names,
`ctex` + the Fandol fonts so nothing depends on a system CJK font.

The deck is written for an audience meeting the Robertson problem for the first time,
so the main line is *not* ordered by computation stage. It runs:

> 要解决什么 → 答案长什么样 → 难在哪 → 怎么做 → 四条结论 → 总结

Twelve main frames carry that line (title page + 12). Everything technical that only
a reader who already knows the vocabulary would want — the spectrum and the structural
zero, `S(t)` against the brief's check values, the stability regions, the observed
orders, the four-track boundary table, the `y2(40)` digit counts, the full Newton
sweep, the tolerance switch, the Jacobian check, the damped-Newton cost table, the
adaptive norm failure, the file manifest, the Tutorial-2 summary — lives in the
fourteen appendix frames and is only shown on request. The first solution draft had
these interleaved, which is why it read as a table of contents rather than an argument;
see Entry 12 of the transparency log.

```sh
cd slides
tectonic -X compile robertson_slides.tex --outdir build   # writes build/robertson_slides.pdf
```

Notes on how the deck is kept honest:

* **No number in a frame is typed by hand from memory.** Every table cell and every
  quoted figure was copied from `results/tables/*.md` or `results/summary.json` after a
  successful `run_all.py`, and then re-checked cell by cell against those files.
  One draft row (an EE non-negativity row) did *not* survive that check — the values in
  it were not in any table, so it was replaced with rows taken from T5/T13 and logged.
  Entries 15–18 are the counterpart on the *code* side: the same discipline applied to a
  claim rather than a cell.
* Figures are included from `../results/figures/` as **PDF**, so the deck stays vector
  and the annotations inside F9–F15 remain readable when zoomed.
* The build is warning-clean apart from a 13.8 pt overfull `\vbox` in metropolis's own
  title-page template. Nothing is clipped on the rendered page; the box is simply taller
  than the text area. **No frame of ours needs a `[shrink]` key.** The rewrite removed the
  need for the original five: the frames that formerly needed `[shrink=8]` now fit
  structurally (two 4-column numeric tables were transposed, a full-width three-panel figure
  moved inside a column, the cost table normalised to a common `10^5` unit), the two
  `[shrink=2]` frames no longer exist as main frames, and the transparency-log frame was
  re-laid out so that its 18 entries fit as a full-width list under a two-column header.
  The three frames added in the Tutorial-2 pass were held to the same rule: the two figure
  frames put the figure *inside* a column rather than full-width above the text, and the
  four-entry nested list in the "what's new" frame was split out into a frame of its own
  rather than squeezed. Each of those three went through a `[shrink]`-free fix across two
  compile passes — 112 pt, 93 pt and 95 pt overfull down to zero.
* `slides/build/` is throwaway. The delivered file is `slides/robertson_slides.pdf`.
