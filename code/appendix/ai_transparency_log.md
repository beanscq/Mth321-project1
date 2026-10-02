# AI transparency log

Course requirement: record where AI assistance was used, and — more importantly —
where it was **wrong** and how the error was found. Each entry states the wrong claim,
how it was detected, and what the corrected statement is.

---

## Entry 1 — the reduced Jacobian was taken as a submatrix (found before any code was run)

**Claimed.** Eliminating `y3 = 1 - y1 - y2` gives the upper-left 2×2 block of `J`,
with `det = 0.04 · 6e7·y2 = 2.4e6·y2`.

**Consequence.** `S(40)` came out as `5.2e5` against the brief's check value `1.58e5`,
and `S(1e-2)` as `5.5e4` against `5.4e3`. This was reported as a suspected
*discrepancy in the brief*.

**How it was caught.** The elimination was redone properly. Substituting
`y3 = 1 - y1 - y2` and differentiating with respect to `(y1, y2)` is a projection,
`B_red = S·J·T` with `T = ∂y/∂z`, not a submatrix extraction. The determinant picks up
a second term:

```
det(B_red) = 2·k2·y2·(k1 + k3·y2) = 2.4e6·y2 + 6e11·y2^2
```

At `t = 40`, `y2 = 9.19e-6` so `k3·y2 = 0.092` is comparable with `k1 = 0.04` and the
extra term matters by a factor 3.3.

**Corrected statement.** All three brief check values reproduce to within 0.4%:
`3.0097e3` vs `3.0e3`, `5.4221e3` vs `5.4e3`, `1.5840e5` vs `1.58e5`. The brief was
right and the suspicion was unfounded.

**Why this matters beyond the arithmetic.** `|λ|max` depends only on `tr(J)`, which is
identical under both conventions, so `|λ|max(40) = 3.39e3` looked correct even while
the ratio was wrong. The ratio is the discriminating check, and the brief explicitly
says the check values "are check values, not substitutes for the student's own
computation". Reproducing **only** `|λ|max` would have hidden the error.

**Permanent guard.** `tests/test_model.py::test_det_reduced_equals_sum_of_principal_2x2_minors`
asserts `tr(J) == tr(B_red)` and `m2(J) == det(B_red)`; `eigen.consistency_check` runs
the same certificate over the whole trajectory (max relative deviation 1.8e-15).

---

## Entry 2 — the digit-agreement metric was defined so that it could not work

**Claimed.** "Digits retained by the tolerance repeat" computed as
`-log10( max_t |Δ_i(t)| / max_t|y_i(t)| )` with an **elementwise** scale,
`max(|a_i(t)|, |b_i(t)|)`.

**Consequence.** The metric returned **4.6 digits**, suggesting the two reference runs
disagreed badly, even though every method produced the same `y(40)` to ten digits.

**How it was caught.** `tests/test_reference.py` failed, and a direct diagnostic showed
the maximum **absolute** disagreement between the two runs was `2.1e-12` while
`y2` and `y3` start at exactly zero (`y2(0) = y3(0) = 0`) and pass through ~`1e-20`.
Dividing an absolute disagreement of `1e-12` by a value of `1e-20` produces nonsense.

**Corrected statement.** Two numbers, both reported:

* maximum absolute disagreement = `2.1e-12` (this is what limits any error measurement
  built on the reference);
* significant digits of the solution **profile**, i.e. scaling by the component's own
  magnitude over the trajectory → **10.8 digits**.

**Permanent guard.** `test_elementwise_scaling_of_the_digit_metric_is_a_trap` encodes
the failing configuration.

---

## Entry 3 — wrong polynomial for the RK4 real-axis stability limit

**Claimed.** Roots of `R(x) - 1` obtained by calling `np.roots` on the coefficients of
`R(x)` itself.

**Consequence.** No valid root found; `stability_region.real_axis_stability_interval`
raised. Had the roots simply been hard-coded, the interval would never have been derived
from the stability polynomial at all.

**Corrected statement.** `R(x) = 1 ⟺ x·(1 + x/2 + x²/6 + x³/24) = 0`, so the limit is
the negative real root of the **cubic** `x³ + 4x² + 12x + 24 = 0`, giving
`-2.7853`. This matches the textbook value and yields the factor `2.7853/2 = 1.39`
advantage of RK4 over explicit Euler.

**Permanent guard.** `test_stability_region_real_axis_intervals` and
`test_rk4_does_not_escape_the_stiffness_restriction`.

---

## Entry 4 — the "fixed grid resolves the eigenvalue collision" assumption

**Claimed.** Plotting `Im λ(t)` on the standard 200-point comparison grid shows the
conjugate-pair window.

**Consequence.** The figure was a flat line: the window spans ~5% in `t` while consecutive
grid points are 12% apart. `run_all.py` reported **0** grid states with negative
discriminant, which initially reads like "the window does not exist".

**Corrected statement.** The window is real:
`t ∈ [1.630e-08, 1.705e-08]`, 9 states on a dedicated dense early-time reference,
`|Im λ|/|Re λ| ≤ 1.29e-2`. A separate `reference.zoom_reference()` trajectory is used
for the figure, it is *not* part of the comparison grid, and F5c says so in its title.
The analytic prediction (root of `0.99933·q² - 0.08·q + 0.0016` with `q = 6e7·y2`) is
tested directly in `test_eigen.py`.

**Methodological note.** This is a case where a coarse grid makes a real feature
invisible, and "not observed" must not be reported as "does not exist".

---

## Entry 5 — the uniform-step driver silently took the whole span in its first step

**Claimed / written.** Inside the step loop, the step length was computed as
`hh = t_end - t` (the *remaining* span) instead of the requested step size.

**Consequence.** The very first step of every uniform run was of length `t_end = 40`
(or `4`, or `1` — whatever the span was), and only the second step onward used `h`. The
bug is invisible for a method that is stable at large steps and catastrophic otherwise:
explicit Euler and RK4 immediately produced `|y| ~ 1e11` and `|y| ~ 5e6` and the run
"diverged" at `t = 0.01` for *every* step size, including step sizes that should have been
perfectly stable.

**How it was caught.** `tests/test_integrators.py` asserted the textbook cost identity
`rhs_calls == n_steps + 1` for explicit Euler and failed with `5 == 3 + 1`; the traceback
also showed the trajectory stopping at `t = 0.03` for a run with `t_end = 1.0`. The
related symptoms — a divided-by-zero warning in the Hermite interpolant (it saw duplicated
node times) and an RK4 "order of convergence" run that produced garbage — pointed at the
same place.

**Corrected statement.** Node times are formed as `t0 + n·h_used` rather than accumulated,
and the final node is pinned to `t_end`, so the node array is strictly increasing and
cannot contain a duplicate time. The interpolation routine additionally returns `nan` for
a run with fewer than two nodes instead of clipping to an empty index range.

**Permanent guard.** `test_the_step_size_lands_exactly_on_the_end_time`,
`test_cost_counters_are_consistent_with_the_expected_per_step_counts`,
`test_dense_output_at_the_nodes_reproduces_the_stored_states` and
`test_explicit_euler_diverges_on_the_stiff_span`.

---

## Entry 6 — the brief's statement about inexact Newton solves does not transfer quantitatively to this system

**Claimed.** Following the brief's remark that "an inexact nonlinear solve produces an
invariant defect controlled by the nonlinear residual", a Newton-tolerance sweep was
planned on the assumption that the measured invariant defect would grow roughly linearly
with the stopping tolerance.

**Consequence.** The prediction is *algebraically true but quantitatively empty here*.
`(1,1,1)^T` is a **left null vector** of `J`, therefore

```
(1,1,1)^T (I - hJ)^{-1} = (1,1,1)^T
```

and one Newton update maps any iterate to `sum(z) = sum(y_n)` **exactly**, for any
tolerance. Measured: the invariant defect is `~1e-14` for *every* run in the P2 tables,
including the runs that diverge and go strongly negative at `h = 3e-3`. A tolerance sweep
built on this metric would have produced a flat line and the wrong conclusion
("Newton tolerance does not matter").

**Corrected statement.** Both statements are reported, and they are different:

* the *bound* `|invariant defect| ≤ 3·||G||_∞` holds (asserted in
  `test_newton.py::test_residual_sum_equals_the_linear_invariant_defect_of_one_step`);
* the *actual* defect is governed by roundoff, not by the tolerance, because the Newton
  correction lies in the invariant-preserving subspace. So the P3 tolerance sweep must be
  judged on the **solution error** and on the Newton iteration count, not on the invariant
  defect, and panel F9 must plot them as separate curves.

**Permanent guard.** `test_newton.py::test_one_newton_step_restores_the_linear_invariant_exactly`
pins down the exact invariance, and asserts that the residual *after* that step is still
non-negligible — the two facts together are the point.

---

## Entry 7 — the empirical stability boundary was expected to sit exactly on the frozen bound

**Claimed.** Before running anything, the plan was to report "the empirical boundary equals
`2/|λ|max`", treating the frozen-Jacobian estimate as a prediction to be confirmed.

**Consequence.** It is bracketed, not confirmed. On a 22-point logarithmic sweep:
explicit Euler is stable at `h = 5.4159e-04` and diverges at `h = 6.2464e-04` against a
prediction of `5.8949e-04`; RK4 is stable at `8.3088e-04` and diverges at `9.5827e-04`
against a prediction of `8.2095e-04`. RK4 survives 1.2% **beyond** its bound, explicit
Euler fails 8% **short** of it.

**Corrected statement.** The estimate is reported as a *local diagnostic* bracketed by the
two adjacent sweep points, and the residual disagreement is attributed to what it is: the
bound is frozen-linear-local, `J(t)` varies, and the system is strongly non-normal (the
linear invariant forces a structural zero eigenvalue), so eigenvalue real parts do not
control transient growth.

**A second, unplanned result from the same sweep.** The three "boundaries" do **not**
coincide. For explicit Euler, stability / non-negativity / accuracy all fail at the same
`h`, because the unstable mode alternates in sign and produces negative concentrations.
For RK4 they separate: accuracy fails at `7.2041e-04`, stability only at `8.3088e-04`.
That window — stable and non-negative but inaccurate — is the concrete content behind
"it did not blow up is not an accuracy claim", and it would have been missed by a
single-threshold sweep.

**Permanent guard.** `test_sweep_finds_the_stability_boundary_where_the_frozen_bound_says`
brackets the prediction with two step sizes chosen from the bound itself, and
`T6_sweep_boundaries.md` reports the three boundaries as three separate columns.

---

## Entry 8 — the Newton tolerance was assumed to be a free knob that can always be tightened

**Claimed.** The first plan for the implicit solver treated the Newton stopping tolerance as
a pure cost/accuracy dial: pick it as small as the budget allows and the implicit step gets
as accurate as we want. The predicted risk was only "too loose breaks the invariant".

**Consequence.** The stopping test is applied to the **predictor** residual, and that residual
is `O(h^2)`. At `h = 1e-3` it is ~`1e-6`, so any tolerance at or above `1e-6` is satisfied
*before Newton does anything*: the loop exits with **zero updates** and implicit Euler
silently degenerates to explicit Euler — same under-resolution, but still paying the implicit
bookkeeping. Measured in `T11_newton_sweep.md` at `h = 1e-3`:

* `tol = 1e-14` … `1e-10`: solution error sits on a plateau of `7.3549e-06` — this is the
  **discretisation** floor, and tightening Newton further changes nothing (the invariant
  defect is already `~1e-14` at every point of the sweep);
* `tol = 1e-2`: 60.0% of accepted steps perform no Newton update at all, and the error
  inflates by a factor `1.96e7` (`1.44e2` vs the plateau).

**Corrected statement.** The tolerance must be set *just left of the break*, not at machine
zero. The break is predicted by the predictor residual (`O(h^2)`), so it tracks `h`; at
`h = 1e-3` the reported recommendation is `tol = 1e-8` (error inflation `0.88`, i.e. no
penalty, at `0.31` Newton updates per accepted step) and the first tolerance that breaks it
is `1e-6` (inflation `2.67`). Panel F10 shows the two facts side by side: the **invariant
defect is flat** for every tolerance (Entry 6) while the **solution error** is not.

**Permanent guard.** `tests/test_analysis_p3.py` pins the `O(h^2)` predictor-residual
scaling and asserts that the recommended tolerance is strictly tighter than the first
breaking one; `T11b_newton_sweep_summary.md` reports the plateau, the recommendation and
the break as three separate quantities.

---

## Entry 9 — the absolute L-infinity norm cannot drive an adaptive controller on this problem

**Claimed.** A standard step-doubling controller was first built with the plain absolute
`L-infinity` norm of the local error estimate, `max_i |e_i|`, on the reasoning that a single
global tolerance is simpler and easier to explain.

**Consequence.** It misses the stiff mode entirely and produces **physical nonsense while
looking stable**. `y_2` is `O(1e-5)`, so its error scale is `~1e-10`; an absolute bound of
`1e-6` therefore accepts steps far beyond the RK4 real-axis stability limit. Measured in
`T12_adaptive.md`:

* `tol = 1e-6` (absolute): no NaN, but `y_2` reaches `-4.77e-05` and the full-state error is
  `7.83e-03` — a run that is numerically "fine" and physically impossible;
* `tol = 1e-4` (absolute): the trajectory is consumed by the instability and **aborts at
  `t = 1.9`** with `min y = -9.03e2`.

**Corrected statement.** The working norm is **componentwise-scaled (mixed)**,
`max_i |e_i| / (atol_i + |y_i|)`, with a per-component absolute floor. The absolute-norm
controller is **kept in the deliverables as a deliberately failing control experiment**, not
deleted: `T12` lists both norms and F11c plots the absolute-norm global error running ~`1e6`
above the mixed one before it aborts. This is the honest version of "the norm matters".

**Permanent guard.** `tests/test_adaptive.py::test_absolute_norm_controller_misses_the_stiff_mode`
asserts the absolute-norm run is non-physical (negative `y_2` and/or early abort) while the
mixed-norm run at the same tolerance stays non-negative and accurate.

---

## Entry 10 — the linear invariant is preserved exactly by the implicit step, so it must be dropped as an accuracy monitor

**Claimed.** The project brief asks to "check the conservation law"; the natural
simplification was to use `|sum(y) - 1|` as the single pass/fail quality gate, on the
assumption that a run which conserves is a run you can trust.

**Consequence.** Conservation certifies nothing here. Because `(1,1,1)^T` is a left null
vector of `J` (Entry 6), the implicit step preserves the sum to roundoff **for every run**.
Measured in `T13_four_track.md`: the worst track-1 case shown is `1.1e-13` on a run with
`min y = -921.8` and full-state error `1.08e10`. A conservation-only gate would have passed
garbage, and it is flat across the whole step-size sweep.

**Corrected statement.** Four diagnostics are kept in **separate columns** and never merged:
(1) conservation, (2) non-negativity, (3) full-state error against our own reference,
(4) the terminal value `y_2(40)` quoted with its number of confirmed digits. `T14` reports
**which track fails first** per method — and the answer differs:

* RK4: **accuracy first** at `h = 7.2041e-04`, then non-negativity / stability together at
  `8.3088e-04` (the window is Entry 7's "stable but inaccurate");
* explicit Euler: non-negativity and accuracy fail **together** at `5.4159e-04`;
* implicit Euler: none of the four tracks fail anywhere in the swept range up to
  `h = 3e-3`, so `T14` reports the boundary as **not found within the sweep** rather than
  inventing a number.

**Permanent guard.** `tests/test_analysis_p3.py` asserts the four tracks are returned as
distinct values (not collapsed), that the conservation track is flat while the error track
is not, and that a non-converged run is reported as "boundary not reached" instead of being
assigned the largest swept step.

---

## Entry 11 — a slide row quoted diagnostics that were not in any table

**Claimed.** A draft frame of the beamer deck (`四轨诊断：那个"稳定但不准"的窗口`) carried a
three-row summary table. The third row read
`h = 1.5e-3, EE, mass defect 4e-15, min_i y_i = -13.7, error 4.8e-2`, and the second row
gave the RK4 mass defect as `2e-14`.

**Consequence.** Neither row could be reproduced. `min_i y_i = -13.7` appears **nowhere** in
`T5_step_sweep.md` or `T13_four_track.md`; the EE runs near `h = 1.5e-3` give
`-67.3` (`h = 1.47e-3`) and `-10.4` (`h = 1.70e-3`). The RK4 mass defect at `8.3088e-4` is
`6.2e-15`, not `2e-14` — `2e-14` is the EE value at `h = 1.5e-4`, i.e. a value copied from a
different row. The `4.8e-2` error also matched no run (the nearest, `4.51e-2`, belongs to
EE at `h = 1.105e-3`). Nothing in the deck's *argument* depended on these numbers — which is
exactly what makes the error dangerous: it would have shipped.

**How it was caught.** Every table cell in the deck was checked against
`results/tables/*.md` and `results/summary.json` after the last successful `run_all.py`,
cell by cell, before the deck was presented.

**Corrected statement.** The row was replaced with three rows taken directly from
`T13_four_track.md`:

| `h` | method | mass defect | `min_i y_i` | error | verdict |
|---|---|---|---|---|---|
| `1.50e-4` | all three | `1–2e-14` | `0.0` | `4.6e-10 … 1.2e-6` | all four tracks pass |
| `8.31e-4` | RK4 | `6.2e-15` | `0.0` | **`1.24e-4`** | accuracy fails |
| `1.27e-3` | EE | `2.4e-15` | **`-42.9`** | **`2.3e-2`** | only conservation survives |

The middle row is untouched in substance and is still the point of the frame: a run that is
stable, non-negative and exactly conservative, and still wrong.

**Permanent guard.** The deck's header comment states that it reads only `results/` and that
no number in it is hand-written from memory. Because a slide has no test suite, the guard is
procedural: the table is regenerated by copying cells from `T13_four_track.md`, and any
future edit to that frame has to redo the same comparison.

---

## Entry 12 — the first deck was ordered by computation stage, so it explained nothing

**Claimed.** The first complete deck (21 pages) was correct and followed the order in which the
work was done: outline → problem → reference → spectrum → stability → frozen bound →
convergence → four-track → terminal value → Newton → adaptive → cost → conclusion. Every number
in it traced to a table, and every frame had been checked cell by cell (Entry 11).

**Consequence.** Correct and unreadable. The reviewer's report was that the deck "讲的一点也不清楚".
The question being answered, the method, and the answer were all present, but never stated in
that order. Concretely:

* the outline frame listed section titles (`谱结构 / 稳定域与冻结界 / 四轨诊断 / 匹配精度成本`),
  which is a table of contents, not a statement of the problem;
* the `五个待答问题` frame asked about `S(t)`, the frozen bound and the observed order — sub-tasks
  whose *motivation* had not been given yet;
* the answer to each question was buried in its frame title (`冻结界 vs 实测崩溃点：被夹住，不是
  被证实`), with no "so what" sentence anywhere.

Nothing in the report was wrong. The defect was purely one of exposition, and it is the kind that
survives every correctness guard this project has: the numbers all traced, the build was
warning-clean, and every frame fit. A per-cell audit cannot detect a bad argument order.

**How it was caught.** Only by reading the deck as a listener rather than as its author. No script
or test in this repository can catch it. The check that works is to state the argument in prose,
with no frame in front of you, and see whether the sentences connect.

**Corrected statement.** The main line was rewritten to run
*要解决什么 → 答案长什么样 → 难在哪 → 怎么做 → 四条结论 → 总结*, in twelve frames. Frames were
reordered, not merely relabelled: the reference solution now appears *before* the difficulty,
each of the four conclusions carries an explicit verdict sentence, and everything only a reader
who already knows the vocabulary would want — spectrum, stability regions, observed orders, the
boundary table, the digit counts, the Newton sweep, the adaptive failure, the file manifest —
moved into ten appendix frames. One new figure, `F13_two_clocks`, was built for the `难在哪`
frame so stiffness is stated before any method is introduced; it carries no new computation, it
re-plots `1/|Re λ_fast|` and `1/|Re λ_slow|` from the existing reference solution.

**Permanent guard.** The README's slides section records the main line and the rule that technical
frames belong in the appendix, so a later edit that re-interleaves them is a visible deviation
rather than a silent regression.

---

## Entry 13 — the same bad row was still in the README, one stage later

**Claimed.** `README.md`, in the section `1. The four diagnostics disagree`, carried a three-row
table summarising `T13_four_track.md`, introduced by the sentence "Measured on the `h`-sweep
(T13)". Its third row read `h = 1.5e-3, Explicit Euler, mass defect 4e-15, min_i y_i = -13.7,
error 4.8e-2`, and its second row gave the RK4 mass defect at `8.31e-4` as `2e-14`.

**Consequence.** Identical to Entry 11, and for the same reason: `1.5e-3` is not a point on the
sweep grid (`T13` runs `1.4701e-3` and then `1.6955e-3`), `-13.7` appears in no sweep table, and
the RK4 defect at `8.3088e-4` is `6.2e-15`. The first row's stated spread `1e-14 … 2e-13` also
overstates the measured one, which is `9.5e-15 … 2.2e-14`.

**How it was caught.** While translating the README into Chinese. Entry 11 fixed the **slide** and
logged the error, but the README was the *source* of that slide row: the correction had been
applied downstream only. Fixing one place where a wrong number is *reported* is not the same as
fixing every place it was *copied to*.

**Corrected statement.** The three rows now read, verbatim from `T13_four_track.md`:

| `h` | method | mass defect | `min_i y_i` | error | verdict |
|---|---|---|---|---|---|
| `1.50e-4` | all three | `9.5e-15 … 2.2e-14` | `0.0` | `4.6e-10 … 1.2e-6` | all four tracks pass |
| `8.31e-4` | RK4 | `6.2e-15` | `0.0` | **`1.24e-4`** | accuracy fails |
| `1.27e-3` | EE | `2.4e-15` | **`-42.9`** | **`2.3e-2`** | only conservation survives |

**Permanent guard.** Entry 11's guard was procedural and scoped to the deck. It should have been
scoped to the *value*: when a hand-transcribed number turns out to be unreproducible, grep the
whole repository for it instead of fixing only the occurrence that was noticed —

```sh
grep -rn '13\.7' --include='*.md' --include='*.tex' .
```

The check is cheap, and the failure mode — one bad row propagating into every document that
quotes it — is exactly what this log exists to prevent.

---

## Entry 14 — three more hand-quoted cells, found by re-checking the README against its own tables

**Claimed.** `README.md`, section `4. Adaptive step size`, carries a three-row table describing
the failure of the plain absolute `L∞` norm, introduced as the runs tabulated in `T12_adaptive.md`.
Its last column gave the mass defect of each run as `1.9e-15` (absolute, `tol = 1e-4`), `2.2e-14`
(absolute, `tol = 1e-6`) and `2.1e-15` (scaled, `tol = 1e-6`).

**Consequence.** Those three cells do not match the table. The measured values are `1.136868e-13`,
`7.771561e-15` and `5.773160e-15`. Every *other* cell of the table matched to the digit
(`−903` vs `-9.030743e+02`, `2.17e-03` vs `2.168641e-03`, `−4.77e-05` vs `-4.766346e-05`,
`9.93e-09` vs `9.930489e-09`), which is what establishes that the README is quoting these three
runs and not three others. The argument survives the correction — all three defects are `≤1e-13`,
so the point "the invariant defect reports nothing wrong in either norm" is unchanged — but the
numbers were not reproducible from the file the sentence cites.

**How it was caught.** Not by a test. Found while translating the README into Chinese, by walking
every table cell in the document back to `results/`. Three documents had now been checked this way
(the deck in Entry 11, the README in Entries 13 and 14) and each pass found at least one cell that
did not reproduce.

**Corrected statement.**

| norm | tol | reached | `min_i y_i` | global error | mass defect |
|---|---|---|---|---|---|
| absolute `L∞` | `1e-4` | `t = 1.9` only | `−903` | `2.17e-03` | `1.1e-13` |
| absolute `L∞` | `1e-6` | `40` | `−4.77e-05` | `7.83e-03` | `7.8e-15` |
| scaled | `1e-6` | `40` | `0.0` | `9.93e-09` | `5.8e-15` |

**One apparent discrepancy that turned out to be correct.** The same pass flagged the claim in
`P2 results` that explicit Euler's "first divergence" is at `t = 9.8e-3`, because no column of
`T5_step_sweep.md` or `T13_four_track.md` contains it — the earliest `t_first_negative` among the
diverging runs is `32.9` (at `h = 6.2464e-4`). It is nevertheless right: `9.8e-3` is
`step_sweep`'s `t_first_divergence`, the **minimum `abort_t` over all runs that aborted**
(`0.009777082518576457` in `summary.json`), a quantity that is computed and stored but never
printed as its own table column, and that is *not* the same thing as the first negative component
of the first diverging run. So the lesson cuts both ways: a number that reproduces nowhere in the
tables is a defect (Entries 11–14), but a number that reproduces nowhere *in the columns you
happened to look at* is not necessarily a defect. Chase the definition before "fixing" it.

**Permanent guard.** The re-check is mechanical and cheap — walk each number in a prose document
back to its source table — and it has now found a defect in every document it has been applied to.
It should be run over the LaTeX report as well before that document is called done, and the
per-document counts recorded here are the evidence that it is worth the time.

---

## Entry 15 — the damped-Newton tests were written against the wrong system, and the Armijo test was wrong

**Context.** Tutorial 2 asks for full Newton *and* a damped variant, with the acceptance test for a
backtracking step stated as a sufficient decrease of the merit function. Adding the variant was
mechanical; the interesting part is that the first version of it was wrong in two independent ways
and the test suite initially *agreed* with it.

**Defect 1 — the test system hid the failure.** The first damping tests used the scalar
`g(z) = z³ − 2z + 2` from `z₀ = 0`, the usual textbook illustration, and asserted that damped
Newton converges where undamped Newton does not. It does not. Undamped Newton on that polynomial
does not diverge: it **cycles** between `0` and `1` (`z ← z − g/g′` maps `0 ↦ 1 ↦ 0`), and a
cycling iteration is not detected by an "iteration count exceeded" criterion that also has to
cover genuine divergence. The assertion passed only because of Defect 2.

**Defect 2 — the Armijo constant was off by a factor of two.** With merit `φ(z) = ½‖G(z)‖²` the
sufficient-decrease test is

```
‖G(z + α d)‖²  ≤  (1 − 2 c α) ‖G(z)‖² ,     c ∈ (0, ½)
```

and the implementation used `(1 − c α)`. A weaker test accepts steps that are not actually a
decrease of the merit function, and combined with `α_min = 1e-8` it accepted a *non-root* iterate:
shrinking `α` by eight orders of magnitude makes `‖G‖` small for reasons that have nothing to do
with being near a solution. The two defects together produced a green test asserting something
false.

**How it was caught.** By asking what the test would do if the implementation were replaced by
something known to be wrong. Cycling and divergence are different, and the acceptance test can be
checked on paper: differentiating `½‖G‖²` along the Newton direction gives the factor `2`, so the
constant `c α` is not available.

**Corrected statement.** Three changes, all in the code rather than in the prose:

* `α_min` default raised from `1e-8` to `1e-6`, so a backtrack that has to shrink by more than six
  orders of magnitude is reported as `FAIL_DAMPING` instead of being accepted;
* the Armijo test rewritten to `(1 − 2 c α)` with `c ∈ (0, ½)`, and `c` exposed as `armijo_c` so
  that the "stricter constant costs more trials, same root" statement can be *measured*;
* the damping tests moved off the textbook cubic and onto **this problem's** first implicit-Euler
  step from `y(0)`, which is the system the claim is actually about. On that system the honest
  statement is the one in `T16`: damping changes the cost, not the answer — the same root at every
  `h` to `1e-11`, `3–21` directions without it, `3–6` with it, and the tutorial's "at most 10
  updates" criterion met at every swept step.

**Consequence for the deliverable.** `T16_newton_damping.md` reports `dirs` and `trials` in
separate columns. A backtracking step costs extra residual evaluations, and a table that counted
only converged updates would make damping look free.

---

## Entry 16 — two bugs in the Jacobian checker, one of which made it reject a correct matrix

**Claimed.** The tutorial's rule is that the analytic Jacobian must be verified against an
independent difference approximation, at more than one state, perturbing the component that
controls the column.

**Defect 1 — the difference matrix was built transposed.** The first implementation perturbed
input component `j` and wrote the result into **row** `j`, producing `J_Dᵀ`. At the initial state
`J` is nearly triangular with one large diagonal entry, so a transposed approximation still scores
a *plausible-looking* `η` there, and the check passed. The convention is easy to state and easy to
get backwards: perturbing `y_j` produces `column j` of `J`, as
`(J)[:, j] ≈ (f(y + ε e_j) − f(y)) / ε`.

**Defect 2 — one tuned perturbation was reused for two different functions.** The perturbed
system needs `J_F = I − h J_f`, not `J_f`, and the checker differenced both. The perturbation that
minimises `η` is a property of **the function being differenced**, and `F(z) = z − y_n − h f(z)`
is not `f`. Reusing `f`'s tuned value gave `η(J_F, J_D) = 0.39` on a matrix that is exactly
right — so the *checker* failed a *correct* Jacobian, which is the more dangerous of the two
failure modes because the natural response is to "fix" the Jacobian.

**How it was caught.** Defect 1 by testing the checker against a deliberately transposed matrix
and requiring detection, at a state and perturbation where detection is possible. Defect 2 by a
test asserting `η` for `J_F` is small; when it came back at 0.39 the arithmetic was checked first
(the matrix was right) and the shared perturbation second.

**Corrected statement.** `jacobian_check.py` now tunes the perturbation per function
(`state_eta_tuned` for `J_f`, `state_eta_residual_tuned` and `residual_tuned_eps` for `J_F`), and
the reported numbers are in `T17`:

| state | `η` @ rule of thumb | `η` @ tuned | `η(J_F)` @ its own tuned value |
|---|---|---|---|
| initial | `4.470e-01` | `3.048e-05` | `3.006e-07` |
| transient | `2.503e-04` | `1.935e-08` | `1.077e-07` |
| end | `1.318e-04` | `9.250e-09` | `8.712e-08` |

**The part of this that matters more than the bugs.** The table above is the argument for the
tutorial's "no universal absolute requirement". With the rule-of-thumb perturbation
`√ε_mach · max(1, |y_j|)`, the **correct** Jacobian at `y(0)` scores `4.47e-01` — a gate at `1e-8`
rejects it. And the sweep in `T17` has its minimum at `ε = 1e-12`, four orders of magnitude below
the rule-of-thumb value, rising on both sides: cancellation to the left, truncation to the right.
A check that reports `η` without the state and the perturbation is not a check.

**And the check's blind spot, demonstrated rather than described.** Deleting the whole `10⁴ y₂y₃`
reaction from `J_f` and re-running gives a separation of **exactly 1.000** at `y(0)`: there
`y₂ = y₃ = 0`, so the four entries the bug touches are zero in the correct matrix too, and the two
matrices are identical. **A single-state check at the initial condition certifies a wrong
Jacobian.** At the end state, with a tuned perturbation, the same bug separates by `9.1e+07`.

---

## Entry 17 — "a loose Newton tolerance only loses accuracy" was false, and the estimate that should have caught it cannot

**Claimed.** Earlier in this project (P3, section 3, and Entry 8) the Newton stopping tolerance was
treated as an accuracy knob: too loose and the algebraic error is large, tight enough and it stops
mattering. That is true, and it was measured. But it is not the whole effect, and the missing part
was found by taking Tutorial 2's separation of *stability*, *Newton convergence* and *time
accuracy* at face value.

**What is actually true.** `ImplicitEuler` starts from the explicit-Euler predictor
`y_pred = y_n + h f(y_n)`, and Newton tests the residual before taking any update. Expanding,

```
G(y_pred) = y_pred − y_n − h f(y_pred) = −h² J_f f + O(h³)
```

so the predictor is accepted — **zero Newton updates**, and the step is literally an explicit-Euler
step — whenever

```
h  ≤  h* = sqrt( tol / ‖J_f f‖_∞ ) .
```

At `t = 1.0227`, `‖J_f f‖_∞ = 8.2324e-03`, so `h* = 1.1e-02` at `tol = 1e-6`. Explicit Euler's
measured stability bound on this problem is `h < 5.895e-4`. The tolerance therefore does not
degrade a stable method — past a computable threshold it **substitutes an unstable one**, and the
controller would accept steps nineteen times the stability limit.

**The estimate does not flag it.** Both Euler methods are first order, so `E = ‖y_f − y_c‖` keeps
its `O(h²)` shape either way: the fitted `h`-scaling exponent is `2.00` in every regime, and at
`h = 10⁻³` a fully explicit trial and a fully Newton-corrected trial agree to `0.07%` (`2.910e-11`
against `2.908e-11`). The estimator answers "how large is the local error of the method I ran" and
it answers that correctly. It cannot answer "which method did I run" — both of these are the same
method to first order. A first-order error estimate cannot distinguish two first-order methods.

**Defect found while measuring it.** The first version of the contamination test asserted
`max(E)/min(E) > 1.5` across the whole `h`-sweep, and it failed: at `h = 1e-4` every row took the
same update pattern, so the spread was **exactly 1.0**. The premise was wrong, not the assertion —
contamination is a property of the *update pattern*, not of `h`. Regrouping the rows by
`(h, iters_each)` showed the real structure, which is sharper than the original hypothesis:

* within a group — rows with the same coarse/half/half update triple — the spread is **exactly
  1.000000**, i.e. the estimate is reproducible to the printed digit;
* the outlying rows are **exactly** the ones whose triple is `(1,0,0)` — the coarse side corrected
  once, the half sides not corrected at all — and they are exactly **3.00×** the group value.

That is the tutorial's "the Newton errors must be small enough not to dominate the difference"
stated as a measurement instead of a warning: `E` is a difference between two sides, so if the two
sides were solved to different algebraic accuracy the mismatch lands directly in the estimate.

**Why both cheap monitors miss it.** Conservation is unaffected by construction — `(1,1,1)ᵀ` is a
left null vector of `J_f`, so one update restores the invariant exactly at *any* tolerance, and at
a loose tolerance there are no updates at all, so the predictor already conserves it. The local
error estimate is blind as above. A measured trajectory error against the reference is the only one
of the three monitors that sees the change, which is why the Newton tolerance is reported beside
every result in this project rather than assumed.

**Corrected statement.** `T18_newton_tolerance_vs_estimator.md` carries two grids, because neither
one alone is convincing: the estimate itself, and the update triple behind each cell. The
recommendation in section 3 (`tol = 1e-8`) is unchanged and is now *better* supported — `1e-8` is
also inside the stable regime at the step sizes the study uses, while `1e-6` is not.

---

## Entry 18 — T16's conclusion was true of the protocol it measured and false of the one the driver uses

**What was claimed.** T16 measures a single implicit-Euler step from `y(0)` starting Newton at the
*old state*, and finds that backtracking holds the update count in `[3, 6]` while the full step grows
`3 → 21`. Section 5 of the README then explained the result with a sentence about the driver:
"Implicit Euler starts from the explicit-Euler predictor, and for large `h` that predictor is a poor
initial guess".

**What is actually true.** That sentence is the opposite of the measurement. `ImplicitEuler._step`
warm-starts Newton from `y_pred = y + h f(t_n, y)` — the free explicit-Euler predictor — and that
guess is *good* enough that damping is never needed. Repeating the comparison on the driver's own
path over `[0, 40]`: `0` backtracks in 44 000 steps at two step sizes, update totals identical
(`4014` vs `4014` at `h = 1e-2`, `40 015` vs `40 015` at `h = 1e-3`), terminal value identical to the
last digit, and the damped variant pays `12 029` rhs against `8 015` — a factor `1.50` for a guard
that never fires. The free predictor, not the damping, is what keeps the driver's Newton monotone.

**Why the error survived.** T16's table is correct and its protocol is stated on line 3 of the file.
What went wrong is the step *from the table to the prose*: the measured protocol (cold start) and the
system's protocol (predictor start) are different experiments, and the second one had never been run.
A table can be right while the sentence explaining it describes something else.

**What changed.** `damping_table` now runs both protocols and emits both tables, so the distinction
cannot be lost again — the second table is generated, not narrated. Section 5 of both READMEs was
rewritten and the overclaim named rather than quietly deleted. Entry 15 is the same class of mistake
(a test written against a system other than the one under study), which suggests the failure mode to
watch for here is **measuring a stand-in and writing prose about the original**.

**Not a defect in the code.** `damping` is correctly implemented and correctly tested; the finding is
about where it is worth paying for. It stays available, stays off by default, and the price of
turning it on is now measured: `+50%` rhs evaluations on this problem.

---

## Smaller defects caught by the test suite in P2

* `zip(*ser)` applied to a dictionary iterates over its *keys*, so grouping sweep results
  by method unpacked a string. Two figure panels failed with
  `ValueError: too many values to unpack`. Fixed to `zip(*ser[name])`.
* `bounds[key]` indexed the frozen-bound dictionary by display name (`"Explicit Euler"`)
  instead of by method key. Fixed by building an explicit name → bound map.
* `\ge` is not a matplotlib mathtext command (`\geq` is); the F8 panel title raised
  `ParseFatalException: Unknown symbol: \ge`.
* The Newton loop originally ran `maxiter + 1` linear solves for `maxiter = 4`, so
  `test_unreachable_tolerance_is_reported_not_hidden` failed with `iters == 5`.

---

## Smaller defects caught by the test suite in P3

* **The sub-step ratio was computed with the wrong factor.** The first version reported
  `median(h_controller) / h_frozen` and gave a ratio near `2.1`, which read as "the
  controller overshoots the stability bound by 110%" — a false alarm. The accepted solution
  is propagated with **two** half-steps, so the step that actually has to be stable is
  `h/2`, not `h`. Renamed to `h_substep_median_over_frozen = 0.5·median(h)/h_frozen`, which
  puts the plateau at `1.06–1.07`. The column name now states the factor rather than hiding
  it, because getting this wrong flips the conclusion from "stability-limited" to
  "unstable".
* **The adaptive driver probed unstable step sizes on purpose and numpy complained.** While
  the controller is still shrinking `h` it evaluates `y_2^2` at values that overflow, and
  `matplotlib`/`numpy` emitted `RuntimeWarning: overflow encountered` in the thousands. The
  three trial steps in `StepDoublingAdaptive` are now wrapped in
  `with np.errstate(all="ignore")`. The warnings were cosmetic — the rejected attempt is
  discarded — but a run that prints thousands of warnings trains the reader to ignore them.
* **F11b was illegible.** The first version plotted the per-node `err/tol` ratio for all
  four tolerances as raw lines; with ~20 000 nodes per run the panel was a solid block. It
  now plots the median with a 5th–95th percentile band over 60 logarithmic time bins
  (`_binned_stats`), which is what made the `err/tol ≈ 0.17–0.59` plateau visible at all.
* **Annotation collisions in F9/F10.** Arrow annotations placed at fixed axes-fraction
  positions landed on top of the data. They were relocated to the free corners with an
  opaque white `bbox`, so they stay readable when the data moves (F9a, F10a, F10b).

---

## Smaller defects caught while building the cost model and the slides (P4)

* **F12's annotations collided with the data and with each other.** The first work–precision
  panel put an arrow annotation at a fixed axes fraction, which landed on the curve it was
  pointing at, and a second annotation covering the legend. Both were moved to free corners
  with opaque white `bbox`es, and the two conclusions that answer *different* questions
  ("EE is cheapest at exactly `1e-6`" vs "RK4 gets `8.9e-8` for the same money") were split
  into one annotation each instead of being merged into a single misleading claim.
* **T15 silently dropped RK4.** At every target error down to `1e-7`, RK4's cheapest stable
  step already beats the target, so log–log interpolation has nothing to interpolate and the
  method vanished from the table. A missing row reads as "not measured". The table now
  carries a `status` column (`interpolated` / `already_better_than_target` /
  `not_reached_in_sweep`) and emits explicit rows with `null` costs, so the absence is stated
  rather than implied.
* **The cost driver could have plotted a silently truncated curve.** `tools/render_p4_cost.py`
  joins T7 (which carries the `accurate` flag) to T9 (which carries the cost counters) on
  `(method, h_used)`. If the two tables were produced by different runs the keys would not
  match and a naive join would drop the mismatched rows without complaint. The script now
  **raises** on a key mismatch.
* **20 of the 20 beamer frames overflowed their text area on the first build.** Ten of them
  by more than 40 pt, one by 126 pt, with the last line of the `Newton 容差` frame visibly
  cut off. The fix was structural, not cosmetic: the base font went to 10 pt, three
  four-column numeric tables that could not fit a narrow column were **transposed** (the
  frozen-bound table and the adaptive table), the two frames carrying a full-width
  three-panel figure were re-laid out with the figure in a column, and the cost table was
  rescaled to a common `10^5` unit so its columns no longer collided. Three frames that
  still exceeded the frame carry `[shrink=8]` and a further two carry `[shrink=2]` for
  overflows of 0.32 pt and 2.5 pt; the shrink costs under 1 pt of font size in every case,
  and no content was deleted to make a frame fit. The only warning left in the build is a
  13.8 pt overfull `\vbox` inside metropolis's own title-page template, which does not clip
  anything on the rendered page. (Superseded by Entry 12 and the later re-layout: no frame now
  needs a `[shrink]` key at all. The first two frames of that list no longer exist as main
  frames, the last two were fixed structurally, and the transparency-log frame was re-laid out
  so its 14 entries fit without one.)
* **An unescaped `_` in `\texttt`.** `\code{run_all.py}` in the outline frame aborted the
  build with `Missing $ inserted`. The first compile failure was reported as a warning by
  `tectonic` and only surfaced when the log was read without a `grep` filter — reading the
  filtered output had hidden the actual error line.

---

## What the AI did **not** do

* No number in `results/` was written by hand; every one is produced by `run_all.py`.
* SciPy is used **only** as the reference oracle (`src/analysis/reference.py`). Explicit
  Euler, RK4, implicit Euler and the Newton solver are implemented from scratch in
  `src/integrators/`; the cost counters are those of our own code, and
  `test_cost_counters_are_consistent_with_the_expected_per_step_counts` checks them
  against the textbook per-step counts.
* The brief's orientation values are used only as regression targets
  (`tests/test_model.py`), never as a substitute for our own computation.
* No number in the beamer deck was transcribed from memory: every cell was copied out of
  `results/tables/*.md` or `results/summary.json` and then verified against those files.
  Entry 11 is the one that failed that check.
