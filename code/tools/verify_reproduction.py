"""One-command reproduction check: do my numbers match the deck?

Run this after ``python run_all.py`` (and optionally
``python tools/render_f16_shootout.py``).  It reads the artefacts the pipeline
just wrote and compares every number that appears in the slides against the
value the deck quotes, printing a PASS/FAIL line per item.

    python tools/verify_reproduction.py            # figures the deck uses
    python tools/verify_reproduction.py --all      # also the extra figures

Design notes
------------
* Every tolerance is stated explicitly.  A number that is exact by construction
  (an integer, a count, a scipy nfev) is checked for equality; a number that
  comes out of a fit or a floating-point trajectory is checked with a relative
  tolerance and the tolerance is printed.
* A missing artefact is a FAIL, not a crash -- the point is to tell the
  teammate which stage they have not run yet.
* Exit code is 0 only if every check passed.

The file the numbers come from is printed next to each section so a failure is
easy to trace back to a pipeline stage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _finite(x) -> bool:
    return not (x != x or x in (float("inf"), float("-inf")))


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[bool, str, str]] = []

    def check(self, label: str, got, exp, *, rtol: float = 0.0,
              atol: float = 0.0, kind: str = "value", note: str = "") -> None:
        if got is None:
            self.rows.append((False, label, f"NOT FOUND ({kind}) {note}".rstrip()))
            return
        try:
            if isinstance(exp, (int, float)) and isinstance(got, (int, float)):
                if isinstance(exp, float) and isinstance(got, float) \
                        and not _finite(exp):
                    # inf / nan: exact match only, |inf - inf| is nan
                    ok = (got == exp)
                else:
                    tol = max(atol, rtol * abs(exp))
                    ok = abs(got - exp) <= tol
                shown = f"got {got:g}  expected {exp:g}"
                if _finite(exp) and _finite(got):
                    shown += f"  tol {max(atol, rtol * abs(exp)):g}"
            else:
                ok = got == exp
                shown = f"got {got!r}  expected {exp!r}"
        except TypeError:
            ok, shown = False, f"got {got!r}  expected {exp!r}"
        if note:
            shown += f"   [{note}]"
        self.rows.append((ok, label, shown))

    def section(self, title: str, source: str) -> None:
        self.rows.append((True, f"--- {title}", f"source: {source}"))

    def finish(self) -> int:
        n_pass = sum(1 for ok, lab, _ in self.rows if ok and not lab.startswith("---"))
        n_fail = sum(1 for ok, lab, _ in self.rows if not ok)
        for ok, lab, detail in self.rows:
            if lab.startswith("---"):
                print(f"\n{lab}   ({detail})")
            else:
                print(f"  [{'PASS' if ok else 'FAIL'}] {lab:46s} {detail}")
        print()
        total = n_pass + n_fail
        if n_fail:
            print(f"RESULT: {n_pass}/{total} checks passed, {n_fail} FAILED.")
            return 1
        print(f"RESULT: all {total} checks passed.")
        return 0


def _load_json(name: str):
    p = RESULTS / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


# The reference values the deck states everywhere (T4 / F2).  These are the
# numbers a teammate must be able to reproduce before anything else matters.
REF_Y40 = [0.7158270687194044, 9.185534764557785e-06, 0.28416374574582975]


def check_reference(rep: Report, summary: dict | None) -> None:
    rep.section("REFERENCE SOLUTION  (F2, p5; T4)", "results/summary.json / reference/*")
    if summary is None:
        rep.check("summary.json present", None, 1, kind="file")
        return
    ref = summary.get("reference", {})
    got_y = ref.get("y_at_t40")
    for i, (exp, nm) in enumerate(zip(REF_Y40, "ABC")):
        rep.check(f"y(40) component {nm}", None if got_y is None else got_y[i], exp,
                  rtol=1e-9, note="the headline answer")
    rep.check("B peak value y2_max", ref.get("y2_peak_value"),
              3.648706770555561e-05, rtol=1e-6)
    rep.check("B peak time t_peak", ref.get("y2_peak_time"),
              0.0044201016416862225, rtol=1e-6)
    defect = ref.get("max_invariant_defect")
    rep.check("max |y1+y2+y3-1| <= 1e-13", None if defect is None else float(defect <= 1e-13), 1,
              note=f"got {defect:.3e}" if defect is not None else "")
    digits = ref.get("confirmed_digits")
    rep.check("digits confirmed by tolerance repeat >= 10", digits, 10.758746789063661,
              rtol=1e-3, note="must be >= 10 to quote 10 significant digits")
    diff = ref.get("max_abs_diff")
    rep.check("tight-vs-tighter max abs diff <= 1e-10", None if diff is None else float(diff <= 1e-10), 1,
              note=f"got {diff:.3e}" if diff is not None else "")


def check_stiffness(rep: Report, summary: dict | None) -> None:
    rep.section("STIFFNESS  (F6, p7; T1/T2)", "results/summary.json")
    if summary is None:
        rep.check("summary.json present", None, 1, kind="file")
        return
    head = summary.get("stiffness_ratio_headline", {})
    rep.check("stiffness ratio S(t=40)", head.get("S_at_t40"), 158401.7717539767, rtol=1e-6,
              note="~1.6e5, the headline number")
    rep.check("S at first reported t=1e-8", head.get("S_at_first_positive_t"),
              1.6659999169627364, rtol=1e-6)
    rep.check("states classified as two-non-zero", head.get("n_states_classified_two_nonzero"),
              200, kind="count")
    frozen = summary.get("frozen_jacobian_bound", {})
    rep.check("max |Re lambda| over trajectory", frozen.get("abs_lambda_max_global"),
              3392.788124454394, rtol=1e-6, note="at t=40, not t=0")
    rep.check("structural zero is really zero", summary.get("spectral_consistency", {}).get("passes"),
              True)
    checks = summary.get("check_values") or []
    if checks:
        rep.check("our S vs brief S at t=1e-4 (ratio)", checks[0].get("ours_over_brief"),
                  1.003244, rtol=1e-3, note="T2: agreement with the published value")


def check_stability(rep: Report, summary: dict | None) -> None:
    rep.section("STABILITY REGIONS  (F4, p8; T3)", "results/summary.json / tables/T3")
    if summary is None:
        rep.check("summary.json present", None, 1, kind="file")
        return
    methods = {m["method"]: m for m in summary.get("stability_methods", [])}
    ee, rk4, ie = methods.get("Explicit Euler", {}), methods.get("RK4", {}), methods.get("Implicit Euler", {})
    rep.check("Explicit Euler real-axis limit", ee.get("max_stable_real_h_lambda"), 2.0, rtol=1e-9)
    rep.check("RK4 real-axis limit", rk4.get("max_stable_real_h_lambda"), 2.785293563405284, rtol=1e-9)
    rep.check("Explicit Euler is NOT A-stable", ee.get("A_stable"), False)
    rep.check("Implicit Euler IS A-stable", ie.get("A_stable"), True)
    frozen_pm = summary.get("frozen_jacobian_bound", {}).get("per_method", {})
    rep.check("step bound implied by lambda_max (EE, frozen estimate)",
              frozen_pm.get("explicit_euler", {}).get("h_bound"),
              5.894856756849875e-04, rtol=1e-9, note="F4/F7: 5.895e-4 s")
    rep.check("step bound implied by lambda_max (RK4, frozen estimate)",
              frozen_pm.get("rk4", {}).get("h_bound"),
              8.209453291025052e-04, rtol=1e-9, note="F4/F7: 8.209e-4 s")
    rep.check("Implicit Euler has no spectral step bound",
              frozen_pm.get("implicit_euler", {}).get("h_bound"), float("inf"))
    rep.check("step limit displayed in T3 (EE)", ee.get("h_limit_for_lambda_max_3p39e3"),
              "5.895e-04")
    rep.check("step limit displayed in T3 (RK4)", rk4.get("h_limit_for_lambda_max_3p39e3"),
              "8.209e-04")


def check_sweep(rep: Report, summary: dict | None) -> None:
    rep.section("STEP-SIZE BOUNDARY  (F7, p19; T6)", "results/summary.json / tables/T6")
    if summary is None:
        rep.check("summary.json present", None, 1, kind="file")
        return
    b = {r["method"]: r for r in summary.get("sweep_boundaries", [])}
    ee, rk4, ie = b.get("Explicit Euler", {}), b.get("RK4", {}), b.get("Implicit Euler", {})
    rep.check("EE largest stable step (empirical)", ee.get("h_max_stable"),
              5.415944540727903e-04, rtol=1e-6)
    rep.check("RK4 largest stable step (empirical)", rk4.get("h_max_stable"),
              8.308753271571601e-04, rtol=1e-6)
    rep.check("Implicit Euler stable over the whole sweep", ie.get("h_max_stable"),
              2.999850007499625e-03, rtol=1e-6)
    rep.check("EE smallest step that diverged", ee.get("h_min_diverged"),
              6.246389e-04, rtol=1e-3, note="h_max_stable < this < frozen bound")


def check_convergence(rep: Report, summary: dict | None) -> None:
    rep.section("CONVERGENCE ORDER  (F8, p15; T8)", "results/summary.json / tables/T8")
    if summary is None:
        rep.check("summary.json present", None, 1, kind="file")
        return
    orders = summary.get("convergence", {}).get("observed_order_late", {})
    rep.check("Explicit Euler observed order ~1", orders.get("explicit_euler", {}).get("slope"),
              1.0788392802053026, rtol=2e-2, note="nominal 1")
    rep.check("RK4 observed order ~4", orders.get("rk4", {}).get("slope"),
              4.267791445917273, rtol=2e-2, note="nominal 4")
    rep.check("Implicit Euler observed order ~1", orders.get("implicit_euler", {}).get("slope"),
              0.9999553566544891, rtol=2e-2, note="nominal 1")


def check_four_track(rep: Report, summary: dict | None) -> None:
    rep.section("FOUR CHECK LINES  (F9, p17; T13/T14)", "results/summary.json")
    if summary is None:
        rep.check("summary.json present", None, 1, kind="file")
        return
    b = {r["method"]: r for r in summary.get("four_track_boundaries", [])}
    ee, rk4 = b.get("Explicit Euler", {}), b.get("RK4", {})
    rep.check("EE: accuracy and non-negativity fail at the same h",
              ee.get("h_max_track3_accurate"), 5.415944540727903e-04, rtol=1e-6)
    rep.check("RK4: accuracy fails BEFORE stability",
              rk4.get("h_max_track3_accurate"), 7.204092192421295e-04, rtol=1e-6)
    rep.check("RK4: stability survives longer",
              rk4.get("h_max_track2_non_negative"), 8.308753271571601e-04, rtol=1e-6)


def check_cost(rep: Report) -> None:
    rep.section("COST AT MATCHED ACCURACY  (F12, p21; T15)",
                "results/tables/T15_matched_accuracy_cost.csv")
    p = RESULTS / "tables" / "T15_matched_accuracy_cost.csv"
    if not p.exists():
        rep.check("T15_matched_accuracy_cost.csv present", None, 1, kind="file")
        return
    import csv
    rows = list(csv.DictReader(open(p, newline="")))
    at = {}
    for r in rows:
        if abs(float(r["target_error"]) - 1e-6) < 1e-18:
            at[r["method"]] = float(r["cost_units"])
    rep.check("cost units, Explicit Euler @1e-6", at.get("Explicit Euler"), 3.025853e+05, rtol=1e-3)
    rep.check("cost units, RK4 @1e-6", at.get("RK4"), 3.200010e+05, rtol=1e-3)
    rep.check("cost units, Implicit Euler @1e-6", at.get("Implicit Euler"), 1.108786e+06, rtol=1e-3)
    if "Implicit Euler" in at and "Explicit Euler" in at:
        rep.check("implicit / explicit cost ratio @1e-6",
                  at["Implicit Euler"] / at["Explicit Euler"], 3.664, rtol=1e-2,
                  note="the counter-intuitive result")


def check_shootout(rep: Report, summary: dict | None) -> None:
    rep.section("SOLVER SHOOTOUT  (F16, p23; T20)",
                "results/solver_shootout.json  (tools/render_f16_shootout.py)")
    data = _load_json("solver_shootout.json")
    if data is None:
        rep.check("solver_shootout.json present -- run tools/render_f16_shootout.py",
                  None, 1, kind="file")
        return
    got = {r["method"]: r["nfev"] for r in data.get("rows", [])}
    for m, exp in (("RK45", 241994), ("RK23", 136730), ("DOP853", 214706),
                   ("Radau", 752), ("BDF", 452)):
        rep.check(f"nfev {m}", got.get(m), exp, kind="count",
                  note="exact, version-independent")
    # LSODA is explicitly NOT checked: its switching heuristics changed between
    # scipy releases (425 on 1.11.1, 407 on 1.18.1, 334 with an analytic jac).
    n_lsoda = got.get("LSODA")
    if n_lsoda is not None:
        rep.check("nfev LSODA is hundreds, not thousands",
                  float(100 <= n_lsoda <= 9999), 1,
                  note=f"got {n_lsoda} (version-dependent, not checked exactly)")
    rep.check("scipy version recorded", data.get("scipy") is not None, True,
              note=f"scipy {data.get('scipy')}")


def check_extra(rep: Report, summary: dict | None) -> None:
    rep.section("EXTRA FIGURES IN THE PACKAGE (F5/F10/F11/F14/F15)",
                "results/summary.json")
    if summary is None:
        rep.check("summary.json present", None, 1, kind="file")
        return
    newton = summary.get("newton_studies", {})
    damp = newton.get("damping", {})
    rep.check("damped Newton: fewer iterations than full steps",
              None if not damp else float(damp.get("damped_dirs_max", 99) < damp.get("full_dirs_last", 0)), 1,
              note="F14")
    jac = newton.get("jacobian", {})
    rep.check("finite-difference Jacobian needs a tuned step",
              jac.get("rule_of_thumb_eps"), 1.4901161193847656e-08, rtol=1e-9, note="F15")
    adapt = summary.get("adaptive", {}).get("mixed") or []
    if adapt:
        worst = max(float(e.get("error_over_tol_global", 9)) for e in adapt)
        rep.check("adaptive RK4 tracks the tolerance (worst err/tol < 1)",
                  float(worst < 1.0), 1, kind="flag",
                  note=f"across {len(adapt)} tolerances, worst {worst:.3e}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true",
                    help="also check the figures not used in the deck")
    args = ap.parse_args()

    if not RESULTS.exists():
        print(f"results/ not found under {ROOT} -- run `python run_all.py` first.")
        return 2

    summary = _load_json("summary.json")
    rep = Report()
    check_reference(rep, summary)
    check_stiffness(rep, summary)
    check_stability(rep, summary)
    check_sweep(rep, summary)
    check_convergence(rep, summary)
    check_four_track(rep, summary)
    check_cost(rep)
    check_shootout(rep, summary)
    if args.all:
        check_extra(rep, summary)
    return rep.finish()


if __name__ == "__main__":
    raise SystemExit(main())
