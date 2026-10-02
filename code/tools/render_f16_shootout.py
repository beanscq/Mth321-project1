"""Render F16 -- the Task 4 solver shootout (non-stiff vs stiff in scipy).

Why this exists
---------------
Task 4 of the brief asks for a comparison of a non-stiff and a stiff solver on
this problem.  That comparison uses *scipy's own* solvers, not the three
integrators we wrote ourselves, so it does not belong in ``run_all.py`` (which
reproduces our own methods).  This driver is the missing piece: it runs the
same problem, the same tolerances and the same output grid through six
``scipy.integrate.solve_ivp`` methods and writes F16 plus a table.

Determinism
-----------
``nfev`` (the number of right-hand-side evaluations) is reproducible to the
digit for RK45 / RK23 / DOP853 / Radau / BDF on any machine.  Two things are
NOT reproducible and must never be used as a pass/fail criterion:

* **Wall-clock time** -- depends on the CPU and on what else is running.
* **LSODA's nfev** -- LSODA switches between an Adams (explicit) and a BDF
  (implicit) formulation at run time; the switching heuristics changed between
  SciPy releases.  Measured: 425 on SciPy 1.11.1, 407 on SciPy 1.18.1, and 334
  if an analytic Jacobian is passed.  Judge LSODA by its order of magnitude
  (hundreds, not hundreds of thousands), not by its last digit.

``jac`` is deliberately NOT passed.  That is how the figure in the deck was
generated, and it keeps the numbers identical across SciPy versions for the
five solvers that matter.  (For the explicit methods the argument is ignored
anyway; for LSODA it silently changes the count.)

Usage
-----
    python tools/render_f16_shootout.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import robertson as R          # noqa: E402

FIGDIR = ROOT / "results" / "figures"
TABDIR = ROOT / "results" / "tables"

# Same tolerances for every solver: this is the whole point of the comparison.
RTOL = 1.0e-6
ATOL = [1.0e-10, 1.0e-12, 1.0e-10]

# (method, kind) -- kind drives the colour, exactly as in the deck's legend.
SOLVERS = [
    ("RK45",     "non-stiff"),
    ("RK23",     "non-stiff"),
    ("DOP853",   "non-stiff"),
    ("Radau",    "stiff"),
    ("BDF",      "stiff"),
    ("LSODA",    "auto"),
]

# Values quoted in the deck / the reproduction checklist.  ``None`` = the count
# is SciPy-version dependent and is not checked for equality.
EXPECTED_NFEV = {
    "RK45": 241994,
    "RK23": 136730,
    "DOP853": 214706,
    "Radau": 752,
    "BDF": 452,
    "LSODA": None,
}

C_MAIN = "#1F2937"
C_A = "#185FA5"
C_B = "#A32D2D"
C_ACCENT = "#854F0B"
KIND_COLOR = {"non-stiff": C_B, "stiff": C_A, "auto": C_ACCENT}
KIND_LABEL = {"non-stiff": "non-stiff (explicit)", "stiff": "stiff (implicit)",
              "auto": "auto (LSODA)"}


def run_one(method: str) -> dict:
    """One solver, one measurement.  Times only the solve, not the import."""
    calls = {"n": 0}

    def rhs(t, y):
        calls["n"] += 1
        return R.f(t, y)

    t0 = time.perf_counter()
    sol = solve_ivp(rhs, (0.0, 40.0), R.Y0, method=method,
                    rtol=RTOL, atol=ATOL)
    wall = time.perf_counter() - t0

    if not sol.success:
        raise RuntimeError(f"{method} failed: {sol.message}")

    return {
        "method": method,
        "nfev": int(sol.nfev),
        "wall_time_s": wall,
        "n_steps_accepted": int(sol.t.size),
        "y_at_t40": sol.y[:, -1].tolist(),
        "y2_at_t40": float(sol.y[R.I_B, -1]),
        # cross-check that the counter we instrumented matches scipy's own
        "nfev_instrumented": calls["n"],
    }


def render(rows: list[dict], reference: np.ndarray, outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    names = [r["method"] for r in rows]
    colors = [KIND_COLOR[k] for _, k in SOLVERS]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.4))

    ax = axes[0]
    ax.bar(x, [r["nfev"] for r in rows], color=colors, width=0.68)
    ax.set_yscale("log")
    ax.set_ylabel(r"function evaluations  nfev  (log)")
    ax.set_title(r"(a)  cost: how many $f$ evaluations")
    for xi, r in zip(x, rows):
        ax.annotate(f"{r['nfev']:,}", (xi, r["nfev"]),
                    textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=8.5, color=C_MAIN)

    ax = axes[1]
    ax.bar(x, [r["wall_time_s"] * 1e3 for r in rows], color=colors, width=0.68)
    ax.set_ylabel("wall-clock time  (ms)")
    ax.set_title("(b)  wall-clock time")
    for xi, r in zip(x, rows):
        ax.annotate(f"{r['wall_time_s'] * 1e3:,.0f}", (xi, r["wall_time_s"] * 1e3),
                    textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=8.5, color=C_MAIN)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.grid(True, axis="x", alpha=0.0)

    handles = [plt.Rectangle((0, 0), 1, 1, color=KIND_COLOR[k]) for k in
               ("non-stiff", "stiff", "auto")]
    fig.legend(handles, [KIND_LABEL[k] for k in ("non-stiff", "stiff", "auto")],
               loc="lower center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    speedup = max(r["nfev"] for r in rows) / min(r["nfev"] for r in rows)
    fig.suptitle("F16  scipy  solve_ivp: non-stiff vs stiff, same tolerances"
                 f"   (best/worst nfev = {speedup:,.0f}x)",
                 fontsize=11, color=C_MAIN)
    fig.tight_layout(rect=(0, 0.045, 1, 0.95))

    written = []
    for fmt in ("pdf", "png"):
        p = outdir / f"F16_solver_shootout.{fmt}"
        fig.savefig(p, bbox_inches="tight", format=fmt)
        written.append(p)
    plt.close(fig)
    return written


T20_COLUMNS = ["method", "kind", "nfev", "wall_time_ms", "y2_at_t40",
               "abs_err_y2_vs_reference", "speedup_vs_slowest"]


def write_table(rows: list[dict], reference: np.ndarray, outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    slowest = max(r["nfev"] for r in rows)
    table = []
    for r, (_, kind) in zip(rows, SOLVERS):
        table.append({
            "method": r["method"],
            "kind": kind,
            "nfev": r["nfev"],
            "wall_time_ms": f"{r['wall_time_s'] * 1e3:.1f}",
            "y2_at_t40": f"{r['y2_at_t40']:.6e}",
            "abs_err_y2_vs_reference":
                f"{abs(r['y2_at_t40'] - float(reference[R.I_B])):.3e}",
            "speedup_vs_slowest": f"{slowest / r['nfev']:.1f}",
        })

    written = []
    csv_path = outdir / "T20_solver_shootout.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=T20_COLUMNS)
        w.writeheader()
        w.writerows(table)
    written.append(csv_path)

    md_path = outdir / "T20_solver_shootout.md"
    with open(md_path, "w") as fh:
        fh.write("| " + " | ".join(T20_COLUMNS) + " |\n")
        fh.write("|" + "---|" * len(T20_COLUMNS) + "\n")
        for t in table:
            fh.write("| " + " | ".join(str(t[c]) for c in T20_COLUMNS) + " |\n")
    written.append(md_path)
    return written


def main() -> int:
    # Reference at t = 40: prefer our own tight reference, fall back to the
    # brief's orientation values if the pipeline has not been run yet.
    ref_path = ROOT / "results" / "reference" / "reference.npz"
    if ref_path.exists():
        d = np.load(ref_path, allow_pickle=True)
        t = np.asarray(d["t"], float)
        y = np.asarray(d["y"], float)
        reference = y[int(np.argmin(np.abs(t - 40.0)))]
        src = ref_path.relative_to(ROOT)
    else:
        reference = R.BRIEF_Y_AT_T40
        src = "model.BRIEF_Y_AT_T40 (pipeline not run)"
    print(f"reference y(40) from {src}: {reference}")

    rows = []
    print(f"scipy {scipy.__version__}  |  rtol={RTOL}  atol={ATOL}  |  no jac passed")
    ok_all = True
    for method, _kind in SOLVERS:
        r = run_one(method)
        rows.append(r)
        # For the explicit methods every RHS call must be visible to scipy's
        # counter.  The implicit ones additionally poke the RHS to build a
        # numerical Jacobian and scipy books those calls under ``njev``, so the
        # instrumented count legitimately runs higher -- do not assert there.
        if _kind == "non-stiff":
            assert r["nfev"] == r["nfev_instrumented"], (
                f"{method}: scipy nfev={r['nfev']} but our counter saw "
                f"{r['nfev_instrumented']} -- the RHS is not being called as "
                f"assumed")
        exp = EXPECTED_NFEV[method]
        if exp is None:
            verdict = "n/a (version-dependent)"
        elif r["nfev"] == exp:
            verdict = "MATCH"
        else:
            verdict = f"MISMATCH (expected {exp:,})"
            ok_all = False
        print(f"  {method:7s} nfev={r['nfev']:>9,}  "
              f"wall={r['wall_time_s'] * 1e3:7.1f} ms  "
              f"y2(40)={r['y2_at_t40']:.6e}  {verdict}")

    written = render(rows, np.asarray(reference, float), FIGDIR)
    written += write_table(rows, np.asarray(reference, float), TABDIR)

    payload = {
        "scipy": scipy.__version__,
        "rtol": RTOL, "atol": ATOL, "t_span": [0.0, 40.0],
        "jac_passed": False,
        "reference_y2_at_t40": float(np.asarray(reference, float)[R.I_B]),
        "rows": [{k: v for k, v in r.items() if k != "y_at_t40"} for r in rows],
        "note": "nfev is deterministic for RK45/RK23/DOP853/Radau/BDF; "
                "wall_time_s is machine-dependent and LSODA's nfev is "
                "SciPy-version-dependent -- neither is a pass/fail criterion.",
    }
    out = ROOT / "results" / "solver_shootout.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")

    for p in written:
        print("  wrote", p.relative_to(ROOT))
    print("  wrote", out.relative_to(ROOT))
    if not ok_all:
        print("RESULT: some nfev values do not match the deck.", file=sys.stderr)
        return 1
    print("RESULT: all version-independent nfev values match the deck.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
