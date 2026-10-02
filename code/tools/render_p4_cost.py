"""Render the P4 cost deliverables (F12, T15) from the P2 tables already on disk.

Why this exists
---------------
F12 and T15 are pure post-processing of ``T7_convergence.csv`` (which carries the
``accurate`` flag) and ``T9_cost.csv`` (which carries the cost counters).  A full
``run_all.py`` pass reproduces them, but that re-integrates the whole step sweep
for ~10 minutes of machine time.  This driver re-renders just the two artefacts
from the frozen P2 numbers, so the figures can be iterated on without touching
any result that is already verified.

It deliberately does NOT recompute anything: it joins the two tables on
``(method, h_used)`` and refuses to run if the join is not one-to-one, so a
mismatch is an error rather than a silently truncated figure.

Usage
-----
    python tools/render_p4_cost.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reporting import figures as F            # noqa: E402
from src.reporting import tables as T             # noqa: E402

FIGDIR = ROOT / "results" / "figures"
TABDIR = ROOT / "results" / "tables"
TARGETS = [1e-5, 3e-6, 1e-6, 1e-7]


def _read(stem: str) -> list[dict]:
    with open(TABDIR / f"{stem}.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def _flag(v) -> bool:
    return str(v).strip().lower() in ("yes", "true", "1")


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())
    import json
    summary = json.loads((ROOT / "results" / "summary.json").read_text())
    frozen = summary["frozen_jacobian_bound"]

    conv = {(r["method"], r["h_used"]): r for r in _read("T7_convergence")}
    cost = {(r["method"], r["h_used"]): r for r in _read("T9_cost")}
    if set(conv) != set(cost):
        missing = set(conv) ^ set(cost)
        print(f"REFUSING: T7/T9 keys disagree on {len(missing)} rows: "
              f"{sorted(missing)[:4]}", file=sys.stderr)
        return 1

    rows = []
    for key, c in cost.items():
        k = conv[key]
        rows.append({
            "method": c["method"],
            "h_used": float(c["h_used"]),
            "error_late": float(c["error_late"]),
            "cost_units": float(c["cost_units"]),
            "wall_time_s": float(c["wall_time_s"]),
            "stable": _flag(k["stable"]),
            "non_negative": _flag(k["non_negative"]),
            "accurate": _flag(k["accurate"]),
        })

    n_bad = sum(1 for r in rows if not r["accurate"])
    print(f"joined {len(rows)} rows; {n_bad} not accurate (drawn hollow)")

    written = F.fig_work_precision(rows, frozen, cfg, FIGDIR)
    for p in written:
        print("  wrote", p.relative_to(ROOT))

    by_method = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)
    t15 = T.matched_accuracy_rows(by_method, TARGETS)
    written = T.write_table(t15, "T15_matched_accuracy_cost", TABDIR,
                            columns=T.MATCHED_ACCURACY_COLUMNS)
    for p in written:
        print("  wrote", p.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
