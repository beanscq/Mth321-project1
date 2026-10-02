"""Render F13 (the two clocks) from artefacts already on disk.

F13 is the one figure written for the *explanation* rather than for a
measurement: it shows the fast and the slow characteristic time of the
Robertson system, which is the whole reason the problem is hard.

It needs only the tight reference solution and its Jacobian spectrum.  Both are
on disk (``results/reference/reference.npz`` and ``results/raw/spectral.npz``),
so this driver re-derives the spectrum in a few milliseconds instead of
re-running the ~10 minute pipeline.

It also prints the four numbers the slide quotes, so that none of them is
transcribed from memory or from a previous run.

Usage
-----
    python tools/render_f13_clock.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis import eigen as E          # noqa: E402
from src.reporting import figures as F       # noqa: E402


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())
    ref = np.load(ROOT / "results" / "reference" / "reference.npz")
    t = np.asarray(ref["t"], float)
    y = np.asarray(ref["y"], float)

    spec = E.analyse_from_series(t, y,
                                 zero_rtol=float(cfg["eigen"]["zero_eigenvalue_rtol"]))

    pos = t > 0
    tau_f = 1.0 / np.abs(spec.lam_fast.real)[pos]
    tau_s = 1.0 / np.abs(spec.lam_slow.real)[pos]

    i_peak = int(np.argmax(y[:, 1]))
    print(f"tau_fast   t=1e-8 : {tau_f[0]:.4g} s")
    print(f"tau_fast   t=40   : {tau_f[-1]:.4g} s")
    print(f"tau_slow   t=1e-8 : {tau_s[0]:.4g} s")
    print(f"tau_slow   t=40   : {tau_s[-1]:.4g} s")
    print(f"gap        t=1e-8 : {tau_s[0] / tau_f[0]:.4g}x")
    print(f"gap        t=40   : {tau_s[-1] / tau_f[-1]:.4g}x")
    print(f"explicit bound at t=40 : 2*tau_fast = {2.0 * tau_f[-1]:.4g}")
    print(f"B peak                 : {y[i_peak, 1]:.4e} at t = {t[i_peak]:.4e} s")
    print(f"B lifetime estimate    : {y[i_peak, 1] / 0.04:.4g} s  (= y2 / production rate)")
    print(f"A lifetime             : {1.0 / 0.04:.4g} s")

    written = F.fig_clock_scales(t, y, spec, cfg, ROOT / "results" / "figures")
    for p in written:
        print("  wrote", p.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
