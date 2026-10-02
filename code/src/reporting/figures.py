"""All figures.  Every figure states in its caption metadata what question it
answers; the module-level constants below record axes and data source.

Figure register
---------------
F13 is the one figure written for the *explanation* rather than for a
measurement: it shows the two clocks that make the problem stiff.

P0/P1
F2  reference solution            : log10 t | y1,y2,y3            -> chemical picture
F4  absolute stability regions    : Re(h*lam) | Im(h*lam)          -> why explicit methods are limited
F5  Jacobian spectrum             : log10 t | Re lam, |lam|, Im lam-> structural zero + degeneracy
F6  stiffness ratio S(t)          : log10 t | S(t)                 -> how stiffness grows
P2
F7  frozen bound vs step sweep    : h | error, min(y), trajectories-> is the local estimate trustworthy?
F8  uniform-step convergence      : h | error (log-log), order     -> observed order and its limits
P3
F9  four-track diagnostics        : h | defect, min y, error, y2(40)-> the four tests disagree
F10 Newton tolerance sweep        : tol | defect, error, iterations-> when is the algebraic solve free?
F11 adaptive step size            : t | h(t), err/tol, error vs tol-> does the controller track the tolerance?
P4
F12 work-precision                : cost | error, and h | error      -> which method is cheapest at EQUAL accuracy?
F13 two clocks (explanatory)      : log10 t | concentrations, 1/|Re lam| -> why the step size is locked
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..analysis import eigen as E
from ..analysis import stability_region as SR
from ..model import robertson as R
from . import tables as T

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "font.size": 10,
    "axes.titlesize": 10.5,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

C_MAIN = "#1F2937"
C_A = "#185FA5"
C_B = "#A32D2D"
C_C = "#27500A"
C_ACCENT = "#854F0B"
C_GREY = "#5F5E5A"
C_B3 = "#7B4FA8"          # threshold curve in F5(b)

# method identity, used consistently in F7 and F8
METHOD_STYLE = {
    "Explicit Euler": dict(color=C_B, marker="o", ls="-"),
    "RK4": dict(color=C_C, marker="s", ls="-"),
    "Implicit Euler": dict(color=C_A, marker="^", ls="-"),
}

# tolerance families, used in F10 and F11
TOL_COLORS = ["#A32D2D", "#854F0B", "#27500A", "#185FA5", "#7B4FA8"]

# compact method names for in-figure annotations (F12)
SHORT_NAME = {"Explicit Euler": "EE", "RK4": "RK4", "Implicit Euler": "IE"}


# ---------------------------------------------------------------------------
def save_figure(fig, stem: str, cfg: dict, outdir: Path) -> list[Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in cfg["figures"].get("formats", ["png"]):
        p = outdir / f"{stem}.{fmt}"
        fig.savefig(p, bbox_inches="tight", format=fmt)
        written.append(p)
    plt.close(fig)
    return written


# ---------------------------------------------------------------------------
# F2 -- reference solution
# ---------------------------------------------------------------------------
def fig_reference_solution(t, y, cfg, outdir):
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    pos = t > 0

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.1))

    ax = axes[0]
    ax.semilogx(t[pos], y[pos, R.I_A], color=C_A, lw=1.6, label=r"$y_1=[A]$")
    ax.semilogx(t[pos], y[pos, R.I_C], color=C_C, lw=1.6, label=r"$y_3=[C]$")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("normalised concentration")
    ax.set_title("(a) Slow species: A is consumed, C accumulates")
    ax.legend(loc="center left")

    ax = axes[1]
    ax.loglog(t[pos], y[pos, R.I_B], color=C_B, lw=1.6)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$y_2=[B]$")
    ax.set_title(r"(b) Short-lived intermediate B: rise, peak, monotone decay")
    i_peak = int(np.argmax(y[:, R.I_B]))
    if t[i_peak] > 0:
        ax.plot([t[i_peak]], [y[i_peak, R.I_B]], marker="o", ms=5, color=C_ACCENT)
        ax.annotate(f"peak $y_2={y[i_peak, R.I_B]:.3e}$\n$t={t[i_peak]:.3e}$",
                    xy=(t[i_peak], y[i_peak, R.I_B]),
                    xytext=(0.32, 0.78), textcoords="axes fraction",
                    fontsize=8.5, color=C_ACCENT,
                    arrowprops=dict(arrowstyle="-", color=C_ACCENT, lw=0.7))

    ax = axes[2]
    defect = np.maximum(R.invariant_defect(y), 1e-22)
    running = np.maximum.accumulate(defect)
    ax.loglog(t[pos], defect[pos], color=C_GREY, lw=0.7, alpha=0.55,
              label="per output time")
    ax.loglog(t[pos], running[pos], color=C_ACCENT, lw=1.6,
              label="running maximum")
    ax.axhline(np.finfo(float).eps, color="#5F5E5A", ls="--", lw=0.9,
               label=r"machine $\epsilon$")
    ax.set_ylim(1e-22, 1e-13)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$|y_1+y_2+y_3-1|$")
    ax.set_title("(c) Linear invariant holds at roundoff level")
    ax.legend(loc="upper left", fontsize=8.5)

    fig.suptitle("F2  Tight-tolerance reference (Radau, tightened tolerances)",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    return save_figure(fig, "F2_reference_solution", cfg, outdir)


# ---------------------------------------------------------------------------
# F4 -- absolute stability regions
# ---------------------------------------------------------------------------
def fig_stability_regions(t, y, cfg, outdir):
    scfg = cfg["stability_region"]
    re_range = tuple(scfg["re_range"])
    im_range = tuple(scfg["im_range"])
    n = int(scfg["grid_n"])
    h_list = list(scfg["overlay_h"])

    t = np.asarray(t, float)
    y = np.asarray(y, float)
    pos = t > 0
    logt = np.log10(t[pos])

    methods = ["Explicit Euler", "RK4", "Implicit Euler"]
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6))

    for ax, method in zip(axes, methods):
        margin = SR.stability_margin_grid(method, re_range, im_range, n)
        re, im, _ = SR.stability_grid(re_range, im_range, n)

        ax.contourf(re, im, margin, levels=[-1e9, 0.0],
                    colors=["#DCEBF8"], alpha=0.9, zorder=1)
        ax.contour(re, im, margin, levels=[0.0],
                   colors=["#185FA5"], linewidths=1.1, zorder=2)

        # actual spectra h*lam(t), coloured by log10 t (two eigenvalues per time)
        for h in h_list:
            z = SR.overlay_spectra(h, t[pos], y[pos])
            c_vals = np.repeat(logt, z.size // logt.size)
            sc = ax.scatter(z.real, z.imag, c=c_vals, cmap="viridis",
                            s=10, zorder=4, vmin=logt.min(), vmax=logt.max())
        # real-axis limit
        a, _ = SR.real_axis_stability_interval(method)
        if np.isfinite(a):
            ax.axvline(a, color=C_B, ls="--", lw=1.0, zorder=3)

        ax.axhline(0.0, color=C_GREY, lw=0.5, zorder=0)
        ax.axvline(0.0, color=C_GREY, lw=0.5, zorder=0)
        ax.set_xlim(*re_range)
        ax.set_ylim(*im_range)
        ax.set_xlabel(r"$\mathrm{Re}(h\lambda)$")
        if method == methods[0]:
            ax.set_ylabel(r"$\mathrm{Im}(h\lambda)$")

        interval = "unbounded ($\\lambda$ in left half plane)" if not np.isfinite(a) \
            else f"[{a:.4f}, 0]"
        ax.set_title(f"{method}  (order {SR.METHOD_ORDER[method]})\n"
                     f"stable real interval {interval}"
                     + ("\nA-stable, L-stable" if method == "Implicit Euler" else ""))

    cb = fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.015)
    cb.set_label(r"$\log_{10} t$  (spectrum of $h\lambda$, $h\in$ "
                 + ", ".join(f"{h:g}" for h in h_list) + ")")
    fig.suptitle("F4  Absolute stability regions and the actual Robertson spectrum "
                 r"$h\lambda(t)$", y=1.03, fontsize=12)
    return save_figure(fig, "F4_stability_regions", cfg, outdir)


# ---------------------------------------------------------------------------
# F5 -- Jacobian spectrum
# ---------------------------------------------------------------------------
def fig_eigenvalues(t, y, spec: E.SpectralAnalysis, cfg, outdir,
                    zoom=None, zero_rtol: float = 1e-8):
    """``zoom`` is an optional ``(t, y)`` pair from a dense early-time reference
    used to resolve the conjugate-pair window that the fixed grid steps over."""
    t = np.asarray(t, float)
    pos = t > 0
    tp = t[pos]
    logt = np.log10(tp)

    lam_f = spec.lam_fast[pos]
    lam_s = spec.lam_slow[pos]

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.3))

    ax = axes[0]
    ax.semilogx(tp, np.abs(lam_f.real), color=C_B, lw=1.6,
                label=r"$|\mathrm{Re}\,\lambda_{\mathrm{fast}}|$")
    ax.semilogx(tp, np.abs(lam_s.real), color=C_A, lw=1.6,
                label=r"$|\mathrm{Re}\,\lambda_{\mathrm{slow}}|$")
    ax.set_yscale("log")
    ax.axhline(R.K1, color=C_GREY, ls=":", lw=1.0, label=r"$k_1=0.04$")
    i40 = int(np.argmin(np.abs(tp - 40.0)))
    ax.annotate(f"{np.abs(lam_f.real[i40]):.2f}", xy=(tp[i40], np.abs(lam_f.real[i40])),
                xytext=(0.05, 0.12), textcoords="axes fraction",
                fontsize=8.5, color=C_B)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$|\mathrm{Re}\,\lambda|$")
    ax.set_title("(a) The two non-zero eigenvalues\n(real, negative, 5 decades apart)")
    ax.legend(loc="lower left")

    ax = axes[1]
    abs_lam = np.abs(spec.lam_full[pos].real)
    ax.loglog(tp, np.maximum(abs_lam[:, 0], 1e-22), color=C_B, lw=1.5,
              label=r"$|\mathrm{Re}\,\lambda|_{\max}$")
    ax.loglog(tp, np.maximum(abs_lam[:, 1], 1e-22), color=C_A, lw=1.5, label="2nd")
    ax.loglog(tp, np.maximum(abs_lam[:, 2], 1e-22), color=C_ACCENT, lw=1.5, marker=".",
              ms=3, label=r"structural $\lambda_3 = 0$")
    ax.loglog(tp, zero_rtol * abs_lam[:, 0], color=C_B3,
              ls="--", lw=1.0, label=f"zero threshold ({zero_rtol:g}$\\times$max)")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$|\mathrm{Re}\,\lambda|$")
    ax.set_title("(b) All three eigenvalues of $J$\n"
                 r"$(1,1,1)^T J \equiv 0 \Rightarrow \lambda_3 = 0$ exactly")
    ax.legend(loc="lower left", fontsize=8)

    ax = axes[2]
    if zoom is not None:
        tz, yz = zoom
        lamz = R.reduced_eigenvalues_analytic(yz)
        imz = np.abs(lamz.imag).max(axis=-1)
        ax.loglog(tz, np.maximum(imz, 1e-14), color=C_ACCENT, lw=1.6)
        pos_z = imz > 0
        if np.any(pos_z):
            t_lo, t_hi = tz[pos_z].min(), tz[pos_z].max()
            ax.axvspan(t_lo, t_hi, color=C_B, alpha=0.12)
            ratio = np.max(np.abs(lamz.imag) / np.maximum(np.abs(lamz.real), 1e-300))
            ax.annotate(f"conjugate pair\n$t\\in[{t_lo:.2e},\\,{t_hi:.2e}]$\n"
                        f"$|\\mathrm{{Im}}|/|\\mathrm{{Re}}|\\leq{ratio:.1e}$",
                        xy=(np.sqrt(t_lo * t_hi), np.max(imz)),
                        xytext=(0.05, 0.10), textcoords="axes fraction",
                        fontsize=8.5, color=C_ACCENT,
                        arrowprops=dict(arrowstyle="-", color=C_ACCENT, lw=0.7))
        ax.set_xlim(tz.min(), tz.max())
        ax.set_title("(c) $|\\mathrm{Im}\\,\\lambda|$ on a dense early-time\n"
                     "reference (the fixed grid steps over the window)")
    else:
        ax.set_title("(c) $|\\mathrm{Im}\\,\\lambda|$")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$|\mathrm{Im}\,\lambda|$")

    fig.suptitle("F5  Jacobian spectrum along the reference trajectory", y=1.03, fontsize=12)
    return save_figure(fig, "F5_jacobian_spectrum", cfg, outdir)


# ---------------------------------------------------------------------------
# F6 -- stiffness ratio
# ---------------------------------------------------------------------------
def fig_stiffness_ratio(t, y, spec: E.SpectralAnalysis, cfg, outdir):
    t = np.asarray(t, float)
    pos = t > 0
    tp = t[pos]
    S = spec.stiffness_ratio[pos]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.4))

    ax = axes[0]
    ax.loglog(tp, S, color=C_MAIN, lw=1.8, label="our computation")
    for t_c, s_c in E.BRIEF_STIFFNESS_RATIO.items():
        ax.plot([t_c], [s_c], marker="s", ms=6, mfc="none", mec=C_B, mew=1.4,
                ls="none", label="brief check values" if t_c == 1e-4 else None)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$S(t)$")
    ax.set_title(r"(a) $S(t)$ from the two non-zero eigenvalues only")
    ax.legend(loc="upper left")

    ax = axes[1]
    ax.semilogx(tp, S, color=C_MAIN, lw=1.8)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$S(t)$")
    ax.set_title("(b) Same data, log-time / linear ratio\n(growth is concentrated at late $t$)")

    fig.suptitle(r"F6  Stiffness ratio $S(t)=\max|\mathrm{Re}\lambda|/\min|\mathrm{Re}\lambda|$"
                 r" over the two non-zero eigenvalues", y=1.03, fontsize=12)
    return save_figure(fig, "F6_stiffness_ratio", cfg, outdir)


# ---------------------------------------------------------------------------
# F7 -- frozen-Jacobian bound vs empirical step sweep (P2)
# ---------------------------------------------------------------------------
def _sweep_series(rows, key):
    """Group a sweep result list into ``{method: [(h, value), ...]}``."""
    out: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        out.setdefault(r["method"], []).append((float(r["h_used"]), float(r[key])))
    for v in out.values():
        v.sort()
    return out


def fig_frozen_bound_vs_sweep(sweep_rows, frozen, cfg, outdir, pair=None):
    """``pair`` is an optional ``[(label, Trajectory), ...]`` for the trajectory
    panel that shows what "unstable" and "negative" actually look like."""
    crit = cfg["criteria"]
    tol_acc = float(crit["accurate_tol"])
    bounds = frozen["per_method"]
    name_to_bound = {v["method"]: float(v["h_bound"]) for v in bounds.values()}

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.6))

    # ---- (a) error vs h, with the frozen bound ---------------------------
    ax = axes[0]
    for r in sweep_rows:
        st = METHOD_STYLE[r["method"]]
        ax.plot([r["h_used"]], [max(r["error_late"], 1e-16)], ls="none",
                marker=st["marker"], ms=5.5,
                mfc=st["color"] if r["stable"] else "none",
                mec=st["color"], mew=1.3, zorder=4)
    for name, st in METHOD_STYLE.items():
        ser = _sweep_series([r for r in sweep_rows if r["method"] == name], "error_late")
        if name in ser:
            h, e = zip(*ser[name])
            ax.plot(h, np.maximum(e, 1e-16), color=st["color"], lw=1.2, alpha=0.75)
        b = name_to_bound[name]
        if np.isfinite(b):
            ax.axvline(b, color=st["color"], ls=":", lw=1.4, alpha=0.9)
    ax.axhline(tol_acc, color=C_MAIN, ls="--", lw=1.1)
    ax.annotate(rf"accuracy target {tol_acc:g}", xy=(0.02, tol_acc),
                xycoords=("axes fraction", "data"), fontsize=8.5, color=C_MAIN,
                va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"step size $h$")
    ax.set_ylabel(r"full-state error $\|y_h-y_{\rm ref}\|_\infty$")
    ax.set_title("(a) Error vs step size across the stability boundary\n"
                 "filled = stable, open = diverged; dotted lines = frozen bound")
    ax.plot([], [], ls="none", marker="o", mfc=C_GREY, mec=C_GREY,
            label="stable run")
    ax.plot([], [], ls="none", marker="o", mfc="none", mec=C_GREY,
            label="diverged run")
    ax.legend(loc="lower right", fontsize=8.5)

    # ---- (b) componentwise non-negativity on the same sweep --------------
    ax = axes[1]
    for name, st in METHOD_STYLE.items():
        ser = _sweep_series([r for r in sweep_rows if r["method"] == name],
                            "min_component_nodes")
        if not ser:
            continue
        h, m = zip(*ser[name])
        ax.plot(h, m, color=st["color"], marker=st["marker"], ms=4.5, lw=1.2,
                label=name)
    ax.axhline(0.0, color=C_MAIN, lw=1.2)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_xlabel(r"step size $h$")
    ax.set_ylabel(r"$\min_i y_i$ over all accepted steps")
    for name, st in METHOD_STYLE.items():
        b = name_to_bound[name]
        if np.isfinite(b):
            ax.axvline(b, color=st["color"], ls=":", lw=1.4, alpha=0.9)
    ax.set_title("(b) Non-negativity is a separate diagnostic\n"
                 "a negative concentration is chemically inadmissible")
    ax.legend(loc="lower left", fontsize=8.5)

    # ---- (c) what the two sides of the boundary look like ----------------
    ax = axes[2]
    if pair:
        for label, tr, ls in pair:
            first = ls == "-"
            for idx, col, letter in ((R.I_A, C_A, "1"),
                                     (R.I_B, C_B, "2"),
                                     (R.I_C, C_C, "3")):
                ax.plot(tr.t, tr.y[:, idx], color=col, ls=ls, lw=1.3,
                        label=f"$y_{letter}$" if first else None)
        ax.set_yscale("symlog", linthresh=1.0e-6)
        lo = min(float(tr.y.min()) for _, tr, _ in pair)
        hi = max(float(tr.y.max()) for _, tr, _ in pair)
        ax.set_ylim(min(2.0 * lo, -0.5), 4.0 * hi)
        lines = ["dashed = the unstable run, which does not reach $t=40$"]
        for label, tr, _ls in pair:
            lines.append(f"{label}:  min $y_i = {tr.min_component():+.2e}$"
                         f" over $[0,{tr.t_covered:.1f}]$")
        ax.text(0.02, 0.02, "\n".join(lines), transform=ax.transAxes,
                fontsize=8, va="bottom",
                bbox=dict(fc="white", ec=C_GREY, lw=0.6, alpha=0.9))
    ax.axhline(0.0, color=C_MAIN, lw=1.0)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("normalised concentration (symlog)")
    ax.set_title("(c) Explicit Euler either side of $h_{\\rm crit}$\n"
                 "the stable run stays in $[0,1]$, the other does not")
    ax.legend(loc="center left", fontsize=8)

    fig.suptitle("F7  Frozen-Jacobian estimate vs an empirical step sweep",
                 y=1.03, fontsize=12)
    fig.tight_layout()
    return save_figure(fig, "F7_frozen_bound_vs_sweep", cfg, outdir)


# ---------------------------------------------------------------------------
# F8 -- uniform-step convergence (P2)
# ---------------------------------------------------------------------------
def _error_series(rows, key):        # rows already filtered to one method
    return sorted((float(r["h_used"]), float(r[key])) for r in rows)


def fig_convergence(rows_by_method, orders_late, orders_full, cfg, outdir):
    err_floor = float(cfg["criteria"]["reference_error_floor"])
    nominal = {"explicit_euler": 1, "rk4": 4, "implicit_euler": 1}

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.6))

    def _panel(ax, orders, title):
        for key, rows in rows_by_method.items():
            st = METHOD_STYLE[rows[0]["method"]]
            ser = _error_series(rows, orders[key]["error_key"])
            h = np.array([p[0] for p in ser])
            e = np.array([p[1] for p in ser])
            ok = np.isfinite(e) & (e > err_floor)
            stab = np.array([r["stable"] for r in
                             sorted(rows, key=lambda r: r["h_used"])])
            ax.loglog(h[ok & stab], e[ok & stab], ls="none", marker=st["marker"],
                      ms=6, mfc=st["color"], mec=st["color"], zorder=4,
                      label=rows[0]["method"])
            ax.loglog(h[~ok | ~stab], e[~ok | ~stab], ls="none",
                      marker=st["marker"], ms=6, mfc="none", mec=st["color"],
                      mew=1.2, zorder=3)
            p = orders[key]["slope"]
            if np.isfinite(p):
                hw = np.array(orders[key]["h_window"])
                anchor = np.interp(np.log(hw[-1]), np.log(h), np.log(np.maximum(e, 1e-30)))
                ax.plot(hw, np.exp(anchor + p * (np.log(hw) - np.log(hw[-1]))),
                        color=st["color"], lw=1.0, ls="-", alpha=0.8)
                ax.annotate(f"fitted $p={p:.2f}$", xy=(0.42, 0.06 + 0.09 *
                            list(rows_by_method).index(key)),
                            xycoords="axes fraction", fontsize=8.5,
                            color=st["color"])
        ax.axhline(err_floor, color=C_GREY, ls=":", lw=1.0)
        ax.annotate("reference error floor", xy=(0.02, err_floor),
                    xycoords=("axes fraction", "data"), fontsize=8,
                    color=C_GREY, va="bottom")
        ax.set_xlabel(r"step size $h$")
        ax.set_ylabel(r"error $\|y_h-y_{\rm ref}\|_\infty$")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=8.5)

    _panel(axes[0], orders_late,
           "(a) Error on $t\\geq10^{-3}$\n(asymptotic region, fitted slopes shown)")
    _panel(axes[1], orders_full,
           "(b) Error on the WHOLE grid\n(cubic-Hermite dense output does not cap the order)")

    # ---- (c) per-pair observed order ------------------------------------
    ax = axes[2]
    for key, rows in rows_by_method.items():
        st = METHOD_STYLE[rows[0]["method"]]
        pairs = orders_late[key]["pairs"]
        if not pairs:
            continue
        hc = [p["h_coarse"] for p in pairs]
        ax.semilogx(hc, [p["order"] for p in pairs], color=st["color"],
                    marker=st["marker"], ms=5, lw=1.2, label=rows[0]["method"])
        ax.axhline(nominal[key], color=st["color"], ls=":", lw=1.1, alpha=0.9)
    ax.set_xlabel(r"coarse step size of the pair, $h_{\rm coarse}$")
    ax.set_ylabel(r"observed order  $p=\log(e_1/e_2)/\log(h_1/h_2)$")
    ax.set_title("(c) Approach to the asymptotic order\n"
                 "dotted lines = nominal order 1 / 4 / 1")
    ax.set_ylim(0.0, 5.2)
    ax.legend(loc="center right", fontsize=8.5)

    fig.suptitle("F8  Uniform-step convergence, observed order and its limits",
                 y=1.03, fontsize=12)
    fig.tight_layout()
    return save_figure(fig, "F8_convergence", cfg, outdir)


# ---------------------------------------------------------------------------
# F9 -- the four diagnostics, kept apart (P3)
# ---------------------------------------------------------------------------
def fig_four_track_diagnostics(sweep_rows, frozen, cfg, outdir,
                               ref_y2_t40: float, ref_abs_tol: float):
    """Four panels, four diagnostics, deliberately **not** combined.

    The point of the figure is that they disagree: the conservation panel is
    flat for every run including the ones that diverge and go strongly
    negative, while the other three panels span eight orders of magnitude.
    """
    crit = cfg["criteria"]
    tol_acc = float(crit["accurate_tol"])
    bounds = {v["method"]: float(v["h_bound"]) for v in frozen["per_method"].values()}

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.4))

    # ---- (a) track 1: conservation --------------------------------------
    ax = axes[0, 0]
    for name, st in METHOD_STYLE.items():
        rs = sorted([r for r in sweep_rows if r["method"] == name],
                    key=lambda r: r["h_used"])
        h = np.array([r["h_used"] for r in rs])
        d = np.maximum([r["mass_defect_max"] for r in rs], 1e-18)
        ax.loglog(h, d, color=st["color"], marker=st["marker"], ms=5, lw=1.1,
                  mfc=st["color"], label=name)
    band = np.finfo(float).eps
    ax.axhline(band, color=C_MAIN, ls="--", lw=1.1)
    ax.annotate(f"double precision floor {band:.1e}", xy=(0.02, band),
                xycoords=("axes fraction", "data"), fontsize=8, color=C_MAIN,
                va="bottom")
    worst = min(sweep_rows, key=lambda r: r["min_component_nodes"])
    ax.annotate(
        "the worst run here\n"
        f"{worst['method']}, $h={worst['h_used']:.3g}$:\n"
        f"$\\min_i y_i={worst['min_component_nodes']:+.1f}$, "
        f"error $={worst['error_full']:.2e}$\n"
        f"mass defect $={worst['mass_defect_max']:.1e}$",
        xy=(worst["h_used"], max(worst["mass_defect_max"], 1e-18)),
        xytext=(0.04, 0.04), textcoords="axes fraction", fontsize=8.5,
        color=C_ACCENT,
        bbox=dict(fc="white", ec=C_ACCENT, lw=0.6, alpha=0.92),
        arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=0.8))
    ax.set_ylim(1e-18, 1e-10)
    ax.set_xlabel(r"step size $h$")
    ax.set_ylabel(r"$|y_1+y_2+y_3-1|_{\max}$")
    ax.set_title("(a) TRACK 1 - conservation is exact and therefore useless\n"
                 "flat at roundoff for every run, including the negative ones")
    ax.legend(loc="upper right", fontsize=8.5)

    # ---- (b) track 2: componentwise non-negativity ----------------------
    ax = axes[0, 1]
    for name, st in METHOD_STYLE.items():
        rs = sorted([r for r in sweep_rows if r["method"] == name],
                    key=lambda r: r["h_used"])
        h = np.array([r["h_used"] for r in rs])
        m = np.array([r["min_component_nodes"] for r in rs])
        ax.plot(h, m, color=st["color"], marker=st["marker"], ms=5, lw=1.1,
                label=name)
        b = bounds.get(name, np.inf)
        if np.isfinite(b):
            ax.axvline(b, color=st["color"], ls=":", lw=1.4)
    ax.axhline(0.0, color=C_MAIN, lw=1.2)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-6)
    ax.set_xlabel(r"step size $h$")
    ax.set_ylabel(r"$\min_i y_i$ over accepted steps")
    ax.set_title("(b) TRACK 2 - non-negativity fails at a DIFFERENT $h$\n"
                 "dotted lines = frozen-Jacobian estimate (EE and RK4)")
    ax.legend(loc="lower left", fontsize=8.5)

    # ---- (c) track 3: full-state error ----------------------------------
    ax = axes[1, 0]
    for name, st in METHOD_STYLE.items():
        rs = sorted([r for r in sweep_rows if r["method"] == name],
                    key=lambda r: r["h_used"])
        h = np.array([r["h_used"] for r in rs])
        e = np.maximum([r["error_full"] for r in rs], 1e-18)
        s = np.array([r["stable"] for r in rs])
        ax.loglog(h[s], e[s], color=st["color"], marker=st["marker"], ms=5,
                  lw=1.1, label=name)
        if np.any(~s):
            ax.loglog(h[~s], e[~s], ls="none", marker=st["marker"], ms=6,
                      mfc="none", mec=st["color"], mew=1.3)
    ax.axhline(tol_acc, color=C_MAIN, ls="--", lw=1.1)
    ax.annotate(f"accuracy target {tol_acc:g}", xy=(0.02, tol_acc),
                xycoords=("axes fraction", "data"), fontsize=8.5, color=C_MAIN,
                va="bottom")
    ax.set_xlabel(r"step size $h$")
    ax.set_ylabel(r"$\|y_h-y_{\rm ref}\|_\infty$")
    ax.set_title("(c) TRACK 3 - the error is what actually distinguishes them\n"
                 "open markers = the run diverged before $t=40$")
    ax.legend(loc="lower right", fontsize=8.5)

    # ---- (d) track 4: the terminal value --------------------------------
    ax = axes[1, 1]
    for name, st in METHOD_STYLE.items():
        rs = sorted([r for r in sweep_rows if r["method"] == name],
                    key=lambda r: r["h_used"])
        h = np.array([r["h_used"] for r in rs])
        d = np.abs(np.array([r["y2_at_t40"] for r in rs]) - ref_y2_t40)
        ok = np.isfinite(d) & (d > 0.0)
        ax.loglog(h[ok], d[ok], color=st["color"], marker=st["marker"], ms=5,
                  lw=1.1, label=name)
        if np.any(~ok):
            ax.plot(h[~ok], np.full(np.count_nonzero(~ok), 1e-3), ls="none",
                    marker="x", ms=6, mec=st["color"], mew=1.4)
    ax.axhspan(0.0, ref_abs_tol, color=C_ACCENT, alpha=0.16, lw=0)
    ax.annotate(f"reference uncertainty {ref_abs_tol:.1e}\n"
                "(our own tolerance repeat; nothing below is meaningful)",
                xy=(0.02, ref_abs_tol), xycoords=("axes fraction", "data"),
                fontsize=8, color=C_ACCENT, va="bottom")
    ax.set_ylim(1e-13, 3e-3)
    ax.set_xlabel(r"step size $h$")
    ax.set_ylabel(r"$|y_2(40)-y_2^{\rm ref}(40)|$")
    ax.set_title("(d) TRACK 4 - the terminal value, quoted with its digits\n"
                 "$\\times$ = no value: the run never reached $t=40$")
    ax.legend(loc="lower right", fontsize=8.5)

    fig.suptitle("F9  Four diagnostics, four verdicts: conservation cannot stand in for accuracy",
                 y=1.0, fontsize=12.5)
    fig.tight_layout()
    return save_figure(fig, "F9_four_track_diagnostics", cfg, outdir)


# ---------------------------------------------------------------------------
# F10 -- Newton stopping-tolerance sweep (P3)
# ---------------------------------------------------------------------------
def fig_newton_tolerance_sweep(rows, stats, cfg, outdir):
    """Three panels: what the invariant defect sees, what the error sees, and
    what the algebraic solve costs."""
    rs = sorted(rows, key=lambda r: r["newton_tol"])
    tol = np.array([r["newton_tol"] for r in rs])
    defect = np.array([max(r["mass_defect_max"], 1e-18) for r in rs])
    envelope = np.array([r["invariant_bound_3tol"] for r in rs])
    err = np.array([r["error_full"] for r in rs])
    err40 = np.array([r["error_at_t40"] for r in rs])
    nostep = np.array([r["fraction_steps_without_newton"] for r in rs])
    iters = np.array([r["newton_iters_per_step"] for r in rs])
    plateau = float(stats["error_plateau"])
    budget = float(stats["max_inflation"])
    h = float(stats["h"])

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.7))

    # ---- (a) the invariant defect sees nothing --------------------------
    ax = axes[0]
    ax.loglog(tol, defect, color=C_A, marker="o", ms=6, lw=1.4,
              label="measured invariant defect")
    ax.loglog(tol, envelope, color=C_GREY, ls="--", lw=1.3,
              label=r"envelope $3\,\mathrm{tol}$ (theory)")
    ax.axhline(np.finfo(float).eps, color=C_MAIN, ls=":", lw=1.1,
               label="double precision floor")
    bad = max(rs, key=lambda r: r["newton_tol"])
    ax.annotate(f"$\\mathrm{{tol}}={bad['newton_tol']:g}$: solution error "
                f"$={bad['error_full']:.1e}$,\n"
                f"$\\min_i y_i={bad['min_component']:+.1f}$, defect "
                f"$={bad['mass_defect_max']:.1e}$",
                xy=(bad["newton_tol"], max(bad["mass_defect_max"], 1e-18)),
                xytext=(0.05, 0.06), textcoords="axes fraction", fontsize=8.2,
                color=C_ACCENT,
                bbox=dict(fc="white", ec=C_ACCENT, lw=0.6, alpha=0.92),
                arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=0.8))
    ax.set_xlabel(r"Newton stopping tolerance $\mathrm{tol}$")
    ax.set_ylabel(r"$|y_1+y_2+y_3-1|_{\max}$")
    ax.set_title("(a) The invariant defect cannot see the algebraic error\n"
                 "every Newton iterate preserves the sum exactly")
    ax.legend(loc="upper left", fontsize=8.2)
    ax.set_ylim(1e-18, 1.0)

    # ---- (b) the solution error does ------------------------------------
    ax = axes[1]
    ax.loglog(tol, err, color=C_B, marker="o", ms=6, lw=1.5,
              label=r"$\|y_h-y_{\rm ref}\|_\infty$")
    ax.loglog(tol, err40, color=C_C, marker="s", ms=5, lw=1.2, alpha=0.85,
              label=r"$\|y_h(40)-y_{\rm ref}(40)\|_\infty$")
    ax.axhline(plateau, color=C_MAIN, ls="--", lw=1.1)
    ax.axhspan(plateau / budget, plateau * budget, color=C_ACCENT, alpha=0.14, lw=0)
    rec = float(stats["recommended_newton_tol"])
    brk = stats["first_tolerance_that_breaks_it"]
    i_rec = int(np.argmin(np.abs(tol - rec)))
    ax.plot([tol[i_rec]], [err[i_rec]], marker="*", ms=15, color=C_C,
            mec="white", mew=0.7, zorder=6)
    ax.annotate(f"recommended $\\mathrm{{tol}}={rec:g}$\n"
                f"(inflation {stats['recommended_inflation']:.2f})",
                xy=(tol[i_rec], err[i_rec]), xytext=(0.30, 0.16),
                textcoords="axes fraction", fontsize=8.2, color=C_C,
                bbox=dict(fc="white", ec=C_C, lw=0.6, alpha=0.92),
                arrowprops=dict(arrowstyle="->", color=C_C, lw=0.8))
    ax.annotate(f"discretisation plateau $={plateau:.2e}$",
                xy=(0.55, plateau), xycoords=("axes fraction", "data"),
                fontsize=8.2, color=C_MAIN, va="bottom",
                bbox=dict(fc="white", ec=C_MAIN, lw=0.5, alpha=0.9))
    if brk is not None:
        ax.axvspan(min(tol), float(brk) * 0.999, color=C_GREY, alpha=0.10, lw=0)
        ax.annotate("Newton steps are SKIPPED here:\n"
                    "the predictor residual is already\n"
                    f"$O(h^2)=({stats['predictor_residual_scaling_h2']:.0e})$, "
                    "below $\\mathrm{tol}$",
                    xy=(0.02, 0.60), xycoords="axes fraction", fontsize=8.0,
                    color=C_GREY,
                    bbox=dict(fc="white", ec=C_GREY, lw=0.5, alpha=0.92))
    ax.set_xlabel(r"Newton stopping tolerance $\mathrm{tol}$")
    ax.set_ylabel("error")
    ax.set_title("(b) ... but the solution error does\n"
                 "the stopping rule must sit left of the break")
    ax.legend(loc="lower left", fontsize=8.2)

    # ---- (c) what the tolerance costs -----------------------------------
    ax = axes[2]
    ax.semilogx(tol, iters, color=C_A, marker="^", ms=6, lw=1.5,
                label="Newton updates per accepted step")
    ax.set_xlabel(r"Newton stopping tolerance $\mathrm{tol}$")
    ax.set_ylabel("Newton updates per step")
    ax.set_ylim(-0.05, 1.4)
    ax2 = ax.twinx()
    ax2.semilogx(tol, nostep, color=C_ACCENT, marker="o", ms=5, lw=1.3, ls="--",
                 label="steps that skip Newton entirely")
    ax2.set_ylabel("fraction of steps with zero Newton updates")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center left", fontsize=8.2)
    ax.set_title("(c) Newton is often FREE, and sometimes skipped\n"
                 "tight tolerances average about one update per step")

    fig.suptitle(f"F10  Newton stopping-tolerance sweep at $h={h:g}$: "
                 "the algebraic error must be negligible against the discretisation error",
                 y=1.03, fontsize=12.5)
    fig.tight_layout()
    return save_figure(fig, "F10_newton_tolerance_sweep", cfg, outdir)


# ---------------------------------------------------------------------------
# F11 -- adaptive step-size control (P3)
# ---------------------------------------------------------------------------
def _binned_stats(t, values, edges):
    """Median and 5th/95th percentile of ``values`` in each time bin."""
    idx = np.digitize(np.asarray(t, dtype=float), edges) - 1
    n = len(edges) - 1
    med = np.full(n, np.nan)
    lo = np.full(n, np.nan)
    hi = np.full(n, np.nan)
    for k in range(n):
        m = idx == k
        if not np.any(m):
            continue
        v = values[m]
        med[k] = np.nanmedian(v)
        lo[k] = np.nanpercentile(v, 5)
        hi[k] = np.nanpercentile(v, 95)
    return med, lo, hi


def fig_adaptive(levels, cfg, outdir):
    """The controller adapting, tracking the tolerance, and the one norm that
    is fooled by the disparity of the component scales."""
    mixed = sorted(levels["mixed"], key=lambda r: r["tol"])
    control = sorted(levels["absolute_inf"], key=lambda r: r["tol"])
    h_frozen = float(levels["frozen_rk4_bound"])
    floor = float(cfg["criteria"]["reference_error_floor"])
    t_end_plot = float(mixed[0]["trajectory"].t_end)

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.8))

    # ---- (a) the step size really moves ---------------------------------
    ax = axes[0]
    for i, r in enumerate(mixed):
        tr = r["trajectory"]
        t = np.asarray(tr.t, dtype=float)
        h = np.asarray(tr.h_at_node, dtype=float)
        ok = np.isfinite(h)
        ax.loglog(t[ok], h[ok], color=TOL_COLORS[i % len(TOL_COLORS)], lw=1.2,
                  label=f"$\\mathrm{{tol}}={r['tol']:g}$")
    loose = mixed[-1]
    rej_t = np.asarray(loose["trajectory"].reject_t, dtype=float)
    rej_h = np.asarray(loose["trajectory"].reject_h, dtype=float)
    if rej_t.size:
        keep = rej_t > 0
        ax.plot(rej_t[keep], rej_h[keep], ls="none", marker="x", ms=3.2,
                color=C_GREY, alpha=0.4, zorder=1,
                label=f"rejected attempts ({rej_t.size})")
    ax.axhline(h_frozen, color=C_MAIN, ls="--", lw=1.3)
    ax.axhline(2.0 * h_frozen, color=C_MAIN, ls=":", lw=1.1)
    ax.annotate("frozen-Jacobian bound $h=8.21\\times10^{-4}$\n"
                "(the accepted solution is propagated\n"
                "with two half-steps, so its effective\n"
                "step is $h/2$: dotted line = $2h_{\\rm frozen}$)",
                xy=(0.02, 2.0 * h_frozen), xycoords=("axes fraction", "data"),
                fontsize=7.8, color=C_MAIN, va="bottom")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"controller step $h$")
    ax.set_title("(a) The step size is not constant: it collapses through the\n"
                 "induction layer, then plateaus on the stability bound")
    ax.legend(loc="lower left", fontsize=7.8, ncol=1)

    # ---- (b) the LOCAL error estimate tracks the tolerance --------------
    ax = axes[1]
    edges = np.logspace(np.log10(1e-7), np.log10(t_end_plot), 61)
    centres = np.sqrt(edges[:-1] * edges[1:])
    for i, r in enumerate(mixed):
        tr = r["trajectory"]
        t = np.asarray(tr.t, dtype=float)
        ratio = tr.err_over_tol()
        ok = np.isfinite(ratio) & (t > 0)
        col = TOL_COLORS[i % len(TOL_COLORS)]
        med, lo, hi = _binned_stats(t[ok], ratio[ok], edges)
        ax.fill_between(centres, lo, hi, color=col, alpha=0.18, lw=0)
        ax.semilogx(centres, med, color=col, lw=1.6,
                    label=f"$\\mathrm{{tol}}={r['tol']:g}$")
        ax.annotate(f"{r['err_over_tol_median']:.2f}",
                    xy=(centres[-1], med[-1]), xytext=(4, 0),
                    textcoords="offset points", fontsize=8.0, color=col,
                    va="center")
    ax.axhline(1.0, color=C_MAIN, lw=1.3)
    ax.annotate("acceptance threshold: $\\mathrm{err}/\\mathrm{tol}=1$",
                xy=(0.02, 1.0), xycoords=("axes fraction", "data"),
                fontsize=8.2, color=C_MAIN, va="bottom")
    ax.set_ylim(1e-4, 3.0)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"local error estimate $/$ requested tolerance")
    ax.set_title("(b) The controlled quantity does track the tolerance\n"
                 "median over log-time bins, band = 5th-95th percentile")
    ax.legend(loc="lower left", fontsize=7.8)

    # ---- (c) global error, and the norm that fails ----------------------
    ax = axes[2]
    tm = np.array([r["tol"] for r in mixed])
    em = np.array([r["error_full"] for r in mixed])
    tc = np.array([r["tol"] for r in control])
    ec = np.array([r["error_full"] for r in control])
    ax.loglog(tm, np.maximum(em, 1e-16), color=C_A, marker="^", ms=7, lw=1.6,
              label="componentwise-scaled norm")
    ax.loglog(tc, np.maximum(ec, 1e-16), color=C_B, marker="o", ms=7, lw=0,
              ls="none", label="absolute infinity norm (control)")
    ref = em[0] / tm[0]
    ax.loglog(tm, ref * tm, color=C_GREY, ls="--", lw=1.1,
              label=r"slope 1 guide, anchored at $\mathrm{tol}=10^{-4}$")
    ax.axhline(floor, color=C_MAIN, ls=":", lw=1.1)
    ax.annotate("reference error floor", xy=(0.03, floor),
                xycoords=("axes fraction", "data"), fontsize=8.0,
                color=C_MAIN, va="bottom")
    for r in control:
        if r["error_full"] > 1e-4:
            ax.annotate(
                f"$\\mathrm{{tol}}={r['tol']:g}$:\nerror $={r['error_full']:.1e}$,\n"
                f"$\\min_i y_i={r['min_component']:+.1e}$",
                xy=(r["tol"], max(r["error_full"], 1e-16)),
                xytext=(0.05, 0.62), textcoords="axes fraction", fontsize=7.8,
                color=C_B,
                bbox=dict(fc="white", ec=C_B, lw=0.6, alpha=0.92),
                arrowprops=dict(arrowstyle="->", color=C_B, lw=0.8))
            break
    aborted = [r for r in control if r["aborted"]]
    if aborted:
        ax.annotate(f"aborted at $t={aborted[0]['t_covered']:.1f}$",
                    xy=(min(tc), float(np.max(ec))), fontsize=7.6, color=C_B,
                    xytext=(0, -26), textcoords="offset points")
    ax.set_xlabel(r"requested tolerance")
    ax.set_ylabel("full-state error over the covered span")
    ax.set_title("(c) What the tolerance buys globally - and the norm\n"
                 "that silently loses the stiff mode")
    ax.legend(loc="lower right", fontsize=7.8)

    fig.suptitle("F11  RK4 step-doubling controller: step size, error tracking, and a norm that fails",
                 y=1.03, fontsize=12.5)
    fig.tight_layout()
    return save_figure(fig, "F11_adaptive_stepping", cfg, outdir)


# ---------------------------------------------------------------------------
# F12 -- work-precision: cost at MATCHED accuracy (P4)
# ---------------------------------------------------------------------------
def _pareto(points):
    """Non-dominated (cost, error) pairs, minimising both.  Returns ascending cost."""
    pts = sorted(set((float(c), float(e)) for c, e in points
                     if np.isfinite(c) and np.isfinite(e) and e > 0))
    front, best = [], np.inf
    for c, e in pts:
        if e < best:
            front.append((c, e))
            best = e
    return front


def fig_work_precision(rows, frozen, cfg, outdir, target_err=1e-6):
    """Cost to reach a given error.  Answers: is the implicit method actually cheaper?

    ``rows`` carry ``method``, ``h_used``, ``error_late``, ``cost_units`` and the
    booleans ``stable`` / ``accurate``.  Points that are NOT accurate are drawn hollow
    and are excluded from the fitted curves, because a run that diverged is cheap and
    meaningless - plotting it on a cost axis without a marker would flatter it.
    """
    by_method: dict[str, list] = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))

    # ---------------- (a) error vs cost -------------------------------------
    ax = axes[0]
    crossings = {}
    for name, rs in by_method.items():
        st = METHOD_STYLE[name]
        acc = sorted([r for r in rs if r["accurate"]],
                     key=lambda r: r["cost_units"])
        bad = [r for r in rs if not r["accurate"]]
        c = np.array([float(r["cost_units"]) for r in acc])
        e = np.array([float(r["error_late"]) for r in acc])
        ax.loglog(c, e, color=st["color"], marker=st["marker"], ms=6.5,
                  lw=1.6, label=name, zorder=4)
        if bad:
            ax.loglog([float(r["cost_units"]) for r in bad],
                      [float(r["error_late"]) for r in bad], ls="none",
                      marker=st["marker"], ms=7, mfc="none", mec=st["color"],
                      mew=1.3, zorder=3)
        cx = T.cost_at_error(list(zip(c, e)), target_err)
        if cx:
            crossings[name] = cx

    # one annotation for all non-converged points, not one per method
    bad_all = [r for r in rows if not r["accurate"]]
    if bad_all:
        worst = max(bad_all, key=lambda r: r["error_late"])
        ax.annotate("hollow = step past the stability\nlimit: cheap, but the run\n"
                    "is wrong",
                    xy=(float(worst["cost_units"]), float(worst["error_late"])),
                    xytext=(0.035, 0.62), textcoords="axes fraction",
                    fontsize=7.8, color=C_GREY, ha="left", va="top",
                    bbox=dict(fc="white", ec=C_GREY, lw=0.6, alpha=0.94),
                    arrowprops=dict(arrowstyle="->", color=C_GREY, lw=0.9))

    # Pareto frontier over every point (accurate or not)
    front = _pareto([(r["cost_units"], r["error_late"]) for r in rows])
    if len(front) > 1:
        fx = [p[0] for p in front]
        fy = [p[1] for p in front]
        ax.plot(fx, fy, color=C_GREY, ls="--", lw=1.3, zorder=2,
                label="Pareto frontier")

    # matched-accuracy line
    ax.axhline(target_err, color=C_ACCENT, ls=":", lw=1.4, zorder=1)
    for name, cx in crossings.items():
        st = METHOD_STYLE[name]
        ax.plot([cx], [target_err], marker="*", ms=13, color=st["color"],
                mec="white", mew=0.7, zorder=6)
    if crossings:
        rank = sorted(crossings, key=crossings.get)
        txt = "\n".join([r"cost to reach $10^{-6}$:"]
                        + [f"{SHORT_NAME[n]:<4s}{crossings[n]:.1e}" for n in rank]
                        + [f"EE is {crossings[rank[-1]] / crossings[rank[0]]:.1f}x"
                           " cheaper than IE"])
        ax.annotate(txt, xy=(0.965, 0.975), xycoords="axes fraction",
                    fontsize=8.3, ha="right", va="top", family="monospace",
                    bbox=dict(fc="white", ec=C_ACCENT, lw=0.9, alpha=0.96))

    # RK4 never gets as inaccurate as 1e-6 while staying stable, so it does not
    # cross the dotted line at all - that is the point, and it is stated, not hidden.
    rk4_acc = [r for r in by_method.get("RK4", []) if r["accurate"]]
    if rk4_acc:
        knee = min(rk4_acc, key=lambda r: float(r["cost_units"]))
        ax.annotate(f"RK4 at its cheapest stable step is\nalready at "
                    f"{float(knee['error_late']):.1e} - better than the\n"
                    f"dotted line, for {float(knee['cost_units']):.1e}",
                    xy=(float(knee["cost_units"]), float(knee["error_late"])),
                    xytext=(0.27, 0.30), textcoords="axes fraction",
                    fontsize=8.0, color=C_C, ha="left", va="top",
                    bbox=dict(fc="white", ec=C_C, lw=0.7, alpha=0.94),
                    arrowprops=dict(arrowstyle="->", color=C_C, lw=0.9))

    ax.set_xlabel(r"cost  (RHS / Jacobian / LU-solve units)")
    ax.set_ylabel(r"error on $t\geq10^{-3}$  $\|y_h-y_{\rm ref}\|_\infty$")
    ax.set_title("(a) Work-precision: cost to reach a given accuracy\n"
                 "dotted = matched accuracy, stars = where each curve crosses it")
    ax.legend(loc="lower left", fontsize=8.4)

    # ---------------- (b) error vs h, with the frozen bounds ----------------
    ax = axes[1]
    for name, rs in by_method.items():
        st = METHOD_STYLE[name]
        acc = sorted([r for r in rs if r["accurate"]], key=lambda r: r["h_used"])
        bad = sorted([r for r in rs if not r["accurate"]], key=lambda r: r["h_used"])
        if acc:
            ax.loglog([float(r["h_used"]) for r in acc],
                      [float(r["error_late"]) for r in acc], color=st["color"],
                      marker=st["marker"], ms=6.5, lw=1.6, label=name, zorder=4)
        if bad:
            ax.loglog([float(r["h_used"]) for r in bad],
                      [float(r["error_late"]) for r in bad], ls="none",
                      marker=st["marker"], ms=7, mfc="none", mec=st["color"],
                      mew=1.3, zorder=3)

    if bad_all:
        worst_b = min(bad_all, key=lambda r: float(r["h_used"]))
        ax.annotate("$(h,error)$ past\nthe bound",
                    xy=(float(worst_b["h_used"]), float(worst_b["error_late"])),
                    xytext=(0, -30), textcoords="offset points",
                    fontsize=7.8, color=C_GREY, ha="center", va="top",
                    bbox=dict(fc="white", ec=C_GREY, lw=0.6, alpha=0.94),
                    arrowprops=dict(arrowstyle="->", color=C_GREY, lw=0.9))

    # the two frozen bounds, labels staggered so they cannot collide
    for key, ylab in (("explicit_euler", 0.05), ("rk4", 0.34)):
        hb = float(frozen["per_method"][key]["h_bound"])
        st = METHOD_STYLE[frozen["per_method"][key]["method"]]
        ax.axvline(hb, color=st["color"], ls="--", lw=1.2, alpha=0.85)
        ax.annotate(f"$h_{{\\rm frozen}}={hb:.3e}$",
                    xy=(hb, ylab), xycoords=("data", "axes fraction"),
                    rotation=90, fontsize=7.8, color=st["color"], va="bottom",
                    ha="right")
    ax.annotate("an explicit step cannot grow past its\n"
                "frozen bound, so $h$ - and with it the\n"
                "cost - is capped from below",
                xy=(0.03, 0.97), xycoords="axes fraction", fontsize=8.2,
                ha="left", va="top", color=C_GREY,
                bbox=dict(fc="white", ec=C_GREY, lw=0.7, alpha=0.94))

    ax.set_xlabel(r"step size $h$")
    ax.set_ylabel(r"error on $t\geq10^{-3}$")
    ax.set_title("(b) Why the cost curve bends: $h$ is capped by stability,\n"
                 "so cost can only be paid down by going left")

    fig.suptitle("F12  Cost at matched accuracy - the implicit method is not automatically cheaper",
                 y=1.02, fontsize=12.5)
    fig.tight_layout()
    return save_figure(fig, "F12_work_precision", cfg, outdir)


# ---------------------------------------------------------------------------
# F13 -- the two clocks (explanatory; this is the "why is it hard" figure)
# ---------------------------------------------------------------------------
def fig_clock_scales(t, y, spec: E.SpectralAnalysis, cfg, outdir):
    """The "why is it hard" picture: what the system looks like, and its clocks.

    (a) the three concentrations on a log-time axis: B is a blip that peaks in
        the first few milliseconds, A and C drift for the whole 40 seconds;
    (b) the characteristic time ``1/|Re lambda|`` of the two non-zero modes,
        together with the step bound an explicit method inherits from the fast
        one, ``h < 2/|Re lambda_fast|``.

    (b) is the whole reason the problem is expensive, and it is the panel to
    look at before any method is compared: the fast clock falls by five decades
    across the run, the slow clock does not move, and the allowed step is
    pinned to the fast one for the entire 40 seconds.  Nothing about the
    accuracy one wants enters this picture.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    pos = t > 0
    tp = t[pos]

    lam_fast = np.abs(np.asarray(spec.lam_fast, dtype=complex).real)[pos]
    lam_slow = np.abs(np.asarray(spec.lam_slow, dtype=complex).real)[pos]
    tau_fast = 1.0 / lam_fast
    tau_slow = 1.0 / lam_slow
    bound = 2.0 * tau_fast

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.05))

    # ---------------- (a) the three species --------------------------------
    ax = axes[0]
    ax.semilogx(tp, y[pos, R.I_A], color=C_A, lw=1.9, label=r"$y_1=[A]$")
    ax.semilogx(tp, y[pos, R.I_C], color=C_C, lw=1.9, label=r"$y_3=[C]$")
    ax.set_xlabel(r"$t$  (s)")
    ax.set_ylabel("concentration   (linear)")
    ax.set_ylim(-0.04, 1.10)

    axr = ax.twinx()
    axr.loglog(tp, y[pos, R.I_B], color=C_B, lw=1.9, label=r"$y_2=[B]$  (right)")
    axr.set_ylabel(r"$y_2=[B]$   (log)", color=C_B)
    axr.set_ylim(1e-12, 1e-3)
    axr.tick_params(axis="y", colors=C_B, labelsize=8.5)
    axr.spines["right"].set_visible(True)
    axr.spines["right"].set_color(C_B)
    axr.spines["top"].set_visible(False)

    i_peak = int(np.argmax(y[:, R.I_B]))
    axr.plot([t[i_peak]], [y[i_peak, R.I_B]], marker="o", ms=5.5, color=C_B,
             mec="white", mew=0.8, zorder=5)
    axr.annotate(f"B peaks at {y[i_peak, R.I_B]:.2e}\n"
                 f"at $t={t[i_peak]*1e3:.1f}$ ms, then decays",
                 xy=(t[i_peak], y[i_peak, R.I_B]),
                 xytext=(0.28, 0.13), textcoords="axes fraction",
                 fontsize=8.4, color=C_B, va="bottom",
                 bbox=dict(fc="white", ec=C_B, lw=0.7, alpha=0.95),
                 arrowprops=dict(arrowstyle="->", color=C_B, lw=0.9))

    h_a, l_a = ax.get_legend_handles_labels()
    h_r, l_r = axr.get_legend_handles_labels()
    ax.legend(h_a + h_r, l_a + l_r, loc="center right", fontsize=8.3,
              framealpha=0.95)
    ax.set_title("(a) The three species on one time axis\n"
                 "(B is plotted on a log scale, right)", fontsize=10.5)

    # ---------------- (b) the clocks ---------------------------------------
    ax = axes[1]
    ax.fill_between(tp, tau_fast, tau_slow, color=C_ACCENT, alpha=0.09,
                    lw=0, zorder=1)
    ax.loglog(tp, tau_slow, color=C_A, lw=2.0, zorder=4,
              label=r"slow clock  $1/|\mathrm{Re}\lambda_{\rm slow}|$")
    ax.loglog(tp, tau_fast, color=C_B, lw=2.0, zorder=4,
              label=r"fast clock  $1/|\mathrm{Re}\lambda_{\rm fast}|$")
    ax.loglog(tp, bound, color=C_B, ls="--", lw=1.4, zorder=3,
              label=r"explicit step bound  $h<2/|\mathrm{Re}\lambda_{\rm fast}|$")

    ax.set_xlabel(r"$t$  (s)")
    ax.set_ylabel("characteristic time  (s)")
    ax.set_ylim(2e-5, 6e2)
    ax.set_xlim(tp[0] * 0.8, tp[-1] * 3.0)

    gap_lo = tau_slow[0] / tau_fast[0]
    gap_hi = tau_slow[-1] / tau_fast[-1]
    ax.annotate(f"ratio  {gap_lo:.1f}x $\\rightarrow$ {gap_hi:.1e}x",
                xy=(0.035, 0.045), xycoords="axes fraction",
                fontsize=8.6, va="bottom",
                bbox=dict(fc="white", ec=C_GREY, lw=0.8, alpha=0.96))

    ax.annotate("the fast clock never stops:\n"
                "the bound stays down here",
                xy=(1.0, bound[np.argmin(np.abs(tp - 1.0))]),
                xytext=(0.985, 0.05), textcoords="axes fraction",
                fontsize=8.4, color=C_ACCENT, va="bottom", ha="right",
                bbox=dict(fc="white", ec=C_ACCENT, lw=0.8, alpha=0.96),
                arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=0.9))

    ax.legend(loc="upper left", fontsize=8.4, framealpha=0.95)
    ax.set_title("(b) The step size follows the fast clock,\n"
                 "the run has to last as long as the slow one", fontsize=10.5)

    fig.suptitle("F13  What the system looks like, and the two clocks that make it expensive",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    return save_figure(fig, "F13_two_clocks", cfg, outdir)
