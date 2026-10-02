#!/usr/bin/env python3
"""Single entry point: one command reproduces every artefact.

    python run_all.py                 # all stages
    python run_all.py --stages reference eigen
    python run_all.py --list

Stages:
    reference    tight-tolerance reference + tolerance-repeat digit retention
    eigen        Jacobian spectrum, structural zero, stiffness ratio
    stability    absolute stability regions of the three methods
    integrators  self-implemented EE / RK4 / IE: uniform-step convergence study
    sweep        empirical step sweep vs the frozen-Jacobian estimate
    diagnostics  four tracks kept apart + endpoint y2(40) from the implicit method
    newtonsweep  Newton stopping-tolerance sweep (implicit Euler)
    adaptive     RK4 step-doubling controller, scaled vs absolute norm
    figures      F2, F4, F5, F6, F7, F8, F9, F10, F11, F13
                 (F12 -- the cost model -- is rendered by tools/render_p4_cost.py,
                  which joins T7 to T9 without rerunning the pipeline)
Reserved for P4: report, slides.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.analysis import adaptive_study as ADAPT             # noqa: E402
from src.analysis import convergence as CONV                 # noqa: E402
from src.analysis import diagnostics as DIAG                 # noqa: E402
from src.analysis import eigen as EIG                        # noqa: E402
from src.analysis import metrics as MET                      # noqa: E402
from src.analysis import newton_sweep as NS                  # noqa: E402
from src.analysis import reference as REF                    # noqa: E402
from src.analysis import stability_region as SR              # noqa: E402
from src.analysis import step_sweep as SW                    # noqa: E402
from src.grid import describe, output_times                  # noqa: E402
from src.integrators import make as make_method              # noqa: E402
from src.model import robertson as R                         # noqa: E402
from src.reporting import figures as FIG                     # noqa: E402
from src.reporting import tables as TAB                      # noqa: E402

STAGES = ["reference", "eigen", "stability", "integrators", "sweep",
          "diagnostics", "newtonsweep", "adaptive", "newtonstudies", "figures"]

METHOD_KEYS = ("explicit_euler", "rk4", "implicit_euler")


def method_kwargs(cfg: dict, key: str) -> dict:
    """Constructor arguments for one method, taken from the config."""
    m = cfg.get("methods", {}).get(key, {})
    if key == "implicit_euler":
        return {"newton_tol": float(m.get("newton_tol", 1.0e-12)),
                "maxiter": int(m.get("newton_maxiter", 50))}
    return {}


# ---------------------------------------------------------------------------
def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.complexfloating,)):
        return [float(o.real), float(o.imag)]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, complex):
        return [o.real, o.imag]
    raise TypeError(f"not serialisable: {type(o)}")


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


# ---------------------------------------------------------------------------
class Context:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.results = ROOT / "results"
        self.t = output_times(
            n_log=cfg["output_grid"]["n_log"],
            t_min=cfg["output_grid"]["t_min"],
            t_max=cfg["output_grid"]["t_max"],
            include_t0=cfg["output_grid"]["include_t0"],
        )
        self.ref = None
        self.spectral = None
        self.zoom = None
        self.convergence = None
        self.orders = None
        self.sweep_rows = None
        self.sweep_boundaries = None
        self.frozen = None
        self.four_track = None
        self.endpoint_rows = None
        self.endpoint_meta = None
        self.newton_rows = None
        self.newton_stats = None
        self.adaptive = None
        self.summary: dict = {}
        self.timings: dict = {}


# ---------------------------------------------------------------------------
def stage_reference(ctx: Context) -> None:
    t0 = time.perf_counter()
    bundle = REF.build_reference_set(ctx.t, ctx.cfg)
    summary, runs = bundle["summary"], bundle["runs"]
    ctx.ref = summary

    out = ctx.results / "reference"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "reference.npz",
        t=summary["t_eval"],
        y=summary["y_reference"],
        y_tight=runs["primary_tight"].y,
    )
    _dump(out / "metadata.json", {
        "grid": describe(ctx.t),
        "confirmed_digits_vs_tolerance_repeat": summary["confirmed_digits_vs_tolerance_repeat"],
        "confirmed_digits_per_component": summary["confirmed_digits_per_component"],
        "cross_check_digits": summary["cross_check_digits"],
        "runs": {k: v.meta() for k, v in runs.items()},
    })

    char = REF.characterise_reference(summary)
    ctx.summary["reference"] = char
    _dump(out / "characterisation.json", char)

    rows = [{
        "quantity": "min digits confirmed by tolerance repeat",
        "value": summary["confirmed_digits_vs_tolerance_repeat"],
    }]
    for k, v in summary["cross_check_digits"].items():
        rows.append({"quantity": f"digits vs {k}", "value": v["min_digits"]})
    rows += [{"quantity": k, "value": v} for k, v in char.items()
             if isinstance(v, (int, float)) and not isinstance(v, bool)]
    TAB.write_table(rows, "T4_reference_precision", ctx.results / "tables",
                    columns=["quantity", "value"])

    ctx.timings["reference"] = time.perf_counter() - t0


def stage_eigen(ctx: Context) -> None:
    t0 = time.perf_counter()
    assert ctx.ref is not None, "run the 'reference' stage first"
    t = ctx.ref["t_eval"]
    y = ctx.ref["y_reference"]

    spec = EIG.analyse_trajectory(t, y, zero_rtol=ctx.cfg["eigen"]["zero_eigenvalue_rtol"])
    ctx.spectral = spec

    out = ctx.results / "raw"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "spectral.npz",
        t=spec.t,
        lam_full=spec.lam_full,
        lam_reduced=spec.lam_reduced,
        is_zero=spec.is_zero,
        zero_alignment=spec.zero_alignment,
        stiffness_ratio=spec.stiffness_ratio,
        discriminant=spec.discriminant,
    )

    cons = EIG.consistency_check(t, y)
    ctx.summary["spectral_consistency"] = cons
    ctx.summary["spectral_summary"] = TAB.spectral_summary_rows(t, y)
    ctx.summary["check_values"] = TAB.check_value_rows(t, y)

    TAB.write_table(TAB.spectral_summary_rows(t, y), "T1_spectral_summary",
                    ctx.results / "tables")
    TAB.write_table(TAB.check_value_rows(t, y), "T2_brief_check_values",
                    ctx.results / "tables")

    # dense early-time reference: the fixed grid steps over the conjugate-pair
    # window, so it is located on a dedicated zoom trajectory instead
    t_zoom, y_zoom = REF.zoom_reference(
        t_lo=ctx.cfg["output_grid"]["t_min"], t_hi=3.0e-7, n=600)
    ctx.zoom = (t_zoom, y_zoom)
    window = EIG.complex_pair_window(t_zoom, y_zoom)
    ctx.summary["complex_pair_window"] = window

    pos = t > 0.0
    ctx.summary["stiffness_ratio_headline"] = {
        "S_at_first_positive_t": float(spec.stiffness_ratio[pos][0]),
        "t_of_first_positive": float(t[pos][0]),
        "S_at_t40": float(spec.stiffness_ratio[-1]),
        "min_zero_alignment": float(spec.zero_alignment[pos].min()),
        "n_states_classified_two_nonzero": int(np.count_nonzero(spec.n_nonzero[pos] == 2)),
        "n_states_in_complex_pair_window_on_grid":
            int(np.count_nonzero(spec.discriminant[pos] < 0.0)),
    }
    ctx.timings["eigen"] = time.perf_counter() - t0


def stage_stability(ctx: Context) -> None:
    t0 = time.perf_counter()
    rows = SR.summary_table()
    TAB.write_table(rows, "T3_stability_methods", ctx.results / "tables",
                    columns=["method", "order", "real_axis_interval", "A_stable",
                             "max_stable_real_h_lambda", "h_limit_for_lambda_max_3p39e3"])
    ctx.summary["stability_methods"] = rows
    ctx.timings["stability"] = time.perf_counter() - t0


def stage_integrators(ctx: Context) -> None:
    """Uniform-step convergence study for the three self-implemented methods."""
    t0 = time.perf_counter()
    assert ctx.ref is not None, "run the 'reference' stage first"
    t = ctx.ref["t_eval"]
    y = ctx.ref["y_reference"]
    crit = MET.Criteria.from_config(ctx.cfg)

    h_levels = list(ctx.cfg["uniform"]["h_levels"])
    t_end = float(ctx.cfg["uniform"]["t_end"])

    rows_by_method: dict[str, list[dict]] = {}
    for key in METHOD_KEYS:
        rows = []
        for h in h_levels:
            integ = make_method(key, **method_kwargs(ctx.cfg, key))
            traj = integ.integrate(float(h), t_end=t_end,
                                   divergence_bound=crit.divergence_bound)
            row = MET.assess_run(traj, t, y, crit)
            row["method_key"] = key
            rows.append(row)
        rows_by_method[key] = rows

    ctx.convergence = rows_by_method
    nominal = {"explicit_euler": 1, "rk4": 4, "implicit_euler": 1}
    floor = float(ctx.cfg["criteria"]["reference_error_floor"])
    orders_late = {k: CONV.observed_order(v, error_key=ctx.cfg["convergence"]["error_key"],
                                          error_floor=floor, method_key=k)
                   for k, v in rows_by_method.items()}
    orders_full = {k: CONV.observed_order(v, error_key=ctx.cfg["convergence"]["error_key_secondary"],
                                          error_floor=floor, method_key=k)
                   for k, v in rows_by_method.items()}
    ctx.orders = (orders_late, orders_full)

    TAB.write_table(TAB.convergence_rows(rows_by_method), "T7_convergence",
                    ctx.results / "tables", columns=TAB.CONVERGENCE_COLUMNS)
    TAB.write_table(TAB.observed_order_rows(orders_late, orders_full, nominal),
                    "T8_observed_order", ctx.results / "tables")
    TAB.write_table(TAB.cost_rows(rows_by_method), "T9_cost",
                    ctx.results / "tables", columns=TAB.COST_COLUMNS)

    # raw error curves, for replotting without re-integrating
    raw = ctx.results / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        raw / "convergence.npz",
        **{f"{k}__h": np.array([r["h_used"] for r in v]) for k, v in rows_by_method.items()},
        **{f"{k}__err_late": np.array([r["error_late"] for r in v]) for k, v in rows_by_method.items()},
        **{f"{k}__err_full": np.array([r["error_full"] for r in v]) for k, v in rows_by_method.items()},
        **{f"{k}__stable": np.array([r["stable"] for r in v]) for k, v in rows_by_method.items()},
    )

    ctx.summary["convergence"] = {
        "h_levels": h_levels,
        "t_end": t_end,
        "observed_order_late": {k: {kk: v[kk] for kk in
                                    ("slope", "slope_stderr", "r2", "n_levels_used", "h_window")}
                                for k, v in orders_late.items()},
        "observed_order_full_grid": {k: {kk: v[kk] for kk in
                                         ("slope", "slope_stderr", "r2", "n_levels_used")}
                                     for k, v in orders_full.items()},
        "runs": TAB.convergence_rows(rows_by_method),
    }
    ctx.timings["integrators"] = time.perf_counter() - t0


def stage_sweep(ctx: Context) -> None:
    """Empirical step sweep, and the frozen-Jacobian estimate it tests."""
    t0 = time.perf_counter()
    assert ctx.ref is not None
    t = ctx.ref["t_eval"]
    y = ctx.ref["y_reference"]
    crit = MET.Criteria.from_config(ctx.cfg)
    scfg = ctx.cfg["step_sweep"]

    hs = SW.h_grid(scfg["h_min"], scfg["h_max"], scfg["n_points"])
    rows = SW.run_sweep(scfg["methods"], hs, t, y, ctx.cfg, crit,
                        t_end=float(ctx.cfg["uniform"]["t_end"]))
    boundaries = SW.boundary_summary(rows)
    frozen = SW.frozen_jacobian_bound(t, y)

    ctx.sweep_rows = rows
    ctx.sweep_boundaries = boundaries
    ctx.frozen = frozen

    TAB.write_table(TAB.step_sweep_rows(rows), "T5_step_sweep",
                    ctx.results / "tables", columns=TAB.SWEEP_COLUMNS)
    TAB.write_table(TAB.sweep_boundary_rows(boundaries, frozen),
                    "T6_sweep_boundaries", ctx.results / "tables")

    ctx.summary["frozen_jacobian_bound"] = frozen
    ctx.summary["sweep_boundaries"] = boundaries
    ctx.summary["step_sweep"] = SW.strip_trajectories(rows)
    ctx.timings["sweep"] = time.perf_counter() - t0


def stage_diagnostics(ctx: Context) -> None:
    """Four tracks kept apart, plus the terminal value from the implicit method.

    Reuses the step-sweep rows (all four diagnostics are already columns there)
    and adds the endpoint table the brief asks for explicitly.
    """
    t0 = time.perf_counter()
    assert ctx.sweep_rows is not None, "run the 'sweep' stage first"
    assert ctx.ref is not None
    char = ctx.summary["reference"]

    four = DIAG.first_failing_track(ctx.sweep_rows)
    ctx.four_track = four
    TAB.write_table(DIAG.four_track_rows(ctx.sweep_rows), "T13_four_track",
                    ctx.results / "tables", columns=DIAG.FOUR_TRACK_COLUMNS)
    TAB.write_table(TAB.four_track_boundary_rows(four), "T14_four_track_boundaries",
                    ctx.results / "tables")

    rows, meta = DIAG.endpoint_rows(ctx.cfg, ctx.t, ctx.ref["y_reference"], char)
    ctx.endpoint_rows = rows
    ctx.endpoint_meta = meta
    TAB.write_table(rows, "T10_y2_endpoint", ctx.results / "tables",
                    columns=DIAG.ENDPOINT_COLUMNS)

    ctx.summary["four_track_boundaries"] = four
    ctx.summary["endpoint"] = meta
    ctx.summary["endpoint_rows"] = rows
    ctx.timings["diagnostics"] = time.perf_counter() - t0


def stage_newtonsweep(ctx: Context) -> None:
    """Newton stopping-tolerance sweep: the algebraic error must be negligible."""
    t0 = time.perf_counter()
    assert ctx.ref is not None
    t, y = ctx.ref["t_eval"], ctx.ref["y_reference"]
    crit = MET.Criteria.from_config(ctx.cfg)

    rows = NS.sweep(ctx.cfg, t, y, crit)
    stats = NS.analyse(rows, max_inflation=float(ctx.cfg["newton_sweep"]["max_inflation"]))
    ctx.newton_rows = rows
    ctx.newton_stats = stats

    TAB.write_table(TAB.newton_sweep_rows(rows), "T11_newton_sweep",
                    ctx.results / "tables", columns=TAB.NEWTON_SWEEP_COLUMNS)
    TAB.write_table(
        [{"quantity": k, "value": v} for k, v in stats.items() if not isinstance(v, dict)],
        "T11b_newton_sweep_summary", ctx.results / "tables",
        columns=["quantity", "value"])

    ctx.summary["newton_sweep"] = stats
    ctx.summary["newton_sweep_rows"] = rows
    ctx.timings["newtonsweep"] = time.perf_counter() - t0


def stage_adaptive(ctx: Context) -> None:
    """RK4 step-doubling controller at several tolerances, both error norms."""
    t0 = time.perf_counter()
    assert ctx.ref is not None
    t, y = ctx.ref["t_eval"], ctx.ref["y_reference"]
    crit = MET.Criteria.from_config(ctx.cfg)

    levels = ADAPT.run_levels(ctx.cfg, t, y, crit, keep_trajectories=True)
    ctx.adaptive = levels

    TAB.write_table(TAB.adaptive_rows(levels), "T12_adaptive",
                    ctx.results / "tables", columns=TAB.ADAPTIVE_COLUMNS)

    ctx.summary["adaptive"] = {
        norm: [{k: v for k, v in r.items() if k != "trajectory"} for r in rows]
        for norm, rows in levels.items() if isinstance(rows, list)
    }
    ctx.summary["adaptive"]["frozen_rk4_bound"] = levels["frozen_rk4_bound"]
    ctx.timings["adaptive"] = time.perf_counter() - t0


def stage_figures(ctx: Context) -> None:
    t0 = time.perf_counter()
    assert ctx.ref is not None and getattr(ctx, "spectral", None) is not None
    t, y = ctx.ref["t_eval"], ctx.ref["y_reference"]
    figdir = ctx.results / "figures"

    written = []
    written += FIG.fig_reference_solution(t, y, ctx.cfg, figdir)
    written += FIG.fig_stability_regions(t, y, ctx.cfg, figdir)
    written += FIG.fig_eigenvalues(
        t, y, ctx.spectral, ctx.cfg, figdir,
        zoom=getattr(ctx, "zoom", None),
        zero_rtol=ctx.cfg["eigen"]["zero_eigenvalue_rtol"],
    )
    written += FIG.fig_stiffness_ratio(t, y, ctx.spectral, ctx.cfg, figdir)
    written += FIG.fig_clock_scales(t, y, ctx.spectral, ctx.cfg, figdir)

    if getattr(ctx, "sweep_rows", None) and getattr(ctx, "endpoint_meta", None):
        pair = _explicit_euler_pair(ctx.sweep_rows)
        written += FIG.fig_frozen_bound_vs_sweep(
            ctx.sweep_rows, ctx.frozen, ctx.cfg, figdir, pair=pair)
        written += FIG.fig_four_track_diagnostics(
            ctx.sweep_rows, ctx.frozen, ctx.cfg, figdir,
            ref_y2_t40=float(ctx.endpoint_meta["reference_y2_at_t40"]),
            ref_abs_tol=float(ctx.endpoint_meta["reference_max_abs_diff"]),
        )
    if getattr(ctx, "convergence", None):
        orders_late, orders_full = ctx.orders
        written += FIG.fig_convergence(ctx.convergence, orders_late, orders_full,
                                       ctx.cfg, figdir)
    if getattr(ctx, "newton_rows", None):
        written += FIG.fig_newton_tolerance_sweep(ctx.newton_rows, ctx.newton_stats,
                                                  ctx.cfg, figdir)
    if getattr(ctx, "adaptive", None):
        written += FIG.fig_adaptive(ctx.adaptive, ctx.cfg, figdir)

    ctx.summary["figures"] = [str(p.relative_to(ROOT)) for p in written]
    ctx.timings["figures"] = time.perf_counter() - t0


def stage_newtonstudies(ctx: Context) -> None:
    """The three implicit-method studies, kept in step with everything else.

    They live in ``tools/render_newton_studies.py`` because they need no state
    from the other stages -- they re-read ``results/reference/reference.npz``
    themselves.  Running them from here anyway is the point: the README claims
    one command reproduces every table and figure, and a claim like that is
    either true or it is a lie.
    """
    t0 = time.perf_counter()
    import importlib.util

    path = ROOT / "tools" / "render_newton_studies.py"
    spec = importlib.util.spec_from_file_location("render_newton_studies", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ctx.summary["newton_studies"] = mod.main()
    ctx.timings["newtonstudies"] = time.perf_counter() - t0


def _explicit_euler_pair(rows, key="explicit_euler"):
    """The two adjacent sweep points that bracket the stability boundary."""
    rs = [r for r in rows if r["method_key"] == key]
    stable = [r for r in rs if r["stable"]]
    diverged = [r for r in rs if not r["stable"]]
    if not stable or not diverged:
        return None
    a = max(stable, key=lambda r: r["h_used"])
    b = min(diverged, key=lambda r: r["h_used"])
    return [(f"$h={a['h_used']:.3g}$", a["trajectory"], "-"),
            (f"$h={b['h_used']:.3g}$", b["trajectory"], "--")]


REGISTRY = {
    "reference": stage_reference,
    "eigen": stage_eigen,
    "stability": stage_stability,
    "integrators": stage_integrators,
    "sweep": stage_sweep,
    "diagnostics": stage_diagnostics,
    "newtonsweep": stage_newtonsweep,
    "adaptive": stage_adaptive,
    "newtonstudies": stage_newtonstudies,
    "figures": stage_figures,
}


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    ap.add_argument("--stages", nargs="*", default=STAGES, choices=STAGES)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)

    if args.list:
        print("stages:", ", ".join(STAGES))
        return 0

    cfg = yaml.safe_load(Path(args.config).read_text())
    ctx = Context(cfg)

    print(f"comparison grid: {describe(ctx.t)}")
    for name in args.stages:
        print(f"[stage] {name} ...", flush=True)
        REGISTRY[name](ctx)

    _dump(ctx.results / "summary.json", ctx.summary)
    _dump(ctx.results / "timings.json", ctx.timings)

    print("\n=== headline numbers ===")
    ref = ctx.summary.get("reference", {})
    if ref:
        print(f"  confirmed reference digits     : {ref['confirmed_digits']:.1f}")
        print(f"  y(40) from our reference       : {np.array2string(np.array(ref['y_at_t40']), precision=10)}")
        print(f"  max |y1+y2+y3-1| (reference)   : {ref['max_invariant_defect']:.3e}")
    sp = ctx.summary.get("stiffness_ratio_headline", {})
    if sp:
        print(f"  S(t) first positive time       : {sp['S_at_first_positive_t']:.4g} at t={sp['t_of_first_positive']:g}")
        print(f"  S(40)                          : {sp['S_at_t40']:.4g}")
        print(f"  states classified 2 non-zero   : {sp['n_states_classified_two_nonzero']}")
    win = ctx.summary.get("complex_pair_window", {})
    if win:
        print(f"  complex-pair window (dense)    : {win.get('n_negative_discriminant', 0)} states, "
              f"t in [{win.get('t_lo', float('nan')):.3e}, {win.get('t_hi', float('nan')):.3e}]")
        print(f"  max |Im|/|Re| in window        : {win.get('max_abs_im_over_abs_re', float('nan')):.2e}")
        print(f"  window states on the fixed grid: {sp.get('n_states_in_complex_pair_window_on_grid', 0)}")
    cv = ctx.summary.get("spectral_consistency", {})
    if cv:
        print(f"  m2(J) == det(B_red)            : {'PASS' if cv['passes'] else 'FAIL'}"
              f"  (max rel diff {cv['max_rel_diff_m2_vs_det_reduced']:.2e})")
    ol = (ctx.summary.get("convergence") or {}).get("observed_order_late", {})
    if ol:
        print("\n  --- observed convergence order (t >= t_cut) ---")
        nom = {"explicit_euler": 1, "rk4": 4, "implicit_euler": 1}
        for k, v in ol.items():
            print(f"  {k:<16s} nominal {nom[k]}"
                  f"  fitted {v['slope']:.3f} +/- {v['slope_stderr']:.3f}"
                  f"  (R^2={v['r2']:.4f}, {v['n_levels_used']} levels)")
    print("\n  --- empirical stability boundary vs frozen-Jacobian estimate ---")
    for e in (ctx.summary.get("sweep_boundaries") or []):
        fb = (ctx.summary.get("frozen_jacobian_bound") or {}).get("per_method", {}).get(e["method_key"], {})
        print(f"  {e['method']:<16s} empirical h_max {e['h_max_stable']:.4g}"
              f"  frozen bound {fb.get('h_bound', float('nan')):.4g}"
              f"  first divergence at h={e['h_min_diverged']:.4g}")

    ep = ctx.summary.get("endpoint") or {}
    if ep:
        print("\n  --- terminal value y2(40) from our implicit method ---")
        print(f"  our reference y2(40)          : {ep['reference_y2_at_t40']:.10e}"
              f"  ({ep['reference_confirmed_digits']:.1f} digits;"
              f" abs uncertainty {ep['reference_max_abs_diff']:.1e})")
        print(f"  quotable from our reference   : {ep['our_y2_confirmable_value']:.10e}")
        for r in (ctx.summary.get("endpoint_rows") or []):
            print(f"  IE h={r['h_used']:.3e} tol={r['newton_tol']:.0e}"
                  f"  y2(40)={r['y2_at_t40']:.10e}"
                  f"  {r['digits_vs_reference']:.2f} digits vs reference")
        print(f"  brief orientation value       : {ep['brief_y_at_t40']}")
        print(f"  digits of it our reference confirms (y1,y2,y3): "
              f"{[round(v, 2) for v in ep['brief_digits_confirmed_by_our_reference']]}")

    nsw = ctx.summary.get("newton_sweep") or {}
    if nsw:
        lo = nsw["loose_end"]
        print("\n  --- Newton stopping-tolerance sweep ---")
        print(f"  discretisation plateau        : {nsw['error_plateau']:.3e}"
              f"  (at tol={nsw['error_plateau_tol']:.0e})")
        print(f"  recommended stopping tolerance: {nsw['recommended_newton_tol']:.0e}"
              f"  (inflation {nsw['recommended_inflation']:.3f},"
              f" {nsw['recommended_newton_iters_per_step']:.2f} updates/step)")
        print(f"  first tolerance that breaks it: {nsw['first_tolerance_that_breaks_it']:.0e}"
              f"  (inflation {nsw['first_breaking_inflation']:.2f})")
        print(f"  predictor residual scale h^2  : {nsw['predictor_residual_scaling_h2']:.1e}")
        print(f"  loose end tol={lo['tol']:.0e}: error {lo['error_full']:.2e},"
              f" min y {lo['min_component']:+.2f},"
              f" mass defect {lo['mass_defect_max']:.2e},"
              f" {lo['fraction_steps_without_newton'] * 100:.0f}% of steps skipped Newton")

    ad = ctx.summary.get("adaptive") or {}
    if ad.get("mixed"):
        print("\n  --- adaptive step size (RK4 step doubling) ---")
        print(f"  frozen RK4 bound              : {ad['frozen_rk4_bound']:.4g}")
        for r in ad["mixed"]:
            print(f"  mixed tol={r['tol']:.0e}: err {r['error_full']:.2e},"
                  f" h_med {r['h_median_controller']:.3e}"
                  f" (sub-step = {r['h_substep_median_over_frozen']:.2f} x frozen bound),"
                  f" err/tol med {r['err_over_tol_median']:.2f},"
                  f" acc/rej {r['n_accepted']}/{r['n_rejected']}")
        for r in ad.get("absolute_inf", []):
            print(f"  abs-inf tol={r['tol']:.0e}: err {r['error_full']:.2e},"
                  f" min y {r['min_component']:+.2e},"
                  f" reached t={r['t_covered']:.1f}"
                  f"{'  ABORTED: ' + r['abort_reason'] if r['aborted'] else ''}")

    for name, dt in ctx.timings.items():
        print(f"  [timing] {name:<12s} {dt:7.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
