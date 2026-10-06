"""Compare mixed-norm adaptive RK4 with a local stability-capped variant.

The two controllers share every parameter and numerical operation except that
the stability-aware variant caps the outer step using the reduced-Jacobian
eigenvalues and the classical RK4 stability polynomial.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


T0 = 0.0
T1 = 40.0
Y0 = np.array([1.0, 0.0, 0.0])
TOLERANCES = (1.0e-10, 1.0e-8, 1.0e-6, 1.0e-4)
ATOL_RATIO = np.array([1.0e-10, 1.0e-13, 1.0e-10])
H0 = 1.0e-6
H_MIN = 1.0e-12
H_MAX = 5.0e-2
ERROR_SAFETY = 0.9
STABILITY_SAFETY = 0.9
SHRINK_LIMIT = 0.2
GROWTH_LIMIT = 5.0
RICHARDSON_DIVISOR = 15.0

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "output" / "raw"
FIGURE_DIR = ROOT / "output" / "figures"


def rhs(_t: float, y: np.ndarray) -> np.ndarray:
    y1, y2, y3 = y
    return np.array(
        [
            -0.04 * y1 + 1.0e4 * y2 * y3,
            0.04 * y1 - 1.0e4 * y2 * y3 - 3.0e7 * y2**2,
            3.0e7 * y2**2,
        ]
    )


def reduced_jacobian(y: np.ndarray) -> np.ndarray:
    y1, y2, _ = y
    return np.array(
        [
            [-0.04 - 1.0e4 * y2, 1.0e4 * (1.0 - y1 - 2.0 * y2)],
            [
                0.04 + 1.0e4 * y2,
                1.0e4 * (y1 + 2.0 * y2 - 1.0) - 6.0e7 * y2,
            ],
        ]
    )


def rk4_stability_polynomial(z: np.ndarray) -> np.ndarray:
    return 1.0 + z + z**2 / 2.0 + z**3 / 6.0 + z**4 / 24.0


def outer_step_is_stable(h_outer: float, eigenvalues: np.ndarray) -> bool:
    values = rk4_stability_polynomial(0.5 * h_outer * eigenvalues)
    return bool(np.max(np.abs(values)) <= 1.0 + 32.0 * np.finfo(float).eps)


def local_stability_cap(y: np.ndarray, h_upper: float = H_MAX) -> float:
    """Largest outer step up to h_upper whose two RK4 half-steps are stable."""
    eigenvalues = np.linalg.eigvals(reduced_jacobian(y))
    if outer_step_is_stable(h_upper, eigenvalues):
        return h_upper

    low = 0.0
    high = h_upper
    for _ in range(60):
        middle = 0.5 * (low + high)
        if outer_step_is_stable(middle, eigenvalues):
            low = middle
        else:
            high = middle
    return low


def rk4_step(t: float, y: np.ndarray, h: float) -> np.ndarray:
    k1 = rhs(t, y)
    k2 = rhs(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = rhs(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = rhs(t + h, y + h * k3)
    return y + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def mixed_error_indicator(error: np.ndarray, y: np.ndarray) -> float:
    return float(np.max(np.abs(error) / (ATOL_RATIO + np.abs(y))))


def controller_factor(tolerance: float, indicator: float) -> float:
    if indicator == 0.0:
        return GROWTH_LIMIT
    raw = ERROR_SAFETY * (tolerance / indicator) ** 0.2
    return float(np.clip(raw, SHRINK_LIMIT, GROWTH_LIMIT))


@dataclass
class RunSummary:
    controller: str
    tolerance: float
    reached_t40: bool
    accepted_steps: int
    rejected_steps: int
    rhs_calls: int
    jacobian_eigenvalue_evaluations: int
    global_error: float
    endpoint_error: float
    minimum_concentration: float
    final_y2: float
    maximum_outer_step: float
    cap_active_accepted_percent: float
    minimum_h_over_hstab: float
    median_h_over_hstab: float
    maximum_h_over_hstab: float
    accepted_steps_above_hstab: int
    trial_proposals_above_stability_safety_cap: int


@dataclass
class RunData:
    summary: RunSummary
    times: np.ndarray
    states: np.ndarray
    steps: np.ndarray
    stability_caps: np.ndarray
    cap_active: np.ndarray


def integrate(tolerance: float, stability_aware: bool) -> RunData:
    t = T0
    y = Y0.copy()
    h_proposed = H0
    proposal_was_capped = False
    times = [t]
    states = [y.copy()]
    steps = []
    stability_caps = []
    cap_active_flags = []
    rejected = 0
    rhs_calls = 0
    eigenvalue_evaluations = 0
    proposals_above_cap = 0

    while t < T1:
        h_stab = local_stability_cap(y)
        if stability_aware:
            eigenvalue_evaluations += 1
        safe_cap = STABILITY_SAFETY * h_stab
        raw_trial = min(h_proposed, H_MAX, T1 - t)
        proposal_above_cap = raw_trial > safe_cap * (1.0 + 1.0e-13)
        h = min(raw_trial, safe_cap) if stability_aware else raw_trial
        cap_active_this_trial = stability_aware and (
            proposal_was_capped or proposal_above_cap
        )
        if proposal_above_cap or proposal_was_capped:
            proposals_above_cap += 1

        if h < H_MIN:
            raise RuntimeError(f"step fell below h_min at t={t:.16g}")

        whole = rk4_step(t, y, h)
        half = rk4_step(t, y, 0.5 * h)
        fine = rk4_step(t + 0.5 * h, half, 0.5 * h)
        rhs_calls += 12

        error = (fine - whole) / RICHARDSON_DIVISOR
        indicator = mixed_error_indicator(error, y)
        h_error = h * controller_factor(tolerance, indicator)

        if indicator <= tolerance and np.all(np.isfinite(fine)):
            t = min(t + h, T1)
            y = fine
            times.append(t)
            states.append(y.copy())
            steps.append(h)
            stability_caps.append(h_stab)
            cap_active_flags.append(cap_active_this_trial)

            if stability_aware and t < T1:
                next_stability_cap = local_stability_cap(y)
                eigenvalue_evaluations += 1
                next_safe_cap = STABILITY_SAFETY * next_stability_cap
                proposal_was_capped = h_error > next_safe_cap * (1.0 + 1.0e-13)
                h_proposed = min(h_error, next_safe_cap, H_MAX)
            else:
                proposal_was_capped = False
                h_proposed = min(h_error, H_MAX)
        else:
            rejected += 1
            if stability_aware:
                proposal_was_capped = h_error > safe_cap * (1.0 + 1.0e-13)
                h_proposed = min(max(H_MIN, h_error), safe_cap, H_MAX)
            else:
                proposal_was_capped = False
                h_proposed = max(H_MIN, h_error)

    times_array = np.asarray(times)
    states_array = np.asarray(states)
    steps_array = np.asarray(steps)
    caps_array = np.asarray(stability_caps)
    ratios = steps_array / caps_array
    cap_flags_array = np.asarray(cap_active_flags, dtype=bool)

    common_grid = np.concatenate(([0.0], np.geomspace(1.0e-8, T1, 200)))
    numerical_common = cubic_hermite_values(times_array, states_array, common_grid)
    reference_common = reference_solution(common_grid)
    global_error = float(np.max(np.abs(numerical_common - reference_common)))
    endpoint_error = float(abs(states_array[-1, 1] - reference_common[-1, 1]))

    summary = RunSummary(
        controller="stability_aware" if stability_aware else "standard_mixed",
        tolerance=tolerance,
        reached_t40=bool(abs(times_array[-1] - T1) <= 8.0 * np.finfo(float).eps * T1),
        accepted_steps=len(steps_array),
        rejected_steps=rejected,
        rhs_calls=rhs_calls,
        jacobian_eigenvalue_evaluations=eigenvalue_evaluations,
        global_error=global_error,
        endpoint_error=endpoint_error,
        minimum_concentration=float(np.min(states_array)),
        final_y2=float(states_array[-1, 1]),
        maximum_outer_step=float(np.max(steps_array)),
        cap_active_accepted_percent=float(100.0 * np.mean(cap_flags_array)),
        minimum_h_over_hstab=float(np.min(ratios)),
        median_h_over_hstab=float(np.median(ratios)),
        maximum_h_over_hstab=float(np.max(ratios)),
        accepted_steps_above_hstab=int(np.count_nonzero(ratios > 1.0 + 1.0e-12)),
        trial_proposals_above_stability_safety_cap=proposals_above_cap,
    )
    return RunData(
        summary=summary,
        times=times_array,
        states=states_array,
        steps=steps_array,
        stability_caps=caps_array,
        cap_active=cap_flags_array,
    )


def cubic_hermite_values(
    times: np.ndarray, states: np.ndarray, evaluation_times: np.ndarray
) -> np.ndarray:
    values = np.empty((len(evaluation_times), states.shape[1]))
    interval_indices = np.searchsorted(times, evaluation_times, side="right") - 1
    interval_indices = np.clip(interval_indices, 0, len(times) - 2)
    for output_index, (target, interval_index) in enumerate(
        zip(evaluation_times, interval_indices)
    ):
        if target == times[-1]:
            values[output_index] = states[-1]
            continue
        left_t = times[interval_index]
        right_t = times[interval_index + 1]
        step = right_t - left_t
        theta = (target - left_t) / step
        left_y = states[interval_index]
        right_y = states[interval_index + 1]
        left_f = rhs(left_t, left_y)
        right_f = rhs(right_t, right_y)
        values[output_index] = (
            (2.0 * theta**3 - 3.0 * theta**2 + 1.0) * left_y
            + (theta**3 - 2.0 * theta**2 + theta) * step * left_f
            + (-2.0 * theta**3 + 3.0 * theta**2) * right_y
            + (theta**3 - theta**2) * step * right_f
        )
    return values


def reference_solution(times: np.ndarray) -> np.ndarray:
    solution = solve_ivp(
        rhs,
        (T0, T1),
        Y0,
        method="Radau",
        t_eval=times,
        rtol=1.0e-12,
        atol=np.array([1.0e-16, 1.0e-22, 1.0e-16]),
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y.T


def write_outputs(runs: list[RunData]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(runs[0].summary))
    with (RAW_DIR / "stability_aware_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(run.summary) for run in runs)

    with (RAW_DIR / "stability_aware_comparison.json").open("w") as stream:
        json.dump([asdict(run.summary) for run in runs], stream, indent=2)

    for run in runs:
        label = f"{run.summary.controller}_tol_{run.summary.tolerance:.0e}"
        np.savez_compressed(
            RAW_DIR / f"{label}.npz",
            times=run.times,
            states=run.states,
            steps=run.steps,
            stability_caps=run.stability_caps,
            cap_active=run.cap_active,
        )


def make_figure(runs: list[RunData]) -> None:
    selected_tolerance = 1.0e-6
    standard = next(
        run
        for run in runs
        if run.summary.controller == "standard_mixed"
        and run.summary.tolerance == selected_tolerance
    )
    aware = next(
        run
        for run in runs
        if run.summary.controller == "stability_aware"
        and run.summary.tolerance == selected_tolerance
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)
    axes[0].loglog(standard.times[1:], standard.steps, label="standard mixed", lw=1.2)
    axes[0].loglog(aware.times[1:], aware.steps, label="stability-aware", lw=1.2)
    axes[0].loglog(
        aware.times[1:],
        STABILITY_SAFETY * aware.stability_caps,
        label=r"$0.9h_{\mathrm{stab}}$",
        color="black",
        ls="--",
        lw=1.0,
    )
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("outer step size")
    axes[0].set_title(r"Step histories at tolerance $10^{-6}$")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend()

    for controller, marker, label in (
        ("standard_mixed", "o", "standard mixed"),
        ("stability_aware", "s", "stability-aware"),
    ):
        subset = [run.summary for run in runs if run.summary.controller == controller]
        axes[1].loglog(
            [row.global_error for row in subset],
            [row.rhs_calls for row in subset],
            marker=marker,
            label=label,
        )
        for row in subset:
            axes[1].annotate(
                f"{row.tolerance:.0e}",
                (row.global_error, row.rhs_calls),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    axes[1].set_xlabel(r"full-state error $E_{\mathrm{global}}$")
    axes[1].set_ylabel("RHS evaluations")
    axes[1].set_title("Work--precision comparison")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend()

    fig.savefig(FIGURE_DIR / "F17_stability_aware_adaptive_rk4.pdf")
    fig.savefig(FIGURE_DIR / "F17_stability_aware_adaptive_rk4.png", dpi=220)
    plt.close(fig)


def main() -> None:
    runs = []
    for tolerance in TOLERANCES:
        runs.append(integrate(tolerance, stability_aware=False))
        runs.append(integrate(tolerance, stability_aware=True))
    write_outputs(runs)
    make_figure(runs)
    for run in runs:
        print(json.dumps(asdict(run.summary), sort_keys=True))


if __name__ == "__main__":
    main()
