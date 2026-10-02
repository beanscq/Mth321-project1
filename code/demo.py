"""一键跑通全部核心结论。

    python demo.py            # 完整版，约 2 分钟
    python demo.py --quick    # 快速版，约 30 秒（收敛阶梯只跑 3 层）

输出：控制台报告 + results/ 下的两张图。
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

import analysis
import model
import solvers

OUT = pathlib.Path(__file__).resolve().parent / "results"


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------------
def section_1_selfcheck() -> None:
    rule("1. 模型自检：守恒恒等式与解析 Jacobian")
    rng = np.random.default_rng(0)
    worst_sum = 0.0
    worst_jac = 0.0
    for _ in range(200):
        y = rng.random(3)
        y = y / y.sum()                                  # 落在守恒面上
        f_sum = float(np.max(np.abs(model.f(0.0, y).sum())))
        worst_sum = max(worst_sum, f_sum / max(np.max(np.abs(model.f(0.0, y))), 1e-300))
        h = 1e-7
        fd = np.column_stack([(model.f(0.0, y + h * e) - model.f(0.0, y - h * e)) / (2 * h)
                              for e in np.eye(3)])
        J = model.jacobian(0.0, y)
        # 用相对误差：J 的元素量级到 6e7，绝对值本身没有可比性
        worst_jac = max(worst_jac, float(np.max(np.abs(fd - J)) / np.max(np.abs(J))))
    print(f"  守恒面上随机 200 个状态：相对 |(1,1,1)·f| / |f|_max 最大 = {worst_sum:.3e}")
    print(f"  解析 Jacobian 对比中心差分：最大相对偏差 = {worst_jac:.3e}")
    print("  -> 守恒律是这个系统的代数性质，任何相容格式都自动满足它。")


def section_2_stiffness(y_ref: np.ndarray, t_grid: np.ndarray) -> tuple:
    rule("2. 刚性有多强：谱、刚性比、以及它出现在什么时候")
    lam = model.eigenvalues_nonzero(y_ref[-1])
    S40 = model.stiffness_ratio(y_ref[-1])
    abs_max, t_at = analysis.max_abs_lambda(t_grid, y_ref)
    print(f"  t = 40 处 J 的非零特征值：{lam.real[0]:.4f}, {lam.real[1]:.4f}")
    print(f"  t = 40 处刚性比 S = |Re lam_fast| / |Re lam_slow| = {S40:.4e}")
    print(f"  整条轨迹上 |Re lam|_max = {abs_max:.4f}，出现在 t = {t_at:.4f}")
    print("  -> 初值 t = 0 处三个分量完全退化（慢特征值也是 0），用初值的谱估步长会")
    print("     严重高估可用的 h；必须取整条轨迹上的最大值。")
    return abs_max, S40


def section_3_step_limit(abs_max: float, quick: bool) -> None:
    rule("3. 显式方法的稳定步长上界：理论 vs 实测")
    print(f"  |Re lam|_max = {abs_max:.4f}，冻结谱给出的上界 h <= limit / |lam|_max：")
    for key, d in analysis.stability_bound(abs_max).items():
        if np.isfinite(d["h_bound"]):
            print(f"    {d['label']:12} 负实轴稳定区间端点 {d['real_axis_limit']:.4f}"
                  f"  ->  h < {d['h_bound']:.4e}")
        else:
            print(f"    {d['label']:12} A 稳定，谱不限制步长（h_bound = inf）")

    n = 7 if quick else 11
    hs = np.logspace(np.log10(1.5e-4), np.log10(3.0e-3), n)
    stable = lambda h: not solvers.make("explicit_euler").integrate(h=h).stats.aborted
    ok = [h for h in hs if stable(h)]
    bad = [h for h in hs if not stable(h)]

    print(f"\n  实测（显式 Euler，{n} 个步长先粗扫，再对分收紧边界）：")
    lo = max(ok) if ok else np.nan
    hi = min(bad) if bad else np.nan
    if np.isfinite(lo) and np.isfinite(hi):
        # 稳定/发散的交界处对分，把区间收紧到 1% 以内
        for _ in range(6):
            mid = np.sqrt(lo * hi)
            if stable(mid):
                lo = mid
            else:
                hi = mid
        print(f"    最大稳定步长   h = {lo:.6e}")
        print(f"    最小发散步长   h = {hi:.6e}   （区间已收窄到 {100 * (hi / lo - 1):.1f}%）")
        print("  -> 冻结谱给的是**充分条件**（满足它一定稳），只是偏保守：")
        print("     |lambda|_max 只在 t 接近 40 的最后几步才出现，而最快模态在此之前")
        print("     一直被强阻尼，末段已经没有足够时间把它长到发散判据，所以实测阈值")
        print(f"     比理论值高约 {(lo / analysis.REAL_AXIS_LIMIT['explicit_euler'] * abs_max - 1) * 100:.0f}%。")


def section_4_convergence(t_grid: np.ndarray, y_ref: np.ndarray,
                          quick: bool) -> dict:
    rule("4. 实测收敛阶（每层都是定步长，误差比对高精度参考解）")
    hs = [5.0e-4, 2.5e-4, 1.25e-4] if quick else \
         [1.0e-3, 5.0e-4, 2.5e-4, 1.25e-4, 6.25e-5]
    print(f"  步长阶梯 h = {', '.join(f'{x:.3g}' for x in hs)}")
    print()
    print(f"  {'方法':<11}{'h':>10}{'步数':>10}{'误差 error_late':>18}")
    print("  " + "-" * 49)

    rows_by_method, fits = {}, {}
    for key in solvers.all_keys():
        rows = []
        for h in hs:
            traj = solvers.make(key).integrate(h=h)
            rows.append(analysis.assess(traj, t_grid, y_ref))
        fit = analysis.observed_order(rows)
        rows_by_method[key] = rows
        fits[key] = fit
        for r in rows:
            mark = "" if r["stable"] else "   发散，剔除"
            print(f"  {r['method']:<11}{r['h_used']:>10.3e}{r['n_steps']:>10}"
                  f"{r['error_late']:>18.3e}{mark}")
        print(f"  {'':<11}{'least-squares':>10}  {fit['n_levels_used']} 层参与拟合"
              f"   实测阶 {fit['slope']:.3f}   R^2 = {fit['r2']:.4f}"
              f"   h 窗口 [{fit['h_window'][0]:.2e}, {fit['h_window'][1]:.2e}]")
        print()

    print("  理论阶：显式 Euler 1，RK4 4，隐式 Euler 1")
    for key, fit in fits.items():
        print(f"    {solvers.label(key):<12} 实测 {fit['slope']:.3f}"
              f"   理论 {solvers.METHODS[key][2]}")
    print("  -> RK4 的实测阶能到 4，前提是稠密输出用三次 Hermite；")
    print("     若换成线性插值，插值误差 O(h^2) 会把这一栏压到 2。")
    return rows_by_method


def section_5_four_tracks(t_grid: np.ndarray, y_ref: np.ndarray) -> None:
    rule("5. 四道诊断：守恒漂亮不等于算对了")
    print("  统一取 h = 3e-3（远大于显式方法的稳定上界）：")
    print()
    print(f"  {'方法':<12}{'走到':>9}{'守恒残差':>12}{'最低浓度':>12}{'误差':>12}{'判定':>8}")
    print("  " + "-" * 66)
    for key in solvers.all_keys():
        traj = solvers.make(key).integrate(h=3.0e-3)
        d = analysis.assess(traj, t_grid, y_ref)
        verdict = "成立" if d["stable"] else "崩溃"
        print(f"  {d['method']:<12}{d['t_covered']:>9.4f}{d['mass_defect_max']:>12.2e}"
              f"{d['min_component']:>12.4g}{d['error_full']:>12.3e}{verdict:>8}")
    print()
    print("  显式 Euler 这一行就是全部要点：守恒残差 1e-13（看着满分），")
    print("  但最低浓度是负的 -921.8、误差 1e10 —— 如果只查守恒，这份垃圾会被放行。")
    print("  隐式 Euler 在同一个 h 下一路走到 t = 40，但它的误差仍然是 O(h)：")
    print("  「没崩」是稳健性结论，不是精度结论。")


def section_6_endpoint(t_grid: np.ndarray, y_ref: np.ndarray) -> None:
    rule("6. 终点值 y2(40)：我们自己的高精度值 vs 题面给的参考值")
    y40 = y_ref[-1]
    print(f"  本程序参考解  y(40) = [{y40[0]:.12f}, {y40[1]:.12e}, {y40[2]:.12f}]")
    print(f"  题面参考值     y(40) = {np.array2string(model.BRIEF_Y_AT_T40, precision=7)}")
    diff = np.abs(y40 - model.BRIEF_Y_AT_T40)
    print(f"  两者偏差       {diff[0]:.3e}, {diff[1]:.3e}, {diff[2]:.3e}")
    print(f"  -> 题面 y2(40) 的小数位只能确认到 {model.BRIEF_Y_AT_T40[1]:.4e} 即 5 位有效数字；")
    print("     我们自己的参考值给到 y2(40) = "
          f"{y40[1]:.12e}，比题面多约 5 位。")

    print("\n  各方法能承认到几位（与参考解对比）：")
    for key in solvers.all_keys():
        for h in (1.0e-3, 2.5e-4):
            traj = solvers.make(key).integrate(h=h)
            if not traj.covers_full_span:
                continue
            v = float(traj.dense([40.0])[0][model.I_B])
            digits = -np.log10(max(abs(v - y40[1]) / abs(y40[1]), 1e-16))
            print(f"    {solvers.label(key):<12} h = {h:.1e}   y2(40) = {v:.10e}"
                  f"   吻合 {min(digits, 15.5):5.1f} 位")


def section_7_figures(rows_by_method: dict, t_grid: np.ndarray,
                      y_ref: np.ndarray) -> list:
    rule("7. 出图")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 图上的标签一律用 ASCII：默认字体 DejaVu Sans 没有中文字形，
    # 写中文会变成一串方框，而且换台机器还得去配字体。
    en = {"explicit_euler": "Explicit Euler", "rk4": "RK4",
          "implicit_euler": "Implicit Euler"}

    if not OUT.is_dir():
        OUT.mkdir(parents=True, exist_ok=True)
    made = []

    # 图一：收敛阶
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=160)
    color = {"explicit_euler": "#1f77b4", "rk4": "#d62728", "implicit_euler": "#2ca02c"}
    for key, rows in rows_by_method.items():
        h = np.array([r["h_used"] for r in rows])
        e = np.array([r["error_late"] for r in rows])
        m = np.isfinite(e) & (e > 0)
        ax.loglog(h[m], e[m], "o-", color=color[key], label=en[key])
        hs = np.array([h[m][0], h[m][-1]])
        p = solvers.METHODS[key][2]
        ax.loglog(hs, e[m][-1] * (hs / h[m][-1]) ** p, "--", lw=0.8,
                  color=color[key], alpha=0.5)
    ax.set_xlabel("step size h")
    ax.set_ylabel("max error vs reference")
    ax.set_title("Observed convergence order (dashed = theoretical slope)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    f1 = OUT / "convergence.png"
    fig.savefig(f1)
    plt.close(fig)
    made.append(f1)

    # 图二：轨迹 + 负浓度
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=160)
    ax.loglog(t_grid[1:], y_ref[1:, model.I_B], "-", color="k", lw=1.6,
              label="reference")
    for key in solvers.all_keys():
        traj = solvers.make(key).integrate(h=5.0e-4)
        ax.loglog(traj.t[1:], np.maximum(traj.y[1:, model.I_B], 1e-30), "--",
                  color=color[key], lw=1.0, label=f"{en[key]} h=5e-4")
    bad = solvers.make("explicit_euler").integrate(h=3.0e-3)
    ax.loglog(bad.t[1:], np.abs(bad.y[1:, model.I_B]) + 1e-30, ":",
              color="crimson", lw=2.0, label="Explicit Euler h=3e-3 (blows up)")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$y_2$ (|value|)")
    ax.set_title(r"Stiff transient: $y_2$ peaks near $t=4.4\times10^{-3}$, then decays")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    f2 = OUT / "stiff_transient.png"
    fig.savefig(f2)
    plt.close(fig)
    made.append(f2)

    for p in made:
        print(f"  已写出 {p.relative_to(OUT.parent)}")
    return made


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Robertson 刚性系统：数值核心演示")
    ap.add_argument("--quick", action="store_true", help="减少收敛阶梯层数，快速跑完")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    t_grid = solvers.output_grid()
    print(f"输出网格：{t_grid.size} 个点（t = 0 加上 [1e-8, 40] 上 200 个对数点）")

    y_ref = analysis.reference_solution(t_grid)
    print(f"高精度参考解：Radau，rtol = 1e-10，atol 逐分量 [1e-14, 1e-20, 1e-14]")

    section_1_selfcheck()
    abs_max, _ = section_2_stiffness(y_ref, t_grid)
    section_3_step_limit(abs_max, args.quick)
    rows = section_4_convergence(t_grid, y_ref, args.quick)
    section_5_four_tracks(t_grid, y_ref)
    section_6_endpoint(t_grid, y_ref)
    section_7_figures(rows, t_grid, y_ref)

    print()
    print("=" * 78)
    print(f"完成，用时 {time.perf_counter() - t0:.1f} s")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
