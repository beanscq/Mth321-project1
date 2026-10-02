"""分析层：高精度参考解、四道诊断、实测收敛阶、稳定步长上界。

这里最容易被做错、也最值得强调的是**四道诊断必须分开记**：

    1) 守恒残差     |y1 + y2 + y3 - 1|
    2) 非负性       min_i y_i >= 0（逐分量）
    3) 全状态误差   相对高精度参考解
    4) 终点值       y2(40)

为什么不能只看守恒：守恒量 (1,1,1)·y 对任何相容的 Runge-Kutta 方法都是
代数恒等式（隐式方法则精确到 Newton 残差），所以一个**完全错误、甚至已经变成
负浓度**的解，守恒残差照样可以是 1e-13 级别的好看。demo.py 第 5 节就是拿这一点
做反面教材：显式 Euler 在 h = 3e-3 时守恒残差 1.1e-13，同时最低浓度 -921.8、
误差 1e10 —— 只查守恒的话，这份垃圾会被放行。
"""

from __future__ import annotations

import numpy as np

import model
import solvers

ERROR_FLOOR_DEFAULT = 1.0e-11      # 参考解自身的不确定度量级
T_CUT_LATE = 1.0e-3                # 误差统计里"晚段"的起点


# --------------------------------------------------------------------------
# 高精度参考解（唯一使用 SciPy 的地方）
# --------------------------------------------------------------------------
def reference_solution(t_eval: np.ndarray, rtol: float = 1.0e-10,
                       atol=(1.0e-14, 1.0e-20, 1.0e-14)) -> np.ndarray:
    """用 SciPy 的 Radau 求高精度参考解，返回形状 (len(t_eval), 3)。

    atol 必须逐分量给：y2 常年生活在 1e-5 ~ 1e-10，与 y1、y3 差十几个量级，
    单一标量 atol 要么浪费精度要么把 y2 直接抹平。
    """
    from scipy.integrate import solve_ivp

    sol = solve_ivp(model.f, model.T_SPAN, model.Y0, method="Radau",
                    t_eval=np.asarray(t_eval, dtype=float),
                    rtol=rtol, atol=np.asarray(atol, dtype=float))
    if not sol.success:
        raise RuntimeError(f"参考解求解失败：{sol.message}")
    return sol.y.T


# --------------------------------------------------------------------------
# 刚性有多强 / 步长能开多大
# --------------------------------------------------------------------------
# 各方法在负实轴上的绝对稳定区间右端（|h*lambda| 的上限）
REAL_AXIS_LIMIT = {
    "explicit_euler": 2.0000000000000000,
    "rk4":            2.7852935634052840,
    "implicit_euler": float("inf"),
}


def stability_bound(abs_lambda_max: float) -> dict:
    """由冻结谱给出的步长上界 h <= limit / |lambda|_max。

    隐式方法没有这个限制（A 稳定），所以给 inf —— 但请记住这只是"不会炸"，
    不代表精度够（它是一阶方法，误差还是 O(h)）。
    """
    out = {}
    for key, lim in REAL_AXIS_LIMIT.items():
        out[key] = {
            "label": solvers.label(key),
            "real_axis_limit": lim,
            "h_bound": float(lim / abs_lambda_max) if np.isfinite(lim) else float("inf"),
        }
    return out


def max_abs_lambda(t_grid: np.ndarray, y_grid: np.ndarray) -> tuple:
    """整条轨迹上 |Re lambda| 的最大值及其发生时刻。

    注意是"整条轨迹上的最大值"，不是 t = 0 处的值：这个系统的 |lambda|_max
    出现在 t 接近 40 的地方，用初值处的谱去估步长会严重高估可用的 h。
    """
    vals = np.array([np.max(np.abs(model.eigenvalues_nonzero(y).real))
                     for y in y_grid])
    i = int(np.argmax(vals))
    return float(vals[i]), float(t_grid[i])


# --------------------------------------------------------------------------
# 单次运行的诊断包
# --------------------------------------------------------------------------
def assess(traj: solvers.Trajectory, t_grid: np.ndarray, y_ref: np.ndarray,
           accurate_tol: float = 1.0e-4, nonneg_tol: float = 1.0e-12) -> dict:
    """对一次定步长运行给出完整诊断。

    非负性在**步节点**上测量，即方法真正产出过的状态，而不是插值出来的曲线：
    Hermite 在节点之间可能轻度越界，把插值当"方法产出"是不诚实的读法。
    """
    t_grid = np.asarray(t_grid, dtype=float)
    y_ref = np.asarray(y_ref, dtype=float)

    covered = t_grid <= traj.t_covered
    y_num = traj.dense(t_grid[covered])
    err = np.abs(y_num - y_ref[covered])
    error_full = float(err.max()) if err.size else float("inf")

    late = covered & (t_grid >= T_CUT_LATE)
    error_late = (float(np.max(np.abs(traj.dense(t_grid[late]) - y_ref[late])))
                  if np.any(late) else float("nan"))

    if traj.covers_full_span:
        y40 = traj.dense([traj.t_end])[0]
    else:
        y40 = np.full(y_ref.shape[1], np.nan)

    min_nodes = traj.min_component
    below = traj.y.min(axis=1) < -nonneg_tol

    return {
        "method": traj.method,
        "h_requested": float(traj.stats.h_requested),
        "h_used": float(traj.stats.h_used),
        "n_steps": int(traj.stats.n_steps),
        "t_covered": traj.t_covered,
        # --- 四道诊断，刻意分开 -----------------------------------------
        "mass_defect_max": float(np.max(model.invariant_defect(traj.y))),
        "min_component": float(min_nodes),
        "t_first_negative": (float(traj.t[int(np.argmax(below))])
                             if np.any(below) else float("nan")),
        "error_full": error_full,
        "error_late": error_late,
        "y2_at_t40": float(y40[model.I_B]),
        # --- 状态标志 ---------------------------------------------------
        "stable": not traj.stats.aborted,
        "non_negative": bool(min_nodes >= -nonneg_tol),
        "accurate": bool(not traj.stats.aborted and traj.covers_full_span
                         and error_full <= accurate_tol),
        "abort_reason": traj.stats.abort_reason,
        # --- 成本 -------------------------------------------------------
        "rhs_calls": int(traj.stats.rhs_calls),
        "jac_calls": int(traj.stats.jac_calls),
        "lu_solves": int(traj.stats.lu_solves),
        "newton_iters": int(traj.stats.newton_iters),
        "cost_units": float(traj.stats.cost()),
    }


# --------------------------------------------------------------------------
# 实测收敛阶
# --------------------------------------------------------------------------
def observed_order(rows: list, error_key: str = "error_late",
                   error_floor: float = ERROR_FLOOR_DEFAULT) -> dict:
    """对 log e = p log h + c 做最小二乘拟合，并给出逐步阶与 R^2。

    两个剔除规则，第二个才是关键：

    * 运行中途发散（根本没走到 t = 40）的层不要；
    * 误差已经掉到参考解自身不确定度量级的层不要 —— 那里的"误差"是
      参考解的地板，不是方法的误差。用穿过地板点的数据拟合出来的"收敛阶"
      没有任何意义，这是学生最常犯的错。
    """
    pts = []
    for r in rows:
        e, h = r.get(error_key, np.nan), r.get("h_used", np.nan)
        if not np.isfinite(e) or not np.isfinite(h) or h <= 0.0:
            continue
        if e <= error_floor or not r.get("stable", True):
            continue
        pts.append((float(h), float(e)))
    pts.sort(key=lambda p: -p[0])          # 从粗到细

    out = {"n_levels_used": len(pts),
           "h_window": [min((p[0] for p in pts), default=np.nan),
                        max((p[0] for p in pts), default=np.nan)],
           "pairs": [{"h_coarse": h1, "h_fine": h2,
                      "order": float(np.log(e1 / e2) / np.log(h1 / h2))}
                     for (h1, e1), (h2, e2) in zip(pts, pts[1:])]}
    if len(pts) < 2:
        out.update({"slope": float("nan"), "r2": float("nan")})
        return out

    x = np.log(np.array([p[0] for p in pts]))
    y = np.log(np.array([p[1] for p in pts]))
    A = np.vstack([x, np.ones_like(x)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - (slope * x + intercept)
    ss_res, ss_tot = float(np.sum(resid ** 2)), float(np.sum((y - y.mean()) ** 2))
    out.update({"slope": float(slope),
                "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
                "fit": list(zip(x.tolist(), y.tolist()))})
    return out
