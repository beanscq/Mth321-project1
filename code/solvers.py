"""三种定步长积分器 + 三次 Hermite 稠密输出。

设计上只有三条规矩，但它们决定了结果可不可信：

1) 每个方法只实现"走一步"，步进循环、发散检测、计算量记账都在基类里。
   这样三种方法的成本可以直接对比，不存在口径不一致。

2) 稠密输出用 **三次 Hermite**（用步两端已知的 y 与 f 构造），
   误差 O(h^4)。如果图省事用线性插值，插值误差 O(h^2) 会把 RK4 的
   实测收敛阶从 4 压到 2，整套收敛性研究直接作废。

3) f 在步节点上的值全部是"顺手捡的"，不额外花钱：
   显式 Euler 下一步的第一次求值就是它；RK4 的 k1 就是它；
   隐式 Euler 的 Newton 最后一次残差求值已经算过 f。
   所以稠密输出的右端项额外开销恰好为 0，下面统计的次数就是真实调用次数。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import model

# 状态一旦离开这个球就被判定为发散。精确解恒在单纯形 [0,1]^3 内，
# 所以 1e3 是极宽松的界，任何有意义的解都碰不到它。
DIVERGENCE_BOUND = 1.0e3

# 成本模型：以"一次右端项求值"为单位，算一次 f、一次 Jacobian、解一次方程各记 1。
COST_WEIGHTS = {"rhs": 1.0, "jac": 1.0, "lu": 1.0}


# --------------------------------------------------------------------------
# 比较用的公共输出网格
# --------------------------------------------------------------------------
def output_grid(n_log: int = 200, t_min: float = 1.0e-8,
                t_max: float = 40.0, include_t0: bool = True) -> np.ndarray:
    """t = 0 加上 n_log 个对数等距点（严格递增）。

    两端点按字面值写入：10**log10(40.0) 在一个 libm 下是 39.99999999999999、
    在另一个下是 40.000000000000007，末点一旦越过 t_max，solve_ivp 会直接
    用 "Values in t_eval are not within t_span" 拒掉整个网格 —— 网格就变成
    平台相关的了。端点属于定义的一部分，不是舍入产物，所以写死。
    """
    if n_log < 2:
        raise ValueError("n_log 必须 >= 2")
    if not 0.0 < t_min < t_max:
        raise ValueError("需要 0 < t_min < t_max")
    t = np.logspace(np.log10(t_min), np.log10(t_max), int(n_log))
    t[0] = t_min
    t[-1] = t_max
    if include_t0:
        t = np.concatenate(([0.0], t))
    return t


def hermite_dense(tn: np.ndarray, yn: np.ndarray, fn: np.ndarray, tq) -> np.ndarray:
    """在节点 (tn, yn, fn) 上做三次 Hermite 插值。

    查询点落在 [tn[0], tn[-1]] 之外时返回 nan —— 一个在 t = 40 之前就崩掉的
    运行结果，绝不允许被悄悄外推成"看起来还算完整"的曲线。
    """
    tn = np.asarray(tn, dtype=float)
    yn = np.asarray(yn, dtype=float)
    fn = np.asarray(fn, dtype=float)
    tq = np.atleast_1d(np.asarray(tq, dtype=float))
    out = np.full((tq.size, yn.shape[1]), np.nan)
    if tn.size < 2:                       # 死在第一步里的运行
        return out

    inside = (tq >= tn[0]) & (tq <= tn[-1])
    if not np.any(inside):
        return out
    q = tq[inside]
    i = np.clip(np.searchsorted(tn, q, side="right") - 1, 0, tn.size - 2)

    t0, t1 = tn[i], tn[i + 1]
    h = (t1 - t0)[:, None]
    th = ((q - t0) / (t1 - t0))[:, None]

    h00 = 1.0 - 3.0 * th ** 2 + 2.0 * th ** 3
    h10 = th - 2.0 * th ** 2 + th ** 3
    h01 = 3.0 * th ** 2 - 2.0 * th ** 3
    h11 = th ** 3 - th ** 2

    out[inside] = (h00 * yn[i] + h10 * h * fn[i]
                   + h01 * yn[i + 1] + h11 * h * fn[i + 1])
    return out


# --------------------------------------------------------------------------
# 统计与轨迹
# --------------------------------------------------------------------------
@dataclass
class StepStats:
    """一次运行允许我们声明的全部成本信息。"""

    method: str = ""
    order: int = 0
    h_requested: float = float("nan")
    h_used: float = float("nan")
    n_steps: int = 0
    rhs_calls: int = 0
    jac_calls: int = 0
    lu_solves: int = 0
    newton_iters: int = 0
    newton_zero_update_steps: int = 0     # 预测步已满足停止准则 ⇒ 这一步等价于显式 Euler
    aborted: bool = False
    abort_reason: str = ""
    abort_t: float = float("nan")

    def cost(self, weights: dict | None = None) -> float:
        w = COST_WEIGHTS if weights is None else weights
        return (w["rhs"] * self.rhs_calls
                + w["jac"] * self.jac_calls
                + w["lu"] * self.lu_solves)


@dataclass
class Trajectory:
    """一条数值解：步节点上的 (t, y, f) 加统计信息。"""

    t: np.ndarray
    y: np.ndarray
    f: np.ndarray
    stats: StepStats
    method: str = ""
    order: int = 0
    t_end: float = model.T_END

    @property
    def t_covered(self) -> float:
        return float(self.t[-1])

    @property
    def covers_full_span(self) -> bool:
        return self.t_covered >= self.t_end - 1e-12 * max(self.t_end, 1.0)

    @property
    def min_component(self) -> float:
        return float(np.min(self.y))

    def dense(self, tq) -> np.ndarray:
        return hermite_dense(self.t, self.y, self.f, tq)


# --------------------------------------------------------------------------
# 基类：步进循环
# --------------------------------------------------------------------------
class Integrator:
    """三种方法共同的骨架；子类只需要实现 _step。"""

    name = "?"
    order = 0

    def __init__(self, rhs=None, jac=None, **options):
        self.rhs = model.f if rhs is None else rhs
        self.jac = model.jacobian if jac is None else jac
        self.options = dict(options)

    def _step(self, t, y, f_cur, h, stats):
        raise NotImplementedError

    def integrate(self, h: float, t_end: float = model.T_END, t0: float = 0.0,
                  y0=None, divergence_bound: float = DIVERGENCE_BOUND) -> Trajectory:
        """在 [t0, t_end] 上定步长积分。

        请求的 h 向下取整成"整数步刚好落到 t_end"：nsteps = ceil(span/h)、
        h_used = span/nsteps。两个数都记下来，因为收敛性研究必须用真正走过的步长
        作图，而不是用请求值。
        """
        y = np.array(model.Y0 if y0 is None else y0, dtype=float).reshape(-1)
        span = float(t_end - t0)
        if span <= 0.0:
            raise ValueError("t_end 必须大于 t0")
        if h <= 0.0:
            raise ValueError("h 必须为正")
        nsteps = max(1, int(np.ceil(span / h)))
        h_used = span / nsteps

        st = StepStats(method=self.name, order=self.order,
                       h_requested=float(h), h_used=float(h_used))
        f_cur = self.rhs(t0, y)
        st.rhs_calls += 1

        ts = [float(t0)]
        ys = [y.copy()]
        fs = [np.asarray(f_cur, dtype=float).copy()]

        for n in range(nsteps):
            # 节点时刻写成 t0 + n*h_used 而不是累加，末点钉死在 t_end，
            # 保证节点严格递增、不会出现重复时刻（重复会让 Hermite 除零）
            t = t0 + n * h_used
            t_next = t_end if n == nsteps - 1 else t0 + (n + 1) * h_used
            y_new, f_new = self._step(t, y, f_cur, t_next - t, st)

            finite = bool(np.all(np.isfinite(y_new)))
            worst = float(np.max(np.abs(y_new))) if finite else np.inf
            if (not finite) or worst > divergence_bound:
                st.aborted = True
                st.abort_reason = ("状态非有限" if not finite
                                   else f"max|y| 超过 {divergence_bound:g}")
                st.abort_t = float(t_next)
                break

            y = np.asarray(y_new, dtype=float)
            f_cur = np.asarray(f_new, dtype=float)
            st.n_steps += 1
            ts.append(float(t_next))
            ys.append(y.copy())
            fs.append(f_cur.copy())

        return Trajectory(t=np.asarray(ts), y=np.asarray(ys), f=np.asarray(fs),
                          stats=st, method=self.name, order=self.order,
                          t_end=float(t_end))


# --------------------------------------------------------------------------
# 方法一：显式（前向）Euler
# --------------------------------------------------------------------------
class ExplicitEuler(Integrator):
    """y_{n+1} = y_n + h f(t_n, y_n)，一阶，每步一次右端项求值。

    绝对稳定区只有单位圆盘 |1 + h*lambda| <= 1，即负实轴上 -2 <= h*lambda <= 0。
    取 |lambda|_max = 3392.79（t = 40 处）就得到 h < 5.895e-4。
    这是把 Jacobian 冻结在最坏一瞬间的线性化估计：它是**充分**的（照着取一定稳），
    但偏保守 —— 实测要到 h ≈ 6.2e-4 才真的崩（见 demo.py 第 3 节）。
    """

    name = "显式 Euler"
    order = 1

    def _step(self, t, y, f_cur, h, st):
        y_new = y + h * f_cur
        f_new = self.rhs(t + h, y_new)
        st.rhs_calls += 1                 # 下一步的 k1，同时又是 Hermite 的右端斜率
        return y_new, f_new


# --------------------------------------------------------------------------
# 方法二：经典四阶 Runge-Kutta
# --------------------------------------------------------------------------
class RK4(Integrator):
    """k1..k4 的标准组合，四阶，每步四次右端项求值。

    提高阶数并不能解决刚性：RK4 的负实轴稳定区间是 [-2.7853, 0]，
    只比显式 Euler 的 [-2, 0] 宽 1.39 倍，所以在 |lambda|_max = 3392.79 下
    仍然要求 h < 8.209e-4。
    """

    name = "RK4"
    order = 4

    def _step(self, t, y, f_cur, h, st):
        k1 = f_cur
        k2 = self.rhs(t + 0.5 * h, y + (0.5 * h) * k1)
        k3 = self.rhs(t + 0.5 * h, y + (0.5 * h) * k2)
        k4 = self.rhs(t + h, y + h * k3)
        st.rhs_calls += 3
        y_new = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        f_new = self.rhs(t + h, y_new)    # 第 4 次，供下一步和稠密输出
        st.rhs_calls += 1
        return y_new, f_new


# --------------------------------------------------------------------------
# Newton 迭代（隐式方法的内核）
# --------------------------------------------------------------------------
@dataclass
class NewtonResult:
    x: np.ndarray
    converged: bool
    iters: int
    residual: float
    rhs_calls: int = 0
    jac_calls: int = 0
    lu_solves: int = 0
    f_used: np.ndarray | None = None
    residual_history: list = field(default_factory=list)


def newton_solve(residual_fn, x0: np.ndarray, tol: float, jac_fn,
                 maxiter: int = 50) -> NewtonResult:
    """解 G(x) = 0。

    收敛判据用**非线性残差** ||G||_inf，而不是 Newton 修正量 ||dx||。
    原因不是口味问题：对这个系统有

        sum G(z) = sum(z - y_n - h f(t_{n+1}, z)) = sum(z) - sum(y_n)

    因为 (1,1,1)^T f == 0。于是"这一步引入的守恒残差"正好被 ||G||_inf 以
    3*||G||_inf 控制住；而对 ||dx|| 加容差得不到这个关系。
    """
    x = np.array(x0, dtype=float).reshape(-1)
    hist, rhs_calls, jac_calls, lu_solves = [], 0, 0, 0
    converged, f_used = False, None

    for _ in range(maxiter + 1):
        G, f_used = residual_fn(x)
        rhs_calls += 1
        G = np.asarray(G, dtype=float)
        r = float(np.max(np.abs(G)))
        hist.append(r)
        if not np.isfinite(r):
            break
        if r <= tol:
            converged = True
            break
        Jn = np.asarray(jac_fn(x), dtype=float)
        jac_calls += 1
        try:
            dx = np.linalg.solve(Jn, -G)
        except np.linalg.LinAlgError:
            break
        lu_solves += 1
        if float(np.max(np.abs(dx))) == 0.0:
            break                          # 线性模型已到根，残差却仍高于容差：无法再推进
        x = x + dx

    return NewtonResult(x=x, converged=converged, iters=jac_calls,
                        residual=hist[-1] if hist else float("inf"),
                        rhs_calls=rhs_calls, jac_calls=jac_calls,
                        lu_solves=lu_solves,
                        f_used=None if f_used is None else np.asarray(f_used, dtype=float),
                        residual_history=hist)


# --------------------------------------------------------------------------
# 方法三：隐式（后向）Euler + Newton
# --------------------------------------------------------------------------
class ImplicitEuler(Integrator):
    """y_{n+1} = y_n + h f(t_{n+1}, y_{n+1})，一阶，每步解一个 3x3 非线性方程组。

    * A 稳定且 L 稳定：R(z) = 1/(1-z) 把整个左半平面映进单位圆，且 R(inf) = 0。
      所以步长**不受刚性限制** —— 但这是"稳"的结论，不是"准"的结论：
      局部截断误差仍然是 O(h)，绝不能把"没崩"讲成"算对了"。

    * 预测值取一步显式 Euler，复用了节点上已经存好的 f，不额外花代价。

    * Newton 容差是头等重要的旋钮，默认取 1e-12。放松它会让方法**悄悄退化成
      显式方法**，而显式方法在大步长下正好会炸（demo.py 第 5 节会看到退化的样子）。
    """

    name = "隐式 Euler"
    order = 1

    def __init__(self, rhs=None, jac=None, newton_tol: float = 1.0e-12,
                 maxiter: int = 50, **options):
        super().__init__(rhs=rhs, jac=jac, **options)
        self.newton_tol = float(newton_tol)
        self.maxiter = int(maxiter)

    def _step(self, t, y, f_cur, h, st):
        tn = t + h
        y_pred = y + h * f_cur             # 显式 Euler 预测，零额外开销

        def residual_fn(z):
            fz = self.rhs(tn, z)
            return z - y - h * fz, fz

        def jac_fn(z):
            return np.eye(z.size) - h * self.jac(tn, z)

        res = newton_solve(residual_fn, y_pred, self.newton_tol, jac_fn,
                           maxiter=self.maxiter)

        st.rhs_calls += res.rhs_calls
        st.jac_calls += res.jac_calls
        st.lu_solves += res.lu_solves
        st.newton_iters += res.iters
        if res.iters == 0:
            st.newton_zero_update_steps += 1     # 预测步直接就是答案
        if not res.converged:
            # 稠密输出要诚实：存的斜率必须属于真正返回的那个状态
            res.f_used = self.rhs(tn, res.x)
            st.rhs_calls += 1
        return res.x, res.f_used


# --------------------------------------------------------------------------
METHODS = {
    "explicit_euler": ("显式 Euler", ExplicitEuler, 1),
    "rk4":            ("RK4", RK4, 4),
    "implicit_euler": ("隐式 Euler", ImplicitEuler, 1),
}


def make(key: str, **kwargs) -> Integrator:
    """按名字构造一个方法。"""
    if key not in METHODS:
        raise KeyError(f"未知方法 {key!r}；可选 {sorted(METHODS)}")
    return METHODS[key][1](**kwargs)


def label(key: str) -> str:
    return METHODS[key][0]


def all_keys() -> list:
    return list(METHODS)
