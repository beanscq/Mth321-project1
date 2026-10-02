"""Robertson 刚性化学反应系统 —— 模型定义。

本文件只描述"问题本身"，不含任何数值方法；所有求解器都从这里拿右端项。

反应网络
--------
    A     -> B          k1 = 0.04
    B + B -> B + C      k2 = 3e7
    B + C -> A + C      k3 = 1e4

对应的 ODE（本文件是唯一权威定义，报告里引用的方程必须与此一致）：

    y1' = -k1*y1 + k3*y2*y3
    y2' = +k1*y1 - k3*y2*y3 - k2*y2**2
    y3' = +k2*y2**2

后面每一步分析都建立在下面三条性质上：

1) 线性守恒律。三式相加，右端逐项抵消（+1e4*y2*y3 对 -1e4*y2*y3，
   -3e7*y2**2 对 +3e7*y2**2），所以

       (y1 + y2 + y3)' == 0 ，从 y(0) = (1,0,0) 出发恒有 y1+y2+y3 == 1。

   等价的说法是：Jacobian J(y) 永远含一个严格等于 0 的特征值。
   这条性质对任何相容的 Runge-Kutta 方法都是代数恒等式 —— 也就是说，
   守恒残差好看并不代表算得准（见 demo.py 第 5 节）。

2) 刚性来源。k1 = 0.04 慢慢给 B 供料，k2 = 3e7 极其迅速地消耗 B，
   两个速率相差 1e8，于是快、慢两个时间尺度同时存在，必须用小步长控制稳定性。

3) 浓度必须非负。出现负浓度不是"精度差一点"，而是化学上不可接受的解。
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# 权威参数 —— 不要修改
# --------------------------------------------------------------------------
K1 = 0.04        # A -> B
K2 = 3.0e7       # B + B -> B + C
K3 = 1.0e4       # B + C -> A + C

Y0 = np.array([1.0, 0.0, 0.0])      # 归一化初始浓度 (A, B, C)
T_SPAN = (0.0, 40.0)                # 求解区间
T_END = T_SPAN[1]
TOTAL_CONCENTRATION = 1.0

I_A, I_B, I_C = 0, 1, 2

# 题面给出的参考终点值，仅用作对照，不替代我们自己算的高精度参考解
BRIEF_Y_AT_T40 = np.array([0.7158271, 9.1855e-6, 0.2841637])


# --------------------------------------------------------------------------
# 右端项与 Jacobian
# --------------------------------------------------------------------------
def f(t: float, y: np.ndarray) -> np.ndarray:
    """右端项 f(t, y)。对 t 显式（自治系统，t 只占位置）。"""
    y = np.asarray(y, dtype=float)
    y1, y2, y3 = y[I_A], y[I_B], y[I_C]
    r1 = K1 * y1            # A -> B
    r2 = K2 * y2 ** 2       # B + B -> B + C  （净消耗 1 个 B、生成 1 个 C）
    r3 = K3 * y2 * y3       # B + C -> A + C
    return np.array([-r1 + r3,
                     r1 - r2 - r3,
                     r2])


def jacobian(t: float, y: np.ndarray) -> np.ndarray:
    """解析 Jacobian J(y) = df/dy。

        J = [[ -0.04,              1e4*y3,       1e4*y2],
             [  0.04,  -1e4*y3 - 6e7*y2,        -1e4*y2],
             [     0,             6e7*y2,            0 ]]

    手写解析式而不是用有限差分：隐式方法的 Newton 迭代每一步都要用它，
    差分近似会把 Newton 的收敛速度拖成线性。
    """
    y = np.asarray(y, dtype=float)
    y2, y3 = y[I_B], y[I_C]
    return np.array([
        [-K1,                        K3 * y3,        K3 * y2],
        [K1,   -K3 * y3 - 2.0 * K2 * y2,            -K3 * y2],
        [0.0,             2.0 * K2 * y2,                  0.0],
    ])


# --------------------------------------------------------------------------
# 守恒量与谱
# --------------------------------------------------------------------------
def invariant_defect(y: np.ndarray) -> np.ndarray:
    """守恒残差 |y1 + y2 + y3 - 1|。"""
    y = np.asarray(y, dtype=float)
    return np.abs(y.sum(axis=-1) - TOTAL_CONCENTRATION)


def reduced_jacobian(t: float, y: np.ndarray) -> np.ndarray:
    """消元 y3 = 1 - y1 - y2 后的 2x2 Jacobian。

    记 z = (y1, y2)，则 B_red = S * J * T，其中 S 取前两个分量、T = d y / d z。
    注意这是一个真正的投影，**不是** J 左上角的 2x2 子块：左上块丢掉了
    det 里 6e11 * y2**2 那一项，在 t = 40 处会差 3.3 倍。

    因为结构特征值恒为 0，B_red 的两个特征值正好就是 J 的两个非零特征值。
    """
    S = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    T = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    return S @ jacobian(t, y) @ T


def eigenvalues_nonzero(y: np.ndarray) -> np.ndarray:
    """J 的两个非零特征值（实部降序）。

    用 B_red 的迹与行列式解析求根，而不是 np.linalg.eig：
    在 B 的诱导层里这两根接近重根，通用特征值求解器在这里会掉精度。
    """
    y = np.asarray(y, dtype=float)
    y2, y3 = y[I_B], y[I_C]
    tr = -K1 - K3 * y3 - 2.0 * K2 * y2
    det = (2.0 * K2 * y2) * (K1 + K3 * y2)
    disc = np.sqrt(np.asarray(tr * tr - 4.0 * det, dtype=complex))
    lam = np.array([(tr + disc) / 2.0, (tr - disc) / 2.0])
    return lam[np.argsort(-np.abs(lam.real))]


def stiffness_ratio(y: np.ndarray) -> float:
    """刚性比 S = |Re lambda_fast| / |Re lambda_slow|。

    t = 0 处三个分量完全退化（慢特征值也是 0），比值没有意义，
    所以统计时从第一个正时刻 1e-8 开始。
    """
    lam = eigenvalues_nonzero(y)
    slow = abs(lam[1].real)
    if slow == 0.0:
        return float("inf")
    return float(abs(lam[0].real) / slow)
