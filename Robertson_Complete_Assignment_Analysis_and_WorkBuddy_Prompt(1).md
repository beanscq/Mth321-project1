# Robertson Problem（Topic ③）完整题目分析与 WorkBuddy 协作 Prompt

> 本文以两份课程文件为准：`Project 1 Brief`（全项目共同要求）与 `Problem Pack`（Topic ③ 的专门要求）。此前仅根据 Project Brief 所做的分析需要更新；本题并不是只要求一般性的 Robertson 数值实验，而是给出了**非常具体、可复现的 protocol 与报告问题**。

---

# Part A：老师到底要你们完成什么？

## 1. 这是一项什么类型的作业？

这是一项**带明确技术要求的开放式数值分析项目**，不是一道只有唯一数值答案的化学题。

老师希望你们：

1. 从 Robertson 化学反应系统出发；
2. 自己实现和比较不同 ODE 数值方法；
3. 用稳定性、特征值、误差、守恒性、非负性、参考解和成本等证据，说明结果为什么可信；
4. 回答一个围绕“刚性化学动力学”的研究问题；
5. 交付可复现代码、报告和展示。

因此，它同时是：

- **化学动力学问题**：反应网络、质量作用定律、反应中间体、时间尺度；
- **常微分方程问题**：三元非线性初值问题；
- **刚性数值分析问题**：显式法的稳定性限制、隐式法与 Newton 迭代；
- **计算科学项目**：自行构建参考解、实验设计、成本—精度比较、可复现性。

老师的核心要求可以概括为：不要只“算出一个解”，而要构建一条**可信的证据链**。

---

## 2. 老师已经替你们规定好的内容（不能随意改）

### 2.1 共同项目要求

无论选哪个标准 deterministic ODE topic，团队都需要：


| 模块      | 老师要求                                                                       |
| ------- | -------------------------------------------------------------------------- |
| 三种方法    | 显式 Euler（1 阶）、RK4（4 阶）、一种隐式方法；隐式法的非线性方程用 Newton 迭代解决                       |
| 稳定性     | 推导并绘制三种方法的绝对稳定域                                                            |
| 特征值与步长  | 对系统进行线性化，讨论相关特征值范围，给出显式方法的实际步长限制                                           |
| 局部稳定性声明 | 必须说明 $$h\lambda$$ 稳定域覆盖只是“冻结 Jacobian”的局部诊断，**不是**完整的非线性稳定性证明；还必须用试验检查这个预测 |
| 自适应控制   | 实现基本 adaptive step-size control；步长必须实际改变，最终误差应接近设置的容差                      |
| 验证      | 自建高精度参考解；至少两个方法给出误差—步长 log-log 图与观测收敛阶                                     |
| 成本—精度   | 先定义成本，再在**匹配精度目标**下比较；说明没有计入哪些成本                                           |
| 交付      | ≥8 页 LaTeX 报告、可运行的代码 zip/README/`run_all.py`、slides PDF                    |


`solve_ivp(Radau/BDF)` 可以作为独立高精度 oracle，但不能替代你们自己实现 Euler、RK4 与隐式方法。

### 2.2 Topic ③ 给出的固定模型

你们必须使用以下 Robertson ODE（这是 authoritative model）：

$$
\begin{aligned}
y_1'&=-0.04y_1+10^4y_2y_3,\\
y_2'&=0.04y_1-10^4y_2y_3-3\times10^7y_2^2,\\
y_3'&=3\times10^7y_2^2,
\end{aligned}
\qquad y(0)=(1,0,0)^T.
$$

时间区间固定为：

$$
0\le t\le40.
$$

### 2.3 Topic ③ 的固定实验 protocol

以下细节是完整题目新补充的重点，报告和代码应严格遵守：

1. **输出时间点**
  - 必须包含 $$t=0$$；
  - 还必须包含从 $$10^{-8}$$ 到 $$40$$ 的至少 200 个对数均匀时间点；
  - 允许每个求解器采用自己的内部步长；
  - 但每种方法与参考解都要在**同一组输出时间点**比较。
2. **参考解**
  - 使用紧容差的 `Radau`、`BDF` 或 `LSODA`；
  - 将容差再缩小 100 倍重新计算；
  - 最终报告时，只保留这两次参考解中不变的有效数字。
3. **均匀步长收敛实验**
  - 至少使用四个连续减半的步长；
  - 对每一次运行，单独标明它是否：**稳定（stable）**、**非负（non-negative）**、**准确（accurate）**。
4. **刚性比的定义**
  - 不能把 $$3\times3$$ Jacobian 中的零特征值放入刚性比；
  - 必须仅使用两个非零特征值，定义为：

$$
S(t)=\frac{\max_{\lambda_j\ne0}|\operatorname{Re}(\lambda_j(t))|}
{\min_{\lambda_j\ne0}|\operatorname{Re}(\lambda_j(t))|}.
$$

- 或者可利用守恒关系消去 $$y_3=1-y_1-y_2$$，再使用约化 $$2\times2$$ Jacobian 的两个特征值；这通常更干净。
- 必须说明判断“零特征值”的数值阈值；
- $$t=0$$ 为退化点，刚性比没有意义；图应从第一个正的对数时间点开始。

5. **额外指定的数值结果**
  - 需要报告：在若干容差/设置下，**你们的隐式方法给出的终点 **$$y_2(40)$$**；**
  - 并与自建高精度参考解交叉检查。
6. **Newton 容差试验**
  - 隐式方法的 Newton 迭代不用追求机器精度；
  - 但应选择足够严格的停止准则，使 Newton 代数误差相对于时间离散误差可忽略；
  - 用 Newton stopping tolerance sweep 证明这一点。

---

# Part B：化学部分究竟在哪里？

## 3. 反应网络与速率常数

题目给出以下助记反应标签：

$$
A\xrightarrow{k_1}B,
\qquad
B+B\xrightarrow{k_2}B+C,
\qquad
B+C\xrightarrow{k_3}A+C,
$$

其中：

$$
k_1=0.04,
\qquad k_2=3\times10^7,
\qquad k_3=10^4.
$$

请注意：反应标签只用于帮助理解；**ODE 是权威定义**。尤其对于 $$B+B\to B+C$$，不能因为有两个 $$B$$ 就在 ODE 的 $$3\times10^7y_2^2$$ 项前额外乘以 2。题目已经明确禁止这样做。

令 $$y_1,y_2,y_3$$ 分别代表 $$A,B,C$$ 的归一化浓度。质量作用定律速率为：

$$
r_1=0.04y_1,
\qquad r_2=3\times10^7y_2^2,
\qquad r_3=10^4y_2y_3.
$$

相应的物种收支为：

$$
y_1'=-r_1+r_3,
\qquad y_2'=r_1-r_2-r_3,
\qquad y_3'=r_2.
$$

这正好恢复题目中的 ODE。

## 4. 该体系的化学故事线

### 4.1 $$B$$ 是短寿命中间体

初始时只有 $$A$$。慢反应 $$A\to B$$ 先生成少量 $$B$$。但一旦 $$B$$ 存在，二次反应速率

$$
r_2=3\times10^7y_2^2
$$

可能迅速变大，因此 $$B$$ 很快又转化为 $$C$$。这使 $$B$$ 只出现一个非常小、短暂的峰值，而不是大量积累。

### 4.2 快慢时间尺度是刚性的物理来源

速率常数从 $$0.04$$ 到 $$3\times10^7$$，跨度约为 $$10^8$$。化学上，这意味着系统同时有：

- 慢的 $$A\to B$$ 生成过程；
- 很快的 $$B$$ 消耗过程；
- 较慢的长期组成变化。

数值上，快过程对应大幅负实部特征值；即使长期轨迹变化很慢，显式法仍可能必须用非常小步长来控制快模态。这就是 Robertson 体系刚性的根源。

### 4.3 总质量守恒

把三条 ODE 相加：

$$
\frac{d}{dt}(y_1+y_2+y_3)=0.
$$

因此：

$$
y_1(t)+y_2(t)+y_3(t)=1.
$$

这是封闭化学系统的物质量守恒，也解释了为什么轨迹总在一个二维平面（invariant plane）上。它还导致完整 $$3\times3$$ Jacobian 始终有一个结构性零特征值。

### 4.4 非负性

浓度应满足：

$$
y_i(t)\ge0.
$$

若数值结果出现负浓度，不能仅描述为“有一点误差”；它是化学上不可能的状态，表明该方法/步长在这个问题上不可靠。

---

# Part C：本题真正要做的实验与结论

## 5. 必须回答的 Topic ③ 专门问题

### 5.1 Jacobian、非零特征值与刚性比

建议首先写出完整 Jacobian：

$$
J(y)=
\begin{pmatrix}
-0.04 & 10^4y_3 & 10^4y_2\\
0.04 & -10^4y_3-6\times10^7y_2 & -10^4y_2\\
0 & 6\times10^7y_2 & 0
\end{pmatrix}.
$$

要求：

- 沿高精度参考解计算 Jacobian 特征值；
- 用对数时间画两个非零特征值的实部/数值；
- 用对数时间画刚性比 $$S(t)$$；
- 解释零特征值来自守恒平面，而不是“无限刚性”；
- 解释 $$t=0$$ 退化时为何不报告刚性比。

题目提供的方向性 check values（不可直接照抄充当结果）是：


| 时间          | 非零特征值刚性比约为         |
| -----------: | ------------------: |
| $$10^{-4}$$ | $$3.0\times10^3$$  |
| $$10^{-2}$$ | $$5.4\times10^3$$  |
| $$40$$      | $$1.58\times10^5$$ |


并且在 $$t=40$$，最大特征值模约为 $$3.39\times10^3$$。之所以不是 $$3\times10^7$$，是因为 $$y_2=O(10^{-5})$$，实际 Jacobian 项会被小浓度缩小。

### 5.2 显式 Euler 的稳定性、准确性与非负性

对测试方程 $$u'=\lambda u$$，显式 Euler 的稳定函数为：

$$
R(z)=1+z,
\qquad z=h\lambda.
$$

其负实轴稳定区间是：

$$
-2\le h\lambda\le0.
$$

利用 $$|\lambda_{\max}|\approx3.39\times10^3$$，局部估计为：

$$
h<\frac{2}{3.39\times10^3}\approx5.9\times10^{-4}.
$$

但这只是局部、冻结 Jacobian 的提示；必须做 empirical step sweep，观察不同步长时：

- 是否数值爆炸或出现明显振荡；
- 是否出现负浓度；
- 是否质量守恒；
- 是否在全状态误差上足够准确。

关键结论：**守恒并不等于准确**。某些 Runge–Kutta 离散法可代数保持线性守恒关系，但仍可能给出严重失真的甚至负的浓度。

### 5.3 与 implicit Euler 的公平比较

隐式 Euler：

$$
y_{n+1}=y_n+h f(y_{n+1}).
$$

每一步通过 Newton 解：

$$
G(z)=z-y_n-hf(z)=0,
$$

$$
z^{(k+1)}=z^{(k)}-
\left[I-hJ(z^{(k)})\right]^{-1}G(z^{(k)}).
$$

要比较的不是“implicit Euler 会不会 blow up”，而是：

- 在同一误差目标下，implicit Euler 能否用更大步长；
- 得到同样准确度时，需要多少步、多少 RHS evaluations、多少 Newton iterations、多少 wall time；
- 在不同 Newton stopping tolerances 下，守恒缺陷和最终误差如何变化；
- 终点 $$y_2(40)$$ 是否与独立紧容差参考值一致。

### 5.4 共同要求中的 RK4、稳定域、自适应与收敛阶

虽然 Topic ③ 专门段落重点点名 Euler 与 implicit Euler，完整项目仍需要：

- 实现 **RK4** 并纳入稳定域、收敛阶、误差与成本比较；
- 画 Euler、RK4、implicit Euler 三个方法的绝对稳定域；
- 至少对两个方法做四个以上连续减半步长的误差—步长 log-log 图；
- 实现自适应方法（推荐 RK4 step-doubling），并展示步长随早期快瞬态变化；
- 清楚报告 accepted/rejected steps、误差与容差关系。

### 5.5 必须分开报告的四个诊断


| 诊断         | 定义/做法                                               | 它回答什么         | 它不能单独证明什么   |
| ---------- | --------------------------------------------------- | ------------- | ----------- |
| 守恒         | $$                                                  | y_1+y_2+y_3-1 | $$          |
| 非负性        | $$\min_{t,i}y_i(t)$$                                | 是否出现化学不可能的浓度  | 不量化接近参考解的误差 |
| 全状态误差      | $$\max_{t\in\mathcal T}|y_h(t)-y_{ref}(t)|_\infty$$ | 数值轨迹距参考解多远    | 不直接说明物理结构   |
| 终点 $$y_2$$ | 比较 $$y_2(40)$$                                      | 对微小中间体最终量是否可信 | 不代表整段轨迹都准确  |


---

## 6. 一个非常适合本题的主研究问题

建议直接采用：

> **How do the chemically induced fast and slow time scales in the Robertson reaction network affect the stability, accuracy, positivity, and cost of explicit and implicit ODE solvers?**
>
> Robertson 反应网络中由化学反应速率差异产生的快慢时间尺度，如何影响显式和隐式 ODE 求解器的稳定性、准确性、浓度非负性与计算成本？

这个问题能同时覆盖：

- 化学解释：速率常数跨越约 $$10^8$$；$$B$$ 是短寿命中间体；
- 数学解释：守恒平面、零特征值、非零谱的刚性比；
- 数值解释：绝对稳定域、显式步长约束、隐式鲁棒性；
- 实验结论：匹配精度下的成本与物理可行性。

---

## 7. 推荐的报告结构

1. **Background and research question**：化学反应网络、为何这是刚性问题、研究问题。
2. **Model and chemical structure**：ODE、质量作用定律、初值、守恒平面、非负性。
3. **Numerical methods**：Euler、RK4、implicit Euler + Newton；Newton 残差和停止准则。
4. **Reference solution and validation protocol**：同一组对数输出时间、Radau/BDF 加严容差检查、误差范数。
5. **Stability and stiffness analysis**：稳定函数/稳定域；Jacobian 非零特征值、刚性比、局部 $$h\lambda$$ overlay。
6. **Numerical experiments**：
  - 参考轨迹与反应速率；
  - 均匀步长收敛阶；
  - Euler step sweep（稳定/准确/非负）；
  - implicit Euler matched-error comparison；
  - Newton tolerance sweep；
  - adaptive RK4；
  - 成本—精度。
7. **Conclusion**：用自己的测量结果回答主问题，不预设“哪个方法必胜”。
8. **Appendix**：AI transparency log、实现细节、额外表格。

---

# Part D：可直接发给 WorkBuddy 的 Prompt

将以下内容完整复制给 WorkBuddy。之后再告诉它“先做第 X 项”。

```text
我正在完成 Numerical Analysis of ODEs and PDEs 的 Project 1，选择 Topic ③：Chemical kinetics — Robertson problem。请作为我的协作助手，严格根据以下完整作业要求帮助我完成项目。不要把它简化成泛泛的 ODE 示例；必须同时保留化学动力学解释与数值分析验证。

====================
A. 项目性质与交付物
====================
这是一个 5 人团队的开放式数值分析项目。我们要像数值顾问一样，对真实的化学反应体系给出可靠、可验证的数值结论，而不是只调用一个黑箱 solver 或复现教材图。

最终交付：
1. 至少 8 页 LaTeX 报告，结构为 Background → Model → Methods → Stability → Numerical experiments → Conclusions；
2. 代码 zip，含 README 和 run_all.py，运行后可生成全部图表和表格；
3. 10 分钟 presentation slides PDF；
4. AI transparency log 等课程要求的附录。

共同硬性要求：
- 自己实现 Explicit Euler（1阶）、RK4（4阶）、一种 implicit method（建议 implicit Euler）；
- 隐式方法产生的非线性方程用 Newton iteration 求解；
- 推导并绘制三种方法的 absolute-stability regions；
- 做 Jacobian eigenvalue / stiffness 分析，并将 h*lambda overlay 作为局部 frozen-Jacobian diagnostic；明确说明它不是完整 nonlinear stability proof，并用实际 step-size experiment 检验；
- 实现 basic adaptive step-size control（推荐 RK4 step-doubling）；展示步长实际改变，误差接近所设 tolerance；
- 自己构建 tight-tolerance reference solution；至少两个方法有 error-vs-step-size log-log plot 和 observed convergence order；
- 做 matched-accuracy cost comparison。先定义成本单位（RHS evaluations、Newton iterations、accepted/rejected steps、wall time 等），说明没有计入哪些成本；
- 不得只用 scipy solver 代替核心方法实现。SciPy Radau/BDF/LSODA 只能当独立高精度 reference/oracle。

====================
B. authoritative Robertson model
====================
使用以下 ODE，不可改动：

 y1' = -0.04*y1 + 1e4*y2*y3
 y2' =  0.04*y1 - 1e4*y2*y3 - 3e7*y2^2
 y3' =  3e7*y2^2
 y(0) = [1, 0, 0]
 t in [0, 40]

化学助记反应：
A -> B                  k1 = 0.04
B + B -> B + C          k2 = 3e7
B + C -> A + C          k3 = 1e4

重要：反应标签只是 mnemonic；上述 ODE 是 authoritative。对于 B+B 反应，不要给 ODE 中的 3e7*y2^2 再额外乘以 2。

化学解释必须进入报告：
- y1,y2,y3 是 A,B,C 的归一化浓度；
- B 是短寿命中间体：A 慢慢产生 B，B 又因很快的二次反应迅速被消耗并转化为 C；
- k1 到 k2 跨度约 1e8，快慢化学时间尺度共存是刚性的物理根源；
- 总质量守恒：y1+y2+y3=1；
- 浓度必须 non-negative，负浓度是化学上不可能的数值结果。

====================
C. Topic ③ 的专门、固定 protocol
====================
1. 输出时间必须包括 t=0，且至少有 200 个从 1e-8 到 40 的 logarithmically spaced output times。允许 solver 自选内部步长，但所有方法和 reference 的结果必须插值/输出在同一组 times 上比较。

2. 建立 tight reference：用 solve_ivp 的 Radau、BDF 或 LSODA。用紧容差运行一次，再把 tolerances 缩小 100 倍运行一次；报告中只保留两个 reference run 都不变的有效数字。

3. Uniform-step convergence：至少四个连续 halved step sizes。每次运行须分别记录 stable / non-negative / accurate 状态。

4. 线性守恒关系：(1,1,1)^T f(y)=0。因此 full 3x3 Jacobian 永远有 structural zero eigenvalue。绝对不能把该零特征值放进 stiffness ratio，否则 ratio 无意义或无限大。

5. Jacobian 为：
J(y) = [[-0.04, 1e4*y3, 1e4*y2],
        [ 0.04, -1e4*y3-6e7*y2, -1e4*y2],
        [ 0, 6e7*y2, 0]]

6. 刚性比只能用两个 nonzero eigenvalues：
S(t) = max_{lambda_j != 0}|Re(lambda_j)| / min_{lambda_j != 0}|Re(lambda_j)|.
也可先消去 y3=1-y1-y2，使用 reduced 2x2 Jacobian。必须说明判断 zero eigenvalue 的数值 threshold；t=0 是 degenerate initial state，刚性比不应在那里报告，图从第一个正的 log time 开始。

7. 参考方向值（仅用于 debug，不可替代我们自己的计算）：nonzero-eigenvalue stiffness ratio 约为 3.0e3 at t=1e-4，5.4e3 at t=1e-2，1.58e5 at t=40；at t=40, |lambda|max 约为 3.39e3。因为 y2=O(1e-5)，不是直接把 3e7 当作 eigenvalue。

8. Frozen-Jacobian 估计给 explicit Euler 的 local bound：h < 2/(3.39e3) ≈ 5.9e-4。必须做 empirical step sweep 检查；它不是 sufficient nonlinear stability theorem。Jacobian 随 state 变化，eigenvalues 也可能在 non-normal 系统中具有局限。

9. 题目指定的报告问题：
- plot both nonzero Jacobian eigenvalues and stiffness ratio vs logarithmic time；解释 structural zero eigenvalue 和 initial degeneracy；
- show where explicit Euler becomes unstable, inaccurate, or negative；
- compare implicit Euler at matched error；“does not blow up”不等于“accurate”；
- track mass conservation y1+y2+y3=1、componentwise non-negativity、full-state error as separate diagnostics；守恒本身很弱，错误/负浓度解也可能守恒；
- report final y2(40) from our implicit method at several settings/tolerances，并和自己的 tight reference cross-check；
- 对 implicit method 做 Newton stopping tolerance sweep：Newton 不必到 machine precision，但 Newton algebraic error 必须相对于 time discretisation error 可忽略。展示 invariant defect/solution error 如何随 Newton tolerance 变化。

10. Sanity checks：
- 一致实现的 RK method 应代数保持 linear invariant；
- implicit Newton 若解得不精确会出现 invariant defect，该 defect 与 nonlinear residual 有关；
- t=40 independent orientation value roughly y=(0.7158271, 9.1855e-6, 0.2841637)，但报告只能写我们 tightened reference 确认的位数；
- y2 在早期峰值后应无 spurious oscillation 地下降；所有分量非负，和为 1。

====================
D. 推荐主研究问题
====================
请围绕这个问题组织整个项目：
“How do the chemically induced fast and slow time scales in the Robertson reaction network affect the stability, accuracy, positivity, and cost of explicit and implicit ODE solvers?”

中文：Robertson 反应网络中由化学反应速率差异产生的快慢时间尺度，如何影响显式和隐式 ODE 求解器的稳定性、准确性、浓度非负性与计算成本？

====================
E. 希望你协作时遵守的工作方式
====================
- 先给出简明计划，再一步一步完成；不要一次输出大量未经验证的代码。
- 每段代码都说明放在哪个文件、输入输出是什么、如何测试。
- 数值结论必须区分：reference evidence、实验测量、理论预期；不要编造实验结果。
- 将质量守恒、非负性、full-state error、终点 y2(40) 分开处理，不能以其中一项替代其他项。
- 对每一张图给出：横/纵轴、采用什么数据、它回答什么问题、预期怎样解释。
- 若我要求你写报告，请使用严谨但学生可讲清的英文，并保留中文解释帮助我理解。
- 若我要求你写代码，请优先 Python + NumPy + SciPy + Matplotlib，并构建可复现目录：src/、tests/、results/、run_all.py、README.md。

现在先不要自行选择后续任务。请先回复：
1. 你理解到的“必做工作包”清单；
2. 推荐的代码模块/文件结构；
3. 推荐的实验执行顺序；
4. 你认为最容易犯的 5 个错误。
然后等待我指定下一步。
```

---

## 8. 建议你给 WorkBuddy 的第一条后续指令

在贴完上面的 prompt 后，可以继续发送：

> 请先完成第 1 阶段：为项目设计一个可运行的 Python 代码结构和任务清单；先不要生成全部代码。请列出每个文件的函数、输入输出、计数器、单元测试，以及 `run_all.py` 的执行顺序。

这能避免它直接生成一大堆无法验证、难以整合的代码。

---

## 9. 本次更新相对于之前分析的重要修正

- Topic ③ 的完整题目其实给出了固定的对数输出时间 protocol；
- Jacobian 的结构性零特征值必须被排除在刚性比之外；
- 需要处理 $$t=0$$ 初始退化；
- 需要特别报告 implicit method 的终点 $$y_2(40)$$；
- 需要做 Newton stopping-tolerance sweep；
- 反应常数/反应标签的顺序应按完整 Problem Pack 解释：$$B+B\to B+C$$ 对应 $$3\times10^7y_2^2$$，$$B+C\to A+C$$ 对应 $$10^4y_2y_3$$；
- “守恒”不能作为“准确”的替代证明；负浓度和 full-state error 必须独立检查。

