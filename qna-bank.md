# Robertson 刚性 ODE · Q&A 演练稿

**对应 16 页 PPT ｜ 40 题 ｜ ⭐ = 必背 15 题（老师最可能问）**

> **老师的范围**：不问具体代码，主要考 **ODE 基础** 和 **模型细节**。
> 所以下面每一题的答案，**只用 PPT 上出现过的词和数字**，不引入新东西。
>
> **答题三句骨架**：① 一句结论（就是 PPT 上那句话）→ ② 一句为什么 / 一个数字 → ③ 指一下在哪页。
> **英文答句里加粗的，就是屏幕上的原词**，照着念即可；`译` 是中文意思，自己听懂用；`页` 是这题答案在哪一页，老师追问时可以直接翻。

---

## 必背 TOP 15（先背这 15 句，其余都是它们的变体）

| # | 一句话答案（背这句） |
|---|---|
| M1 | It is the **1966 stiff benchmark**: **three species**, **three reactions**, rate constants from **0.04 to 3×10⁷**. |
| M4 | **Adding the three equations, every term cancels in pairs** — so **y₁+y₂+y₃ ≡ 1**, always. |
| S1 | **One very fast process and one very slow process standing side by side** — that is stiff. |
| S2 | The two speeds **are eigenvalues of the Jacobian**; their ratio is **1.58×10⁵** at t = 40. |
| S6 | Every explicit method has a **step ceiling**: **EE 5.9×10⁻⁴ s, RK4 8.2×10⁻⁴ s**; **implicit Euler has none**. |
| S9 | **Stability decides whether you can keep going; accuracy decides whether the answer is right** — not the same thing. |
| N1 | **Explicit Euler only looks at now**; **implicit Euler looks ahead**, so **every step solves an equation**. |
| N3 | We **measure** the order from the fit: **1.08 / 4.27 / 1.00** — **all three match the textbook**. |
| N5 | The rates are **nonlinear** (y₂², y₂y₃), so **every implicit step solves an equation by Newton** — that is why an implicit step costs **3×**. |
| N10 | **Implicit is steadier, but not cheaper**: at error 10⁻⁶ it costs **11 against 3.0**. |
| R1 | We **tighten the tolerance until the digits stop changing**, then **cross-check against BDF and Radau**. |
| R4 | Explicit Euler at h = 3×10⁻³: **conservation 1.1×10⁻¹³ looks perfect, yet min concentration −921.8**. |
| R5 | The number we hand in: **y₂(40) = 9.185535×10⁻⁶**. |
| R6 | Same tolerance: **RK45 241,994 evaluations vs BDF 452** — about **100× faster**. |
| R7 | **Stability forces every explicit step below 10⁻⁴ s** — so use an **implicit (stiff) solver**. |

---

## A. 模型细节（Model）

### M1 ⭐ What is the Robertson problem?
- **Q**：What is the Robertson problem?（这是什么问题？）
- **答**：It is a **1966 stiff benchmark**: **three species** A, B and C **interconvert through three reactions**, and the **rate constants span 0.04 to 3×10⁷**. We start from **y(0) = (1, 0, 0)** and solve on **t in [0, 40]** seconds.
- **译**：1966 年的刚性基准题：三个物种经三个反应互相转化，速率常数从 0.04 跨到 3×10⁷；初值 (1,0,0)，在 t∈[0,40] 上求解。
- **页**：P2

### M2 Which reactions? Write the model.
- **Q**：What are the three reactions?（三个反应是什么？）
- **答**：**A goes to B** (k₁ = 0.04), **B plus B gives B plus C** (k₂ = 3×10⁷), and **B plus C gives A plus C** (k₃ = 1×10⁴). By the **law of mass action**, each rate is **rate constant × product of reactant concentrations**, so **y₁′ = −r₁ + r₃**, **y₂′ = +r₁ − r₂ − r₃**, **y₃′ = +r₂**.
- **译**：A→B、B+B→B+C、B+C→A+C；按质量作用定律，速率 = 速率常数 × 反应物浓度之积，于是三条方程如左。
- **页**：P2

### M3 Is this a linear system?
- **Q**：Is the system linear?（是线性系统吗？）
- **答**：No, it is **nonlinear**: the rates contain **y₂ squared** and **y₂ times y₃**. That is exactly why an implicit step can no longer be solved by hand and needs **Newton's method**.
- **译**：不是。速率里有 y₂² 和 y₂y₃；正因如此隐式步不能直接解，必须用 Newton 迭代。
- **页**：P2、P7

### M4 ⭐ Why is y₁+y₂+y₃ always exactly 1?
- **Q**：Why is the total concentration conserved?（总浓度为什么守恒？）
- **答**：**Add the three equations** and **every term cancels in pairs** — r₁ against r₁, r₂ against r₂, r₃ against r₃ — so **d/dt(y₁+y₂+y₃) = 0**, and from **(1,0,0)** we always have **y₁+y₂+y₃ = 1**. In the Jacobian language, **(1,1,1)ᵀJ = 0**: the system has **one structural zero eigenvalue**.
- **译**：三式相加逐项抵消，总和恒定（恒为 1）。用 Jacobian 说就是 (1,1,1)ᵀJ=0，含一个结构零特征值。
- **页**：P2、P4、P11

### M5 Over what interval, and is it autonomous?
- **Q**：What time interval? Does f depend on t?（时间区间？f 显含 t 吗？）
- **答**：**t from 0 to 40** seconds. The system is **autonomous**: the right-hand side depends on the state **y** only; **t** just sits in the argument for the solver's convenience.
- **译**：0 到 40 秒。系统自治，右端只依赖状态 y。
- **页**：P2

### M6 Why does B peak so early?
- **Q**：Why does B peak at t ≈ 0.004 s?（B 为什么这么早到峰？）
- **答**：**B jumps up from essentially zero**, but the moment it appears, **k₂ = 3×10⁷** consumes it as **y₂²** — the faster it builds, the faster it burns. So B **peaks at t ≈ 0.004 s** at only **3.65×10⁻⁵**, then **decays fast**.
- **译**：B 一旦生成就被 k₂ 以 y₂² 的速率迅速消耗，越堆越烧，所以峰很早、很低（3.65×10⁻⁵ @ t≈0.004 s），随后快速衰减。
- **页**：P3

### M7 Why are A and C slow, and why is y₂(40) so small?
- **Q**：Why do A and C move slowly? Why is y₂(40) so tiny?（A、C 为什么慢？终点 B 为什么这么小？）
- **答**：After the initial spike, **B is consumed as fast as it is produced**, so the remaining motion is controlled by the slow feed **k₁ = 0.04**: **A slides from 1 down to 0.72** and **C rises from 0 to 0.28** on a **tens-of-seconds** timescale, **which has nothing to do with B**. So at t = 40 only **9.2×10⁻⁶** of B is left.
- **译**：快瞬态过后，B 生成多快就被消耗多快，剩下的变化由慢速率 k₁=0.04 控制：A 从 1→0.72、C 从 0→0.28，时间尺度是几十秒，与 B 无关；t=40 时 B 只剩 9.2×10⁻⁶。
- **页**：P3

### M8 What has to be handed in?
- **Q**：What do you hand in?（要交什么？）
- **答**：**Three things**: the **three concentration curves** y₁(t), y₂(t), y₃(t); **one specific number, y₂(40)**; and an explanation of **why this number can be believed**.
- **译**：三件：三条浓度曲线、一个数 y₂(40)、以及"为什么这个数可信"的说明。
- **页**：P2、P3、P15

---

## B. 刚性 & ODE 基础（Stiffness）

### S1 ⭐ What does "stiff" mean?
- **Q**：What does stiffness mean in this problem?（这题里"刚性"指什么？）
- **答**：**One very fast process and one very slow process standing side by side, with a large gap between them.** The fast world has to be computed **together with** the slow one, and **that is the annoying part**.
- **译**：一个极快的过程和一个极慢的过程并存、差距很大；快的部分必须和慢的一起算，这就是麻烦所在。
- **页**：P4

### S2 ⭐ How do you measure stiffness?
- **Q**：How do you measure "stiff"?（怎么量化刚度？）
- **答**：By the **two speeds of the system**: they are the **eigenvalues of the Jacobian**, and a mode decays at speed **|Re λ|**. At **t = 40** we get **λ = −3392.79** (a clock of **2.95×10⁻⁴ s**) and **λ = −0.0214** (**46.7 s**), so the **stiffness ratio** is **S = 1.58×10⁵**.
- **译**：用系统的两个速度衡量：即 Jacobian 的特征值，模式按 |Re λ| 衰减。t=40 时 λ=−3392.79（时钟 2.95e-4 s）与 −0.0214（46.7 s），比值 S=1.58e5。
- **页**：P4

### S3 Why do you need the Jacobian?
- **Q**：为什么要用雅可比？（Why the Jacobian?）
- **答**：Give the state a small perturbation **δ**: **f(y+δ) = f(y) + J(y)δ + O(δ²)**, so **δ′ = J(y)δ** and each mode decays like **e^{λt}**. The Jacobian, **J(y) = ∂f/∂y**, is what turns "how fast" into algebra.
- **译**：给状态一个扰动 δ，展开得 δ′=J(y)δ，模式按 e^{λt} 衰减；雅可比 J=∂f/∂y 把"多快"变成可算的东西。
- **页**：P7、P4

### S4 Why is one eigenvalue exactly zero?
- **Q**：One eigenvalue is zero — why?（为什么有一个零特征值？）
- **答**：Because **adding the three columns of J gives (1,1,1)ᵀJ = 0**: moving along the direction **(1,1,1)** changes nothing, since that direction is exactly **conservation**. So **one mode is a structural zero**, and the two **live** speeds come from the **reduced 2×2 Jacobian**.
- **译**：因为 J 的三列相加为零，沿 (1,1,1)（守恒方向）不变，所以有一个结构零特征值；两个"活"的速度来自降阶的 2×2 雅可比。
- **页**：P4

### S5 What is the test equation? What is a stability region?
- **Q**：What is the test equation?（什么是测试方程？）
- **答**：**y′ = λy**. One step of a method **multiplies y by a factor R(z)** with **z = hλ**, and the step is **safe wherever |R(z)| ≤ 1**. On the negative real axis, **explicit Euler is safe to z = −2**, **RK4 to z ≈ −2.79**, **implicit Euler everywhere**.
- **译**：y′=λy。一步相当于把 y 乘以 R(z)（z=hλ），|R(z)|≤1 就是安全区。负实轴上 EE 到 −2，RK4 到 −2.79，隐式 Euler 全覆盖。
- **页**：P5

### S6 ⭐ Is there a limit on the step size?
- **Q**：Do you have a step-size limit?（步长有上限吗？）
- **答**：Yes — every explicit method has a **step ceiling**, and it comes from the **fastest eigenvalue of this problem**, **|λ| = 3392.79 s⁻¹** at t = 40, times each method's **real-axis edge**: **Explicit Euler 5.9×10⁻⁴ s** (2 ÷ 3392.79) and **RK4 8.2×10⁻⁴ s** (2.79 ÷ 3392.79). **Implicit Euler has none.**
- **译**：有。天花板由最快特征值 |λ|=3392.79（t=40）与方法的负实轴边界决定：EE 5.9e-4 s、RK4 8.2e-4 s；隐式 Euler 没有。
- **页**：P5

### S7 What happens if the step is too large?
- **Q**：What if the step is too big?（步长太大会怎样？）
- **答**：If the step is too large, **the error is amplified at every step**, and the solution **collapses**: at **Explicit Euler with step 3×10⁻³** the concentration reads **−921.8**, which is **chemically unacceptable**.
- **译**：误差每步被放大，解崩溃：EE h=3e-3 时浓度读到 −921.8，化学上不可接受。
- **页**：P5、P11

### S8 What does that ceiling cost you?
- **Q**：What does the ceiling cost you?（这个上限的代价是什么？）
- **答**：With a ceiling of **5.9×10⁻⁴ s**, covering **40 seconds takes at least 6.8×10⁴ steps** — that is just 40 divided by the ceiling. In practice **you need even more**.
- **译**：上限 5.9e-4 s，40 秒至少要 6.8e4 步（就是 40 除以上限）；实际只会更多。
- **页**：P5

### S9 ⭐ Is stability the same as accuracy?
- **Q**：Is stability the same as accuracy?（稳定等于准确吗？）
- **答**：No. **Stability decides whether you can keep going; accuracy decides whether the answer is right.** The clearest example: a run with **conservation 1.1×10⁻¹³ — it looks perfect — yet min concentration −921.8 and error 1.08×10¹⁰**.
- **译**：不是。稳定性管"能不能算下去"，精度管"算得对不对"。反例：守恒 1.1e-13 看着完美，但 min y = −921.8、误差 1.08e10。
- **页**：P11

### S10 Can the step be relaxed halfway?
- **Q**：Can you use a big step in the easy part?（后半段能不能放大步长？）
- **答**：No — **the ceiling is set by the fastest process, and the fastest is at t = 40, the very end of the run**, so it **binds for the whole integration**. The sweep confirms it: the **first step that dies** is only about 15% above the largest stable one.
- **译**：不能。上限由最快的过程决定，而它恰好在 t=40（全程末尾）最大，所以整段计算都被卡住；实测也证实：崩掉的步长只比跑通的大 15% 左右。
- **页**：P5、P10

---

## C. 方法 & 数值基础（Methods）

### N1 ⭐ What is the difference between the two Euler methods?
- **Q**：Explicit vs implicit Euler?（显式与隐式欧拉的区别？）
- **答**：**Explicit Euler only looks at now**: **y_{n+1} = y_n + h f(t_n, y_n)** — **one evaluation**, then **walk straight along the tangent**. **Implicit Euler looks ahead**: **y_{n+1} = y_n + h f(t_{n+1}, y_{n+1})** — the unknown sits **on both sides**, so **every step solves an equation**.
- **译**：显式只用当前斜率、一次求值、沿切线走直线；隐式未知量在两边，每步都要解方程。
- **页**：P7

### N2 How does RK4 work?
- **Q**：How does RK4 take a step?（RK4 怎么走一步？）
- **答**：**RK4 samples the step before committing**: **four slopes** — **left end, midpoint twice, right end** — and takes a weighted average, **y_{n+1} = y_n + h(k₁+2k₂+2k₃+k₄)/6**. That is **four evaluations per step**, in exchange for **4th-order accuracy**.
- **译**：RK4 用四个斜率（左端、两次中点、右端）加权平均；每步 4 次求值换来 4 阶精度。
- **页**：P7

### N3 ⭐ Do your methods really have the right order?
- **Q**：How do you check the order?（阶数怎么验证？）
- **答**：We do not **claim** the order, we **measure it from the fit**: on a **log–log plot** a p-th-order method is a **straight line of slope p**. Measured: **Explicit Euler 1.08**, **RK4 4.27**, **Implicit Euler 1.00** — **all three match the textbook**, so the implementations are **correct**.
- **译**：不声称而是测量：log-log 图上 p 阶方法是一条斜率为 p 的直线。测得 1.08 / 4.27 / 1.00，与教科书一致，说明实现正确。
- **页**：P8

### N4 Why does one implicit step cost more?
- **Q**：Why is an implicit step more expensive?（隐式一步为什么更贵？）
- **答**：Because every step **solves a nonlinear system**, by **Newton's method**: each correction costs **one f, one Jacobian and one solve**. So **each step costs about 3× more** than explicit Euler — the trade is that the step itself **may be much larger**.
- **译**：每步要解非线性方程组（Newton）：每次迭代花 1 次 f、1 次雅可比、1 次求解，单步约贵 3 倍；换来的是步长可以大得多。
- **页**：P7、P13

### N5 ⭐ Can you explain Newton's method in your solver?
- **Q**：What does Newton's method do here?（Newton 迭代怎么做？）
- **答**：Every step must **solve an equation**, because the rates are **nonlinear** (y₂² and y₂y₃), so the unknown sits **on both sides**. Newton fixes a first guess step by step until both sides agree, and each correction costs **one f, one Jacobian, one solve** — that is why **an implicit step costs about 3× an explicit one**.
- **译**：速率是非线性的（y₂²、y₂y₃），未知量出现在等式两边，所以每步要解方程；Newton 从一个初猜出发逐次修正直到两边一致，每次修正花 1 次 f、1 次雅可比、1 次求解 —— 所以隐式一步约是显式的 3 倍成本。
- **页**：P7、P13
- **追问**：If he asks “what equation exactly?” — the step condition $y_{n+1}=y_n+h\,f(y_{n+1})$ itself: keep guessing $y_{n+1}$ until both sides agree. （PPT 上说到「每次迭代就是解一个 3×3 的线性方程组」为止，**不写迭代公式，也不要展开**。）

### N6 Why did you write the Jacobian by hand?
- **Q**：Why an analytic Jacobian?（为什么手写解析雅可比？）
- **答**：Two reasons: **implicit Euler needs it** — Newton solves the step equation with the Jacobian — and our integrators **count Jacobian calls** as part of the cost. The analytic form is exact and only a few lines.
- **译**：两个原因：隐式 Euler 的 Newton 迭代需要雅可比；而且我们的积分器把雅可比调用也计进成本。解析形式精确，代码也只有几行。
- **页**：P4、P13

### N8 What are rtol and atol?
- **Q**：What are rtol and atol?（两个容差是什么？）
- **答**：**rtol** is the **relative** tolerance and **atol** the **absolute** one; the solver keeps its estimated error below them. Our stiff run uses **rtol = 1e−6**, and every solver in the comparison ran **at the same tolerance**, so the comparison is fair.
- **译**：rtol 是相对容差、atol 是绝对容差；我们的刚性求解用 rtol=1e-6，图里所有求解器都在同一容差下跑，比较才公平。
- **页**：P14

### N9 What solver did you use for the reference? And what is nfev?
- **Q**：什么求解器？nfev 是什么？
- **答**：Two **stiff solvers**, **BDF** and **Radau**, are cross-checked against each other. **nfev** means the number of **function evaluations** — how many times the solver called the right-hand side — which is a fair way to compare work.
- **译**：用两个刚性求解器 BDF 和 Radau 互相校验；nfev = 右端求值次数，是公平的工作量度量。
- **页**：P6、P14

### N10 ⭐ Since implicit takes big steps, is it cheaper?
- **Q**：隐式更省吗？（Is implicit cheaper?）
- **答**：**Implicit is steadier, but not cheaper.** At an error of **10⁻⁶**, **Explicit Euler costs 3.0, RK4 3.2, and Implicit Euler 11** — implicit Euler is **only 1st order**, so it must take very small steps, each 3× the cost. **Stability buys a bigger step, not accuracy.**
- **译**：不更省。目标误差 1e-6 时 EE 3.0、RK4 3.2、IE 11——IE 只有 1 阶，必须取很小的步，每步又贵 3 倍。稳定性买到的是"更大的步"，不是"更高的精度"。
- **页**：P12

### N11 Which method would you pick for a problem like this?
- **Q**：你会选哪个方法？为什么？
- **答**：For a **stiff problem**, use an **implicit (stiff) solver** — that is our rule of thumb. Among the three textbook methods, **RK4 is the best value** here: it matches implicit at 10⁻⁶ for cost, but it is **4th order**, and it is the only one that reaches **10⁻⁷**.
- **译**：刚性题就用隐式（刚性）求解器；三个方法里 RK4 最划算——与隐式同价位但 4 阶，还是唯一能做到 1e-7 的。
- **页**：P14

---

## D. 验证 & 结果（Verification & results）

### R1 ⭐ How do you know the reference is right?
- **Q**：参考解凭什么可信？
- **答**：We **tighten the tolerance until the digits stop changing**, and only quote the digits that survive; then we **cross-check against two off-the-shelf stiff solvers, BDF and Radau**. Independent implementations agreeing is what makes the reference trustworthy.
- **译**：反复调紧容差直到数字不再变，只引用活下来的位数；再跟两个现成的刚性求解器 BDF、Radau 交叉对照 —— 独立实现彼此吻合，参考解才可信。
- **页**：P6

### R2 What are the checks?
- **Q**：你做了哪些检查？
- **答**：**Three checks plus the endpoint**, **one failure each**: **Conservation** — whether **y₁+y₂+y₃ stays at 1**; **Non-negativity** — whether **a concentration ever goes negative**; **Accuracy** — **point-by-point against the reference**; and the fourth card, **Endpoint** — whether **y₂(40) matches**. **The first three checks are mutually independent**, and the nastiest combination is **"the first two pass, the third is way off"**.
- **译**：三道检查加终点值，各管一种失效：守恒、非负、精度（逐点比对）、终点值。前三道相互独立，最坏组合是"前两道过、第三道差得远"。
- **页**：P9

### R3 Then why keep the conservation check?
- **Q**：守恒检查还有意义吗？
- **答**：It **cannot fail**: **add the three equations** and **the terms cancel in pairs**, so **every method keeps it automatically**, good run or bad. The **10⁻¹³** in the plot is just the computer's **rounding error** — **this check can never fail**, and the other two are the checks that can.
- **译**：它不可能失败：三式相加逐项抵消，任何方法都自动保持。图里那个 1e-13 只是舍入误差——这道检查永远不会不及格，能抓错的是另外两道。
- **页**：P11

### R4 ⭐ What is the most striking single run?
- **Q**：最扎眼的一次运行是？
- **答**：**Explicit Euler at step 3×10⁻³**: **conservation 1.1×10⁻¹³ — it looks perfect — yet min concentration −921.8 and error 1.08×10¹⁰**. **Checking conservation alone would have waved this through** — so **surviving is not correct**.
- **译**：EE 取 h=3e-3：守恒 1.1e-13 看着完美，但 min y=−921.8、误差 1.08e10。只看守恒就会放它过——"活着"不等于"正确"。
- **页**：P11

### R5 ⭐ What number do you hand in?
- **Q**：要交的数是多少？
- **答**：**y₂(40) = 9.185535×10⁻⁶**, safe to quote **to 10 digits**. We also report the **peak 3.65×10⁻⁵ at t ≈ 0.004 s**, because it is **4 times larger** than the endpoint.
- **译**：y₂(40)=9.185535e-6，可报 10 位有效数字；同时报峰值 3.65e-5（t≈0.004 s），它比终点大 4 倍。
- **页**：P6、P15

### R6 ⭐ How big is the stiff vs non-stiff difference?
- **Q**：刚性和非刚性差多少？
- **答**：At **the same tolerance**, the explicit solvers needed **241,994 evaluations for RK45** and 214,706 for DOP853, while the implicit ones needed only **752 for Radau and 452 for BDF** — **about 500× fewer evaluations**, and about **100× in wall-clock time**. **All four give the same y₂(40).**
- **译**：同一容差下，显式的 RK45 要 241,994 次求值、DOP853 要 214,706 次；隐式的 Radau 只要 752 次、BDF 452 次 —— 求值次数少约 500 倍，按运行时间约 100 倍；四个求解器算出的 y₂(40) 一模一样。
- **页**：P14
- **追问（口径）**：PPT 标题写的是 **100×**（按时间的大致口径），图上是**求值次数**，比值约 **530×**。若老师追问精确值，**先确认他问的是时间还是次数**，再给对应的数。

### R7 ⭐ Why are the non-stiff solvers so slow?
- **Q**：非刚性为什么这么慢？
- **答**：**Stability forces every explicit step below 10⁻⁴ s**, so **40 seconds takes hundreds of thousands of steps**; an **implicit method has no such limit**. So the **rule of thumb: for a stiff problem, use an implicit (stiff) solver.**
- **译**：稳定性把每个显式步压在 1e-4 s 以下，40 秒就要几十万步；隐式没有这个限制。口诀：刚性题用隐式求解器。
- **页**：P14

### R8 Can you reproduce all these numbers?
- **Q**：这些数能复现吗？
- **答**：Yes: **every number in the report is produced by our own code** — **not a single number is typed by hand**. SciPy is used **only for the reference cross-check**; **EE / RK4 / IE and Newton are all implemented by us**, each under 20 lines.
- **译**：可以。报告里每个数字都由我们自己的代码产出，没有一个数字是手敲的；SciPy 只用于参考解交叉校验，EE/RK4/IE 与 Newton 全部自己实现，每段不到 20 行。
- **页**：报告内容（PPT 已不单列这一页）

---

## E. 开放题 & 收尾（Open）

### O1 What is the main lesson?
- **Q**：这个项目你学到什么？
- **答**：Three things: check accuracy **point-by-point, never by conservation**; for a **stiff problem, use a stiff solver**; and **be careful with tolerances**, because a **loose tolerance quietly changes the method**.
- **译**：三点：精度要逐点比、不能靠守恒；刚性题用刚性求解器；容差要小心，松容差会悄悄改变方法本身。
- **页**：P15

### O2 What are the limitations?
- **Q**：有什么局限？
- **答**：The stability bound is a **diagnostic, not a theorem**: we predicted **5.90×10⁻⁴** and the largest step that survived was **5.42×10⁻⁴** — **8% apart**, good enough to guide, not to prove. And our cost model is **counting**: one f, one Jacobian, one solve, each worth **1**.
- **译**：稳定性界是判据不是定理：预测 5.90 vs 实测 5.42，差 8%，够指导但不构成证明；成本模型是计数的近似。
- **页**：P10、P15

### O3 Why is this problem a benchmark?
- **Q**：为什么这题是基准题？
- **答**：Because it is **small but hard**: only **three species and three reactions**, yet **nonlinear** and with a **stiffness ratio of 1.58×10⁵** — and because there is a **known reference answer**, any method can be checked against it.
- **译**：因为它小但难：三个物种、三个反应，非线性，刚度比 1.58e5；且有公认参考解，任何方法都能拿来对照。
- **页**：P2

### O4 If he does ask about code
- **Q**：Show me the code / how is the model coded?
- **答**：**The model is just the three reactions written as f(t, y)** — on the slide we show that f, and the **Jacobian** it implies on page 4. Explicit Euler, RK4, Implicit Euler and Newton are **written by us**, each under 20 lines; **the reference comes from the off-the-shelf stiff solvers BDF and Radau**, and **every figure comes out of this code**.
- **译**：模型就是三条反应写成 f(t,y)（PPT 第 2 页给出方程，第 4 页给出它对应的雅可比）；EE/RK4/IE 与 Newton 都是我们自己写的，每段不到 20 行；参考解来自 SciPy 的 BDF 与 Radau，所有图都出自这段代码。
- **页**：P2、P4

---

## 救场句（Q&A 专用，背下来）

| 情况 | 说这句 |
|---|---|
| 没听清 | **“Sorry, could you repeat the question a bit more slowly?”** |
| 想确认问题 | **“Do you mean … ?”** |
| 完全不会 | **“That is a good question. Let me check the data and come back to it.”** |
| 答到一半卡住 | **“Let me put it in one sentence.”**（然后念该页标题） |
| 收尾 / 抢时间 | **“That is exactly what conclusion 2 shows.”**（把问题交回 PPT） |

---

## 红线（这几句话绝对不要说错）

1. ❌ “**Implicit is cheaper.**” → ✅ **Implicit is steadier, but not cheaper.**
2. ❌ “**Conservation passed, so the answer is right.**” → ✅ **This check can never fail; only point-by-point comparison bites.**
3. ❌ “**Higher order means a bigger allowed step.**” → ✅ For explicit methods the ceiling comes from **stability** (the fastest eigenvalue), **not from the order**; RK4's ceiling is only slightly larger (2.79 vs 2.00).
4. ❌ “**The stiff solver is 530× faster.**” → ✅ Say **about 100×** (as on the slide); if asked to be precise, state the basis — **time ≈ 110×**, **evaluations ≈ 530×**.
5. ❌ “**RK4 is implicit** / **BDF is explicit**.” → ✅ RK4 is explicit; BDF is a stiff (implicit) solver.
6. ❌ “**λ is positive / the system is unstable.**” → ✅ **Both eigenvalues are negative**, so every mode decays.
7. ❌ 现场心算（如 2 ÷ 3392.79）。 → ✅ 直接报 **the ceiling is 5.9×10⁻⁴ s**。

---

## 数字怎么念（Q&A 常用）

| 屏幕上 | 念成 |
|---|---|
| 0.04 / 3×10⁷ / 1×10⁴ | zero point zero four / three times ten to the seven / one times ten to the four |
| 1.58×10⁵ | one point five eight times ten to the five |
| 5.9×10⁻⁴ / 8.2×10⁻⁴ | five point nine / eight point two times ten to the minus four |
| 3392.79 | three thousand three hundred ninety-two point seven nine |
| 3.65×10⁻⁵ | three point six five times ten to the minus five |
| 9.185535×10⁻⁶ | nine point one eight five three five times ten to the minus six |
| 1.1×10⁻¹³ | one point one times ten to the minus thirteen |
| 1.08×10¹⁰ | one point zero eight times ten to the ten |
| 241,994 / 452 | two hundred forty-one thousand nine hundred ninety-four / four hundred fifty-two |
| y₂(40) | **y two at forty** |
| nfev | **N-F-E-V** |
| BDF / RK45 / DOP853 | **B-D-F** / **R-K-four-five** / **D-O-P-eight-five-three** |
