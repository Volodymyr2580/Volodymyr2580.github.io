---
layout: page
permalink: /notes/AI/RL_3/index.html
title: 强化学习的数学原理 3
---
上一章我们学习了贝尔曼方程和State Value

强化学习的最终目的是要找到某种最优策略。本章我们将引入optimal state value和optimal policies两个核心概念以及Bellman optimality equation一个重要的工具。

一种trivial的想法是利用上一章中定义的action value，我们对某个状态，考虑每一种action对应的action value，如果有某个action能带来最大的action value，我们可以直观认为这是一个Optimal action

### Optimal state values and optimal policies

考虑两个policies $\pi_1$和$\pi_2$，如果：

$$v_{\pi_1}(s)\geq v_{\pi_2}(s), \forall s\in\mathcal{S}$$

我们认为$\pi_1$要优于$\pi_2$，进一步，如果有一个policy比其他每一个都要优，自然是optimal policy。数学上表述为：

**Def 3.1** A policy $\pi^*$ is optimal if $v_{\pi^*}(s) \geq v_\pi(s), \text{for all }s\in \mathcal{S} \text{ and for any other policy } \pi$ 。The state values of $\pi^*$ are the optimal state values.

自然的问题是：这样的policy一定存在吗？若存在，其唯一吗？策略应该混合吗？如何得到这样的optimal policy？

### Bellman optimality equation

BOE，研究最优策略的工具。
对每一个state $s\in \mathcal{S}$, BOE的逐元素表达式为：

$$
\begin{align}
v(s) &= \max_{\pi(s) \in \Pi(s)} \sum_{a \in \mathcal{A}} \pi(a|s) \left( \sum_{r \in \mathcal{R}} p(r|s, a)r + \gamma \sum_{s' \in \mathcal{S}} p(s'|s, a)v(s') \right)\\
& = 
\max_{\pi(s) \in \Pi(s)} \sum_{a \in \mathcal{A}} \pi(a|s)q(s, a)

\end{align}
$$

其中$v(s),v(s')$都是待求解的量。

$$q(s, a) \doteq \sum_{r \in \mathcal{R}} p(r|s, a)r + \gamma \sum_{s' \in \mathcal{S}} p(s'|s, a)v(s')
$$

第一步是求解max，考虑到$\sum_a \pi(a|s) =1$

$$\sum_{a\in\mathcal{A}} \pi(a|s)q(s,a) \leq \sum_{a\in\mathcal{A}}\pi(a|s) \max_{a\in\mathcal{A}} q(s,a)=\max_{a\in\mathcal{A}} q(s,a)$$

Suppose $a^*=\text{arg } max_a q(s,a)$，上式的等号取自

$$
\pi(a|s) = \begin{cases} 
1, & a = a^*, \\
0, & a \neq a^*.
\end{cases}
$$

总结来说, optimal policy是那个选择了拥有最大$q(s,a)$的action的policy

将逐state的方程组合起来会得到BOE的矩阵形式：

$$v = \max_{\pi \in \Pi}(r_\pi+\gamma P_\pi v)$$

其中v表示state value的向量，max是逐元素取max。

$$
[r_\pi]_s \doteq \sum_{a \in \mathcal{A}} \pi(a|s) \sum_{r \in \mathcal{R}} p(r|s, a)r
$$

$$
[P_\pi]_{s,s'} = p(s'|s) \doteq \sum_{a \in \mathcal{A}} \pi(a|s)p(s'|s, a)
$$

考虑上式右侧所得到的optimal policy $\pi$实际上取决于v，因此可以表示为v的函数

$$
f(v) \doteq \max_{\pi \in \Pi}(r_\pi + \gamma P_\pi v)
$$

因此BOE的简洁形式为：

$$v=f(v)$$

即f的一个不动点。

根据压缩映射原理(contraction mapping theorem), 若映射f 是压缩映射——即$\forall x_1, x_2, \exists \gamma \in (0,1)$，使得

$$||f(x_1)-f(x_2)||\leq \gamma||x_1-x_2||$$

则其存在唯一的不动点$x^*$，且可以通过迭代$x_{k+1}=f(x_k)$得到。

那BOE右侧的函数f是不是一个压缩映射呢？答案是肯定的。

实际上max对应的就是$L_{\infty}$范数，因此有for any $v_1,v_2 \in \mathbb{R}^{|S|}$,

$$||f(v_1)-f(v_2)||_{\infty}\leq \gamma||v_1-v_2||_{\infty}$$

Proof:

Suppose $\pi_1^* \doteq \text{arg }max_\pi (r_\pi+\gamma P_\pi v_1)$ and $\pi_2^* \doteq \text{arg }max_\pi (r_\pi+\gamma P_\pi v_2)$

$$\begin{align}
f(v_1) &= \max_\pi (r_\pi + \gamma P_\pi v_1) = r_{\pi_1^*} + \gamma P_{\pi_1^*} v_1 \geq r_{\pi_2^*} + \gamma P_{\pi_2^*} v_1, \\

f(v_2) &= \max_\pi (r_\pi + \gamma P_\pi v_2) = r_{\pi_2^*} + \gamma P_{\pi_2^*} v_2 \geq r_{\pi_1^*} + \gamma P_{\pi_1^*} v_2,
\end{align}$$

Thus,

$$\begin{align}
f(v_1) - f(v_2) &= r_{\pi_1^*} + \gamma P_{\pi_1^*} v_1 - (r_{\pi_2^*} + \gamma P_{\pi_2^*} v_2) \\ 
&\leq r_{\pi_1^*} + \gamma P_{\pi_1^*} v_1 - (r_{\pi_1^*} + \gamma P_{\pi_1^*} v_2) 
&= \gamma P_{\pi_1^*} (v_1 - v_2)\\
\end{align}$$

Similarly,

$$f(v_2) - f(v_1) \leq \gamma P_{\pi_2^*} (v_2 - v_1)$$

Define

$$
z \doteq \max \left\{ |\gamma P_{\pi_2^*}(v_1 - v_2)|, |\gamma P_{\pi_1^*}(v_1 - v_2)| \right\} \in \mathbb{R}^{|\mathcal{S}|}
$$

上式中出现的运算符都是elementwise的，容易看到

$$ -z \leq \gamma P_{\pi_2^*}(v_1 - v_2) \leq f(v_1) - f(v_2) \leq \gamma P_{\pi_1^*}(v_1 - v_2) \leq z, $$
so, 

$$ |f(v_1) - f(v_2)| \leq z. $$
which means,

$$  \|f(v_1) - f(v_2)\|_\infty \leq \|z\|_\infty, $$

另一方面, 

$$z_i = \max \{\gamma |p_i^T(v_1 - v_2)|, \gamma |q_i^T(v_1 - v_2)|\} $$

其中,$p_i^T,q_i^T$分别是$P_{\pi_1^*},P_{\pi_2^*}$的第i行，其各个元素非负且和为1，则

$$ |p_i^T(v_1 - v_2)| \leq p_i^T|v_1 - v_2| \leq \|v_1 - v_2\|_\infty. $$

类似有 

$$|q_i^T(v_1 - v_2)| \leq q_i^T|v_1 - v_2| \leq \|v_1 - v_2\|_\infty  $$

Thus, $z_i \leq \gamma ||v_1-v_2||_{\infty}$

$$ \|f(v_1) - f(v_2)\|_\infty \leq \gamma \|v_1 - v_2\|_\infty, $$

#### BOE求解最优策略
从上一节的数学原理可以看到，我们只需从一个initial guess $v_0$出发，然后做迭代$v_{k+1}=f(v_k)$即可收敛到唯一存在的不动点$v^*$，即最优state value vector

这一迭代策略有一个单独名字叫value iteration。一旦$v^*$求解出来，optimal policy $\pi^* = \text{arg } \max_{\pi \in \Pi}(r_\pi+\gamma P_\pi v^*)$

In fact, 还有一条性质

$$\text{for any policy} \pi, v^* = v_{\pi^*} \geq v_\pi$$

we can see it from the following facts:

$$ v^* - v_\pi \geq (r_\pi + \gamma P_\pi v^*) - (r_\pi + \gamma P_\pi v_\pi) = \gamma P_\pi (v^* - v_\pi) $$

$$v^* - v_\pi \geq \gamma P_\pi (v^* - v_\pi)\geq \gamma^2 P_\pi^2 (v^* - v_\pi)\geq \cdots\geq 
\lim_{n \to \infty} \gamma^n P_\pi^n (v^* - v_\pi) = 0
$$

有一个自然的问题是，贪心算法是不是Optimal的？下面一个定理回答了这个问题

Theorem 3.5 (Greedy optimal policy).} For any $s \in \mathcal{S}$, the deterministic greedy policy

$$
\pi^*(a|s) = \begin{cases} 
1, & a = a^*(s), \\
0, & a \neq a^*(s),
\end{cases}
$$

是一个optimal policy，其中

$$a^*(s)= \text{arg } \max_a q^*(a,s)$$

$$q^*(s, a) \doteq \sum_{r \in \mathcal{R}} p(r|s, a)r + \gamma \sum_{s' \in \mathcal{S}} p(s'|s, a)v^*(s').$$

这其实可以从elementwise的BOE解形式得到：

$$ \pi^*(s) = \arg\max_{\pi \in \Pi} \sum_{a \in \mathcal{A}} \pi(a|s) \left( \underbrace{\sum_{r \in \mathcal{R}} p(r|s, a)r + \gamma \sum_{s' \in \mathcal{S}} p(s'|s, a)v^*(s')}_{q^*(s, a)} \right), \quad s \in \mathcal{S}. $$
It is clear that RHS is maximized if $\pi(s)$ selects the action with the greatest $q^*(s,a)$

总结来说，最优的optimal value是唯一存在的，但其对应的optimal policy 解可以有无穷多。贴现因子、回报值、和一些概率分布都会影响BOE的解。
