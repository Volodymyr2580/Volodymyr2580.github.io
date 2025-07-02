---
layout: page
permalink: /notes/AI/RL_4/index.html
title: 强化学习的数学原理 4
---
上一章我们定义了Optimal Policy和optimal value两个用于求解最优策略的核心概念和手段——BOE。本章节我们进一步介绍3个算法——value iteration, policy iteration & truncated policy iteration

### Value iteration
回忆：

$$v_{k+1}=\max_{\pi \in \Pi}(r_\pi+\gamma P_\pi v_k)$$

每一步更新都要分两步走，第一步先根据$v_k$算出$\pi_{k+1}$, 称为policy update; 第二步是value update：

$$ v_{k+1} = r_{\pi_{k+1}} + \gamma P_{\pi_{k+1}} v_k, $$

下面我们逐元素地来细看一下：
第一步：

$$ \pi_{k+1}(s) = \arg\max_\pi \sum_a \pi(a|s) \left( \underbrace{\sum_r p(r|s, a)r + \gamma \sum_{s'} p(s'|s, a)v_k(s')}_{q_k(s, a)} \right), \quad s \in \mathcal{S}. $$

而上式的解其实是：

$$ \pi_{k+1}(a|s) = \begin{cases} 
1, & a = a_k^*(s), \\
0, & a \neq a_k^*(s),
\end{cases} $$

其中$a_k^*(s)=arg max_a q_k(s,a)$ ,如果a有多解的话，可以任选一个，不会影响收敛过程。可见新的policy $\pi_{k+1}$选择的是使得$q_k(s,a)$最大的action，这样的policy叫做greedy

第二步：

$$v_{k+1}(s) = \sum_a \pi_{k+1}(a|s) \left( \underbrace{\sum_r p(r|s, a)r + \gamma \sum_{s'} p(s'|s, a)v_k(s')}_{q_k(s, a)} \right), \quad s \in \mathcal{S}. $$
即：

$$v_{k+1}(s)=\max_a q_k(s,a)$$

总结来说，上述步骤可以表示为：

$$v_k(s) \rightarrow q_k(s, a) \rightarrow \text{new greedy policy } \pi_{k+1}(s) \rightarrow \text{new value } v_{k+1}(s) = \max_a q_k(s, a)$$

这里需要注意的问题是，虽然$v_k$最终会收敛到optimal state value，但过程的中间量$v_k$不一定是一个state value,相应的$q_k$也不一定是一个action value


### Policy iteration
每个迭代分两步走，第一步叫做policy evaluation，即根据当前的policy计算state value

$$v_{\pi_k} = r_{\pi_k} + \gamma P_{\pi_k} v_{\pi_k}, $$

第二步叫做policy improvement，计算到$v_k$之后，就可以得到$\pi_{k+1}$

$$\pi_{k+1} = \arg\max_\pi (r_\pi + \gamma P_\pi v_{\pi_k}). $$

注意这里和value iteration有本质的区别，我们的更新主体是policy。但自然会有以下疑问：（1）$v_{\pi_k}$如何计算 （2）为什么$\pi_{k+1}$会优于$\pi_k$ （3）这样的迭代策略为什么会收敛到optimal policy?

#### （1） solve $v_{\pi_k}$
我们在第二章中介绍过Bellman equation的两种解法。第一种是闭式解——$v_{\pi_k}=(I-\gamma P_{\pi_k})^{-1}r_{\pi_k}$,但计算矩阵逆会消耗大量计算量。
第二种是迭代解，

$$ v_{\pi_k}^{(j+1)} = r_{\pi_k} + \gamma P_{\pi_k} v_{\pi_k}^{(j)}, \quad j = 0, 1, 2, \ldots $$

也就是在整个迭代过程中还嵌套了一个迭代解对象，并且理论上要趋于无穷时才会收敛，但实际上会设置一个标准，比如$||v_{\pi_k}^{(j+1)} -v_{\pi_k}^{(j)} ||$小于某个阈值或者迭代次数j超过某个限定数后。

#### (2)
引理：(Policy improvement) 如果$\pi_{k+1} = argmax_\pi (r_\pi +\gamma P_\pi v_{\pi_k})$, 则$v_{\pi_{k+1}}\geq v_{\pi_k}$。
Proof:
state values满足Bellman equations

$$v_{\pi_{k+1}} = r_{\pi_{k+1}} + \gamma P_{\pi_{k+1}} v_{\pi_{k+1}},  v_{\pi_k} = r_{\pi_k} + \gamma P_{\pi_k} v_{\pi_k}. $$
根据$\pi_{k+1}$的定义：

$$
 r_{\pi_{k+1}} + \gamma P_{\pi_{k+1}} v_{\pi_k} \geq r_{\pi_k} + \gamma P_{\pi_k} v_{\pi_k}. $$

 Then,

 $$\begin{align}
  v_{\pi_k} - v_{\pi_{k+1}} &= (r_{\pi_k} + \gamma P_{\pi_k} v_{\pi_k}) - (r_{\pi_{k+1}} + \gamma P_{\pi_{k+1}} v_{\pi_{k+1}}) \\
& \leq (r_{\pi_{k+1}} + \gamma P_{\pi_{k+1}} v_{\pi_k}) - (r_{\pi_{k+1}} + \gamma P_{\pi_{k+1}} v_{\pi_{k+1}}) \\
&\leq \gamma P_{\pi_{k+1}} (v_{\pi_k} - v_{\pi_{k+1}}). 
 \end{align}$$
 因此，

 $$\begin{align}
  v_{\pi_k} - v_{\pi_{k+1}} &\leq \gamma^2 P_{\pi_{k+1}}^2 (v_{\pi_k} - v_{\pi_{k+1}}) \leq \ldots \leq \gamma^n P_{\pi_{k+1}}^n (v_{\pi_k} - v_{\pi_{k+1}}) \\  
  &\leq \lim_{n \to \infty} \gamma^n P_{\pi_{k+1}}^n (v_{\pi_k} - v_{\pi_{k+1}}) = 0. 
  \end{align}$$
  
#### (3)
我们有:

$$
 v_{\pi_0} \leq v_{\pi_1} \leq v_{\pi_2} \leq \ldots \leq v_{\pi_k} \leq \ldots \leq v^*. 
$$

 根据单调有界收敛，$\{v_{\pi_k}\}$序列收敛到一个常数$v_\infty$，下面一个定理说明了$v_\infty=v^*$
Theorem (Convergence of policy iteration) policy iteration产生的state value sequence $\{v_{\pi_k}\}_{k=0}^\infty \to v^*$,且$\{\pi_{k}\}$收敛到一个optimal policy

这个定理也会得到一个有意思的结论：即从同一个initial guess出发，policy iteration会比value iteration收敛得更快。

事实上，我们可以通过value iteration的收敛性来说明policy iteration的收敛性，我们再引入一个序列$\{v_{\pi_k}\}$由下式生成

$$ v_{k+1} = f(v_k) = \max_\pi (r_\pi + \gamma P_\pi v_k). $$
即value iteration

By induction, for k=0, we can always find a $v_0$ such that $v_{\pi_0} \geq v_0$ for any $\pi_0$

for $k \geq 0$, suppose that $v_{\pi_k}\geq v_k$

for k+1, we have

$$\begin{align}
v_{\pi_{k+1}} - v_{k+1} &= (r_{\pi_{k+1}} + \gamma P_{\pi_{k+1}} v_{\pi_{k+1}}) - \max\limits_{\pi} (r_\pi + \gamma P_\pi v_k) \\
& \geq (r_{\pi_{k+1}} + \gamma P_{\pi_{k+1}} v_{\pi_k}) - \max\limits_{\pi} (r_\pi + \gamma P_\pi v_k) \\
& = (r_{\pi_{k+1}} + \gamma P_{\pi_{k+1}} v_{\pi_k}) - (r_{\pi_k'} + \gamma P_{\pi_k'} v_k) \quad (\text{suppose } \pi_k' = \arg\max\limits_{\pi} (r_\pi + \gamma P_\pi v_k)) \\
& \geq (r_{\pi_k'} + \gamma P_{\pi_k'} v_{\pi_k}) - (r_{\pi_k'} + \gamma P_{\pi_k'} v_k) \quad (\text{because } \pi_{k+1} = \arg\max\limits_{\pi} (r_\pi + \gamma P_\pi v_{\pi_k})) \\
&= \gamma P_{\pi_k'} (v_{\pi_k} - v_k). 
\end{align}$$

由于$v_{\pi_k}-v_k \geq 0$, $P_{\pi_k'}$ 非负，则$v_{\pi_{k+1}} - v_{k+1} \geq 0$

所以由数学归纳法，$v_k \leq v_{\pi_k} \leq v^*$for any $k\geq 0$
因此policy iteration必收敛，且收敛得更快


#### Elementwise implementation
第一步：

$$ v_{\pi_k}^{(j+1)}(s) = \sum_a \pi_k(a|s) \left( \sum_r p(r|s, a)r + \gamma \sum_{s'} p(s'|s, a)v_{\pi_k}^{(j)}(s') \right), \quad s \in \mathcal{S}, $$
迭代计算$v_{\pi_k}$

第二步

$$ \pi_{k+1}(s) = \arg\max\limits_{\pi} \sum_a \pi(a|s) \left( \underbrace{\sum_r p(r|s, a)r + \gamma \sum_{s'} p(s'|s, a)v_{\pi_k}(s')}_{q_{\pi_k}(s, a)} \right), \quad s \in \mathcal{S}, $$
更新Policy


### Truncated policy iteration
前两者的比较
![[RL.asset/pi_vi.png]]
![[RL.asset/piviprocess.png]]
所谓截断truncated,其实就是在policy iteration计算$v_{\pi_k}$的过程中计算了几步的问题？如果计算无穷步，则趋近理论的policy iteration,如果只计算一步，其实计算得到的就是value iteration的$v_1$,所以当计算有限步长时，是介于两者之间的某种算法。

![[RL.asset/TPI.png]]