---
layout: page
permalink: /notes/AI/RL_5/index.html
title: 强化学习的数学原理 5
---
上一章我们讲了基于模型寻找最优策略的方法，value iteration和policy iteration，我们证明了policy iteration能够比value iteration更快得收敛，其中可控的参数是在每一步迭代中计算$v_\pi$时需要启动一个新的迭代，而这一步往往采用截断的手段，我们说明了如果只计算一步，truncated policy iteration退化为value iteration，而如果计算无限步，则趋于理论的policy iteration。

本章我们要介绍的概念是一种不依赖于模型的强化学习算法——蒙特卡洛法。
当我们没有模型的时候，我们就需要大量的agent与环境交互的数据。如果数据和模型都没有，那就别做了。

### MC Basic
将policy iteration转变为model-free版本：
回忆policy iteration每一个迭代有两步，第一步是policy evaluation, 我们通过计算state value $v_{\pi_k}$得到，第二步是policy improvement，计算$v_{\pi_k}$对应的greedy policy：
$$ 
\begin{align}
\pi_{k+1}(s) &= \arg\max_{\pi} \sum_a \pi(a|s) \left[ \sum_r p(r|s, a)r + \gamma \sum_{s'} p(s'|s, a)v_{\pi_k}(s') \right]  \\
 &= \arg\max\limits_{\pi} \sum_a \pi(a|s) q_{\pi_k}(s, a), \quad s \in \mathcal{S}
 \end{align}
 $$
 注意action values在两步中起到关键的作用，回忆action values的两种计算方法：
 （1） 基于模型, 先从Bellman equation解出$v_{\pi_k}$，然后根据下式计算action values
 $$ q_{\pi_k}(s, a) = \sum_r p(r|s, a)r + \gamma \sum_{s'} p(s'|s, a)v_{\pi_k}(s'). $$
 （2） model-free approach, 回忆action value的定义是给定状态s，选择action a后得到的期望回报：
 $$\begin{align} q_{\pi_k}(s, a) &= \mathbb{E}[G_t|S_t = s, A_t = a] \\
&= \mathbb{E}[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots |S_t = s, A_t = a], \end{align}$$
既然是期望值，就可以通过MC的方法求得——从(s,a)出发，agent根据policy$\pi_k$与环境交互n个episodes（每个episode要和环境交互若干次直到轨道结束），action value就可以估计为：
$$ q_{\pi_k}(s, a) = \mathbb{E}[G_t|S_t = s, A_t = a] \approx \frac{1}{n} \sum_{i=1}^n g_{\pi_k}^{(i)}(s, a). $$
其中, $g_{\pi_k}^{(i)}(s, a)$是第i个episode交互得到的return，根据强大数律，当交互次数足够多时，上式几乎处处收敛。

因此，MC Basic Algorithm从初始policy $\pi_0$出发，每一步iteration有两小步
（1）Policy evaluation
（2）Policy improvement
![[MCBasic.png]]

### MC Exploring Starts
如何更有效地利用我们的samples呢？
假设根据某一个policy $\pi$我们得到了an episode of samples:
$$s_1 \xrightarrow{a_2} s_2 \xrightarrow{a_4} s_1  \xrightarrow{a_2} s_2 \xrightarrow {a_3}s_5  \xrightarrow{a_1} \cdots $$
每一次一对(state, action)出现，就称一次访问(visit)。MC Basic算法采用的是initial-visit strategy, 即这个episode只用于估计起始访问对的action value。但实际上也可以用于计算其他的点对，如下式分解：
$$\begin{align}
 s_1 \xrightarrow{a_2} s_2 \xrightarrow {a_4} s_1 \xrightarrow{a_2} s_2 \xrightarrow{a_3} &s_5 \xrightarrow{a_1} \cdots \quad \text{[original episode]} \\
 s_2 \xrightarrow{a_4} s_1 \xrightarrow{a_2} s_2  \xrightarrow{a_3} &s_5 \xrightarrow{a_1}  \cdots \quad \text{subepisode starting from } (s_2, a_4) \\
 s_1 \xrightarrow{a_2} s_2  \xrightarrow{a_3} &s_5 \xrightarrow{a_1}  \cdots \quad \text{subepisode starting from } (s_1, a_2) \\
 s_2 \xrightarrow{a_3} &s_5  \xrightarrow{a_1}  \ldots \quad \text{subepisode starting from} (s_2, a_3) ]\\
 &s_5 \xrightarrow{a_1}  \cdots \quad \text{subepisode starting from } (s_5, a_1) 
\end{align}$$
这条轨道上的每一个pair都可以看做一个新的episode的起点。显然，同一对(s,a)在一个轨道上可能出现多次，如果我们只考虑第一次，就是first-visit strategy,如果考虑每一次，就是every-visit策略。后者显然在利用效率上更优。

接下来是如何更有效率地更新policy?

之前的想法必须等到全部的episodes跑完才能更新，因为需要计算平均值。现在的想法是只跑一个episode直接更新。
![[MCefficient.png]]

### MC $\epsilon$-Greedy
一个策略称为软(soft)的，若在任一状态，有正概率取各个行动。这种情况下，如果我们跑了一个足够长的episode,就应该能访问各个pair足够多次。

一种常见的软策略是$\epsilon$-greedy策略。Suppose $\epsilon \in [0,1]$,
$$ \pi(a|s) = \begin{cases} 
1 - \frac{\epsilon}{|\mathcal{A}(s)|}(|\mathcal{A}(s)| - 1), & \text{for the greedy action}, \\
\frac{\epsilon}{|\mathcal{A}(s)|}, & \text{for the other } |\mathcal{A}(s)| - 1 \text{ actions},
\end{cases} $$
其中$|\mathcal{A}(s)|$指代的是state s相关的行为数。

如何实现这一策略呢？我们可以取一个服从$[0,1]$均匀分布的随机数x，如果$x\geq \epsilon$，选择greedy action, 否则，以$1/|\mathcal{A}(s)|$的概率随机选择一个行为。

![[MCGreedy.png]]