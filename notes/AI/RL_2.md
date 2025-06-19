---
layout: page
permalink: /notes/AI/RL_2/index.html
title: 强化学习的数学原理 2
-




上一节介绍了关于我们刻画一个agent与环境交互行为的基本概念。
这一章笔记的记法会有所不同，我决定先按照书本学习一遍，然后再做笔记梳理，这样思路会按照我的习惯来，并不会follow原书的chain。
# State Values and Bellman Equation

这一节的核心概念是state value——the average reward that an agent can obtain if it follows a given policy. state value越大，相应的policy越优。分析state value的工具是贝尔曼方程(Bellman equation)，这一过程被称为policy evaluation。

回忆上一讲中介绍的一个基本概念return,是我的某一条确定性轨道上每一步的奖励乘上贴现因子的代数和：
$$\text{discounted return} = \sum_{i=0}^{\infty}\gamma^ir_i$$
但这个定义不适用于我们上一节最后提到的MDPs模型，并且为了复杂化和自然化我们的policy，我们往往是需要做混合的（这里混合mix其实用了博弈论的表述习惯，意思是每一步的action和reward都会取成一个随机变量(Random Variable, R.V.)）。因此，我们需要引入state value和action value两个概念。

## State value & Action value
需要先引入必要的记号：考虑时间步$t=0,1,2,\cdots$ ，在时间步t，agent位于状态$S_t$，根据选定的某个policy $\pi$ ,下一个action是$A_t$，状态为$S_{t+1}$，得到的immediate reward是$R_{t+1}$

$$S_t\xrightarrow[]{A_t}S_{t+1},R_{t+1}$$
上式中涉及的四个都是R.V.。$S_t,S_{t+1}\in \mathcal{S},A_t\in\mathcal{A}(S_t),R_{t+1}\in\mathcal{R}(S_t,A_t)$
从t开始，我们能得到一个state-action-reward轨道：
$$S_t\xrightarrow[]{A_t}S_{t+1},R_{t+1}\xrightarrow[]{A_{t+1}}S_{t+2},R_{t+2}\xrightarrow \cdots$$
根据定义，沿着轨道的贴现回报(discounted return)：
$$G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots$$
注意$G_t$也是一个R.V.，因此我们可以把state value定义为它的期望：
$$v_\pi(s) =\mathbb{E}[G_t|S_t=s]$$
注意$v_\pi(s)$是一个依赖于policy $\pi$和状态s的函数，但与时间步t无关，并且要注意这里取的期望是对action,state和reward同时取的期望。

进一步，我们可以定义在状态s执行action a之后的期望回报，也就是action value:

$$q_\pi(s,a)=\mathbb{E}[G_t|S_t=s,A_t=a]$$
这个定义是自然的。
根据条件期望的性质，自然有以下式子成立：
$$v_\pi(s)=\sum_{a\in \mathcal{A}}\pi(a|s)q_\pi(s,a)$$
## Bellman equation
贝尔曼方程的推导是极其自然的，类比随机过程中处理马氏过程我们常用的首步分析法，我们可以将两个状态之间的state value或者action value通过递推的方法关联起来，从而解出我们的state value or action value.

接下来我们推导贝尔曼方程，显然有(by definition)
$$G_t=R_{t+1}+\gamma G_{t+1}$$
Thus, 
$$v_\pi(s)=\mathbb{E}[G_t|S_t=s]=\mathbb{E}[R_{t+1}|S_t=s]+\gamma\mathbb{E}[G_{t+1}|S_t=s]$$
利用全概率公式：
$$\mathbb{E}[R_{t+1}|S_t=s]=\sum_{a\in\mathcal{A}}\pi(a|s)\mathbb{E}[R_{t+1}|S_t=s,A_t=a]=\sum_{a\in\mathcal{A}}\pi(a|s)\sum_{r\in\mathcal{R}}p(r|s,a)r$$$$\begin{align}
\mathbb{E}[G_{t+1}|S_t=s]&=\sum_{s'\in\mathcal{S}}\mathbb{E}[G_{t+1}|S_t=s,S_{t+1}=s']p(s'|s)\\
&=\sum_{s'\in\mathcal{S}}\mathbb{E}[G_{t+1}|S_{t+1}=s']p(s'|s) \quad \text{due to Markov property} \\
&=\sum_{s'\in\mathcal{S}}v_\pi(s')p(s'|s)\\
&=\sum_{s'\in\mathcal{S}}v_\pi(s')\sum_{a\in\mathcal{A}}p(s'|s,a)\pi(a|s)
\end{align}$$
代入原式就得到了贝尔曼方程（上述的代换是自然的，把未知的条件期望转化为显式的由待求量$v_\pi$和已知的转移概率组成的方程）
$$v_\pi(s)=\sum_{a\in\mathcal{A}}\pi(a|s)[\sum_{r\in\mathcal{R}}p(r|s,a)r+\gamma\sum_{s'\in\mathcal{S}}p(s'|s,a)v_\pi(s')], \text{for all }s\in \mathcal{S}$$

上式还可以改写为：
$$v_\pi(s)=\sum_{a\in\mathcal{A}}\pi(a|s)\sum_{r\in\mathcal{R}}\sum_{s'\in\mathcal{S}}p(s',r|s,a)[r+\gamma v_\pi(s')],$$
Or
$$\begin{align}
v_\pi(s)&=\sum_{a\in\mathcal{A}}\pi(a|s)\sum_{r\in\mathcal{R}}p(r|s,a)r+\gamma\sum_{a\in\mathcal{A}}\pi(a|s)\sum_{s'\in\mathcal{S}}p(s'|s,a)v_\pi(s'), \\
&=r_\pi(s)+\gamma\sum_{s'\in\mathcal{S}}p_\pi(s'|s)v_\pi(s')
\end{align}$$
其中：$p_\pi(s'|s)=\sum_{a\in\mathcal{A}}\pi(a|s)p(s'|s,a)$，这一形式中，$r_\pi(s)$表示在s处的所有即时奖励(immediate rewards)的平均，$p_\pi(s'|s)$则是在policy $\pi$下从状态$s\to s'$的转移概率。

我们可以根据这种表达式写出贝尔曼方程的矩阵形式。
把状态依次标为：$s_i, i=1,2,\cdots,n$，$n=|\mathcal{S}|$ , let $v_\pi=[v_\pi(s_1),v_\pi(s_2),\cdots,v_\pi(s_n)]^T \in \mathbb{R}^n$,$r_\pi=[r_\pi(s_1),r_\pi(s_2),\cdots,r_\pi(s_n)]^T \in \mathbb{R}^n$ and $P_\pi \in \mathbb{R}^{n\times n}$
其中$[P_\pi]_{ij}=p_\pi(s_j|s_i)$ 

则矩阵形式的贝尔曼方程为：
$$v_\pi=r_\pi+\gamma P_\pi v_\pi$$
其中$P_\pi$就是转移概率矩阵(conditioned on policy $\pi$)

如此贝尔曼方程的解就可以表达为：
$$v_\pi =(I-\gamma P_\pi)^{-1}r_\pi$$
这个前提是矩阵$(I-\gamma P_\pi)$可逆，由于$\gamma \in(0,1)$并且$P_\pi$是转移概率矩阵。根据Gershgorin圆盘定理，矩阵的每一个复特征值都位于至少一个Gershgorin圆盘内，而第i个圆盘的中心位于$1-\gamma p_\pi(s_i|s_i)$，半径大小等于第i行非对角线元素的绝对值之和$R_i=\sum_{j\neq i}\gamma p_\pi(s_j|s_i)<1-\gamma p_\pi(s_i|s_i)$ 

其中小于号是利用了矩阵$P_\pi$第i行的元素和为1

所以没有0特征值，因此可逆。

此外，
$$(I-\gamma P_\pi)^{-1}=I+\gamma P_\pi+(\gamma P_\pi)^2+\cdots\geq I\geq 0$$
这里的大于等于号是逐元素比较的。

但上式涉及到矩阵求逆，当我们状态数n增大时，这会导致巨大的计算消耗。相应地有迭代解：
$$v_{k+1}=r_\pi+\gamma P_\pi v_k, k=0,1,2,\cdots$$


对应地也有action value的贝尔曼方程：
$$q_\pi(s,a)=\sum_{r \in \mathcal{R}}p(r|s,a)r+\gamma\sum_{s'\in \mathcal{S}}p(s'|s,a)\sum_{a' \in \mathcal{A}(s')}\pi(a'|s')q_\pi(s',a')$$
其矩阵形式是：
$$q_\pi=\tilde{r}+\gamma P \Pi q_\pi$$
其中：$[q_\pi]_{(s,a)}=q_\pi(s,a)$, $[\tilde{r}]_{(s,a)}=\sum_{r\in\mathcal{R}} p(r|s,a)r$,$[P]_{(s,a),s'}=p(s'|s,a)$,$\Pi$是一个分块对角阵，每一块是一个$1\times |\mathcal{A}|$的向量：$\Pi_{s',(s',a')}=\pi(a'|s')$

最后，从直观上来看，state value越大说明一个policy越优，但只是直观。

