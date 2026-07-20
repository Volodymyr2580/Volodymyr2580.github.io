---
layout: page
permalink: /notes/exams/old-vault/old-vault-002/index.html
title: 25春期中 葛颢
---

> Imported from old Obsidian vault on 2026-07-06. Source: `25春期中 葛颢.md`

（1）

(a) 构造一个非周期的马尔科夫链，使得其存在不唯一的平稳分布，且说明这与遍历定理不冲突

(b) 对分支过程，若$P(\xi = 0) =\frac{1}{3},P(\xi = 2) =\frac{2}{3}$ ,求其灭绝概率

(2)

(a) $$P=\begin{pmatrix}0.5 & 0.3 &0.2\\0.2 & 0.5&0.3\\0&0&1\end{pmatrix}$$
求$E_0\tau_2$
(b) 设有限状态$S=\{1,2,\cdots,K\}$上的马氏链的转移矩阵$P$是双随机的，即
$$\sum_{i=1}^Kp_{ij}=1\quad \forall i,j\in S$$
求P的平稳分布

（3）设$\gamma$是$\{0,1,2\}$上的概率测度，满足$\gamma(0)>\gamma(2)>0$，随机变量$\{\xi_n\}_{n\geq1}$ i.i.d，同分布于$\gamma$。
设$Y_0=0,Y_{n+1}=max\{0,Y_n+\xi_{n+1}-1\},\forall n \geq 0$ 
求证$Y_n$是$\mathbb{N}=\{0,1,2,\cdots\}$上的不可约马氏链，且正常返。

(4) 设$\{X_n\}_{n\geq0}$ 是一个不可约非周期正常返的马氏链，状态空间S，平稳分布记为$\pi$。若存在非负函数$V:S\to [1,+\infty]$以及常数$\lambda\in (0,1)$和正数$b<+\infty$ ,使得对任意$i\in S$，有$E[V(X_1)|X_0=i]\leq \lambda V(i)+b$
证明：存在非负常数$M<+\infty$，$\rho \in (0,1)$，使得
$$E[V(X_n)|X_0=i]\leq MV(i)\rho^n+\frac{b}{1-\lambda}$$

(5) 求证对任意非负整数$L,N$，任意状态$i,j$，有如下不等式：
$$\sum_{n=L}^{n=N+L} p_{ij}(n)\leq\sum_{n=0}^{n=N} p_{jj}(n)$$

(6) 设$o$是定义在非负整数轴上的不可约马氏链的给定状态，求证：该马氏链常返，如果下列方程存在解：
$$u_i \geq \sum_{j\neq o} p_{ij}u_j \quad \forall i \neq o$$
且$u_i \to +\infty$ 当$i\to +\infty$ 

