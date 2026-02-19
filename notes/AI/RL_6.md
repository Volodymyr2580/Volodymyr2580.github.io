---
layout: page
permalink: /notes/AI/RL_6/index.html
title: 强化学习的数学原理 6
---
上一章我们学习了蒙特卡洛法，本章我们会为下一章的内容——Temporal Difference——作铺垫，学习Stochastic Approximation的内容。

### Robbins-Monro 算法
假设我们想找到方程的根：
$$g(w)=0,$$
许多优化问题都能化归到一个寻根的问题，譬如$J(w)$是一个目标函数，那原优化问题等价于寻找$\nabla_w J(w)=0$的根。

通常，函数g是未知的，我们通常只能得到其一个含噪声的观测：
$$\tilde{g}(w,\eta)=g(w)+\eta$$
RM算法的解可以迭代地表示为：
$$w_{k+1}=w_k-a_k \tilde{g}(w_k,\eta_k), \quad k=1,2,3,\cdots$$
其中$w_k$是第k次估计。

![[Pasted image 20250717202455.png]]
![[Pasted image 20250717202536.png]]
![[Pasted image 20250717202543.png]]

### SGD
SGD其实是一种特殊的RM算法？
考虑以下优化问题：
$$\min_w J(w)=\mathbb{E}[f(w,X)]$$
其中X是一个随机变量。根据梯度下降算法：
$$w_{k+1}=w_k-\alpha_k \nabla_w J(w_k)=w_k-\alpha_k \mathbb{E}[\nabla_w f(w_k,X)]$$
