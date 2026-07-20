---
layout: page
permalink: /notes/fwi/old-vault/old-vault-023/index.html
title: 地物波形反演02
---

> Imported from old Obsidian vault on 2026-07-06. Source: `地物波形反演02.md`
[[地物波形反演01]] #地球物理 #反演 
对$E=e^Te$,where $e=d_{pre}-d_{obs}$,suppose $d_{pre}=Gm$
do $\frac{\partial E}{\partial m_q}=0$ we get $G^TGm=G^Td$

如何只有一个点？
$d=\begin{pmatrix}1 & x_1\end{pmatrix}\begin{pmatrix}a \\ b\end{pmatrix}$
$$G^TG=\begin{pmatrix}1 & x_1 \\ x_1 & x_1^2\end{pmatrix}$$
存在一个0特征值，则参数$m$的一个dimension就失去了观测。$G^TG$没有Inverse，就有无穷多解。

**Exercise**
suppose $d=a+bx+cy$

### 反演解的形式 
Overdetermined Underdetermined mixed-determined

Example:
Two squares of velocity parameters $v_1,v_2$, shoot rays to measure.

对Geophysical Problem, 大多是Mixed-determined
What do we do？

利用a priori information --- velocity, density... about model
define: data error $E_d=e^Te$ ;
model parameter information:$L=|m|_p^{\frac{1}{p}}$
define $E=E_d+\epsilon^2 L$ 设为两者的线性组合，然后 minimize E instead of $E_d$

取L是$L_2 \quad Norm$  $E=e^Te+\epsilon^2 m^Tm$ （保持一个steady state，取model parameter较小，符合客观世界整体认知规律）
$Solve\quad min_m E$
整体关于m的参数仍是抛物线形式，最小点仍可以有FOC给出
$\frac{\partial E}{\partial m}=0$
$$\begin{align}E&=\sum_i^N[d_i-\sum_j^MG_{ij}m_j][d_i-\sum_k^MG_{ik}m_k]+\epsilon^2\sum_j^Mm_j^2 \\
&=\sum_j^M\sum_k^Mm_jm_k\sum_i^NG_{ij}G_{ik}-2\sum_j^Mm_j\sum_i^NG_{ij}d_i+\sum_i^N d_id_i+\epsilon^2\sum_j^Mm_j^2
\end{align}$$
$$\frac{\partial E}{\partial m}=0 \leftrightarrow [G^TG+\epsilon^2I]m=G^T d$$
↑damped Least Square 
加上model的prior information；$m_{est}-m_p$最小化
可以define不同的矩阵D $[D(m-m_p)]^T[D(m-m_p)]$最小，D可以有特殊的作用。

第二种
已知平均
$$E_m^{(2)}=[m_{est}-\langle m\rangle]^T[m_{est}-\langle m\rangle]$$ 第三种 对model的smooth约束
first-order: $\frac{\partial m}{\partial x}$ 希望均匀
Second-order: $\frac{\partial^2m}{\partial x^2}$ 希望变化率小
添加此类信息？$\frac{\partial m}{\partial x}=\lim_{\Delta x}\frac{m(x+\Delta x)-m(x)}{\Delta x}$
取Matrix D表示差分形式，作用在我们的$m_{est}-\langle m\rangle>$ 上，取其energy最小
$$[D(m_{est}-\langle m\rangle)]^T[D(m_{est}-\langle m\rangle)]$$
$\frac{\partial E}{\partial m}=0$会得到：
$$(G^TG+\varepsilon^2D^TD)m_{est}=G^Td+\varepsilon^2D^TD\langle m\rangle$$
$D^TD\equiv W_m$ 相当于对模型参数做了一个weighting
另一种，对观测数据做weighting 考虑到数据质量上有差异。
do $Q d$ ,$Q=diag\{\lambda_1,\lambda_2,\cdots,\lambda_n\}$ 
$e=d_{est}-d{obs}$, let $E=(Qe)^T(Qe)+\varepsilon^2E_m$ 
still let$\frac{\partial E}{\partial m}=0$,we get:
$$(G^TW_dG+\varepsilon^2W_m)m_{est}=G^TW_dd+\varepsilon^2W_m\langle m\rangle$$
where $W_d=Q^TQ$ 
对上式分解：
$$F^TFm_{est}=F^T\tilde{d}$$
so that$Fm_{est}=\bar{f}$
where $F=\begin{pmatrix}W_d^{1/2}G \\ \varepsilon D\end{pmatrix}$,$f=\begin{pmatrix} W_d^{1/2}\tilde{d} \\ \varepsilon D \langle m \rangle \end{pmatrix}$





