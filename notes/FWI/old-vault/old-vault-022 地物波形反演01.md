---
layout: page
permalink: /notes/fwi/old-vault/old-vault-022/index.html
title: 地物波形反演01
---

> Imported from old Obsidian vault on 2026-07-06. Source: `地物波形反演01.md`
#地球物理 #反演
Taylor's Theorem:
$$f(x+\Delta x)=f(x)+\nabla f(x)^T\Delta x +\frac{1}{2}\Delta x^T\nabla^2f(x)\Delta x
+\cdots$$
where $\nabla f(x)$ is a column vector: 
$$\nabla f(x)=(\frac{\partial f}{\partial x_i}) $$ Compare two notations: Numerator layout VS Denomiorator layout
for Numerator layout,$d=(d_1,\cdots,d_n)^T$
$$\frac{\partial d}{\partial x}=(\frac{\partial d_1}{\partial x},\cdots,\frac{\partial d_n}{\partial x})^T$$
$$\frac{\partial \vec{y}}{\partial \vec{x}}=(\frac{\partial y_i}{\partial x_j})$$
Numerator layout的意思就是求完导后的维度与分子相同，也就是纵向同y，横向同x

### Linear Inverse Problem
data: $d_obs=[\quad]^T$
model parameter: $m=[\quad]^T$
quantitative model theory Forward:$f(m)=d$
$Do:\quad m^{est}\to f(m^{est}) \to d_{pre} \leftrightarrow d_{obs} \to Inverse \to m^{new\_est}$
由于存在噪声和各向异性介质，可以找到一个真实model的approximation，但永远找不到真实解。

#### Types of Theory
**Implicit Theory**:$f(d,m)=c$ 数据和模型耦合无法拆开。
Example: 给一个物体称重，我们能够观测其Length、Height、Mass、width
Want to know $\rho$ 
$$d_m=m_{\rho}d_Ld_Hd_W$$

**Explicit Theory**
$$\hat{d}=f(\hat{m})$$
Example：给一个长方形，观测其周长与面积，模型参数为长和宽，其是decoupled

**Linear explicit theory**
$$\bar{d}=\bar{G}\bar{m}$$
Example: for some instance made of gold and quartz given an observable volume V and mass M, we have $V_g+V_q=V$,$\rho_gV_g+\rho_qV_q=M$

**Linear Implicit**
$$\bar{F}\begin{pmatrix}
\bar{d} \\
\bar{m}
\end{pmatrix}=0$$

##### Simple Example: Fitting a straight line
For a given 1D-temperature-data distribution relevant to time t $\{T_j\}$
最小二乘法

##### Acoustic tomography
一个二维介质均匀分为16块，宽h，每块内的速度是各向同性的$\{v_j\}$ 16个速度参数未知
做观测，横向和纵向穿过一条的走时$\{T_k\}$
We have :
$$T=HS$$
$T=\frac{d}{v}\sigma$ d是distance,v是velocity,$\sigma$是slowness
##### HW：能不能通过一个setup确定16个参数。

#### X-ray Imaging
Theory: $\frac{dI}{ds}=-C(x,y)I$
$I$是intensity of x-ray, 即data；$s:$distance
$C(x,y):$ absorption coefficient

积分得到$I_i=I_0exp(-\int_{beam-i}C(x,y)ds)$ 取log后得：$lnI_0-lnI_i=\int C(x,y) ds$
做Taylor 展开：
$$lnI_i=lnI_0+\frac{1}{I_0}(I_i-I_0)+O((I_i-I_0)^2)$$
再对积分作网格离散。......

#### Least Square solution
Theoretically,$Gm=d$, G known ; we have $G m_{est}=d_{est}$
define error vector $e=d_{obs}-d_{est}$
How to measure?

$L_1 \quad Norm$: $|e|=\sum_i|e_i|$
$L_2 \quad Norm$: $|e|_2=(\sum_i|e_i|^2)^{1/2}$ Euclidian Length
$L_p \quad Norm$: $|e|_p=(\sum_i|e_i|^p)^{1/p}$

Difference between different norms:
p越大，范数对最大模的那一项越敏感。
$L_{\infty} \quad Norm$ : $|e|_{\infty}=max|e_i|$

作反演时常见的几种error分布：Gaussian 集中于某一区域； or Tail 在更大区域上有分布
对存在大的error，一般取比较小的norm；如果error比较小一般就使用$L_2$范数







