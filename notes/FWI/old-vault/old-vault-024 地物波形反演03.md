---
layout: page
permalink: /notes/fwi/old-vault/old-vault-024/index.html
title: 地物波形反演03
---

> Imported from old Obsidian vault on 2026-07-06. Source: `地物波形反演03.md`
[[地物波形反演02]][[地物波形反演03]]

$m_{est}=Md+v=G^{-g}d$
$G^{-g}$ is called generalized Inverse
$d_{est}=Gm$
so we have
$$GG^{-g}d_{obs}=d_{est}$$
另一方面
$d_{obs}=Gm_{true}$
we have:
$$m_{est}=G^{-g}Gm_{true}$$
HW:
1. 一条直线上若干观测点 $(x_i,y_i), y_i$加了随机噪声，反演直线.构建此Inverse问题的两个resolution matrix
2. $$d_i=\sum_{j=1}^Mexp(-c_iz_j)m_j$$
对不同的c,构造两个resolution matrix, 画出Model resolution matrix，分析其性质。

Discrete Laplace Transform
$$d(c)=\int_0^{\infty}exp(-cz)m(z)dz$$ $$d_i=\sum_{j=1}^Mexp(-cz_j)m_j$$

## Probability Theory

Random Variable: 我们观测的数据$d_i$总是带有误差，视为某种随机量

对观测值做统计学上的描述histogram（直方图） 去统计某个d出现的次数
当观测次数足够多，能够趋向这个观测数据的分布

#### Typical value and scattering range
1. peak, maximum likelihood
2. median 中位数
3. mean
4. scattering range: 算variance 方差 $q(d)=(d-\langle d \rangle)^2$
用 $q(d)p(d)$ 去 measure

$$cov(d_1,d_2)=\int\int(d_1-\langle d_1 \rangle)(d_2-\langle d_2\rangle)p(d_1,d_2)dd_1dd_2$$
$$matrix form: \begin{pmatrix}\sigma_1^2 & \sigma_{12}^2 \\ \sigma_{21}^2 & \sigma_2^2\end{pmatrix}$$
Multivariate Gaussian
$$N(\bar{d},Cov)=\frac{1}{(2\pi)^{N/2}||Cov||^{1/2}}exp(-\frac{1}{2}(d-\bar{d})^TCov^{-1}(d-\bar{d})))$$
### Maximum Likelihood for Linear Gaussian Inverse problem

求解极大似然函数 极值条件 最终得到的Gaussian参数实际就是观测数据的均值和variance

Linear Problem 代入gaussian
$Gm=d$
$$P(d) \propto exp[-\frac{1}{2}(d_{obs}-Gm)^TCov(d_{obs})(d_{obs}-Gm)]$$
$$L=log(P(d))=-\frac{1}{2}(d_{obs}-Gm)^TCov(d_{obs})(d_{obs}-Gm)$$
如何在概率的角度加入先验信息？认为有一个先验分布，取其和data的分布的joint

较多的分布可以用Gaussian来表示

$$P(m)\propto exp(-\frac{1}{2}(m-\langle m\rangle)^TCov(m)(m-\langle m\rangle))$$
$$P(d,m)=P(d)P(m)$$
therefore
$$L=d_{norm}+m_{norm}$$


对应先前讲的Linear case
$$Fm=f$$
$$F=\begin{pmatrix}cov(d)^{-1/2}G \\ cov(m)^{-1/2}I\end{pmatrix}, f=\begin{pmatrix}cov(d)^{-1/2}d_{obs} \\ cov(m)^{-1/2}\langle m \rangle\end{pmatrix}$$
对大型矩阵，especially dense matrix,转置求逆等操作相当耗时

## Iterative solution
objective function/ misfit function/ energy function/ cost function
$$E=d^Td+m^Tm$$
workflow:
$m_0$ initial model/guess
while $E(m)\geq c$ do $k=1,2,\cdots,N$
	a. compute a search direction $\Delta m_k$ 
	b. compute a step length $\alpha_k$
	c. $m_{k+1}=m_{k}+\alpha_k\Delta m_k$
	d. k to k+1

#### gradient descent
steepest descent: $m_{k+1}=m_{k}+\alpha_k\Delta m_k$;$\Delta m_k=\nabla E(m)$
在高维问题中，发现很多梯度走的方向是重复的。


更有效率的方法CG conjugate gradient 共轭梯度法

### CG
对一个线性方程
$$Ax=b$$
等价于求解
$$min\quad f(x)=\frac{1}{2}x^TAx-b^Tx+c$$
$$f'(x)=\frac{1}{2}A^Tx+\frac{1}{2}Ax-b$$
Suppose A is symmetric:
$$f'(x)=Ax-b$$
let $f'(x)=0$ station point: $Ax_0=b$

define: 
1. error: $e_{(i)}=x_{(i)}-x_0$
2. residual: $r_{(i)}=b-Ax_{(i)}$
in fact, $r_{(i)}=-f'(x_{(i)}), r_{(i)}=-Ae_{(i)}$

更新：
$$x_{(1)}=x_{(0)}+\alpha r_{(0)}$$
如何定步长？
$$\frac{df(x_{(1)})}{d\alpha}=f'(x_{(1)})^T\frac{dx_{(1)}}{d\alpha}=f'(x_{(1)})^Tr_{(0)}$$
let $f'(x_{(1)})^Tr_{(0)}=0$ 所以$x_{(1)}$处的导数会和$x_{(0)}$处的导数方向是正交的

equal to $$(b-A(x_{(0)}+\alpha r_{(0)}))^Tr_{(0)}=0$$
$$(b-Ax_{(0)})^Tr_{(0)}=\alpha r_{(0)}^TAr_{(0)}$$
$$\alpha = \frac{r_{(0)}^Tr_{(0)}}{r_{(0)}^TAr_{(0)}}$$
以上这样迭代的过程中每一次迭代的方向都有重复

CG的思路是构造一组搜索方向，使得新的搜索方向没有重复

想要： $$d_{(i)}^Te_{(i+1)}=d_{(i)}^T(e_{(i)}+\alpha_{(i)}d_{(i)})=0$$
构建一组方向使得步长能够计算出来

A-orthogonal: 在矩阵A定义的实内积下正交

现在找$$d_{(i)}^TAe_{(i+1)}=0$$
ultimately we will have:
$$\alpha = \frac{d_{(i)}^TAe_{(i)}}{d_{(i)}^TAd_{(i)}}$$
CG:
$$\beta_i=\frac{r_i^Tr_i}{r_{i-1}^Tr_{i-1}}$$
$$d_{i+1}=r_{i+1}+\beta_{i+1}d_{i}$$

