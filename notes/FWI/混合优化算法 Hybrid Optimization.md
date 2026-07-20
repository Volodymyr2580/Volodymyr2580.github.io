---
layout: page
permalink: /notes/FWI/混合优化算法 Hybrid Optimization/index.html
title: Hybrid Optimization for FWI
---

主迭代框架：局部优化+全局优化混合+Tikhonov正则化
![Hybrid optimization main iteration framework](hybrid.asset/hybrid1.png)

## 算法伪代码
![Hybrid optimization pseudocode](hybrid.asset/hybrid3.png)

Cooling schedule
![Hybrid optimization cooling schedule](hybrid.asset/hybrid4.png)
$T_0=1$为起始温度，参数c决定了冷却速度，Normally, Slow(c=0.05), Moderate(c=0.8), fast(c=10)

Adam 优化器的超参数一般设置为：$\beta_1=0.9, \beta_2=0.999$
