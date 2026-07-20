---
layout: page
permalink: /notes/ai/old-vault/old-vault-033/index.html
title: 深度学习基本原理（一）
---

> Imported from old Obsidian vault on 2026-07-06. Source: `深度学习基本原理（一）.md`
万有逼近定理。NN is complex enough to approximate any continuous function
但无法回答神经网络拟合和多项式拟合的区别？

泛化谜团：过参数化的NN不会明显过拟合；经验上倾向于 符合直观；通常泛化性好

NN的隐式偏好是什么？

##### 频率原则
Q：真实数据过于复杂？What to do？
“从轮廓到细节” 
三角级数展开：$S_N(x)=\sum_{n=-N}^{N} C_ne^{i2\pi \frac{n}{P}x}$
频率原则：NN按照从低频至高频的顺序拟合 （普通的全连接）
对高频成分的跳跃和间断拟合，就需要高频

如何设计实验去思考 影响拟合顺序的是频率还是振幅？

一维：两层全连接
$h(x)=\sum a_j\sigma(w_jx+b_i)$然后做傅里叶变换$\bar{h}(k)\approx \sum a_j exp(\frac{ib_j}{w_j}exp(-|\frac{\pi k}{2w_i}|))$
定义频率k处的损失：$L(k)=\frac{1}{2}|\bar{h}(k)-\bar{f}(k)|^2$
根据Parseval等式：空间域的损失函数与频率域相同
通过傅里叶域中的损失计算梯度，可以看成是每一个单独的频率成分对更新梯度的贡献

低频梯度几乎必然大于高频梯度

#### 初始化如何影响NN训练
常见初始化：$scale \sim m^{-\lambda}$

