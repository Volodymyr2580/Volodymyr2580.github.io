---
layout: page
permalink: /notes/复习笔记/old-vault/old-vault-027/index.html
title: 报告书面建议
---

> Imported from old Obsidian vault on 2026-07-06. Source: `报告书面建议.md`
### 第一组：解释性？attention机制
吴境广 
5-utr 测序手段——数据来源 不同实验室不同pc ；测的是表达量是细胞类型在某瞬间的表达量（时变）
result：什么是PC效应抗性；比较好的识别效果。识别出未分类的细胞。
可能训练集里没有出现的cell type，在测试机出现——噪声perturbation

Method——Attention
表达量序列（n行1列）$C_{n\times1}$
$t_{k\times1}=WC$ 表示token的参与k个通路/功能的情况，其中W是设计好的，由一个preconditioner $M$乘 hadamard积（或者说是掩码吧），指出在第j个通路中会有哪些基因参与？
m次重复得到m个$t_i$ 形成$T_{k\times m}$再加上一个CLS层使得$I=\begin{pmatrix}CLS \\ T\end{pmatrix}$ 
再对I做Attention。

随机掩码（不影响结果）？（先复现，再看一下随机掩码）
困难问题——骨细胞分类 

建议：新细胞类型？是否可以用noise perturbation来做data augmentation
掩码的矩阵听起来很像优化问题中的一个preconditioner；稀疏性？有效性？
为什么会有out of distribution的效果？是哪部分起的效果？做消融实验。

对数据集的质量做有效评估？

（1） 针对掩码矩阵M：您可以在报告中解释清楚其形成T的m次过程，以及如果把掩码改成随机掩码时具体的实现方式。另一方面

### 第三组：胡昌泰 2100012251@stu.pku.edu.cn
基因组补全——统计技术，对未直接进行基因分型的个体的基因组分配最可能的基因型来推断缺失数据
HLA基因座 
未对相型进行编码？
SNP数据
序列到标签的任务，Transformer

不好给建议啊
创新点？

第四组：刘任达 Neural SDE liurenda@stu.pku.edu.cn
建模金融市场宏观经济
新闻数据？EPU Index
低频or高频
