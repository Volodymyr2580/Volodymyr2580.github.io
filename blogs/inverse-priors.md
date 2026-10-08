---
layout: lab
bilingual: true
permalink: /blogs/inverse-priors/index.html
title: 从反演问题到生成模型：为什么我开始重新理解“先验”
comments: false
lab_nav: Blogs
lab_subtitle: blue-sky lab notebook
---

<nav class="lab-simple-switch" aria-label="Back" data-i18n-aria-label="Back">
  <a href="{{ '/blogs/' | relative_url }}" data-i18n="← Blogs">← Blogs</a>
</nav>

<section class="lab-hero" aria-labelledby="article-title">
  <p class="lab-eyebrow" data-i18n="Blog Entry / 公众号母稿">Blog Entry</p>
  <h1 id="article-title" data-i18n="从反演问题到生成模型：为什么我开始重新理解“先验”">From Inverse Problems to Generative Models: Rethinking Priors</h1>
  <p class="lab-lead" data-i18n="这类详情页的重点不是炫技，而是让长文、公式、补充材料和公众号同步状态都能安静地放在同一个页面里。">This article page brings long-form writing, equations, supplementary material, and the status of the WeChat version together in one place.</p>
</section>

<div class="lab-detail-layout">
  <article class="lab-article-paper">
    <header class="lab-article-head">
      <p class="lab-eyebrow" data-i18n="Featured Observation">Featured Observation</p>
      <h1 data-i18n="从反演问题到生成模型：为什么我开始重新理解“先验”">From Inverse Problems to Generative Models: Rethinking Priors</h1>
      <div class="lab-article-meta">
        <span data-i18n="Research Essay">Research Essay</span>
        <span class="lab-stamp" data-i18n="Wechat draft">Wechat draft</span>
        <span>FWI</span>
        <span data-i18n="12 min read">12 min read</span>
      </div>
    </header>

    <div class="lab-article-body">
      <p data-i18n="我一开始把“先验”理解成一种外加约束：数据不够时，给模型加一点额外偏好。但最近读 FWI 和生成模型相关论文时，我越来越觉得这个说法太粗糙。">I initially understood a prior as an added constraint: when data are insufficient, introduce an extra preference for the model. But while reading papers on FWI and generative models recently, I have increasingly felt that this description is too crude.</p>

      <blockquote data-i18n="更准确的问题也许不是“要不要加先验”，而是：当观测数据不足以唯一决定答案时，我们希望模型借用哪一种结构性知识？">Perhaps the more precise question is not whether to add a prior, but what kind of structural knowledge we want the model to draw on when observations cannot uniquely determine the answer.</blockquote>

      <h2 id="motivation" data-i18n="Motivation">Motivation</h2>
      <p data-i18n="在反演问题里，我们常常只有间接观测。速度模型、地下结构、边界条件和噪声都会影响最终结果。单纯优化数据 misfit 很容易得到看似合理但物理上不稳定的解。">In inverse problems, we often have only indirect observations. Velocity models, subsurface structures, boundary conditions, and noise all influence the result. Optimizing the data misfit alone can readily produce solutions that look plausible but are physically unstable.</p>

      <div class="lab-formula-box">min<sub>m</sub> L(d, F(m)) + λ R(m)</div>

      <p data-i18n="这个表达式看起来简单，但真正困难的是右边的 R(m)：它到底是在惩罚什么？是在鼓励光滑、边缘、稀疏，还是某种从数据集中学来的地质结构？">The expression looks simple, but the real difficulty lies in R(m): what does it penalize? Does it encourage smoothness, edges, sparsity, or geological structures learned from a dataset?</p>

      <h2 id="math-view" data-i18n="A Small Mathematical View">A Small Mathematical View</h2>
      <p data-i18n="如果把生成模型看成一个结构分布的近似，那么先验就不再只是正则项，而是一个“可采样的知识空间”。这会改变我们看待实验失败的方式：失败不一定来自优化器，也可能来自训练分布与反演目标之间的错位。">If a generative model approximates a distribution over structures, a prior becomes more than a regularization term: it becomes a space of knowledge that we can sample. This changes how we interpret failed experiments. The optimizer may not be at fault; the training distribution may instead be misaligned with the inversion target.</p>

      <div class="lab-note-box">
        <b data-i18n="Revision note">Revision note</b>
        <p data-i18n="v0.3 版本准备补一张示意图：把 classical regularization、learned prior 和 diffusion prior 放在同一张坐标轴上比较。">For version 0.3, I plan to add a diagram comparing classical regularization, learned priors, and diffusion priors within a common set of axes.</p>
      </div>

      <h2 id="experiment-note" data-i18n="Experiment Note">Experiment Note</h2>
      <p data-i18n="最近一次二维速度模型实验里，我发现 score prior 接入后结果并不稳定。复盘后更像是尺度问题：训练样本的速度范围、归一化方式和反演数据的物理尺度没有对齐。">In a recent experiment with a two-dimensional velocity model, adding a score-based prior produced unstable results. Looking back, this seems more like a scale mismatch: the velocity range and normalization of the training samples were not aligned with the physical scale of the inversion data.</p>

      <h2 id="wechat-version" data-i18n="Wechat Version">Wechat Version</h2>
      <p data-i18n="公众号版本可以删去部分公式，把主线改成三个问题：为什么反演会不适定？先验到底在提供什么？生成模型为什么可能成为新的先验表达方式？">The WeChat version could omit some equations and focus on three questions: Why are inverse problems ill-posed? What does a prior provide? Why might generative models offer a new way to represent priors?</p>
    </div>
  </article>
</div>
