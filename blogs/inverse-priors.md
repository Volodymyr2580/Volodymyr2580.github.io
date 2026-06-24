---
layout: lab
permalink: /blogs/inverse-priors/index.html
title: 从反演问题到生成模型：为什么我开始重新理解“先验”
comments: false
lab_nav: Blogs
lab_subtitle: blue-sky lab notebook
---

<section class="lab-hero" aria-labelledby="article-title">
  <p class="lab-eyebrow">Blog Entry / 公众号母稿</p>
  <h1 id="article-title">从反演问题到生成模型：为什么我开始重新理解“先验”</h1>
  <p class="lab-lead">这类详情页的重点不是炫技，而是让长文、公式、补充材料和公众号同步状态都能安静地放在同一个页面里。</p>
</section>

<div class="lab-detail-layout">
  <aside class="lab-detail-rail">
    <section class="lab-rail-box">
      <h2>Entry Info</h2>
      <p>网页长稿保留完整推导和修订历史，公众号版本可以只截取主线叙事。</p>
      <div class="lab-rail-list">
        <div class="lab-rail-item"><span>Date</span><b>2026-06-24</b></div>
        <div class="lab-rail-item"><span>Status</span><b>Draft</b></div>
        <div class="lab-rail-item"><span>Version</span><b>v0.3</b></div>
      </div>
    </section>
    <section class="lab-rail-box">
      <h2>On This Page</h2>
      <nav class="lab-toc-list" aria-label="Article table of contents">
        <a href="#motivation">Motivation</a>
        <a href="#math-view">A Small Mathematical View</a>
        <a href="#experiment-note">Experiment Note</a>
        <a href="#wechat-version">Wechat Version</a>
      </nav>
    </section>
  </aside>

  <article class="lab-article-paper">
    <header class="lab-article-head">
      <p class="lab-eyebrow">Featured Observation</p>
      <h1>从反演问题到生成模型：为什么我开始重新理解“先验”</h1>
      <div class="lab-article-meta">
        <span>Research Essay</span>
        <span class="lab-stamp">Wechat draft</span>
        <span>FWI</span>
        <span>12 min read</span>
      </div>
    </header>

    <div class="lab-article-body">
      <p>我一开始把“先验”理解成一种外加约束：数据不够时，给模型加一点额外偏好。但最近读 FWI 和生成模型相关论文时，我越来越觉得这个说法太粗糙。</p>

      <blockquote>更准确的问题也许不是“要不要加先验”，而是：当观测数据不足以唯一决定答案时，我们希望模型借用哪一种结构性知识？</blockquote>

      <h2 id="motivation">Motivation</h2>
      <p>在反演问题里，我们常常只有间接观测。速度模型、地下结构、边界条件和噪声都会影响最终结果。单纯优化数据 misfit 很容易得到看似合理但物理上不稳定的解。</p>

      <div class="lab-formula-box">min<sub>m</sub> L(d, F(m)) + λ R(m)</div>

      <p>这个表达式看起来简单，但真正困难的是右边的 R(m)：它到底是在惩罚什么？是在鼓励光滑、边缘、稀疏，还是某种从数据集中学来的地质结构？</p>

      <h2 id="math-view">A Small Mathematical View</h2>
      <p>如果把生成模型看成一个结构分布的近似，那么先验就不再只是正则项，而是一个“可采样的知识空间”。这会改变我们看待实验失败的方式：失败不一定来自优化器，也可能来自训练分布与反演目标之间的错位。</p>

      <div class="lab-note-box">
        <b>Revision note</b>
        v0.3 版本准备补一张示意图：把 classical regularization、learned prior 和 diffusion prior 放在同一张坐标轴上比较。
      </div>

      <h2 id="experiment-note">Experiment Note</h2>
      <p>最近一次二维速度模型实验里，我发现 score prior 接入后结果并不稳定。复盘后更像是尺度问题：训练样本的速度范围、归一化方式和反演数据的物理尺度没有对齐。</p>

      <h2 id="wechat-version">Wechat Version</h2>
      <p>公众号版本可以删去部分公式，把主线改成三个问题：为什么反演会不适定？先验到底在提供什么？生成模型为什么可能成为新的先验表达方式？</p>
    </div>
  </article>
</div>
