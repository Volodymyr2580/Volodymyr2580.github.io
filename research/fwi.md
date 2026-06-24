---
layout: lab
permalink: /research/fwi/index.html
title: Full Waveform Inversion
comments: false
lab_nav: Research Log
lab_subtitle: research logbook
---

<section class="lab-hero" aria-labelledby="topic-title">
  <p class="lab-eyebrow">Research Topic / Topic 01</p>
  <h1 id="topic-title">Full Waveform Inversion</h1>
  <p class="lab-lead">一个 topic 页面像研究文件夹的封面：它告诉你这个方向在问什么、已经积累了什么、最近做过哪些实验、下一步要推进哪里。</p>
</section>

<div class="lab-detail-layout">
  <aside class="lab-detail-rail">
    <section class="lab-rail-box">
      <h2>Topic Status</h2>
      <p>这个侧栏适合放 topic 的稳定信息，而不是每篇文章都变化的正文。</p>
      <div class="lab-rail-list">
        <div class="lab-rail-item"><span>Stage</span><b>Active</b></div>
        <div class="lab-rail-item"><span>Lectures</span><b>06</b></div>
        <div class="lab-rail-item"><span>Logs</span><b>09</b></div>
        <div class="lab-rail-item"><span>Open Qs</span><b>04</b></div>
      </div>
    </section>
    <section class="lab-rail-box">
      <h2>Topic Index</h2>
      <nav class="lab-toc-list" aria-label="Topic table of contents">
        <a href="#overview">Overview</a>
        <a href="#lecture-sequence">Lecture Sequence</a>
        <a href="#latest-experiment">Latest Experiment</a>
        <a href="#recent-timeline">Recent Timeline</a>
        <a href="#open-questions">Open Questions</a>
      </nav>
    </section>
  </aside>

  <article class="lab-article-paper">
    <header class="lab-article-head">
      <p class="lab-eyebrow">Active Research Line</p>
      <h1>Full Waveform Inversion</h1>
      <p class="lab-lead">从 PDE 正问题、伴随状态法、优化景观到深度先验，逐步整理一条可复现的学习路径。</p>
      <div class="lab-topic-summary">
        <div><b>06</b><span>lecture drafts</span></div>
        <div><b>09</b><span>experiment logs</span></div>
        <div><b>04</b><span>paper notes</span></div>
        <div><b>17</b><span>open questions</span></div>
      </div>
    </header>

    <div class="lab-article-body">
      <h2 id="overview">Overview</h2>
      <p>这个 topic 关注如何从波场观测恢复地下速度结构。我的当前理解是：FWI 的困难不仅来自数值计算量，也来自目标函数的几何形状、频率信息缺失和先验表达方式。</p>

      <h2 id="lecture-sequence">Lecture Sequence</h2>
      <div class="lab-experiment-table">
        <div class="lab-experiment-row"><b>Lecture 01</b><span>PDE 正问题和离散化：从波动方程到可计算的 forward operator。</span></div>
        <div class="lab-experiment-row"><b>Lecture 02</b><span>伴随状态法：如何避免显式构造巨大的 Jacobian。</span></div>
        <div class="lab-experiment-row"><b>Lecture 03</b><span>Cycle skipping：为什么局部极小值会如此顽固。</span></div>
        <div class="lab-experiment-row"><b>Lecture 04</b><span>Regularization and priors：从 TV 到 learned prior。</span></div>
      </div>

      <h2 id="latest-experiment">Latest Experiment</h2>
      <div class="lab-note-box">
        <b>Experiment log</b>
        目的：测试 TV regularization 能否改善低频缺失时的早期迭代方向。初步结果显示边界变清楚了，但过强正则会牺牲深部结构。
      </div>

      <h2 id="recent-timeline">Recent Timeline</h2>
      <div class="lab-timeline">
        <article class="lab-timeline-item">
          <time>Jun 22, 2026</time>
          <div>
            <h3>测试 TV regularization 对 cycle skipping 的影响</h3>
            <p>新认识：正则项不是“越强越稳”，它也会改变模型可表达的结构。</p>
          </div>
        </article>
        <article class="lab-timeline-item">
          <time>Jun 16, 2026</time>
          <div>
            <h3>把 score prior 接入二维速度模型重建</h3>
            <p>失败原因更可能是训练分布与反演数据尺度不匹配，而不是 sampling 本身。</p>
          </div>
        </article>
        <article class="lab-timeline-item">
          <time>Jun 08, 2026</time>
          <div>
            <h3>整理伴随状态法讲义 v0.2</h3>
            <p>把矩阵推导改写成连续形式，后续更容易连接 PDE 约束优化。</p>
          </div>
        </article>
      </div>

      <h2 id="open-questions">Open Questions</h2>
      <p>下一步我想弄清楚 learned prior 在不同尺度上的稳定性：它究竟是在帮助优化，还是在把模型推向训练集里最常见的结构？</p>
    </div>
  </article>
</div>
