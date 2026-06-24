---
layout: lab
permalink: /research/index.html
title: Research Log
comments: false
lab_nav: Research Log
lab_subtitle: research logbook
---

<section class="lab-hero" aria-labelledby="research-title">
  <p class="lab-eyebrow">Research Log / 科研日志记录本</p>
  <h1 id="research-title">按 topic 记录目的、假设、实验、结果与新的认识。</h1>
  <p class="lab-lead">每个 topic 都是一条可以长期推进的研究线：从朴素问题开始，逐步沉淀讲义、论文笔记、实验记录和下一步计划。</p>
</section>

<div class="lab-workspace">
  <aside class="lab-rail">
    <section class="lab-rail-box">
      <h2>Lab Template</h2>
      <p>每条实验记录固定包含 motivation, hypothesis, setup, result, insight, next step。</p>
      <div class="lab-rail-list">
        <div class="lab-rail-item"><span>Active topics</span><b>04</b></div>
        <div class="lab-rail-item"><span>Experiment logs</span><b>12</b></div>
        <div class="lab-rail-item"><span>Open questions</span><b>17</b></div>
      </div>
    </section>
    <section class="lab-rail-box">
      <h2>Review Rhythm</h2>
      <p>每周整理一次失败实验和新增认识，每月把散乱记录合并成一份 topic lecture。</p>
    </section>
  </aside>

  <section class="lab-sheet">
    <div class="lab-section-head">
      <h2>Active Topics</h2>
      <span>research lines that can grow over semesters</span>
    </div>

    <div class="lab-topic-grid">
      <article class="lab-topic-card">
        <p class="lab-label">Topic 01</p>
        <h3><a href="{{ '/research/fwi/' | relative_url }}">Full Waveform Inversion</a></h3>
        <p>从 PDE 正问题、伴随状态法、优化景观到深度先验，整理一条可复现的学习路径。</p>
        <div class="lab-topic-foot"><span><b>06</b>lectures</span><span><b>09</b>logs</span><span><b>04</b>papers</span></div>
      </article>
      <article class="lab-topic-card">
        <p class="lab-label">Topic 02</p>
        <h3>Score-based Generative Models</h3>
        <p>把 SDE、score matching、flow matching 和反问题中的先验建模放在一起比较。</p>
        <div class="lab-topic-foot"><span><b>04</b>lectures</span><span><b>03</b>logs</span><span><b>07</b>papers</span></div>
      </article>
      <article class="lab-topic-card">
        <p class="lab-label">Topic 03</p>
        <h3>Scientific Computing Notes</h3>
        <p>记录 Devito、Python 数值实验、服务器环境和可复现实验脚本中的关键坑点。</p>
        <div class="lab-topic-foot"><span><b>03</b>guides</span><span><b>05</b>logs</span><span><b>02</b>todos</span></div>
      </article>
      <article class="lab-topic-card">
        <p class="lab-label">Topic 04</p>
        <h3>Mathematical Tools</h3>
        <p>把齐次化、变分法、泛函分析和概率直觉整理成面向研究问题的讲义。</p>
        <div class="lab-topic-foot"><span><b>08</b>notes</span><span><b>02</b>logs</span><span><b>11</b>questions</span></div>
      </article>
    </div>

    <div class="lab-section-head">
      <h2>Recent Experiment Matrix</h2>
      <span>purpose -> setup -> result -> next step</span>
    </div>

    <div class="lab-matrix">
      <article class="lab-matrix-row">
        <b>Jun 22 / FWI</b>
        <div>
          <strong>测试 TV regularization 对 cycle skipping 的影响</strong>
          <p>目的：验证简单正则项能否改善低频缺失时的早期迭代方向。</p>
        </div>
        <span>New insight</span>
      </article>
      <article class="lab-matrix-row">
        <b>Jun 16 / Prior</b>
        <div>
          <strong>把 score prior 接入二维速度模型重建</strong>
          <p>结果不稳定，但暴露出训练分布和反演数据尺度不匹配的问题。</p>
        </div>
        <span>Needs rerun</span>
      </article>
      <article class="lab-matrix-row">
        <b>Jun 08 / Lecture</b>
        <div>
          <strong>整理伴随状态法讲义 v0.2</strong>
          <p>把目标函数梯度推导从矩阵形式改写成连续形式，方便和 PDE 约束连接。</p>
        </div>
        <span>Updated</span>
      </article>
    </div>
  </section>
</div>
