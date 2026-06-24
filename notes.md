---
layout: lab
permalink: /notes/index.html
title: Notes
comments: false
lab_nav: Notes
lab_subtitle: new notes bench
---

<section class="lab-hero" aria-labelledby="notes-title">
  <p class="lab-eyebrow">Notes / 新笔记工作台</p>
  <h1 id="notes-title">从零开始搭建新的学习笔记入口。</h1>
  <p class="lab-lead">旧笔记先统一放入 Old Archives；新的 Notes 将按更稳定的 topic、lecture、reference 和 problem log 结构逐步生长。</p>
</section>

<div class="lab-workspace">
  <aside class="lab-rail">
    <section class="lab-rail-box">
      <h2>Note Protocol</h2>
      <p>新笔记优先服务长期复习和科研复用，而不是临时堆放材料。</p>
      <div class="lab-rail-list">
        <div class="lab-rail-item"><span>Topic notes</span><b>00</b></div>
        <div class="lab-rail-item"><span>Lecture drafts</span><b>00</b></div>
        <div class="lab-rail-item"><span>Reference logs</span><b>00</b></div>
      </div>
    </section>
    <section class="lab-rail-box">
      <h2>Old Material</h2>
      <p>历史笔记没有删除，也没有批量移动；它们被集中索引到归档页，方便慢慢筛选和重写。</p>
    </section>
  </aside>

  <section class="lab-sheet">
    <div class="lab-section-head">
      <h2>New Notes Structure</h2>
      <span>clean start for future study notes</span>
    </div>

    <div class="lab-topic-grid">
      <article class="lab-topic-card">
        <p class="lab-label">Notebook 01</p>
        <h3>Mathematical Tools</h3>
        <p>用于整理研究中反复出现的数学工具：变分法、PDE、概率直觉、优化和线性代数。</p>
        <div class="lab-topic-foot"><span><b>00</b>notes</span><span><b>00</b>examples</span><span><b>00</b>remarks</span></div>
      </article>
      <article class="lab-topic-card">
        <p class="lab-label">Notebook 02</p>
        <h3>Machine Learning</h3>
        <p>用于重写深度学习、生成模型、强化学习和科学机器学习相关笔记。</p>
        <div class="lab-topic-foot"><span><b>00</b>notes</span><span><b>00</b>papers</span><span><b>00</b>codes</span></div>
      </article>
      <article class="lab-topic-card">
        <p class="lab-label">Notebook 03</p>
        <h3>Geophysics</h3>
        <p>用于组织地球物理、行星科学、反演问题和数值模拟相关材料。</p>
        <div class="lab-topic-foot"><span><b>00</b>notes</span><span><b>00</b>logs</span><span><b>00</b>figures</span></div>
      </article>
      <article class="lab-topic-card">
        <p class="lab-label">Archive</p>
        <h3><a href="{{ '/old-archives/' | relative_url }}">Old Archives</a></h3>
        <p>旧 Blogs 和 Notes 的统一入口。旧内容暂时保留原路径，后续可以逐篇筛选、重写或迁移。</p>
        <div class="lab-topic-foot"><span><b>01</b>archive</span><span><b>many</b>old files</span><span><b>safe</b>links</span></div>
      </article>
    </div>

    <div class="lab-section-head">
      <h2>Next Notes To Build</h2>
      <span>empty by design</span>
    </div>

    <div class="lab-entry-list">
      <article class="lab-entry-card">
        <time>Plan</time>
        <div>
          <h3>FWI 基础讲义：从波动方程到 forward operator</h3>
          <p>新 Notes 的第一批内容可以从 Research Log 中已经稳定的讲义抽出来。</p>
        </div>
        <span class="lab-state">Todo</span>
      </article>
      <article class="lab-entry-card">
        <time>Plan</time>
        <div>
          <h3>Score-based models 的最小数学骨架</h3>
          <p>把旧 AI 笔记中可复用的部分整理成更清楚的定义、例子和 remark。</p>
        </div>
        <span class="lab-state">Todo</span>
      </article>
    </div>
  </section>
</div>
