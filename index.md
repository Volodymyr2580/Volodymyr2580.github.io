---
layout: lab
title: About Me
comments: false
lab_nav: About Me
lab_subtitle: personal academic homepage
---

<section class="lab-hero" aria-labelledby="home-title">
  <p class="lab-eyebrow">About Me / 个人学术主页</p>
  <h1 id="home-title">Wenzhe Sheng</h1>
  <p class="lab-lead">Undergraduate researcher at Peking University, School of Earth and Space Sciences. I use this website to keep research topics, public writing, notes, and personal progress in one calm place.</p>
</section>

<div class="lab-workspace">
  <aside class="lab-rail">
    <section class="lab-rail-box">
      <h2>Profile</h2>
      <p>盛文哲 / Volodymyr. Undergraduate student in Earth and Space Sciences at Peking University.</p>
      <div class="lab-rail-list">
        <div class="lab-rail-item"><span>Institution</span><b>PKU</b></div>
        <div class="lab-rail-item"><span>Field</span><b>Geophysics</b></div>
        <div class="lab-rail-item"><span>Focus</span><b>AI + Inversion</b></div>
      </div>
    </section>
    <section class="lab-rail-box">
      <h2>Contact</h2>
      <p><a href="mailto:{{ site.owner.email }}">{{ site.owner.email }}</a></p>
      <p><a href="https://github.com/{{ site.owner.github }}">GitHub: {{ site.owner.github }}</a></p>
    </section>
  </aside>

  <section class="lab-sheet">
    <div class="lab-section-head">
      <h2>Research Interests</h2>
      <span>current academic direction</span>
    </div>
    <div class="lab-topic-grid">
      <article class="lab-topic-card">
        <p class="lab-label">Research 01</p>
        <h3>AI for Scientific Problems</h3>
        <p>Machine learning and deep learning methods for scientific discovery, inverse problems, and physical modeling.</p>
        <div class="lab-topic-foot"><span><b>AI</b>methods</span><span><b>ML</b>models</span><span><b>Sci</b>tasks</span></div>
      </article>
      <article class="lab-topic-card">
        <p class="lab-label">Research 02</p>
        <h3>Geophysics and Inversion</h3>
        <p>Wave propagation, subsurface imaging, full waveform inversion, and planetary radar data interpretation.</p>
        <div class="lab-topic-foot"><span><b>FWI</b>topic</span><span><b>PDE</b>models</span><span><b>Radar</b>data</span></div>
      </article>
      <article class="lab-topic-card">
        <p class="lab-label">Research 03</p>
        <h3>Mathematical Tools</h3>
        <p>Stochastic analysis, optimization, statistical field theory, and the mathematical language behind models.</p>
        <div class="lab-topic-foot"><span><b>Math</b>notes</span><span><b>Proof</b>ideas</span><span><b>Tools</b>reuse</span></div>
      </article>
      <article class="lab-topic-card">
        <p class="lab-label">Writing</p>
        <h3>Learning and Public Notes</h3>
        <p>I keep this site as a long-term archive for research logs, public essays, lecture drafts, and study notes.</p>
        <div class="lab-topic-foot"><span><b>Blogs</b>public</span><span><b>Notes</b>study</span><span><b>Logs</b>research</span></div>
      </article>
    </div>

    <div class="lab-section-head">
      <h2>Site Map</h2>
      <span>where things live now</span>
    </div>
    <div class="lab-entry-list">
      <article class="lab-entry-card"><time>Write</time><div><h3><a href="{{ '/blogs/' | relative_url }}">Blogs</a></h3><p>公众号同步文章和公开写作母稿。</p></div><span class="lab-state">New</span></article>
      <article class="lab-entry-card"><time>Research</time><div><h3><a href="{{ '/research/' | relative_url }}">Research Log</a></h3><p>按 topic 记录讲义、实验、论文阅读和认知迭代。</p></div><span class="lab-state">Active</span></article>
      <article class="lab-entry-card"><time>Study</time><div><h3><a href="{{ '/notes/' | relative_url }}">Notes</a></h3><p>新的学习笔记入口，从零开始搭建。</p></div><span class="lab-state">Reset</span></article>
      <article class="lab-entry-card"><time>Archive</time><div><h3><a href="{{ '/old-archives/' | relative_url }}">Old Archives</a></h3><p>旧 Blogs 和旧 Notes 的集中索引。</p></div><span class="lab-state">Preserved</span></article>
    </div>
  </section>
</div>
