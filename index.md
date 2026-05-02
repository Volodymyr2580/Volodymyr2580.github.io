---
layout: page
academic_home: true
comments: false
---

<section class="home-hero" aria-labelledby="home-title">
  <div class="home-hero__portrait">
    <img src="{{ '/wenzhe.jpg' | relative_url }}" alt="Wenzhe Sheng">
  </div>
  <div class="home-hero__content">
    <h1 id="home-title">Wenzhe Sheng</h1>
    <p class="home-hero__role">Undergraduate Researcher</p>
    <p class="home-hero__affiliation">Peking University</p>
    <p class="home-hero__school">School of Earth and Space Sciences</p>
    <p class="home-hero__summary">
      I work at the intersection of <em>AI</em> and <em>geophysics</em>.
      My current research focuses on deep learning methods for scientific problems,
      especially reinforcement learning, stochastic analysis, statistical field theory,
      and full waveform inversion for ChangE'4 Lunar Penetrating Radar data.
    </p>
    <div class="home-hero__links" aria-label="Primary contact links">
      <a href="mailto:{{ site.owner.email }}"><i class="icon-mail"></i> {{ site.owner.email }}</a>
      <a href="https://github.com/{{ site.owner.github }}"><i class="icon-github"></i> GitHub</a>
    </div>
  </div>
</section>

<section class="home-section home-about" aria-labelledby="about-me">
  <div class="home-section__heading">
    <h2 id="about-me">About Me</h2>
  </div>
  <div class="home-section__body">
    <p>
      Here is <strong>Wenzhe Sheng</strong> (Volodymyr, 盛文哲). I am an undergraduate student
      in the School of Earth and Space Sciences at Peking University, China.
    </p>
    <p>
      大家好，我是盛文哲，目前是北京大学地球与空间科学学院的本科生。
      如果您对我的研究、学习笔记或主页内容感兴趣，欢迎通过邮件或社交媒体与我联系。
    </p>
    <p>
      I value clarity in thinking and writing, and I use this website to keep track of research,
      learning notes, technical experiments, and personal reflections.
    </p>
  </div>
  <aside class="home-profile-facts" aria-label="Profile facts">
    <p><strong>Peking University</strong><span>Beijing, China</span></p>
    <p><strong>Undergraduate</strong><span>Earth & Space Sciences</span></p>
    <p><strong>Research Focus</strong><span>AI, Geophysics, Inverse Problems</span></p>
    <p><strong>Languages</strong><span>中文 | English</span></p>
  </aside>
</section>

<section class="home-section home-research" aria-labelledby="research-interests">
  <div class="home-section__heading">
    <h2 id="research-interests">Research Interests</h2>
  </div>
  <div class="research-grid">
    <article>
      <span class="research-icon">AI</span>
      <h3>AI</h3>
      <p>Machine learning and deep learning methods for scientific discovery and inverse problems.</p>
    </article>
    <article>
      <span class="research-icon">GEO</span>
      <h3>Geophysics</h3>
      <p>Seismic and radar wave propagation, subsurface imaging, and planetary science.</p>
    </article>
    <article>
      <span class="research-icon">RL</span>
      <h3>Reinforcement Learning</h3>
      <p>Sequential decision making and control in high-dimensional physical systems.</p>
    </article>
    <article>
      <span class="research-icon">SA</span>
      <h3>Stochastic Analysis</h3>
      <p>Stochastic processes, random fields, and uncertainty quantification in geosciences.</p>
    </article>
    <article>
      <span class="research-icon">SFT</span>
      <h3>Statistical Field Theory</h3>
      <p>Field-theoretic methods for complex systems and scalable inference.</p>
    </article>
    <article>
      <span class="research-icon">FWI</span>
      <h3>Full Waveform Inversion</h3>
      <p>Theory and algorithms for FWI with a focus on ChangE'4 Lunar Penetrating Radar data.</p>
    </article>
  </div>
</section>

<section class="home-section home-updates" aria-labelledby="news-updates">
  <div class="home-section__heading">
    <h2 id="news-updates">News and Updates</h2>
  </div>
  <div class="update-list">
    <p><time datetime="2026-05-02">2026.05.02</time><span>Redesigned this homepage for a cleaner academic reading experience.</span></p>
    <p><time datetime="2026-04-24">2026.04.24</time><span>Updated personal tracking materials and site assets.</span></p>
    <p><time datetime="2026-03-30">2026.03.30</time><span>Continued organizing blogs, diary records, and learning notes.</span></p>
    <p><time datetime="2025-05-24">2025.05.24</time><span>Started using this homepage to record research, learning, and personal life.</span></p>
  </div>
</section>

<section class="home-section home-explore" aria-labelledby="explore-more">
  <div class="home-section__heading">
    <h2 id="explore-more">Explore More</h2>
  </div>
  <div class="explore-list">
    <article>
      <h3><a href="{{ '/blogs/' | relative_url }}">Blogs</a></h3>
      <p>Thoughts on research, methods, learning, and science.</p>
      <a class="text-link" href="{{ '/blogs/' | relative_url }}">Browse all posts</a>
    </article>
    <article>
      <h3><a href="{{ '/notes/' | relative_url }}">Notes</a></h3>
      <p>Study notes, course materials, technical summaries, and references.</p>
      <a class="text-link" href="{{ '/notes/' | relative_url }}">Browse all notes</a>
    </article>
    <article>
      <h3><a href="{{ '/diary/' | relative_url }}">Diary</a></h3>
      <p>A research and learning diary for reflections and records.</p>
      <a class="text-link" href="{{ '/diary/' | relative_url }}">Browse diary</a>
    </article>
  </div>
</section>

<section class="home-section home-site-note" aria-labelledby="about-homepage">
  <div class="home-section__heading">
    <h2 id="about-homepage">About this homepage</h2>
  </div>
  <div class="home-section__body">
    <p>
      This website is built with Jekyll and GitHub Pages. I will keep updating notes,
      tutorials, blogs, and diary entries as a public record of my research and learning.
    </p>
  </div>
</section>
