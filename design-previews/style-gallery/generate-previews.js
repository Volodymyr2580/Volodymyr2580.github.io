const fs = require("fs");
const path = require("path");

const outDir = __dirname;

const posts = [
  ["Jun 24", "从反演问题到生成模型：重新理解先验", "公众号母稿 / Research essay"],
  ["Jun 18", "齐次化不只是技巧，而是一种尺度观察", "Math note"],
  ["Jun 10", "读论文时怎样把“没懂”变成可推进的问题", "Learning reflection"],
];

const logs = [
  ["FWI", "测试 TV regularization 对 cycle skipping 的影响", "New insight"],
  ["Score Models", "把 score prior 接入二维速度模型重建", "Needs rerun"],
  ["Adjoint", "整理伴随状态法讲义 v0.2", "Lecture updated"],
];

const topics = ["Full Waveform Inversion", "Score-based Generative Models", "Scientific Computing", "Mathematical Tools"];

const themes = [
  {
    id: "01-scholar-notes",
    name: "清冷学术札记风",
    note: "少卡片、细线、留白，像一个长期维护的学术笔记入口。",
    className: "scholar",
  },
  {
    id: "02-japanese-diary",
    name: "日式个人博客风",
    note: "更柔和，有生活感，适合公众号同步和轻量日记。",
    className: "japanese",
  },
  {
    id: "03-fomalhaut-inspired",
    name: "Fomalhaut 参考站风",
    note: "保留个人博客门户、侧栏、文章流，但避免直接照搬。",
    className: "fomalhaut",
  },
  {
    id: "04-library-catalog",
    name: "复古图书馆目录风",
    note: "像档案柜和馆藏目录，适合 topic 长期归档。",
    className: "library",
  },
  {
    id: "05-typographic-minimal",
    name: "极简 Typographic 风",
    note: "几乎不用装饰，靠编号、字体层级和秩序感成立。",
    className: "typeonly",
  },
  {
    id: "06-obsidian-wiki",
    name: "Obsidian 知识库风",
    note: "左侧 topic 树，右侧内容流，强调知识网络。",
    className: "wiki",
  },
  {
    id: "07-lab-notebook",
    name: "实验室 Lab Notebook 风",
    note: "强调目的、假设、设置、结果和下一步，适合实验归档。",
    className: "lab",
  },
  {
    id: "08-literary-magazine",
    name: "文学杂志 Essay 风",
    note: "更像独立刊物，适合公开文章和公众号长文。",
    className: "magazine",
  },
  {
    id: "09-latex-handout",
    name: "黑白数学讲义风",
    note: "接近 LaTeX handout，适合数学推导、定义、remark。",
    className: "latex",
  },
  {
    id: "10-research-map",
    name: "个人研究地图风",
    note: "首页像 topic map，把讲义、论文和实验挂到研究节点上。",
    className: "map",
  },
];

function postList() {
  return posts.map(([date, title, meta], index) => `
    <article class="post-item">
      <time>${date}</time>
      <div>
        <h3>${title}</h3>
        <p>${meta}</p>
      </div>
      <span>${index === 0 ? "Draft" : "Read"}</span>
    </article>
  `).join("");
}

function logList() {
  return logs.map(([topic, title, result]) => `
    <article class="log-item">
      <b>${topic}</b>
      <div>
        <h3>${title}</h3>
        <p>Motivation -> setup -> result -> interpretation -> next step</p>
      </div>
      <span>${result}</span>
    </article>
  `).join("");
}

function topicCloud() {
  return topics.map((topic, index) => `
    <a class="topic topic-${index + 1}" href="#">
      <strong>${topic}</strong>
      <span>${index + 3} lectures / ${index + 5} logs</span>
    </a>
  `).join("");
}

function page(theme) {
  return `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${theme.name}</title>
  <link rel="stylesheet" href="./preview-styles.css">
</head>
<body class="${theme.className}">
  <main class="sheet">
    <header class="site-head">
      <a class="brand" href="#">
        <img src="../../images/bigheadwenzhe.jpg" alt="Wenzhe Sheng">
        <span>Wenzhe Sheng</span>
      </a>
      <nav>
        <a>About</a>
        <a class="active">Blogs</a>
        <a>Research Log</a>
        <a>Notes</a>
      </nav>
    </header>

    <section class="style-title">
      <p>Style ${theme.id.slice(0, 2)}</p>
      <h1>${theme.name}</h1>
      <span>${theme.note}</span>
    </section>

    <section class="split">
      <article class="panel blog-panel">
        <div class="panel-head">
          <p>Blogs / 公众号同步</p>
          <h2>公开文章、数学讲解和研究随笔</h2>
        </div>
        <div class="feature">
          <p class="label">Featured Draft</p>
          <h3>从反演问题到生成模型：为什么我开始重新理解“先验”</h3>
          <p>网页保留更完整的推导、参考文献和实验附录；公众号版本负责讲清主线。</p>
        </div>
        <div class="list">
          ${postList()}
        </div>
      </article>

      <article class="panel research-panel">
        <div class="panel-head">
          <p>Research Log / 科研工作台</p>
          <h2>按 topic 记录问题、讲义、实验和认知迭代</h2>
        </div>
        <div class="topics">
          ${topicCloud()}
        </div>
        <div class="list logs">
          ${logList()}
        </div>
      </article>
    </section>
  </main>
</body>
</html>`;
}

const css = `
*{box-sizing:border-box}
body{margin:0;color:var(--ink);background:var(--bg);font-family:var(--font);font-size:16px;line-height:1.58}
a{text-decoration:none;color:inherit}
.sheet{width:min(1180px,calc(100vw - 72px));margin:0 auto;padding:26px 0 44px}
.site-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:38px}
.brand{display:flex;align-items:center;gap:12px;font-weight:700}
.brand img{width:38px;height:38px;border-radius:50%;object-fit:cover;border:1px solid var(--line)}
nav{display:flex;gap:20px;color:var(--muted);font-size:14px}
nav .active{color:var(--ink);font-weight:700}
.style-title{margin-bottom:28px}
.style-title p{margin:0 0 8px;color:var(--accent);font-size:12px;font-weight:800;letter-spacing:.12em;text-transform:uppercase}
.style-title h1{margin:0;font-size:44px;line-height:1.12;letter-spacing:0}
.style-title span{display:block;margin-top:10px;color:var(--muted);max-width:780px}
.split{display:grid;grid-template-columns:1fr 1fr;gap:24px;align-items:start}
.panel{min-height:720px;background:var(--paper);border:1px solid var(--line);border-radius:var(--radius);padding:24px}
.panel-head{margin-bottom:18px}
.panel-head p,.label{margin:0 0 7px;color:var(--accent);font-size:12px;font-weight:800;letter-spacing:.08em;text-transform:uppercase}
.panel-head h2{margin:0;font-size:27px;line-height:1.22;letter-spacing:0}
.feature{padding:20px;margin-bottom:20px;border:1px solid var(--line);background:var(--soft);border-radius:calc(var(--radius) - 2px)}
.feature h3{margin:0 0 10px;font-size:22px;line-height:1.32}
.feature p:last-child{margin:0;color:var(--muted)}
.list{display:grid;gap:11px}
.post-item,.log-item{display:grid;grid-template-columns:70px minmax(0,1fr)64px;gap:14px;align-items:start;padding:14px 0;border-bottom:1px solid var(--line)}
.post-item:last-child,.log-item:last-child{border-bottom:0}
.post-item time,.post-item span,.log-item b,.log-item span{color:var(--muted);font-size:12px;font-weight:700}
.post-item h3,.log-item h3{margin:0 0 3px;font-size:17px;line-height:1.35}
.post-item p,.log-item p{margin:0;color:var(--muted);font-size:13px}
.topics{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:22px}
.topic{display:block;padding:14px;border:1px solid var(--line);border-radius:calc(var(--radius) - 2px);background:var(--soft)}
.topic strong{display:block;margin-bottom:6px;font-size:15px;line-height:1.28}
.topic span{color:var(--muted);font-size:12px}

.scholar{--bg:#f7f8f6;--paper:#fff;--ink:#172026;--muted:#59646d;--line:#d7ded9;--accent:#2e6f63;--soft:#f4f7f4;--radius:4px;--font:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC",Arial,sans-serif}
.scholar .panel{box-shadow:none}.scholar .site-head{border-bottom:1px solid var(--line);padding-bottom:18px}.scholar .feature{border-left:4px solid var(--accent)}

.japanese{--bg:#fbfaf6;--paper:#fffdf8;--ink:#2c2925;--muted:#766f68;--line:#e7ded0;--accent:#b05c48;--soft:#fff7eb;--radius:10px;--font:"Yu Gothic","Hiragino Sans","Noto Sans SC",Arial,sans-serif}
.japanese .style-title h1{font-weight:650}.japanese .panel{border-color:#eadfce}.japanese .feature{background:#fff6e7}.japanese nav .active{border-bottom:2px solid #d4a373}

.fomalhaut{--bg:#eef2f6;--paper:#fff;--ink:#192231;--muted:#637083;--line:#d6dee8;--accent:#4c6fb3;--soft:#f2f6fc;--radius:8px;--font:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC",Arial,sans-serif}
.fomalhaut .sheet{width:min(1200px,calc(100vw - 56px))}.fomalhaut .site-head{background:#fff;border:1px solid var(--line);border-radius:8px;padding:15px 18px}.fomalhaut .panel{box-shadow:0 12px 30px rgba(50,70,100,.08)}.fomalhaut .feature{background:linear-gradient(135deg,#f7fbff,#eef3ff)}

.library{--bg:#eee7d6;--paper:#fffaf0;--ink:#302819;--muted:#756852;--line:#d8c9aa;--accent:#8b4e23;--soft:#f7ecd5;--radius:2px;--font:Georgia,"Noto Serif SC","Times New Roman",serif}
.library .brand img{border-radius:3px}.library .site-head{border-bottom:3px double var(--line);padding-bottom:16px}.library .style-title h1{font-weight:500}.library .topic{border-left:5px solid #b88b4a}

.typeonly{--bg:#fff;--paper:#fff;--ink:#111;--muted:#666;--line:#222;--accent:#111;--soft:#fff;--radius:0;--font:"Helvetica Neue",Arial,"Noto Sans SC",sans-serif}
.typeonly .sheet{width:min(1120px,calc(100vw - 90px))}.typeonly .site-head{border-bottom:2px solid #111;padding-bottom:14px}.typeonly .panel{border-width:0;border-top:2px solid #111;padding-left:0;padding-right:0}.typeonly .feature{border-width:0 0 1px 0;padding-left:0}.typeonly .topic{border-width:0 0 1px 0;padding-left:0}

.wiki{--bg:#20242c;--paper:#272c35;--ink:#f0f3f6;--muted:#aab4bf;--line:#3a414d;--accent:#8cc8ff;--soft:#222a35;--radius:6px;--font:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC",Arial,sans-serif}
.wiki .brand img{border-color:#4a5564}.wiki .site-head{border-bottom:1px solid var(--line);padding-bottom:16px}.wiki .feature{background:#1f2936}.wiki .topic{background:#202a35}.wiki .panel{box-shadow:inset 0 1px 0 rgba(255,255,255,.03)}

.lab{--bg:#edf1ee;--paper:#fbfdfb;--ink:#1f2d28;--muted:#5d6e67;--line:#cfdad4;--accent:#247256;--soft:#edf8f2;--radius:6px;--font:"IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC",Arial,sans-serif}
.lab .feature{background:#fff;border-left:6px solid #247256}.lab .post-item,.lab .log-item{grid-template-columns:82px minmax(0,1fr)90px}.lab .topic strong:before{content:"EXP ";color:#247256;font-size:11px;margin-right:4px}

.magazine{--bg:#f6f1ea;--paper:#fffaf3;--ink:#201913;--muted:#766a5f;--line:#dfd0bf;--accent:#9c3f32;--soft:#fbefe7;--radius:0;--font:"Iowan Old Style",Georgia,"Noto Serif SC",serif}
.magazine .style-title h1{font-size:54px;font-weight:500}.magazine .panel-head h2{font-size:34px;font-weight:500}.magazine .panel{padding:30px}.magazine .feature h3{font-size:28px;font-weight:500}

.latex{--bg:#fdfdfb;--paper:#fff;--ink:#111;--muted:#555;--line:#aaa;--accent:#174a8b;--soft:#fafafa;--radius:0;--font:"Latin Modern Roman","Computer Modern",Georgia,"Noto Serif SC",serif}
.latex .style-title h1{font-size:40px;text-align:center}.latex .style-title{text-align:center;border-bottom:1px solid #aaa;padding-bottom:18px}.latex .panel-head h2:before{content:"§ ";color:#174a8b}.latex .feature{border:1px solid #888}.latex .feature:before{content:"Remark.";font-weight:700;margin-right:6px}.latex nav{font-family:Arial,sans-serif}

.map{--bg:#eef4f1;--paper:#fbfffd;--ink:#162621;--muted:#5e7069;--line:#cddbd4;--accent:#3666a3;--soft:#edf7ff;--radius:14px;--font:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC",Arial,sans-serif}
.map .topics{position:relative;grid-template-columns:1fr 1fr;margin-top:18px}.map .topic{border-radius:999px;text-align:center;background:#fff}.map .topic-1{border-color:#3666a3}.map .topic-2{border-color:#2f7b68}.map .topic-3{border-color:#a26b2b}.map .topic-4{border-color:#965c75}.map .feature{border-radius:18px}.map .panel{border-radius:18px}

@media(max-width:900px){.sheet{width:calc(100vw - 32px)}.split{grid-template-columns:1fr}.panel{min-height:auto}.style-title h1{font-size:34px}nav{display:none}}
`;

fs.writeFileSync(path.join(outDir, "preview-styles.css"), css, "utf8");

for (const theme of themes) {
  fs.writeFileSync(path.join(outDir, `${theme.id}.html`), page(theme), "utf8");
}

const gallery = `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Style Gallery</title>
  <style>
    *{box-sizing:border-box}
    body{margin:0;background:#f4f4f1;color:#172026;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC",Arial,sans-serif}
    main{width:1320px;margin:0 auto;padding:28px 0 40px}
    header{margin-bottom:22px;border-bottom:1px solid #d9ddd8;padding-bottom:16px}
    h1{margin:0;font-size:34px}
    p{margin:8px 0 0;color:#667}
    .grid{display:grid;grid-template-columns:repeat(2,1fr);gap:18px}
    figure{margin:0;background:#fff;border:1px solid #d8ddd8;border-radius:8px;overflow:hidden}
    img{display:block;width:100%;height:360px;object-fit:cover;object-position:top}
    figcaption{display:flex;justify-content:space-between;gap:16px;padding:12px 14px;font-weight:700}
    figcaption span{color:#69747c;font-weight:500}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Blogs + Research Log：10 种设计风格预览</h1>
      <p>每张缩略图都包含 Blogs 和 Research Log 两个区域，用来快速比较整体气质。</p>
    </header>
    <section class="grid">
      ${themes.map((theme) => `<figure><img src="./${theme.id}.png" alt="${theme.name}"><figcaption>${theme.id.slice(0,2)} ${theme.name}<span>${theme.note}</span></figcaption></figure>`).join("")}
    </section>
  </main>
</body>
</html>`;

fs.writeFileSync(path.join(outDir, "gallery.html"), gallery, "utf8");
