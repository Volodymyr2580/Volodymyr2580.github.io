# AGENTS.md

本文件是给 Codex/AI Agent 维护本仓库时使用的协作说明。这个仓库是 Wenzhe Sheng 的个人主页，优先目标是稳定展示个人信息、研究兴趣、博客、日记和学习笔记。

## 用户背景与沟通方式

- 用户是前后端开发初学者。涉及 Jekyll、GitHub Pages、HTML、CSS、Less、JavaScript、Markdown、Git 等概念时，需要用中文解释清楚背景、作用和风险。
- 不要只给结论。改动前后应简要说明为什么这么改，以及它会影响网站的哪些部分。
- 如果遇到多种实现方案，优先选择对初学者更容易理解、维护成本更低、对现有站点侵入更小的方案。

## 项目概览

这是一个 GitHub Pages/Jekyll 风格的静态个人主页。

- `_config.yml`：Jekyll 站点配置，包含站点标题、作者信息、导航链接、Markdown/MathJax 配置等。
- `index.md`、`awards.md`、`publications.md`、`blogs.md`、`diary.md`、`notes.md`：主要页面内容入口。
- `blogs/`：博客文章，主要是 Markdown 文件，也包含文章图片。
- `diary/`：日记/记录类 Markdown 内容。
- `notes/`：学习笔记、课程资料、图片、PDF 等内容。这里包含大量个人资料和素材，修改前要特别谨慎。
- `_layouts/`：Jekyll 页面模板，控制页面整体 HTML 结构。
- `_includes/`：Jekyll 可复用片段，如导航、页头、页脚、作者简介、脚本引用。
- `assets/less/`：Less 样式源文件。
- `assets/css/`：浏览器实际加载的 CSS 文件，当前主要是 `assets/css/main.css`。
- `assets/js/`：站点 JavaScript，包含第三方库和本地脚本。
- `images/`、`wenzhe.jpg`、`weight_progress.png`：站点图片资产。
- `weight_monitor.py`、`weights.csv`：体重记录/图表相关脚本和数据。

## 技术栈说明

- Jekyll：静态网站生成器，会把 Markdown、Liquid 模板和配置组合成 HTML 页面。
- GitHub Pages：常用于托管 Jekyll 静态站点，推送到 GitHub 后自动发布。
- Markdown：大部分正文内容使用 Markdown 编写。
- Liquid：Jekyll 模板语言，常见于 `_layouts/` 和 `_includes/` 里的 `{{ ... }}`、`{% ... %}`。
- Less/CSS：样式系统。Less 是 CSS 的预处理语言，`assets/css/main.css` 是最终页面直接引用的样式文件。
- MathJax：用于渲染 LaTeX 数学公式，页面模板中已有配置。

## 安全与文件操作规则

禁止批量删除文件或目录。

不要使用：

- `del /s`
- `rd /s`
- `rmdir /s`
- `Remove-Item -Recurse`
- `rm -rf`

需要删除文件时，只能一次删除一个明确路径的文件。

正确示例：

```powershell
Remove-Item "C:\path\to\file.txt"
```

如果需要批量删除文件，应停止操作，并请求用户手动删除。

其他注意事项：

- 不要删除或重命名 `blogs/`、`diary/`、`notes/`、`images/` 中的内容，除非用户明确指定具体文件。
- 不要清理、压缩或移动大量图片/PDF/笔记素材，除非用户明确要求并确认范围。
- 不要覆盖用户已有文章内容。修改 Markdown 正文前，先确认该文件与任务直接相关。
- 工作区可能已有用户未提交改动。不要回滚、重置或覆盖不属于本次任务的改动。
- 不要使用破坏性 Git 命令，例如 `git reset --hard` 或 `git checkout -- <file>`，除非用户明确要求。

## 编码与中文内容

- 优先使用 UTF-8 保存文本文件。
- 当前部分中文内容可能存在乱码。修复乱码前应先判断来源，避免二次破坏。
- 新增中文内容时，直接写正常中文，不要使用错误编码或转义乱码。
- 如果修改 README、主页简介、博客或笔记中的中文，需要在回复中提醒用户已注意编码问题。

## 改版与样式规则

用户认为当前个人主页排版一般，后续可能会进行视觉和布局改造。改版时遵循：

- 优先保持现有 Jekyll 结构，不要轻易迁移到 React/Vue/Next.js 等新框架。
- 先从 `_layouts/`、`_includes/`、`assets/less/`、`assets/css/main.css` 理解现有主题，再改样式。
- 导航、作者信息、主页图片、博客/笔记入口是核心体验，改版时不能破坏。
- 个人主页应偏学术、清晰、可信、易阅读，不要做成浮夸的营销落地页。
- 页面应兼顾桌面端和移动端，尤其检查导航、头像/照片、长标题、公式、图片是否溢出。
- 不要把所有内容塞进大量装饰性卡片。博客、笔记、日记列表应优先保证可扫描和可阅读。
- 涉及 MathJax、代码高亮、文章图片展示时，要确认原有功能仍可用。

## 内容维护规则

- Markdown 文件通常需要保留 YAML front matter，例如开头的 `---` 配置块。
- 新增博客或笔记时，文件名、标题、分类和链接应与现有目录习惯一致。
- 图片引用尽量使用仓库内已有相对路径或现有站点约定，避免随意引入外部不稳定链接。
- 学术信息、邮箱、社交链接、个人经历等属于用户身份信息，修改前要谨慎核对。
- `_config.yml` 中的 `site.links` 控制顶部导航，改导航时要同步考虑页面是否存在。

## 验证建议

优先使用项目已有方式验证。如果本地环境支持 Jekyll，可尝试：

```powershell
bundle exec jekyll serve
```

或：

```powershell
jekyll serve
```

如果没有 Ruby/Jekyll 环境，不要强行安装大批依赖；先说明当前无法本地预览，并给出可检查的文件级改动说明。

静态检查时至少关注：

- 页面 Markdown front matter 是否完整。
- Liquid 标签是否闭合。
- 导航链接是否与实际页面路径匹配。
- CSS 修改是否影响移动端。
- 中文是否仍为正常 UTF-8 文本。

## Git 提交规则

每次提交 git commit 时，遵循以下格式：

```text
<type>(<scope>): <summary>

<正文：描述本次变更的背景与动机>

Agent-Task: <原始任务描述或任务 ID>
Agent-Model: <使用的模型，如 gpt-4o、gemini-2.5-pro>
Agent-Decision: <关键设计决策及理由>
Agent-Limitation: <已知局限或后续 TODO>
```

示例：

```text
docs(agents): add repository guidance for AI collaborators

Document the structure, safety rules, and maintenance workflow for the personal homepage so future agent work can avoid damaging content assets.

Agent-Task: 为个人主页项目编写 AGENTS.md
Agent-Model: GPT-5
Agent-Decision: 保留现有 Jekyll/GitHub Pages 结构，强调安全删除、编码、内容资产保护和初学者解释要求。
Agent-Limitation: 未配置本地 Jekyll 预览环境，后续视觉改版仍需浏览器验证。
```

## 当前已知情况

- 本仓库根目录当前没有明确的 `Gemfile` 或 `package.json`，所以它更像一个直接托管在 GitHub Pages 上的 Jekyll 静态站，而不是 Node 前端工程。
- 当前 PowerShell 环境可能无法直接调用 `git` 命令；如需提交或查看状态，应先确认 Git 是否可用。
- README 和部分页面中可见中文乱码，后续整理内容时建议优先修复编码问题。
