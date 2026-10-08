# Particle Animation

互动实验室的第一个子项目。提供爱心、花朵、土星和烟花四种粒子造型，保留原 ParticleAnimation 的着色器、图片和音乐，增加主页导航入口、手动控制、摄像头按需启用、失败提示和手机布局。

## 网站结构

- `/playground/`：由根目录 `playground.md` 生成的 Jekyll 栏目页，中英文文案使用 `_data/i18n.json`。
- `/playground/particle-animation/`：独立的全屏静态应用，包含已构建的 HTML、CSS、JavaScript 和媒体文件。
- `_playground/particle-animation/`：可维护源码。以下划线开头的源码目录不进入 Jekyll 站点；网站继续使用现有 Jekyll 发布方式，无需把整个站点迁移至 React。

## 修改与构建

在本目录打开终端。首次安装依赖：

```powershell
npm ci
```

预览与构建：

```powershell
npm run dev
npm run build
```

构建结果写入仓库的 `playground/particle-animation/`。源码修改后应重新构建，将源码和构建结果一并提交。GitHub Pages 不需要运行 Node.js。Vite 使用相对资源路径，避免子目录部署时音乐和图片失效。

构建设置 `emptyOutDir: false`，不会批量清理文件；固定输出文件名便于覆盖更新。若未来产生不再使用的文件，应逐个确认后处理。

## 交互与依赖

- 默认展示爱心；点击造型后聚合，按钮控制聚合／散开，滑块控制旋转。
- 摄像头默认关闭，开启后在浏览器本地运行 MediaPipe Hands。握拳聚合、张手散开、手腕转动控制旋转。手动操作聚合／散开或旋转会退出摄像头模式并停止视频轨道。
- 手势模型来自固定版本的 jsDelivr HTTPS 地址，需要网络访问。权限被拒绝、设备不可用或模型加载失败时，可继续使用手动控制。
- 音乐在首次点击后加载和播放，默认不自动播放。
- 桌面使用 64,000 个粒子，首次打开的窄屏设备使用 24,000 个粒子；隐藏标签页时跳过绘制和手势推理。
- 原项目来源：`E:\i^2计划\cg's part\cgthebest-project\cgthebest\ParticleAnimation`。本仓库保存独立源码副本，未搬走原项目文件。

## 验证范围

已在 Chromium 验证栏目导航、中英文切换、造型切换、聚合／散开、颜色、旋转、音乐播放暂停、拒绝摄像头后的降级，以及 1440×900、390×844、320×640 布局。模拟摄像头验证了识别模型加载、640 像素视频输入和关闭后的轨道释放；真实手部识别效果仍需要带摄像头的设备实测。

没有本地 Ruby/Jekyll 环境时，Liquid/Markdown 预览只能检查页面与交互，不能替代 GitHub Pages 的最终 Jekyll 构建与线上访问验证。
