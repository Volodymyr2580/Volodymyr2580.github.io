# Beachball Plotter

Playground 的双力偶震源机制球绘图工具。正式页面位于 `playground/beachball/index.html`，从 `playground.md` 进入。无需构建、联网库或计算服务器；也可以直接用浏览器打开 HTML 文件。

## 文件与维护

- `playground/beachball/index.html`：页面、参数说明、开源来源。
- `playground/beachball/style.css`：仅作用于此独立页面的样式。
- `playground/beachball/mechanism.js`：向量、矩张量、主轴、等面积投影与节面区域。
- `playground/beachball/app.js`：输入联动、SVG 绘制、初动探针、SVG/PNG 导出。
- `_playground/beachball/`：维护说明、独立参考数据及检查脚本；不是第二份网站源码。
- `design-previews/beachball/`：浏览器截图、导出示例及检查结果；项目配置已排除该目录的发布。

新增入口时仅向 `playground.md` 和 `_data/i18n.json` 添加了本工具对应的条目，保留已有小岛实验入口与其他未提交内容。所有新增中文文件使用 UTF-8。

## 科学约定

采用 Aki–Richards 约定，坐标为北、东、下（NED），三项输入单位均为度：

- 走向从北顺时针，范围 0–360°；360° 与 0° 等价。
- 倾角 0–90°，倾向在走向右侧。
- 滑动角 −180–180°；+90° 为逆断层，−90° 为正断层，0° 为左旋走滑。
- 纯双力偶，单位滑动向量与单位法向量正交，标量矩归一化为 1。

矩张量与相对 P 波辐射系数：

$$
M = s n^{\mathsf T} + n s^{\mathsf T}, \qquad A(r) = r^{\mathsf T} M r.
$$

其中 $r$ 是从震源向外的单位射线。$A>0$ 填色并标为压缩初动；$A<0$ 留白并标为膨胀初动。P/T 分别是最小/最大特征值对应的轴，因此 P 落在负初动区、T 落在正初动区，不能按轴名倒置填色。初动分类将 $|A|<10^{-7}$ 视为数值零。

投影为下半球 Lambert 等面积投影。单位圆中，圆心为竖直向下、圆周为水平。探针出射角从竖直向下起算，仅接受 0–90°，不使用台站入射角或从向上方向起算的出射角。主轴读数采用方位角和相对水平面的俯角；竖直轴方位未定义。水平轴的两个对径端点等价，程序选取稳定代表。

填色直接使用两个节面同号半球区域的交集，不依赖像素轮廓追踪。投影曲线以 0.5° 的球面角间隔折线化，所以 SVG 是纯矢量，但曲线仍有有限采样精度。极性与轴读数通过解析向量计算。输入断层面为实线，辅助面为虚线；没有额外证据时无法用机制球识别真实断层。

初动探针和图中的正负号均是理论预测，不是观测数据。未建模地球传播、自由表面、仪器响应，也不支持非双力偶矩张量。

## 开源参考与复现

优先参考并核对以下开源实现及说明：

- [ObsPy 1.4.2 Beachball](https://github.com/obspy/obspy/blob/1.4.2/obspy/imaging/beachball.py)：输入参数与坐标约定。
- [ObsPy 1.4.2 MoPaD](https://github.com/obspy/obspy/blob/1.4.2/obspy/imaging/scripts/mopad.py)：独立矩张量计算参考。
- [USGS 机制球说明](https://www.usgs.gov/programs/earthquake-hazards/focal-mechanisms-or-beachballs)：下半球与初动填色解释。

ObsPy/MoPaD 使用 LGPL-3.0。本工具独立实现标准向量几何，未复制或分发其程序源码。测试使用已安装的 ObsPy 1.4.2 生成数值参考，不作为网站运行时依赖。

`generate-reference.py` 通过 MoPaD 的 `get_M()` 产生 NED 矩张量，再用 NumPy `eigh` 求独立参考 P/T 轴。使用矩阵特征分解是因为此环境中 MoPaD 的 `get_p_axis()` 在一维向量形状上会报错。固定随机种子为 2580，覆盖 10 个典型/边界机制和 100 个随机机制。主轴比较允许向量整体变号。

在仓库根目录执行：

```powershell
# 仅当需要重新生成参考数据时运行，要求该 Python 环境已有 ObsPy 1.4.2 与 NumPy。
python _playground/beachball/generate-reference.py

# 无第三方 Node 依赖的数学和投影检查。
node --test _playground/beachball/mechanism.test.cjs

# 浏览器检查需要可用的 playwright、pngjs 与 Chromium。
# 如果依赖不是当前项目的本地包，可将 NODE_PATH 指向它们所在的 node_modules。
# 如果 Chromium 不在 Playwright 默认位置，可设置 CHROMIUM_PATH 为浏览器可执行文件。
node _playground/beachball/browser.test.cjs
```

浏览器脚本临时启动仅监听 `127.0.0.1` 的静态服务，完成后关闭；不发布网站。

## 2026-10-28 检查记录

- 6 组数学测试通过：110 组 ObsPy 矩张量/P/T 轴对照、典型初动、投影往返与方位、超过 15,000 个区域采样、节面零振幅与连续性、非法输入。
- 10 个独立参考机制的 1,413 个浏览器实际绘制像素与 ObsPy 辐射符号一致；避开边界抗锯齿与探针标记。
- 1440、1000、768、720、390、320 px 宽度无横向溢出；人工查看桌面与手机截图。
- 验证示例按钮、数值框、滑块、键盘、无效值处理、点击探针、节面零极性、图层和黑白模式。
- SVG 导出可独立解析和显示，无栅格嵌图；PNG 为 1680 × 2040，含参数、投影、极性与探针说明。
- 主页面浏览器控制台无错误或警告；直接从本地文件打开也可绘图。
- 未找到可用的 Ruby/Jekyll 命令，未运行全站构建。新页面是原样发布的静态 HTML；入口的 front matter 与 Liquid 结构保持完整。
- 未提交或推送 Git，也未修改线上网站。
