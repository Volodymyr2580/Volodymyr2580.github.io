---
layout: page
permalink: /diary/index.html
title: Diary
---

# 科研日志
总算熬过了大三，该开始好好规划一下接下来6年左右的科研生活了。

准备在这里记录我的科研进展，类似每日记录，等6年后回过头来看这些会是什么想法呢哈哈！

开一个论文阅读的链接：
- [论文阅读](https://Volodymyr2580.github.io/diary/paper)

在做科研工作的时候，感觉程序会有很大的阻挠，但我通常都是遇到问题找ai解决，也不长记性，这导致了很多的重复劳动。所以，我觉得再开一个coding的链接，用于记录我在处理服务器端、linux系统、python程序和有关方库的使用方法和错误经历，这会是很有用的。

用一个想法勉励自己！为了解决高通货膨胀带来的恶性影响，政府需要下定决心，不能用铸币税来部分缓解眼前问题，而是要财政紧缩，经历痛苦的财政改革，才能根治问题！培养一个习惯、或者是改掉一个坏习惯，我想也应该是如此。

- [Coding](https://Volodymyr2580.github.io/diary/coding)

## 2025年

### 六月

#### 18日
今天刚刚是结束期末第二天，小放一天假。思考一下接下来的工作安排。

1. 安装一个虚拟机linux系统，或者用WSL子系统？每次都要连接服务器再使用devito有点麻烦。在linux系统上学习devito的使用方法，并用devito完成全波形反演的程序设计（还需要额外写一套嵌套进pytorch的nn.Function框架中的，先完成前者的基础版）。
2. 上次组会汇报时出现的若干问题未解决——一个是关于实际数据预处理的；另一个应该是用于验证先前一套程序的合理性的，但我现在也觉得很尴尬，感觉在程序这块周转浪费了太多时间，感觉很不舒服。先是gprMax可能浪费了一个月，然后又是GPR-FWI-Py又耗费一个多月，实在有点难顶，现在又想转移到devito上去，多少有点矫情。
3. 研读论文，全波形反演的相关研究看得还是太少，有关的实验设计和应该出现的结果自己心里没有概念。昨天在写ai4s的大作业报告的时候就感觉，很多论文被堆放在自己的文件夹中，然后可能当时读的时候印象比较深刻，等后续要写作的时候很有可能就会出现没有印象了，这部分会出现很多的重复劳动。所以，我在想能不能在我的主页这边做记录，每读一篇文章都记录下它的主要工作和结果。首先要给读的论文分档次，一个是强相关的，几乎是基于他的研究工作的，那需要细读、精读，要记录周全他的idea和experiment；一个是弱相关的，只是略读他的idea和主要工作，可以让ai辅助记录主要想法、主要工作和主要结果。另一类是出于兴趣阅读的，比如之前读到的有关SGD的深度学习原理解释有关的。然后我把他们的原始论文pdf也传到我的github仓库里，还可以添加超链接，检索方便。非常合理！

# 学习日志

与科研日志区分开的是学习日志。在这里会区别开我做的科研工作，我希望我能在完成科研任务的同时保持理论知识的学习，尤其是数学、物理和深度学习方向，也希望自己能在之后的生活中保证阅读量，所以也会记录一些读书的想法和读的书。

## 体重记录与可视化

<div id="weight-tracker" style="margin-top:1.5rem">
  <div class="weight-form" style="display:flex;gap:.5rem;align-items:center;flex-wrap:wrap">
    <label>日期：<input type="date" id="weight-date"></label>
    <label>体重(kg)：<input type="number" id="weight-value" step="0.1" min="30" max="200" placeholder="例如 67.5"></label>
    <button id="weight-add">添加记录</button>
    <button id="weight-clear" style="margin-left:.5rem">清空本地数据</button>
  </div>
  <div style="margin-top:1rem">
    <canvas id="weight-chart" height="140"></canvas>
  </div>
  <p style="font-size:12px;color:#666;margin-top:.5rem">说明：数据仅保存在当前浏览器的本地存储中；更换设备或清空浏览器数据会丢失。如需云端同步可后续升级。</p>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
(function(){
  const LS_KEY = 'diary_weight_records';
  function todayStr(){
    const d = new Date();
    const m = String(d.getMonth()+1).padStart(2,'0');
    const day = String(d.getDate()).padStart(2,'0');
    return `${d.getFullYear()}-${m}-${day}`;
  }
  function loadData(){
    try {
      const raw = localStorage.getItem(LS_KEY);
      const arr = raw ? JSON.parse(raw) : [];
      return arr.filter(r => r && r.date && r.weight !== undefined)
                .sort((a,b)=> a.date.localeCompare(b.date));
    } catch(e) { return []; }
  }
  function saveData(arr){ localStorage.setItem(LS_KEY, JSON.stringify(arr)); }
  let records = loadData();
  const dateEl = document.getElementById('weight-date');
  const valEl = document.getElementById('weight-value');
  dateEl.value = dateEl.value || todayStr();
  let chart;
  function render(){
    const labels = records.map(r=>r.date);
    const data = records.map(r=>+r.weight);
    const minW = data.length? Math.min(...data) : 50;
    const maxW = data.length? Math.max(...data) : 80;
    const yMin = Math.floor(minW-1);
    const yMax = Math.ceil(maxW+1);
    if(!chart){
      chart = new Chart(document.getElementById('weight-chart'), {
        type: 'line',
        data: { labels, datasets: [{ label: '体重(kg)', data, borderColor: '#2d6cdf', backgroundColor: 'rgba(45,108,223,0.15)', tension: 0.25, fill: true, pointRadius: 3 }] },
        options: { responsive: true, plugins:{legend:{display:true}}, scales: { y: { min: yMin, max: yMax } } }
      });
    } else {
      chart.data.labels = labels;
      chart.data.datasets[0].data = data;
      chart.options.scales.y.min = yMin;
      chart.options.scales.y.max = yMax;
      chart.update();
    }
  }
  render();
  document.getElementById('weight-add').addEventListener('click', function(){
    const date = dateEl.value || todayStr();
    const w = parseFloat(valEl.value);
    if(!date){ alert('请输入日期'); return; }
    if(isNaN(w)){ alert('请输入体重'); return; }
    if(w < 30 || w > 200){ alert('体重范围30~200kg'); return; }
    const idx = records.findIndex(r=>r.date === date);
    if(idx >= 0){ records[idx].weight = w; } else { records.push({date, weight: w}); }
    records.sort((a,b)=> a.date.localeCompare(b.date));
    saveData(records);
    render();
    valEl.value = '';
  });
  document.getElementById('weight-clear').addEventListener('click', function(){
    if(confirm('确定清空本地体重记录？此操作不可恢复。')){
      records = [];
      saveData(records);
      render();
    }
  });
})();
</script>


