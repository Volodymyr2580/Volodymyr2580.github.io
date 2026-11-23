// An intentionally qualitative illustration, with normalized depth and time.
// This is not a physical forward solver or an inversion result.
export function createRadarDemo(container, { lang, onPing }) {
  const en = lang === 'en';
  container.innerHTML = `
    <h3>${en ? 'Listen beneath the surface' : '听见地下的回声'}</h3>
    <canvas class="experiment-canvas" width="900" height="420" aria-label="${en ? 'Schematic radar pulse and reflections' : '雷达发射与分层反射的概念示意'}"></canvas>
    <label>${en ? 'Move the buried interface' : '改变地下界面的位置'} <input type="range" min="30" max="78" value="54" aria-label="${en ? 'Relative interface depth' : '界面相对深度'}"></label>
    <button class="primary-button" type="button">${en ? 'Send a radar pulse' : '发射探测波'}</button>
    <p class="note">${en ? 'Qualitative illustration · depth and time are normalized; no physical units or simulated data.' : '定性概念示意 · 深度与时间为相对量，不是数值正演或实测数据。'}</p>`;
  const canvas = container.querySelector('canvas');
  const ctx = canvas.getContext('2d');
  const range = container.querySelector('input');
  const button = container.querySelector('button');
  let latestTime = 0;
  let started = null;
  let depth = .54;
  function send() { started = latestTime; onPing(); }
  function changeDepth() { depth = Number(range.value) / 100; started = null; }
  button.addEventListener('click', send);
  range.addEventListener('input', changeDepth);

  function render(time, reducedMotion) {
    latestTime = time;
    if (!ctx) return;
    const t = started === null ? -1 : Math.min((time - started) / 3, 1);
    const elapsed = reducedMotion && t >= 0 ? 1 : t;
    const layerY = 100 + depth * 270;
    ctx.fillStyle = '#101e30'; ctx.fillRect(0, 0, 900, 420);
    ctx.fillStyle = '#293d49'; ctx.fillRect(40, 100, 380, 275);
    ctx.fillStyle = '#4c6562'; ctx.fillRect(40, layerY, 380, 375 - layerY);
    ctx.fillStyle = '#6e8772'; ctx.fillRect(40, 100, 380, 5);
    ctx.strokeStyle = '#9aba8a'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(40, layerY); ctx.lineTo(420, layerY); ctx.stroke();
    ctx.fillStyle = '#f3eddc'; ctx.font = '20px sans-serif';
    ctx.fillText(en ? 'TRANSMIT / RECEIVE' : '发射 / 接收', 42, 43);
    ctx.fillText(en ? 'ECHO TRACE' : '回波示意', 500, 43);
    ctx.fillStyle = '#acddec'; ctx.fillRect(215, 70, 30, 30);
    ctx.fillStyle = '#b4c9d3'; ctx.font = '18px sans-serif';
    ctx.fillText(en ? 'surface' : '地表', 55, 92);
    ctx.fillText(en ? 'interface' : '地下界面', 55, layerY + 29);
    ctx.fillText(en ? 'relative time →' : '相对时间 →', 675, 365);

    if (elapsed >= 0) {
      const phase = Math.min(elapsed * 2, 1);
      const pulseY = elapsed < .5 ? 100 + (layerY - 100) * phase : layerY - (layerY - 100) * (elapsed * 2 - 1);
      ctx.strokeStyle = elapsed < .5 ? '#acddec' : '#f1be85';
      ctx.lineWidth = 4;
      ctx.beginPath(); ctx.moveTo(177, pulseY); ctx.lineTo(283, pulseY); ctx.stroke();
      ctx.setLineDash([6, 10]); ctx.lineWidth = 2; ctx.globalAlpha = .45;
      ctx.beginPath(); ctx.moveTo(230, 108); ctx.lineTo(230, layerY); ctx.stroke();
      ctx.setLineDash([]); ctx.globalAlpha = 1;
    }
    ctx.strokeStyle = '#426071'; ctx.lineWidth = 1;
    for (let x = 500; x <= 855; x += 71) { ctx.beginPath(); ctx.moveTo(x, 90); ctx.lineTo(x, 330); ctx.stroke(); }
    ctx.beginPath(); ctx.moveTo(495, 210); ctx.lineTo(855, 210); ctx.stroke();
    ctx.strokeStyle = '#acddec'; ctx.lineWidth = 3; ctx.beginPath();
    for (let i = 0; i <= 355; i++) {
      const p = i / 355;
      const wave = (x, width) => (1 - 2 * (x / width) ** 2) * Math.exp(-((x / width) ** 2));
      const signal = elapsed < 0 || p > elapsed ? 0 : 68 * wave(p - .1, .027) + 44 * wave(p - (.25 + depth * .7), .037);
      if (i === 0) ctx.moveTo(500, 210 - signal); else ctx.lineTo(500 + i, 210 - signal);
    }
    ctx.stroke();
    const isRunning = t >= 0 && t < 1 && !reducedMotion;
    button.disabled = isRunning;
    button.textContent = isRunning ? (en ? 'Listening for echoes…' : '正在聆听回声…') : (en ? 'Send a radar pulse' : '发射探测波');
  }
  return { render, destroy() { button.removeEventListener('click', send); range.removeEventListener('input', changeDepth); } };
}
