/* UI and self-contained SVG rendering. All calculations stay in the browser. */
(function () {
  'use strict';
  const B = window.Beachball;
  const $ = id => document.getElementById(id);
  const keys = ['strike', 'dip', 'rake'];
  const C = 280, R = 215;
  const palette = { ink: '#0b2242', blue: '#245d88', muted: '#596e85', white: '#ffffff', gold: '#a75021', black: '#1a1a1a' };
  let model = B.mechanism(0, 45, -90);
  let probe = {azimuth: 45, takeoff: 60};
  const format = (n, digits = 1) => (Math.abs(n) < .5 * 10**-digits ? 0 : n).toFixed(digits);
  const angle = n => `${format(n)}°`;
  const xy = v => B.project(v).map(x => C + R*x);
  const path = (points, closed = false) => points.length ? points.map((p, i) => `${i ? 'L' : 'M'}${format(C+R*p[0], 3)},${format(C+R*p[1], 3)}`).join(' ') + (closed ? ' Z' : '') : '';
  const text = (x, y, value, extra = '') => `<text x="${format(x, 3)}" y="${format(y, 3)}" ${extra}>${value}</text>`;

  function axis(v, name, fill) {
    const [x, y] = xy(B.lower(v));
    return `<g data-axis="${name}" transform="translate(${x},${y})"><circle r="14" fill="${name === 'P' ? palette.white : fill}" stroke="${palette.white}" stroke-width="5"/><circle r="14" fill="none" stroke="${fill}" stroke-width="1.7"/><text y="5" text-anchor="middle" fill="${name === 'P' ? fill : palette.white}" font-size="15" font-weight="700">${name}</text></g>`;
  }

  function draw() {
    const fill = $('monochrome').checked ? palette.black : palette.blue;
    document.documentElement.style.setProperty('--fill', fill);
    const np = B.halfDisk(model.n), sp = B.halfDisk(model.s);
    const nn = B.halfDisk(model.n.map(x => -x)), sn = B.halfDisk(model.s.map(x => -x));
    const r = B.direction(probe.azimuth, probe.takeoff);
    const a = B.amplitude(model, r), sign = B.polarity(a);
    const symbol = sign > 0 ? '+' : sign < 0 ? '−' : '0';
    const label = sign > 0 ? '压缩初动 · 向外' : sign < 0 ? '膨胀初动 · 向内' : '节面方向 · 零初动';
    let guides = '';
    for (let degrees = 0; degrees < 360; degrees += 10) {
      const t = degrees*Math.PI/180, major = degrees%30 === 0;
      guides += `<path d="M${C+(R+5)*Math.sin(t)},${C-(R+5)*Math.cos(t)} L${C+(R+(major?12:8))*Math.sin(t)},${C-(R+(major?12:8))*Math.cos(t)}" stroke="${palette.muted}" stroke-width="${major?1:.5}"/>`;
    }
    let nodes = '';
    if ($('show-planes').checked) {
      for (const [i, plane] of [np, sp].entries()) {
        const dash = i ? 'stroke-dasharray="7 5"' : '';
        nodes += `<path d="${path(plane.node)}" fill="none" stroke="${palette.white}" stroke-width="3.4" ${dash}/><path data-plane="${i+1}" d="${path(plane.node)}" fill="none" stroke="${palette.ink}" stroke-width="1.4" ${dash}/>`;
      }
    }
    let samples = '';
    if ($('show-polarities').checked) {
      for (let az = 22.5; az < 360; az += 45) {
        const ray = B.direction(az, 58), amplitude = B.amplitude(model, ray);
        if (Math.abs(amplitude) < .06) continue;
        const [x, y] = xy(ray);
        if ($('show-axes').checked && [model.p, model.t].some(v => {
          const [ax, ay] = xy(B.lower(v)); return Math.hypot(x-ax, y-ay) < 32;
        })) continue;
        samples += text(x, y+6, amplitude > 0 ? '+' : '−', `text-anchor="middle" font-size="21" fill="${amplitude > 0 ? palette.white : palette.ink}"`);
      }
    }
    const [px, py] = xy(r);
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 560 560" width="560" height="560" role="img" aria-labelledby="beachball-title beachball-description" style="font-family:'Segoe UI','Microsoft YaHei',sans-serif">
      <title id="beachball-title">震源机制球：走向 ${angle(model.strike)}，倾角 ${angle(model.dip)}，滑动角 ${angle(model.rake)}</title>
      <desc id="beachball-description">下半球等面积投影。深色压缩初动，白色膨胀初动。P 为压缩轴，T 为张性轴。金色探针：${label}。</desc>
      <defs><clipPath id="positive-normal"><path d="${path(np.region, true)}"/></clipPath><clipPath id="negative-normal"><path d="${path(nn.region, true)}"/></clipPath><clipPath id="sphere"><circle cx="${C}" cy="${C}" r="${R}"/></clipPath></defs>
      <rect width="560" height="560" fill="${palette.white}"/>
      <g fill="${palette.ink}" font-size="15" text-anchor="middle">${text(C, 35, 'N')}${text(530, C+5, 'E')}${text(C, 537, 'S')}${text(30, C+5, 'W')}</g>
      ${guides}<circle cx="${C}" cy="${C}" r="${R}" fill="${palette.white}"/>
      <g clip-path="url(#sphere)"><path d="${path(sp.region, true)}" fill="${fill}" clip-path="url(#positive-normal)"/><path d="${path(sn.region, true)}" fill="${fill}" clip-path="url(#negative-normal)"/>${nodes}${samples}</g>
      <circle cx="${C}" cy="${C}" r="${R}" fill="none" stroke="${palette.ink}" stroke-width="1.7"/>
      ${$('show-axes').checked ? axis(model.p, 'P', fill)+axis(model.t, 'T', fill) : ''}
      <g data-probe="true" transform="translate(${px},${py})"><circle r="6.5" fill="${palette.gold}" stroke="${palette.white}" stroke-width="2"/><circle r="10" fill="none" stroke="${palette.gold}" stroke-width="1.5"/></g>
      </svg>`;
    $('plot').innerHTML = svg;
    for (const name of ['p', 't']) {
      const o = B.orientation(model[name]);
      $(''+name+'-readout').textContent = `方位 ${o.azimuth === null ? '未定义' : angle(o.azimuth)} / 俯角 ${angle(o.plunge)}`;
    }
    $('parameter-summary').textContent = `φ ${angle(model.strike)} / δ ${angle(model.dip)} / λ ${angle(model.rake)}`;
    $('polarity-symbol').textContent = symbol;
    $('polarity-label').textContent = label;
    $('amplitude-readout').textContent = `A = ${a >= 0 ? '+' : ''}${format(a, 4)}`;
    document.querySelectorAll('[data-preset]').forEach(button => {
      button.setAttribute('aria-pressed', String(button.dataset.preset.split(',').every((v, i) => Number(v) === model[keys[i]])));
    });
  }

  function validate(ids, errorId, message) {
    const valid = ids.map(id => {
      const el = $(id), okay = el.value.trim() !== '' && Number.isFinite(el.valueAsNumber) && el.checkValidity();
      el.setAttribute('aria-invalid', String(!okay)); return okay;
    }).every(Boolean);
    $(errorId).hidden = valid;
    $(errorId).textContent = valid ? '' : message;
    $('export-svg').disabled = $('export-png').disabled = !$('input-error').hidden || !$('probe-error').hidden;
    return valid;
  }

  function updateParameters() {
    if (!validate(keys, 'input-error', '请输入有效角度：走向 0–360°，倾角 0–90°，滑动角 −180–180°，最多一位小数。图中保留上次有效结果。')) return;
    model = B.mechanism(...keys.map(k => $(k).valueAsNumber));
    keys.forEach(k => { $(k+'-range').value = $(k).value; });
    draw();
  }

  keys.forEach(k => {
    $(k).addEventListener('input', updateParameters);
    $(k+'-range').addEventListener('input', () => { $(k).value = $(k+'-range').value; updateParameters(); });
  });
  document.querySelectorAll('[data-preset]').forEach(button => button.addEventListener('click', () => {
    button.dataset.preset.split(',').forEach((v, i) => { $(keys[i]).value = v; });
    updateParameters();
  }));
  ['show-axes', 'show-planes', 'show-polarities', 'monochrome'].forEach(id => $(id).addEventListener('change', draw));
  ['azimuth', 'takeoff'].forEach(id => $(id).addEventListener('input', () => {
    if (!validate(['azimuth', 'takeoff'], 'probe-error', '方位角须为 0–360°，出射角须为 0–90°，最多一位小数；保留上次有效探针。')) return;
    probe = {azimuth: $('azimuth').valueAsNumber, takeoff: $('takeoff').valueAsNumber}; draw();
  }));
  $('plot').addEventListener('click', event => {
    const svg = $('plot').querySelector('svg');
    const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(svg.getScreenCTM().inverse());
    const ray = B.unproject((point.x-C)/R, (point.y-C)/R);
    if (!ray) return;
    const azimuth = Math.hypot(ray[0], ray[1]) < 1e-10 ? 0 : (Math.atan2(ray[1], ray[0])*180/Math.PI+360)%360;
    probe = {azimuth: Number(format(azimuth)), takeoff: Number(format(Math.acos(Math.max(0, Math.min(1, ray[2])))*180/Math.PI))};
    $('azimuth').value = probe.azimuth; $('takeoff').value = probe.takeoff;
    validate(['azimuth', 'takeoff'], 'probe-error', ''); draw();
  });

  function exportMarkup() {
    const copy = $('plot').querySelector('svg').cloneNode(true);
    copy.setAttribute('viewBox', '0 0 560 680');
    copy.setAttribute('width', '1120'); copy.setAttribute('height', '1360');
    copy.querySelector('rect').setAttribute('height', '680');
    copy.insertAdjacentHTML('beforeend', `<g text-anchor="middle" fill="${palette.ink}" font-size="13">
      ${text(C, 568, `Strike ${angle(model.strike)} / Dip ${angle(model.dip)} / Rake ${angle(model.rake)}`)}
      ${text(C, 592, 'Lower hemisphere · Equal area · Double couple')}
      ${text(C, 616, 'Filled: compression (+) / White: dilatation (−)')}
      ${text(C, 640, `Probe: azimuth ${angle(probe.azimuth)}, takeoff ${angle(probe.takeoff)}`)}
      ${text(C, 662, `A = ${format(B.amplitude(model, B.direction(probe.azimuth, probe.takeoff)), 4)} · Takeoff measured from down`)}
      </g>`);
    return new XMLSerializer().serializeToString(copy);
  }
  function download(blob, suffix) {
    const url = URL.createObjectURL(blob), link = document.createElement('a');
    link.href = url; link.download = `beachball-${model.strike}-${model.dip}-${model.rake}.${suffix}`;
    document.body.appendChild(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1500);
  }
  $('export-svg').addEventListener('click', () => {
    download(new Blob([exportMarkup()], {type: 'image/svg+xml;charset=utf-8'}), 'svg');
    $('export-status').textContent = 'SVG 已生成。';
  });
  $('export-png').addEventListener('click', async () => {
    const button = $('export-png'); button.disabled = true;
    const url = URL.createObjectURL(new Blob([exportMarkup()], {type: 'image/svg+xml;charset=utf-8'}));
    try {
      const image = new Image();
      await new Promise((resolve, reject) => { image.onload = resolve; image.onerror = reject; image.src = url; });
      const canvas = document.createElement('canvas'); canvas.width = 1680; canvas.height = 2040;
      canvas.getContext('2d').drawImage(image, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
      if (!blob) throw new Error('PNG encoding failed');
      download(blob, 'png'); $('export-status').textContent = 'PNG 已生成。';
    } catch (error) {
      $('export-status').classList.remove('sr-only');
      $('export-status').textContent = 'PNG 导出失败，请使用 SVG 下载。';
    } finally {
      URL.revokeObjectURL(url); button.disabled = !$('input-error').hidden || !$('probe-error').hidden;
    }
  });
  draw();
})();
