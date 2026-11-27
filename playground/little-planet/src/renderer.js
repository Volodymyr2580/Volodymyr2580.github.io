const DIRECTIONS = { south: 0, west: 1, east: 2, north: 3 };
const LABELS = {
  zh: { radar: '雷达站', ice: '极地观测站', library: '林间图书馆', camp: '营地' },
  en: { radar: 'Radar station', ice: 'Ice observatory', library: 'Forest library', camp: 'Camp' },
};

const clamp = (value, minimum, maximum) => Math.max(minimum, Math.min(maximum, value));
const finite = (value, fallback = 0) => Number.isFinite(value) ? value : fallback;

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Unable to load island artwork: ${url}`));
    image.src = url;
  });
}

// Generated sprites vary slightly within their cells. Locate actual alpha
// bounds once, keeping the feet stable rather than stretching whole cells.
function readSpriteFrames(image) {
  const sheet = document.createElement('canvas');
  sheet.width = image.naturalWidth;
  sheet.height = image.naturalHeight;
  const context = sheet.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('Canvas 2D is not available.');
  context.drawImage(image, 0, 0);
  const pixels = context.getImageData(0, 0, sheet.width, sheet.height).data;
  const frames = [];
  for (let row = 0; row < 4; row += 1) {
    const poses = [];
    for (let column = 0; column < 4; column += 1) {
      const left = Math.floor(column * sheet.width / 4);
      const right = Math.floor((column + 1) * sheet.width / 4);
      const top = Math.floor(row * sheet.height / 4);
      const bottom = Math.floor((row + 1) * sheet.height / 4);
      let x0 = right; let y0 = bottom; let x1 = left; let y1 = top;
      for (let y = top; y < bottom; y += 1) {
        for (let x = left; x < right; x += 1) {
          if (pixels[(y * sheet.width + x) * 4 + 3] < 100) continue;
          x0 = Math.min(x0, x); x1 = Math.max(x1, x);
          y0 = Math.min(y0, y); y1 = Math.max(y1, y);
        }
      }
      if (x0 > x1 || y0 > y1) throw new Error('Explorer sprite has an empty animation frame.');
      poses.push({ x: x0, y: y0, width: x1 - x0 + 1, height: y1 - y0 + 1 });
    }
    frames.push(poses);
  }
  return frames;
}

function seededStars() {
  let seed = 91377;
  const random = () => {
    seed = (Math.imul(1664525, seed) + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  return Array.from({ length: 140 }, () => ({
    x: random(), y: random(), size: random() > .89 ? 2 : 1,
    phase: random() * Math.PI * 2, speed: .25 + random() * .4,
    color: random() > .91 ? '#d6ad78' : random() > .45 ? '#7bb7d2' : '#d0e3e4',
  }));
}

export async function createRenderer(canvas, { islandUrl, explorerUrl, world, placements, onProgress = () => {} }) {
  const context = canvas.getContext('2d', { alpha: false });
  if (!context) throw new Error('Canvas 2D is not available.');
  onProgress('正在铺好岛上的小径…');
  const [island, explorer] = await Promise.all([loadImage(islandUrl), loadImage(explorerUrl)]);
  onProgress('小探险家正在整理背包…');
  const frames = readSpriteFrames(explorer);
  const stars = seededStars();
  let width = 1;
  let height = 1;
  let pixelRatio = 1;
  let baseScale = 1;
  let mobile = false;
  let initialSize = true;
  let zoom = 1;
  let targetZoom = 1;
  let scale = 1;
  let originX = 0;
  let originY = 0;
  let centerX = world.width / 2;
  let centerY = world.height / 2;
  let focusX = 0;
  let focusY = 0;
  let cameraReady = false;
  let previousFrameTime = 0;
  let destroyed = false;

  function updateTransform() {
    scale = baseScale * zoom;
    originX = focusX - centerX * scale;
    originY = focusY - centerY * scale;
  }

  function resize() {
    if (destroyed) return;
    const rect = canvas.getBoundingClientRect();
    width = Math.max(1, rect.width);
    height = Math.max(1, rect.height);
    pixelRatio = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.round(width * pixelRatio);
    canvas.height = Math.round(height * pixelRatio);
    mobile = width <= 760;
    const landscape = height <= 560 && width >= 600;
    const top = landscape ? 92 : mobile ? 172 : 120;
    const bottom = landscape ? 58 : mobile ? 180 : 64;
    const margin = mobile ? 15 : 38;
    const availableHeight = Math.max(150, height - top - bottom);
    baseScale = Math.min((width - margin * 2) / world.width, availableHeight / world.height);
    focusX = width / 2;
    focusY = top + availableHeight * .5;
    if (initialSize) {
      targetZoom = mobile ? 1.85 : 1;
      zoom = targetZoom;
      initialSize = false;
    }
    updateTransform();
    context.imageSmoothingEnabled = false;
  }

  function toScreen(point) {
    return { x: originX + point.x * scale, y: originY + point.y * scale };
  }

  function screenToWorld(clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    return { x: (clientX - rect.left - originX) / scale, y: (clientY - rect.top - originY) / scale };
  }

  function drawStars(time) {
    context.fillStyle = '#101e30';
    context.fillRect(0, 0, width, height);
    const count = Math.max(48, Math.min(stars.length, Math.round(width * height / 7500)));
    for (let index = 0; index < count; index += 1) {
      const star = stars[index];
      const x = Math.round(star.x * width);
      const y = Math.round(star.y * height);
      context.globalAlpha = .25 + .3 * (.5 + .5 * Math.sin(time * star.speed + star.phase));
      context.fillStyle = star.color;
      context.fillRect(x, y, star.size, star.size);
      if (star.size > 1) {
        context.globalAlpha *= .5;
        context.fillRect(x - 3, y, 8, 2);
        context.fillRect(x, y - 3, 2, 8);
      }
    }
    context.globalAlpha = 1;
  }

  function drawRing(point, time, selected = false) {
    if (!point) return;
    const p = toScreen(point);
    const pulse = selected ? 1 : 1 + Math.sin(time * 3) * .08;
    context.strokeStyle = selected ? '#e3f3c5' : '#d8ebdd';
    context.fillStyle = 'rgba(229,244,202,.13)';
    context.lineWidth = 1.5;
    context.beginPath();
    context.ellipse(p.x, p.y + 1, 19 * scale * pulse, 9 * scale * pulse, 0, 0, Math.PI * 2);
    context.fill();
    context.stroke();
  }

  function drawRoute(state, time) {
    if (state.route?.length) {
      context.lineWidth = 1.5;
      context.strokeStyle = 'rgba(249,239,206,.75)';
      context.setLineDash([2, 6]);
      context.beginPath();
      const first = toScreen(state.player);
      context.moveTo(first.x, first.y);
      for (const waypoint of state.route) {
        const p = toScreen(waypoint);
        context.lineTo(p.x, p.y);
      }
      context.stroke();
      context.setLineDash([]);
    }
    if (state.target) drawRing(state.target, time, true);
    else if (state.nearby && placements[state.nearby]) drawRing(placements[state.nearby], time);
  }

  function drawSparks(time) {
    const fire = { x: world.width * .651, y: world.height * .638 };
    for (let index = 0; index < 7; index += 1) {
      const phase = (time * .3 + index / 7) % 1;
      const p = toScreen({ x: fire.x + Math.sin(index * 3.6 + phase * 2) * 12, y: fire.y - phase * 45 });
      context.globalAlpha = (1 - phase) * .65;
      context.fillStyle = index % 2 ? '#f6b964' : '#ffdd9e';
      context.fillRect(Math.round(p.x), Math.round(p.y), Math.max(1, scale * 2), Math.max(1, scale * 3));
    }
    context.globalAlpha = 1;
  }

  function drawRadarPulse(state) {
    if (state.pulse == null || !placements.radar) return;
    const elapsed = Math.max(0, finite(state.time) - state.pulse);
    if (elapsed > 3.2) return;
    const p = toScreen(placements.radar);
    context.lineWidth = 1.5;
    for (let index = 0; index < 3; index += 1) {
      const age = elapsed - index * .45;
      if (age < 0 || age > 2.2) continue;
      context.globalAlpha = (1 - age / 2.2) * .65;
      context.strokeStyle = '#a6e3e7';
      context.beginPath();
      context.ellipse(p.x, p.y, (15 + age * 95) * scale, (6 + age * 36) * scale, 0, 0, Math.PI * 2);
      context.stroke();
    }
    context.globalAlpha = 1;
  }

  function drawExplorer(state, time) {
    const row = DIRECTIONS[state.direction] ?? 0;
    const column = state.moving && !state.reducedMotion ? Math.floor(time * 8) % 4 : 0;
    const frame = frames[row][column];
    const p = toScreen(state.player);
    const spriteHeight = 70 * scale;
    const spriteWidth = spriteHeight * frame.width / frame.height;
    context.fillStyle = 'rgba(23,29,22,.35)';
    context.beginPath();
    context.ellipse(p.x, p.y + 1, 17 * scale, 6 * scale, 0, 0, Math.PI * 2);
    context.fill();
    context.imageSmoothingEnabled = false;
    context.drawImage(explorer, frame.x, frame.y, frame.width, frame.height,
      Math.round(p.x - spriteWidth / 2), Math.round(p.y - spriteHeight),
      Math.round(spriteWidth), Math.round(spriteHeight));
  }

  function drawLabels(state) {
    const names = LABELS[state.lang] || LABELS.zh;
    let index = 0;
    for (const [id, place] of Object.entries(placements)) {
      index += 1;
      const p = toScreen({ x: place.labelX, y: place.labelY });
      if (p.x < -60 || p.x > width + 60 || p.y < 0 || p.y > height + 25) continue;
      const selected = state.nearby === id;
      const visited = state.visited?.has(id);
      const label = names[id] || id;
      context.font = `${mobile ? 10 : 11}px "Microsoft YaHei", "Segoe UI", sans-serif`;
      const textWidth = context.measureText(label).width;
      const boxWidth = Math.ceil(textWidth + 30);
      const boxHeight = 25;
      const x = Math.round(p.x - boxWidth / 2);
      const y = Math.round(p.y - boxHeight / 2);
      context.fillStyle = selected ? 'rgba(30,54,66,.96)' : 'rgba(17,32,46,.87)';
      context.fillRect(x, y, boxWidth, boxHeight);
      context.strokeStyle = selected ? '#c0d9cb' : 'rgba(225,232,222,.6)';
      context.lineWidth = 1;
      context.strokeRect(x + .5, y + .5, boxWidth - 1, boxHeight - 1);
      context.fillStyle = visited ? '#adcd95' : '#a8c9d6';
      context.fillRect(x + 8, y + 10, 4, 4);
      context.fillStyle = '#f1ebd8';
      context.textBaseline = 'middle';
      context.fillText(label, x + 19, y + boxHeight / 2 + .5);
    }
  }

  function render(state) {
    if (destroyed || !state.player) return;
    const now = performance.now();
    const elapsed = previousFrameTime ? clamp((now - previousFrameTime) / 1000, 0, .08) : 1;
    previousFrameTime = now;
    const ease = state.reducedMotion ? 1 : 1 - Math.exp(-elapsed * 8);
    zoom += (targetZoom - zoom) * ease;
    const follow = clamp((zoom - 1) / .65, 0, 1);
    const targetX = world.width / 2 + (state.player.x - world.width / 2) * follow;
    const targetY = world.height / 2 + (state.player.y - 16 - world.height / 2) * follow;
    if (!cameraReady) {
      centerX = targetX;
      centerY = targetY;
      cameraReady = true;
    } else {
      centerX += (targetX - centerX) * ease;
      centerY += (targetY - centerY) * ease;
    }
    updateTransform();
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    const time = state.reducedMotion ? 0 : finite(state.time);
    drawStars(time);
    context.imageSmoothingEnabled = false;
    context.drawImage(island, originX, originY, world.width * scale, world.height * scale);
    drawSparks(time);
    drawRoute(state, time);
    drawRadarPulse(state);
    drawExplorer(state, time);
    drawLabels(state);
  }

  resize();
  const observer = typeof ResizeObserver === 'function' ? new ResizeObserver(resize) : null;
  observer?.observe(canvas);

  return {
    render,
    resize,
    screenToWorld,
    setZoom(value) { targetZoom = clamp(finite(value, targetZoom), .8, 2.5); },
    getZoom() { return targetZoom; },
    destroy() { destroyed = true; observer?.disconnect(); },
  };
}
