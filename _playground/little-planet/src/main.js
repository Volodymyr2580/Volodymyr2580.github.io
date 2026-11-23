import { PLACES, TEXT } from './content.js';
import { WORLD, SPAWN, PLACEMENTS, WALKABLE } from './world-map.js';
import { createNavigation } from './navigation.js';
import { createRenderer } from './renderer.js';
import { createRadarDemo } from './radar.js';

const $ = (id) => document.getElementById(id);
const canvas = $('world');
const dialog = $('info-dialog');
const nav = createNavigation({ ...WORLD, walkablePolygons: WALKABLE, cellSize: 10 });
const reducedPreference = window.matchMedia('(prefers-reduced-motion: reduce)');
const placeIds = new Set(PLACES.map((place) => place.id));
const storageKey = 'volodymyr-planet-v1';
const asset = (name) => new URL(`../assets/${name}`, import.meta.url).href;
// The lightweight preview server does not compile the main Jekyll site.
// Keep its content links useful by opening the public site from local previews.
const isLocalPreview = ['localhost', '127.0.0.1', '[::1]'].includes(location.hostname);
const siteLink = (href) => isLocalPreview && href.startsWith('/') ? new URL(href, 'https://volodymyr2580.github.io').href : href;
const escapeHTML = (value) => String(value).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
let stored = {};
try { stored = JSON.parse(localStorage.getItem(storageKey)) || {}; } catch { /* Storage is optional. */ }
const state = {
  player: { ...SPAWN }, moving: false, direction: 'south', time: 0,
  reducedMotion: reducedPreference.matches, visited: new Set(Array.isArray(stored.visited) ? stored.visited.filter((id) => placeIds.has(id)) : []),
  nearby: null, lang: stored.lang === 'en' ? 'en' : 'zh', route: [], target: null, pulse: null,
};
let renderer = null;
let pendingPlace = null;
let dialogPlace = null;
let radar = null;
let lastFrame = performance.now();
let animation = 0;
let toastTimer = 0;
let loadingAttempt = 0;
let initialized = false;
let soundEnabled = false;
let soundContext = null;
let lastFootstep = 0;
const keys = new Set();
const stick = { x: 0, y: 0, pointerId: null };
const words = () => TEXT[state.lang];
const choose = (zh, en) => state.lang === 'en' ? en : zh;
const placeById = (id) => PLACES.find((place) => place.id === id);

function save() {
  try { localStorage.setItem(storageKey, JSON.stringify({ visited: [...state.visited], lang: state.lang })); } catch { /* Private browsing may block storage. */ }
}
function toast(message) {
  clearTimeout(toastTimer); $('toast').textContent = message; $('toast').hidden = false;
  toastTimer = setTimeout(() => { $('toast').hidden = true; }, 3500);
}
function clearMovement() {
  keys.clear(); stick.x = stick.y = 0; stick.pointerId = null;
  $('joystick-knob').style.transform = ''; $('joystick').classList.remove('active');
  state.moving = false;
}
function stopRoute() { state.route = []; state.target = null; pendingPlace = null; }
function updateProgress() {
  $('progress-label').textContent = choose(`已发现 ${state.visited.size} / 4 个地点`, `${state.visited.size} / 4 places discovered`);
}
function setLabel(id, label) {
  const element = $(id); element.setAttribute('aria-label', label); element.title = label;
}
function translate() {
  document.documentElement.lang = state.lang === 'en' ? 'en' : 'zh-CN';
  document.querySelector('.brand-subtitle').textContent = choose('一座小岛，一些好奇心', 'A LITTLE ISLAND. A LITTLE CURIOSITY.');
  document.querySelector('.back-link span').textContent = words().back;
  document.querySelector('.back-link').setAttribute('aria-label', words().back);
  document.querySelector('.back-link').href = siteLink('/playground/');
  $('map-button').querySelector('span').textContent = choose('地图', 'Map');
  $('directory-button').querySelector('span').textContent = words().places;
  $('language-button').textContent = state.lang === 'en' ? '中' : 'EN';
  setLabel('language-button', state.lang === 'en' ? '切换到中文' : 'Switch to English');
  $('reset-button').querySelector('span').textContent = words().resetPosition;
  $('interact-button').querySelector('span').textContent = words().interactButton;
  $('location-hint').querySelector('.eyebrow').textContent = choose('停下来，发现一点什么', 'A LITTLE DISCOVERY');
  $('map-panel').querySelector('h2').textContent = choose('岛上有什么', 'Around the island');
  document.querySelector('.map-description').textContent = choose('选一个地点，沿着小径走过去。', 'Choose a place. Follow the path.');
  const guides = document.querySelector('.movement-guide').children;
  guides[1].textContent = choose('移动', 'move'); guides[3].textContent = choose('点击小径，也能出发', 'or click a path to walk');
  setLabel('close-dialog', choose('关闭内容，继续探索', 'Close and keep exploring'));
  setLabel('map-close', choose('关闭地图', 'Close map'));
  setLabel('zoom-in', choose('放大视野', 'Zoom in')); setLabel('zoom-out', choose('缩小视野', 'Zoom out'));
  setLabel('sound-button', soundEnabled ? words().soundOff : words().soundOn);
  setLabel('motion-button', state.reducedMotion ? choose('恢复环境动画', 'Resume ambient animation') : choose('暂停环境动画', 'Pause ambient animation'));
  $('motion-button').setAttribute('aria-pressed', String(state.reducedMotion));
  canvas.setAttribute('aria-label', choose('像素探索小岛。WASD 或方向键移动，E 或空格互动。也可通过地点图鉴阅读全部内容。', 'Pixel exploration island. WASD or arrow keys to move, E or Space to interact. All content is also available in the field guide.'));
  $('map-panel').setAttribute('aria-label', choose('小岛地图', 'Island map'));
  dialog.setAttribute('aria-label', words().places);
  $('joystick').setAttribute('aria-label', words().touchMove);
  $('retry-button').textContent = choose('再试一次', 'Try again');
  updateProgress(); updateNearby(true); renderMap();
}

function updateNearby(force = false) {
  let nearest = null;
  let distance = Infinity;
  for (const [id, p] of Object.entries(PLACEMENTS)) {
    const d = Math.hypot(state.player.x - p.x, state.player.y - p.y);
    if (d < p.radius && d < distance) { distance = d; nearest = id; }
  }
  if (nearest === state.nearby && !force) return;
  state.nearby = nearest;
  $('location-hint').hidden = !nearest || dialog.open;
  if (nearest) {
    const content = placeById(nearest).content[state.lang];
    $('location-title').textContent = content.title;
    $('location-description').textContent = content.summary;
  }
}

function renderMap() {
  const entries = PLACES.map((place, i) => {
    const c = place.content[state.lang]; const visited = state.visited.has(place.id);
    return `<button type="button" data-go="${place.id}" data-visited="${visited}"><span class="location-number">0${i + 1}</span><span>${escapeHTML(c.title)}<small>${visited ? words().visited : words().unvisited}</small></span><span aria-hidden="true" class="map-arrow">→</span></button>`;
  }).join('');
  $('map-destinations').innerHTML = `<div class="mini-world" aria-hidden="true"><img src="${asset('island.png')}" alt=""/><i id="mini-player"></i>${PLACES.map((place, i) => `<b style="left:${PLACEMENTS[place.id].x / WORLD.width * 100}%;top:${PLACEMENTS[place.id].y / WORLD.height * 100}%">${i + 1}</b>`).join('')}</div>${entries}`;
  $('map-destinations').querySelectorAll('[data-go]').forEach((button) => button.addEventListener('click', () => goToPlace(button.dataset.go)));
}
function toggleMap(show = $('map-panel').hidden) {
  $('map-panel').hidden = !show; $('map-button').setAttribute('aria-expanded', String(show));
  if (show) { renderMap(); $('map-close').focus(); } else canvas.focus({ preventScroll: true });
}

function openDialog(id = null, discover = false) {
  clearMovement(); stopRoute(); toggleMap(false);
  radar?.destroy(); radar = null; dialogPlace = id;
  if (discover && id && !state.visited.has(id)) {
    state.visited.add(id); save(); updateProgress(); chime('discovery');
    if (state.visited.size === 4) toast(words().completed);
  }
  const content = $('dialog-content');
  if (!id) {
    content.innerHTML = `<span class="eyebrow">VOLODYMYR'S PLANET / FIELD GUIDE</span><h2>${words().places}</h2><p class="place-summary">${choose('每条小径，都通往一点好奇心。你可以走过去，也可以在这里直接阅读。', 'Every path leads to a little curiosity. Walk there, or read right here.')}</p>` + PLACES.map((place, i) => {
      const c = place.content[state.lang];
      return `<article class="journal-entry"><span class="eyebrow">0${i + 1} / ${state.visited.has(place.id) ? words().visited : words().unvisited}</span><h3>${escapeHTML(c.title)}</h3><p>${escapeHTML(c.summary)}</p><div class="journal-actions"><button type="button" class="secondary-button" data-read="${place.id}">${choose('直接阅读', 'Read here')}</button><button type="button" class="visit-button" data-go="${place.id}" ${!renderer ? 'disabled' : ''}>${choose('走过去', 'Walk there')} →</button></div></article>`;
    }).join('');
    content.querySelectorAll('[data-read]').forEach((button) => button.addEventListener('click', () => openDialog(button.dataset.read)));
    content.querySelectorAll('[data-go]').forEach((button) => button.addEventListener('click', () => { dialog.close(); goToPlace(button.dataset.go); }));
  } else {
    const c = placeById(id).content[state.lang];
    content.innerHTML = `<button class="guide-back" type="button">← ${words().places}</button><span class="eyebrow">${escapeHTML(c.kicker)}</span><h2>${escapeHTML(c.title)}</h2><p class="place-summary">${escapeHTML(c.summary)}</p>${c.body.split('\n\n').map((p) => `<p>${escapeHTML(p)}</p>`).join('')}<div class="place-links">${c.links.map((link) => `<a class="place-link" href="${escapeHTML(siteLink(link.href))}" ${link.href.startsWith('mailto:') ? '' : 'target="_blank" rel="noopener"'}><span>${escapeHTML(link.label)}</span><span aria-hidden="true">↗</span></a>`).join('')}</div>${id === 'radar' ? '<section class="experiment"></section>' : ''}<button class="secondary-button keep-exploring" type="button">${words().close} →</button>`;
    content.querySelector('.guide-back').addEventListener('click', () => openDialog());
    content.querySelector('.keep-exploring').addEventListener('click', () => dialog.close());
    if (id === 'radar') radar = createRadarDemo(content.querySelector('.experiment'), { lang: state.lang, onPing: () => { state.pulse = state.time; chime('radar'); } });
  }
  if (!dialog.open) dialog.showModal();
  dialog.scrollTop = 0; $('close-dialog').focus(); $('location-hint').hidden = true;
}

function walkTo(point, destination = null) {
  if (!renderer) return;
  const path = nav.findPath(state.player, point);
  if (path.length < 2) {
    if (destination && state.nearby === destination) openDialog(destination, true);
    else toast(choose('沿着浅色小径走吧。', 'Follow the pale paths.'));
    return;
  }
  state.route = path.slice(1); state.target = { ...path.at(-1) }; pendingPlace = destination;
}
function goToPlace(id) {
  toggleMap(false);
  if (!renderer) { openDialog(id); return; }
  if (state.nearby === id) { openDialog(id, true); return; }
  walkTo(PLACEMENTS[id], id);
  toast(choose(`前往${placeById(id).content.zh.title}…`, `Walking to ${placeById(id).content.en.title}…`));
}
function interact() { if (state.nearby) openDialog(state.nearby, true); }

function update(dt) {
  if (dialog.open || !renderer) { state.moving = false; return; }
  let dx = (keys.has('KeyD') || keys.has('ArrowRight') ? 1 : 0) - (keys.has('KeyA') || keys.has('ArrowLeft') ? 1 : 0) + stick.x;
  let dy = (keys.has('KeyS') || keys.has('ArrowDown') ? 1 : 0) - (keys.has('KeyW') || keys.has('ArrowUp') ? 1 : 0) + stick.y;
  const manual = Math.hypot(dx, dy) > .08;
  const speed = 150;
  const before = { ...state.player };
  if (manual) { stopRoute(); const n = Math.max(1, Math.hypot(dx, dy)); dx = dx / n * speed * dt; dy = dy / n * speed * dt; }
  else if (state.route.length) {
    let next = state.route[0];
    // Reach the exact validated corner before taking its next segment.
    // Skipping even a few pixels early can cut outside a narrow curved path.
    while (next && Math.hypot(next.x - state.player.x, next.y - state.player.y) < 1e-5) { state.route.shift(); next = state.route[0]; }
    if (next) { const distance = Math.hypot(next.x - state.player.x, next.y - state.player.y); const step = Math.min(speed * dt, distance); dx = (next.x - state.player.x) / distance * step; dy = (next.y - state.player.y) / distance * step; }
    else { dx = dy = 0; }
  } else { dx = dy = 0; }
  if (dx || dy) state.player = nav.move(state.player, { x: dx, y: dy });
  const movedX = state.player.x - before.x; const movedY = state.player.y - before.y;
  state.moving = Math.hypot(movedX, movedY) > .02;
  if (state.moving) {
    state.direction = Math.abs(movedX) > Math.abs(movedY) ? (movedX < 0 ? 'west' : 'east') : (movedY < 0 ? 'north' : 'south');
    if (state.time - lastFootstep > .3) { chime('step'); lastFootstep = state.time; }
  }
  updateNearby();
  if (!state.route.length && state.target) {
    const arrival = pendingPlace; state.target = null; pendingPlace = null;
    if (arrival && state.nearby === arrival) openDialog(arrival, true);
  }
}

function chime(kind) {
  if (!soundEnabled || !soundContext || soundContext.state !== 'running') return;
  const tones = kind === 'discovery' ? [523, 659, 784] : kind === 'radar' ? [784, 1046] : [110];
  tones.forEach((frequency, i) => {
    const oscillator = soundContext.createOscillator(); const gain = soundContext.createGain();
    const at = soundContext.currentTime + i * .09;
    oscillator.type = kind === 'step' ? 'triangle' : 'sine'; oscillator.frequency.value = frequency;
    gain.gain.setValueAtTime(0, at); gain.gain.linearRampToValueAtTime(kind === 'step' ? .009 : .035, at + .015); gain.gain.exponentialRampToValueAtTime(.0001, at + .2);
    oscillator.connect(gain); gain.connect(soundContext.destination); oscillator.start(at); oscillator.stop(at + .22);
    oscillator.onended = () => { oscillator.disconnect(); gain.disconnect(); };
  });
}
async function toggleSound() {
  if (soundEnabled) { soundEnabled = false; } else {
    try { const Context = window.AudioContext || window.webkitAudioContext; if (!Context) throw new Error('Audio unavailable'); soundContext ||= new Context(); await soundContext.resume(); soundEnabled = true; chime('discovery'); }
    catch { toast(choose('当前浏览器无法播放声音，仍可继续探索。', 'Sound is unavailable. You can keep exploring.')); }
  }
  $('sound-button').setAttribute('aria-pressed', String(soundEnabled)); translate();
}

const movementKeys = new Set(['KeyW', 'KeyA', 'KeyS', 'KeyD', 'ArrowUp', 'ArrowLeft', 'ArrowDown', 'ArrowRight']);
window.addEventListener('keydown', (event) => {
  if (dialog.open || /INPUT|TEXTAREA|SELECT/.test(event.target.tagName) || event.ctrlKey || event.metaKey || event.altKey) return;
  if (movementKeys.has(event.code)) { event.preventDefault(); keys.add(event.code); }
  else if (event.code === 'KeyE' || (event.code === 'Space' && (event.target === canvas || event.target === document.body))) { event.preventDefault(); if (!event.repeat) interact(); }
  else if (event.code === 'KeyM' && !event.repeat) { event.preventDefault(); toggleMap(); }
  else if (event.code === 'Escape' && !$('map-panel').hidden) toggleMap(false);
});
window.addEventListener('keyup', (event) => keys.delete(event.code));
window.addEventListener('blur', () => { clearMovement(); stopRoute(); });
document.addEventListener('visibilitychange', () => {
  if (document.hidden) { clearMovement(); stopRoute(); if (soundContext?.state === 'running') soundContext.suspend().catch(() => {}); }
  else { lastFrame = performance.now(); if (soundEnabled && soundContext) soundContext.resume().catch(() => {}); }
});

canvas.addEventListener('pointerdown', (event) => {
  if (event.button !== 0 || !renderer || dialog.open) return;
  canvas.focus({ preventScroll: true });
  if (!$('map-panel').hidden) toggleMap(false);
  const point = renderer.screenToWorld(event.clientX, event.clientY);
  const target = Object.entries(PLACEMENTS).find(([, p]) => Math.hypot(point.x - p.labelX, point.y - p.labelY) < 65 || Math.hypot(point.x - p.x, point.y - p.y) < 35);
  if (target) goToPlace(target[0]); else walkTo(point);
});
canvas.addEventListener('wheel', (event) => { event.preventDefault(); if (renderer) renderer.setZoom(renderer.getZoom() + (event.deltaY < 0 ? .12 : -.12)); }, { passive: false });

function moveStick(event) {
  const rect = $('joystick').getBoundingClientRect(); const limit = rect.width * .3;
  const dx = event.clientX - rect.left - rect.width / 2; const dy = event.clientY - rect.top - rect.height / 2;
  const distance = Math.hypot(dx, dy); const factor = distance > limit ? limit / distance : 1;
  stick.x = dx * factor / limit; stick.y = dy * factor / limit;
  $('joystick-knob').style.transform = `translate(${dx * factor}px, ${dy * factor}px)`;
}
$('joystick').addEventListener('pointerdown', (event) => {
  if (stick.pointerId !== null || dialog.open) return;
  event.preventDefault(); stick.pointerId = event.pointerId; $('joystick').setPointerCapture(event.pointerId); $('joystick').classList.add('active'); moveStick(event);
});
$('joystick').addEventListener('pointermove', (event) => { if (event.pointerId === stick.pointerId) moveStick(event); });
for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) $('joystick').addEventListener(name, (event) => { if (event.pointerId === stick.pointerId) clearMovement(); });

$('directory-button').addEventListener('click', () => openDialog());
$('interact-button').addEventListener('click', interact);
$('map-button').addEventListener('click', () => toggleMap());
$('map-close').addEventListener('click', () => toggleMap(false));
$('close-dialog').addEventListener('click', () => dialog.close());
dialog.addEventListener('close', () => { if (dialog.open) return; radar?.destroy(); radar = null; dialogPlace = null; clearMovement(); updateNearby(true); canvas.focus({ preventScroll: true }); });
dialog.addEventListener('click', (event) => { if (event.target === dialog) { const r = dialog.getBoundingClientRect(); if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) dialog.close(); } });
$('language-button').addEventListener('click', () => { state.lang = state.lang === 'zh' ? 'en' : 'zh'; save(); translate(); if (dialog.open) openDialog(dialogPlace); });
$('reset-button').addEventListener('click', () => { clearMovement(); stopRoute(); state.player = { x: PLACEMENTS.camp.x, y: PLACEMENTS.camp.y }; state.direction = 'south'; updateNearby(true); toast(choose('回到营地了。慢慢探索，不必着急。', 'Back at camp. Take your time.')); });
$('zoom-in').addEventListener('click', () => renderer?.setZoom(renderer.getZoom() + .2));
$('zoom-out').addEventListener('click', () => renderer?.setZoom(renderer.getZoom() - .2));
$('motion-button').addEventListener('click', () => { state.reducedMotion = !state.reducedMotion; translate(); });
$('sound-button').addEventListener('click', toggleSound);
reducedPreference.addEventListener('change', (event) => { state.reducedMotion = event.matches; translate(); });
$('retry-button').addEventListener('click', init);

function frame(now) {
  const dt = Math.min((now - lastFrame) / 1000, .045); lastFrame = now;
  if (!document.hidden) {
    state.time += dt; update(dt); renderer?.render(state); radar?.render(state.time, state.reducedMotion);
    const dot = $('mini-player'); if (dot && !$('map-panel').hidden) { dot.style.left = `${state.player.x / WORLD.width * 100}%`; dot.style.top = `${state.player.y / WORLD.height * 100}%`; }
  }
  animation = requestAnimationFrame(frame);
}
async function init() {
  const attempt = ++loadingAttempt;
  $('loading').hidden = false; $('retry-button').hidden = true; $('loading-label').textContent = words().loading;
  try {
    const next = await createRenderer(canvas, { islandUrl: asset('island.png'), explorerUrl: asset('explorer.png'), world: WORLD, placements: PLACEMENTS, onProgress: () => {} });
    if (attempt !== loadingAttempt) { next.destroy(); return; }
    renderer?.destroy(); renderer = next; initialized = true;
    $('loading').hidden = true; translate();
    if (!animation) animation = requestAnimationFrame(frame);
    toast(choose('点击小径或用方向键移动。走近建筑，按 E 发现故事。', 'Click a path or use arrow keys. Press E near a place to discover its story.'));
  } catch (error) {
    if (attempt !== loadingAttempt) return;
    $('loading').hidden = true; toast(words().error); openDialog();
    $('dialog-content').insertAdjacentHTML('afterbegin', `<p class="note">${escapeHTML(words().error)}</p><button type="button" class="secondary-button" id="reload-scene">${choose('重新加载场景', 'Reload scene')}</button>`);
    $('reload-scene').addEventListener('click', () => { dialog.close(); init(); });
    if (!animation) animation = requestAnimationFrame(frame);
  }
}

// Read-only diagnostics enabled explicitly for local browser verification.
if (new URLSearchParams(location.search).has('debug')) Object.defineProperty(window, '__planetDebug', { value: Object.freeze({
  snapshot: () => ({ ready: initialized, player: { ...state.player }, nearby: state.nearby, visited: [...state.visited], routeLength: state.route.length, lang: state.lang, moving: state.moving, reducedMotion: state.reducedMotion, zoom: renderer?.getZoom(), dialog: dialogPlace, soundEnabled }),
  walkable: (x, y) => nav.isWalkable(x, y),
}) });
window.addEventListener('pagehide', (event) => { clearMovement(); stopRoute(); if (!event.persisted) { cancelAnimationFrame(animation); renderer?.destroy(); soundContext?.close().catch(() => {}); } });
translate(); init();
