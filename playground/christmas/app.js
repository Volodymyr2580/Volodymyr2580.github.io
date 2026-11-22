(() => {
  'use strict';

  // Public demo only. Real letters must not be stored here as readable text.
  const demo = {
    recipient: 'Alex',
    code: 'SNOW-2026',
    paragraphs: [
      'If this letter found you, the little post office is doing its job.',
      'I hope this Christmas gives you a quiet moment to catch your breath, something warm to drink, and a good reason to look forward to the new year.',
      'Thank you for stopping by. Merry Christmas!',
    ],
  };

  const pick = (id) => document.getElementById(id);
  const scene = document.querySelector('.scene');
  const artwork = pick('post-office');
  const motion = matchMedia('(prefers-reduced-motion: reduce)');
  const mail = pick('mail');
  const sealed = pick('sealed');
  const envelope = pick('break-seal');
  const letter = pick('letter');
  const greeting = pick('letter-greeting');
  const recipient = pick('recipient');
  const code = pick('pickup-code');
  const status = pick('pickup-status');
  let delivery;

  // Fractions are measured in the original images, before object-fit crops them.
  function artPoint(feature) {
    const portrait = artwork.currentSrc.includes('portrait');
    const coordinates = portrait
      ? { slot: [.746, .798], lamp: [.48, .637] }
      : { slot: [.831, .617], lamp: [.664, .290] };
    const [x, y] = coordinates[feature];
    const rect = artwork.getBoundingClientRect();
    if (!artwork.naturalWidth || !rect.width || !rect.height) return null;
    const zoom = Math.max(rect.width / artwork.naturalWidth, rect.height / artwork.naturalHeight);
    const width = artwork.naturalWidth * zoom;
    const height = artwork.naturalHeight * zoom;
    return {
      x: rect.left + (rect.width - width) * (portrait ? .5 : 1) + width * x,
      y: rect.top + (rect.height - height) * (portrait ? 0 : .5) + height * y,
      width,
    };
  }

  function placeSceneDetails() {
    const lamp = artPoint('lamp');
    if (!lamp) return;
    const rect = scene.getBoundingClientRect();
    const element = pick('lantern');
    element.style.setProperty('--lamp-x', (lamp.x - rect.left) + 'px');
    element.style.setProperty('--lamp-y', (lamp.y - rect.top) + 'px');
    const slot = artPoint('slot');
    const waiting = pick('waiting-letter');
    waiting.style.setProperty('--slot-x', (slot.x - rect.left) + 'px');
    waiting.style.setProperty('--slot-y', (slot.y - rect.top) + 'px');
    waiting.style.setProperty('--slot-letter-width', slot.width * (artwork.currentSrc.includes('portrait') ? .075 : .047) + 'px');
  }

  function deliver() {
    if (motion.matches || typeof envelope.animate !== 'function') return;
    const slot = artPoint('slot');
    const rect = envelope.getBoundingClientRect();
    const visibleSlot = slot && slot.x > 0 && slot.x < innerWidth && slot.y > 0 && slot.y < innerHeight;
    const x = visibleSlot ? slot.x - (rect.left + rect.width / 2) : 0;
    const y = visibleSlot ? slot.y - (rect.top + rect.height / 2) : 42;
    delivery = envelope.animate([
      { transform: `translate(${x}px, ${y - 5}px) rotate(-8deg) scale(.18)`, clipPath: 'inset(85% 0 0)', opacity: 0 },
      { transform: `translate(${x * .94}px, ${y + 9}px) rotate(-11deg) scale(.25)`, clipPath: 'inset(0)', opacity: 1, offset: .24 },
      { transform: 'translate(0, 0) rotate(0) scale(1)', clipPath: 'inset(0)', opacity: 1 },
    ], { duration: 820, easing: 'cubic-bezier(.24,.66,.28,1)' });
  }

  pick('pickup').addEventListener('submit', (event) => {
    event.preventDefault();
    const name = recipient.value.trim().normalize('NFKC').toLowerCase();
    const token = code.value.trim().toUpperCase();
    if (name !== demo.recipient.toLowerCase() || token !== demo.code) {
      status.textContent = 'We couldn’t find that letter. Please check your name and secret code.';
      return;
    }
    status.textContent = '';
    code.value = '';
    pick('envelope-recipient').textContent = 'To: ' + demo.recipient;
    greeting.textContent = 'Dear ' + demo.recipient + ',';
    pick('letter-body').replaceChildren(...demo.paragraphs.map((line) => {
      const paragraph = document.createElement('p');
      paragraph.textContent = line;
      return paragraph;
    }));
    sealed.hidden = false;
    letter.hidden = true;
    mail.showModal();
    pick('waiting-letter').hidden = true;
    deliver();
  });
  for (const input of [recipient, code]) {
    input.addEventListener('input', () => { status.textContent = ''; });
  }
  envelope.addEventListener('click', () => {
    delivery?.cancel();
    sealed.hidden = true;
    letter.hidden = false;
    greeting.focus({ preventScroll: true });
  });
  mail.addEventListener('close', () => {
    delivery?.cancel();
    delivery = undefined;
    pick('waiting-letter').hidden = false;
  });
  artwork.addEventListener('load', placeSceneDetails);
  if (artwork.complete) placeSceneDetails();
  pick('pickup-fields').disabled = false;

  const canvas = pick('snow');
  const context = canvas.getContext('2d');
  let flakes = [];
  let width = 0;
  let height = 0;
  let frame = 0;
  let last = 0;

  function flake(initial) {
    const depth = Math.random();
    return {
      x: Math.random() * width,
      y: initial ? Math.random() * height : -8,
      radius: .55 + depth ** 3 * 2.9,
      speed: 15 + depth ** 2 * 51,
      alpha: .24 + depth * .44,
      depth,
      phase: Math.random() * Math.PI * 2,
    };
  }

  function snow(time, animate) {
    const elapsed = last ? Math.min((time - last) / 1000, .035) : 0;
    last = time;
    context.clearRect(0, 0, width, height);
    for (const item of flakes) {
      if (animate) {
        item.y += item.speed * elapsed;
        item.x += (9 + 18 * item.depth + Math.sin(time / 1130 + item.phase) * 12) * elapsed;
        if (item.y > height + 8) Object.assign(item, flake(false));
        if (item.x > width + 8) item.x = -8;
        if (item.x < -8) item.x = width + 8;
      }
      context.fillStyle = 'rgba(231, 242, 255, ' + item.alpha + ')';
      context.beginPath();
      context.arc(item.x, item.y, item.radius, 0, Math.PI * 2);
      context.fill();
    }
    frame = animate ? requestAnimationFrame(tick) : 0;
  }

  function tick(time) { snow(time, true); }

  function syncSnow() {
    if (!context) return;
    cancelAnimationFrame(frame);
    last = 0;
    const still = motion.matches;
    pick('lantern').style.animationPlayState = still ? 'paused' : 'running';
    snow(performance.now(), !still && !document.hidden);
  }

  function layout() {
    placeSceneDetails();
    if (!context) return;
    width = scene.clientWidth;
    height = scene.clientHeight;
    const density = Math.min(devicePixelRatio || 1, 1.5);
    canvas.width = Math.ceil(width * density);
    canvas.height = Math.ceil(height * density);
    context.setTransform(density, 0, 0, density, 0, 0);
    flakes = Array.from({ length: Math.min(155, Math.max(44, Math.round(width * height / 9200))) }, () => flake(true));
    syncSnow();
  }

  motion.addEventListener('change', syncSnow);
  document.addEventListener('visibilitychange', syncSnow);
  new ResizeObserver(layout).observe(scene);
  layout();
})();
