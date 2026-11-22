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
  const letterBody = pick('letter-body');
  const letterSignature = pick('letter-signature');
  const letterDate = pick('letter-date');
  const quill = pick('quill-pen');
  const soundBtn = pick('toggle-sound');
  const skipBtn = pick('skip-writing');
  const replayBtn = pick('replay-writing');
  const recipient = pick('recipient');
  const code = pick('pickup-code');
  const status = pick('pickup-status');

  let delivery;
  let openTimers = [];
  let isOpening = false;
  let writeRaf = 0;
  let writeChars = [];
  let writeIndex = 0;
  let isWriting = false;

  // Subtle Web Audio API quill-on-parchment & paper rustle synthesizer
  let audioCtx = null;
  let noiseBuffer = null;
  let soundEnabled = true;
  let lastStrokeAudio = 0;

  function ensureAudio() {
    if (!soundEnabled || motion.matches) return null;
    try {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      if (!Ctx) return null;
      if (!audioCtx) {
        audioCtx = new Ctx();
        const sampleRate = audioCtx.sampleRate;
        const length = Math.floor(sampleRate * 0.6);
        noiseBuffer = audioCtx.createBuffer(1, length, sampleRate);
        const data = noiseBuffer.getChannelData(0);
        // Pink-ish smoothed noise for soft paper texture (never harsh static)
        let prev = 0;
        for (let i = 0; i < length; i += 1) {
          const white = Math.random() * 2 - 1;
          prev = (prev + 0.06 * white) / 1.06;
          data[i] = prev * 3.2;
        }
      }
      if (audioCtx.state === 'suspended') {
        audioCtx.resume().catch(() => {});
      }
      return audioCtx;
    } catch {
      return null;
    }
  }

  function playPaperRustle() {
    const ctx = ensureAudio();
    if (!ctx || !noiseBuffer) return;
    try {
      const now = ctx.currentTime;
      const src = ctx.createBufferSource();
      src.buffer = noiseBuffer;
      const filter = ctx.createBiquadFilter();
      filter.type = 'bandpass';
      filter.frequency.setValueAtTime(850, now);
      filter.frequency.exponentialRampToValueAtTime(1450, now + 0.14);
      filter.frequency.exponentialRampToValueAtTime(620, now + 0.36);
      filter.Q.value = 1.4;

      const gain = ctx.createGain();
      gain.gain.setValueAtTime(0.0001, now);
      gain.gain.linearRampToValueAtTime(0.055, now + 0.06);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.38);

      src.connect(filter).connect(gain).connect(ctx.destination);
      src.start(now, 0, 0.4);
    } catch {}
  }

  function playQuillStroke(ch, nowMs) {
    if (!soundEnabled || motion.matches || !ch || ch === ' ') return;
    if (nowMs - lastStrokeAudio < 46) return;
    lastStrokeAudio = nowMs;
    const ctx = ensureAudio();
    if (!ctx || !noiseBuffer) return;
    try {
      const t = ctx.currentTime;
      const dur = 0.048 + Math.random() * 0.026;
      const offset = Math.random() * Math.max(0.05, noiseBuffer.duration - dur - 0.02);

      const src = ctx.createBufferSource();
      src.buffer = noiseBuffer;

      const band = ctx.createBiquadFilter();
      band.type = 'bandpass';
      const baseFreq = ',.;:!?'.includes(ch) ? 1650 : 1950 + Math.random() * 650;
      band.frequency.setValueAtTime(baseFreq, t);
      band.frequency.linearRampToValueAtTime(baseFreq * (0.9 + Math.random() * 0.2), t + dur);
      band.Q.value = 2.4;

      const lowpass = ctx.createBiquadFilter();
      lowpass.type = 'lowpass';
      lowpass.frequency.value = 3400;

      const gain = ctx.createGain();
      const peak = 0.032 + Math.random() * 0.015;
      gain.gain.setValueAtTime(0.0001, t);
      gain.gain.linearRampToValueAtTime(peak, t + 0.012);
      gain.gain.exponentialRampToValueAtTime(0.0001, t + dur);

      src.connect(band).connect(lowpass).connect(gain).connect(ctx.destination);
      src.start(t, offset, dur + 0.01);
    } catch {}
  }

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

  function clearOpenTimers() {
    for (const id of openTimers) clearTimeout(id);
    openTimers = [];
    isOpening = false;
    sealed.classList.remove('is-opening', 'is-exiting');
    envelope.classList.remove('is-opening');
  }

  function stopWriting() {
    cancelAnimationFrame(writeRaf);
    writeRaf = 0;
    isWriting = false;
  }

  function appendCharSpans(container, text, targetList) {
    for (const ch of Array.from(text)) {
      const span = document.createElement('span');
      span.className = 'ink-char';
      span.textContent = ch;
      container.appendChild(span);
      targetList.push({ el: span, ch });
    }
  }

  function prepareLetterContent(recipientName, paragraphs) {
    const chars = [];
    greeting.replaceChildren();
    appendCharSpans(greeting, 'Dear ' + recipientName + ',', chars);
    if (chars.length) chars[chars.length - 1].pauseAfter = 170;

    letterBody.replaceChildren();
    paragraphs.forEach((line, idx) => {
      const p = document.createElement('p');
      appendCharSpans(p, line, chars);
      if (chars.length) chars[chars.length - 1].pauseAfter = idx === paragraphs.length - 1 ? 180 : 140;
      letterBody.appendChild(p);
    });

    letterSignature.replaceChildren();
    appendCharSpans(letterSignature, 'Warm wishes,', chars);
    if (chars.length) chars[chars.length - 1].pauseAfter = 110;
    letterSignature.appendChild(document.createElement('br'));
    const em = document.createElement('em');
    appendCharSpans(em, 'Wenzhe', chars);
    if (chars.length) chars[chars.length - 1].pauseAfter = 120;
    letterSignature.appendChild(em);

    letterDate.replaceChildren();
    appendCharSpans(letterDate, 'Christmas 2026', chars);

    return chars;
  }

  function finishWritingImmediately() {
    stopWriting();
    for (const item of writeChars) {
      item.el.classList.add('is-written');
    }
    writeIndex = writeChars.length;
    quill.classList.remove('is-writing', 'is-lifting');
    skipBtn.hidden = true;
    replayBtn.hidden = false;
  }

  function moveQuillToChar(el, stepIndex) {
    const letterRect = letter.getBoundingClientRect();
    const charRect = el.getBoundingClientRect();
    if (!charRect.width && !charRect.height) return null;
    const x = charRect.right - letterRect.left + letter.scrollLeft;
    const y = charRect.bottom - letterRect.top + letter.scrollTop - 3;
    // Gentle rhythmic nib oscillation to mimic brisk cursive strokes
    const strokeY = Math.sin(stepIndex * 1.45) * 2.2;
    const tilt = Math.cos(stepIndex * 1.15) * 4.5;
    quill.style.transform = `translate3d(${x.toFixed(1)}px, ${(y + strokeY).toFixed(1)}px, 0) rotate(${tilt.toFixed(1)}deg)`;

    // Keep the active writing line comfortably visible inside scrollable mobile viewports
    const visibleBottom = letter.scrollTop + letter.clientHeight - 44;
    if (y > visibleBottom) {
      letter.scrollTop = y - letter.clientHeight + 72;
    }
    return { x, y };
  }

  function charDelay(item) {
    if (item.pauseAfter) return item.pauseAfter;
    const ch = item.ch;
    if (ch === ' ') return 13;
    if (',;'.includes(ch)) return 95;
    if ('.!?'.includes(ch)) return 135;
    return 24;
  }

  function startQuillWriting(initialDelay = 0) {
    stopWriting();
    for (const item of writeChars) {
      item.el.classList.remove('is-written');
    }
    writeIndex = 0;

    if (motion.matches || !writeChars.length) {
      finishWritingImmediately();
      return;
    }

    isWriting = true;
    skipBtn.hidden = false;
    replayBtn.hidden = true;
    quill.classList.remove('is-writing', 'is-lifting');
    letter.scrollTop = 0;

    let lastPos = null;
    let quillShown = false;
    let nextTime = performance.now() + initialDelay;

    function step(now) {
      if (!isWriting) return;
      if (!quillShown && now >= nextTime - 120) {
        lastPos = moveQuillToChar(writeChars[0].el, 0);
        quill.classList.add('is-writing');
        quillShown = true;
      }

      let stepsThisFrame = 0;
      while (writeIndex < writeChars.length && now >= nextTime && stepsThisFrame < 3) {
        const current = writeChars[writeIndex];
        current.el.classList.add('is-written');
        playQuillStroke(current.ch, now);
        const pos = moveQuillToChar(current.el, writeIndex);
        if (pos) lastPos = pos;
        nextTime += charDelay(current);
        writeIndex += 1;
        stepsThisFrame += 1;
      }

      if (writeIndex >= writeChars.length) {
        isWriting = false;
        writeRaf = 0;
        skipBtn.hidden = true;
        replayBtn.hidden = false;
        // Lift the quill gracefully off the parchment
        if (lastPos) {
          quill.classList.add('is-lifting');
          quill.style.transform = `translate3d(${(lastPos.x + 18).toFixed(1)}px, ${(lastPos.y - 20).toFixed(1)}px, 0) rotate(14deg)`;
        }
        return;
      }

      writeRaf = requestAnimationFrame(step);
    }

    writeRaf = requestAnimationFrame(step);
  }

  function revealLetter() {
    clearOpenTimers();
    sealed.hidden = true;
    letter.hidden = false;
    greeting.focus({ preventScroll: true });
    // Wait for the parchment unfold animation to settle before touching the quill down
    startQuillWriting(560);
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
    clearOpenTimers();
    stopWriting();
    pick('envelope-recipient').textContent = 'To: ' + demo.recipient;
    writeChars = prepareLetterContent(demo.recipient, demo.paragraphs);
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
    if (isOpening) return;
    delivery?.cancel();
    if (motion.matches) {
      revealLetter();
      return;
    }
    isOpening = true;
    playPaperRustle();
    sealed.classList.add('is-opening');
    envelope.classList.add('is-opening');

    openTimers.push(setTimeout(() => {
      sealed.classList.add('is-exiting');
    }, 680));

    openTimers.push(setTimeout(() => {
      revealLetter();
    }, 960));
  });

  soundBtn.addEventListener('click', () => {
    soundEnabled = !soundEnabled;
    soundBtn.setAttribute('aria-pressed', soundEnabled ? 'true' : 'false');
    soundBtn.textContent = soundEnabled ? 'Sound: On' : 'Sound: Off';
    if (soundEnabled) ensureAudio();
  });

  skipBtn.addEventListener('click', () => {
    finishWritingImmediately();
  });

  replayBtn.addEventListener('click', () => {
    ensureAudio();
    startQuillWriting(140);
  });

  mail.addEventListener('close', () => {
    delivery?.cancel();
    delivery = undefined;
    clearOpenTimers();
    stopWriting();
    quill.classList.remove('is-writing', 'is-lifting');
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
