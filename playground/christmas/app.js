(() => {
  'use strict';

  // Public demo only. Real letters live encrypted in vault/ and are built from private-christmas/.
  const demo = {
    recipient: 'Alex',
    code: 'SNOW-2026',
    envelopeRecipient: 'To: Alex',
    greeting: 'Dear Alex,',
    paragraphs: [
      'If this letter found you, the little post office is doing its job.',
      'I hope this Christmas gives you a quiet moment to catch your breath, something warm to drink, and a good reason to look forward to the new year.',
      'Thank you for stopping by. Merry Christmas!',
    ],
    signaturePrefix: 'Warm wishes,',
    signatureName: 'Wenzhe',
    date: 'Christmas 2026',
    video: null,
  };

  const pick = (id) => document.getElementById(id);
  const scene = document.querySelector('.scene');
  const artwork = pick('post-office');
  const motion = matchMedia('(prefers-reduced-motion: reduce)');
  const mail = pick('mail');
  const sealed = pick('sealed');
  const envelope = pick('break-seal');
  const letterWrap = pick('letter-wrap');
  const letter = pick('letter');
  const greeting = pick('letter-greeting');
  const letterBody = pick('letter-body');
  const letterSignature = pick('letter-signature');
  const letterDate = pick('letter-date');
  const letterVideoWrap = pick('letter-video-wrap');
  const letterVideoCaption = pick('letter-video-caption');
  const letterVideoStatus = pick('letter-video-status');
  const letterVideoFrame = pick('letter-video-frame');
  const letterVideo = pick('letter-video');
  const quill = pick('quill-pen');
  const skipBtn = pick('skip-writing');
  const replayBtn = pick('replay-writing');
  const pickupForm = pick('pickup');
  const pickupFields = pick('pickup-fields');
  const recipient = pick('recipient');
  const code = pick('pickup-code');
  const status = pick('pickup-status');
  const bgm = pick('bgm');
  const bgmToggle = pick('bgm-toggle');
  const bgmLabel = bgmToggle ? bgmToggle.querySelector('.bgm-label') : null;

  let delivery;
  let openTimers = [];
  let isOpening = false;
  let writeRaf = 0;
  let writeChars = [];
  let writeIndex = 0;
  let isWriting = false;
  let currentVideo = null;
  let activeVideoObjectUrl = null;
  let videoDecryptGeneration = 0;
  let userScrolledLetter = false;
  let vaultIndexCache = null;

  // Background music controller (Anson Seabra - Once Upon a Christmas Eve)
  const TARGET_BGM_VOLUME = 0.45;
  let bgmFadeRaf = 0;
  let userPausedBgm = false;
  let bgmPausedForVideo = false;

  function normalizeName(value) {
    return (value || '').trim().normalize('NFKC').toLowerCase().replace(/\s+/g, '');
  }

  function normalizeCode(value) {
    return (value || '').trim().normalize('NFKC').toUpperCase().replace(/\s+/g, '');
  }

  function b64ToBytes(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i += 1) {
      bytes[i] = bin.charCodeAt(i);
    }
    return bytes;
  }

  function bytesToHex(bytes) {
    let hex = '';
    for (let i = 0; i < bytes.length; i += 1) {
      hex += bytes[i].toString(16).padStart(2, '0');
    }
    return hex;
  }

  async function deriveSlotKeys(normName, normCode, saltB64, iterations) {
    const subtle = window.crypto && window.crypto.subtle;
    if (!subtle) {
      throw new Error('Secure context required for encrypted letters.');
    }
    const enc = new TextEncoder();
    const secretBytes = enc.encode(`${normName}\0${normCode}`);
    const baseKey = await subtle.importKey('raw', secretBytes, 'PBKDF2', false, ['deriveBits']);
    const bits = await subtle.deriveBits(
      {
        name: 'PBKDF2',
        salt: b64ToBytes(saltB64),
        iterations,
        hash: 'SHA-256',
      },
      baseKey,
      96 * 8
    );
    const dk = new Uint8Array(bits);
    const tagDigest = new Uint8Array(await subtle.digest('SHA-256', dk.slice(0, 32)));
    return {
      tag: bytesToHex(tagDigest).slice(0, 32),
      encKey: dk.slice(32, 64),
      macKey: dk.slice(64, 96),
    };
  }

  async function decryptBlob(encKeyBytes, macKeyBytes, encryptedBytes) {
    const subtle = window.crypto.subtle;
    if (encryptedBytes.length < 48) {
      throw new Error('Corrupted encrypted payload.');
    }
    const iv = encryptedBytes.slice(0, 16);
    const macOffset = encryptedBytes.length - 32;
    const dataToVerify = encryptedBytes.slice(0, macOffset);
    const ciphertext = encryptedBytes.slice(16, macOffset);
    const mac = encryptedBytes.slice(macOffset);

    const macKey = await subtle.importKey(
      'raw',
      macKeyBytes,
      { name: 'HMAC', hash: 'SHA-256' },
      false,
      ['verify']
    );
    const valid = await subtle.verify('HMAC', macKey, mac, dataToVerify);
    if (!valid) {
      throw new Error('Authentication failed.');
    }

    const encKey = await subtle.importKey('raw', encKeyBytes, 'AES-CTR', false, ['decrypt']);
    const plaintextBuf = await subtle.decrypt(
      { name: 'AES-CTR', counter: iv, length: 64 },
      encKey,
      ciphertext
    );
    return new Uint8Array(plaintextBuf);
  }

  async function loadVaultIndex() {
    if (vaultIndexCache) return vaultIndexCache;
    const res = await fetch('vault/index.json', { cache: 'no-store' });
    if (res.status === 404) return null;
    if (!res.ok) throw new Error('Failed to load vault index.');
    vaultIndexCache = await res.json();
    return vaultIndexCache;
  }

  async function unlockVaultLetter(rawName, rawCode) {
    const index = await loadVaultIndex();
    if (!index || !Array.isArray(index.slots) || !index.slots.length) {
      return null;
    }
    const normName = normalizeName(rawName);
    const normCode = normalizeCode(rawCode);
    const { tag, encKey, macKey } = await deriveSlotKeys(
      normName,
      normCode,
      index.saltB64,
      index.iterations || 120000
    );
    const slot = index.slots.find((entry) => entry.tag === tag);
    if (!slot) return null;

    const slotPlain = await decryptBlob(encKey, macKey, b64ToBytes(slot.payload));
    const slotInfo = JSON.parse(new TextDecoder().decode(slotPlain));
    const letterRes = await fetch(slotInfo.letterFile, { cache: 'no-store' });
    if (!letterRes.ok) throw new Error('Failed to load encrypted letter.');

    const letterCombinedKey = b64ToBytes(slotInfo.letterKeyB64);
    const letterPlain = await decryptBlob(
      letterCombinedKey.slice(0, 32),
      letterCombinedKey.slice(32, 64),
      new Uint8Array(await letterRes.arrayBuffer())
    );
    return JSON.parse(new TextDecoder().decode(letterPlain));
  }

  function syncBgmUi(playing) {
    if (!bgmToggle) return;
    bgmToggle.classList.toggle('is-playing', playing);
    bgmToggle.setAttribute('aria-pressed', playing ? 'true' : 'false');
    if (bgmLabel) bgmLabel.textContent = playing ? 'Playing' : 'Music';
  }

  function fadeBgmVolume(from, to, durationMs, onDone) {
    cancelAnimationFrame(bgmFadeRaf);
    if (!bgm) return;
    const start = performance.now();
    bgm.volume = from;
    function step(now) {
      const progress = Math.min(1, (now - start) / durationMs);
      bgm.volume = from + (to - from) * progress;
      if (progress < 1) {
        bgmFadeRaf = requestAnimationFrame(step);
      } else {
        bgmFadeRaf = 0;
        if (onDone) onDone();
      }
    }
    bgmFadeRaf = requestAnimationFrame(step);
  }

  function startBgm() {
    if (!bgm || userPausedBgm) return;
    if (!bgm.paused) {
      syncBgmUi(true);
      return;
    }
    bgm.volume = 0;
    const promise = bgm.play();
    if (promise && typeof promise.then === 'function') {
      promise.then(() => {
        syncBgmUi(true);
        fadeBgmVolume(0, TARGET_BGM_VOLUME, 1400);
      }).catch(() => {
        syncBgmUi(false);
      });
    } else {
      syncBgmUi(true);
      fadeBgmVolume(0, TARGET_BGM_VOLUME, 1400);
    }
  }

  function pauseBgm() {
    if (!bgm) return;
    syncBgmUi(false);
    if (bgm.paused) return;
    fadeBgmVolume(bgm.volume, 0, 320, () => {
      bgm.pause();
    });
  }

  if (bgmToggle && bgm) {
    bgmToggle.addEventListener('click', (event) => {
      event.stopPropagation();
      if (bgm.paused) {
        userPausedBgm = false;
        startBgm();
      } else {
        userPausedBgm = true;
        pauseBgm();
      }
    });

    startBgm();
    const unlockBgm = () => {
      if (!userPausedBgm && bgm.paused) startBgm();
    };
    window.addEventListener('pointerdown', unlockBgm, { once: true, passive: true });
    window.addEventListener('keydown', unlockBgm, { once: true, passive: true });
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

  function prepareLetterContent(letterData) {
    const chars = [];
    greeting.replaceChildren();
    appendCharSpans(greeting, letterData.greeting || '', chars);
    if (chars.length) chars[chars.length - 1].pauseAfter = 190;

    letterBody.replaceChildren();
    const paragraphs = Array.isArray(letterData.paragraphs) ? letterData.paragraphs : [];
    paragraphs.forEach((line, idx) => {
      const p = document.createElement('p');
      appendCharSpans(p, String(line), chars);
      if (chars.length) chars[chars.length - 1].pauseAfter = idx === paragraphs.length - 1 ? 190 : 150;
      letterBody.appendChild(p);
    });

    letterSignature.replaceChildren();
    const sigPrefix = letterData.signaturePrefix !== undefined ? String(letterData.signaturePrefix).trim() : 'Warm wishes,';
    const sigName = letterData.signatureName || 'Wenzhe';
    if (sigPrefix) {
      appendCharSpans(letterSignature, sigPrefix, chars);
      if (chars.length) chars[chars.length - 1].pauseAfter = 110;
      letterSignature.appendChild(document.createElement('br'));
    }
    const em = document.createElement('em');
    appendCharSpans(em, sigName, chars);
    if (chars.length) chars[chars.length - 1].pauseAfter = 130;
    letterSignature.appendChild(em);

    letterDate.replaceChildren();
    appendCharSpans(letterDate, letterData.date || 'Christmas 2026', chars);

    return chars;
  }

  function revokeActiveVideoUrl() {
    if (activeVideoObjectUrl) {
      URL.revokeObjectURL(activeVideoObjectUrl);
      activeVideoObjectUrl = null;
    }
  }

  function prepareVideoAttachment(videoConfig) {
    videoDecryptGeneration += 1;
    const gen = videoDecryptGeneration;
    currentVideo = videoConfig && (videoConfig.vaultFile || videoConfig.src) ? videoConfig : null;

    if (!letterVideoWrap || !letterVideo) return;
    letterVideo.pause();
    letterVideo.removeAttribute('src');
    letterVideo.removeAttribute('poster');
    revokeActiveVideoUrl();
    letterVideoWrap.classList.remove('is-visible');
    letterVideoWrap.hidden = true;
    letterVideoStatus.hidden = true;
    letterVideoStatus.textContent = '';
    letterVideoFrame.hidden = true;

    if (!currentVideo) return;
    letterVideoCaption.textContent = currentVideo.caption || '';

    if (currentVideo.src) {
      letterVideo.src = currentVideo.src;
      if (currentVideo.poster) letterVideo.poster = currentVideo.poster;
      letterVideoFrame.hidden = false;
      letterVideo.load();
      return;
    }

    if (currentVideo.vaultFile && currentVideo.keyB64) {
      letterVideoStatus.textContent = 'Preparing your attached video…';
      letterVideoStatus.hidden = false;
      (async () => {
        try {
          const res = await fetch(currentVideo.vaultFile, { cache: 'no-store' });
          if (!res.ok) throw new Error('Video download failed');
          const combinedKey = b64ToBytes(currentVideo.keyB64);
          const decryptedBytes = await decryptBlob(
            combinedKey.slice(0, 32),
            combinedKey.slice(32, 64),
            new Uint8Array(await res.arrayBuffer())
          );
          if (gen !== videoDecryptGeneration) return;
          const blob = new Blob([decryptedBytes], { type: currentVideo.mimeType || 'video/mp4' });
          activeVideoObjectUrl = URL.createObjectURL(blob);
          letterVideo.src = activeVideoObjectUrl;
          letterVideoStatus.hidden = true;
          letterVideoStatus.textContent = '';
          letterVideoFrame.hidden = false;
          letterVideo.load();
        } catch {
          if (gen !== videoDecryptGeneration) return;
          letterVideoStatus.textContent = 'Unable to load the attached video right now.';
          letterVideoStatus.hidden = false;
        }
      })();
    }
  }

  function revealVideoAttachment() {
    if (!currentVideo || !letterVideoWrap) return;
    letterVideoWrap.hidden = false;
    requestAnimationFrame(() => {
      letterVideoWrap.classList.add('is-visible');
    });
  }

  function resetVideoAttachment() {
    videoDecryptGeneration += 1;
    if (letterVideo) {
      letterVideo.pause();
      try { letterVideo.currentTime = 0; } catch {}
      letterVideo.removeAttribute('src');
    }
    revokeActiveVideoUrl();
    if (bgmPausedForVideo) {
      bgmPausedForVideo = false;
      if (!userPausedBgm) startBgm();
    }
  }

  if (letterVideo) {
    letterVideo.addEventListener('play', () => {
      if (bgm && !bgm.paused) {
        bgmPausedForVideo = true;
        pauseBgm();
      }
    });
    const resumeBgmAfterVideo = () => {
      if (bgmPausedForVideo) {
        bgmPausedForVideo = false;
        if (!userPausedBgm) startBgm();
      }
    };
    letterVideo.addEventListener('pause', resumeBgmAfterVideo);
    letterVideo.addEventListener('ended', resumeBgmAfterVideo);
  }

  const markUserScroll = () => {
    if (isWriting) userScrolledLetter = true;
  };
  letter.addEventListener('wheel', markUserScroll, { passive: true });
  letter.addEventListener('touchstart', markUserScroll, { passive: true });

  function finishWritingImmediately() {
    stopWriting();
    for (const item of writeChars) {
      item.el.classList.add('is-written');
    }
    writeIndex = writeChars.length;
    quill.classList.remove('is-writing', 'is-lifting');
    skipBtn.hidden = true;
    replayBtn.hidden = false;
    revealVideoAttachment();
  }

  function moveQuillToChar(el, stepIndex) {
    const letterRect = letter.getBoundingClientRect();
    const charRect = el.getBoundingClientRect();
    if (!charRect.width && !charRect.height) return null;
    const x = charRect.right - letterRect.left + letter.scrollLeft;
    const y = charRect.bottom - letterRect.top + letter.scrollTop - 3;
    const strokeY = Math.sin(stepIndex * 1.45) * 2.2;
    const tilt = Math.cos(stepIndex * 1.15) * 4.5;
    quill.style.transform = `translate3d(${x.toFixed(1)}px, ${(y + strokeY).toFixed(1)}px, 0) rotate(${tilt.toFixed(1)}deg)`;

    if (!userScrolledLetter) {
      const visibleBottom = letter.scrollTop + letter.clientHeight - 56;
      if (y > visibleBottom) {
        letter.scrollTop = y - letter.clientHeight + 88;
      }
    }
    return { x, y };
  }

  function isCjk(ch) {
    const codePoint = ch.codePointAt(0) || 0;
    return (
      (codePoint >= 0x4e00 && codePoint <= 0x9fff) ||
      (codePoint >= 0x3400 && codePoint <= 0x4dbf) ||
      (codePoint >= 0x3000 && codePoint <= 0x303f) ||
      (codePoint >= 0xff00 && codePoint <= 0xffef)
    );
  }

  function charDelay(item) {
    if (item.pauseAfter) return item.pauseAfter;
    const ch = item.ch;
    if (ch === ' ') return 13;
    if (',;，、；：'.includes(ch)) return 120;
    if ('.!?。！？…'.includes(ch)) return 185;
    return isCjk(ch) ? 46 : 22;
  }

  function startQuillWriting(initialDelay = 140) {
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
    userScrolledLetter = false;
    skipBtn.hidden = false;
    replayBtn.hidden = true;
    if (letterVideoWrap) {
      letterVideoWrap.classList.remove('is-visible');
      letterVideoWrap.hidden = true;
    }
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
        revealVideoAttachment();
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
    letterWrap.hidden = false;
    greeting.focus({ preventScroll: true });
    startQuillWriting(560);
  }

  function openLetterDialog(letterData) {
    status.textContent = '';
    code.value = '';
    clearOpenTimers();
    stopWriting();
    pick('envelope-recipient').textContent = letterData.envelopeRecipient || ('To: ' + demo.recipient);
    writeChars = prepareLetterContent(letterData);
    prepareVideoAttachment(letterData.video);
    sealed.hidden = false;
    letterWrap.hidden = true;
    mail.showModal();
    pick('waiting-letter').hidden = true;
    deliver();
  }

  pickupForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    const rawName = recipient.value;
    const rawCode = code.value;
    const normName = normalizeName(rawName);
    const normCode = normalizeCode(rawCode);

    if (normName === normalizeName(demo.recipient) && normCode === normalizeCode(demo.code)) {
      openLetterDialog(demo);
      return;
    }

    pickupFields.disabled = true;
    status.textContent = 'Checking the post office shelves…';
    try {
      const unlocked = await unlockVaultLetter(rawName, rawCode);
      if (!unlocked) {
        status.textContent = 'We couldn’t find that letter. Please check your name and secret code.';
        return;
      }
      openLetterDialog(unlocked);
    } catch {
      status.textContent = 'We couldn’t open the mailbox right now. Please try again.';
    } finally {
      pickupFields.disabled = false;
    }
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
    sealed.classList.add('is-opening');
    envelope.classList.add('is-opening');

    openTimers.push(setTimeout(() => {
      sealed.classList.add('is-exiting');
    }, 680));

    openTimers.push(setTimeout(() => {
      revealLetter();
    }, 960));
  });

  skipBtn.addEventListener('click', () => {
    finishWritingImmediately();
  });

  replayBtn.addEventListener('click', () => {
    startQuillWriting(140);
  });

  mail.addEventListener('close', () => {
    delivery?.cancel();
    delivery = undefined;
    clearOpenTimers();
    stopWriting();
    resetVideoAttachment();
    quill.classList.remove('is-writing', 'is-lifting');
    pick('waiting-letter').hidden = false;
  });

  artwork.addEventListener('load', placeSceneDetails);
  if (artwork.complete) placeSceneDetails();
  pickupFields.disabled = false;

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
