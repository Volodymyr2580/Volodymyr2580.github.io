/* File Bridge v1. No persistent storage, analytics, or automatic file acceptance. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const ALPHABET = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
  const CHUNK = 16 * 1024, WINDOW = 1024 * 1024, MEMORY_LIMIT = 128 * 1024 * 1024;
  const encoder = new TextEncoder();
  const diskSupported = typeof window.showSaveFilePicker === 'function';
  let session = null, tx = null, rx = null, queue = [], retainedBytes = 0;
  const records = new Set();
  const CHAT_LIMIT = 1000, CHAT_HISTORY = 200;
  const BUILD = '20261109-pairing-1';
  const chatPending = new Map();
  const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
  const hex = bytes => Array.from(new Uint8Array(bytes), b => b.toString(16).padStart(2, '0')).join('');
  const randomHex = () => hex(crypto.getRandomValues(new Uint8Array(16)));
  const formatCode = code => code.match(/.{1,4}/g).join('-');
  const sizeText = n => n < 1024 ? `${n} B` : n < 1024 ** 2 ? `${(n / 1024).toFixed(1)} KiB` : n < 1024 ** 3 ? `${(n / 1024 ** 2).toFixed(1)} MiB` : `${(n / 1024 ** 3).toFixed(2)} GiB`;
  const safeName = name => name.replace(/[\x00-\x1f\x7f/\\:*?"<>|]/g, '_').replace(/[. ]+$/, '').slice(0, 180) || 'download';
  function notice(text) { $('notice').textContent = text; $('notice').hidden = !text; }
  function status(title, detail, connected = false) {
    $('connection-state').textContent = title;
    $('connection-state').classList.toggle('connected', connected);
    $('connection-detail').textContent = detail;
  }
  function diagnose(s, event) {
    s.diagnostics ||= [];
    const pc = s.conn?.peerConnection;
    s.diagnostics.push({event, stage: s.stage, ice: pc?.iceConnectionState || 'none', remoteDescription: !!pc?.remoteDescription});
    if (s.diagnostics.length > 15) s.diagnostics.shift();
    $('connection-debug').hidden = false;
    $('connection-diagnostic').textContent = JSON.stringify({version: BUILD, role: s.host ? 'create' : 'join', turnConfigured: s.turnConfigured, events: s.diagnostics}, null, 2);
  }
  function pairingStage(s, stage) {
    s.stage = stage;
    const descriptions = {
      signal: ['正在连接', '1/3 · 正在联系配对服务…'],
      waiting: ['等待另一台设备', '配对码已就绪，请在另一台电脑输入。'],
      connecting: ['正在连接', '2/3 · 正在协商设备直连，请保持两端页面打开…'],
      authenticating: ['正在验证', '3/3 · 网络已连通，正在验证配对码…']
    };
    if (descriptions[stage]) status(...descriptions[stage]);
    diagnose(s, stage);
  }
  function pairingDeadline(s, milliseconds) {
    clearTimeout(s.timer);
    s.timer = setTimeout(() => {
      if (session !== s || s.authenticated) return;
      const message = s.stage === 'signal' ? '联系配对服务超时，请检查网络后重试。'
        : s.stage === 'authenticating' ? '网络已连通，但配对验证超时。请双方刷新后重试。'
        : '设备直连超时。请检查两端浏览器的本地网络权限、VPN、防火墙或 Wi-Fi 设备隔离；也可换同一手机热点重试。';
      failPairing(s, message, 'timeout');
    }, milliseconds);
  }
  function failPairing(s, message, reason) {
    if (session !== s) return;
    diagnose(s, reason);
    // A failed guest attempt must not invalidate the host's otherwise live room.
    if (s.host && !s.authenticated && s.peer?.open && s.conn) {
      clearTimeout(s.timer);
      const conn = s.conn; s.conn = null;
      s.releaseCandidates?.(); s.releaseCandidates = null;
      conn.close(); s.chain = Promise.resolve();
      pairingStage(s, 'waiting');
      notice(`${message} 当前配对码仍有效，对方可以用同一码重试。`);
      refresh(); return;
    }
    stop(); notice(message);
  }
  function refresh() {
    const connected = !!session?.authenticated;
    $('create').disabled = !!session;
    $('join').disabled = !!session;
    $('join-code').disabled = !!session;
    $('network-settings').disabled = !!session;
    $('disconnect').hidden = !session;
    $('choose-files').disabled = !connected;
    $('drop-zone').setAttribute('aria-disabled', String(!connected));
    $('drop-hint').textContent = connected ? '双方都可发送 · 支持多选 · 接收前需确认' : '连接后即可发送，支持多选文件';
    $('queue-count').textContent = queue.length ? `· ${queue.length} 个排队中` : '';
    $('empty-history').hidden = records.size > 0;
    $('chat-panel').hidden = !connected;
    document.querySelector('.connection-notes').hidden = connected;
    $('chat-help').textContent = session?.chatEnabled ? '仅本次连接 · 断开或刷新后清空' : '对方页面版本较旧，请双方刷新后重新连接以启用聊天。';
    refreshChat();
  }
  function refreshChat() {
    const enabled = !!session?.authenticated && !!session.chatEnabled;
    $('chat-input').disabled = !enabled;
    $('send-chat').disabled = !enabled || !$('chat-input').value.trim() || chatPending.size >= 20;
    $('chat-count').textContent = `${$('chat-input').value.length} / ${CHAT_LIMIT}`;
    $('chat-empty').hidden = $('chat-log').childElementCount > 0;
    $('clear-chat').disabled = !$('chat-log').childElementCount && !$('chat-input').value;
  }
  function clearChat() {
    for (const item of chatPending.values()) clearTimeout(item.timer);
    chatPending.clear(); $('chat-log').replaceChildren(); $('chat-input').value = '';
    $('chat-latest').hidden = true; $('chat-feedback').textContent = ''; refreshChat();
  }
  function latestChat() {
    const log = $('chat-log'); log.scrollTop = log.scrollHeight; $('chat-latest').hidden = true;
  }
  function appendChat(text, own, id) {
    const log = $('chat-log'), follow = log.scrollHeight - log.scrollTop - log.clientHeight < 45;
    const row = document.createElement('div'); row.className = `chat-message${own ? ' own' : ''}`; row.dataset.id = id;
    const meta = document.createElement('div'); meta.className = 'chat-meta';
    const author = document.createElement('span'); author.textContent = own ? '我' : '对方';
    const time = document.createElement('time'); time.dateTime = new Date().toISOString(); time.textContent = new Date().toLocaleTimeString('zh-CN', {hour: '2-digit', minute: '2-digit', hour12: false});
    const body = document.createElement('p'); body.className = 'chat-text'; body.textContent = text;
    const delivery = document.createElement('span'); delivery.className = 'chat-delivery'; delivery.textContent = own ? '发送中…' : '';
    meta.append(author, time); row.append(meta, body, delivery); log.append(row);
    while (log.childElementCount > CHAT_HISTORY) {
      const first = log.firstElementChild, pending = chatPending.get(first.dataset.id);
      if (pending) { clearTimeout(pending.timer); chatPending.delete(first.dataset.id); }
      first.remove();
    }
    refreshChat();
    if (own || follow) latestChat(); else $('chat-latest').hidden = false;
    return delivery;
  }
  function sendChat() {
    const s = session, text = $('chat-input').value.trim();
    if (!s?.authenticated || !s.chatEnabled || !text) return;
    if (text.length > CHAT_LIMIT) { $('chat-feedback').textContent = '消息过长，请分段发送。'; return; }
    if (chatPending.size >= 20 || s.conn.dataChannel.bufferedAmount > 2 * WINDOW) {
      $('chat-feedback').textContent = '连接繁忙，请稍后再发送；输入内容已保留。'; return;
    }
    const id = randomHex(), delivery = appendChat(text, true, id);
    try {
      send({type: 'chat', id, text}, s);
      const timer = setTimeout(() => {
        chatPending.delete(id); delivery.textContent = '暂未确认送达'; refreshChat();
      }, 30000);
      chatPending.set(id, {delivery, timer});
      $('chat-input').value = ''; $('chat-feedback').textContent = '';
    } catch { delivery.textContent = '发送失败'; $('chat-feedback').textContent = '消息未发出，输入内容已保留。'; }
    refreshChat(); $('chat-input').focus();
  }
  function readCode(value) {
    let text = value.trim();
    if (/^https?:/i.test(text)) text = new URL(text).hash.slice(1);
    if (text.startsWith('join=')) text = text.slice(5);
    text = text.toUpperCase().replace(/[\s-]/g, '');
    if (text.length !== 16 || [...text].some(c => !ALPHABET.includes(c))) throw new Error('请输入完整的 16 位配对码，或粘贴邀请链接。');
    return text;
  }
  function iceConfig() {
    const iceServers = [{urls: ['stun:stun.cloudflare.com:3478', 'stun:stun.l.google.com:19302']}];
    const urls = $('turn-urls').value.trim().split(/\s+/).filter(Boolean);
    if (urls.length) {
      if (urls.length > 6 || urls.some(url => !/^turns?:[a-z0-9.[\]:-]+(?:\?transport=(?:tcp|udp))?$/i.test(url))) throw new Error('请填写有效的 turn: 或 turns: 地址，每行一个，不含用户名或密码。');
      if (!$('turn-user').value.trim() || !$('turn-password').value) throw new Error('请补全中转服务的用户名和密码。');
      iceServers.push({urls, username: $('turn-user').value.trim(), credential: $('turn-password').value});
    }
    if ($('relay-only').checked && !urls.length) throw new Error('仅中转模式需要先填写 TURN 服务。');
    return {iceServers, iceTransportPolicy: $('relay-only').checked ? 'relay' : 'all'};
  }
  function send(message, s = session) {
    if (!s?.conn?.open) throw new Error('连接已断开，请重新配对。');
    s.conn.send(JSON.stringify(message));
  }
  function makeRecord(name, size, direction) {
    const row = document.createElement('div'); row.className = 'transfer-row';
    const head = document.createElement('div'); head.className = 'file-head';
    const title = document.createElement('div'); title.className = 'file-name'; title.textContent = `${direction === 'send' ? '↑' : '↓'} ${name}`;
    const actions = document.createElement('div'); actions.className = 'file-actions';
    const cancel = document.createElement('button'); cancel.className = 'text-button'; cancel.textContent = '取消';
    const note = document.createElement('p'); note.className = 'file-note'; note.textContent = `${sizeText(size)} · 等待接收确认`;
    const progress = document.createElement('progress'); progress.className = 'file-progress'; progress.max = Math.max(size, 1); progress.value = 0; progress.setAttribute('aria-label', `${name} 的传输进度`);
    actions.append(cancel); head.append(title, actions); row.append(head, note, progress); $('transfers').prepend(row);
    const record = {row, note, progress, cancel, actions, size, done: false, started: 0, lastPaint: 0, url: null};
    records.add(record); refresh(); return record;
  }
  function paint(record, bytes, force = false) {
    const now = performance.now();
    if (!force && now - record.lastPaint < 150) return;
    record.lastPaint = now;
    record.progress.value = bytes;
    const seconds = Math.max((now - record.started) / 1000, .01);
    const rate = bytes / seconds;
    record.note.textContent = `${sizeText(bytes)} / ${sizeText(record.size)} · ${rate ? `${sizeText(rate)}/s` : '准备中'} · ${Math.floor(bytes / Math.max(record.size, 1) * 100)}%`;
  }
  function finish(record, text, error = false) {
    record.done = true; record.cancel.remove(); record.note.textContent = `${sizeText(record.size)} · ${text}`;
    record.note.classList.toggle('error', error);
    if (!error) record.progress.value = record.progress.max;
  }
  function cancelTx(message, inform = true) {
    const t = tx; if (!t) return;
    tx = null;
    if (inform) { try { send({type: 'cancel', id: t.id}); } catch {} }
    finish(t.record, message, true); t.file = null;
    setTimeout(startNext, 0);
  }
  function cancelRx(message, inform = true) {
    const r = rx; if (!r) return;
    rx = null; $('incoming').hidden = true;
    if (inform) { try { send({type: 'reject', id: r.id, reason: message}); } catch {} }
    if (r.reserved) retainedBytes -= r.size;
    r.parts = [];
    if (r.writer) r.writer.abort().catch(() => {});
    finish(r.record, message, true);
  }
  function stop(message = '已断开连接。重新创建或加入即可继续。') {
    const old = session; session = null;
    clearChat();
    for (const item of queue) { finish(item.record, '连接结束，已移出队列', true); item.file = null; }
    queue = [];
    if (old) { clearTimeout(old.timer); clearTimeout(old.expiry); old.releaseCandidates?.(); old.conn?.close(); old.peer?.destroy(); }
    cancelTx('连接结束，未完成；请重新发送', false); cancelRx('连接结束，未完成；请重新发送', false);
    $('pair-actions').hidden = false; $('room').hidden = true;
    $('room-code').textContent = ''; status('尚未连接', message); refresh();
  }
  async function start(host) {
    if (session) return;
    try {
      if (!['https:', 'http:'].includes(location.protocol) || !window.isSecureContext || !window.Peer || !window.RTCPeerConnection || !crypto.subtle) throw new Error('请使用支持 WebRTC 的浏览器，通过 HTTPS 网站或 localhost 打开本页；不能直接双击 HTML 文件。');
      const config = iceConfig();
      // Each character contributes five random bits (80 bits per room code).
      let code = '';
      if (host) {
        for (const b of crypto.getRandomValues(new Uint8Array(16))) code += ALPHABET[b % 32];
      } else code = readCode($('join-code').value);
      const s = {code, host, authenticated: false, conn: null, peer: null, chain: Promise.resolve(), challenge: randomHex(), turnConfigured: config.iceServers.length > 1};
      session = s; refresh(); notice(''); pairingStage(s, 'signal'); pairingDeadline(s, 30000);
      const digest = hex(await crypto.subtle.digest('SHA-256', encoder.encode(`file-bridge-v1:${code}`)));
      s.key = await crypto.subtle.importKey('raw', encoder.encode(code), {name: 'HMAC', hash: 'SHA-256'}, false, ['sign', 'verify']);
      if (session !== s) return;
      const target = `fb1-${digest.slice(0, 40)}`;
      s.peer = new Peer(host ? target : `fb1-guest-${randomHex()}`, {config, debug: 0});
      s.peer.on('open', () => {
        if (session !== s) return;
        if (host) {
          clearTimeout(s.timer);
          s.expiry = setTimeout(() => { if (session === s && !s.authenticated) stop('配对码已过期，请重新创建连接。'); }, 15 * 60 * 1000);
          pairingStage(s, 'waiting');
        } else attach(s, s.peer.connect(target, {serialization: 'raw', reliable: true}));
        $('pair-actions').hidden = true; $('room').hidden = false; $('room-code').textContent = formatCode(code);
        $('room-caption').textContent = host ? '在另一台电脑输入配对码' : '正在加入临时连接';
        $('room-help').textContent = '等待配对时，15 分钟后失效。仅分享给你自己的设备。';
      });
      s.peer.on('connection', conn => {
        if (session !== s || !host || s.conn || conn.serialization !== 'raw') { conn.close(); return; }
        attach(s, conn);
      });
      s.peer.on('error', error => {
        if (session !== s) return;
        if (error.type === 'webrtc') {
          // Provider-level WebRTC errors have no connection ID and may belong to
          // a canceled attempt. Let the active channel or its deadline decide.
          diagnose(s, 'peer:webrtc'); return;
        }
        if (s.authenticated && s.conn?.open && ['network', 'socket-error', 'socket-closed'].includes(error.type)) {
          notice('配对服务暂时断开，现有文件连接仍可使用。断开后需重新配对。'); return;
        }
        const messages = {'peer-unavailable': '未找到对方，请确认配对码正确、对方页面已创建连接且未关闭。', 'unavailable-id': '配对码暂时不可用，请重新创建。', 'network': '无法联系公共配对服务，请检查网络或稍后重试。'};
        failPairing(s, messages[error.type] || `连接协商失败（${error.type || 'unknown'}），请重试；可展开下方“连接诊断”查看阶段。`, `peer:${error.type || 'unknown'}`);
      });
      s.peer.on('disconnected', () => { if (session === s && !s.authenticated) { stop(); notice('配对服务已断开，请重新创建或加入。'); } });
    } catch (error) { stop(); notice(error.message); }
  }
  function deferRemoteCandidates(s, conn) {
    const pc = conn.peerConnection, add = pc.addIceCandidate;
    let pending = [], disposed = false;
    const current = () => !disposed && session === s && s.conn === conn;
    function flush() {
      if (!current() || !pc.remoteDescription) return;
      const batch = pending; pending = [];
      for (const item of batch) {
        // PeerJS 1.5.5 handles trickled candidates without awaiting remote SDP.
        // Queue only this connection's candidates until its SDP has been applied.
        Promise.resolve().then(() => current() ? add.call(pc, item.candidate) : undefined)
          .then(item.resolve, error => current() ? item.reject(error) : item.resolve());
      }
    }
    pc.addIceCandidate = function (candidate) {
      if (!current()) return Promise.resolve();
      if (pc.remoteDescription) return add.call(pc, candidate);
      return new Promise((resolve, reject) => { pending.push({candidate, resolve, reject}); });
    };
    pc.addEventListener('signalingstatechange', flush);
    return () => {
      disposed = true; pc.removeEventListener('signalingstatechange', flush);
      pc.addIceCandidate = add;
      // Cancellation discards queued candidates; it is not a new negotiation error.
      for (const item of pending) item.resolve(); pending = [];
    };
  }
  function attach(s, conn) {
    s.conn = conn; s.chain = Promise.resolve(); s.challenge = randomHex();
    s.signed = false; s.verified = false; s.remoteReady = false; s.chatEnabled = false;
    s.releaseCandidates = deferRemoteCandidates(s, conn);
    notice(''); pairingStage(s, 'connecting'); pairingDeadline(s, 60000);
    conn.on('iceStateChanged', state => { if (session === s && s.conn === conn) diagnose(s, `ice:${state}`); });
    const beginHandshake = () => {
      if (session !== s || s.conn !== conn || s.stage === 'authenticating' || s.authenticated) return;
      pairingStage(s, 'authenticating'); pairingDeadline(s, 15000);
      send({type: 'challenge', nonce: s.challenge}, s);
    };
    conn.on('open', beginHandshake);
    conn.on('data', data => {
      // Serialize asynchronous writes, ensuring 'end' cannot overtake the last disk write.
      s.chain = s.chain.then(() => session === s && s.conn === conn ? onData(s, data, conn) : undefined).catch(() => {
        if (session === s && s.conn === conn) {
          if (!s.authenticated) failPairing(s, '配对验证失败，请确认双方使用最新页面后重试。', 'handshake-error');
          else { stop(); notice('传输协议或文件写入出错，已停止连接。未完成的文件请重新发送。'); }
        }
      });
    });
    conn.on('close', () => {
      if (session !== s || s.conn !== conn) return;
      if (!s.authenticated) failPairing(s, '对方在配对完成前断开了连接。', 'channel-closed');
      else stop('对方已断开连接。未完成的文件需要重新发送。');
    });
    conn.on('error', error => {
      if (session !== s || s.conn !== conn) return;
      failPairing(s, '设备直连协商中断，请检查网络权限、VPN 或防火墙后重试。', `channel:${error.type || 'error'}`);
    });
    if (conn.open) beginHandshake();
  }
  async function connected(s) {
    if (session !== s || s.authenticated || !s.verified || !s.remoteReady) return;
    s.authenticated = true; clearTimeout(s.timer); clearTimeout(s.expiry);
    s.stage = 'connected'; diagnose(s, 'connected');
    $('room-caption').textContent = '两台设备已连接'; $('room-help').textContent = '现在双方都可以发送文件。关闭或刷新页面会结束连接。';
    status('已连接', '加密连接已建立 · 正在确认网络路径', true); refresh();
    try {
      const stats = await s.conn.peerConnection.getStats();
      let pair;
      stats.forEach(stat => { if (stat.type === 'transport' && stat.selectedCandidatePairId) pair = stats.get(stat.selectedCandidatePairId); });
      if (!pair) stats.forEach(stat => { if (stat.type === 'candidate-pair' && stat.nominated && stat.state === 'succeeded') pair = stat; });
      const local = pair && stats.get(pair.localCandidateId), remote = pair && stats.get(pair.remoteCandidateId);
      const path = local?.candidateType === 'relay' || remote?.candidateType === 'relay' ? 'TURN 中转' : pair ? '设备直连' : '路径未识别';
      if (session === s) status('已连接', `${path} · 加密传输 · 双方均可发送`, true);
    } catch { if (session === s) status('已连接', '加密传输 · 双方均可发送', true); }
  }
  async function onData(s, data, conn) {
    if (data instanceof ArrayBuffer) {
      if (!s.authenticated || data.byteLength < 8 || data.byteLength > CHUNK + 8) throw new Error('Invalid binary frame');
      const view = new DataView(data), r = rx;
      if (!r || view.getUint32(0) !== r.id) return; // Late frames from a canceled transfer.
      if (!r.accepted || view.getUint32(4) !== r.sequence || r.received + data.byteLength - 8 > r.size) throw new Error('Unexpected chunk');
      const chunk = data.slice(8); if (!chunk.byteLength) throw new Error('Empty chunk');
      if (r.writer) {
        try { await r.writer.write(chunk); }
        catch (error) { if (rx !== r || session !== s) return; throw error; }
      } else r.parts.push(chunk);
      if (session !== s || rx !== r) return;
      r.sequence++; r.received += chunk.byteLength; r.activity = Date.now(); paint(r.record, r.received);
      if (r.received - r.lastAck >= 64 * 1024 || r.received === r.size) { r.lastAck = r.received; send({type: 'ack', id: r.id, bytes: r.received}); }
      return;
    }
    if (typeof data !== 'string' || data.length > 8192) throw new Error('Invalid message');
    const m = JSON.parse(data);
    if (!m || typeof m.type !== 'string') throw new Error('Invalid message');
    if (m.type === 'challenge') {
      if (s.signed || !/^[a-f0-9]{32}$/.test(m.nonce)) throw new Error('Invalid challenge');
      s.signed = true;
      const proof = hex(await crypto.subtle.sign('HMAC', s.key, encoder.encode(`file-bridge:${s.host ? 'host' : 'guest'}:${m.nonce}`)));
      if (session === s && s.conn === conn) send({type: 'proof', proof}, s);
      return;
    }
    if (m.type === 'proof') {
      if (s.verified || !/^[a-f0-9]{64}$/.test(m.proof)) throw new Error('Invalid proof');
      const proof = new Uint8Array(m.proof.match(/../g).map(b => parseInt(b, 16)));
      const valid = await crypto.subtle.verify('HMAC', s.key, proof, encoder.encode(`file-bridge:${s.host ? 'guest' : 'host'}:${s.challenge}`));
      if (!valid) throw new Error('Wrong code');
      if (session !== s || s.conn !== conn) return;
      s.verified = true; send({type: 'ready', features: ['chat-v1']}, s); await connected(s); return;
    }
    if (m.type === 'ready') {
      if (!s.verified || s.remoteReady) throw new Error('Invalid ready');
      s.chatEnabled = Array.isArray(m.features) && m.features.includes('chat-v1');
      s.chatSeen = new Set(); s.remoteReady = true; await connected(s); return;
    }
    if (!s.authenticated) throw new Error('Not authenticated');
    if (m.type === 'chat' || m.type === 'chat-ack') {
      if (!s.chatEnabled || typeof m.id !== 'string' || !/^[a-f0-9]{32}$/.test(m.id)) throw new Error('Invalid chat message');
      if (m.type === 'chat') {
        if (typeof m.text !== 'string' || !m.text.trim() || m.text.length > CHAT_LIMIT) throw new Error('Invalid chat text');
        if (!s.chatSeen.has(m.id)) {
          appendChat(m.text, false, m.id); s.chatSeen.add(m.id);
          if (s.chatSeen.size > CHAT_HISTORY) s.chatSeen.delete(s.chatSeen.values().next().value);
        }
        send({type: 'chat-ack', id: m.id}, s);
      } else {
        const pending = chatPending.get(m.id);
        if (pending) { clearTimeout(pending.timer); pending.delivery.textContent = '已送达'; chatPending.delete(m.id); refreshChat(); }
      }
      return;
    }
    if (!Number.isInteger(m.id) || m.id < 1 || m.id > 0xffffffff) throw new Error('Invalid transfer ID');
    switch (m.type) {
      case 'offer': {
        if (typeof m.name !== 'string' || !m.name.length || m.name.length > 255 || !Number.isSafeInteger(m.size) || m.size < 0 || m.size > 1024 ** 4) throw new Error('Invalid file metadata');
        if (rx) { send({type: 'reject', id: m.id, reason: '正在接收其他文件，请稍后重试'}); return; }
        if (!diskSupported && retainedBytes + m.size > MEMORY_LIMIT) { send({type: 'reject', id: m.id, reason: '接收端内存额度不足，请清空已下载记录，或使用桌面 Chrome / Edge'}); return; }
        const record = makeRecord(m.name, m.size, 'receive');
        rx = {id: m.id, name: safeName(m.name), size: m.size, received: 0, lastAck: 0, sequence: 0, accepted: false, parts: [], record};
        record.cancel.onclick = () => cancelRx('已取消接收');
        $('incoming-name').textContent = m.name; $('incoming-size').textContent = sizeText(m.size); $('incoming').hidden = false;
        $('accept').disabled = false; $('reject').disabled = false;
        $('receive-hint').textContent = diskSupported ? '请选择新的保存位置；若选中已有文件，完成接收时会替换它。' : '先接收到浏览器内存，完成后点击“下载”。合计上限 128 MiB。';
        break;
      }
      case 'accept':
        if (tx?.id === m.id && !tx.started) { tx.started = true; tx.activity = Date.now(); tx.record.started = performance.now(); pump(tx, s); }
        break;
      case 'ack':
        if (tx?.id === m.id) {
          if (!Number.isSafeInteger(m.bytes) || m.bytes < tx.acked || m.bytes > tx.sent) throw new Error('Invalid ack');
          tx.acked = m.bytes; tx.activity = Date.now(); paint(tx.record, tx.acked);
        }
        break;
      case 'end': {
        const r = rx; if (!r || r.id !== m.id) return;
        if (!r.accepted || r.received !== r.size) throw new Error('Incomplete file');
        if (r.writer) {
          try { await r.writer.close(); }
          catch (error) { if (rx !== r || session !== s) return; throw error; }
        }
        if (session !== s || rx !== r) return;
        if (!r.writer) {
          r.record.url = URL.createObjectURL(new Blob(r.parts, {type: 'application/octet-stream'})); r.parts = [];
          const link = document.createElement('a'); link.href = r.record.url; link.download = r.name; link.textContent = '下载';
          r.record.actions.append(link);
        }
        finish(r.record, r.writer ? '已保存到所选位置' : '已接收，请点击下载');
        send({type: 'complete', id: r.id, saved: !!r.writer}); rx = null; break;
      }
      case 'complete':
        if (tx?.id === m.id) {
          if (!tx.ended || tx.sent !== tx.file.size) throw new Error('Premature completion');
          finish(tx.record, m.saved === true ? '对方已保存' : '对方已接收，待下载'); tx = null; startNext();
        }
        break;
      case 'reject': if (tx?.id === m.id) cancelTx(typeof m.reason === 'string' ? m.reason.slice(0, 200) : '对方拒绝接收', false); break;
      case 'cancel': if (rx?.id === m.id) cancelRx('发送方已取消', false); break;
      default: throw new Error('Unknown message');
    }
  }
  async function accept() {
    const r = rx, s = session; if (!r || r.accepted || r.accepting) return;
    r.accepting = true; $('accept').disabled = true;
    try {
      if (diskSupported) {
        const handle = await window.showSaveFilePicker({suggestedName: r.name});
        if (rx !== r || session !== s) return;
        const writer = await handle.createWritable();
        if (rx !== r || session !== s) { await writer.abort(); return; }
        r.writer = writer;
      } else {
        if (retainedBytes + r.size > MEMORY_LIMIT) throw new Error('内存额度不足，请先下载并清空已完成记录。');
        retainedBytes += r.size; r.reserved = true;
      }
      r.accepted = true; r.activity = Date.now(); r.record.started = performance.now(); $('incoming').hidden = true;
      send({type: 'accept', id: r.id}); paint(r.record, 0, true);
    } catch (error) {
      if (rx === r) {
        if (error.name === 'AbortError') { r.accepting = false; $('accept').disabled = false; }
        else { cancelRx('无法写入保存位置，请重新发送'); notice(error.message); }
      }
    }
  }
  function enqueue(files) {
    if (!session?.authenticated) { notice('请先连接另一台设备，再选择文件。'); return; }
    if (queue.length + files.length > 100) { notice('一次最多排队 100 个文件，请分批发送。'); return; }
    notice('');
    for (const file of files) {
      if (file.size > 1024 ** 4 || file.name.length > 255) { notice('部分文件过大或文件名过长，未加入队列。'); continue; }
      const record = makeRecord(file.name, file.size, 'send');
      const item = {file, record}; queue.push(item);
      record.note.textContent = `${sizeText(file.size)} · 等待发送`;
      record.cancel.onclick = () => { queue = queue.filter(entry => entry !== item); finish(record, '已移出队列', true); item.file = null; refresh(); };
    }
    startNext(); refresh();
  }
  function startNext() {
    if (tx || !session?.authenticated || !queue.length) { refresh(); return; }
    const item = queue.shift();
    const id = crypto.getRandomValues(new Uint32Array(1))[0] || 1;
    tx = {...item, id, sent: 0, acked: 0, sequence: 0, started: false, ended: false};
    item.record.cancel.onclick = () => cancelTx('已取消发送');
    item.record.note.textContent = `${sizeText(item.file.size)} · 等待对方确认接收`;
    try { send({type: 'offer', id, name: item.file.name, size: item.file.size}); }
    catch { cancelTx('连接中断', false); }
    refresh();
  }
  async function pump(t, s) {
    try {
      while (tx === t && session === s && t.sent < t.file.size) {
        // Receiver acknowledgements bound both network buffering and pending disk writes.
        if (t.sent - t.acked >= WINDOW || s.conn.dataChannel.bufferedAmount > WINDOW) { await sleep(12); continue; }
        const chunk = await t.file.slice(t.sent, t.sent + CHUNK).arrayBuffer();
        if (tx !== t || session !== s) return;
        const frame = new ArrayBuffer(chunk.byteLength + 8), view = new DataView(frame);
        view.setUint32(0, t.id); view.setUint32(4, t.sequence++); new Uint8Array(frame, 8).set(new Uint8Array(chunk));
        s.conn.send(frame); t.sent += chunk.byteLength;
      }
      if (tx === t && session === s) { t.ended = true; send({type: 'end', id: t.id}); }
    } catch { if (tx === t) cancelTx('读取或发送失败，请重新发送'); }
  }
  async function copy(link) {
    if (!session) return;
    const value = link ? `${location.origin}${location.pathname}#join=${session.code}` : formatCode(session.code);
    try { await navigator.clipboard.writeText(value); notice(link ? '邀请链接已复制。在另一台电脑打开后点击“连接对方”。' : '配对码已复制。'); }
    catch { notice('剪贴板不可用，请选中页面上的配对码手动复制。'); }
  }
  $('create').onclick = () => start(true);
  $('join-form').onsubmit = event => { event.preventDefault(); start(false); };
  $('disconnect').onclick = () => { stop(); notice(''); };
  $('copy-code').onclick = () => copy(false); $('copy-link').onclick = () => copy(true);
  $('chat-form').onsubmit = event => { event.preventDefault(); sendChat(); };
  $('chat-input').oninput = () => { $('chat-feedback').textContent = ''; refreshChat(); };
  $('chat-input').onkeydown = event => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing && event.keyCode !== 229) { event.preventDefault(); sendChat(); }
  };
  $('clear-chat').onclick = clearChat; $('chat-latest').onclick = latestChat;
  $('chat-log').onscroll = () => {
    const log = $('chat-log'); if (log.scrollHeight - log.scrollTop - log.clientHeight < 45) $('chat-latest').hidden = true;
  };
  $('choose-files').onclick = () => $('file-input').click();
  $('file-input').onchange = event => { enqueue([...event.target.files]); event.target.value = ''; };
  $('accept').onclick = accept; $('reject').onclick = () => cancelRx('对方拒绝接收');
  $('clear-history').onclick = () => {
    for (const record of records) {
      if (!record.done) continue;
      if (record.url) { URL.revokeObjectURL(record.url); retainedBytes -= record.size; }
      record.row.remove(); records.delete(record);
    }
    refresh();
  };
  window.addEventListener('dragover', e => e.preventDefault());
  window.addEventListener('drop', e => e.preventDefault());
  $('drop-zone').ondragover = event => { event.preventDefault(); if (session?.authenticated) $('drop-zone').classList.add('dragover'); };
  $('drop-zone').ondragleave = () => $('drop-zone').classList.remove('dragover');
  $('drop-zone').ondrop = event => {
    event.preventDefault(); $('drop-zone').classList.remove('dragover');
    if ([...event.dataTransfer.items].some(item => item.webkitGetAsEntry?.()?.isDirectory)) { notice('暂不支持文件夹，请先压缩成 ZIP 文件再发送。'); return; }
    enqueue([...event.dataTransfer.files]);
  };
  window.addEventListener('beforeunload', event => {
    if (tx || rx || queue.length || [...records].some(r => r.url)) { event.preventDefault(); event.returnValue = ''; }
  });
  window.addEventListener('pagehide', () => { stop(); for (const record of records) if (record.url) URL.revokeObjectURL(record.url); });
  setInterval(() => {
    if (tx?.started && Date.now() - tx.activity > 90000) cancelTx('90 秒未收到进度，已取消；请重新发送');
    if (rx?.accepted && Date.now() - rx.activity > 90000) cancelRx('90 秒未收到数据，已取消；请重新发送');
  }, 5000);
  $('storage-mode').textContent = diskSupported ? '接收方式：直接写入硬盘' : '接收方式：内存 · 合计 128 MiB';
  $('accept').textContent = diskSupported ? '选择保存位置并接收' : '接收到浏览器';
  function readInvite() {
    if (!location.hash) return;
    try {
      if (session) notice('请先断开当前连接，再打开新的邀请链接。');
      else { $('join-code').value = formatCode(readCode(location.hash.slice(1))); notice('邀请已填入，请点击“连接对方”。'); }
    } catch { notice('邀请链接无效，请向另一台电脑重新获取配对码。'); }
    history.replaceState(null, '', location.pathname + location.search);
  }
  window.addEventListener('hashchange', readInvite);
  readInvite();
  refresh();
})();
