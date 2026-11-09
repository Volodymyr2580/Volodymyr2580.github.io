/* Real WebRTC through a local PeerServer; no mocked transport. */
const {chromium} = require('playwright');
const {PeerServer} = require('peer');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const http = require('node:http');
const {createHash} = require('node:crypto');
const {execFileSync} = require('node:child_process');
const root = path.resolve(__dirname, '../..');
const output = path.join(root, 'design-previews/file-transfer');
const hash = b => createHash('sha256').update(b).digest('hex');
const server = http.createServer(async (req, res) => {
  try {
    let file = path.resolve(root, '.' + decodeURIComponent(new URL(req.url, 'http://localhost').pathname));
    if (!file.startsWith(root + path.sep)) { res.writeHead(403).end(); return; }
    if ((await fs.stat(file)).isDirectory()) file = path.join(file, 'index.html');
    res.setHeader('Content-Type', {'.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css', '.jpg': 'image/jpeg'}[path.extname(file)] || 'application/octet-stream');
    res.end(await fs.readFile(file));
  } catch { res.writeHead(404).end(); }
});
let signalServer;
const signaling = PeerServer({port: 0, host: '127.0.0.1', path: '/', alive_timeout: 60000}, s => { signalServer = s; });
const report = {checks: [], errors: []};
const check = text => { report.checks.push(text); console.log('PASS', text); };
(async () => {
  await fs.mkdir(output, {recursive: true});
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  if (!signalServer?.listening) await new Promise(resolve => setTimeout(resolve, 300));
  const port = signalServer.address().port;
  const url = `http://127.0.0.1:${server.address().port}/playground/file-transfer/`;
  const browser = await chromium.launch({headless: true, executablePath: process.env.CHROMIUM_PATH || undefined});
  const contexts = [];
  async function page({disk = false, cloud = false, legacy = false} = {}) {
    const context = await browser.newContext({viewport: {width: 1440, height: 1100}, acceptDownloads: true}); contexts.push(context);
    if (!cloud) await context.route('**/vendor/peerjs-1.5.5.min.js', async route => {
      const library = await fs.readFile(path.join(root, 'playground/file-transfer/vendor/peerjs-1.5.5.min.js'), 'utf8');
      await route.fulfill({contentType: 'text/javascript', body: library + `\n;window.Peer=class extends window.Peer {constructor(id,opts){super(id,{...opts,host:'127.0.0.1',port:${port},secure:false,config:{iceServers:[]}});window.__peer=this;}};`});
    });
    if (legacy) await context.route('**/app.js?*', route => route.fulfill({contentType: 'text/javascript', body: execFileSync('git', ['show', 'd5fd9ca:playground/file-transfer/app.js'], {cwd: root}).toString('utf8')}));
    await context.addInitScript(({disk}) => {
      window.__writes = []; window.__aborted = false; window.__closed = false;
      Object.defineProperty(window, 'showSaveFilePicker', {value: disk ? async () => ({createWritable: async () => ({
        write: async chunk => { if (window.__failWrite) throw new Error('disk full'); if (window.__writeDelay) await new Promise(r => setTimeout(r, window.__writeDelay)); window.__writes.push(chunk); },
        close: async () => { window.__closed = true; }, abort: async () => { window.__aborted = true; }
      })}) : undefined, configurable: true});
    }, {disk});
    const p = await context.newPage(); p.on('pageerror', error => report.errors.push(error.message));
    await p.goto(url); return p;
  }
  async function pair(a, b) {
    await a.locator('#create').click();
    await a.waitForFunction(() => document.getElementById('room-code').textContent.length > 0);
    const code = await a.locator('#room-code').textContent();
    await b.locator('#join-code').fill(code); await b.locator('#join').click();
    await Promise.all([a, b].map(p => p.waitForFunction(() => document.getElementById('connection-state').textContent === '已连接', null, {timeout: 40000})));
    return code;
  }
  async function offer(a, b, name, data) {
    await a.locator('#file-input').setInputFiles({name, mimeType: 'application/octet-stream', buffer: data});
    await b.locator('#incoming').waitFor({state: 'visible'});
    assert.equal(await b.locator('#incoming-name').textContent(), name);
  }
  async function download(p, name) {
    const row = p.locator('.transfer-row').filter({has: p.locator('.file-name', {hasText: name})}).first();
    await row.locator('a').waitFor();
    const waiting = p.waitForEvent('download'); await row.locator('a').click();
    const d = await waiting; return fs.readFile(await d.path());
  }
  try {
    const a = await page(), b = await page();
    assert.equal(await a.locator('#choose-files').isDisabled(), true);
    assert.equal(await a.locator('#chat-panel').isVisible(), false);
    await a.screenshot({path: path.join(output, 'desktop.png'), fullPage: true});
    for (const width of [1440, 1000, 768, 700, 390, 320]) {
      await a.setViewportSize({width, height: 1000});
      assert.ok(await a.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `overflow ${width}`);
      if (width === 390) await a.screenshot({path: path.join(output, 'mobile.png'), fullPage: true});
    }
    await a.setViewportSize({width: 1440, height: 1100}); check('Responsive layout: 1440/1000/768/700/390/320; disabled offline controls');
    await a.locator('#join-code').fill('BAD'); await a.locator('#join').click();
    assert.match(await a.locator('#notice').textContent(), /16 位/); check('Invalid pairing code rejected');
    const code = await pair(a, b); assert.match(code, /^[A-Z2-9]{4}(?:-[A-Z2-9]{4}){3}$/);
    check('Two isolated browser contexts pair over real WebRTC and mutual code authentication');
    assert.equal(await a.locator('#chat-panel').isVisible(), true);
    assert.equal(await a.locator('#send-chat').isDisabled(), true);
    await a.locator('#chat-input').fill('你好，另一台电脑！'); await a.locator('#chat-input').press('Enter');
    await b.waitForFunction(() => document.querySelector('.chat-text')?.textContent === '你好，另一台电脑！');
    await a.waitForFunction(() => document.querySelector('.chat-delivery')?.textContent === '已送达');
    await b.locator('#chat-input').fill('第一行'); await b.locator('#chat-input').press('Shift+Enter'); await b.locator('#chat-input').press('End'); await b.locator('#chat-input').press('a');
    assert.equal(await b.locator('#chat-input').inputValue(), '第一行\na');
    await b.locator('#chat-input').fill('<img src=x onerror=alert(1)>\n纯文字，不执行 HTML'); await b.locator('#send-chat').click();
    await a.waitForFunction(() => document.querySelectorAll('.chat-text').length === 2);
    assert.equal(await a.locator('.chat-text').last().textContent(), '<img src=x onerror=alert(1)>\n纯文字，不执行 HTML');
    assert.equal(await a.locator('#chat-log img').count(), 0);
    check('Chat: bidirectional text, multiline keyboard input, delivery receipts and literal HTML');
    await a.locator('#chat-input').fill('输入法确认');
    await a.locator('#chat-input').evaluate(el => el.dispatchEvent(new KeyboardEvent('keydown', {key:'Enter', isComposing:true, bubbles:true})));
    assert.equal(await a.locator('#chat-input').inputValue(), '输入法确认');
    assert.equal(await a.locator('.chat-text').count(), 2);
    await a.locator('#chat-input').evaluate(el => { el.value='x'.repeat(1001); el.dispatchEvent(new Event('input')); }); await a.locator('#send-chat').click();
    assert.match(await a.locator('#chat-feedback').textContent(), /过长/);
    await a.locator('#chat-input').fill('   \n'); assert.equal(await a.locator('#send-chat').isDisabled(), true);
    await a.locator('#chat-input').fill(''); check('Chat: IME confirmation does not send; blank and over-limit messages rejected');
    await a.screenshot({path: path.join(output, 'chat-desktop.png'), fullPage: true});
    for (const width of [1000, 768, 700, 390, 320]) {
      await a.setViewportSize({width, height:1000}); assert.ok(await a.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `chat overflow ${width}`);
      if (width === 390) await a.screenshot({path: path.join(output, 'chat-mobile.png'), fullPage: true});
    }
    await a.setViewportSize({width:1440,height:1100});
    await a.locator('#clear-chat').click(); assert.equal(await a.locator('.chat-text').count(), 0); assert.equal(await b.locator('.chat-text').count(), 2);
    check('Chat: responsive layout and local-only clearing');
    const bytes = Buffer.alloc(3 * 1024 * 1024 + 79); for (let i = 0; i < bytes.length; i++) bytes[i] = (i * 17 + 23) % 256;
    await offer(a, b, '实验数据.bin', bytes); await b.locator('#accept').click();
    await b.locator('#chat-input').fill('文件正在传，可以继续聊天。'); await b.locator('#send-chat').click();
    await a.waitForFunction(() => document.querySelector('.chat-text')?.textContent === '文件正在传，可以继续聊天。');
    assert.equal(hash(await download(b, '实验数据.bin')), hash(bytes));
    await a.waitForFunction(() => document.querySelector('.file-note').textContent.includes('对方已接收，待下载'));
    check('3 MiB multi-window file downloaded with independent SHA-256 match; sender distinguishes received from saved');
    check('Chat and file transfer share the connection without corrupting file bytes');
    await b.evaluate(() => {
      const conn=Object.values(window.__peer.connections).flat()[0];
      for(let i=0;i<205;i++) conn.send(JSON.stringify({type:'chat',id:i.toString(16).padStart(32,'0'),text:`history-${i}`}));
    });
    await a.waitForFunction(() => document.querySelector('.chat-message:last-child .chat-text')?.textContent === 'history-204');
    assert.equal(await a.locator('.chat-message').count(),200); assert.equal(await a.locator('.chat-text').first().textContent(),'history-5');
    await a.locator('#chat-log').evaluate(el=>{el.scrollTop=0;el.dispatchEvent(new Event('scroll'));});
    await b.locator('#chat-input').fill('新消息提示'); await b.locator('#send-chat').click(); await a.locator('#chat-latest').waitFor({state:'visible'});
    await a.locator('#chat-latest').click(); assert.ok(await a.locator('#chat-log').evaluate(el=>el.scrollHeight-el.scrollTop-el.clientHeight<45));
    check('Chat: history capped at 200; reading earlier messages shows a jump-to-latest button');
    await a.clock.install();
    await b.evaluate(() => { const conn=Object.values(window.__peer.connections).flat()[0]; conn.__originalSend=conn.send;conn.send=function(data){if(typeof data==='string'&&JSON.parse(data).type==='chat-ack')return;return this.__originalSend(data);}; });
    await a.locator('#chat-input').fill('回执超时检查');await a.locator('#send-chat').click();
    await b.waitForFunction(()=>document.querySelector('.chat-message:last-child .chat-text')?.textContent==='回执超时检查');
    await a.clock.runFor(31000);assert.equal(await a.locator('.chat-delivery').last().textContent(),'暂未确认送达');
    await a.clock.resume();await b.evaluate(()=>{const conn=Object.values(window.__peer.connections).flat()[0];conn.send=conn.__originalSend;});
    check('Chat: missing acknowledgement never becomes delivered');
    await offer(b, a, 'reverse.txt', Buffer.from('另一台电脑 → 这台电脑\nUTF-8')); await a.locator('#accept').click();
    assert.equal((await download(a, 'reverse.txt')).toString(), '另一台电脑 → 这台电脑\nUTF-8'); check('Reverse direction UTF-8 file');
    await offer(a, b, 'empty.txt', Buffer.alloc(0)); await b.locator('#accept').click(); assert.equal((await download(b, 'empty.txt')).length, 0); check('Zero-byte file');
    await offer(a, b, '<img onerror=alert(1)>.txt', Buffer.from('safe')); assert.equal(await b.locator('#incoming-name img').count(), 0); await b.locator('#reject').click();
    await a.waitForFunction(() => document.querySelector('.file-note').textContent.includes('拒绝')); check('Untrusted file names rendered as text; rejection returned to sender');
    await a.locator('#file-input').setInputFiles([{name: 'queue1.txt', mimeType: 'text/plain', buffer: Buffer.from('first')}, {name: 'queue2.txt', mimeType: 'text/plain', buffer: Buffer.from('second')}]);
    await b.locator('#incoming').waitFor({state: 'visible'}); await b.locator('#reject').click();
    await b.waitForFunction(() => document.getElementById('incoming-name').textContent === 'queue2.txt'); await b.locator('#accept').click();
    assert.equal((await download(b, 'queue2.txt')).toString(), 'second'); check('Multi-file queue advances after rejection');
    await offer(a, b, 'cancel.txt', bytes); await a.locator('.transfer-row').first().locator('button').click(); await b.locator('#incoming').waitFor({state: 'hidden'}); check('Cancel pending offer clears receiver prompt');
    await a.locator('#clear-history').click(); await b.locator('#clear-history').click();
    assert.equal(await b.locator('.transfer-row').count(), 0); check('History clear revokes completed downloads');
    // Validate memory cap without allocating the announced file.
    await a.evaluate(() => Object.values(window.__peer.connections).flat()[0].send(JSON.stringify({type: 'offer', id: 987654, name: 'too-big.bin', size: 129 * 1024 * 1024})));
    await new Promise(resolve => setTimeout(resolve, 150)); assert.equal(await b.locator('#incoming').isVisible(), false); check('Memory receiver refuses an offer above 128 MiB');
    await offer(a, b, 'disconnect.bin', bytes); await a.locator('#disconnect').click();
    await b.waitForFunction(() => document.getElementById('connection-state').textContent === '尚未连接'); assert.equal(await b.locator('#incoming').isVisible(), false); check('Disconnect clears partial state');
    for (const p of [a,b]) { assert.equal(await p.locator('#chat-panel').isVisible(), false); assert.equal(await p.locator('.chat-text').count(),0); assert.equal(await p.locator('#chat-input').inputValue(),''); }
    check('Chat: disconnect erases both sides of the temporary conversation');
    await pair(a, b); await offer(a, b, 'again.txt', Buffer.from('again')); await b.locator('#accept').click(); assert.equal((await download(b, 'again.txt')).toString(), 'again'); check('Re-pair and resend after disconnect');
    await a.locator('#disconnect').click();
    const disk = await page({disk: true}); await pair(a, disk);
    await offer(a, disk, 'streamed.bin', bytes); await disk.locator('#accept').click();
    await disk.waitForFunction(() => window.__closed);
    const diskHash = await disk.evaluate(async () => {
      const blob = new Blob(window.__writes), digest = await crypto.subtle.digest('SHA-256', await blob.arrayBuffer());
      return Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, '0')).join('');
    }); assert.equal(diskHash, hash(bytes));
    await a.waitForFunction(() => document.querySelector('.file-note').textContent.includes('对方已保存'));
    check('Streaming save adapter receives exact bytes and closes before sender reports saved (OS picker stubbed)');
    await disk.evaluate(() => { window.__writeDelay = 80; window.__aborted = false; });
    await offer(a, disk, 'cancel-active.bin', Buffer.alloc(8 * 1024 * 1024)); await disk.locator('#accept').click();
    await disk.waitForFunction(() => document.querySelector('.file-progress').value > 0);
    await disk.locator('.transfer-row').first().locator('button').click();
    await disk.waitForFunction(() => window.__aborted); await a.waitForFunction(() => document.querySelector('.file-note').textContent.includes('取消'));
    check('Cancellation during slow disk writes aborts sink and notifies sender');
    await disk.evaluate(() => { window.__writeDelay = 0; window.__failWrite = true; });
    await offer(a, disk, 'disk-full.bin', bytes); await disk.locator('#accept').click();
    await disk.waitForFunction(() => document.getElementById('connection-state').textContent === '尚未连接');
    assert.match(await disk.locator('#notice').textContent(), /写入出错/); check('Disk write failure closes connection without success acknowledgement');
    const invited = await page(); await invited.goto(`${url}#join=ABCD-EFGH-JKLM-NPQR`); await invited.waitForFunction(() => document.getElementById('join-code').value === 'ABCD-EFGH-JKLM-NPQR'); assert.equal(new URL(invited.url()).hash, ''); check('Invite fragment fills join form without auto-connect and is removed from address bar');
    await invited.locator('details').first().locator('summary').click(); await invited.locator('#relay-only').check(); await invited.locator('#create').click(); assert.match(await invited.locator('#notice').textContent(), /TURN/); check('Relay-only mode rejects missing TURN configuration');
    // Both directions may transfer concurrently without sharing a receive buffer.
    await b.waitForFunction(() => document.getElementById('connection-state').textContent === '尚未连接');
    await pair(a, b);
    await offer(a, b, 'duplex-a.bin', bytes); await offer(b, a, 'duplex-b.txt', Buffer.from('duplex'));
    await b.locator('#accept').click(); await a.locator('#accept').click();
    assert.equal(hash(await download(b, 'duplex-a.bin')), hash(bytes)); assert.equal((await download(a, 'duplex-b.txt')).toString(), 'duplex'); check('Simultaneous bidirectional transfers keep bytes isolated');
    await a.screenshot({path: path.join(output, 'connected.png'), fullPage: true});
    // A repeated authentication proof is rejected even on an established channel.
    await offer(a, b, 'premature.txt', Buffer.from('not sent'));
    await b.evaluate(() => { const conn = Object.values(window.__peer.connections).flat()[0]; conn.send(JSON.stringify({type:'proof',proof:'0'.repeat(64)})); });
    await a.waitForFunction(() => document.getElementById('connection-state').textContent === '尚未连接'); check('Replayed/invalid authentication proof terminates the connection');
    const legacy = await page({legacy:true}); await pair(a,legacy);
    assert.equal(await a.locator('#chat-input').isDisabled(),true); assert.match(await a.locator('#chat-help').textContent(),/版本较旧/);
    await offer(a,legacy,'legacy.txt',Buffer.from('backward compatible')); await legacy.locator('#accept').click(); assert.equal((await download(legacy,'legacy.txt')).toString(),'backward compatible');
    await a.locator('#disconnect').click(); check('Old client compatibility: chat disabled, file transfer remains usable');
    // Public signaling is an environment smoke test, separate from deterministic checks.
    if (process.env.TEST_PUBLIC_SIGNAL === '1') {
      const c = await page({cloud: true}), d = await page({cloud: true});
      try { await pair(c, d); await offer(c, d, 'public-signaling.txt', Buffer.from('public signaling, actual data')); await d.locator('#accept').click(); assert.equal((await download(d, 'public-signaling.txt')).toString(), 'public signaling, actual data'); report.publicSignaling = 'PASS: same-host browser contexts, public PeerJS signaling, real WebRTC bytes'; }
      catch (e) { report.publicSignaling = `UNVERIFIED: ${await c.locator('#notice').textContent()} / ${await d.locator('#notice').textContent()} / ${e.message.slice(0, 120)}`; }
      console.log(report.publicSignaling);
    }
    assert.deepEqual(report.errors, []); check('No uncaught browser errors');
  } finally {
    await fs.writeFile(path.join(output, 'test-results.json'), JSON.stringify(report, null, 2));
    for (const context of contexts) await context.close();
    await browser.close(); server.close(); signalServer.close();
  }
})().then(() => process.exit(0), error => { console.error(error); process.exit(1); });
