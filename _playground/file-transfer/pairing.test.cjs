const {chromium} = require('playwright');
const {PeerServer} = require('peer');
const fs = require('node:fs/promises');
const path = require('node:path');
const http = require('node:http');
const assert = require('node:assert/strict');
const {execFileSync} = require('node:child_process');
const root = path.resolve(__dirname, '../..');
const server = http.createServer(async (req, res) => {
  try {
    let file = path.resolve(root, '.' + new URL(req.url, 'http://localhost').pathname);
    if (!file.startsWith(root + path.sep)) return res.writeHead(403).end();
    if ((await fs.stat(file)).isDirectory()) file = path.join(file, 'index.html');
    res.setHeader('Content-Type', {'.html':'text/html; charset=utf-8','.js':'text/javascript','.css':'text/css','.jpg':'image/jpeg'}[path.extname(file)] || 'application/octet-stream');
    res.end(await fs.readFile(file));
  } catch { res.writeHead(404).end(); }
});
let signal;
PeerServer({port:0,host:'127.0.0.1'}, s => { signal = s; });
(async () => {
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  if (!signal?.listening) await new Promise(resolve => setTimeout(resolve, 200));
  const browser = await chromium.launch({headless:true, executablePath:process.env.CHROMIUM_PATH});
  const url = `http://127.0.0.1:${server.address().port}/playground/file-transfer/`;
  const contexts = [], checks = [], errors = [];
  async function page({slowSDP = false, slowProof = false, blockSDP = false, blockProof = false} = {}) {
    const c = await browser.newContext(); contexts.push(c);
    await c.route('**/vendor/peerjs-1.5.5.min.js', async route => route.fulfill({contentType:'text/javascript',body:await fs.readFile(path.join(root,'playground/file-transfer/vendor/peerjs-1.5.5.min.js'),'utf8') + `\nwindow.Peer=class extends window.Peer{constructor(id,opts){super(id,{...opts,host:'127.0.0.1',port:${signal.address().port},secure:false,config:{iceServers:[]}});window.__peer=this;window.__peerErrors=[];this.on('error',e=>window.__peerErrors.push({type:e.type,message:e.message}));}};`}));
    if (process.env.PAIRING_PROBE === '1') await c.route('**/app.js?*', route => route.fulfill({contentType:'text/javascript',body:execFileSync('git',['show','3a4a139:playground/file-transfer/app.js'],{cwd:root}).toString('utf8')}));
    await c.addInitScript(({slowSDP,slowProof,blockSDP,blockProof}) => {
      if (slowSDP) {
        const original = RTCPeerConnection.prototype.setRemoteDescription;
        RTCPeerConnection.prototype.setRemoteDescription = async function (...args) { await new Promise(r => setTimeout(r,800)); return original.apply(this,args); };
      }
      if (slowProof) {
        const sign = crypto.subtle.sign.bind(crypto.subtle);
        crypto.subtle.sign = async (...args) => { await new Promise(r => setTimeout(r,2000)); return sign(...args); };
      }
      if (blockSDP) RTCPeerConnection.prototype.setRemoteDescription = () => new Promise(() => {});
      if (blockProof) crypto.subtle.sign = () => new Promise(() => {});
      window.__iceFailures=[];
      const add = RTCPeerConnection.prototype.addIceCandidate;
      RTCPeerConnection.prototype.addIceCandidate=async function(...args){
        try{return await add.apply(this,args);}catch(e){window.__iceFailures.push({name:e.name,message:e.message});throw e;}
      };
    }, {slowSDP,slowProof,blockSDP,blockProof});
    const p=await c.newPage(); p.on('pageerror',e=>errors.push(e.message)); await p.goto(url); return p;
  }
  const snapshot = p => p.evaluate(() => ({state:document.getElementById('connection-state').textContent,notice:document.getElementById('notice').textContent,errors:window.__peerErrors,iceFailures:window.__iceFailures}));
  async function create(p) { await p.locator('#create').click(); await p.waitForFunction(()=>document.getElementById('room-code').textContent.length>0); return p.locator('#room-code').textContent(); }
  async function join(p,code) { await p.locator('#join-code').fill(code); await p.locator('#join').click(); }
  async function paired(a,b) { await Promise.all([a,b].map(p=>p.waitForFunction(()=>document.getElementById('connection-state').textContent==='已连接',null,{timeout:12000}))); }
  function pass(text) { checks.push(text); console.log('PASS', text); }
  try {
    const host=await page(), guest=await page({slowSDP:true});
    await join(guest,await create(host));
    if (process.env.PAIRING_PROBE === '1') {
      await new Promise(r=>setTimeout(r,4000));
      const result=await Promise.all([host,guest].map(snapshot));console.log(JSON.stringify(result));
      await fs.writeFile(path.join(root,'design-previews/file-transfer/delayed-sdp-before.json'),JSON.stringify(result,null,2));
      return;
    }
    await paired(host,guest);pass('Pairing survives slow remote SDP application');
    assert.deepEqual(await guest.evaluate(()=>window.__peerErrors),[]);
    await guest.locator('#chat-input').fill('慢速协商后也能聊天');await guest.locator('#send-chat').click();
    await host.waitForFunction(()=>document.querySelector('.chat-text')?.textContent==='慢速协商后也能聊天');
    await host.locator('#disconnect').click();
    const slowHost=await page({slowSDP:true,slowProof:true}), slowGuest=await page({slowSDP:true,slowProof:true});
    await join(slowGuest,await create(slowHost));await paired(slowHost,slowGuest);
    pass('Both peers may apply SDP and calculate handshake proofs slowly');await slowHost.locator('#disconnect').click();
    const room=await page(), blocked=await page({blockProof:true}), retry=await page();
    const code=await create(room);await room.clock.install();await join(blocked,code);
    await room.waitForFunction(()=>document.getElementById('connection-state').textContent==='正在验证');
    await room.evaluate(()=>window.__stale=Object.values(window.__peer.connections).flat()[0]);
    await room.clock.runFor(16000);
    assert.equal(await room.locator('#room-code').textContent(),code);assert.equal(await room.locator('#connection-state').textContent(),'等待另一台设备');
    assert.match(await room.locator('#notice').textContent(),/验证超时/);
    const diag=await room.locator('#connection-diagnostic').textContent();assert.ok(!diag.includes(code.replaceAll('-','')));
    await room.clock.resume();await join(retry,code);await paired(room,retry);
    await room.evaluate(()=>{window.__stale.emit('close');window.__stale.emit('data','not-json');});
    await room.evaluate(()=>window.__peer.emit('error',{type:'webrtc'}));
    await retry.locator('#chat-input').fill('同一个配对码重试成功');await retry.locator('#send-chat').click();
    await room.waitForFunction(()=>document.querySelector('.chat-text')?.textContent==='同一个配对码重试成功');
    pass('Handshake timeout preserves room; retry succeeds; old connection events are ignored');
    await room.locator('#disconnect').click();
    const networkHost=await page(), blockedNetwork=await page({blockSDP:true}), networkRetry=await page();
    const networkCode=await create(networkHost);await networkHost.clock.install();await join(blockedNetwork,networkCode);
    await networkHost.waitForFunction(()=>document.getElementById('connection-detail').textContent.startsWith('2/3'));
    await networkHost.clock.runFor(61000);
    assert.match(await networkHost.locator('#notice').textContent(),/直连超时/);assert.equal(await networkHost.locator('#room-code').textContent(),networkCode);
    await networkHost.clock.resume();await join(networkRetry,networkCode);await paired(networkHost,networkRetry);
    pass('Unopened-channel timeout discards queued candidates and keeps room reusable');
    await networkHost.locator('#disconnect').click();
    const badCode=await page();await join(badCode,'AAAA-BBBB-CCCC-DDDD');
    await badCode.waitForFunction(()=>document.getElementById('notice').textContent.includes('未找到对方'));
    assert.equal(await badCode.locator('#join').isDisabled(),false);pass('Unavailable room gives actionable error and enables retry');
    assert.deepEqual(errors, []);
  } finally {
    await fs.writeFile(path.join(root,'design-previews/file-transfer/pairing-test-results.json'),JSON.stringify({checks,errors},null,2));
    for (const c of contexts) await c.close(); await browser.close(); server.close(); signal.close();
  }
})().then(()=>process.exit(0),e=>{console.error(e);process.exit(1);});
