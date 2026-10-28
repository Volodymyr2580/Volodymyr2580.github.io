/* Run with Playwright and pngjs on NODE_PATH, or available as normal packages. */
const {chromium} = require('playwright');
const {PNG} = require('pngjs');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const http = require('node:http');
const reference = require('./reference.json');
const root = path.resolve(__dirname, '../..');
const output = path.join(root, 'design-previews/beachball');
const types = {'.html':'text/html; charset=utf-8','.css':'text/css','.js':'text/javascript','.jpg':'image/jpeg','.svg':'image/svg+xml','.png':'image/png'};
const server = http.createServer(async (req,res) => {
  try {
    let file=path.resolve(root, '.'+decodeURIComponent(new URL(req.url,'http://localhost').pathname));
    if(file!==root&&!file.startsWith(root+path.sep)) {res.writeHead(403).end();return;}
    if((await fs.stat(file)).isDirectory()) file=path.join(file,'index.html');
    res.setHeader('Content-Type',types[path.extname(file)]||'application/octet-stream');
    res.end(await fs.readFile(file));
  } catch {res.writeHead(404).end();}
});

(async()=>{
  await fs.mkdir(output,{recursive:true});
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const browser=await chromium.launch({headless:true, ...(process.env.CHROMIUM_PATH ? {executablePath:process.env.CHROMIUM_PATH} : {})});
  const page=await browser.newPage({viewport:{width:1440,height:1100},deviceScaleFactor:1});
  const errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  page.on('console',m=>{if(['error','warning'].includes(m.type()))errors.push(m.text());});
  const url=`http://127.0.0.1:${server.address().port}/playground/beachball/`;
  try {
    await page.goto(url); await page.locator('#plot svg').waitFor();
    assert.match(await page.locator('#p-readout').textContent(),/未定义.*90.0°/);
    await page.screenshot({path:path.join(output,'desktop.png'),fullPage:true});
    for(const width of [1000,768,720,390,320]) {
      await page.setViewportSize({width,height:1000});
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),`overflow ${width}`);
      if(width===390) await page.screenshot({path:path.join(output,'mobile.png'),fullPage:true});
    }
    await page.setViewportSize({width:1440,height:1100});
    await page.locator('[data-preset="0,45,90"]').click();
    await page.locator('#takeoff').fill('0');
    assert.match(await page.locator('#polarity-label').textContent(),/压缩/);
    await page.locator('#dip').fill('91');
    assert.equal(await page.locator('#input-error').isVisible(),true);
    assert.equal(await page.locator('#export-svg').isDisabled(),true);
    assert.match(await page.locator('#parameter-summary').textContent(),/45.0°/);
    await page.locator('#dip').fill('');
    assert.equal(await page.locator('#input-error').isVisible(),true);
    await page.locator('[data-preset="0,45,-90"]').click();
    assert.equal(await page.locator('#input-error').isVisible(),false);
    assert.match(await page.locator('#polarity-label').textContent(),/膨胀/);
    await page.locator('#strike-range').evaluate(el=>{el.value='123.4';el.dispatchEvent(new Event('input',{bubbles:true}));});
    assert.equal(await page.locator('#strike').inputValue(),'123.4');
    await page.locator('#strike').focus(); await page.keyboard.press('ArrowUp');
    assert.equal(await page.locator('#strike-range').inputValue(),'123.5');
    await page.locator('#azimuth').fill('361');
    assert.equal(await page.locator('#probe-error').isVisible(),true);
    const box=await page.locator('#plot svg').boundingBox();
    await page.mouse.click(box.x+box.width/2,box.y+box.height/2);
    assert.equal(await page.locator('#probe-error').isVisible(),false);
    assert.ok(Number(await page.locator('#takeoff').inputValue())<0.5,'center click within browser pixel precision');
    await page.locator('#azimuth').fill('0');await page.locator('#takeoff').fill('90');
    await page.locator('[data-preset="0,90,0"]').click();
    assert.match(await page.locator('#polarity-label').textContent(),/零初动/);

    // Verify actual browser-painted pixels against the independent ObsPy tensors.
    for(const id of ['show-axes','show-planes','show-polarities'])await page.locator('#'+id).uncheck();
    let pixelChecks=0;
    for(const record of reference.cases.slice(0,10)) {
      await page.evaluate(sdr=>{
        ['strike','dip','rake'].forEach((id,i)=>{const el=document.getElementById(id);el.step='any';el.value=sdr[i];});
        document.getElementById('rake').dispatchEvent(new Event('input',{bubbles:true}));
      },record.sdr);
      const png=PNG.sync.read(await page.locator('#plot svg').screenshot());
      for(let y=-.85;y<.9;y+=.13)for(let x=-.85;x<.9;x+=.13){
        const r2=x*x+y*y;if(r2>.85)continue;
        const k=Math.sqrt(2-r2),r=[-y*k,x*k,1-r2];
        const amp=r.reduce((sum,v,i)=>sum+v*record.matrix[i].reduce((s,w,j)=>s+w*r[j],0),0);
        if(Math.abs(amp)<.08)continue;
        const px=Math.round((280+215*x)*png.width/560),py=Math.round((280+215*y)*png.height/560);
        const idx=(py*png.width+px)*4;
        const rgb=[...png.data.subarray(idx,idx+3)];
        // Probe is at the north rim, outside this sampling domain.
        const expected=amp>0?[36,93,136]:[255,255,255];
        assert.ok(rgb.every((v,i)=>Math.abs(v-expected[i])<4),`pixel ${record.sdr} ${x}/${y}: ${rgb} expected ${expected}`);
        pixelChecks++;
      }
    }
    for(const id of ['show-axes','show-planes','show-polarities'])await page.locator('#'+id).check();
    await page.locator('[data-preset="35,60,125"]').click();
    await page.locator('#azimuth').fill('45');await page.locator('#takeoff').fill('60');
    await page.screenshot({path:path.join(output,'oblique.png'),fullPage:true});
    await page.locator('#monochrome').check();
    assert.ok((await page.locator('#plot').innerHTML()).includes('#1a1a1a'));
    await page.locator('#monochrome').uncheck();
    for(const ext of ['svg','png']){
      const pending=page.waitForEvent('download');await page.locator('#export-'+ext).click();
      const download=await pending;const file=path.join(output,'example.'+ext);await download.saveAs(file);
      assert.equal(await download.failure(),null);
      if(ext==='svg'){
        const svg=await fs.readFile(file,'utf8');assert.match(svg,/Strike 35.0°/);assert.match(svg,/viewBox="0 0 560 680"/);
        assert.ok(!svg.includes('<image'),'SVG must remain vector');
      }else{const png=PNG.sync.read(await fs.readFile(file));assert.equal(png.width,1680);assert.equal(png.height,2040);}
    }
    // The exported SVG must load independently, with no site styles or scripts.
    const svgPage=await browser.newPage();
    await svgPage.goto(`http://127.0.0.1:${server.address().port}/design-previews/beachball/example.svg`);
    await svgPage.locator('svg').waitFor();
    assert.equal(await svgPage.locator('parsererror').count(),0);
    await svgPage.goto('about:blank');
    const exportedSVG=(await fs.readFile(path.join(output,'example.svg'))).toString('base64');
    await svgPage.setContent(`<html><body style="margin:0"><img style="display:block;width:560px" src="data:image/svg+xml;base64,${exportedSVG}"></body></html>`);
    await svgPage.locator('img').evaluate(img=>img.decode());
    await svgPage.locator('img').screenshot({path:path.join(output,'export-preview.png')});
    await svgPage.close();
    const localPage=await browser.newPage();
    await localPage.goto(require('node:url').pathToFileURL(path.join(root,'playground/beachball/index.html')).href);
    await localPage.locator('#plot svg').waitFor();
    await localPage.locator('[data-preset="0,45,90"]').click();
    assert.match(await localPage.locator('#parameter-summary').textContent(),/λ 90.0°/);
    await localPage.close();
    assert.deepEqual(errors,[]);
    const summary={viewports:[1440,1000,768,720,390,320],pixelChecks,referenceMechanisms:10,
      checks:['presets','numeric inputs','range inputs','keyboard','invalid inputs','probe click','nodal polarity','layer toggles','monochrome','SVG download','PNG download'],consoleErrors:errors};
    await fs.writeFile(path.join(output,'browser-checks.json'),JSON.stringify(summary,null,2)+'\n');
    console.log(JSON.stringify(summary,null,2));
  }finally{await browser.close();server.close();}
})().catch(e=>{console.error(e);server.close();process.exitCode=1;});
