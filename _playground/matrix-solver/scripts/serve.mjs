import { createServer } from 'node:http';
import { createReadStream } from 'node:fs';
import { stat, realpath } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
await import('./build.mjs');
const root = await realpath(fileURLToPath(new URL('../../../', import.meta.url)));
const port = Number(process.env.MATRIX_PORT || 4174);
const mime = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.json': 'application/json; charset=utf-8', '.txt': 'text/plain; charset=utf-8', '.png': 'image/png' };
const server = createServer(async (req, res) => {
  try {
    if (!['GET', 'HEAD'].includes(req.method)) { res.writeHead(405).end(); return; }
    let pathname = decodeURIComponent(req.url.split('?')[0]);
    if (pathname === '/') { res.writeHead(302, { Location: '/playground/matrix-solver/' }).end(); return; }
    if (/[\\\0:]/.test(pathname) || pathname.split('/').some((part) => part.startsWith('.'))) { res.writeHead(403).end(); return; }
    if (pathname.endsWith('/')) pathname += 'index.html';
    const file = await realpath(path.join(root, pathname));
    const rel = path.relative(root, file);
    if (rel.startsWith('..') || path.isAbsolute(rel)) { res.writeHead(403).end(); return; }
    const info = await stat(file);
    if (!info.isFile()) { res.writeHead(404).end(); return; }
    res.writeHead(200, { 'Content-Type': mime[path.extname(file)] || 'application/octet-stream', 'Cache-Control': 'no-store', 'Content-Length': info.size });
    if (req.method === 'HEAD') res.end(); else { const stream = createReadStream(file); stream.on('error', () => res.destroy()); stream.pipe(res); }
  } catch { res.writeHead(404).end('Not found. Main-site pages require Jekyll or the published site.'); }
});
server.listen(port, '127.0.0.1', () => console.log(`Matrix Lab: http://127.0.0.1:${port}/playground/matrix-solver/`));
server.on('error', (error) => { console.error(error.message); process.exitCode = 1; });
