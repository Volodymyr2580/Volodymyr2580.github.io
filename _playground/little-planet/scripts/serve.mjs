import { createReadStream } from 'node:fs';
import { realpath, stat } from 'node:fs/promises';
import { createServer } from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const repositoryRoot = fileURLToPath(new URL('../../../', import.meta.url));
const canonicalRoot = await realpath(repositoryRoot);
const host = '127.0.0.1';
const port = Number(process.env.PORT ?? 4173);
const types = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.txt': 'text/plain; charset=utf-8',
  '.md': 'text/plain; charset=utf-8',
  '.svg': 'image/svg+xml; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.avif': 'image/avif',
  '.gif': 'image/gif',
  '.ico': 'image/x-icon',
  '.pdf': 'application/pdf',
  '.mp3': 'audio/mpeg',
  '.ogg': 'audio/ogg',
  '.wav': 'audio/wav',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
};

if (!Number.isInteger(port) || port < 1 || port > 65535) {
  throw new Error('PORT must be an integer between 1 and 65535.');
}

function isWithinRoot(candidate) {
  const relative = path.relative(canonicalRoot, candidate);
  return relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative);
}

function respond(request, response, status, message, headers = {}) {
  response.writeHead(status, { 'Content-Type': 'text/plain; charset=utf-8', ...headers });
  response.end(request.method === 'HEAD' ? undefined : message);
}

const server = createServer(async (request, response) => {
  if (request.method !== 'GET' && request.method !== 'HEAD') {
    respond(request, response, 405, 'Only GET and HEAD are supported.', { Allow: 'GET, HEAD' });
    return;
  }
  let pathname;
  try {
    pathname = decodeURIComponent((request.url ?? '/').split(/[?#]/, 1)[0]);
  } catch {
    respond(request, response, 400, 'Malformed URL.');
    return;
  }
  // Reject traversal before URL or filesystem normalization can erase it.
  const segments = pathname.split('/').filter(Boolean);
  if (!pathname.startsWith('/') || /[\\\0:]/.test(pathname)
      || segments.some((segment) => segment.startsWith('.'))) {
    respond(request, response, 403, 'Path is not available.');
    return;
  }
  try {
    let requestedPath = path.join(repositoryRoot, ...segments);
    let canonicalPath = await realpath(requestedPath);
    if (!isWithinRoot(canonicalPath)) {
      respond(request, response, 403, 'Path is outside the repository.');
      return;
    }
    let details = await stat(canonicalPath);
    if (details.isDirectory()) {
      if (!pathname.endsWith('/')) {
        response.writeHead(301, { Location: `${encodeURI(pathname)}/` });
        response.end();
        return;
      }
      requestedPath = path.join(requestedPath, 'index.html');
      canonicalPath = await realpath(requestedPath);
      if (!isWithinRoot(canonicalPath)) {
        respond(request, response, 403, 'Path is outside the repository.');
        return;
      }
      details = await stat(canonicalPath);
    }
    if (!details.isFile()) {
      respond(request, response, 404, 'File not found.');
      return;
    }
    response.writeHead(200, {
      'Content-Type': types[path.extname(canonicalPath).toLowerCase()] ?? 'application/octet-stream',
      'Content-Length': details.size,
      'Cache-Control': 'no-store',
      'X-Content-Type-Options': 'nosniff',
    });
    if (request.method === 'HEAD') {
      response.end();
      return;
    }
    const stream = createReadStream(canonicalPath);
    stream.on('error', () => response.destroy());
    response.on('close', () => stream.destroy());
    stream.pipe(response);
  } catch (error) {
    const missing = error.code === 'ENOENT' || error.code === 'ENOTDIR';
    respond(request, response, missing ? 404 : 500,
      missing ? 'File not found. This static preview does not build Jekyll pages.' : 'Unable to read this file.');
  }
});

server.on('error', (error) => {
  console.error(`Preview failed: ${error.message}`);
  process.exitCode = 1;
});

server.listen(port, host, () => {
  console.log(`Source preview: http://${host}:${port}/_playground/little-planet/`);
  console.log(`Built preview:  http://${host}:${port}/playground/little-planet/`);
  console.log('Static files only; links to the main site require Jekyll or GitHub Pages.');
  console.log('Press Ctrl+C to stop.');
});
