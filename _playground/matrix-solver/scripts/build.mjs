import { build } from 'esbuild';
import { mkdir, readFile, writeFile, copyFile, realpath } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
const root = fileURLToPath(new URL('../', import.meta.url));
const repo = path.resolve(root, '../..');
const out = path.join(repo, 'playground', 'matrix-solver');
await mkdir(path.join(out, 'assets'), { recursive: true });
if (!path.relative(await realpath(repo), await realpath(out)).startsWith('playground' + path.sep)) throw new Error('Unexpected build destination');
// Compile in memory first; leave the prior build intact if compilation fails.
const built = await build({
  absWorkingDir: root,
  entryPoints: { app: 'src/main.js', worker: 'src/worker.js' },
  outdir: path.join(out, 'assets'),
  bundle: true, format: 'esm', target: ['es2022'], platform: 'browser',
  minify: true, legalComments: 'linked', write: false,
});
for (const file of built.outputFiles) await writeFile(file.path, file.contents);
await copyFile(path.join(root, 'index.html'), path.join(out, 'index.html'));
await copyFile(path.join(root, 'src/style.css'), path.join(out, 'assets/app.css'));
const license = await readFile(path.join(root, 'node_modules/ml-matrix/LICENSE'), 'utf8');
await writeFile(path.join(out, 'assets/ml-matrix-LICENSE.txt'), license, 'utf8');
const vendorNotices = [];
for (const name of ['ml-matrix', 'is-any-array', 'ml-array-max', 'ml-array-min', 'ml-array-rescale']) {
  const folder = path.join(root, 'node_modules', name);
  const metadata = JSON.parse(await readFile(path.join(folder, 'package.json'), 'utf8'));
  vendorNotices.push(`${name} ${metadata.version}\n${await readFile(path.join(folder, 'LICENSE'), 'utf8')}`);
}
await writeFile(path.join(out, 'assets/THIRD-PARTY-LICENSES.txt'), vendorNotices.join('\n\n'), 'utf8');
console.log(`Built Matrix Lab at ${out}; no files deleted.`);
