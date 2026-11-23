import { copyFile, mkdir, readFile, readdir, realpath, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const projectRoot = fileURLToPath(new URL('../', import.meta.url));
const repositoryRoot = path.resolve(projectRoot, '../..');
const outputRoot = path.join(repositoryRoot, 'playground', 'little-planet');
const requiredFiles = [
  'index.html',
  'src/main.js',
  'src/style.css',
  'assets/island.png',
  'assets/explorer.png',
];
const allowedAssets = new Set([
  '.png', '.jpg', '.jpeg', '.webp', '.avif', '.gif', '.svg',
  '.mp3', '.ogg', '.wav', '.woff', '.woff2',
]);

function isWithin(root, candidate) {
  const relative = path.relative(root, candidate);
  return relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative);
}

async function checkSource(relativePath) {
  const sourcePath = path.join(projectRoot, relativePath);
  const canonicalPath = await realpath(sourcePath);
  if (!isWithin(await realpath(projectRoot), canonicalPath)) {
    throw new Error(`Source points outside the project: ${relativePath}`);
  }
  const details = await stat(sourcePath);
  if (!details.isFile() || details.size === 0) {
    throw new Error(`Source is not a nonempty file: ${relativePath}`);
  }
}

async function checkPng(relativePath) {
  const data = await readFile(path.join(projectRoot, relativePath));
  const signature = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  if (data.length < 24 || !data.subarray(0, 8).equals(signature)) {
    throw new Error(`Invalid PNG image: ${relativePath}`);
  }
}

try {
  // Finish all input checks before copying, so a missing image cannot leave a partial build.
  for (const file of requiredFiles) await checkSource(file);
  await checkPng('assets/island.png');
  await checkPng('assets/explorer.png');

  const sourceEntries = await readdir(path.join(projectRoot, 'src'), { withFileTypes: true });
  const assetEntries = await readdir(path.join(projectRoot, 'assets'), { withFileTypes: true });
  const files = [
    'index.html',
    'src/style.css',
    ...sourceEntries.filter((entry) => entry.isFile() && entry.name.endsWith('.js'))
      .map((entry) => `src/${entry.name}`),
    ...assetEntries.filter((entry) => entry.isFile() && allowedAssets.has(path.extname(entry.name).toLowerCase()))
      .map((entry) => `assets/${entry.name}`),
  ];
  for (const file of files) await checkSource(file);

  await mkdir(outputRoot, { recursive: true });
  const canonicalOutput = await realpath(outputRoot);
  if (!isWithin(await realpath(repositoryRoot), canonicalOutput)) {
    throw new Error('The build destination points outside the repository.');
  }
  for (const file of files) {
    const destination = path.join(outputRoot, file);
    if (!isWithin(outputRoot, destination)) throw new Error(`Invalid destination: ${file}`);
    await mkdir(path.dirname(destination), { recursive: true });
    if (!isWithin(canonicalOutput, await realpath(path.dirname(destination)))) {
      throw new Error(`Destination directory points outside the build: ${file}`);
    }
    try {
      if (!isWithin(canonicalOutput, await realpath(destination))) {
        throw new Error(`Destination file points outside the build: ${file}`);
      }
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
    }
    await copyFile(path.join(projectRoot, file), destination);
  }
  console.log(`Built ${files.length} files: ${outputRoot}`);
  console.log('No files were deleted. Preview: http://127.0.0.1:4173/playground/little-planet/');
} catch (error) {
  console.error(`Build failed: ${error.message}`);
  process.exitCode = 1;
}
