import { dimension, scalar, parseMatrixText, EXAMPLES } from './input.js';
import { resultView, matrixText } from './result-view.js';
import { UI } from './locale.js';

const $ = (id) => document.getElementById(id);
let lang = 'zh';
try { if (localStorage.getItem('site-language') === 'en') lang = 'en'; } catch { /* Language preferences are optional. */ }
const t = (key) => UI[lang][key] || key;
const choose = (zh, en) => lang === 'en' ? en : zh;
let rows = 3;
let cols = 3;
let values = EXAMPLES.symmetric.map((row) => row.map(String));
let current = null;
let inputAtCalculation = null;
let renderedMatrices = {};
let activeWorker = null;
let job = 0;
let timeout = 0;
let statusKey = 'exampleLoaded';
let emptyState = 'initial';
const originalEmpty = $('results').innerHTML;
const custom = document.createElement('option'); custom.value = ''; custom.disabled = true;
$('examples').prepend(custom);
$('dimensions-form').noValidate = true;

function message(key, detail = '') {
  statusKey = key;
  $('compute-status').textContent = t(key) + (detail ? ` ${detail}` : '');
}
function translateStatic() {
  document.querySelectorAll('[data-i18n]').forEach((element) => { element.textContent = t(element.dataset.i18n); });
  document.documentElement.lang = lang === 'en' ? 'en' : 'zh-CN';
  document.title = lang === 'en' ? 'Matrix Lab · Matrix Solver' : 'Matrix Lab · 矩阵求解器';
  $('language-button').textContent = lang === 'en' ? '中文' : 'EN';
  $('language-button').setAttribute('aria-label', lang === 'en' ? '切换到中文' : 'Switch to English');
  custom.textContent = t('customExample');
  $('mode-label').textContent = rows === cols ? t('squareMode') : t('rectangleMode');
  $('identity-button').textContent = rows === cols ? t('identity') : choose('对角填 1', 'Diagonal 1s');
  $('compute-button').title = t('computeHint');
  $('matrix-editor').setAttribute('aria-label', choose('矩阵元素输入区', 'Matrix entry editor'));
  $('paste-input').setAttribute('aria-label', choose('粘贴矩阵文本', 'Paste matrix text'));
  document.querySelectorAll('.matrix-grid input').forEach((element) => element.setAttribute('aria-label', `${t('matrixCell')} A[${+element.dataset.row + 1}, ${+element.dataset.col + 1}]`));
  if (['localhost', '127.0.0.1'].includes(location.hostname)) document.querySelector('.back-link').href = 'https://volodymyr2580.github.io/playground/';
  message(statusKey);
}
function renderEmpty(state = emptyState) {
  emptyState = state;
  if (state === 'initial') $('results').innerHTML = originalEmpty;
  else $('results').innerHTML = `<div class="empty-results"><div class="empty-matrix" aria-hidden="true"><span>λ</span><span>0</span><span>0</span><span>σ</span></div><h3>${state === 'computing' ? t('computing') : t('staleTitle')}</h3><p>${state === 'computing' ? choose('正在核对矩阵与分解残差。', 'Computing factors and checking residuals.') : t('stale')}</p></div>`;
  translateStatic();
}
function clearError() {
  $('input-error').hidden = true; $('dimension-error').hidden = true;
  document.querySelectorAll('[aria-invalid="true"]').forEach((element) => element.removeAttribute('aria-invalid'));
}
function errorMessage(error) {
  const keys = { dimension: 'dimensionHelp', 'zero-denominator': 'zeroDenominator', 'paste-empty': 'pasteEmpty' };
  return `${error.row ? `A[${error.row}, ${error.col}]: ` : ''}${t(keys[error.message] || (UI[lang][error.message] ? error.message : 'calculationFailed'))}`;
}
function showError(error, id = 'input-error') {
  $(id).textContent = errorMessage(error); $(id).hidden = false;
  if (error.row) document.querySelector(`input[data-row="${error.row - 1}"][data-col="${error.col - 1}"]`)?.setAttribute('aria-invalid', 'true');
}
function finishWorker() {
  activeWorker?.terminate(); activeWorker = null; clearTimeout(timeout);
  $('compute-button').disabled = false; $('cancel-button').hidden = true;
  $('results').setAttribute('aria-busy', 'false');
}
function invalidate() {
  const hadResults = !!current || !!activeWorker;
  ++job; finishWorker(); current = null; renderedMatrices = {}; inputAtCalculation = null;
  $('copy-all-button').disabled = $('download-button').disabled = true;
  clearError();
  if (hadResults || emptyState !== 'initial') renderEmpty('stale');
  message('stale');
}
function drawEditor() {
  $('matrix-editor').innerHTML = '';
  const grid = document.createElement('div'); grid.className = 'matrix-grid'; grid.style.setProperty('--cols', cols);
  const fragment = document.createDocumentFragment();
  values.forEach((row, i) => row.forEach((value, j) => {
    const input = document.createElement('input');
    input.type = 'text'; input.value = value; input.inputMode = 'text'; input.autocomplete = 'off'; input.spellcheck = false;
    input.dataset.row = i; input.dataset.col = j;
    input.setAttribute('aria-label', `${t('matrixCell')} A[${i + 1}, ${j + 1}]`);
    fragment.append(input);
  }));
  grid.append(fragment); $('matrix-editor').append(grid);
  $('input-shape').textContent = `${rows} × ${cols}`;
  $('rows').value = rows; $('cols').value = cols;
  translateStatic();
}
function appliedDimensions() {
  const m = dimension($('rows').value); const n = dimension($('cols').value);
  if (m !== rows || n !== cols) throw new Error('pendingDimensions');
}
function readInput() {
  appliedDimensions();
  return values.map((row, i) => row.map((value, j) => {
    try { return scalar(value); } catch (error) {
      error.row = i + 1; error.col = j + 1;
      const input = document.querySelector(`input[data-row="${i}"][data-col="${j}"]`);
      input.setAttribute('aria-invalid', 'true'); input.focus({ preventScroll: true }); throw error;
    }
  }));
}
function showResult() {
  const rendered = resultView(current, lang, Number($('precision').value));
  renderedMatrices = rendered.matrices; $('results').innerHTML = rendered.html;
  $('copy-all-button').disabled = $('download-button').disabled = false;
  $('results').querySelectorAll('[data-copy]').forEach((button) => button.addEventListener('click', () => copy(matrixText(renderedMatrices[button.dataset.copy].values))));
}
function compute() {
  clearError();
  let matrix;
  try { matrix = readInput(); }
  catch (error) { invalidate(); showError(error, error.message === 'dimension' || error.message === 'pendingDimensions' ? 'dimension-error' : 'input-error'); return; }
  ++job; finishWorker(); const id = job;
  current = null; renderedMatrices = {}; inputAtCalculation = matrix;
  $('copy-all-button').disabled = $('download-button').disabled = true;
  statusKey = 'computing'; renderEmpty('computing');
  $('compute-button').disabled = true; $('cancel-button').hidden = false;
  $('results').setAttribute('aria-busy', 'true');
  try {
    activeWorker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    activeWorker.onmessage = (event) => {
      if (event.data.id !== job) return;
      finishWorker();
      if (event.data.error) { showError(new Error('calculationFailed')); renderEmpty('stale'); message('calculationFailed'); return; }
      current = event.data.result; showResult();
      message('complete', choose(`· ${rows} × ${cols} · 可复制或下载结果`, `· ${rows} × ${cols} · ready to copy or download`));
    };
    activeWorker.onerror = (event) => { event.preventDefault(); if (id !== job) return; finishWorker(); renderEmpty('stale'); showError(new Error('loadingError')); message('loadingError'); };
    timeout = setTimeout(() => { if (id !== job) return; ++job; finishWorker(); renderEmpty('stale'); showError(new Error('timeout')); message('timeout'); }, 12000);
    activeWorker.postMessage({ id, matrix });
  } catch { finishWorker(); renderEmpty('stale'); showError(new Error('loadingError')); message('loadingError'); }
}
async function copy(text) {
  try { await navigator.clipboard.writeText(text); message('copied'); }
  catch {
    let fallback = $('copy-fallback');
    if (!fallback) {
      const block = document.createElement('div'); block.className = 'notice'; block.id = 'copy-fallback-block';
      const label = document.createElement('label'); label.htmlFor = 'copy-fallback'; label.id = 'copy-fallback-label'; block.append(label);
      fallback = document.createElement('textarea'); fallback.id = 'copy-fallback'; fallback.readOnly = true; block.append(fallback);
      $('results').prepend(block);
    }
    $('copy-fallback-label').textContent = t('copyFailed'); fallback.value = text; fallback.focus({ preventScroll: true }); fallback.select(); message('copyFailed');
  }
}
function completeText() {
  const matrices = [{ label: 'A', values: inputAtCalculation }, ...Object.values(renderedMatrices)];
  const diagnostics = {
    mode: current.mode, warnings: current.warnings, numericalRank: current.rank, tolerance: current.tolerance, conditionNumber: current.conditionNumber,
    eigenbasisReliable: current.evd?.diagonalizable,
    eigenbasisCondition: current.evd?.basisCondition, eigenResidual: current.evd?.residual,
    inverseResiduals: current.inverse ? [current.inverse.leftResidual, current.inverse.rightResidual] : undefined,
    penroseResiduals: current.pseudoinverse?.penroseResiduals,
    svdResidual: current.svd?.residual, orthogonalityU: current.svd?.orthogonalityU, orthogonalityV: current.svd?.orthogonalityV,
  };
  const validity = current.evd && !current.evd.diagonalizable ? choose('未获得可靠的数值对角化。以下只满足特征值方程 AV ≈ VΛ。', 'A reliable numerical diagonalization is unavailable. Only the eigen-equation AV ≈ VΛ is provided.') : '';
  return [`Matrix Lab — ${rows} × ${cols}`, validity, `Numerical diagnostics\n${JSON.stringify(diagnostics, (_, value) => typeof value === 'number' && !Number.isFinite(value) ? String(value) : value, 2)}`, ...matrices.map((matrix) => `${matrix.label}\n${matrixText(matrix.values)}`)].filter(Boolean).join('\n\n');
}

$('dimensions-form').addEventListener('submit', (event) => {
  event.preventDefault();
  try {
    const m = dimension($('rows').value); const n = dimension($('cols').value);
    const next = Array.from({ length: m }, (_, i) => Array.from({ length: n }, (_, j) => values[i]?.[j] ?? '0'));
    invalidate(); rows = m; cols = n; values = next; $('examples').value = ''; drawEditor(); message('dimensionsReady');
  } catch (error) { showError(error, 'dimension-error'); }
});
for (const id of ['rows', 'cols']) $(id).addEventListener('input', () => { invalidate(); message('pendingDimensions'); });
$('matrix-editor').addEventListener('input', (event) => {
  if (!event.target.matches('input')) return;
  values[Number(event.target.dataset.row)][Number(event.target.dataset.col)] = event.target.value;
  invalidate(); $('examples').value = '';
});
$('matrix-editor').addEventListener('keydown', (event) => {
  if (event.key !== 'Enter' || event.ctrlKey || event.metaKey || !event.target.matches('input')) return;
  event.preventDefault();
  const i = Number(event.target.dataset.row), j = Number(event.target.dataset.col);
  const next = document.querySelector(`input[data-row="${(i + (j + 1 >= cols ? 1 : 0)) % rows}"][data-col="${(j + 1) % cols}"]`);
  next?.focus(); next?.select();
});
function fill(next) {
  try { appliedDimensions(); } catch (error) { showError(error, 'dimension-error'); return; }
  invalidate(); values = next.map((row) => row.map(String)); $('examples').value = ''; drawEditor();
}
$('identity-button').addEventListener('click', () => fill(Array.from({ length: rows }, (_, i) => Array.from({ length: cols }, (_, j) => i === j ? 1 : 0))));
$('random-button').addEventListener('click', () => fill(Array.from({ length: rows }, () => Array.from({ length: cols }, () => Math.floor(Math.random() * 11) - 5))));
$('clear-button').addEventListener('click', () => fill(Array.from({ length: rows }, () => Array(cols).fill(0))));
$('examples').addEventListener('change', () => {
  const name = $('examples').value; if (!EXAMPLES[name]) return;
  invalidate(); rows = EXAMPLES[name].length; cols = EXAMPLES[name][0].length; values = EXAMPLES[name].map((row) => row.map(String)); drawEditor(); message('exampleLoaded');
});
function paste(text) {
  clearError();
  try { appliedDimensions(); const matrix = parseMatrixText(text, rows, cols); fill(matrix); message('dimensionsReady'); }
  catch (error) { showError(error); $('paste-input').setAttribute('aria-invalid', 'true'); }
}
$('paste-button').addEventListener('click', () => paste($('paste-input').value));
$('matrix-editor').addEventListener('paste', (event) => {
  const text = event.clipboardData?.getData('text/plain') || '';
  if (/\t/.test(text) || /[\n;]/.test(text.trim())) { event.preventDefault(); paste(text); }
});
$('compute-button').addEventListener('click', compute);
$('cancel-button').addEventListener('click', () => { ++job; finishWorker(); renderEmpty('stale'); message('cancelled'); });
$('precision').addEventListener('change', () => { if (current) showResult(); });
$('copy-all-button').addEventListener('click', () => { if (current) copy(completeText()); });
$('download-button').addEventListener('click', () => {
  if (!current) return;
  const payload = { input: inputAtCalculation, ...current, metadata: { arithmetic: 'IEEE 754 double precision', library: 'ml-matrix 6.15.1', complex: '{re, im}', nonfiniteScalars: 'Infinity is encoded as a string.', displayDigits: Number($('precision').value) } };
  const blob = new Blob([JSON.stringify(payload, (_, value) => typeof value === 'number' && !Number.isFinite(value) ? String(value) : value, 2)], { type: 'application/json;charset=utf-8' });
  const url = URL.createObjectURL(blob); const a = document.createElement('a'); a.href = url; a.download = `matrix-${rows}x${cols}.json`; document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000); message('downloadReady');
});
$('language-button').addEventListener('click', () => {
  lang = lang === 'en' ? 'zh' : 'en';
  try { localStorage.setItem('site-language', lang === 'en' ? 'en' : 'zh-CN'); } catch { /* Optional. */ }
  translateStatic(); if (current) showResult(); else renderEmpty();
});
window.addEventListener('keydown', (event) => { if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') { event.preventDefault(); compute(); } });
window.addEventListener('pagehide', () => { ++job; finishWorker(); });
drawEditor(); renderEmpty(); message('exampleLoaded');
