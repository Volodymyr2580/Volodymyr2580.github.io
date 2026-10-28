export const MAX_DIMENSION = 50;
const decimal = /^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i;

export function dimension(value) {
  const raw = String(value).trim();
  const number = Number(raw);
  if (!/^\d+$/.test(raw) || !Number.isInteger(number) || number < 1 || number > MAX_DIMENSION) {
    throw new Error('dimension');
  }
  return number;
}

export function scalar(value) {
  const text = String(value).trim();
  if (!text) throw new Error('empty');
  const parts = text.split('/').map((part) => part.trim());
  if (parts.length > 2 || parts.some((part) => !decimal.test(part))) throw new Error('number');
  const numerator = Number(parts[0]);
  const denominator = parts.length === 2 ? Number(parts[1]) : 1;
  if (!Number.isFinite(numerator) || !Number.isFinite(denominator)) throw new Error('range');
  // Detect nonzero decimals that underflowed during parsing.
  if (parts.some((part) => Number(part) === 0 && /[1-9]/.test(part.split(/e/i)[0]))) throw new Error('range');
  if (denominator === 0) throw new Error('zero-denominator');
  const result = numerator / denominator;
  if (!Number.isFinite(result) || Math.abs(result) > 1e150 || (result !== 0 && Math.abs(result) < 1e-150) || (result === 0 && numerator !== 0)) throw new Error('range');
  return Object.is(result, -0) ? 0 : result;
}

export function parseMatrixText(text, rows, cols) {
  // Preserve tabs at row edges: spreadsheet blank cells must not disappear.
  const trimOuterSpace = (value) => value.replace(/^[ \r\n]+|[ \r\n]+$/g, '');
  let source = trimOuterSpace(String(text));
  if (!source.trim()) throw new Error('paste-empty');
  // MATLAB-style surrounding brackets are optional. Nested brackets and
  // arbitrary expressions are deliberately not evaluated.
  if (source.startsWith('[') && source.endsWith(']')) source = trimOuterSpace(source.slice(1, -1));
  const lines = source.split(/\r?\n|;/).map((line) => line.replace(/^ +| +$/g, '')).filter((line) => line.trim());
  const cells = lines.map((line) => line.split(/ *, *| *\t *| +/));
  if (cells.length !== rows || cells.some((row) => row.length !== cols)) throw new Error('shape');
  return cells.map((row, i) => row.map((cell, j) => {
    try { return scalar(cell); } catch (error) {
      error.row = i + 1; error.col = j + 1; throw error;
    }
  }));
}

export const EXAMPLES = {
  symmetric: [[4, 1, 1], [1, 3, 0], [1, 0, 2]],
  tall: [[1, 0], [0, 1], [1, 1]],
  wide: [[1, 0, 1], [0, 1, 1]],
  singular: [[1, 2, 3], [2, 4, 6], [1, 1, 1]],
  complex: [[0, -1], [1, 0]],
  jordan: [[1, 1], [0, 1]],
};
