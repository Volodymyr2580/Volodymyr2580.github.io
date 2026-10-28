import { Matrix, SingularValueDecomposition, EigenvalueDecomposition, inverse } from 'ml-matrix';

const EPSILON = Number.EPSILON;
const CONDITION_WARNING = 1 / Math.sqrt(EPSILON);
const MAX_DIMENSION = 50;
const MIN_MAGNITUDE = 1e-150;
const MAX_MAGNITUDE = 1e150;
const zero = (m, n) => Array.from({ length: m }, () => Array(n).fill(0));
const identity = (n) => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => Number(i === j)));
const transpose = (a) => a[0].map((_, j) => a.map((row) => row[j]));
const rescale = (a, scale) => a.map((row) => row.map((value) => value * scale));

function multiply(a, b) {
  const output = zero(a.length, b[0].length);
  for (let i = 0; i < a.length; i++) {
    for (let k = 0; k < b.length; k++) {
      for (let j = 0; j < b[0].length; j++) output[i][j] += a[i][k] * b[k][j];
    }
  }
  return output;
}

function norm(a) {
  let result = 0;
  for (const row of a) for (const value of row) result = Math.hypot(result, value);
  return result;
}

function relativeDifference(a, b) {
  let difference = 0;
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < a[0].length; j++) difference = Math.hypot(difference, a[i][j] - b[i][j]);
  }
  const denominator = norm(b);
  return denominator === 0 ? (difference === 0 ? 0 : Infinity) : difference / denominator;
}

function assertFiniteMatrix(a, label) {
  if (a.some((row) => row.some((value) => !Number.isFinite(value)))) throw new Error(`${label}: non-finite numerical output`);
  return a;
}

function validate(input) {
  if (!Array.isArray(input) || input.length < 1 || input.length > MAX_DIMENSION || !Array.isArray(input[0])) throw new Error('invalid-dimensions');
  const rows = input.length;
  const cols = input[0].length;
  if (cols < 1 || cols > MAX_DIMENSION) throw new Error('invalid-dimensions');
  let magnitude = 0;
  for (let i = 0; i < rows; i++) {
    if (!Array.isArray(input[i]) || input[i].length !== cols) throw new Error('invalid-shape');
    for (let j = 0; j < cols; j++) {
      const value = input[i][j];
      if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error(`invalid-entry:${i + 1},${j + 1}`);
      const absolute = Math.abs(value);
      if (absolute > MAX_MAGNITUDE || (absolute !== 0 && absolute < MIN_MAGNITUDE)) throw new Error(`entry-out-of-range:${i + 1},${j + 1}`);
      magnitude = Math.max(magnitude, absolute);
    }
  }
  return { rows, cols, scale: magnitude || 1 };
}

function thinSvd(scaled) {
  const svd = new SingularValueDecomposition(new Matrix(scaled), { autoTranspose: true });
  const k = Math.min(scaled.length, scaled[0].length);
  const u = svd.leftSingularVectors.to2DArray().map((row) => row.slice(0, k));
  const v = svd.rightSingularVectors.to2DArray().map((row) => row.slice(0, k));
  const singularValues = svd.diagonal.slice(0, k);
  assertFiniteMatrix(u, 'svd-u'); assertFiniteMatrix(v, 'svd-v');
  if (singularValues.some((value) => !Number.isFinite(value) || value < 0)) throw new Error('svd-invalid-spectrum');
  return { u, v, vt: transpose(v), singularValues };
}

function pseudoinverse(scaled, decomposition, cutoff, scale) {
  const { u, v, singularValues } = decomposition;
  const values = zero(scaled[0].length, scaled.length);
  for (let p = 0; p < singularValues.length; p++) {
    if (singularValues[p] <= cutoff) continue;
    const reciprocal = 1 / singularValues[p];
    for (let i = 0; i < values.length; i++) {
      for (let j = 0; j < values[0].length; j++) values[i][j] += v[i][p] * reciprocal * u[j][p];
    }
  }
  // Scaling cancels in the four Moore-Penrose identities, avoiding overflow
  // when users enter uniformly very large or very small matrices.
  const ax = multiply(scaled, values);
  const xa = multiply(values, scaled);
  const penroseResiduals = [
    relativeDifference(multiply(ax, scaled), scaled),
    relativeDifference(multiply(xa, values), values),
    relativeDifference(transpose(ax), ax),
    relativeDifference(transpose(xa), xa),
  ];
  return { values: assertFiniteMatrix(rescale(values, 1 / scale), 'pseudoinverse'), penroseResiduals };
}

const complex = (re = 0, im = 0) => ({ re, im });
const cabs = (a) => Math.hypot(a.re, a.im);
const cmul = (a, b) => complex(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
const csub = (a, b) => complex(a.re - b.re, a.im - b.im);
function cdivide(a, b) {
  // Ratios avoid squaring either a very small or a large pivot.
  if (Math.abs(b.re) >= Math.abs(b.im)) {
    const ratio = b.im / b.re;
    const denominator = b.re + b.im * ratio;
    return complex((a.re + a.im * ratio) / denominator, (a.im - a.re * ratio) / denominator);
  }
  const ratio = b.re / b.im;
  const denominator = b.im + b.re * ratio;
  return complex((a.re * ratio + a.im) / denominator, (a.im * ratio - a.re) / denominator);
}

function complexInverse(matrix) {
  const n = matrix.length;
  const augmented = matrix.map((row, i) => [...row.map((value) => ({ ...value })), ...Array.from({ length: n }, (_, j) => complex(Number(i === j)))]);
  for (let k = 0; k < n; k++) {
    let pivot = k;
    for (let i = k + 1; i < n; i++) if (cabs(augmented[i][k]) > cabs(augmented[pivot][k])) pivot = i;
    if (cabs(augmented[pivot][k]) <= EPSILON * n) throw new Error('eigenbasis-singular');
    [augmented[k], augmented[pivot]] = [augmented[pivot], augmented[k]];
    const divisor = { ...augmented[k][k] };
    for (let j = 0; j < 2 * n; j++) augmented[k][j] = cdivide(augmented[k][j], divisor);
    for (let i = 0; i < n; i++) {
      if (i === k) continue;
      const factor = { ...augmented[i][k] };
      for (let j = 0; j < 2 * n; j++) augmented[i][j] = csub(augmented[i][j], cmul(factor, augmented[k][j]));
    }
  }
  const result = augmented.map((row) => row.slice(n));
  if (result.some((row) => row.some((value) => !Number.isFinite(value.re) || !Number.isFinite(value.im)))) throw new Error('eigenbasis-inverse-invalid');
  return result;
}

function eigenbasisCondition(vectors) {
  const n = vectors.length;
  const hasComplex = vectors.some((row) => row.some((value) => value.im !== 0));
  // This real embedding has each complex singular value twice, so its
  // condition number is exactly the complex basis condition number.
  let realBasis;
  if (hasComplex) {
    realBasis = zero(2 * n, 2 * n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      realBasis[i][j] = realBasis[i + n][j + n] = vectors[i][j].re;
      realBasis[i + n][j] = vectors[i][j].im;
      realBasis[i][j + n] = -vectors[i][j].im;
    }
  } else realBasis = vectors.map((row) => row.map((value) => value.re));
  const singular = thinSvd(realBasis).singularValues;
  const cutoff = realBasis.length * EPSILON * singular[0];
  const fullRank = singular.every((value) => value > cutoff);
  return { fullRank, condition: fullRank ? singular[0] / singular.at(-1) : Infinity };
}

function eigendecomposition(scaled, scale, warnings) {
  const n = scaled.length;
  const evd = new EigenvalueDecomposition(new Matrix(scaled));
  const real = evd.realEigenvalues;
  const imaginary = evd.imaginaryEigenvalues;
  const blockVectors = assertFiniteMatrix(evd.eigenvectorMatrix.to2DArray(), 'eigenvectors');
  if ([...real, ...imaginary].some((value) => !Number.isFinite(value))) throw new Error('eigen-invalid-spectrum');
  const vectors = Array.from({ length: n }, () => Array.from({ length: n }, () => complex()));
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      if (imaginary[j] > 0) {
        if (j + 1 >= n || imaginary[j + 1] >= 0) throw new Error('eigen-invalid-conjugate-pair');
        vectors[i][j] = complex(blockVectors[i][j], blockVectors[i][j + 1]);
      } else if (imaginary[j] < 0) {
        if (j === 0 || imaginary[j - 1] <= 0) throw new Error('eigen-invalid-conjugate-pair');
        vectors[i][j] = complex(blockVectors[i][j - 1], -blockVectors[i][j]);
      } else vectors[i][j] = complex(blockVectors[i][j]);
    }
    let length = 0;
    for (let i = 0; i < n; i++) length = Math.hypot(length, vectors[i][j].re, vectors[i][j].im);
    if (!Number.isFinite(length) || length === 0) throw new Error('eigen-invalid-vector');
    for (let i = 0; i < n; i++) vectors[i][j] = complex(vectors[i][j].re / length, vectors[i][j].im / length);
  }

  let difference = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    let left = complex();
    for (let k = 0; k < n; k++) {
      left.re += scaled[i][k] * vectors[k][j].re;
      left.im += scaled[i][k] * vectors[k][j].im;
    }
    const right = cmul(vectors[i][j], complex(real[j], imaginary[j]));
    difference = Math.hypot(difference, left.re - right.re, left.im - right.im);
  }
  const denominator = norm(scaled) * Math.sqrt(n);
  const residual = denominator === 0 ? (difference === 0 ? 0 : Infinity) : difference / denominator;
  const basis = eigenbasisCondition(vectors);
  let diagonalizable = basis.fullRank && basis.condition <= CONDITION_WARNING && residual < 1e-8;
  if (!basis.fullRank) warnings.push('not-diagonalizable');
  else if (basis.condition > CONDITION_WARNING) warnings.push('eigenbasis-ill-conditioned');
  if (residual >= 1e-8) warnings.push('eigen-residual');
  let inverseVectors = null;
  if (diagonalizable) {
    try { inverseVectors = complexInverse(vectors); }
    catch { diagonalizable = false; warnings.push('eigenbasis-ill-conditioned'); }
  }
  const values = real.map((value, j) => complex(value * scale, imaginary[j] * scale));
  const diagonal = values.map((value, i) => values.map((_, j) => i === j ? { ...value } : complex()));
  return { values, vectors, diagonal, inverseVectors, residual, diagonalizable, basisCondition: basis.condition };
}

/**
 * Real finite matrices only. Dimensions 1..50; nonzero entries 1e-150..1e150.
 * Residuals use Frobenius norms, computed after a shared scale normalization.
 * Numerical rank uses max(m,n) * Number.EPSILON * sigmaMax. A false
 * `diagonalizable` means a reliable numerical diagonalization was not found;
 * this floating-point test is not a symbolic proof of defectiveness.
 */
export function analyzeMatrix(input) {
  const { rows, cols, scale } = validate(input);
  const scaled = input.map((row) => row.map((value) => value / scale));
  const decomposition = thinSvd(scaled);
  const relativeTolerance = Math.max(rows, cols) * EPSILON;
  const cutoff = relativeTolerance * decomposition.singularValues[0];
  const rank = decomposition.singularValues.filter((value) => value > cutoff).length;
  const k = Math.min(rows, cols);
  const conditionNumber = rank < k ? Infinity : decomposition.singularValues[0] / decomposition.singularValues.at(-1);
  const result = {
    rows, cols, mode: rows === cols ? 'square' : 'rectangular', rank,
    relativeTolerance, tolerance: cutoff * scale, conditionNumber,
    warnings: [], inverse: null, pseudoinverse: null, evd: null, svd: null,
  };
  if (rank < k) result.warnings.push('rank-deficient');
  else if (conditionNumber > CONDITION_WARNING) result.warnings.push('ill-conditioned');

  if (rows === cols) {
    if (rank === rows) {
      try {
        const scaledInverse = assertFiniteMatrix(inverse(new Matrix(scaled)).to2DArray(), 'inverse');
        const expected = identity(rows);
        result.inverse = {
          values: assertFiniteMatrix(rescale(scaledInverse, 1 / scale), 'inverse'),
          leftResidual: relativeDifference(multiply(scaled, scaledInverse), expected),
          rightResidual: relativeDifference(multiply(scaledInverse, scaled), expected),
        };
      } catch { result.warnings.push('inverse-failed'); }
    }
    if (!result.inverse) result.pseudoinverse = pseudoinverse(scaled, decomposition, cutoff, scale);
    try { result.evd = eigendecomposition(scaled, scale, result.warnings); }
    catch (error) { result.evdError = error.message; result.warnings.push('eigen-failed'); }
  } else {
    result.pseudoinverse = pseudoinverse(scaled, decomposition, cutoff, scale);
    const scaledS = decomposition.singularValues.map((value, i) => decomposition.singularValues.map((_, j) => i === j ? value : 0));
    result.svd = {
      u: decomposition.u,
      s: rescale(scaledS, scale),
      vt: decomposition.vt,
      singularValues: decomposition.singularValues.map((value) => value * scale),
      residual: relativeDifference(multiply(multiply(decomposition.u, scaledS), decomposition.vt), scaled),
      orthogonalityU: relativeDifference(multiply(transpose(decomposition.u), decomposition.u), identity(k)),
      orthogonalityV: relativeDifference(multiply(decomposition.vt, decomposition.v), identity(k)),
    };
  }
  result.warnings = [...new Set(result.warnings)];
  return result;
}
