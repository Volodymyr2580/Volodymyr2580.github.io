import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { analyzeMatrix } from '../src/engine.js';

const reference = JSON.parse(await readFile(new URL('./fixtures/reference.json', import.meta.url), 'utf8'));
const transpose = (a) => a[0].map((_, j) => a.map((row) => row[j]));
const multiply = (a, b) => a.map((row) => b[0].map((_, j) => row.reduce((sum, value, k) => sum + value * b[k][j], 0)));
const magnitude = (a) => Math.max(...a.flat().map(Math.abs));
const identity = (n) => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => Number(i === j)));
const complexMultiply = (a, b) => a.map((row) => b[0].map((_, j) => row.reduce((sum, v, k) => ({ re: sum.re + v.re * b[k][j].re - v.im * b[k][j].im, im: sum.im + v.re * b[k][j].im + v.im * b[k][j].re }), { re: 0, im: 0 })));
const asComplex = (a) => a.map((row) => row.map((re) => ({ re, im: 0 })));
const unpackComplex = (a) => a.map((row) => row.flatMap((v) => [v.re, v.im]));

function relativeError(actual, expected) {
  assert.equal(actual.length, expected.length, 'row count');
  actual.forEach((row, index) => {
    assert.equal(row.length, expected[index].length, 'column count');
    assert.ok(row.every(Number.isFinite), 'matrix must contain finite numbers');
  });
  const scale = magnitude(expected);
  if (!scale) return magnitude(actual);
  const error = Math.hypot(...actual.flat().map((value, index) => value / scale - expected.flat()[index] / scale));
  return error / Math.hypot(...expected.flat().map((value) => value / scale));
}

function matchSpectrum(actual, expected, matrixScale) {
  assert.equal(actual.length, expected.length);
  const remaining = actual.map((value) => {
    assert.ok(Number.isFinite(value.re) && Number.isFinite(value.im), 'eigenvalues must be finite');
    return value;
  });
  for (const value of expected) {
    const distances = remaining.map((candidate) => Math.hypot(candidate.re / (matrixScale || 1) - value.re / (matrixScale || 1), candidate.im / (matrixScale || 1) - value.im / (matrixScale || 1)));
    const best = Math.min(...distances);
    assert.ok(best < 2e-9, `spectrum mismatch for ${JSON.stringify(value)}: ${best}`);
    remaining.splice(distances.indexOf(best), 1);
  }
}

function eigenpairResidual(a, eig) {
  const scale = magnitude(a) || 1;
  const aScaled = a.map((row) => row.map((value) => value / scale));
  const vectors = eig.vectors;
  assert.equal(vectors.length, a.length);
  const errors = [];
  for (let column = 0; column < a.length; column++) {
    const lambda = { re: eig.values[column].re / scale, im: eig.values[column].im / scale };
    const norm = Math.hypot(...vectors.map((row) => Math.hypot(row[column].re, row[column].im)));
    assert.ok(norm > 0 && Number.isFinite(norm), 'each eigenvector must be nonzero and finite');
    for (let i = 0; i < a.length; i++) {
      const re = aScaled[i].reduce((sum, value, k) => sum + value * vectors[k][column].re, 0);
      const im = aScaled[i].reduce((sum, value, k) => sum + value * vectors[k][column].im, 0);
      const v = vectors[i][column];
      errors.push((re - (v.re * lambda.re - v.im * lambda.im)) / norm);
      errors.push((im - (v.re * lambda.im + v.im * lambda.re)) / norm);
    }
  }
  return Math.hypot(...errors) / Math.max(Math.hypot(...aScaled.flat()), 1);
}

for (const fixture of reference.cases) {
  test(`independent NumPy reference: ${fixture.name}`, () => {
    const a = fixture.matrix;
    const result = analyzeMatrix(a);
    assert.equal(result.rank, fixture.rank, 'numerical rank vs NumPy');
    assert.equal(result.rows, a.length);
    assert.equal(result.cols, a[0].length);

    if (fixture.inverse) {
      assert.ok(result.inverse, 'a numerically nonsingular square matrix must have an inverse');
      assert.ok(relativeError(result.inverse.values, fixture.inverse) < 2e-9, 'inverse vs NumPy');
    } else if (result.mode === 'square') {
      assert.equal(result.inverse, null, 'numerically singular square matrix must not report an ordinary inverse');
    }

    if (result.mode === 'rectangular' || fixture.inverse === null) {
      assert.ok(result.pseudoinverse, 'pseudoinverse must be available');
      const p = result.pseudoinverse.values;
      assert.ok(relativeError(p, fixture.pseudoinverse) < 2e-9, 'pseudoinverse vs NumPy');
      const ap = multiply(a, p);
      const pa = multiply(p, a);
      assert.ok(relativeError(multiply(ap, a), a) < 2e-9, 'A A+ A = A');
      assert.ok(relativeError(multiply(pa, p), p) < 2e-9, 'A+ A A+ = A+');
      assert.ok(relativeError(transpose(ap), ap) < 2e-9, '(A A+)ᵀ = A A+');
      assert.ok(relativeError(transpose(pa), pa) < 2e-9, '(A+ A)ᵀ = A+ A');
    }

    if (result.mode === 'rectangular') {
      const { u, s, vt, singularValues } = result.svd;
      const k = Math.min(a.length, a[0].length);
      assert.deepEqual([u.length, u[0].length, s.length, s[0].length, vt.length, vt[0].length], [a.length, k, k, k, k, a[0].length], 'economy SVD shapes');
      assert.ok(relativeError([singularValues], [fixture.singularValues]) < 2e-9, 'singular values vs NumPy');
      assert.ok(relativeError(multiply(multiply(u, s), vt), a) < 2e-9, 'SVD reconstruction');
      assert.ok(relativeError(multiply(transpose(u), u), identity(k)) < 2e-9, 'orthonormal U columns');
      assert.ok(relativeError(multiply(vt, transpose(vt)), identity(k)) < 2e-9, 'orthonormal V columns');
    } else {
      assert.ok(result.evd, `EVD must be returned: ${result.evdError ?? ''}`);
      matchSpectrum(result.evd.values, fixture.eigenvalues, magnitude(a));
      assert.ok(eigenpairResidual(a, result.evd) < 2e-9, 'complex A V = V Lambda identity');
      assert.equal(result.evd.diagonalizable, fixture.diagonalizable, 'reliable eigenvector basis classification');
      if (!fixture.diagonalizable) {
        assert.equal(result.evd.inverseVectors, null, 'defective eigenvector basis must not be inverted');
      } else {
        assert.ok(result.evd.inverseVectors, 'reliable basis has an inverse');
        const { vectors, inverseVectors, diagonal, values } = result.evd;
        for (let i = 0; i < a.length; i++) for (let j = 0; j < a.length; j++) {
          assert.deepEqual(diagonal[i][j], i === j ? values[i] : { re: 0, im: 0 }, 'Lambda must contain its eigenvalues on the diagonal');
        }
        assert.ok(relativeError(unpackComplex(complexMultiply(vectors, inverseVectors)), unpackComplex(asComplex(identity(a.length)))) < 2e-9, 'complex V V^-1 = I');
        const scale = magnitude(a) || 1;
        const diagonalScaled = diagonal.map((row) => row.map((value) => ({ re: value.re / scale, im: value.im / scale })));
        const reconstructed = complexMultiply(complexMultiply(vectors, diagonalScaled), inverseVectors);
        const expected = asComplex(a.map((row) => row.map((value) => value / scale)));
        assert.ok(relativeError(unpackComplex(reconstructed), unpackComplex(expected)) < 2e-9, 'complex V Lambda V^-1 reconstructs A');
      }
    }
  });
}
