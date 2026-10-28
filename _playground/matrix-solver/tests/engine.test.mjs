import test from 'node:test';
import assert from 'node:assert/strict';
import { analyzeMatrix } from '../src/engine.js';

const close = (actual, expected, tolerance = 1e-11) => assert.ok(Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)), `${actual} differs from ${expected}`);
function matrixClose(actual, expected, tolerance = 1e-11) {
  assert.equal(actual.length, expected.length);
  actual.forEach((row, i) => {
    assert.equal(row.length, expected[i].length);
    row.forEach((value, j) => close(value, expected[i][j], tolerance));
  });
}
function complexMultiply(a, b) {
  return a.map((row) => b[0].map((_, j) => row.reduce((sum, value, k) => ({
    re: sum.re + value.re * b[k][j].re - value.im * b[k][j].im,
    im: sum.im + value.re * b[k][j].im + value.im * b[k][j].re,
  }), { re: 0, im: 0 })));
}
function reconstruction(result, original) {
  const { vectors, diagonal, inverseVectors } = result.evd;
  const actual = complexMultiply(complexMultiply(vectors, diagonal), inverseVectors);
  actual.forEach((row, i) => row.forEach((value, j) => { close(value.re, original[i][j]); close(value.im, 0); }));
}

test('symmetric square: analytic inverse, eigenvalues, eigenvector reconstruction', () => {
  const a = [[4, 1], [1, 3]];
  const result = analyzeMatrix(a);
  assert.equal(result.mode, 'square'); assert.equal(result.rank, 2); assert.equal(result.svd, null);
  matrixClose(result.inverse.values, [[3 / 11, -1 / 11], [-1 / 11, 4 / 11]]);
  close(result.evd.values[0].re, (7 - Math.sqrt(5)) / 2);
  close(result.evd.values[1].re, (7 + Math.sqrt(5)) / 2);
  assert.ok(result.inverse.leftResidual < 1e-14 && result.inverse.rightResidual < 1e-14);
  reconstruction(result, a);
});

test('nonsymmetric real diagonalizable matrix gives a true eigenbasis inverse', () => {
  const a = [[1, 3], [0, 2]];
  const result = analyzeMatrix(a);
  matrixClose(result.inverse.values, [[1, -1.5], [0, .5]]);
  assert.deepEqual(result.evd.values.map((value) => value.re).sort(), [1, 2]);
  assert.equal(result.evd.diagonalizable, true); reconstruction(result, a);
});

test('real rotation produces conjugate complex eigenvectors with correct signs', () => {
  const a = [[0, -1], [1, 0]];
  const result = analyzeMatrix(a);
  assert.deepEqual(result.evd.values, [{ re: 0, im: 1 }, { re: 0, im: -1 }]);
  assert.ok(result.evd.residual < 1e-14);
  close(result.evd.basisCondition, 1);
  reconstruction(result, a);
});

test('Jordan block does not claim a reliable diagonalization', () => {
  const result = analyzeMatrix([[1, 1], [0, 1]]);
  assert.equal(result.evd.diagonalizable, false); assert.equal(result.evd.inverseVectors, null);
  assert.ok(result.warnings.includes('not-diagonalizable') || result.warnings.includes('eigenbasis-ill-conditioned'));
  assert.ok(result.inverse);
});

test('repeated eigenvalues alone do not imply a defective matrix', () => {
  const a = [[3, 0, 0], [0, 3, 0], [0, 0, 3]];
  const result = analyzeMatrix(a);
  assert.equal(result.evd.diagonalizable, true);
  reconstruction(result, a);
});

test('singular square reports no ordinary inverse and a Moore-Penrose fallback', () => {
  const result = analyzeMatrix([[1, 2], [2, 4]]);
  assert.equal(result.inverse, null); assert.equal(result.rank, 1);
  assert.ok(result.warnings.includes('rank-deficient'));
  matrixClose(result.pseudoinverse.values, [[1 / 25, 2 / 25], [2 / 25, 4 / 25]]);
  result.pseudoinverse.penroseResiduals.forEach((value) => assert.ok(value < 1e-14));
});

test('single-entry matrices include zero without NaN outputs', () => {
  const zero = analyzeMatrix([[0]]);
  assert.equal(zero.inverse, null); matrixClose(zero.pseudoinverse.values, [[0]]);
  assert.deepEqual(zero.pseudoinverse.penroseResiduals, [0, 0, 0, 0]);
  assert.equal(zero.evd.residual, 0); assert.equal(zero.evd.diagonalizable, true);
  const nonzero = analyzeMatrix([[-4]]);
  matrixClose(nonzero.inverse.values, [[-.25]]); close(nonzero.evd.values[0].re, -4);
});

test('tall full-rank matrix: known pseudoinverse and economy factors', () => {
  const result = analyzeMatrix([[1, 0], [0, 1], [1, 1]]);
  assert.equal(result.mode, 'rectangular'); assert.equal(result.inverse, null); assert.equal(result.evd, null);
  matrixClose(result.pseudoinverse.values, [[2 / 3, -1 / 3, 1 / 3], [-1 / 3, 2 / 3, 1 / 3]]);
  assert.equal(result.svd.u.length, 3); assert.equal(result.svd.u[0].length, 2);
  assert.equal(result.svd.vt.length, 2); assert.equal(result.svd.vt[0].length, 2);
  close(result.svd.singularValues[0], Math.sqrt(3)); close(result.svd.singularValues[1], 1);
  assert.ok(result.svd.residual < 1e-14);
});

test('wide full-rank matrix uses the transposed economy SVD consistently', () => {
  const result = analyzeMatrix([[1, 0, 1], [0, 1, 1]]);
  matrixClose(result.pseudoinverse.values, [[2 / 3, -1 / 3], [-1 / 3, 2 / 3], [1 / 3, 1 / 3]]);
  assert.equal(result.svd.u.length, 2); assert.equal(result.svd.u[0].length, 2);
  assert.equal(result.svd.vt.length, 2); assert.equal(result.svd.vt[0].length, 3);
  assert.ok(result.svd.residual < 1e-14);
  assert.ok(result.svd.orthogonalityU < 1e-14 && result.svd.orthogonalityV < 1e-14);
});

test('rectangular rank-one and zero matrices satisfy all four Penrose identities', () => {
  const result = analyzeMatrix([[1, 2, 3], [2, 4, 6]]);
  assert.equal(result.rank, 1); assert.equal(result.conditionNumber, Infinity);
  matrixClose(result.pseudoinverse.values, [[1 / 70, 2 / 70], [2 / 70, 4 / 70], [3 / 70, 6 / 70]]);
  result.pseudoinverse.penroseResiduals.forEach((value) => assert.ok(value < 1e-14));
  const zero = analyzeMatrix([[0, 0, 0], [0, 0, 0]]);
  assert.equal(zero.rank, 0); assert.deepEqual(zero.svd.singularValues, [0, 0]);
  matrixClose(zero.pseudoinverse.values, [[0, 0], [0, 0], [0, 0]]);
  assert.equal(zero.svd.residual, 0);
  assert.ok(zero.svd.orthogonalityU < 1e-14 && zero.svd.orthogonalityV < 1e-14);
});

test('single row and column SVDs preserve dimensions and coefficients', () => {
  matrixClose(analyzeMatrix([[3, 4]]).pseudoinverse.values, [[3 / 25], [4 / 25]]);
  matrixClose(analyzeMatrix([[3], [4]]).pseudoinverse.values, [[3 / 25, 4 / 25]]);
});

test('uniform scaling protects inverse, EVD, SVD and relative residuals', () => {
  for (const scale of [1e-150, 1e149]) {
    const square = analyzeMatrix([[2 * scale, 0], [0, 4 * scale]]);
    close(square.inverse.values[0][0] * scale, .5);
    close(square.inverse.values[1][1] * scale, .25);
    close(square.evd.values[0].re / scale, 2); close(square.evd.values[1].re / scale, 4);
    const rectangular = analyzeMatrix([[3 * scale, 4 * scale]]);
    close(rectangular.pseudoinverse.values[0][0] * scale, 3 / 25);
    close(rectangular.svd.singularValues[0] / scale, 5);
    assert.ok(rectangular.svd.residual < 1e-14);
    rectangular.pseudoinverse.penroseResiduals.forEach((value) => assert.ok(value < 1e-14));
  }
});

test('rank cutoff is scale relative and ill-conditioning is explicit', () => {
  const near = analyzeMatrix([[1, 0], [0, 1e-12]]);
  assert.equal(near.rank, 2); assert.ok(near.inverse); assert.ok(near.warnings.includes('ill-conditioned'));
  const unresolved = analyzeMatrix([[1, 0], [0, 1e-18]]);
  assert.equal(unresolved.rank, 1); assert.equal(unresolved.inverse, null);
  close(unresolved.relativeTolerance, 2 * Number.EPSILON);
  close(unresolved.tolerance, 2 * Number.EPSILON);
});

test('input is never modified and results are structured-cloneable', () => {
  const input = [[1, 2, 4], [3, 5, 7]];
  const before = structuredClone(input);
  const result = analyzeMatrix(input);
  assert.deepEqual(input, before);
  assert.deepEqual(structuredClone(result), result);
});

test('invalid shapes, nonreal values and unsupported magnitudes are rejected', () => {
  for (const value of [[], [[]], [[1], [2, 3]], [[NaN]], [[Infinity]], [['2']], [[true]], [[{ re: 1, im: 0 }]], [[1e151]], [[1e-151]], Array.from({ length: 51 }, () => [1])]) assert.throws(() => analyzeMatrix(value));
});
