import test from 'node:test';
import assert from 'node:assert/strict';
import { scalar, dimension, parseMatrixText } from '../src/input.js';

test('dimensions require whole numbers in the supported range', () => {
  assert.equal(dimension(' 50 '), 50);
  for (const value of ['', 0, -1, 51, 1.5, '2e1', 'Infinity']) assert.throws(() => dimension(value));
});
test('numeric fields accept decimals, scientific notation and one fraction without evaluation', () => {
  assert.equal(scalar(' -1.25e-3 '), -.00125);
  assert.equal(scalar('1 / 3'), 1 / 3);
  assert.equal(scalar('2e2/4e1'), 5);
  assert.equal(Object.is(scalar('-0'), -0), false);
  for (const value of ['', '1/0', '1//2', '1+2', '2i', 'NaN', 'Infinity', '1,000', 'alert(1)', '1e200', '1e-200', '1e-999', '1e999/1e999', '1e-100/1e250']) assert.throws(() => scalar(value), value);
});
test('bulk paste supports Excel, whitespace and MATLAB row separators', () => {
  const wanted = [[1, 2], [3, 4]];
  assert.deepEqual(parseMatrixText('1\t2\r\n3\t4\r\n', 2, 2), wanted);
  assert.deepEqual(parseMatrixText('[1, 2; 3, 4]', 2, 2), wanted);
  assert.deepEqual(parseMatrixText('1  2\n3 4', 2, 2), wanted);
  assert.throws(() => parseMatrixText('1,;2,3', 2, 2));
  assert.throws(() => parseMatrixText('1 2 3\n4 5 6', 2, 2));
  for (const text of ['1\t\t2', '\t1\t2', '1\t2\t']) {
    assert.throws(() => parseMatrixText(text, 1, 2), /shape/);
    assert.throws(() => parseMatrixText(text, 1, 3), /empty/);
  }
});
