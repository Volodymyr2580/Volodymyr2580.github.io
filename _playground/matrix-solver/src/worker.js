import { analyzeMatrix } from './engine.js';
self.onmessage = (event) => {
  const { id, matrix } = event.data;
  const start = performance.now();
  try { self.postMessage({ id, result: analyzeMatrix(matrix), milliseconds: performance.now() - start }); }
  catch (error) { self.postMessage({ id, error: error.message || 'calculation-failed' }); }
};
