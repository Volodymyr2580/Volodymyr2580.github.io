export const escapeHTML = (value) => String(value).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
function real(value, digits, full) {
  if (value === Infinity) return '∞';
  if (value === -Infinity) return '−∞';
  if (!Number.isFinite(value)) return '—';
  if (value === 0) return '0';
  if (full) return String(value);
  return String(Number(value.toPrecision(digits)));
}
export function numberText(value, digits = 6, full = false) {
  if (typeof value === 'number') return real(value, digits, full);
  const { re, im } = value;
  if (im === 0) return real(re, digits, full);
  const imaginary = `${Math.abs(im) === 1 ? '' : real(Math.abs(im), digits, full)}i`;
  if (re === 0) return `${im < 0 ? '-' : ''}${imaginary}`;
  return `${real(re, digits, full)} ${im < 0 ? '-' : '+'} ${imaginary}`;
}
export const matrixText = (data) => data.map((row) => row.map((value) => numberText(value, 17, true)).join('\t')).join('\n');

export function resultView(result, lang, digits) {
  const pick = (zh, en) => lang === 'en' ? en : zh;
  const text = (value) => numberText(value, digits);
  const matrices = {};
  const matrix = (id, label, symbol, values, description = '') => {
    matrices[id] = { label: `${label} (${symbol})`, values };
    return `<section class="result-section" data-result="${id}"><div class="result-heading"><div><h3>${label}</h3><span class="formula">${symbol}</span><span class="matrix-dimensions">${values.length} × ${values[0].length}</span></div><button type="button" class="matrix-copy" data-copy="${id}">${pick('复制矩阵', 'Copy matrix')}</button></div><div class="matrix-wrap" tabindex="0" role="region" aria-label="${escapeHTML(label)}"><table class="matrix-table" aria-label="${escapeHTML(label)}"><tbody>${values.map((row) => `<tr>${row.map((value) => `<td title="${escapeHTML(numberText(value, 17, true))}">${escapeHTML(text(value))}</td>`).join('')}</tr>`).join('')}</tbody></table></div>${description ? `<p class="result-note">${description}</p>` : ''}</section>`;
  };
  const metric = (label, value) => `<span class="metric${value > 1e-8 || !Number.isFinite(value) ? ' warning' : ''}">${label}<strong>${real(value, 3, false)}</strong></span>`;
  const warnings = {
    'rank-deficient': pick('当前阈值下数值秩不足。伪逆会舍弃不超过阈值的奇异值。', 'Numerically rank deficient at the stated cutoff. The pseudoinverse discards singular values at or below it.'),
    'ill-conditioned': pick('矩阵的条件数较大，微小输入误差可能被放大；低残差不保证结果对扰动稳定。', 'This matrix is ill-conditioned. Small input errors may be amplified; a small residual does not guarantee stability.'),
    'not-diagonalizable': pick('未找到足够独立的特征向量，无法可靠对角化（数值判断）。仍可查看特征值及各列特征向量。', 'No numerically independent eigenbasis was found. A reliable diagonalization is unavailable; eigenvalues and individual eigenvectors remain available.'),
    'eigenbasis-ill-conditioned': pick('特征向量组成的矩阵过于病态，不展示不稳定的对角化重建。', 'The eigenvector matrix is too ill-conditioned for a reliable diagonalization.'),
    'eigen-residual': pick('特征值方程的残差偏大，请谨慎使用这组特征向量。', 'The eigen-equation residual is large. Treat these eigenvectors with caution.'),
    'inverse-failed': pick('未能可靠计算普通逆矩阵，以下单独给出伪逆。', 'A reliable ordinary inverse could not be computed. A pseudoinverse is provided separately.'),
    'eigen-failed': pick('特征值计算未能完成；已保留可用的求逆结果。', 'The eigenvalue calculation failed. Available inverse results are preserved.'),
  };
  let html = `<div class="summary-grid"><div class="summary-item"><small>${pick('矩阵形状', 'Shape')}</small><strong>${result.rows} × ${result.cols}</strong></div><div class="summary-item"><small>${pick('数值秩', 'Numerical rank')}</small><strong>${result.rank} / ${Math.min(result.rows, result.cols)}</strong></div><div class="summary-item"><small>${pick('2-范数条件数', '2-norm condition')}</small><strong>${text(result.conditionNumber)}</strong></div></div>`;
  html += result.warnings.map((code) => `<p class="notice warning" data-warning="${code}">${warnings[code] || escapeHTML(code)}</p>`).join('');
  if (result.inverse) {
    html += matrix('inverse', pick('逆矩阵', 'Inverse'), 'A⁻¹', result.inverse.values);
    html += `<div class="metrics">${metric('‖AA⁻¹ − I‖F / ‖I‖F', result.inverse.leftResidual)}${metric('‖A⁻¹A − I‖F / ‖I‖F', result.inverse.rightResidual)}</div>`;
  } else if (result.mode === 'square') {
    html += `<p class="notice" data-result="inverse-unavailable">${pick('当前数值阈值下无法给出可靠的普通逆矩阵。A⁺ 是伪逆，不应当作 A⁻¹ 使用。', 'No reliable ordinary inverse is available at this numerical cutoff. A⁺ below is a pseudoinverse, not A⁻¹.')}</p>`;
  }
  if (result.pseudoinverse) html += matrix('pseudoinverse', pick('Moore–Penrose 伪逆', 'Moore–Penrose pseudoinverse'), 'A⁺', result.pseudoinverse.values, pick('使用截断奇异值分解计算；保留大于数值阈值的奇异值。', 'Computed from SVD, retaining only singular values above the numerical cutoff.'));
  if (result.evd) {
    const e = result.evd;
    html += `<section class="result-section"><div class="result-heading"><h3>${pick('特征值分解', 'Eigendecomposition')}</h3><span class="formula">${e.diagonalizable ? 'A = VΛV⁻¹' : 'AV ≈ VΛ'}</span></div><p class="result-note">${pick('特征向量按列排列，与下方特征值顺序一一对应。', 'Columns of V correspond to the eigenvalues in the order below.')}</p><div class="eigen-values">${e.values.map((value, i) => `<span><small>λ<sub>${i + 1}</sub></small><b>${escapeHTML(text(value))}</b></span>`).join('')}</div><div class="metrics">${metric('‖AV − VΛ‖F / (‖A‖F ‖V‖F)', e.residual)}</div></section>`;
    html += matrix('eigenvectors', pick('特征向量矩阵', 'Eigenvector matrix'), 'V', e.vectors);
    html += matrix('eigenvalues', pick('特征值对角矩阵', 'Eigenvalue diagonal matrix'), 'Λ', e.diagonal);
    if (e.inverseVectors) html += `<details class="extra-matrix"><summary>${pick('查看特征向量矩阵的逆 V⁻¹', 'Show the inverse eigenvector matrix V⁻¹')}</summary>${matrix('inverse-eigenvectors', pick('特征向量矩阵的逆', 'Inverse eigenvector matrix'), 'V⁻¹', e.inverseVectors)}</details>`;
  }
  if (result.svd) {
    const s = result.svd;
    html += `<section class="result-section"><div class="result-heading"><h3>${pick('奇异值分解', 'Singular value decomposition')}</h3><span class="formula">A = UΣVᵀ</span></div><p class="result-note">${pick('紧凑形状：k = min(m, n)。U 为 m×k，Σ 为 k×k，Vᵀ 为 k×n。零奇异值仍列出。', 'Economy-sized factors: k = min(m, n); U is m×k, Σ is k×k, and Vᵀ is k×n. Zero singular values are retained in these factors.')}</p><div class="spectrum">${s.singularValues.slice(0, 12).map((value, i) => `<div class="spectrum-row"><span class="spectrum-label">σ${i + 1}</span><span class="spectrum-track"><i class="spectrum-bar" style="width:${s.singularValues[0] === 0 ? 0 : Math.max(0, value / s.singularValues[0] * 100)}%"></i></span><span class="spectrum-value">${text(value)}</span></div>`).join('')}</div>${s.singularValues.length > 12 ? `<p class="result-note">${pick('图中显示前 12 个奇异值；Σ 包含完整谱。', 'The first 12 singular values are plotted; Σ contains the full spectrum.')}</p>` : ''}<div class="metrics">${metric('‖UΣVᵀ − A‖F / ‖A‖F', s.residual)}${metric('‖UᵀU − I‖F / ‖I‖F', s.orthogonalityU)}${metric('‖VᵀV − I‖F / ‖I‖F', s.orthogonalityV)}</div></section>`;
    html += matrix('u', pick('左奇异向量', 'Left singular vectors'), 'U', s.u);
    html += matrix('sigma', pick('奇异值对角矩阵', 'Singular value diagonal matrix'), 'Σ', s.s);
    html += matrix('vt', pick('右奇异向量的转置', 'Transposed right singular vectors'), 'Vᵀ', s.vt);
  }
  html += `<details class="numerical-details"><summary>${pick('数值阈值与校验', 'Numerical cutoff and checks')}</summary><p class="result-note">${pick('数值秩采用 σ > τ，τ = max(m,n) × ε × σ₁，ε 为双精度机器精度。', 'Rank uses σ > τ, where τ = max(m,n) × ε × σ₁ and ε is double-precision machine epsilon.')}<br>τ = ${real(result.tolerance, 6, false)}; ε = ${Number.EPSILON}</p>`;
  if (result.evd) html += `<p class="result-note">κ₂(V) = ${text(result.evd.basisCondition)}. ${pick('数值可对角化判定不等于符号证明。', 'The numerical diagonalizability check is not a symbolic proof.')}</p>`;
  if (result.pseudoinverse) {
    html += `<p class="result-note">${pick('四个 Moore–Penrose 条件的相对 Frobenius 残差（分母为等式右侧范数）：', 'Relative Frobenius residuals of the four Moore–Penrose conditions (normalized by the right-hand side):')}</p><div class="metrics">`;
    const labels = ['AA⁺A = A', 'A⁺AA⁺ = A⁺', '(AA⁺)ᵀ = AA⁺', '(A⁺A)ᵀ = A⁺A'];
    html += result.pseudoinverse.penroseResiduals.map((value, i) => metric(labels[i], value)).join('') + '</div>';
  }
  html += `<p class="result-note">${pick('输入整体缩放后进行计算，以减轻上溢与下溢。零矩阵的精确零残差记为 0。显示位数只影响排版；复制和 JSON 下载保留完整计算精度。', 'Calculations use a globally scaled matrix to reduce overflow and underflow. Exact zero-matrix residuals are reported as 0. Display precision only affects formatting; copy and JSON retain full computed precision.')}</p></details>`;
  return { html, matrices };
}
