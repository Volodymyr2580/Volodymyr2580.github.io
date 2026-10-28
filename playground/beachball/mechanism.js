/* Independent double-couple geometry; NED coordinates, Aki–Richards convention.
 * Reference/validation: ObsPy 1.4.2 MoPaD. See _playground/beachball/README.md.
 * No third-party implementation is bundled in this browser module.
 */
(function (root) {
  'use strict';
  const RAD = Math.PI / 180;
  const dot = (a, b) => a.reduce((sum, x, i) => sum + x * b[i], 0);
  const scale = (a, k) => a.map(x => x * k);
  const unit = a => scale(a, 1 / Math.hypot(...a));
  const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

  function mechanism(strike, dip, rake) {
    if (![strike, dip, rake].every(Number.isFinite) || strike < 0 || strike > 360 || dip < 0 || dip > 90 || rake < -180 || rake > 180) {
      throw new RangeError('Strike: 0–360°, dip: 0–90°, rake: −180–180°.');
    }
    const f = strike * RAD, d = dip * RAD, l = rake * RAD;
    const n = [-Math.sin(d)*Math.sin(f), Math.sin(d)*Math.cos(f), -Math.cos(d)];
    const s = [Math.cos(l)*Math.cos(f)+Math.sin(l)*Math.cos(d)*Math.sin(f),
      Math.cos(l)*Math.sin(f)-Math.sin(l)*Math.cos(d)*Math.cos(f), -Math.sin(l)*Math.sin(d)];
    const matrix = n.map((x, i) => n.map((y, j) => x*s[j]+s[i]*y));
    return { strike, dip, rake, n, s, matrix,
      p: unit(s.map((x, i) => x-n[i])), t: unit(s.map((x, i) => x+n[i])) };
  }

  // Axis signs are arbitrary. Pick a stable representative on the lower hemisphere.
  function lower(v) {
    let a = unit(v);
    if (Math.abs(a[2]) < 1e-12) {
      a[2] = 0;
      if (a[0] < -1e-12 || (Math.abs(a[0]) < 1e-12 && a[1] < 0)) a = scale(a, -1);
    } else if (a[2] < 0) a = scale(a, -1);
    return a;
  }

  // Lambert azimuthal equal-area, disk radius 1; x east, y screen-down (south).
  // This primitive does not flip equatorial endpoints; nodal arcs must stay continuous.
  function project(v) {
    const d = clamp(v[2], 0, 1);
    return [v[1] / Math.sqrt(1+d), -v[0] / Math.sqrt(1+d)];
  }

  function unproject(x, y) {
    const r2 = x*x+y*y;
    if (!Number.isFinite(r2) || r2 > 1+1e-12) return null;
    const k = Math.sqrt(2-clamp(r2, 0, 1));
    return [-y*k, x*k, 1-clamp(r2, 0, 1)];
  }

  function direction(azimuth, takeoff) {
    const a = azimuth*RAD, i = takeoff*RAD;
    return [Math.sin(i)*Math.cos(a), Math.sin(i)*Math.sin(a), Math.cos(i)];
  }

  function orientation(v) {
    const a = lower(v);
    return { vector: a, azimuth: Math.hypot(a[0], a[1]) < 1e-10 ? null :
      (Math.atan2(a[1], a[0])/RAD+360)%360, plunge: Math.asin(clamp(a[2], 0, 1))/RAD };
  }

  const amplitude = (m, r) => dot(r, m.matrix.map(row => dot(row, r)));
  const polarity = a => Math.abs(a) < 1e-7 ? 0 : Math.sign(a);

  // Boundary of {r: down >= 0 and v·r >= 0}. The nodal semicircle plus the
  // positive equatorial semicircle form a closed, projected polygon. Intersect
  // same-sign half-disks for A = 2(n·r)(s·r) > 0. This avoids raster contours.
  function halfDisk(v, steps = 360) {
    const horizontal = Math.hypot(v[0], v[1]);
    if (horizontal < 1e-12) {
      const circle = Array.from({length: steps*2+1}, (_, k) => [Math.cos(k*Math.PI/steps), Math.sin(k*Math.PI/steps)]);
      return { region: v[2] > 0 ? circle : [], node: circle };
    }
    const a = [-v[1]/horizontal, v[0]/horizontal, 0];
    let b = unit(cross(v, a));
    if (b[2] < 0) b = scale(b, -1);
    const c = [v[0]/horizontal, v[1]/horizontal, 0];
    const node = Array.from({length: steps+1}, (_, k) => {
      const t = k*Math.PI/steps;
      return project(a.map((x, i) => x*Math.cos(t)+b[i]*Math.sin(t)));
    });
    const rim = Array.from({length: steps}, (_, k) => {
      const t = (k+1)*Math.PI/steps;
      return project(a.map((x, i) => -x*Math.cos(t)+c[i]*Math.sin(t)));
    });
    return {region: node.concat(rim), node};
  }

  const api = { mechanism, lower, project, unproject, direction, orientation, amplitude, polarity, halfDisk, dot };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.Beachball = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
