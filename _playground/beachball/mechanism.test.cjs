const test = require('node:test');
const assert = require('node:assert/strict');
const B = require('../../playground/beachball/mechanism.js');
const reference = require('./reference.json');
const close = (a, b, eps = 1e-10) => assert.ok(Math.abs(a-b) < eps, `${a} != ${b}`);

test('110 mechanisms match independent ObsPy NED tensors and eigensystem axes', () => {
  for (const record of reference.cases) {
    const m = B.mechanism(...record.sdr);
    m.matrix.forEach((row, i) => row.forEach((x, j) => close(x, record.matrix[i][j])));
    close(Math.abs(B.dot(m.p, record.p)), 1);
    close(Math.abs(B.dot(m.t, record.t)), 1);
    close(B.amplitude(m, m.p), -1); close(B.amplitude(m, m.t), 1);
    close(B.dot(m.n, m.s), 0);
  }
});

test('normal / reverse center polarity and strike-slip quadrants', () => {
  close(B.amplitude(B.mechanism(0,45,-90), [0,0,1]), -1);
  close(B.amplitude(B.mechanism(0,45,90), [0,0,1]), 1);
  const m = B.mechanism(0,90,0);
  for (const [az, sign] of [[45,1],[135,-1],[225,1],[315,-1]]) {
    close(B.amplitude(m, B.direction(az,90)), sign);
  }
  close(B.orientation(B.mechanism(0,45,-90).p).plunge, 90);
  assert.equal(B.orientation(B.mechanism(0,45,-90).p).azimuth, null);
});

test('equal-area projection round trips, cardinal orientation, horizontal axes', () => {
  for (let a=0;a<360;a+=17) for (let i=0;i<=90;i+=3) {
    const v=B.direction(a,i), p=B.project(v), roundtrip=B.unproject(...p);
    v.forEach((x,k)=>close(x, roundtrip[k]));
  }
  close(B.project([1,0,0])[1], -1); close(B.project([0,1,0])[0], 1);
  assert.equal(B.unproject(1.1,0), null);
  assert.equal(B.unproject(NaN,0), null);
  const a = B.lower([0,-1,0]), b=B.lower([0,1,0]);
  a.forEach((x,i)=>close(x,b[i]));
});

// Point-in-polygon tests validate the actual region geometry, independently of
// SVG clip-path painting. Exclude only a narrow band at the sampled boundary.
function inside(point, poly) {
  let result=false;
  for(let i=0,j=poly.length-1;i<poly.length;j=i++) {
    const [xi,yi]=poly[i], [xj,yj]=poly[j], [x,y]=point;
    if ((yi>y)!==(yj>y) && x<(xj-xi)*(y-yi)/(yj-yi)+xi) result=!result;
  }
  return result;
}
test('projected SVG fill regions match radiation signs including degenerate planes', () => {
  let checks=0;
  for(const record of reference.cases) {
    const m=B.mechanism(...record.sdr);
    const [np,sp,nn,sn]=[m.n,m.s,m.n.map(x=>-x),m.s.map(x=>-x)].map(v=>B.halfDisk(v).region);
    for(let az=3;az<360;az+=19) for(let takeoff=2;takeoff<90;takeoff+=11) {
      const v=B.direction(az,takeoff), a=B.amplitude(m,v);
      if(Math.abs(a)<.002) continue;
      const p=B.project(v);
      const filled=(inside(p,np)&&inside(p,sp))||(inside(p,nn)&&inside(p,sn));
      assert.equal(filled,a>0,`Fill mismatch SDR ${record.sdr}, ray ${az}/${takeoff}`); checks++;
    }
  }
  assert.ok(checks>15000);
});

test('nodal curves are zeros of radiation and remain continuous', () => {
  for(const record of reference.cases) {
    const m=B.mechanism(...record.sdr);
    for(const v of [m.n,m.s]) {
      const points=B.halfDisk(v).node;
      points.forEach((p,i)=>{
        const ray=B.unproject(...p); assert.ok(ray); close(B.amplitude(m,ray),0,1e-8);
        if(i) assert.ok(Math.hypot(p[0]-points[i-1][0],p[1]-points[i-1][1])<.03);
      });
    }
  }
});

test('reject invalid mechanisms instead of drawing silent nonsense', () => {
  for(const sdr of [[NaN,45,0],[Infinity,45,0],[0,-1,0],[0,91,0],[0,45,181],[-1,0,0],[361,0,0]]) {
    assert.throws(()=>B.mechanism(...sdr), RangeError);
  }
});
