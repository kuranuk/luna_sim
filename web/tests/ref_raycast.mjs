// Reference ray caster for tests: exact cell walk (Amanatides-Woo) over the triangulated grid with
// Möller-Trumbore per triangle. Independent of raycast.js (no max-mipmap, no acceptance window).
function tri(a, b, c, o, d) {
  const e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
  const p = [d[1] * e2[2] - d[2] * e2[1], d[2] * e2[0] - d[0] * e2[2], d[0] * e2[1] - d[1] * e2[0]];
  const det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
  if (Math.abs(det) < 1e-20) return Infinity;
  const s = [o[0] - a[0], o[1] - a[1], o[2] - a[2]], u = (s[0] * p[0] + s[1] * p[1] + s[2] * p[2]) / det;
  if (u < -1e-12 || u > 1 + 1e-12) return Infinity;
  const q = [s[1] * e1[2] - s[2] * e1[1], s[2] * e1[0] - s[0] * e1[2], s[0] * e1[1] - s[1] * e1[0]];
  const v = (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]) / det;
  if (v < -1e-12 || u + v > 1 + 1e-12) return Infinity;
  const t = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) / det;
  return t >= 0 ? t : Infinity;
}
export function cellHit(hm, i, j, o, d) {
  const { z, n, res, x0, y0 } = hm;
  const xa = x0 + j * res, xb = xa + res, ya = y0 + i * res, yb = ya + res;
  const A = [xa, ya, z[i * n + j]], B = [xb, ya, z[i * n + j + 1]], Cc = [xb, yb, z[(i + 1) * n + j + 1]], D = [xa, yb, z[(i + 1) * n + j]];
  return Math.min(tri(A, B, Cc, o, d), tri(A, Cc, D, o, d));
}
export function refTrace(hm, o, d, tmax) {
  const { n, res, x0, y0 } = hm, nq = n - 1;
  let u = (o[0] - x0) / res, v = (o[1] - y0) / res;
  // enter the grid box
  let t0 = 0, t1 = tmax;
  for (const [p, dd] of [[u, d[0] / res], [v, d[1] / res]]) {
    if (Math.abs(dd) < 1e-15) { if (p < 0 || p > nq) return Infinity; continue; }
    let a = (0 - p) / dd, b = (nq - p) / dd; if (a > b) [a, b] = [b, a];
    t0 = Math.max(t0, a); t1 = Math.min(t1, b);
  }
  if (t0 > t1) return Infinity;
  const du = d[0] / res, dv = d[1] / res;
  let uu = u + du * t0, vv = v + dv * t0;
  let j = Math.min(Math.max(Math.floor(uu), 0), nq - 1), i = Math.min(Math.max(Math.floor(vv), 0), nq - 1);
  const sj = du > 0 ? 1 : du < 0 ? -1 : 0, si = dv > 0 ? 1 : dv < 0 ? -1 : 0;
  let tx = sj ? ((sj > 0 ? j + 1 : j) - u) / du : Infinity, ty = si ? ((si > 0 ? i + 1 : i) - v) / dv : Infinity;
  const dtx = sj ? Math.abs(1 / du) : Infinity, dty = si ? Math.abs(1 / dv) : Infinity;
  let best = Infinity;
  for (let guard = 0; guard < 4 * nq + 10; guard++) {
    // test this cell and its neighbours along the boundary (robust to ties); accept the minimum
    const th = cellHit(hm, i, j, o, d);
    if (th <= t1) best = Math.min(best, th);
    const tnext = Math.min(tx, ty);
    if (best <= tnext + 1e-9) return best; // nothing nearer can come from later cells
    if (tnext > t1) return best;
    if (tx < ty) { j += sj; tx += dtx; } else { i += si; ty += dty; }
    if (i < 0 || j < 0 || i >= nq || j >= nq) return best;
  }
  return best;
}

