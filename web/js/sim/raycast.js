// Height-field ray casting: max-mipmap hierarchical traversal over the triangulated grid
// (port of lunarsim/core/raycast/heightfield.py: same algorithm and tolerances).

const EPS_STEP = 1e-7, EPS_BARY = 1e-9, EPS_T = 1e-6, EPS_Z = 1e-6, EPS_DIR = 1e-12;

function triHit(ax, ay, az, bx, by, bz, cx, cy, cz, ox, oy, oz, dx, dy, dz) {
  const e1x = bx - ax, e1y = by - ay, e1z = bz - az, e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
  const px = dy * e2z - dz * e2y, py = dz * e2x - dx * e2z, pz = dx * e2y - dy * e2x;
  const det = e1x * px + e1y * py + e1z * pz;
  if (Math.abs(det) < 1e-18) return Infinity;
  const inv = 1 / det, sx = ox - ax, sy = oy - ay, sz = oz - az;
  const u = (sx * px + sy * py + sz * pz) * inv;
  if (u < -EPS_BARY || u > 1 + EPS_BARY) return Infinity;
  const qx = sy * e1z - sz * e1y, qy = sz * e1x - sx * e1z, qz = sx * e1y - sy * e1x;
  const v = (dx * qx + dy * qy + dz * qz) * inv;
  if (v < -EPS_BARY || u + v > 1 + EPS_BARY) return Infinity;
  return (e2x * qx + e2y * qy + e2z * qz) * inv;
}

export class HeightfieldCaster {
  constructor(hm) {
    this.hm = hm;
    const { z, n } = hm;
    // Pyramid: level 0 = max of each quad's 4 corners; then 2x2 max pooling down to 1x1.
    let w = n - 1, h = n - 1, lvl = new Float64Array(w * h);
    for (let i = 0; i < h; i++) {
      for (let j = 0; j < w; j++) {
        lvl[i * w + j] = Math.max(z[i * n + j], z[i * n + j + 1], z[(i + 1) * n + j], z[(i + 1) * n + j + 1]);
      }
    }
    const data = [lvl], ws = [w], hs = [h];
    while (w > 1 || h > 1) {
      const nw = (w + 1) >> 1, nh = (h + 1) >> 1, nxt = new Float64Array(nw * nh);
      for (let i = 0; i < nh; i++) {
        for (let j = 0; j < nw; j++) {
          let m = -Infinity;
          for (let di = 0; di < 2; di++) {
            const ii = 2 * i + di;
            if (ii >= h) continue;
            for (let dj = 0; dj < 2; dj++) { const jj = 2 * j + dj; if (jj < w && lvl[ii * w + jj] > m) m = lvl[ii * w + jj]; }
          }
          nxt[i * nw + j] = m;
        }
      }
      lvl = nxt; w = nw; h = nh;
      data.push(lvl); ws.push(w); hs.push(h);
    }
    this.data = data; this.ws = ws; this.hs = hs;
    let zmin = Infinity, zmax = -Infinity;
    for (let k = 0; k < z.length; k++) { if (z[k] < zmin) zmin = z[k]; if (z[k] > zmax) zmax = z[k]; }
    this.zmin = zmin; this.zmax = zmax;
  }

  /** Nearest hit distance along the unit ray within [t0, t1] (Infinity if none). */
  trace(ox, oy, oz, dx, dy, dz, t0 = 0, t1 = Infinity) {
    const { z, n, res, x0, y0 } = this.hm;
    const nq = n - 1, nlev = this.data.length;
    const u0 = (ox - x0) / res, v0 = (oy - y0) / res;
    const du = Math.abs(dx) > EPS_DIR ? dx / res : 0, dv = Math.abs(dy) > EPS_DIR ? dy / res : 0;
    let tlo = t0, thi = t1;
    // Slab clipping against the grid's bounding box (u, v in quad units, z in m).
    for (let ax = 0; ax < 3; ax++) {
      const p = ax === 0 ? u0 : ax === 1 ? v0 : oz, d = ax === 0 ? du : ax === 1 ? dv : dz;
      const lo = ax === 2 ? this.zmin - EPS_Z : 0, hi = ax === 2 ? this.zmax + EPS_Z : nq;
      if (Math.abs(d) < 1e-15) {
        if (p < lo || p > hi) return Infinity;
      } else {
        let ta = (lo - p) / d, tb = (hi - p) / d;
        if (ta > tb) { const s = ta; ta = tb; tb = s; }
        if (ta > tlo) tlo = ta;
        if (tb < thi) thi = tb;
      }
    }
    if (tlo > thi) return Infinity;
    let t = tlo, level = nlev - 1;
    while (t <= thi) {
      const s = 1 << level, lw = this.ws[level], lh = this.hs[level];
      const u = u0 + du * t, v = v0 + dv * t;
      const j = Math.min(Math.max(Math.floor(u / s), 0), lw - 1);
      const i = Math.min(Math.max(Math.floor(v / s), 0), lh - 1);
      const tx = du > 0 ? ((j + 1) * s - u0) / du : du < 0 ? (j * s - u0) / du : Infinity;
      const ty = dv > 0 ? ((i + 1) * s - v0) / dv : dv < 0 ? (i * s - v0) / dv : Infinity;
      const tc = Math.min(tx, ty, thi);
      const zlow = Math.min(oz + dz * t, oz + dz * tc);
      if (zlow > this.data[level][i * lw + j] + EPS_Z) {
        t = Math.max(t, tc) + EPS_STEP;
        if (level < nlev - 1) level++;
        continue;
      }
      if (level > 0) { level--; continue; }
      const xa = x0 + j * res, xb = xa + res, ya = y0 + i * res, yb = ya + res;
      const z00 = z[i * n + j], z01 = z[i * n + j + 1], z10 = z[(i + 1) * n + j], z11 = z[(i + 1) * n + j + 1];
      // Validate each triangle separately (one plane's hit behind the segment must not mask the other).
      const h1 = triHit(xa, ya, z00, xb, ya, z01, xb, yb, z11, ox, oy, oz, dx, dy, dz);
      const h2 = triHit(xa, ya, z00, xb, yb, z11, xa, yb, z10, ox, oy, oz, dx, dy, dz);
      const lo = Math.max(t - EPS_T, t0), hi = tc + EPS_T;
      let th = Infinity;
      if (lo <= h1 && h1 <= hi) th = h1;
      if (lo <= h2 && h2 <= hi && h2 < th) th = h2;
      if (th < Infinity) return th;
      t = Math.max(t, tc) + EPS_STEP;
    }
    return Infinity;
  }

  occluded(ox, oy, oz, dx, dy, dz, tMax = Infinity) {
    return this.trace(ox, oy, oz, dx, dy, dz, 0, tMax) < Infinity;
  }
}
