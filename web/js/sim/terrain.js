// Terrain generation (simplified browser port of lunarsim/core/terrain).
// Order: fractal -> tilt -> craters (oldest first) -> rocks. Single layer; the fractal is
// periodic (FFT on the grid itself), which is seamless but repeats beyond the map edge.
//
// Grid: z[i * n + j], i = south -> north (y), j = west -> east (x);
// x_j = x0 + j * res, y_i = y0 + i * res, centred on the origin.
// Parameter defaults and references follow lunarsim/config/schema.py (several are sample values).

import { fft2 } from './fft.js';
import { rng } from './rng.js';

export const LABEL = { PLAIN: 0, CRATER_INTERIOR: 1, CRATER_RIM: 2, ROCK: 3 };

const RING_IN = 1.1, RING_OUT = 1.6; // reference-plane ring [crater radii]
const REP_IN = 0.9, REP_OUT = 1.1; // pre-impact surface replacement ramp
const RIM_IN = 0.85, RIM_OUT = 1.15; // label bands

// --------------------------------------------------------------------------- fractal

export function rmsSlope(z, n, res, lag) {
  let s = 0, cnt = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j + lag < n; j++) { const d = z[i * n + j + lag] - z[i * n + j]; s += d * d; cnt++; }
  }
  for (let i = 0; i + lag < n; i++) {
    for (let j = 0; j < n; j++) { const d = z[(i + lag) * n + j] - z[i * n + j]; s += d * d; cnt++; }
  }
  return Math.sqrt(s / cnt) / (lag * res);
}

/** Self-affine fractal: PSD ∝ (k² + k0²)^(-β/2), β = 2H + 2, scaled to the RMS slope at `baseline`. */
export function fractal(n, res, p, seed) {
  const r = rng(seed, 'terrain/fractal');
  const re = new Float64Array(n * n), im = new Float64Array(n * n);
  if (!p.enabled || p.rmsSlopeDeg <= 0) return re;
  for (let k = 0; k < n * n; k++) re[k] = r.normal();
  fft2(re, im, n);
  const beta = 2 * p.hurst + 2, k0 = p.rolloffWavelength ? 1 / p.rolloffWavelength : 0;
  for (let a = 0; a < n; a++) {
    const ky = (a <= n / 2 ? a : a - n) / (n * res);
    for (let b = 0; b < n; b++) {
      const kx = (b <= n / 2 ? b : b - n) / (n * res);
      const amp = a === 0 && b === 0 ? 0 : Math.pow(kx * kx + ky * ky + k0 * k0, -beta / 4);
      re[a * n + b] *= amp; im[a * n + b] *= amp;
    }
  }
  fft2(re, im, n, true);
  let mean = 0;
  for (let k = 0; k < n * n; k++) mean += re[k];
  mean /= n * n;
  for (let k = 0; k < n * n; k++) re[k] -= mean;
  const lag = Math.max(1, Math.round(p.baseline / res));
  const s = rmsSlope(re, n, res, lag);
  const scale = s > 0 ? Math.tan(p.rmsSlopeDeg * Math.PI / 180) / s : 0;
  for (let k = 0; k < n * n; k++) re[k] *= scale;
  return re;
}

// --------------------------------------------------------------------------- craters

function i0e(x) { // exp(-|x|) I0(x), Abramowitz & Stegun 9.8.1 / 9.8.2
  const ax = Math.abs(x);
  if (ax <= 3.75) {
    const t = (x / 3.75) ** 2;
    return Math.exp(-ax) * (1 + t * (3.5156229 + t * (3.0899424 + t * (1.2067492
      + t * (0.2659732 + t * (0.0360768 + t * 0.0045813))))));
  }
  const t = 3.75 / ax;
  return (0.39894228 + t * (0.01328592 + t * (0.00225319 + t * (-0.00157565 + t * (0.00916281
    + t * (-0.02057706 + t * (0.02635537 + t * (-0.01647633 + t * 0.00392377)))))))) / Math.sqrt(ax);
}

const NG = 33, NR = 257;
const tableCache = new Map();

/**
 * Radial profiles / R vs degradation g: parabolic bowl, rim and ejecta ∝ (r/R)^-n (fresh), smoothed
 * by an isotropic Gaussian of sigma = smoothing * g when degraded and rescaled so rim-to-floor = d(g).
 */
export function profileTable(shape) {
  const key = JSON.stringify(shape);
  if (tableCache.has(key)) return tableCache.get(key);
  const rmax = shape.ejectaExtent, n = shape.ejectaExponent;
  const rho = Float64Array.from({ length: NR }, (_, k) => rmax * k / (NR - 1));
  const dr = rho[1], table = [];
  for (let gi = 0; gi < NG; gi++) {
    const g = gi / (NG - 1);
    const depth = 2 * (shape.depthFresh + g * (shape.depthDegraded - shape.depthFresh)); // D = 2R = 2
    const rim = depth * (shape.rimFresh / shape.depthFresh), floor = depth - rim, tail = rmax ** -n;
    let f = Float64Array.from(rho, (r) => (r < 1 ? -floor + (floor + rim) * r * r
      : r < rmax ? rim * (r ** -n - tail) / (1 - tail) : 0));
    const s = shape.smoothingDegraded * g;
    if (s > 0) {
      const out = new Float64Array(NR);
      for (let a = 0; a < NR; a++) {
        let acc = 0;
        for (let b = 0; b < NR; b++) {
          const w = (b === 0 || b === NR - 1 ? 0.5 : 1) * dr;
          acc += f[b] * Math.exp(-((rho[a] - rho[b]) ** 2) / (2 * s * s)) * i0e(rho[a] * rho[b] / (s * s)) * rho[b] * w / (s * s);
        }
        out[a] = acc;
      }
      let hi = -Infinity, lo = Infinity;
      for (let a = 0; a < NR; a++) {
        if (rho[a] <= 1.2) hi = Math.max(hi, out[a]);
        if (rho[a] < 1) lo = Math.min(lo, out[a]);
      }
      for (let a = 0; a < NR; a++) out[a] *= depth / (hi - lo);
      f = out;
    }
    for (let a = 0; a < NR; a++) if (rho[a] >= rmax) f[a] = 0;
    table.push(f);
  }
  const res = { rho, table, rmax };
  tableCache.set(key, res);
  return res;
}

/** Crater population: cumulative N(>D) = c D^-2 per m², c = 1.54 × equilibrium fraction. */
export function sampleCraters(extent, p, seed) {
  if (!p.enabled) return [];
  const c = 1.54 * p.equilibriumFraction, b = 2;
  const margin = 0.5 * p.dMax * p.shape.ejectaExtent;
  const xmin = extent[0] - margin, xmax = extent[1] + margin, ymin = extent[2] - margin, ymax = extent[3] + margin;
  const lo = p.dMin ** -b, hi = p.dMax ** -b;
  const count = rng(seed, 'terrain/craters/count').poisson((xmax - xmin) * (ymax - ymin) * c * (lo - hi));
  const rs = rng(seed, 'terrain/craters/size'), rp = rng(seed, 'terrain/craters/position');
  const rg = rng(seed, 'terrain/craters/degradation');
  const out = [];
  for (let k = 0; k < count; k++) {
    const d = (lo - rs.random() * (lo - hi)) ** (-1 / b);
    const g = rg.beta(p.degradationA, p.degradationB);
    const dd = p.shape.depthFresh + g * (p.shape.depthDegraded - p.shape.depthFresh);
    out.push({ x: rp.uniform(xmin, xmax), y: rp.uniform(ymin, ymax), d, g,
      depth: dd * d, rim: dd * d * (p.shape.rimFresh / p.shape.depthFresh) });
  }
  out.sort((a, q) => q.g - a.g); // oldest (most degraded) first = stamping order
  return out;
}

export function stampCraters(hm, craters, shape, labels) {
  const { z, n, res, x0, y0 } = hm;
  const { rho: prho, table, rmax } = profileTable(shape);
  const nr = prho.length, drho = prho[1], prof = new Float64Array(nr);
  for (const cr of craters) {
    const R = cr.d / 2, reach = rmax * R;
    const j0 = Math.max(0, Math.floor((cr.x - reach - x0) / res)), j1 = Math.min(n, Math.ceil((cr.x + reach - x0) / res) + 1);
    const i0 = Math.max(0, Math.floor((cr.y - reach - y0) / res)), i1 = Math.min(n, Math.ceil((cr.y + reach - y0) / res) + 1);
    if (j0 >= j1 || i0 >= i1) continue;
    const t = Math.min(Math.max(cr.g, 0), 1) * (NG - 1), ti = Math.min(Math.floor(t), NG - 2), tf = t - ti;
    for (let m = 0; m < nr; m++) prof[m] = (1 - tf) * table[ti][m] + tf * table[ti + 1][m];
    // Pre-impact reference plane z = a + b dx + c dy fitted on the ring (normal equations).
    let cnt = 0, sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0, sz = 0, sxz = 0, syz = 0, pz = 0;
    for (let i = i0; i < i1; i++) {
      const dy = y0 + res * i - cr.y;
      for (let j = j0; j < j1; j++) {
        const dx = x0 + res * j - cr.x, r = Math.hypot(dx, dy) / R, zz = z[i * n + j];
        pz += zz;
        if (r >= RING_IN && r <= RING_OUT) {
          cnt++; sx += dx; sy += dy; sxx += dx * dx; sxy += dx * dy; syy += dy * dy; sz += zz; sxz += dx * zz; syz += dy * zz;
        }
      }
    }
    let a = cnt ? sz / cnt : pz / ((i1 - i0) * (j1 - j0)), bb = 0, cc = 0;
    if (cnt >= 3 && cnt >= 0.5 * Math.PI * (RING_OUT ** 2 - RING_IN ** 2) * (R / res) ** 2) {
      const m00 = cnt, m01 = sx, m02 = sy, m11 = sxx, m12 = sxy, m22 = syy;
      const c00 = m11 * m22 - m12 * m12, c01 = m02 * m12 - m01 * m22, c02 = m01 * m12 - m02 * m11;
      const det = m00 * c00 + m01 * c01 + m02 * c02;
      if (Math.abs(det) > 1e-12 * Math.abs(m00 * m11 * m22)) {
        const c11 = m00 * m22 - m02 * m02, c12 = m01 * m02 - m00 * m12, c22 = m00 * m11 - m01 * m01;
        a = (c00 * sz + c01 * sxz + c02 * syz) / det;
        bb = (c01 * sz + c11 * sxz + c12 * syz) / det;
        cc = (c02 * sz + c12 * sxz + c22 * syz) / det;
      }
    }
    const lab = labels && cr.d >= 4 * res;
    for (let i = i0; i < i1; i++) {
      const dy = y0 + res * i - cr.y;
      for (let j = j0; j < j1; j++) {
        const dx = x0 + res * j - cr.x, r = Math.hypot(dx, dy) / R;
        let delta = 0;
        if (r <= rmax) {
          const u = r / drho, m = Math.min(Math.floor(u), nr - 2), f = u - m;
          delta = R * ((1 - f) * prof[m] + f * prof[m + 1]);
        }
        const w = Math.min(Math.max((REP_OUT - r) / (REP_OUT - REP_IN), 0), 1);
        const k = i * n + j, zz = z[k];
        z[k] = zz + (delta + w * (a + bb * dx + cc * dy - zz));
        if (lab) {
          if (r < RIM_IN) labels[k] = LABEL.CRATER_INTERIOR;
          else if (r < RIM_OUT) labels[k] = LABEL.CRATER_RIM;
        }
      }
    }
  }
}

// --------------------------------------------------------------------------- rocks

const qOfK = (k) => 1.79 + 0.152 / k; // Golombek & Rapp (1997): F_k(D) = k exp(-q D)

function rockDiameters(k, dMin, dMax, area, r) {
  if (dMax <= dMin || k <= 0) return [];
  const N = 200, d = [], cum = [0], q = qOfK(k);
  for (let i = 0; i < N; i++) d.push(dMin * (dMax / dMin) ** (i / (N - 1)));
  const dens = d.map((x) => k * q * Math.exp(-q * x) / (0.25 * Math.PI * x * x));
  for (let i = 1; i < N; i++) cum.push(cum[i - 1] + 0.5 * (dens[i] + dens[i - 1]) * (d[i] - d[i - 1]));
  const total = cum[N - 1], count = r.poisson(total * area), out = [];
  for (let c = 0; c < count; c++) {
    const u = r.random() * total;
    let lo = 0, hi = N - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (cum[mid] <= u) lo = mid; else hi = mid; }
    out.push(d[lo] + (u - cum[lo]) / Math.max(cum[hi] - cum[lo], 1e-300) * (d[hi] - d[lo]));
  }
  return out;
}

export function sampleRocks(extent, p, craters, seed) {
  if (!p.enabled) return [];
  const out = [];
  const make = (r, d, x, y) => ({ x, y, d, h: p.heightRatio * d, axis: r.uniform(p.axisRatioMin, 1), orient: r.uniform(0, Math.PI) });
  const rb = rng(seed, 'terrain/rocks/background');
  const area = (extent[1] - extent[0]) * (extent[3] - extent[2]);
  for (const d of rockDiameters(p.k, p.dMin, p.dMax, area, rb)) {
    out.push(make(rb, d, rb.uniform(extent[0], extent[1]), rb.uniform(extent[2], extent[3])));
  }
  if (p.ejecta) { // blocks around fresh craters (rim / proximal ejecta annulus 0.8-2.5 R)
    const re = rng(seed, 'terrain/rocks/ejecta');
    for (const c of craters) {
      const kE = p.ejectaK * (1 - c.g);
      if (kE <= 1e-6 || c.d < p.ejectaMinCraterD) continue;
      const dMax = Math.min(p.dMax, 0.29 * c.d ** 0.66); // TODO(ref): Bart & Melosh (2010)
      const rin = 0.4 * c.d, rout = 1.25 * c.d;
      for (const d of rockDiameters(kE, p.dMin, dMax, Math.PI * (rout * rout - rin * rin), re)) {
        const rr = Math.sqrt(re.uniform(rin * rin, rout * rout)), th = re.uniform(0, 2 * Math.PI);
        out.push(make(re, d, c.x + rr * Math.cos(th), c.y + rr * Math.sin(th)));
      }
    }
  }
  return out;
}

export function stampRocks(hm, rocks, labels) {
  const { z, n, res, x0, y0 } = hm;
  for (const rk of rocks) {
    const a = rk.d / 2, b = a * rk.axis;
    const j0 = Math.max(0, Math.floor((rk.x - a - x0) / res)), j1 = Math.min(n, Math.ceil((rk.x + a - x0) / res) + 1);
    const i0 = Math.max(0, Math.floor((rk.y - a - y0) / res)), i1 = Math.min(n, Math.ceil((rk.y + a - y0) / res) + 1);
    const co = Math.cos(rk.orient), si = Math.sin(rk.orient);
    for (let i = i0; i < i1; i++) {
      const dy = y0 + res * i - rk.y;
      for (let j = j0; j < j1; j++) {
        const dx = x0 + res * j - rk.x;
        const u = (dx * co + dy * si) / a, v = (-dx * si + dy * co) / b, s = 1 - u * u - v * v;
        if (s > 0) { z[i * n + j] += rk.h * Math.sqrt(s); if (labels) labels[i * n + j] = LABEL.ROCK; }
      }
    }
  }
}

// --------------------------------------------------------------------------- assembly

/** Build the terrain of a scenario: { z, labels, n, res, x0, y0, extent, craters, rocks }. */
export function buildTerrain(sc) {
  const t = sc.terrain, n = t.n, res = t.res;
  const x0 = -0.5 * (n - 1) * res, y0 = x0;
  const z = fractal(n, res, t.fractal, sc.seed);
  if (t.tilt.slopeDeg > 0) {
    const az = t.tilt.downhillAzimuthDeg * Math.PI / 180, grade = Math.tan(t.tilt.slopeDeg * Math.PI / 180);
    const gx = Math.sin(az), gy = Math.cos(az);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) z[i * n + j] -= grade * ((x0 + j * res) * gx + (y0 + i * res) * gy);
  }
  const labels = new Uint8Array(n * n);
  const hm = { z, labels, n, res, x0, y0 };
  const extent = [x0, x0 + (n - 1) * res, y0, y0 + (n - 1) * res];
  const craters = sampleCraters(extent, t.craters, sc.seed);
  stampCraters(hm, craters, t.craters.shape, labels);
  const rocks = sampleRocks(extent, t.rocks, craters, sc.seed);
  stampRocks(hm, rocks, labels);
  return { ...hm, extent, craters, rocks };
}

/** Elevation [m] on the triangulated surface (quads split along (i, j)-(i+1, j+1)). */
export function heightAt(hm, x, y) {
  const { z, n, res, x0, y0 } = hm;
  const u = Math.min(Math.max((x - x0) / res, 0), n - 1), v = Math.min(Math.max((y - y0) / res, 0), n - 1);
  const j = Math.min(Math.floor(u), n - 2), i = Math.min(Math.floor(v), n - 2);
  const fu = u - j, fv = v - i, z00 = z[i * n + j], z11 = z[(i + 1) * n + j + 1];
  if (fu >= fv) { const zx = z[i * n + j + 1]; return z00 + fu * (zx - z00) + fv * (z11 - zx); }
  const zy = z[(i + 1) * n + j];
  return z00 + fv * (zy - z00) + fu * (z11 - zy);
}

/** Node gradient (central differences, one-sided at edges) -> unit normal at node (i, j). */
export function nodeNormal(hm, i, j) {
  const { z, n, res } = hm;
  const jm = Math.max(j - 1, 0), jp = Math.min(j + 1, n - 1), im = Math.max(i - 1, 0), ip = Math.min(i + 1, n - 1);
  const gx = (z[i * n + jp] - z[i * n + jm]) / ((jp - jm) * res), gy = (z[ip * n + j] - z[im * n + j]) / ((ip - im) * res);
  const nn = Math.hypot(gx, gy, 1);
  return [-gx / nn, -gy / nn, 1 / nn];
}

/** Smooth unit normal at (x, y): bilinear interpolation of node gradients. */
export function normalAt(hm, x, y) {
  const { n, res, x0, y0 } = hm;
  const u = Math.min(Math.max((x - x0) / res, 0), n - 1), v = Math.min(Math.max((y - y0) / res, 0), n - 1);
  const j = Math.min(Math.floor(u), n - 2), i = Math.min(Math.floor(v), n - 2);
  const fu = u - j, fv = v - i;
  let gx = 0, gy = 0;
  for (let di = 0; di < 2; di++) {
    for (let dj = 0; dj < 2; dj++) {
      const nm = nodeNormal(hm, i + di, j + dj), w = (dj ? fu : 1 - fu) * (di ? fv : 1 - fv);
      gx += w * (-nm[0] / nm[2]); gy += w * (-nm[1] / nm[2]);
    }
  }
  const nn = Math.hypot(gx, gy, 1);
  return [-gx / nn, -gy / nn, 1 / nn];
}

/** Terrain label of the nearest node. */
export function labelAt(hm, x, y) {
  const { n, res, x0, y0, labels } = hm;
  const j = Math.min(Math.max(Math.round((x - x0) / res), 0), n - 1), i = Math.min(Math.max(Math.round((y - y0) / res), 0), n - 1);
  return labels[i * n + j];
}
