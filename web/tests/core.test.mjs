// Run: node --test web/tests
import assert from 'node:assert/strict';
import test from 'node:test';

import { fft2 } from '../js/sim/fft.js';
import { HeightfieldCaster } from '../js/sim/raycast.js';
import { rng } from '../js/sim/rng.js';
import { sunAstronomical, sunManual } from '../js/sim/sun.js';
import { buildTerrain, fractal, heightAt, rmsSlope, sampleCraters, stampCraters } from '../js/sim/terrain.js';
import { SCENARIOS } from '../js/sim/scenarios.js';

const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg}: ${a} vs ${b} (tol ${tol})`);

test('rng: reproducible, independent streams, distributions', () => {
  const a = rng(1, 'x'), b = rng(1, 'x'), c = rng(1, 'y');
  const sa = Array.from({ length: 5 }, () => a.next()), sb = Array.from({ length: 5 }, () => b.next());
  assert.deepEqual(sa, sb);
  assert.notDeepEqual(sa, Array.from({ length: 5 }, () => c.next()));
  const r = rng(3, 'stats');
  let s = 0, s2 = 0, p = 0;
  const N = 200000;
  for (let i = 0; i < N; i++) { const x = r.normal(); s += x; s2 += x * x; p += r.poisson(4.5); }
  close(s / N, 0, 0.01, 'normal mean'); close(s2 / N, 1, 0.01, 'normal var'); close(p / N, 4.5, 0.02, 'poisson mean');
});

test('fft2 round trip', () => {
  const n = 16, re = new Float64Array(n * n), im = new Float64Array(n * n), r = rng(0, 'fft');
  for (let k = 0; k < n * n; k++) re[k] = r.normal();
  const orig = re.slice();
  fft2(re, im, n); fft2(re, im, n, true);
  for (let k = 0; k < n * n; k++) close(re[k], orig[k], 1e-12, 'roundtrip');
});

test('fractal: target RMS slope at the baseline', () => {
  const p = { enabled: true, hurst: 0.85, rmsSlopeDeg: 6, baseline: 1, rolloffWavelength: null };
  const z = fractal(256, 0.25, p, 7);
  close(rmsSlope(z, 256, 0.25, 4), Math.tan(6 * Math.PI / 180), 1e-9, 'rms slope');
});

test('craters: expected count of the equilibrium SFD', () => {
  const p = SCENARIOS.mare_plain.terrain.craters;
  const ext = [-64, 64, -64, 64], margin = 0.5 * p.dMax * p.shape.ejectaExtent;
  const area = (128 + 2 * margin) ** 2, lam = area * 1.54 * p.equilibriumFraction * (p.dMin ** -2 - p.dMax ** -2);
  const n = sampleCraters(ext, p, 11).length;
  assert.ok(Math.abs(n - lam) < 4 * Math.sqrt(lam), `count ${n} vs expected ${lam}`);
});

test('terrain build is deterministic and crater depth is right for a fresh crater', () => {
  const sc = structuredClone(SCENARIOS.mare_plain);
  sc.terrain.n = 128;
  const a = buildTerrain(sc), b = buildTerrain(sc);
  assert.deepEqual(a.z, b.z);
  const flat = structuredClone(sc);
  flat.terrain.fractal.enabled = false; flat.terrain.rocks.enabled = false;
  flat.terrain.craters.enabled = false;
  const t = buildTerrain(flat);
  stampCraters(t, [{ x: 0, y: 0, d: 8, g: 0, depth: 1.2, rim: 0.288 }], flat.terrain.craters.shape, t.labels);
  close(heightAt(t, 0, 0), -(1.2 - 0.288), 0.02, 'floor');
  // Rim crest (0.288 m) is sampled on the 0.2 m grid: nodes at r = 0.975 / 1.025 R give ~0.25 m.
  close(heightAt(t, 4, 0), 0.248, 0.01, 'rim');
});

test('raycast matches a brute-force march', () => {
  const sc = structuredClone(SCENARIOS.mare_plain);
  sc.terrain.n = 128;
  const hm = buildTerrain(sc), caster = new HeightfieldCaster(hm), r = rng(5, 'rays');
  for (let k = 0; k < 300; k++) {
    const ox = r.uniform(-10, 10), oy = r.uniform(-10, 10), oz = heightAt(hm, ox, oy) + r.uniform(0.5, 3);
    let dx = r.normal(), dy = r.normal(), dz = -r.uniform(0.05, 1);
    const nn = Math.hypot(dx, dy, dz); dx /= nn; dy /= nn; dz /= nn;
    const t = caster.trace(ox, oy, oz, dx, dy, dz);
    let tb = Infinity; // brute force: first sign change of (ray z - surface z), refined by bisection
    let prev = oz - heightAt(hm, ox, oy);
    for (let s = 0.01; s < 60; s += 0.01) {
      const x = ox + dx * s, y = oy + dy * s;
      if (x < hm.extent[0] || x > hm.extent[1] || y < hm.extent[2] || y > hm.extent[3]) break;
      const cur = oz + dz * s - heightAt(hm, x, y);
      if (prev > 0 && cur <= 0) {
        let lo = s - 0.01, hi = s;
        for (let it = 0; it < 40; it++) {
          const m = 0.5 * (lo + hi);
          if (oz + dz * m - heightAt(hm, ox + dx * m, oy + dy * m) > 0) lo = m; else hi = m;
        }
        tb = 0.5 * (lo + hi);
        break;
      }
      prev = cur;
    }
    if (tb < Infinity) close(t, tb, 1e-6, `ray ${k}`);
  }
});

test('sun: analytic model equals the Python implementation', () => {
  const ref = [
    { lat: -89.5, lon: 0, t: '2026-12-20T00:00:00Z', az: 48.7512717040939, el: 1.6122693412614981, earth: [-0.12593455455698488, 0.9856239225736192, 0.11263201684745715] },
    { lat: 10, lon: 20, t: '2027-03-05T12:34:00Z', az: 315.19247648836733, el: -75.12456514613925, earth: [-0.3743594673945459, -0.14558176447919477, 0.9157843299725085] },
    { lat: -45, lon: -120, t: '2030-07-01T06:00:00Z', az: 285.9289318803903, el: 16.20158614829591, earth: [0.8930132393034784, -0.2749721474641594, -0.35625506669752344] },
  ];
  for (const r of ref) {
    const s = sunAstronomical(r.lat, r.lon, new Date(r.t));
    close(s.az * 180 / Math.PI, r.az, 1e-7, 'az'); close(s.el * 180 / Math.PI, r.el, 1e-7, 'el');
    for (let k = 0; k < 3; k++) close(s.earthDir[k], r.earth[k], 1e-9, 'earth');
  }
  const m = sunManual(90, 30);
  close(m.dir[0], Math.cos(Math.PI / 6), 1e-12, 'manual east');
});
