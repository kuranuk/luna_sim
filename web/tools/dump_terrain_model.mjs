// Dump numbers of the browser terrain model for the cross-check against the Python implementation
// and the published relations (tests/test_web_sim.py): crater profiles, crater and rock populations,
// fractal statistics. Prints one JSON object.
//   node web/tools/dump_terrain_model.mjs

import { SCENARIOS } from '../js/sim/scenarios.js';
import { buildTerrain, fractal, profileTable, rmsSlope, sampleCraters, sampleRocks, stampCraters, stampRocks, heightAt } from '../js/sim/terrain.js';

const sc = structuredClone(SCENARIOS.south_pole_lowsun), t = sc.terrain;
const out = { shape: t.craters.shape, craterParams: { fraction: t.craters.equilibriumFraction, dMin: t.craters.dMin, dMax: t.craters.dMax } };

// 1. crater profiles / R at fixed radii for several degradation states
const { rho, table } = profileTable(t.craters.shape);
const at = (row, r) => { const u = r / rho[1], m = Math.min(Math.floor(u), rho.length - 2); return row[m] + (u - m) * (row[m + 1] - row[m]); };
out.rhos = [0, 0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2, 3, 3.9];
out.profiles = {};
for (const g of [0, 0.25, 0.5, 0.75, 1]) {
  const ti = g * (table.length - 1), i = Math.min(Math.floor(ti), table.length - 2), f = ti - i;
  const row = table[i].map((v, k) => (1 - f) * v + f * table[i + 1][k]);
  out.profiles[g] = out.rhos.map((r) => at(row, r));
}

// 2. crater population on a large area (many seeds pooled)
const extent = [-200, 200, -200, 200], margin = 0.5 * t.craters.dMax * t.craters.shape.ejectaExtent;
out.craterArea = (400 + 2 * margin) ** 2;
const ds = [], gs = [];
out.craterCounts = [];
for (let seed = 1; seed <= 8; seed++) {
  const cs = sampleCraters(extent, t.craters, seed);
  out.craterCounts.push(cs.length);
  for (const c of cs) { ds.push(c.d); gs.push(c.g); }
}
ds.sort((a, b) => a - b);
out.craterDiameterQuantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999].map((q) => [q, ds[Math.floor(q * (ds.length - 1))]]);
out.craterMeanG = gs.reduce((a, b) => a + b, 0) / gs.length;

// 3. one fresh and one fully degraded 20 m crater stamped on a flat 0.05 m grid: measured shape
function measure(g) {
  const n = 2048, res = 0.05, hm = { z: new Float64Array(n * n), n, res, x0: -(n - 1) * res / 2, y0: -(n - 1) * res / 2 };
  const s = t.craters.shape, dd = s.depthFresh + g * (s.depthDegraded - s.depthFresh), d = 20;
  stampCraters(hm, [{ x: 0, y: 0, d, g, depth: dd * d, rim: dd * d * s.rimFresh / s.depthFresh }], s, null);
  let floor = Infinity, crest = -Infinity, crestR = 0;
  for (let k = 0; k <= 1400; k++) { const r = k * 0.01, z = heightAt(hm, r, 0); if (z < floor) floor = z; if (z > crest) { crest = z; crestR = r; } }
  return { g, d, depthOverD: (crest - floor) / d, rimOverD: crest / d, crestRadiusOverR: crestR / (d / 2),
    ejecta: [1.5, 2, 3].map((q) => heightAt(hm, q * d / 2, 0) / crest) };
}
out.stamped = [measure(0), measure(1)];

// 4. rocks: background population pooled over seeds (no craters)
const rp = { ...t.rocks, ejecta: false };
out.rockParams = { k: rp.k, dMin: rp.dMin, dMax: rp.dMax, heightRatio: rp.heightRatio, axisRatioMin: rp.axisRatioMin };
const rext = [-300, 300, -300, 300];
out.rockArea = 600 * 600;
const rd = [];
let areaSum = 0;
out.rockCounts = [];
for (let seed = 1; seed <= 8; seed++) {
  const rs = sampleRocks(rext, rp, [], seed);
  out.rockCounts.push(rs.length);
  for (const r of rs) { rd.push(r.d); areaSum += Math.PI / 4 * r.d * r.d; }
}
out.rockCircleAreaFraction = areaSum / (8 * out.rockArea); // counting each rock as a circle of diameter d
out.rockAreaFractionAbove = [0.5, 0.75, 1.0, 1.5].map((D) => [D, rd.filter((d) => d >= D).reduce((a, d) => a + Math.PI / 4 * d * d, 0) / (8 * out.rockArea)]);
{ // stamped footprint and height of one rock
  const n = 256, res = 0.01, hm = { z: new Float64Array(n * n), n, res, x0: -1.275, y0: -1.275 };
  stampRocks(hm, [{ x: 0, y: 0, d: 1.2, h: 0.6, axis: 0.7, orient: 0.4 }], null);
  let cells = 0, top = 0;
  for (const v of hm.z) { if (v > 0) cells++; if (v > top) top = v; }
  out.rockStamp = { area: cells * res * res, expectedArea: Math.PI * 0.6 * 0.6 * 0.7, top };
}

// 5. fractal: RMS slope at the baseline and structure-function exponent (Hurst)
{
  const n = 1024, res = 0.2, z = fractal(n, res, t.fractal, 7);
  out.fractal = { hurst: t.fractal.hurst, rmsSlopeDeg: t.fractal.rmsSlopeDeg, baseline: t.fractal.baseline,
    measuredSlopeDeg: Math.atan(rmsSlope(z, n, res, Math.round(t.fractal.baseline / res))) * 180 / Math.PI,
    structure: [1, 2, 4, 8, 16, 32, 64].map((lag) => [lag * res, rmsSlope(z, n, res, lag) * lag * res]) };
  const a = out.fractal.structure, x = a.map((p) => Math.log(p[0])), y = a.map((p) => Math.log(p[1]));
  const mx = x.reduce((s, v) => s + v, 0) / x.length, my = y.reduce((s, v) => s + v, 0) / y.length;
  out.fractal.measuredHurst = x.reduce((s, v, k) => s + (v - mx) * (y[k] - my), 0) / x.reduce((s, v) => s + (v - mx) ** 2, 0);
  let mean = 0; for (const v of z) mean += v; out.fractal.mean = mean / z.length;
}

// 6. full terrain: tilt recovered by a plane fit
{
  const s2 = structuredClone(sc); s2.terrain.n = 512;
  s2.terrain.tilt = t.tilt = { slopeDeg: 3, downhillAzimuthDeg: 50 }; // config-only model
  for (const k of ['fractal', 'craters', 'rocks']) s2.terrain[k].enabled = false;
  const hm = buildTerrain(s2), az = t.tilt.downhillAzimuthDeg * Math.PI / 180;
  const down = (heightAt(hm, 0, 0) - heightAt(hm, 10 * Math.sin(az), 10 * Math.cos(az))) / 10; // drop per metre downhill
  out.tilt = { slopeDeg: t.tilt.slopeDeg, measuredDeg: Math.atan(down) * 180 / Math.PI,
    across: heightAt(hm, 10 * Math.cos(az), -10 * Math.sin(az)) - heightAt(hm, 0, 0) };
}
console.log(JSON.stringify(out));
