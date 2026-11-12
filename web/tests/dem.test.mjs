// Real-DEM base terrain: interpolation, terrain build, shipped LOLA asset.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { resolveConfig } from '../js/sim/config.js';
import { demHeight, demProblems, parseDem } from '../js/sim/dem.js';
import { SCENARIOS } from '../js/sim/scenarios.js';
import { buildTerrain, heightAt } from '../js/sim/terrain.js';
import { World } from '../js/sim/world.js';
import { loadDem } from '../tools/load_dem.mjs';

const META = (n, res) => ({ format: 'lunarsim-dem/1', site: { lat: -89.5, lon: 0 }, n, res, x0: -0.5 * (n - 1) * res, y0: -0.5 * (n - 1) * res, data: 'x.f32' });
function synthetic(n, res, f) {
  const meta = META(n, res), a = new Float32Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) a[i * n + j] = f(meta.x0 + j * res, meta.y0 + i * res);
  return parseDem(meta, a.buffer);
}
const bare = () => { // south pole scenario with every synthetic model off, small grid
  const sc = structuredClone(SCENARIOS.south_pole_lowsun);
  Object.assign(sc.terrain, { n: 128, res: 1 });
  sc.terrain.fractal.enabled = false; sc.terrain.craters.enabled = false; sc.terrain.rocks.enabled = false;
  sc.objects = [];
  return sc;
};

test('bicubic interpolation: exact at nodes, reproduces a plane, clamps outside', () => {
  const plane = synthetic(17, 10, (x, y) => 0.1 * x - 0.05 * y + 3);
  for (const [x, y] of [[0, 0], [12.3, -45.6], [-61, 54.9], [33.3, 0.1]]) { // interior (edges use a one-sided stencil)
    assert.ok(Math.abs(demHeight(plane, x, y) - (0.1 * x - 0.05 * y + 3)) < 1e-4);
  }
  const bumpy = synthetic(17, 10, (x, y) => Math.sin(x / 30) * Math.cos(y / 25));
  for (let i = 0; i < 17; i++) for (let j = 0; j < 17; j++) {
    assert.ok(demHeight(bumpy, bumpy.x0 + j * 10, bumpy.y0 + i * 10) - bumpy.z[i * 17 + j] === 0);
  }
  assert.ok(Math.abs(demHeight(bumpy, 13, 7) - Math.sin(13 / 30) * Math.cos(7 / 25)) < 5e-3); // smooth function: small error
  assert.equal(demHeight(bumpy, 1e4, 0), demHeight(bumpy, 80, 0));
});

test('parseDem rejects wrong format, size and non-finite values', () => {
  assert.throws(() => parseDem({ ...META(8, 1), format: 'x' }, new ArrayBuffer(256)));
  assert.throws(() => parseDem(META(8, 1), new ArrayBuffer(100)));
  const a = new Float32Array(64); a[5] = NaN;
  assert.throws(() => parseDem(META(8, 1), a.buffer));
});

test('terrain with all synthetic models off equals the DEM; the tilt is not applied', () => {
  const dem = synthetic(33, 10, (x, y) => 0.08 * x + 2 * Math.sin(y / 40)), sc = bare();
  const t = buildTerrain(sc, dem);
  assert.equal(t.dem, true);
  for (const [x, y] of [[0, 0], [10.5, -20.25], [-60, 33]]) assert.ok(Math.abs(heightAt(t, x, y) - demHeight(dem, x, y)) < 2e-3);
  const noDem = buildTerrain(sc, null); // without the data: procedural (tilt 3 deg)
  assert.ok(Math.abs(heightAt(noDem, 10, 0) - heightAt(noDem, 0, 0)) > 0.3);
});

test('with a DEM only small-scale detail is added on top', () => {
  const dem = synthetic(33, 10, () => 0), sc = structuredClone(SCENARIOS.south_pole_lowsun);
  Object.assign(sc.terrain, { n: 256, res: 0.5 });
  sc.objects = [];
  const t = buildTerrain(sc, dem);
  assert.ok(t.craters.length > 10);
  assert.ok(Math.max(...t.craters.map((c) => c.d)) <= 40); // larger craters are the DEM's business
  let lo = Infinity, hi = -Infinity;
  for (const v of t.z) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  assert.ok(hi - lo < 25 && hi - lo > 1); // relief of the detail only (no 400 m undulation, no tilt)
});

test('site mismatch and too small DEM are refused', () => {
  const sc = bare();
  assert.throws(() => buildTerrain(sc, synthetic(9, 10, () => 0)), /はみ出します/);
  const other = structuredClone(sc); other.site = { lat: 0, lon: 0 };
  assert.equal(demProblems(synthetic(33, 10, () => 0), other).length, 1);
});

test('shipped LOLA DEM: south pole preset uses it; metadata and slope are as documented', () => {
  const sc = bare(), dem = loadDem(sc);
  assert.deepEqual(dem.site, { lat: -89.5, lon: 0 });
  assert.equal(dem.n, 129); assert.equal(dem.res, 10);
  assert.ok(Math.abs(demHeight(dem, 0, 0)) < 1.5); // heights are relative to the site
  const gx = (demHeight(dem, 10, 0) - demHeight(dem, -10, 0)) / 20, gy = (demHeight(dem, 0, 10) - demHeight(dem, 0, -10)) / 20;
  const slope = Math.atan(Math.hypot(gx, gy)) * 180 / Math.PI;
  assert.ok(Math.abs(slope - dem.stats.slope_at_site_deg) < 0.5, `slope ${slope}`);
  assert.ok(slope > 5 && slope < 12);
  assert.ok(gx > 0); // ground rises to the east (downhill azimuth ~267 deg)
  for (const size of [[512, 0.25], [1024, 0.2], [1024, 0.5]]) { // every size of the UI fits inside the DEM
    Object.assign(sc.terrain, { n: size[0], res: size[1] });
    assert.deepEqual(demProblems(dem, sc), []);
  }
  const w = new World(bare(), { dem });
  assert.equal(w.demMissing, false);
  assert.ok(Math.abs(heightAt(w.terrain, 20, 5) - demHeight(dem, 20, 5)) < 2e-3);
  assert.equal(new World(bare()).demMissing, true); // no data given: procedural fallback, flagged
  for (const k of ['mare_plain', 'crater_field']) assert.equal(SCENARIOS[k].terrain.dem, undefined);
});

test('config: terrain.dem is validated and can be switched off', () => {
  assert.equal(resolveConfig({ base: 'south_pole_lowsun', terrain: { dem: { enabled: false } } }).terrain.dem.enabled, false);
  assert.throws(() => resolveConfig({ base: 'south_pole_lowsun', terrain: { dem: { enabled: true, url: 'https://example.com/a.json' } } }), /terrain\.dem\.url/);
  assert.throws(() => resolveConfig({ base: 'south_pole_lowsun', terrain: { dem: { enabled: true, url: '../a.json' } } }), /terrain\.dem\.url/);
});
