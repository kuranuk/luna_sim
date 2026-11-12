// Surface model toggles (路面モデル tab): each model can be switched off on its own.
import assert from 'node:assert/strict';
import test from 'node:test';

import { SCENARIOS } from '../js/sim/scenarios.js';
import { buildTerrain, heightAt } from '../js/sim/terrain.js';
import { World } from '../js/sim/world.js';

const small = () => { const sc = structuredClone(SCENARIOS.south_pole_lowsun); sc.terrain.n = 256; sc.cameras = []; return sc; };

test('terrain models switch off individually', () => {
  const all = buildTerrain(small());
  assert.ok(all.craters.length > 0 && all.rocks.length > 0);
  let sc = small(); sc.terrain.rocks.background = false; // only the ejecta blocks remain
  const ej = buildTerrain(sc);
  assert.ok(ej.rocks.length < all.rocks.length);
  sc = small(); sc.terrain.rocks.background = false; sc.terrain.rocks.ejecta = false;
  assert.equal(buildTerrain(sc).rocks.length, 0);
  sc = small(); sc.terrain.craters.enabled = false;
  assert.equal(buildTerrain(sc).craters.length, 0);
  // tilt off: a pure plane when everything else is off too
  sc = small(); for (const k of ['fractal', 'craters', 'rocks']) sc.terrain[k].enabled = false;
  sc.terrain.tilt = { slopeDeg: 3, downhillAzimuthDeg: 50 }; // config-only model (no preset uses it any more)
  const tilted = buildTerrain(sc);
  sc.terrain.tilt.enabled = false;
  const flat = buildTerrain(sc);
  assert.ok(Math.abs(heightAt(tilted, 10, 10) - heightAt(tilted, -10, -10)) > 0.5);
  assert.equal(heightAt(flat, 10, 10), 0); assert.equal(heightAt(flat, -10, -10), 0);
});

test('rover slip and sideways drift switch off', () => {
  const sc = small();
  for (const k of ['fractal', 'craters', 'rocks']) sc.terrain[k].enabled = false; // 3° tilted plane
  sc.terrain.tilt = { slopeDeg: 3, downhillAzimuthDeg: 50 };
  sc.lidars = []; sc.imu = null; sc.objects = [];
  const run = (flags) => {
    const w = new World(sc);
    Object.assign(w.scenario.rover, flags);
    w.rover.reset(0, 0, -2.44); w.setCommand(0.2, 0); // uphill (the plane falls towards azimuth 50°)
    w.advance(10, { scans: false });
    return { x: w.rover.x, y: w.rover.y, slip: w.rover.status.slip, vy: w.rover.status.vy };
  };
  const on = run({}), off = run({ slipEnabled: false, lateralEnabled: false });
  assert.ok(on.slip > 0.02, `slip ${on.slip}`);
  assert.equal(off.slip, 0); assert.equal(off.vy, 0);
  assert.notDeepEqual([on.x, on.y], [off.x, off.y]);
});
