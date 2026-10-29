import assert from 'node:assert/strict';
import test from 'node:test';

import { resolveConfig } from '../js/sim/config.js';
import { matVec } from '../js/sim/frames.js';
import { renderLidar } from '../js/sim/lidar.js';
import { buildObjects, OBJECT_LABEL, objectSunVisibility, rayCylinder } from '../js/sim/objects.js';
import { SCENARIOS } from '../js/sim/scenarios.js';
import { heightAt } from '../js/sim/terrain.js';
import { World } from '../js/sim/world.js';

function flat(extra = {}) {
  const sc = structuredClone(SCENARIOS.mare_plain);
  sc.cameras = []; sc.terrain.n = 1024; sc.terrain.res = 0.5; // ±256 m: the tower stands on the grid
  sc.terrain.fractal.enabled = sc.terrain.craters.enabled = sc.terrain.rocks.enabled = false;
  return Object.assign(sc, extra);
}

test('presets carry the 10 m x 50 m tower 200 m ahead of the start', () => {
  const o = SCENARIOS.south_pole_lowsun.objects[0];
  assert.deepEqual([o.type, o.x, o.y, o.diameter, o.height], ['cylinder', 200, 0, 10, 50]);
  const w = new World(flat()), c = w.objects[0], g = heightAt(w.terrain, 200, 0);
  assert.equal(c.outside, false);
  assert.ok(Math.abs(c.zt - g - 50) < 1e-9 && c.zb < g && Math.abs(c.r - 5) < 1e-12);
  const sc = structuredClone(SCENARIOS.south_pole_lowsun); sc.terrain.n = 256; sc.cameras = [];
  assert.equal(new World(sc).objects[0].outside, true); // the small grid does not reach 200 m
});

test('ray / cylinder intersection: side, top cap, misses, inside', () => {
  const c = { cx: 10, cy: 0, r: 2, zb: -1, zt: 5 };
  let h = rayCylinder(0, 0, 1, 1, 0, 0, c, 0, Infinity);
  assert.ok(Math.abs(h.t - 8) < 1e-12); assert.deepEqual(h.n, [-1, 0, 0]);
  const s = Math.SQRT1_2; // from above onto the cap
  h = rayCylinder(10 - 3, 0, 8, s, 0, -s, c, 0, Infinity);
  assert.ok(Math.abs(h.t - 3 / s) < 1e-12); assert.deepEqual(h.n, [0, 0, 1]);
  assert.equal(rayCylinder(0, 3, 1, 1, 0, 0, c, 0, Infinity), null); // passes beside
  assert.equal(rayCylinder(0, 0, 6, 1, 0, 0, c, 0, Infinity), null); // passes above
  assert.equal(rayCylinder(0, 0, 1, 1, 0, 0, c, 0, 7.9), null); // beyond t1
  h = rayCylinder(10, 0, 1, 1, 0, 0, c, 0, Infinity); // from inside: the far wall
  assert.ok(Math.abs(h.t - 2) < 1e-12);
  const d = [10, 2.2, 0.5], n = Math.hypot(...d); // oblique ray near the edge: lands on the wall
  h = rayCylinder(0, -0.5, 0, d[0] / n, d[1] / n, d[2] / n, c, 0, Infinity);
  const p = [d[0] / n * h.t, -0.5 + d[1] / n * h.t];
  assert.ok(Math.abs(Math.hypot(p[0] - 10, p[1]) - 2) < 1e-9);
});

test('LiDAR sees the tower (object label) when it is within range; terrain behind it is hidden', () => {
  const sc = flat();
  sc.objects = [{ name: 'tower', type: 'cylinder', x: 40, y: 0, diameter: 10, height: 50 }];
  const w = new World(sc), l = w.lidars.find((x) => x.cfg.name === 'spin32');
  const pose = w.sensorPose(l.cfg.mount), scan = renderLidar(l.cfg, l.pattern, pose, w.terrain, w.caster, null, 0, 1);
  let onTower = 0;
  for (let k = 0; k < scan.count; k++) {
    if (scan.labels[k] !== OBJECT_LABEL) continue;
    onTower++;
    const x = scan.world[3 * k], y = scan.world[3 * k + 1];
    assert.ok(Math.abs(Math.hypot(x - 40, y) - 5) < 0.2, `point off the wall: ${x}, ${y}`); // ± range noise
  }
  assert.ok(onTower > 100, `${onTower} points on the tower`);
  // a beam straight at the tower hits the wall at 35 m - sensor x
  const d = [1, 0, 0], h = w.caster.traceHit(pose.t[0], pose.t[1], pose.t[2], ...d);
  assert.equal(h.object.name, 'tower'); assert.ok(Math.abs(h.t - (35 - pose.t[0])) < 1e-9);
  assert.ok(w.caster.terrain.trace(pose.t[0], pose.t[1], pose.t[2], ...d) === Infinity);
});

test('the tower casts a sun shadow on the terrain and its sunward wall is lit', () => {
  const w = new World(flat());
  w.setSunManual(90, 20); // sun in the east, 20° up: the shadow points west of the tower
  const c = w.objects[0], s = w.sun.dir, L = 50 / Math.tan(20 * Math.PI / 180);
  const lit = (x, y) => !w.caster.occluded(x, y, heightAt(w.terrain, x, y) + 2e-3, s[0], s[1], s[2]);
  assert.equal(lit(c.cx - 6, 0), false); // just behind the tower
  assert.equal(lit(c.cx - L + 3, 0), false); // near the shadow tip (~137 m long)
  assert.equal(lit(c.cx - L - 5, 0), true); // beyond the tip
  assert.equal(lit(c.cx - 20, 8), true); // beside the shadow
  const v = objectSunVisibility([c], w.caster, s, 64, 64)[0];
  assert.equal(v.vis[40 * 64 + 0], 255); // wall facing +x (sunward), mid height: lit
  assert.equal(v.vis[40 * 64 + 32], 0); // facing -x: dark
});

test('config: objects merged by name, new ones from the template, validated', () => {
  const sc = resolveConfig({ objects: [{ name: 'tower', x: 120 }, { name: 'mast2', y: 30, height: 20 }] });
  assert.deepEqual(sc.objects.map((o) => [o.name, o.x, o.y, o.diameter, o.height]), [['tower', 120, 0, 10, 50], ['mast2', 0, 30, 10, 20]]);
  assert.equal(resolveConfig({ objects: [] }).objects.length, 0);
  assert.throws(() => resolveConfig({ objects: [{ name: 'b', type: 'box' }] }), /cylinder/);
  assert.throws(() => resolveConfig({ objects: [{ name: 'tower', diameter: -1 }] }), /diameter/);
  const w = new World(Object.assign(flat(), { objects: [{ name: 't', type: 'cylinder', x: 30, y: 0, diameter: 4, height: 9, enabled: false }] }));
  assert.equal(w.objects.length, 0);
  assert.equal(buildObjects({ objects: [{ name: 'q', type: 'cylinder', x: 0, y: 0, diameter: 2, height: 3 }] }, w.terrain).length, 1);
  void matVec;
});
