// SPAD / spinning LiDAR ray casting against an independent reference (tests/ref_raycast.mjs).
import assert from 'node:assert/strict';
import test from 'node:test';

import { matVec } from '../js/sim/frames.js';
import { NOISE_LABEL, renderLidar } from '../js/sim/lidar.js';
import { SCENARIOS } from '../js/sim/scenarios.js';
import { heightAt } from '../js/sim/terrain.js';
import { World } from '../js/sim/world.js';
import { cellHit, refTrace } from './ref_raycast.mjs';

function roughWorld() { // south-pole terrain (craters + rocks) on a smaller grid
  const sc = structuredClone(SCENARIOS.south_pole_lowsun);
  sc.terrain.n = 256; sc.terrain.res = 0.2; sc.cameras = [];
  return new World(sc);
}

test('reference caster agrees with an all-cells brute force', () => {
  const w = roughWorld(), hm = w.terrain;
  for (let k = 0; k < 150; k++) {
    const ox = (k % 15) * 1.5 - 11 + 0.123, oy = Math.floor(k / 15) * 2 - 10 + 0.377, oz = heightAt(hm, ox, oy) + 0.3 + (k % 7) * 0.3;
    const az = k * 2.399, el = -0.02 - (k % 11) * 0.05, d = [Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el)];
    const L = 8, tr = refTrace(hm, [ox, oy, oz], d, L);
    let tb = Infinity;
    for (let i = 0; i < hm.n - 1; i++) {
      for (let j = 0; j < hm.n - 1; j++) {
        const x = hm.x0 + j * hm.res, y = hm.y0 + i * hm.res;
        if (Math.abs(x - ox) < L + 1 && Math.abs(y - oy) < L + 1) tb = Math.min(tb, cellHit(hm, i, j, [ox, oy, oz], d));
      }
    }
    assert.equal(tr, tb > L ? Infinity : tb, `ray ${k}`);
  }
});

test('every SPAD and spinning LiDAR beam matches the reference (hits, misses, distance)', () => {
  const w = roughWorld(), hm = w.terrain;
  for (const name of ['spad_flash', 'spin32']) {
    const l = w.lidars.find((x) => x.cfg.name === name);
    let hits = 0;
    for (let k = 0; k < 3; k++) {
      w.rover.reset(-15 + 12 * k, 8 - 9 * k, 0.4 + 2.1 * k);
      const pose = w.sensorPose(l.cfg.mount), o = pose.t, tmax = l.cfg.rangeMax * 1.05;
      for (let b = 0; b < l.pattern.n; b++) {
        const d = matVec(pose.R, [l.pattern.dirs[3 * b], l.pattern.dirs[3 * b + 1], l.pattern.dirs[3 * b + 2]]);
        const tc = w.caster.trace(o[0], o[1], o[2], d[0], d[1], d[2], 0, tmax), tr = refTrace(hm, o, d, tmax);
        if (tr === Infinity) assert.equal(tc, Infinity, `${name} pose ${k} beam ${b}: false hit at ${tc}`);
        else { assert.ok(Math.abs(tc - tr) < 1e-9, `${name} pose ${k} beam ${b}: ${tc} vs ${tr}`); hits++; }
      }
    }
    assert.ok(hits > 1000, `${name}: only ${hits} hits`);
  }
});

test('sun shadow rays (SPAD background) match the reference', () => {
  const w = roughWorld(), hm = w.terrain, s = w.sun.dir;
  let lit = 0;
  for (let k = 0; k < 3000; k++) {
    const x = -25 + ((k * 7.919) % 50), y = -25 + ((k * 13.37) % 50), z = heightAt(hm, x, y) + 2e-3;
    const oc = w.caster.occluded(x, y, z, s[0], s[1], s[2]);
    assert.equal(oc, refTrace(hm, [x, y, z], s, Infinity) < Infinity, `point ${k}`);
    if (!oc) lit++;
  }
  assert.ok(lit > 300 && lit < 2700, `lit ${lit}`); // low sun: a mix of light and shadow
});

test('edge rays: along grid lines, through vertices, vertical, horizontal, from outside the grid', () => {
  const w = roughWorld(), hm = w.terrain, g = hm.x0 + 100 * hm.res, gy = hm.y0 + 100 * hm.res;
  const cases = [
    [[g, gy, 3], [1, 0, -0.1]], [[g, gy, 3], [0, 1, -0.1]], [[g, gy, 3], [-1, 0, -0.1]], [[g, gy, 3], [1, 1, -0.05]],
    [[g + 0.03, gy + 0.07, 10], [0, 0, -1]], [[g, gy, 10], [0, 0, -1]], [[hm.x0 - 30, 0.11, 2], [1, 0, -0.03]],
    [[0, 0, heightAt(hm, 0, 0) + 0.5], [1, 0, 0]], [[0.2, 0.2, heightAt(hm, 0.2, 0.2) + 0.4], [-1, -1, 0]],
  ];
  for (const [o, d0] of cases) {
    const nn = Math.hypot(...d0), d = d0.map((x) => x / nn);
    const tc = w.caster.trace(o[0], o[1], o[2], d[0], d[1], d[2], 0, 500), tr = refTrace(hm, o, d, 500);
    assert.ok(tc === tr || Math.abs(tc - tr) < 1e-9, `${JSON.stringify(o)} ${JSON.stringify(d)}: ${tc} vs ${tr}`);
  }
});

test('SPAD flash geometry on a flat plane: pixel directions and ranges are analytic', () => {
  const sc = structuredClone(SCENARIOS.mare_plain);
  sc.terrain.n = 512; sc.terrain.res = 0.5; sc.cameras = []; // ±128 m: every hit within 100 m is on the grid
  sc.terrain.fractal.enabled = sc.terrain.craters.enabled = sc.terrain.rocks.enabled = false;
  const w = new World(sc), l = w.lidars.find((x) => x.cfg.name === 'spad_flash'), c = l.cfg;
  const pose = w.sensorPose(c.mount), h = pose.t[2] - heightAt(w.terrain, pose.t[0], pose.t[1]);
  const pitch = c.mount.pitchDeg * Math.PI / 180;
  // pixel (row 0, col 0) looks up-left, the last pixel down-right (FLU: +y left, +z up)
  const d0 = l.pattern.dirs.slice(0, 3), dn = l.pattern.dirs.slice(3 * (l.pattern.n - 1));
  assert.ok(d0[1] > 0 && d0[2] > 0 && dn[1] < 0 && dn[2] < 0);
  assert.ok(Math.abs(Math.atan2(d0[1], d0[0]) - (c.hfovDeg / 2) * (1 - 1 / c.width) * Math.PI / 180) < 0.01);
  for (let b = 0; b < l.pattern.n; b += 37) {
    const d = matVec(pose.R, [l.pattern.dirs[3 * b], l.pattern.dirs[3 * b + 1], l.pattern.dirs[3 * b + 2]]);
    const t = w.caster.trace(pose.t[0], pose.t[1], pose.t[2], d[0], d[1], d[2], 0, 105);
    const expected = d[2] < 0 ? h / -d[2] : Infinity;
    if (expected <= 105) assert.ok(Math.abs(t - expected) < 1e-9, `beam ${b}: ${t} vs ${expected}`);
    else assert.equal(t, Infinity);
  }
  // the centre row looks down by the mount pitch: range = h / sin(pitch)
  const mid = (c.height / 2) * c.width + c.width / 2, dm = matVec(pose.R, l.pattern.dirs.slice(3 * mid, 3 * mid + 3));
  assert.ok(Math.abs(Math.asin(-dm[2]) - pitch) < 0.01);
});

test('SPAD points lie on their beams at the true range (± timing noise), labels and frames consistent', () => {
  const w = roughWorld(), hm = w.terrain, l = w.lidars.find((x) => x.cfg.name === 'spad_flash');
  w.rover.reset(3, -4, 0.7);
  const pose = w.sensorPose(l.cfg.mount), out = renderLidar(l.cfg, l.pattern, pose, hm, w.caster, w.sun, 0, 1);
  const sigma1 = 0.5 * 299792458 * Math.hypot(l.cfg.spad.jitter, l.cfg.spad.pulseFwhm / 2.3548); // S = 1
  let sum = 0, m = 0;
  for (let k = 0; k < out.count; k++) {
    const p = [out.points[4 * k], out.points[4 * k + 1], out.points[4 * k + 2]], r = Math.hypot(...p);
    const wp = matVec(pose.R, p).map((v, a) => v + pose.t[a]);
    for (let a = 0; a < 3; a++) assert.ok(Math.abs(wp[a] - out.world[3 * k + a]) < 1e-4);
    const b = out.beam[k], db = l.pattern.dirs.slice(3 * b, 3 * b + 3);
    for (let a = 0; a < 3; a++) assert.ok(Math.abs(p[a] / r - db[a]) < 1e-6, 'point off its beam');
    if (out.labels[k] === NOISE_LABEL) continue;
    const tr = refTrace(hm, pose.t, matVec(pose.R, db), 200);
    assert.ok(Math.abs(r - tr) < 6 * sigma1, `beam ${b}: range ${r} vs ${tr}`);
    sum += r - tr; m++;
  }
  assert.ok(m > 0.3 * out.beams, `${m} detections`);
  assert.ok(Math.abs(sum / m) < 2e-3, `range bias ${sum / m}`);
});

test('field of view can be changed: patterns, clamping, ray casting and camera intrinsics', () => {
  const w = roughWorld(), hm = w.terrain, l = w.lidars.find((x) => x.cfg.name === 'spad_flash');
  const cfg = w.setFov('lidar', 'spad_flash', { hfovDeg: 120, vfovDeg: 90 });
  assert.equal(cfg.hfovDeg, 120); assert.equal(l.pattern.n, 160 * 120);
  const d0 = l.pattern.dirs; // first pixel centre: (1 - 1/W) of the half angle
  assert.ok(Math.abs(Math.atan2(d0[1], d0[0]) - Math.atan(Math.tan(60 * Math.PI / 180) * (1 - 1 / 160))) < 1e-12);
  const pose = w.sensorPose(l.cfg.mount);
  for (let b = 0; b < l.pattern.n; b += 7) {
    const d = matVec(pose.R, l.pattern.dirs.slice(3 * b, 3 * b + 3));
    const tc = w.caster.trace(pose.t[0], pose.t[1], pose.t[2], d[0], d[1], d[2], 0, 105), tr = refTrace(hm, pose.t, d, 105);
    assert.ok(tc === tr || Math.abs(tc - tr) < 1e-9, `beam ${b}`);
  }
  assert.equal(w.setFov('lidar', 'spad_flash', { hfovDeg: 500, vfovDeg: -3 }).hfovDeg, 170);
  assert.equal(l.cfg.vfovDeg, 1);
  const spin = w.setFov('lidar', 'spin32', { vMinDeg: 10, vMaxDeg: 5 });
  assert.ok(spin.vMaxDeg > spin.vMinDeg);
  const sc = structuredClone(SCENARIOS.mare_plain); sc.terrain.n = 64;
  const wc = new World(sc), K0 = wc.cameras[0].K.fx;
  wc.setFov('camera', 'navcam', { hfovDeg: 40 });
  assert.ok(Math.abs(wc.cameras[0].K.fx - 0.5 * 640 / Math.tan(20 * Math.PI / 180)) < 1e-9 && wc.cameras[0].K.fx > K0);
  assert.equal(wc.scenario.cameras[0].hfovDeg, 40); // the hello message carries the new value
});
