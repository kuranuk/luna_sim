// SPAD flash LiDAR: photon statistics equal the Python model; the range image has no holes.
import assert from 'node:assert/strict';
import test from 'node:test';

import { matVec } from '../js/sim/frames.js';
import { flashSolidAngle, NOISE_LABEL, poissonThreshold, renderLidar, spadPixel, spadSetup } from '../js/sim/lidar.js';
import { SCENARIOS } from '../js/sim/scenarios.js';
import { facetNormalAt, heightAt, normalAt } from '../js/sim/terrain.js';
import { World } from '../js/sim/world.js';

test('Poisson threshold = scipy poisson.isf(q) + 1', () => {
  // (lam, q, T) from scipy.stats.poisson.isf(q, lam) + 1
  for (const [lam, q, T] of [[0.002, 2.994e-6, 2], [101.1, 2.994e-6, 151], [125.9, 2.994e-6, 181], [60.76, 2.994e-6, 100], [3, 1e-3, 11], [0, 1e-3, 1]]) {
    assert.equal(poissonThreshold(lam, q)[0], T, `lam ${lam}`);
  }
  const [T, tail] = poissonThreshold(5000, 1e-6); // Cornish-Fisher branch: close to the exact 5341
  assert.ok(Math.abs(T - 5341) <= 2, `T ${T}`); assert.equal(tail, 1e-6);
});

test('SPAD rates, window background and thresholds equal lunarsim/core/sensors/spad.py', () => {
  const cfg = structuredClone(SCENARIOS.mare_plain.lidars[0]);
  cfg.mount.pitchDeg = 0; cfg.spad.pulseEnergy = 1e-6; // the reference values were computed for 1 uJ
  const su = spadSetup(cfg, cfg.width * cfg.height);
  assert.ok(Math.abs(su.wT - 2e-9) < 1e-20); assert.equal(su.nWindows, 334);
  // [r, cos, mu0] -> [n_s, phi, mu_B, T, p_s] from spad_rates / _bg_window_mean / spad_threshold
  const ref = [
    [[5, 0.6, 0], [6.15, 1000, 0.002, 2, 0.9979]], [[50, 0.07, 0], [0.007176, 1000, 0.002, 2, 0.00715]],
    [[10, 0.35, 0.5], [0.8969, 1.141e8, 101.1, 151, 0.2766]], [[60, 0.06, 0.9], [0.004271, 2.054e8, 125.9, 181, 0.001396]],
    [[95, 0.04, 0.2], [0.001136, 4.564e7, 60.76, 100, 0.0007794]],
  ];
  for (const [[r, cos, mu0], [ns0, phi0, muB0, T0, pS0]] of ref) {
    const ns = su.K * cfg.reflectance * cos / (r * r), phi = cfg.spad.dcr + cfg.reflectance * mu0 * su.bgK, st = spadPixel(cfg, su, ns, phi);
    for (const [got, want] of [[ns, ns0], [phi, phi0], [st.muB, muB0], [st.pS, pS0]]) assert.ok(Math.abs(got / want - 1) < 2e-3, `${got} vs ${want}`);
    assert.equal(st.T, T0);
    assert.ok(st.pFalse <= cfg.spad.pfa && st.pFalse > 0.3 * cfg.spad.pfa);
  }
  assert.ok(Math.abs(flashSolidAngle({ projection: 'pinhole', hfovDeg: 60, vfovDeg: 45 }) - 4 * Math.asin(Math.sin(Math.PI / 6) * Math.sin(Math.PI / 8))) < 1e-12);
});

function scene(preset, flat, sunEl) {
  const sc = structuredClone(SCENARIOS[preset]); sc.cameras = []; sc.objects = [];
  if (flat) sc.terrain.fractal.enabled = sc.terrain.craters.enabled = sc.terrain.rocks.enabled = false;
  else sc.terrain.n = 512;
  const w = new World(sc);
  if (sunEl !== null) w.setSunManual(135, sunEl);
  return w;
}

test('range image has no holes: every pixel that sees the surface within rangeMax returns a range', () => {
  for (const [name, w] of [['flat, dark', scene('mare_plain', true, -5)], ['flat, sun 30°', scene('mare_plain', true, 30)],
    ['flat, sun 75°', scene('mare_plain', true, 75)], ['south pole', scene('south_pole_lowsun', false, null)], ['crater field', scene('crater_field', false, null)]]) {
    const l = w.lidars.find((x) => x.cfg.name === 'spad_flash');
    assert.equal(l.cfg.spad.pulseEnergy, 5e-4);
    for (const [x, y, yaw] of [[0, 0, 0], [12, -9, 2.2]]) {
      w.rover.reset(x, y, yaw);
      const s = renderLidar(l.cfg, l.pattern, w.sensorPose(l.cfg.mount), w.terrain, w.caster, w.sun, 0, 1);
      assert.equal(s.range.length, l.cfg.width * l.cfg.height);
      let hits = 0, holes = 0, falseAlarms = 0, worst = 0;
      for (let b = 0; b < s.beams; b++) {
        const hit = !Number.isNaN(s.rangeTrue[b]), got = !Number.isNaN(s.range[b]);
        if (hit && !got) holes++;
        if (hit) hits++;
        if (!hit && got) falseAlarms++;
        if (hit && got) worst = Math.max(worst, Math.abs(s.range[b] - s.rangeTrue[b]));
      }
      assert.ok(hits > 10000, `${name}: ${hits} surface pixels`);
      assert.equal(holes, 0, `${name}: ${holes} holes in ${hits} surface pixels`);
      assert.ok(falseAlarms <= 12, `${name}: ${falseAlarms} false alarms`); // P_fa 1e-3 on the sky pixels
      assert.ok(worst < 0.05, `${name}: range error ${worst}`); // timing noise + TDC bin, walk corrected
      // the point list is the same measurement: range[beam] * direction
      for (let k = 0; k < s.count; k += 97) {
        const r = Math.hypot(s.points[4 * k], s.points[4 * k + 1], s.points[4 * k + 2]);
        assert.ok(Math.abs(r - s.range[s.beam[k]]) < 1e-4);
        if (s.labels[k] === NOISE_LABEL) assert.ok(Number.isNaN(s.rangeTrue[s.beam[k]]));
      }
    }
  }
});

test('the old 1 uJ budget loses far / sunlit returns (physics), and silhouettes use the facet that is hit', () => {
  const w = scene('mare_plain', true, 30), l = w.lidars.find((x) => x.cfg.name === 'spad_flash');
  l.cfg.spad.pulseEnergy = 1e-6;
  const s = renderLidar(l.cfg, l.pattern, w.sensorPose(l.cfg.mount), w.terrain, w.caster, w.sun, 0, 1);
  let far = 0, farGot = 0, near = 0, nearGot = 0;
  for (let b = 0; b < s.beams; b++) {
    const t = s.rangeTrue[b];
    if (t > 40) { far++; if (!Number.isNaN(s.range[b])) farGot++; }
    if (t < 8) { near++; if (!Number.isNaN(s.range[b])) nearGot++; }
  }
  assert.equal(nearGot, near); assert.ok(far > 500 && farGot < 0.05 * far, `${farGot}/${far}`);
  // a ray that hits a surface whose smooth normal faces away still has a positive facet cosine
  const wr = scene('south_pole_lowsun', false, null), hm = wr.terrain, lr = wr.lidars[0], pose = wr.sensorPose(lr.cfg.mount);
  let checked = 0;
  for (let b = 0; b < lr.pattern.n; b++) {
    const d = matVec(pose.R, lr.pattern.dirs.slice(3 * b, 3 * b + 3)), t = wr.caster.trace(pose.t[0], pose.t[1], pose.t[2], d[0], d[1], d[2], 0, 100);
    if (t === Infinity) continue;
    const p = pose.t.map((v, k) => v + d[k] * t), n = normalAt(hm, p[0], p[1]);
    if (-(n[0] * d[0] + n[1] * d[1] + n[2] * d[2]) >= 0) continue;
    const f = facetNormalAt(hm, p[0], p[1]);
    assert.ok(Math.abs(f[0] * d[0] + f[1] * d[1] + f[2] * d[2]) > 0, 'facet cosine');
    assert.ok(Math.abs(heightAt(hm, p[0], p[1]) - p[2]) < 1e-6);
    checked++;
  }
  assert.ok(checked >= 1, `${checked} silhouette pixels`);
});

test('range walk: corrected by default, present (towards the sensor) when switched off', () => {
  const w = scene('mare_plain', true, -5), l = w.lidars.find((x) => x.cfg.name === 'spad_flash');
  const bias = () => {
    const s = renderLidar(l.cfg, l.pattern, w.sensorPose(l.cfg.mount), w.terrain, w.caster, w.sun, 0, 1);
    let sum = 0, n = 0;
    for (let b = 0; b < s.beams; b++) if (s.rangeTrue[b] < 8 && !Number.isNaN(s.range[b])) { sum += s.range[b] - s.rangeTrue[b]; n++; }
    return sum / n;
  };
  assert.ok(Math.abs(bias()) < 1e-3);
  l.cfg.spad.walkCorrection = false;
  const b = bias();
  assert.ok(b < -0.05 && b > -0.5, `near-range walk ${b} m`);
});
