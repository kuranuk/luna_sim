// World: terrain + sun + rover + sensors, advanced in fixed sub-steps (pure JS, no DOM).
// Rover sub-steps end on IMU sample instants (as World.step in the Python simulator);
// LiDAR and camera frames are taken on the first sub-step boundary at or after their due time.
// Cameras need a GPU renderer injected as world.cameraRenderer (see ../gl/camera_gl.js); without
// one (Node, no WebGL2) camera frames are skipped.

import { expose, intrinsics } from './camera.js';
import { compose, DEG, mountPose, quat } from './frames.js';
import { Imu } from './imu.js';
import { renderLidar, scanPattern } from './lidar.js';
import { HeightfieldCaster } from './raycast.js';
import { Rover, WaypointFollower } from './rover.js';
import { sunAstronomical, sunManual } from './sun.js';
import { buildTerrain, heightAt, nodeNormal } from './terrain.js';

export class World {
  constructor(scenario) {
    this.scenario = structuredClone(scenario);
    const t0 = performance.now();
    this.terrain = buildTerrain(this.scenario);
    this.caster = new HeightfieldCaster(this.terrain);
    this.buildMs = performance.now() - t0;
    this.lidars = this.scenario.lidars.map((cfg) => ({ cfg, pattern: scanPattern(cfg), frame: 0, next: 0, enabled: cfg.enabled !== false }));
    this.cameras = (this.scenario.cameras || []).map((cfg) => ({ cfg, K: intrinsics(cfg), frame: 0, next: 0, enabled: cfg.enabled !== false }));
    this.lightsOn = true;
    this.cameraRenderer = null;
    this.sunVersion = 0;
    this.mountVersion = 0;
    this.imu = this.scenario.imu ? new Imu(this.scenario.imu, this.scenario.seed, this.scenario.site.lat) : null;
    this.setSunFromScenario();
    this.reset();
  }

  height = (x, y) => heightAt(this.terrain, x, y);

  reset() {
    this.time = 0;
    this.nextPoseLog = 0;
    this.follower = null;
    this.rover = new Rover(this.scenario.rover, this.height);
    for (const l of [...this.lidars, ...this.cameras]) { l.frame = 0; l.next = 0; }
    if (this.imu) { this.imu.reset(); this.imuSamples = [this.imu.push(this.sensorPose(this.scenario.imu.mount))]; }
    this.trail = [[this.rover.x, this.rover.y, this.rover.contact.t[2]]];
  }

  setSunFromScenario() {
    const s = this.scenario.sun;
    this.sun = s.mode === 'astronomical'
      ? sunAstronomical(this.scenario.site.lat, this.scenario.site.lon, new Date(s.utc))
      : sunManual(s.azDeg, s.elDeg);
    this.sunVersion++;
    this._sunlitFull = null;
  }

  setSunManual(azDeg, elDeg) { Object.assign(this.scenario.sun, { mode: 'manual', azDeg, elDeg }); this.setSunFromScenario(); }

  setSunUtc(utc) { Object.assign(this.scenario.sun, { mode: 'astronomical', utc }); this.setSunFromScenario(); }

  /**
   * Change a sensor's field of view [deg] without rebuilding the terrain. LiDAR flash: { hfovDeg,
   * vfovDeg }; spinning: { vMinDeg, vMaxDeg }; camera: { hfovDeg } (square pixels, so the vertical
   * angle follows the aspect ratio). Values are clamped to valid ranges; returns the new config.
   * Flash SPAD: each pixel still gets 1/N of the pulse (same signal per pixel), but sees a larger
   * solid angle, hence more solar background and a higher detection threshold.
   */
  setFov(kind, name, p) {
    const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, +v));
    if (kind === 'lidar') {
      const l = this.lidars.find((x) => x.cfg.name === name);
      if (!l) throw new Error(`no LiDAR ${name}`);
      const c = l.cfg;
      if (c.kind === 'flash') {
        if (p.hfovDeg !== undefined) c.hfovDeg = clamp(p.hfovDeg, 1, 170);
        if (p.vfovDeg !== undefined) c.vfovDeg = clamp(p.vfovDeg, 1, 170);
      } else {
        const lo = clamp(p.vMinDeg ?? c.vMinDeg, -90, 89), hi = clamp(p.vMaxDeg ?? c.vMaxDeg, lo + 0.1, 90);
        c.vMinDeg = lo; c.vMaxDeg = hi;
      }
      l.pattern = scanPattern(c);
      return c;
    }
    const cam = this.cameras.find((x) => x.cfg.name === name);
    if (!cam) throw new Error(`no camera ${name}`);
    if (p.hfovDeg !== undefined) cam.cfg.hfovDeg = clamp(p.hfovDeg, 5, 170);
    delete cam.cfg.fx; delete cam.cfg.fy;
    cam.K = intrinsics(cam.cfg);
    return cam.cfg;
  }

  /** Move a sensor or light: mount = { x, y, z [m] (body FLU), rollDeg, pitchDeg, yawDeg } (partial ok). */
  setMount(kind, name, mount) {
    const list = kind === 'lidar' ? this.lidars.map((l) => l.cfg) : kind === 'camera' ? this.cameras.map((c) => c.cfg)
      : (this.scenario.lights || []);
    const cfg = list.find((c) => c.name === name);
    if (!cfg) throw new Error(`no ${kind} ${name}`);
    for (const k of ['x', 'y', 'z', 'rollDeg', 'pitchDeg', 'yawDeg']) {
      if (mount[k] === undefined) continue;
      const v = +mount[k];
      if (!Number.isFinite(v)) throw new Error(`${name}.mount.${k} = ${mount[k]}`);
      cfg.mount[k] = k === 'pitchDeg' ? Math.min(90, Math.max(-90, v)) // pitch clamped, roll / yaw wrapped
        : k.endsWith('Deg') ? ((v + 180) % 360 + 360) % 360 - 180 : Math.min(50, Math.max(-50, v));
    }
    this.mountVersion++;
    return cfg;
  }

  /** setFov + setMount in one call (params may hold field-of-view keys and `mount`). */
  setSensor(kind, name, params) {
    const { mount, ...fov } = params;
    if (mount) this.setMount(kind, name, mount);
    if (kind !== 'light' && Object.keys(fov).length) this.setFov(kind, name, fov);
    const list = kind === 'lidar' ? this.lidars.map((l) => l.cfg) : kind === 'camera' ? this.cameras.map((c) => c.cfg) : this.scenario.lights;
    return list.find((c) => c.name === name);
  }

  setCommand(v, omega) { this.follower = null; this.rover.setCommand(v, omega); }

  setWaypoints(points, speed) { this.follower = new WaypointFollower(points, speed); }

  sensorPose(mount) { return compose(this.rover.pose, mountPose(mount)); }

  /**
   * Advance by dt [s]. Returns { imu: [samples], lidar: [{name, frame, time, pose, scan}],
   * camera: [{name, frame, time, pose, image}], poses: [[t, x, y, z, qw, qx, qy, qz] at 10 Hz] }.
   * scans: 'all' renders every due LiDAR frame at its sub-step (deterministic, for closed-loop
   * runs); 'latest' renders at most one frame per LiDAR at the end of the call and skips the
   * missed slots (interactive real-time use); false renders none.
   */
  advance(dt, { control = 0.1, scans = 'all' } = {}) {
    const out = { imu: [], lidar: [], camera: [], poses: [] };
    const h = this.imu ? this.imu.period : 0.01;
    let remaining = dt;
    while (remaining > 1e-12) {
      if (this.follower && (this.ctrlLeft === undefined || this.ctrlLeft <= 1e-12)) {
        const [v, w] = this.follower.command(this.rover.x, this.rover.y, this.rover.yaw);
        this.rover.setCommand(v, w);
        this.ctrlLeft = control;
        if (this.follower.done) this.follower = null;
      }
      let step = Math.min(remaining, h);
      if (this.imu) step = Math.min(step, Math.max(this.imu.nextTime - this.time, 0));
      if (step > 0) this.rover.step(step);
      this.time += step; remaining -= step;
      if (this.imu && Math.abs(this.imu.nextTime - this.time) < 1e-9) this.time = this.imu.nextTime; // no drift
      if (this.ctrlLeft !== undefined) this.ctrlLeft -= step;
      if (this.imu) {
        while (this.imu.nextTime <= this.time + 1e-9) out.imu.push(this.imu.push(this.sensorPose(this.scenario.imu.mount)));
      }
      while (this.nextPoseLog <= this.time + 1e-9) { // ground-truth trajectory at 10 Hz
        out.poses.push([this.time, ...this.rover.contact.t, ...quat(this.rover.contact.R)]);
        this.nextPoseLog += 0.1;
      }
      if (scans === 'all') this.takeFrames(out, false);
    }
    if (scans === 'latest') this.takeFrames(out, true);
    const last = this.trail[this.trail.length - 1];
    if (Math.hypot(this.rover.x - last[0], this.rover.y - last[1]) > 0.05) this.trail.push([this.rover.x, this.rover.y, this.rover.contact.t[2]]);
    return out;
  }

  /** Frames of all due sensors; skipMissed: one frame per sensor, missed slots dropped. */
  takeFrames(out, skipMissed) {
    const cams = this.cameraRenderer ? this.cameras : [];
    for (const [list, key, fn] of [[this.lidars, 'lidar', (l) => this.scan(l)], [cams, 'camera', (c) => this.shoot(c)]]) {
      for (const l of list) {
        if (!l.enabled || this.time + 1e-9 < l.next) continue;
        out[key].push(fn(l));
        const period = 1 / l.cfg.rateHz;
        l.next += skipMissed ? period * Math.max(1, Math.ceil((this.time + 1e-9 - l.next) / period)) : period;
      }
    }
  }

  /** Render one camera frame (GPU radiance -> sensor model). */
  shoot(c) {
    const pose = this.sensorPose(c.cfg.mount);
    const t0 = performance.now(), radiance = this.cameraRenderer.render(this, c, pose), t1 = performance.now();
    const img = expose(radiance, c.cfg.sensor, c.cfg.name, c.frame, this.scenario.seed);
    return { name: c.cfg.name, frame: c.frame++, time: this.time, pose, gpuMs: t1 - t0, sensorMs: performance.now() - t1,
      image: { ...img, width: c.K.width, height: c.K.height, bits: c.cfg.sensor.bits, K: c.K } };
  }

  /** Full-resolution sun visibility per node (cached until the sun changes). */
  sunlitFull() {
    if (!this._sunlitFull) this._sunlitFull = this.sunlit(1);
    return this._sunlitFull;
  }

  scan(l) {
    const pose = this.sensorPose(l.cfg.mount);
    const scan = renderLidar(l.cfg, l.pattern, pose, this.terrain, this.caster, this.sun, l.frame, this.scenario.seed);
    return { name: l.cfg.name, frame: l.frame++, time: this.time, pose, scan };
  }

  state() {
    const r = this.rover, c = r.contact;
    return {
      time: this.time,
      rover: {
        position: c.t, quaternion: quat(c.R), R: c.R, x: r.x, y: r.y, yawDeg: r.yaw / DEG,
        rollDeg: c.roll / DEG, pitchDeg: c.pitch / DEG, slopeDeg: c.slope / DEG, odometer: r.odometer,
        command: { v: r.v, omega: r.omega }, slip: r.status.slip, warnings: r.status.warnings,
        waypointsRemaining: this.follower ? this.follower.points.length - this.follower.index : 0,
      },
      sun: { azDeg: this.sun.az / DEG, elDeg: this.sun.el / DEG, dir: this.sun.dir },
    };
  }

  /** Per-node direct-sun visibility (0/1) at every `stride`-th node (shadow rays), for display. */
  sunlit(stride = 1) {
    const hm = this.terrain, { z, n, res, x0, y0 } = hm, s = this.sun.dir;
    const m = Math.floor((n - 1) / stride) + 1, out = new Uint8Array(m * m);
    if (this.sun.el <= 0) return { lit: out, m, stride };
    for (let a = 0; a < m; a++) {
      for (let b = 0; b < m; b++) {
        const i = a * stride, j = b * stride, nm = nodeNormal(hm, i, j);
        if (nm[0] * s[0] + nm[1] * s[1] + nm[2] * s[2] <= 0) continue;
        const ox = x0 + j * res + 2e-3 * nm[0], oy = y0 + i * res + 2e-3 * nm[1], oz = z[i * n + j] + 2e-3 * nm[2];
        out[a * m + b] = this.caster.occluded(ox, oy, oz, s[0], s[1], s[2]) ? 0 : 1;
      }
    }
    return { lit: out, m, stride };
  }
}
