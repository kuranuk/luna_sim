// Rover: skid-steer kinematics, static terrain contact, slope-dependent slip, warnings
// (port of lunarsim/core/rover). Commands: forward speed v [m/s], yaw rate omega [rad/s].
// Ground track (x, y) = contact centre, yaw from East counter-clockwise.

import { DEG, eulerZYX } from './frames.js';

export function wheelOffsets(cfg) {
  const hx = cfg.wheelbase / 2, hy = cfg.track / 2;
  return [[hx, hy], [hx, -hy], [-hx, hy], [-hx, -hy]];
}

/** Rest the rover on the terrain: each wheel on the highest point of a 3x3 footprint, plane fit. */
export function settle(height, cfg, x, y, yaw) {
  const w = wheelOffsets(cfg), lx = cfg.contactHalfLengthRatio * cfg.wheelRadius, ly = cfg.wheelWidth / 2;
  const foot = [];
  for (const fu of [-lx, 0, lx]) for (const fv of [-ly, 0, ly]) foot.push([fu, fv]);
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  let roll = 0, pitch = 0, a = 0, wz = [0, 0, 0, 0], wx = [0, 0, 0, 0], wy = [0, 0, 0, 0];
  for (let it = 0; it < 4; it++) {
    const R = eulerZYX(roll, pitch, yaw);
    let sn = 0, su = 0, sv = 0, suu = 0, suv = 0, svv = 0, sz = 0, suz = 0, svz = 0;
    for (let k = 0; k < 4; k++) {
      wx[k] = x + R[0] * w[k][0] + R[1] * w[k][1];
      wy[k] = y + R[3] * w[k][0] + R[4] * w[k][1];
      let m = -Infinity;
      for (const f of foot) {
        const bx = w[k][0] + f[0], by = w[k][1] + f[1];
        m = Math.max(m, height(x + R[0] * bx + R[1] * by, y + R[3] * bx + R[4] * by));
      }
      wz[k] = m;
      const u = (wx[k] - x) * cy + (wy[k] - y) * sy, v = -(wx[k] - x) * sy + (wy[k] - y) * cy;
      sn++; su += u; sv += v; suu += u * u; suv += u * v; svv += v * v; sz += m; suz += u * m; svz += v * m;
    }
    // Solve [[n, Σu, Σv], [Σu, Σuu, Σuv], [Σv, Σuv, Σvv]] (a, b, c) = (Σz, Σuz, Σvz)
    const c00 = suu * svv - suv * suv, c01 = sv * suv - su * svv, c02 = su * suv - sv * suu;
    const det = sn * c00 + su * c01 + sv * c02;
    const c11 = sn * svv - sv * sv, c12 = su * sv - sn * suv, c22 = sn * suu - su * su;
    a = (c00 * sz + c01 * suz + c02 * svz) / det;
    const b = (c01 * sz + c11 * suz + c12 * svz) / det, c = (c02 * sz + c12 * suz + c22 * svz) / det;
    const nrm = Math.sqrt(1 + b * b + c * c);
    const rollN = Math.asin(c / nrm), pitchN = Math.atan2(-b, 1);
    const done = Math.abs(rollN - roll) < 1e-12 && Math.abs(pitchN - pitch) < 1e-12;
    roll = rollN; pitch = pitchN;
    if (done) break;
  }
  const R = eulerZYX(roll, pitch, yaw);
  const t = [x + cfg.originHeight * R[2], y + cfg.originHeight * R[5], a + cfg.originHeight * R[8]];
  const g = [-R[6], -R[7], -R[8]]; // gravity (0, 0, -1) in the body frame = R^T (0,0,-1)
  const tiltLat = Math.abs(Math.atan2(g[1], -g[2])), tiltLon = Math.abs(Math.atan2(g[0], -g[2]));
  return {
    R, t, roll, pitch, yaw, wheels: wx.map((_, k) => [wx[k], wy[k], wz[k]]),
    slope: Math.acos(Math.min(1, Math.max(-1, R[8]))),
    lateralMargin: Math.atan2(cfg.track / 2, cfg.cgHeight) - tiltLat,
    longitudinalMargin: Math.atan2(cfg.wheelbase / 2, cfg.cgHeight) - tiltLon,
  };
}

export class Rover {
  constructor(cfg, height) {
    this.cfg = cfg; this.height = height;
    this.reset(cfg.x, cfg.y, cfg.yawDeg * DEG);
  }

  reset(x, y, yaw) {
    Object.assign(this, { x, y, yaw, v: 0, omega: 0, odometer: 0 });
    this.contact = settle(this.height, this.cfg, x, y, yaw);
    this.status = { slip: 0, vx: 0, vy: 0, yawRate: 0, warnings: this.warnings(0, 0) };
  }

  setCommand(v, omega) {
    const c = this.cfg;
    this.v = Math.max(-c.maxSpeed, Math.min(c.maxSpeed, v));
    this.omega = Math.max(-c.maxYawRateDeg * DEG, Math.min(c.maxYawRateDeg * DEG, omega));
  }

  slipFor(uphill) {
    const c = this.cfg, crit = c.criticalSlopeDeg * DEG;
    if (uphill >= 0) return Math.min(1, c.slipFlat + (1 - c.slipFlat) * (uphill / crit) ** c.slipExponent);
    return -Math.min(1, c.downhillSkid * (-uphill / crit) ** c.slipExponent);
  }

  warnings(slip, vCmd) {
    const c = this.cfg, ct = this.contact, w = [];
    if (ct.slope > c.maxSlopeDeg * DEG) w.push('slope_exceeded');
    const margin = Math.min(ct.lateralMargin, ct.longitudinalMargin);
    if (margin < 0) w.push('tipped_over');
    else if (margin < c.tipoverMarginDeg * DEG) w.push('tipover_risk');
    if (vCmd !== 0 && slip >= 1) w.push('stalled');
    return w;
  }

  /** Advance by h [s] (one sub-step; exact arc for constant velocities). */
  step(h) {
    if (h <= 0 || this.status.warnings.includes('tipped_over')) return this.status;
    const c = this.cfg, ct = this.contact, v = this.v;
    const slip = v !== 0 ? this.slipFor(v >= 0 ? -ct.pitch : ct.pitch) : 0;
    const vx = v * (1 - slip), vy = -c.lateralCoefficient * Math.abs(v) * Math.sin(ct.roll), wz = this.omega;
    const half = 0.5 * wz * h, yawMid = this.yaw + half, chord = Math.abs(half) > 1e-12 ? Math.sin(half) / half : 1;
    const vh = vx * Math.cos(ct.pitch);
    this.x += (vh * Math.cos(yawMid) - vy * Math.sin(yawMid)) * h * chord;
    this.y += (vh * Math.sin(yawMid) + vy * Math.cos(yawMid)) * h * chord;
    this.yaw = ((this.yaw + wz * h + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
    this.odometer += Math.abs(vx) * h;
    this.contact = settle(this.height, c, this.x, this.y, this.yaw);
    this.status = { slip, vx, vy, yawRate: wz, warnings: this.warnings(slip, v) };
    return this.status;
  }

  get pose() { return { R: this.contact.R, t: this.contact.t }; }
}

/** Waypoint follower (port of follower.py): ω = k ε, v = v_ref · max(0, cos ε). */
export class WaypointFollower {
  constructor(points, speed, tolerance = 0.3, headingGain = 1.5) {
    Object.assign(this, { points, speed, tolerance, headingGain, index: 0 });
  }

  get done() { return this.index >= this.points.length; }

  command(x, y, yaw) {
    while (!this.done) {
      const [px, py] = this.points[this.index];
      if (Math.hypot(px - x, py - y) > this.tolerance) break;
      this.index++;
    }
    if (this.done) return [0, 0];
    const [px, py] = this.points[this.index];
    let err = Math.atan2(py - y, px - x) - yaw;
    err = ((err + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
    return [this.speed * Math.max(0, Math.cos(err)), this.headingGain * err];
  }
}
