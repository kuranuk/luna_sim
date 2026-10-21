// 6-axis IMU driven by the rover trajectory (port of lunarsim/core/sensors/imu.py).
// Truth at sample k (period h, sensor frame S, world ENU W):
//   ω_S = rotvec(R_{k-1}^T R_k) / h + R_k^T Ω_moon,   Ω_moon = Ω (0, cos φ, sin φ) in ENU
//   f_S = R_k^T (a_W - g_W),  a_W = (p_k - 2 p_{k-1} + p_{k-2}) / h²,  g_W = (0, 0, -g)
// low-passed (filterOrder first-order stages at bandwidthHz; stands in for suspension + sensor
// filtering), then  y = truth + b + σ/√h n,  b_{k+1} = b_k + σ_rw √h n'  (Kalibr convention).
// Noise in blocks of 1024 samples from streams sensor/imu/{name}/block/{b}: independent of how
// simulation time is split into steps.

import { matTVec, matMul, rotvec, transpose } from './frames.js';
import { rng } from './rng.js';

const BLOCK = 1024;

export class Imu {
  constructor(cfg, seed, latitudeDeg) {
    this.cfg = cfg; this.seed = seed; this.period = 1 / cfg.rateHz;
    const la = latitudeDeg * Math.PI / 180;
    this.omegaMoon = [0, cfg.moonRotation * Math.cos(la), cfg.moonRotation * Math.sin(la)];
    this.reset();
  }

  reset() {
    this.k = 0; this.pos = []; this.Rprev = null; this.filt = null; this.blockIndex = -1; this.block = null;
    const r = rng(this.seed, `sensor/imu/${this.cfg.name}/bias`), c = this.cfg;
    this.bg = [0, 1, 2].map(() => c.gyroBias * r.normal());
    this.ba = [0, 1, 2].map(() => c.accelBias * r.normal());
  }

  get nextTime() { return this.k * this.period; }

  noise(k) {
    const b = Math.floor(k / BLOCK);
    if (b !== this.blockIndex) {
      const r = rng(this.seed, `sensor/imu/${this.cfg.name}/block/${b}`);
      this.block = Float64Array.from({ length: BLOCK * 12 }, () => r.normal());
      this.blockIndex = b;
    }
    return this.block.subarray((k % BLOCK) * 12, (k % BLOCK) * 12 + 12);
  }

  /** Record the sensor pose {R, t} at nextTime; returns the sample. */
  push(pose) {
    const c = this.cfg, h = this.period;
    this.pos.push(pose.t.slice());
    if (this.pos.length > 3) this.pos.shift();
    while (this.pos.length < 3) this.pos.unshift(pose.t.slice()); // at rest before the first sample
    const [p0, p1, p2] = this.pos;
    const aW = [0, 1, 2].map((i) => (p2[i] - 2 * p1[i] + p0[i]) / (h * h));
    aW[2] += c.gravity;
    const f = matTVec(pose.R, aW);
    const wRel = this.Rprev ? rotvec(matMul(transpose(this.Rprev), pose.R)).map((x) => x / h) : [0, 0, 0];
    const wm = matTVec(pose.R, this.omegaMoon);
    this.Rprev = pose.R.slice();
    let truth = [wRel[0] + wm[0], wRel[1] + wm[1], wRel[2] + wm[2], f[0], f[1], f[2]];
    if (c.bandwidthHz) {
      if (!this.filt) this.filt = Array.from({ length: c.filterOrder }, () => truth.slice());
      else {
        const alpha = 1 - Math.exp(-2 * Math.PI * c.bandwidthHz * h);
        let x = truth;
        for (let s = 0; s < c.filterOrder; s++) {
          this.filt[s] = this.filt[s].map((y, i) => y + alpha * (x[i] - y));
          x = this.filt[s];
        }
      }
      truth = this.filt[c.filterOrder - 1].slice();
    }
    const n = this.noise(this.k), sq = Math.sqrt(h);
    const sample = {
      t: this.nextTime,
      gyro: [0, 1, 2].map((i) => truth[i] + this.bg[i] + c.gyroNoise / sq * n[i]),
      accel: [0, 1, 2].map((i) => truth[3 + i] + this.ba[i] + c.accelNoise / sq * n[3 + i]),
      gyroTrue: truth.slice(0, 3), accelTrue: truth.slice(3), gyroBias: this.bg.slice(), accelBias: this.ba.slice(),
    };
    this.bg = this.bg.map((b, i) => b + c.gyroRandomWalk * sq * n[6 + i]);
    this.ba = this.ba.map((b, i) => b + c.accelRandomWalk * sq * n[9 + i]);
    this.k++;
    return sample;
  }
}
