// LiDAR simulation (simplified port of lunarsim/core/sensors/{lidar,spad}.py).
//
// Patterns (sensor frame FLU): flash (W x H pixels over hfov x vfov) and spinning
// (channels evenly spaced in elevation, columns every hRes counter-clockwise from -180°).
// Flash projection (cfg.projection):
//   'azel' (default): azimuth / elevation grid whose azimuth axis stays the body's vertical axis; the
//      mount pitch is added to the elevation of every row and the mount yaw to the azimuth (roll is not
//      used). Each row is a cone of constant elevation about the body z axis: on flat ground every row is
//      an arc of constant range around the sensor and the side edges are radial lines;
//   'equiangular': pixel (r, c) at azimuth / elevation evenly spaced over hfov / vfov, i.e. each
//      row is a cone of constant elevation (on flat ground with zero pitch: an arc of constant range);
//   'pinhole': flat focal-plane array behind an ideal lens, tan(angle) evenly spaced (rows are planes
//      through the sensor: straight lines on flat ground);
//   'equidistant': f-theta lens, angle from the optical axis proportional to the image radius.
// Points are returned in the sensor frame at the scan pose (no motion distortion).
//
// "spad" model (photon counting, per pixel, `pulses` laser cycles accumulated per frame):
//   signal photons per pulse  n_s = E_pix ρ cosθ A_rx τ_rx / (π r²) · λ/(hc) · PDE,
//     E_pix = E_pulse τ_tx Ω_pix / Ω_fov (flood illumination)
//   background rate  Φ = ρ F_λ Δλ cos(i_sun) / π · A_rx Ω_pix τ_rx · λ/(hc) · PDE + DCR  (sunlit only)
//   multi-event detection with dead time: armed fraction 1 / (1 + Φ τ_d)
//   counts in the peak bin  S ~ Poisson(N (1 - e^{-n_s f_peak}) armed + B),  B = N Φ t_bin armed
//   threshold T: Poisson tail of B over the range gate ≤ P_fa; detected if S ≥ T;
//   range noise σ_r = (c/2) sqrt(σ_jit² + σ_pulse²) / sqrt(S); otherwise a false alarm with P_fa.
// "gaussian" model: p_det = 1 / (1 + (S/S50)^-γ), S = ρ cosθ / r², σ(r) = sqrt(σ0² + (k r)²).
// All device parameters are generic sample values (no specific sensor).

import { eulerZYX, matTVec, matVec } from './frames.js';
import { rng } from './rng.js';
import { LABEL, labelAt, normalAt } from './terrain.js';

const C = 299792458, H = 6.62607015e-34;
export const NOISE_LABEL = 254;

/** Unit beam directions in the sensor frame, Float64Array (N x 3), plus ring/time per beam. */
export function scanPattern(cfg) {
  const dirs = [], ring = [], time = [];
  if (cfg.kind === 'flash') {
    const proj = cfg.projection || 'azel';
    let Rm = null; // azel: body directions are mapped back into the sensor frame with R_mount^T
    if (proj === 'azel') {
      const m = cfg.mount || {};
      Rm = eulerZYX((m.rollDeg || 0) * Math.PI / 180, (m.pitchDeg || 0) * Math.PI / 180, (m.yawDeg || 0) * Math.PI / 180);
    }
    const ha = cfg.hfovDeg * Math.PI / 360, va = cfg.vfovDeg * Math.PI / 360; // half angles [rad]
    const th = Math.tan(ha), tv = Math.tan(va);
    for (let r = 0; r < cfg.height; r++) {
      for (let c = 0; c < cfg.width; c++) {
        const sx = 1 - (c + 0.5) / cfg.width * 2, sy = 1 - (r + 0.5) / cfg.height * 2; // +1 = left / top edge
        if (proj === 'pinhole') {
          const y = sx * th, z = sy * tv, nn = Math.hypot(1, y, z);
          dirs.push(1 / nn, y / nn, z / nn);
        } else if (proj === 'equidistant') {
          const ax = sx * ha, ay = sy * va, theta = Math.hypot(ax, ay), s = theta > 0 ? Math.sin(theta) / theta : 1;
          dirs.push(Math.cos(theta), ax * s, ay * s);
        } else if (proj === 'azel') {
          const m = cfg.mount || {}, az = sx * ha + (m.yawDeg || 0) * Math.PI / 180, el = sy * va - (m.pitchDeg || 0) * Math.PI / 180;
          dirs.push(...matTVec(Rm, [Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el)]));
        } else { // equiangular
          const az = sx * ha, el = sy * va;
          dirs.push(Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el));
        }
        ring.push(r); time.push(0);
      }
    }
  } else {
    const cols = Math.round(360 / cfg.hResDeg), period = 1 / cfg.rateHz;
    for (let c = 0; c < cols; c++) {
      const az = (-180 + c * cfg.hResDeg) * Math.PI / 180;
      for (let ch = 0; ch < cfg.channels; ch++) {
        const el = (cfg.vMinDeg + (cfg.vMaxDeg - cfg.vMinDeg) * (cfg.channels > 1 ? ch / (cfg.channels - 1) : 0.5)) * Math.PI / 180;
        dirs.push(Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el)); ring.push(ch); time.push(c / cols * period);
      }
    }
  }
  return { dirs: Float64Array.from(dirs), ring: Uint16Array.from(ring), time: Float32Array.from(time), n: ring.length };
}

/** Solid angle [sr] of the flash field (pinhole: rectangular pyramid; others: azimuth x elevation patch). */
export function flashSolidAngle(cfg) {
  const ha = cfg.hfovDeg * Math.PI / 360, va = cfg.vfovDeg * Math.PI / 360;
  const proj = cfg.projection || 'azel';
  if (proj === 'pinhole') return 4 * Math.asin(Math.sin(ha) * Math.sin(va));
  if (proj === 'azel') { // elevation band shifted by the pitch
    const p = ((cfg.mount && cfg.mount.pitchDeg) || 0) * Math.PI / 180;
    return 2 * ha * (Math.sin(Math.min(va - p, Math.PI / 2)) - Math.sin(Math.max(-va - p, -Math.PI / 2)));
  }
  return 4 * ha * Math.sin(va);
}

function poissonThreshold(lam, tail) { // smallest k with P(X >= k) <= tail, X ~ Poisson(lam)
  if (lam > 60) return Math.ceil(lam + 5.5 * Math.sqrt(lam) + 1);
  let p = Math.exp(-lam), cdf = p, k = 0;
  while (1 - cdf > tail && k < 1000) { k++; p *= lam / k; cdf += p; }
  return k + 1;
}

/**
 * Simulate one frame. pose = T_world_sensor {R, t}; sun = { dir, el } or null.
 * Returns { points: Float32Array N*4 (x, y, z, intensity; sensor frame), world: Float32Array N*3,
 * labels: Uint8Array, ring, time, beam (Uint32Array: beam / pixel index, flash: row * width + col), count, beams }.
 */
export function renderLidar(cfg, pattern, pose, hm, caster, sun, frameIndex, seed) {
  const r = rng(seed, `sensor/lidar/${cfg.name}/frame/${String(frameIndex).padStart(6, '0')}`);
  const nb = pattern.n, dirs = pattern.dirs, o = pose.t;
  const pts = new Float32Array(nb * 4), wpts = new Float32Array(nb * 3), lab = new Uint8Array(nb);
  const ring = new Uint16Array(nb), time = new Float32Array(nb), beam = new Uint32Array(nb);
  let m = 0;
  const spad = cfg.model === 'spad' ? cfg.spad : null;
  let K = 0, bgK = 0, nBins = 0, peakFrac = 0, sigmaT = 0;
  if (spad) {
    const omegaFov = flashSolidAngle(cfg);
    const omegaPix = omegaFov / nb, aRx = Math.PI * (spad.rxAperture / 2) ** 2, phJ = spad.wavelength / (H * C);
    K = spad.pulseEnergy * spad.txTransmission * (omegaPix / omegaFov) * aRx * spad.rxTransmission / Math.PI * phJ * spad.pde;
    bgK = spad.solarSpectral * spad.filterNm / Math.PI * aRx * omegaPix * spad.rxTransmission * phJ * spad.pde;
    nBins = Math.ceil(2 * cfg.rangeMax / C / spad.binWidth);
    const sp = spad.pulseFwhm / 2.3548;
    peakFrac = Math.min(1, spad.binWidth / (sp * Math.sqrt(2 * Math.PI)));
    sigmaT = Math.sqrt(spad.jitter ** 2 + sp ** 2);
  }
  const sunOk = sun && sun.el > 0;
  for (let b = 0; b < nb; b++) {
    const ds = [dirs[3 * b], dirs[3 * b + 1], dirs[3 * b + 2]], dw = matVec(pose.R, ds);
    const t = caster.trace(o[0], o[1], o[2], dw[0], dw[1], dw[2], 0, cfg.rangeMax * 1.05);
    const u1 = r.random(), u2 = r.random(), g1 = r.normal(); // fixed draws per beam
    let range = NaN, inten = 0, label = 0, hit = t >= cfg.rangeMin && t <= cfg.rangeMax;
    let rho = cfg.reflectance, cosI = 0, nrm = null, px = 0, py = 0, pz = 0;
    if (hit) {
      px = o[0] + dw[0] * t; py = o[1] + dw[1] * t; pz = o[2] + dw[2] * t;
      nrm = normalAt(hm, px, py);
      cosI = Math.max(0, -(nrm[0] * dw[0] + nrm[1] * dw[1] + nrm[2] * dw[2]));
      label = labelAt(hm, px, py) === LABEL.ROCK ? LABEL.ROCK : 0;
    }
    if (spad) {
      let phi = spad.dcr;
      if (hit && sunOk) {
        const mu0 = nrm[0] * sun.dir[0] + nrm[1] * sun.dir[1] + nrm[2] * sun.dir[2];
        if (mu0 > 0 && !caster.occluded(px + 2e-3 * nrm[0], py + 2e-3 * nrm[1], pz + 2e-3 * nrm[2], sun.dir[0], sun.dir[1], sun.dir[2])) {
          phi += rho * mu0 * bgK / (sun.distanceAu || 1) ** 2;
        }
      }
      const armed = 1 / (1 + phi * spad.deadTime), B = spad.pulses * phi * spad.binWidth * armed;
      const T = poissonThreshold(B, spad.pfa / nBins);
      const ns = hit ? K * rho * cosI / (t * t) : 0;
      const S = r.poisson(spad.pulses * (1 - Math.exp(-ns * peakFrac)) * armed + B);
      if (hit && S >= T) {
        range = t + 0.5 * C * sigmaT / Math.sqrt(Math.max(S, 1)) * g1;
        inten = Math.min(1, S / spad.pulses);
      } else if (u1 < spad.pfa) { // false alarm: a random bin of the gate
        range = Math.max(cfg.rangeMin, u2 * cfg.rangeMax); inten = 0; label = NOISE_LABEL;
      }
    } else if (hit) {
      const S = rho * cosI / (t * t), S50 = cfg.referenceReflectance / cfg.rangeMax ** 2;
      const p = S > 0 ? 1 / (1 + (S / S50) ** -cfg.steepness) : 0;
      if (u1 < p) { range = t + Math.hypot(cfg.sigma0, cfg.sigmaPerM * t) * g1; inten = Math.min(1, rho * cosI); }
    }
    if (!Number.isNaN(range)) {
      pts[4 * m] = ds[0] * range; pts[4 * m + 1] = ds[1] * range; pts[4 * m + 2] = ds[2] * range; pts[4 * m + 3] = inten;
      wpts[3 * m] = o[0] + dw[0] * range; wpts[3 * m + 1] = o[1] + dw[1] * range; wpts[3 * m + 2] = o[2] + dw[2] * range;
      lab[m] = label; ring[m] = pattern.ring[b]; time[m] = pattern.time[b]; beam[m] = b;
      m++;
    }
  }
  return { points: pts.slice(0, 4 * m), world: wpts.slice(0, 3 * m), labels: lab.slice(0, m),
    ring: ring.slice(0, m), time: time.slice(0, m), beam: beam.slice(0, m), count: m, beams: nb };
}
