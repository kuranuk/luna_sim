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
import { OBJECT_LABEL } from './objects.js';
import { facetNormalAt, LABEL, labelAt, normalAt } from './terrain.js';

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

// ---- SPAD statistics (as lunarsim/core/sensors/spad.py)

const Z_CACHE = new Map();
function normalQuantileUpper(q) { // z with P(Z > z) = q (Acklam's rational approximation, |err| < 1.2e-9 rel.)
  if (Z_CACHE.has(q)) return Z_CACHE.get(q);
  const p = 1 - q, a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
  const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
  const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
  const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
  let z;
  if (p > 1 - 0.02425) { const t = Math.sqrt(-2 * Math.log(1 - p)); z = -(((((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4]) * t + c[5]) / ((((d[0] * t + d[1]) * t + d[2]) * t + d[3]) * t + 1); }
  else if (p < 0.02425) { const t = Math.sqrt(-2 * Math.log(p)); z = (((((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4]) * t + c[5]) / ((((d[0] * t + d[1]) * t + d[2]) * t + d[3]) * t + 1); }
  else { const t = p - 0.5, r = t * t; z = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * t / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1); }
  Z_CACHE.set(q, z);
  return z;
}

/**
 * Detection threshold: smallest T with P(X >= T) <= q for X ~ Poisson(lam) (scipy: poisson.isf(q) + 1).
 * Exact summation for lam <= 400, Cornish-Fisher quantile beyond. Returns [T, P(X >= T)].
 */
export function poissonThreshold(lam, q) {
  if (lam <= 400) {
    let pmf = Math.exp(-lam), sf = 1 - pmf, k = 0; // sf = P(X > k)
    while (sf > q && k < 5000) { k++; pmf *= lam / k; sf -= pmf; }
    return [k + 1, Math.max(sf, 0)];
  }
  const z = normalQuantileUpper(q);
  return [Math.floor(lam + z * Math.sqrt(lam) + (z * z - 1) / 6) + 2, q];
}

function binomial(r, n, p) { // moderate accuracy is enough: counts feed a threshold test
  if (p <= 0) return 0;
  if (p >= 1) return n;
  const v = n * p * (1 - p);
  if (v > 9) return Math.min(n, Math.max(0, Math.round(n * p + Math.sqrt(v) * r.normal())));
  if (p < 0.1) return Math.min(n, r.poisson(n * p));
  if (p > 0.9) return Math.max(0, n - r.poisson(n * (1 - p)));
  let k = 0;
  for (let i = 0; i < n; i++) if (r.random() < p) k++;
  return k;
}

// First-photon arrival of a Gaussian pulse with n mean photons per cycle: mean shift and std in units
// of the pulse sigma (range walk towards the sensor as n grows). Table over log10 n in [-3, 3].
let FP_TABLE = null;
function firstPhotonMoments(n) {
  if (!FP_TABLE) {
    const M = 121, tab = new Float64Array(2 * M), Z = 281;
    for (let m = 0; m < M; m++) {
      const nn = 10 ** (-3 + 6 * m / (M - 1));
      let w0 = 0, w1 = 0, w2 = 0;
      for (let k = 0; k < Z; k++) {
        const z = -7 + 14 * k / (Z - 1), g = Math.exp(-0.5 * z * z), G = 0.5 * (1 + erf(z / Math.SQRT2)), f = g * Math.exp(-nn * G);
        w0 += f; w1 += f * z; w2 += f * z * z;
      }
      const mean = w1 / w0;
      tab[2 * m] = mean; tab[2 * m + 1] = Math.sqrt(Math.max(w2 / w0 - mean * mean, 0));
    }
    FP_TABLE = { tab, M };
  }
  if (!(n > 1e-3)) return [-n / (2 * Math.sqrt(Math.PI)), 1]; // first order in n
  const { tab, M } = FP_TABLE, u = Math.min((Math.log10(n) + 3) / 6 * (M - 1), M - 1), i = Math.min(Math.floor(u), M - 2), f = u - i;
  return [tab[2 * i] + f * (tab[2 * i + 2] - tab[2 * i]), tab[2 * i + 1] + f * (tab[2 * i + 3] - tab[2 * i + 1])];
}
function erf(x) { // Abramowitz & Stegun 7.1.26 (|err| < 1.5e-7)
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return x >= 0 ? y : -y;
}

/** Per-frame SPAD constants of a flash LiDAR config. */
export function spadSetup(cfg, nb) {
  const sp = cfg.spad, omegaFov = flashSolidAngle(cfg), omegaPix = omegaFov / nb;
  const aRx = Math.PI * (sp.rxAperture / 2) ** 2, phJ = sp.wavelength / (H * C);
  const wBins = Math.max(1, Math.ceil(2 * sp.pulseFwhm / sp.binWidth)), wT = wBins * sp.binWidth;
  return {
    K: sp.pulseEnergy * sp.txTransmission * (omegaPix / omegaFov) * aRx * sp.rxTransmission / Math.PI * phJ * sp.pde, // n_s = K rho cos / r²
    bgK: sp.solarSpectral * sp.filterNm / Math.PI * aRx * omegaPix * sp.rxTransmission * phJ * sp.pde, // phi_sun = bgK rho mu0
    wT, nWindows: Math.max(1, Math.ceil(2 * cfg.rangeMax / C / wT)),
    sigmaG: Math.sqrt((sp.pulseFwhm / 2.3548200450309493) ** 2 + sp.jitter ** 2),
  };
}

/** Detection statistics of one pixel: armed fraction, background per window, threshold, signal probability. */
export function spadPixel(cfg, su, ns, phi) {
  const sp = cfg.spad, armed = 1 / (1 + phi * sp.deadTime);
  const muB = sp.pulses * (1 - Math.exp(-phi * armed * su.wT));
  const [T, pTail] = poissonThreshold(muB, sp.pfa / su.nWindows);
  return { armed, muB, T, pFalse: Math.min(1, su.nWindows * pTail), pS: armed * (1 - Math.exp(-ns)) };
}

/**
 * Simulate one frame. pose = T_world_sensor {R, t}; sun = { dir, el } or null.
 * Returns { points: Float32Array N*4 (x, y, z, intensity; sensor frame), world: Float32Array N*3,
 * labels: Uint8Array, ring, time, beam (Uint32Array: beam / pixel index, flash: row * width + col), count, beams,
 * range: Float32Array per beam (measured range [m], NaN = no return; flash: the range image, row-major),
 * rangeTrue: Float32Array per beam (geometric range, NaN = no surface within rangeMax; ground truth) }.
 *
 * "spad" (photon counting, as lunarsim/core/sensors/spad.py, multi-event mode): signal photons per cycle
 * n_s = E_pix rho cos(theta) A_rx tau_rx / (pi r²) lambda/(hc) PDE; background rate Phi = sunlit surface in
 * the filter band + DCR; armed fraction 1 / (1 + Phi tau_d); counts in a search window of ~2 pulse FWHM:
 * Binomial(N, armed (1 - e^-n_s)) + Poisson(N (1 - e^(-Phi armed w))); detected if >= T, T from the Poisson
 * tail of the background over all windows of the gate (P_fa per pixel and frame); range = window centroid
 * (first-photon range walk, TDC quantisation, background photons uniform in the window); otherwise a
 * false alarm with the probability that background alone crosses T, at a uniform range (label 254).
 */
export function renderLidar(cfg, pattern, pose, hm, caster, sun, frameIndex, seed) {
  const r = rng(seed, `sensor/lidar/${cfg.name}/frame/${String(frameIndex).padStart(6, '0')}`);
  const nb = pattern.n, dirs = pattern.dirs, o = pose.t;
  const pts = new Float32Array(nb * 4), wpts = new Float32Array(nb * 3), lab = new Uint8Array(nb);
  const ring = new Uint16Array(nb), time = new Float32Array(nb), beam = new Uint32Array(nb);
  const rangeImg = new Float32Array(nb).fill(NaN), rangeTrue = new Float32Array(nb).fill(NaN);
  let m = 0;
  const spad = cfg.model === 'spad' ? cfg.spad : null, su = spad ? spadSetup(cfg, nb) : null;
  const sunOk = sun && sun.el > 0;
  for (let b = 0; b < nb; b++) {
    const ds = [dirs[3 * b], dirs[3 * b + 1], dirs[3 * b + 2]], dw = matVec(pose.R, ds);
    const hitInfo = caster.traceHit ? caster.traceHit(o[0], o[1], o[2], dw[0], dw[1], dw[2], 0, cfg.rangeMax * 1.05) : null;
    const t = hitInfo ? hitInfo.t : caster.trace(o[0], o[1], o[2], dw[0], dw[1], dw[2], 0, cfg.rangeMax * 1.05);
    const u1 = r.random(), u2 = r.random(), g1 = r.normal();
    let range = NaN, inten = 0, label = 0;
    const hit = t >= cfg.rangeMin && t <= cfg.rangeMax;
    let rho = cfg.reflectance, cosI = 0, nrm = null, px = 0, py = 0, pz = 0;
    if (hit) {
      rangeTrue[b] = t;
      px = o[0] + dw[0] * t; py = o[1] + dw[1] * t; pz = o[2] + dw[2] * t;
      const obj = hitInfo && hitInfo.object;
      nrm = obj ? hitInfo.n : normalAt(hm, px, py);
      cosI = Math.max(0, -(nrm[0] * dw[0] + nrm[1] * dw[1] + nrm[2] * dw[2]));
      if (!obj && cosI < 0.05) { // silhouettes: the smooth shading normal can face away from a ray that
        const f = facetNormalAt(hm, px, py); // does hit the surface -> use the facet actually hit
        cosI = Math.max(cosI, Math.abs(f[0] * dw[0] + f[1] * dw[1] + f[2] * dw[2]));
      }
      label = obj ? OBJECT_LABEL : labelAt(hm, px, py) === LABEL.ROCK ? LABEL.ROCK : 0;
    }
    if (spad) {
      let phi = spad.dcr;
      if (hit && sunOk) {
        const mu0 = nrm[0] * sun.dir[0] + nrm[1] * sun.dir[1] + nrm[2] * sun.dir[2];
        if (mu0 > 0 && !caster.occluded(px + 2e-3 * nrm[0], py + 2e-3 * nrm[1], pz + 2e-3 * nrm[2], sun.dir[0], sun.dir[1], sun.dir[2])) {
          phi += rho * mu0 * su.bgK / (sun.distanceAu || 1) ** 2;
        }
      }
      const ns = hit ? su.K * rho * cosI / (t * t) : 0, st = spadPixel(cfg, su, ns, phi);
      const uSig = hit ? binomial(r, spad.pulses, st.pS) : 0, uBg = r.poisson(st.muB), k = uSig + uBg;
      if (hit && k >= st.T) {
        const [shift, sd] = firstPhotonMoments(ns);
        const varSig = (sd * su.sigmaG) ** 2 + spad.binWidth ** 2 / 12;
        const varT = (uSig * varSig + uBg * su.wT ** 2 / 12) / Math.max(k, 1) ** 2;
        // walkCorrection (default): the sensor subtracts the calibrated range walk for the measured intensity
        const walk = spad.walkCorrection === false ? shift * su.sigmaG : 0;
        range = t + 0.5 * C * (walk + Math.sqrt(varT) * g1);
        inten = Math.min(1, uSig / spad.pulses);
      } else if (u1 < st.pFalse) { // false alarm: background alone crossed the threshold somewhere in the gate
        range = cfg.rangeMin + u2 * (cfg.rangeMax - cfg.rangeMin); inten = 0; label = NOISE_LABEL;
      }
    } else if (hit) {
      const S = rho * cosI / (t * t), S50 = cfg.referenceReflectance / cfg.rangeMax ** 2;
      const p = S > 0 ? 1 / (1 + (S / S50) ** -cfg.steepness) : 0;
      if (u1 < p) { range = t + Math.hypot(cfg.sigma0, cfg.sigmaPerM * t) * g1; inten = Math.min(1, rho * cosI); }
    }
    if (!Number.isNaN(range)) {
      rangeImg[b] = range;
      pts[4 * m] = ds[0] * range; pts[4 * m + 1] = ds[1] * range; pts[4 * m + 2] = ds[2] * range; pts[4 * m + 3] = inten;
      wpts[3 * m] = o[0] + dw[0] * range; wpts[3 * m + 1] = o[1] + dw[1] * range; wpts[3 * m + 2] = o[2] + dw[2] * range;
      lab[m] = label; ring[m] = pattern.ring[b]; time[m] = pattern.time[b]; beam[m] = b;
      m++;
    }
  }
  return { points: pts.slice(0, 4 * m), world: wpts.slice(0, 3 * m), labels: lab.slice(0, m),
    ring: ring.slice(0, m), time: time.slice(0, m), beam: beam.slice(0, m), count: m, beams: nb, range: rangeImg, rangeTrue };
}
