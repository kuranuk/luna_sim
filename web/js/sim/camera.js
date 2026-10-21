// Camera and spotlight models without any GPU code (port of lunarsim/core/sensors/camera.py,
// camera_noise.py and illumination/lights.py). The GPU renderer (../gl/camera_gl.js) produces
// the in-band radiance image; expose() turns it into digital numbers with shot / read noise.
// Pinhole only (no distortion). Optical frame: OpenCV (x right, y down, z forward).

import { rng } from './rng.js';

const PLANCK = 6.62607015e-34, C_LIGHT = 299792458;

/** fx, fy, cx, cy for intrinsics { width, height, hfovDeg, fx?, fy?, cx?, cy? } (pixel-centre origin). */
export function intrinsics(c) {
  const fx = c.fx ?? 0.5 * c.width / Math.tan(c.hfovDeg * Math.PI / 360);
  return { width: c.width, height: c.height, fx, fy: c.fy ?? fx, cx: c.cx ?? 0.5 * (c.width - 1), cy: c.cy ?? 0.5 * (c.height - 1) };
}

/** Electrons per second per unit in-band radiance [e- / (s W m^-2 sr^-1)]. */
export function electronsPerRadiance(s) {
  const etendue = Math.PI / (4 * s.fNumber ** 2) * s.pixelPitch ** 2;
  return etendue * s.transmission * s.quantumEfficiency * s.wavelength / (PLANCK * C_LIGHT);
}

export const gainOf = (s) => s.gain ?? s.fullWell / (2 ** s.bits - 1);

/** Exposure [s] putting the aePercentile of the noise-free signal at aeTarget x full well. */
export function autoExposure(radiance, s) {
  const k = electronsPerRadiance(s), sorted = Float32Array.from(radiance).sort();
  const idx = (s.aePercentile / 100) * (sorted.length - 1), lo = Math.floor(idx), hi = Math.min(lo + 1, sorted.length - 1);
  const ref = (sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo)) * k + s.darkCurrent; // numpy 'linear' percentile
  return ref <= 0 ? s.aeMaxExposure : Math.min(s.aeMaxExposure, s.aeTarget * s.fullWell / ref);
}

/**
 * Digitise a radiance image [W m^-2 sr^-1] (row-major, top row first).
 * Returns { dn: Uint16Array, exposure, saturated (count) }. Noise stream:
 * sensor/camera/{name}/frame/{index:06d} (same name as the Python simulator; values differ).
 */
export function expose(radiance, s, name, frameIndex, seed) {
  const t = s.autoExposure ? autoExposure(radiance, s) : s.exposure;
  const r = rng(seed, `sensor/camera/${name}/frame/${String(frameIndex).padStart(6, '0')}`);
  const k = electronsPerRadiance(s) * t, dark = s.darkCurrent * t, g = gainOf(s), maxDn = 2 ** s.bits - 1;
  const dn = new Uint16Array(radiance.length);
  let saturated = 0;
  for (let p = 0; p < radiance.length; p++) {
    let e = r.poisson(Math.max(radiance[p], 0) * k + dark);
    if (s.readNoise > 0) e += s.readNoise * r.normal();
    const satWell = e >= s.fullWell;
    const v = Math.round(Math.min(Math.max(e, 0), s.fullWell) / g) + s.blackLevel;
    if (satWell || v >= maxDn) saturated++;
    dn[p] = Math.min(Math.max(v, 0), maxDn);
  }
  return { dn, exposure: t, saturated };
}

/** Relative spot intensity vs off-axis angle [rad]: 1 inside inner, smoothstep to 0 at outer. */
export function spotProfile(theta, inner, outer) {
  const t = Math.min(Math.max((theta - inner) / (outer - inner), 0), 1);
  return 1 - t * t * (3 - 2 * t);
}

/** ∫ f dΩ [sr] (exact inner cone + trapezoid on the ramp, as lights.py). */
export function profileSolidAngle(inner, outer) {
  const n = 4001, h = (outer - inner) / (n - 1);
  let sum = 0;
  for (let k = 0; k < n; k++) {
    const th = inner + k * h, y = spotProfile(th, inner, outer) * Math.sin(th);
    sum += k === 0 || k === n - 1 ? 0.5 * y : y;
  }
  return 2 * Math.PI * (1 - Math.cos(inner) + h * sum);
}

/** On-axis in-band radiant intensity [W/sr] of a spotlight config. */
export function spotIntensity(l) {
  const inner = l.beamHalfAngleDeg * Math.PI / 180, outer = l.fieldHalfAngleDeg * Math.PI / 180;
  return { i0: l.fluxLm / l.efficacyLmPerW * l.bandFraction / profileSolidAngle(inner, outer), inner, outer };
}
