// Scenario presets (browser counterparts of configs/*.yaml): monocular navcam + two headlights,
// SPAD flash + spinning LiDAR, IMU.
// Physical defaults and their sources: lunarsim/config/schema.py, docs/references.md.

const CRATER_SHAPE = {
  depthFresh: 0.15, // d/D fresh (Stopar et al. 2017; secondary citation)
  depthDegraded: 0.05, // d/D degraded (Basilevsky et al. 2018; secondary citation)
  rimFresh: 0.036, // rim height / D (Pike 1977)
  ejectaExponent: 3, // ejecta ∝ (r/R)^-3 (McGetchin et al. 1973)
  ejectaExtent: 4, // [crater radii]
  smoothingDegraded: 0.3, // Gaussian sigma at g = 1 [radii], sample value
};

const ROVER = {
  wheelbase: 1.0, track: 0.8, wheelRadius: 0.15, wheelWidth: 0.2, contactHalfLengthRatio: 0.4,
  originHeight: 0.5, cgHeight: 0.45, maxSpeed: 0.2, maxYawRateDeg: 20, maxSlopeDeg: 20,
  tipoverMarginDeg: 5, slipFlat: 0.02, criticalSlopeDeg: 25, slipExponent: 2, downhillSkid: 0.05,
  lateralCoefficient: 0.3, x: 0, y: 0, yawDeg: 0, // generic sample values (no real vehicle)
};

const LIDARS = [
  { // SPAD flash LiDAR (primary). Generic sample parameters, see LidarSpadConfig.
    name: 'spad_flash', kind: 'flash', model: 'spad', rateHz: 2, projection: 'equiangular', // or 'pinhole' / 'equidistant'
    width: 160, height: 120, hfovDeg: 60, vfovDeg: 45,
    // Mast-mounted: 3.7 m above flat ground (body origin is originHeight = 0.5 m up), 22° down
    // (user setting 2026-10). Mount z is in the body frame (FLU); pitch > 0 = looking down.
    mount: { x: 0.55, y: 0, z: 3.2, rollDeg: 0, pitchDeg: 22, yawDeg: 0 },
    rangeMin: 0.3, rangeMax: 100, reflectance: 0.3,
    spad: { wavelength: 905e-9, pulseEnergy: 1e-6, txTransmission: 0.9, rxAperture: 0.01, rxTransmission: 0.8,
      pde: 0.2, filterNm: 1.0, solarSpectral: 1.0, dcr: 1000, pulseFwhm: 1e-9, jitter: 1e-10, binWidth: 2.5e-10,
      pulses: 1000, pfa: 1e-3, deadTime: 10e-9 },
  },
  { // Rotating 32-beam LiDAR, simple Gaussian model (for comparison).
    name: 'spin32', kind: 'spinning', model: 'gaussian', rateHz: 10,
    channels: 32, vMinDeg: -25, vMaxDeg: 15, hResDeg: 0.2,
    mount: { x: 0.5, y: 0, z: 0.8, rollDeg: 0, pitchDeg: 0, yawDeg: 0 },
    rangeMin: 0.3, rangeMax: 100, reflectance: 0.3, referenceReflectance: 0.1, steepness: 4,
    sigma0: 0.02, sigmaPerM: 1e-4,
  },
];

const IMU = { // MEMS-class sample values (Kalibr/EuRoC convention), see ImuConfig
  name: 'imu0', rateHz: 100, mount: { x: 0, y: 0, z: 0.3, rollDeg: 0, pitchDeg: 0, yawDeg: 0 },
  gyroNoise: 1.7e-4, gyroRandomWalk: 2e-5, gyroBias: 1e-3, accelNoise: 2e-3, accelRandomWalk: 3e-3, accelBias: 2e-2,
  bandwidthHz: 5, filterOrder: 2, gravity: 1.6243, moonRotation: 2.6617e-6,
};

const CAMERA_SENSOR = { // CameraSensorConfig defaults (sample values; band_fraction TODO(ref) as in schema.py)
  autoExposure: true, exposure: 5e-4, aePercentile: 99.5, aeTarget: 0.8, aeMaxExposure: 1.0,
  fNumber: 4, pixelPitch: 5.5e-6, transmission: 0.9, quantumEfficiency: 0.5, wavelength: 600e-9,
  bandFraction: 0.35, fullWell: 20000, readNoise: 5, darkCurrent: 10, bits: 12, gain: null, blackLevel: 0,
};
const camera = (name, rateHz, hfovDeg, mount) => ({
  name, enabled: true, rateHz, width: 640, height: 480, hfovDeg,
  mount: { x: 0, y: 0, z: 0, rollDeg: 0, pitchDeg: 0, yawDeg: 0, ...mount }, sensor: { ...CAMERA_SENSOR },
});
// Generic sample geometry (no real vehicle). Mount yaw: 0 = forward, +90 = left, 180 = backward.
const CAMERAS = [
  camera('navcam', 2, 70, { x: 0.5, z: 1.0, pitchDeg: 12 }), // as configs/south_pole_lowsun.yaml
  camera('side_left', 1, 90, { y: 0.45, z: 1.0, pitchDeg: 20, yawDeg: 90 }),
  camera('side_right', 1, 90, { y: -0.45, z: 1.0, pitchDeg: 20, yawDeg: -90 }),
  camera('rear', 1, 90, { x: -0.55, z: 1.0, pitchDeg: 15, yawDeg: 180 }),
];

const LIGHT = { // SpotlightConfig defaults (generic sample values)
  enabled: true, fluxLm: 3000, efficacyLmPerW: 300, bandFraction: 0.8, beamHalfAngleDeg: 15, fieldHalfAngleDeg: 30,
};
const LIGHTS = [
  { ...LIGHT, name: 'headlight_left', mount: { x: 0.6, y: 0.3, z: 0.4, rollDeg: 0, pitchDeg: 15, yawDeg: 0 } },
  { ...LIGHT, name: 'headlight_right', mount: { x: 0.6, y: -0.3, z: 0.4, rollDeg: 0, pitchDeg: 15, yawDeg: 0 } },
];

/** Defaults for sensors added by a config file (see config.js). */
export const TEMPLATES = {
  camera: CAMERAS[0], light: LIGHTS[0],
  lidar: { flash: LIDARS[0], spinning: LIDARS[1] },
};

export const HAPKE = { w: 0.38, b: 0.25, c: 0.5, bs0: 1.5, hs: 0.05, bc0: 0.0, hc: 0.05, thetaBarDeg: 23.657 }; // HapkeConfig

const base = (name, seed, terrain, sun, site) => ({
  name, seed, site, sun, terrain,
  rover: { ...ROVER }, lidars: structuredClone(LIDARS), imu: { ...IMU, mount: { ...IMU.mount } },
  cameras: structuredClone(CAMERAS), lights: structuredClone(LIGHTS), reflectance: { ...HAPKE },
});

export const SCENARIOS = {
  south_pole_lowsun: base('south_pole_lowsun', 2026, {
    n: 1024, res: 0.2,
    fractal: { enabled: true, hurst: 0.85, rmsSlopeDeg: 6, baseline: 1, rolloffWavelength: 400 },
    tilt: { slopeDeg: 3, downhillAzimuthDeg: 50 },
    craters: { enabled: true, equilibriumFraction: 0.02, dMin: 0.5, dMax: 60, degradationA: 1, degradationB: 1, shape: CRATER_SHAPE },
    rocks: { enabled: true, k: 0.05, dMin: 0.4, dMax: 4, heightRatio: 0.5, axisRatioMin: 0.6, ejecta: true, ejectaK: 0.15, ejectaMinCraterD: 20 },
  }, { mode: 'astronomical', utc: '2026-12-20T00:00:00Z', azDeg: 49, elDeg: 1.6 }, { lat: -89.5, lon: 0 }),
  mare_plain: base('mare_plain', 42, {
    n: 1024, res: 0.2,
    fractal: { enabled: true, hurst: 0.9, rmsSlopeDeg: 4, baseline: 1, rolloffWavelength: 200 },
    tilt: { slopeDeg: 0, downhillAzimuthDeg: 0 },
    craters: { enabled: true, equilibriumFraction: 0.02, dMin: 0.5, dMax: 60, degradationA: 1, degradationB: 1, shape: CRATER_SHAPE },
    rocks: { enabled: true, k: 0.02, dMin: 0.4, dMax: 3, heightRatio: 0.5, axisRatioMin: 0.6, ejecta: false, ejectaK: 0.15, ejectaMinCraterD: 30 },
  }, { mode: 'manual', utc: '2026-01-01T00:00:00Z', azDeg: 135, elDeg: 30 }, { lat: 0, lon: 0 }),
  crater_field: base('crater_field', 11, {
    n: 1024, res: 0.2,
    fractal: { enabled: true, hurst: 0.9, rmsSlopeDeg: 3, baseline: 1, rolloffWavelength: 200 },
    tilt: { slopeDeg: 0, downhillAzimuthDeg: 0 },
    craters: { enabled: true, equilibriumFraction: 0.05, dMin: 0.5, dMax: 80, degradationA: 1, degradationB: 1, shape: CRATER_SHAPE },
    rocks: { enabled: true, k: 0.02, dMin: 0.4, dMax: 5, heightRatio: 0.5, axisRatioMin: 0.6, ejecta: true, ejectaK: 0.15, ejectaMinCraterD: 15 },
  }, { mode: 'manual', utc: '2026-01-01T00:00:00Z', azDeg: 100, elDeg: 12 }, { lat: 0, lon: 0 }),
};

export const SIZES = { // grid nodes (power of two) x resolution
  '128 m / 0.25 m': { n: 512, res: 0.25 },
  '205 m / 0.2 m': { n: 1024, res: 0.2 },
  '512 m / 0.5 m': { n: 1024, res: 0.5 },
};
