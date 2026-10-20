// Small 3-D math helpers. Rotation matrices are row-major arrays of 9 numbers.
// Frames (as in the Python simulator): world ENU, rover body FLU, sensors FLU.
// Euler ZYX: R = Rz(yaw) Ry(pitch) Rx(roll); pitch > 0 = nose down, roll > 0 = left side up.

export const DEG = Math.PI / 180;

export function eulerZYX(roll, pitch, yaw) {
  const cr = Math.cos(roll), sr = Math.sin(roll), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  return [
    cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
    sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
    -sp, cp * sr, cp * cr,
  ];
}

export function matMul(A, B) {
  const C = new Array(9);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) C[i * 3 + j] = A[i * 3] * B[j] + A[i * 3 + 1] * B[3 + j] + A[i * 3 + 2] * B[6 + j];
  }
  return C;
}

export const matVec = (R, v) => [
  R[0] * v[0] + R[1] * v[1] + R[2] * v[2], R[3] * v[0] + R[4] * v[1] + R[5] * v[2], R[6] * v[0] + R[7] * v[1] + R[8] * v[2],
];
export const matTVec = (R, v) => [
  R[0] * v[0] + R[3] * v[1] + R[6] * v[2], R[1] * v[0] + R[4] * v[1] + R[7] * v[2], R[2] * v[0] + R[5] * v[1] + R[8] * v[2],
];
export const transpose = (R) => [R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8]];

/** Pose composition: (Ra, ta) ∘ (Rb, tb). */
export const compose = (a, b) => ({ R: matMul(a.R, b.R), t: matVec(a.R, b.t).map((c, k) => c + a.t[k]) });

/** T_body_mount from a mount {x, y, z, rollDeg, pitchDeg, yawDeg}. */
export const mountPose = (m) => ({ R: eulerZYX(m.rollDeg * DEG, m.pitchDeg * DEG, m.yawDeg * DEG), t: [m.x, m.y, m.z] });

/** Rotation vector (axis * angle) of a rotation matrix. */
export function rotvec(R) {
  const c = Math.min(1, Math.max(-1, (R[0] + R[4] + R[8] - 1) / 2)), ang = Math.acos(c);
  const w = [R[7] - R[5], R[2] - R[6], R[3] - R[1]];
  if (ang < 1e-8) return w.map((x) => 0.5 * x);
  if (Math.PI - ang < 1e-6) { // near pi: axis from the diagonal
    const ax = [Math.sqrt(Math.max((R[0] + 1) / 2, 0)), Math.sqrt(Math.max((R[4] + 1) / 2, 0)), Math.sqrt(Math.max((R[8] + 1) / 2, 0))];
    if (R[1] < 0) ax[1] = -ax[1];
    if (R[2] < 0) ax[2] = -ax[2];
    return ax.map((x) => x * ang);
  }
  const s = ang / (2 * Math.sin(ang));
  return w.map((x) => x * s);
}

/** Quaternion (w, x, y, z), w >= 0, of a rotation matrix. */
export function quat(R) {
  const tr = R[0] + R[4] + R[8];
  let q;
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    q = [0.25 * s, (R[7] - R[5]) / s, (R[2] - R[6]) / s, (R[3] - R[1]) / s];
  } else if (R[0] > R[4] && R[0] > R[8]) {
    const s = Math.sqrt(1 + R[0] - R[4] - R[8]) * 2;
    q = [(R[7] - R[5]) / s, 0.25 * s, (R[1] + R[3]) / s, (R[2] + R[6]) / s];
  } else if (R[4] > R[8]) {
    const s = Math.sqrt(1 + R[4] - R[0] - R[8]) * 2;
    q = [(R[2] - R[6]) / s, (R[1] + R[3]) / s, 0.25 * s, (R[5] + R[7]) / s];
  } else {
    const s = Math.sqrt(1 + R[8] - R[0] - R[4]) * 2;
    q = [(R[3] - R[1]) / s, (R[2] + R[6]) / s, (R[5] + R[7]) / s, 0.25 * s];
  }
  return q[0] < 0 ? q.map((x) => -x) : q;
}
