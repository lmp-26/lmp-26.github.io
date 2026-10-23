// Franka Panda kinematics (modified DH from Franka's docs) and a damped least-squares IK.
// Matrices are row-major arrays of 16.

const DH = [
  // a, d, alpha
  [0, 0.333, 0],
  [0, 0, -Math.PI / 2],
  [0, 0.316, Math.PI / 2],
  [0.0825, 0, Math.PI / 2],
  [-0.0825, 0.384, -Math.PI / 2],
  [0, 0, Math.PI / 2],
  [0.088, 0, Math.PI / 2],
];
const FLANGE = 0.107;
// < 1 lets the gripper tilt slightly instead of missing far targets
const W_ORIENT = 0.3;
const TCP = 0.1034;
const LOWER = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973];
const UPPER = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973];
export const Q_READY = [0, -Math.PI / 4, 0, -3 * Math.PI / 4, 0, Math.PI / 2, Math.PI / 4];

function mul(A, B) {
  const C = new Array(16);
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      C[r * 4 + c] = A[r * 4] * B[c] + A[r * 4 + 1] * B[4 + c] + A[r * 4 + 2] * B[8 + c] + A[r * 4 + 3] * B[12 + c];
    }
  }
  return C;
}
const rotX = (t) => [1, 0, 0, 0, 0, Math.cos(t), -Math.sin(t), 0, 0, Math.sin(t), Math.cos(t), 0, 0, 0, 0, 1];
const rotZ = (t) => [Math.cos(t), -Math.sin(t), 0, 0, Math.sin(t), Math.cos(t), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const trans = (x, y, z) => [1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, z, 0, 0, 0, 1];
const pos = (T) => [T[3], T[7], T[11]];
const zAxis = (T) => [T[2], T[6], T[10]];

// Joint frames, the points where each joint axis starts, and the flange/hand/TCP frames.
export function fk(q) {
  let T = trans(0, 0, 0);
  const frames = [];
  const axisPts = [];
  for (let i = 0; i < 7; i++) {
    const [a, d, al] = DH[i];
    T = mul(T, mul(rotX(al), trans(a, 0, 0)));
    axisPts.push(pos(T));
    T = mul(T, mul(rotZ(q[i]), trans(0, 0, d)));
    frames.push(T);
  }
  const flange = mul(T, trans(0, 0, FLANGE));
  const hand = mul(flange, rotZ(-Math.PI / 4));
  const tcp = mul(hand, trans(0, 0, TCP));
  return { frames, axisPts, flange, hand, tcp };
}

const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

function solve6(A, b) {
  const n = 6;
  const M = A.map((row, i) => row.concat([b[i]]));
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
    x[r] = s / M[r][r];
  }
  return x;
}

// Top-down TCP orientation of the ready pose.
const R_DOWN = (() => {
  const T = fk(Q_READY).tcp;
  return [T[0], T[1], T[2], T[4], T[5], T[6], T[8], T[9], T[10]];
})();

function ikStep(q, target, lambda) {
  const { frames, tcp } = fk(q);
  const p = pos(tcp);
  const e = [target[0] - p[0], target[1] - p[1], target[2] - p[2], 0, 0, 0];
  // orientation error: 0.5 * sum(current column x desired column)
  for (let k = 0; k < 3; k++) {
    const cur = [tcp[k], tcp[4 + k], tcp[8 + k]];
    const des = [R_DOWN[k], R_DOWN[3 + k], R_DOWN[6 + k]];
    const c = cross(cur, des);
    e[3] += 0.5 * W_ORIENT * c[0];
    e[4] += 0.5 * W_ORIENT * c[1];
    e[5] += 0.5 * W_ORIENT * c[2];
  }
  const J = [[], [], [], [], [], []];
  for (let i = 0; i < 7; i++) {
    const z = zAxis(frames[i]);
    const o = pos(frames[i]);
    const v = cross(z, [p[0] - o[0], p[1] - o[1], p[2] - o[2]]);
    for (let r = 0; r < 3; r++) {
      J[r][i] = v[r];
      J[3 + r][i] = W_ORIENT * z[r];
    }
  }
  const JJt = J.map((ri, r) => J.map((rj, c) => {
    let s = r === c ? lambda * lambda : 0;
    for (let i = 0; i < 7; i++) s += ri[i] * rj[i];
    return s;
  }));
  const Jt = (y) => Array.from({ length: 7 }, (_, i) => y.reduce((s, v, r) => s + J[r][i] * v, 0));
  const dq = Jt(solve6(JJt, e));
  // pull toward the ready pose inside the null space
  const zq = q.map((v, i) => 0.1 * (Q_READY[i] - v));
  const Jz = J.map((row) => row.reduce((s, v, i) => s + v * zq[i], 0));
  const back = Jt(solve6(JJt, Jz));
  const out = q.map((v, i) => Math.min(UPPER[i], Math.max(LOWER[i], v + dq[i] + zq[i] - back[i])));
  return { q: out, err: Math.hypot(e[0], e[1], e[2]) + Math.hypot(e[3], e[4], e[5]) };
}

export function ik(target, q0, iters) {
  let q = q0.slice();
  for (let k = 0; k < iters; k++) {
    const r = ikStep(q, target, 0.05);
    q = r.q;
    if (r.err < 2e-4) break;
  }
  return q;
}

// Joint angles for each point, warm-started along the path.
export function ikPath(points) {
  const qs = [];
  let q = ik(points[0], Q_READY, 300);
  qs.push(q);
  for (let i = 1; i < points.length; i++) {
    q = ik(points[i], q, 40);
    qs.push(q);
  }
  return qs;
}
