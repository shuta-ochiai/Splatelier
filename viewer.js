// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// A small WebGL2 viewer for antimatter15 `.splat` files (32 bytes per splat: position float3,
// linear scale float3, RGBA u8, quaternion u8x4 w-x-y-z). Splats are drawn as screen-space
// ellipses (EWA projection of the 3D covariance), sorted back to front in a worker, and blended
// with premultiplied alpha. The world is y-up (ARKit); the camera orbits a target point.

const VERT = `#version 300 es
precision highp float;
precision highp int;
uniform highp usampler2D u_data;
uniform mat4 u_view;          // world -> camera (x right, y down, z forward)
uniform vec2 u_focal;         // pixels
uniform vec2 u_viewport;      // pixels
in vec2 a_corner;             // quad corner in [-1, 1]
in uint a_index;
out vec4 v_color;
out vec2 v_pos;
const float EXTENT = 3.0;     // the quad covers 3 sigma
void main() {
  ivec2 tc = ivec2((a_index & 1023u) << 1, a_index >> 10);
  uvec4 t0 = texelFetch(u_data, tc, 0);
  uvec4 t1 = texelFetch(u_data, tc + ivec2(1, 0), 0);
  vec4 cam = u_view * vec4(uintBitsToFloat(t0.xyz), 1.0);
  if (cam.z < 0.02) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
  vec2 ndc = vec2(u_focal.x * cam.x / cam.z, -u_focal.y * cam.y / cam.z) / (0.5 * u_viewport);
  // Cheap early out for centers far off screen; the exact test (by footprint) is below.
  if (any(greaterThan(abs(ndc), vec2(4.0)))) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }

  vec3 s = uintBitsToFloat(t1.xyz);
  vec4 q = normalize((vec4(t1.w & 255u, (t1.w >> 8) & 255u, (t1.w >> 16) & 255u, t1.w >> 24) - 128.0) / 128.0);
  float w = q.x, x = q.y, y = q.z, z = q.w;
  mat3 R = mat3(
    1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y),
    2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x),
    2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y));
  mat3 M = R * mat3(s.x, 0.0, 0.0, 0.0, s.y, 0.0, 0.0, 0.0, s.z);
  mat3 sigma = M * transpose(M);

  float iz = 1.0 / cam.z;
  mat3 J = mat3(
    u_focal.x * iz, 0.0, 0.0,
    0.0, u_focal.y * iz, 0.0,
    -u_focal.x * cam.x * iz * iz, -u_focal.y * cam.y * iz * iz, 0.0);
  mat3 T = J * mat3(u_view);
  mat3 c2 = T * sigma * transpose(T);
  float a = c2[0][0] + 0.3, b = c2[0][1], c = c2[1][1] + 0.3;

  float mid = 0.5 * (a + c);
  float rad = length(vec2(0.5 * (a - c), b));
  float l1 = mid + rad, l2 = max(mid - rad, 0.1);
  vec2 axis = abs(b) < 1e-9 ? (a >= c ? vec2(1.0, 0.0) : vec2(0.0, 1.0)) : normalize(vec2(b, l1 - a));
  // Cull by footprint, not center: a big splat whose center is off screen can still cover it.
  float radius = min(EXTENT * sqrt(l1), 2048.0);
  if (any(greaterThan(abs(ndc * 0.5 * u_viewport) - radius, 0.5 * u_viewport))) {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return;
  }
  vec2 major = radius * axis;
  vec2 minor = min(EXTENT * sqrt(l2), 2048.0) * vec2(-axis.y, axis.x);
  vec2 off = a_corner.x * major + a_corner.y * minor;          // pixels, y down
  gl_Position = vec4(ndc + vec2(off.x, -off.y) * 2.0 / u_viewport, 0.0, 1.0);

  v_color = vec4(t0.w & 255u, (t0.w >> 8) & 255u, (t0.w >> 16) & 255u, t0.w >> 24) / 255.0;
  v_pos = a_corner * EXTENT;
}`;

const FRAG = `#version 300 es
precision highp float;
in vec4 v_color;
in vec2 v_pos;
out vec4 fragColor;
void main() {
  float d2 = dot(v_pos, v_pos);
  if (d2 > 9.0) discard;
  float alpha = min(0.99, v_color.a * exp(-0.5 * d2));
  if (alpha < 1.0 / 255.0) discard;
  fragColor = vec4(v_color.rgb * alpha, alpha);
}`;

// Mesh: vertex colors with a little headlight shading so the shape reads.
const MESH_VERT = `#version 300 es
precision highp float;
uniform mat4 u_view;
uniform vec2 u_focal;
uniform vec2 u_viewport;
in vec3 a_pos;
in vec3 a_normal;
in vec4 a_color;
out vec3 v_color;
out vec3 v_normal;
out vec3 v_cam;
void main() {
  vec4 cam = u_view * vec4(a_pos, 1.0);
  float n = 0.02, f = 200.0;
  gl_Position = vec4(
    u_focal.x * cam.x / (0.5 * u_viewport.x),
    -u_focal.y * cam.y / (0.5 * u_viewport.y),
    cam.z * (f + n) / (f - n) - 2.0 * f * n / (f - n),
    cam.z);
  v_color = a_color.rgb;
  v_normal = mat3(u_view) * a_normal;
  v_cam = cam.xyz;
}`;

const MESH_FRAG = `#version 300 es
precision highp float;
uniform float u_gray;          // 1: plain gray, lit from the camera, to show the shape
in vec3 v_color;
in vec3 v_normal;
in vec3 v_cam;
out vec4 fragColor;
void main() {
  float l = abs(dot(normalize(v_normal), normalize(v_cam)));
  vec3 base = mix(v_color, vec3(0.78), u_gray);
  float shade = mix(0.35 + 0.75 * l, 0.12 + 0.88 * l, u_gray);
  fragColor = vec4(base * shade, 1.0);
}`;

// Sorts splat indices far to near with a 16-bit counting sort on camera depth.
const SORT_WORKER = `
let pos = null, n = 0, depth = null, key = null;
const counts = new Uint32Array(65536);
onmessage = (e) => {
  if (e.data.positions) {
    pos = e.data.positions; n = pos.length / 3;
    depth = new Int32Array(n); key = new Uint16Array(n);     // reused by every sort
    return;
  }
  const v = e.data.view;                       // column-major 4x4; depth = row 2
  const r0 = v[2], r1 = v[6], r2 = v[10], r3 = v[14];
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < n; i++) {
    const d = (r0 * pos[3 * i] + r1 * pos[3 * i + 1] + r2 * pos[3 * i + 2] + r3) * 4096;
    depth[i] = d;
    if (d < lo) lo = d;
    if (d > hi) hi = d;
  }
  const scale = 65535 / Math.max(1, hi - lo);
  counts.fill(0);
  for (let i = 0; i < n; i++) { const k = 65535 - (((depth[i] - lo) * scale) | 0); key[i] = k; counts[k]++; }
  for (let i = 1; i < 65536; i++) counts[i] += counts[i - 1];
  const order = new Uint32Array(n);
  for (let i = n - 1; i >= 0; i--) order[--counts[key[i]]] = i;
  postMessage({ order, id: e.data.id }, [order.buffer]);
};`;

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}

function link(gl, vs, fs) {
  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  return prog;
}

// Parses the page's binary mesh: "SPLM", u32 vertex count, u32 triangle count, f32 positions,
// u8 RGBA colors, u32 indices. Returns the arrays plus smooth per-vertex normals.
export function parseMesh(buffer) {
  const head = new Uint32Array(buffer, 0, 3);
  if (new TextDecoder().decode(new Uint8Array(buffer, 0, 4)) !== 'SPLM') throw new Error('not a mesh file');
  const nv = head[1], nt = head[2];
  let o = 12;
  const pos = new Float32Array(buffer, o, nv * 3); o += nv * 12;
  const color = new Uint8Array(buffer, o, nv * 4); o += nv * 4;
  const index = new Uint32Array(buffer.slice(o, o + nt * 12));
  const normal = new Float32Array(nv * 3);
  for (let t = 0; t < nt; t++) {
    const a = index[3 * t] * 3, b = index[3 * t + 1] * 3, c = index[3 * t + 2] * 3;
    const ux = pos[b] - pos[a], uy = pos[b + 1] - pos[a + 1], uz = pos[b + 2] - pos[a + 2];
    const vx = pos[c] - pos[a], vy = pos[c + 1] - pos[a + 1], vz = pos[c + 2] - pos[a + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    for (const i of [a, b, c]) { normal[i] += nx; normal[i + 1] += ny; normal[i + 2] += nz; }
  }
  return { pos, color, index, normal, triangles: nt };
}

export class SplatViewer {
  constructor(canvas) {
    this.canvas = canvas;
    // No MSAA: splats are soft already, and it costs a lot of fill rate with this much overdraw.
    const gl = canvas.getContext('webgl2', { antialias: false, premultipliedAlpha: true, alpha: true, depth: true });
    if (!gl) throw new Error('WebGL2 is not available');
    this.gl = gl;
    const prog = link(gl, VERT, FRAG);
    gl.useProgram(prog);
    this.u = {
      data: gl.getUniformLocation(prog, 'u_data'),
      view: gl.getUniformLocation(prog, 'u_view'),
      focal: gl.getUniformLocation(prog, 'u_focal'),
      viewport: gl.getUniformLocation(prog, 'u_viewport'),
    };
    gl.uniform1i(this.u.data, 0);

    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    const quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, 1, 1, -1, 1]), gl.STATIC_DRAW);
    const aCorner = gl.getAttribLocation(prog, 'a_corner');
    gl.enableVertexAttribArray(aCorner);
    gl.vertexAttribPointer(aCorner, 2, gl.FLOAT, false, 0, 0);
    this.indexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.indexBuffer);
    const aIndex = gl.getAttribLocation(prog, 'a_index');
    gl.enableVertexAttribArray(aIndex);
    gl.vertexAttribIPointer(aIndex, 1, gl.UNSIGNED_INT, 0, 0);
    gl.vertexAttribDivisor(aIndex, 1);
    this.texture = gl.createTexture();
    this.splatProg = prog;

    this.meshProg = link(gl, MESH_VERT, MESH_FRAG);
    this.mu = {
      view: gl.getUniformLocation(this.meshProg, 'u_view'),
      focal: gl.getUniformLocation(this.meshProg, 'u_focal'),
      viewport: gl.getUniformLocation(this.meshProg, 'u_viewport'),
      gray: gl.getUniformLocation(this.meshProg, 'u_gray'),
    };
    this.meshVao = gl.createVertexArray();
    this.meshBuffers = [];
    this.meshCount = 0;
    this.mode = 'splats';

    this.worker = new Worker(URL.createObjectURL(new Blob([SORT_WORKER], { type: 'text/javascript' })));
    this.worker.onmessage = (e) => this.onSorted(e.data);
    this.sortId = 0;
    this.sortBusy = false;
    this.sortedView = null;
    this.count = 0;
    this.drawCount = 0;

    // Orbit camera.
    this.target = [0, 0, 0];
    this.yaw = 0;
    this.pitch = 0.3;
    this.dist = 2;
    this.fovY = (50 * Math.PI) / 180;
    this.autoSway = true;
    this.swayBase = 0;
    this.swayAmp = 0.25;
    this.dirty = true;
    this.installControls();
    this.resizeObserver = new ResizeObserver(() => { this.dirty = true; });
    this.resizeObserver.observe(canvas);
    this.paused = false;
    if ('IntersectionObserver' in window) {
      new IntersectionObserver((entries) => {
        this.paused = !entries[entries.length - 1].isIntersecting;
        if (!this.paused) this.dirty = true;
      }).observe(canvas);
    }
    this.t0 = performance.now();
    // The frame timestamp (aligned to the display refresh) drives the motion, so steps are even.
    const loop = (ts) => { this.frame(ts); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
  }

  // `buffer` holds .splat records. `cam` = {target, yaw, pitch, dist, fovY?, sway?}.
  setSplats(buffer, cam) {
    const gl = this.gl;
    const n = Math.floor(buffer.byteLength / 32);
    const words = new Uint32Array(buffer, 0, n * 8);
    const f32 = new Float32Array(buffer, 0, n * 8);
    const height = Math.ceil(n / 1024);
    const tex = new Uint32Array(2048 * height * 4);
    const positions = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const s = i * 8, d = i * 8;
      tex[d] = words[s]; tex[d + 1] = words[s + 1]; tex[d + 2] = words[s + 2]; tex[d + 3] = words[s + 6];
      tex[d + 4] = words[s + 3]; tex[d + 5] = words[s + 4]; tex[d + 6] = words[s + 5]; tex[d + 7] = words[s + 7];
      positions[i * 3] = f32[s]; positions[i * 3 + 1] = f32[s + 1]; positions[i * 3 + 2] = f32[s + 2];
    }
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32UI, 2048, height, 0, gl.RGBA_INTEGER, gl.UNSIGNED_INT, tex);
    this.count = n;
    this.drawCount = 0;
    this.sortedView = null;
    this.sortId++;
    this.sortBusy = false;
    this.worker.postMessage({ positions }, [positions.buffer]);
    if (cam) {
      this.target = cam.target.slice();
      this.yaw = cam.yaw; this.pitch = cam.pitch; this.dist = cam.dist;
      this.fovY = ((cam.fovY ?? 50) * Math.PI) / 180;
      this.swayBase = cam.yaw;
      this.swayAmp = cam.sway ?? 0.25;
      this.home = { ...cam };
      this.autoSway = true;
      this.t0 = performance.now();
    }
    this.dirty = true;
  }

  // `mesh` comes from parseMesh(). Passing null clears it.
  setMesh(mesh) {
    const gl = this.gl;
    for (const b of this.meshBuffers) gl.deleteBuffer(b);
    this.meshBuffers = [];
    this.meshCount = 0;
    if (mesh) {
      gl.bindVertexArray(this.meshVao);
      const attr = (name, data, size, type, normalized) => {
        const b = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, b);
        gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
        const loc = gl.getAttribLocation(this.meshProg, name);
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, size, type, normalized, 0, 0);
        this.meshBuffers.push(b);
      };
      attr('a_pos', mesh.pos, 3, gl.FLOAT, false);
      attr('a_normal', mesh.normal, 3, gl.FLOAT, false);
      attr('a_color', mesh.color, 4, gl.UNSIGNED_BYTE, true);
      const ib = gl.createBuffer();
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, mesh.index, gl.STATIC_DRAW);
      this.meshBuffers.push(ib);
      this.meshCount = mesh.index.length;
      gl.bindVertexArray(null);
    }
    this.dirty = true;
  }

  // 'splats', 'mesh' (vertex colors) or 'shape' (gray mesh).
  setMode(mode) {
    this.mode = mode;
    this.dirty = true;
  }

  // Points the orbit camera from `eye` at `target` (world up stays up).
  lookAt(eye, target) {
    this.distGoal = null;
    const d = sub(eye, target);
    const dist = Math.hypot(d[0], d[1], d[2]) || 1;
    this.target = target.slice();
    this.dist = dist;
    this.yaw = Math.atan2(d[0], d[2]);
    this.pitch = Math.asin(clamp(d[1] / dist, -1, 1));
    this.dirty = true;
  }

  // An orbit around `target` that the idle viewer plays: {target, closed, theta0, theta1, r[], h[]},
  // with r (horizontal distance) and h (height) sampled evenly from theta0 to theta1 (azimuth, the
  // viewer's yaw). The camera moves at a constant speed along the curve (by arc length, not by
  // angle): a closed orbit takes `secondsPerTurn` per lap; an open arc goes back and forth with a
  // short ramp at each end. It starts at the azimuth nearest the home view. Null goes back to the
  // gentle sway.
  setPath(path, secondsPerTurn = 60) {
    this.path = path && path.r && path.r.length > 1 ? path : null;
    if (this.path) {
      const p = this.path, span = p.theta1 - p.theta0;
      // Cumulative length along a dense sampling of the curve, for constant-speed playback.
      const m = (p.r.length - 1) * 8 + 1;
      const len = new Float64Array(m);
      let prev = this.orbitEye(0);
      for (let j = 1; j < m; j++) {
        const e = this.orbitEye(j / (m - 1));
        len[j] = len[j - 1] + Math.hypot(e[0] - prev[0], e[1] - prev[1], e[2] - prev[2]);
        prev = e;
      }
      this.pathLen = len;
      this.pathSeconds = Math.max(15, (secondsPerTurn * Math.abs(span)) / (2 * Math.PI));
      const yaw = this.home ? this.home.yaw : this.yaw;
      const turns = (((yaw - p.theta0) / (2 * Math.PI)) % 1 + 1) % 1;
      let u0 = p.closed ? turns : clamp((turns * 2 * Math.PI) / span, 0, 1);
      if (!p.closed && turns * 2 * Math.PI > span) u0 = 0;
      const s0 = len[Math.round(u0 * (m - 1))] / len[m - 1];
      this.pathOffset = s0 * this.pathSeconds;   // start there, moving forward
    }
    this.restartAuto();
  }

  // Eye on the orbit at u in [0, 1] (evenly in azimuth).
  orbitEye(u) {
    const p = this.path, n = p.r.length;
    const x = u * (n - 1), i = Math.min(n - 2, Math.floor(x)), f = x - i;
    const r = p.r[i] + (p.r[i + 1] - p.r[i]) * f;
    const h = p.h[i] + (p.h[i + 1] - p.h[i]) * f;
    const th = p.theta0 + (p.theta1 - p.theta0) * u;
    const t = p.target;
    return [t[0] + r * Math.sin(th), t[1] + h, t[2] + r * Math.cos(th)];
  }

  restartAuto() {
    this.autoSway = true;
    this.t0 = performance.now();
    this.clock = 0;
    if (this.path) { const { eye, target } = this.pathPose(0); this.lookAt(eye, target); }
    this.dirty = true;
  }

  // Orbit position `time` seconds into the demo, at constant speed along the curve.
  pathPose(time) {
    const p = this.path, len = this.pathLen, m = len.length, L = len[m - 1];
    const k = (time + this.pathOffset) / this.pathSeconds;
    // One direction only: a closed orbit just keeps going; an open arc starts over at its
    // beginning (the frame loop fades across the jump).
    const s = ((k % 1) + 1) % 1;                      // fraction of the length travelled
    // Arc length → curve parameter.
    const target = s * L;
    let lo = 0, hi = m - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (len[mid] <= target) lo = mid; else hi = mid; }
    const f = (target - len[lo]) / Math.max(1e-9, len[hi] - len[lo]);
    return { eye: this.orbitEye((lo + clamp(f, 0, 1)) / (m - 1)), target: p.target };
  }

  // For a viewer without splats (a mesh only): the view that Reset and the orbit start from.
  setHome(cam) {
    this.home = { ...cam, target: cam.target.slice() };
    this.fovY = ((cam.fovY ?? 50) * Math.PI) / 180;
    this.setCamera(cam);
  }

  resetView() {
    if (this.path) this.restartAuto();
    else if (this.home) this.setCamera(this.home);
  }

  setCamera(cam) {
    this.distGoal = null;
    this.target = cam.target.slice();
    this.yaw = cam.yaw; this.pitch = cam.pitch; this.dist = cam.dist;
    this.swayBase = cam.yaw;
    this.autoSway = true;
    this.t0 = performance.now();
    this.dirty = true;
  }

  onSorted({ order, id }) {
    this.sortBusy = false;
    if (id !== this.sortId) return;
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.indexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, order, gl.DYNAMIC_DRAW);
    this.drawCount = order.length;
    this.dirty = true;
    if (this.onFirstFrame) { const f = this.onFirstFrame; this.onFirstFrame = null; requestAnimationFrame(() => f()); }
  }

  viewMatrix() {
    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const eye = [
      this.target[0] + this.dist * cp * Math.sin(this.yaw),
      this.target[1] + this.dist * sp,
      this.target[2] + this.dist * cp * Math.cos(this.yaw),
    ];
    const f = norm(sub(this.target, eye));
    const r = norm(cross(f, [0, 1, 0]));
    const dn = cross(f, r);
    // Column-major; rows are right, down, forward.
    return new Float32Array([
      r[0], dn[0], f[0], 0,
      r[1], dn[1], f[1], 0,
      r[2], dn[2], f[2], 0,
      -dot(r, eye), -dot(dn, eye), -dot(f, eye), 1,
    ]);
  }

  frame(ts = performance.now()) {
    if (this.paused) { this.lastTs = ts; return; }      // off screen: draw nothing
    const gl = this.gl;
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    const w = Math.round(this.canvas.clientWidth * dpr), h = Math.round(this.canvas.clientHeight * dpr);
    if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; this.dirty = true; }
    const dt = this.lastTs == null ? 0 : Math.min(0.1, Math.max(0, (ts - this.lastTs) / 1000));
    this.lastTs = ts;
    if (this.distGoal != null) {
      // Ease toward the wheel's goal in log space (≈0.12 s time constant).
      const a = 1 - Math.exp(-dt / 0.12);
      this.dist = Math.exp(Math.log(this.dist) + (Math.log(this.distGoal) - Math.log(this.dist)) * a);
      if (Math.abs(Math.log(this.dist / this.distGoal)) < 1e-3) { this.dist = this.distGoal; this.distGoal = null; }
      this.dirty = true;
    }
    const hasContent = this.count > 0 || (this.mode !== 'splats' && this.meshCount > 0);
    if (this.autoSway && hasContent) {
      const t = Math.max(0, ts - this.t0) / 1000;
      if (this.path) {
        const { eye, target } = this.pathPose(t);
        this.lookAt(eye, target);
        // An open arc jumps back to its start at the end of each pass: fade out and in across it.
        let alpha = 1;
        if (!this.path.closed) {
          const P = this.pathSeconds, x = ((((t + this.pathOffset) % P) + P) % P);
          alpha = clamp(Math.min(x, P - x) / 0.6, 0, 1);
        }
        this.canvas.style.opacity = String(alpha);
      } else {
        this.yaw = this.swayBase + this.swayAmp * Math.sin(t * 0.35);
        this.canvas.style.opacity = '1';
      }
      this.dirty = true;
    }
    if (!this.dirty || !hasContent) return;
    this.dirty = false;
    const view = this.viewMatrix();
    const fy = (0.5 * h) / Math.tan(0.5 * this.fovY);
    gl.viewport(0, 0, w, h);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (this.mode !== 'splats') {
      if (!this.meshCount) return;
      gl.enable(gl.DEPTH_TEST);
      gl.disable(gl.BLEND);
      gl.useProgram(this.meshProg);
      gl.uniform1f(this.mu.gray, this.mode === 'shape' ? 1 : 0);
      gl.uniformMatrix4fv(this.mu.view, false, view);
      gl.uniform2f(this.mu.focal, fy, fy);
      gl.uniform2f(this.mu.viewport, w, h);
      gl.bindVertexArray(this.meshVao);
      gl.drawElements(gl.TRIANGLES, this.meshCount, gl.UNSIGNED_INT, 0);
      return;
    }
    if (!this.sortBusy && (!this.sortedView || viewChanged(view, this.sortedView))) {
      this.sortBusy = true;
      this.sortedView = view;
      this.worker.postMessage({ view, id: this.sortId });
    }
    if (!this.drawCount) return;
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(this.splatProg);
    gl.uniformMatrix4fv(this.u.view, false, view);
    gl.uniform2f(this.u.focal, fy, fy);
    gl.uniform2f(this.u.viewport, w, h);
    gl.bindVertexArray(this.vao);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, this.drawCount);
  }

  installControls() {
    const c = this.canvas;
    const pointers = new Map();
    let last = null;
    const stop = () => { this.autoSway = false; this.canvas.style.opacity = '1'; };
    c.addEventListener('pointerdown', (e) => {
      c.setPointerCapture(e.pointerId);
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      last = gesture(pointers);
      stop();
    });
    c.addEventListener('pointermove', (e) => {
      if (!pointers.has(e.pointerId)) return;
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      const g = gesture(pointers);
      if (!last || g.n !== last.n) { last = g; return; }
      const dx = g.x - last.x, dy = g.y - last.y;
      const pan = g.n >= 2 || e.shiftKey || (e.buttons & 2) !== 0;
      if (pan) this.pan(dx, dy);
      else {
        this.yaw -= dx * 0.006;
        this.pitch = clamp(this.pitch + dy * 0.006, -1.45, 1.45);
      }
      if (g.n >= 2 && last.span > 0) {
        this.distGoal = null;
        this.dist = clamp(this.dist * (last.span / g.span), MIN_DIST, 100);
      }
      last = g;
      this.dirty = true;
    });
    const up = (e) => { pointers.delete(e.pointerId); last = gesture(pointers); };
    c.addEventListener('pointerup', up);
    c.addEventListener('pointercancel', up);
    c.addEventListener('contextmenu', (e) => e.preventDefault());
    // The wheel sets where the zoom is heading; frame() eases the distance there, so a mouse
    // wheel's coarse notches don't jump.
    c.addEventListener('wheel', (e) => {
      e.preventDefault();
      stop();
      const dy = e.deltaY * (e.deltaMode === 1 ? 33 : e.deltaMode === 2 ? 400 : 1);
      this.distGoal = clamp((this.distGoal ?? this.dist) * Math.exp(dy * 0.0015), MIN_DIST, 100);
      this.dirty = true;
    }, { passive: false });
    c.addEventListener('dblclick', () => this.resetView());
  }

  pan(dx, dy) {
    const v = this.viewMatrix();
    const h = this.canvas.clientHeight;
    const k = (2 * this.dist * Math.tan(0.5 * this.fovY)) / h;
    const r = [v[0], v[4], v[8]], dn = [v[1], v[5], v[9]];
    for (let i = 0; i < 3; i++) this.target[i] -= (r[i] * dx + dn[i] * dy) * k;
  }
}

function gesture(pointers) {
  const ps = [...pointers.values()];
  if (!ps.length) return null;
  const x = ps.reduce((s, p) => s + p.x, 0) / ps.length;
  const y = ps.reduce((s, p) => s + p.y, 0) / ps.length;
  const span = ps.length >= 2 ? Math.hypot(ps[0].x - ps[1].x, ps[0].y - ps[1].y) : 0;
  return { n: ps.length, x, y, span };
}

function viewChanged(a, b) {
  let d = 0;
  for (let i = 0; i < 16; i++) d += Math.abs(a[i] - b[i]);
  return d > 1e-3;
}
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
// Closest the orbit camera gets to its target (m): nearer, it ends up inside the subject.
const MIN_DIST = 0.2;
