// Juan in 3D: the model from /web/model/ (scripts/model/web_model.py), posed with the live joint angles.
// The torso: in the simulator (--sim) its true position and orientation; on the robot, the IMU's tilt and heading,
// standing on the lower foot. Detail: off / boxes / low / high (remembered in this browser). Drag to orbit,
// scroll or pinch to zoom, right-drag to pan.
import * as THREE from './vendor/three.module.min.js';
import {OrbitControls} from './vendor/OrbitControls.js';

export const DETAILS = ['off', 'boxes', 'low', 'high'];
const KEY = 'juan.view3d.detail';
const FEET = ['left_foot_link', 'right_foot_link'];
const model = {json: null, bins: {}};

function stored(key, fallback) { try { return localStorage.getItem(key) || fallback; } catch (e) { return fallback; } }
function store(key, v) { try { localStorage.setItem(key, v); } catch (e) { /* private mode: not remembered */ } }

async function loadJson() {
  if (!model.json) model.json = await (await fetch('/web/model/juan.json')).json();
  return model.json;
}
async function loadBin(level) {
  if (!model.bins[level]) model.bins[level] = await (await fetch('/web/model/' + model.json.levels[level].file)).arrayBuffer();
  return model.bins[level];
}
const quat = q => new THREE.Quaternion(q[1], q[2], q[3], q[0]);   // MuJoCo (w, x, y, z)

export class View3D {
  constructor(container, {follow = true} = {}) {
    this.el = container;
    this.detail = stored(KEY, 'low');
    if (!DETAILS.includes(this.detail)) this.detail = 'low';
    this.follow = follow;
    this.bodies = {};            // name -> {frame, joint, holder, axis}
    this.dirty = true;
    this.ready = this.init();
  }

  async init() {
    const r = this.renderer = new THREE.WebGLRenderer({antialias: true});
    r.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.el.appendChild(r.domElement);
    r.domElement.style.cssText = 'width:100%;height:100%;display:block;border-radius:8px;touch-action:none';
    const scene = this.scene = new THREE.Scene();
    scene.background = new THREE.Color(matchMedia('(prefers-color-scheme: dark)').matches ? 0x15181c : 0xe9edf2);
    const cam = this.camera = new THREE.PerspectiveCamera(40, 1, 0.01, 50);
    cam.up.set(0, 0, 1);
    cam.position.set(0.55, -0.6, 0.38);
    this.controls = new OrbitControls(cam, r.domElement);
    this.controls.target.set(0, 0, 0.17);
    this.controls.addEventListener('change', () => { this.dirty = true; });
    scene.add(new THREE.HemisphereLight(0xffffff, 0x445566, 1.6));
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(1, -1.5, 2.5);
    scene.add(sun);
    const grid = new THREE.GridHelper(10, 100, 0x8899aa, 0x8899aa);
    grid.rotation.x = Math.PI / 2;
    grid.material.transparent = true; grid.material.opacity = 0.6;
    this.grid = grid;
    scene.add(grid);
    const arrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0, 0.002), 0.25, 0x2563eb, 0.05, 0.03);
    this.heading = arrow;
    scene.add(arrow);
    new ResizeObserver(() => this.resize()).observe(this.el);
    this.resize();

    const json = await loadJson();
    this.root = new THREE.Group();
    scene.add(this.root);
    for (const b of json.bodies) {
      const frame = new THREE.Group(), joint = new THREE.Group(), holder = new THREE.Group();
      if (b.parent) { frame.position.fromArray(b.pos); frame.quaternion.copy(quat(b.quat)); }
      frame.add(joint); joint.add(holder);
      (b.parent ? this.bodies[b.parent].joint : this.root).add(frame);
      this.bodies[b.name] = {frame, joint, holder, info: b,
                             axis: b.joint ? new THREE.Vector3().fromArray(b.joint.axis) : null};
    }
    await this.setDetail(this.detail);
    const loop = () => {
      requestAnimationFrame(loop);
      if (this.dirty && this.el.offsetParent !== null && this.detail !== 'off') {
        this.dirty = false;
        this.renderer.render(this.scene, this.camera);
      }
    };
    loop();
  }

  resize() {
    const w = this.el.clientWidth || 1, h = this.el.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.dirty = true;
  }

  async setDetail(level) {
    this.detail = level;
    store(KEY, level);
    this.el.hidden = level === 'off';
    if (level === 'off' || !model.json) return;
    const bin = level === 'boxes' ? null : await loadBin(level);
    if (this.detail !== level) return;   // changed again while loading
    for (const [name, b] of Object.entries(this.bodies)) {
      for (const c of [...b.holder.children]) { b.holder.remove(c); c.geometry.dispose(); }
      if (level === 'boxes') {
        const [lo, hi] = b.info.box, size = hi.map((v, i) => Math.max(v - lo[i], 0.005));
        const mesh = new THREE.Mesh(new THREE.BoxGeometry(...size),
                                    new THREE.MeshStandardMaterial({color: 0x9aa4b1, roughness: 0.8}));
        mesh.position.set((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2);
        b.holder.add(mesh);
        continue;
      }
      const part = model.json.levels[level].bodies[name];
      if (!part) continue;
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(bin, part.pos, part.verts * 3), 3));
      g.setAttribute('color', new THREE.BufferAttribute(new Uint8Array(bin, part.col, part.verts * 3), 3, true));
      g.setIndex(new THREE.BufferAttribute(new Uint32Array(bin, part.idx, part.tris * 3), 1));
      g.computeBoundingBox(); g.computeBoundingSphere();
      b.holder.add(new THREE.Mesh(g, new THREE.MeshStandardMaterial({vertexColors: true, flatShading: true,
                                                                     roughness: 0.75, metalness: 0.05, side: THREE.DoubleSide})));
    }
    this.dirty = true;
  }

  // The page's state (core.js R.S): joint angles, and the torso from the simulator or the IMU.
  update(S) {
    if (!this.root || !S || this.detail === 'off') return;
    for (const [name, j] of Object.entries(S.joints || {})) {
      const b = Object.values(this.bodies).find(x => x.info.joint && x.info.joint.name === name);
      if (b && j.deg !== null && j.deg !== undefined) b.joint.quaternion.setFromAxisAngle(b.axis, j.deg * Math.PI / 180);
    }
    const before = this.root.position.clone();
    if (S.sim) {
      this.root.position.fromArray(S.sim.pos);
      this.root.quaternion.copy(quat(S.sim.quat));
    } else if (S.imu && S.imu.gravity) {
      const g = new THREE.Vector3().fromArray(S.imu.gravity).normalize();
      const tilt = new THREE.Quaternion().setFromUnitVectors(g, new THREE.Vector3(0, 0, -1));
      const yaw = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), S.imu.yaw || 0);
      this.root.quaternion.copy(yaw.multiply(tilt));
      this.root.position.set(0, 0, 0);
      this.root.updateMatrixWorld(true);
      let low = Infinity;   // stand on the lower foot
      for (const f of FEET) if (this.bodies[f]) low = Math.min(low, new THREE.Box3().setFromObject(this.bodies[f].holder).min.z);
      if (Number.isFinite(low)) this.root.position.z = -low;
    }
    this.root.updateMatrixWorld(true);
    const torso = new THREE.Vector3().setFromMatrixPosition(this.root.matrixWorld);
    this.heading.position.set(torso.x, torso.y, 0.002);
    const fwd = new THREE.Vector3(1, 0, 0).applyQuaternion(this.root.quaternion); fwd.z = 0;
    if (fwd.lengthSq() > 1e-6) this.heading.setDirection(fwd.normalize());
    if (this.follow) {   // the camera moves with the robot (orbiting still works)
      const d = this.root.position.clone().sub(before); d.z = 0;
      this.camera.position.add(d);
      this.controls.target.add(d);
      this.grid.position.set(Math.round(torso.x), Math.round(torso.y), 0);
      this.controls.update();
    }
    this.dirty = true;
  }

  home() {
    const p = this.root ? this.root.position : new THREE.Vector3();
    this.camera.position.set(p.x + 0.55, p.y - 0.6, 0.38);
    this.controls.target.set(p.x, p.y, 0.17);
    this.controls.update();
    this.dirty = true;
  }
}
