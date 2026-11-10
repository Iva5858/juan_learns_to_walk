// Drive: start a policy, then steer it. Every input sets the same command (forward m/s, sideways m/s, turn
// rad/s), sent 10 times a second while it changes and every 0.25 s otherwise:
//   * forward speed is a set speed (cruise): number keys 0-6 (0.1 m/s each; Shift+1/2 backward), the preset
//     buttons or the slider set it; holding up / down ramps it; the joystick drives it directly while held;
//   * turning and side-stepping last while held (keys, buttons, the joystick's sideways axis, a gamepad).
// The robot caps the command to what the policy was trained for; the page shows both.
import {$, R, battery, call, esc, has, n1, onState} from './core.js';

const RAMP = 0.2;                 // m/s per second while up / down is held
const DECEL = 0.5;                // m/s per second at most when slowing down or stopping: the robot logs of
                                  // 2026-11-03 fell 5 times within 0.8 s of an instant stop from 0.4-0.6 m/s
const PRESETS = [-0.15, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
const DEFAULT_RANGES = {forward_range: [-0.15, 0.6], lateral_range: [-0.15, 0.15], yaw_rate_range: [-0.6, 0.6]};
const DEAD = 0.15;                // gamepad stick dead zone

const st = {
  cruise: 0,                      // set forward speed
  hold: {turnL: 0, turnR: 0, sideL: 0, sideR: 0, up: 0, down: 0},
  stick: null,                    // {x, y} in -1..1 while the on-screen joystick is held
  pad: null,                      // the same from a gamepad
  turnRate: 0.4, sideSpeed: 0.1, stickHold: false,
  sent: null, sentAt: 0, last: performance.now(), vxOut: 0,
};

const ranges = () => (R.S && R.S.walker.ranges) || DEFAULT_RANGES;
const clamp = (v, r) => Math.min(r[1], Math.max(r[0], v));
const round = v => Math.round(v * 100) / 100 || 0;
const running = () => R.S && R.S.walker.running;

export function build(root) {
  root.innerHTML = `
  <div class="drive">
    <div>
      <div class="card">
        <div class="row">
          <select id="dPolicy" style="flex:1;min-width:0"></select>
          <button class="go" id="dStart">Start</button>
          <button class="danger" id="dStop" title="Esc">Stop (hold)</button>
          <button id="dPickup" title="P: stop, logged as a pickup, not a fall">Pick up</button>
        </div>
        <div class="row">
          <span class="sub">Lying face down:</span>
          <select id="dGetup" style="flex:1;min-width:0"></select>
          <label class="sub" title="Once up, the walking policy above takes over, standing; unticked, the getup policy keeps it standing (30 s)"><input type="checkbox" id="dGetupWalk"> then walker</label>
          <button class="go" id="dGetupBtn" title="Getup policy until up (one try, 6 s), then standing">Get up</button>
        </div>
        <div class="row">
          <span class="sub">Before starting:</span>
          <button class="small" id="dBoot" title="Straight legs, then the home crouch">Boot</button>
          <button class="small" id="dHome">Home</button>
          <button class="small" id="dRelax">Relax all</button>
          <span class="sub" id="dNote"></span>
        </div>
      </div>
      <div class="card" style="margin-top:12px">
        <div class="row"><b>While walking</b><span class="sub">(everything: Status tab; the keys drive from any tab)</span></div>
        <div style="display:grid;grid-template-columns:110px 1fr;gap:10px;align-items:center">
          <svg class="horizon" id="dHorizon" viewBox="0 0 110 70" style="height:70px;margin:0"></svg>
          <div class="kv" id="dVitals" style="margin:0"></div></div>
      </div>
      <div class="card" style="margin-top:12px">
        <div class="sub">Command</div>
        <div class="cmdline" id="dCmd">-</div>
        <div class="sub" id="dSent"></div>
        <div class="row presets" id="dPresets"></div>
        <div class="row"><span class="sub">Speed</span>
          <input type="range" id="dSlider" min="-0.15" max="0.6" step="0.01" value="0" style="flex:1">
          <b id="dSliderV" style="min-width:64px;text-align:right">0.00 m/s</b></div>
        <div class="row">
          <label class="sub">Turn <select id="dTurnRate"><option>0.2</option><option selected>0.4</option><option>0.6</option></select> rad/s</label>
          <label class="sub">Side <select id="dSide"><option>0.05</option><option selected>0.1</option><option>0.15</option></select> m/s</label>
          <label class="sub"><input type="checkbox" id="dStickHold"> joystick keeps its speed when let go</label>
        </div>
        <div class="row"><label class="sub">Servo move time per step
          <select id="dMs"><option value="0">0 ms (as in training)</option><option value="20">20 ms</option></select></label></div>
      </div>
    </div>
    <div>
      <div class="card">
        <div class="stickrow">
          <div class="stick" id="dStick"><div class="knob" id="dKnob"></div>
            <span class="lbl" style="top:6px;left:calc(50% - 10px)">fwd</span><span class="lbl" style="bottom:6px;left:calc(50% - 14px)">back</span>
            <span class="lbl" style="left:8px;top:calc(50% - 8px)">turn L</span><span class="lbl" style="right:8px;top:calc(50% - 8px)">turn R</span></div>
          <div class="holds">
            <button data-hold="turnL" title="← held">⟲ turn</button><button data-hold="turnR" title="→ held">turn ⟳</button>
            <button data-hold="sideL" title=", held">← side</button><button data-hold="sideR" title=". held">side →</button>
            <button id="dStand" style="grid-column:1/-1" title="Space">Stand still</button>
          </div>
        </div>
        <div class="sub" id="dPadNote" style="text-align:center;margin-top:6px"></div>
      </div>
      <div class="card" style="margin-top:12px">
        <div class="row"><b>Camera</b>
          <button class="small" id="dCamBtn">Show</button>
          <select id="dCamView"><option value="raw">Raw</option><option value="stabilized">Stabilized</option></select>
          <span class="sub" id="dCamNote">Off until shown (the stream costs CPU).</span></div>
        <img class="cam" id="dCam" alt="" hidden>
      </div>
      <div class="card sub" style="margin-top:12px" id="dStats"></div>
      <div class="card sub" style="margin-top:12px">
        Slowing down and stopping (<kbd>Space</kbd>, lower speeds) ease off at 0.5 m/s per second: stopping
        dead from a fast walk tipped the robot over. <kbd>Esc</kbd> stops the policy at once.
        Keys: <kbd>0</kbd>–<kbd>6</kbd> forward speed (0.1 m/s each), <kbd>Shift</kbd>+<kbd>1</kbd>/<kbd>2</kbd> backward;
        <kbd>↑</kbd><kbd>↓</kbd> held: faster / slower; <kbd>←</kbd><kbd>→</kbd> held: turn; <kbd>,</kbd><kbd>.</kbd> held: sideways;
        <kbd>Space</kbd> stand still; <kbd>Esc</kbd> stop; <kbd>P</kbd> pick up. Gamepad: left stick drives, right stick
        sideways, A stand still, B stop.
        It stands still by itself if this page goes quiet for 2 s and stops after 6 s (keep it in front, screen on);
        it relaxes everything past 40° of tilt.
      </div>
    </div>
  </div>`;

  $('dPresets').innerHTML = PRESETS.map(v => `<button data-v="${v}">${v < 0 ? 'back ' + -v : v === 0 ? '0' : v}</button>`).join('');
  $('dPresets').onclick = e => { const v = e.target.dataset.v; if (v !== undefined) setCruise(+v); };
  $('dSlider').oninput = e => setCruise(+e.target.value);
  $('dTurnRate').onchange = e => { st.turnRate = +e.target.value; };
  $('dSide').onchange = e => { st.sideSpeed = +e.target.value; };
  $('dStickHold').onchange = e => { st.stickHold = e.target.checked; };
  $('dMs').onchange = e => call('policy/move_ms', {ms: +e.target.value});
  $('dStart').onclick = start;
  $('dGetupBtn').onclick = () => { standStill(); call('policy/getup', {getup: $('dGetup').value, walker: $('dGetupWalk').checked ? $('dPolicy').value : null}); };
  $('dStop').onclick = () => call('policy/stop');
  $('dPickup').onclick = () => call('policy/stop', {why: 'pickup'});
  $('dBoot').onclick = () => call('boot');
  $('dHome').onclick = () => call('pose', {name: 'home'});
  $('dRelax').onclick = () => call('relax');
  $('dStand').onclick = standStill;
  for (const b of root.querySelectorAll('[data-hold]')) holdButton(b, b.dataset.hold);
  joystick($('dStick'), $('dKnob'));
  $('dCamBtn').onclick = () => camera(!camOn);
  $('dCamView').onchange = () => camera(camOn);
  document.addEventListener('keydown', keydown);
  document.addEventListener('keyup', keyup);
  window.addEventListener('blur', releaseAll);
  requestAnimationFrame(tick);
}

function setCruise(v) { st.cruise = clamp(v, ranges().forward_range); }   // (rounded when sent: a ramp adds tiny steps)
function standStill() { st.cruise = 0; releaseAll(); }
function releaseAll() { for (const k in st.hold) st.hold[k] = 0; st.stick = null; }

function start() {
  const name = $('dPolicy').value;
  if (!name) return;
  if (!window.confirm(`Start ${name}? The robot must be at Home, hanging or standing on the floor (harness on). It starts standing still.`)) return;
  standStill();
  call('policy/start', {name}, {confirm: false});
}

// ---- inputs ----
function holdButton(b, key) {
  const on = e => { e.preventDefault(); st.hold[key] = 1; b.classList.add('sel'); b.setPointerCapture?.(e.pointerId); };
  const off = () => { st.hold[key] = 0; b.classList.remove('sel'); };
  b.addEventListener('pointerdown', on);
  for (const ev of ['pointerup', 'pointercancel', 'lostpointercapture']) b.addEventListener(ev, off);
}

function joystick(el, knob) {
  const move = e => {
    const r = el.getBoundingClientRect(), R0 = r.width / 2;
    let x = (e.clientX - r.left - R0) / (R0 - 35), y = (e.clientY - r.top - R0) / (R0 - 35);
    const len = Math.hypot(x, y);
    if (len > 1) { x /= len; y /= len; }
    st.stick = {x, y};
    knob.style.left = (R0 - 35 + x * (R0 - 35)) + 'px';
    knob.style.top = (R0 - 35 + y * (R0 - 35)) + 'px';
  };
  const release = () => {
    if (st.stick && st.stickHold) setCruise(-st.stick.y * ranges().forward_range[1]);
    else if (st.stick) st.cruise = 0;
    st.stick = null;
    knob.style.left = knob.style.top = '';
  };
  el.addEventListener('pointerdown', e => { e.preventDefault(); el.setPointerCapture(e.pointerId); move(e); });
  el.addEventListener('pointermove', e => { if (st.stick) move(e); });
  for (const ev of ['pointerup', 'pointercancel']) el.addEventListener(ev, release);
}

const KEYS = {ArrowLeft: 'turnL', ArrowRight: 'turnR', Comma: 'sideL', Period: 'sideR', ArrowUp: 'up', ArrowDown: 'down'};
function typing(e) { return ['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName) && e.target.type !== 'range' && e.target.type !== 'checkbox'; }
function keydown(e) {
  if (typing(e) || e.metaKey || e.ctrlKey || e.altKey) return;
  const hold = KEYS[e.code] || KEYS[e.key];
  if (hold) { e.preventDefault(); st.hold[hold] = 1; return; }
  const digit = /^Digit([0-6])$/.exec(e.code);
  if (digit && !e.repeat) {
    e.preventDefault();
    const d = +digit[1];
    return setCruise(e.shiftKey ? -0.05 - 0.05 * d : d / 10);   // Shift+1: back 0.1, Shift+2: back 0.15
  }
  if (e.code === 'Space') { e.preventDefault(); standStill(); }
  else if (e.code === 'Escape') call('policy/stop');
  else if (e.code === 'KeyP') call('policy/stop', {why: 'pickup'});
}
function keyup(e) {
  const hold = KEYS[e.code] || KEYS[e.key];
  if (hold) st.hold[hold] = 0;
}

let padButtons = [];
function gamepad() {
  const gp = [...(navigator.getGamepads ? navigator.getGamepads() : [])].find(g => g && g.connected);
  if (!gp) { st.pad = null; return null; }
  const ax = k => (Math.abs(gp.axes[k] || 0) > DEAD ? gp.axes[k] : 0);
  const pressed = gp.buttons.map(b => b.pressed);
  if (pressed[0] && !padButtons[0]) standStill();
  if (pressed[1] && !padButtons[1]) call('policy/stop');
  padButtons = pressed;
  st.pad = {x: ax(0), y: ax(1), side: ax(2)};
  return gp.id;
}

// ---- the command ----
function command(dt) {
  const r = ranges();
  const h = st.hold;
  if (h.up || h.down) setCruise(st.cruise + (h.up - h.down) * RAMP * dt);
  let vx = st.cruise, vy = (h.sideL - h.sideR) * st.sideSpeed, wz = (h.turnL - h.turnR) * st.turnRate;
  const drive = st.stick || (st.pad && (st.pad.x || st.pad.y) ? st.pad : null);
  if (drive) {
    vx = -drive.y * (drive.y < 0 ? r.forward_range[1] : -r.forward_range[0]);
    wz = -drive.x * r.yaw_rate_range[1];
  }
  if (st.pad && st.pad.side) vy = -st.pad.side * r.lateral_range[1];
  vx = clamp(vx, r.forward_range);
  // slow down gently: toward zero no faster than DECEL (speeding up and reversing past zero are not limited)
  if (Math.abs(vx) < Math.abs(st.vxOut) && Math.sign(vx) !== -Math.sign(st.vxOut))
    vx = st.vxOut - Math.sign(st.vxOut) * Math.min(Math.abs(st.vxOut - vx), DECEL * dt);
  st.vxOut = vx;
  return [round(vx), round(clamp(vy, r.lateral_range)), round(clamp(wz, r.yaw_rate_range))];
}

function fmt(c) {
  return c.every(v => v === 0) ? 'stand still' : `fwd ${c[0].toFixed(2)} · side ${c[1].toFixed(2)} m/s · turn ${c[2].toFixed(2)} rad/s`;
}

function tick(now) {
  const dt = Math.min(0.1, (now - st.last) / 1000);
  st.last = now;
  const pad = gamepad();
  if (!running()) st.vxOut = 0;   // the next start begins standing, not easing off an old speed
  const c = command(dt);
  const changed = !st.sent || c.some((v, k) => v !== st.sent[k]);
  if (running() && (changed ? now - st.sentAt > 100 : now - st.sentAt > 250)) {
    st.sent = c; st.sentAt = now;
    call('policy/cmd', {vx: c[0], vy: c[1], wz: c[2]}, {confirm: false});
  }
  $('dCmd').textContent = fmt(c);
  const fr = ranges().forward_range;
  $('dSlider').value = st.cruise;
  $('dSliderV').textContent = st.cruise.toFixed(2) + ' m/s';
  for (const b of $('dPresets').children) {
    const v = +b.dataset.v;
    b.classList.toggle('sel', v === round(st.cruise));
    b.disabled = v < fr[0] - 1e-9 || v > fr[1] + 1e-9;
    b.title = b.disabled ? 'beyond what this policy was trained for' : '';
  }
  $('dPadNote').textContent = pad ? `Gamepad: ${pad.slice(0, 40)}` : st.stick ? 'joystick held' : '';
  requestAnimationFrame(tick);
}

// ---- vitals: what matters while walking ----
const lvl = (v, warn, bad) => !has(v) ? '' : v >= bad ? 'bad' : v >= warn ? 'warn' : '';
function vitals(S) {
  const m = S.imu, w = S.walker, p = S.pi || {}, lim = S.limits || {servo_temp: 60, pi_temp: 75};
  const tilt = m ? Math.hypot(m.lean_forward, m.lean_right) : null;
  if (m) {
    const roll = Math.max(-60, Math.min(60, m.lean_right)), pitch = Math.max(-45, Math.min(45, m.lean_forward));
    $('dHorizon').innerHTML = `<g transform="translate(55 35) rotate(${-roll}) translate(0 ${-pitch * 0.8})">
      <rect x="-120" y="0" width="240" height="120" fill="var(--band)"/><line x1="-120" x2="120" y1="0" y2="0" stroke="var(--accent)" stroke-width="2"/></g>
      <line x1="35" x2="75" y1="35" y2="35" stroke="var(--text)" stroke-width="3"/>`;
  }
  const b = battery(S);
  const temps = Object.values(S.joints).filter(j => has(j.temp));
  const hot = temps.length ? temps.reduce((a, j) => (j.temp > a.temp ? j : a)) : null;
  const busMax = (S.buses || []).map(x => x.pass_max).filter(has);
  const linkCls = !has(w.link_max) ? '' : w.link_max > w.link_limits[0] ? 'bad' : w.link_max > 0.8 ? 'warn' : '';
  $('dVitals').innerHTML = `
    <div>Tilt<b class="${lvl(tilt, 10, 25)}">${m ? n1(tilt) + '°' : '?'}</b>${m ? `fwd ${n1(m.lean_forward)} · right ${n1(m.lean_right)}` : esc(S.imu_error || 'no IMU')}</div>
    <div>Battery<b class="${b ? b.level : ''}">${b ? b.volts.toFixed(2) + ' V' : '?'}</b>${b ? '~' + b.pct + '%' : ''}</div>
    <div>Hottest servo<b class="${hot ? lvl(hot.temp, lim.servo_temp - 10, lim.servo_temp) : ''}">${hot ? hot.temp + '°C' : '?'}</b>${hot ? 'servo ' + hot.id : ''}</div>
    <div>Pi<b class="${lvl(p.temp, lim.pi_temp - 10, lim.pi_temp)}">${has(p.temp) ? p.temp.toFixed(0) + '°C' : '?'}</b>CPU ${has(p.cpu) ? p.cpu.toFixed(0) + '%' : '?'}${p.undervolt || p.throttled ? ' <span class="bad">power low</span>' : ''}</div>
    <div>Wi-Fi<b class="${!has(p.wifi_dbm) ? '' : p.wifi_dbm < -75 ? 'bad' : p.wifi_dbm < -65 ? 'warn' : ''}">${has(p.wifi_dbm) ? p.wifi_dbm.toFixed(0) + ' dBm' : '?'}</b>link gap ${has(w.link_max) ? `<span class="${linkCls}">${n1(w.link_max)} s</span>` : '-'}</div>
    <div>Timing<b class="${lvl(w.tick_max, 15, 20)}">${has(w.tick_ms) ? n1(w.tick_ms) + ' ms' : '-'}</b>step (max ${n1(w.tick_max)}) · bus ${busMax.length ? n1(Math.max(...busMax)) + ' ms' : '?'}</div>`;
}

// ---- state ----
let camOn = false;
function camera(on) {
  camOn = on;
  const stab = $('dCamView').value === 'stabilized';
  $('dCam').hidden = !on;
  $('dCam').src = on ? (stab ? '/camera-stabilized.mjpg' : '/camera.mjpg') + '?' + Date.now() : '';
  $('dCamBtn').textContent = on ? 'Hide' : 'Show';
}

onState((S, full) => {
  const w = S.walker;
  vitals(S);
  if (full) {
    const sel = $('dPolicy');
    const gs = $('dGetup'), gp = w.getup_policies || [];
    const opt = p => `<option value="${esc(p)}">${esc((w.labels || {})[p] ? `${w.labels[p]} · ${p}` : p)}</option>`;
    if (gs.options.length !== gp.length) gs.innerHTML = gp.map(opt).join('') || '<option value="">no getup policy</option>';
    if (sel.options.length !== w.policies.length) {
      sel.innerHTML = w.policies.map(opt).join('');
      sel.value = w.policy || w.default || w.policies[w.policies.length - 1];
    }
    const c = S.camera;
    $('dCamView').disabled = !c.stabilization_configured;
    if ($('dCamView').disabled && $('dCamView').value !== 'raw') { $('dCamView').value = 'raw'; if (camOn) camera(true); }
    $('dCamNote').textContent = !c.available ? 'Camera unavailable: ' + c.error : c.error && camOn ? 'Camera error: ' + c.error
      : c.stabilization_error ? 'Stabilized view unavailable: ' + c.stabilization_error : camOn ? '' : 'Off until shown (the stream costs CPU).';
  }
  if (w.running) $('dPolicy').value = w.policy;
  $('dPolicy').disabled = w.running;
  $('dStart').disabled = w.running || !w.policies.length || S.mode !== 'idle';
  $('dGetupBtn').disabled = w.running || !(w.getup_policies || []).length || S.mode !== 'idle';
  for (const id of ['dBoot', 'dHome']) $(id).disabled = S.mode !== 'idle';
  $('dMs').value = w.move_ms;
  const r = w.ranges;
  $('dNote').textContent = r ? `this policy: ${r.forward_range[0]} to ${r.forward_range[1]} m/s, side ±${r.lateral_range[1]}, turn ±${r.yaw_rate_range[1]}` : '';
  if (w.running && w.command) {
    const got = w.command.map(round), sent = st.sent || [0, 0, 0];
    const capped = sent.some((v, k) => Math.abs(v - got[k]) > 0.005);
    $('dSent').innerHTML = `robot: ${fmt(got)}${w.command_source && w.command_source !== 'page' ? ` (from ${esc(w.command_source)})` : ''}`
      + (capped && w.command_source === 'page' ? ' <span class="warn">capped to what the policy was trained for</span>' : '')
      + (w.steer_owner ? ` <span class="warn">${esc(w.steer_owner)} is steering</span>` : '');
  } else {
    $('dSent').textContent = w.status && w.status !== 'off' ? w.status : 'not walking: Start sends the command';
  }
  const linkCls = w.link_max > w.link_limits[0] ? 'bad' : w.link_max > 0.8 ? 'warn' : 'ok';
  $('dStats').innerHTML = has(w.ticks)
    ? `${n1(w.seconds, 0)} s · step ${n1(w.tick_ms)} ms (max ${n1(w.tick_max)}) of 20 · late ${w.overruns} times · link gap max
       <b class="${linkCls}">${n1(w.link_max)} s</b>${w.log ? ' · log ' + esc(w.log) : ''}`
    : 'No walk yet.';
});

