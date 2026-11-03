// Setup: poses, calibration (zero, directions, limits, the servos' stored limits, the IMU) and every joint.
// Calibration buttons stay locked until "Unlock calibration" is ticked, so nothing gets pressed by accident.
import {$, R, call, esc, has, n1, onState} from './core.js';

const LABEL = {hip_roll: 'Hip roll', hip_pitch: 'Hip pitch', knee: 'Knee', ankle: 'Ankle'};
const label = name => `${name.startsWith('left') ? 'L' : 'R'} ${LABEL[name.replace(/^(left|right)_/, '')]}`;
let selected = null, lastInput = 0;

export function build(root) {
  root.innerHTML = `
  <div class="card">
    <div class="row"><b>Poses</b>
      <button id="uBoot" title="Straight legs over 2 s, then the home crouch over 2.5 s">Boot: straight → home</button>
      <button id="uStraight">Straight</button><button id="uHome">Home</button>
      <button id="uRelax">Relax all</button><button id="uScan">Rescan servos</button>
      <label class="sub">Move time <input id="uMs" type="number" min="0" max="5000" step="50" value="300"> ms</label></div>
  </div>
  <div class="card" style="margin-top:12px">
    <div class="row"><b>Joints</b><span class="sub">Tap a row to move it, set its limits or flip it.</span></div>
    <div class="tablewrap"><table><thead><tr><th>Joint</th><th>Servo</th><th>Angle</th><th>Target</th><th>Torque</th>
      <th>Range</th><th>Temp</th></tr></thead><tbody id="uJoints"></tbody></table></div>
    <div id="uDetail" style="margin-top:10px"></div>
  </div>
  <div class="card" style="margin-top:12px">
    <div class="row"><b>Calibration</b><span class="sub" id="uCalState"></span>
      <label style="margin-left:auto"><input type="checkbox" id="uUnlock"> Unlock calibration</label></div>
    <fieldset id="uCal" disabled><ol class="steps">
      <li><b>Zero.</b> Relax all, hold both legs perfectly straight (a jig or a flat wall helps), then
        <button class="small" data-act="cal/zero">Set zero: legs straight</button>
        <span class="sub">Every angle here is the simulator's: legs straight = 0°.</span></li>
      <li><b>Directions.</b> On each joint, torque on and +5°: it must move the way its "+ =" says. If not, Flip it.</li>
      <li><b>Limits.</b> Relax a joint, move it by hand to where it touches something, back off a few degrees, Lower here / Upper here.
        Until both are set, a joint uses the simulator's range.</li>
      <li><b>Store in the servos.</b> <button class="small" data-act="cal/write_limits">Write limits to servos</button>
        <button class="small" id="uResetLimits">Reset servo limits</button></li>
      <li><b>IMU.</b> In its card on <a href="#status">Status</a>, next to its readings: axes, bus, zero gyro, level.</li>
    </ol></fieldset>
  </div>`;
  $('uBoot').onclick = () => call('boot');
  $('uStraight').onclick = () => call('pose', {name: 'straight'});
  $('uHome').onclick = () => call('pose', {name: 'home'});
  $('uRelax').onclick = () => call('relax');
  $('uScan').onclick = () => call('scan');
  $('uUnlock').onchange = e => { $('uCal').disabled = !e.target.checked; render(R.S); };
  root.querySelector('#uCal').addEventListener('click', e => { const a = e.target.dataset.act; if (a) call(a); });
  $('uResetLimits').onclick = () => { if (confirm("Reset every servo's stored limits to the full 0-1000?")) call('cal/write_limits', {reset: true}, {confirm: false}); };
  $('uJoints').onclick = e => { const tr = e.target.closest('tr'); if (tr) { selected = selected === tr.dataset.j ? null : tr.dataset.j; render(R.S); } };
  $('uDetail').addEventListener('click', detailClick);
  $('uDetail').addEventListener('input', e => { if (e.target.id === 'uSlider') { lastInput = Date.now(); send(selected, +e.target.value); } });
}

// one move at a time to the servo: a dragged slider sends its latest value, not every step
let pending = null, busy = false;
function send(name, deg) {
  pending = {name, deg};
  if (busy) return;
  busy = true;
  (async () => {
    while (pending) {
      const p = pending; pending = null;
      await call('move', {joint: p.name, deg: p.deg, ms: +$('uMs').value || 300});
    }
    busy = false;
  })();
}

function detailClick(e) {
  const b = e.target.closest('button');
  if (!b || !selected) return;
  const j = R.S.joints[selected], d = b.dataset;
  if (d.nudge) {
    const from = j.torque && has(j.target_deg) ? j.target_deg : j.deg;
    if (!has(from)) return;
    const deg = Math.max(j.limits[0], Math.min(j.limits[1], from + +d.nudge));
    $('uSlider').value = deg; lastInput = Date.now(); send(selected, deg);
  } else if (d.act) call(d.act, {joint: selected, ...(d.end ? {end: d.end} : {})});
  else if (d.torque) call(j.torque ? 'relax' : 'hold', {joint: selected});
}

function rangeSvg(j, W) {
  const lo = Math.min(j.sim[0], j.limits[0]) - 5, hi = Math.max(j.sim[1], j.limits[1]) + 5;
  const x = d => (6 + (d - lo) / (hi - lo) * (W - 12)).toFixed(1);
  let s = `<rect x="${x(j.sim[0])}" y="14" width="${x(j.sim[1]) - x(j.sim[0])}" height="12" rx="3" fill="var(--band)"><title>simulator range</title></rect>`;
  if (j.limits_set) s += `<rect x="${x(j.limits[0])}" y="12" width="${x(j.limits[1]) - x(j.limits[0])}" height="16" rx="3" fill="none" stroke="var(--accent)" stroke-width="1.5"/>`;
  s += `<line x1="${x(0)}" x2="${x(0)}" y1="8" y2="32" stroke="var(--muted)" stroke-dasharray="2 2"/>`;
  s += `<text x="${x(j.home)}" y="38" font-size="9" font-weight="700" text-anchor="middle" fill="var(--muted)">H</text>`;
  if (j.torque && has(j.target_deg)) s += `<line x1="${x(j.target_deg)}" x2="${x(j.target_deg)}" y1="10" y2="30" stroke="var(--accent)" stroke-width="2"/>`;
  if (has(j.deg)) s += `<circle cx="${x(j.deg)}" cy="20" r="5" fill="var(--text)"/>`;
  return s;
}

function detail(S) {
  const el = $('uDetail');
  if (!selected) { el.innerHTML = ''; el.dataset.j = ''; return; }
  const j = S.joints[selected], cal = !$('uCal').disabled;
  if (el.dataset.j !== selected) {
    el.dataset.j = selected;
    el.innerHTML = `<div class="row"><b>${label(selected)}</b><span class="sub">servo ${j.id} · + = ${esc(j.meaning)}</span>
        <button class="small" data-torque="1" id="uTq">Torque</button></div>
      <svg class="range" id="uRange" viewBox="0 0 300 40"></svg>
      <input type="range" id="uSlider" step="0.5" style="width:100%">
      <div class="row">${[-5, -1, 1, 5].map(d => `<button data-nudge="${d}">${d > 0 ? '+' : '−'}${Math.abs(d)}°</button>`).join('')}
        <button data-act="relax">Relax</button></div>
      <div class="row"><button class="small cal" data-act="cal/limit" data-end="lo">Lower here</button>
        <button class="small cal" data-act="cal/limit" data-end="hi">Upper here</button>
        <button class="small cal" data-act="cal/clear_limits">Clear limits</button>
        <button class="small cal" data-act="cal/flip">Flip</button></div>
      <div class="sub" id="uLim"></div>`;
  }
  el.querySelectorAll('button.cal').forEach(b => { b.disabled = !cal; });
  $('uTq').textContent = j.torque ? 'Torque on' : j.torque === false ? 'Torque off' : 'Torque ?';
  $('uTq').classList.toggle('on', !!j.torque);
  const svg = $('uRange'), W = Math.max(100, Math.round(svg.clientWidth || 300));
  svg.setAttribute('viewBox', `0 0 ${W} 40`);
  svg.innerHTML = rangeSvg(j, W);
  const sl = $('uSlider');
  sl.min = j.limits[0]; sl.max = j.limits[1]; sl.disabled = !j.found || j.locked || S.mode !== 'idle';
  if (Date.now() - lastInput > 1500) {  // relaxed: it follows the joint, so touching it never makes it jump
    const v = j.torque && has(j.target_deg) ? j.target_deg : j.deg;
    if (has(v)) sl.value = v;
  }
  const narrow = j.limits_set && (j.limits[0] > j.sim[0] + 1 || j.limits[1] < j.sim[1] - 1);
  const mem = j.info && j.info.angle_limits ? ` · in servo memory: ${j.info.angle_limits[0]}–${j.info.angle_limits[1]} counts` : '';
  $('uLim').innerHTML = `Range ${n1(j.limits[0])}° to ${n1(j.limits[1])}° ` + (j.limits_set ? '(set on the robot)' : '(simulator range: limits not set)')
    + mem + ` · counts ${has(j.pos) ? j.pos : '-'} · ${j.vin ? j.vin.toFixed(2) + ' V' : ''}`
    + (narrow ? `<div class="warn">Narrower than the simulator's ${j.sim[0]}° to ${j.sim[1]}°: the policy will be clipped here.</div>` : '');
}

function render(S) {
  if (!S) return;
  $('uJoints').innerHTML = Object.entries(S.joints).map(([n, j]) => `<tr class="joint${n === selected ? ' sel' : ''}${!j.found ? ' off' : ''}${j.locked ? ' locked' : ''}" data-j="${n}">
    <td>${label(n)}</td><td>${j.id}</td><td>${j.found ? n1(j.deg) + '°' : 'not found'}</td>
    <td>${j.torque && has(j.target_deg) ? n1(j.target_deg) + '°' : '-'}</td><td>${j.torque ? 'on' : j.torque === false ? 'off' : '?'}</td>
    <td>${n1(j.limits[0], 0)}…${n1(j.limits[1], 0)}°${j.limits_set ? '' : ' (sim)'}</td>
    <td class="${j.temp >= S.limits.servo_temp ? 'bad' : j.temp >= S.limits.servo_temp - 10 ? 'warn' : ''}">${has(j.temp) ? j.temp + '°C' : '?'}</td></tr>`).join('');
  detail(S);
  const idle = S.mode === 'idle';
  for (const id of ['uBoot', 'uStraight', 'uHome']) $(id).disabled = !idle;
  $('uCalState').textContent = S.zero_set ? 'zero set' : 'not calibrated';
}

onState(S => render(S));
