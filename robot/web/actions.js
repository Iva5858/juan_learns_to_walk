// Actions: gestures, dance, the victory spin, playful mode, greeting faces, voice, sounds, speech. Any action
// declared in robot/actions.py (groups poses / personality / voice) that takes no required parameter and
// isn't one of these gets its own button under "More" by itself: a new behavior needs no page change.
import {$, R, call, esc, onState} from './core.js';

const SHOWN = new Set(['gesture', 'gesture/stop', 'playful', 'faces', 'listen', 'voice', 'sound', 'say', 'volume', 'spin',
                       'boot', 'pose']);
let built = false;

export function build(root) {
  root.innerHTML = `
  <div class="grid">
    <div class="card"><h3>Gestures <span class="sub" id="aStatus"></span></h3>
      <div class="sub">From Home, standing still; each one checked in the simulator.</div>
      <div class="row" id="aGestures"></div>
      <div class="row"><label class="sub">Dance BPM <input id="aBpm" type="number" min="70" max="120" step="5" value="100"></label>
        <button class="danger small" id="aStopGesture">Stop gesture</button></div>
      <div class="row"><button id="aSpin" title="A full turn in place while walking, then a bounce">Victory spin</button>
        <span class="sub">while the walking policy runs</span></div></div>
    <div class="card"><h3>Modes</h3>
      <div class="row"><button id="aPlayful">Playful mode</button><span class="sub">idle moves and chirps every 10–25 s at Home</span></div>
      <div class="row"><button id="aFaces">Greet faces</button><span class="sub">a bounce when someone appears</span></div>
      <div class="row"><button id="aListen">Voice commands</button><span class="sub">"Juan, walk" / "stop" / "dance"; steers a running policy</span></div>
      <div class="sub" id="aModeNote"></div></div>
    <div class="card"><h3>Voice</h3>
      <div class="row" id="aEmotions"></div><div class="row" id="aFiles"></div>
      <div class="row"><input id="aSay" type="text" placeholder="Say something (needs espeak-ng)" style="flex:1;min-width:0">
        <button class="small" id="aSayBtn">Say</button></div>
      <div class="row"><label class="sub">Volume <input id="aVol" type="range" min="0" max="1" step="0.05"></label>
        <span class="sub" id="aVoiceNote"></span></div></div>
    <div class="card" id="aMoreCard" hidden><h3>More</h3><div class="row" id="aMore"></div></div>
  </div>`;
  $('aGestures').onclick = e => {
    const g = e.target.dataset.g;
    if (g) call('gesture', g === 'dance' ? {name: g, bpm: +$('aBpm').value} : {name: g});
  };
  $('aStopGesture').onclick = () => call('gesture/stop');
  $('aSpin').onclick = () => call('spin');
  $('aPlayful').onclick = () => call('playful', {on: !R.S.personality.playful});
  $('aFaces').onclick = () => call('faces', {on: !R.S.personality.greet_faces});
  $('aListen').onclick = () => call('listen', {on: !R.S.personality.listening});
  $('aEmotions').onclick = e => { if (e.target.dataset.e) call('sound', {emotion: e.target.dataset.e}); };
  $('aFiles').onclick = e => { if (e.target.dataset.f) call('sound', {file: e.target.dataset.f}); };
  const say = () => { const t = $('aSay').value.trim(); if (t) call('say', {text: t}); };
  $('aSayBtn').onclick = say;
  $('aSay').onkeydown = e => { if (e.key === 'Enter') say(); };
  $('aVol').onchange = e => call('volume', {v: +e.target.value});
  $('aMore').onclick = e => { if (e.target.dataset.a) call(e.target.dataset.a); };
}

const toggle = (id, text, on) => { $(id).textContent = `${text}: ${on ? 'on' : 'off'}`; $(id).classList.toggle('on', on); };

onState((S, full) => {
  const P = S.personality, V = P.voice;
  if (!built) {
    $('aGestures').innerHTML = P.gestures.map(g => `<button class="small" title="${esc(g.description)}" data-g="${esc(g.name)}">${esc(g.label)}</button>`).join('');
    $('aEmotions').innerHTML = V.emotions.map(e => `<button class="small" data-e="${esc(e)}">${esc(e)}</button>`).join('');
    $('aVol').value = V.volume;
    const more = Object.values(R.catalog).filter(a => ['poses', 'personality', 'voice'].includes(a.group) && !SHOWN.has(a.name)
      && Object.values(a.params).every(p => 'default' in p));
    $('aMore').innerHTML = more.map(a => `<button data-a="${esc(a.name)}" title="${esc(a.doc)}">${esc(a.label)}</button>`).join('');
    $('aMoreCard').hidden = !more.length;
    built = true;
  }
  if (full && $('aFiles').dataset.n !== String(V.files.length)) {
    $('aFiles').innerHTML = V.files.map(f => `<button class="small" data-f="${esc(f)}">${esc(f.replace('.wav', ''))}</button>`).join('');
    $('aFiles').dataset.n = V.files.length;
  }
  $('aStatus').textContent = S.gesture ? 'playing ' + S.gesture : '';
  const idle = S.mode === 'idle';
  for (const b of $('aGestures').children) b.disabled = !idle;
  $('aSpin').disabled = !S.walker.running || P.spinning;
  toggle('aPlayful', 'Playful mode', P.playful);
  toggle('aFaces', 'Greet faces', P.greet_faces);
  toggle('aListen', 'Voice commands', P.listening);
  const notes = [P.faces_error && 'camera/faces: ' + P.faces_error, P.listener_error && 'voice: ' + P.listener_error,
                 P.faces && P.faces.faces ? `sees ${P.faces.faces} face(s)` : '', P.spinning ? 'spinning' : ''].filter(Boolean);
  $('aModeNote').innerHTML = notes.map(esc).join(' · ');
  $('aVoiceNote').innerHTML = !V.backend ? '<span class="bad">no audio output found</span>'
    : V.error ? `<span class="warn">${esc(V.error)}</span>` : V.playing ? 'playing ' + esc(V.playing) : esc(V.backend + (V.tts ? ' + ' + V.tts : ''));
});
