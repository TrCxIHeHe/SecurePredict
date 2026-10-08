"""
SecurePredict Control Room (3D digital twin) - pure HTML/three.js builder.

Receives the payload computed by src/twin/timeline.py (every status, probability, SHAP value
and alert already produced by the frozen models) and plays it back. This file contains no
model logic and no hard-coded outputs; legend thresholds come from the payload.

Illustrative twin: telemetry is simulated; network flows are real benchmark flows assigned
to machines by the simulator.
"""

from __future__ import annotations

import json

THREE_CDN = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"


def build_twin_html(payload: dict, height: int = 780) -> str:
    data = json.dumps(payload).replace("</", "<\\/")
    return TEMPLATE.replace("__PAYLOAD__", data).replace("__CDN__", THREE_CDN).replace("__H__", str(height))


TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8">
<style>
*{box-sizing:border-box}
html,body{margin:0;background:#070b14;color:#d9e3f5;font:13px/1.45 system-ui,"Segoe UI",Arial,sans-serif}
#app{position:relative;width:100%;height:__H__px;overflow:hidden;background:radial-gradient(circle at 50% 30%,#0e1730,#070b14 70%)}
.panel{position:absolute;background:rgba(9,15,29,.74);backdrop-filter:blur(9px);-webkit-backdrop-filter:blur(9px);
       border:1px solid rgba(130,160,215,.18);border-radius:12px;box-shadow:0 8px 28px rgba(0,0,0,.35)}
#top{left:12px;right:12px;top:10px;height:48px;display:flex;align-items:center;gap:10px;padding:0 12px}
.brand{font-weight:700;letter-spacing:.3px;white-space:nowrap}.brand b{color:#58b6ff}.brand span{color:#7f93b3;font-weight:500;margin-left:6px}
button{background:#14223d;color:#d9e3f5;border:1px solid #2a3d63;border-radius:8px;padding:5px 10px;cursor:pointer;font:inherit}
button:hover{background:#1c3157}button.on{background:#1f4f8f;border-color:#3b78c9}
#scrub{flex:1;accent-color:#58b6ff;min-width:80px}
#clock{font-variant-numeric:tabular-nums;color:#9fb3d6;white-space:nowrap}
#gpill{padding:4px 10px;border-radius:999px;font-weight:700;font-size:12px;white-space:nowrap}
#fleet{left:12px;top:68px;width:244px;bottom:156px;overflow:auto;padding:8px}
#detail{right:12px;top:68px;width:300px;bottom:156px;overflow:auto;padding:12px}
#bottom{position:absolute;left:12px;right:12px;bottom:10px;height:136px;display:flex;gap:12px}
#heatwrap,#log{position:relative;border-radius:12px}
#heatwrap{flex:1.5;padding:8px 10px}#log{flex:1;overflow:auto;padding:8px 10px}
.card{border:1px solid #1f2e4d;border-radius:10px;padding:7px 9px;margin-bottom:7px;cursor:pointer;background:rgba(18,28,50,.6)}
.card:hover{border-color:#3b5588}.card.sel{border-color:#58b6ff;background:rgba(30,60,110,.45)}
.ch{display:flex;align-items:center;gap:7px}.ch b{flex:1}.dot{width:10px;height:10px;border-radius:50%}
.lv{font-size:10.5px;font-weight:700;padding:1px 7px;border-radius:999px;color:#06101f}
.cm{display:flex;justify-content:space-between;color:#8fa4c8;font-size:11.5px}
canvas.spark{width:100%;height:34px;display:block;margin:3px 0}
.sec{margin:12px 0 5px;color:#7f93b3;font-size:11px;text-transform:uppercase;letter-spacing:.8px}
.big{font-size:30px;font-weight:700;font-variant-numeric:tabular-nums;line-height:1}
.barwrap{position:relative;height:9px;background:#14213a;border-radius:6px;overflow:visible;margin:6px 0}
.bar{height:100%;border-radius:6px}.mark{position:absolute;top:-4px;width:2px;height:17px;background:#aab8d4}
.mark i{position:absolute;top:18px;left:-14px;font-style:normal;font-size:9.5px;color:#8fa4c8;white-space:nowrap}
.row{display:flex;align-items:center;gap:8px;margin:4px 0;font-size:12px}.row .k{width:104px;color:#9fb3d6}.row .v{width:76px;text-align:right;font-variant-numeric:tabular-nums}
.row .b{flex:1;height:6px;background:#14213a;border-radius:4px}.row .b div{height:100%;border-radius:4px;background:#58b6ff}
.why{color:#b9c7e3;font-size:12px;background:rgba(20,34,61,.7);border-left:3px solid #58b6ff;padding:6px 8px;border-radius:6px}
.verdict{padding:5px 8px;border-radius:7px;font-weight:600;font-size:12px;margin-top:6px}
.ev{display:flex;gap:7px;padding:3px 2px;border-bottom:1px solid rgba(60,80,120,.25);cursor:pointer;font-size:12px}
.ev:hover{background:rgba(40,70,120,.25)}.ev .t{color:#7f93b3;width:52px;flex:none;font-variant-numeric:tabular-nums}
.ev .c{width:9px;height:9px;border-radius:50%;margin-top:4px;flex:none}
#heat{width:100%;height:calc(100% - 20px);display:block;cursor:pointer}
#heatinfo{height:18px;font-size:11px;color:#8fa4c8}
#tip{position:absolute;display:none;pointer-events:none;background:rgba(9,15,29,.95);border:1px solid #3a4d70;border-radius:7px;padding:5px 8px;font-size:12px;white-space:pre;z-index:5}
.hint{color:#6f84a8;font-size:11px;margin-top:8px}
@media (max-width:980px){#fleet{display:none}#detail{width:250px}}
@media (max-width:700px){#detail{display:none}}
</style></head>
<body><div id="app">
 <div id="top" class="panel">
   <div class="brand">Secure<b>Predict</b><span>Control Room</span></div>
   <button id="play">Pause</button>
   <span id="speeds"><button data-s="0.5">0.5x</button><button data-s="1" class="on">1x</button><button data-s="2">2x</button><button data-s="4">4x</button></span>
   <input id="scrub" type="range" min="0" value="0" step="1">
   <div id="clock"></div><div id="gpill"></div><button id="reset">Overview</button>
 </div>
 <div id="fleet" class="panel"></div>
 <div id="detail" class="panel"></div>
 <div id="bottom">
   <div id="heatwrap" class="panel"><div id="heatinfo"></div><canvas id="heat"></canvas></div>
   <div id="log" class="panel"></div>
 </div>
 <div id="tip"></div>
</div>
<script src="__CDN__"></script>
<script>
const D = __PAYLOAD__;
(function(){
const T = D.frames.length, M = D.machines.length, LV = D.levels, NET = D.netAvailable, TH = D.thresholds;
const COL = {"OK":0x2ecc71,"WARNING":0xf5a623,"NETWORK ALERT":0xa66bff,"ALARM":0xff4d4f,"CRITICAL":0xff1f6b};
const css = n => '#' + n.toString(16).padStart(6,'0');
const $ = id => document.getElementById(id);
const app = $('app');
let tick = 0, playing = true, speed = 1, sel = 0, acc = 0, lastInteract = -99;

// ---------- three.js scene ----------
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x070b14);
scene.fog = new THREE.FogExp2(0x070b14, 0.016);
const camera = new THREE.PerspectiveCamera(48, 1, 0.1, 300);
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
app.insertBefore(renderer.domElement, app.firstChild);
renderer.domElement.style.cssText = 'position:absolute;left:0;top:0;width:100%;height:100%';

scene.add(new THREE.HemisphereLight(0x7088b8, 0x0a0f1c, 0.95));
const sun = new THREE.DirectionalLight(0xffffff, 0.7); sun.position.set(12, 22, 10); scene.add(sun);

const floor = new THREE.Mesh(new THREE.PlaneGeometry(90, 70), new THREE.MeshStandardMaterial({color:0x0d1424, metalness:.35, roughness:.75}));
floor.rotation.x = -Math.PI/2; scene.add(floor);
const grid = new THREE.GridHelper(90, 90, 0x1d2c48, 0x131d33); grid.position.y = 0.02; scene.add(grid);
(function zone(){ const pts=[[-17,-7],[17,-7],[17,16],[-17,16]].map(p=>new THREE.Vector3(p[0],0.04,p[1]));
  const l=new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({color:0x2c5a8f,transparent:true,opacity:.7})); scene.add(l); })();

const dust = (function(){ const n=260, a=new Float32Array(n*3); for(let i=0;i<n;i++){a[i*3]=(Math.random()-.5)*60;a[i*3+1]=Math.random()*14+.5;a[i*3+2]=(Math.random()-.5)*50;}
  const g=new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(a,3));
  const p=new THREE.Points(g,new THREE.PointsMaterial({color:0x5b7fb8,size:.09,transparent:true,opacity:.55})); scene.add(p); return p; })();

function label(){ const c=document.createElement('canvas'); c.width=360; c.height=110;
  const tex=new THREE.CanvasTexture(c); const s=new THREE.Sprite(new THREE.SpriteMaterial({map:tex,transparent:true,depthWrite:false}));
  s.scale.set(4.2,1.28,1); s.userData={c:c,tex:tex,key:''}; return s; }
function paintLabel(s, a, b, color){ const key=a+'|'+b+'|'+color; if(s.userData.key===key) return; s.userData.key=key;
  const c=s.userData.c, g=c.getContext('2d'); g.clearRect(0,0,360,110);
  g.fillStyle='rgba(9,15,29,.88)'; g.fillRect(0,0,360,110); g.strokeStyle=color; g.lineWidth=5; g.strokeRect(2.5,2.5,355,105);
  g.fillStyle='#e8eef9'; g.font='bold 34px Arial'; g.fillText(a,16,46); g.fillStyle=color; g.font='bold 27px Arial'; g.fillText(b,16,88);
  s.userData.tex.needsUpdate=true; }

// ---------- layout ----------
const posOf = i => { const c=i%3, r=Math.floor(i/3); return new THREE.Vector3((c-1)*10.5, 0, r*10.5 + 1.5); };
const SW = new THREE.Vector3(0,0,-11);
const machines = [], pickables = [], links = [];

function mat(c, m, r){ return new THREE.MeshStandardMaterial({color:c, metalness:m, roughness:r}); }
function buildMachine(i){
  const g = new THREE.Group(); g.position.copy(posOf(i));
  const o = {g:g, col:new THREE.Color(COL.OK), goal:new THREE.Color(COL.OK), lastLevel:'OK'};
  const base = new THREE.Mesh(new THREE.BoxGeometry(5.4,0.3,4.4), mat(0x16213a,.4,.7)); base.position.y=.15; g.add(base);
  o.bodyMat = new THREE.MeshStandardMaterial({color:0x2b3d60, metalness:.6, roughness:.38, emissive:0x000000, emissiveIntensity:.6});
  const body = new THREE.Mesh(new THREE.BoxGeometry(3.7,2.3,2.9), o.bodyMat); body.position.y=1.45; g.add(body); body.userData.idx=i; pickables.push(body);
  const hood = new THREE.Mesh(new THREE.BoxGeometry(3.1,.5,2.3), mat(0x3a4f78,.5,.4)); hood.position.y=2.85; g.add(hood);
  o.screenMat = new THREE.MeshStandardMaterial({color:0x111111, emissive:0x2ecc71, emissiveIntensity:.9});
  const screen = new THREE.Mesh(new THREE.BoxGeometry(1.1,.7,.08), o.screenMat); screen.position.set(1.0,1.8,1.47); g.add(screen);
  const house = new THREE.Mesh(new THREE.CylinderGeometry(.55,.55,.9,24), mat(0x5d6f94,.7,.3)); house.position.y=3.5; g.add(house);
  o.spindle = new THREE.Mesh(new THREE.CylinderGeometry(.24,.24,1.3,16), mat(0xd5deec,.9,.2)); o.spindle.position.y=2.55; g.add(o.spindle);
  const tool = new THREE.Mesh(new THREE.ConeGeometry(.2,.6,12), mat(0xa3b0c6,.9,.2)); tool.rotation.x=Math.PI; tool.position.y=1.75; o.spindle.add(tool); tool.position.y=-.9;
  // andon stack light
  const pole = new THREE.Mesh(new THREE.CylinderGeometry(.06,.06,2.6,8), mat(0x8896b0,.8,.3)); pole.position.set(-2.2,1.6,1.6); g.add(pole);
  o.lamps = [[0xff3b3b,3.0],[0xffb02e,2.55],[0x34e07a,2.1]].map(function(a){ const m2=new THREE.MeshStandardMaterial({color:0x222222,emissive:a[0],emissiveIntensity:.05});
    const l=new THREE.Mesh(new THREE.CylinderGeometry(.22,.22,.36,14), m2); l.position.set(-2.2,a[1]+.35,1.6); g.add(l); return m2; });
  // tool-wear tower
  o.wear = []; for(let k=0;k<10;k++){ const m3=new THREE.MeshStandardMaterial({color:0x222a3a,emissive:0xffd166,emissiveIntensity:0});
    const b=new THREE.Mesh(new THREE.BoxGeometry(.34,.17,.34), m3); b.position.set(2.3,.55+k*.22,-1.6); g.add(b); o.wear.push(m3); }
  // floor ring, shield, beacon, light
  o.ringMat = new THREE.MeshBasicMaterial({color:COL.OK, transparent:true, opacity:.75, side:THREE.DoubleSide});
  o.ring = new THREE.Mesh(new THREE.RingGeometry(3.3,3.55,56), o.ringMat); o.ring.rotation.x=-Math.PI/2; o.ring.position.y=.05; g.add(o.ring);
  o.shieldMat = new THREE.MeshBasicMaterial({color:0xff4d4f, wireframe:true, transparent:true, opacity:.35});
  o.shield = new THREE.Mesh(new THREE.SphereGeometry(3.9,26,16), o.shieldMat); o.shield.position.y=1.8; o.shield.visible=false; g.add(o.shield);
  o.beaconMat = new THREE.MeshBasicMaterial({color:COL.CRITICAL, transparent:true, opacity:.28, blending:THREE.AdditiveBlending, depthWrite:false, side:THREE.DoubleSide});
  o.beacon = new THREE.Mesh(new THREE.CylinderGeometry(.55,1.3,16,20,1,true), o.beaconMat); o.beacon.position.y=9; o.beacon.visible=false; g.add(o.beacon);
  o.light = new THREE.PointLight(COL.OK, .55, 13); o.light.position.y=4; g.add(o.light);
  o.label = label(); o.label.position.y=5.6; g.add(o.label);
  scene.add(g); machines.push(o);
}
for(let i=0;i<M;i++) buildMachine(i);

// network core switch
const swMat = new THREE.MeshStandardMaterial({color:0x1d3a63, emissive:0x2a6bd6, emissiveIntensity:.5, metalness:.5, roughness:.4});
const sw = new THREE.Mesh(new THREE.BoxGeometry(7,1.3,1.8), swMat); sw.position.set(SW.x,.8,SW.z); scene.add(sw);
for(let k=0;k<8;k++){ const port=new THREE.Mesh(new THREE.BoxGeometry(.4,.18,.12), new THREE.MeshBasicMaterial({color:0x4dd2ff})); port.position.set(-2.8+k*.8,.8,SW.z+.93); scene.add(port); }
const swLabel = label(); swLabel.position.set(0,3.2,SW.z); scene.add(swLabel);

for(let i=0;i<M;i++){
  const p = posOf(i), a = new THREE.Vector3(0,.9,SW.z+1), b = new THREE.Vector3(p.x,.9,p.z-2.6);
  const mid = a.clone().lerp(b,.5); mid.y = 3.2;
  const curve = new THREE.QuadraticBezierCurve3(a, mid, b);
  const m = new THREE.MeshBasicMaterial({color:0x2d6f9c, transparent:true, opacity:.55});
  scene.add(new THREE.Mesh(new THREE.TubeGeometry(curve, 28, .06, 6, false), m)); links.push({curve:curve, mat:m});
}

// packets
const packets = []; const pGeo = new THREE.SphereGeometry(.17, 10, 8);
function spawn(i, color, dir, size){ const m = new THREE.Mesh(pGeo, new THREE.MeshBasicMaterial({color:color})); m.scale.setScalar(size||1); scene.add(m);
  packets.push({m:m, i:i, t:0, dir:dir, v:.55 + Math.random()*.15}); if(packets.length>220){ const old=packets.shift(); scene.remove(old.m); old.m.material.dispose(); } }

// ---------- camera ----------
let theta = .5, phi = 1.0, radius = 38, tTheta = theta, tPhi = phi, tRadius = radius;
const target = new THREE.Vector3(0,1.5,5), goal = new THREE.Vector3(0,1.5,5);
let focused = false;
function focusMachine(i){ const p = posOf(i); goal.set(p.x, 2.2, p.z); tRadius = 17; tPhi = 1.0; focused = true; }
function overview(){ goal.set(0,1.5,5); tRadius = 38; tPhi = 1.0; focused = false; }
function placeCam(){
  theta += (tTheta - theta)*.1; phi += (tPhi - phi)*.1; radius += (tRadius - radius)*.08; target.lerp(goal, .08);
  camera.position.set(target.x + radius*Math.sin(phi)*Math.sin(theta), target.y + radius*Math.cos(phi), target.z + radius*Math.sin(phi)*Math.cos(theta));
  camera.lookAt(target);
}
const el = renderer.domElement; let drag=false, lx=0, ly=0, downX=0, downY=0;
el.addEventListener('mousedown', e => { drag=true; lx=downX=e.clientX; ly=downY=e.clientY; lastInteract=clock.getElapsedTime(); });
window.addEventListener('mouseup', e => { if(!drag) return; drag=false;
  if(Math.hypot(e.clientX-downX, e.clientY-downY) < 5){ const i=pick(e); if(i>=0){ select(i,true); } } });
window.addEventListener('mousemove', e => { if(drag){ tTheta -= (e.clientX-lx)*.006; tPhi = Math.max(.35, Math.min(1.45, tPhi-(e.clientY-ly)*.005)); lx=e.clientX; ly=e.clientY; lastInteract=clock.getElapsedTime(); } hover(e); });
el.addEventListener('wheel', e => { e.preventDefault(); tRadius = Math.max(9, Math.min(60, tRadius + e.deltaY*.03)); lastInteract=clock.getElapsedTime(); }, {passive:false});
const ray = new THREE.Raycaster(), mouse = new THREE.Vector2(), tip = $('tip');
function pick(e){ const r = el.getBoundingClientRect(); mouse.set(((e.clientX-r.left)/r.width)*2-1, -((e.clientY-r.top)/r.height)*2+1);
  ray.setFromCamera(mouse, camera); const h = ray.intersectObjects(pickables)[0]; return h ? h.object.userData.idx : -1; }
function hover(e){ if(drag){ tip.style.display='none'; return; } const i = pick(e);
  if(i<0){ tip.style.display='none'; return; } const f = D.frames[tick][i], r = app.getBoundingClientRect();
  tip.textContent = D.machines[i] + '  [' + f.fu + ']\nfailure probability ' + (f.p*100).toFixed(2) + '%\n' + f.rpm + ' rpm | ' + f.tq + ' Nm | wear ' + f.wr + ' min' + (NET ? '\nflow attack prob ' + (f.np*100).toFixed(1) + '%' : '');
  tip.style.display='block'; tip.style.left=(e.clientX-r.left+14)+'px'; tip.style.top=(e.clientY-r.top+14)+'px'; }

// ---------- HUD ----------
const fleet = $('fleet'), detail = $('detail'), logEl = $('log'), heat = $('heat'), hinfo = $('heatinfo');
const cards = [];
D.machines.forEach((n,i) => { const d = document.createElement('div'); d.className='card';
  d.innerHTML = '<div class="ch"><span class="dot"></span><b>'+n+'</b><span class="lv"></span></div><canvas class="spark" width="220" height="34"></canvas><div class="cm"><span class="pp"></span><span class="nn"></span></div>';
  d.addEventListener('click', () => select(i,true)); fleet.appendChild(d); cards.push(d); });
const scrub = $('scrub'); scrub.max = T-1;
const fmtClock = t => { const m = t*D.tickMinutes, h = Math.floor(m/60); return 'T+' + String(h).padStart(2,'0') + ':' + String(Math.round(m%60)).padStart(2,'0'); };

function spark(cv, i){ const g = cv.getContext('2d'), w = cv.width, h = cv.height; g.clearRect(0,0,w,h);
  [TH.early, TH.primary].forEach(v => { g.strokeStyle='rgba(170,190,225,.25)'; g.beginPath(); g.moveTo(0,h-v*(h-3)-1); g.lineTo(w,h-v*(h-3)-1); g.stroke(); });
  g.strokeStyle = css(COL[D.frames[tick][i].fu]); g.lineWidth = 1.6; g.beginPath();
  for(let t=0;t<=tick;t++){ const x = t/(T-1)*w, y = h - D.frames[t][i].p*(h-3) - 1; if(t===0) g.moveTo(x,y); else g.lineTo(x,y); } g.stroke();
  g.fillStyle='rgba(255,255,255,.12)'; g.fillRect(tick/(T-1)*w, 0, 1, h); }

function barRow(k, v, frac, color){ return '<div class="row"><span class="k">'+k+'</span><span class="b"><div style="width:'+Math.max(2,Math.min(100,frac*100))+'%;background:'+(color||'#58b6ff')+'"></div></span><span class="v">'+v+'</span></div>'; }
function shapRows(list){ if(!list || !list.length) return '<div class="hint">none for this tick</div>'; const mx = Math.max.apply(null, list.map(s=>Math.abs(s[1]))) || 1;
  return list.map(s => barRow(s[0], (s[1]>0?'+':'') + s[1].toFixed(2), Math.abs(s[1])/mx, s[1]>0?'#ff6b6b':'#3fd68f')).join(''); }

function renderDetail(){ const f = D.frames[tick][sel], col = css(COL[f.fu]);
  let h = '<div style="display:flex;align-items:center;gap:8px"><b style="font-size:16px">'+D.machines[sel]+'</b><span style="color:#8fa4c8">type '+f.ty+'</span><span class="lv" style="margin-left:auto;background:'+col+'">'+f.fu+'</span></div>';
  h += '<div class="why" style="border-color:'+col+';margin-top:8px">'+f.wy+'</div>';
  h += '<div class="sec">Failure probability (XGBoost)</div><div class="big" style="color:'+css(COL[f.st==='HIGH RISK'?'ALARM':f.st==='EARLY WARNING'?'WARNING':'OK'])+'">'+(f.p*100).toFixed(2)+'%</div>';
  h += '<div class="barwrap"><div class="bar" style="width:'+Math.min(100,f.p*100)+'%;background:'+col+'"></div><div class="mark" style="left:'+TH.early*100+'%"><i>early '+TH.early+'</i></div><div class="mark" style="left:'+TH.primary*100+'%"><i>primary '+TH.primary+'</i></div></div><div style="height:12px"></div>';
  h += '<div class="sec">Telemetry (simulated)</div>' + barRow('Speed', f.rpm+' rpm', (f.rpm-1000)/1200) + barRow('Torque', f.tq+' Nm', f.tq/80) + barRow('Tool wear', f.wr+' min', f.wr/300, '#ffd166') +
        barRow('Air temp', f.air+' K', (f.air-295)/10) + barRow('Process temp', f.pr+' K', (f.pr-305)/10);
  h += '<div class="sec">Why this probability (SHAP, log-odds)</div>' + shapRows(f.sh);
  if(NET){ const v = f.ta&&f.nf ? ['Attack flow correctly flagged','#3a1620'] : f.ta&&!f.nf ? ['MISSED attack flow','#4a3510'] : !f.ta&&f.nf ? ['False alarm on benign flow','#4a3510'] : ['Benign flow correctly passed','#10301f'];
    h += '<div class="sec">Network flow this tick (Random Forest)</div>' + barRow('Attack prob', (f.np*100).toFixed(1)+'%', f.np, f.nf?'#ff6b6b':'#4dd2ff') +
         '<div class="row"><span class="k">Simulated source</span><span style="flex:1;text-align:right">'+f.nt+'</span></div><div class="verdict" style="background:'+v[1]+'">'+v[0]+'</div>';
    if(f.ns && f.ns.length) h += '<div class="sec">Flow drivers (SHAP, probability)</div>' + shapRows(f.ns);
  } else h += '<div class="sec">Network</div><div class="hint">No benchmark flows loaded: run src/data/export_demo_flows.py</div>';
  h += '<div class="hint">Use the controls under this view to open this exact reading in the Machine Health or Network Security page.</div>';
  detail.innerHTML = h; }

function renderLog(){ const ev = D.events.filter(e => e.t <= tick).reverse(); logEl.innerHTML = ev.length ? ev.map((e,k) =>
  '<div class="ev" data-t="'+e.t+'" data-m="'+e.m+'"><span class="c" style="background:'+css(COL[e.lvl])+'"></span><span class="t">'+fmtClock(e.t)+'</span><span>'+e.text+'</span></div>').join('') : '<div class="hint">No alerts yet. Alerts appear when the models cross their thresholds.</div>';
  Array.prototype.forEach.call(logEl.querySelectorAll('.ev'), n => n.addEventListener('click', () => { setTick(+n.dataset.t); select(+n.dataset.m, true); })); }

function drawHeat(){ const r = heat.getBoundingClientRect(); if(!r.width) return; heat.width = r.width*(window.devicePixelRatio||1); heat.height = r.height*(window.devicePixelRatio||1);
  const g = heat.getContext('2d'), w = heat.width, h = heat.height, cw = w/T, rh = h/M; g.clearRect(0,0,w,h);
  for(let m=0;m<M;m++) for(let t=0;t<T;t++){ const lv = D.frames[t][m].fu; g.fillStyle = lv==='OK' ? 'rgba(46,204,113,.28)' : css(COL[lv]); g.fillRect(t*cw, m*rh+1, Math.ceil(cw), rh-2); }
  g.fillStyle='rgba(255,255,255,.9)'; g.fillRect(tick*cw, 0, Math.max(2,cw*.6), h);
  g.fillStyle='rgba(255,255,255,.7)'; g.font=(10*(window.devicePixelRatio||1))+'px Arial'; for(let m=0;m<M;m++) g.fillText(D.machines[m], 4, m*rh+rh*.65);
  if(D.window && D.window.end > D.window.start){ g.strokeStyle='rgba(255,255,255,.35)'; g.setLineDash([4,3]); g.strokeRect(D.window.start*cw, D.window.target*rh+.5, (D.window.end-D.window.start)*cw, rh-1); g.setLineDash([]); } }
heat.addEventListener('click', e => { const r = heat.getBoundingClientRect(); setTick(Math.max(0,Math.min(T-1, Math.floor((e.clientX-r.left)/r.width*T)))); select(Math.max(0,Math.min(M-1, Math.floor((e.clientY-r.top)/r.height*M))), true); });
heat.addEventListener('mousemove', e => { const r = heat.getBoundingClientRect(); const t = Math.max(0,Math.min(T-1, Math.floor((e.clientX-r.left)/r.width*T))), m = Math.max(0,Math.min(M-1, Math.floor((e.clientY-r.top)/r.height*M)));
  hinfo.textContent = 'Fused status timeline - ' + D.machines[m] + ' at ' + fmtClock(t) + ': ' + D.frames[t][m].fu + (D.window&&D.window.end>D.window.start?'  (dashed box = injected event window)':''); });

// ---------- tick application ----------
function applyTick(spawnPackets){
  const fr = D.frames[tick]; let worst = 0, wm = 0;
  fr.forEach((f,i) => { const o = machines[i], lv = LV.indexOf(f.fu); if(lv > worst){ worst = lv; wm = 1; } else if(lv === worst && lv>0) wm++;
    o.goal.setHex(COL[f.fu]); o.level = f.fu; o.f = f;
    o.lamps[0].emissiveIntensity = f.st==='HIGH RISK' ? 1.6 : .05; o.lamps[1].emissiveIntensity = f.st==='EARLY WARNING' ? 1.6 : .05; o.lamps[2].emissiveIntensity = f.st==='NORMAL' ? 1.4 : .05;
    const lit = Math.round(Math.min(1, f.wr/300)*10); o.wear.forEach((m,k) => m.emissiveIntensity = k < lit ? (k>=7 ? 1.1 : .55) : 0);
    o.shield.visible = !!(NET && (f.cf || f.nf)); o.shieldMat.opacity = f.cf ? .5 : .2; o.beacon.visible = f.fu === 'CRITICAL';
    paintLabel(o.label, D.machines[i], f.fu + '  ' + (f.p*100).toFixed(1) + '%', css(COL[f.fu]));
    if(spawnPackets){ spawn(i, 0x2ecc71, -1, .7);
      if(NET) spawn(i, f.nf ? 0xff4d4f : (f.ta ? 0xffa31a : 0x4dd2ff), 1, f.nf ? 1.5 : 1);
      links[i].mat.color.setHex(NET && f.nf ? 0xff4d4f : 0x2d6f9c); links[i].mat.opacity = NET && f.nf ? .9 : .55; } });
  const gp = $('gpill'); gp.textContent = worst === 0 ? 'SITE: ALL OK' : 'SITE: ' + LV[worst] + (wm>1 ? ' x'+wm : ''); gp.style.background = css(COL[LV[worst]]); gp.style.color = '#06101f';
  paintLabel(swLabel, 'Core switch / IDS', NET ? (fr.filter(f=>f.nf).length + ' flagged flows') : 'feed not loaded', NET && fr.some(f=>f.nf) ? '#ff6b6b' : '#4dd2ff');
  $('clock').textContent = fmtClock(tick) + '  (' + (tick+1) + '/' + T + ')'; scrub.value = tick;
  cards.forEach((c,i) => { const f = fr[i], col = css(COL[f.fu]); c.classList.toggle('sel', i===sel); c.querySelector('.dot').style.background = col;
    const lv = c.querySelector('.lv'); lv.textContent = f.fu; lv.style.background = col; c.querySelector('.pp').textContent = 'fail ' + (f.p*100).toFixed(1) + '%';
    c.querySelector('.nn').textContent = NET ? 'flow ' + (f.np*100).toFixed(0) + '%' : ''; spark(c.querySelector('canvas'), i); });
  renderDetail(); renderLog(); drawHeat(); }

function setTick(t){ tick = Math.max(0, Math.min(T-1, t)); acc = 0; applyTick(false); }
function select(i, focus){ sel = i; if(focus) focusMachine(i); applyTick(false); }
$('reset').onclick = () => { overview(); };
$('play').onclick = () => { if(tick >= T-1){ tick = 0; } playing = !playing; $('play').textContent = playing ? 'Pause' : 'Play'; };
Array.prototype.forEach.call($('speeds').querySelectorAll('button'), b => b.onclick = () => { speed = +b.dataset.s; Array.prototype.forEach.call($('speeds').querySelectorAll('button'), x => x.classList.toggle('on', x===b)); });
scrub.oninput = () => setTick(+scrub.value);
window.addEventListener('keydown', e => { if(e.code==='Space'){ e.preventDefault(); $('play').click(); } if(e.code==='ArrowRight') setTick(tick+1); if(e.code==='ArrowLeft') setTick(tick-1); });
function resize(){ const w = app.clientWidth, h = app.clientHeight; renderer.setSize(w,h); camera.aspect = w/h; camera.updateProjectionMatrix(); drawHeat(); }
window.addEventListener('resize', resize);

// ---------- loop ----------
const clock = new THREE.Clock(); let prev = 0;
function animate(){
  requestAnimationFrame(animate);
  const t = clock.getElapsedTime(), dt = Math.min(.1, t - prev); prev = t;
  if(playing){ acc += dt*1000*speed; if(acc >= 900){ acc = 0; if(tick < T-1){ tick++; applyTick(true); } else { playing=false; $('play').textContent='Replay'; } } }
  machines.forEach((o,i) => { const f = o.f || D.frames[0][i];
    o.col.lerp(o.goal, .12); o.bodyMat.emissive.copy(o.col); o.screenMat.emissive.copy(o.col); o.ringMat.color.copy(o.col); o.light.color.copy(o.col);
    const lvl = LV.indexOf(f.fu), pulse = .5 + .5*Math.sin(t*(lvl>=3?7:lvl>=1?3.2:1.2));
    o.bodyMat.emissiveIntensity = .08 + (lvl===0?0:.5*pulse); o.light.intensity = .45 + (lvl===0?0:.9*pulse);
    o.ring.rotation.z += .004 + lvl*.003; o.ringMat.opacity = .45 + .4*pulse;
    o.spindle.rotation.y += Math.min(f.rpm,3000)/1500 * .14;
    const shake = (f.fu==='ALARM'||f.fu==='CRITICAL') ? .035 : 0; o.g.position.x = posOf(i).x + (Math.random()-.5)*shake; o.g.position.z = posOf(i).z + (Math.random()-.5)*shake;
    if(o.shield.visible){ o.shield.rotation.y += .01; o.shield.scale.setScalar(1 + .04*Math.sin(t*6)); }
    if(o.beacon.visible){ o.beaconMat.opacity = .16 + .22*pulse; o.beacon.rotation.y += .02; } });
  for(let k=packets.length-1;k>=0;k--){ const p = packets[k]; p.t += dt*p.v; if(p.t >= 1){ scene.remove(p.m); p.m.material.dispose(); packets.splice(k,1); continue; }
    p.m.position.copy(links[p.i].curve.getPoint(p.dir > 0 ? p.t : 1-p.t)); }
  swMat.emissiveIntensity = .4 + .15*Math.sin(t*3); dust.rotation.y += .0004;
  if(!focused && !drag && t - lastInteract > 5) tTheta += .0009;
  placeCam(); renderer.render(scene, camera); }

// defaults: focus the machine with the highest peak risk so the first view is meaningful
(function init(){ let best = 0, bv = -1; for(let i=0;i<M;i++){ let mx = 0; for(let t=0;t<T;t++) mx = Math.max(mx, LV.indexOf(D.frames[t][i].fu)*10 + D.frames[t][i].p); if(mx > bv){ bv = mx; best = i; } }
  sel = (D.window && D.window.fault || D.window && D.window.cyber) ? D.window.target : best; resize(); applyTick(true); animate(); })();
})();
</script></body></html>
"""
