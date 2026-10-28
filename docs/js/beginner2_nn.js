"use strict";
/* Extra beginner steps for lessons 03 (neural networks) and 09 (network algorithms).
   Written for readers with no machine learning experience and minimal chemistry.
   Each step assumes only the earlier beginner steps. */

/* ---------- a tiny network that trains in a blink: d inputs, k hidden neurons, one output ---------- */
function bnInit(d, k, seed, span){
  var r = rng(seed), n = k * d + 3 * k + 1, p = new Float64Array(n), j, i;
  for (j = 0; j < k; j++) {
    for (i = 0; i < d; i++) p[j * d + i] = (r() * 2 - 1) * 1.2;
    if (d === 1) { var s = r() < 0.5 ? -1 : 1, kn = -span + 2 * span * (j + 0.5) / k; p[j] = s * (0.6 + r() * 0.8); p[k * d + j] = -p[j] * kn; } else p[k * d + j] = (r() * 2 - 1) * 0.8;
    p[k * d + k + j] = (r() * 2 - 1) * 0.5;
  }
  return { d: d, k: k, p: p };
}
function bnRun(m, x, relu){
  var d = m.d, k = m.k, p = m.p, h = new Array(k), y = p[k * d + 2 * k], j, i, z;
  for (j = 0; j < k; j++) { z = p[k * d + j]; for (i = 0; i < d; i++) z += p[j * d + i] * x[i]; h[j] = relu === false ? z : Math.max(0, z); y += p[k * d + k + j] * h[j]; }
  return { h: h, y: y };
}
function bnTrain(m, X, Y, steps, lr, relu){
  var d = m.d, k = m.k, p = m.p, np = p.length, n = Y.length, g = new Float64Array(np), m1 = new Float64Array(np), m2 = new Float64Array(np), z = new Float64Array(k), h = new Float64Array(k), t, s, j, i, q;
  for (t = 1; t <= steps; t++) {
    g.fill(0);
    for (s = 0; s < n; s++) {
      var x = X[s], y = p[k * d + 2 * k];
      for (j = 0; j < k; j++) { var zz = p[k * d + j]; for (i = 0; i < d; i++) zz += p[j * d + i] * x[i]; z[j] = zz; h[j] = relu === false ? zz : Math.max(0, zz); y += p[k * d + k + j] * h[j]; }
      var e = 2 * (y - Y[s]) / n; g[k * d + 2 * k] += e;
      for (j = 0; j < k; j++) { g[k * d + k + j] += e * h[j]; var dh = e * p[k * d + k + j]; if (relu !== false && z[j] <= 0) dh = 0; g[k * d + j] += dh; for (i = 0; i < d; i++) g[j * d + i] += dh * x[i]; }
    }
    var c1 = 1 - Math.pow(0.9, t), c2 = 1 - Math.pow(0.999, t);
    for (q = 0; q < np; q++) { m1[q] = 0.9 * m1[q] + 0.1 * g[q]; m2[q] = 0.999 * m2[q] + 0.001 * g[q] * g[q]; p[q] -= lr * (m1[q] / c1) / (Math.sqrt(m2[q] / c2) + 1e-8); }
  }
  return m;
}
/* a seeded sample of compounds, features and target rescaled so that 0 is average */
function bnData(key, feats, n, seed, pool){
  var ck = EP + key; if (D.cache[ck]) return D.cache[ck];
  var r = rng(seed), sh = (pool || D.TRAIN).slice(), a, j, t; for (a = sh.length - 1; a > 0; a--) { j = Math.floor(r() * (a + 1)); t = sh[a]; sh[a] = sh[j]; sh[j] = t; }
  var idx = sh.slice(0, n), st = feats.map(function(f){ var v = idx.map(function(i){ return D.COL[f][i]; }), m = mean(v), s = Math.sqrt(mean(v.map(function(q){ return (q - m) * (q - m); }))) || 1; return { m: m, s: s }; }), ys = idx.map(function(i){ return D.Y[i]; }), my = mean(ys), sy = Math.sqrt(mean(ys.map(function(q){ return (q - my) * (q - my); }))) || 1;
  var X = idx.map(function(i){ return feats.map(function(f, q){ return (D.COL[f][i] - st[q].m) / st[q].s; }); }), Yz = idx.map(function(i){ return (D.Y[i] - my) / sy; });
  return (D.cache[ck] = { idx: idx, X: X, Yz: Yz, st: st, my: my, sy: sy, feats: feats });
}
function bnZ(S, i){ return S.feats.map(function(f, q){ return (D.COL[f][i] - S.st[q].m) / S.st[q].s; }); }
function bnR2(pred, truth){ var m = mean(truth), a = 0, b = 0, i; for (i = 0; i < truth.length; i++) { a += (truth[i] - pred[i]) * (truth[i] - pred[i]); b += (truth[i] - m) * (truth[i] - m); } return b ? 1 - a / b : 0; }
function bnLine(X, Y){ var n = X.length, mx = mean(X.map(function(x){ return x[0]; })), my = mean(Y), sxy = 0, sxx = 0, i; for (i = 0; i < n; i++) { sxy += (X[i][0] - mx) * (Y[i] - my); sxx += (X[i][0] - mx) * (X[i][0] - mx); } var w = sxx ? sxy / sxx : 0; return { w: w, b: my - w * mx }; }
function bnSeg(items, cur, attr){ return '<span class="seg" ' + attr + '>' + items.map(function(it, i){ return '<button type="button" data-v="' + i + '"' + (i === cur ? ' class="on"' : '') + '>' + esc(it) + '</button>'; }).join('') + '</span>'; }

/* ---------- 03 step: neurons side by side ---------- */
LESSONS.nn.add({ level: 'beginner', short: 'Neurons side by side', title: 'Many neurons, each casting a vote',
  change: 'Click a neuron to switch it off, and click again to switch it on. Press Next compound to try another compound.',
  watch: 'One neuron is a sum with dials. A network uses several neurons side by side. Here four neurons read the same two features of a compound, logP (how oily it is) and TPSA (how much of its surface attracts water). Each neuron has its own dials, so each reacts to a different pattern. A neuron stays silent, which means it gives 0, when its sum is below zero. A final neuron collects the four answers. Each answer is multiplied by its own dial and added up, and the total is the prediction. Switch neurons off and watch the vote total change. Then look at the chart: with all four on, the dots lie closer to the dashed line. The computer set all the dials, using 260 training compounds.',
  build: function(api){
    var dg = api.el('pn', 'left:28px;top:28px;width:520px;height:408px', '<h4>The network</h4><canvas class="dg" width="490" height="236"></canvas><div class="vt" style="font-size:12.5px;line-height:1.45;color:' + C.char + '"></div>');
    var sc = api.el('pn', 'left:568px;top:28px;width:364px;height:408px', '<h4>Predicted against measured, 260 training compounds</h4><canvas class="sc" width="334" height="300"></canvas><div class="sn" style="font-size:12.5px;margin-top:4px;color:' + C.char + '"></div>');
    var ctl = api.addControl('Neurons <label><input type="checkbox" data-n="0" checked> 1</label> <label><input type="checkbox" data-n="1" checked> 2</label> <label><input type="checkbox" data-n="2" checked> 3</label> <label><input type="checkbox" data-n="3" checked> 4</label> <button type="button" class="btn sm ghost" data-next>Next compound</button>');
    var on = [1, 1, 1, 1], cmp = 7, M = null, S = null, boxes = qsa('[data-n]', ctl), FE = ['logP', 'TPSA'];
    function prep(){ S = bnData('b2', FE, 260, 21); var key = EP + 'b2net'; if (!D.cache[key]) { var m = bnInit(2, 4, 5); bnTrain(m, S.X, S.Yz, 900, 0.03, true); D.cache[key] = m; } M = D.cache[key]; }
    function predAll(){ var out = new Float64Array(D.N), se = 0; S.idx.forEach(function(i, q){ var r = bnRun(M, S.X[q]), y = M.p[M.k * 2 + 2 * M.k]; for (var j = 0; j < 4; j++) y += on[j] ? M.p[M.k * 2 + M.k + j] * r.h[j] : 0; out[i] = S.my + S.sy * y; se += (out[i] - D.Y[i]) * (out[i] - D.Y[i]); }); return { p: out, rm: Math.sqrt(se / S.idx.length) }; }
    function draw(){
      if (!M) return; var g = cv(qs('.dg', dg)), x = g.x, q = ((cmp % S.idx.length) + S.idx.length) % S.idx.length, r = bnRun(M, S.X[q]), xs = [60, 245, 430], ys = [[84, 152], [30, 86, 142, 198], [114]];
      x.textAlign = 'center'; x.font = '600 11px Aptos, "Segoe UI", sans-serif';
      for (var j = 0; j < 4; j++) { [0, 1].forEach(function(i){ var w = M.p[j * 2 + i]; x.strokeStyle = w >= 0 ? C.red : '#6F8196'; x.globalAlpha = on[j] ? 0.25 + 0.5 * Math.min(1, Math.abs(w)) : 0.08; x.lineWidth = 1 + 2.5 * Math.min(1, Math.abs(w)); x.beginPath(); x.moveTo(xs[0] + 20, ys[0][i]); x.lineTo(xs[1] - 20, ys[1][j]); x.stroke(); }); var vw = M.p[M.k * 2 + M.k + j]; x.strokeStyle = vw >= 0 ? C.red : '#6F8196'; x.globalAlpha = on[j] ? 0.3 + 0.5 * Math.min(1, Math.abs(vw)) : 0.08; x.lineWidth = 1 + 2.5 * Math.min(1, Math.abs(vw)); x.beginPath(); x.moveTo(xs[1] + 20, ys[1][j]); x.lineTo(xs[2] - 22, ys[2][0]); x.stroke(); }
      x.globalAlpha = 1;
      function node(cx, cy, fill, stroke, dash, label, sub){ x.fillStyle = fill; x.strokeStyle = stroke; x.lineWidth = 2; x.setLineDash(dash ? [4, 3] : []); x.beginPath(); x.arc(cx, cy, 19, 0, 6.3); x.fill(); x.stroke(); x.setLineDash([]); x.fillStyle = C.char; x.font = '700 12px Aptos, "Segoe UI", sans-serif'; x.fillText(label, cx, cy + 4); if (sub) { x.font = '11px Aptos, "Segoe UI", sans-serif'; x.fillText(sub, cx, cy + 34); } }
      node(xs[0], ys[0][0], '#fff', C.char, false, S.X[q][0].toFixed(1), 'logP'); node(xs[0], ys[0][1], '#fff', C.char, false, S.X[q][1].toFixed(1), 'TPSA');
      for (j = 0; j < 4; j++) { var act = on[j] && r.h[j] > 0; node(xs[1], ys[1][j], !on[j] ? '#EEF0F3' : (act ? '#FBE9E9' : '#fff'), on[j] ? C.char : '#B8C0CA', !on[j] || !act, on[j] ? r.h[j].toFixed(1) : 'off', ''); x.font = '600 10.5px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'left'; x.strokeStyle = 'rgba(255,255,255,.95)'; x.lineWidth = 3; x.strokeText('neuron ' + (j + 1), xs[1] + 24, ys[1][j] + 4); x.fillStyle = C.mid; x.fillText('neuron ' + (j + 1), xs[1] + 24, ys[1][j] + 4); x.textAlign = 'center'; }
      var pr = predAll(), y = M.p[M.k * 2 + 2 * M.k], votes = [];
      for (j = 0; j < 4; j++) { var v = on[j] ? M.p[M.k * 2 + M.k + j] * r.h[j] : 0; votes.push(v); y += v; }
      node(xs[2], ys[2][0], '#FBE9E9', C.char, false, (S.my + S.sy * y).toFixed(1), 'prediction');
      x.textAlign = 'left'; x.fillStyle = C.mid; x.font = '11px Aptos, "Segoe UI", sans-serif'; x.fillText('Line thickness: how strongly a dial counts. Red adds, blue-grey subtracts.', 4, g.h - 3);
      var base = S.my + S.sy * M.p[M.k * 2 + 2 * M.k], meas = D.Y[S.idx[q]];
      qs('.vt', dg).innerHTML = '<div><b>This compound</b>, measured value ' + meas.toFixed(2) + '. Base amount <b>' + base.toFixed(2) + '</b>' + votes.map(function(v, j){ return ', neuron ' + (j + 1) + ' <b style="color:' + (!on[j] ? C.mid : (v >= 0 ? C.dred : '#4B5A6B')) + '">' + (!on[j] ? 'off' : (v >= 0 ? '+' : '') + (S.sy * v).toFixed(2)) + '</b>'; }).join('') + '. Total: <b style="color:' + C.red + '">' + (S.my + S.sy * y).toFixed(2) + '</b>.</div>';
      scatterPV(qs('.sc', sc), pr.p, S.idx, '', { max: 260 });
      var offN = on.filter(function(v){ return !v; }).length;
      qs('.sn', sc).innerHTML = 'Typical miss: <b style="color:' + C.red + '">' + pr.rm.toFixed(2) + '</b> ' + esc(unitWord()) + ' units. ' + (offN === 0 ? 'All four neurons are on.' : offN + ' neuron' + (offN > 1 ? 's are' : ' is') + ' off, so the predictions get worse or shift.');
    }
    boxes.forEach(function(b){ b.addEventListener('change', function(){ on[+b.dataset.n] = b.checked ? 1 : 0; draw(); }); });
    qs('[data-next]', ctl).addEventListener('click', function(){ cmp = Math.floor(Math.random() * 260); draw(); });
    qs('.dg', dg).addEventListener('click', function(e){ var c = qs('.dg', dg), rc = c.getBoundingClientRect(), px = (e.clientX - rc.left) / rc.width * 490, py = (e.clientY - rc.top) / rc.height * 236; [30, 86, 142, 198].forEach(function(yy, j){ if (Math.hypot(px - 245, py - yy) < 22) { on[j] = on[j] ? 0 : 1; boxes[j].checked = !!on[j]; draw(); } }); });
    function render(){ prep(); draw(); }
    async function play(tok){
      on = [1, 1, 1, 1]; boxes.forEach(function(b){ b.checked = true; }); render(); api.cap('Four neurons read the same two features. Each one has its own dials, so each gives its own answer.'); await api.sleep(3400);
      for (var j = 0; j < 4; j++) { api.checkTok(tok); on[j] = 0; boxes[j].checked = false; draw(); api.cap('Neuron ' + (j + 1) + ' is switched off. The prediction and the typical miss change.'); await api.sleep(2200); on[j] = 1; boxes[j].checked = true; draw(); await api.sleep(500); }
      api.cap('The final neuron adds up the four votes. Together they predict better than any one alone.'); await api.sleep(3000); api.cap(null);
    }
    return { render: render, play: play };
  } });
GUIDE.steps['nn/Neurons side by side'] = { notice: 'Switching a neuron off changes the total and makes the dots drift from the dashed line.', q: 'Why use several neurons instead of one?', a: 'Each neuron reacts to a different pattern in the features. Adding their votes gives a better prediction than any single neuron can.' };

/* ---------- 03 step: the bend ---------- */
LESSONS.nn.add({ level: 'beginner', short: 'Why a network can bend', title: 'The switch that lets a network bend',
  change: 'Drag the slider to give the network more neurons. Tick the box to take the switch away.',
  watch: 'A single neuron makes a straight line. Real data often curves. A neuron has a switch: it stays silent until its sum rises above zero, and then it passes the sum through. After that point the neuron adds a slope to the picture. Each neuron switches on at a different place, so adding neurons adds bends, and the curve can follow the dots better. Each dot is a compound: across is logP (how oily it is) and up is the lab value. The grey dashed line is the best straight line. The red curve is the network. Tick the box to remove the switch. Without the switch, the neurons only add dials to dials, and the network can draw nothing but a straight line, however many neurons it has.',
  build: function(api){
    var cp = api.el('pn', 'left:28px;top:28px;width:620px;height:408px', '<h4>Lab value against logP, 260 training compounds</h4><canvas width="590" height="352"></canvas>');
    var tp = api.el('pn', 'left:668px;top:28px;width:264px;height:408px;overflow:auto', '<div class="bt" style="font-size:13px;line-height:1.5;color:' + C.char + '"></div>');
    var ctl = api.addControl('Neurons <input type="range" min="1" max="8" step="1" value="3" style="width:200px"> <b class="kv">3</b> <label><input type="checkbox" data-sw> Take the switch away</label>');
    var rg = qs('input[type=range]', ctl), kv = qs('.kv', ctl), sw = qs('[data-sw]', ctl), S = null, line = null;
    function net(k){ var key = EP + 'b3net' + k; if (!D.cache[key]) { var m = bnInit(1, k, 3 + k, 2.2); bnTrain(m, S.X, S.Yz, 1500, 0.03, true); D.cache[key] = m; } return D.cache[key]; }
    function draw(){
      S = bnData('b3', ['logP'], 260, 4); line = bnLine(S.X, S.Yz); var k = +rg.value, relu = !sw.checked, m = net(k); kv.textContent = k;
      var g = cv(qs('canvas', cp)), x = g.x, box = { l: 52, r: g.w - 14, t: 14, b: g.h - 40 }, zlo = -2.6, zhi = 2.6, ylo = D.ylo, yhi = D.yhi, sx = function(z){ return box.l + (z - zlo) / (zhi - zlo) * (box.r - box.l); }, sy = function(v){ return box.b - (v - ylo) / (yhi - ylo) * (box.b - box.t); };
      x.strokeStyle = C.silver; x.fillStyle = C.mid; x.font = '11px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'right'; niceTicks(ylo, yhi, 5).v.forEach(function(v){ x.beginPath(); x.moveTo(box.l, sy(v)); x.lineTo(box.r, sy(v)); x.stroke(); x.fillText(v.toFixed(1), box.l - 6, sy(v) + 4); });
      x.textAlign = 'center'; [-2, -1, 0, 1, 2].forEach(function(z){ x.fillText((S.st[0].m + S.st[0].s * z).toFixed(1), sx(z), box.b + 16); });
      x.fillStyle = C.char; x.font = '600 12px Aptos, "Segoe UI", sans-serif'; x.fillText('logP (how oily the compound is)', (box.l + box.r) / 2, box.b + 34); x.save(); x.translate(13, (box.t + box.b) / 2); x.rotate(-Math.PI / 2); x.fillText(unitWord(), 0, 0); x.restore();
      x.fillStyle = 'rgba(52,55,63,.4)'; S.idx.forEach(function(i, q){ x.beginPath(); x.arc(sx(S.X[q][0]), sy(D.Y[i]), 2.6, 0, 6.3); x.fill(); });
      function lineAt(z){ return S.my + S.sy * (line.w * z + line.b); }
      x.strokeStyle = C.mid; x.lineWidth = 2; x.setLineDash([6, 5]); x.beginPath(); x.moveTo(sx(zlo), sy(lineAt(zlo))); x.lineTo(sx(zhi), sy(lineAt(zhi))); x.stroke(); x.setLineDash([]);
      var f = function(z){ return relu ? S.my + S.sy * bnRun(m, [z]).y : lineAt(z); };
      x.strokeStyle = C.red; x.lineWidth = 3.4; x.beginPath(); for (var z = zlo; z <= zhi + 1e-9; z += 0.05) { var px = sx(z), py = sy(clamp(f(z), ylo - 1, yhi + 1)); z === zlo ? x.moveTo(px, py) : x.lineTo(px, py); } x.stroke();
      var seL = 0, seN = 0; S.idx.forEach(function(i, q){ seL += (lineAt(S.X[q][0]) - D.Y[i]) * (lineAt(S.X[q][0]) - D.Y[i]); seN += (f(S.X[q][0]) - D.Y[i]) * (f(S.X[q][0]) - D.Y[i]); });
      var rl = Math.sqrt(seL / S.idx.length), rn = Math.sqrt(seN / S.idx.length);
      qs('.bt', tp).innerHTML = '<div class="ol">Typical miss</div><div style="display:flex;gap:18px"><div><div class="big" style="font-size:28px;color:' + C.mid + '">' + rl.toFixed(2) + '</div><div style="font-size:12px">straight line</div></div><div><div class="big" style="font-size:28px;color:' + C.red + '">' + rn.toFixed(2) + '</div><div style="font-size:12px">network</div></div></div><p style="margin-top:12px">' + (!relu ? 'The switch is gone, so the network draws exactly the straight line. It has ' + k + ' neuron' + (k > 1 ? 's' : '') + ' and no bends.' : (k === 1 ? 'One neuron with its switch gives one bend: a flat piece, then a slope.' : 'With ' + k + ' neurons the curve can bend up to ' + k + ' times. ' + (rl - rn > 0.02 ? 'It follows the dots more closely than the straight line.' : 'It follows the dots a little more closely than the straight line. These data are noisy, so the gain is small.'))) + '</p><p style="font-size:12px;color:' + C.mid + '">A smaller typical miss means the curve is closer to the dots. These dots are the training compounds, so this shows how well the network fits what it studied.</p>';
    }
    rg.addEventListener('input', draw); sw.addEventListener('change', draw);
    async function play(tok){
      sw.checked = false; rg.value = 1; draw(); api.cap('One neuron: a flat piece and then a slope. That is one bend.'); await api.sleep(3000);
      await api.anim(3600, function(t){ rg.value = Math.round(1 + 7 * t); draw(); }); api.cap('More neurons give more bends, so the curve follows the dots more closely.'); await api.sleep(3000);
      sw.checked = true; draw(); api.cap('Without the switch the neurons cannot bend. Stacking them still gives a straight line.'); await api.sleep(3400); sw.checked = false; draw(); api.cap(null);
    }
    return { render: draw, play: play };
  } });
GUIDE.steps['nn/Why a network can bend'] = { notice: 'Without the switch the red curve is the same straight line, however many neurons there are.', q: 'What does the switch in each neuron do?', a: 'It keeps the neuron silent until its sum is above zero. Each neuron switches on at a different place, and that adds a bend to the curve.' };

/* ---------- 03 step: too big a network memorises ---------- */
var BN4 = { cache: {} };
LESSONS.nn.add({ level: 'beginner', short: 'A network that is too big', title: 'A network with too many neurons memorises',
  change: 'Choose the size of the network and compare the two scores.',
  watch: 'Here only 40 compounds are used to train. They are the filled dots. Another 90 compounds are held back as the test compounds, the hollow dots, and the network never studies them. A tiny network cannot bend enough and misses the pattern. A huge network has so many dials that it can pass through every filled dot, and the curve twists to do it. That is memorising. It scores very well on the practice compounds and badly on the test compounds. Score here is R2: 1 is perfect, and 0 is no better than always guessing the average. A score below zero is worse than guessing the average.',
  build: function(api){
    var SZ = [2, 6, 60], sel = 1, S = null, T = null;
    var cp = api.el('pn', 'left:28px;top:28px;width:560px;height:408px', '<h4>Lab value against logP</h4><canvas width="530" height="350"></canvas>');
    var tp = api.el('pn', 'left:608px;top:28px;width:324px;height:408px;overflow:auto', '<div class="mt4" style="font-size:13px;line-height:1.5;color:' + C.char + '"></div>');
    var ctl = api.addControl('Network size ' + bnSeg(['Tiny: 2 neurons', 'Medium: 6 neurons', 'Huge: 60 neurons'], sel, 'data-sz'));
    function setup(){
      S = bnData('b4', ['logP'], 40, 5); var te = bnData('b4t', ['logP'], 90, 19, D.TEST), key = EP + 'b4';
      T = { idx: te.idx, X: te.idx.map(function(i){ return [(D.COL.logP[i] - S.st[0].m) / S.st[0].s]; }), Y: te.idx.map(function(i){ return D.Y[i]; }) };
      if (!BN4.cache[key]) { BN4.cache[key] = SZ.map(function(k){ var m = bnInit(1, k, 11 + k, 2.2); bnTrain(m, S.X, S.Yz, k > 20 ? 6000 : (k > 4 ? 800 : 1500), k > 20 ? 0.01 : 0.03, true); var ptr = S.X.map(function(x, q){ return S.my + S.sy * bnRun(m, x).y; }), pte = T.X.map(function(x){ return S.my + S.sy * bnRun(m, x).y; }); return { m: m, tr: bnR2(ptr, S.idx.map(function(i){ return D.Y[i]; })), te: bnR2(pte, T.Y) }; }); }
      return BN4.cache[key];
    }
    function bar(label, v, col){ var w = clamp(v, 0, 1) * 100; return '<div style="margin:10px 0"><div style="display:flex;justify-content:space-between;align-items:baseline"><b>' + label + '</b><span class="big" style="font-size:26px;margin:0;color:' + col + '">' + v.toFixed(2) + '</span></div><div style="height:14px;background:' + C.light + ';border-radius:4px;overflow:hidden"><i style="display:block;height:14px;width:' + w + '%;background:' + col + ';transition:width .4s"></i></div>' + (v < 0 ? '<div style="font-size:11px;color:' + C.red + '">Below zero: worse than guessing the average.</div>' : '') + '</div>'; }
    function draw(){
      var R = setup(), r = R[sel], g = cv(qs('canvas', cp)), x = g.x, box = { l: 52, r: g.w - 14, t: 14, b: g.h - 40 }, zlo = -2.6, zhi = 2.6, ylo = D.ylo, yhi = D.yhi, sx = function(z){ return box.l + (z - zlo) / (zhi - zlo) * (box.r - box.l); }, sy = function(v){ return box.b - (v - ylo) / (yhi - ylo) * (box.b - box.t); };
      x.strokeStyle = C.silver; x.fillStyle = C.mid; x.font = '11px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'right'; niceTicks(ylo, yhi, 5).v.forEach(function(v){ x.beginPath(); x.moveTo(box.l, sy(v)); x.lineTo(box.r, sy(v)); x.stroke(); x.fillText(v.toFixed(1), box.l - 6, sy(v) + 4); });
      x.textAlign = 'center'; [-2, -1, 0, 1, 2].forEach(function(z){ x.fillText((S.st[0].m + S.st[0].s * z).toFixed(1), sx(z), box.b + 16); }); x.fillStyle = C.char; x.font = '600 12px Aptos, "Segoe UI", sans-serif'; x.fillText('logP (how oily the compound is)', (box.l + box.r) / 2, box.b + 34); x.save(); x.translate(13, (box.t + box.b) / 2); x.rotate(-Math.PI / 2); x.fillText(unitWord(), 0, 0); x.restore();
      x.strokeStyle = 'rgba(52,55,63,.5)'; x.lineWidth = 1.5; T.idx.forEach(function(i, q){ var px = sx(clamp(T.X[q][0], zlo, zhi)); x.beginPath(); x.arc(px, sy(D.Y[i]), 3, 0, 6.3); x.stroke(); });
      x.strokeStyle = C.red; x.lineWidth = 3.2; x.beginPath(); for (var z = zlo; z <= zhi + 1e-9; z += 0.04) { var py = sy(clamp(S.my + S.sy * bnRun(r.m, [z]).y, ylo - 1.5, yhi + 1.5)); z === zlo ? x.moveTo(sx(z), py) : x.lineTo(sx(z), py); } x.stroke();
      x.fillStyle = C.char; S.idx.forEach(function(i, q){ x.beginPath(); x.arc(sx(S.X[q][0]), sy(D.Y[i]), 4.2, 0, 6.3); x.fill(); });
      x.textAlign = 'left'; x.font = '11px Aptos, "Segoe UI", sans-serif'; x.fillStyle = C.mid; x.fillText('filled: 40 training compounds', box.l + 6, box.t + 12); x.fillText('hollow: 90 test compounds', box.l + 6, box.t + 26);
      var msg = sel === 0 ? 'Two neurons give only a gentle bend. The curve misses much of the pattern, so both scores are low.' : (sel === 1 ? 'Six neurons can follow the main trend without chasing every dot. The test score is low because lab values are noisy, but it holds up.' : 'Sixty neurons have more dials than there are compounds, so the curve can bend to touch the filled dots.' + (r.tr > r.te ? ' The practice score is far above the test score.' : ''));
      qs('.mt4', tp).innerHTML = '<div class="ol">' + SZ[sel] + ' neurons</div>' + bar('Practice compounds', r.tr, C.char) + bar('Test compounds', r.te, C.red) + '<p style="margin-top:10px">' + msg + '</p><p style="font-size:12px;color:' + C.mid + '">What matters is the score on the test compounds. A network that is only good at the compounds it studied is not useful for new ones.</p>';
    }
    qsa('[data-sz] button', ctl).forEach(function(b){ b.addEventListener('click', function(){ sel = +b.dataset.v; qsa('[data-sz] button', ctl).forEach(function(q){ q.classList.toggle('on', q === b); }); draw(); }); });
    async function play(tok){ for (var k = 0; k < 3; k++) { api.checkTok(tok); sel = k; qsa('[data-sz] button', ctl).forEach(function(q){ q.classList.toggle('on', +q.dataset.v === k); }); draw(); api.cap(k === 0 ? 'A tiny network: too simple to follow the pattern.' : (k === 1 ? 'A medium network follows the trend.' : 'A huge network threads through every filled dot, and then it does worse on the hollow dots.')); await api.sleep(3400); } sel = 1; qsa('[data-sz] button', ctl).forEach(function(q){ q.classList.toggle('on', +q.dataset.v === 1); }); draw(); api.cap(null); }
    return { render: draw, play: play };
  } });
GUIDE.steps['nn/A network that is too big'] = { notice: 'The huge network passes through the filled dots, and the hollow dots are far from its curve.', q: 'Why is a high practice score not enough?', a: 'A network can score well on compounds it has studied by memorising them. Only the score on test compounds shows whether it learned a pattern that holds for new ones.' };

/* ---------- 09 shared: one dial and a hill of misses ---------- */
function bhData(){
  var key = EP + 'bh'; if (D.cache[key]) return D.cache[key]; var S = bnData('bh', ['logP'], 260, 31), n = S.X.length, mxy = 0, i;
  for (i = 0; i < n; i++) mxy += S.X[i][0] * S.Yz[i]; mxy /= n; var myy = mean(S.Yz.map(function(v){ return v * v; })), mxx = mean(S.X.map(function(v){ return v[0] * v[0]; }));
  S.loss = function(w){ var s = 0; for (var q = 0; q < n; q++) { var e = w * S.X[q][0] - S.Yz[q]; s += e * e; } return s / n; };
  S.best = mxy / mxx; return (D.cache[key] = S);
}
function bhCurve(el, S, o){
  o = o || {}; var g = cv(el), x = g.x, box = { l: 52, r: g.w - 16, t: 16, b: g.h - 40 }, wlo = -3, whi = 3, lmax = S.loss(wlo) * 1.02, sx = function(w){ return box.l + (w - wlo) / (whi - wlo) * (box.r - box.l); }, sy = function(l){ return box.b - Math.min(l, lmax * 1.3) / lmax * (box.b - box.t); };
  x.strokeStyle = C.silver; x.fillStyle = C.mid; x.font = '11px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'center'; [-3, -2, -1, 0, 1, 2, 3].forEach(function(w){ x.beginPath(); x.moveTo(sx(w), box.t); x.lineTo(sx(w), box.b); x.stroke(); x.fillText(w, sx(w), box.b + 16); });
  x.fillStyle = C.char; x.font = '600 12px Aptos, "Segoe UI", sans-serif'; x.fillText('the dial setting', (box.l + box.r) / 2, box.b + 34); x.save(); x.translate(13, (box.t + box.b) / 2); x.rotate(-Math.PI / 2); x.fillText('how big the misses are', 0, 0); x.restore();
  x.textAlign = 'right'; x.fillStyle = C.mid; x.font = '11px Aptos, "Segoe UI", sans-serif'; x.fillText('higher is worse', box.r, box.t + 12);
  x.strokeStyle = C.char; x.lineWidth = 3; x.beginPath(); for (var w = wlo; w <= whi + 1e-9; w += 0.03) { var py = sy(S.loss(w)); w === wlo ? x.moveTo(sx(w), py) : x.lineTo(sx(w), py); } x.stroke();
  if (o.trail) { x.strokeStyle = 'rgba(225,36,42,.45)'; x.lineWidth = 1.5; x.beginPath(); o.trail.forEach(function(w, i){ var px = sx(clamp(w, wlo, whi)), p2 = sy(S.loss(clamp(w, wlo, whi))); i ? x.lineTo(px, p2) : x.moveTo(px, p2); }); x.stroke(); o.trail.forEach(function(w){ x.fillStyle = 'rgba(225,36,42,.35)'; x.beginPath(); x.arc(sx(clamp(w, wlo, whi)), sy(S.loss(clamp(w, wlo, whi))), 3, 0, 6.3); x.fill(); }); }
  if (o.marks) o.marks.forEach(function(m){ x.fillStyle = m.col; x.beginPath(); x.arc(sx(m.w), sy(S.loss(m.w)), 5, 0, 6.3); x.fill(); if (m.label) { x.fillStyle = C.char; x.font = '600 11px Aptos, "Segoe UI", sans-serif'; x.textAlign = m.dx > 0 ? 'left' : (m.dx < 0 ? 'right' : 'center'); x.fillText(m.label, sx(m.w) + (m.dx || 0), sy(S.loss(m.w)) + (m.dx ? 4 : (m.below ? 18 : -10))); } });
  var w0 = o.w, off = Math.abs(w0) > 3.2, wc = clamp(w0, wlo, whi), bx = sx(wc), by = sy(S.loss(wc));
  if (o.arrow !== false && !off) { var sl = 2 * (w0 - S.best), ang = Math.atan(sl * (box.b - box.t) / lmax / ((box.r - box.l) / (whi - wlo))), L = 70; x.strokeStyle = C.red; x.lineWidth = 2.5; x.beginPath(); x.moveTo(bx - Math.cos(ang) * L / 2, by + Math.sin(ang) * L / 2); x.lineTo(bx + Math.cos(ang) * L / 2, by - Math.sin(ang) * L / 2); x.stroke(); }
  x.fillStyle = off ? C.mid : C.red; x.strokeStyle = '#fff'; x.lineWidth = 2; x.beginPath(); x.arc(bx, by, 9, 0, 6.3); x.fill(); x.stroke();
  g.sx = sx; g.sy = sy; g.box = box; g.wlo = wlo; g.whi = whi; return g;
}

/* ---------- 09 step: rolling downhill ---------- */
LESSONS.netalgo.add({ level: 'beginner', short: 'Rolling downhill', title: 'Training is rolling a ball downhill',
  change: 'Choose a step size, then press Take one step or Roll 25 steps. Press Start again to try another size.',
  watch: 'Training a model means finding dial settings that make the misses small. Picture a hill. The position of the ball across is the setting of one dial, and the height of the hill is how big the misses are, the loss. The best setting is the bottom of the valley. The red line shows how steep the hill is where the ball sits. The rule for training is simple: roll the ball a little way downhill, then look at the slope again. The step size, called the learning rate, matters. With tiny steps the ball takes ages to arrive. With steps that are just right it arrives fast. With steps that are too big it jumps across the valley and can fly off. The hill here is built from 260 real compounds: the dial is how strongly logP counts toward the prediction.',
  build: function(api){
    var etas = [0.04, 0.4, 1.05], names = ['Small steps', 'Steps that are just right', 'Steps that are too big'], sel = 1, w = -2.4, steps = 0, trail = [-2.4], S = null, run = 0;
    var cp = api.el('pn', 'left:28px;top:28px;width:600px;height:408px', '<h4>How big the misses are, for each dial setting</h4><canvas width="570" height="350"></canvas>');
    var tp = api.el('pn', 'left:648px;top:28px;width:284px;height:408px;overflow:auto', '<div class="rt" style="font-size:13px;line-height:1.5;color:' + C.char + '"></div>');
    var ctl = api.addControl('Step size ' + bnSeg(names, sel, 'data-e') + ' <button type="button" class="btn sm" data-one>Take one step</button> <button type="button" class="btn sm" data-roll>Roll 25 steps</button> <button type="button" class="btn sm ghost" data-reset>Start again</button>');
    function slope(v){ return 2 * (v - S.best); }
    function draw(){
      S = bhData(); var lo = S.loss(w), sl = slope(w), off = Math.abs(w) > 6;
      bhCurve(qs('canvas', cp), S, { w: w, trail: trail });
      qs('.rt', tp).innerHTML = '<div class="ol">Where the ball is</div><div style="display:flex;gap:18px"><div><div class="big" style="font-size:28px;color:' + C.red + '">' + (off ? 'off' : w.toFixed(2)) + '</div><div style="font-size:12px">dial setting</div></div><div><div class="big" style="font-size:28px">' + (off ? '-' : lo.toFixed(2)) + '</div><div style="font-size:12px">misses</div></div></div><p style="margin-top:10px">Steps taken: <b>' + steps + '</b></p>' + (off ? '<p style="color:' + C.red + ';font-weight:700">The ball flew off the hill. The steps were too big.</p>' : '<p>' + (Math.abs(sl) < 0.05 ? 'The ground is flat here. The ball is at the bottom.' : 'The hill slopes ' + (sl > 0 ? 'up to the right, so the ball rolls left.' : 'up to the left, so the ball rolls right.')) + '</p>') + '<p style="font-size:12px;color:' + C.mid + '">The best dial setting is about <b>' + S.best.toFixed(2) + '</b>. The rule for each step: new setting = old setting minus the step size times the slope.</p>';
    }
    function once(){ if (Math.abs(w) > 6) return; w = w - etas[sel] * slope(w); steps++; trail.push(w); if (trail.length > 60) trail.shift(); }
    function reset(){ run++; w = -2.4; steps = 0; trail = [w]; draw(); }
    qsa('[data-e] button', ctl).forEach(function(b){ b.addEventListener('click', function(){ sel = +b.dataset.v; qsa('[data-e] button', ctl).forEach(function(q){ q.classList.toggle('on', q === b); }); reset(); }); });
    qs('[data-one]', ctl).addEventListener('click', function(){ once(); draw(); });
    qs('[data-reset]', ctl).addEventListener('click', reset);
    qs('[data-roll]', ctl).addEventListener('click', async function(){ var id = ++run; for (var i = 0; i < 25; i++) { if (id !== run) return; once(); draw(); await delay(ANIM ? 140 : 0); } });
    async function play(tok){
      for (var k = 0; k < 3; k++) { api.checkTok(tok); sel = k; qsa('[data-e] button', ctl).forEach(function(q){ q.classList.toggle('on', +q.dataset.v === k); }); reset(); api.cap(names[k] + ': watch the ball.'); var id = ++run; for (var i = 0; i < 22; i++) { api.checkTok(tok); if (id !== run) return; once(); draw(); await api.sleep(150); } await api.sleep(900); }
      sel = 1; qsa('[data-e] button', ctl).forEach(function(q){ q.classList.toggle('on', +q.dataset.v === 1); }); reset(); api.cap(null);
    }
    return { render: function(){ draw(); }, play: play };
  } });
GUIDE.steps['netalgo/Rolling downhill'] = { notice: 'Small steps creep, good steps arrive, and big steps overshoot.', q: 'Why can steps that are too big make things worse?', a: 'Each step jumps past the bottom of the valley to a point that is even higher, so the misses grow instead of shrinking.' };

/* ---------- 09 step: which way and how far ---------- */
LESSONS.netalgo.add({ level: 'beginner', short: 'Which way to turn the dial', title: 'The slope says which way to turn, and how far',
  change: 'Drag the dial. Read what happens to the misses when the dial is nudged left and right.',
  watch: 'How does the computer know which way is downhill? It nudges the dial. It moves the dial a tiny bit to the right and checks how big the misses are, then a tiny bit to the left and checks again. If the misses grow when the dial moves right, the hill rises to the right, so the dial should turn left. The size of the change tells how steep the hill is. A steep hill means the dial is far from the best setting, so a big turn is safe. A flat hill means the dial is close, so only a small turn is needed. This difference in misses, divided by the size of the nudge, is called the slope, or gradient.',
  build: function(api){
    var w = -1.8, S = null, NUDGE = 0.2;
    var cp = api.el('pn', 'left:28px;top:28px;width:600px;height:408px', '<h4>How big the misses are, for each dial setting</h4><canvas width="570" height="350"></canvas>');
    var tp = api.el('pn', 'left:648px;top:28px;width:284px;height:408px;overflow:auto', '<div class="rt" style="font-size:13px;line-height:1.5;color:' + C.char + '"></div>');
    var ctl = api.addControl('The dial <input type="range" min="-3" max="3" step="0.01" value="-1.8" style="width:300px"> <b class="wv"></b> <button type="button" class="btn sm" data-turn>Turn it the way the slope says</button>');
    var rg = qs('input', ctl), wv = qs('.wv', ctl);
    function draw(){
      S = bhData(); w = +rg.value; wv.textContent = w.toFixed(2); var a = S.loss(w - NUDGE), b = S.loss(w + NUDGE), c = S.loss(w), sl = (b - a) / (2 * NUDGE);
      bhCurve(qs('canvas', cp), S, { w: w, arrow: false, marks: [{ w: Math.max(-3, w - NUDGE), col: '#6F8196', label: 'left', dx: -10 }, { w: Math.min(3, w + NUDGE), col: C.dred, label: 'right', dx: 10 }] });
      var flat = Math.abs(sl) < 0.08, dir = sl > 0 ? 'left' : 'right';
      qs('.rt', tp).innerHTML = '<div class="ol">Nudge the dial by ' + NUDGE.toFixed(1) + '</div><div style="display:grid;grid-template-columns:1fr auto;gap:2px 10px;font-size:13px"><span>Misses now</span><b>' + c.toFixed(2) + '</b><span>Misses after nudging left</span><b style="color:#4B5A6B">' + a.toFixed(2) + '</b><span>Misses after nudging right</span><b style="color:' + C.dred + '">' + b.toFixed(2) + '</b></div><p style="margin-top:10px">' + (flat ? 'The two nudges give almost the same misses. The ground is flat, so the dial is at the bottom of the valley.' : 'Nudging ' + (b > a ? 'right makes' : 'left makes') + ' the misses <b>grow</b>, and nudging ' + (b > a ? 'left makes' : 'right makes') + ' them <b>shrink</b>. So turn the dial to the <b style="color:' + C.red + '">' + dir + '</b>.') + '</p><div class="ol" style="margin-top:8px">How steep</div><div style="height:12px;background:' + C.light + ';border-radius:3px;overflow:hidden"><i style="display:block;height:12px;width:' + clamp(Math.abs(sl) / 6 * 100, 2, 100) + '%;background:' + C.red + '"></i></div><p style="font-size:12px;margin-top:6px">' + (Math.abs(sl) > 2.5 ? 'Very steep: far from the bottom, so a big turn is safe.' : (flat ? 'Flat.' : 'Gentle: close to the bottom, so turn only a little.')) + '</p>';
    }
    rg.addEventListener('input', draw);
    qs('[data-turn]', ctl).addEventListener('click', function(){ S = bhData(); var sl = (S.loss(+rg.value + NUDGE) - S.loss(+rg.value - NUDGE)) / (2 * NUDGE), from = +rg.value, to = clamp(from - 0.4 * sl, -3, 3); animateFree(900, function(t){ rg.value = from + (to - from) * t; draw(); }); });
    async function play(tok){
      rg.value = -1.8; draw(); api.cap('Start with the dial on the left side of the hill.'); await api.sleep(2800);
      api.cap('The misses are lower when we nudge the dial right, so right is downhill.'); await api.sleep(3200);
      var to = bhData().best; await api.anim(2800, function(t){ rg.value = -1.8 + (to + 1.8) * t; draw(); }); api.cap('Near the bottom the ground is flat, and the two nudges agree.'); await api.sleep(3000);
      rg.value = 2.4; draw(); api.cap('From the far side, nudging left lowers the misses, so the dial should turn left.'); await api.sleep(3200); api.cap(null);
    }
    return { render: draw, play: play };
  } });
GUIDE.steps['netalgo/Which way to turn the dial'] = { notice: 'The side where the misses are smaller is the way downhill. Near the bottom both sides agree.', q: 'How does a computer find out which way is downhill?', a: 'It nudges the dial a little each way and sees which nudge makes the misses smaller.' };

/* ---------- 09 step: counting the dials ---------- */
LESSONS.netalgo.add({ level: 'beginner', short: 'Counting the dials', title: 'A bigger network has more dials, and needs more examples',
  change: 'Press the plus and minus buttons to change the number of neurons in each layer, and read the count of dials.',
  watch: 'Every dial a network can turn is called a parameter. There are two kinds: a weight on each line between neurons, and a base amount on each neuron. Count them. The network here reads 2 features, passes them through a first layer of neurons, optionally a second layer, and ends in one output. Add neurons and the count of dials grows fast, because every neuron in one layer connects to every neuron in the next. Dials need examples to set them. The chart shows a real test: networks of different sizes learned a 3D shape, a double helix like DNA, from 100, 1,000 or 10,000 examples. A score of 0.5 is guessing and 1.0 is perfect. With few examples even a huge network could not use its dials.',
  build: function(api){
    var n1 = 3, n2 = 0, J = null;
    var dp = api.el('pn', 'left:28px;top:28px;width:500px;height:408px', '<h4>The network</h4><canvas width="470" height="230"></canvas><div class="cn" style="font-size:12.5px;line-height:1.5;color:' + C.char + '"></div>');
    var rp = api.el('pn', 'left:548px;top:28px;width:384px;height:408px;overflow:auto', '<div class="cr" style="font-size:13px;line-height:1.5;color:' + C.char + '"></div>');
    var ctl = api.addControl('First layer <button type="button" class="btn sm ghost" data-a="-1">-</button> <b class="a1"></b> <button type="button" class="btn sm ghost" data-a="1">+</button> neurons &nbsp; Second layer <button type="button" class="btn sm ghost" data-b="-1">-</button> <b class="a2"></b> <button type="button" class="btn sm ghost" data-b="1">+</button> neurons');
    function count(){ var w1 = 2 * n1, b1 = n1, w2 = n2 ? n1 * n2 : 0, b2 = n2, wo = (n2 || n1), bo = 1; return { w1: w1, b1: b1, w2: w2, b2: b2, wo: wo, bo: bo, total: w1 + b1 + w2 + b2 + wo + bo }; }
    function drawNet(){
      var g = cv(qs('canvas', dp)), x = g.x, layers = [2, n1].concat(n2 ? [n2] : []).concat([1]), cols = layers.length, xs = layers.map(function(_, i){ return 50 + i * (g.w - 100) / (cols - 1); });
      function ys(c, i){ var n = layers[c], gap = Math.min(34, (g.h - 40) / n); return g.h / 2 + (i - (n - 1) / 2) * gap; }
      x.lineWidth = 1; for (var c = 0; c < cols - 1; c++) for (var i = 0; i < layers[c]; i++) for (var j = 0; j < layers[c + 1]; j++) { x.strokeStyle = 'rgba(111,129,150,.45)'; x.beginPath(); x.moveTo(xs[c], ys(c, i)); x.lineTo(xs[c + 1], ys(c + 1, j)); x.stroke(); }
      for (c = 0; c < cols; c++) for (i = 0; i < layers[c]; i++) { x.fillStyle = c === 0 ? '#fff' : (c === cols - 1 ? '#FBE9E9' : '#EDF2F4'); x.strokeStyle = C.char; x.lineWidth = 1.6; x.beginPath(); x.arc(xs[c], ys(c, i), 7, 0, 6.3); x.fill(); x.stroke(); }
      x.fillStyle = C.mid; x.font = '600 11px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'center'; x.fillText('2 features', xs[0], g.h - 6); x.fillText('first layer', xs[1], g.h - 6); if (n2) x.fillText('second layer', xs[2], g.h - 6); x.fillText('output', xs[cols - 1], g.h - 6);
    }
    function near(params, runs){ var best = 0, bd = 1e9; runs.forEach(function(r, i){ var d = Math.abs(Math.log(r.params) - Math.log(params)); if (d < bd) { bd = d; best = i; } }); return best; }
    function draw(){
      qs('.a1', ctl).textContent = n1; qs('.a2', ctl).textContent = n2 || 'none'; var c = count(); drawNet();
      qs('.cn', dp).innerHTML = '<div><b>Weights</b> (one per line): 2 times ' + n1 + ' = ' + c.w1 + (n2 ? ', then ' + n1 + ' times ' + n2 + ' = ' + c.w2 + ', then ' + n2 + ' to the output = ' + c.wo : ', then ' + n1 + ' to the output = ' + c.wo) + '.</div><div><b>Base amounts</b> (one per neuron): ' + (n2 ? n1 + ' + ' + n2 + ' + 1 output = ' + (c.b1 + c.b2 + c.bo) : n1 + ' + 1 output = ' + (c.b1 + c.bo)) + '.</div><div style="margin-top:4px">Total dials: <b style="color:' + C.red + ';font-size:20px">' + c.total.toLocaleString() + '</b></div>';
      var html = '<div class="ol">A real test of network size</div>';
      if (J) {
        html += '<p style="font-size:12px">Score on a 3D shape that no network had studied, for the tested network nearest to yours in size. 0.5 is guessing.</p>';
        ['100', '1000', '10000'].forEach(function(sz){ var runs = J.runs[sz], i = near(c.total, runs), r = runs[i], acc = r.ck[r.ck.length - 1].acc, w = clamp((acc - 0.5) / 0.5, 0, 1) * 100; html += '<div style="margin:11px 0"><div style="display:flex;justify-content:space-between"><b>' + Number(sz).toLocaleString() + ' examples</b><span class="big" style="font-size:22px;margin:0;color:' + (acc > 0.8 ? C.red : C.char) + '">' + acc.toFixed(2) + '</span></div><div style="height:14px;background:' + C.light + ';border-radius:4px;overflow:hidden"><i style="display:block;height:14px;width:' + Math.max(w, 1.5) + '%;background:' + (acc > 0.8 ? C.red : C.bg) + '"></i></div><div style="font-size:11px;color:' + C.mid + '">tested network with ' + r.params.toLocaleString() + ' dials</div></div>'; });
        html += '<p style="margin-top:8px">With 100 examples, no network does well, however many dials it has. With 10,000 examples a network with thousands of dials does very well. More dials only help when there are enough examples to set them.</p>';
      } else html += '<p>Loading the saved test.</p>';
      qs('.cr', rp).innerHTML = html;
    }
    function bump(which, d){ if (which === 'a') n1 = clamp(n1 + d, 1, 12); else n2 = clamp(n2 + d, 0, 12); draw(); }
    qsa('[data-a]', ctl).forEach(function(b){ b.addEventListener('click', function(){ bump('a', +b.dataset.a); }); });
    qsa('[data-b]', ctl).forEach(function(b){ b.addEventListener('click', function(){ bump('b', +b.dataset.b); }); });
    async function render(){ draw(); if (!J) { J = await loadJSON('scaling'); draw(); } }
    async function play(tok){
      await render(); n1 = 2; n2 = 0; draw(); api.cap('Two features and two neurons. Count the lines and the neurons: each one is a dial.'); await api.sleep(3200);
      n1 = 6; draw(); api.cap('Six neurons: more lines, so many more dials.'); await api.sleep(2600); n2 = 6; draw(); api.cap('Add a second layer and every neuron links to every neuron in the next layer. The count jumps.'); await api.sleep(3400);
      n1 = 12; n2 = 12; draw(); api.cap('Twelve and twelve. Dials need examples to set them. Look at the chart for what happens with only 100 examples.'); await api.sleep(3600); n1 = 3; n2 = 0; draw(); api.cap(null);
    }
    return { render: render, play: play };
  } });
GUIDE.steps['netalgo/Counting the dials'] = { notice: 'Adding a second layer makes the number of dials jump.', q: 'Why does a bigger network need more examples?', a: 'Each dial has to be set by the examples. With too few examples the extra dials just memorise them and do not learn a pattern.' };
