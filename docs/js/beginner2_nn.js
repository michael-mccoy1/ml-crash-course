"use strict";
/* Beginner steps for lessons 03 (neural networks) and 09 (network algorithms).
   Written for chemists with no machine learning background. Each step assumes only the earlier beginner steps. */

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
LESSONS.nn.add({ level: 'beginner', short: 'Neurons side by side', title: 'Hidden layers',
  change: 'Select a neuron to switch it off, and select it again to switch it on. Press Next compound to change the compound.',
  watch: 'A hidden layer applies several neurons in parallel to the same inputs. Each hidden neuron computes a weighted sum of the features plus a bias, then applies ReLU: output = max(0, sum). The output neuron forms a weighted sum of the hidden outputs plus a bias, and that sum is the prediction. Here four hidden neurons receive two features, logP and TPSA, standardised to zero mean and unit variance. Line colour gives the sign of each weight and line width its magnitude. Switching a hidden neuron off removes its contribution to the output. The scatter plot shows predicted against measured values for the 260 training compounds. The weights were fitted to those compounds with the Adam optimiser.',
  build: function(api){
    var dg = api.el('pn', 'left:28px;top:28px;width:520px;height:408px', '<h4>Network with one hidden layer</h4><canvas class="dg" width="490" height="236"></canvas><div class="vt" style="font-size:12.5px;line-height:1.45;color:' + C.char + '"></div>');
    var sc = api.el('pn', 'left:568px;top:28px;width:364px;height:408px', '<h4>Predicted against measured, 260 training compounds</h4><canvas class="sc" width="334" height="300"></canvas><div class="sn" style="font-size:12.5px;margin-top:4px;color:' + C.char + '"></div>');
    var ctl = api.addControl('Hidden neurons <label><input type="checkbox" data-n="0" checked> 1</label> <label><input type="checkbox" data-n="1" checked> 2</label> <label><input type="checkbox" data-n="2" checked> 3</label> <label><input type="checkbox" data-n="3" checked> 4</label> <button type="button" class="btn sm ghost" data-next>Next compound</button>');
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
      x.textAlign = 'left'; x.fillStyle = C.mid; x.font = '11px Aptos, "Segoe UI", sans-serif'; x.fillText('Line width: weight magnitude. Red: positive weight. Blue-grey: negative weight.', 4, g.h - 3);
      var base = S.my + S.sy * M.p[M.k * 2 + 2 * M.k], meas = D.Y[S.idx[q]];
      qs('.vt', dg).innerHTML = '<div><b>This compound</b>: measured value ' + meas.toFixed(2) + '. Output bias <b>' + base.toFixed(2) + '</b>' + votes.map(function(v, j){ return ', hidden neuron ' + (j + 1) + ' <b style="color:' + (!on[j] ? C.mid : (v >= 0 ? C.dred : '#4B5A6B')) + '">' + (!on[j] ? 'off' : (v >= 0 ? '+' : '') + (S.sy * v).toFixed(2)) + '</b>'; }).join('') + '. Prediction: <b style="color:' + C.red + '">' + (S.my + S.sy * y).toFixed(2) + '</b>.</div>';
      scatterPV(qs('.sc', sc), pr.p, S.idx, '', { max: 260 });
      var offN = on.filter(function(v){ return !v; }).length;
      qs('.sn', sc).innerHTML = 'RMSE: <b style="color:' + C.red + '">' + pr.rm.toFixed(2) + '</b> ' + esc(unitWord()) + ' units. ' + (offN === 0 ? 'All four hidden neurons are on.' : offN + ' hidden neuron' + (offN > 1 ? 's are' : ' is') + ' off. The output sums the remaining contributions.');
    }
    boxes.forEach(function(b){ b.addEventListener('change', function(){ on[+b.dataset.n] = b.checked ? 1 : 0; draw(); }); });
    qs('[data-next]', ctl).addEventListener('click', function(){ cmp = Math.floor(Math.random() * 260); draw(); });
    qs('.dg', dg).addEventListener('click', function(e){ var c = qs('.dg', dg), rc = c.getBoundingClientRect(), px = (e.clientX - rc.left) / rc.width * 490, py = (e.clientY - rc.top) / rc.height * 236; [30, 86, 142, 198].forEach(function(yy, j){ if (Math.hypot(px - 245, py - yy) < 22) { on[j] = on[j] ? 0 : 1; boxes[j].checked = !!on[j]; draw(); } }); });
    function render(){ prep(); draw(); }
    async function play(tok){
      on = [1, 1, 1, 1]; boxes.forEach(function(b){ b.checked = true; }); render(); api.cap('Four hidden neurons receive the same two features. Each has its own weights, so each produces a different output.'); await api.sleep(3400);
      for (var j = 0; j < 4; j++) { api.checkTok(tok); on[j] = 0; boxes[j].checked = false; draw(); api.cap('Hidden neuron ' + (j + 1) + ' is switched off. The prediction and the RMSE change.'); await api.sleep(2200); on[j] = 1; boxes[j].checked = true; draw(); await api.sleep(500); }
      api.cap('The output neuron sums the weighted hidden outputs. Compare the RMSE with all four on against each neuron removed.'); await api.sleep(3000); api.cap(null);
    }
    return { render: render, play: play };
  } });
GUIDE.steps['nn/Neurons side by side'] = { notice: 'Switching a hidden neuron off removes its contribution and changes the RMSE.', q: 'What does one hidden neuron contribute to the prediction?', a: 'Its ReLU output multiplied by its output weight. The prediction is the sum of these contributions plus the output bias.' };

/* ---------- 03 step: the bend ---------- */
LESSONS.nn.add({ level: 'beginner', short: 'Why a network can bend', title: 'Nonlinearity',
  change: 'Drag the slider to change the number of hidden neurons. Select the checkbox to remove ReLU.',
  watch: 'A linear model predicts a straight line. A neuron with ReLU, max(0, w x + b), is flat until its weighted sum becomes positive and then rises linearly, so it contributes one bend. A hidden layer with k neurons places up to k bends at different positions, and the output layer adds their weighted outputs. Each point is a compound: logP on the horizontal axis and the measured value on the vertical axis. The dashed line is the least-squares line. The red curve is the network. Selecting the checkbox replaces ReLU with the identity function. The network then reduces to a linear model, whatever the number of neurons.',
  build: function(api){
    var cp = api.el('pn', 'left:28px;top:28px;width:620px;height:408px', '<h4>Measured value against logP, 260 training compounds</h4><canvas width="590" height="352"></canvas>');
    var tp = api.el('pn', 'left:668px;top:28px;width:264px;height:408px;overflow:auto', '<div class="bt" style="font-size:13px;line-height:1.5;color:' + C.char + '"></div>');
    var ctl = api.addControl('Hidden neurons <input type="range" min="1" max="8" step="1" value="3" style="width:200px"> <b class="kv">3</b> <label><input type="checkbox" data-sw> Remove ReLU (linear activation)</label>');
    var rg = qs('input[type=range]', ctl), kv = qs('.kv', ctl), sw = qs('[data-sw]', ctl), S = null, line = null;
    function net(k){ var key = EP + 'b3net' + k; if (!D.cache[key]) { var m = bnInit(1, k, 3 + k, 2.2); bnTrain(m, S.X, S.Yz, 1500, 0.03, true); D.cache[key] = m; } return D.cache[key]; }
    function draw(){
      S = bnData('b3', ['logP'], 260, 4); line = bnLine(S.X, S.Yz); var k = +rg.value, relu = !sw.checked, m = net(k); kv.textContent = k;
      var g = cv(qs('canvas', cp)), x = g.x, box = { l: 52, r: g.w - 14, t: 14, b: g.h - 40 }, zlo = -2.6, zhi = 2.6, ylo = D.ylo, yhi = D.yhi, sx = function(z){ return box.l + (z - zlo) / (zhi - zlo) * (box.r - box.l); }, sy = function(v){ return box.b - (v - ylo) / (yhi - ylo) * (box.b - box.t); };
      x.strokeStyle = C.silver; x.fillStyle = C.mid; x.font = '11px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'right'; niceTicks(ylo, yhi, 5).v.forEach(function(v){ x.beginPath(); x.moveTo(box.l, sy(v)); x.lineTo(box.r, sy(v)); x.stroke(); x.fillText(v.toFixed(1), box.l - 6, sy(v) + 4); });
      x.textAlign = 'center'; [-2, -1, 0, 1, 2].forEach(function(z){ x.fillText((S.st[0].m + S.st[0].s * z).toFixed(1), sx(z), box.b + 16); });
      x.fillStyle = C.char; x.font = '600 12px Aptos, "Segoe UI", sans-serif'; x.fillText('logP', (box.l + box.r) / 2, box.b + 34); x.save(); x.translate(13, (box.t + box.b) / 2); x.rotate(-Math.PI / 2); x.fillText(unitWord(), 0, 0); x.restore();
      x.fillStyle = 'rgba(52,55,63,.4)'; S.idx.forEach(function(i, q){ x.beginPath(); x.arc(sx(S.X[q][0]), sy(D.Y[i]), 2.6, 0, 6.3); x.fill(); });
      function lineAt(z){ return S.my + S.sy * (line.w * z + line.b); }
      x.strokeStyle = C.mid; x.lineWidth = 2; x.setLineDash([6, 5]); x.beginPath(); x.moveTo(sx(zlo), sy(lineAt(zlo))); x.lineTo(sx(zhi), sy(lineAt(zhi))); x.stroke(); x.setLineDash([]);
      var f = function(z){ return relu ? S.my + S.sy * bnRun(m, [z]).y : lineAt(z); };
      x.strokeStyle = C.red; x.lineWidth = 3.4; x.beginPath(); for (var z = zlo; z <= zhi + 1e-9; z += 0.05) { var px = sx(z), py = sy(clamp(f(z), ylo - 1, yhi + 1)); z === zlo ? x.moveTo(px, py) : x.lineTo(px, py); } x.stroke();
      var seL = 0, seN = 0; S.idx.forEach(function(i, q){ seL += (lineAt(S.X[q][0]) - D.Y[i]) * (lineAt(S.X[q][0]) - D.Y[i]); seN += (f(S.X[q][0]) - D.Y[i]) * (f(S.X[q][0]) - D.Y[i]); });
      var rl = Math.sqrt(seL / S.idx.length), rn = Math.sqrt(seN / S.idx.length);
      qs('.bt', tp).innerHTML = '<div class="ol">RMSE on the training compounds</div><div style="display:flex;gap:18px"><div><div class="big" style="font-size:28px;color:' + C.mid + '">' + rl.toFixed(2) + '</div><div style="font-size:12px">linear fit</div></div><div><div class="big" style="font-size:28px;color:' + C.red + '">' + rn.toFixed(2) + '</div><div style="font-size:12px">network</div></div></div><p style="margin-top:12px">' + (!relu ? 'ReLU is removed, so the network reduces to the linear fit. It has ' + k + ' hidden neuron' + (k > 1 ? 's' : '') + ' and no bends.' : (k === 1 ? 'One ReLU neuron gives one bend: a flat segment followed by a linear segment.' : 'With ' + k + ' neurons the curve can have up to ' + k + ' bends. ' + (rl - rn > 0.02 ? 'The RMSE is lower than for the linear fit.' : 'The RMSE is slightly lower than for the linear fit. The gain is small because the data are noisy.'))) + '</p><p style="font-size:12px;color:' + C.mid + '">These are training compounds, so the RMSE measures fit to the training data and not generalisation.</p>';
    }
    rg.addEventListener('input', draw); sw.addEventListener('change', draw);
    async function play(tok){
      sw.checked = false; rg.value = 1; draw(); api.cap('One ReLU neuron: a flat segment and a linear segment, which is one bend.'); await api.sleep(3000);
      await api.anim(3600, function(t){ rg.value = Math.round(1 + 7 * t); draw(); }); api.cap('More hidden neurons add bends, so the curve can follow the training data more closely.'); await api.sleep(3000);
      sw.checked = true; draw(); api.cap('Without ReLU the layers compose into a linear function, so the output remains a straight line.'); await api.sleep(3400); sw.checked = false; draw(); api.cap(null);
    }
    return { render: draw, play: play };
  } });
GUIDE.steps['nn/Why a network can bend'] = { notice: 'With ReLU removed, the curve is the linear fit for any number of neurons.', q: 'Why does a network without ReLU reduce to a linear model?', a: 'A composition of weighted sums is itself a weighted sum, so stacked linear layers give a linear function. ReLU makes each neuron piecewise linear, and a sum of such neurons can bend.' };

/* ---------- 03 step: too big a network memorises ---------- */
var BN4 = { cache: {} };
LESSONS.nn.add({ level: 'beginner', short: 'A network that is too big', title: 'Capacity and overfitting',
  change: 'Select a network size and compare the training and test R2.',
  watch: 'Capacity is the range of functions a network can represent, and it grows with the number of weights. Here a network is trained on 40 compounds (filled points) and evaluated on 90 test compounds (open points) that were not used in training. A network with 2 hidden neurons has too little capacity and underfits. A network with 60 hidden neurons has more weights than training compounds and can pass close to every training point. That is overfitting: the training R2 is high and the test R2 is low. R2 is 1 for a perfect fit and 0 for a model no better than predicting the mean. R2 below 0 is worse than predicting the mean.',
  build: function(api){
    var SZ = [2, 6, 60], sel = 1, S = null, T = null;
    var cp = api.el('pn', 'left:28px;top:28px;width:560px;height:408px', '<h4>Measured value against logP</h4><canvas width="530" height="350"></canvas>');
    var tp = api.el('pn', 'left:608px;top:28px;width:324px;height:408px;overflow:auto', '<div class="mt4" style="font-size:13px;line-height:1.5;color:' + C.char + '"></div>');
    var ctl = api.addControl('Hidden neurons ' + bnSeg(['2', '6', '60'], sel, 'data-sz'));
    function setup(){
      S = bnData('b4', ['logP'], 40, 5); var te = bnData('b4t', ['logP'], 90, 19, D.TEST), key = EP + 'b4';
      T = { idx: te.idx, X: te.idx.map(function(i){ return [(D.COL.logP[i] - S.st[0].m) / S.st[0].s]; }), Y: te.idx.map(function(i){ return D.Y[i]; }) };
      if (!BN4.cache[key]) { BN4.cache[key] = SZ.map(function(k){ var m = bnInit(1, k, 11 + k, 2.2); bnTrain(m, S.X, S.Yz, k > 20 ? 6000 : (k > 4 ? 800 : 1500), k > 20 ? 0.01 : 0.03, true); var ptr = S.X.map(function(x, q){ return S.my + S.sy * bnRun(m, x).y; }), pte = T.X.map(function(x){ return S.my + S.sy * bnRun(m, x).y; }); return { m: m, tr: bnR2(ptr, S.idx.map(function(i){ return D.Y[i]; })), te: bnR2(pte, T.Y) }; }); }
      return BN4.cache[key];
    }
    function bar(label, v, col){ var w = clamp(v, 0, 1) * 100; return '<div style="margin:10px 0"><div style="display:flex;justify-content:space-between;align-items:baseline"><b>' + label + '</b><span class="big" style="font-size:26px;margin:0;color:' + col + '">' + v.toFixed(2) + '</span></div><div style="height:14px;background:' + C.light + ';border-radius:4px;overflow:hidden"><i style="display:block;height:14px;width:' + w + '%;background:' + col + ';transition:width .4s"></i></div>' + (v < 0 ? '<div style="font-size:11px;color:' + C.red + '">Below zero: worse than predicting the mean.</div>' : '') + '</div>'; }
    function draw(){
      var R = setup(), r = R[sel], g = cv(qs('canvas', cp)), x = g.x, box = { l: 52, r: g.w - 14, t: 14, b: g.h - 40 }, zlo = -2.6, zhi = 2.6, ylo = D.ylo, yhi = D.yhi, sx = function(z){ return box.l + (z - zlo) / (zhi - zlo) * (box.r - box.l); }, sy = function(v){ return box.b - (v - ylo) / (yhi - ylo) * (box.b - box.t); };
      x.strokeStyle = C.silver; x.fillStyle = C.mid; x.font = '11px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'right'; niceTicks(ylo, yhi, 5).v.forEach(function(v){ x.beginPath(); x.moveTo(box.l, sy(v)); x.lineTo(box.r, sy(v)); x.stroke(); x.fillText(v.toFixed(1), box.l - 6, sy(v) + 4); });
      x.textAlign = 'center'; [-2, -1, 0, 1, 2].forEach(function(z){ x.fillText((S.st[0].m + S.st[0].s * z).toFixed(1), sx(z), box.b + 16); }); x.fillStyle = C.char; x.font = '600 12px Aptos, "Segoe UI", sans-serif'; x.fillText('logP', (box.l + box.r) / 2, box.b + 34); x.save(); x.translate(13, (box.t + box.b) / 2); x.rotate(-Math.PI / 2); x.fillText(unitWord(), 0, 0); x.restore();
      x.strokeStyle = 'rgba(52,55,63,.5)'; x.lineWidth = 1.5; T.idx.forEach(function(i, q){ var px = sx(clamp(T.X[q][0], zlo, zhi)); x.beginPath(); x.arc(px, sy(D.Y[i]), 3, 0, 6.3); x.stroke(); });
      x.strokeStyle = C.red; x.lineWidth = 3.2; x.beginPath(); for (var z = zlo; z <= zhi + 1e-9; z += 0.04) { var py = sy(clamp(S.my + S.sy * bnRun(r.m, [z]).y, ylo - 1.5, yhi + 1.5)); z === zlo ? x.moveTo(sx(z), py) : x.lineTo(sx(z), py); } x.stroke();
      x.fillStyle = C.char; S.idx.forEach(function(i, q){ x.beginPath(); x.arc(sx(S.X[q][0]), sy(D.Y[i]), 4.2, 0, 6.3); x.fill(); });
      x.textAlign = 'left'; x.font = '11px Aptos, "Segoe UI", sans-serif'; x.fillStyle = C.mid; x.fillText('filled: 40 training compounds', box.l + 6, box.t + 12); x.fillText('open: 90 test compounds', box.l + 6, box.t + 26);
      var msg = sel === 0 ? 'Two hidden neurons give only a gentle bend. The curve misses much of the pattern, so both R2 values are low (underfitting).' : (sel === 1 ? 'Six hidden neurons give a more flexible curve. The test R2 is low because the lab values are noisy and only 40 compounds are used for training.' : 'Sixty hidden neurons give more weights than there are training compounds, so the curve can pass close to the filled points.' + (r.tr > r.te ? ' The training R2 is far above the test R2.' : ''));
      qs('.mt4', tp).innerHTML = '<div class="ol">' + SZ[sel] + ' hidden neurons</div>' + bar('Training R2', r.tr, C.char) + bar('Test R2', r.te, C.red) + '<p style="margin-top:10px">' + msg + '</p><p style="font-size:12px;color:' + C.mid + '">The test R2 measures generalisation. A high training R2 alone does not show that the network predicts new compounds.</p>';
    }
    qsa('[data-sz] button', ctl).forEach(function(b){ b.addEventListener('click', function(){ sel = +b.dataset.v; qsa('[data-sz] button', ctl).forEach(function(q){ q.classList.toggle('on', q === b); }); draw(); }); });
    async function play(tok){ for (var k = 0; k < 3; k++) { api.checkTok(tok); sel = k; qsa('[data-sz] button', ctl).forEach(function(q){ q.classList.toggle('on', +q.dataset.v === k); }); draw(); api.cap(k === 0 ? 'Two hidden neurons: too little capacity to follow the pattern.' : (k === 1 ? 'Six hidden neurons: a more flexible curve.' : 'Sixty hidden neurons: the curve passes close to the training points, and the test R2 is below the training R2.')); await api.sleep(3400); } sel = 1; qsa('[data-sz] button', ctl).forEach(function(q){ q.classList.toggle('on', +q.dataset.v === 1); }); draw(); api.cap(null); }
    return { render: draw, play: play };
  } });
GUIDE.steps['nn/A network that is too big'] = { notice: 'The 60-neuron curve passes close to the filled points, while the open test points can lie far from it.', q: 'Why is a high training R2 not sufficient?', a: 'A network with enough capacity can fit the training compounds, noise included. Only the test R2 shows whether the pattern generalises to new compounds.' };

/* ---------- 09 shared: the loss as a function of one weight ---------- */
function bhData(){
  var key = EP + 'bh'; if (D.cache[key]) return D.cache[key]; var S = bnData('bh', ['logP'], 260, 31), n = S.X.length, mxy = 0, i;
  for (i = 0; i < n; i++) mxy += S.X[i][0] * S.Yz[i]; mxy /= n; var myy = mean(S.Yz.map(function(v){ return v * v; })), mxx = mean(S.X.map(function(v){ return v[0] * v[0]; }));
  S.loss = function(w){ var s = 0; for (var q = 0; q < n; q++) { var e = w * S.X[q][0] - S.Yz[q]; s += e * e; } return s / n; };
  S.best = mxy / mxx; return (D.cache[key] = S);
}
function bhCurve(el, S, o){
  o = o || {}; var g = cv(el), x = g.x, box = { l: 52, r: g.w - 16, t: 16, b: g.h - 40 }, wlo = -3, whi = 3, lmax = S.loss(wlo) * 1.02, sx = function(w){ return box.l + (w - wlo) / (whi - wlo) * (box.r - box.l); }, sy = function(l){ return box.b - Math.min(l, lmax * 1.3) / lmax * (box.b - box.t); };
  x.strokeStyle = C.silver; x.fillStyle = C.mid; x.font = '11px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'center'; [-3, -2, -1, 0, 1, 2, 3].forEach(function(w){ x.beginPath(); x.moveTo(sx(w), box.t); x.lineTo(sx(w), box.b); x.stroke(); x.fillText(w, sx(w), box.b + 16); });
  x.fillStyle = C.char; x.font = '600 12px Aptos, "Segoe UI", sans-serif'; x.fillText('weight', (box.l + box.r) / 2, box.b + 34); x.save(); x.translate(13, (box.t + box.b) / 2); x.rotate(-Math.PI / 2); x.fillText('loss (mean squared error)', 0, 0); x.restore();
  x.textAlign = 'right'; x.fillStyle = C.mid; x.font = '11px Aptos, "Segoe UI", sans-serif'; x.fillText('higher loss is worse', box.r, box.t + 12);
  x.strokeStyle = C.char; x.lineWidth = 3; x.beginPath(); for (var w = wlo; w <= whi + 1e-9; w += 0.03) { var py = sy(S.loss(w)); w === wlo ? x.moveTo(sx(w), py) : x.lineTo(sx(w), py); } x.stroke();
  if (o.trail) { x.strokeStyle = 'rgba(225,36,42,.45)'; x.lineWidth = 1.5; x.beginPath(); o.trail.forEach(function(w, i){ var px = sx(clamp(w, wlo, whi)), p2 = sy(S.loss(clamp(w, wlo, whi))); i ? x.lineTo(px, p2) : x.moveTo(px, p2); }); x.stroke(); o.trail.forEach(function(w){ x.fillStyle = 'rgba(225,36,42,.35)'; x.beginPath(); x.arc(sx(clamp(w, wlo, whi)), sy(S.loss(clamp(w, wlo, whi))), 3, 0, 6.3); x.fill(); }); }
  if (o.marks) o.marks.forEach(function(m){ x.fillStyle = m.col; x.beginPath(); x.arc(sx(m.w), sy(S.loss(m.w)), 5, 0, 6.3); x.fill(); if (m.label) { x.fillStyle = C.char; x.font = '600 11px Aptos, "Segoe UI", sans-serif'; x.textAlign = m.dx > 0 ? 'left' : (m.dx < 0 ? 'right' : 'center'); x.fillText(m.label, sx(m.w) + (m.dx || 0), sy(S.loss(m.w)) + (m.dx ? 4 : (m.below ? 18 : -10))); } });
  var w0 = o.w, off = Math.abs(w0) > 3.2, wc = clamp(w0, wlo, whi), bx = sx(wc), by = sy(S.loss(wc));
  if (o.arrow !== false && !off) { var sl = 2 * (w0 - S.best), ang = Math.atan(sl * (box.b - box.t) / lmax / ((box.r - box.l) / (whi - wlo))), L = 70; x.strokeStyle = C.red; x.lineWidth = 2.5; x.beginPath(); x.moveTo(bx - Math.cos(ang) * L / 2, by + Math.sin(ang) * L / 2); x.lineTo(bx + Math.cos(ang) * L / 2, by - Math.sin(ang) * L / 2); x.stroke(); }
  x.fillStyle = off ? C.mid : C.red; x.strokeStyle = '#fff'; x.lineWidth = 2; x.beginPath(); x.arc(bx, by, 9, 0, 6.3); x.fill(); x.stroke();
  g.sx = sx; g.sy = sy; g.box = box; g.wlo = wlo; g.whi = whi; return g;
}

/* ---------- 09 step: rolling downhill ---------- */
LESSONS.netalgo.add({ level: 'beginner', short: 'Rolling downhill', title: 'Gradient descent in one dimension',
  change: 'Select a learning rate, then press Take one step or Run 25 steps. Press Reset to start again.',
  watch: 'Training adjusts the weights to minimise the loss, here the average squared error over the training compounds. With one weight the loss is a curve: the horizontal axis is the weight and the vertical axis is the loss. The minimum of the curve is the best weight. Gradient descent starts from an initial weight and repeats one update: new weight = old weight minus learning rate times gradient, where the gradient is the slope of the loss at the current weight. The red line is the tangent at the current weight. A small learning rate converges slowly. A moderate learning rate converges in a few updates. A learning rate that is too large overshoots the minimum on every update, and the loss grows (divergence). The curve is computed from 260 training compounds: the weight multiplies standardised logP in a linear model.',
  build: function(api){
    var etas = [0.04, 0.4, 1.05], names = ['0.04', '0.4', '1.05'], sel = 1, w = -2.4, steps = 0, trail = [-2.4], S = null, run = 0;
    var cp = api.el('pn', 'left:28px;top:28px;width:600px;height:408px', '<h4>Loss as a function of the weight</h4><canvas width="570" height="350"></canvas>');
    var tp = api.el('pn', 'left:648px;top:28px;width:284px;height:408px;overflow:auto', '<div class="rt" style="font-size:13px;line-height:1.5;color:' + C.char + '"></div>');
    var ctl = api.addControl('Learning rate ' + bnSeg(names, sel, 'data-e') + ' <button type="button" class="btn sm" data-one>Take one step</button> <button type="button" class="btn sm" data-roll>Run 25 steps</button> <button type="button" class="btn sm ghost" data-reset>Reset</button>');
    function slope(v){ return 2 * (v - S.best); }
    function draw(){
      S = bhData(); var lo = S.loss(w), sl = slope(w), off = Math.abs(w) > 6;
      bhCurve(qs('canvas', cp), S, { w: w, trail: trail });
      qs('.rt', tp).innerHTML = '<div class="ol">Current state</div><div style="display:flex;gap:18px"><div><div class="big" style="font-size:28px;color:' + C.red + '">' + (off ? 'off scale' : w.toFixed(2)) + '</div><div style="font-size:12px">weight</div></div><div><div class="big" style="font-size:28px">' + (off ? '-' : lo.toFixed(2)) + '</div><div style="font-size:12px">loss</div></div></div><p style="margin-top:10px">Updates: <b>' + steps + '</b></p>' + (off ? '<p style="color:' + C.red + ';font-weight:700">The weight has diverged: the learning rate is too large.</p>' : '<p>' + (Math.abs(sl) < 0.05 ? 'The slope is near zero: the weight is at the minimum.' : 'The slope is ' + (sl > 0 ? 'positive, so the update decreases the weight.' : 'negative, so the update increases the weight.')) + '</p>') + '<p style="font-size:12px;color:' + C.mid + '">The minimum is at a weight of about <b>' + S.best.toFixed(2) + '</b>. Update rule: new weight = old weight minus learning rate times slope.</p>';
    }
    function once(){ if (Math.abs(w) > 6) return; w = w - etas[sel] * slope(w); steps++; trail.push(w); if (trail.length > 60) trail.shift(); }
    function reset(){ run++; w = -2.4; steps = 0; trail = [w]; draw(); }
    qsa('[data-e] button', ctl).forEach(function(b){ b.addEventListener('click', function(){ sel = +b.dataset.v; qsa('[data-e] button', ctl).forEach(function(q){ q.classList.toggle('on', q === b); }); reset(); }); });
    qs('[data-one]', ctl).addEventListener('click', function(){ once(); draw(); });
    qs('[data-reset]', ctl).addEventListener('click', reset);
    qs('[data-roll]', ctl).addEventListener('click', async function(){ var id = ++run; for (var i = 0; i < 25; i++) { if (id !== run) return; once(); draw(); await delay(ANIM ? 140 : 0); } });
    async function play(tok){
      for (var k = 0; k < 3; k++) { api.checkTok(tok); sel = k; qsa('[data-e] button', ctl).forEach(function(q){ q.classList.toggle('on', +q.dataset.v === k); }); reset(); api.cap('Learning rate ' + names[k] + ': successive updates of the weight.'); var id = ++run; for (var i = 0; i < 22; i++) { api.checkTok(tok); if (id !== run) return; once(); draw(); await api.sleep(150); } await api.sleep(900); }
      sel = 1; qsa('[data-e] button', ctl).forEach(function(q){ q.classList.toggle('on', +q.dataset.v === 1); }); reset(); api.cap(null);
    }
    return { render: function(){ draw(); }, play: play };
  } });
GUIDE.steps['netalgo/Rolling downhill'] = { notice: 'A small learning rate converges slowly, a moderate one converges quickly, and a large one overshoots.', q: 'Why can a learning rate that is too large increase the loss?', a: 'Each update moves the weight past the minimum to a point where the loss is higher, so the loss grows instead of falling.' };

/* ---------- 09 step: which way and how far ---------- */
LESSONS.netalgo.add({ level: 'beginner', short: 'Which way to turn the dial', title: 'The gradient',
  change: 'Drag the weight. Read the loss after a small decrease and a small increase of the weight.',
  watch: 'Gradient descent needs the gradient: the derivative of the loss with respect to a weight. It can be estimated by finite differences. Increase the weight by a small step and recompute the loss, then decrease it by the same step and recompute. The difference between the two losses divided by twice the step estimates the slope. If the loss is lower at the larger weight, the slope is negative and the weight should increase. The magnitude of the slope is large far from the minimum and near zero at the minimum. Backpropagation, described in a later step, computes the same quantity exactly and for every weight at once.',
  build: function(api){
    var w = -1.8, S = null, NUDGE = 0.2;
    var cp = api.el('pn', 'left:28px;top:28px;width:600px;height:408px', '<h4>Loss as a function of the weight</h4><canvas width="570" height="350"></canvas>');
    var tp = api.el('pn', 'left:648px;top:28px;width:284px;height:408px;overflow:auto', '<div class="rt" style="font-size:13px;line-height:1.5;color:' + C.char + '"></div>');
    var ctl = api.addControl('Weight <input type="range" min="-3" max="3" step="0.01" value="-1.8" style="width:300px"> <b class="wv"></b> <button type="button" class="btn sm" data-turn>Apply one update</button>');
    var rg = qs('input', ctl), wv = qs('.wv', ctl);
    function draw(){
      S = bhData(); w = +rg.value; wv.textContent = w.toFixed(2); var a = S.loss(w - NUDGE), b = S.loss(w + NUDGE), c = S.loss(w), sl = (b - a) / (2 * NUDGE);
      bhCurve(qs('canvas', cp), S, { w: w, arrow: false, marks: [{ w: Math.max(-3, w - NUDGE), col: '#6F8196', label: 'w - 0.2', dx: -10 }, { w: Math.min(3, w + NUDGE), col: C.dred, label: 'w + 0.2', dx: 10 }] });
      var flat = Math.abs(sl) < 0.08;
      qs('.rt', tp).innerHTML = '<div class="ol">Finite difference, step ' + NUDGE.toFixed(1) + '</div><div style="display:grid;grid-template-columns:1fr auto;gap:2px 10px;font-size:13px"><span>Loss at the current weight</span><b>' + c.toFixed(2) + '</b><span>Loss at weight minus ' + NUDGE.toFixed(1) + '</span><b style="color:#4B5A6B">' + a.toFixed(2) + '</b><span>Loss at weight plus ' + NUDGE.toFixed(1) + '</span><b style="color:' + C.dred + '">' + b.toFixed(2) + '</b></div><p style="margin-top:10px">' + (flat ? 'The two losses are almost equal. The slope is near zero, so the weight is at the minimum.' : 'The loss is lower at the weight ' + (b > a ? 'minus' : 'plus') + ' the step, so the slope is <b>' + (sl > 0 ? 'positive' : 'negative') + '</b> and the weight should <b style="color:' + C.red + '">' + (sl > 0 ? 'decrease' : 'increase') + '</b>.') + '</p><div class="ol" style="margin-top:8px">Slope magnitude</div><div style="height:12px;background:' + C.light + ';border-radius:3px;overflow:hidden"><i style="display:block;height:12px;width:' + clamp(Math.abs(sl) / 6 * 100, 2, 100) + '%;background:' + C.red + '"></i></div><p style="font-size:12px;margin-top:6px">' + (Math.abs(sl) > 2.5 ? 'Large slope: far from the minimum.' : (flat ? 'Near zero: at the minimum.' : 'Small slope: close to the minimum.')) + '</p>';
    }
    rg.addEventListener('input', draw);
    qs('[data-turn]', ctl).addEventListener('click', function(){ S = bhData(); var sl = (S.loss(+rg.value + NUDGE) - S.loss(+rg.value - NUDGE)) / (2 * NUDGE), from = +rg.value, to = clamp(from - 0.4 * sl, -3, 3); animateFree(900, function(t){ rg.value = from + (to - from) * t; draw(); }); });
    async function play(tok){
      rg.value = -1.8; draw(); api.cap('The weight starts to the left of the minimum.'); await api.sleep(2800);
      api.cap('The loss is lower at the larger weight, so the slope is negative and the weight should increase.'); await api.sleep(3200);
      var to = bhData().best; await api.anim(2800, function(t){ rg.value = -1.8 + (to + 1.8) * t; draw(); }); api.cap('Near the minimum the two loss values are almost equal.'); await api.sleep(3000);
      rg.value = 2.4; draw(); api.cap('To the right of the minimum the loss is lower at the smaller weight, so the weight should decrease.'); await api.sleep(3200); api.cap(null);
    }
    return { render: draw, play: play };
  } });
GUIDE.steps['netalgo/Which way to turn the dial'] = { notice: 'The side with the lower loss gives the descent direction. Near the minimum both sides are equal.', q: 'How is the gradient estimated by finite differences?', a: 'Evaluate the loss at the weight plus and minus a small step. The difference divided by twice the step approximates the derivative.' };

/* ---------- 09 step: counting the dials ---------- */
LESSONS.netalgo.add({ level: 'beginner', short: 'Counting the dials', title: 'Parameter count',
  change: 'Use the plus and minus buttons to change the number of neurons in each hidden layer. Read the parameter count.',
  watch: 'The parameters of a network are its weights and biases. Each connection between two neurons has one weight, and each hidden or output neuron has one bias. A fully connected layer from m inputs to n neurons has m times n weights and n biases. The network here takes 2 features, has one or two hidden layers, and ends in one output neuron. The count grows with the product of adjacent layer widths. The chart shows a saved experiment: networks of different sizes were trained to predict the signed distance to the surface of a 3D double helix, so that its sign classifies a point as inside or outside. They used 100, 1,000 or 10,000 training points and were scored by balanced accuracy on 2,000 held-out points near the surface (0.5 is chance and 1.0 is perfect). With 100 training points no network scores well, whatever its parameter count.',
  build: function(api){
    var n1 = 3, n2 = 0, J = null;
    var dp = api.el('pn', 'left:28px;top:28px;width:500px;height:408px', '<h4>Fully connected network</h4><canvas width="470" height="230"></canvas><div class="cn" style="font-size:12.5px;line-height:1.5;color:' + C.char + '"></div>');
    var rp = api.el('pn', 'left:548px;top:28px;width:384px;height:408px;overflow:auto', '<div class="cr" style="font-size:13px;line-height:1.5;color:' + C.char + '"></div>');
    var ctl = api.addControl('Hidden layer 1 <button type="button" class="btn sm ghost" data-a="-1">-</button> <b class="a1"></b> <button type="button" class="btn sm ghost" data-a="1">+</button> neurons &nbsp; Hidden layer 2 <button type="button" class="btn sm ghost" data-b="-1">-</button> <b class="a2"></b> <button type="button" class="btn sm ghost" data-b="1">+</button> neurons');
    function count(){ var w1 = 2 * n1, b1 = n1, w2 = n2 ? n1 * n2 : 0, b2 = n2, wo = (n2 || n1), bo = 1; return { w1: w1, b1: b1, w2: w2, b2: b2, wo: wo, bo: bo, total: w1 + b1 + w2 + b2 + wo + bo }; }
    function drawNet(){
      var g = cv(qs('canvas', dp)), x = g.x, layers = [2, n1].concat(n2 ? [n2] : []).concat([1]), cols = layers.length, xs = layers.map(function(_, i){ return 50 + i * (g.w - 100) / (cols - 1); });
      function ys(c, i){ var n = layers[c], gap = Math.min(34, (g.h - 40) / n); return g.h / 2 + (i - (n - 1) / 2) * gap; }
      x.lineWidth = 1; for (var c = 0; c < cols - 1; c++) for (var i = 0; i < layers[c]; i++) for (var j = 0; j < layers[c + 1]; j++) { x.strokeStyle = 'rgba(111,129,150,.45)'; x.beginPath(); x.moveTo(xs[c], ys(c, i)); x.lineTo(xs[c + 1], ys(c + 1, j)); x.stroke(); }
      for (c = 0; c < cols; c++) for (i = 0; i < layers[c]; i++) { x.fillStyle = c === 0 ? '#fff' : (c === cols - 1 ? '#FBE9E9' : '#EDF2F4'); x.strokeStyle = C.char; x.lineWidth = 1.6; x.beginPath(); x.arc(xs[c], ys(c, i), 7, 0, 6.3); x.fill(); x.stroke(); }
      x.fillStyle = C.mid; x.font = '600 11px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'center'; x.fillText('2 inputs', xs[0], g.h - 6); x.fillText('hidden layer 1', xs[1], g.h - 6); if (n2) x.fillText('hidden layer 2', xs[2], g.h - 6); x.fillText('output', xs[cols - 1], g.h - 6);
    }
    function near(params, runs){ var best = 0, bd = 1e9; runs.forEach(function(r, i){ var d = Math.abs(Math.log(r.params) - Math.log(params)); if (d < bd) { bd = d; best = i; } }); return best; }
    function draw(){
      qs('.a1', ctl).textContent = n1; qs('.a2', ctl).textContent = n2 || 'none'; var c = count(); drawNet();
      qs('.cn', dp).innerHTML = '<div><b>Weights</b> (one per connection): 2 times ' + n1 + ' = ' + c.w1 + (n2 ? ', then ' + n1 + ' times ' + n2 + ' = ' + c.w2 + ', then ' + n2 + ' to the output = ' + c.wo : ', then ' + n1 + ' to the output = ' + c.wo) + '.</div><div><b>Biases</b> (one per neuron): ' + (n2 ? n1 + ' + ' + n2 + ' + 1 output = ' + (c.b1 + c.b2 + c.bo) : n1 + ' + 1 output = ' + (c.b1 + c.bo)) + '.</div><div style="margin-top:4px">Total parameters: <b style="color:' + C.red + ';font-size:20px">' + c.total.toLocaleString() + '</b></div>';
      var html = '<div class="ol">Saved experiment</div>';
      if (J) {
        html += '<p style="font-size:12px">Balanced accuracy on held-out points for the saved network nearest in size to the one shown. 0.5 is chance.</p>';
        ['100', '1000', '10000'].forEach(function(sz){ var runs = J.runs[sz], i = near(c.total, runs), r = runs[i], acc = r.ck[r.ck.length - 1].acc, w = clamp((acc - 0.5) / 0.5, 0, 1) * 100; html += '<div style="margin:11px 0"><div style="display:flex;justify-content:space-between"><b>' + Number(sz).toLocaleString() + ' training points</b><span class="big" style="font-size:22px;margin:0;color:' + (acc > 0.8 ? C.red : C.char) + '">' + acc.toFixed(2) + '</span></div><div style="height:14px;background:' + C.light + ';border-radius:4px;overflow:hidden"><i style="display:block;height:14px;width:' + Math.max(w, 1.5) + '%;background:' + (acc > 0.8 ? C.red : C.bg) + '"></i></div><div style="font-size:11px;color:' + C.mid + '">saved network with ' + r.params.toLocaleString() + ' parameters</div></div>'; });
        html += '<p style="margin-top:8px">With 100 training points no network scores well. With 10,000 points, networks of about 10,000 parameters or more score above 0.9. More parameters help only when the training set can determine them.</p>';
      } else html += '<p>Loading the saved experiment.</p>';
      qs('.cr', rp).innerHTML = html;
    }
    function bump(which, d){ if (which === 'a') n1 = clamp(n1 + d, 1, 12); else n2 = clamp(n2 + d, 0, 12); draw(); }
    qsa('[data-a]', ctl).forEach(function(b){ b.addEventListener('click', function(){ bump('a', +b.dataset.a); }); });
    qsa('[data-b]', ctl).forEach(function(b){ b.addEventListener('click', function(){ bump('b', +b.dataset.b); }); });
    async function render(){ draw(); if (!J) { J = await loadJSON('scaling'); draw(); } }
    async function play(tok){
      await render(); n1 = 2; n2 = 0; draw(); api.cap('Two inputs and two hidden neurons. Each connection has a weight and each neuron has a bias.'); await api.sleep(3200);
      n1 = 6; draw(); api.cap('Six hidden neurons: the number of weights grows with the layer width.'); await api.sleep(2600); n2 = 6; draw(); api.cap('A second hidden layer connects every neuron in layer 1 to every neuron in layer 2, so the count increases sharply.'); await api.sleep(3400);
      n1 = 12; n2 = 12; draw(); api.cap('Twelve and twelve neurons. In the saved experiment, the score with 100 training points stays below 0.6 at every size.'); await api.sleep(3600); n1 = 3; n2 = 0; draw(); api.cap(null);
    }
    return { render: render, play: play };
  } });
GUIDE.steps['netalgo/Counting the dials'] = { notice: 'A second hidden layer increases the parameter count sharply.', q: 'Why does a larger network need more training data?', a: 'The data must determine every parameter. With too few training points the extra parameters fit noise (overfitting) instead of a pattern.' };
