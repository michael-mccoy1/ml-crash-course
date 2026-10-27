"use strict";
/* Lesson 9: Network algorithms. Gradient descent, backpropagation, initialisation, weight decay, attention, the next-token loss and parameter counts. */
var L8 = new Lesson('netalgo', 'Network algorithms');
L8.optsHTML = function(){ return sharedOptions(); };
L8.onData = function(){ };

function naSd(a){ var m = mean(a); return Math.sqrt(mean(a.map(function(v){ return (v - m) * (v - m); }))) || 1; }
function naRow(label, value, color){ return '<div class="rrow">' + label + '<b' + (color ? ' style="color:' + color + '"' : '') + '>' + value + '</b></div>'; }
function naDebounce(fn, ms){ var t = null; return function(){ clearTimeout(t); t = setTimeout(fn, ms); }; }
function naFmt(v, n){ return !isFinite(v) ? 'diverged' : (Math.abs(v) >= 1e4 ? v.toExponential(1) : v.toFixed(n === undefined ? 3 : n)); }

/* ---------- Step 1: gradient descent on a loss surface ---------- */
L8.add({ short: 'Gradient descent', title: 'How a model walks down the loss surface',
  change: 'Click the map to set a start. Drag Learning rate and Steps. Switch centring on and off.',
  watch: 'Training searches for the weight and bias with the lowest loss, the average squared miss. The map shows the loss for every pair of weight (across) and bias (up), and darker red is higher. Gradient descent steps downhill. Each step is the learning rate times the slope. Momentum also keeps a running sum of earlier steps. Adam divides each step by the recent size of its gradient. If the feature is not centred, the valley is long and thin: the slope is steep across it and shallow along it. Plain gradient descent must then use a small learning rate to stay stable, and it crawls along the valley. Adam rescales each direction, so it can use a larger rate. Centring and scaling the feature makes the valley round.',
  build: function(api){
    var mp = api.el('pn', 'left:28px;top:28px;width:620px;height:408px', '<h4>Loss for every weight and bias. Click to set the start.</h4><canvas width="596" height="344" style="margin-top:2px"></canvas>');
    var sp = api.el('pn', 'left:668px;top:28px;width:264px;height:408px;overflow:auto', '<h4>Loss at each step</h4><canvas class="c2" width="240" height="170"></canvas><div class="rd" style="margin-top:6px"></div>');
    var ctl = api.addControl('Feature <select class="sf"></select> <label><input class="cn" type="checkbox"> Centre and scale</label> Learning rate <input class="lr" type="range" min="-3" max="0" step="0.1" value="-0.5" style="width:110px"><b class="lv">0.316</b> Steps <input class="st" type="range" min="1" max="300" step="1" value="120" style="width:110px"><b class="sv">120</b>');
    var sf = qs('.sf', ctl), cn = qs('.cn', ctl), lr = qs('.lr', ctl), st = qs('.st', ctl), lv = qs('.lv', ctl), sv = qs('.sv', ctl), S = null, start = null, W0 = 0, W1 = 0, B0 = 0, B1 = 0;
    function fillSel(){ var c = sf.value || 'MW'; sf.innerHTML = D.BASE.map(function(x){ return '<option>' + x + '</option>'; }).join(''); sf.value = D.BASE.indexOf(c) >= 0 ? c : D.BASE[0]; }
    function stats(){
      var f = sf.value, centre = cn.checked, x = D.TRAIN.map(function(i){ return D.COL[f][i]; }), y = D.TRAIN.map(function(i){ return D.Y[i]; }), mx = mean(x), sx = naSd(x), my = mean(y), sy = naSd(y);
      var X = x.map(function(v){ return (centre ? v - mx : v) / sx; }), Y = y.map(function(v){ return (v - my) / sy; }), n = X.length, exx = 0, ex = 0, exy = 0, ey = 0, eyy = 0;
      for (var i = 0; i < n; i++) { exx += X[i] * X[i]; ex += X[i]; exy += X[i] * Y[i]; ey += Y[i]; eyy += Y[i] * Y[i]; } exx /= n; ex /= n; exy /= n; ey /= n; eyy /= n;
      var ws = (exy - ex * ey) / (exx - ex * ex), bs = ey - ws * ex; return { exx: exx, ex: ex, exy: exy, ey: ey, eyy: eyy, ws: ws, bs: bs, lmin: eyy - 2 * ws * exy - 2 * bs * ey + ws * ws * exx + 2 * ws * bs * ex + bs * bs };
    }
    function loss(w, b){ return S.eyy - 2 * w * S.exy - 2 * b * S.ey + w * w * S.exx + 2 * w * b * S.ex + b * b; }
    function run(opt, rate, n){
      var w = start[0], b = start[1], vw = 0, vb = 0, mw = 0, mb = 0, sw = 0, sb = 0, pts = [[w, b, loss(w, b)]];
      for (var t = 1; t <= n; t++) {
        var gw = -2 * S.exy + 2 * w * S.exx + 2 * b * S.ex - 0, gb = -2 * S.ey + 2 * w * S.ex + 2 * b;
        if (opt === 'gd') { w -= rate * gw; b -= rate * gb; }
        else if (opt === 'mom') { vw = 0.9 * vw + gw; vb = 0.9 * vb + gb; w -= rate * vw; b -= rate * vb; }
        else { mw = 0.9 * mw + 0.1 * gw; mb = 0.9 * mb + 0.1 * gb; sw = 0.999 * sw + 0.001 * gw * gw; sb = 0.999 * sb + 0.001 * gb * gb; var c1 = 1 - Math.pow(0.9, t), c2 = 1 - Math.pow(0.999, t); w -= rate * (mw / c1) / (Math.sqrt(sw / c2) + 1e-8); b -= rate * (mb / c1) / (Math.sqrt(sb / c2) + 1e-8); }
        var L = loss(w, b); pts.push([w, b, L]); if (!isFinite(L) || L > 1e8) break;
      }
      return pts;
    }
    function resetStart(){ S = stats(); var R = 2, Rb = 2 + 2 * Math.abs(S.ex); W0 = S.ws - R; W1 = S.ws + R; B0 = S.bs - Rb; B1 = S.bs + Rb; start = [S.ws - 1.5, S.bs + (cn.checked ? 1.5 : 1.5 * S.ex + 2)]; }
    function draw(){
      lv.textContent = Math.pow(10, +lr.value).toFixed(3); sv.textContent = st.value; var rate = Math.pow(10, +lr.value), n = +st.value;
      var g = cv(qs('canvas', mp)), x = g.x, box = { l: 44, r: g.w - 8, t: 8, b: g.h - 36 }, cw = 6, sx = function(w){ return box.l + (w - W0) / (W1 - W0) * (box.r - box.l); }, sy = function(b){ return box.b - (b - B0) / (B1 - B0) * (box.b - box.t); };
      for (var px = box.l; px < box.r; px += cw) for (var py = box.t; py < box.b; py += cw) { var w = W0 + (px + cw / 2 - box.l) / (box.r - box.l) * (W1 - W0), b = B0 + (box.b - py - cw / 2) / (box.b - box.t) * (B1 - B0), lev = Math.floor(Math.log10(Math.max(loss(w, b) - S.lmin, 1e-9) / 0.01 + 1) * 2.2); x.fillStyle = 'rgba(225,36,42,' + (0.04 + 0.075 * Math.min(lev, 10)).toFixed(3) + ')'; x.fillRect(px, py, cw, cw); }
      x.strokeStyle = C.silver; x.strokeRect(box.l, box.t, box.r - box.l, box.b - box.t); x.fillStyle = C.char; x.font = '600 12px Aptos, Segoe UI, sans-serif'; x.textAlign = 'center'; x.fillText('weight', (box.l + box.r) / 2, g.h - 6); x.save(); x.translate(12, (box.t + box.b) / 2); x.rotate(-Math.PI / 2); x.fillText('bias', 0, 0); x.restore();
      x.save(); x.beginPath(); x.rect(box.l, box.t, box.r - box.l, box.b - box.t); x.clip();
      var cols = { gd: C.char, mom: '#6F8196', adam: C.red }, res = {};
      ['gd', 'mom', 'adam'].forEach(function(o){ var pts = run(o, rate, n); res[o] = pts; x.strokeStyle = cols[o]; x.lineWidth = 2; x.beginPath(); pts.forEach(function(p, i){ i ? x.lineTo(sx(p[0]), sy(p[1])) : x.moveTo(sx(p[0]), sy(p[1])); }); x.stroke(); var e = pts[pts.length - 1]; x.fillStyle = cols[o]; x.beginPath(); x.arc(sx(e[0]), sy(e[1]), 4, 0, 6.3); x.fill(); });
      x.restore(); x.fillStyle = '#fff'; x.strokeStyle = C.char; x.lineWidth = 2; x.beginPath(); x.arc(sx(start[0]), sy(start[1]), 6, 0, 6.3); x.fill(); x.stroke(); x.fillStyle = C.char; x.beginPath(); x.moveTo(sx(S.ws), sy(S.bs) - 8); x.lineTo(sx(S.ws) + 7, sy(S.bs) + 6); x.lineTo(sx(S.ws) - 7, sy(S.bs) + 6); x.closePath(); x.fill();
      var series = ['gd', 'mom', 'adam'].map(function(o){ return { pts: res[o].map(function(p, i){ return [i, Math.log10(Math.max(p[2] - S.lmin, 1e-9))]; }).filter(function(p){ return isFinite(p[1]) && p[1] < 6; }), color: cols[o], nodot: true, w: 2.2 }; });
      lineChart(qs('.c2', sp), series, { xl: 'step', yl: 'log10 of the excess loss', x0: 0, x1: Math.max(2, n), y0: -6, y1: 3, ydec: 0, xdec: 0 });
      qs('.rd', sp).innerHTML = [['Gradient descent', 'gd'], ['Momentum', 'mom'], ['Adam', 'adam']].map(function(r){ var e = res[r[1]], L = e[e.length - 1][2]; return '<div class="rrow"><span><b style="color:' + cols[r[1]] + '">' + r[0] + '</b></span><b>' + (isFinite(L) && L < 1e8 ? 'excess ' + naFmt(L - S.lmin, 3) : 'diverged') + '</b></div>'; }).join('') + '<p class="note" style="font-size:12px;margin-top:6px">Triangle: the lowest loss. White dot: the start. The excess loss is the loss minus its lowest value.</p>';
    }
    var cvs = qs('canvas', mp); cvs.addEventListener('pointerdown', function(ev){ var r = cvs.getBoundingClientRect(), sc = r.width / cvs.getAttribute('width'), px = (ev.clientX - r.left) / sc, py = (ev.clientY - r.top) / sc, box = { l: 44, r: 596 - 8, t: 8, b: 344 - 36 }; if (px < box.l || px > box.r || py < box.t || py > box.b) return; start = [W0 + (px - box.l) / (box.r - box.l) * (W1 - W0), B0 + (box.b - py) / (box.b - box.t) * (B1 - B0)]; draw(); });
    sf.addEventListener('change', function(){ resetStart(); draw(); }); cn.addEventListener('change', function(){ resetStart(); draw(); }); lr.addEventListener('input', naDebounce(draw, 40)); st.addEventListener('input', naDebounce(draw, 40));
    return { render: function(){ fillSel(); if (!sf.value) sf.value = 'MW'; resetStart(); draw(); } };
  } });

/* ---------- Step 2: backpropagation by hand ---------- */
var NA_NAMES = ['W1 input 1 to hidden 1', 'W1 input 1 to hidden 2', 'W1 input 2 to hidden 1', 'W1 input 2 to hidden 2', 'b1 hidden 1', 'b1 hidden 2', 'W2 hidden 1 to output', 'W2 hidden 2 to output', 'b2 output'];
var NA_P0 = [0.6, -0.5, 0.4, 0.7, 0.1, -0.1, 0.8, -0.6, 0.05];
function naNet(p, x, y){
  var z = [p[0] * x[0] + p[2] * x[1] + p[4], p[1] * x[0] + p[3] * x[1] + p[5]], h = [Math.max(0, z[0]), Math.max(0, z[1])], yh = p[6] * h[0] + p[7] * h[1] + p[8];
  return { z: z, h: h, yh: yh, L: (yh - y) * (yh - y) };
}
function naGrad(p, x, y){
  var f = naNet(p, x, y), dy = 2 * (f.yh - y), dh = [dy * p[6], dy * p[7]], dz = [f.z[0] > 0 ? dh[0] : 0, f.z[1] > 0 ? dh[1] : 0];
  return { f: f, dy: dy, dh: dh, dz: dz, g: [dz[0] * x[0], dz[1] * x[0], dz[0] * x[1], dz[1] * x[1], dz[0], dz[1], dy * f.h[0], dy * f.h[1], dy] };
}
L8.add({ short: 'Backpropagation', title: 'How a network finds the gradient of every weight',
  change: 'Switch between Forward and Backward. Press Next compound, then Take one step.',
  watch: 'To change a weight, training needs the slope of the loss with respect to that weight, the gradient. Backpropagation finds all of them with the chain rule. The forward pass computes each neuron in turn, from the inputs to the prediction, and the loss. The backward pass starts at the loss and works toward the inputs, reusing each result: the slope at a neuron is the slope after it times the local slope. A neuron whose ReLU output is zero passes back a zero, so its input weights get no gradient for this compound. The table checks every gradient against a numerical estimate, the loss change when the weight moves a tiny amount in each direction.',
  build: function(api){
    var lp = api.el('pn', 'left:28px;top:28px;width:470px;height:408px;overflow:auto', '<h4 class="th"></h4><div class="tx" style="font-size:13px;line-height:1.55;color:' + C.char + '"></div>');
    var rp = api.el('pn', 'left:518px;top:28px;width:414px;height:408px;overflow:auto', '<h4>Gradient of each weight</h4><div class="tb"></div>');
    var ctl = api.addControl('<span class="seg" data-v><button type="button" data-view="0" class="on">Forward</button><button type="button" data-view="1">Backward</button></span> <button type="button" class="btn sm ghost" data-next>Next compound</button> <button type="button" class="btn sm" data-step>Take one step</button> <button type="button" class="btn sm ghost" data-reset>Reset weights</button>');
    var P = NA_P0.slice(), view = 0, k = 0, SAMPLES = null, note = '', RATE = 0.1;
    function sample(){ if (!SAMPLES) { SAMPLES = []; for (var q = 0; q < 12; q++) SAMPLES.push(D.TRAIN[Math.floor((q + 0.5) * D.TRAIN.length / 12)]); } var i = SAMPLES[k % SAMPLES.length], fa = D.COL.logP, fb = D.COL.TPSA, la = D.TRAIN.map(function(t){ return fa[t]; }), lb = D.TRAIN.map(function(t){ return fb[t]; }), ys = D.TRAIN.map(function(t){ return D.Y[t]; });
      return { x: [(fa[i] - mean(la)) / naSd(la), (fb[i] - mean(lb)) / naSd(lb)], y: (D.Y[i] - mean(ys)) / naSd(ys) }; }
    function draw(){
      var s = sample(), r = naGrad(P, s.x, s.y), f = r.f, n2 = function(v){ return (v >= 0 ? '' : '') + v.toFixed(3); };
      qs('.th', lp).textContent = view === 0 ? 'Forward pass, loss ' + f.L.toFixed(3) : 'Backward pass, loss ' + f.L.toFixed(3);
      var t;
      if (view === 0) t = '<p>Inputs for this compound, centred and scaled: logP <b>' + n2(s.x[0]) + '</b>, TPSA <b>' + n2(s.x[1]) + '</b>. The measured value, centred and scaled, is <b>' + n2(s.y) + '</b>.</p>' +
        '<p>Hidden 1: z = ' + n2(P[0]) + ' times ' + n2(s.x[0]) + ' + ' + n2(P[2]) + ' times ' + n2(s.x[1]) + ' + ' + n2(P[4]) + ' = <b>' + n2(f.z[0]) + '</b>, so ReLU gives h = <b>' + n2(f.h[0]) + '</b>.</p>' +
        '<p>Hidden 2: z = ' + n2(P[1]) + ' times ' + n2(s.x[0]) + ' + ' + n2(P[3]) + ' times ' + n2(s.x[1]) + ' + ' + n2(P[5]) + ' = <b>' + n2(f.z[1]) + '</b>, so ReLU gives h = <b>' + n2(f.h[1]) + '</b>.</p>' +
        '<p>Output: ŷ = ' + n2(P[6]) + ' times ' + n2(f.h[0]) + ' + ' + n2(P[7]) + ' times ' + n2(f.h[1]) + ' + ' + n2(P[8]) + ' = <b>' + n2(f.yh) + '</b>.</p>' +
        '<p>Loss = (ŷ - y)² = (' + n2(f.yh) + ' - ' + n2(s.y) + ')² = <b>' + n2(f.L) + '</b>.</p>';
      else t = '<p>Slope at the output: dL/dŷ = 2(ŷ - y) = <b>' + n2(r.dy) + '</b>.</p>' +
        '<p>Output weights: dL/dW2 = dL/dŷ times h, so <b>' + n2(r.g[6]) + '</b> and <b>' + n2(r.g[7]) + '</b>. The output bias gets <b>' + n2(r.g[8]) + '</b>.</p>' +
        '<p>Slope at each hidden output: dL/dh = dL/dŷ times W2, so <b>' + n2(r.dh[0]) + '</b> and <b>' + n2(r.dh[1]) + '</b>.</p>' +
        '<p>Through ReLU: dL/dz = dL/dh if z is above 0, else 0. That gives <b>' + n2(r.dz[0]) + '</b> and <b>' + n2(r.dz[1]) + '</b>' + (f.z[0] <= 0 || f.z[1] <= 0 ? '. A hidden neuron is switched off for this compound, so it passes back zero.' : '.') + '</p>' +
        '<p>Hidden weights: dL/dW1 = dL/dz times input, and dL/db1 = dL/dz.</p>';
      qs('.tx', lp).innerHTML = t + (note ? '<p style="color:' + C.red + ';font-weight:700">' + note + '</p>' : '');
      var eps = 1e-5, rows = P.map(function(v, i){ var a = P.slice(), b = P.slice(); a[i] += eps; b[i] -= eps; var num = (naNet(a, s.x, s.y).L - naNet(b, s.x, s.y).L) / (2 * eps); return '<tr style="text-align:right;border-top:1px solid ' + C.silver + '"><td style="text-align:left">' + NA_NAMES[i] + '</td><td>' + n2(v) + '</td><td><b>' + n2(r.g[i]) + '</b></td><td>' + n2(num) + '</td></tr>'; }).join('');
      qs('.tb', rp).innerHTML = '<table style="width:100%;border-collapse:collapse;font-size:12px;color:' + C.char + '"><tr style="text-align:right;font-weight:700;line-height:1.15"><td style="text-align:left">Weight</td><td>Value</td><td>Backprop gradient</td><td>Numerical check</td></tr>' + rows + '</table><p class="note" style="font-size:12px;margin-top:6px">The last two columns match. A weight update subtracts the learning rate (0.1 here) times the gradient.</p>';
    }
    qsa('[data-view]', ctl).forEach(function(b){ b.addEventListener('click', function(){ view = +b.dataset.view; qsa('[data-view]', ctl).forEach(function(x){ x.classList.toggle('on', x === b); }); draw(); }); });
    qs('[data-next]', ctl).addEventListener('click', function(){ k++; note = ''; draw(); });
    qs('[data-reset]', ctl).addEventListener('click', function(){ P = NA_P0.slice(); note = ''; draw(); });
    qs('[data-step]', ctl).addEventListener('click', function(){ var s = sample(), r = naGrad(P, s.x, s.y), before = r.f.L; P = P.map(function(v, i){ return v - RATE * r.g[i]; }); var after = naNet(P, s.x, s.y).L; note = 'One step: loss ' + before.toFixed(3) + ' to ' + after.toFixed(3) + ' on this compound.'; draw(); });
    return { render: draw };
  } });

/* ---------- Step 3: initialisation and depth ---------- */
function naDeep(act, depth, scale){
  var W = 64, B = 48, r = rng(77), x = [], l, i, j, b, h = [];
  for (b = 0; b < B; b++) { var v = new Float64Array(W); for (i = 0; i < W; i++) v[i] = gauss(r); h.push(v); }
  var c = act === 'relu' ? 2 : 1, sd = scale * Math.sqrt(c / W), Ws = [], Zs = [], fstd = [];
  fstd.push(1);
  for (l = 0; l < depth; l++) {
    var Wm = new Float64Array(W * W); for (i = 0; i < Wm.length; i++) Wm[i] = gauss(r) * sd; Ws.push(Wm);
    var Z = [], H = [], s2 = 0, cnt = 0;
    for (b = 0; b < B; b++) { var z = new Float64Array(W), a = new Float64Array(W); for (j = 0; j < W; j++) { var t = 0; for (i = 0; i < W; i++) t += h[b][i] * Wm[i * W + j]; z[j] = t; a[j] = act === 'relu' ? Math.max(0, t) : (act === 'tanh' ? Math.tanh(t) : 1 / (1 + Math.exp(-t))); s2 += a[j] * a[j]; cnt++; } Z.push(z); H.push(a); }
    Zs.push(Z); h = H; fstd.push(Math.sqrt(s2 / cnt));
  }
  var g = []; for (b = 0; b < B; b++) { var o = new Float64Array(W); for (j = 0; j < W; j++) o[j] = 1 / W; g.push(o); }
  var gn = new Array(depth);
  for (l = depth - 1; l >= 0; l--) {
    var s3 = 0, cnt2 = 0, ng = [];
    for (b = 0; b < B; b++) { var dz = new Float64Array(W), z2 = Zs[l][b]; for (j = 0; j < W; j++) { var ap = act === 'relu' ? (z2[j] > 0 ? 1 : 0) : (act === 'tanh' ? 1 - Math.tanh(z2[j]) * Math.tanh(z2[j]) : (function(){ var sg = 1 / (1 + Math.exp(-z2[j])); return sg * (1 - sg); })()); dz[j] = g[b][j] * ap; s3 += dz[j] * dz[j]; cnt2++; }
      var prev = new Float64Array(W); for (i = 0; i < W; i++) { var q = 0; for (j = 0; j < W; j++) q += Ws[l][i * W + j] * dz[j]; prev[i] = q; } ng.push(prev); }
    gn[l] = Math.sqrt(s3 / cnt2); g = ng;
  }
  return { fstd: fstd, gn: gn };
}
L8.add({ short: 'Initialisation', title: 'Why the starting weights decide whether a deep network can learn',
  change: 'Choose an activation. Drag Depth and Weight scale and watch the signal and the gradient.',
  watch: 'A network starts from random weights, and their size matters. Each layer multiplies the signal by its weights. If the weights are too small the signal shrinks layer by layer until it is almost zero. If they are too large it grows, or a saturating activation such as tanh or sigmoid flattens. The gradient travels back through the same layers, so it shrinks or grows in the same way. This is the vanishing or exploding gradient. A weight scale of 1 here is the standard choice for the activation: He initialisation for ReLU, which draws weights with variance 2 divided by the number of inputs, and Xavier initialisation for tanh and sigmoid, which uses 1 divided by the number of inputs when a layer has as many outputs as inputs, as here.',
  build: function(api){
    var p1 = api.el('pn', 'left:28px;top:28px;width:446px;height:408px', '<h4>Size of the signal at each layer</h4><canvas class="c1" width="420" height="330"></canvas>');
    var p2 = api.el('pn', 'left:486px;top:28px;width:446px;height:408px', '<h4>Size of the gradient at each layer, against the last layer</h4><canvas class="c2" width="420" height="330"></canvas>');
    var ctl = api.addControl('Activation <select class="ac"><option value="relu">ReLU</option><option value="tanh">tanh</option><option value="sig">sigmoid</option></select> Depth <input class="dp" type="range" min="2" max="30" step="1" value="20" style="width:110px"><b class="dv">20</b> Weight scale <input class="sc" type="range" min="-1" max="1" step="0.1" value="0" style="width:110px"><b class="sv">1.0</b>');
    var ac = qs('.ac', ctl), dp = qs('.dp', ctl), sc = qs('.sc', ctl), dv = qs('.dv', ctl), sv = qs('.sv', ctl);
    function draw(){
      var act = ac.value, depth = +dp.value, scale = Math.pow(10, +sc.value); dv.textContent = depth; sv.textContent = scale.toFixed(1);
      var r = naDeep(act, depth, scale), fs = r.fstd.map(function(v, i){ return [i, Math.log10(Math.max(v, 1e-12))]; }), gl = r.gn[r.gn.length - 1], gs = r.gn.map(function(v, i){ return [i + 1, Math.log10(Math.max(v / gl, 1e-30))]; });
      function ch(el, pts, label){ var ys = pts.map(function(p){ return p[1]; }), hi = Math.max(1, Math.ceil(Math.max.apply(null, ys))), lo0 = Math.min(-2, Math.floor(Math.min.apply(null, ys))), lo = hi - 4 * Math.ceil((hi - lo0) / 4); var g = lineChart(el, [{ pts: pts, color: C.red, w: 2.8 }], { xl: 'layer', yl: label, x0: pts[0][0], x1: Math.max(pts[pts.length - 1][0], pts[0][0] + 1), y0: lo, y1: hi, ydec: 0, xdec: 0 }); if (g.sy && lo < 0 && hi > 0) { g.x.setLineDash([4, 4]); g.x.strokeStyle = C.char; g.x.beginPath(); g.x.moveTo(g.box.l, g.sy(0)); g.x.lineTo(g.box.r, g.sy(0)); g.x.stroke(); g.x.setLineDash([]); } }
      ch(qs('.c1', p1), fs, 'log10 of the signal size'); ch(qs('.c2', p2), gs, 'log10 of the gradient, relative to the last layer');
    }
    ac.addEventListener('change', draw); dp.addEventListener('input', naDebounce(draw, 60)); sc.addEventListener('input', naDebounce(draw, 60));
    return { render: draw };
  } });

/* ---------- Step 4: weight decay and batch size on the real network ---------- */
var NA_WD = [0, 0.0001, 0.001, 0.01];
L8.add({ short: 'Weight decay', title: 'How a penalty on the weights changes what a network learns',
  change: 'Choose a batch size and press Run comparison.',
  watch: 'Weight decay adds a penalty to the loss for large weights, so training prefers small ones. Small weights give a smoother function that follows noise less. The gradient of an L2 penalty is the strength times the weight, so each step also pulls every weight a little toward zero. The batch size is how many compounds are used for each weight update. Small batches give noisier updates and more of them per epoch. Here the same network is trained four times, with decay of 0, 0.0001, 0.001 and 0.01, for 50 epochs with Adam and no dropout. The lines show the error on compounds held back during training.',
  build: function(api){
    var cp = api.el('pn', 'left:28px;top:28px;width:560px;height:408px', '<h4>Validation error against epoch</h4><canvas width="536" height="344"></canvas>');
    var sp = api.el('pn', 'left:608px;top:28px;width:324px;height:408px;overflow:auto', '<h4>Four runs</h4><div class="sd"></div>');
    var ctl = api.addControl('Batch size <select class="bs"><option>16</option><option selected>64</option><option>256</option></select> <button type="button" class="btn sm" data-run>Run comparison</button>');
    var bs = qs('.bs', ctl), busy = false, COL = [C.char, '#6F8196', '#F78C85', C.red];
    function store(){ D.cache.na = D.cache.na || { wd: {} }; return D.cache.na; }
    function draw(){
      var runs = store().wd[bs.value] || [], series = runs.map(function(r, i){ return { pts: r.va.map(function(v, e){ return [e + 1, v]; }), color: COL[i], w: 2.2, label: 'decay ' + NA_WD[i], nodot: i < runs.length - 1 }; });
      if (!series.length) { var g0 = cv(qs('canvas', cp)); g0.x.fillStyle = C.char; g0.x.fillText('Press Run comparison.', 60, 40); qs('.sd', sp).innerHTML = '<p class="note">No run yet.</p>'; return; }
      lineChart(qs('canvas', cp), series, { xl: 'epoch', yl: 'validation error (MSE)', ydec: 2, xdec: 0, x0: 1 });
      qs('.sd', sp).innerHTML = runs.map(function(r, i){ return '<div style="padding:8px 0;border-top:1px solid ' + C.silver + '"><b style="color:' + COL[i] + '">Decay ' + NA_WD[i] + '</b>' + naRow('Test R2', r.r2.toFixed(3)) + naRow('Lowest validation error', r.bestVa.toFixed(3)) + naRow('Sum of squared weights', r.wn.toFixed(0)) + '</div>'; }).join('') + (runs.length < 4 ? '<p class="note">Fitted ' + runs.length + ' of 4.</p>' : '<p class="note" style="font-size:12px">Sum of squared weights is the size the penalty acts on. A stronger decay should lower it.</p>');
    }
    async function run(){
      if (busy) return; busy = true; var key = bs.value, runs = store().wd[key] = [];
      for (var i = 0; i < NA_WD.length; i++) {
        api.busy('Training ' + (i + 1) + ' of 4'); await tick(); await tick();
        var res = await nnTrain({ sub: D.TRAIN, wd: NA_WD[i], batch: +key, seed: 0, params: { layers: 2, units: 32, drop: 0, lr: 0.003, epochs: 50, early: false } });
        var wn = 0; res.model.W.forEach(function(w){ for (var k = 0; k < w.length; k++) wn += w[k] * w[k]; });
        runs.push({ va: res.hist.va, r2: res.r2, bestVa: Math.min.apply(null, res.hist.va), wn: wn }); draw();
      }
      busy = false; api.idle(); draw();
    }
    qs('[data-run]', ctl).addEventListener('click', run); bs.addEventListener('change', function(){ if (!store().wd[bs.value]) run(); else draw(); });
    return { render: async function(){ if (!store().wd[bs.value]) await run(); else draw(); } };
  } });

/* ---------- Step 5: attention by hand ---------- */
var NA_TOK = ['C', 'C', '(', '=', 'O', ')', 'N'];
var NA_K = [[0.9, 0.4], [0.7, 0.8], [-0.9, 0.5], [-0.3, -0.9], [0.4, -0.8], [-0.8, -0.4], [0.2, 0.2]];
var NA_V = [[1, 0], [0.8, 0.6], [-0.6, 0.8], [0, -1], [0.6, -0.8], [-1, 0], [0.3, 0.3]];
L8.add({ short: 'Attention', title: 'Attention is a weighted lookup',
  change: 'Drag the query point. Choose which token is asking, and change the divisor.',
  watch: 'Each token carries a key, a value and, when it is the one asking, a query. In real models these are vectors with dozens of numbers. Here they have two so you can see them. The score between the query and a key is their dot product: large when the two point the same way. Scores are divided by a constant, then a softmax turns them into weights that are positive and add to 1. The output is the weighted sum of the values. The divisor is the square root of the vector length, which stops the scores growing as vectors get longer. A smaller divisor makes the weights sharper. Tokens after the asking token are hidden, because a GPT may not look ahead.',
  build: function(api){
    var kp = api.el('pn', 'left:28px;top:28px;width:420px;height:408px', '<h4>Query and keys. Drag the red point.</h4><canvas width="396" height="344"></canvas>');
    var wp = api.el('pn', 'left:468px;top:28px;width:464px;height:408px;overflow:auto', '<h4>Scores, weights and output</h4><div class="wb"></div>');
    var ctl = api.addControl('Asking token <select class="qt">' + NA_TOK.map(function(t, i){ return '<option value="' + i + '">' + (i + 1) + ': ' + t + '</option>'; }).join('') + '</select> Divide scores by <input class="dv" type="range" min="0.5" max="8" step="0.1" value="1.4" style="width:130px"><b class="dn">1.4</b>');
    var qt = qs('.qt', ctl), dvs = qs('.dv', ctl), dn = qs('.dn', ctl), q = [0.8, 0.5], drag = false; qt.value = '6';
    function draw(){
      var pos = +qt.value, div = +dvs.value; dn.textContent = div.toFixed(1);
      var sc = [], mx = -1e9, i; for (i = 0; i <= pos; i++) { sc.push((q[0] * NA_K[i][0] + q[1] * NA_K[i][1]) / div); mx = Math.max(mx, sc[i]); }
      var ex = sc.map(function(s){ return Math.exp(s - mx); }), z = ex.reduce(function(a, b){ return a + b; }, 0), w = ex.map(function(e){ return e / z; }), out = [0, 0]; for (i = 0; i <= pos; i++) { out[0] += w[i] * NA_V[i][0]; out[1] += w[i] * NA_V[i][1]; }
      var g = cv(qs('canvas', kp)), x = g.x, cx = g.w / 2, cy = (g.h - 20) / 2, S = 130; x.strokeStyle = C.silver; x.beginPath(); x.moveTo(8, cy); x.lineTo(g.w - 8, cy); x.moveTo(cx, 8); x.lineTo(cx, g.h - 28); x.stroke();
      for (i = 0; i < NA_K.length; i++) { var X = cx + NA_K[i][0] * S, Y = cy - NA_K[i][1] * S, hid = i > pos; x.globalAlpha = hid ? 0.25 : 1; x.fillStyle = C.char; x.beginPath(); x.arc(X, Y, 5 + (hid ? 0 : w[i] * 16), 0, 6.3); x.fill(); x.fillStyle = C.char; x.font = '600 12px Aptos, Segoe UI, sans-serif'; x.textAlign = 'left'; x.fillText((i + 1) + ': ' + NA_TOK[i], X + 9, Y - 8); x.globalAlpha = 1; }
      var qx = cx + q[0] * S, qy = cy - q[1] * S; x.strokeStyle = C.red; x.lineWidth = 2.6; x.beginPath(); x.moveTo(cx, cy); x.lineTo(qx, qy); x.stroke(); x.fillStyle = C.red; x.beginPath(); x.arc(qx, qy, 8, 0, 6.3); x.fill(); x.fillStyle = C.char; x.font = '600 11.5px Aptos, Segoe UI, sans-serif'; x.textAlign = 'center'; x.fillText('Circle size shows the attention weight. Faded tokens are hidden.', g.w / 2, g.h - 6);
      var rows = ''; for (i = 0; i < NA_K.length; i++) { var hidden = i > pos; rows += '<tr style="text-align:right;border-top:1px solid ' + C.silver + ';opacity:' + (hidden ? 0.35 : 1) + '"><td style="text-align:left">' + (i + 1) + ': ' + esc(NA_TOK[i]) + '</td><td>' + (hidden ? 'hidden' : (q[0] * NA_K[i][0] + q[1] * NA_K[i][1]).toFixed(2)) + '</td><td>' + (hidden ? '' : sc[i].toFixed(2)) + '</td><td style="width:120px">' + (hidden ? '' : '<div class="pbar" style="height:9px;margin:2px 0"><i style="background:' + C.red + ';width:' + (w[i] * 100).toFixed(0) + '%;transition:none"></i></div>') + '</td><td><b>' + (hidden ? '' : w[i].toFixed(2)) + '</b></td></tr>'; }
      qs('.wb', wp).innerHTML = '<table style="width:100%;border-collapse:collapse;font-size:13px;color:' + C.char + '"><tr style="text-align:right;font-weight:700;line-height:1.15"><td style="text-align:left">Token</td><td>Dot product</td><td>Divided</td><td></td><td>Weight</td></tr>' + rows + '</table>' +
        '<p style="font-size:13px;margin:10px 0 2px;color:' + C.char + '">Output = sum of weight times value = (<b>' + out[0].toFixed(2) + '</b>, <b>' + out[1].toFixed(2) + '</b>)</p><p class="note" style="font-size:12px">The weights add to ' + w.reduce(function(a, b){ return a + b; }, 0).toFixed(2) + '. For vectors of length 2 the usual divisor is the square root of 2, which is 1.4.</p>';
    }
    var cvs = qs('canvas', kp); function move(ev){ var r = cvs.getBoundingClientRect(), sc = r.width / 396, px = (ev.clientX - r.left) / sc, py = (ev.clientY - r.top) / sc; q = [clamp((px - 198) / 130, -2, 2), clamp(-(py - 162) / 130, -2, 2)]; draw(); }
    cvs.addEventListener('pointerdown', function(ev){ drag = true; cvs.setPointerCapture(ev.pointerId); move(ev); }); cvs.addEventListener('pointermove', function(ev){ if (drag) move(ev); }); cvs.addEventListener('pointerup', function(){ drag = false; });
    qt.addEventListener('change', draw); dvs.addEventListener('input', draw);
    return { render: draw };
  } });

/* ---------- Step 6: next-token loss ---------- */
var NA_V6 = ['C', 'N', 'O', '(', ')', '='];
L8.add({ short: 'Next-token loss', title: 'The number a GPT lowers during training',
  change: 'Drag the scores. Choose the correct token. Press Take one step.',
  watch: 'At each position the model outputs one score, a logit, for every token in its vocabulary. A softmax turns the scores into probabilities that add to 1. The loss is the cross-entropy: minus the natural log of the probability given to the correct token. It is 0 when that probability is 1 and grows as the probability falls. Perplexity is e raised to the loss, the number of equally likely tokens the model is choosing between. The gradient with respect to each score is the probability minus 1 for the correct token and the probability alone for the rest, so training raises the correct score and lowers the others.',
  build: function(api){
    var pn = api.el('pn', 'left:28px;top:28px;width:904px;height:408px', '<div class="nl"></div>');
    var ctl = api.addControl('Correct token <select class="ct">' + NA_V6.map(function(t, i){ return '<option value="' + i + '">' + esc(t) + '</option>'; }).join('') + '</select> <button type="button" class="btn sm" data-step>Take one step</button> <button type="button" class="btn sm ghost" data-reset>Reset scores</button>');
    var ct = qs('.ct', ctl), logits = [0.5, 1.2, -0.3, 0.8, -1, 0.1], note = '';
    function probs(){ var m = Math.max.apply(null, logits), e = logits.map(function(l){ return Math.exp(l - m); }), z = e.reduce(function(a, b){ return a + b; }, 0); return e.map(function(v){ return v / z; }); }
    function draw(){
      var c = +ct.value, p = probs(), L = -Math.log(p[c]), host = qs('.nl', pn);
      if (!qs('.nlr', pn)) { host.innerHTML = '<div style="display:grid;grid-template-columns:1fr 250px;gap:26px"><div class="nlr"></div><div class="nlq"></div></div>'; }
      qs('.nlr', pn).innerHTML = '<table style="width:100%;border-collapse:collapse;font-size:14px;color:' + C.char + '"><tr style="font-weight:700;text-align:right;line-height:1.2"><td style="text-align:left;width:90px">Token</td><td style="text-align:left;width:230px">Score (logit)</td><td>Probability</td><td style="width:160px"></td><td>Gradient</td></tr>' + NA_V6.map(function(t, i){ var g = p[i] - (i === c ? 1 : 0); return '<tr style="border-top:1px solid ' + C.silver + ';text-align:right"><td style="text-align:left;padding:8px 0"><b>' + esc(t) + '</b>' + (i === c ? ' <span style="color:' + C.red + ';font-size:12px">correct</span>' : '') + '</td><td style="text-align:left"><input class="lg" data-i="' + i + '" type="range" min="-5" max="5" step="0.1" value="' + logits[i] + '" style="width:150px"> <b>' + logits[i].toFixed(1) + '</b></td><td>' + p[i].toFixed(3) + '</td><td><div class="pbar" style="height:10px;margin:0"><i style="background:' + (i === c ? C.red : C.char) + ';width:' + (p[i] * 100).toFixed(0) + '%;transition:none"></i></div></td><td style="color:' + (g < 0 ? C.red : C.char) + '">' + (g >= 0 ? '+' : '') + g.toFixed(3) + '</td></tr>'; }).join('') + '</table>';
      qs('.nlq', pn).innerHTML = '<div class="big" style="color:' + C.red + '">' + L.toFixed(2) + '</div><div class="note" style="font-size:12.5px;color:' + C.char + ';margin-bottom:10px">loss = -ln(' + p[c].toFixed(3) + ')</div>' + naRow('Perplexity', Math.exp(L).toFixed(2)) + naRow('Loss of a uniform guess', Math.log(NA_V6.length).toFixed(2)) + (note ? '<p style="font-size:13px;color:' + C.red + ';font-weight:700;margin-top:10px">' + note + '</p>' : '') + '<p class="note" style="font-size:12px;margin-top:10px">A negative gradient (red) means raising that score lowers the loss. A step subtracts the learning rate (1.0 here) times the gradient from each score.</p>';
      qsa('.lg', pn).forEach(function(r){ r.addEventListener('input', function(){ logits[+r.dataset.i] = +r.value; note = ''; draw(); }); });
    }
    qs('[data-step]', ctl).addEventListener('click', function(){ var c = +ct.value, p = probs(), before = -Math.log(p[c]); logits = logits.map(function(l, i){ return l - 1.0 * (p[i] - (i === c ? 1 : 0)); }); var after = -Math.log(probs()[c]); note = 'One step: loss ' + before.toFixed(2) + ' to ' + after.toFixed(2) + '.'; draw(); });
    qs('[data-reset]', ctl).addEventListener('click', function(){ logits = [0.5, 1.2, -0.3, 0.8, -1, 0.1]; note = ''; draw(); }); ct.addEventListener('change', function(){ note = ''; draw(); });
    return { render: draw };
  } });

/* ---------- Step 7: counting parameters ---------- */
L8.add({ short: 'Counting parameters', title: 'Where a transformer keeps its parameters',
  change: 'Change the model size. Press Load lesson 04 GPT to load its settings and compare the count.',
  watch: 'A parameter is one learned number. A transformer keeps most of them in two places per layer: the attention weights, which are four square matrices of the model width d, and the feed-forward network, which has two matrices between d and f times d. The count per layer grows with d squared, so doubling the width roughly quadruples it. The embedding table holds one vector of d numbers per token, and a learned position table holds one per position. The output layer maps d back to the vocabulary, and it can share the embedding table. Storing the weights as 32-bit numbers takes 4 bytes each.',
  build: function(api){
    var lp = api.el('pn', 'left:28px;top:28px;width:560px;height:408px', '<h4>Parameters by part</h4><div class="bk"></div>');
    var rp = api.el('pn', 'left:608px;top:28px;width:324px;height:408px;overflow:auto', '<h4>Totals</h4><div class="tt"></div>');
    var ctl = api.addControl('Vocab <select class="vo"><option>50</option><option>1000</option><option>32000</option></select> Width <select class="dm"><option>32</option><option selected>64</option><option>128</option><option>256</option><option>512</option><option>1024</option><option>4096</option></select> Layers <input class="ly" type="range" min="1" max="48" step="1" value="2" style="width:100px"><b class="lv">2</b> Feed-forward <select class="ff"><option>2</option><option selected>4</option><option>8</option></select> Context <select class="cx"><option selected>48</option><option>512</option><option>2048</option></select> <label><input class="tie" type="checkbox"> Share output weights</label> <button type="button" class="btn sm ghost" data-gpt style="white-space:nowrap">Load lesson 04 GPT</button>');
    var vo = qs('.vo', ctl), dm = qs('.dm', ctl), ly = qs('.ly', ctl), lv = qs('.lv', ctl), ff = qs('.ff', ctl), cx = qs('.cx', ctl), tie = qs('.tie', ctl);
    function count(){ var V = +vo.value, d = +dm.value, L = +ly.value, f = +ff.value, T = +cx.value, emb = V * d, pos = T * d, att = L * (4 * d * d + 4 * d), fn = L * (2 * f * d * d + (f + 1) * d), ln = L * 4 * d + 2 * d, out = tie.checked ? 0 : d * V; return { emb: emb, pos: pos, att: att, fn: fn, ln: ln, out: out, total: emb + pos + att + fn + ln + out }; }
    function draw(){
      lv.textContent = ly.value; var c = count(), parts = [['Token embeddings', c.emb], ['Position embeddings', c.pos], ['Attention', c.att], ['Feed-forward', c.fn], ['Layer norms', c.ln], ['Output layer', c.out]], mx = Math.max.apply(null, parts.map(function(p){ return p[1]; })) || 1;
      qs('.bk', lp).innerHTML = parts.map(function(p){ return '<div style="display:grid;grid-template-columns:160px 1fr 110px;gap:10px;align-items:center;margin:11px 0;font-size:14px;color:' + C.char + '"><span>' + p[0] + '</span><div class="pbar" style="height:14px;margin:0"><i style="background:' + C.red + ';width:' + (p[1] / mx * 100).toFixed(1) + '%;transition:none"></i></div><b style="text-align:right">' + p[1].toLocaleString() + '</b></div>'; }).join('') + '<p class="note" style="font-size:12.5px;margin-top:12px">Per layer: attention 4d² + 4d, feed-forward 2fd² + (f + 1)d, two layer norms 4d. Embeddings V times d, positions T times d, output d times V.</p><div id="chk" style="font-size:13px;margin-top:6px"></div>';
      var mb = c.total * 4 / 1048576; qs('.tt', rp).innerHTML = '<div class="big" style="color:' + C.red + '">' + c.total.toLocaleString() + '</div><div class="note" style="font-size:12px;color:' + C.char + ';margin-bottom:10px">parameters</div>' + naRow('Weights in 32-bit floats', mb < 1 ? (mb * 1024).toFixed(0) + ' KB' : (mb >= 1024 ? (mb / 1024).toFixed(1) + ' GB' : mb.toFixed(1) + ' MB')) + naRow('Share in attention and feed-forward', Math.round((c.att + c.fn) / c.total * 100) + '%') + naRow('Attention score table per layer', (+cx.value * +cx.value).toLocaleString() + ' entries per head');
    }
    qs('[data-gpt]', ctl).addEventListener('click', function(){ var gv = String(G.V || 50); if (![].slice.call(vo.options).some(function(o){ return o.value === gv; })) { var op = document.createElement('option'); op.textContent = gv; vo.insertBefore(op, vo.firstChild); } vo.value = gv; dm.value = String(DM); ly.value = G.NL || 2; ff.value = '4'; cx.value = String(TLEN); tie.checked = false; draw(); var c = count(), V = G.V || 0, d = DM, L = G.NL || 2, T = TLEN, exp = V * d + T * d + L * (12 * d * d + 13 * d) + 2 * d + d * V; var el = qs('#chk', lp); if (el && G.params) el.innerHTML = 'This course\'s GPT in lesson 04 has <b>' + G.params.toLocaleString() + '</b> parameters with vocabulary ' + V + ', width ' + d + ', ' + L + ' layers and context ' + T + '. The formula gives <b>' + exp.toLocaleString() + '</b>' + (exp === G.params ? ', an exact match.' : '.'); else if (el) el.innerHTML = 'Open lesson 04 and train or load the GPT to compare with its real count.'; });
    [vo, dm, ff, cx, tie].forEach(function(e){ e.addEventListener('change', draw); }); ly.addEventListener('input', draw);
    return { render: draw };
  } });

L8.glossary([
  ['Loss', 'A number that measures how bad a prediction is. Training lowers it.'],
  ['Gradient', 'The slope of the loss with respect to a weight. It points uphill.'],
  ['Learning rate', 'The size of each weight update, as a multiple of the gradient.'],
  ['Momentum', 'Keeping a running sum of earlier gradients so steps follow a steady direction.'],
  ['Adam', 'An optimiser that divides each step by the recent size of the gradient.'],
  ['Backpropagation', 'Finding every gradient by applying the chain rule from the loss back to the inputs.'],
  ['Chain rule', 'The slope of a chain of steps is the product of the slopes of each step.'],
  ['Vanishing gradient', 'A gradient that shrinks toward zero as it travels back through many layers.'],
  ['Initialisation', 'The random values weights start from. Their size sets how well signals and gradients travel.'],
  ['Weight decay', 'A penalty on large weights that pulls each weight toward zero at every step.'],
  ['Batch size', 'How many compounds are used for one weight update.'],
  ['Query, key and value', 'The three vectors attention uses. A query is scored against keys, and the scores weight the values.'],
  ['Softmax', 'A function that turns scores into positive weights that add to 1.'],
  ['Logit', 'The raw score a model gives a token before the softmax.'],
  ['Cross-entropy', 'Minus the natural log of the probability the model gave the correct token.'],
  ['Perplexity', 'e raised to the loss. The number of equally likely tokens the model is choosing between.']
]);
setUses(L8, [[], [], [], [], [], [], []]);
