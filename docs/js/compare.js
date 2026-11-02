"use strict";
/* Lesson 6: Compare */
var L5 = new Lesson('compare', 'Compare');
L5.optsHTML = function(){ return optGroup('Endpoint', seg([['HLM', 'HLM clearance'], ['logD', 'logD']], EP, 'data-ep'), 'Scores are kept per endpoint.') + '<div class="og"><div class="ol">What is listed</div><div class="hint">All models trained on the selected endpoint, plus reference runs. Train models in the other lessons first.</div><div class="status" id="cst"></div></div>'; };
L5.onData = function(){ };
function cmpRows(){
  var rows = [], names = Object.keys(D.scores); names.forEach(function(k){ var m = D.scores[k]; rows.push({ n: k + (m.params ? ' (' + m.params + (m.auto ? ', default options' : '') + ')' : (m.auto ? ' (default options)' : '')), v: m.r2, t: m.secs, ref: false, m: m }); });
  var ft = EXTRAS && EXTRAS.chemprop_ft; if (ft && EP === 'HLM') { var NM = { scratch: 'from scratch', frozen: 'frozen encoder', full: 'full fine-tuning' }; Object.keys(NM).forEach(function(md){ var r = ft.runs.filter(function(x){ return x.mode === md && x.n === 1000; }); if (r.length) rows.push({ n: 'Chemprop, 1,000 HLM compounds, ' + NM[md], v: mean(r.map(function(x){ return x.r2; })), t: mean(r.map(function(x){ return x.secs; })), ref: true }); }); }
  var refs = (EXTRAS && EXTRAS.refs && EXTRAS.refs[EP]) || []; refs.forEach(function(r){ rows.push({ n: r[0], v: r[1], t: r[2], ref: true }); }); return rows.sort(function(a, b){ return b.v - a.v; });
}
var POP = null;
function populate(api){
  if (D.popDone) return Promise.resolve(); if (POP) return POP; var d = D;
  POP = (async function(){
    try { if (api) api.busy('Training the default models'); await tick(); await tick();
      if (activeFeats().length) { ['tree', 'forest', 'boost'].forEach(function(m){ if (!d.scores[TNAMES[m]]) recordMode(m, true); }); await tick(); if (!d.scores['Neural network']) { await nnReady(null); if (d.scores['Neural network']) d.scores['Neural network'].auto = true; } }
      if (window.gptLoadDemo) await gptLoadDemo(); d.popDone = true;
    } catch (e) { console.error(e); } finally { POP = null; if (api) api.idle(); }
  })();
  return POP;
}
function gptRow(){ return G && G.trained ? { n: 'GPT (' + G.NL + ' layers, ' + G.steps + ' steps)', t: G.secs } : null; }

L5.add({ short: 'Leaderboard', title: 'Every model you trained, with accuracy and time',
  change: 'Train models in the other lessons. Their rows replace the defaults.',
  watch: 'This chart ranks every model by R2 on the test compounds. R2 is the fraction of the variance of the measured values that the model explains: 1 is perfect and 0 equals predicting the mean. Red bars are models trained in this course. Until a model is trained, a red bar shows default settings. Blue-grey bars are reference runs made earlier with more features and longer training.',
  build: function(api){
    var pn = api.el('pn', 'left:28px;top:20px;width:904px;height:480px;overflow:auto', '<div class="lb"></div><p class="note" style="font-size:12px;margin-top:12px">R2 is measured on test compounds. Time covers fitting only. The GPT predicts tokens, not a value, so it has no R2.</p>');
    function draw(prog){
      var rows = cmpRows(), g = gptRow(), mx = Math.max(0.5, Math.max.apply(null, rows.map(function(r){ return r.v; }))), html = '<div class="hd">Model</div><div class="hd">R2 on test compounds</div><div class="hd" style="text-align:right">R2</div><div class="hd" style="text-align:right">Training time</div>';
      rows.forEach(function(r, i){ var vis = clamp(prog * rows.length - i * 0.6, 0, 1); html += '<div class="nm" style="opacity:' + (vis > 0.05 ? 1 : 0.15) + '">' + esc(r.n) + '</div><div class="tr"><i class="' + (r.ref ? 'ref' : '') + '" style="width:' + (clamp(r.v, 0, mx) / mx * 100 * vis) + '%;transition:none"></i></div><div class="vv">' + r.v.toFixed(2) + '</div><div class="vv" style="font-weight:500;color:#34373F">' + (r.t === null || r.t === undefined ? 'n/a' : fmtT(r.t)) + '</div>'; });
      if (g) html += '<div class="nm">' + esc(g.n) + '</div><div class="tr"></div><div class="vv">n/a</div><div class="vv" style="font-weight:500;color:#34373F">' + fmtT(g.t) + '</div>';
      qs('.lb', pn).innerHTML = html; var n = Object.keys(D.scores).length; var st = $('cst'); if (st) st.textContent = n ? n + ' model' + (n > 1 ? 's' : '') + ' trained on ' + EP + (g ? ' and a GPT' : '') + '.' : 'Nothing trained on ' + EP + ' yet.'; }
    async function render(){ await populate(api); draw(1); }
    async function play(tok){ draw(0); var rows = cmpRows(); await api.beat(Object.keys(D.scores).length ? 'Models trained in this session, next to the reference runs.' : 'No model has been trained on this endpoint yet. Only the reference runs are shown.', 2600); await api.anim(ANIM ? 3200 : 1, function(t){ draw(t); }); api.cap('Compare R2 first, then the training time that each gain in R2 required.'); await api.sleep(3600); }
    return { render: render, play: play };
  } });

L5.add({ short: 'Accuracy against time', title: 'Accuracy against training time',
  change: 'Train other settings and watch the points move.',
  watch: 'Each point is one model. The y axis is its R2 on the test compounds and the x axis is its fit time. The time axis is logarithmic, so each gridline is ten times longer than the one before. A model at the top left is both accurate and fast.',
  build: function(api){
    var pp = api.el('pn', 'left:28px;top:28px;width:640px;height:408px', '<h4>Test R2 against training time (log scale)</h4><canvas width="618" height="350"></canvas>');
    var sp = api.el('pn', 'left:688px;top:28px;width:244px;height:408px', '<h4>How to read it</h4><div class="rd" style="font-size:12.5px;color:#34373F"></div>');
    function pts(){ var out = []; Object.keys(D.scores).forEach(function(k){ var m = D.scores[k]; if (m.secs > 0) out.push({ n: k, t: Math.max(m.secs, 0.003), v: m.r2, ref: false }); }); ((EXTRAS && EXTRAS.refs && EXTRAS.refs[EP]) || []).forEach(function(r){ if (r[2]) out.push({ n: r[0].split(',')[0], t: r[2], v: r[1], ref: true }); }); return out; }
    function draw(prog){
      var P = pts(), g = cv(qs('canvas', pp)), x = g.x, box = { l: 52, r: g.w - 20, t: 14, b: g.h - 44 }; if (!P.length) { x.fillStyle = C.mid; x.fillText('Train a model first', 60, 40); return; }
      var lt = P.map(function(p){ return Math.log10(p.t); }), lo = Math.floor(Math.min.apply(null, lt)) - 0.2, hi = Math.ceil(Math.max.apply(null, lt)) + 0.2, vmx = Math.max(0.5, Math.max.apply(null, P.map(function(p){ return p.v; })) * 1.1), vmn = Math.min(0, Math.min.apply(null, P.map(function(p){ return p.v; })));
      var sx = function(l){ return box.l + (l - lo) / (hi - lo) * (box.r - box.l); }, sy = function(v){ return box.b - (v - vmn) / (vmx - vmn) * (box.b - box.t); }; x.strokeStyle = C.silver; x.fillStyle = C.mid; x.textAlign = 'right'; x.font = '12px Aptos, Segoe UI, sans-serif';
      for (var k = 0; k <= 4; k++) { var v = vmn + (vmx - vmn) * k / 4, y = sy(v); x.beginPath(); x.moveTo(box.l, y); x.lineTo(box.r, y); x.stroke(); x.fillText(v.toFixed(2), box.l - 6, y + 4); } x.textAlign = 'center'; for (var e = Math.ceil(lo); e <= hi; e++) { x.beginPath(); x.moveTo(sx(e), box.t); x.lineTo(sx(e), box.b); x.stroke(); var s = Math.pow(10, e); x.fillText(s >= 1 ? s + ' s' : s + ' s', sx(e), box.b + 16); }
      x.fillStyle = C.char; x.font = '600 12px Aptos, Segoe UI, sans-serif'; x.fillText('training time', (box.l + box.r) / 2, g.h - 6); x.save(); x.translate(13, (box.t + box.b) / 2); x.rotate(-Math.PI / 2); x.fillText('test R2', 0, 0); x.restore();
      P.forEach(function(p, i){ var vis = clamp(prog * P.length - i * 0.7, 0, 1); if (vis <= 0) return; var X = sx(Math.log10(p.t)), Y = sy(p.v); x.globalAlpha = vis; x.fillStyle = p.ref ? C.bg : C.red; x.beginPath(); x.arc(X, Y, 7 * (0.5 + 0.5 * vis), 0, 6.3); x.fill(); x.globalAlpha = 1; });
      var placed = [], lab = P.map(function(p){ return { p: p, X: sx(Math.log10(p.t)), Y: sy(p.v) }; }).filter(function(o, i){ return clamp(prog * P.length - i * 0.7, 0, 1) > 0.5; });
      lab.forEach(function(o){ x.font = '600 11px Aptos, Segoe UI, sans-serif'; var t = o.p.n.length > 26 ? o.p.n.slice(0, 26) : o.p.n, w = x.measureText(t).width, c = [[o.X + 11, o.Y + 4], [o.X - 11 - w, o.Y + 4], [o.X - w / 2, o.Y - 11], [o.X - w / 2, o.Y + 19], [o.X + 11, o.Y - 12], [o.X + 11, o.Y + 20]], pick = null;
        c.forEach(function(q){ if (pick) return; var r = { l: q[0] - 2, r: q[0] + w + 2, t: q[1] - 11, b: q[1] + 4 }; if (r.l < box.l - 4 || r.r > g.w - 2 || r.t < box.t - 4 || r.b > box.b + 4) return; if (placed.some(function(z){ return r.l < z.r && r.r > z.l && r.t < z.b && r.b > z.t; })) return; pick = { q: q, r: r }; });
        if (pick) { placed.push(pick.r); x.fillStyle = C.char; x.textAlign = 'left'; x.fillText(t, pick.q[0], pick.q[1]); } });
      qs('.rd', sp).innerHTML = '<p><b style="color:' + C.red + '">Red</b>: trained in this session.</p><p><b style="color:#A1B1C4">Blue grey</b>: reference runs that report a time.</p><p>Moving right costs time. Moving up buys accuracy. A point that is both slower and less accurate than another is dominated.</p><p class="note" style="color:#34373F">The time axis is logarithmic, so each gridline is ten times slower.</p>'; }
    async function render(){ await populate(api); draw(1); }
    async function play(tok){ draw(0); await api.beat('Each model is placed by its test R2 and its training time.', 2400); await api.anim(ANIM ? 3000 : 1, function(t){ draw(t); }); api.cap('The top left corner is both accurate and fast.'); await api.sleep(3200); }
    return { render: render, play: play };
  } });

/* Ensembles of model types. Predictions for the 308 held-out training compounds and the 617 test compounds of the HLM scaffold split, from five model types
   (data/ensemble_preds.json, made offline from the public HLM set). The weights are fitted on the held-out training compounds only. */
var ENS = null;
function ensProj(v){
  var u = v.slice().sort(function(a, b){ return b - a; }), css = 0, th = 0, i;
  for (i = 0; i < u.length; i++) { css += u[i]; var t = (css - 1) / (i + 1); if (u[i] - t > 0) th = t; }
  return v.map(function(x){ return Math.max(x - th, 0); });
}
function ensWeights(P, y){
  var k = P.length, n = y.length, w = P.map(function(){ return 1 / k; }), tr = 0, i, j, it;
  for (j = 0; j < k; j++) for (i = 0; i < n; i++) tr += P[j][i] * P[j][i];
  var lr = 0.5 * n / (2 * tr);
  for (it = 0; it < 4000; it++) {
    var g = new Array(k).fill(0);
    for (i = 0; i < n; i++) { var p = 0; for (j = 0; j < k; j++) p += w[j] * P[j][i]; var e = p - y[i]; for (j = 0; j < k; j++) g[j] += 2 * e * P[j][i] / n; }
    w = ensProj(w.map(function(x, q){ return x - lr * g[q]; }));
  }
  return w;
}
function ensRmse(a, y){ var s = 0; for (var i = 0; i < y.length; i++) s += (a[i] - y[i]) * (a[i] - y[i]); return Math.sqrt(s / y.length); }
function ensR2(a, y){ var m = mean(y), s = 0, t = 0; for (var i = 0; i < y.length; i++) { s += (a[i] - y[i]) * (a[i] - y[i]); t += (y[i] - m) * (y[i] - m); } return 1 - s / t; }
function ensCorr(a, b){ var ma = mean(a), mb = mean(b), s = 0, x = 0, z = 0; for (var i = 0; i < a.length; i++) { s += (a[i] - ma) * (b[i] - mb); x += (a[i] - ma) * (a[i] - ma); z += (b[i] - mb) * (b[i] - mb); } return s / Math.sqrt(x * z); }
var ENS_SHORT = { 'LightGBM': 'LGBM', 'Random forest': 'RF', 'Neural network': 'NN', 'Nearest neighbours': 'kNN', 'Chemprop': 'CP' };
var ENS_DESC = { 'LightGBM': 'Descriptors and Morgan bits', 'Random forest': 'Descriptors and Morgan bits', 'Neural network': 'Descriptors', 'Nearest neighbours': 'Morgan bits, Tanimoto, k = 5', 'Chemprop': 'Message passing network, lesson 06' };

L5.add({ short: 'Model-type ensembles', title: 'Ensembles of model types',
  change: 'Tick models to include them. Compare the two ensembles with the best single model.',
  watch: 'This step combines the predictions of five model types fitted to the HLM training compounds: LightGBM, a random forest, a neural network on descriptors, a nearest-neighbour model on fingerprints, and the saved Chemprop run of lesson 06. The simple average gives every included model the same weight. The weighted average uses weights that are non-negative, sum to 1 and minimise the squared error on 308 training compounds held out from fitting, so the 617 test compounds play no part in choosing them. The dot plot gives the RMSE of every model and of both ensembles on the test compounds, in log10 units, with values at the assay floor included as in the Leaderboard. The matrix gives the correlation between the prediction errors of each pair of models.',
  build: function(api){
    var lp = api.el('pn', 'left:28px;top:20px;width:344px;height:500px;overflow:auto', '<h4>Models</h4><div class="mt"></div>');
    var cp = api.el('pn', 'left:388px;top:20px;width:544px;height:262px', '<h4>RMSE on 617 test compounds</h4><canvas width="516" height="214"></canvas>');
    var mp = api.el('pn', 'left:388px;top:294px;width:282px;height:226px', '<h4>Correlation of errors</h4><div class="cm"></div>');
    var rp = api.el('pn', 'left:682px;top:294px;width:250px;height:226px;overflow:auto', '<div class="rd" style="font-size:12px;line-height:1.5;color:' + C.char + '"></div>');
    var sel = null, W = null;
    function compute(){
      var J = ENS, names = J.models, S = names.filter(function(n, i){ return sel[i]; }), R = {};
      R.S = S; R.single = {}; names.forEach(function(n){ R.single[n] = { rmse: ensRmse(J.pred_test[n], J.y_test), r2: ensR2(J.pred_test[n], J.y_test), vrmse: ensRmse(J.pred_val[n], J.y_val) }; });
      if (S.length >= 2) {
        var avg = J.y_test.map(function(_, i){ return mean(S.map(function(n){ return J.pred_test[n][i]; })); });
        R.avg = { rmse: ensRmse(avg, J.y_test), r2: ensR2(avg, J.y_test) };
        var w = ensWeights(S.map(function(n){ return J.pred_val[n]; }), J.y_val), wp = J.y_test.map(function(_, i){ var s = 0; S.forEach(function(n, q){ s += w[q] * J.pred_test[n][i]; }); return s; });
        R.w = w; R.wt = { rmse: ensRmse(wp, J.y_test), r2: ensR2(wp, J.y_test) };
        var E = S.map(function(n){ return J.pred_test[n].map(function(v, i){ return v - J.y_test[i]; }); });
        R.corr = S.map(function(a, i){ return S.map(function(b, j){ return ensCorr(E[i], E[j]); }); });
        var cs = [], i, j; for (i = 0; i < S.length; i++) for (j = i + 1; j < S.length; j++) cs.push(R.corr[i][j]); R.mc = mean(cs);
      }
      var best = null; S.forEach(function(n){ if (!best || R.single[n].rmse < R.single[best].rmse) best = n; }); R.best = best;
      return R;
    }
    function draw(){
      if (!ENS) return; var J = ENS, R = compute(), names = J.models;
      qs('.mt', lp).innerHTML = '<table style="width:100%;border-collapse:collapse;font-size:12px"><tr style="color:' + C.mid + ';font-size:11px"><td></td><td style="padding:2px 4px">Model</td><td style="text-align:right;padding:2px 4px">Test RMSE</td><td style="text-align:right;padding:2px 4px">Test R2</td><td style="text-align:right;padding:2px 4px">Weight</td></tr>' +
        names.map(function(n, i){ var q = R.S.indexOf(n), wv = R.w && q >= 0 ? R.w[q].toFixed(2) : (sel[i] ? 'n/a' : ''); return '<tr style="border-top:1px solid ' + C.silver + (sel[i] ? '' : ';color:' + C.mid) + '"><td style="padding:5px 4px"><input type="checkbox" data-i="' + i + '"' + (sel[i] ? ' checked' : '') + '></td><td style="padding:5px 4px"><b>' + esc(n) + '</b><br><span style="font-size:11px;color:' + C.mid + '">' + esc(ENS_DESC[n] || '') + '</span></td><td style="text-align:right;padding:5px 4px">' + R.single[n].rmse.toFixed(3) + '</td><td style="text-align:right;padding:5px 4px">' + R.single[n].r2.toFixed(2) + '</td><td style="text-align:right;padding:5px 4px">' + wv + '</td></tr>'; }).join('') + '</table>' +
        '<p style="font-size:11.5px;line-height:1.45;color:' + C.char + ';margin:10px 0 0">All models were fitted to the same 2,162 training compounds. The weights are fitted on the 308 compounds held out from them. Chemprop stopped training on those 308 compounds, so its held-out predictions are somewhat more favourable than its test predictions.</p>';
      qsa('input[type=checkbox]', lp).forEach(function(b){ b.addEventListener('change', function(){ sel[+b.dataset.i] = b.checked; if (sel.filter(Boolean).length < 1) { sel[+b.dataset.i] = true; } draw(); }); });
      /* dot plot */
      var g = cv(qs('canvas', cp)), x = g.x, rows = names.map(function(n, i){ return { n: n, v: R.single[n].rmse, k: 's', on: sel[i] }; });
      rows.sort(function(a, b){ return a.v - b.v; }); if (R.avg) { rows.push({ n: 'Simple average', v: R.avg.rmse, k: 'e', on: true }); rows.push({ n: 'Weighted average', v: R.wt.rmse, k: 'w', on: true }); }
      var vs = rows.map(function(r){ return r.v; }), lo = Math.floor(Math.min.apply(null, vs) * 20) / 20 - 0.05, hi = Math.ceil(Math.max.apply(null, vs) * 20) / 20 + 0.0, L = 134, Rr = g.w - 16, T = 10, B = g.h - 36, rh = (B - T) / rows.length;
      var px = function(v){ return L + (v - lo) / (hi - lo) * (Rr - L); };
      x.strokeStyle = C.silver; x.fillStyle = C.mid; x.font = '11px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'center';
      for (var t = Math.ceil(lo * 20) / 20; t <= hi + 1e-9; t += 0.05) { x.beginPath(); x.moveTo(px(t), T); x.lineTo(px(t), B); x.stroke(); x.fillText(t.toFixed(2), px(t), B + 14); }
      x.fillStyle = C.char; x.font = '600 11.5px Aptos, "Segoe UI", sans-serif'; x.fillText('RMSE of log10 clearance, test compounds', (L + Rr) / 2, g.h - 4);
      if (R.best) { x.strokeStyle = C.char; x.setLineDash([4, 3]); x.beginPath(); x.moveTo(px(R.single[R.best].rmse), T); x.lineTo(px(R.single[R.best].rmse), B); x.stroke(); x.setLineDash([]); }
      rows.forEach(function(r, i){ var y = T + rh * (i + 0.5); x.textAlign = 'right'; x.font = (r.k === 's' ? '12px' : '600 12px') + ' Aptos, "Segoe UI", sans-serif'; x.fillStyle = r.on ? C.char : C.bg; x.fillText(r.n + (r.on ? '' : ' (not included)'), L - 8, y + 4);
        x.fillStyle = r.k === 's' ? (r.on ? C.char : C.silver) : (r.k === 'e' ? C.red : C.dred); x.beginPath(); x.arc(px(r.v), y, r.k === 's' ? 5 : 6.5, 0, 6.2832); x.fill();
        x.fillStyle = r.on ? C.char : C.bg; x.textAlign = 'left'; x.font = '11px Aptos, "Segoe UI", sans-serif'; x.fillText(r.v.toFixed(3), px(r.v) + 10, y + 4); });
      /* correlation matrix */
      var S = R.S, cm = qs('.cm', mp);
      if (S.length < 2) cm.innerHTML = '<div style="font-size:12px;color:' + C.mid + '">Include at least two models.</div>';
      else cm.innerHTML = '<table style="border-collapse:collapse;font-size:11px;width:100%"><tr><td></td>' + S.map(function(n){ return '<td style="text-align:center;padding:1px 2px;color:' + C.mid + '">' + esc(ENS_SHORT[n] || n) + '</td>'; }).join('') + '</tr>' + S.map(function(a, i){ return '<tr><td style="padding:1px 3px 1px 0;color:' + C.mid + ';white-space:nowrap">' + esc(ENS_SHORT[a] || a) + '</td>' + S.map(function(b, j){ var c = R.corr[i][j], al = clamp((c - 0.4) / 0.6, 0, 1); return '<td style="text-align:center;padding:4px 2px;background:rgba(225,36,42,' + (0.08 + 0.55 * al).toFixed(2) + ');color:' + C.char + '">' + c.toFixed(2) + '</td>'; }).join('') + '</tr>'; }).join('') + '</table>';
      /* readout */
      var rd = qs('.rd', rp), bs = R.best ? R.single[R.best].rmse : null;
      if (!R.avg) rd.innerHTML = 'One model is included. Include a second model to form an ensemble.';
      else {
        var d1 = R.avg.rmse - bs, d2 = R.wt.rmse - bs;
        rd.innerHTML = '<div><b>Best included single model</b>: ' + esc(R.best) + ', RMSE ' + bs.toFixed(3) + '.</div><div><b>Simple average</b>: RMSE ' + R.avg.rmse.toFixed(3) + ', R2 ' + R.avg.r2.toFixed(2) + '. It is ' + Math.abs(d1).toFixed(3) + (d1 > 0 ? ' higher' : ' lower') + ' than the best single model.</div><div><b>Weighted average</b>: RMSE ' + R.wt.rmse.toFixed(3) + ', R2 ' + R.wt.r2.toFixed(2) + '. It is ' + Math.abs(d2).toFixed(3) + (d2 > 0 ? ' higher' : ' lower') + ' than the best single model.</div><div><b>Mean correlation of errors</b>: ' + R.mc.toFixed(2) + '.</div><div style="font-size:11px;color:' + C.mid + ';margin-top:6px">LGBM LightGBM, RF random forest, NN neural network, kNN nearest neighbours, CP Chemprop. Pearson correlation of prediction errors on the test compounds.</div>';
      }
    }
    async function render(){
      if (!ENS) { api.busy('Loading predictions'); try { ENS = await loadJSON('ensemble_preds'); } finally { api.idle(); } }
      if (!sel) sel = ENS.models.map(function(){ return true; });
      draw();
    }
    async function play(tok){
      await render(); sel = ENS.models.map(function(){ return true; }); draw();
      await api.beat('All five model types are included. The dashed line marks the best single model.', 3200);
      await api.beat('The simple average and the weighted average are placed against it.', 3000);
      sel = ENS.models.map(function(n){ return n === 'LightGBM' || n === 'Chemprop'; }); draw();
      await api.beat('With only LightGBM and Chemprop included, the same two ensembles are recomputed.', 3400);
      sel = ENS.models.map(function(){ return true; }); draw(); api.cap(null);
    }
    return { render: render, play: play };
  } });

L5.add({ short: 'Discussion', title: 'Questions about a result',
  change: 'Select a question and answer it from the scores obtained.',
  watch: 'A score depends on how it was produced. These six questions cover the split, the model type, the metric, censored values, assay noise and the GPT. Select one and answer it from the results of the earlier lessons.',
  build: function(api){
    var qs_ = ['Would the same model win on another split of the data? What would you run to find out?', 'Why do tree ensembles often match or beat neural networks on a few thousand compounds?', 'Which score did you report, and on which compounds? Would a colleague get the same answer from the same data?', 'Does handling of censored values change which model looks best? Try the filters in each model lesson.', 'How does assay noise limit the best possible R2? How would you estimate that limit?', 'The GPT has no R2. What would you do with it in a project: predict, generate, or embed compounds?'];
    var els = qs_.map(function(q, i){ return api.el('pn fd', 'left:' + (28 + (i % 2) * 456) + 'px;top:' + (28 + Math.floor(i / 2) * 134) + 'px;width:448px;height:122px', '<h4>Question ' + (i + 1) + '</h4><div style="font-size:15px;line-height:1.45;color:#34373F">' + esc(q) + '</div>'); });
    function render(){ els.forEach(function(e){ e.classList.add('on'); }); }
    async function play(tok){ els.forEach(function(e){ e.classList.remove('on'); }); for (var i = 0; i < els.length; i++) { els[i].classList.add('on'); await api.sleep(1200); } }
    return { render: render, play: play };
  } });

L5.glossary([
  ['R2', 'Fraction of the variance of the measured values explained by the predictions on the test compounds. 1 is perfect and 0 equals predicting the mean.'],
  ['Spearman', 'Rank correlation between predictions and measurements. It ignores the exact values.'], ['RMSE', 'Root mean squared error, in the units of the endpoint.'],
  ['Training time', 'How long fitting the model took, excluding drawing and animation.'], ['Reference run', 'A published or previously measured result shown for comparison.'],
  ['Dominated model', 'A model that another model beats on both accuracy and speed.'], ['Test compounds', 'Compounds held back from training, chosen by scaffold.'],
  ['Assay noise', 'Random measurement error in the assay data. It limits the R2 any model can reach.']
]);
setUses(L5, [['endpoint'], ['endpoint'], []]);
