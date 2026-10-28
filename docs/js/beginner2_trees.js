"use strict";
/* Extra beginner steps for lessons 02 (trees) and 08 (tree algorithms).
   Written for chemists with no machine learning experience.
   Each step assumes only the earlier beginner steps. */

/* ---------- shared helpers ---------- */
function bt2Sample(key, n, seed){
  var k = EP + key; if (D.cache[k]) return D.cache[k]; var r = rng(seed), sh = D.TRAIN.slice();
  for (var a = sh.length - 1; a > 0; a--) { var j = Math.floor(r() * (a + 1)), t = sh[a]; sh[a] = sh[j]; sh[j] = t; }
  return (D.cache[k] = sh.slice(0, n));
}
/* sorted prefix sums of the lab value for one feature, over all training compounds: any cut can then be scored at once */
function bt2Pre(f){
  var k = EP + 'pre' + f; if (D.cache[k]) return D.cache[k]; var idx = D.TRAIN.slice().sort(function(a, b){ return D.COL[f][a] - D.COL[f][b]; }), n = idx.length, xs = new Float64Array(n), cs = new Float64Array(n + 1), cs2 = new Float64Array(n + 1);
  idx.forEach(function(i, q){ xs[q] = D.COL[f][i]; var y = D.Y[i]; cs[q + 1] = cs[q] + y; cs2[q + 1] = cs2[q] + y * y; });
  return (D.cache[k] = { xs: xs, cs: cs, cs2: cs2, n: n });
}
/* RMSE of the predictions when the first m compounds form one group and the rest form another */
function bt2Miss(P, m){ var n = P.n, T = P.cs[n], T2 = P.cs2[n], sl = P.cs2[m] - P.cs[m] * P.cs[m] / m, sr = (T2 - P.cs2[m]) - (T - P.cs[m]) * (T - P.cs[m]) / (n - m); return Math.sqrt(Math.max(0, sl + sr) / n); }
function bt2Base(P){ var n = P.n, T = P.cs[n]; return Math.sqrt(Math.max(0, P.cs2[n] / n - (T / n) * (T / n))); }
function bt2Best(P){ var best = 20, bv = Infinity; for (var m = 20; m <= P.n - 20; m++) { var v = bt2Miss(P, m); if (v < bv) { bv = v; best = m; } } return { m: best, v: bv }; }
function bt2Num(v, d){ return (Math.abs(v) < Math.pow(10, -d) / 2 ? 0 : v).toFixed(d); }

/* ====================================================================
   Lesson 02, step: how a tree chooses its question
   ==================================================================== */
LESSONS.trees.add({ level: 'beginner', short: 'Choosing the best question', title: 'Split selection',
  change: 'Drag the cut position. Select a feature. Press Best cut to move to the minimum.',
  watch: 'A tree node splits the compounds on one feature at one threshold, for example logP at most 3. Each side predicts the mean lab value of its compounds. The residual of a compound is its lab value minus the prediction. RMSE is the root of the mean squared residual over all compounds, so large residuals weigh more. The tree evaluates every feature and every threshold, and keeps the split with the lowest RMSE. The curve on the right shows the RMSE at every threshold of the selected feature. Its minimum is the best split for that feature.',
  build: function(api){
    var feat = 'logP', m = 0, FE = ['logP', 'TPSA', 'MW'];
    var cp = api.el('pn', 'left:28px;top:28px;width:560px;height:408px', '<h4 class="t"></h4><canvas width="530" height="350"></canvas>');
    var rp = api.el('pn', 'left:608px;top:28px;width:324px;height:408px;overflow:hidden', '<div class="nb" style="font-size:13px;line-height:1.45;color:' + C.char + '"></div><canvas class="c2" width="294" height="180" style="margin-top:4px"></canvas>');
    var ctl = api.addControl('Feature ' + inSeg(FE, 0, 'data-f') + ' Cut position <input type="range" class="cut" style="width:240px"> <button type="button" class="btn sm" data-best>Best cut</button>');
    var rg = qs('.cut', ctl);
    function setup(){ var P = bt2Pre(feat), b = bt2Best(P); rg.min = 20; rg.max = P.n - 20; rg.step = Math.max(1, Math.round(P.n / 300)); return { P: P, b: b }; }
    function draw(){
      var S = setup(), P = S.P, n = P.n; m = clamp(m || Math.round(n * 0.15), 20, n - 20); rg.value = m;
      var cutV = P.xs[m], xlo = P.xs[Math.floor(n * 0.01)], xhi = P.xs[Math.floor(n * 0.99)], miss = bt2Miss(P, m), base = bt2Base(P), ml = P.cs[m] / m, mr = (P.cs[n] - P.cs[m]) / (n - m);
      qs('.t', cp).textContent = 'Compounds split at the cut. Each side predicts its mean.';
      var g = cv(qs('canvas', cp)), x = g.x, box = { l: 48, r: g.w - 14, t: 14, b: g.h - 40 }, sx = function(v){ return box.l + (clamp(v, xlo, xhi) - xlo) / (xhi - xlo) * (box.r - box.l); }, sy = function(v){ return box.b - (v - D.ylo) / (D.yhi - D.ylo) * (box.b - box.t); };
      axes(g, box, [xlo, xhi], [D.ylo, D.yhi], feat, 'lab value', 4);
      var cx = sx(cutV); x.fillStyle = 'rgba(161,177,196,.18)'; x.fillRect(box.l, box.t, cx - box.l, box.b - box.t); x.fillStyle = 'rgba(225,36,42,.07)'; x.fillRect(cx, box.t, box.r - cx, box.b - box.t);
      x.fillStyle = 'rgba(52,55,63,.5)'; bt2Sample('bt2dots', 380, 4).forEach(function(i){ x.beginPath(); x.arc(sx(D.COL[feat][i]), sy(D.Y[i]), 2.6, 0, 6.3); x.fill(); });
      x.strokeStyle = C.char; x.lineWidth = 2; x.beginPath(); x.moveTo(box.l + 4, sy(ml)); x.lineTo(cx - 4, sy(ml)); x.stroke(); x.strokeStyle = C.red; x.beginPath(); x.moveTo(cx + 4, sy(mr)); x.lineTo(box.r - 4, sy(mr)); x.stroke();
      x.strokeStyle = C.char; x.setLineDash([5, 4]); x.lineWidth = 1.5; x.beginPath(); x.moveTo(cx, box.t); x.lineTo(cx, box.b); x.stroke(); x.setLineDash([]);
      x.font = '700 12px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'center'; x.fillStyle = C.char; x.fillText('mean ' + ml.toFixed(2), (box.l + cx) / 2, box.t + 14); x.fillStyle = C.red; x.fillText('mean ' + mr.toFixed(2), (cx + box.r) / 2, box.t + 14);
      var rows = FE.filter(function(f){ return D.COL[f]; }).map(function(f){ var Q = bt2Pre(f), bb = bt2Best(Q); return { f: f, m: bb.m, v: bb.v, c: Q.xs[bb.m] }; }), win = rows.reduce(function(a, b){ return b.v < a.v ? b : a; });
      qs('.nb', rp).innerHTML = '<div class="ol">Split: ' + feat + ' at most ' + fmtFeat(feat, cutV) + '</div><div style="display:flex;gap:18px;align-items:flex-end"><div><div class="big" style="color:' + C.red + ';font-size:30px;margin:0">' + miss.toFixed(2) + '</div><div style="font-size:12px">RMSE with this split</div></div><div><div class="big" style="font-size:22px;margin:0;color:' + C.mid + '">' + base.toFixed(2) + '</div><div style="font-size:12px">RMSE with no split</div></div></div>' +
        '<div style="margin-top:8px;font-size:12.5px">Best threshold for ' + feat + ': at most <b>' + fmtFeat(feat, S.b.m < n ? P.xs[S.b.m] : 0) + '</b>, RMSE <b>' + S.b.v.toFixed(2) + '</b>. ' + (Math.abs(miss - S.b.v) < 0.003 ? '<b style="color:#1E6B40">The cut is at the minimum.</b>' : 'The cut is not at the minimum.') + '</div>' +
        '<div style="margin-top:6px;font-size:12.5px">Lowest RMSE by feature: ' + rows.map(function(r){ return (r.f === win.f ? '<b>' : '') + r.f + ' ' + r.v.toFixed(2) + (r.f === win.f ? '</b>' : ''); }).join(', ') + '. The tree splits on <b>' + win.f + '</b> first.</div>';
      var pts = []; for (var q = 20; q <= n - 20; q += Math.max(1, Math.round(n / 120))) if (P.xs[q] >= xlo && P.xs[q] <= xhi) pts.push([P.xs[q], bt2Miss(P, q)]);
      var lo = Math.floor((S.b.v - 0.01) * 50) / 50, hi = Math.ceil((base + 0.01) * 50) / 50;
      lineChart(qs('.c2', rp), [{ pts: pts, color: C.char, w: 2.4, nodot: true, label: 'every threshold' }, { pts: [[xlo, base], [xhi, base]], color: C.bg, w: 1.6, dash: [5, 4], nodot: true, label: 'no split' }, { pts: [[cutV, miss]], color: C.red, w: 2, label: 'selected cut' }], { xl: feat + ' threshold', yl: 'RMSE', x0: xlo, x1: xhi, y0: lo, y1: hi });
    }
    rg.addEventListener('input', function(){ m = +rg.value; draw(); });
    qsa('[data-f] button', ctl).forEach(function(b){ b.addEventListener('click', function(){ feat = FE[+b.dataset.v]; qsa('[data-f] button', ctl).forEach(function(z){ z.classList.toggle('on', z === b); }); m = Math.round(bt2Pre(feat).n * 0.15); draw(); }); });
    qs('[data-best]', ctl).addEventListener('click', function(){ var from = m, to = bt2Best(bt2Pre(feat)).m; animateFree(900, function(t){ m = Math.round(from + (to - from) * t); draw(); }); });
    async function render(){ draw(); }
    async function play(tok){
      var P = bt2Pre(feat), n = P.n, b = bt2Best(P).m; m = Math.round(n * 0.12); draw(); api.cap('A threshold near the lower edge isolates a few compounds. The other group holds almost all of them, so the RMSE barely falls.'); await api.sleep(3600);
      await api.anim(3600, function(t){ m = Math.round(n * 0.12 + (n * 0.88 - n * 0.12) * t); draw(); }); api.cap('Moving the threshold changes the two groups and the RMSE.'); await api.sleep(2200);
      var from = m; await api.anim(1400, function(t){ m = Math.round(from + (b - from) * t); draw(); }); api.cap('The minimum of the curve is the best threshold. The tree applies it, then repeats the search within each group.'); await api.sleep(3600); api.cap(null);
    }
    return { render: render, play: play };
  } });

/* ====================================================================
   Lesson 02, step: random forest
   ==================================================================== */
LESSONS.trees.add({ level: 'beginner', short: 'A crowd of trees', title: 'Random forests',
  change: 'Drag the number of trees. Press Next compound to select another compound.',
  watch: 'The prediction of a single tree depends on the particular training compounds it was fitted to, so it has high variance. A random forest fits many trees. Each tree is trained on a bootstrap sample of the training compounds, and each split considers a random subset of the features. The trees therefore differ. The forest prediction is the mean of the tree predictions, and averaging reduces the variance. Each dot is the prediction of one tree for one compound. The red line is the forest mean. The dashed line is the lab value. The figures on the right compare the RMSE of a single tree with the RMSE of the forest on 300 test compounds.',
  build: function(api){
    var K = 6, cur = 0, NT = 15, fr = null;
    var cp = api.el('pn', 'left:28px;top:28px;width:600px;height:408px', '<h4>Prediction of each tree for this compound</h4><canvas width="570" height="350"></canvas>');
    var rp = api.el('pn', 'left:648px;top:28px;width:284px;height:408px;overflow:auto', '<div class="nb" style="font-size:13px;line-height:1.45;color:' + C.char + '"></div>');
    var ctl = api.addControl('Trees in the forest <input type="range" min="1" max="15" step="1" value="6" style="width:200px"> <b class="kv">6</b> <button type="button" class="btn sm ghost" data-next>Next compound</button>');
    var rg = qs('input', ctl), kv = qs('.kv', ctl);
    function prep(){
      var key = EP + 'crowd'; if (D.cache[key]) return D.cache[key]; var feats = ['logP', 'TPSA', 'MW'].filter(function(f){ return D.COL[f]; }), f = fitForest(feats, D.TRAIN, NT, 6, 7), r = rng(12), sh = D.TEST.slice();
      for (var a = sh.length - 1; a > 0; a--) { var j = Math.floor(r() * (a + 1)), t = sh[a]; sh[a] = sh[j]; sh[j] = t; }
      var ev = sh.slice(0, 300), one = [], crowd = [], acc = new Float64Array(D.N);
      f.per.forEach(function(p, k){ one.push(rmse(ev, p)); for (var i = 0; i < D.N; i++) acc[i] += p[i]; var avg = new Float64Array(D.N); for (var q = 0; q < D.N; q++) avg[q] = acc[q] / (k + 1); crowd.push(rmse(ev, avg)); });
      return (D.cache[key] = { fr: f, order: sh, one: mean(one), crowd: crowd });
    }
    function draw(){
      var S = prep(), i = S.order[cur % S.order.length], k = K; kv.textContent = k;
      var g = cv(qs('canvas', cp)), x = g.x, box = { l: 30, r: g.w - 24, t: 30, b: g.h - 54 }, sx = function(v){ return box.l + (clamp(v, D.ylo, D.yhi) - D.ylo) / (D.yhi - D.ylo) * (box.r - box.l); };
      var guesses = S.fr.per.slice(0, k).map(function(p){ return p[i]; }), avg = mean(guesses), truth = D.Y[i];
      x.strokeStyle = C.silver; x.lineWidth = 1; x.fillStyle = C.mid; x.font = '11px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'center'; var tk = niceTicks(D.ylo, D.yhi, 6); tk.v.forEach(function(v){ x.beginPath(); x.moveTo(sx(v), box.t); x.lineTo(sx(v), box.b); x.stroke(); x.fillText(tickText(v, tk.step), sx(v), box.b + 16); });
      x.fillStyle = C.char; x.font = '600 12px Aptos, "Segoe UI", sans-serif'; x.fillText('predicted value (' + unitWord() + ')', (box.l + box.r) / 2, box.b + 36);
      var rj = rng(i + 5); guesses.forEach(function(v, q){ var y = box.t + 22 + (q % 8) * ((box.b - box.t - 44) / 8) + rj() * 6; x.fillStyle = 'rgba(52,55,63,.7)'; x.beginPath(); x.arc(sx(v), y, 6, 0, 6.3); x.fill(); x.strokeStyle = 'rgba(52,55,63,.18)'; x.beginPath(); x.moveTo(sx(v), y); x.lineTo(sx(avg), y); x.stroke(); });
      x.strokeStyle = C.red; x.lineWidth = 3; x.beginPath(); x.moveTo(sx(avg), box.t); x.lineTo(sx(avg), box.b); x.stroke(); x.fillStyle = C.red; x.font = '700 12px Aptos, "Segoe UI", sans-serif'; x.textAlign = sx(avg) > box.r - 120 ? 'right' : 'left'; x.fillText('forest mean ' + avg.toFixed(2), sx(avg) + (x.textAlign === 'left' ? 6 : -6), box.t - 6 + 4);
      x.strokeStyle = C.char; x.lineWidth = 2.5; x.setLineDash([6, 4]); x.beginPath(); x.moveTo(sx(truth), box.t + 10); x.lineTo(sx(truth), box.b); x.stroke(); x.setLineDash([]); x.fillStyle = C.char; x.textAlign = Math.abs(sx(truth) - sx(avg)) < 150 ? (sx(truth) < sx(avg) ? 'right' : 'left') : 'left'; x.fillText('lab ' + truth.toFixed(2), sx(truth) + (x.textAlign === 'left' ? 6 : -6), box.b - 6);
      var spread = Math.max.apply(null, guesses) - Math.min.apply(null, guesses);
      qs('.nb', rp).innerHTML = '<div class="ol">This compound</div><p style="margin:2px 0 8px">The ' + k + ' tree' + (k > 1 ? 's' : '') + ' predict between <b>' + Math.min.apply(null, guesses).toFixed(2) + '</b> and <b>' + Math.max.apply(null, guesses).toFixed(2) + '</b>. The forest mean is <b style="color:' + C.red + '">' + avg.toFixed(2) + '</b>. The lab value is <b>' + truth.toFixed(2) + '</b>.</p>' +
        '<div class="ol" style="margin-top:12px">On 300 test compounds</div><div style="display:flex;gap:16px;margin-top:4px"><div><div class="big" style="color:' + C.mid + ';font-size:28px;margin:0">' + S.one.toFixed(2) + '</div><div style="font-size:12px">RMSE, single tree (mean of ' + NT + ')</div></div><div><div class="big" style="color:' + C.red + ';font-size:28px;margin:0">' + S.crowd[k - 1].toFixed(2) + '</div><div style="font-size:12px">RMSE, forest of ' + k + '</div></div></div>' +
        '<p style="margin-top:10px">' + (k === 1 ? 'A forest of one tree is a single tree.' : (S.crowd[k - 1] < S.one - 0.005 ? 'The forest RMSE is lower than the mean RMSE of a single tree.' : 'The forest RMSE is about equal to the mean RMSE of a single tree here.')) + ' RMSE is in ' + esc(unitWord()) + ' units. Lower is better.</p>' +
        '<p style="font-size:12px;color:' + C.mid + '">Each tree was fitted to a bootstrap sample of the training compounds and considered a random subset of the features at each split.</p>';
    }
    rg.addEventListener('input', function(){ K = +rg.value; draw(); });
    qs('[data-next]', ctl).addEventListener('click', function(){ cur++; draw(); });
    async function render(){ draw(); }
    async function play(tok){
      K = 1; rg.value = 1; draw(); api.cap('A single tree gives one prediction, which can be far from the lab value.'); await api.sleep(3000);
      await api.anim(4200, function(t){ K = Math.max(1, Math.round(1 + 14 * t)); rg.value = K; draw(); }); api.cap('Each added tree contributes a prediction. The mean stabilises and is usually closer to the lab value.'); await api.sleep(3400); api.cap(null);
    }
    return { render: render, play: play };
  } });

/* ====================================================================
   Lesson 02, step: gradient boosting
   ==================================================================== */
LESSONS.trees.add({ level: 'beginner', short: 'Fixing mistakes round by round', title: 'Gradient boosting',
  change: 'Drag the number of boosting rounds, or press Walk through.',
  watch: 'A forest fits its trees independently and averages them. Gradient boosting fits them in sequence. The first model is the mean of the training values. Each later round fits a small tree to the residuals of the current model, and adds that tree to the model scaled by the learning rate. The red line is the model. The thin red bars are the residuals of the training compounds. They shrink as rounds are added. Here the trees use logP only, so the model is a curve in two dimensions. The chart on the right shows the RMSE on test compounds after each round.',
  build: function(api){
    var R = 6, MAXR = 30;
    var cp = api.el('pn', 'left:28px;top:28px;width:560px;height:408px', '<h4>Model after the selected number of rounds</h4><canvas width="530" height="350"></canvas>');
    var rp = api.el('pn', 'left:608px;top:28px;width:324px;height:408px;overflow:hidden', '<div class="nb" style="font-size:13px;line-height:1.45;color:' + C.char + '"></div><canvas class="c2" width="294" height="200" style="margin-top:4px"></canvas>');
    var ctl = api.addControl('Boosting rounds <input type="range" min="0" max="30" step="1" value="6" style="width:240px"> <b class="rv">6</b>');
    var rg = qs('input', ctl), rv = qs('.rv', ctl);
    function prep(){
      var key = EP + 'boostbeg'; if (D.cache[key]) return D.cache[key]; var m = fitBoost(['logP'], D.TRAIN, MAXR, 0.3, 2, 1), miss = [], base = new Float64Array(D.N).fill(m.base); miss.push(rmse(D.TEST, base)); m.snaps.forEach(function(s){ miss.push(rmse(D.TEST, s.p)); });
      var dots = bt2Sample('bt2boost', 360, 6).slice().sort(function(a, b){ return D.COL.logP[a] - D.COL.logP[b]; });
      return (D.cache[key] = { m: m, miss: miss, dots: dots, base: base });
    }
    function draw(){
      var S = prep(); rv.textContent = R; var p = R === 0 ? S.base : S.m.snaps[R - 1].p, xs = S.dots.map(function(i){ return D.COL.logP[i]; }), xlo = sortedCol('logP')[Math.floor(D.N * 0.01)], xhi = sortedCol('logP')[Math.floor(D.N * 0.99)];
      var g = cv(qs('canvas', cp)), x = g.x, box = { l: 48, r: g.w - 14, t: 14, b: g.h - 40 }, sx = function(v){ return box.l + (clamp(v, xlo, xhi) - xlo) / (xhi - xlo) * (box.r - box.l); }, sy = function(v){ return box.b - (v - D.ylo) / (D.yhi - D.ylo) * (box.b - box.t); };
      axes(g, box, [xlo, xhi], [D.ylo, D.yhi], 'logP', 'lab value', 4);
      S.dots.forEach(function(i, q){ if (q % 4 !== 0) return; x.strokeStyle = 'rgba(225,36,42,.28)'; x.lineWidth = 1.4; x.beginPath(); x.moveTo(sx(D.COL.logP[i]), sy(D.Y[i])); x.lineTo(sx(D.COL.logP[i]), sy(p[i])); x.stroke(); });
      x.fillStyle = 'rgba(52,55,63,.5)'; S.dots.forEach(function(i){ x.beginPath(); x.arc(sx(D.COL.logP[i]), sy(D.Y[i]), 2.5, 0, 6.3); x.fill(); });
      x.strokeStyle = C.red; x.lineWidth = 3.2; x.lineJoin = 'round'; x.beginPath(); S.dots.forEach(function(i, q){ var px = sx(D.COL.logP[i]), py = sy(p[i]); q ? x.lineTo(px, py) : x.moveTo(px, py); }); x.stroke();
      var mi = S.miss[R], m0 = S.miss[0], lo = Math.floor((Math.min.apply(null, S.miss) - 0.01) * 50) / 50, hi = Math.ceil((m0 + 0.01) * 50) / 50, pts = S.miss.slice(0, R + 1).map(function(v, q){ return [q, v]; });
      qs('.nb', rp).innerHTML = '<div class="ol">' + (R === 0 ? 'Round 0: training mean only' : 'After ' + R + ' round' + (R > 1 ? 's' : '')) + '</div><div style="display:flex;gap:16px;align-items:flex-end"><div><div class="big" style="color:' + C.red + ';font-size:30px;margin:0">' + mi.toFixed(2) + '</div><div style="font-size:12px">RMSE on test compounds</div></div></div><p style="margin-top:8px">' + (R === 0 ? 'Every compound has the same prediction, so the residuals are large.' : (R < 6 ? 'Each round reduces the residuals, and the bars shorten.' : 'Later rounds reduce the residuals by less, so the gain in RMSE becomes small.')) + '</p>';
      lineChart(qs('.c2', rp), [{ pts: pts, color: C.red, w: 2.6, label: 'RMSE, test compounds' }], { xl: 'boosting rounds', yl: 'RMSE', x0: 0, x1: MAXR, y0: lo, y1: hi });
    }
    rg.addEventListener('input', function(){ R = +rg.value; draw(); });
    async function render(){ draw(); }
    async function play(tok){
      R = 0; rg.value = 0; draw(); api.cap('Round 0: the model is the training mean. The thin bars are the residuals.'); await api.sleep(3200);
      await api.anim(5200, function(t){ R = Math.round(MAXR * t); rg.value = R; draw(); }); api.cap('Each round adds a small tree fitted to the remaining residuals. The bars shorten and the test RMSE falls.'); await api.sleep(3600); api.cap(null);
    }
    return { render: render, play: play };
  } });

/* ====================================================================
   Lesson 08, step: how a cut is scored
   ==================================================================== */
LESSONS.treealgo.add({ level: 'beginner', short: 'Scoring a cut', title: 'Split criterion',
  change: 'Drag the cut position, or press Best cut. Compare the SSE before and after the split.',
  watch: 'A tree compares candidate splits with a criterion. For squared error, take a group of compounds and its mean lab value. The residual of a compound is its lab value minus the group mean. The sum of squared errors (SSE) is the sum of the squared residuals. Here each squared residual is drawn as the area of a square with the residual as its side. A split divides the group in two, and each part has its own mean and SSE. The gain of the split is the SSE before the split minus the total SSE of the two parts. The tree applies the split with the largest gain.',
  build: function(api){
    var k = 0, N = 10;
    var cp = api.el('pn', 'left:28px;top:28px;width:904px;height:408px', '<canvas width="880" height="384"></canvas>');
    var ctl = api.addControl('Cut after compound number <input type="range" min="0" max="9" step="1" value="0" style="width:200px"> <b class="kv">none</b> <button type="button" class="btn sm" data-best>Best cut</button>');
    var rg = qs('input', ctl), kv = qs('.kv', ctl);
    function pick(){
      var key = EP + 'sqcut'; if (D.cache[key]) return D.cache[key]; var r = rng(21), sh = D.TRAIN.slice(); for (var a = sh.length - 1; a > 0; a--) { var j = Math.floor(r() * (a + 1)), t = sh[a]; sh[a] = sh[j]; sh[j] = t; }
      var ids = sh.slice(0, N).sort(function(a, b){ return D.COL.logP[a] - D.COL.logP[b]; }); return (D.cache[key] = ids);
    }
    function sse(vals){ if (!vals.length) return 0; var m = mean(vals); return vals.reduce(function(s, v){ return s + (v - m) * (v - m); }, 0); }
    function cutSSE(ids, c){ var v = ids.map(function(i){ return D.Y[i]; }); return c === 0 ? sse(v) : sse(v.slice(0, c)) + sse(v.slice(c)); }
    function bestCut(ids){ var b = 1, bv = Infinity; for (var c = 1; c < ids.length; c++) { var v = cutSSE(ids, c); if (v < bv) { bv = v; b = c; } } return b; }
    function draw(){
      var ids = pick(), v = ids.map(function(i){ return D.Y[i]; }), g = cv(qs('canvas', cp)), x = g.x, W = g.w, left = 60, slot = (W - left - 20) / N, top = 66, pb = 236, sb = g.h - 12;
      var lo = Math.min.apply(null, v) - 0.15, hi = Math.max.apply(null, v) + 0.15, sy = function(val){ return pb - (val - lo) / (hi - lo) * (pb - top); };
      var groups = k === 0 ? [[0, N]] : [[0, k], [k, N]], total0 = cutSSE(ids, 0), tot = cutSSE(ids, k), maxMiss = Math.max.apply(null, v.map(function(y){ return Math.abs(y - mean(v)); })) || 1, scale = 52 / Math.max(maxMiss, 0.4);
      kv.textContent = k === 0 ? 'none' : k;
      x.font = '11px Aptos, "Segoe UI", sans-serif'; x.fillStyle = C.mid; x.textAlign = 'right'; x.fillText('lab value', 54, top - 20); x.textAlign = 'left'; x.fillText('Compounds ordered by logP, lowest on the left. The line is the mean of each group.', left, 14);
      x.strokeStyle = C.silver; x.lineWidth = 1; var tk = niceTicks(lo, hi, 4); x.textAlign = 'right'; tk.v.forEach(function(t){ x.beginPath(); x.moveTo(left - 4, sy(t)); x.lineTo(W - 20, sy(t)); x.stroke(); x.fillText(tickText(t, tk.step), left - 8, sy(t) + 4); });
      groups.forEach(function(gr, gi){ var m = mean(v.slice(gr[0], gr[1])), x0 = left + gr[0] * slot, x1 = left + gr[1] * slot; x.fillStyle = gi ? 'rgba(225,36,42,.07)' : 'rgba(161,177,196,.2)'; x.fillRect(x0 + 2, top - 6, x1 - x0 - 4, pb - top + 12); x.strokeStyle = gi ? C.red : C.char; x.lineWidth = 2.6; x.beginPath(); x.moveTo(x0 + 6, sy(m)); x.lineTo(x1 - 6, sy(m)); x.stroke(); x.fillStyle = gi ? C.red : C.char; x.font = '700 11px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'center'; x.fillText('mean ' + m.toFixed(2), (x0 + x1) / 2, top - 14);
        for (var q = gr[0]; q < gr[1]; q++) { var cxp = left + (q + 0.5) * slot, miss = v[q] - m; x.strokeStyle = gi ? 'rgba(225,36,42,.6)' : 'rgba(52,55,63,.5)'; x.lineWidth = 1.5; x.beginPath(); x.moveTo(cxp, sy(v[q])); x.lineTo(cxp, sy(m)); x.stroke(); x.fillStyle = C.char; x.beginPath(); x.arc(cxp, sy(v[q]), 5, 0, 6.3); x.fill();
          var side = Math.abs(miss) * scale; x.fillStyle = gi ? 'rgba(225,36,42,.28)' : 'rgba(111,129,150,.38)'; x.strokeStyle = gi ? C.red : '#6F8196'; x.lineWidth = 1.2; x.fillRect(cxp - side / 2, sb - side - 14, side, side); x.strokeRect(cxp - side / 2, sb - side - 14, side, side); x.fillStyle = C.char; x.font = '10.5px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'center'; x.fillText((miss * miss).toFixed(2), cxp, sb); } });
      x.fillStyle = C.mid; x.font = '600 11px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'left'; x.fillText('Square side: residual. Number: squared residual.', left, pb + 24);
      var gain = total0 - tot; x.textAlign = 'right'; x.font = '700 14px Aptos, "Segoe UI", sans-serif'; x.fillStyle = C.char; x.fillText('SSE, no split: ' + total0.toFixed(2), W - 20, pb + 24);
      x.fillStyle = C.red; x.fillText(k === 0 ? 'Select a cut position to see the gain.' : 'SSE with this split: ' + tot.toFixed(2) + '.  Gain: ' + gain.toFixed(2), W - 20, pb + 44);
      if (k > 0 && k === bestCut(ids)) { x.fillStyle = '#1E6B40'; x.fillText('This split has the largest gain for these ten compounds.', W - 20, pb + 64); }
    }
    rg.addEventListener('input', function(){ k = +rg.value; draw(); });
    qs('[data-best]', ctl).addEventListener('click', function(){ k = bestCut(pick()); rg.value = k; draw(); });
    async function render(){ draw(); }
    async function play(tok){
      k = 0; rg.value = 0; draw(); api.cap('With no split, all ten compounds share one mean. The squares show the squared residuals.'); await api.sleep(3600);
      k = 2; rg.value = 2; draw(); api.cap('A poorly placed split leaves the squares large. The gain is small.'); await api.sleep(3200);
      k = bestCut(pick()); rg.value = k; draw(); api.cap('The best split gives two groups with small residuals. The squares shrink and the gain is largest. The tree applies this split.'); await api.sleep(4200); api.cap(null);
    }
    return { render: render, play: play };
  } });

/* ====================================================================
   Lesson 08, step: bins
   ==================================================================== */
LESSONS.treealgo.add({ level: 'beginner', short: 'Fewer places to cut', title: 'Histogram binning',
  change: 'Select a number of bins. Compare the best split the bins allow with the best split overall.',
  watch: 'An exact search tests a threshold between every pair of neighbouring values of a feature. With n compounds that is up to n - 1 thresholds per feature at every node. Histogram-based libraries first assign each value to one of a fixed number of bins, then test only the bin edges. The search is faster, but the best exact threshold can fall inside a bin and be missed. This figure uses 120 compounds so that the thresholds are visible. Grey lines are the allowed thresholds. The red line is the best allowed split. The dashed black line is the best exact split.',
  build: function(api){
    var NB = 8, BINS = [3, 5, 8, 12, 20, 40, 120], NS = 120;
    var cp = api.el('pn', 'left:28px;top:28px;width:600px;height:408px', '<h4>Compounds by logP. Grey lines mark the allowed thresholds</h4><canvas width="570" height="350"></canvas>');
    var rp = api.el('pn', 'left:648px;top:28px;width:284px;height:408px;overflow:auto', '<div class="nb" style="font-size:13px;line-height:1.45;color:' + C.char + '"></div>');
    var ctl = api.addControl('Number of bins ' + inSeg(BINS.map(function(b){ return b === NS ? 'all' : b; }), 2, 'data-b'));
    function data(){
      var key = EP + 'binsd'; if (D.cache[key]) return D.cache[key]; var ids = bt2Sample('bt2bin', NS, 31).slice().sort(function(a, b){ return D.COL.logP[a] - D.COL.logP[b]; }), y = ids.map(function(i){ return D.Y[i]; }), xs = ids.map(function(i){ return D.COL.logP[i]; });
      var cs = [0], cs2 = [0]; y.forEach(function(v, q){ cs.push(cs[q] + v); cs2.push(cs2[q] + v * v); });
      function gainAt(m){ var n = NS, T = cs[n], T2 = cs2[n], sl = cs2[m] - cs[m] * cs[m] / m, sr = (T2 - cs2[m]) - (T - cs[m]) * (T - cs[m]) / (n - m); return (T2 - T * T / n) - (sl + sr); }
      var bestAll = 1, bg = -1; for (var m = 1; m < NS; m++) { var gg = gainAt(m); if (gg > bg) { bg = gg; bestAll = m; } }
      return (D.cache[key] = { ids: ids, y: y, xs: xs, gainAt: gainAt, bestAll: bestAll, bg: bg });
    }
    function draw(){
      var S = data(), nb = NB, edges = []; for (var q = 1; q < nb; q++) { var m = Math.round(q * NS / nb); if (m > 0 && m < NS && edges.indexOf(m) < 0) edges.push(m); } if (nb >= NS) { edges = []; for (var q2 = 1; q2 < NS; q2++) edges.push(q2); }
      var bestB = edges[0], bgv = -1; edges.forEach(function(m){ var gg = S.gainAt(m); if (gg > bgv) { bgv = gg; bestB = m; } });
      var g = cv(qs('canvas', cp)), x = g.x, box = { l: 48, r: g.w - 14, t: 14, b: g.h - 40 }, xlo = Math.min.apply(null, S.xs), xhi = Math.max.apply(null, S.xs), sx = function(v){ return box.l + (v - xlo) / (xhi - xlo) * (box.r - box.l); }, sy = function(v){ return box.b - (v - D.ylo) / (D.yhi - D.ylo) * (box.b - box.t); };
      axes(g, box, [xlo, xhi], [D.ylo, D.yhi], 'logP', 'lab value', 4);
      var mid = function(m){ return (S.xs[m - 1] + S.xs[m]) / 2; };
      x.strokeStyle = 'rgba(111,129,150,.55)'; x.lineWidth = 1; edges.forEach(function(m){ var px = sx(mid(m)); x.beginPath(); x.moveTo(px, box.t); x.lineTo(px, box.b); x.stroke(); });
      x.fillStyle = 'rgba(52,55,63,.7)'; S.xs.forEach(function(v, q){ x.beginPath(); x.arc(sx(v), sy(S.y[q]), 3, 0, 6.3); x.fill(); });
      x.strokeStyle = C.char; x.setLineDash([6, 4]); x.lineWidth = 2; var pa = sx(mid(S.bestAll)); x.beginPath(); x.moveTo(pa, box.t); x.lineTo(pa, box.b); x.stroke(); x.setLineDash([]);
      x.strokeStyle = C.red; x.lineWidth = 3; var pb2 = sx(mid(bestB)); x.beginPath(); x.moveTo(pb2, box.t); x.lineTo(pb2, box.b); x.stroke();
      var keep = bgv / S.bg * 100, nFull = D.TRAIN.length;
      qs('.nb', rp).innerHTML = '<div class="ol">' + (nb >= NS ? 'No binning: every value' : nb + ' bins') + '</div><div style="display:flex;gap:16px"><div><div class="big" style="color:' + C.red + ';font-size:30px;margin:0">' + edges.length + '</div><div style="font-size:12px">thresholds tested</div></div><div><div class="big" style="font-size:30px;margin:0;color:' + C.mid + '">' + Math.round(keep) + '%</div><div style="font-size:12px">of the maximum gain</div></div></div>' +
        '<p style="margin-top:10px">' + (keep > 99.5 ? 'The best allowed split is the best exact split.' : (keep > 90 ? 'The best allowed split is close to the best exact split.' : 'The best exact threshold fell inside a bin and was missed.')) + '</p>' +
        '<p>With all <b>' + nFull.toLocaleString() + '</b> training compounds, an exact search tests up to <b>' + (nFull - 1).toLocaleString() + '</b> thresholds per feature. With 64 bins it tests at most <b>63</b>. Tree libraries use bins like these because the gain lost is small.</p><p style="font-size:12px;color:' + C.mid + '">Gain is the reduction in SSE, as in the previous step.</p>';
    }
    qsa('[data-b] button', ctl).forEach(function(b){ b.addEventListener('click', function(){ NB = BINS[+b.dataset.v]; qsa('[data-b] button', ctl).forEach(function(z){ z.classList.toggle('on', z === b); }); draw(); }); });
    async function render(){ draw(); }
    async function play(tok){
      for (var q = BINS.length - 1; q >= 0; q--) { api.checkTok(tok); NB = BINS[q]; qsa('[data-b] button', ctl).forEach(function(z){ z.classList.toggle('on', +z.dataset.v === q); }); draw(); api.cap(q === BINS.length - 1 ? 'Every value is a candidate: many thresholds, slow search.' : (q === 0 ? 'Three bins: few thresholds, but the best split can be missed.' : 'Fewer bins give fewer thresholds. The best allowed split is often nearly as good.')); await api.sleep(2200); }
      NB = 8; qsa('[data-b] button', ctl).forEach(function(z){ z.classList.toggle('on', +z.dataset.v === 2); }); draw(); api.cap(null);
    }
    return { render: render, play: play };
  } });

/* ====================================================================
   Lesson 08, step: random draws with replacement
   ==================================================================== */
LESSONS.treealgo.add({ level: 'beginner', short: 'Random draws', title: 'Bootstrap sampling',
  change: 'Press New sample for another bootstrap sample, or press Repeat 200 samples.',
  watch: 'A random forest fits each tree to a bootstrap sample. A bootstrap sample draws n compounds from the n training compounds at random, with replacement. A compound can therefore appear more than once, and some do not appear at all. The compounds absent from a sample are out-of-bag for that tree. The tree was not fitted to them, so they provide a test-like score for it. The circles show 24 compounds, and the circle area increases with the number of times a compound was drawn.',
  build: function(api){
    var N = 24, counts = new Array(N).fill(0), draws = [], avg = null, rnd = Math.random;
    var cp = api.el('pn', 'left:28px;top:28px;width:560px;height:408px', '<h4>Bootstrap sample: 24 draws with replacement from 24 compounds</h4><canvas width="530" height="350"></canvas>');
    var rp = api.el('pn', 'left:608px;top:28px;width:324px;height:408px;overflow:auto', '<div class="nb" style="font-size:13px;line-height:1.45;color:' + C.char + '"></div>');
    var ctl = api.addControl('<button type="button" class="btn sm" data-draw>New sample</button> <button type="button" class="btn sm ghost" data-rep>Repeat 200 samples</button>');
    function newDraw(){ counts = new Array(N).fill(0); draws = []; for (var q = 0; q < N; q++) { var c = Math.floor(rnd() * N); counts[c]++; draws.push(c); } }
    function draw(upTo){
      var shown = upTo === undefined ? draws.length : upTo, cs = new Array(N).fill(0); for (var q = 0; q < shown; q++) cs[draws[q]]++;
      var g = cv(qs('canvas', cp)), x = g.x, cols = 8, rows = 3, cw = g.w / cols, ch = (g.h - 20) / rows;
      for (var i = 0; i < N; i++) { var cx = (i % cols + 0.5) * cw, cy = Math.floor(i / cols) * ch + ch / 2 + 6, n = cs[i], r = n === 0 ? 17 : 17 + n * 5;
        x.beginPath(); x.arc(cx, cy, r, 0, 6.3); if (n === 0) { x.fillStyle = '#fff'; x.fill(); x.setLineDash([4, 3]); x.strokeStyle = C.bg; x.lineWidth = 1.8; x.stroke(); x.setLineDash([]); } else { x.fillStyle = n === 1 ? 'rgba(52,55,63,.82)' : (n === 2 ? 'rgba(225,36,42,.8)' : C.dred); x.fill(); }
        x.fillStyle = n === 0 ? C.mid : '#fff'; x.font = '600 12px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'center'; x.fillText('#' + (i + 1), cx, cy + 4); if (n > 1) { x.fillStyle = C.char; x.font = '700 11px Aptos, "Segoe UI", sans-serif'; x.fillText(n + ' times', cx, cy + r + 13); } }
      var z = cs.filter(function(c){ return c === 0; }).length, one = cs.filter(function(c){ return c === 1; }).length, two = cs.filter(function(c){ return c === 2; }).length, more = cs.filter(function(c){ return c >= 3; }).length;
      qs('.nb', rp).innerHTML = '<div class="ol">This sample (' + shown + ' of ' + N + ' draws)</div>' + [['Not drawn (out-of-bag)', z, C.bg], ['Drawn once', one, C.char], ['Drawn twice', two, C.red], ['Drawn three or more times', more, C.dred]].map(function(r){ return '<div style="display:grid;grid-template-columns:150px 1fr 26px;gap:8px;align-items:center;margin:6px 0;font-size:12.5px"><span>' + r[0] + '</span><div style="height:12px;background:' + C.light + ';border-radius:3px"><i style="display:block;height:12px;width:' + (r[1] / N * 100) + '%;border-radius:3px;background:' + r[2] + '"></i></div><b style="text-align:right">' + r[1] + '</b></div>'; }).join('') +
        '<p style="margin-top:10px">' + (shown < N ? 'Sampling in progress.' : 'Out-of-bag in this sample: <b>' + z + '</b> of ' + N + ' compounds, <b>' + Math.round(100 * z / N) + '%</b>.') + '</p>' +
        '<div class="ol" style="margin-top:10px">Repeated samples</div><p style="margin:2px 0">' + (avg === null ? 'Press Repeat 200 samples to estimate the mean out-of-bag share.' : 'Over 200 samples, the mean out-of-bag share was <b style="color:' + C.red + '">' + avg.toFixed(0) + '%</b>. For large n it approaches 1/e, about 37%.') + '</p>' +
        '<p style="font-size:12px;color:' + C.mid + '">Each tree in a forest has its own bootstrap sample. Its out-of-bag compounds give a test-like score without a separate test set.</p>';
    }
    qs('[data-draw]', ctl).addEventListener('click', function(){ newDraw(); draw(); });
    qs('[data-rep]', ctl).addEventListener('click', function(){ var tot = 0; for (var r = 0; r < 200; r++) { var c = new Array(N).fill(0); for (var q = 0; q < N; q++) c[Math.floor(rnd() * N)]++; tot += c.filter(function(v){ return v === 0; }).length / N; } avg = tot / 200 * 100; draw(); });
    async function render(){ if (!draws.length) newDraw(); draw(); }
    async function play(tok){
      newDraw(); draw(0); api.cap('24 compounds are sampled with replacement, one draw at a time.'); await api.sleep(2600);
      for (var q = 1; q <= N; q++) { api.checkTok(tok); draw(q); await api.sleep(ANIM ? 170 : 0); }
      api.cap('Some compounds were drawn more than once. The dashed circles were not drawn. They are out-of-bag for this tree.'); await api.sleep(3800); api.cap(null);
    }
    return { render: render, play: play };
  } });

/* ---------- guide entries ---------- */
GUIDE.steps['trees/Choosing the best question'] = { notice: 'The RMSE is lowest at the best threshold. The tree applies that split and repeats the search within each group.', q: 'Why does the tree evaluate every feature and every threshold?', a: 'No threshold is known in advance to be best. The tree computes the RMSE for all of them and keeps the lowest.' };
GUIDE.steps['trees/A crowd of trees'] = { notice: 'The forest mean is usually nearer the lab value than most single tree predictions.', q: 'Why do the trees in a forest give different predictions?', a: 'Each tree is fitted to a different bootstrap sample and considers a random subset of features at each split.' };
GUIDE.steps['trees/Fixing mistakes round by round'] = { notice: 'The residual bars shorten quickly at first and slowly later.', q: 'What is each new tree fitted to?', a: 'The residuals of the current model.' };
GUIDE.steps['treealgo/Scoring a cut'] = { notice: 'The best split gives the smallest total SSE.', q: 'Why square the residuals?', a: 'Squaring makes large residuals count much more than small ones, so a split that leaves a few large residuals scores badly. It also makes the criterion differentiable.' };
GUIDE.steps['treealgo/Fewer places to cut'] = { notice: 'Even with few bins, the best allowed split keeps most of the gain.', q: 'What is the cost of binning?', a: 'The best exact threshold may fall inside a bin and be missed, so the chosen split can be slightly worse.' };
GUIDE.steps['treealgo/Random draws'] = { notice: 'About one compound in three is not drawn.', q: 'Why are out-of-bag compounds useful?', a: 'The tree was not fitted to them, so they give a test-like score for that tree.' };
