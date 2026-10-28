"use strict";
/* Extra beginner steps for lessons 06 and 07.
   Written for chemists with no machine learning experience. Simulated data are labelled as such on the page. */

/* ---------- shared helpers ---------- */
function m2Gauss(r){ var u = Math.max(r(), 1e-9), v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(6.283185307 * v); }
function m2Sd(a){ var m = mean(a); return Math.sqrt(mean(a.map(function(v){ return (v - m) * (v - m); }))) || 1; }
function m2Pick(arr, n, seed){ var r = rng(seed), sh = arr.slice(); for (var a = sh.length - 1; a > 0; a--) { var j = Math.floor(r() * (a + 1)), t = sh[a]; sh[a] = sh[j]; sh[j] = t; } return sh.slice(0, n); }
function m2Score(ys, ps){ var my = mean(ys), a = 0, b = 0; for (var i = 0; i < ys.length; i++) { a += (ys[i] - ps[i]) * (ys[i] - ps[i]); b += (ys[i] - my) * (ys[i] - my); } return b ? 1 - a / b : 0; }
/* a plain scatter plot with axes, for pairs of numbers that are not tied to the dataset */
function m2Scatter(el, xs, ys, xr, yr, xl, yl, o){
  o = o || {}; var g = cv(el), box = { l: 52, r: g.w - 14, t: 14, b: g.h - 40 }, x = g.x; axes(g, box, xr, yr, xl, yl, 4);
  var sx = function(v){ return box.l + (v - xr[0]) / (xr[1] - xr[0]) * (box.r - box.l); }, sy = function(v){ return box.b - (v - yr[0]) / (yr[1] - yr[0]) * (box.b - box.t); };
  if (o.diag) { x.strokeStyle = C.char; x.setLineDash([5, 5]); x.beginPath(); x.moveTo(sx(xr[0]), sy(yr[0])); x.lineTo(sx(xr[1]), sy(yr[1])); x.stroke(); x.setLineDash([]); }
  if (o.under) o.under(x, sx, sy, box);
  x.fillStyle = o.fill || 'rgba(225,36,42,.6)'; for (var i = 0; i < xs.length; i++) { x.beginPath(); x.arc(sx(clamp(xs[i], xr[0], xr[1])), sy(clamp(ys[i], yr[0], yr[1])), o.r || 3.4, 0, 6.3); x.fill(); }
  if (o.over) o.over(x, sx, sy, box); g.sx = sx; g.sy = sy; g.box = box; return g;
}
function m2Big(label, num, sub, col){ return '<div style="margin:0 0 9px"><div class="ol" style="margin-bottom:0">' + label + '</div><div style="display:flex;align-items:baseline;gap:10px"><span class="big" style="color:' + (col || C.red) + ';font-size:30px;margin:0">' + num + '</span></div><div style="font-size:12px;color:' + C.char + '">' + sub + '</div></div>'; }

/* ---------- 06 Compare: evaluation metrics ---------- */
LESSONS.compare.add({ level: 'beginner', short: 'What the scores mean', title: 'Evaluation metrics',
  change: 'Drag the prediction noise and the shrinkage toward the mean. Read the three metrics.',
  watch: 'Three metrics are reported for a regression model, each on the test set. R2 is the fraction of the variance of the measured values that the predictions explain: 1 is perfect, 0 equals predicting the mean of the measured values for every compound, and a negative value is worse than that. RMSE, the root mean squared error, is the typical prediction error in the units of the measured value. Spearman correlation is the rank correlation between predictions and measurements, so it ignores the exact values and reflects only the ordering. The four small panels use simulated points to show how R2 relates to scatter. The large panel uses measured values from the data set with simulated predictions, so the prediction quality can be set directly.',
  build: function(api){
    var tp = api.el('pn', 'left:28px;top:20px;width:904px;height:176px', '<h4>Simulated points, one per compound. x is the measured value, y is the prediction, the dashed line is y = x.</h4><canvas width="880" height="140"></canvas>');
    var cp = api.el('pn', 'left:28px;top:206px;width:520px;height:314px', '<h4>Measured values from the data set, simulated predictions</h4><canvas width="490" height="262"></canvas>');
    var np = api.el('pn', 'left:568px;top:206px;width:364px;height:314px;overflow:auto', '<div class="nb" style="font-size:13px;line-height:1.45;color:' + C.char + '"></div>');
    var ctl = api.addControl('Prediction noise <input type="range" data-s min="0.05" max="1.2" step="0.01" value="0.45" style="width:150px"> Shrinkage toward the mean <input type="range" data-k min="0.2" max="1" step="0.01" value="1" style="width:150px">');
    var rs = qs('[data-s]', ctl), rk = qs('[data-k]', ctl), mini = null, S = null;
    function drawMini(){
      var g = cv(qs('canvas', tp)), x = g.x, rr = rng(21), n = 60, M = [], E = [], k, i;
      if (!mini) { for (i = 0; i < n; i++) { M.push(clamp(m2Gauss(rr), -2.2, 2.2)); E.push(clamp(m2Gauss(rr), -2.2, 2.2)); } mini = { M: M, E: E }; }
      M = mini.M; E = mini.E; var targets = [0, 0.3, 0.6, 0.9], notes = ['equal to predicting the mean', 'wide scatter', 'moderate scatter', 'narrow scatter'];
      for (k = 0; k < 4; k++) {
        var ox = 6 + k * 219, w = 200, top = 4, bot = 92, s = 0, lo_ = 0, hi_ = 1.6; if (k) { for (var it = 0; it < 24; it++) { s = (lo_ + hi_) / 2; if (m2Score(M, M.map(function(m, j){ return m + s * E[j]; })) > targets[k]) lo_ = s; else hi_ = s; } } var P = M.map(function(m, j){ return k === 0 ? 0.04 * E[j] : m + s * E[j]; }), sc = m2Score(M, P), sx = function(v){ return ox + 8 + (clamp(v, -2.8, 2.8) + 2.8) / 5.6 * (w - 16); }, sy = function(v){ return bot - (clamp(v, -2.8, 2.8) + 2.8) / 5.6 * (bot - top); };
        x.strokeStyle = C.silver; x.lineWidth = 1; x.strokeRect(ox, top, w, bot - top); x.strokeStyle = C.char; x.setLineDash([4, 4]); x.beginPath(); x.moveTo(sx(-2.8), sy(-2.8)); x.lineTo(sx(2.8), sy(2.8)); x.stroke(); x.setLineDash([]);
        x.fillStyle = 'rgba(225,36,42,.65)'; M.forEach(function(m, j){ x.beginPath(); x.arc(sx(m), sy(P[j]), 2.8, 0, 6.3); x.fill(); });
        x.fillStyle = C.char; x.textAlign = 'center'; x.font = '700 13px Aptos, "Segoe UI", sans-serif'; x.fillText('R2 ' + (k === 0 ? '0' : sc.toFixed(1)), ox + w / 2, bot + 16); x.font = '11px Aptos, "Segoe UI", sans-serif'; x.fillStyle = C.mid; x.fillText(notes[k], ox + w / 2, bot + 31);
      }
    }
    function prep(){ if (S) return S; var idx = m2Pick(D.TEST, 150, 3), r = rng(31), E = idx.map(function(){ return m2Gauss(r); }), ys = idx.map(function(i){ return D.Y[i]; }); S = { idx: idx, E: E, mu: mean(ys), sd: m2Sd(ys) }; return S; }
    function draw(){
      drawMini(); var P = prep(), s = +rs.value, k = +rk.value, pred = new Float64Array(D.N);
      P.idx.forEach(function(i, j){ var raw = D.Y[i] + s * P.sd * P.E[j]; pred[i] = P.mu + k * (raw - P.mu); });
      scatterPV(qs('canvas', cp), pred, P.idx, '', { max: 150 });
      var sc = r2(P.idx, pred), rm = rmse(P.idx, pred), sp = spearman(P.idx, pred);
      qs('.nb', np).innerHTML = m2Big('R2', sc.toFixed(2), 'Variance explained. 1 is perfect, 0 equals predicting the mean.') + m2Big('RMSE', rm.toFixed(2), 'In ' + esc(unitWord()) + ' units. Lower is better.', C.char) + m2Big('Spearman', sp.toFixed(2), 'Rank correlation. 1 is a perfect ordering.', '#6F8196') +
        (k < 0.9 ? '<p style="font-size:12.5px;color:' + C.dred + '">Shrinking the predictions toward the mean raises RMSE and lowers R2. The ordering of the compounds is unchanged, so Spearman barely moves.</p>' : '<p style="font-size:12.5px">Reduce the shrinkage slider. R2 and RMSE depend on the values of the predictions. Spearman depends only on their order.</p>');
    }
    rs.addEventListener('input', draw); rk.addEventListener('input', draw);
    async function play(tok){
      rk.value = 1; rs.value = 1.1; draw(); api.cap('At high prediction noise the points form a cloud and R2 is close to zero.'); await api.sleep(3000);
      await api.anim(2600, function(t){ rs.value = 1.1 - 1.05 * t; draw(); }); api.cap('As the noise falls, the points approach the dashed line and R2 rises toward 1.'); await api.sleep(3000);
      await api.anim(2200, function(t){ rk.value = 1 - 0.7 * t; draw(); }); api.cap('Shrinking the predictions toward the mean lowers R2 but leaves the ordering, and so Spearman, unchanged.'); await api.sleep(3200); api.cap(null);
    }
    return { render: draw, play: play };
  } });

/* ---------- 06 Compare: assay repeatability ceiling ---------- */
LESSONS.compare.add({ level: 'beginner', short: 'Why no score is perfect', title: 'Assay repeatability ceiling',
  change: 'Drag the noise slider. Press Measure again to draw new repeat measurements.',
  watch: 'Repeat measurements of the same compound differ by assay noise. A model cannot predict the random component of a measurement, so even a model that returned the true value for every compound would score below 1 against a noisy measurement. The maximum R2 attainable is the ceiling set by assay repeatability. Each point here is one compound. The x axis is a first measurement and the y axis is a second measurement of the same compound. The true values are taken from the data set. Both measurements are simulated by adding Gaussian noise with the standard deviation set by the slider, so the repeatability can be varied. The ceiling is the R2 of the noise-free values scored against one noisy measurement.',
  build: function(api){
    var seed = 1, S = null;
    var cp = api.el('pn', 'left:28px;top:20px;width:520px;height:500px', '<h4>Two simulated measurements of the same compounds</h4><canvas width="490" height="440"></canvas>');
    var np = api.el('pn', 'left:568px;top:20px;width:364px;height:500px;overflow:auto', '<div class="nb" style="font-size:13px;line-height:1.45;color:' + C.char + '"></div>');
    var ctl = api.addControl('Measurement noise (SD) <input type="range" min="0.05" max="0.8" step="0.01" value="0.3" style="width:200px"> <b class="nv"></b> <button type="button" class="btn sm ghost" data-again>Measure again</button>');
    var rg = qs('input', ctl), nv = qs('.nv', ctl);
    function prep(){ var idx = m2Pick(D.TRAIN, 130, 4); S = { idx: idx, T: idx.map(function(i){ return D.Y[i]; }) }; var r = rng(100 + seed); S.E1 = idx.map(function(){ return m2Gauss(r); }); S.E2 = idx.map(function(){ return m2Gauss(r); }); }
    function draw(){
      if (!S) prep(); var sd = +rg.value, M1 = S.T.map(function(t, j){ return t + sd * S.E1[j]; }), M2 = S.T.map(function(t, j){ return t + sd * S.E2[j]; });
      nv.textContent = sd < 0.2 ? 'low' : (sd < 0.5 ? 'moderate' : 'high');
      m2Scatter(qs('canvas', cp), M1, M2, [D.ylo, D.yhi], [D.ylo, D.yhi], 'first measurement', 'second measurement', { diag: true });
      var rep = m2Score(M1, M2), ceil = m2Score(M1, S.T), diff = Math.sqrt(mean(M1.map(function(v, j){ return (v - M2[j]) * (v - M2[j]); })));
      qs('.nb', np).innerHTML = m2Big('R2 between repeats', rep.toFixed(2), 'The second measurement scored as a prediction of the first.', C.char) + m2Big('Ceiling for any model', ceil.toFixed(2), 'R2 of the noise-free values scored against one noisy measurement.') + m2Big('RMSE between repeats', diff.toFixed(2), 'In ' + esc(unitWord()) + ' units.', '#6F8196') +
        '<p style="font-size:12.5px;margin-top:6px">A test R2 close to the ceiling cannot be improved by a better model. Inter-run and inter-day variability add to this noise and depend on the assay. Obtain the repeatability from the assay owner.</p>';
    }
    rg.addEventListener('input', draw); qs('[data-again]', ctl).addEventListener('click', function(){ seed++; prep(); draw(); });
    async function play(tok){
      rg.value = 0.05; draw(); api.cap('At low noise the second measurement almost always matches the first.'); await api.sleep(3000);
      await api.anim(3200, function(t){ rg.value = 0.05 + 0.6 * t; draw(); }); api.cap('As the noise grows, repeat measurements diverge and the ceiling for any model falls with them.'); await api.sleep(3200); api.cap(null);
    }
    return { render: draw, play: play };
  } });

/* ---------- 07 Role of AI: applicability domain ---------- */
LESSONS.ai.add({ level: 'beginner', short: 'What a model has not seen', title: 'Applicability domain',
  change: 'Drag the slider to change the share of training compounds retained, lowest logP first.',
  watch: 'A trained model is reliable only within the region of feature space covered by its training compounds, its applicability domain. Here a regression tree is trained on the compounds with the lowest logP, up to a limit set by the slider, using logP, TPSA and MW as features. It is then evaluated on test compounds. Each vertical line joins the measured value (dark dot) to the prediction (end of the line). The shaded band is the logP range of the training compounds. Grey lines are inside the band and red lines outside it. RMSE is reported separately for the two groups.',
  build: function(api){
    var cp = api.el('pn', 'left:28px;top:20px;width:610px;height:500px', '<h4>Test compounds: measured value (dark dot) and prediction (end of the line)</h4><canvas width="580" height="440"></canvas>');
    var np = api.el('pn', 'left:658px;top:20px;width:274px;height:500px;overflow:auto', '<div class="nb" style="font-size:13px;line-height:1.45;color:' + C.char + '"></div>');
    var ctl = api.addControl('Share of training compounds retained, lowest logP first <input type="range" min="30" max="95" step="5" value="40" style="width:200px"> <b class="pv"></b>');
    var rg = qs('input', ctl), pv = qs('.pv', ctl), FE = ['logP', 'TPSA', 'MW'];
    function draw(){
      var share = +rg.value / 100, tr = D.TRAIN, lp = D.COL.logP, sorted = tr.map(function(i){ return lp[i]; }).sort(function(a, b){ return a - b; }), cut = sorted[Math.min(sorted.length - 1, Math.floor(share * sorted.length))], lo0 = sorted[0];
      pv.textContent = Math.round(share * 100) + '%'; var key = EP + 'unseen' + rg.value, T = D.cache[key];
      if (!T) { var sub = tr.filter(function(i){ return lp[i] <= cut; }), tree = fitTree(FE.filter(function(f){ return D.COL[f]; }), sub, 5, 15, 1); T = D.cache[key] = { pred: treeAll(tree, D.N), n: sub.length }; }
      var idx = m2Pick(D.TEST, 260, 6), xs = idx.map(function(i){ return lp[i]; }), xlo = Math.min(-1, Math.floor(Math.min.apply(null, xs))), xhi = Math.min(9, Math.ceil(Math.max.apply(null, xs)));
      var inS = 0, inN = 0, outS = 0, outN = 0, outB = 0; idx.forEach(function(i){ var d = D.Y[i] - T.pred[i], e = d * d; if (lp[i] <= cut) { inS += e; inN++; } else { outS += e; outN++; outB += d; } });
      m2Scatter(qs('canvas', cp), [], [], [xlo, xhi], [D.ylo, D.yhi], 'logP', 'measured value', { under: function(x, sx, sy, box){
        x.fillStyle = 'rgba(161,177,196,.22)'; x.fillRect(sx(Math.max(xlo, lo0)), box.t, sx(Math.min(xhi, cut)) - sx(Math.max(xlo, lo0)), box.b - box.t); x.fillStyle = C.char; x.font = '700 11px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'center'; x.fillText('training range', (sx(Math.max(xlo, lo0)) + sx(Math.min(xhi, cut))) / 2, box.t + 14); if (cut < xhi) { x.fillStyle = C.red; x.fillText('outside training range', (sx(cut) + sx(xhi)) / 2, box.t + 14); }
        idx.forEach(function(i){ var inside = lp[i] <= cut; x.strokeStyle = inside ? 'rgba(111,129,150,.55)' : 'rgba(225,36,42,.6)'; x.lineWidth = 1.4; x.beginPath(); x.moveTo(sx(lp[i]), sy(clamp(D.Y[i], D.ylo, D.yhi))); x.lineTo(sx(lp[i]), sy(clamp(T.pred[i], D.ylo, D.yhi))); x.stroke(); if (!inside) { x.fillStyle = C.red; x.fillRect(sx(lp[i]) - 2, sy(clamp(T.pred[i], D.ylo, D.yhi)) - 2, 4, 4); } });
        x.fillStyle = 'rgba(52,55,63,.8)'; idx.forEach(function(i){ x.beginPath(); x.arc(sx(lp[i]), sy(clamp(D.Y[i], D.ylo, D.yhi)), 2.6, 0, 6.3); x.fill(); }); } });
      var ri = inN ? Math.sqrt(inS / inN) : NaN, ro = outN ? Math.sqrt(outS / outN) : NaN;
      qs('.nb', np).innerHTML = '<div class="ol">Training set</div><div style="margin-bottom:10px"><b>' + T.n.toLocaleString() + '</b> compounds with logP up to <b>' + cut.toFixed(1) + '</b>.</div>' + m2Big('RMSE inside the training range', isNaN(ri) ? 'n/a' : ri.toFixed(2), inN + ' test compounds. Grey lines.', '#6F8196') + m2Big('RMSE outside the training range', isNaN(ro) ? 'n/a' : ro.toFixed(2), outN + ' test compounds. Red lines.') +
        (outN && Math.abs(outB / outN) > 0.1 ? m2Big('Mean error outside the range', (outB / outN > 0 ? 'predictions too low by ' : 'predictions too high by ') + Math.abs(outB / outN).toFixed(2), 'The errors share a sign, so they are systematic.', C.char) : '') + '<p style="font-size:12.5px">' + (isNaN(ro) ? 'Almost every test compound is inside the training range.' : 'A tree returns the mean of the nearest leaf, so its predictions stay constant beyond the training range while the measured values keep changing. The red squares mark those predictions. Move the slider left to enlarge the region outside the training range and compare the two RMSE values.') + '</p>';
    }
    rg.addEventListener('input', draw);
    async function play(tok){
      rg.value = 90; draw(); api.cap('With 90% of the training compounds retained, few test compounds fall outside the training range.'); await api.sleep(3000);
      await api.anim(3600, function(t){ rg.value = Math.round((90 - 60 * t) / 5) * 5; draw(); }); api.cap('As the training range narrows, more test compounds fall outside it and the RMSE there is larger.'); await api.sleep(3400); api.cap(null);
    }
    return { render: draw, play: play };
  } });

/* ---------- 07 Role of AI: assessing a prediction ---------- */
LESSONS.ai.add({ level: 'beginner', short: 'Checking an answer', title: 'Assessing a prediction',
  change: 'Select Show assessment for each case.',
  watch: 'A model returns a value for any input, whether or not the value is supported. Three criteria determine how far a prediction can be relied on. Can the value be checked against a measurement? Does the compound fall within the training range of the model? Is there a reported test-set performance for compounds like it? The three cases below are constructed for this exercise. For each, the assessment lists the three criteria and the conclusion.',
  build: function(api){
    var ch = [false, false, false];
    var cp = api.el('pn', 'left:28px;top:20px;width:904px;height:500px;overflow:auto', '<div class="cl"></div>');
    var ctl = api.addControl('<button type="button" class="btn sm ghost" data-reset>Hide assessments</button>');
    function stats(){
      var key = EP + 'chk'; if (D.cache[key]) return D.cache[key]; var fs = ['logP', 'TPSA', 'MW'].filter(function(f){ return D.COL[f]; }), t = fitTree(fs, D.TRAIN, 4, 20, 1), mw = D.TRAIN.map(function(i){ return D.COL.MW[i]; }), mwMax = Math.max.apply(null, mw), med = {}; fs.forEach(function(f){ var a = D.TRAIN.map(function(i){ return D.COL[f][i]; }).sort(function(p, q){ return p - q; }); med[f] = a[Math.floor(a.length / 2)]; });
      var tp = treeAll(t, D.N), big = Object.assign({}, med); big.MW = Math.round(mwMax * 1.3);
      return (D.cache[key] = { tree: t, mwMax: mwMax, med: med, big: big, pMed: predictVals(t, D.B, med), pBig: predictVals(t, D.B, big), miss: rmse(D.TEST, tp) });
    }
    function draw(){
      var s = stats(), claims = [
        { text: 'A generative model proposes a new structure and states a clearance of 0.40 for it.', note: 'Constructed for this exercise. The statement is text produced by the model, not the output of a trained regression model.', crit: ['No. No measurement exists for the structure.', 'Not assessed. No applicability domain is defined.', 'None reported.'],
          why: 'Not supported. A generated value has no measurement behind it and no test-set performance. A structure proposed by a generative model needs a prediction from a trained model, and then an assay.' },
        { text: 'A regression tree trained on compounds up to a molecular weight of <b>' + Math.round(s.mwMax) + '</b> returns <b>' + s.pBig.toFixed(2) + '</b> for a compound with molecular weight <b>' + s.big.MW + '</b>.', note: 'The compound is constructed for this exercise.', crit: ['Yes, by an assay.', 'No. The molecular weight is above the maximum of the training set.', 'Reported for the whole test set, but not valid for this compound.'],
          why: 'Not supported. The compound lies outside the applicability domain. A tree returns the value of the nearest leaf, which is the value it returns for the heaviest training compounds, whatever the true value is.' },
        { text: 'The same model returns <b>' + s.pMed.toFixed(2) + '</b> for a compound with logP <b>' + s.med.logP.toFixed(1) + '</b>, TPSA <b>' + Math.round(s.med.TPSA) + '</b> and molecular weight <b>' + Math.round(s.med.MW) + '</b>.', note: 'The compound is constructed for this exercise. The feature values are the medians of the training set.', crit: ['Yes, by an assay.', 'Yes. All three features are at the training median.', 'Yes. RMSE ' + s.miss.toFixed(2) + ' ' + esc(unitWord()) + ' units on the scaffold-split test set.'],
          why: 'Supported, within the stated error. The compound is inside the applicability domain and a test-set RMSE is reported. The prediction remains an estimate, so confirm it by measurement before a decision that depends on it.' }];
      var labels = ['Checkable against a measurement', 'Within the training range', 'Test-set performance reported'];
      qs('.cl', cp).innerHTML = claims.map(function(c, i){ var a = ch[i]; return '<div style="border:1px solid ' + C.silver + ';border-radius:8px;padding:12px 14px;margin:0 0 12px;background:#fff"><div class="ol">Case ' + (i + 1) + '</div><div style="font-size:14px;line-height:1.5;color:' + C.char + '">' + c.text + '</div><div style="font-size:11.5px;color:' + C.mid + ';margin:2px 0 8px">' + c.note + '</div>' + (a ? '<table style="width:100%;border-collapse:collapse;font-size:13px;margin:6px 0">' + c.crit.map(function(t, k){ return '<tr style="border-top:1px solid ' + C.silver + '"><td style="padding:5px 8px 5px 0;width:230px;font-weight:600">' + labels[k] + '</td><td style="padding:5px 0">' + t + '</td></tr>'; }).join('') + '</table><div style="font-size:13px;line-height:1.5;margin-top:6px;color:' + C.char + '"><b>Conclusion. </b>' + c.why + '</div>' : '<button type="button" class="btn sm ghost" data-c="' + i + '">Show assessment</button>') + '</div>'; }).join('');
      qsa('[data-c]', cp).forEach(function(b){ b.addEventListener('click', function(){ ch[+b.dataset.c] = true; draw(); }); });
    }
    qs('[data-reset]', ctl).addEventListener('click', function(){ ch = [false, false, false]; draw(); });
    async function play(tok){ ch = [false, false, false]; draw(); api.cap('Three cases. Each is assessed against the same three criteria.'); await api.sleep(2600); for (var i = 0; i < 3; i++) { api.checkTok(tok); ch[i] = true; draw(); api.cap(i === 0 ? 'A generated value has no measurement and no reported performance.' : (i === 1 ? 'The compound is outside the training range of the model.' : 'The compound is inside the training range and a test-set error is reported.')); await api.sleep(3200); } api.cap(null); }
    return { render: draw, play: play };
  } });

/* ---------- guide entries ---------- */
GUIDE.steps['compare/What the scores mean'] = { notice: 'Shrinking the predictions toward the mean lowers R2 and raises RMSE, and leaves Spearman unchanged.', q: 'Can a model have a high Spearman correlation and a low R2?', a: 'Yes. The ordering of the compounds can be correct while the predicted values are shifted or compressed, so the errors are large.' };
GUIDE.steps['compare/Why no score is perfect'] = { notice: 'The ceiling falls as the measurement noise rises.', q: 'Why can a model that returns the true values not reach R2 of 1?', a: 'The measurement it is scored against contains random noise, which no model can predict.' };
GUIDE.steps['ai/What a model has not seen'] = { notice: 'RMSE outside the training range is larger than inside it.', q: 'Why is the error larger outside the training range?', a: 'The model has no training compounds there. A tree returns the value of its nearest leaf, so predictions stay constant while the measured values change.' };
GUIDE.steps['ai/Checking an answer'] = { notice: 'Only the third case meets all three criteria.', q: 'What are the three criteria?', a: 'The value can be checked against a measurement, the compound is within the training range, and a test-set performance is reported.' };
