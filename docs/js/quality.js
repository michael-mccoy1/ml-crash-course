"use strict";
/* Lesson 2: Data quality, censoring and filtering */
var L1 = new Lesson('quality', 'Data quality');
function qRange(f){ var lo = Infinity, hi = -Infinity; for (var i = 0; i < D.N; i++) { var v = D.COL[f][i]; if (v < lo) lo = v; if (v > hi) hi = v; } return [Math.floor(lo), Math.ceil(hi)]; }
L1.optsHTML = function(){
  return sharedOptions() +
    optGroup('Censored values', seg([['keep', 'Keep'], ['train', 'Drop from training'], ['both', 'Drop from both']], 'keep', 'data-cens'), 'Values at the assay floor. Only HLM has them.') +
    optRange('qz', 'Outlier cutoff (SD)', 0, 5, 0.5, 0, '0 means off. Removes training compounds this many SD from the mean.') +
    optRange('qmwlo', 'Minimum MW', 100, 400, 10, 100, '') + optRange('qmwhi', 'Maximum MW', 400, 900, 10, 900, '') +
    optRange('qlplo', 'Minimum logP', -3, 3, 0.5, -3, '') + optRange('qlphi', 'Maximum logP', 3, 9, 0.5, 9, 'Sliders at their ends mean no limit.') +
    optGroup('Use in other lessons', '<label style="display:flex;gap:8px;align-items:center;font-size:.86rem"><input type="checkbox" id="qapply" style="accent-color:#E1242A"> Train the other lessons on the filtered set</label>', 'The unseen set stays unfiltered there.') +
    '<div class="og"><div class="ol">Current filters</div><div class="rbox" id="qsum"></div></div>';
};
var qTimer = null;
function syncFilterControls(){
  if (!D) return; var mw = qRange('MW'), lp = qRange('logP');
  [['qmwlo', mw[0], mw[1]], ['qmwhi', mw[0], mw[1]], ['qlplo', lp[0], lp[1]], ['qlphi', lp[0], lp[1]]].forEach(function(a, k){ var el = $(a[0]); if (!el) return; el.min = a[1]; el.max = a[2]; el.step = k < 2 ? 10 : 0.5; el.value = k % 2 === 0 ? a[1] : a[2]; var o = $('o-' + a[0]); if (o) o.textContent = el.value; });
  var z = $('qz'); if (z) { z.value = 0; $('o-qz').textContent = '0'; }
  qsa('.seg[data-cens] button').forEach(function(b){ b.classList.toggle('on', b.dataset.v === 'keep'); b.disabled = !D.cens && b.dataset.v !== 'keep'; b.style.opacity = !D.cens && b.dataset.v !== 'keep' ? 0.4 : 1; });
  FILT.cens = 'keep'; FILT.z = 0; FILT.mw = null; FILT.lp = null; D.cache.exp = {};
}
function readFilters(){
  var mw = [+$('qmwlo').value, +$('qmwhi').value], lp = [+$('qlplo').value, +$('qlphi').value], r1 = [+$('qmwlo').min, +$('qmwhi').max], r2_ = [+$('qlplo').min, +$('qlphi').max];
  FILT.mw = (mw[0] <= r1[0] && mw[1] >= r1[1]) ? null : mw; FILT.lp = (lp[0] <= r2_[0] && lp[1] >= r2_[1]) ? null : lp; FILT.z = +$('qz').value; FILT.apply = $('qapply').checked;
  var o = $('o-qz'); if (o && FILT.z === 0) o.textContent = 'off';
}
function qSummary(){ var el = $('qsum'); if (!el || !D) return; var tr = filteredIdx(false), te = filteredIdx(true); el.innerHTML = '<div class="rrow">Training kept<b>' + tr.length + ' of ' + D.TRAIN.length + '</b></div><div class="rrow">Unseen kept<b>' + te.length + ' of ' + D.TEST.length + '</b></div>' + (filterOn() ? '' : '<div class="hint">No filter is active.</div>'); }
L1.onOpts = function(root){
  syncFilterControls(); readFilters(); qSummary();
  var fire = function(){ readFilters(); qSummary(); D.cache.expv = (D.cache.expv || 0) + 1; clearTimeout(qTimer); qTimer = setTimeout(function(){ L1.refresh(); if (FILT.apply) ['trees', 'nn', 'compare'].forEach(function(k){ if (LESSONS[k] && LESSONS[k].built) LESSONS[k].refresh(); }); }, 250); };
  qsa('input[type=range]', qs('aside', root)).forEach(function(r){ if (r.id.indexOf('q') === 0) r.addEventListener('input', fire); }); $('qapply').addEventListener('change', fire);
  qsa('.seg[data-cens] button', root).forEach(function(b){ b.addEventListener('click', function(){ if (b.disabled) return; qsa('.seg[data-cens] button', root).forEach(function(x){ x.classList.toggle('on', x === b); }); FILT.cens = b.dataset.v; fire(); }); });
};
L1.onData = function(){ if (L1.built) { syncFilterControls(); readFilters(); qSummary(); } };
featListeners.push(function(){ if (L1.built) { D.cache.exp = {}; L1.refresh(); } });

/* experiment: train on a filtered training set, score on the full and the filtered unseen sets */
function runExp(f){
  var feats = activeFeats(); if (!feats.length) return null; D.cache.exp = D.cache.exp || {}; var key = JSON.stringify([f, feats]); if (D.cache.exp[key]) return D.cache.exp[key];
  var tr = filteredIdx(false, f); if (tr.length < 60) return (D.cache.exp[key] = { tr: tr.length, te: 0, bad: true });
  var m = fitBoost(feats, tr, 80, 0.1, 3), te = filteredIdx(true, f), unc = D.cens ? D.TEST.filter(function(i){ return !D.cens[i]; }) : D.TEST;
  return (D.cache.exp[key] = { tr: tr.length, te: te.length, r2All: r2(D.TEST, m.F), r2Kept: te.length > 10 ? r2(te, m.F) : NaN, r2Unc: r2(unc, m.F), F: m.F, secs: m.secs });
}
function drawHist(canvas, vals, o){
  o = o || {}; var g = cv(canvas), x = g.x, box = { l: 46, r: g.w - 12, t: 14, b: g.h - 40 }, lo = D.ylo, hi = D.yhi, nb = 44, bins = new Array(nb).fill(0), grey = new Array(nb).fill(0);
  vals.forEach(function(i){ var b = clamp(Math.floor((D.Y[i] - lo) / (hi - lo) * nb), 0, nb - 1); if (o.out && o.out(i)) grey[b]++; else bins[b]++; }); var mx = Math.max.apply(null, bins.map(function(v, k){ return v + grey[k]; })) || 1, bw = (box.r - box.l) / nb, prog = o.prog === undefined ? 1 : o.prog;
  x.strokeStyle = C.silver; x.fillStyle = C.mid; x.textAlign = 'right'; for (var k = 0; k <= 4; k++) { var y = box.b - k / 4 * (box.b - box.t); x.beginPath(); x.moveTo(box.l, y); x.lineTo(box.r, y); x.stroke(); x.fillText(Math.round(mx * k / 4), box.l - 6, y + 4); }
  x.textAlign = 'center'; for (var j = 0; j <= 5; j++) x.fillText((lo + (hi - lo) * j / 5).toFixed(1), box.l + (box.r - box.l) * j / 5, box.b + 16); x.fillStyle = C.char; x.font = '600 12px Aptos, Segoe UI, sans-serif'; x.fillText(unitWord(), (box.l + box.r) / 2, box.b + 33); x.save(); x.translate(12, (box.t + box.b) / 2); x.rotate(-Math.PI / 2); x.fillText('compounds', 0, 0); x.restore();
  for (var b = 0; b < nb; b++) { var vis = clamp(prog * nb - b, 0, 1), h1 = bins[b] / mx * (box.b - box.t) * vis, h2 = grey[b] / mx * (box.b - box.t) * vis, isFloor = D.floor !== undefined && lo + (b + 0.5) * (hi - lo) / nb <= D.floor + (hi - lo) / nb && lo + (b + 0.5) * (hi - lo) / nb >= D.floor - (hi - lo) / nb * 0.01;
    x.fillStyle = (o.mark && isFloor) ? C.red : C.bg; x.fillRect(box.l + b * bw + 1, box.b - h1, bw - 2, h1); x.fillStyle = 'rgba(52,55,63,.28)'; x.fillRect(box.l + b * bw + 1, box.b - h1 - h2, bw - 2, h2); }
  g.box = box; g.bw = bw; g.mx = mx; return g;
}

/* ---------- Q1: look at the target ---------- */
L1.add({ short: 'Look at the target', title: 'Look at the distribution before modelling anything',
  change: 'Press Play. Switch the endpoint to logD to compare.',
  watch: 'HLM has a tall bar at the lowest value. Those compounds are below what the assay can measure.',
  build: function(api){
    var plot = api.el('pn', 'left:28px;top:28px;width:600px;height:408px', '<h4>Measured values of the training compounds</h4><canvas width="580" height="350"></canvas>');
    var side = api.el('pn', 'left:648px;top:28px;width:284px;height:408px', '<h4>Facts</h4><div class="sf"></div>');
    function facts(){ var tr = D.TRAIN, ys = tr.map(function(i){ return D.Y[i]; }), m = mean(ys), sd = Math.sqrt(mean(ys.map(function(v){ return (v - m) * (v - m); }))), cn = D.cens ? tr.filter(function(i){ return D.cens[i]; }).length : 0;
      qs('.sf', side).innerHTML = '<div class="nums" style="font-size:13px"><div><span>Compounds</span><b>' + tr.length + '</b></div><div><span>Mean</span><b>' + m.toFixed(2) + '</b></div><div><span>Spread (SD)</span><b>' + sd.toFixed(2) + '</b></div><div><span>At the assay floor</span><b style="color:' + C.red + '">' + (D.cens ? pct(cn, tr.length) + '%' : 'none') + '</b></div></div>' +
        (D.cens ? '<p style="font-size:12.5px;margin-top:12px;color:#4a4f5a">The assay cannot report clearance below ' + D.floor.toFixed(2) + '. Every slower compound is written as exactly that value. The real values are lower, but nobody knows by how much. This is called <b>censoring</b>.</p>' : '<p style="font-size:12.5px;margin-top:12px;color:#4a4f5a">logD values here are measured directly and spread across the whole range. There is no assay floor, so censoring is not an issue for this endpoint.</p>'); }
    function render(){ drawHist(qs('canvas', plot), D.TRAIN, { mark: !!D.cens }); facts(); }
    async function play(tok){ facts(); drawHist(qs('canvas', plot), D.TRAIN, { prog: 0 }); api.cap('Each bar counts training compounds in a range of ' + (D.key === 'HLM' ? 'clearance' : 'logD') + ' values.');
      await api.anim(2400, function(t){ drawHist(qs('canvas', plot), D.TRAIN, { prog: t }); });
      if (D.cens) { drawHist(qs('canvas', plot), D.TRAIN, { mark: true }); api.cap('One bar towers over the others. ' + pct(D.TRAIN.filter(function(i){ return D.cens[i]; }).length, D.TRAIN.length) + '% of compounds sit exactly at the assay floor.'); await api.sleep(3600); api.key('A pile-up at one value means censoring. A model cannot learn the true differences inside it.'); }
      else { api.cap('The values spread smoothly across the range, with no pile-up at either end.'); await api.sleep(3000); api.key('Look at the distribution first. It tells you what the assay can and cannot resolve.'); } }
    return { render: render, play: play };
  } });

/* two-model approach for censored data: classifier for the floor, regressor for the rest */
function runHurdle(){
  var feats = activeFeats(); if (!feats.length || !D.cens) return null; D.cache.hur = D.cache.hur || {}; var key = feats.join(','); if (D.cache.hur[key]) return D.cache.hur[key];
  var t0 = performance.now(), ind = Float32Array.from(D.cens), dsI = Object.assign({}, D, { Y: ind }), clf = fitBoost(feats, D.TRAIN, 80, 0.1, 3, 0, dsI), unc = D.TRAIN.filter(function(i){ return !D.cens[i]; }), reg = fitBoost(feats, unc, 80, 0.1, 3), H = new Float64Array(D.N);
  for (var i = 0; i < D.N; i++) { var p = clamp(clf.F[i], 0, 1); H[i] = p * D.floor + (1 - p) * reg.F[i]; }
  var uncT = D.TEST.filter(function(i){ return !D.cens[i]; }), accN = D.TEST.filter(function(i){ return (clf.F[i] > 0.5 ? 1 : 0) === D.cens[i]; }).length;
  return (D.cache.hur[key] = { F: H, regF: reg.F, r2All: r2(D.TEST, H), r2Unc: r2(uncT, reg.F), acc: accN / D.TEST.length, secs: (performance.now() - t0) / 1000 });
}

/* ---------- Q2: what censoring does ---------- */
L1.add({ short: 'Handling censoring', title: 'Three ways to treat the values at the assay floor',
  change: 'Press Play. Change the Features in the options to see whether the result holds.',
  watch: 'Hollow dots were censored in the lab. Each score is shown for all unseen compounds and for the measurable ones only.',
  build: function(api){
    var names = ['Keep them all', 'Drop them from training', 'Two models'], subs = ['One model, trained on everything', 'Train only on measurable values', 'Classifier for the floor, regressor for the rest'], P = [], T = [];
    for (var k = 0; k < 3; k++) { P.push(api.el('pn', 'left:' + (28 + k * 307) + 'px;top:28px;width:290px;height:316px', '<h4>' + names[k] + '</h4><div style="font-size:11px;color:#6B7280;margin:-3px 0 2px">' + subs[k] + '</div><canvas width="266" height="250"></canvas>')); T.push(api.el('pn', 'left:' + (28 + k * 307) + 'px;top:356px;width:290px;height:82px', '<div class="tt"></div>')); }
    function sc(canvas, F){ var g = cv(canvas), x = g.x, box = { l: 36, r: g.w - 8, t: 8, b: g.h - 32 }, lo = D.ylo, hi = D.yhi; axes(g, box, [lo, hi], [lo, hi], 'measured', 'predicted', 3); var sx = function(v){ return box.l + (v - lo) / (hi - lo) * (box.r - box.l); }, sy = function(v){ return box.b - (clamp(v, lo, hi) - lo) / (hi - lo) * (box.b - box.t); };
      x.strokeStyle = C.char; x.setLineDash([4, 4]); x.beginPath(); x.moveTo(sx(lo), sy(lo)); x.lineTo(sx(hi), sy(hi)); x.stroke(); x.setLineDash([]); var step = Math.ceil(D.TEST.length / 220);
      D.TEST.forEach(function(i, k){ if (k % step) return; if (D.cens[i]) { x.strokeStyle = 'rgba(52,55,63,.6)'; x.lineWidth = 1.1; x.beginPath(); x.arc(sx(D.Y[i]), sy(F[i]), 2.6, 0, 6.3); x.stroke(); } else { x.fillStyle = 'rgba(225,36,42,.7)'; x.beginPath(); x.arc(sx(D.Y[i]), sy(F[i]), 2.6, 0, 6.3); x.fill(); } }); }
    function tile(el, all, unc, extra){ qs('.tt', el).innerHTML = '<div style="display:flex;gap:22px"><div><div class="lbl">R2 all unseen</div><div class="big" style="font-size:22px">' + all.toFixed(2) + '</div></div><div><div class="lbl">R2 measurable only</div><div class="big" style="font-size:22px;color:' + C.red + '">' + unc.toFixed(2) + '</div></div></div>' + (extra ? '<div class="note" style="font-size:11.5px;color:#6B7280">' + extra + '</div>' : ''); }
    function results(){ var a = runExp({ cens: 'keep', z: 0, mw: null, lp: null }), b = runExp({ cens: 'train', z: 0, mw: null, lp: null }), c = runHurdle(); return a && b && c ? { a: a, b: b, c: c } : null; }
    function render(){ if (!D.cens) { P.forEach(function(p){ cv(qs('canvas', p)); }); T.forEach(function(t){ qs('.tt', t).innerHTML = ''; }); qs('.tt', T[0]).innerHTML = '<span class="note">This endpoint has no censored values. Switch to HLM clearance.</span>'; return; }
      var r = results(); if (!r) return; sc(qs('canvas', P[0]), r.a.F); sc(qs('canvas', P[1]), r.b.F); sc(qs('canvas', P[2]), r.c.F); tile(T[0], r.a.r2All, r.a.r2Unc, 'trained on ' + r.a.tr + ' compounds'); tile(T[1], r.b.r2All, r.b.r2Unc, 'trained on ' + r.b.tr + ' compounds'); tile(T[2], r.c.r2All, r.c.r2Unc, 'floor classifier accuracy ' + Math.round(r.c.acc * 100) + '%'); }
    async function play(tok){ if (!D.cens) { render(); api.cap('Switch the endpoint to HLM clearance to see censoring.'); return; }
      P.forEach(function(p){ p.style.opacity = 0.15; }); T.forEach(function(t){ t.style.opacity = 0.15; }); var r = results(); if (!r) return;
      api.cap('Hollow dots are censored. The question is what to do with them when training.'); await api.sleep(2600);
      for (var k = 0; k < 3; k++) { P[k].style.transition = T[k].style.transition = 'opacity .6s'; P[k].style.opacity = 1; T[k].style.opacity = 1; var s = [r.a, r.b, r.c][k]; sc(qs('canvas', P[k]), s.F); tile(T[k], s.r2All, s.r2Unc, ''); if (k === 0) api.cap('Keep everything: predictions pile up at the floor and the ranking among measurable compounds is poor.'); if (k === 1) api.cap('Drop them from training: measurable compounds are ranked better, but the model never learns that some compounds are stable.'); if (k === 2) api.cap('Two models: one predicts whether a compound hits the floor, the other predicts the value when it does not.'); await api.sleep(3600); }
      render(); api.key('Report stability and potency separately, or use two models. One number for both hides the problem.'); }
    return { render: render, play: play };
  } });

/* ---------- Q3: outliers ---------- */
L1.add({ short: 'Extreme values', title: 'Remove training compounds far from the mean',
  change: 'Drag the cutoff control below, or the Outlier cutoff in the options.',
  watch: 'Grey bars are training compounds that would be removed. Check the unseen R2 as the cutoff tightens.',
  build: function(api){
    var plot = api.el('pn', 'left:28px;top:28px;width:600px;height:408px', '<h4>Training compounds, grey = removed</h4><canvas width="580" height="350"></canvas>');
    var side = api.el('pn', 'left:648px;top:28px;width:284px;height:408px', '<h4>Effect</h4><div class="se"></div>');
    var ctl = api.addControl('Cutoff (SD) <input type="range" min="0" max="5" step="0.5" value="0" style="width:200px"><b class="zv">off</b>'); var zr = qs('input', ctl);
    function stats(z){ if (!D.zstat) { passFilter(D.TRAIN[0], false, { cens: 'keep', z: 1 }); } var rem = z > 0 ? D.TRAIN.filter(function(i){ return Math.abs(D.Y[i] - D.zstat.m) / D.zstat.sd > z; }) : []; return rem; }
    function draw(z){ var f = { cens: FILT.cens === 'both' ? 'train' : FILT.cens, z: z, mw: null, lp: null }; var rem = stats(z), rset = {}; rem.forEach(function(i){ rset[i] = 1; });
      drawHist(qs('canvas', plot), D.TRAIN, { out: function(i){ return rset[i]; }, mark: false }); var base = runExp({ cens: 'keep', z: 0, mw: null, lp: null }), cur = runExp({ cens: 'keep', z: z, mw: null, lp: null }); qs('.zv', ctl).textContent = z > 0 ? z : 'off';
      qs('.se', side).innerHTML = '<div class="nums" style="font-size:13px"><div><span>Removed from training</span><b>' + rem.length + '</b></div><div><span>Training compounds left</span><b>' + (D.TRAIN.length - rem.length) + '</b></div></div>' + (base && cur && !cur.bad ? '<div class="lbl" style="margin-top:12px">Unseen R2</div><div class="big" style="color:' + C.red + '">' + cur.r2All.toFixed(2) + '</div><div class="note" style="font-size:12px;color:#6B7280">without any filter: ' + base.r2All.toFixed(2) + '<br>change: ' + (cur.r2All - base.r2All >= 0 ? '+' : '') + (cur.r2All - base.r2All).toFixed(3) + '</div>' : '<p class="note">Too few compounds left.</p>') + '<p style="font-size:12.5px;margin-top:12px;color:#4a4f5a">The cutoff is counted in standard deviations from the training mean. A very strict cutoff throws away real information along with the noise.</p>'; }
    zr.addEventListener('input', function(){ api.tok++; api.clearTexts(); draw(+zr.value); });
    function render(){ var z = +zr.value; draw(z); }
    async function play(tok){ zr.value = 0; draw(0); await api.beat('With no cutoff every training compound is used.', 2200); var steps = [4, 3, 2.5, 2, 1.5]; for (var k = 0; k < steps.length; k++) { zr.value = steps[k]; draw(steps[k]); api.cap('Cutoff at ' + steps[k] + ' SD. Grey compounds are removed from training.'); await api.sleep(2200); }
      zr.value = 0; draw(0); api.key('Outlier removal helps only if the removed values are errors. It can also remove the compounds you most want to predict.'); }
    return { render: render, play: play };
  } });

/* ---------- Q4: chemical space range ---------- */
L1.add({ short: 'Chemical space', title: 'Restrict training to the property range you care about',
  change: 'Move Minimum or Maximum MW and logP in the options.',
  watch: 'Grey dots on the map are excluded. The score is shown for all unseen compounds and for the unseen compounds that stay inside the range.',
  build: function(api){
    var mapP = api.el('pn', 'left:28px;top:28px;width:560px;height:408px', '<h4>Chemical space map, grey = outside the range</h4><canvas width="540" height="352"></canvas>');
    var side = api.el('pn', 'left:608px;top:28px;width:324px;height:408px', '<h4>Effect</h4><div class="sr"></div>');
    function draw(f){
      var g = cv(qs('canvas', mapP)), x = g.x, W = g.w, H = g.h, sx = function(v){ return 14 + (v + 1) / 2 * (W - 28); }, sy = function(v){ return H - 14 - (v + 1) / 2 * (H - 28); };
      for (var i = 0; i < D.N; i++) { var inr = passFilter(i, D.test[i] === 1, f), te = D.test[i]; x.fillStyle = !inr ? 'rgba(52,55,63,.18)' : (te ? 'rgba(225,36,42,.85)' : 'rgba(161,177,195,.65)'); x.beginPath(); x.arc(sx(D.mx[i]), sy(D.my[i]), te ? 2.8 : 2.2, 0, 6.3); x.fill(); }
      x.fillStyle = C.char; x.font = '600 12px Aptos, Segoe UI, sans-serif'; x.fillText('red: unseen    blue grey: training    dark grey: outside the range', 14, 14); }
    function stat(f){ var base = runExp({ cens: 'keep', z: 0, mw: null, lp: null }), cur = runExp(f); qs('.sr', side).innerHTML = (cur && !cur.bad ? '<div class="nums" style="font-size:13px"><div><span>Training kept</span><b>' + cur.tr + ' of ' + D.TRAIN.length + '</b></div><div><span>Unseen inside the range</span><b>' + cur.te + ' of ' + D.TEST.length + '</b></div></div><div class="lbl" style="margin-top:12px">R2 on all unseen compounds</div><div class="big" style="color:' + C.red + '">' + cur.r2All.toFixed(2) + '</div><div class="note" style="font-size:12px;color:#6B7280">no filter: ' + (base ? base.r2All.toFixed(2) : '') + '</div><div class="lbl" style="margin-top:12px">R2 on unseen compounds inside the range</div><div class="big">' + (isNaN(cur.r2Kept) ? 'n/a' : cur.r2Kept.toFixed(2)) + '</div>' : '<p class="note">Too few compounds left. Widen the range.</p>') + '<p style="font-size:12.5px;margin-top:12px;color:#4a4f5a">A narrow range gives a model that is accurate inside it and unreliable outside it. This range is the model\'s applicability domain.</p>'; }
    function f_(){ return { cens: 'keep', z: 0, mw: FILT.mw, lp: FILT.lp }; }
    function render(){ var f = f_(); draw(f); stat(f); }
    async function play(tok){ var mw = [+$('qmwlo').min, +$('qmwhi').max]; draw({ cens: 'keep', z: 0, mw: null, lp: null }); stat({ cens: 'keep', z: 0, mw: null, lp: null }); await api.beat('Start with every compound.', 2000);
      var hi0 = mw[1], steps = [0.85, 0.7, 0.55]; for (var k = 0; k < steps.length; k++) { var hi = Math.round((mw[0] + (hi0 - mw[0]) * steps[k]) / 10) * 10, f = { cens: 'keep', z: 0, mw: [mw[0], hi], lp: null }; draw(f); stat(f); api.cap('Maximum molecular weight set to ' + hi + '. Compounds above it leave the training set and the unseen set.'); await api.sleep(2700); }
      var lp = [+$('qlplo').min, +$('qlphi').max], f2 = { cens: 'keep', z: 0, mw: null, lp: [1, 4] }; draw(f2); stat(f2); api.cap('Now keep only logP between 1 and 4, a typical drug-like window.'); await api.sleep(3000); render(); api.key('Restricting the range tunes a model to one region of chemistry. Say which region when you report the score.'); }
    return { render: render, play: play };
  } });

/* ---------- Q5: the test-set trap ---------- */
L1.add({ short: 'The test set trap', title: 'Filtering the unseen set raises a score without improving the model',
  change: 'Press Play. Set filters in the options to see how your own filters behave.',
  watch: 'One trained model is scored on the honest unseen set, on your filtered unseen set, and after removing its worst predictions.',
  build: function(api){
    var plot = api.el('pn', 'left:28px;top:28px;width:560px;height:408px', '<h4>R2 of one model on three versions of the unseen set</h4><canvas width="540" height="350"></canvas>');
    var side = api.el('pn', 'left:608px;top:28px;width:324px;height:408px', '<h4>Reading the bars</h4><div class="sk"></div>');
    function bars(){ var f = filterOn() ? { cens: FILT.cens, z: FILT.z, mw: FILT.mw, lp: FILT.lp } : { cens: 'keep', z: 0, mw: null, lp: null }, r = runExp(f); if (!r || r.bad) return null;
      var errs = D.TEST.map(function(i){ return [i, Math.abs(D.Y[i] - r.F[i])]; }).sort(function(a, b){ return a[1] - b[1]; }), keep = errs.slice(0, Math.round(errs.length * 0.9)).map(function(e){ return e[0]; });
      return { r: r, list: [['Honest unseen set', r.r2All, C.char], [filterOn() ? 'Your filtered unseen set' : 'Unseen set with no filters', filterOn() ? r.r2Kept : r.r2All, C.bg], ['Worst 10% of predictions removed', r2(keep, r.F), C.red]] }; }
    function draw(prog){
      var b = bars(), g = cv(qs('canvas', plot)), x = g.x; if (!b) { x.fillStyle = C.mid; x.fillText('Too few compounds left', 30, 40); return; }
      var box = { l: 60, r: g.w - 20, t: 20, b: g.h - 70 }, vals = b.list, mx = Math.max(0.5, Math.max.apply(null, vals.map(function(v){ return isNaN(v[1]) ? 0 : v[1]; })) * 1.2), bw = (box.r - box.l) / vals.length;
      x.strokeStyle = C.silver; x.fillStyle = C.mid; x.textAlign = 'right'; for (var k = 0; k <= 4; k++) { var y = box.b - k / 4 * (box.b - box.t); x.beginPath(); x.moveTo(box.l, y); x.lineTo(box.r, y); x.stroke(); x.fillText((mx * k / 4).toFixed(2), box.l - 6, y + 4); }
      vals.forEach(function(v, i){ var vis = clamp(prog * vals.length - i, 0, 1), val = isNaN(v[1]) ? 0 : v[1], hh = Math.max(0, val) / mx * (box.b - box.t) * vis, bx = box.l + i * bw + bw * 0.16; x.fillStyle = v[2]; x.fillRect(bx, box.b - hh, bw * 0.68, hh); x.fillStyle = C.char; x.textAlign = 'center'; x.font = '700 15px Aptos, Segoe UI, sans-serif'; if (vis > 0.9) x.fillText(isNaN(v[1]) ? 'n/a' : v[1].toFixed(2), bx + bw * 0.34, box.b - hh - 8); x.font = '12px Aptos, Segoe UI, sans-serif'; var words = v[0].split(' '), line = '', ly = box.b + 18; words.forEach(function(w){ if ((line + ' ' + w).length > 18) { x.fillText(line.trim(), bx + bw * 0.34, ly); line = w; ly += 14; } else line += ' ' + w; }); x.fillText(line.trim(), bx + bw * 0.34, ly); });
      qs('.sk', side).innerHTML = '<div class="nums" style="font-size:13px"><div><span>Training compounds used</span><b>' + b.r.tr + '</b></div><div><span>Unseen compounds kept</span><b>' + b.r.te + ' of ' + D.TEST.length + '</b></div></div><p style="font-size:12.5px;margin-top:12px;color:#4a4f5a">The first bar is the honest score. The last bar throws away the compounds the model got most wrong, so R2 jumps, yet the model is identical. Any filter applied to the unseen set can do a milder version of the same thing.</p>'; }
    function render(){ draw(1); }
    async function play(tok){ draw(0); await api.beat('One model is trained once. Only the exam changes.', 2600); await api.anim(3400, function(t){ draw(t); }); api.cap('Removing the worst 10% of predictions looks like a big improvement. It is not one.'); await api.sleep(3800); api.key('Filter the training set to improve a model. Never filter the unseen set to improve a score.'); }
    return { render: render, play: play };
  } });

/* ---------- Q6: combine ---------- */
L1.add({ short: 'Putting it together', title: 'Add one change at a time and read both scores',
  change: 'Press Play. Each group of bars is a fresh model scored on the unfiltered unseen set.',
  watch: 'Light bars: all unseen compounds. Red bars: measurable compounds only. Some changes help one and hurt the other.',
  build: function(api){
    var plot = api.el('pn', 'left:28px;top:28px;width:904px;height:408px', '<h4>Unseen R2 as the data handling changes (unfiltered unseen compounds)</h4><canvas width="880" height="350"></canvas>');
    function plan(){ var base = { cens: 'keep', z: 0, mw: null, lp: null }, p = [{ n: 'No filters', f: base }];
      if (D.cens) { p.push({ n: 'Drop censored from training', f: { cens: 'train', z: 0, mw: null, lp: null } }); p.push({ n: 'Two models', hur: true }); }
      var last = D.cens ? { cens: 'train', z: 0, mw: null, lp: null } : base; p.push({ n: 'Plus outliers beyond 3 SD', f: { cens: last.cens, z: 3, mw: null, lp: null } }); p.push({ n: 'Plus MW 200 to 600', f: { cens: last.cens, z: 3, mw: [200, 600], lp: null } }); p.push({ n: 'Plus logP 0 to 5', f: { cens: last.cens, z: 3, mw: [200, 600], lp: [0, 5] } }); return p; }
    function draw(prog){
      var P = plan(), res = P.map(function(s){ if (s.hur) { var h = runHurdle(); return { r2All: h.r2All, r2Unc: h.r2Unc }; } return runExp(s.f); }), g = cv(qs('canvas', plot)), x = g.x, box = { l: 60, r: g.w - 20, t: 24, b: g.h - 62 }, vals = [];
      res.forEach(function(r){ vals.push(r && !r.bad ? r.r2All : 0); vals.push(r && !r.bad && D.cens ? r.r2Unc : 0); }); var mx = Math.max(0.4, Math.max.apply(null, vals) * 1.2), bw = (box.r - box.l) / P.length;
      x.strokeStyle = C.silver; x.fillStyle = C.mid; x.textAlign = 'right'; for (var k = 0; k <= 4; k++) { var y = box.b - k / 4 * (box.b - box.t); x.beginPath(); x.moveTo(box.l, y); x.lineTo(box.r, y); x.stroke(); x.fillText((mx * k / 4).toFixed(2), box.l - 6, y + 4); }
      P.forEach(function(s, i){ var r = res[i], vis = clamp(prog * P.length - i, 0, 1), bx = box.l + i * bw + bw * 0.1, w2 = bw * 0.36;
        [[r && !r.bad ? r.r2All : 0, C.bg], [r && !r.bad && D.cens ? r.r2Unc : null, C.red]].forEach(function(b, j){ if (b[1] === null && b[0] === null) return; var v = b[0]; if (j === 1 && !D.cens) return; var hh = Math.max(0, v) / mx * (box.b - box.t) * vis, xx = bx + j * (w2 + 4); x.fillStyle = b[1]; x.fillRect(xx, box.b - hh, w2, hh); if (vis > 0.9) { x.fillStyle = C.char; x.textAlign = 'center'; x.font = '700 13px Aptos, Segoe UI, sans-serif'; x.fillText(v.toFixed(2), xx + w2 / 2, box.b - hh - 6); } });
        x.font = '12px Aptos, Segoe UI, sans-serif'; x.fillStyle = C.char; x.textAlign = 'center'; var words = s.n.split(' '), line = '', ly = box.b + 18; words.forEach(function(w){ if ((line + ' ' + w).length > 16) { x.fillText(line.trim(), bx + w2 + 2, ly); line = w; ly += 14; } else line += ' ' + w; }); x.fillText(line.trim(), bx + w2 + 2, ly); });
      x.font = '600 12px Aptos, Segoe UI, sans-serif'; x.textAlign = 'left'; x.fillStyle = C.bg; x.fillRect(box.r - 250, 8, 11, 11); x.fillStyle = C.char; x.fillText('all unseen compounds', box.r - 234, 18); x.fillStyle = C.red; x.fillRect(box.r - 100, 8, 11, 11); x.fillStyle = C.char; x.fillText('measurable only', box.r - 84, 18); }
    function render(){ draw(1); }
    async function play(tok){ draw(0); await api.beat('Every group is a fresh model scored on the same unfiltered unseen compounds.', 2800); await api.anim(6200, function(t){ draw(t); }); api.cap('Handling censoring changes the scores a lot. Outlier and range filters barely move them here.'); await api.sleep(3800); api.key('Test each change on its own and keep only what raises the honest score for the question you care about.'); }
    return { render: render, play: play };
  } });

L1.glossary([
  ['Censoring', 'A measurement reported at the assay limit because the true value is beyond what the assay can read.'],
  ['Assay floor', 'The lowest value the assay can report. Compounds below it are written as exactly this value.'],
  ['Outlier', 'A value far from the others. It may be an error or a real, unusual compound.'],
  ['Standard deviation (SD)', 'A measure of spread. A value 3 SD from the mean is rare in a bell-shaped distribution.'],
  ['Applicability domain', 'The region of chemical space where a model has seen enough examples to be trusted.'],
  ['Filtering the training set', 'Removing compounds before fitting, so the model learns from cleaner data.'],
  ['Filtering the unseen set', 'Removing hard compounds from the exam. It raises the score without improving the model.'],
  ['Honest score', 'A score on unseen compounds that have not been filtered or tuned on.'],
  ['MW', 'Molecular weight in daltons.'],
  ['logP', 'Preference of a compound for oil over water.']
]);
