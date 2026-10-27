"use strict";
/* Training-data filters shared by every lesson, and the filtering section each lesson adds. */
function qRange(f, d){ d = d || D; var lo = Infinity, hi = -Infinity; for (var i = 0; i < d.N; i++) { var v = d.COL[f][i]; if (v < lo) lo = v; if (v > hi) hi = v; } return [Math.floor(lo), Math.ceil(hi)]; }
var FQ = { timer: null, pending: false };

function fRow(q, label){ return '<div class="fr"><label class="row">' + label + '<output data-o="' + q + '"></output></label><input type="range" data-q="' + q + '"></div>'; }
function filterGroup(o){
  o = o || {}; var lab = o.labels !== false;
  return '<details class="filtgrp" open><summary>Filter training data</summary>' +
    (lab ? '<div class="fr"><div class="ol" style="margin:6px 0 4px">Censored values</div>' + seg([['keep', 'Keep'], ['train', 'Drop from training'], ['both', 'Drop from both']], FILT.cens, 'data-cens') + '<div class="hint">Values at the assay floor. HLM only.</div></div>' + fRow('z', 'Outlier cutoff (SD)') : '') +
    fRow('mwlo', 'Minimum MW') + fRow('mwhi', 'Maximum MW') + fRow('lplo', 'Minimum logP') + fRow('lphi', 'Maximum logP') +
    '<div class="rbox qsum"' + (lab ? '' : ' data-gpt="1"') + '></div><div class="hint">' + (lab ? 'Filters apply to the training set. The test set stays whole unless you choose Drop from both.' : 'These choose which Biogen molecules the GPT reads.') + '</div></details>';
}
function filterSummary(){
  if (!D) return; var tr = filteredIdx(false), te = filteredIdx(true), html = '<div class="rrow">Training kept<b>' + tr.length + ' of ' + D.TRAIN.length + '</b></div>' + (te.length !== D.TEST.length ? '<div class="rrow">Test kept<b>' + te.length + ' of ' + D.TEST.length + '</b></div>' : '');
  qsa('.qsum').forEach(function(el){ if (el.dataset.gpt) { if (!window.G || !G.allIds) { el.innerHTML = ''; return; } var d = DATA.HLM, all = 0, kept = 0; for (var i = 0; i < d.N; i++) if (G.allIds[i]) { all++; if (passRange(d, i)) kept++; } el.innerHTML = '<div class="rrow">Molecules kept<b>' + kept + ' of ' + all + '</b></div>'; } else el.innerHTML = html; });
}
function syncFilterControls(reset){
  if (!D) return; if (reset) { FILT.cens = 'keep'; FILT.z = 0; FILT.mw = null; FILT.lp = null; D.cache.exp = {}; }
  var mw = qRange('MW'), lp = qRange('logP'), vals = { z: FILT.z, mwlo: FILT.mw ? FILT.mw[0] : mw[0], mwhi: FILT.mw ? FILT.mw[1] : mw[1], lplo: FILT.lp ? FILT.lp[0] : lp[0], lphi: FILT.lp ? FILT.lp[1] : lp[1] };
  qsa('input[data-q]').forEach(function(el){ var q = el.dataset.q; if (q === 'z') { el.min = 0; el.max = 5; el.step = 0.5; } else { var r = q.indexOf('mw') === 0 ? mw : lp; el.min = r[0]; el.max = r[1]; el.step = q.indexOf('mw') === 0 ? 1 : 0.5; } el.value = vals[q]; });
  qsa('output[data-o]').forEach(function(o){ var q = o.dataset.o; o.textContent = q === 'z' && !FILT.z ? 'off' : vals[q]; });
  qsa('.seg[data-cens] button').forEach(function(b){ b.classList.toggle('on', b.dataset.v === FILT.cens); b.disabled = !D.cens && b.dataset.v !== 'keep'; b.style.opacity = !D.cens && b.dataset.v !== 'keep' ? 0.4 : 1; });
  filterSummary();
}
function readFilters(){
  var g = function(q){ var el = qs('input[data-q="' + q + '"]'); return el ? +el.value : null; }, el0 = qs('input[data-q="mwlo"]'), el1 = qs('input[data-q="lplo"]'); if (!el0) return;
  var mw = [g('mwlo'), g('mwhi')], lp = [g('lplo'), g('lphi')], r1 = [+el0.min, +qs('input[data-q="mwhi"]').max], r2_ = [+el1.min, +qs('input[data-q="lphi"]').max];
  FILT.mw = (mw[0] <= r1[0] && mw[1] >= r1[1]) ? null : mw; FILT.lp = (lp[0] <= r2_[0] && lp[1] >= r2_[1]) ? null : lp; var z = g('z'); if (z !== null) FILT.z = z;
}
function flushFilter(){
  if (HOLD) { FQ.pending = true; return; } FQ.pending = false;
  LORDER.forEach(function(id){ var L = LESSONS[id]; if (L && L.built) L.refresh(); });
}
function filterChanged(){ filterSummary(); clearTimeout(FQ.timer); FQ.timer = setTimeout(flushFilter, 260); }
document.addEventListener('input', function(e){
  var t = e.target; if (!t || !t.dataset || !t.dataset.q) return; var q = t.dataset.q, v = +t.value;
  var pair = { mwlo: 'mwhi', mwhi: 'mwlo', lplo: 'lphi', lphi: 'lplo' }[q]; if (pair) { var o = qs('input[data-q="' + pair + '"]'); if (o && ((q.slice(-2) === 'lo' && v > +o.value) || (q.slice(-2) === 'hi' && v < +o.value))) { qsa('input[data-q="' + pair + '"]').forEach(function(x){ x.value = v; }); qsa('output[data-o="' + pair + '"]').forEach(function(x){ x.textContent = v; }); } }
  qsa('input[data-q="' + q + '"]').forEach(function(x){ if (x !== t) x.value = t.value; }); qsa('output[data-o="' + q + '"]').forEach(function(x){ x.textContent = q === 'z' && v === 0 ? 'off' : t.value; });
  readFilters(); filterChanged();
});
document.addEventListener('click', function(e){
  var b = e.target.closest ? e.target.closest('.seg[data-cens] button') : null; if (!b || b.disabled) return; FILT.cens = b.dataset.v; qsa('.seg[data-cens] button').forEach(function(x){ x.classList.toggle('on', x.dataset.v === FILT.cens); }); filterChanged();
});
window.addEventListener('pointerup', function(){ if (FQ.pending) setTimeout(flushFilter, 80); }, true);
function setFilters(f){ FILT.cens = f.cens || 'keep'; FILT.z = f.z || 0; FILT.mw = f.mw || null; FILT.lp = f.lp || null; syncFilterControls(); clearTimeout(FQ.timer); flushFilter(); }
function filtCopy(f){ f = f || FILT; return { cens: f.cens, z: f.z, mw: f.mw ? f.mw.slice() : null, lp: f.lp ? f.lp.slice() : null }; }
var NOFILT = { cens: 'keep', z: 0, mw: null, lp: null };

/* ---------- experiments: train on a filtered set, score on whole and filtered test sets ---------- */
function scoreExp(F, trn, f, secs){
  var te = filteredIdx(true, f), unc = D.cens ? D.TEST.filter(function(i){ return !D.cens[i]; }) : D.TEST;
  return { tr: trn, te: te.length, r2All: r2(D.TEST, F), r2Kept: te.length > 10 ? r2(te, F) : NaN, r2Unc: r2(unc, F), F: F, secs: secs };
}
function runExp(f){
  var feats = activeFeats(); if (!feats.length) return null; D.cache.exp = D.cache.exp || {}; var key = JSON.stringify(['trees', EP, feats, filtKey(f)]); if (D.cache.exp[key]) return D.cache.exp[key];
  var tr = filteredIdx(false, f); if (tr.length < 60) return (D.cache.exp[key] = { tr: tr.length, te: 0, bad: true });
  var m = fitBoost(feats, tr, 80, 0.1, 3); return (D.cache.exp[key] = scoreExp(m.F, tr.length, f, m.secs));
}
async function expNN(f){
  var feats = activeFeats(); if (!feats.length) return null; D.cache.exp = D.cache.exp || {}; var p = NP(), ep = Math.min(p.epochs, 40), key = JSON.stringify(['nn', EP, feats, filtKey(f), p.layers, p.units, p.drop, p.lr, ep, p.early]); if (D.cache.exp[key]) return D.cache.exp[key];
  var tr = filteredIdx(false, f); if (tr.length < 120) return (D.cache.exp[key] = { tr: tr.length, te: 0, bad: true });
  var res = await nnTrain({ params: { epochs: ep }, filt: f }); return (D.cache.exp[key] = scoreExp(res.pred, tr.length, f, res.secs));
}
function runHurdle(){
  var feats = activeFeats(); if (!feats.length || !D.cens) return null; D.cache.hur = D.cache.hur || {}; var key = feats.join(','); if (D.cache.hur[key]) return D.cache.hur[key];
  var t0 = performance.now(), ind = Float32Array.from(D.cens), dsI = Object.assign({}, D, { Y: ind }), clf = fitBoost(feats, D.TRAIN, 80, 0.1, 3, 0, dsI), unc = D.TRAIN.filter(function(i){ return !D.cens[i]; }), reg = fitBoost(feats, unc, 80, 0.1, 3), H = new Float64Array(D.N);
  for (var i = 0; i < D.N; i++) { var p = clamp(clf.F[i], 0, 1); H[i] = p * D.floor + (1 - p) * reg.F[i]; }
  var uncT = D.TEST.filter(function(i){ return !D.cens[i]; }), accN = D.TEST.filter(function(i){ return (clf.F[i] > 0.5 ? 1 : 0) === D.cens[i]; }).length;
  return (D.cache.hur[key] = { F: H, regF: reg.F, r2All: r2(D.TEST, H), r2Unc: r2(uncT, reg.F), acc: accN / D.TEST.length, secs: (performance.now() - t0) / 1000 });
}

/* ---------- drawing ---------- */
function hopts(o){ o.out = function(i){ return !passFilter(i, false); }; return o; }
function drawHist(canvas, vals, o){
  o = o || {}; var g = cv(canvas), x = g.x, box = { l: o.l || 42, r: g.w - 8, t: 10, b: g.h - 38 }, val = o.val || function(i){ return D.Y[i]; }, lo = o.lo !== undefined ? o.lo : D.ylo, hi = o.hi !== undefined ? o.hi : D.yhi, nb = o.nb || 40, bins = new Array(nb).fill(0), grey = new Array(nb).fill(0);
  vals.forEach(function(i){ var b = clamp(Math.floor((val(i) - lo) / (hi - lo) * nb), 0, nb - 1); if (o.out && o.out(i)) grey[b]++; else bins[b]++; }); var mx = Math.max.apply(null, bins.map(function(v, k){ return v + grey[k]; })) || 1, ystep = niceStep(mx, 4), bw = (box.r - box.l) / nb; mx = Math.ceil(mx / ystep - 1e-9) * ystep;
  x.strokeStyle = C.silver; x.fillStyle = C.mid; x.textAlign = 'right'; for (var yv = 0; yv <= mx + ystep * 1e-6; yv += ystep) { var y = box.b - yv / mx * (box.b - box.t); x.beginPath(); x.moveTo(box.l, y); x.lineTo(box.r, y); x.stroke(); x.fillText(Math.round(yv), box.l - 6, y + 4); }
  var xtk = niceTicks(lo, hi, 5); x.textAlign = 'center'; xtk.v.forEach(function(v){ var tx = tickText(v, xtk.step, o.dec), tw = x.measureText(tx).width, xp = box.l + (v - lo) / (hi - lo) * (box.r - box.l); x.fillText(tx, Math.max(tw / 2 + 1, Math.min(xp, g.w - 2 - tw / 2)), box.b + 15); }); x.fillStyle = C.char; x.font = '600 12px Aptos, Segoe UI, sans-serif'; x.fillText(o.unit || unitWord(), (box.l + box.r) / 2, box.b + 31);
  for (var b = 0; b < nb; b++) { var h1 = bins[b] / mx * (box.b - box.t), h2 = grey[b] / mx * (box.b - box.t), isFloor = o.mark && D.floor !== undefined && D.floor >= lo + b * (hi - lo) / nb - 1e-9 && D.floor < lo + (b + 1) * (hi - lo) / nb;
    x.fillStyle = isFloor ? C.red : C.bg; x.fillRect(box.l + b * bw + 1, box.b - h1, bw - 2, h1); x.fillStyle = 'rgba(52,55,63,.32)'; x.fillRect(box.l + b * bw + 1, box.b - h1 - h2, bw - 2, h2); }
  g.box = box; g.bw = bw; g.mx = mx; return g;
}
function drawBars(canvas, groups, o){
  o = o || {}; var g = cv(canvas), x = g.x, box = { l: 38, r: g.w - 8, t: 16, b: g.h - 40 }, all = []; groups.forEach(function(gr){ gr.bars.forEach(function(b){ all.push(isNaN(b.v) ? 0 : b.v); }); });
  var mx = Math.max(o.min || 0.4, Math.max.apply(null, all) * 1.2), gw = (box.r - box.l) / groups.length;
  x.strokeStyle = C.silver; x.fillStyle = C.mid; x.textAlign = 'right'; for (var k = 0; k <= 4; k++) { var y = box.b - k / 4 * (box.b - box.t); x.beginPath(); x.moveTo(box.l, y); x.lineTo(box.r, y); x.stroke(); x.fillText((mx * k / 4).toFixed(2), box.l - 5, y + 4); }
  groups.forEach(function(gr, gi){ var n = gr.bars.length, bw = Math.min(46, (gw - 18) / n), x0 = box.l + gi * gw + (gw - bw * n - (n - 1) * 4) / 2;
    gr.bars.forEach(function(b, bi){ var v = isNaN(b.v) ? 0 : Math.max(0, b.v), hh = v / mx * (box.b - box.t), bx = x0 + bi * (bw + 4); x.fillStyle = b.color; x.fillRect(bx, box.b - hh, bw, hh); x.fillStyle = C.char; x.textAlign = 'center'; x.font = '700 13px Aptos, Segoe UI, sans-serif'; x.fillText(isNaN(b.v) ? 'n/a' : b.v.toFixed(2), bx + bw / 2, box.b - hh - 5); });
    x.font = '600 11.5px Aptos, Segoe UI, sans-serif'; x.fillStyle = C.char; x.textAlign = 'center'; var words = gr.label.split(' '), line = '', ly = box.b + 16; words.forEach(function(w){ if ((line + ' ' + w).trim().length > 14) { x.fillText(line.trim(), box.l + gi * gw + gw / 2, ly); line = w; ly += 13; } else line += ' ' + w; }); x.fillText(line.trim(), box.l + gi * gw + gw / 2, ly); x.font = '12px Aptos, Segoe UI, sans-serif'; });
  return g;
}
function scatterCens(canvas, F, label){
  var g = cv(canvas), x = g.x, box = { l: 34, r: g.w - 8, t: 8, b: g.h - 32 }, lo = D.ylo, hi = D.yhi; axes(g, box, [lo, hi], [lo, hi], 'measured', 'predicted', 3); var sx = function(v){ return box.l + (v - lo) / (hi - lo) * (box.r - box.l); }, sy = function(v){ return box.b - (clamp(v, lo, hi) - lo) / (hi - lo) * (box.b - box.t); };
  x.strokeStyle = C.char; x.setLineDash([4, 4]); x.beginPath(); x.moveTo(sx(lo), sy(lo)); x.lineTo(sx(hi), sy(hi)); x.stroke(); x.setLineDash([]); var step = Math.ceil(D.TEST.length / 220);
  D.TEST.forEach(function(i, k){ if (k % step) return; if (D.cens && D.cens[i]) { x.strokeStyle = 'rgba(52,55,63,.6)'; x.lineWidth = 1.1; x.beginPath(); x.arc(sx(D.Y[i]), sy(F[i]), 2.6, 0, 6.3); x.stroke(); } else { x.fillStyle = 'rgba(225,36,42,.7)'; x.beginPath(); x.arc(sx(D.Y[i]), sy(F[i]), 2.6, 0, 6.3); x.fill(); } });
  if (label) { x.fillStyle = C.char; x.font = '700 13px Aptos, Segoe UI, sans-serif'; x.textAlign = 'left'; x.fillText(label, box.l + 6, box.t + 14); } return g;
}

/* ---------- the filtering section every lesson has ---------- */
function filterPresets(){
  var p = [['No filter', NOFILT]]; if (D && D.cens) p.push(['Drop censored from training', { cens: 'train', z: 0, mw: null, lp: null }]); p.push(['Drop outliers beyond 2.5 SD', { cens: 'keep', z: 2.5, mw: null, lp: null }]);
  var mw = qRange('MW'); p.push(['Keep MW 250 to 450', { cens: 'keep', z: 0, mw: [Math.max(mw[0], 250), Math.min(mw[1], 450)], lp: null }]); if (D && D.cens) p.push(['Censored out and 3 SD', { cens: 'train', z: 3, mw: null, lp: null }]); return p;
}
function sameFilter(a, b){ return JSON.stringify(filtCopy(a)) === JSON.stringify(filtCopy(b)); }
function presetBar(api, top){
  var bar = api.el('presets', 'left:28px;top:' + top + 'px;width:904px;height:30px'); function draw(){ bar.innerHTML = '<span class="lbl" style="align-self:center;margin-right:4px">Presets</span>' + filterPresets().map(function(p, i){ return '<button type="button" data-i="' + i + '"' + (sameFilter(p[1], FILT) ? ' class="on"' : '') + '>' + esc(p[0]) + '</button>'; }).join(''); qsa('button', bar).forEach(function(b){ b.addEventListener('click', function(){ setFilters(filterPresets()[+b.dataset.i][1]); }); }); } draw(); return { draw: draw };
}
function filterStepDef(kind){
  if (kind === 'gpt') return gptFilterStep();
  var nn = kind === 'nn', what = nn ? 'a neural network' : 'a gradient boosting model';
  return { short: 'Filtering the data', uses: nn ? ['endpoint', 'features', 'filters', 'nlayers', 'nunits', 'nepochs'] : ['endpoint', 'features', 'filters'],
    title: 'Remove compounds, then compare scores',
    watch: 'Filtering means removing some compounds from training before the model learns. You can drop compounds at the assay floor, outliers (values far from the rest, counted in standard deviations, SD) or molecules outside a size range. Grey bars on the left show which compounds are removed. The middle chart compares test R2 without a filter (grey) and with it (red). The right chart shows the new predictions.',
    change: 'Click a preset above the charts.',
    build: function(api){
      var pr = presetBar(api, 18);
      var P1 = api.el('pn', 'left:28px;top:60px;width:288px;height:376px', '<h4>Training compounds</h4><canvas width="264" height="236"></canvas><div class="f1 nums" style="font-size:12.5px;margin-top:8px"></div>');
      var P2 = api.el('pn', 'left:336px;top:60px;width:288px;height:376px', '<h4>R2 on test compounds</h4><canvas width="264" height="236"></canvas><div class="f2 nums" style="font-size:12.5px;margin-top:8px"></div>');
      var P3 = api.el('pn', 'left:644px;top:60px;width:288px;height:376px', '<h4>Predictions with your filters</h4><canvas width="264" height="236"></canvas><div class="f3 nums" style="font-size:12.5px;margin-top:8px"></div>');
      var seq = 0;
      function counts(){
        var tr = D.TRAIN, kept = filteredIdx(false).length, cn = D.cens ? tr.filter(function(i){ return D.cens[i]; }).length : 0;
        drawHist(qs('canvas', P1), tr, hopts({ mark: !!D.cens }));
        qs('.f1', P1).innerHTML = '<div><span>Compounds in training</span><b>' + tr.length + '</b></div><div><span>Removed by filters</span><b>' + (tr.length - kept) + '</b></div>' + (D.cens ? '<div><span>At the assay floor (red bar)</span><b>' + pct(cn, tr.length) + '%</b></div>' : '');
      }
      async function render(){
        var my = ++seq; counts(); var feats = activeFeats(), g2 = cv(qs('canvas', P2)), g3 = cv(qs('canvas', P3)); pr.draw();
        if (!feats.length) { qs('.f2', P2).innerHTML = '<span class="note">Switch on at least one feature.</span>'; qs('.f3', P3).innerHTML = ''; return; }
        var f = filtCopy(), base, cur;
        if (nn) { api.busy('Training a network for each setting'); await waitRelease(); if (my !== seq) return; await tick(); base = await expNN(NOFILT); if (my !== seq) return; cur = filterOn(f) ? await expNN(f) : base; api.idle(); }
        else { base = runExp(NOFILT); cur = filterOn(f) ? runExp(f) : base; }
        if (my !== seq) return; if (!base || !cur || cur.bad) { qs('.f2', P2).innerHTML = '<span class="note">Too few compounds left. Loosen the filters.</span>'; qs('.f3', P3).innerHTML = ''; return; }
        var groups = [{ label: D.cens ? 'Every test compound' : 'Test compounds', bars: [{ v: base.r2All, color: C.bg }, { v: cur.r2All, color: C.red }] }]; if (D.cens) groups.push({ label: 'Real lab values only', bars: [{ v: base.r2Unc, color: C.bg }, { v: cur.r2Unc, color: C.red }] });
        drawBars(qs('canvas', P2), groups); var d = cur.r2All - base.r2All;
        qs('.f2', P2).innerHTML = '<div><span><i style="display:inline-block;width:10px;height:10px;background:' + C.bg + ';margin-right:6px"></i>No filter</span><b>' + base.r2All.toFixed(3) + '</b></div><div><span><i style="display:inline-block;width:10px;height:10px;background:' + C.red + ';margin-right:6px"></i>Your filters</span><b>' + cur.r2All.toFixed(3) + '</b></div><div><span>Change</span><b style="color:' + (d > 0.0005 ? '#34373F' : (d < -0.0005 ? C.red : C.char)) + '">' + (d >= 0 ? '+' : '') + d.toFixed(3) + '</b></div>';
        scatterCens(qs('canvas', P3), cur.F, filterOn(f) ? 'filtered' : 'no filter'); qs('.f3', P3).innerHTML = '<div><span>Trained on</span><b>' + cur.tr + ' compounds</b></div><div><span>Training time</span><b>' + fmtT(cur.secs) + '</b></div>' + (D.cens ? '<div><span>Hollow dots</span><b>censored</b></div>' : '');
      }
      return { render: render };
    } };
}

/* ---------- the GPT reads molecules, so only range filters apply ---------- */
function passRange(d, i, f){ f = f || FILT; if (f.mw && (d.COL.MW[i] < f.mw[0] || d.COL.MW[i] > f.mw[1])) return false; if (f.lp && (d.COL.logP[i] < f.lp[0] || d.COL.logP[i] > f.lp[1])) return false; return true; }
function gptFilterStep(){
  return { short: 'Filtering the data', uses: ['filters', 'gsteps'], title: 'Filter which molecules the GPT reads',
    watch: 'A GPT learns only from the molecule text, never from lab values, so only the molecular weight and logP ranges apply here. Narrow a range, retrain the GPT, and the molecules it writes shift to match what it read. The charts compare the molecules it was shown with the molecules it wrote.',
    change: 'Set Minimum MW or Maximum MW in the options panel, press Train in the GPT workbench, then return here.',
    build: function(api){
      var P1 = api.el('pn', 'left:28px;top:28px;width:288px;height:408px', '<h4>Molecular weight of the corpus</h4><canvas width="264" height="250"></canvas><div class="f1 nums" style="font-size:12.5px;margin-top:8px"></div>');
      var P2 = api.el('pn', 'left:336px;top:28px;width:288px;height:408px', '<h4>SMILES length of the corpus</h4><canvas width="264" height="250"></canvas><div class="f2 nums" style="font-size:12.5px;margin-top:8px"></div>');
      var P3 = api.el('pn', 'left:644px;top:28px;width:288px;height:408px', '<h4>Length of molecules it wrote</h4><canvas width="264" height="250"></canvas><div class="f3 nums" style="font-size:12.5px;margin-top:8px"></div>');
      function render(){
        var d = DATA.HLM; if (!d) return; gptSetup(); var all = []; for (var i = 0; i < d.N; i++) if (G.allIds[i]) all.push(i); var mw = qRange('MW', d), out = function(i){ return !passRange(d, i); }, kept = all.filter(function(i){ return !out(i); });
        drawHist(qs('canvas', P1), all, { val: function(i){ return d.COL.MW[i]; }, lo: mw[0], hi: mw[1], unit: 'MW', dec: 0, out: out });
        qs('.f1', P1).innerHTML = '<div><span>Molecules the GPT can read</span><b>' + all.length + '</b></div><div><span>Kept by your range</span><b>' + kept.length + '</b></div><div><span>Mean MW kept</span><b>' + mean(kept.map(function(i){ return d.COL.MW[i]; })).toFixed(0) + '</b></div>';
        drawHist(qs('canvas', P2), all, { val: function(i){ return G.tokLen[i]; }, lo: 0, hi: TLEN, unit: 'tokens', dec: 0, nb: 24, out: out });
        qs('.f2', P2).innerHTML = '<div><span>Mean length, all</span><b>' + mean(all.map(function(i){ return G.tokLen[i]; })).toFixed(1) + ' tokens</b></div><div><span>Mean length, kept</span><b>' + mean(kept.map(function(i){ return G.tokLen[i]; })).toFixed(1) + ' tokens</b></div>';
        var gen = G.gen && G.gen.sig === gptCorpusSig() ? G.gen : null, g = cv(qs('canvas', P3)), x = g.x;
        if (!gen) { x.fillStyle = C.mid; x.textAlign = 'center'; x.fillText('Train the GPT in the workbench to see this.', g.w / 2, g.h / 2); qs('.f3', P3).innerHTML = G.gen ? '<span class="note">The options changed since the last training.</span>' : ''; return; }
        var nb = 24, bins = new Array(nb).fill(0), ref = new Array(nb).fill(0); gen.lens.forEach(function(l){ bins[clamp(Math.floor(l / TLEN * nb), 0, nb - 1)]++; }); kept.forEach(function(i){ ref[clamp(Math.floor(G.tokLen[i] / TLEN * nb), 0, nb - 1)]++; });
        var bx = { l: 30, r: g.w - 8, t: 10, b: g.h - 34 }, mx = Math.max(Math.max.apply(null, bins) / gen.lens.length, Math.max.apply(null, ref) / kept.length) || 1, bw = (bx.r - bx.l) / nb; x.strokeStyle = C.silver; x.beginPath(); x.moveTo(bx.l, bx.b); x.lineTo(bx.r, bx.b); x.stroke();
        for (var b = 0; b < nb; b++) { var h1 = bins[b] / gen.lens.length / mx * (bx.b - bx.t); x.fillStyle = C.red; x.fillRect(bx.l + b * bw + 1, bx.b - h1, bw - 2, h1); }
        x.strokeStyle = C.char; x.lineWidth = 2; x.beginPath(); for (var b2 = 0; b2 < nb; b2++) { var yy = bx.b - ref[b2] / kept.length / mx * (bx.b - bx.t), xx = bx.l + (b2 + 0.5) * bw; b2 ? x.lineTo(xx, yy) : x.moveTo(xx, yy); } x.stroke(); x.lineWidth = 1; x.fillStyle = C.char; x.textAlign = 'center'; x.font = '600 12px Aptos, Segoe UI, sans-serif'; x.fillText('tokens', (bx.l + bx.r) / 2, g.h - 6); x.font = '11px Aptos, Segoe UI, sans-serif'; x.fillText('0', bx.l, bx.b + 14); x.fillText(String(TLEN), bx.r, bx.b + 14);
        qs('.f3', P3).innerHTML = '<div><span><i style="display:inline-block;width:10px;height:10px;background:' + C.red + ';margin-right:6px"></i>Written by the GPT</span><b>' + mean(gen.lens).toFixed(1) + '</b></div><div><span><i style="display:inline-block;width:14px;height:2px;background:' + C.char + ';margin-right:4px;vertical-align:3px"></i>Corpus it read</span><b>' + mean(kept.map(function(i){ return G.tokLen[i]; })).toFixed(1) + '</b></div><div><span>Pass the bracket check</span><b>' + gen.ok + ' of ' + gen.lens.length + '</b></div>';
      }
      return { render: render };
    } };
}

/* ---------- data steps for the first lesson ---------- */
function targetStepDef(){
  return { short: 'The target', uses: ['endpoint', 'filters'], title: 'The values we want to predict',
    watch: 'The target is the value we want the model to predict. Each bar counts compounds whose value falls in that range. In HLM clearance, the tall bar at the far left is the assay floor. These compounds are too stable for the test to measure, so all of them are set to the lowest value it can report. They are censored: the lab knows only that the value is below a limit.',
    change: 'Switch the endpoint to logD to compare, then set a filter. Grey bars are the compounds the filter removes.',
    build: function(api){
      var plot = api.el('pn', 'left:28px;top:28px;width:600px;height:408px', '<h4>Measured values of the training compounds</h4><canvas width="576" height="350"></canvas>');
      var side = api.el('pn', 'left:648px;top:28px;width:284px;height:408px', '<h4>Facts</h4><div class="sf"></div>');
      function render(){
        var tr = D.TRAIN, ys = tr.map(function(i){ return D.Y[i]; }), m = mean(ys), sd = Math.sqrt(mean(ys.map(function(v){ return (v - m) * (v - m); }))), cn = D.cens ? tr.filter(function(i){ return D.cens[i]; }).length : 0;
        drawHist(qs('canvas', plot), tr, hopts({ mark: !!D.cens, nb: 44, l: 46 }));
        qs('.sf', side).innerHTML = '<div class="nums" style="font-size:13px"><div><span>Compounds</span><b>' + tr.length + '</b></div><div><span>Mean</span><b>' + m.toFixed(2) + '</b></div><div><span>Spread (SD)</span><b>' + sd.toFixed(2) + '</b></div><div><span>At the assay floor</span><b style="color:' + C.red + '">' + (D.cens ? pct(cn, tr.length) + '%' : 'none') + '</b></div><div><span>Removed by your filters</span><b>' + (tr.length - filteredIdx(false).length) + '</b></div></div>' +
          (D.cens ? '<p style="font-size:12.5px;margin-top:12px;color:#34373F">The assay cannot report clearance below ' + D.floor.toFixed(2) + '. Every slower compound is written as exactly that value. The true values are lower by an unknown amount. This is censoring.</p>' : '<p style="font-size:12.5px;margin-top:12px;color:#34373F">logD was measured directly and spreads across the whole range. There is no assay floor.</p>');
      }
      return { render: render };
    } };
}
function censorStepDef(){
  var OPTS = [['Keep them all', 'One model learns from every training compound, including those at the assay floor.'], ['Drop them from training', 'The model learns only from compounds that were actually measured.'], ['Two models', 'One model guesses whether a compound is at the floor. A second model predicts the value for the rest.']];
  var sel = 0;
  return { short: 'Handling censoring', uses: ['endpoint', 'features'], title: 'Handling values at the assay floor',
    watch: 'Censored means the lab knew only that a value was below the assay limit, not what it was. About 31% of the HLM compounds are censored. There are three ways to treat them, and each card trains a model one way. The test set holds compounds with real values and compounds at the floor. The left bar scores all of them. The right bar scores only the compounds with a real value.',
    change: 'Click a card to see its predictions on the right.',
    build: function(api){
      var cards = api.el('abs', 'left:28px;top:28px;width:540px;height:410px', ''), sp = api.el('pn', 'left:588px;top:28px;width:344px;height:410px', '<h4 class="sh2"></h4><canvas width="320" height="270"></canvas><div class="lg" style="font-size:12px;line-height:1.5;margin-top:6px"></div>');
      var res = null;
      function render(){
        if (!D.cens) { cards.innerHTML = '<div class="pn" style="position:relative;padding:20px"><b>logD has no censored values.</b><p class="note" style="margin-top:6px">Switch the endpoint to HLM clearance to see this step.</p></div>'; cv(qs('canvas', sp)); qs('.sh2', sp).textContent = ''; qs('.lg', sp).innerHTML = ''; return; }
        var a = runExp(NOFILT), b = runExp({ cens: 'train', z: 0, mw: null, lp: null }), c = runHurdle(); if (!a || !b || !c) return; res = [{ F: a.F, all: a.r2All, unc: a.r2Unc, note: 'trained on ' + a.tr + ' compounds' }, { F: b.F, all: b.r2All, unc: b.r2Unc, note: 'trained on ' + b.tr + ' compounds' }, { F: c.F, all: c.r2All, unc: c.r2Unc, note: 'floor guess correct for ' + Math.round(c.acc * 100) + '% of compounds' }];
        cards.innerHTML = OPTS.map(function(o, i){ var r = res[i], on = i === sel; return '<div data-c="' + i + '" class="pn" style="position:relative;margin-bottom:10px;cursor:pointer;border:' + (on ? '2px solid ' + C.red : '1px solid ' + C.silver) + ';padding:9px 14px"><div style="display:flex;justify-content:space-between;align-items:baseline"><b style="font-size:14px">' + (i + 1) + '. ' + o[0] + '</b><span class="note" style="font-size:11.5px">' + r.note + '</span></div><div style="font-size:12.5px;color:#34373F;margin:2px 0 4px">' + o[1] + '</div><div style="display:grid;grid-template-columns:1fr 1fr;gap:6px 22px">' + '<div>' + '<div style="font-size:11.5px;color:#34373F">Every test compound, floor ones included</div><div class="pbar" style="height:11px;margin:2px 0"><i style="background:' + C.bg + ';width:' + clamp(r.all * 100 / 0.45, 0, 100) + '%;transition:none"></i></div><b style="font-size:13px">' + r.all.toFixed(2) + '</b></div><div><div style="font-size:11.5px;color:#34373F">Test compounds with a real lab value (floor ones left out)</div><div class="pbar" style="height:11px;margin:2px 0"><i style="background:' + C.red + ';width:' + clamp(r.unc * 100 / 0.45, 0, 100) + '%;transition:none"></i></div><b style="font-size:13px;color:' + C.red + '">' + r.unc.toFixed(2) + '</b></div></div></div>'; }).join('') + '<div class="note" style="font-size:12px;line-height:1.45;padding:2px 4px">The bars show R2: 1 means perfect predictions and 0 means no better than always guessing the average. Longer is better.</div>';
        qsa('[data-c]', cards).forEach(function(d){ d.addEventListener('click', function(){ sel = +d.dataset.c; render(); }); });
        qs('.sh2', sp).textContent = 'Predictions of option ' + (sel + 1); scatterCens(qs('canvas', sp), res[sel].F); qs('.lg', sp).innerHTML = '<span style="color:' + C.red + '">Red dot</span>: a measured compound. <b>Hollow dot</b>: a compound at the floor. Dashed line: perfect predictions. The closer the dots are to the dashed line, the better. The axes are the lab value (across) and the model\'s prediction (up).';
      }
      return { render: render };
    } };
}

/* ---------- one input, one weight, one bias: used in the first lesson and again in neural networks ---------- */
function lineStepDef(c){
  return { short: c.short, uses: ['endpoint', 'features'], title: c.title, watch: c.watch, change: c.change,
  build: function(api){
    var P1 = api.el('pn', 'left:28px;top:28px;width:620px;height:392px', '<h4 class="wh"></h4><canvas width="596" height="346"></canvas>');
    var P2 = api.el('pn', 'left:668px;top:28px;width:264px;height:392px', '<div class="wr" style="font-size:13px;line-height:1.45"></div>');
    var P3 = api.el('pn', 'left:28px;top:432px;width:904px;height:92px;display:flex;gap:26px;align-items:center', '<div style="width:190px"><div class="lbl" style="margin-bottom:4px">Input feature</div><select class="wf"></select></div><div style="width:230px"><label class="row">Weight<output class="wv"></output></label><input type="range" class="ww" step="0.01"></div><div style="width:230px"><label class="row">Bias<output class="bv"></output></label><input type="range" class="wb" step="0.01"></div><div style="display:flex;flex-direction:column;gap:6px"><button type="button" class="btn sm" data-best>Find the best weights</button><button type="button" class="btn sm ghost" data-reset>Reset</button></div>');
    var sel = qs('.wf', P3), ws = qs('.ww', P3), bs = qs('.wb', P3), S = { key: '', f: null, x: [], y: [], raw: [], mx: 0, sx: 1, my: 0, sy: 1, w: 0, b: 0, w0: 0, b0: 0, bestW: 0, bestB: 0, cmp: 0, step: -1 };
    function prep(force){
      var feats = activeFeats(); if (!feats.length) { S.f = null; return; } var f = sel.value && feats.indexOf(sel.value) >= 0 ? sel.value : feats[0], key = EP + '|' + f; if (!force && S.key === key) return;
      var idx = D.TRAIN, raw = idx.map(function(i){ return D.COL[f][i]; }), y = idx.map(function(i){ return D.Y[i]; }), mx = mean(raw), sx = Math.sqrt(mean(raw.map(function(v){ return (v - mx) * (v - mx); }))) || 1, my = mean(y), sy = Math.sqrt(mean(y.map(function(v){ return (v - my) * (v - my); }))) || 1;
      S.x = raw.map(function(v){ return (v - mx) / sx; }); S.y = y; S.raw = raw; S.mx = mx; S.sx = sx; S.my = my; S.sy = sy; S.f = f; S.key = key; var cov = 0; for (var i2 = 0; i2 < S.x.length; i2++) cov += S.x[i2] * (y[i2] - my); S.bestW = cov / S.x.length; S.bestB = my;
      S.w0 = 0; S.b0 = D.yhi - (D.yhi - D.ylo) * 0.3; S.w = S.w0; S.b = S.b0; S.step = -1; S.cmp = Math.floor(S.x.length * 0.37);
      var wm = Math.max(0.5, 2 * sy); ws.min = -wm; ws.max = wm; bs.min = D.ylo; bs.max = D.yhi; sel.innerHTML = feats.map(function(q){ return '<option' + (q === f ? ' selected' : '') + '>' + q + '</option>'; }).join('');
    }
    function mse(w, b){ var s = 0, n = S.x.length; for (var i = 0; i < n; i++) { var e = S.y[i] - (w * S.x[i] + b); s += e * e; } return s / n; }
    function sync(){ ws.value = S.w; bs.value = S.b; qs('.wv', P3).textContent = (S.w >= 0 ? '+' : '') + S.w.toFixed(2); qs('.bv', P3).textContent = S.b.toFixed(2); }
    function draw(){
      if (!S.f) { cv(qs('canvas', P1)); qs('.wr', P2).innerHTML = '<span class="note">Switch on at least one feature.</span>'; return; }
      var g = cv(qs('canvas', P1)), x = g.x, box = { l: 46, r: g.w - 10, t: 10, b: g.h - 46 }, X0 = -2.5, X1 = 2.5, lo = D.ylo, hi = D.yhi, sx = function(v){ return box.l + (v - X0) / (X1 - X0) * (box.r - box.l); }, sy = function(v){ return box.b - (v - lo) / (hi - lo) * (box.b - box.t); }, n = S.x.length;
      qs('.wh', P1).textContent = 'Training compounds, ' + S.f + ' against ' + unitWord() + (S.step >= 0 ? '. Step ' + S.step + ' of 90' : '');
      x.strokeStyle = C.silver; x.fillStyle = C.mid; x.textAlign = 'right'; var ytk = niceTicks(lo, hi, 5); ytk.v.forEach(function(v){ var yy = sy(v); x.beginPath(); x.moveTo(box.l, yy); x.lineTo(box.r, yy); x.stroke(); x.fillText(tickText(v, ytk.step), box.l - 6, yy + 4); });
      x.textAlign = 'center'; for (var z = -2; z <= 2; z++) { x.fillText(fmtFeat(S.f, S.mx + z * S.sx), sx(z), box.b + 15); if (z === 0) { x.strokeStyle = 'rgba(52,55,63,.18)'; x.beginPath(); x.moveTo(sx(0), box.t); x.lineTo(sx(0), box.b); x.stroke(); } }
      x.fillStyle = C.char; x.font = '600 12px Aptos, Segoe UI, sans-serif'; x.fillText(S.f + ' (real values; the middle is the average)', (box.l + box.r) / 2, box.b + 34); x.font = '12px Aptos, Segoe UI, sans-serif';
      var step = Math.ceil(n / 420); x.fillStyle = 'rgba(161,177,195,.55)'; for (var i = 0; i < n; i += step) { var px = sx(clamp(S.x[i], X0, X1)); x.beginPath(); x.arc(px, sy(clamp(S.y[i], lo, hi)), 2.6, 0, 6.3); x.fill(); }
      var rs = Math.ceil(n / 34); x.strokeStyle = 'rgba(52,55,63,.5)'; x.lineWidth = 1; for (var j = 0; j < n; j += rs) { var xv = clamp(S.x[j], X0, X1), pr = S.w * S.x[j] + S.b; x.beginPath(); x.moveTo(sx(xv), sy(clamp(S.y[j], lo, hi))); x.lineTo(sx(xv), sy(clamp(pr, lo, hi))); x.stroke(); }
      if (S.showBest) { x.setLineDash([6, 5]); x.strokeStyle = C.char; x.lineWidth = 1.6; x.beginPath(); x.moveTo(sx(X0), sy(clamp(S.bestW * X0 + S.bestB, lo, hi))); x.lineTo(sx(X1), sy(clamp(S.bestW * X1 + S.bestB, lo, hi))); x.stroke(); x.setLineDash([]); }
      x.strokeStyle = C.red; x.lineWidth = 3.4; x.lineCap = 'round'; x.beginPath(); x.moveTo(sx(X0), sy(clamp(S.w * X0 + S.b, lo - 5, hi + 5))); x.lineTo(sx(X1), sy(clamp(S.w * X1 + S.b, lo - 5, hi + 5))); x.stroke(); x.lineWidth = 1;
      var c = S.cmp, zc = S.x[c], pred = S.w * zc + S.b, meas = S.y[c], e = mse(S.w, S.b), eb = mse(S.bestW, S.bestB), aw = Math.abs(S.w), wsent = aw < 0.02 ? 'The weight is near zero, so this input has almost no effect on the prediction.' : 'When ' + S.f + ' rises by one standard deviation (' + fmtFeat(S.f, S.sx) + '), the prediction ' + (S.w > 0 ? 'rises' : 'falls') + ' by ' + aw.toFixed(2) + '. A bigger weight means a stronger push, and a negative weight pushes the other way.';
      x.fillStyle = C.red; x.beginPath(); x.arc(sx(clamp(zc, X0, X1)), sy(clamp(meas, lo, hi)), 5.5, 0, 6.3); x.fill(); x.strokeStyle = '#fff'; x.lineWidth = 2; x.stroke(); x.lineWidth = 1;
      qs('.wr', P2).innerHTML = '<div class="lbl">The model</div><div style="font-size:14px;font-weight:700;margin:3px 0 8px">prediction = ' + (S.w >= 0 ? '' : '-') + aw.toFixed(2) + ' times ' + S.f + ' + ' + S.b.toFixed(2) + '</div><p style="margin:0 0 7px"><b>Weight ' + (S.w >= 0 ? '+' : '') + S.w.toFixed(2) + '.</b> ' + wsent + '</p><p style="margin:0 0 7px"><b>Bias ' + S.b.toFixed(2) + '.</b> The prediction for an average compound.</p>' +
        '<div class="lbl" style="margin-top:8px">The red dot is one compound</div><p style="margin:3px 0 7px">' + S.f + ' ' + fmtFeat(S.f, S.raw[c]) + ' is ' + zc.toFixed(2) + ' standard deviations from average. The model predicts ' + S.w.toFixed(2) + ' times ' + zc.toFixed(2) + ' + ' + S.b.toFixed(2) + ' = <b>' + pred.toFixed(2) + '</b>. The lab measured <b>' + meas.toFixed(2) + '</b>.</p>' +
        '<div class="lbl">Average squared error</div><div class="big" style="color:' + C.red + ';font-size:30px">' + e.toFixed(3) + '</div><div class="note" style="font-size:12px">Lowest possible with one weight: ' + eb.toFixed(3) + '. Lower is better.</div><button type="button" class="btn sm ghost" data-other style="margin-top:8px">Another compound</button>';
      qs('[data-other]', P2).addEventListener('click', function(){ S.cmp = Math.floor(Math.random() * n); draw(); }); sync();
    }
    function onSlide(){ api.tok++; S.w = +ws.value; S.b = +bs.value; S.step = -1; draw(); }
    ws.addEventListener('input', onSlide); bs.addEventListener('input', onSlide);
    sel.addEventListener('change', function(){ api.tok++; S.showBest = false; prep(true); draw(); });
    qs('[data-reset]', P3).addEventListener('click', function(){ api.tok++; S.w = S.w0; S.b = S.b0; S.step = -1; S.showBest = false; draw(); });
    qs('[data-best]', P3).addEventListener('click', async function(){
      if (!S.f) return; api.tok++; var tok = api.tok, n = S.x.length; S.showBest = true; try { for (var it = 0; it <= 90; it++) { api.checkTok(tok); var gw = 0, gb = 0; for (var i = 0; i < n; i++) { var e = S.y[i] - (S.w * S.x[i] + S.b); gw += -2 * e * S.x[i]; gb += -2 * e; } S.w -= 0.12 * gw / n; S.b -= 0.12 * gb / n; S.step = it; draw(); await api.sleep(45); } } catch (e) { if (!(e instanceof Cancel)) console.error(e); }
    });
    function render(){ prep(false); draw(); }
    return { render: render };
  } };
}

/* ---------- seeds and ensembles: the same model run several times, then averaged ---------- */
function seedStepDef(kind){
  var nn = kind === 'nn', K = nn ? 6 : 8, cache = {};
  return { short: 'Seeds and ensembles', uses: nn ? ['endpoint', 'features', 'filters', 'nlayers', 'nunits', 'ndrop', 'nlr'] : ['endpoint', 'features', 'filters', 'tfdepth'],
    title: 'Seeds and ensembles',
    watch: 'Training uses random choices, such as ' + (nn ? 'the starting weights' : 'which compounds each tree sees') + '. The seed is the number that starts them. Change the seed and the same model gives a slightly different score. An ensemble averages the predictions of several runs of the same model, which smooths out that luck. ' + (nn ? 'Each run is a network with your options.' : 'Each run here is a small random forest of 15 trees.'),
    change: 'Press Run seeds below the plot, then drag Runs averaged.',
    build: function(api){
      var cp = api.el('pn', 'left:28px;top:28px;width:560px;height:408px', '<h4>Test R2 of each run, and of the average of the first runs</h4><canvas width="536" height="330"></canvas><div class="note" style="font-size:12px;margin-top:4px">Grey bars: single runs. Red line: average of the first runs. Dashed: mean of the single runs.</div>');
      var sp = api.el('pn', 'left:608px;top:28px;width:324px;height:408px;overflow:auto', '<div class="sd" style="font-size:12.5px;line-height:1.4"></div><h4 style="margin-top:8px">Predictions of the average</h4><canvas width="300" height="160"></canvas>');
      var ctl = api.addControl('<button type="button" class="btn sm" data-run>Run seeds</button> Runs averaged <input type="range" min="1" max="' + K + '" step="1" value="' + K + '" style="width:200px"><b class="kv"></b>'), rs = qs('input', ctl), busy = false;
      function key(){ var p = nn ? NP() : TP(); return JSON.stringify([kind, EP, activeFeats(), filtKey(), nn ? [p.layers, p.units, p.drop, p.lr] : p.fd]); }
      function show(res){
        var k = +rs.value, runs = res.map(function(r){ return r.r2; }), ens = [], acc = new Float64Array(D.N); res.forEach(function(r, i){ for (var j = 0; j < D.N; j++) acc[j] += r.pred[j]; var m = new Float64Array(D.N); for (var q = 0; q < D.N; q++) m[q] = acc[q] / (i + 1); ens.push({ r2: r2(D.TEST, m), pred: m }); });
        var g = cv(qs('canvas', cp)), x = g.x, box = { l: 44, r: g.w - 12, t: 16, b: g.h - 40 }, lo = Math.min(0, Math.min.apply(null, runs.concat(ens.map(function(e){ return e.r2; })))), hi = Math.max(0.3, Math.max.apply(null, runs.concat(ens.map(function(e){ return e.r2; })))) * 1.12, bw = (box.r - box.l) / K, sy = function(v){ return box.b - (v - lo) / (hi - lo) * (box.b - box.t); };
        x.strokeStyle = C.silver; x.fillStyle = C.mid; x.textAlign = 'right'; x.font = '11px Aptos, Segoe UI, sans-serif'; for (var t = 0; t <= 4; t++) { var y = box.b - t / 4 * (box.b - box.t); x.beginPath(); x.moveTo(box.l, y); x.lineTo(box.r, y); x.stroke(); x.fillText((lo + (hi - lo) * t / 4).toFixed(2), box.l - 5, y + 4); }
        res.forEach(function(r, i){ var h0 = sy(Math.max(lo, 0)), y0 = sy(r.r2); x.fillStyle = i < k ? C.bg : 'rgba(209,216,224,.5)'; x.fillRect(box.l + i * bw + bw * 0.2, Math.min(y0, h0), bw * 0.6, Math.abs(h0 - y0)); x.fillStyle = C.char; x.textAlign = 'center'; x.font = '600 11px Aptos, Segoe UI, sans-serif'; x.fillText(r.r2.toFixed(2), box.l + i * bw + bw / 2, y0 - 5); x.fillText('run ' + (i + 1), box.l + i * bw + bw / 2, box.b + 16); });
        x.setLineDash([5, 4]); x.strokeStyle = C.char; x.beginPath(); x.moveTo(box.l, sy(mean(runs))); x.lineTo(box.r, sy(mean(runs))); x.stroke(); x.setLineDash([]);
        x.strokeStyle = C.red; x.lineWidth = 3; x.beginPath(); for (var i2 = 0; i2 < k; i2++) { var X = box.l + i2 * bw + bw / 2, Y = sy(ens[i2].r2); i2 ? x.lineTo(X, Y) : x.moveTo(X, Y); } x.stroke(); x.fillStyle = C.red; for (var i3 = 0; i3 < k; i3++) { x.beginPath(); x.arc(box.l + i3 * bw + bw / 2, sy(ens[i3].r2), 4.5, 0, 6.3); x.fill(); } x.lineWidth = 1;
        scatterPV(qs('canvas', sp), ens[k - 1].pred, D.TEST, 'average of ' + k + ' run' + (k > 1 ? 's' : '') + ', R2 ' + ens[k - 1].r2.toFixed(2), { max: 260 }); qs('.kv', ctl).textContent = k;
        var best = Math.max.apply(null, runs), worst = Math.min.apply(null, runs), e = ens[K - 1].r2;
        qs('.sd', sp).innerHTML = '<div class="lbl">Single runs</div><div class="nums" style="margin:3px 0 7px"><div><span>Best</span><b>' + best.toFixed(2) + '</b></div><div><span>Worst</span><b>' + worst.toFixed(2) + '</b></div><div><span>Mean</span><b>' + mean(runs).toFixed(2) + '</b></div></div><div class="lbl" style="color:' + C.red + '">Average of all ' + K + '</div><div class="big" style="font-size:24px;color:' + C.red + '">' + e.toFixed(2) + '</div><p style="margin:3px 0 0">' + (e > mean(runs) + 0.003 ? 'Better than the typical single run by ' + (e - mean(runs)).toFixed(2) + '.' : 'About the same as a typical single run here.') + '</p>';
      }
      async function run(){
        if (busy) return; if (!activeFeats().length) return; var kk = key(); if (cache[kk]) { show(cache[kk]); return; } busy = true; qs('[data-run]', ctl).disabled = true;
        try { var res = [], feats = activeFeats(), idx = activeTrain();
          for (var s = 1; s <= K; s++) { api.busy('Run ' + s + ' of ' + K + ' with seed ' + s); await tick(); await tick(); var pred; if (nn) { var o = await nnTrain({ params: { epochs: Math.min(NP().epochs, 30) }, seed: s }); pred = o.pred; } else { var fr = fitForest(feats, idx, 15, TP().fd, 100 + s); pred = fr.pred; } res.push({ r2: r2(D.TEST, pred), pred: pred }); }
          cache[kk] = res; api.idle(); show(res);
        } catch (e) { console.error(e); api.idle(); qs('.sd', sp).innerHTML = '<span style="color:' + C.red + '">' + esc(e.message) + '</span>'; }
        busy = false; qs('[data-run]', ctl).disabled = false;
      }
      qs('[data-run]', ctl).addEventListener('click', run); rs.addEventListener('input', function(){ if (cache[key()]) show(cache[key()]); else qs('.kv', ctl).textContent = rs.value; });
      function render(){ var c = cache[key()]; if (c) show(c); else { cv(qs('canvas', cp)); cv(qs('canvas', sp)); qs('.sd', sp).innerHTML = '<span class="note">Press Run seeds to run the same model ' + K + ' times with different seeds.</span>'; qs('.kv', ctl).textContent = rs.value; } }
      return { render: render };
    } };
}
