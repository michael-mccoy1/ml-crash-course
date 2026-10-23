"use strict";
/* Lesson 1: Data and ML basics */
var LOGD = { promise: null, ready: false, model: null, r2: null, feats: null, B: null };
function ensureLogdPred(){
  if (LOGD.promise) return LOGD.promise;
  LOGD.promise = (async function(){
    var az = await loadDataset('logD'); if (!az.ylo) setRange(az); var feats = az.BASE.slice();
    var ev = fitBoost(feats, az.TRAIN, 150, 0.1, 4, 0, az); LOGD.r2 = r2(az.TEST, ev.F, az.Y); LOGD.rmse = rmse(az.TEST, ev.F, az.Y); await tick();
    var all = az.TRAIN.concat(az.TEST), full = fitBoost(feats, all, 150, 0.1, 4, 0, az); LOGD.model = full; LOGD.feats = feats; LOGD.B = az.B; LOGD.secs = full.secs; await tick();
    var h = DATA.HLM || await loadDataset('HLM'), vals = []; for (var i = 0; i < h.N; i++) { var v = {}; feats.forEach(function(f){ v[f] = h.COL[f][i]; }); vals.push(boostPredictVals(full, az.B, v)); }
    addFeature(h, 'logD_pred', vals); LOGD.ready = true; if (D === h) renderChips();
    if (LESSONS.start && LESSONS.start.built) LESSONS.start.refresh();
  })();
  return LOGD.promise;
}

var L0 = new Lesson('start', 'Data and ML basics');
L0.optsHTML = function(){ return sharedOptions() + optGroup('Data', '<div id="dsinfo" class="hint"></div>') + filterGroup(); };
L0.onData = function(){ updateInfo(); };
L0.onOpts = function(){ updateInfo(); };
function updateInfo(){ var e = $('dsinfo'); if (!e || !D) return; var c = D.cens ? D.cens.reduce(function(a, b){ return a + b; }, 0) : 0; e.innerHTML = esc(D.full) + '.<br>' + D.N + ' compounds: ' + D.TRAIN.length + ' to study, ' + D.TEST.length + ' held out by scaffold.' + (c ? '<br>' + pct(c, D.N) + '% of values sit at the assay floor.' : ''); }
function unitWord(){ return D.key === 'HLM' ? 'HLM clearance (log10)' : 'logD'; }

L0.add(lineStepDef({ short: 'What learning means', title: 'A model is a rule fitted to examples',
  watch: 'Each dot is a compound. Across is logP, how oily it is. Up is its clearance, how fast the liver breaks it down. The red line is a rule that predicts clearance from logP. Weight is how steep the line is. Bias is where it sits.',
  change: 'Drag Weight and Bias below the plot, then press Find the best weights.' }));

/* ---------- Step 1: structure to prediction ---------- */
L0.add({ short: 'Structure to prediction', title: 'From structure to prediction',
  change: 'Press Next compound. Untick a feature in the left panel and watch the prediction change.',
  watch: 'A model cannot read a drawing. The molecule becomes text (SMILES), then numbers (features), then a prediction.',
  build: function(api){
    var cur = null;
    var lib = api.el('pn fd', 'left:28px;top:28px;width:300px;height:158px', '<h4>Compound library</h4><div class="lg" style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px"></div>');
    var big = api.el('pn fd', 'left:28px;top:200px;width:300px;height:236px', '<h4>Sample compound</h4><div class="struct" style="height:196px"></div>');
    var smi = api.el('pn fd', 'left:348px;top:28px;width:584px;height:64px', '<h4>SMILES: the same structure as text</h4><div class="smi" style="min-height:20px"></div>');
    var feat = api.el('pn fd', 'left:348px;top:104px;width:276px;height:332px', '<h4>Features from RDKit</h4><div class="fr"></div><div style="font-size:11px;color:#6B7280;margin-top:6px">Bar: position among all compounds. Grey rows are switched off.</div>');
    var mod = api.el('pn fd', 'left:642px;top:104px;width:290px;height:150px', '<h4>Model</h4><div style="display:flex;gap:14px;align-items:center"><div>' + miniTreeSVG(110, 100) + '</div><div class="mt" style="font-size:12px;color:#4a4f5a"></div></div>');
    var pred = api.el('pn fd', 'left:642px;top:266px;width:290px;height:170px', '<h4 class="ph">Prediction</h4><div class="ps"></div>');
    api.addControl('<button type="button" class="btn sm ghost" data-next>Next compound</button>');
    var panels = [lib, big, smi, feat, mod, pred];
    function pickCmp(){ var arr = STRUCTS[EP]; return arr[Math.floor(Math.random() * arr.length)]; }
    function fill(){
      var e = cur, i = e.i, others = []; STRUCTS[EP].forEach(function(x){ if (x.i !== i) others.push(x); }); var r = rng(i + 3); others.sort(function(){ return r() - 0.5; }); var seven = others.slice(0, 7); seven.splice(3, 0, e);
      qs('.lg', lib).innerHTML = seven.map(function(x, k){ return '<div class="struct fl" data-k="' + k + '" style="height:52px;background:#fff;border:1.5px solid ' + (x === e ? C.red : C.silver) + ';border-radius:5px">' + x.svg + '</div>'; }).join('');
      qs('.struct', big).innerHTML = e.svg; qs('.smi', smi).textContent = '';
      var pt = demoTree(), pv = pt ? predictTree(pt, D.B, i) : NaN, mv = D.Y[i], act = activeFeats();
      qs('.fr', feat).innerHTML = D.BASE.map(function(f){ var v = D.COL[f][i], on = FSEL[f] ? 1 : 0; return '<div class="fq" data-on="' + on + '" style="display:grid;grid-template-columns:62px 1fr 48px;gap:8px;align-items:center;margin:6px 0;opacity:0;transition:opacity .4s"><span class="lbl" style="text-transform:none;letter-spacing:0;font-size:12px;color:#34373F">' + f + '</span><div class="pbar" style="margin:0"><i style="background:' + (on ? C.red : C.bg) + ';width:0" data-w="' + Math.round(percentile(f, v) * 100) + '"></i></div><b style="text-align:right" data-v="' + v + '" data-f="' + f + '">0</b></div>'; }).join('');
      qs('.mt', mod).innerHTML = pt ? 'Decision tree of depth 5 fitted on ' + D.TRAIN.length + ' compounds using ' + act.length + ' feature' + (act.length > 1 ? 's' : '') + '.<br>This compound was held back.' : '<span style="color:' + C.red + '">No feature is switched on, so there is nothing to fit.</span>';
      qs('.ph', pred).textContent = 'Prediction: ' + unitWord(); var span = D.yhi - D.ylo;
      qs('.ps', pred).innerHTML = '<div style="font-size:12px;font-weight:600">Predicted by the model <span class="pv" style="float:right;color:' + C.red + '">0.00</span></div><div class="pbar"><i style="background:' + C.red + ';width:0" data-w="' + (isNaN(pv) ? 0 : clamp((pv - D.ylo) / span * 100, 0, 100)) + '"></i></div><div style="font-size:12px;font-weight:600">Measured in the lab <span class="mv" style="float:right">0.00</span></div><div class="pbar"><i style="background:' + C.char + ';width:0" data-w="' + clamp((mv - D.ylo) / span * 100, 0, 100) + '"></i></div><div class="er" style="font-size:12px;color:#6B7280"></div>';
      qs('.ps', pred).dataset.pv = isNaN(pv) ? '' : pv; qs('.ps', pred).dataset.mv = mv; panels.forEach(function(p){ p.classList.remove('on'); }); api.clearTexts();
    }
    function showFinal(){
      panels.forEach(function(p){ p.classList.add('on'); }); qsa('.fl', lib).forEach(function(e){ e.classList.add('on'); }); qs('.smi', smi).textContent = cur.smi;
      qsa('.fq', feat).forEach(function(r){ r.style.opacity = r.dataset.on === '1' ? 1 : 0.35; var b = qs('b', r), v = +b.dataset.v, f = b.dataset.f; b.textContent = fmtFeat(f, v); var i = qs('i', r); i.style.transition = 'none'; i.style.width = i.dataset.w + '%'; });
      var ps = qs('.ps', pred), pv = ps.dataset.pv === '' ? NaN : +ps.dataset.pv, mv = +ps.dataset.mv; qsa('.pbar i', ps).forEach(function(i){ i.style.transition = 'none'; i.style.width = i.dataset.w + '%'; }); qs('.pv', ps).textContent = isNaN(pv) ? 'n/a' : pv.toFixed(2); qs('.mv', ps).textContent = mv.toFixed(2); qs('.er', ps).textContent = isNaN(pv) ? '' : 'Error ' + (pv - mv >= 0 ? '+' : '') + (pv - mv).toFixed(2);
    }
    async function render(){ await loadStructs(); if (!cur || cur._ep !== EP) { cur = pickCmp(); cur._ep = EP; } fill(); showFinal(); }
    qs('[data-next]', api.sbar).addEventListener('click', async function(){ await loadStructs(); cur = pickCmp(); cur._ep = EP; fill(); api.tok++; var t = api.tok; api.stepDef.safe(function(){ return play(t); }); });
    async function play(tok){
      if (!cur) await render(); fill(); await api.sleep(150);
      lib.classList.add('on'); qsa('.fl', lib).forEach(function(e, k){ setTimeout(function(){ e.classList.add('on'); }, ANIM ? k * 140 : 0); }); await api.beat('We start from a library of structures. Each one is a compound a chemist could make.', 2600);
      big.classList.add('on'); await api.beat('Pick one compound and look at its structure.', 2000);
      smi.classList.add('on'); api.cap('The same structure as one line of text. Letters are atoms, symbols mark bonds and branches, digits close rings.'); await typeInto(qs('.smi', smi), cur.smi, api, 70); await api.sleep(1400);
      feat.classList.add('on'); api.cap('RDKit turns the structure into numbers. Each number is one feature the model can read.'); var rows = qsa('.fq', feat);
      for (var k = 0; k < rows.length; k++) { api.checkTok(tok); rows[k].style.opacity = rows[k].dataset.on === '1' ? 1 : 0.35; var b = qs('b', rows[k]), v = +b.dataset.v, f = b.dataset.f, bar = qs('i', rows[k]); bar.style.width = bar.dataset.w + '%'; (function(b, v, f){ animateFree(500, function(t){ b.textContent = (f === 'logP' || f === 'Fsp3') ? (v * t).toFixed(2) : (v * t).toFixed(0); }); })(b, v, f); await api.sleep(260); }
      await api.sleep(1200); mod.classList.add('on'); await api.beat('A model has already learned from many compounds with known answers.', 2400);
      pred.classList.add('on'); api.cap('It predicts a value. The lab measured the same compound, so the two can be compared.'); await api.sleep(500);
      var ps = qs('.ps', pred); qsa('.pbar i', ps).forEach(function(i){ i.style.width = i.dataset.w + '%'; }); if (ps.dataset.pv !== '') countUp(qs('.pv', ps), +ps.dataset.pv, 2); countUp(qs('.mv', ps), +ps.dataset.mv, 2); await api.sleep(2600); showFinal();
    }
    return { render: render, play: play };
  } });

/* ---------- Step 2: descriptors come from the structure ---------- */
L0.add({ short: 'Where features come from', title: 'Features are sums over the atoms RDKit sees',
  change: 'Press Next compound.',
  watch: 'A feature is a number computed from a molecule. Each atom adds a little to logP (how oily) and to TPSA (how polar). The totals are the features.',
  build: function(api){
    var cur = null, mode = 0;
    var box = api.el('pn', 'left:28px;top:28px;width:430px;height:408px', '<h4 class="t">Structure</h4><div class="ly" style="position:relative;height:356px"></div>');
    var side = api.el('pn', 'left:478px;top:28px;width:454px;height:408px', '<div class="sd"></div>');
    api.addControl('<button type="button" class="btn sm ghost" data-next>Next compound</button>');
    function layers(){ qs('.ly', box).innerHTML = ['svg', 'logp', 'tpsa'].map(function(k, n){ return '<div class="struct" data-l="' + k + '" style="position:absolute;inset:0;opacity:' + (n === 0 ? 1 : 0) + ';transition:opacity .7s">' + cur[k] + '</div>'; }).join(''); }
    function show(m){ mode = m; qsa('.struct', box).forEach(function(e){ e.style.opacity = e.dataset.l === ['svg', 'logp', 'tpsa'][m] ? 1 : 0; }); qs('.t', box).textContent = ['Structure', 'Atom contributions to logP', 'Atoms that count toward TPSA'][m]; side_(m); }
    function side_(m){
      var i = cur.i, v = function(f){ return D.COL[f] ? fmtFeat(f, D.COL[f][i]) : ''; }, lg = '<div style="height:10px;border-radius:5px;background:linear-gradient(90deg,#A1B1C3,#F6F8F9,#E1242A);margin:8px 0 2px"></div><div style="display:flex;justify-content:space-between;font-size:11px;color:#6B7280"><span>lowers</span><span>neutral</span><span>raises</span></div>';
      var t = [
        '<h4>Features for this compound</h4><div class="nums" style="font-size:13px">' + D.BASE.map(function(f){ return '<div><span>' + f + '</span><b>' + v(f) + '</b></div>'; }).join('') + '</div>',
        '<h4>logP (lipophilicity)</h4><div class="big" style="color:' + C.red + '">' + v('logP') + '</div><p style="font-size:13px;margin-top:6px">RDKit gives every atom a contribution based on its type and neighbours. Red atoms raise logP and blue grey atoms lower it. The compound logP is the sum over all atoms.</p>' + lg,
        '<h4>TPSA (polar surface area)</h4><div class="big" style="color:' + C.red + '">' + v('TPSA') + '</div><p style="font-size:13px;margin-top:6px">Only polar atoms count: nitrogen and oxygen, with their attached hydrogens. Red atoms add to the surface area. Carbon atoms add nothing.</p><div style="margin-top:10px" class="nums"><div><span>HBD (donors)</span><b>' + v('HBD') + '</b></div><div><span>HBA (acceptors)</span><b>' + v('HBA') + '</b></div></div>'][m];
      qs('.sd', side).innerHTML = t; }
    async function pickCmp(){ await loadStructs(); var arr = (STRUCTS[EP] || []).filter(function(e){ return e.logp; }); if (!arr.length) arr = (STRUCTS.HLM || []).filter(function(e){ return e.logp; }); cur = arr[Math.floor(Math.random() * arr.length)]; layers(); show(0); }
    async function render(){ await pickCmp(); }
    qs('[data-next]', api.sbar).addEventListener('click', async function(){ api.tok++; await pickCmp(); var t = api.tok; api.stepDef.safe(function(){ return play(t); }); });
    async function play(tok){ if (!cur) await pickCmp(); show(0); await api.beat('RDKit reads the structure: atoms, bonds and rings.', 2200); show(1); await api.beat('For logP, each atom is given a contribution. Red raises logP, blue grey lowers it.', 3200); show(2); await api.beat('For TPSA, only polar nitrogen and oxygen atoms count.', 3000); show(0); }
    return { render: render, play: play };
  } });

/* ---------- Step 3: the unseen set ---------- */
L0.add({ short: 'The unseen set', title: 'Hold compounds back to test a model honestly',
  change: 'Switch between random and scaffold split below the picture.',
  watch: 'Some compounds are held back to test the model: the unseen set. A random split leaves close relatives of them in training. A scaffold split removes whole families.',
  build: function(api){
    var mode = 'rand', a = 0;
    var mapP = api.el('pn', 'left:28px;top:28px;width:560px;height:408px', '<h4>Chemical space map of every compound (RDKit fingerprints, projected to 2D)</h4><canvas width="540" height="352" style="margin-top:4px"></canvas>');
    var cnt = api.el('pn', 'left:608px;top:28px;width:324px;height:112px', '<h4>Unseen compounds with a seen scaffold</h4><div class="big" style="color:' + C.red + '"><span class="sv">0</span>%</div><div class="note" style="font-size:12px;color:#6B7280">A scaffold is the ring skeleton of a compound.</div>');
    var hist = api.el('pn', 'left:608px;top:152px;width:324px;height:196px', '<h4>Similarity to the closest training compound</h4><canvas width="300" height="140"></canvas>');
    var r2p = api.el('pn', 'left:608px;top:360px;width:324px;height:76px', '<h4>Quick check, boosted trees</h4><div class="rq" style="font-size:13px">Computing</div>');
    var sel = api.addControl('<select><option value="rand">Random split</option><option value="scaf">Scaffold split</option></select>'); var selEl = qs('select', sel);
    function stats(){ var s = D.sim, seen = D.seen; return { rand: { seen: seen.random, h: s.random.hist, m: s.random.mean }, scaf: { seen: seen.scaffold, h: s.scaffold.hist, m: s.scaffold.mean } }; }
    function drawMap(blend){
      var g = cv(qs('canvas', mapP)), x = g.x, W = g.w, H = g.h, sx = function(v){ return 14 + (v + 1) / 2 * (W - 28); }, sy = function(v){ return H - 14 - (v + 1) / 2 * (H - 28); };
      for (var i = 0; i < D.N; i++) { var ur = D.rtest[i], us = D.test[i], u = blend === null ? 0 : (mode === 'rand' ? lerp(blend < 0 ? 0 : ur, ur, 1) : lerp(ur, us, 1)); if (blend === null) u = 0; else u = mode === 'rand' ? ur : lerp(ur, us, clamp(a, 0, 1)); x.fillStyle = u > 0.5 ? 'rgba(225,36,42,.85)' : (blend === null ? 'rgba(161,177,195,.55)' : 'rgba(161,177,195,.6)'); x.beginPath(); x.arc(sx(D.mx[i]), sy(D.my[i]), u > 0.5 ? 2.8 : 2.1, 0, 6.3); x.fill(); }
      x.fillStyle = C.char; x.font = '600 12px Aptos, Segoe UI, sans-serif'; x.textAlign = 'left'; x.fillText('red: unseen compounds    grey blue: compounds the model studies', 14, 16); }
    function drawHist(blend){
      var s = stats(), cur = s[mode], oth = s[mode === 'rand' ? 'scaf' : 'rand'], g = cv(qs('canvas', hist)), x = g.x, W = g.w, H = g.h, bw = (W - 20) / 20, mx = Math.max.apply(null, cur.h.concat(oth.h)) || 1;
      for (var i = 0; i < 20; i++) { var hh = cur.h[i] / mx * (H - 56) * blend; x.fillStyle = C.red; x.fillRect(10 + i * bw + 1, H - 22 - hh, bw - 2, hh); var ho = oth.h[i] / mx * (H - 56); x.strokeStyle = C.mid; x.lineWidth = 1; x.strokeRect(10 + i * bw + 1.5, H - 22 - ho, bw - 3, ho); }
      x.fillStyle = C.mid; x.font = '11px Aptos, Segoe UI, sans-serif'; x.textAlign = 'center'; [0, 0.25, 0.5, 0.75, 1].forEach(function(v){ x.fillText(v.toFixed(2), 10 + v * (W - 20), H - 8); }); x.textAlign = 'left'; x.fillStyle = C.char; x.fillText('mean ' + cur.m.toFixed(2) + '   (outline: the other split, ' + oth.m.toFixed(2) + ')', 12, 12); }
    function counters(){ qs('.sv', cnt).textContent = Math.round(stats()[mode].seen * 100); }
    function quick(){ var feats = activeFeats(); if (!feats.length) { qs('.rq', r2p).textContent = 'Switch on at least one feature'; return; } var key = 'splitR2' + feats.join(','); if (!D.cache[key]) { var rtr = [], rte = []; D.rtest.forEach(function(t, i){ (t ? rte : rtr).push(i); }); var fr = fitBoost(feats, rtr, 60, 0.1, 3), fs = fitBoost(feats, D.TRAIN, 60, 0.1, 3); D.cache[key] = { rand: r2(rte, fr.F), scaf: r2(D.TEST, fs.F) }; } var q = D.cache[key]; qs('.rq', r2p).innerHTML = 'Random split R2 <b>' + q.rand.toFixed(2) + '</b>, scaffold split R2 <b>' + q.scaf.toFixed(2) + '</b>'; }
    function set(m, instant){ mode = m; selEl.value = m; a = m === 'scaf' ? 1 : 0; drawMap(0); drawHist(1); counters(); }
    selEl.addEventListener('change', function(){ api.tok++; set(selEl.value); api.clearTexts(); });
    function render(){ mode = selEl.value; set(mode); setTimeout(quick, 30); }
    async function play(tok){
      mode = 'rand'; selEl.value = 'rand'; a = 0; drawMap(null); drawHist(0); qs('.sv', cnt).textContent = '0';
      await api.beat('Every dot is a compound, placed by how similar its structure is to the others.', 2600);
      drawMap(0); drawHist(1); counters(); await api.beat('A random split picks unseen compounds anywhere. Many have close relatives in the training set.', 3200);
      mode = 'scaf'; selEl.value = 'scaf'; await api.anim(1600, function(t){ a = t; drawMap(0); drawHist(0.4 + 0.6 * t); qs('.sv', cnt).textContent = Math.round(lerp(stats().rand.seen, stats().scaf.seen, t) * 100); });
      api.cap('A scaffold split holds out whole ring skeletons, so no unseen compound shares one with the training set.'); await api.sleep(3400);
      var s = stats(); var gap = D.key === 'HLM' ? 'In this Biogen set the compounds come from many vendors, so the two splits give similar scores.' : 'In this AstraZeneca set a random split leaks a lot, so a scaffold split is much stricter.';
      api.cap(gap); await api.sleep(3400);
    }
    return { render: render, play: play };
  } });

L0.add(targetStepDef());
L0.add(censorStepDef());
/* ---------- Step 4: underfitting to overfitting ---------- */
L0.add({ short: 'Underfitting and overfitting', title: 'A flexible model can memorise noise',
  change: 'Choose a feature and drag Tree depth below the plot.',
  watch: 'A decision tree is a flowchart of yes or no questions. Depth is how many it may ask. Too few misses the trend. Too many fits the noise.',
  build: function(api){
    var G_ = 200, cur = null, lab = null, depthEl, featSel;
    var plot = api.el('pn', 'left:28px;top:28px;width:620px;height:408px', '<h4 class="pt">Measured value against the feature, with a decision tree fit</h4><canvas width="600" height="352" style="margin-top:2px"></canvas>');
    var side = api.el('pn', 'left:668px;top:28px;width:264px;height:408px', '<h4>Error (RMSE)</h4><div class="lb2"></div>');
    var ctl = api.addControl('Feature <select></select> Depth <input type="range" min="0" max="12" value="3" style="width:180px"><b class="dv">3</b>'); featSel = qs('select', ctl); depthEl = qs('input', ctl);
    function feats(){ var f = activeFeats(); return f.length ? f : D.BASE.slice(); }
    function fillSel(){ var f = feats(), c = featSel.value; featSel.innerHTML = f.map(function(x){ return '<option>' + x + '</option>'; }).join(''); if (f.indexOf(c) >= 0) featSel.value = c; }
    function prep(){
      var feat = featSel.value; D.cache.lab = D.cache.lab || {}; if (D.cache.lab[feat]) { lab = D.cache.lab[feat]; return; }
      var B = fineBins(feat, 200), curves = [], trE = [], teE = [], xs = D.COL[feat], sorted = sortedCol(feat), x0 = sorted[Math.floor(sorted.length * 0.01)], x1 = sorted[Math.floor(sorted.length * 0.99)];
      for (var d = 0; d <= 12; d++) { var t = buildTree(B, [feat], D.Y, Int32Array.from(D.TRAIN), d, 2, 1, rng(1)), cu = new Float32Array(G_), vals = {}; for (var k = 0; k < G_; k++) { vals[feat] = lerp(x0, x1, k / (G_ - 1)); cu[k] = predictVals(t, B, vals); } curves.push(cu);
        var s1 = 0, s2 = 0; D.TRAIN.forEach(function(i){ var e = D.Y[i] - predictTree(t, B, i); s1 += e * e; }); D.TEST.forEach(function(i){ var e = D.Y[i] - predictTree(t, B, i); s2 += e * e; }); trE.push(Math.sqrt(s1 / D.TRAIN.length)); teE.push(Math.sqrt(s2 / D.TEST.length)); }
      var rnd = rng(9), shT = [], shE = []; D.TRAIN.forEach(function(i){ if (rnd() < 0.18) shT.push(i); }); D.TEST.forEach(function(i){ if (rnd() < 0.3) shE.push(i); });
      lab = D.cache.lab[feat] = { feat: feat, curves: curves, trE: trE, teE: teE, x0: x0, x1: x1, shT: shT, shE: shE, best: teE.indexOf(Math.min.apply(null, teE)) };
    }
    function draw(){
      var g = cv(qs('canvas', plot)), x = g.x, box = { l: 50, r: g.w - 12, t: 12, b: g.h - 42 }, yl = D.ylo, yh = D.yhi; axes(g, box, [lab.x0, lab.x1], [yl, yh], lab.feat, unitWord(), 5);
      var f = lab.feat, sx = function(v){ return box.l + (clamp(v, lab.x0, lab.x1) - lab.x0) / (lab.x1 - lab.x0) * (box.r - box.l); }, sy = function(v){ return box.b - (clamp(v, yl, yh) - yl) / (yh - yl) * (box.b - box.t); };
      x.fillStyle = 'rgba(161,177,195,.7)'; lab.shT.forEach(function(i){ x.beginPath(); x.arc(sx(D.COL[f][i]), sy(D.Y[i]), 3, 0, 6.3); x.fill(); }); x.strokeStyle = C.char; x.lineWidth = 1.4; lab.shE.forEach(function(i){ x.beginPath(); x.arc(sx(D.COL[f][i]), sy(D.Y[i]), 3.3, 0, 6.3); x.stroke(); });
      x.strokeStyle = C.red; x.lineWidth = 3.4; x.lineJoin = 'round'; x.beginPath(); for (var k = 0; k < G_; k++) { var px = box.l + k / (G_ - 1) * (box.r - box.l); k ? x.lineTo(px, sy(cur[k])) : x.moveTo(px, sy(cur[k])); } x.stroke();
      x.font = '11px Aptos, Segoe UI, sans-serif'; x.fillStyle = C.char; x.textAlign = 'left'; x.fillText('filled: studied compounds    hollow: unseen compounds', box.l + 6, box.t + 12); qs('.pt', plot).textContent = 'Measured value against ' + f + ', with a decision tree fit'; }
    function info(d){
      var tag = d < lab.best - 1 ? ['Underfitting', C.mid, 'Too simple. It misses the trend and both errors are high.'] : (d <= lab.best + 1 ? ['Good fit', C.char, 'It follows the trend and not the noise. The unseen error is near its lowest.'] : ['Overfitting', C.red, 'It memorises noise. The studied error falls while the unseen error rises.']), mx = Math.max.apply(null, lab.teE.concat(lab.trE));
      qs('.lb2', side).innerHTML = '<span style="display:inline-block;border:1px solid ' + tag[1] + ';color:' + tag[1] + ';border-radius:999px;font-size:12px;font-weight:700;padding:2px 12px">' + tag[0] + '</span><div style="margin-top:14px;font-weight:600">Studied compounds <span style="float:right">' + lab.trE[d].toFixed(2) + '</span></div><div class="pbar"><i style="background:' + C.char + ';width:' + (lab.trE[d] / mx * 100) + '%"></i></div><div style="font-weight:600">Unseen compounds <span style="float:right">' + lab.teE[d].toFixed(2) + '</span></div><div class="pbar"><i style="background:' + C.red + ';width:' + (lab.teE[d] / mx * 100) + '%"></i></div><p style="font-size:12.5px;margin-top:10px;color:#4a4f5a">' + tag[2] + '</p><p style="font-size:12px;color:#6B7280">Depth ' + d + ' means each compound can be asked up to ' + d + ' questions in a row. Best unseen error at depth ' + lab.best + '.</p>'; }
    async function go(d){ var from = new Float32Array(cur), to = lab.curves[d]; qs('.dv', ctl).textContent = d; depthEl.value = d; info(d); await api.anim(600, function(t){ for (var k = 0; k < G_; k++) cur[k] = lerp(from[k], to[k], t); draw(); }); }
    function setDepth(d){ qs('.dv', ctl).textContent = d; depthEl.value = d; cur = new Float32Array(lab.curves[d]); info(d); draw(); }
    depthEl.addEventListener('input', function(){ api.tok++; api.clearTexts(); setDepth(+depthEl.value); });
    featSel.addEventListener('change', function(){ api.tok++; api.clearTexts(); prep(); setDepth(+depthEl.value); });
    function render(){ fillSel(); prep(); setDepth(+depthEl.value); }
    async function play(tok){
      fillSel(); prep(); setDepth(0); await api.beat('Depth 0 predicts the same average for every compound.', 2400);
      for (var d = 1; d <= 12; d++) { await go(d); if (d === 1) api.cap('One question splits the compounds in two groups with two averages.'); if (d === 3) api.cap('More questions follow the trend more closely.'); if (d === Math.min(12, lab.best + 1)) api.cap('Around here the unseen error is lowest. This is the right amount of flexibility.'); if (d === 9) api.cap('Now the line chases single dots, including noise.'); await api.sleep(d < 4 ? 900 : 650); }
      await api.sleep(1200);
    }
    return { render: render, play: play };
  } });

/* ---------- Step 5: how a model is scored ---------- */
L0.add({ short: 'Scoring a model', title: 'R2, Spearman and RMSE measure different things',
  change: 'Drag Model quality below the plot.',
  watch: 'Dots closer to the diagonal are better predictions. R2 and RMSE measure the size of the errors. Spearman measures only the order.',
  build: function(api){
    var q = 100, pv = null, ys = null, idx = null, pm = 0;
    var plot = api.el('pn', 'left:28px;top:28px;width:560px;height:408px', '<h4>Predicted against measured, unseen compounds</h4><canvas width="540" height="352" style="margin-top:2px"></canvas>');
    var tiles = [['R2', C.red], ['Spearman', C.char], ['RMSE', C.char]].map(function(t, k){ return api.el('pn', 'left:608px;top:' + (28 + k * 138) + 'px;width:324px;height:126px', '<h4>' + t[0] + '</h4><div class="big" style="color:' + t[1] + '">0</div><div class="note" style="font-size:12px;color:#6B7280"></div>'); });
    var notes = ['Share of the spread in the answers that the model explains. 1 is perfect, 0 is no better than the average.', 'Do the predictions rank compounds in the right order? It ignores how far off each value is.', 'Typical size of an error, in the units of the endpoint. Smaller is better.'];
    var ctl = api.addControl('Model quality <input type="range" min="0" max="200" value="100" style="width:220px"><span class="qv"></span>'); var rg = qs('input', ctl);
    function prep(){ var t = demoTree() || fitTree(D.BASE, D.TRAIN, 5, 20, 1); idx = D.TEST; ys = idx.map(function(i){ return D.Y[i]; }); pv = idx.map(function(i){ return predictTree(t, D.B, i); }); pm = mean(D.TRAIN.map(function(i){ return D.Y[i]; })); }
    function preds(v){ /* 0 = average only, 100 = decision tree, 200 = perfect */ return idx.map(function(_, k){ return v <= 100 ? pm + (pv[k] - pm) * (v / 100) : pv[k] + (ys[k] - pv[k]) * ((v - 100) / 100); }); }
    function draw(v){
      var p = preds(v), full = new Float64Array(D.N); idx.forEach(function(i, k){ full[i] = p[k]; }); var rr = r2(idx, full), sp = spearman(idx, full), rm = rmse(idx, full);
      var g = cv(qs('canvas', plot)), x = g.x, box = { l: 48, r: g.w - 12, t: 12, b: g.h - 40 }, lo = D.ylo, hi = D.yhi; axes(g, box, [lo, hi], [lo, hi], 'measured', 'predicted', 4);
      var sx = function(v){ return box.l + (v - lo) / (hi - lo) * (box.r - box.l); }, sy = function(v){ return box.b - (clamp(v, lo, hi) - lo) / (hi - lo) * (box.b - box.t); };
      x.strokeStyle = C.char; x.setLineDash([5, 5]); x.beginPath(); x.moveTo(sx(lo), sy(lo)); x.lineTo(sx(hi), sy(hi)); x.stroke(); x.setLineDash([]);
      var step = Math.ceil(idx.length / 220); idx.forEach(function(i, k){ if (k % step) return; var X = sx(ys[k]), Y = sy(p[k]), dI = Math.abs(p[k] - ys[k]); x.strokeStyle = 'rgba(225,36,42,.28)'; x.lineWidth = 1; x.beginPath(); x.moveTo(X, Y); x.lineTo(X, sy(ys[k])); x.stroke(); x.fillStyle = 'rgba(225,36,42,.85)'; x.beginPath(); x.arc(X, Y, 3.3, 0, 6.3); x.fill(); });
      qsa('.big', plot.parentNode); tiles[0].querySelector('.big').textContent = rr.toFixed(2); tiles[1].querySelector('.big').textContent = sp.toFixed(2); tiles[2].querySelector('.big').textContent = rm.toFixed(2);
      tiles.forEach(function(t, k){ t.querySelector('.note').textContent = notes[k]; }); qs('.qv', ctl).textContent = v < 5 ? 'average only' : (v > 195 ? 'perfect' : (Math.abs(v - 100) < 4 ? 'decision tree' : v + '')); }
    rg.addEventListener('input', function(){ api.tok++; api.clearTexts(); q = +rg.value; draw(q); });
    function render(){ prep(); q = +rg.value; draw(q); }
    async function play(tok){ prep(); rg.value = 200; draw(200); await api.beat('A perfect model puts every dot on the diagonal. R2 is 1 and the error is 0.', 2800);
      await api.anim(2600, function(t){ q = 200 - 100 * t; rg.value = q; draw(q); }); api.cap('A real decision tree misses by a lot. Vertical lines show each error.'); await api.sleep(3000);
      await api.anim(2600, function(t){ q = 100 - 100 * t; rg.value = q; draw(q); }); api.cap('Predicting the average for everything gives R2 near 0. Every model must beat this.'); await api.sleep(3000);
      rg.value = 100; draw(100); }
    return { render: render, play: play };
  } });

/* ---------- Step 6: second endpoint, logD feeds HLM ---------- */
L0.add({ short: 'logD feeds HLM', title: 'One model can feed another',
  change: 'Untick features in the left panel. The comparison reruns.',
  watch: 'logD is how oily a compound is at body pH. A model trained on another company\'s logD data predicts it for every compound. That prediction is added as a feature.',
  build: function(api){
    var ab = null;
    var flow = api.el('pn', 'left:28px;top:28px;width:904px;height:130px', '<svg viewBox="0 0 880 100" width="880" height="100"></svg>');
    var sc = api.el('pn', 'left:28px;top:172px;width:430px;height:264px', '<h4>logD_pred against measured HLM</h4><canvas width="410" height="212"></canvas>');
    var res = api.el('pn', 'left:478px;top:172px;width:454px;height:264px', '<h4>Does it help? Boosted trees, same unseen compounds</h4><div class="ab" style="font-size:13px;color:#4a4f5a">Press Run comparison.</div>');
    var ctl = api.addControl('<button type="button" class="btn sm" data-run>Run comparison</button>');
    function drawFlow(on){
      var bw = 150, gap = 33, boxes = [['AstraZeneca logD', '4,200 measured compounds'], ['logD model', LOGD.r2 !== null ? 'boosted trees, unseen R2 ' + LOGD.r2.toFixed(2) : 'trains on load'], ['logD_pred', 'predicted for each HLM compound'], ['HLM model', 'features' + (on ? ' plus logD_pred' : '')], ['HLM prediction', 'log10 clearance']], s = '<defs><marker id="a1" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="' + C.red + '"/></marker></defs>';
      boxes.forEach(function(b, i){ var x = i * (bw + gap), dim = i === 2 && !on; s += '<rect x="' + x + '" y="10" width="' + bw + '" height="68" rx="6" fill="#fff" stroke="' + (i === 2 && on ? C.red : C.silver) + '" stroke-width="' + (i === 2 && on ? 2 : 1) + '" ' + (dim ? 'stroke-dasharray="4 4"' : '') + '/><text x="' + (x + 10) + '" y="32" font-size="13" font-weight="700" fill="' + (dim ? C.mid : C.char) + '">' + b[0] + '</text>';
        var words = b[1].split(' '), lines = [], cur = ''; words.forEach(function(w){ if ((cur + ' ' + w).trim().length > 24) { lines.push(cur); cur = w; } else cur = (cur + ' ' + w).trim(); }); lines.push(cur); lines.forEach(function(l, k){ s += '<text x="' + (x + 10) + '" y="' + (50 + k * 13) + '" font-size="11" fill="#6B7280">' + l + '</text>'; });
        if (i < 4) { var live = !(i === 2 && !on); s += '<line x1="' + (x + bw + 3) + '" y1="44" x2="' + (x + bw + gap - 4) + '" y2="44" stroke="' + (live ? C.red : C.bg) + '" stroke-width="2" ' + (live ? '' : 'stroke-dasharray="4 4"') + ' marker-end="url(#a1)"/>'; } });
      qs('svg', flow).innerHTML = s; }
    function drawScatter(){ if (!D.COL.logD_pred) { var g0 = cv(qs('canvas', sc)); g0.x.fillStyle = C.mid; g0.x.fillText(D.key === 'HLM' ? 'logD model is training in your browser' : 'Switch the endpoint to HLM clearance', 20, 30); return; }
      var f = D.COL.logD_pred, g = cv(qs('canvas', sc)), x = g.x, box = { l: 46, r: g.w - 10, t: 10, b: g.h - 38 }, lo = -1.5, hi = 4.5, yl = D.ylo, yh = D.yhi; axes(g, box, [lo, hi], [yl, yh], 'predicted logD', 'HLM', 4);
      var sx = function(v){ return box.l + (clamp(v, lo, hi) - lo) / (hi - lo) * (box.r - box.l); }, sy = function(v){ return box.b - (clamp(v, yl, yh) - yl) / (yh - yl) * (box.b - box.t); };
      x.fillStyle = 'rgba(161,177,195,.6)'; D.TRAIN.forEach(function(i, k){ if (k % 4 === 0) { x.beginPath(); x.arc(sx(f[i]), sy(D.Y[i]), 2.6, 0, 6.3); x.fill(); } }); x.strokeStyle = C.char; D.TEST.forEach(function(i, k){ if (k % 3 === 0) { x.beginPath(); x.arc(sx(f[i]), sy(D.Y[i]), 2.9, 0, 6.3); x.stroke(); } });
      var all = D.TRAIN.concat(D.TEST), ma = mean(all.map(function(i){ return f[i]; })), mb = mean(all.map(function(i){ return D.Y[i]; })), n = 0, da = 0, db = 0; all.forEach(function(i){ n += (f[i] - ma) * (D.Y[i] - mb); da += (f[i] - ma) * (f[i] - ma); db += (D.Y[i] - mb) * (D.Y[i] - mb); });
      var r = n / Math.sqrt(da * db), sl = n / da; x.strokeStyle = C.red; x.lineWidth = 3; x.beginPath(); x.moveTo(sx(lo + .3), sy(mb + sl * (lo + .3 - ma))); x.lineTo(sx(hi - .3), sy(mb + sl * (hi - .3 - ma))); x.stroke(); x.fillStyle = C.char; x.font = '700 13px Aptos, Segoe UI, sans-serif'; x.textAlign = 'left'; x.fillText('Pearson r = ' + r.toFixed(2), box.l + 8, box.t + 14); }
    var running = false, lastSig = null;
    async function run(force){
      if (D.key !== 'HLM' || running) return; var out = qs('.ab', res), btn = qs('[data-run]', ctl); if (!LOGD.ready) { out.textContent = 'The logD model is still training in your browser.'; return; }
      var base = activeFeats().filter(function(f){ return f !== 'logD_pred'; }); if (!base.length) { out.textContent = 'Switch on at least one descriptor.'; return; } var sig = base.join(',') + '|' + (D.cache.abv || 0); if (!force && sig === lastSig) return;
      running = true; btn.disabled = true; out.textContent = 'Training both models'; await tick(); var r = []; for (var p = 0; p < 2; p++) { await tick(); var m = fitBoost(p ? base.concat(['logD_pred']) : base, D.TRAIN, 100, 0.1, 3); r.push({ r2: r2(D.TEST, m.F), sp: spearman(D.TEST, m.F), tr: r2(D.TRAIN, m.F), secs: m.secs }); }
      var d = r[1].r2 - r[0].r2; out.innerHTML = '<div class="scores" style="display:grid;grid-template-columns:1fr 1fr;gap:10px">' + scoreCard('Without logD_pred', r[0], 0) + scoreCard('With logD_pred', r[1], 1) + '</div><p style="margin-top:12px;font-size:13px"><b>Change in unseen R2: ' + (d >= 0 ? '+' : '') + d.toFixed(3) + '.</b></p>'; btn.disabled = false; running = false; lastSig = sig; ab = r; }
    qs('[data-run]', ctl).addEventListener('click', function(){ run(true); });
    function render(){ drawFlow(FSEL.logD_pred && D.key === 'HLM'); drawScatter(); run(false); }
    featListeners.push(function(){ if (L0.built) drawFlow(FSEL.logD_pred && D.key === 'HLM'); });
    async function play(tok){ render(); if (D.key !== 'HLM') { api.cap('Switch the endpoint to HLM clearance to follow logD into the HLM model.'); return; }
      await api.beat('logD measures how lipophilic a compound is at blood pH. AstraZeneca published 4,200 measured values.', 3200); await api.beat('A model learns logD from those compounds, then predicts logD for every Biogen compound.', 3000);
      api.cap('That prediction is one more feature, logD_pred. Tick it in Features (it is pulsing now) to give it to every model.'); qsa('.featchips label[title]').forEach(function(l){ l.classList.add('pulse'); setTimeout(function(){ l.classList.remove('pulse'); }, 4200); }); await api.sleep(4200); }
    return { render: render, play: play };
  } });

/* ---------- Step 7: predict your own compound (RDKit in the browser) ---------- */
L0.add({ short: 'Predict a compound', title: 'Predict a compound from its structure',
  change: 'Edit the SMILES and press Predict.',
  watch: 'Type a molecule as SMILES. The page predicts its logD, then its clearance.',
  auto: false,
  build: function(api){
    var inp = api.el('pn', 'left:28px;top:28px;width:904px;height:70px', '<h4>SMILES</h4><div style="display:flex;gap:10px"><input type="text" class="si" spellcheck="false" style="font-family:Cascadia Mono,Consolas,monospace"><button type="button" class="btn sm" data-go style="white-space:nowrap">Predict</button></div>');
    var st = api.el('pn', 'left:28px;top:112px;width:300px;height:324px', '<h4>Structure drawn by RDKit</h4><div class="struct" style="height:272px"></div>');
    var ft = api.el('pn', 'left:348px;top:112px;width:270px;height:324px', '<h4>Features computed by RDKit</h4><div class="fr nums" style="font-size:13px"></div>');
    var out = api.el('pn', 'left:638px;top:112px;width:294px;height:324px', '<h4>Predictions</h4><div class="po" style="font-size:13px;color:#4a4f5a">Press Predict.</div>');
    var MAP = { logP: 'CrippenClogP', TPSA: 'tpsa', MW: 'amw', HBD: 'NumHBD', HBA: 'NumHBA', RotB: 'NumRotatableBonds', Fsp3: 'FractionCSP3', AromRings: 'NumAromaticRings' }, hm = null, busy = false;
    var si = qs('.si', inp);
    async function render(){ if (!si.value) { await loadStructs(); si.value = (STRUCTS.HLM[0] || {}).smi || 'CC(=O)Nc1ccc(O)cc1'; } }
    async function predict(){
      if (busy) return; busy = true; var po = qs('.po', out), smi = si.value.trim(); po.textContent = RDK ? 'Computing' : 'Loading RDKit (about 7 MB)'; progress(0.3);
      try { await loadRDKit(); progress(0.6); var svg = rdkSvg(smi, 300, 272); if (!svg) { po.innerHTML = '<span style="color:' + C.red + '">RDKit could not read that SMILES.</span>'; busy = false; progress(1); return; }
        qs('.struct', st).innerHTML = svg; var d = rdkDescriptors(smi), hlmD = DATA.HLM || await loadDataset('HLM'), vals = {}; Object.keys(MAP).forEach(function(k){ vals[k] = d[MAP[k]]; });
        qs('.fr', ft).innerHTML = Object.keys(MAP).map(function(k){ return '<div><span>' + k + '</span><b>' + fmtFeat(k, vals[k]) + '</b></div>'; }).join('');
        await ensureLogdPred(); var logd = boostPredictVals(LOGD.model, LOGD.B, vals); progress(0.8); await tick();
        if (!hm) { var feats = hlmD.BASE.concat(['logD_pred']); hm = fitBoost(feats, hlmD.TRAIN, 120, 0.1, 3, 0, hlmD); }
        var v2 = Object.assign({ logD_pred: logd }, vals), hl = boostPredictVals(hm, hlmD.B, v2);
        var span = hlmD.yhi - hlmD.ylo; po.innerHTML = '<div class="lbl">Predicted logD, first model</div><div class="big" style="color:' + C.char + '">' + logd.toFixed(2) + '</div><div class="lbl" style="margin-top:8px">Predicted HLM clearance (log10), second model</div><div class="big" style="color:' + C.red + '">' + hl.toFixed(2) + '</div><div class="pbar"><i style="background:' + C.red + ';width:' + clamp((hl - hlmD.ylo) / span * 100, 0, 100) + '%"></i></div><p class="note" style="font-size:12px;margin-top:6px">Models trained on about 2,500 compounds with eight descriptors. Treat the numbers as an illustration, not a measurement. ' + (hl <= hlmD.floor + 0.05 ? 'The prediction is at the assay floor, so the compound is likely stable in HLM.' : '') + '</p>'; }
      catch (e) { po.innerHTML = '<span style="color:' + C.red + '">' + esc(e.message) + '</span>'; console.error(e); }
      progress(1); busy = false; }
    qs('[data-go]', inp).addEventListener('click', predict); si.addEventListener('keydown', function(e){ if (e.key === 'Enter') predict(); });
    return { render: render };
  } });

L0.glossary([
  ['Machine learning', 'Software that learns a pattern from examples instead of following hand-written rules.'],
  ['Model', 'A formula with adjustable settings that turns features into a prediction.'],
  ['Feature', 'A number describing a molecule, such as logP or molecular weight.'],
  ['Target', 'What we predict: log10 HLM clearance, or logD.'],
  ['Training compounds', 'The compounds a model is allowed to study.'],
  ['Unseen compounds', 'Compounds held back to test whether the model learned a real pattern or memorised.'],
  ['Scaffold', 'The ring skeleton of a compound, with side chains removed.'],
  ['Scaffold split', 'Unseen compounds share no scaffold with the training set.'],
  ['Overfitting', 'Memorising training data, noise included, so new compounds are predicted worse.'],
  ['R2', 'Share of the spread in the answers explained on unseen compounds. 1 is perfect, 0 is no better than the average.'],
  ['Spearman', 'How well the model ranks compounds, ignoring exact values.'],
  ['RMSE', 'Typical error in the units of the endpoint.'],
  ['SMILES', 'A line of text that describes a structure, such as CC(=O)N.'],
  ['logP', 'Preference of a compound for oil over water, from atom contributions.'],
  ['logD', 'logP corrected for ionisation at pH 7.4.'],
  ['TPSA', 'Polar surface area from nitrogen and oxygen atoms.'],
  ['HLM clearance', 'How fast human liver microsomes remove a compound. Higher means a shorter half-life.'],
  ['Stacking', 'Feeding one model\'s prediction into another model as a feature.'],
  ['RDKit', 'Open-source chemistry toolkit used here for structures, descriptors and fingerprints.']
]);

setUses(L0, [null, ['endpoint', 'features'], ['endpoint'], ['endpoint', 'features'], null, null, ['endpoint', 'features'], ['endpoint', 'features'], ['endpoint', 'features'], []]);
featListeners.push(function(){ if (L0.built) L0.refresh(); });
