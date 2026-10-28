"use strict";
/* Lesson 11, Beginner steps. Written for PK-literate scientists who are new to machine learning.
   Steps: Model families and tasks, Predicted parameters, Error propagation.
   Data: the Biogen HLM set (public), through loadDataset('HLM'). The global endpoint is never read here. */

/* ---------- shared: HLM data, and a boosted-trees model fitted on the training compounds only ---------- */
var MP = { ds: null, F: null, rmse: null, rmseQ: null, list: null, p: null, med: null, lo: 0.7, hi: 3.5 };
async function mpReady(api){
  if (!MP.p) MP.p = (async function(){
    await tick(); var ds = await loadDataset('HLM'); await loadStructs(); await tick();
    var m = fitBoost(ds.BASE.slice(), ds.TRAIN, 150, 0.08, 3, 0, ds), q = ds.TEST.filter(function(i){ return !ds.cens[i]; });
    MP.ds = ds; MP.F = m.F; MP.rmse = rmse(ds.TEST, m.F, ds.Y); MP.rmseQ = rmse(q, m.F, ds.Y);
    MP.list = (STRUCTS.HLM || []).filter(function(e){ return ds.test[e.i] && !ds.cens[e.i]; });
    var ys = q.map(function(i){ return ds.Y[i]; }).sort(function(a, b){ return a - b; }); MP.med = ys[Math.floor(ys.length / 2)]; MP.lo = Math.floor(ys[0] * 20) / 20; MP.hi = Math.ceil(ys[ys.length - 1] * 20) / 20;
  })();
  api.busy('Fitting the HLM model on the training compounds'); try { await MP.p; } finally { api.idle(); }
}
function mpFmt(v, d){ return (+v).toFixed(d); }
function mpSig(v){ v = +v; if (v === 0) return '0'; var a = Math.abs(v); return a >= 100 ? v.toFixed(0) : (a >= 10 ? v.toFixed(1) : (a >= 1 ? v.toFixed(2) : v.toPrecision(2))); }
function mpSegHTML(attr, items, cur){ return '<span class="seg" ' + attr + '>' + items.map(function(it, i){ return '<button type="button" data-v="' + i + '"' + (i === cur ? ' class="on"' : '') + '>' + esc(it) + '</button>'; }).join('') + '</span>'; }

/* ---------- 1. Model families and tasks ---------- */
var MP_FAM = [
  { n: 'Decision tree', ref: 'Lesson 02, Decision tree structure' },
  { n: 'Random forest', ref: 'Lesson 02, Random forests' },
  { n: 'Gradient boosting', ref: 'Lesson 02, Gradient boosting' },
  { n: 'Neural network', ref: 'Lesson 03, Hidden layer' },
  { n: 'Graph network', ref: 'Lesson 03, Message passing' },
  { n: 'Pre-trained model', ref: 'Lesson 05, Transfer strategies' },
  { n: 'GPT and language models', ref: 'Lesson 04, Next-token prediction' }
];
/* r: family index -> [rating, rationale]. P: well matched. S: workable with limits. Families not listed are not suited or not used for the task. */
var MP_TASKS = [
  { n: 'Parameter prediction from structure', d: 'Predict CL, Vss, fu or similar from the structure of a compound.',
    r: { 0: ['S', 'Interpretable, but usually less accurate than an ensemble of trees.'], 1: ['P', 'Robust on descriptor and fingerprint tables of a few thousand compounds.'], 2: ['P', 'Strong on descriptor tables. It had the highest score of the HLM reference runs in lesson 06.'], 3: ['S', 'Needs more data and tuning than trees on tabular descriptors.'], 4: ['S', 'Learns features from the molecular graph with no descriptor table. In the lesson 06 reference runs it scored below boosting on HLM.'], 5: ['S', 'Fine-tuning helps most when few labelled compounds are available.'] } },
  { n: 'Surrogate of a simulator', d: 'Replace a PBPK or ODE simulator with a fast approximation of its outputs.',
    r: { 0: ['S', 'A readable approximation, but coarse.'], 1: ['S', 'Robust, but predictions are piecewise constant and cannot extrapolate beyond the sampled inputs.'], 2: ['P', 'Trains quickly on a table of simulator inputs and one output. The prediction is piecewise constant.'], 3: ['P', 'Smooth regression from simulator inputs to outputs, and cheap to evaluate once trained.'] } },
  { n: 'Covariate screening in population PK', d: 'Identify which covariates explain between-subject variability.',
    r: { 0: ['S', 'Gives readable splits of the covariates, but the tree is unstable between samples.'], 1: ['P', 'Importance measures over many covariates, with nonlinear effects and interactions allowed.'], 2: ['P', 'Handles mixed covariate types and interactions. Importance is ambiguous when covariates are correlated.'] } },
  { n: 'Concentration-time profiles', d: 'Model observed profiles from sparse sampling in many subjects.',
    r: { 2: ['S', 'Time can enter as a feature in a table of observations. The fitted curve is piecewise constant.'], 3: ['P', 'A network that takes time and covariates as inputs, or a sequence model, can fit profiles when many subjects are available.'], 6: ['S', 'The transformer architecture handles sequences, but needs far more data than a typical PK data set.'] } },
  { n: 'Hybrid mechanistic-ML models', d: 'Keep the ODE structure and learn one unknown term.',
    r: { 3: ['P', 'A small network represents one unknown term inside an ODE while the mechanistic structure is kept.'] } },
  { n: 'Molecule generation', d: 'Propose new structures, for example under predicted property constraints.',
    r: { 5: ['S', 'A pre-trained generative model can be fine-tuned on a focused set of structures.'], 6: ['P', 'Autoregressive generation of SMILES. Predicted PK parameters can then filter the generated structures.'] } },
  { n: 'Literature and code assistance', d: 'Draft text, summaries and analysis code.',
    r: { 6: ['P', 'Language models draft text and code. The output has no measurement behind it and needs checking (lesson 07).'] } }
];
LESSONS.mlpk.add({ level: 'beginner', short: 'Model families and tasks', title: 'Model families and PK tasks',
  change: 'Select a task row. The panel below lists the families that fit it, with the reason and the lesson that covers each family.',
  watch: 'The model families in lessons 02 to 05 map onto tasks in PK, PBPK and MIDD workflows. A family is well matched to a task when its inputs, outputs and data requirements fit the task, for example a table of descriptors to a clearance value. The matrix rates seven tasks against seven families. The ratings describe fit to the task, not a ranking of accuracy, and the reason for each rating is shown when the task is selected.',
  build: function(api){
    var cur = 0;
    var mp = api.el('pn', 'left:28px;top:20px;width:904px;height:310px;overflow:hidden', '<div class="mx"></div>');
    var rp = api.el('pn', 'left:28px;top:340px;width:904px;height:180px;overflow:auto', '<div class="rt" style="font-size:13px;line-height:1.45;color:' + C.char + '"></div>');
    function dot(r, on){ if (r === 'P') return '<span style="display:inline-block;width:14px;height:14px;border-radius:50%;background:' + C.red + ';border:2px solid ' + C.red + '"></span>'; if (r === 'S') return '<span style="display:inline-block;width:14px;height:14px;border-radius:50%;background:#fff;border:2px solid ' + C.red + '"></span>'; return '<span style="display:inline-block;width:10px;height:2px;background:' + C.silver + ';vertical-align:middle"></span>'; }
    function draw(){
      var hd = '<tr><th style="text-align:left;padding:2px 8px;font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:' + C.mid + ';background:' + C.light + '">Task</th>' + MP_FAM.map(function(f){ return '<th style="padding:2px 3px;font-size:11px;font-weight:700;line-height:1.15;color:' + C.char + ';background:' + C.light + ';width:92px">' + esc(f.n) + '</th>'; }).join('') + '</tr>';
      var rows = MP_TASKS.map(function(t, k){ var on = k === cur; return '<tr data-k="' + k + '" tabindex="0" style="cursor:pointer;border-top:1px solid ' + C.silver + ';background:' + (on ? '#FBE9E9' : '#fff') + '"><td style="padding:2px 8px;font-weight:' + (on ? 700 : 500) + ';font-size:13px">' + esc(t.n) + '</td>' + MP_FAM.map(function(f, j){ return '<td style="text-align:center;padding:2px 3px">' + dot(t.r[j] ? t.r[j][0] : '') + '</td>'; }).join('') + '</tr>'; }).join('');
      qs('.mx', mp).innerHTML = '<table style="width:100%;border-collapse:collapse;table-layout:fixed"><colgroup><col style="width:228px">' + MP_FAM.map(function(){ return '<col>'; }).join('') + '</colgroup>' + hd + rows + '</table><div style="font-size:12px;margin-top:6px;color:' + C.char + '"><span style="margin-right:20px">' + dot('P') + ' Well matched to the task</span><span style="margin-right:20px">' + dot('S') + ' Workable with limits</span><span>' + dot('') + ' Not suited, or not used for the task</span></div>';
      var t = MP_TASKS[cur], items = Object.keys(t.r).map(Number).sort(function(a, b){ return (t.r[a][0] === 'P' ? 0 : 1) - (t.r[b][0] === 'P' ? 0 : 1) || a - b; });
      qs('.rt', rp).innerHTML = '<div class="ol">' + esc(t.n) + '</div><p style="margin:2px 0 8px">' + esc(t.d) + '</p>' + items.map(function(j){ var e = t.r[j]; return '<div style="display:grid;grid-template-columns:22px 170px 1fr 230px;gap:8px;align-items:start;padding:5px 0;border-top:1px solid ' + C.light + '"><span style="padding-top:2px">' + dot(e[0]) + '</span><b>' + esc(MP_FAM[j].n) + '</b><span>' + esc(e[1]) + '</span><span style="color:' + C.mid + ';font-size:12px">' + esc(MP_FAM[j].ref) + '</span></div>'; }).join('');
      qsa('tr[data-k]', mp).forEach(function(tr){ tr.addEventListener('click', function(){ cur = +tr.dataset.k; draw(); }); tr.addEventListener('keydown', function(e){ if (e.key === 'Enter') { cur = +tr.dataset.k; draw(); } }); });
    }
    async function play(tok){
      for (var k = 0; k < MP_TASKS.length; k++) { api.checkTok(tok); cur = k; draw(); api.cap(MP_TASKS[k].n + ': ' + Object.keys(MP_TASKS[k].r).filter(function(j){ return MP_TASKS[k].r[j][0] === 'P'; }).map(function(j){ return MP_FAM[j].n.toLowerCase(); }).join(' and ') + ' well matched.'); await api.sleep(2600); }
      cur = 0; draw(); api.cap(null);
    }
    return { render: draw, play: play };
  } });

/* ---------- 2. Predicted parameters ---------- */
LESSONS.mlpk.add({ level: 'beginner', short: 'Predicted parameters', title: 'Predicted parameters as PBPK inputs',
  change: 'Select a test compound, or select Next compound. Switch the concentration axis between linear and logarithmic.',
  watch: 'Machine learning enters a PBPK workflow as the source of parameter values. A boosted-trees model, fitted on the HLM training compounds only, predicts log10 intrinsic clearance for a test compound. Both the prediction and the measured value are converted to hepatic clearance with the well-stirred model and used as the metabolic clearance in the same whole-body PBPK model of a 10 mg oral dose. Every other parameter is identical in the two simulations, so the difference between the curves isolates the effect of the prediction error in CLint. The HLM value is the log10 intrinsic clearance in mL/min/kg of this course. It is used as a clearance per kg body weight, with no microsomal protein or liver weight scaling. The other inputs are assumptions, listed on the right, and are not measurements.',
  build: function(api){
    var DOSE = 10, TEND = 48, BW = 70, KA = 1, cur = -1, logY = false, sims = {}, busy = false;
    var sp = api.el('pn', 'left:28px;top:20px;width:290px;height:214px;overflow:hidden', '<h4 class="nm"></h4><div class="struct" style="height:150px"></div><div class="smi" style="font:11px Cascadia Mono,Consolas,monospace;color:' + C.mid + ';word-break:break-all;margin-top:2px;max-height:34px;overflow:hidden"></div>');
    var tp = api.el('pn', 'left:28px;top:246px;width:290px;height:274px;overflow:auto', '<div class="tb" style="font-size:12.5px;color:' + C.char + '"></div>');
    var cp = api.el('pn', 'left:338px;top:20px;width:594px;height:296px', '<h4>Simulated plasma concentration, 10 mg oral dose</h4><canvas width="566" height="254"></canvas>');
    var ap = api.el('pn', 'left:338px;top:326px;width:594px;height:194px;overflow:auto', '<div class="as" style="font-size:12px;line-height:1.45;color:' + C.char + '"></div>');
    var ctl = api.addControl('Test compound <select class="sel" style="width:230px"></select> <button type="button" class="btn sm ghost" data-next>Next compound</button> Axis ' + mpSegHTML('data-ax', ['Linear', 'Logarithmic'], 0));
    var sel = qs('.sel', ctl);
    function sim(e){
      if (sims[e.i]) return sims[e.i]; var ds = MP.ds, o = {}, kinds = { measured: ds.Y[e.i], predicted: MP.F[e.i] }, logp = ds.COL.logP[e.i];
      Object.keys(kinds).forEach(function(k){
        var m = PKE.pbpk.model({ logP: logp, fup: 1, bp: 1, clintU: PKE.pbpk.clintUFromHLM(kinds[k], BW, 1), clr: 0, ka: KA, F: 1, bw: BW }), r = PKE.pbpk.simulate(m, { route: 'oral', dose: DOSE }, TEND, null, 200), cm = 0, tm = 0;
        r.plasma.forEach(function(c, j){ if (c > cm) { cm = c; tm = r.t[j]; } });
        o[k] = { t: r.t, c: r.plasma, auc: r.aucPlasma[r.aucPlasma.length - 1], cmax: cm, tmax: tm, log10: kinds[k], ivive: PKE.fromHLM(kinds[k], { bw: BW }), clh: PKE.pbpk.modelHepatic(m), massError: r.massError, flowKg: m.phys.flows.liverTotal * 1000 / 60 / BW };
      });
      return (sims[e.i] = o);
    }
    function ticksLog(a, b){ var mx = Math.max.apply(null, a.concat(b).map(function(p){ return p[1]; })), o = []; for (var v = -1; v <= Math.ceil(mx); v++) o.push({ v: v, t: v < 0 ? '0.1' : String(Math.pow(10, v)) }); return o; }
    function draw(){
      var e = MP.list[cur], r = sims[e.i]; if (!r) return;
      qs('.nm', sp).textContent = 'Compound ' + (e.i + 1); qs('.struct', sp).innerHTML = e.svg; qs('.smi', sp).textContent = e.smi;
      var f = function(v){ return v > 0 ? Math.log10(Math.max(v * 1000, 0.1)) : -1; };
      var sM = r.measured.t.map(function(t, j){ return [t, logY ? f(r.measured.c[j]) : r.measured.c[j] * 1000]; }), sP = r.predicted.t.map(function(t, j){ return [t, logY ? f(r.predicted.c[j]) : r.predicted.c[j] * 1000]; });
      lineChart(qs('canvas', cp), [{ pts: sM, color: C.char, w: 2.6, nodot: true, label: 'measured CLint' }, { pts: sP, color: C.red, w: 2.6, nodot: true, label: 'predicted CLint' }], { xl: 'time after dose (h)', yl: 'plasma concentration (µg/L)', x0: 0, x1: TEND, y0: logY ? -1 : 0, yticks: logY ? ticksLog(sM, sP) : undefined });
      var fa = r.predicted.auc / r.measured.auc, fc = r.predicted.cmax / r.measured.cmax, row = function(l, a, b, u){ return '<tr style="border-top:1px solid ' + C.silver + '"><td style="padding:3px 4px">' + l + '</td><td style="text-align:right;padding:3px 4px">' + a + '</td><td style="text-align:right;padding:3px 4px;color:' + C.red + ';font-weight:700">' + b + '</td></tr>'; };
      qs('.tb', tp).innerHTML = '<table style="width:100%;border-collapse:collapse"><tr style="font-weight:700;font-size:11px;color:' + C.mid + '"><td></td><td style="text-align:right;padding:2px 4px">Measured</td><td style="text-align:right;padding:2px 4px">Predicted</td></tr>' +
        row('log10 CLint (mL/min/kg)', mpFmt(r.measured.log10, 2), mpFmt(r.predicted.log10, 2)) + row('Hepatic CL, plasma (L/h)', mpFmt(r.measured.ivive.clhPlasmaLh, 1), mpFmt(r.predicted.ivive.clhPlasmaLh, 1)) + row('Extraction ratio', mpFmt(r.measured.ivive.extraction, 2), mpFmt(r.predicted.ivive.extraction, 2)) + row('AUC 0 to 48 h (µg h/L)', mpSig(r.measured.auc * 1000), mpSig(r.predicted.auc * 1000)) + row('Cmax (µg/L)', mpSig(r.measured.cmax * 1000), mpSig(r.predicted.cmax * 1000)) + '</table>' +
        '<div style="margin-top:8px"><b>AUC fold error ' + mpFmt(fa >= 1 ? fa : 1 / fa, 2) + '</b> (' + (fa >= 1 ? 'over' : 'under') + '-predicted). <b>Cmax fold error ' + mpFmt(fc >= 1 ? fc : 1 / fc, 2) + '</b>.</div>';
      qs('.as', ap).innerHTML = '<div class="ol">What the comparison isolates</div>The two curves differ only in the CLint input. The error in log10 CLint for this compound is ' + mpFmt(r.predicted.log10 - r.measured.log10, 2) + '. The test RMSE of the model is ' + mpFmt(MP.rmse, 2) + ', and this compound is shown first because its error is the median error of the test compounds in the list. Fixed for both simulations, as assumptions and not measurements: oral dose 10 mg; body weight 70 kg; fraction unbound in plasma 1; blood to plasma ratio 1; absorption rate constant 1 per h; fraction reaching the gut wall 1; no renal clearance; perfusion-limited distribution with tissue partition coefficients from the Poulin and Theil mixture model, using the Crippen logP of this compound (' + mpFmt(MP.ds.COL.logP[e.i], 2) + '). The PBPK model uses its own liver blood flow, ' + mpFmt(r.measured.flowKg, 1) + ' mL/min/kg. The HLM value is used as intrinsic clearance per kg body weight, without microsomal protein or liver weight scaling. The predicted CLint comes from a boosted-trees model fitted on the training compounds only, and this compound is in the test set.';
    }
    async function pick(k){
      if (busy) return; cur = k; sel.value = String(k); busy = true; var e = MP.list[cur];
      if (!sims[e.i]) { api.busy('Simulating'); await tick(); await tick(); try { sim(e); } finally { api.idle(); } }
      busy = false; draw();
    }
    async function render(){
      await mpReady(api); if (cur < 0) { var ae = MP.list.map(function(e, k){ return [Math.abs(MP.F[e.i] - MP.ds.Y[e.i]), k]; }).sort(function(a, b){ return a[0] - b[0]; }); cur = ae[Math.floor(ae.length / 2)][1]; }
      if (!sel.options.length) { MP.list.forEach(function(e, k){ var o = document.createElement('option'); o.value = k; o.textContent = 'Compound ' + (e.i + 1) + ', measured log10 CLint ' + mpFmt(MP.ds.Y[e.i], 2); sel.appendChild(o); }); }
      await pick(cur);
    }
    sel.addEventListener('change', function(){ pick(+sel.value); });
    qs('[data-next]', ctl).addEventListener('click', function(){ pick((cur + 1) % MP.list.length); });
    qsa('[data-ax] button', ctl).forEach(function(b){ b.addEventListener('click', function(){ logY = +b.dataset.v === 1; qsa('[data-ax] button', ctl).forEach(function(x){ x.classList.toggle('on', x === b); }); draw(); }); });
    async function play(tok){
      await render(); api.cap('A model fitted on the training compounds predicts log10 CLint for a test compound.'); await api.sleep(3200); api.checkTok(tok);
      api.cap('Both the prediction and the measured value go through the same IVIVE and PBPK model.'); await api.sleep(3200); api.checkTok(tok);
      logY = true; qsa('[data-ax] button', ctl).forEach(function(x){ x.classList.toggle('on', +x.dataset.v === 1); }); draw(); api.cap('On a logarithmic axis the terminal slopes show the difference in clearance.'); await api.sleep(3400);
      logY = false; qsa('[data-ax] button', ctl).forEach(function(x){ x.classList.toggle('on', +x.dataset.v === 0); }); draw(); api.cap(null);
    }
    return { render: render, play: play };
  } });

/* ---------- 3. Error propagation ---------- */
LESSONS.mlpk.add({ level: 'beginner', short: 'Error propagation', title: 'Error propagation to exposure',
  change: 'Drag the error SD and the measured CLint. Compare a low and a high CLint at the same error.',
  watch: 'A prediction error in an ADME parameter propagates to the predicted exposure. The test RMSE of the boosted-trees model for HLM is taken as the standard deviation of a lognormal error in CLint, in log10 units. One thousand errors are drawn with a fixed seed, and each is converted to hepatic clearance with the well-stirred model and to AUC with a one-compartment intravenous model, AUC = dose divided by CL. The histogram shows the AUC fold error relative to the AUC at the measured CLint. Extraction saturates as CLint rises, so the same error in CLint changes hepatic clearance little for a high-extraction compound and by the full fold error for a low-extraction compound. The other inputs are assumptions: fraction unbound in plasma 1, blood to plasma ratio 1, hepatic blood flow 20.7 mL/min/kg, 70 kg body weight, and no renal clearance.',
  build: function(api){
    var N = 1000, DOSE = 100, sd = 0.55, ref = 2, rows = null;
    var hp = api.el('pn', 'left:28px;top:20px;width:444px;height:318px', '<h4>AUC fold error, 1000 draws</h4><canvas width="416" height="276"></canvas>');
    var cp = api.el('pn', 'left:488px;top:20px;width:444px;height:318px', '<h4>Hepatic clearance against CLint</h4><canvas width="416" height="276"></canvas>');
    var np = api.el('pn', 'left:28px;top:350px;width:904px;height:170px;overflow:auto', '<div class="ns" style="font-size:12.5px;line-height:1.5;color:' + C.char + '"></div>');
    var ctl = api.addControl('Error SD, log10 units <input class="sd" type="range" min="0.05" max="1.2" step="0.01" style="width:160px"> <b class="sdv"></b> &nbsp; Measured log10 CLint <input class="rf" type="range" step="0.05" style="width:160px"> <b class="rfv"></b>');
    var sdI = qs('.sd', ctl), rfI = qs('.rf', ctl), sdV = qs('.sdv', ctl), rfV = qs('.rfv', ctl);
    var GAUSS = null;
    function normals(){ if (GAUSS) return GAUSS; var r = rng(2026), z = []; for (var i = 0; i < N; i++) { var u = 0; while (u === 0) u = r(); z.push(Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r())); } return (GAUSS = z); }
    function cl(l10){ return PKE.fromHLM(l10, { bw: 70 }).clhPlasmaLh; }
    function calc(){
      var z = normals(), c0 = cl(ref), fe = z.map(function(v){ return c0 / cl(ref + sd * v); }).sort(function(a, b){ return a - b; }), within = fe.filter(function(v){ return v >= 0.5 && v <= 2; }).length / N;
      return { fe: fe, c0: c0, auc0: DOSE / c0, within: within, p5: fe[Math.floor(0.05 * N)], p95: fe[Math.floor(0.95 * N)], e: PKE.fromHLM(ref, { bw: 70 }).extraction };
    }
    function drawHist(R){
      var g = cv(qs('canvas', hp)), x = g.x, L = 46, Rr = g.w - 14, T = 34, B = g.h - 46, lo = -0.8, hi = 0.8, nb = 32, bins = new Array(nb).fill(0), mx;
      R.fe.forEach(function(v){ var l = Math.log10(v), k = Math.floor((l - lo) / (hi - lo) * nb); if (k >= 0 && k < nb) bins[k]++; }); mx = Math.max.apply(null, bins) || 1;
      var px = function(l){ return L + (l - lo) / (hi - lo) * (Rr - L); }, py = function(c){ return B - c / mx * (B - T); };
      x.strokeStyle = C.silver; x.lineWidth = 1; x.fillStyle = C.mid; x.font = '11px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'center';
      [0.2, 0.5, 1, 2, 5].forEach(function(v){ var xx = px(Math.log10(v)); x.beginPath(); x.moveTo(xx, T); x.lineTo(xx, B); x.stroke(); x.fillText(v + (v === 1 ? '' : 'x') , xx, B + 15); });
      x.fillStyle = 'rgba(46,125,79,.12)'; x.fillRect(px(Math.log10(0.5)), T - 12, px(Math.log10(2)) - px(Math.log10(0.5)), B - T + 12);
      bins.forEach(function(c, k){ var x0 = L + k / nb * (Rr - L), w = (Rr - L) / nb - 1; x.fillStyle = C.char; x.fillRect(x0, py(c), w, B - py(c)); });
      x.strokeStyle = C.red; x.lineWidth = 2; x.beginPath(); x.moveTo(px(0), T - 12); x.lineTo(px(0), B); x.stroke();
      [R.p5, R.p95].forEach(function(v){ x.strokeStyle = C.red; x.setLineDash([4, 3]); x.beginPath(); x.moveTo(px(Math.log10(v)), T - 12); x.lineTo(px(Math.log10(v)), B); x.stroke(); x.setLineDash([]); });
      x.fillStyle = C.red; x.font = '600 11px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'center'; x.textAlign = 'right'; x.fillText('5th', px(Math.log10(R.p5)) - 4, T - 2); x.textAlign = 'left'; x.fillText('95th', px(Math.log10(R.p95)) + 4, T - 2);
      x.fillStyle = '#1E6B40'; x.font = '11px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'left'; x.fillText('Green: within 2-fold. Dashed: 5th and 95th percentile.', L, 12);
      x.fillStyle = C.char; x.font = '600 12px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'center'; x.fillText('AUC fold error, predicted over reference (log scale)', (L + Rr) / 2, g.h - 6);
      x.save(); x.translate(13, (T + B) / 2); x.rotate(-Math.PI / 2); x.fillText('draws', 0, 0); x.restore();
    }
    function drawCurve(R){
      var g = cv(qs('canvas', cp)), x = g.x, L = 52, Rr = g.w - 14, T = 14, B = g.h - 46, lo = 0, hi = 4, qh = 20.7 * 70 * 60 / 1000, px = function(l){ return L + (l - lo) / (hi - lo) * (Rr - L); }, py = function(v){ return B - v / (qh * 1.1) * (B - T); }, k;
      x.strokeStyle = C.silver; x.fillStyle = C.mid; x.lineWidth = 1; x.font = '11px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'right';
      [0, 20, 40, 60, 80].forEach(function(v){ x.beginPath(); x.moveTo(L, py(v)); x.lineTo(Rr, py(v)); x.stroke(); x.fillText(String(v), L - 6, py(v) + 4); });
      x.textAlign = 'center'; [0, 1, 2, 3, 4].forEach(function(l){ x.fillText(String(l), px(l), B + 15); });
      var lo2 = ref - sd, hi2 = ref + sd; x.fillStyle = 'rgba(225,36,42,.10)'; x.fillRect(px(lo2), T, px(hi2) - px(lo2), B - T);
      x.strokeStyle = C.bg; x.setLineDash([5, 4]); x.beginPath(); x.moveTo(L, py(qh)); x.lineTo(Rr, py(qh)); x.stroke(); x.setLineDash([]);
      x.fillStyle = C.mid; x.textAlign = 'right'; x.fillText('hepatic blood flow, ' + qh.toFixed(0) + ' L/h', Rr - 4, py(qh) - 5);
      x.strokeStyle = C.char; x.lineWidth = 2.4; x.beginPath(); for (k = 0; k <= 80; k++) { var l = lo + (hi - lo) * k / 80, v = cl(l); k ? x.lineTo(px(l), py(v)) : x.moveTo(px(l), py(v)); } x.stroke();
      [[lo2, 'ref - SD'], [ref, 'reference'], [hi2, 'ref + SD']].forEach(function(p, i){ x.fillStyle = i === 1 ? C.red : C.char; x.beginPath(); x.arc(px(p[0]), py(cl(p[0])), i === 1 ? 5.5 : 4, 0, 6.3); x.fill(); });
      x.fillStyle = C.char; x.font = '600 12px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'center'; x.fillText('log10 CLint (mL/min/kg)', (L + Rr) / 2, g.h - 6);
      x.save(); x.translate(13, (T + B) / 2); x.rotate(-Math.PI / 2); x.fillText('hepatic CL, plasma (L/h)', 0, 0); x.restore();
    }
    function draw(){
      if (!MP.ds) return; var R = calc(); sdV.textContent = mpFmt(sd, 2); rfV.textContent = mpFmt(ref, 2); drawHist(R); drawCurve(R);
      var lowC = cl(ref - sd), highC = cl(ref + sd);
      qs('.ns', np).innerHTML = '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:14px"><div><div class="ol">Within 2-fold</div><div class="big" style="color:' + C.red + ';font-size:30px">' + mpFmt(R.within * 100, 0) + '%</div>of the draws</div><div><div class="ol">5th to 95th percentile</div><div class="big" style="font-size:24px">' + mpFmt(R.p5, 2) + ' to ' + mpFmt(R.p95, 2) + '</div>AUC fold error</div><div><div class="ol">Extraction ratio at reference</div><div class="big" style="font-size:30px">' + mpFmt(R.e, 2) + '</div>well-stirred, fup 1, bp 1</div><div><div class="ol">AUC at reference</div><div class="big" style="font-size:30px">' + mpFmt(R.auc0, 2) + '</div>mg h/L for ' + DOSE + ' mg IV</div></div>' +
        '<p style="margin:10px 0 0">The test RMSE of the HLM boosted-trees model is ' + mpFmt(MP.rmse, 2) + ' log10 units over all ' + MP.ds.TEST.length + ' test compounds, and ' + mpFmt(MP.rmseQ, 2) + ' over the test compounds with a quantified value. The shaded band on the right is the reference CLint plus and minus one error SD: it spans hepatic clearances of ' + mpFmt(lowC, 1) + ' to ' + mpFmt(highC, 1) + ' L/h. AUC is the dose divided by the clearance of a one-compartment intravenous model, so the AUC fold error is the reference clearance divided by the sampled clearance.</p>';
    }
    sdI.addEventListener('input', function(){ sd = +sdI.value; draw(); }); rfI.addEventListener('input', function(){ ref = +rfI.value; draw(); });
    async function render(){ await mpReady(api); rfI.min = MP.lo; rfI.max = MP.hi; if (!rows) { rows = 1; sd = Math.round(MP.rmse * 100) / 100; ref = Math.round(MP.med * 20) / 20; } sdI.value = sd; rfI.value = ref; draw(); }
    async function play(tok){
      await render(); var s0 = sd; sd = Math.round(MP.rmse * 100) / 100; sdI.value = sd;
      ref = 1.2; rfI.value = ref; draw(); api.cap('Low CLint: hepatic clearance is proportional to CLint, so the error in CLint passes to AUC unchanged.'); await api.sleep(3800); api.checkTok(tok);
      ref = 3.2; rfI.value = ref; draw(); api.cap('High CLint: extraction approaches 1 and clearance approaches hepatic blood flow, so the same error changes AUC much less.'); await api.sleep(3800); api.checkTok(tok);
      ref = Math.round(MP.med * 20) / 20; rfI.value = ref; draw(); api.cap(null);
    }
    return { render: render, play: play };
  } });

/* ---------- guide entries ---------- */
GUIDE.steps['mlpk/Model families and tasks'] = { notice: 'Most tasks have one or two well-matched families, and some have none among the model families in this course.', q: 'Why is a random forest well matched to covariate screening but a decision tree only workable?', a: 'A forest averages many trees, so its importance measures are more stable. A single tree can change its splits when the sample changes.' };
GUIDE.steps['mlpk/Predicted parameters'] = { notice: 'The two curves differ only in the CLint input.', q: 'What does the comparison of the two curves isolate?', a: 'The effect of the error in predicted CLint, because every other parameter is the same in both simulations.' };
GUIDE.steps['mlpk/Error propagation'] = { notice: 'At high CLint the same error in CLint changes AUC much less than at low CLint.', q: 'Why does a given CLint error change AUC less for a high-extraction compound?', a: 'Hepatic clearance approaches hepatic blood flow as extraction approaches 1, so it stops depending on CLint.' };
