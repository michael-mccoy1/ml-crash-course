"use strict";
/* Lesson 10, Intermediate steps: PBPK model structure, compartmental fits to public data, and population variability.
   Fixed numbers and their sources are given beside them. Helpers carry the prefix pkb. */

/* ---------- shared helpers ---------- */
/* y ticks for a log10 axis: whole powers of ten between lo and hi */
function pkbLogTicks(lo, hi){ var tk = [], e; for (e = lo; e <= hi; e++) tk.push({ v: e, t: e >= 0 ? String(Math.pow(10, e)) : Math.pow(10, e).toFixed(-e) }); return tk; }
function pkbNice(v, d){ if (!isFinite(v)) return 'n/a'; var a = Math.abs(v); return a >= 100 ? v.toFixed(0) : (a >= 10 ? v.toFixed(1) : (a >= 1 ? v.toFixed(d === undefined ? 2 : d) : v.toFixed(d === undefined ? 3 : d))); }
/* Nelder-Mead simplex minimiser. f takes an array and returns a number. Returns { x, fx, evals }. Standard coefficients: reflection 1, expansion 2, contraction 0.5, shrink 0.5. */
function pkbNM(f, x0, step, maxEval){
  var n = x0.length, S = [x0.slice()], i, j, evals = 0, fv;
  for (i = 0; i < n; i++) { var p = x0.slice(); p[i] += step; S.push(p); }
  fv = S.map(function(p){ evals++; return f(p); });
  function order(){ var idx = fv.map(function(_, k){ return k; }).sort(function(a, b){ return fv[a] - fv[b]; }); S = idx.map(function(k){ return S[k]; }); fv = idx.map(function(k){ return fv[k]; }); }
  while (evals < (maxEval || 2000)) {
    order(); if (Math.abs(fv[n] - fv[0]) < 1e-12 && evals > 100) break;
    var c = new Array(n).fill(0); for (i = 0; i < n; i++) for (j = 0; j < n; j++) c[j] += S[i][j] / n;
    var xr = c.map(function(v, j){ return v + (v - S[n][j]); }), fr = f(xr); evals++;
    if (fr < fv[0]) { var xe = c.map(function(v, j){ return v + 2 * (v - S[n][j]); }), fe = f(xe); evals++; if (fe < fr) { S[n] = xe; fv[n] = fe; } else { S[n] = xr; fv[n] = fr; } }
    else if (fr < fv[n - 1]) { S[n] = xr; fv[n] = fr; }
    else { var xc = c.map(function(v, j){ return v + 0.5 * ((fr < fv[n] ? xr[j] : S[n][j]) - v); }), fc = f(xc); evals++; if (fc < Math.min(fr, fv[n])) { S[n] = xc; fv[n] = fc; } else { for (i = 1; i <= n; i++) { S[i] = S[i].map(function(v, j){ return S[0][j] + 0.5 * (v - S[0][j]); }); fv[i] = f(S[i]); evals++; } } }
  }
  order(); return { x: S[0], fx: fv[0], evals: evals };
}
function pkbRound(a, d){ return a.map(function(v){ return +v.toFixed(d); }); }
var PKB = { fits: {}, sim: null };

/* ---------- 1. PBPK structure ---------- */
LESSONS.pk.add({ level: 'intermediate', short: 'PBPK structure', title: 'PBPK model structure',
  change: 'Select a test compound and an organ. Compare the simulation that uses the predicted CLint with the one that uses the measured value.',
  watch: 'A whole-body PBPK model represents each organ as a well-mixed compartment with a physiological volume and blood flow. Every tissue is perfusion-limited: the tissue and the venous blood leaving it are in instantaneous equilibrium, with the ratio set by a partition coefficient Kp. The gut drains to the liver through the portal vein, and the liver eliminates drug by the well-stirred model. The physiology is fixed (ICRP Publication 89 volumes and flows for a 73 kg adult male, scaled to 70 kg; tissue composition from Ruark 2014 and Poulin and Theil 2002). The Kp method and its unbound-in-tissue default are as implemented in the course engine and have not been checked against the primary paper. The drug-specific inputs come from a model. Hepatic intrinsic clearance is predicted by the boosted-trees model of lesson 01 and lesson 02, trained on the HLM training compounds. logP is computed from the structure. Fraction unbound, blood to plasma ratio, ka and F have no data in this course and are set to 1, with renal clearance 0. Each test compound is simulated twice with the same model, once with the predicted CLint and once with the measured one, for a 10 mg oral dose over 48 h. The difference between the two curves is the effect of the prediction error. The validation number is the fold error of AUC and Cmax, shown for the compound and over all quantified test compounds.',
  build: function(api){
    var ORG = [['brain', 'Brain'], ['heart', 'Heart'], ['kidney', 'Kidney'], ['muscle', 'Muscle'], ['adipose', 'Adipose'], ['skin', 'Skin'], ['bone', 'Bone'], ['rest', 'Rest'], ['gut', 'Gut'], ['liver', 'Liver']],
        PH = PKE.pbpk.reference, sel = 'liver', logAxis = false, cur = -1, DOSE = 10, TEND = 48, BW = 70, KA = 1, busy = false, agg = null;
    var dp = api.el('pn', 'left:28px;top:20px;width:904px;height:236px', '<h4>Whole-body model: organs in parallel between arterial and venous blood</h4><div class="pkd" style="margin-top:2px"></div>');
    var tp = api.el('pn', 'left:28px;top:264px;width:520px;height:256px;overflow:auto', '<h4 class="pkth"></h4><div class="pkt" style="font-size:11.5px;line-height:1.35;color:' + C.char + '"></div>');
    var rp = api.el('pn', 'left:564px;top:264px;width:368px;height:256px;overflow:auto', '<div class="pkr" style="font-size:11.5px;line-height:1.35;color:' + C.char + '"></div><canvas width="336" height="150" style="margin-top:2px"></canvas><div class="pkq" style="font-size:11.5px;line-height:1.35;color:' + C.char + '"></div>');
    var ctl = api.addControl('Test compound <select class="sel" style="width:250px"></select> <button type="button" class="btn sm ghost" data-next>Next compound</button> <span class="seg" data-ax><button type="button" data-v="0" class="on">Linear</button><button type="button" data-v="1">Log</button></span>');
    var sel0 = qs('.sel', ctl);
    function flowOf(n){ var F = PKE.pbpk.referenceScaled(BW).flows; return n === 'liver' ? F.liverTotal : F[n]; }
    function diagram(){
      var P = PKE.pbpk.referenceScaled(BW), V = P.volumes, F = P.flows, w = 62, x0 = 130, dx = 74, y0 = 74, h = 62, aY = 30, vY = 166, s = '', i;
      s += '<line x1="50" y1="' + aY + '" x2="' + (x0 + 9 * dx + w) + '" y2="' + aY + '" stroke="' + C.red + '" stroke-width="3"/><line x1="' + (x0 + 9 * dx + w) + '" y1="' + vY + '" x2="50" y2="' + vY + '" stroke="' + C.mid + '" stroke-width="3"/>';
      s += '<line x1="50" y1="' + vY + '" x2="50" y2="' + (y0 + h) + '" stroke="' + C.mid + '" stroke-width="2"/><line x1="50" y1="' + y0 + '" x2="50" y2="' + aY + '" stroke="' + C.red + '" stroke-width="2"/>';
      s += '<rect x="10" y="' + y0 + '" width="80" height="' + h + '" rx="4" fill="#fff" stroke="' + C.char + '" stroke-width="1.4"/><text x="50" y="' + (y0 + 20) + '" text-anchor="middle" font-weight="700" fill="' + C.char + '">Lung</text><text x="50" y="' + (y0 + 38) + '" text-anchor="middle" font-size="10.5" fill="' + C.char + '">V ' + pkbNice(V.lung, 1) + ' L</text><text x="50" y="' + (y0 + 52) + '" text-anchor="middle" font-size="10.5" fill="' + C.char + '">Q ' + pkbNice(F.cardiacOutput, 0) + ' L/h</text>';
      s += '<text x="' + (x0 - 6) + '" y="' + (aY - 8) + '" text-anchor="start" font-size="11" fill="' + C.red + '">arterial blood</text><text x="' + (x0 - 6) + '" y="' + (vY + 16) + '" text-anchor="start" font-size="11" fill="' + C.mid + '">venous blood</text>';
      ORG.forEach(function(o, k){
        var x = x0 + k * dx, cx = x + w / 2, on = o[0] === sel, vol = o[0] === 'liver' ? V.liver : V[o[0]], q = flowOf(o[0]);
        s += '<line x1="' + cx + '" y1="' + aY + '" x2="' + cx + '" y2="' + y0 + '" stroke="' + C.red + '" stroke-width="1.4" marker-end="url(#pkba)"/>';
        if (o[0] === 'liver') s += '<line x1="' + (cx + 14) + '" y1="' + aY + '" x2="' + (cx + 14) + '" y2="' + y0 + '" stroke="' + C.red + '" stroke-width="1.4" marker-end="url(#pkba)"/>';
        if (o[0] !== 'gut') s += '<line x1="' + cx + '" y1="' + (y0 + h) + '" x2="' + cx + '" y2="' + vY + '" stroke="' + C.mid + '" stroke-width="1.4"/>';
        s += '<g data-org="' + o[0] + '" style="cursor:pointer" tabindex="0"><rect x="' + x + '" y="' + y0 + '" width="' + w + '" height="' + h + '" rx="4" fill="' + (on ? '#FBE9E9' : '#fff') + '" stroke="' + (on ? C.red : C.char) + '" stroke-width="' + (on ? 2.4 : 1.4) + '"/><text x="' + cx + '" y="' + (y0 + 18) + '" text-anchor="middle" font-weight="700" fill="' + C.char + '">' + o[1] + '</text><text x="' + cx + '" y="' + (y0 + 36) + '" text-anchor="middle" font-size="10.5" fill="' + C.char + '">V ' + pkbNice(vol, vol < 0.1 ? 3 : 1) + ' L</text><text x="' + cx + '" y="' + (y0 + 51) + '" text-anchor="middle" font-size="10.5" fill="' + C.char + '">Q ' + pkbNice(q, q >= 100 ? 0 : 1) + ' L/h</text></g>';
        if (o[0] === 'liver') s += '<text x="' + (cx - 4) + '" y="' + (y0 + h + 15) + '" text-anchor="end" font-size="10.5" font-weight="700" fill="' + C.red + '">CLint (ML)</text>';
        if (o[0] === 'kidney') s += '<text x="' + (cx - 4) + '" y="' + (y0 + h + 15) + '" text-anchor="end" font-size="10.5" fill="' + C.mid + '">renal CL = 0</text>';
        if (o[0] === 'gut') s += '<line x1="' + (x + w) + '" y1="' + (y0 + h / 2) + '" x2="' + (x + dx) + '" y2="' + (y0 + h / 2) + '" stroke="' + C.char + '" stroke-width="1.6" marker-end="url(#pkba)"/><line x1="' + (cx - 16) + '" y1="' + (y0 + h + 13) + '" x2="' + (cx - 16) + '" y2="' + (y0 + h + 3) + '" stroke="' + C.char + '" stroke-width="1.6" marker-end="url(#pkba)"/><text x="' + (cx - 16) + '" y="' + (y0 + h + 24) + '" text-anchor="middle" font-size="10.5" fill="' + C.char + '">oral dose</text>';
      });
      qs('.pkd', dp).innerHTML = '<svg viewBox="0 0 880 200" width="100%" style="max-width:880px;display:block;margin:0 auto" font-family="Aptos, Segoe UI, sans-serif" font-size="12"><defs><marker id="pkba" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="' + C.char + '"/></marker></defs>' + s + '</svg>';
    }
    function sim(e){
      PKB.psims = PKB.psims || {}; if (PKB.psims[e.i]) return PKB.psims[e.i]; var ds = MP.ds, o = {}, kinds = { measured: ds.Y[e.i], predicted: MP.F[e.i] }, logp = ds.COL.logP[e.i];
      Object.keys(kinds).forEach(function(k){
        var m = PKE.pbpk.model({ logP: logp, fup: 1, bp: 1, clintU: PKE.pbpk.clintUFromHLM(kinds[k], BW, 1), clr: 0, ka: KA, F: 1, bw: BW }), r = PKE.pbpk.simulate(m, { route: 'oral', dose: DOSE }, TEND, null, 200), cm = 0;
        r.plasma.forEach(function(c){ if (c > cm) cm = c; });
        o[k] = { t: r.t, c: r.plasma, auc: r.aucPlasma[r.aucPlasma.length - 1], cmax: cm, log10: kinds[k], logP: logp };
      });
      return (PKB.psims[e.i] = o);
    }
    function aggregate(){
      if (agg) return agg; var ds = MP.ds, ids = ds.TEST.filter(function(i){ return !ds.cens[i]; }), fe = ids.map(function(i){ var a = PKE.fromHLM(ds.Y[i], { bw: BW }).clhPlasmaLh, b = PKE.fromHLM(MP.F[i], { bw: BW }).clhPlasmaLh, r = b / a; return r >= 1 ? r : 1 / r; }).sort(function(x, y){ return x - y; });
      return (agg = { n: fe.length, med: fe[Math.floor(fe.length / 2)], p90: fe[Math.floor(fe.length * 0.9)], w2: fe.filter(function(x){ return x <= 2; }).length / fe.length });
    }
    function table(){
      var e = MP.list[cur], r = PKB.psims && PKB.psims[e.i]; if (!r) return;
      var badge = function(t, c){ return '<span style="display:inline-block;padding:0 5px;border-radius:3px;font-size:10px;font-weight:700;color:#fff;background:' + c + '">' + t + '</span>'; };
      var rows = [['logP', badge('computed', C.mid), 'RDKit Crippen', r.measured.logP.toFixed(2)], ['log10 CLint (mL/min/kg)', badge('ML prediction', C.red), 'Boosted trees. Measured ' + r.measured.log10.toFixed(2), '<b style="color:' + C.red + '">' + r.predicted.log10.toFixed(2) + '</b>'], ['fup, blood to plasma', badge('assumed', C.mid), 'No data in this course', '1, 1'], ['ka (per h), F', badge('assumed', C.mid), 'No data in this course', '1, 1'], ['Renal clearance', badge('assumed', C.mid), 'Set to 0', '0'], ['Volumes, flows, tissues', badge('fixed', C.char), 'Physiology, 70 kg adult', 'diagram']];
      qs('.pkth', tp).textContent = 'Inputs for compound ' + (e.i + 1);
      qs('.pkt', tp).innerHTML = '<table style="width:100%;border-collapse:collapse;table-layout:fixed"><colgroup><col style="width:29%"><col style="width:22%"><col style="width:35%"><col style="width:14%"></colgroup><tr style="font-weight:700;font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;color:' + C.mid + '"><td>Input</td><td>Source</td><td>Detail</td><td style="text-align:right">Value</td></tr>' + rows.map(function(x){ return '<tr style="border-top:1px solid ' + C.silver + ';vertical-align:middle"><td style="padding:3px 6px 3px 0;font-weight:600">' + x[0] + '</td><td style="padding:3px 6px 3px 0">' + x[1] + '</td><td style="padding:3px 6px 3px 0">' + x[2] + '</td><td style="padding:3px 0;text-align:right">' + x[3] + '</td></tr>'; }).join('') + '</table>';
    }
    function details(){
      var e = MP.list[cur], p = PKB.psims && PKB.psims[e.i], P = PKE.pbpk.referenceScaled(BW), kp = PKE.pbpk.kpPT(p ? p.measured.logP : 2, 1), v = sel === 'liver' ? P.volumes.liver : P.volumes[sel], q = flowOf(sel), nm = ORG.filter(function(o){ return o[0] === sel; })[0][1], k = kp[sel], tau = v * k / q;
      qs('.pkr', rp).innerHTML = '<b>' + nm + '</b>: V ' + pkbNice(v, 2) + ' L, Q ' + pkbNice(q, 1) + ' L/h, Kp ' + pkbNice(k, 2) + ', V Kp / Q ' + (tau < 1 ? (tau * 60).toFixed(0) + ' min' : tau.toFixed(1) + ' h') + '.';
    }
    function chart(){
      var el = qs('canvas', rp), e = MP.list[cur], r = PKB.psims && PKB.psims[e.i]; if (!r) return;
      var f = function(v){ return v > 0 ? Math.log10(Math.max(v * 1000, 0.1)) : -1; }, sM = r.measured.t.map(function(t, j){ return [t, logAxis ? f(r.measured.c[j]) : r.measured.c[j] * 1000]; }), sP = r.predicted.t.map(function(t, j){ return [t, logAxis ? f(r.predicted.c[j]) : r.predicted.c[j] * 1000]; }), o = { xl: 'time after dose (h)', yl: logAxis ? 'ug/L, log scale' : 'ug/L', x0: 0, x1: TEND };
      if (logAxis) { var mx = Math.max.apply(null, sM.concat(sP).map(function(q){ return q[1]; })), hi = Math.ceil(mx); o.y0 = -1; o.y1 = hi; o.yticks = []; for (var tv = -1; tv <= hi; tv++) o.yticks.push({ v: tv, t: tv < 0 ? '0.1' : String(Math.pow(10, tv)) }); } else o.y0 = 0;
      lineChart(el, [{ pts: sM, color: C.char, w: 2.4, nodot: true, label: 'measured CLint' }, { pts: sP, color: C.red, w: 2.4, nodot: true, label: 'predicted CLint' }], o);
      var fa = r.predicted.auc / r.measured.auc, fc = r.predicted.cmax / r.measured.cmax, A = aggregate();
      qs('.pkq', rp).innerHTML = '<b>AUC fold error ' + (fa >= 1 ? fa : 1 / fa).toFixed(2) + '</b> (' + (fa >= 1 ? 'over' : 'under') + '-predicted), <b>Cmax fold error ' + (fc >= 1 ? fc : 1 / fc).toFixed(2) + '</b>. Over ' + A.n + ' quantified test compounds, the AUC fold error has median ' + A.med.toFixed(2) + ' and 90th percentile ' + A.p90.toFixed(2) + ', and ' + (100 * A.w2).toFixed(0) + '% are within 2-fold.';
    }
    function all(){ diagram(); table(); details(); chart(); }
    async function pick(k){
      if (busy) return; cur = k; sel0.value = String(k); busy = true; var e = MP.list[cur];
      PKB.psims = PKB.psims || {};
      if (!PKB.psims[e.i]) { api.busy('Simulating'); await tick(); await tick(); try { sim(e); } finally { api.idle(); } }
      busy = false; all();
    }
    async function render(){
      await mpReady(api);
      if (cur < 0) { var ae = MP.list.map(function(e, k){ return [Math.abs(MP.F[e.i] - MP.ds.Y[e.i]), k]; }).sort(function(a, b){ return a[0] - b[0]; }); cur = ae[Math.floor(ae.length / 2)][1]; }
      if (!sel0.options.length) MP.list.forEach(function(e, k){ var o = document.createElement('option'); o.value = k; o.textContent = 'Compound ' + (e.i + 1) + ', measured log10 CLint ' + MP.ds.Y[e.i].toFixed(2); sel0.appendChild(o); });
      await pick(cur);
    }
    sel0.addEventListener('change', function(){ pick(+sel0.value); });
    qs('[data-next]', ctl).addEventListener('click', function(){ if (MP.list) pick((cur + 1) % MP.list.length); });
    qsa('[data-ax] button', ctl).forEach(function(b){ b.addEventListener('click', function(){ logAxis = +b.dataset.v === 1; qsa('[data-ax] button', ctl).forEach(function(x){ x.classList.toggle('on', x === b); }); if (cur >= 0) chart(); }); });
    dp.addEventListener('click', function(e){ var g = e.target.closest ? e.target.closest('[data-org]') : null; if (!g || cur < 0) return; sel = g.dataset.org; diagram(); details(); });
    dp.addEventListener('keydown', function(e){ if (e.key !== 'Enter') return; var g = e.target.closest ? e.target.closest('[data-org]') : null; if (g && cur >= 0) { sel = g.dataset.org; diagram(); details(); } });
    async function play(tok){
      sel = 'liver'; await render(); api.cap('Organs sit in parallel between the arterial and venous blood. The physiology is fixed.'); await api.sleep(3400); api.checkTok(tok);
      sel = 'adipose'; diagram(); details(); api.cap('Each tissue has a volume, a blood flow and a partition coefficient Kp. Adipose has a large volume and a small flow, so it equilibrates slowly.'); await api.sleep(3800); api.checkTok(tok);
      sel = 'liver'; diagram(); details(); api.cap('The drug-specific input that has data in this course is hepatic CLint. It is predicted by a model and compared with the measured value.'); await api.sleep(3800); api.checkTok(tok);
      logAxis = true; qsa('[data-ax] button', ctl).forEach(function(x){ x.classList.toggle('on', +x.dataset.v === 1); }); chart(); api.cap('The two simulations differ only in the CLint input. On a log axis the terminal slopes show the difference in clearance.'); await api.sleep(3800);
      logAxis = false; qsa('[data-ax] button', ctl).forEach(function(x){ x.classList.toggle('on', +x.dataset.v === 0); }); chart(); api.cap(null);
    }
    return { render: render, play: play };
  } });
GUIDE.steps['pk/PBPK structure'] = { notice: 'Only hepatic CLint is predicted by a model in this step. The other drug-specific inputs are assumed.', q: 'How is the effect of a CLint prediction error on the simulated exposure measured?', a: 'The compound is simulated with the predicted and with the measured CLint, and the fold error of AUC and Cmax is reported, for the compound and over all quantified test compounds.' };

/* ---------- 2. Compartmental fit ---------- */
LESSONS.pk.add({ level: 'intermediate', short: 'Compartmental fit', title: 'NCA and compartmental fits',
  change: 'Select a data set, a subject and a model. Compare the non-compartmental and fitted estimates.',
  watch: 'Two ways to extract parameters from a measured profile. Non-compartmental analysis (NCA) uses the trapezoidal rule and the terminal slope and assumes no model. A compartmental fit chooses a model and minimises the difference between the model and the data. The fits here minimise the sum of squared differences of log concentration, which approximates a proportional error model. Both use the same public data: Theoph (oral theophylline, 12 subjects, 132 observations) and Indometh (intravenous indomethacin, 6 subjects, 66 observations). The Indometh dose is not verified in the public sources, so Indometh results are given for an assumed dose of 1 mg and all clearances and volumes scale in proportion to the true dose. A boosted-trees model (lesson 02) is trained to predict the NCA CL/F and V/F of a Theoph subject from weight and dose per kg. Each subject is held out in turn and the model is trained on the other 11 (leave-one-out). Its error is compared with two baselines: the mean of the other subjects, and a log-linear fit on weight. Twelve subjects and two covariates are too few to train a model, and the comparison shows how much the covariates add to the mean. Indometh has no recorded covariates, so no covariate model can be fitted to it. Lesson 11 covers models that predict parameters before any profile exists (Predicted parameters as PBPK inputs) and sequence models that replace the compartmental model (Sequence models for profiles).',
  build: function(api){
    var DATA = null, ds = 'theoph', sub = 0, mdl = 1;
    var ap = api.el('pn', 'left:28px;top:20px;width:292px;height:500px;overflow:auto', '<h4>Estimates</h4><div class="pkf" style="font-size:11.5px;line-height:1.4;color:' + C.char + '"></div>');
    var cp = api.el('pn', 'left:336px;top:20px;width:596px;height:258px', '<h4 class="pkh1"></h4><canvas width="568" height="214"></canvas>');
    var rs = api.el('pn', 'left:336px;top:288px;width:292px;height:232px', '<h4>Residuals, ln(observed / predicted)</h4><canvas width="262" height="190"></canvas>');
    var ag = api.el('pn', 'left:640px;top:288px;width:292px;height:232px', '<h4 class="pkh3"></h4><canvas width="262" height="190"></canvas>');
    var ctl = api.addControl('Data <span class="seg" data-ds><button type="button" data-v="theoph" class="on">Theoph, oral</button><button type="button" data-v="indometh">Indometh, IV</button></span> Subject <select class="sbj" style="width:70px"></select> <span class="mwrap">Model <span class="seg" data-md><button type="button" data-v="1" class="on">One compartment</button><button type="button" data-v="2">Two compartments</button></span></span>');
    function subj(){ return DATA[ds].subjects[sub]; }
    function usable(s){ var t = [], c = []; for (var i = 0; i < s.t.length; i++) if (s.t[i] > 0 && s.conc[i] > 0) { t.push(s.t[i]); c.push(s.conc[i]); } return { t: t, c: c }; }
    function dose(s){ return ds === 'theoph' ? s.doseMg : 1; }
    function ncaOf(s){
      var t = s.t.slice(), c = s.conc.slice();
      if (ds === 'indometh') { var c0 = c[0] * Math.pow(c[0] / c[1], t[0] / (t[1] - t[0])); t.unshift(0); c.unshift(c0); }
      return PKE.nca(t, c, dose(s), { nTerminal: 3 });
    }
    function modelFn(kind, th, d){
      if (kind === 'oral1') { var ka = Math.exp(th[0]), ke = Math.exp(th[1]), V = Math.exp(th[2]); return PKE.oneComp({ dose: d, V: V, CL: ke * V, ka: ka, F: 1, route: 'oral' }).conc; }
      if (kind === 'iv1') { var V1 = Math.exp(th[0]), k = Math.exp(th[1]); return PKE.oneComp({ dose: d, V: V1, CL: k * V1, route: 'iv' }).conc; }
      var v1 = Math.exp(th[0]), k10 = Math.exp(th[1]), k12 = Math.exp(th[2]), k21 = Math.exp(th[3]); return PKE.twoComp({ dose: d, V1: v1, CL: k10 * v1, Q: k12 * v1, V2: v1 * k12 / k21 }).conc;
    }
    function fit(s, kind){
      var u = usable(s), d = dose(s), n = ncaOf(s), ke0 = isFinite(n.lambdaZ) && n.lambdaZ > 0 ? n.lambdaZ : 0.2, loss = function(th){ var f = modelFn(kind, th, d), e = 0, i; for (i = 0; i < u.t.length; i++) { var p = Math.max(f(u.t[i]), 1e-9), r = Math.log(u.c[i]) - Math.log(p); e += r * r; } return isFinite(e) ? e : 1e9; }, x0, best = null, tries;
      if (kind === 'oral1') tries = [[Math.log(1.5), Math.log(ke0), Math.log(d / (n.cmax || 1) * 1.2)], [Math.log(0.5), Math.log(ke0), Math.log(d / (n.cmax || 1))], [Math.log(4), Math.log(ke0), Math.log(d / (n.cmax || 1) * 1.5)]];
      else if (kind === 'iv1') tries = [[Math.log(d / (s.conc[0] * 1.3)), Math.log(ke0)], [Math.log(d / (s.conc[0] * 2)), Math.log(ke0 * 2)]];
      else tries = [[Math.log(d / (s.conc[0] * 1.2)), Math.log(ke0), Math.log(1), Math.log(1)], [Math.log(d / (s.conc[0] * 1.5)), Math.log(ke0), Math.log(2), Math.log(0.6)], [Math.log(d / (s.conc[0] * 1.1)), Math.log(ke0 * 0.7), Math.log(0.5), Math.log(1.5)]];
      tries.forEach(function(x){ var r = pkbNM(loss, x, 0.5, 1500); r = pkbNM(loss, r.x, 0.2, 1500); if (!best || r.fx < best.fx) best = r; });
      return { th: best.x, sse: best.fx, n: u.t.length, kind: kind, d: d };
    }
    function summary(f){
      var th = f.th, d = f.d, e = Math.exp;
      if (f.kind === 'oral1') { var ka = e(th[0]), ke = e(th[1]), V = e(th[2]); return { cl: ke * V, v: V, thalf: Math.LN2 / ke, auc: d / (ke * V), extra: 'ka ' + pkbNice(ka, 2) + ' per h' }; }
      if (f.kind === 'iv1') { var V1 = e(th[0]), k = e(th[1]); return { cl: k * V1, v: V1, thalf: Math.LN2 / k, auc: d / (k * V1), extra: '' }; }
      var v1 = e(th[0]), k10 = e(th[1]), k12 = e(th[2]), k21 = e(th[3]), s = k10 + k12 + k21, disc = Math.sqrt(s * s - 4 * k10 * k21), beta = (s - disc) / 2, cl = k10 * v1;
      return { cl: cl, v: cl / beta, thalf: Math.LN2 / beta, auc: d / cl, extra: 'Vss ' + pkbNice(v1 * (1 + k12 / k21), 2) + ' L, alpha half-life ' + pkbNice(Math.LN2 / ((s + disc) / 2), 2) + ' h' };
    }
    function ncaTheoph(s){ return PKE.nca(s.t.slice(), s.conc.slice(), s.doseMg, { nTerminal: 3 }); }
    function loo(){
      if (PKB.loo) return PKB.loo;
      var S = DATA.theoph.subjects, n = S.length, nc = S.map(ncaTheoph), X = S.map(function(s){ return [s.wt, s.doseMgPerKg]; }), tg = { cl: nc.map(function(q){ return Math.log(q.clF); }), v: nc.map(function(q){ return Math.log(q.vzF); }) }, out = { n: n, nca: nc, pred: { trees: { cl: [], v: [] }, mean: { cl: [], v: [] }, allo: { cl: [], v: [] } } }, i, j;
      ['cl', 'v'].forEach(function(key){
        for (i = 0; i < n; i++) {
          var tr = [], Xtr = [], ytr = [];
          for (j = 0; j < n; j++) if (j !== i) { Xtr.push(X[j]); ytr.push(tg[key][j]); }
          out.pred.trees[key].push(mlbBoost(Xtr, ytr, { rounds: 100, lr: 0.1, depth: 2, minLeaf: 2 }).predict(X[i]));
          var m = mean(ytr); out.pred.mean[key].push(m);
          var xl = Xtr.map(function(r){ return Math.log(r[0] / 70); }), xm = mean(xl), sxy = 0, sxx = 0;
          xl.forEach(function(v, k){ sxy += (v - xm) * (ytr[k] - m); sxx += (v - xm) * (v - xm); });
          var b = sxx > 0 ? sxy / sxx : 0; out.pred.allo[key].push(m + b * (Math.log(X[i][0] / 70) - xm));
        }
      });
      out.tg = tg;
      out.stat = function(model, key){ var e = out.pred[model][key].map(function(v, k){ return Math.abs(v - tg[key][k]); }); return { gmfe: Math.exp(mean(e)), within2: e.filter(function(x){ return x <= Math.LN2; }).length }; };
      return (PKB.loo = out);
    }
    function getFit(i, kind){ var key = ds + i + kind; if (!PKB.fits[key]) PKB.fits[key] = fit(DATA[ds].subjects[i], kind); return PKB.fits[key]; }
    function kinds(){ return ds === 'theoph' ? ['oral1'] : (mdl === 1 ? ['iv1'] : ['iv1', 'iv2']); }
    function draw(){
      if (!DATA) return; var s = subj(), u = usable(s), n = ncaOf(s), ks = kinds(), tEnd = Math.ceil(Math.max.apply(null, s.t) * 1.05), cmaxD = Math.max.apply(null, s.conc), cmin = Math.min.apply(null, u.c) / 3, i, series = [], cols = { oral1: C.red, iv1: C.red, iv2: C.char };
      var lo = Math.floor(Math.log10(cmin)), hi = Math.ceil(Math.log10(cmaxD * 1.3));
      ks.forEach(function(kd){ var f = getFit(sub, kd), fn = modelFn(kd, f.th, f.d), pts = []; for (i = 0; i <= 200; i++) { var t = tEnd * i / 200 + 1e-4; pts.push([t, Math.log10(Math.max(fn(t), Math.pow(10, lo)))]); } series.push({ pts: pts, color: cols[kd], w: 2.6, nodot: true, label: kd === 'oral1' ? 'one compartment, first-order absorption' : (kd === 'iv1' ? 'one compartment' : 'two compartments') }); });
      qs('.pkh1', cp).textContent = (ds === 'theoph' ? 'Theoph subject ' : 'Indometh subject ') + s.id + ': concentration against time';
      var g = lineChart(qs('canvas', cp), series, { xl: 'time (h)', yl: 'concentration (mg/L, log scale)', x0: 0, x1: tEnd, y0: lo, y1: hi, yticks: pkbLogTicks(lo, hi) });
      if (g && g.sx) { g.x.fillStyle = C.char; s.t.forEach(function(t, k){ if (s.conc[k] > 0) { g.x.beginPath(); g.x.arc(g.sx(t), g.sy(Math.log10(s.conc[k])), 4, 0, 6.3); g.x.fill(); } }); }
      /* residuals */
      var r = cv(qs('canvas', rs)), box = { l: 40, r: r.w - 12, t: 14, b: r.h - 36 }, rmax = 0.2, resid = {};
      ks.forEach(function(kd){ var f = getFit(sub, kd), fn = modelFn(kd, f.th, f.d); resid[kd] = u.t.map(function(t, k){ var v = Math.log(u.c[k]) - Math.log(Math.max(fn(t), 1e-9)); rmax = Math.max(rmax, Math.abs(v)); return v; }); });
      rmax = Math.ceil(rmax * 10) / 10;
      var gr = lineChart(qs('canvas', rs), [{ pts: [[0, 0], [tEnd, 0]], color: C.bg, w: 1.4, nodot: true, dash: [4, 4] }], { xl: 'time (h)', yl: 'ln residual', x0: 0, x1: tEnd, y0: -rmax, y1: rmax });
      if (gr && gr.sx) ks.forEach(function(kd){ gr.x.fillStyle = cols[kd]; u.t.forEach(function(t, k){ gr.x.beginPath(); gr.x.arc(gr.sx(t), gr.sy(resid[kd][k]), 3.4, 0, 6.3); gr.x.fill(); }); });
      /* table */
      var fmt = function(v){ return pkbNice(v, 2); }, unit = ds === 'theoph' ? '/F' : '', nm = { oral1: 'One compartment', iv1: 'One compartment', iv2: 'Two compartments' };
      var qs2 = ks.map(function(kd){ var f = getFit(sub, kd); return { kd: kd, q: summary(f), rm: Math.sqrt(f.sse / f.n) }; });
      var head = '<tr style="font-weight:700;font-size:10px;text-transform:uppercase;letter-spacing:.05em;color:' + C.mid + '"><td></td><td style="text-align:right;padding:2px 3px">NCA</td>' + qs2.map(function(o){ return '<td style="text-align:right;padding:2px 3px;color:' + cols[o.kd] + '">' + nm[o.kd] + '</td>'; }).join('') + '</tr>';
      var line = function(l, a, fn){ return '<tr style="border-top:1px solid ' + C.silver + '"><td style="padding:2px 3px 2px 0">' + l + '</td><td style="text-align:right;font-weight:700;padding:2px 3px">' + a + '</td>' + qs2.map(function(o){ return '<td style="text-align:right;font-weight:700;padding:2px 3px;color:' + cols[o.kd] + '">' + fn(o) + '</td>'; }).join('') + '</tr>'; };
      var html = '<div class="ol">Subject ' + s.id + ', ' + (ds === 'theoph' ? 'oral' : 'IV') + '</div><table style="width:100%;border-collapse:collapse;font-size:11px">' + head + line('AUC0-inf (mg h/L)', fmt(n.aucInf), function(o){ return fmt(o.q.auc); }) + line('CL' + unit + ' (L/h)', fmt(n.clF), function(o){ return fmt(o.q.cl); }) + line('V' + unit + ' (L)', fmt(n.vzF), function(o){ return fmt(o.q.v); }) + line('Half-life (h)', fmt(n.thalf), function(o){ return fmt(o.q.thalf); }) + line('Residual SD of ln c', 'n/a', function(o){ return o.rm.toFixed(3); }) + '</table>' + qs2.map(function(o){ return o.q.extra ? '<div style="font-size:10.5px;color:' + C.mid + '">' + nm[o.kd] + ': ' + o.q.extra + '</div>' : ''; }).join('');
      if (ds === 'theoph') {
        var L = loo(), tr = L.pred.trees, nq = L.nca[sub], fe = function(pr, ob){ var r = Math.exp(pr - ob); return r >= 1 ? r : 1 / r; }, sT = L.stat('trees', 'cl'), sM = L.stat('mean', 'cl'), sA = L.stat('allo', 'cl'), vT = L.stat('trees', 'v'), vM = L.stat('mean', 'v'), vA = L.stat('allo', 'v');
        html += '<div class="ol" style="margin-top:10px">ML, boosted trees, leave-one-out</div><table style="width:100%;border-collapse:collapse">' + [['CL/F predicted (L/h)', fmt(Math.exp(tr.cl[sub])) + ', fold error ' + fe(tr.cl[sub], L.tg.cl[sub]).toFixed(2)], ['V/F predicted (L)', fmt(Math.exp(tr.v[sub])) + ', fold error ' + fe(tr.v[sub], L.tg.v[sub]).toFixed(2)]].map(function(x){ return '<tr style="border-top:1px solid ' + C.silver + '"><td style="padding:2px 4px 2px 0">' + x[0] + '</td><td style="text-align:right;font-weight:700;color:' + C.red + '">' + x[1] + '</td></tr>'; }).join('') + '</table>' +
          '<div style="font-size:11px;line-height:1.35;color:' + C.char + ';margin-top:6px">Geometric mean fold error over the 12 subjects, against the NCA value:<table style="width:100%;border-collapse:collapse;margin-top:3px"><tr style="font-weight:700;color:' + C.mid + '"><td></td><td style="text-align:right">CL/F</td><td style="text-align:right">V/F</td></tr><tr style="border-top:1px solid ' + C.silver + '"><td>Boosted trees</td><td style="text-align:right">' + sT.gmfe.toFixed(2) + '</td><td style="text-align:right">' + vT.gmfe.toFixed(2) + '</td></tr><tr style="border-top:1px solid ' + C.silver + '"><td>Mean of the other 11</td><td style="text-align:right">' + sM.gmfe.toFixed(2) + '</td><td style="text-align:right">' + vM.gmfe.toFixed(2) + '</td></tr><tr style="border-top:1px solid ' + C.silver + '"><td>Log-linear on weight</td><td style="text-align:right">' + sA.gmfe.toFixed(2) + '</td><td style="text-align:right">' + vA.gmfe.toFixed(2) + '</td></tr></table></div>';
      } else html += '<div style="font-size:11px;line-height:1.35;color:' + C.char + ';margin-top:10px">No covariates are recorded for Indometh, so no covariate model can be fitted. The dose is assumed to be 1 mg.</div>';
      qs('.pkf', ap).innerHTML = html;
      /* agreement across subjects */
      var N = DATA[ds].subjects.length, pairs = [], mx = 0;
      for (i = 0; i < N; i++) { var si = DATA[ds].subjects[i], ni = ncaOf(si), fk = ds === 'theoph' ? 'oral1' : (mdl === 1 ? 'iv1' : 'iv2'), fq = summary(getFit(i, fk)); pairs.push([ni.clF, fq.cl, si.id]); mx = Math.max(mx, ni.clF, fq.cl); if (ds === 'theoph') mx = Math.max(mx, Math.exp(loo().pred.trees.cl[i])); }
      qs('.pkh3', ag).textContent = 'CL' + unit + ' across ' + N + ' subjects';
      var a = cv(qs('canvas', ag)), bx = { l: 40, r: a.w - 12, t: 12, b: a.h - 36 }, top = Math.ceil(mx * 1.1);
      var sxa = function(v){ return bx.l + v / top * (bx.r - bx.l); }, sya = function(v){ return bx.b - v / top * (bx.b - bx.t); };
      axes(a, bx, [0, top], [0, top], 'CL' + unit + ' by NCA (L/h)', 'CL' + unit + ' by fit (L/h)', 4);
      a.x.strokeStyle = C.bg; a.x.setLineDash([4, 4]); a.x.beginPath(); a.x.moveTo(sxa(0), sya(0)); a.x.lineTo(sxa(top), sya(top)); a.x.stroke(); a.x.setLineDash([]);
      pairs.forEach(function(p){ a.x.fillStyle = p[2] === s.id ? C.red : C.char; a.x.beginPath(); a.x.arc(sxa(p[0]), sya(p[1]), p[2] === s.id ? 5.5 : 3.8, 0, 6.3); a.x.fill(); });
      if (ds === 'theoph') { var Lp = loo(); a.x.strokeStyle = C.red; a.x.lineWidth = 1.6; Lp.nca.forEach(function(q, k){ var xv = q.clF, yv = Math.exp(Lp.pred.trees.cl[k]); a.x.strokeRect(sxa(xv) - 3.5, sya(Math.min(yv, top)) - 3.5, 7, 7); }); a.x.fillStyle = C.char; a.x.font = '10.5px Aptos, "Segoe UI", sans-serif'; a.x.textAlign = 'left'; a.x.fillText('filled: compartmental fit', bx.l + 6, bx.t + 12); a.x.fillText('squares: boosted trees, leave-one-out', bx.l + 6, bx.t + 25); }
    }
    function fillSubjects(){ var sel = qs('.sbj', ctl); sel.innerHTML = DATA[ds].subjects.map(function(s, i){ return '<option value="' + i + '">' + s.id + '</option>'; }).join(''); sel.value = String(sub); qs('.mwrap', ctl).style.display = ds === 'indometh' ? '' : 'none'; }
    async function render(){
      if (!DATA) { api.busy('Loading the public PK data'); try { DATA = await loadJSON('pk_public'); } finally { api.idle(); } fillSubjects(); }
      api.busy('Fitting'); await tick(); try { draw(); } finally { api.idle(); }
    }
    qsa('[data-ds] button', ctl).forEach(function(b){ b.addEventListener('click', async function(){ ds = b.dataset.v; sub = 0; qsa('[data-ds] button', ctl).forEach(function(x){ x.classList.toggle('on', x === b); }); fillSubjects(); await render(); }); });
    qsa('[data-md] button', ctl).forEach(function(b){ b.addEventListener('click', async function(){ mdl = +b.dataset.v; qsa('[data-md] button', ctl).forEach(function(x){ x.classList.toggle('on', x === b); }); await render(); }); });
    qs('.sbj', ctl).addEventListener('change', async function(){ sub = +this.value; await render(); });
    async function play(tok){
      await render(); ds = 'theoph'; sub = 0; fillSubjects(); qsa('[data-ds] button', ctl).forEach(function(x){ x.classList.toggle('on', x.dataset.v === 'theoph'); }); await render();
      api.cap('NCA and the one-compartment fit give similar clearance for an oral profile when the terminal phase is well sampled.'); await api.sleep(3800);
      for (var k = 1; k < 4; k++) { api.checkTok(tok); sub = k; fillSubjects(); await render(); await api.sleep(1400); }
      api.cap('Across subjects the two estimates track each other. The squares are boosted-trees predictions of CL/F for each subject, made without that subject in the training set.'); await api.sleep(4200);
      ds = 'indometh'; sub = 0; mdl = 1; fillSubjects(); qsa('[data-ds] button', ctl).forEach(function(x){ x.classList.toggle('on', x.dataset.v === 'indometh'); }); qsa('[data-md] button', ctl).forEach(function(x){ x.classList.toggle('on', x.dataset.v === '1'); }); await render();
      api.cap('An IV profile with a distribution phase: the one-compartment residuals have a pattern.'); await api.sleep(3800);
      mdl = 2; qsa('[data-md] button', ctl).forEach(function(x){ x.classList.toggle('on', x.dataset.v === '2'); }); await render(); api.cap('A second compartment removes the pattern in the residuals.'); await api.sleep(3600); api.cap(null);
    }
    return { render: render, play: play };
  } });
GUIDE.steps['pk/Compartmental fit'] = { notice: 'With 12 subjects, a leave-one-out error is the only error estimate available for a covariate model.', q: 'How is the error of a model that predicts CL/F from covariates measured with 12 subjects?', a: 'Each subject is predicted by a model trained on the other 11. The errors are collected over all subjects and compared with the mean of the other subjects as a baseline.' };

/* ---------- 3. Population variability ---------- */
LESSONS.pk.add({ level: 'intermediate', short: 'Population variability', title: 'Population variability',
  change: 'Set the variability, the weight distribution and the number of training subjects. Switch between the allometric fit and boosted trees.',
  watch: 'A population model describes how a parameter differs between individuals: a typical value that depends on covariates, multiplied by a lognormal random effect with standard deviation omega. Here the true clearance of a 70 kg subject is 6 L/h, scaled by (weight / 70) to the power 0.75, with omega of CL as set. A covariate model for the typical value is learned from simulated subjects whose observed ln CL carries the random effect and an estimation error of SD 0.15. Two learners are compared: a log-linear fit, which estimates the allometric exponent, and boosted trees on weight (lesson 02). Both are scored on 400 subjects not used for fitting, against the observed ln CL and against the true typical value. The AUC and Cmax distributions are then simulated for those 400 subjects from the selected model, with the held-out residual SD as its random effect. The weight exponent of 0.75 is a convention (Anderson and Holford 2008, Annu Rev Pharmacol Toxicol 48:303-332), and the typical values are illustrative, not data. Variability does not shrink with more training subjects. The error of the learned typical value does.',
  build: function(api){
    var CLT = 6, VT = 40, KAT = 1, DOSE = 100, TEND = 24, EPS = 0.15, NTEST = 400, A = 0.75;
    var lp = api.el('pn', 'left:28px;top:20px;width:236px;height:500px;overflow:auto', '<h4>Population</h4>' + optRange('pk3ocl', 'Omega of CL (SD of ln CL)', 0, 0.8, 0.05, 0.3) + optRange('pk3ov', 'Omega of V', 0, 0.6, 0.05, 0.2) + optRange('pk3oka', 'Omega of ka', 0, 0.8, 0.05, 0.4) + optRange('pk3wm', 'Median weight (kg)', 50, 100, 1, 70) + optRange('pk3ws', 'Weight SD (kg)', 5, 25, 1, 12) + optRange('pk3n', 'Training subjects', 20, 500, 10, 100) + '<div class="og"><div class="ol">Covariate model</div>' + seg([['0', 'Allometric fit'], ['1', 'Boosted trees']], '1', 'data-md') + '</div><div class="og"><div class="ol">Typical values (illustrative)</div><div class="hint">70 kg subject, oral dose 100 mg: CL 6 L/h, V 40 L, ka 1 per h, F 1. These are not data.</div></div>');
    var sp = api.el('pn', 'left:280px;top:20px;width:652px;height:262px', '<h4>Clearance against weight: training subjects and fitted typical values</h4><canvas width="622" height="216"></canvas>');
    var h1 = api.el('pn', 'left:280px;top:292px;width:206px;height:230px', '<h4>AUC</h4><canvas width="176" height="140"></canvas><div class="s1" style="font-size:11.5px;line-height:1.35;color:' + C.char + '"></div>');
    var h2 = api.el('pn', 'left:496px;top:292px;width:206px;height:230px', '<h4>Cmax</h4><canvas width="176" height="140"></canvas><div class="s2" style="font-size:11.5px;line-height:1.35;color:' + C.char + '"></div>');
    var tb = api.el('pn', 'left:712px;top:292px;width:220px;height:230px', '<h4>Error on 400 test subjects</h4><div class="s3" style="font-size:11.5px;line-height:1.35;color:' + C.char + '"></div>');
    var md = 1;
    function P(){ return { ocl: +$('pk3ocl').value, ov: +$('pk3ov').value, oka: +$('pk3oka').value, wm: +$('pk3wm').value, ws: +$('pk3ws').value, n: +$('pk3n').value }; }
    function wts(n, seed, p){ var z = mlbNormal(seed), out = [], i; for (i = 0; i < n; i++) out.push(Math.min(120, Math.max(35, p.wm + p.ws * z()))); return out; }
    function pct(a, q){ var s = a.slice().sort(function(x, y){ return x - y; }), k = (s.length - 1) * q, lo = Math.floor(k), hi = Math.ceil(k); return s[lo] + (s[hi] - s[lo]) * (k - lo); }
    function hist(el, vals, bins, color, xl){
      var g = cv(el), lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals), w = (hi - lo) / bins || 1, cnt = new Array(bins).fill(0), box = { l: 46, r: g.w - 8, t: 8, b: g.h - 38 };
      vals.forEach(function(v){ var k = Math.min(bins - 1, Math.floor((v - lo) / w)); cnt[k]++; });
      var mx = Math.max.apply(null, cnt); axes(g, box, [lo, hi], [0, mx], xl, 'subjects', 2, { xdec: lo < 10 ? 1 : 0, ydec: 0 });
      cnt.forEach(function(c, k){ var x0 = box.l + k / bins * (box.r - box.l), x1 = box.l + (k + 1) / bins * (box.r - box.l), y = box.b - c / mx * (box.b - box.t); g.x.fillStyle = color; g.x.fillRect(x0 + 1, y, x1 - x0 - 2, box.b - y); });
    }
    function gm(a){ return Math.exp(mean(a.map(Math.log))); }
    function draw(){
      var p = P(), n = p.n, zE = mlbNormal(11), zT = mlbNormal(12), i;
      var Wtr = wts(n, 31, p), ytr = Wtr.map(function(w){ return Math.log(CLT) + A * Math.log(w / 70) + p.ocl * zE() + EPS * zE(); });
      var Wte = wts(NTEST, 41, p), etaTe = Wte.map(function(){ return zT(); }), epsTe = Wte.map(function(){ return zT(); });
      var truthL = function(w){ return Math.log(CLT) + A * Math.log(w / 70); };
      var yte = Wte.map(function(w, k){ return truthL(w) + p.ocl * etaTe[k] + EPS * epsTe[k]; });
      /* log-linear fit of ln CL on ln(weight / 70) */
      var xm = mean(Wtr.map(function(w){ return Math.log(w / 70); })), ym = mean(ytr), sxy = 0, sxx = 0;
      Wtr.forEach(function(w, k){ var dx = Math.log(w / 70) - xm; sxy += dx * (ytr[k] - ym); sxx += dx * dx; });
      var b = sxx > 0 ? sxy / sxx : 0, a0 = ym - b * xm, allo = function(w){ return a0 + b * Math.log(w / 70); };
      var bt = mlbBoost(Wtr.map(function(w){ return [w]; }), ytr, { rounds: 100, lr: 0.1, depth: 2 }), trees = function(w){ return bt.predict([w]); };
      var f = md ? trees : allo;
      var rm = function(fn, ref){ return Math.sqrt(mean(Wte.map(function(w, k){ var d = fn(w) - ref[k]; return d * d; }))); }, truthTe = Wte.map(truthL);
      var rA = rm(allo, yte), rT = rm(trees, yte), tA = rm(allo, truthTe), tT = rm(trees, truthTe), floor = Math.sqrt(p.ocl * p.ocl + EPS * EPS);
      /* chart: clearance (L/h) against weight */
      var xs = [], k; for (k = 0; k <= 60; k++) xs.push(35 + 85 * k / 60);
      var series = [{ pts: xs.map(function(w){ return [w, CLT * Math.pow(w / 70, A)]; }), color: C.bg, w: 2.4, nodot: true, label: 'true typical' }, { pts: xs.map(function(w){ return [w, Math.exp(allo(w))]; }), color: C.char, w: 2.4, nodot: true, label: 'allometric fit', dash: [6, 4] }, { pts: xs.map(function(w){ return [w, Math.exp(trees(w))]; }), color: C.red, w: 2.6, nodot: true, label: 'boosted trees' }];
      var ymax = Math.min(40, Math.ceil(Math.max.apply(null, ytr.map(Math.exp)) * 1.05));
      var g = lineChart(qs('canvas', sp), series, { xl: 'weight (kg)', yl: 'CL (L/h)', x0: 35, x1: 120, y0: 0, y1: ymax });
      if (g && g.sx) { var step = Math.max(1, Math.floor(n / 250)); g.x.fillStyle = 'rgba(111,129,150,0.45)'; for (i = 0; i < n; i += step) { var yy = Math.exp(ytr[i]); if (yy <= ymax) { g.x.beginPath(); g.x.arc(g.sx(Wtr[i]), g.sy(yy), 2.4, 0, 6.3); g.x.fill(); } } }
      /* simulated exposure for the test subjects with the selected model */
      var sres = Math.sqrt(mean(Wte.map(function(w, kk){ var d = yte[kk] - f(w); return d * d; }))), zv = mlbNormal(51), zk = mlbNormal(52), zr = mlbNormal(53), aucs = [], cmaxs = [], aucT = [];
      Wte.forEach(function(w){ var cl = Math.exp(f(w) + sres * zr()), v = VT * (w / 70) * Math.exp(p.ov * zv()), ka = KAT * Math.exp(p.oka * zk()), m = PKE.oneComp({ dose: DOSE, V: v, CL: cl, ka: ka, F: 1, route: 'oral' }); aucs.push(DOSE / cl); cmaxs.push(m.summary.cmax); });
      Wte.forEach(function(w, kk){ aucT.push(DOSE / Math.exp(truthL(w) + p.ocl * etaTe[kk])); });
      hist(qs('canvas', h1), aucs, 14, C.char, 'AUC (mg h/L)'); hist(qs('canvas', h2), cmaxs, 14, C.red, 'Cmax (mg/L)');
      qs('.s1', h1).innerHTML = 'Geometric mean ' + pkbNice(gm(aucs), 1) + ' (true model ' + pkbNice(gm(aucT), 1) + ')<br>5th to 95th: ' + pkbNice(pct(aucs, 0.05), 1) + ' to ' + pkbNice(pct(aucs, 0.95), 1);
      qs('.s2', h2).innerHTML = 'Geometric mean ' + pkbNice(gm(cmaxs), 2) + '<br>5th to 95th: ' + pkbNice(pct(cmaxs, 0.05), 2) + ' to ' + pkbNice(pct(cmaxs, 0.95), 2) + '<br>Residual SD of ln CL: ' + sres.toFixed(2);
      var row = function(l, x, y, on){ return '<tr style="border-top:1px solid ' + C.silver + (on ? ';background:#FBE9E9' : '') + '"><td style="padding:3px 4px 3px 0">' + l + '</td><td style="text-align:right;padding:3px 4px">' + x.toFixed(3) + '</td><td style="text-align:right;padding:3px 0 3px 4px">' + y.toFixed(3) + '</td></tr>'; };
      qs('.s3', tb).innerHTML = '<table style="width:100%;border-collapse:collapse"><tr style="font-weight:700;font-size:10.5px;color:' + C.mid + '"><td></td><td style="text-align:right;padding:2px 4px">vs observed</td><td style="text-align:right;padding:2px 0 2px 4px">vs true typical</td></tr>' + row('Allometric fit', rA, tA, !md) + row('Boosted trees', rT, tT, md) + '</table><div style="margin-top:6px">RMSE of ln CL on 400 test subjects. The floor against observed values is ' + floor.toFixed(3) + ' (omega and estimation error). The fitted exponent is ' + b.toFixed(2) + ' (true 0.75).</div>';
    }
    function sync(){ qsa('input[type=range]', lp).forEach(function(r){ var o = $('o-' + r.id); if (o) o.textContent = r.value; }); }
    ['pk3ocl', 'pk3ov', 'pk3oka', 'pk3wm', 'pk3ws', 'pk3n'].forEach(function(id){ $(id).addEventListener('input', function(){ sync(); draw(); }); });
    qsa('[data-md] button', lp).forEach(function(bn){ bn.addEventListener('click', function(){ md = +bn.dataset.v; qsa('[data-md] button', lp).forEach(function(x){ x.classList.toggle('on', x === bn); }); draw(); }); });
    function setv(id, v){ var e = $(id); e.value = v; var o = $('o-' + id); if (o) o.textContent = v; }
    async function play(tok){
      setv('pk3ocl', 0.3); setv('pk3n', 20); md = 1; qsa('[data-md] button', lp).forEach(function(x){ x.classList.toggle('on', x.dataset.v === '1'); }); draw();
      api.cap('With 20 training subjects the fitted curves differ from the true typical clearance.'); await api.sleep(3600);
      await api.anim(2600, function(t){ setv('pk3n', Math.round((20 + 380 * t) / 10) * 10); draw(); }); api.cap('More training subjects reduce the error against the true typical value. The scatter around it stays at the size of omega.'); await api.sleep(3800);
      md = 0; qsa('[data-md] button', lp).forEach(function(x){ x.classList.toggle('on', x.dataset.v === '0'); }); draw(); api.cap('The log-linear fit contains the true functional form, so its error is lower than that of the trees here.'); await api.sleep(3800); api.cap(null);
    }
    return { render: function(){ sync(); draw(); }, play: play };
  } });
GUIDE.steps['pk/Population variability'] = { notice: 'The scatter of individuals stays when the covariate model improves. The error of the typical value does not.', q: 'Which part of the spread in AUC can a better covariate model reduce?', a: 'The error of the typical value. The between-subject variability is a property of the population and remains.' };
