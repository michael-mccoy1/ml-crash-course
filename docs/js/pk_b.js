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
  change: 'Select an organ. Set logP, fup and CLint, then select Simulate.',
  watch: 'A whole-body PBPK model represents each organ as a well-mixed compartment with a physiological volume and blood flow. Here every tissue is perfusion-limited: the tissue and the venous blood leaving it are in instantaneous equilibrium, with the ratio set by a partition coefficient Kp. The gut drains to the liver through the portal vein, and the liver eliminates drug by the well-stirred model. The physiology is fixed. The drug-specific parameters are the inputs that a machine learning model has to supply. The tables list them and name the model families in this course that could predict each one. Only intrinsic clearance has data in this course.',
  build: function(api){
    var ORG = [['brain', 'Brain'], ['heart', 'Heart'], ['kidney', 'Kidney'], ['muscle', 'Muscle'], ['adipose', 'Adipose'], ['skin', 'Skin'], ['bone', 'Bone'], ['rest', 'Rest'], ['gut', 'Gut'], ['liver', 'Liver']],
        PH = PKE.pbpk.reference, sel = 'liver', logAxis = true, fresh = false;
    var dp = api.el('pn', 'left:28px;top:20px;width:904px;height:236px', '<h4>Whole-body model: organs in parallel between arterial and venous blood</h4><div class="pkd" style="margin-top:2px"></div>');
    var tp = api.el('pn', 'left:28px;top:264px;width:436px;height:256px;overflow:auto', '<h4>Parameters of the model</h4><div class="pkt" style="font-size:11.5px;line-height:1.35;color:' + C.char + '"></div>');
    var rp = api.el('pn', 'left:480px;top:264px;width:452px;height:256px', '<div class="pkr" style="font-size:12px;line-height:1.45;color:' + C.char + '"></div><canvas width="420" height="146" style="margin-top:2px"></canvas>');
    var ctl = api.addControl('logP <input type="range" class="c-lp" min="-1" max="5" step="0.1" value="2" style="width:100px"><b class="v-lp"></b> fup <input type="range" class="c-fu" min="0.02" max="1" step="0.01" value="0.5" style="width:100px"><b class="v-fu"></b> log10 CLint (mL/min/kg) <input type="range" class="c-ci" min="-1" max="3" step="0.1" value="1" style="width:100px"><b class="v-ci"></b> <button type="button" class="btn sm" data-sim>Simulate</button> <span class="seg" data-ax><button type="button" data-v="0">Linear</button><button type="button" data-v="1" class="on">Log</button></span> <span class="st" style="font-size:12px"></span>');
    function par(){ return { logP: +qs('.c-lp', ctl).value, fup: +qs('.c-fu', ctl).value, ci: +qs('.c-ci', ctl).value }; }
    function model(){ var p = par(); return PKE.pbpk.model({ logP: p.logP, fup: p.fup, bp: 1, clintU: PKE.pbpk.clintUFromHLM(p.ci, 70, 1), clr: 0, bw: 70 }); }
    function flowOf(n){ var F = PKE.pbpk.referenceScaled(70).flows; return n === 'liver' ? F.liverTotal : F[n]; }
    function diagram(){
      var P = PKE.pbpk.referenceScaled(70), V = P.volumes, F = P.flows, w = 66, x0 = 130, dx = 74, y0 = 74, h = 62, aY = 30, vY = 176, s = '', i;
      s += '<line x1="50" y1="' + aY + '" x2="' + (x0 + 9 * dx + w) + '" y2="' + aY + '" stroke="' + C.red + '" stroke-width="3"/><line x1="' + (x0 + 9 * dx + w) + '" y1="' + vY + '" x2="50" y2="' + vY + '" stroke="' + C.mid + '" stroke-width="3"/>';
      s += '<line x1="50" y1="' + vY + '" x2="50" y2="' + (y0 + h) + '" stroke="' + C.mid + '" stroke-width="2"/><line x1="50" y1="' + y0 + '" x2="50" y2="' + aY + '" stroke="' + C.red + '" stroke-width="2"/>';
      s += '<rect x="10" y="' + y0 + '" width="80" height="' + h + '" rx="4" fill="#fff" stroke="' + C.char + '" stroke-width="1.4"/><text x="50" y="' + (y0 + 20) + '" text-anchor="middle" font-weight="700" fill="' + C.char + '">Lung</text><text x="50" y="' + (y0 + 38) + '" text-anchor="middle" font-size="10.5" fill="' + C.char + '">V ' + pkbNice(V.lung, 1) + ' L</text><text x="50" y="' + (y0 + 52) + '" text-anchor="middle" font-size="10.5" fill="' + C.char + '">Q ' + pkbNice(F.cardiacOutput, 0) + ' L/h</text>';
      s += '<text x="' + (x0 - 6) + '" y="' + (aY - 8) + '" text-anchor="start" font-size="11" fill="' + C.red + '">arterial blood</text><text x="' + (x0 - 6) + '" y="' + (vY + 16) + '" text-anchor="start" font-size="11" fill="' + C.mid + '">venous blood</text>';
      s += '<line x1="104" y1="' + (vY + 14) + '" x2="104" y2="' + (vY + 2) + '" stroke="' + C.char + '" stroke-width="1.6" marker-end="url(#pkba)"/><text x="104" y="' + (vY + 26) + '" text-anchor="middle" font-size="10.5" fill="' + C.char + '">IV dose</text>';
      ORG.forEach(function(o, k){
        var x = x0 + k * dx, cx = x + w / 2, on = o[0] === sel, vol = o[0] === 'liver' ? V.liver : V[o[0]], q = flowOf(o[0]);
        if (o[0] !== 'gut') s += '<line x1="' + cx + '" y1="' + aY + '" x2="' + cx + '" y2="' + y0 + '" stroke="' + C.red + '" stroke-width="1.4" marker-end="url(#pkba)"/>'; else s += '<line x1="' + cx + '" y1="' + aY + '" x2="' + cx + '" y2="' + y0 + '" stroke="' + C.red + '" stroke-width="1.4" marker-end="url(#pkba)"/>';
        if (o[0] === 'liver') s += '<line x1="' + (cx + 14) + '" y1="' + aY + '" x2="' + (cx + 14) + '" y2="' + y0 + '" stroke="' + C.red + '" stroke-width="1.4" marker-end="url(#pkba)"/>';
        if (o[0] !== 'gut') s += '<line x1="' + cx + '" y1="' + (y0 + h) + '" x2="' + cx + '" y2="' + vY + '" stroke="' + C.mid + '" stroke-width="1.4"/>';
        s += '<g data-org="' + o[0] + '" style="cursor:pointer" tabindex="0"><rect x="' + x + '" y="' + y0 + '" width="' + w + '" height="' + h + '" rx="4" fill="' + (on ? '#FBE9E9' : '#fff') + '" stroke="' + (on ? C.red : C.char) + '" stroke-width="' + (on ? 2.4 : 1.4) + '"/><text x="' + cx + '" y="' + (y0 + 18) + '" text-anchor="middle" font-weight="700" fill="' + C.char + '">' + o[1] + '</text><text x="' + cx + '" y="' + (y0 + 36) + '" text-anchor="middle" font-size="10.5" fill="' + C.char + '">V ' + pkbNice(vol, vol < 0.1 ? 3 : 1) + ' L</text><text x="' + cx + '" y="' + (y0 + 51) + '" text-anchor="middle" font-size="10.5" fill="' + C.char + '">Q ' + pkbNice(q, q >= 100 ? 0 : 1) + ' L/h</text></g>';
        if (o[0] === 'liver') s += '<text x="' + (cx - 4) + '" y="' + (y0 + h + 15) + '" text-anchor="end" font-size="10.5" fill="' + C.dred + '">CLint</text>';
        if (o[0] === 'kidney') s += '<text x="' + (cx - 4) + '" y="' + (y0 + h + 15) + '" text-anchor="end" font-size="10.5" fill="' + C.mid + '">renal CL</text>';
        if (o[0] === 'gut') s += '<line x1="' + (x + w) + '" y1="' + (y0 + h / 2) + '" x2="' + (x + dx) + '" y2="' + (y0 + h / 2) + '" stroke="' + C.char + '" stroke-width="1.6" marker-end="url(#pkba)"/><text x="' + (x + w + 4) + '" y="' + (y0 + h / 2 - 4) + '" font-size="9.5" fill="' + C.char + '">portal</text>';
      });
      qs('.pkd', dp).innerHTML = '<svg viewBox="0 0 880 200" width="100%" style="max-width:880px;display:block;margin:0 auto" font-family="Aptos, Segoe UI, sans-serif" font-size="12"><defs><marker id="pkba" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="' + C.char + '"/></marker></defs>' + s + '</svg>';
    }
    function table(){
      var rows = [['Physiology (fixed)', 'Organ volumes', 'ICRP Publication 89, 73 kg adult male, scaled to 70 kg', 'Not predicted'], ['', 'Blood flows', 'Same source; flows scale with weight to the power 0.75', 'Not predicted'], ['', 'Tissue composition', 'Water, neutral lipid, phospholipid fractions (Ruark 2014; Poulin and Theil 2002)', 'Not predicted'],
        ['Drug-specific', 'Hepatic CLint', 'HLM log10 CLint, mL/min/kg, through IVIVE', 'Trees, networks, pre-trained models (lessons 02 to 05). Data in this course.'], ['', 'fup, blood:plasma ratio', 'Assumed here (fup set above, ratio 1)', 'Same families on other endpoints. No data in this course.'], ['', 'Kp per tissue', 'Poulin and Theil mixture model from logP and fup', 'Could be predicted directly. No data in this course.'],
        ['', 'ka and F', 'Not used for the IV dose here', 'Permeability and solubility endpoints. No data in this course.'], ['', 'Renal clearance', 'Set to 0 here', 'Same families. No data in this course.']];
      qs('.pkt', tp).innerHTML = '<table style="width:100%;border-collapse:collapse"><tr style="font-weight:700;font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;color:' + C.mid + '"><td></td><td>Parameter</td><td>Value or source</td><td>ML in this course</td></tr>' + rows.map(function(r){ return '<tr style="border-top:1px solid ' + C.silver + ';vertical-align:top"><td style="padding:4px 6px 4px 0;font-weight:700;color:' + (r[0] === 'Drug-specific' ? C.dred : C.char) + '">' + r[0] + '</td><td style="padding:4px 6px 4px 0;font-weight:600">' + r[1] + '</td><td style="padding:4px 6px 4px 0">' + r[2] + '</td><td style="padding:4px 0">' + r[3] + '</td></tr>'; }).join('') + '</table>' +
        '<p style="margin:6px 0 0;color:' + C.mid + '">Physiology source: ' + esc(PH.source) + '. The Kp method and its unbound-in-tissue default are as implemented in the course engine and have not been checked against the primary paper.</p>';
    }
    function details(){
      var p = par(), P = PKE.pbpk.referenceScaled(70), kp = PKE.pbpk.kpPT(p.logP, p.fup), v = sel === 'liver' ? P.volumes.liver : P.volumes[sel], q = flowOf(sel), nm = ORG.filter(function(o){ return o[0] === sel; })[0][1], k = kp[sel], tau = v * k / q;
      var tot = Object.keys(P.volumes).reduce(function(a, b){ return a + P.volumes[b]; }, 0);
      qs('.pkr', rp).innerHTML = '<b style="font-size:13px">' + nm + '</b>: volume ' + pkbNice(v, 2) + ' L (' + (100 * v / tot).toFixed(1) + '% of the model volume), blood flow ' + pkbNice(q, 1) + ' L/h (' + (100 * q / P.flows.cardiacOutput).toFixed(1) + '% of cardiac output). Kp ' + pkbNice(k, 2) + ' at logP ' + p.logP.toFixed(1) + ' and fup ' + p.fup.toFixed(2) + '. Equilibration time constant V Kp / Q: ' + (tau < 1 ? (tau * 60).toFixed(0) + ' min' : tau.toFixed(1) + ' h') + '.';
    }
    function chart(){
      var el = qs('canvas', rp), S = PKB.sim;
      if (!S) { var g0 = cv(el); g0.x.fillStyle = C.mid; g0.x.font = '12px Aptos, "Segoe UI", sans-serif'; g0.x.textAlign = 'center'; g0.x.fillText('Select Simulate to run a 10 mg IV bolus over 24 h.', g0.w / 2, g0.h / 2); return; }
      var org = S.res.organs[sel] || S.res.organs.liver, pl = S.res.venousPlasma, cmax = Math.max.apply(null, pl.concat(org)), cmin = cmax / 1e4, pts1 = [], pts2 = [], i, f = function(c){ return logAxis ? Math.log10(Math.max(c, cmin)) : c; };
      for (i = 0; i < S.res.t.length; i++) { pts1.push([S.res.t[i], f(pl[i])]); pts2.push([S.res.t[i], f(org[i])]); }
      var o = { xl: 'time (h)', yl: logAxis ? 'mg/L, log scale' : 'mg/L', x0: 0, x1: S.tEnd };
      if (logAxis) { var lo = Math.floor(Math.log10(cmin)), hi = Math.ceil(Math.log10(cmax)); o.y0 = lo; o.y1 = hi; o.yticks = pkbLogTicks(lo, hi); } else o.y0 = 0;
      lineChart(el, [{ pts: pts1, color: C.red, w: 2.6, nodot: true, label: 'venous plasma' }, { pts: pts2, color: C.char, w: 2.4, nodot: true, label: ORG.filter(function(x){ return x[0] === sel; })[0][1] + ' tissue' }], o);
    }
    function status(t){ qs('.st', ctl).textContent = t || ''; }
    function readouts(){ var p = par(); qs('.v-lp', ctl).textContent = p.logP.toFixed(1); qs('.v-fu', ctl).textContent = p.fup.toFixed(2); qs('.v-ci', ctl).textContent = p.ci.toFixed(1); }
    function all(){ readouts(); diagram(); table(); details(); chart(); }
    async function simulate(){
      var btn = qs('[data-sim]', ctl); btn.disabled = true; status('Simulating'); await tick(); await tick();
      try { var m = model(), dt = PKE.pbpk.stableDt(m), tEnd = 24, n = Math.ceil(tEnd / dt), res = PKE.pbpk.simulate(m, { route: 'iv', dose: 10 }, tEnd, dt, Math.max(1, Math.ceil(n / 420))); PKB.sim = { res: res, tEnd: tEnd, p: par() }; status('Done. Mass balance error ' + res.massError.toExponential(1) + '.'); fresh = true; chart(); }
      catch (e) { console.error(e); status('The simulation failed.'); } finally { btn.disabled = false; }
    }
    qs('[data-sim]', ctl).addEventListener('click', simulate);
    ['.c-lp', '.c-fu', '.c-ci'].forEach(function(c){ qs(c, ctl).addEventListener('input', function(){ readouts(); details(); if (PKB.sim) status('Parameters changed. Select Simulate to update the curves.'); }); });
    qsa('[data-ax] button', ctl).forEach(function(b){ b.addEventListener('click', function(){ logAxis = +b.dataset.v === 1; qsa('[data-ax] button', ctl).forEach(function(x){ x.classList.toggle('on', x === b); }); chart(); }); });
    dp.addEventListener('click', function(e){ var g = e.target.closest ? e.target.closest('[data-org]') : null; if (!g) return; sel = g.dataset.org; diagram(); details(); chart(); });
    dp.addEventListener('keydown', function(e){ if (e.key !== 'Enter') return; var g = e.target.closest ? e.target.closest('[data-org]') : null; if (g) { sel = g.dataset.org; diagram(); details(); chart(); } });
    async function play(tok){
      sel = 'liver'; qs('.c-lp', ctl).value = 2; qs('.c-fu', ctl).value = 0.5; qs('.c-ci', ctl).value = 1; all(); api.cap('Organs sit in parallel between the arterial and venous blood. The gut drains to the liver through the portal vein.'); await api.sleep(3600);
      sel = 'adipose'; all(); api.cap('Each tissue has a volume, a blood flow and a partition coefficient Kp. Adipose has a large volume and a small flow, so it equilibrates slowly.'); await api.sleep(3800);
      sel = 'liver'; all(); api.cap('Hepatic elimination uses the intrinsic clearance, the model input that the HLM data inform.'); await api.sleep(3400);
      await simulate(); api.cap('A 10 mg IV bolus. Tissue curves lag plasma by an amount set by V Kp / Q.'); await api.sleep(3600); api.cap(null);
    }
    return { render: all, play: play };
  } });
GUIDE.steps['pk/PBPK structure'] = { notice: 'The physiology is fixed. The drug-specific parameters are what a machine learning model has to supply.', q: 'Which parameters of this model are inputs that machine learning could supply?', a: 'Hepatic CLint, fraction unbound, blood to plasma ratio, the partition coefficients, ka and F, and renal clearance. The organ volumes and flows are physiology.' };

/* ---------- 2. Compartmental fit ---------- */
LESSONS.pk.add({ level: 'intermediate', short: 'Compartmental fit', title: 'NCA and compartmental fits',
  change: 'Select a data set, a subject and a model. Compare the non-compartmental and fitted estimates.',
  watch: 'Two ways to extract parameters from a measured profile. Non-compartmental analysis (NCA) uses the trapezoidal rule and the terminal slope and assumes no model. A compartmental fit chooses a model and minimises the difference between the model and the data. The fits here minimise the sum of squared differences of log concentration, which approximates a proportional error model. Both use the same public data: Theoph (oral theophylline, 12 subjects, 132 observations) and Indometh (intravenous indomethacin, 6 subjects, 66 observations). The Indometh dose is not verified in the public sources, so Indometh results are given for an assumed dose of 1 mg and all clearances and volumes scale in proportion to the true dose. The fits are the same optimisation that trains a machine learning model: a loss, parameters, and an optimiser.',
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
      var fmt = function(v){ return pkbNice(v, 2); }, unit = ds === 'theoph' ? '/F' : '', rows = [['AUC0-inf (mg h/L)', fmt(n.aucInf)], ['CL' + unit + ' (L/h)', fmt(n.clF)], ['V' + unit + ' (L), Vz', fmt(n.vzF)], ['Terminal half-life (h)', fmt(n.thalf)]];
      var html = '<div class="ol">Non-compartmental (' + (ds === 'theoph' ? 'oral' : 'IV') + ')</div><table style="width:100%;border-collapse:collapse">' + rows.map(function(x){ return '<tr style="border-top:1px solid ' + C.silver + '"><td style="padding:2px 4px 2px 0">' + x[0] + '</td><td style="text-align:right;font-weight:700">' + x[1] + '</td></tr>'; }).join('') + '</table>';
      ks.forEach(function(kd){ var f = getFit(sub, kd), q = summary(f), rm = Math.sqrt(f.sse / f.n); html += '<div class="ol" style="margin-top:10px">' + (kd === 'oral1' ? 'Fit: one compartment, oral' : (kd === 'iv1' ? 'Fit: one compartment' : 'Fit: two compartments')) + '</div><table style="width:100%;border-collapse:collapse">' + [['AUC0-inf (mg h/L)', fmt(q.auc)], ['CL' + unit + ' (L/h)', fmt(q.cl)], ['V' + unit + ' (L)' + (kd === 'iv2' ? ', Vz' : ''), fmt(q.v)], ['Terminal half-life (h)', fmt(q.thalf)], ['Residual SD of ln c', rm.toFixed(3)]].map(function(x){ return '<tr style="border-top:1px solid ' + C.silver + '"><td style="padding:2px 4px 2px 0">' + x[0] + '</td><td style="text-align:right;font-weight:700;color:' + cols[kd] + '">' + x[1] + '</td></tr>'; }).join('') + '</table>' + (q.extra ? '<div style="font-size:11px;color:' + C.mid + '">' + q.extra + '</div>' : ''); });
      html += '<p style="margin:10px 0 0;font-size:11.5px">' + (ds === 'theoph' ? 'Theoph: oral dose in mg from the data set. Observations at t = 0 are excluded from the fit because the model has concentration 0 there and the objective uses logarithms. CL/F and V/F are apparent values because bioavailability is not identifiable from oral data alone.' : 'Indometh: dose assumed to be 1 mg because it is not verified in the public sources. CL and V therefore scale in proportion to the true dose, while the half-lives do not. NCA extrapolates the first two observations back to t = 0.') + '</p>';
      qs('.pkf', ap).innerHTML = html;
      /* agreement across subjects */
      var N = DATA[ds].subjects.length, pairs = [], mx = 0;
      for (i = 0; i < N; i++) { var si = DATA[ds].subjects[i], ni = ncaOf(si), fk = ds === 'theoph' ? 'oral1' : (mdl === 1 ? 'iv1' : 'iv2'), fq = summary(getFit(i, fk)); pairs.push([ni.clF, fq.cl, si.id]); mx = Math.max(mx, ni.clF, fq.cl); }
      qs('.pkh3', ag).textContent = 'CL' + unit + ' across ' + N + ' subjects: NCA against fit';
      var a = cv(qs('canvas', ag)), bx = { l: 40, r: a.w - 12, t: 12, b: a.h - 36 }, top = Math.ceil(mx * 1.1);
      var sxa = function(v){ return bx.l + v / top * (bx.r - bx.l); }, sya = function(v){ return bx.b - v / top * (bx.b - bx.t); };
      axes(a, bx, [0, top], [0, top], 'CL' + unit + ' by NCA (L/h)', 'CL' + unit + ' by fit (L/h)', 4);
      a.x.strokeStyle = C.bg; a.x.setLineDash([4, 4]); a.x.beginPath(); a.x.moveTo(sxa(0), sya(0)); a.x.lineTo(sxa(top), sya(top)); a.x.stroke(); a.x.setLineDash([]);
      pairs.forEach(function(p){ a.x.fillStyle = p[2] === s.id ? C.red : C.char; a.x.beginPath(); a.x.arc(sxa(p[0]), sya(p[1]), p[2] === s.id ? 5.5 : 3.8, 0, 6.3); a.x.fill(); });
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
      api.cap('Across subjects the two estimates track each other. The scatter shows where they differ.'); await api.sleep(3400);
      ds = 'indometh'; sub = 0; mdl = 1; fillSubjects(); qsa('[data-ds] button', ctl).forEach(function(x){ x.classList.toggle('on', x.dataset.v === 'indometh'); }); qsa('[data-md] button', ctl).forEach(function(x){ x.classList.toggle('on', x.dataset.v === '1'); }); await render();
      api.cap('An IV profile with a distribution phase: the one-compartment residuals have a pattern.'); await api.sleep(3800);
      mdl = 2; qsa('[data-md] button', ctl).forEach(function(x){ x.classList.toggle('on', x.dataset.v === '2'); }); await render(); api.cap('A second compartment removes the pattern in the residuals.'); await api.sleep(3600); api.cap(null);
    }
    return { render: render, play: play };
  } });
GUIDE.steps['pk/Compartmental fit'] = { notice: 'Residuals with a pattern show that the model structure is wrong, not the parameter values.', q: 'Why is only dose-normalised information reported for Indometh?', a: 'The dose is not verified in the public sources. Clearances and volumes scale with the dose, while rate constants and half-lives do not.' };

/* ---------- 3. Population variability ---------- */
LESSONS.pk.add({ level: 'intermediate', short: 'Population variability', title: 'Population variability',
  change: 'Set the between-subject variability, the weight distribution and the number of subjects. Compare variability with uncertainty.',
  watch: 'A population model describes how parameters differ between individuals. Each individual parameter is the typical value, scaled by a covariate effect and by a lognormal random effect whose standard deviation is omega. Body weight is the usual covariate, scaled to a 70 kg reference with an exponent of 0.75 on clearance and 1 on volume. This is a convention (Anderson and Holford 2008, Annu Rev Pharmacol Toxicol 48:303-332) and not a universal law. The typical values here are illustrative, not data. The figure separates two quantities that are easy to confuse. Variability is real difference between individuals and does not shrink with more data. Uncertainty is lack of knowledge of the typical value, as when it is predicted by a machine learning model with an error, and shrinks as the prediction improves.',
  build: function(api){
    var CLT = 6, VT = 40, KAT = 1, DOSE = 100, TEND = 24;
    var lp = api.el('pn', 'left:28px;top:20px;width:236px;height:500px;overflow:auto', '<h4>Population</h4>' + optRange('pk3ocl', 'Omega of CL (SD of ln CL)', 0, 0.8, 0.05, 0.3) + optRange('pk3ov', 'Omega of V', 0, 0.6, 0.05, 0.2) + optRange('pk3oka', 'Omega of ka', 0, 0.8, 0.05, 0.4) + optRange('pk3wm', 'Median weight (kg)', 50, 100, 1, 70) + optRange('pk3ws', 'Weight SD (kg)', 0, 25, 1, 12) + optRange('pk3n', 'Subjects', 20, 500, 10, 200) + '<div class="og"><div class="ol">Weight effect</div>' + seg([['1', 'Allometric'], ['0', 'None']], '1', 'data-wt') + '</div><div class="og"><div class="ol">Typical values (illustrative)</div><div class="hint">70 kg subject, oral dose 100 mg: CL 6 L/h, V 40 L, ka 1 per h, F 1. These are not data.</div></div>');
    var sp = api.el('pn', 'left:280px;top:20px;width:652px;height:248px', '<h4>Concentration after an oral dose: individuals, median and 5th to 95th percentile</h4><canvas width="622" height="200"></canvas>');
    var h1 = api.el('pn', 'left:280px;top:278px;width:206px;height:244px', '<h4>AUC</h4><canvas width="176" height="156"></canvas><div class="s1" style="font-size:11.5px;line-height:1.35;color:' + C.char + '"></div>');
    var h2 = api.el('pn', 'left:496px;top:278px;width:206px;height:244px', '<h4>Cmax</h4><canvas width="176" height="156"></canvas><div class="s2" style="font-size:11.5px;line-height:1.35;color:' + C.char + '"></div>');
    var up = api.el('pn', 'left:712px;top:278px;width:220px;height:244px', '<h4>Uncertainty of the typical CL</h4><canvas width="190" height="156"></canvas><div class="s3" style="margin-top:2px"></div>');
    var ctl = api.addControl('Uncertainty of typical CL (SD of ln) <input type="range" class="c-un" min="0" max="1" step="0.05" value="0.4" style="width:150px"><b class="v-un"></b>');
    var allo = true, cache = null;
    function P(){ return { ocl: +$('pk3ocl').value, ov: +$('pk3ov').value, oka: +$('pk3oka').value, wm: +$('pk3wm').value, ws: +$('pk3ws').value, n: +$('pk3n').value, un: +qs('.c-un', ctl).value }; }
    function normals(n, seed){ var r = rng(seed), out = []; for (var i = 0; i < n; i++) { var u = Math.max(r(), 1e-12), v = r(); out.push(Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)); } return out; }
    function pct(a, q){ var s = a.slice().sort(function(x, y){ return x - y; }), k = (s.length - 1) * q, lo = Math.floor(k), hi = Math.ceil(k); return s[lo] + (s[hi] - s[lo]) * (k - lo); }
    function hist(el, vals, bins, color, xl){
      var g = cv(el), lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals), w = (hi - lo) / bins || 1, cnt = new Array(bins).fill(0), box = { l: 46, r: g.w - 8, t: 8, b: g.h - 38 };
      vals.forEach(function(v){ var k = Math.min(bins - 1, Math.floor((v - lo) / w)); cnt[k]++; });
      var mx = Math.max.apply(null, cnt); axes(g, box, [lo, hi], [0, mx], xl, 'subjects', 2, { xdec: lo < 10 ? 1 : 0, ydec: 0 });
      cnt.forEach(function(c, k){ var x0 = box.l + k / bins * (box.r - box.l), x1 = box.l + (k + 1) / bins * (box.r - box.l), y = box.b - c / mx * (box.b - box.t); g.x.fillStyle = color; g.x.fillRect(x0 + 1, y, x1 - x0 - 2, box.b - y); });
    }
    function draw(){
      var p = P(), n = p.n, z = normals(n, 31), W = z.map(function(v){ return Math.min(120, Math.max(35, p.wm + p.ws * v)); }), pop = PKE.population(n, 2026, { CL: { typical: CLT, omega: p.ocl }, V: { typical: VT, omega: p.ov }, ka: { typical: KAT, omega: p.oka } }), i, k, T = [], nt = 97;
      for (k = 0; k < nt; k++) T.push(TEND * k / (nt - 1));
      var prof = [], aucs = [], cmaxs = [];
      for (i = 0; i < n; i++) { var f = allo ? Math.pow(W[i] / 70, 0.75) : 1, fv = allo ? W[i] / 70 : 1, cl = pop[i].CL * f, v = pop[i].V * fv, ka = pop[i].ka, m = PKE.oneComp({ dose: DOSE, V: v, CL: cl, ka: ka, F: 1, route: 'oral' }); prof.push(T.map(function(t){ return m.conc(t); })); aucs.push(DOSE / cl); cmaxs.push(m.summary.cmax); }
      var med = [], lo = [], hi = [];
      for (k = 0; k < nt; k++) { var col = prof.map(function(r){ return r[k]; }); med.push(pct(col, 0.5)); lo.push(pct(col, 0.05)); hi.push(pct(col, 0.95)); }
      var top = Math.ceil(Math.max.apply(null, hi) * 1.1 * 10) / 10, series = [], step = Math.max(1, Math.floor(n / 40));
      for (i = 0; i < n; i += step) series.push({ pts: T.map(function(t, kk){ return [t, prof[i][kk]]; }), color: 'rgba(111,129,150,0.35)', w: 1, nodot: true });
      series.push({ pts: T.map(function(t, kk){ return [t, med[kk]]; }), color: C.red, w: 3, nodot: true, label: 'median' });
      var g = lineChart(qs('canvas', sp), series, { xl: 'time (h)', yl: 'plasma concentration (mg/L)', x0: 0, x1: TEND, y0: 0, y1: top });
      if (g && g.sx) { g.x.fillStyle = 'rgba(225,36,42,0.13)'; g.x.beginPath(); T.forEach(function(t, kk){ var x = g.sx(t), y = g.sy(hi[kk]); kk ? g.x.lineTo(x, y) : g.x.moveTo(x, y); }); for (var kb = nt - 1; kb >= 0; kb--) g.x.lineTo(g.sx(T[kb]), g.sy(lo[kb])); g.x.closePath(); g.x.fill(); g.x.fillStyle = C.char; g.x.font = '11px Aptos, "Segoe UI", sans-serif'; g.x.textAlign = 'right'; g.x.fillText('grey: ' + Math.min(n, Math.ceil(n / step)) + ' individuals. Shaded: 5th to 95th percentile of all ' + n + '.', g.box.r - 4, g.box.t + 12); }
      hist(qs('canvas', h1), aucs, 14, C.char, 'AUC (mg h/L)'); hist(qs('canvas', h2), cmaxs, 14, C.red, 'Cmax (mg/L)');
      var gm = function(a){ return Math.exp(mean(a.map(Math.log))); }, cvp = function(a){ var l = a.map(Math.log), m = mean(l), sd = Math.sqrt(mean(l.map(function(x){ return (x - m) * (x - m); }))); return 100 * Math.sqrt(Math.exp(sd * sd) - 1); };
      qs('.s1', h1).innerHTML = 'Geometric mean ' + pkbNice(gm(aucs), 1) + '<br>5th to 95th: ' + pkbNice(pct(aucs, 0.05), 1) + ' to ' + pkbNice(pct(aucs, 0.95), 1) + '<br>Geometric CV ' + cvp(aucs).toFixed(0) + '%';
      qs('.s2', h2).innerHTML = 'Geometric mean ' + pkbNice(gm(cmaxs), 2) + '<br>5th to 95th: ' + pkbNice(pct(cmaxs, 0.05), 2) + ' to ' + pkbNice(pct(cmaxs, 0.95), 2) + '<br>Geometric CV ' + cvp(cmaxs).toFixed(0) + '%';
      /* uncertainty: median profile of a 70 kg subject for typical CL drawn from lognormal(un) against the individual band */
      var un = p.un, zz = normals(30, 77), ul = [];
      qs('.v-un', ctl).textContent = un.toFixed(2);
      zz.forEach(function(v){ var cl = CLT * Math.exp(un * v), m = PKE.oneComp({ dose: DOSE, V: VT, CL: cl, ka: KAT, F: 1, route: 'oral' }); ul.push({ pts: T.map(function(t){ return [t, m.conc(t)]; }), color: 'rgba(225,36,42,0.35)', w: 1.2, nodot: true }); });
      var typ = PKE.oneComp({ dose: DOSE, V: VT, CL: CLT, ka: KAT, F: 1, route: 'oral' });
      ul.push({ pts: T.map(function(t){ return [t, typ.conc(t)]; }), color: C.char, w: 2.4, nodot: true });
      lineChart(qs('canvas', up), ul, { xl: 'time (h)', yl: 'mg/L', x0: 0, x1: TEND, y0: 0, y1: Math.ceil(Math.max.apply(null, hi) * 1.1 * 10) / 10 });
      qs('.s3', up).innerHTML = '<div style="font-size:10.5px;line-height:1.25;color:' + C.char + '">Red: 30 draws of the typical CL, SD ' + un.toFixed(2) + ' in ln units. Dark: point estimate. This spread is uncertainty, not between-subject variability.</div>';
    }
    function sync(){ qsa('input[type=range]', lp).forEach(function(r){ var o = $('o-' + r.id); if (o) o.textContent = r.value; }); }
    ['pk3ocl', 'pk3ov', 'pk3oka', 'pk3wm', 'pk3ws', 'pk3n'].forEach(function(id){ $(id).addEventListener('input', function(){ sync(); draw(); }); });
    qs('.c-un', ctl).addEventListener('input', draw);
    qsa('[data-wt] button', lp).forEach(function(b){ b.addEventListener('click', function(){ allo = b.dataset.v === '1'; qsa('[data-wt] button', lp).forEach(function(x){ x.classList.toggle('on', x === b); }); draw(); }); });
    function setv(id, v){ var e = $(id); e.value = v; var o = $('o-' + id); if (o) o.textContent = v; }
    async function play(tok){
      setv('pk3ocl', 0.3); setv('pk3ov', 0.2); setv('pk3oka', 0.4); setv('pk3ws', 12); qs('.c-un', ctl).value = 0.4; allo = true; qsa('[data-wt] button', lp).forEach(function(x){ x.classList.toggle('on', x.dataset.v === '1'); }); draw();
      api.cap('Individuals differ in CL, V and ka. The shaded band is the 5th to 95th percentile of the population.'); await api.sleep(3600);
      await api.anim(2400, function(t){ setv('pk3ocl', +(0.3 + 0.4 * t).toFixed(2)); draw(); }); api.cap('Raising omega of CL widens the AUC distribution. The median profile does not move.'); await api.sleep(3400);
      await api.anim(2000, function(t){ qs('.c-un', ctl).value = (0.4 * (1 - t)).toFixed(2); draw(); }); api.cap('Uncertainty of the typical value can be reduced by a better prediction. Variability between individuals cannot.'); await api.sleep(3800); api.cap(null);
    }
    return { render: function(){ sync(); draw(); }, play: play };
  } });
GUIDE.steps['pk/Population variability'] = { notice: 'The spread of individuals stays when the uncertainty of the typical value goes to zero.', q: 'How does uncertainty of a predicted typical clearance differ from between-subject variability?', a: 'Uncertainty is lack of knowledge of the typical value and shrinks with a better prediction. Variability is real difference between individuals and does not.' };
