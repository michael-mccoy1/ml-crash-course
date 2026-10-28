"use strict";
/* Lesson 10, Advanced steps: sensitivity of PBPK exposure, absorption and distribution, DDI models, parameter estimation.
   Sources of fixed numbers and equations are given beside them. Prefix pkc is used for every helper in this file. */

/* ---------- shared helpers ---------- */
var PKC_BASE = { logP: 2.0, fup: 0.5, bp: 1.0, clr: 1.0, ka: 1.0, F: 0.5, bw: 70, dose: 100 };
/* Typical HLM compound and the test RMSE of the lesson 02 boosted-trees model, both computed from the data set. */
var PKC_HLM = null;
async function pkcHLM(){
  if (PKC_HLM) return PKC_HLM;
  var ds = await loadDataset('HLM'), keep = D; D = ds;
  try {
    var q = ds.TRAIN.filter(function(i){ return !(ds.cens && ds.cens[i]); }).map(function(i){ return ds.Y[i]; }).sort(function(a, b){ return a - b; }), med = q[Math.floor(q.length / 2)];
    var m = fitBoost(ds.BASE.slice(), ds.TRAIN, 150, 0.08, 3), rm = rmse(ds.TEST, m.F, ds.Y);
    PKC_HLM = { median: med, rmse: rm, n: q.length };
  } finally { D = keep; }
  return PKC_HLM;
}
function pkcTick(){ return new Promise(function(r){ setTimeout(r, 0); }); }
function pkcFmt(v, d){ if (!isFinite(v)) return 'n/a'; var a = Math.abs(v); return a >= 100 ? v.toFixed(0) : (a >= 10 ? v.toFixed(1) : v.toFixed(d === undefined ? 2 : d)); }
function pkcLog2Fmt(v){ var f = Math.pow(2, v); return (f >= 1 ? '+' : '') + ((f - 1) * 100).toFixed(0) + '%'; }
function pkcSolve(A, b){ /* Gaussian elimination with partial pivoting, returns x for A x = b */
  var n = b.length, M = A.map(function(r, i){ return r.concat([b[i]]); }), i, j, k;
  for (i = 0; i < n; i++) { var p = i; for (j = i + 1; j < n; j++) if (Math.abs(M[j][i]) > Math.abs(M[p][i])) p = j; var t = M[i]; M[i] = M[p]; M[p] = t;
    for (j = i + 1; j < n; j++) { var f = M[j][i] / M[i][i]; for (k = i; k <= n; k++) M[j][k] -= f * M[i][k]; } }
  var x = new Array(n); for (i = n - 1; i >= 0; i--) { var s = M[i][n]; for (j = i + 1; j < n; j++) s -= M[i][j] * x[j]; x[i] = s / M[i][i]; }
  return x;
}
/* Linear system of the PBPK model after an intravenous dose: the 14 blood and tissue amounts of PKE.pbpk (lung, arterial, venous, portal, gut, liver,
   kidney, muscle, adipose, brain, heart, skin, bone, rest). The equations repeat the right-hand side of PKE.pbpk without the absorption terms. */
function pkcMatrix(m){
  var V = m.phys.volumes, Q = m.phys.flows, kp = m.kp, B = m.bp, co = Q.cardiacOutput, qL = Q.liverTotal, n = 14, A = [], j, i;
  var names = ['lung', 'arterial', 'venous', 'portal', 'gut', 'liver', 'kidney', 'muscle', 'adipose', 'brain', 'heart', 'skin', 'bone', 'rest'];
  var qOf = { kidney: Q.kidney, muscle: Q.muscle, adipose: Q.adipose, brain: Q.brain, heart: Q.heart, skin: Q.skin, bone: Q.bone, rest: Q.rest }, SYS = ['kidney', 'muscle', 'adipose', 'brain', 'heart', 'skin', 'bone', 'rest'];
  function f(y){
    var d = new Array(n).fill(0), Ca = y[1] / V.arterial, Cv = y[2] / V.venous, Cp = y[3] / V.portal, cb = {}, venIn = 0, art = 0;
    ['lung', 'gut', 'liver', 'kidney', 'muscle', 'adipose', 'brain', 'heart', 'skin', 'bone', 'rest'].forEach(function(nm){ cb[nm] = B * y[names.indexOf(nm)] / V[nm] / kp[nm]; });
    SYS.forEach(function(nm){ d[names.indexOf(nm)] = qOf[nm] * (Ca - cb[nm]); venIn += qOf[nm] * cb[nm]; art += qOf[nm] * Ca; });
    d[4] = Q.gut * (Ca - cb.gut); art += Q.gut * Ca; d[3] = Q.gut * (cb.gut - Cp);
    var met = m.clintU * m.fup * y[5] / V.liver / kp.liver;
    d[5] = Q.liverArtery * Ca + Q.gut * Cp - qL * cb.liver - met; art += Q.liverArtery * Ca; venIn += qL * cb.liver;
    d[2] = venIn - co * Cv; d[0] = co * (Cv - cb.lung); d[1] = co * cb.lung - art - m.clr * Ca / B;
    return d;
  }
  for (j = 0; j < n; j++) { var e = new Array(n).fill(0); e[j] = 1; var col = f(e); for (i = 0; i < n; i++) { if (!A[i]) A[i] = []; A[i][j] = col[i]; } }
  return A;
}
/* Slowest decay rate of the linear system, by inverse iteration. Returns the terminal rate constant (1/h). */
function pkcTerminalRate(A){
  var n = A.length, v = new Array(n).fill(1), lam = 0, k, i;
  for (k = 0; k < 400; k++) {
    var w = pkcSolve(A, v), nrm = Math.sqrt(w.reduce(function(s, x){ return s + x * x; }, 0)); v = w.map(function(x){ return x / nrm; });
    var Av = A.map(function(r){ return r.reduce(function(s, a, j){ return s + a * v[j]; }, 0); }), num = 0, den = 0; for (i = 0; i < n; i++) { num += v[i] * Av[i]; den += v[i] * v[i]; }
    var nl = num / den; if (k > 5 && Math.abs(nl - lam) < 1e-13 * Math.abs(nl)) { lam = nl; break; } lam = nl;
  }
  return -lam;
}
/* Volume of distribution at steady state with respect to plasma concentration. The model amounts are blood amounts (blood concentration is bp times the plasma
   concentration) and tissue amounts (tissue concentration is Kp times the plasma concentration at equilibrium). */
function pkcVss(m){
  var V = m.phys.volumes, s = m.bp * (V.arterial + V.venous + V.portal);
  ['lung', 'gut', 'liver', 'kidney', 'muscle', 'adipose', 'brain', 'heart', 'skin', 'bone', 'rest'].forEach(function(t){ s += m.kp[t] * V[t]; });
  return s;
}
function pkcBaseModel(clintU, over){
  var o = Object.assign({}, PKC_BASE, over || {}); o.clintU = clintU;
  return PKE.pbpk.model({ logP: o.logP, fup: o.fup, bp: o.bp, clintU: o.clintU, clr: o.clr, ka: o.ka, F: o.F, bw: o.bw });
}
function pkcScaleFlows(m, f){ var Q = m.phys.flows, k; for (k in Q) if (k !== 'liverTotal' && k !== 'cardiacOutput') Q[k] *= f; Q.liverTotal = Q.liverArtery + Q.gut; Q.cardiacOutput = Q.gut + Q.liverArtery + Q.kidney + Q.muscle + Q.adipose + Q.brain + Q.heart + Q.skin + Q.bone + Q.rest; }
function pkcSim(m, tEnd){
  var dt = PKE.pbpk.stableDt(m), every = Math.max(1, Math.round(0.1 / dt)), r = PKE.pbpk.simulate(m, { route: 'oral', dose: PKC_BASE.dose }, tEnd, dt, every), cmax = 0, i;
  for (i = 0; i < r.plasma.length; i++) if (r.plasma[i] > cmax) cmax = r.plasma[i];
  return { auc: r.aucPlasma[r.aucPlasma.length - 1], cmax: cmax, massError: r.massError };
}

/* ---------- 1. Sensitivity ---------- */
var PKC_PARAMS = [
  { id: 'clint', label: 'CLint (unbound hepatic)', ml: 'HLM model of this course' },
  { id: 'fup', label: 'Fraction unbound in plasma', ml: 'none in this course' },
  { id: 'bp', label: 'Blood to plasma ratio', ml: 'none in this course' },
  { id: 'kp', label: 'Tissue partition (all Kp)', ml: 'derived from logP and fup' },
  { id: 'ka', label: 'Absorption rate ka', ml: 'none in this course' },
  { id: 'F', label: 'Fraction reaching gut tissue', ml: 'none in this course' },
  { id: 'clr', label: 'Renal clearance', ml: 'none in this course' },
  { id: 'flows', label: 'Organ blood flows (all)', ml: 'physiology, not predicted' },
  { id: 'vad', label: 'Adipose volume', ml: 'physiology, not predicted' }
];
function pkcPerturbed(clintU, id, f){
  var b = PKC_BASE, o = {};
  if (id === 'clint') return pkcBaseModel(clintU * f);
  if (id === 'fup') return pkcBaseModel(clintU, { fup: Math.min(1, b.fup * f) });
  if (id === 'bp') return pkcBaseModel(clintU, { bp: b.bp * f });
  if (id === 'ka') return pkcBaseModel(clintU, { ka: b.ka * f });
  if (id === 'F') return pkcBaseModel(clintU, { F: Math.min(1, b.F * f) });
  if (id === 'clr') return pkcBaseModel(clintU, { clr: b.clr * f });
  var m = pkcBaseModel(clintU);
  if (id === 'kp') { Object.keys(m.kp).forEach(function(t){ m.kp[t] *= f; }); }
  if (id === 'flows') pkcScaleFlows(m, f);
  if (id === 'vad') m.phys.volumes.adipose *= f;
  return m;
}
LESSONS.pk.add({ level: 'advanced', short: 'Sensitivity', title: 'Parameter sensitivity',
  change: 'Press Run to repeat the analysis. Change the baseline log10 CLint and run again.',
  watch: 'A PBPK prediction is a function of many inputs, and an error in an input changes the prediction by an amount that depends on the model structure. Local one-at-a-time sensitivity analysis changes one input at a time by a fixed factor and records the change in the output. Here each input is multiplied by 0.5 and by 2 for a 100 mg oral dose of a typical compound, and the outputs are AUC from 0 to 24 h and Cmax from the whole-body model in this course. The factor of 2 is applied to every input in the same way, so the bars compare inputs and are not a statement of their real uncertainty. Every baseline value except CLint is an assumption, listed below the chart.',
  build: function(api){
    var cp = api.el('pn', 'left:28px;top:20px;width:604px;height:330px', '<h4>Change in the output for 0.5 and 2 times each input</h4><canvas width="574" height="284"></canvas>');
    var sp = api.el('pn', 'left:644px;top:20px;width:288px;height:330px;overflow:auto', '<div class="sv" style="font-size:12.5px;line-height:1.5;color:' + C.char + '"></div>');
    var mp = api.el('pn', 'left:28px;top:362px;width:904px;height:158px;overflow:auto', '<div class="mv" style="font-size:12.5px;line-height:1.5;color:' + C.char + '"></div>');
    var ctl = api.addControl('Baseline log10 CLint (mL/min/kg) <input type="range" min="-0.5" max="3" step="0.05" value="1.3" style="width:180px"> <b class="cv"></b> <button type="button" class="btn sm" data-run>Run</button> <span class="rs" style="font-size:12px"></span>');
    var rg = qs('input', ctl), cvEl = qs('.cv', ctl), rs = qs('.rs', ctl), res = null, busy = false, H = null;
    function rows(){ return res ? PKC_PARAMS.map(function(p, k){ var r = res.rows[p.id]; return { p: p, lo: r.lo, hi: r.hi, aucLo: Math.log2(r.aucLo / res.base.auc), aucHi: Math.log2(r.aucHi / res.base.auc), cmLo: Math.log2(r.cmLo / res.base.cmax), cmHi: Math.log2(r.cmHi / res.base.cmax) }; }) : []; }
    function draw(){
      var g = cv(qs('canvas', cp)), x = g.x; x.clearRect(0, 0, g.w, g.h);
      if (!res) { x.fillStyle = C.mid; x.font = '13px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'center'; x.fillText(busy ? 'Simulating' : 'Press Run', g.w / 2, g.h / 2); return; }
      var R = rows().sort(function(a, b){ return Math.max(Math.abs(b.aucLo), Math.abs(b.aucHi)) - Math.max(Math.abs(a.aucLo), Math.abs(a.aucHi)); }), lim = 0, left = 188, panelW = (g.w - left - 18) / 2;
      R.forEach(function(r){ lim = Math.max(lim, Math.abs(r.aucLo), Math.abs(r.aucHi), Math.abs(r.cmLo), Math.abs(r.cmHi)); }); lim = Math.max(1, Math.ceil(lim));
      var top = 30, rh = (g.h - top - 34) / R.length, cx = [left + panelW / 2, left + panelW + 8 + panelW / 2];
      x.font = '700 12px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'center'; x.fillStyle = C.char; x.fillText('AUC 0 to 24 h', cx[0], 16); x.fillText('Cmax', cx[1], 16);
      cx.forEach(function(c){ x.strokeStyle = C.silver; x.lineWidth = 1; for (var t = -lim; t <= lim; t++) { var px = c + t / lim * (panelW / 2 - 6); x.beginPath(); x.moveTo(px, top - 4); x.lineTo(px, g.h - 34); x.stroke(); x.fillStyle = C.mid; x.font = '10.5px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'center'; x.fillText((t === 0 ? '1' : (t > 0 ? Math.pow(2, t) : '1/' + Math.pow(2, -t))) + '', px, g.h - 20); } });
      x.fillStyle = C.char; x.font = '11.5px Aptos, "Segoe UI", sans-serif'; x.fillText('output relative to baseline (log scale)', left + panelW + 4, g.h - 5);
      R.forEach(function(r, k){
        var y = top + k * rh + rh / 2; x.textAlign = 'right'; x.fillStyle = r.p.id === 'clint' ? C.dred : C.char; x.font = (r.p.id === 'clint' ? '700 ' : '') + '11.5px Aptos, "Segoe UI", sans-serif'; x.fillText(r.p.label, left - 8, y + 4);
        [[cx[0], r.aucLo, r.aucHi], [cx[1], r.cmLo, r.cmHi]].forEach(function(b){ var c = b[0], sc = (panelW / 2 - 6) / lim;
          x.fillStyle = C.bg; x.fillRect(Math.min(c, c + b[1] * sc), y - rh * 0.3, Math.abs(b[1] * sc), rh * 0.6); x.fillStyle = C.red; x.fillRect(Math.min(c, c + b[2] * sc), y - rh * 0.3, Math.abs(b[2] * sc), rh * 0.6); });
      });
      x.textAlign = 'left'; x.fillStyle = C.bg; x.fillRect(left, g.h - 34 + 14, 0, 0);
    }
    function panels(){
      if (!res) { qs('.sv', sp).innerHTML = busy ? '<div class="ol">Running</div>' : '<div class="ol">Result</div>'; return; }
      var R = rows(), ord = R.slice().sort(function(a, b){ return Math.max(Math.abs(b.aucLo), Math.abs(b.aucHi)) - Math.max(Math.abs(a.aucLo), Math.abs(a.aucHi)); }), rank = ord.map(function(r){ return r.p.id; }).indexOf('clint') + 1, h = PKC_HLM;
      var f = Math.pow(10, h.rmse);
      qs('.sv', sp).innerHTML = '<div class="ol">Baseline</div><div style="margin-bottom:8px">AUC 0 to 24 h <b>' + pkcFmt(res.base.auc) + '</b> mg h/L, Cmax <b>' + pkcFmt(res.base.cmax) + '</b> mg/L.</div><div class="ol">Test error of the course model</div><div>The HLM boosted-trees model has a test RMSE of <b>' + h.rmse.toFixed(2) + '</b> log10 units, a factor of <b>' + f.toFixed(1) + '</b> in CLint. Moving CLint by that factor changes AUC by <b>' + pkcLog2Fmt(Math.log2(res.rmseUp.auc / res.base.auc)) + '</b> (CLint times ' + f.toFixed(1) + ') and <b>' + pkcLog2Fmt(Math.log2(res.rmseDown.auc / res.base.auc)) + '</b> (CLint divided by ' + f.toFixed(1) + '). CLint ranks <b>' + rank + '</b> of ' + R.length + ' by the effect on AUC at a factor of 2.</div>';
    }
    function table(){
      var h = PKC_HLM; qs('.mv', mp).innerHTML = '<div class="ol">Baseline assumptions</div>' +
        '<div>Oral dose 100 mg, body weight 70 kg, logP ' + PKC_BASE.logP.toFixed(1) + ' (sets the tissue partition coefficients by the model of PKE.pbpk), fraction unbound in plasma ' + PKC_BASE.fup + ', blood to plasma ratio ' + PKC_BASE.bp + ', renal clearance ' + PKC_BASE.clr + ' L/h, ka ' + PKC_BASE.ka + ' per h, fraction reaching the gut tissue ' + PKC_BASE.F + '. Hepatic clearance is the well-stirred model on the HLM CLint, used as an intrinsic clearance per kg body weight with no microsomal protein or liver weight scaling. Output grid: simulated to 24 h, concentrations sampled every 0.1 h, AUC integrated in the model. The inputs a machine learning model would supply, with the source in this course: CLint (HLM boosted trees), logP (computed from structure by RDKit). The other inputs have no model in this course. Changing fup also changes the partition coefficients through the model. A change of 2 in fraction reaching the gut tissue and in fup is capped at 1.</div>';
    }
    async function run(){
      if (busy) return; busy = true; var tok = ++api.tok; api.busy('Simulating'); draw(); panels();
      try {
        await pkcHLM(); var clintU = PKE.pbpk.clintUFromHLM(+rg.value, PKC_BASE.bw, 1), t0 = performance.now(), out = { rows: {} }, k = 0, total = PKC_PARAMS.length * 2 + 3, f = Math.pow(10, PKC_HLM.rmse);
        function step(label){ k++; rs.textContent = label + ' (' + k + ' of ' + total + ')'; return pkcTick(); }
        out.base = pkcSim(pkcBaseModel(clintU), 24); await step('Baseline');
        for (var i = 0; i < PKC_PARAMS.length; i++) {
          var p = PKC_PARAMS[i], lo = pkcSim(pkcPerturbed(clintU, p.id, 0.5), 24); await step(p.label); var hi = pkcSim(pkcPerturbed(clintU, p.id, 2), 24); await step(p.label);
          out.rows[p.id] = { aucLo: lo.auc, aucHi: hi.auc, cmLo: lo.cmax, cmHi: hi.cmax };
        }
        out.rmseUp = pkcSim(pkcPerturbed(clintU, 'clint', f), 24); await step('CLint, one test RMSE'); out.rmseDown = pkcSim(pkcPerturbed(clintU, 'clint', 1 / f), 24); await step('CLint, one test RMSE');
        out.secs = (performance.now() - t0) / 1000; res = out; rs.textContent = 'Done in ' + out.secs.toFixed(1) + ' s.';
      } finally { api.idle(); busy = false; }
      draw(); panels();
    }
    qs('[data-run]', ctl).addEventListener('click', run);
    rg.addEventListener('input', function(){ cvEl.textContent = (+rg.value).toFixed(2); });
    async function render(){ await pkcHLM(); rg.value = Math.round(PKC_HLM.median * 20) / 20; cvEl.textContent = (+rg.value).toFixed(2); table(); if (!res) await run(); else { draw(); panels(); } }
    async function play(tok){
      await render(); api.cap('Each bar is the change in the output when one input is multiplied by 2 (red) or by 0.5 (grey).'); await api.sleep(3600);
      api.cap('The inputs are ranked by the larger of the two effects on AUC.'); await api.sleep(3200);
      api.cap('The panel on the right applies the test error of the course CLint model, instead of a factor of 2.'); await api.sleep(3600); api.cap(null);
    }
    return { render: render, play: play };
  } });
GUIDE.steps['pk/Sensitivity'] = { notice: 'The fraction reaching the gut tissue, fup and CLint change AUC the most. The absorption rate ka changes Cmax but not AUC.', q: 'Why is the effect of CLint on AUC smaller than proportional?', a: 'Hepatic clearance saturates at hepatic blood flow as CLint rises, so the extraction ratio approaches 1 and AUC stops falling in proportion to CLint.' };

/* ---------- 2. Absorption and distribution ---------- */
/* Transit compartment absorption (Savic et al. 2007, J Pharmacokinet Pharmacodyn 34:711-726): n transit compartments in series with rate ktr, followed by an absorption
   compartment with first-order rate; ktr = (n + 1) / MTT, where MTT is the mean transit time. Here the absorption compartment also has rate ktr, so that the mean absorption time is MTT.
   The primary paper was not read; the parameterisation is from the PKPDsim vignette on absorption models and a Certara poster that cite it. */
function pkcAbsorb(n, mtt, V, CL, dose, tEnd){
  /* states: n transit compartments (0 to n-1), the absorption compartment (n), the central compartment (n + 1); n + 1 compartments before the central one, each with rate ktr */
  var ktr = (n + 1) / mtt, ke = CL / V, g = function(t, y){ var d = new Array(n + 2).fill(0), i; d[0] = -ktr * y[0]; for (i = 1; i <= n; i++) d[i] = ktr * (y[i - 1] - y[i]); d[n + 1] = ktr * y[n] - ke * y[n + 1]; return d; };
  var y0 = new Array(n + 2).fill(0); y0[0] = dose;
  var r = PKE.rk4(g, y0, 0, tEnd, 0.02), c = r.y.map(function(y){ return y[n + 1] / V; });
  return { t: r.t, c: c };
}
LESSONS.pk.add({ level: 'advanced', short: 'Absorption and distribution', title: 'Absorption and distribution',
  change: 'Change the mean transit time and the number of transit compartments. Change logP and the fraction unbound.',
  watch: 'Absorption and distribution shape the profile independently of clearance. Absorption: a first-order input has an exponential delay with mean 1 / ka, and a chain of n transit compartments in series followed by an absorption compartment, all with rate ktr = (n + 1) / MTT, has the same mean absorption time MTT but a delayed, narrower input. Both curves below have the same dose, clearance, volume and mean absorption time, so AUC is identical and only tmax and Cmax differ. Distribution: in the whole-body model the tissue to plasma partition coefficients Kp set the volume of distribution at steady state, Vss, equal to the blood amount plus the sum of Kp times tissue volume over all tissues, and the terminal half-life follows from the slowest eigenvalue of the model.',
  build: function(api){
    var cp = api.el('pn', 'left:28px;top:20px;width:440px;height:360px', '<h4>Absorption: same mean absorption time</h4><canvas width="410" height="306"></canvas>');
    var dp = api.el('pn', 'left:488px;top:20px;width:444px;height:360px', '<h4>Distribution: whole-body model</h4><canvas width="414" height="306"></canvas>');
    var rp = api.el('pn', 'left:28px;top:392px;width:904px;height:128px;overflow:auto', '<div class="rv" style="font-size:12.5px;line-height:1.5;color:' + C.char + '"></div>');
    var ctl = api.addControl('MTT (h) <input id="pkc2mtt" type="range" min="0.25" max="6" step="0.25" value="2" style="width:110px"> <b class="v1"></b> Transit compartments n <input id="pkc2n" type="range" min="0" max="10" step="1" value="4" style="width:90px"> <b class="v2"></b> logP <input id="pkc2lp" type="range" min="0" max="4" step="0.1" value="2" style="width:100px"> <b class="v3"></b> fup <input id="pkc2fu" type="range" min="0.05" max="1" step="0.05" value="0.5" style="width:100px"> <b class="v4"></b>');
    var PK = { V: 50, CL: 10, dose: 100 };
    function eigen(logP, fup, clintU){ var m = pkcBaseModel(clintU, { logP: logP, fup: fup }), A = pkcMatrix(m), lam = pkcTerminalRate(A), cl = PKE.pbpk.modelHepatic(m).clhPlasma + m.clr, vss = pkcVss(m); return { vss: vss, thalf: Math.LN2 / lam, cl: cl, mrt: vss / cl }; }
    async function draw(){
      var mtt = +$('pkc2mtt').value, n = +$('pkc2n').value, lp = +$('pkc2lp').value, fu = +$('pkc2fu').value; qs('.v1', ctl).textContent = mtt; qs('.v2', ctl).textContent = n; qs('.v3', ctl).textContent = lp.toFixed(1); qs('.v4', ctl).textContent = fu.toFixed(2);
      var tEnd = 24, ch = pkcAbsorb(n, mtt, PK.V, PK.CL, PK.dose, tEnd), fo = PKE.oneComp({ dose: PK.dose, V: PK.V, CL: PK.CL, ka: 1 / mtt, F: 1, route: 'oral' }), foPts = [], chPts = [], i, cmaxC = 0, tmaxC = 0;
      for (i = 0; i < ch.t.length; i += 5) { chPts.push([ch.t[i], ch.c[i]]); if (ch.c[i] > cmaxC) { cmaxC = ch.c[i]; tmaxC = ch.t[i]; } foPts.push([ch.t[i], fo.conc(ch.t[i])]); }
      lineChart(qs('canvas', cp), [{ pts: foPts, color: C.char, w: 2.6, nodot: true, label: 'first-order, ka = 1 / MTT' }, { pts: chPts, color: C.red, w: 2.8, nodot: true, label: n + ' transit compartments' }], { xl: 'time (h)', yl: 'plasma concentration (mg/L)', x0: 0, x1: tEnd, y0: 0 });
      var h = await pkcHLM(), clintU = PKE.pbpk.clintUFromHLM(h.median, PKC_BASE.bw, 1), pts = [], pts2 = [], lps = [], k;
      for (k = 0; k <= 16; k++) { var l = 0.25 * k, e = eigen(l, fu, clintU); pts.push([l, Math.log10(e.vss)]); pts2.push([l, Math.log10(e.thalf)]); }
      var cur = eigen(lp, fu, clintU);
      var tk = function(a, b){ var o = []; for (var e = a; e <= b; e++) o.push({ v: e, t: String(Math.pow(10, e)) }); return o; };
      lineChart(qs('canvas', dp), [{ pts: pts, color: C.char, w: 2.6, nodot: true, label: 'Vss (L)' }, { pts: pts2, color: C.red, w: 2.6, nodot: true, label: 'terminal half-life (h)' }, { pts: [[lp, Math.log10(cur.vss)]], color: C.char, w: 1, label: '' }, { pts: [[lp, Math.log10(cur.thalf)]], color: C.red, w: 1, label: '' }], { xl: 'logP', yl: 'Vss or half-life (log scale)', x0: 0, x1: 4, y0: 0, y1: 3.5, yticks: [{ v: 0, t: '1' }, { v: 0.5, t: '3' }, { v: 1, t: '10' }, { v: 1.5, t: '30' }, { v: 2, t: '100' }, { v: 2.5, t: '300' }, { v: 3, t: '1000' }, { v: 3.5, t: '3000' }] });
      qs('.rv', rp).innerHTML = '<div style="display:flex;gap:26px;flex-wrap:wrap;margin-bottom:6px"><div><div class="ol">First-order input</div><b>tmax ' + pkcFmt(fo.summary.tmax) + ' h, Cmax ' + pkcFmt(fo.summary.cmax) + ' mg/L</b></div><div><div class="ol">Transit chain</div><b>tmax ' + pkcFmt(tmaxC) + ' h, Cmax ' + pkcFmt(cmaxC) + ' mg/L</b></div><div><div class="ol">Whole-body model at logP ' + lp.toFixed(1) + ', fup ' + fu.toFixed(2) + '</div><b>Vss ' + pkcFmt(cur.vss) + ' L, mean residence time ' + pkcFmt(cur.mrt) + ' h, terminal half-life ' + pkcFmt(cur.thalf) + ' h</b></div></div>' +
        '<div>Absorption: one-compartment disposition with V = ' + PK.V + ' L and CL = ' + PK.CL + ' L/h, dose ' + PK.dose + ' mg, complete absorption. The transit chain is dA0/dt = -ktr A0, dAi/dt = ktr (Ai-1 - Ai), with n transit compartments and one absorption compartment at the same rate, ktr = (n + 1) / MTT (Savic et al. 2007). Distribution: perfusion-limited whole-body model, clearance as in the sensitivity step (the HLM median CLint as an assumption, renal 1 L/h), Kp from the model of PKE.pbpk at the chosen logP and fup. Vss = bp (Va + Vv + Vp) + sum of Kp V over tissues. The terminal half-life is ln 2 divided by the slowest eigenvalue of the linear model, which is exact for this model. The models assume neutral compounds.</div>';
    }
    ['pkc2mtt', 'pkc2n', 'pkc2lp', 'pkc2fu'].forEach(function(id){ $(id).addEventListener('input', draw); });
    async function render(){ await draw(); }
    async function play(tok){
      await draw(); api.cap('Both inputs have the same mean absorption time, so AUC is identical. The transit chain delays the peak and raises it.'); await api.sleep(3800);
      await api.anim(2600, function(t){ $('pkc2n').value = Math.round(4 - 4 * t); draw(); }); api.cap('With no transit compartment the chain reduces to the first-order input.'); await api.sleep(3200);
      $('pkc2n').value = 4; await api.anim(3000, function(t){ $('pkc2lp').value = (0.5 + 3 * t).toFixed(1); draw(); }); api.cap('A higher logP raises Kp in adipose and other lipid-rich tissue, so Vss and the terminal half-life increase while CL is unchanged.'); await api.sleep(3800); api.cap(null);
    }
    return { render: render, play: play };
  } });
GUIDE.steps['pk/Absorption and distribution'] = { notice: 'AUC is the same for both inputs. Only tmax and Cmax differ.', q: 'Why does the terminal half-life rise with logP at fixed clearance?', a: 'The volume of distribution rises with the partition coefficients, and the half-life is proportional to the volume divided by the clearance.' };

/* ---------- 3. DDI models ---------- */
/* Equations: ICH M12 Drug interaction studies, Step 4 (21 May 2024), Section 2.1.2.1 (basic method: risk of reversible inhibition excluded if Ki,u > 50 Cmax,u, that is Cmax,u / Ki,u < 0.02,
   with Cmax,u the unbound Cmax at the highest recommended dose at steady state) and Section 7.5.1 (mechanistic static model, AUCR = [1 / ([Ag Bg Cg] (1 - Fg) + Fg)] [1 / ([Ah Bh Ch] fm + (1 - fm))],
   Ah = 1 / (1 + [I]h / Ki), [I]h = fu,p (Cmax + Fa Fg ka Dose / Qh / RB); AUCR between 0.80 and 1.25 indicates low risk; the equation assumes negligible extrahepatic clearance). */
function pkcAucr(fm, x){ return 1 / (fm / (1 + x) + (1 - fm)); }
function pkcDynamic(fm, x, kInh, cl0, V, ka, F, dose, mode, tEnd){
  var f = function(t, y){ var inh = mode === 'decay' ? x * Math.exp(-kInh * t) : (mode === 'const' ? x : 0), cl = cl0 * (fm / (1 + inh) + (1 - fm)); return [-ka * y[0], F * ka * y[0] - cl / V * y[1], y[1] / V]; };
  var r = PKE.rk4(f, [dose, 0, 0], 0, tEnd, 0.02), c = r.y.map(function(y){ return y[1] / V; }), cmax = 0; c.forEach(function(v){ if (v > cmax) cmax = v; });
  return { t: r.t, c: c, auc: r.y[r.y.length - 1][2], cmax: cmax };
}
LESSONS.pk.add({ level: 'advanced', short: 'DDI models', title: 'Drug-drug interaction models',
  change: 'Set the fraction metabolised by the inhibited enzyme and the ratio of inhibitor concentration to Ki.',
  watch: 'A static mechanistic model converts an in vitro inhibition constant into a predicted change in exposure of a victim drug. For reversible inhibition of one enzyme in the liver, with negligible intestinal and extrahepatic clearance, the AUC ratio is AUCR = 1 / (fm / (1 + [I]/Ki) + (1 - fm)), where fm is the fraction of hepatic clearance mediated by the inhibited enzyme, [I] is the unbound inhibitor concentration at the enzyme and Ki the unbound inhibition constant (ICH M12, Section 7.5.1). The static model uses one inhibitor concentration and therefore ignores its time course. The dynamic panel simulates the victim profile with the clearance reduced by the static factor, and with the clearance following an inhibitor concentration that decays after a single dose.',
  build: function(api){
    var cp = api.el('pn', 'left:28px;top:20px;width:444px;height:330px', '<h4>Static AUC ratio</h4><canvas width="414" height="276"></canvas>');
    var dp = api.el('pn', 'left:488px;top:20px;width:444px;height:330px', '<h4>Victim drug profile, dynamic simulation</h4><canvas width="414" height="276"></canvas>');
    var rp = api.el('pn', 'left:28px;top:362px;width:904px;height:158px;overflow:auto', '<div class="dv" style="font-size:12.5px;line-height:1.5;color:' + C.char + '"></div>');
    var ctl = api.addControl('fm <input id="pkc3fm" type="range" min="0" max="1" step="0.05" value="0.8" style="width:120px"> <b class="v1"></b> [I] / Ki <input id="pkc3x" type="range" min="-2" max="2" step="0.05" value="0.6" style="width:150px"> <b class="v2"></b> Inhibitor half-life (h) <input id="pkc3t" type="range" min="1" max="24" step="1" value="6" style="width:100px"> <b class="v3"></b>');
    var P = { cl: 10, V: 50, ka: 1, F: 0.8, dose: 100 };
    function draw(){
      var fm = +$('pkc3fm').value, x = Math.pow(10, +$('pkc3x').value), th = +$('pkc3t').value; qs('.v1', ctl).textContent = fm.toFixed(2); qs('.v2', ctl).textContent = pkcFmt(x); qs('.v3', ctl).textContent = th;
      var curves = [0.2, 0.5, 0.8, 0.95].map(function(f, k){ var pts = [], e; for (e = -2; e <= 2.001; e += 0.1) pts.push([e, pkcAucr(f, Math.pow(10, e))]); return { pts: pts, color: ['#A1B1C4', '#6F8196', '#34373F', '#891515'][k], w: 2.2, nodot: true, label: 'fm ' + f }; });
      var cur = []; for (var e = -2; e <= 2.001; e += 0.1) cur.push([e, pkcAucr(fm, Math.pow(10, e))]);
      var ymax = Math.max(2, Math.ceil(pkcAucr(0.95, 100))), xt = [-2, -1, 0, 1, 2].map(function(v){ return { v: v, t: String(Math.pow(10, v)) }; });
      lineChart(qs('canvas', cp), curves.concat([{ pts: cur, color: C.red, w: 3.4, nodot: true, label: 'selected fm ' + fm.toFixed(2) }, { pts: [[Math.log10(x), pkcAucr(fm, x)]], color: C.red, w: 1, label: '' }]), { xl: '[I] / Ki (log scale)', yl: 'AUC ratio', x0: -2, x1: 2, y0: 1, y1: ymax, xticks: xt });
            var s0 = pkcAucr(fm, 1e-9), st = pkcAucr(fm, x), tEnd = 48, noI = pkcDynamic(fm, x, 0, P.cl, P.V, P.ka, P.F, P.dose, 'none', tEnd), cst = pkcDynamic(fm, x, 0, P.cl, P.V, P.ka, P.F, P.dose, 'const', tEnd), dyn = pkcDynamic(fm, x, Math.LN2 / th, P.cl, P.V, P.ka, P.F, P.dose, 'decay', tEnd), pts = function(r){ var o = [], i; for (i = 0; i < r.t.length; i += 10) o.push([r.t[i], r.c[i]]); return o; };
      lineChart(qs('canvas', dp), [{ pts: pts(noI), color: C.char, w: 2.6, nodot: true, label: 'no inhibitor' }, { pts: pts(cst), color: C.red, w: 2.8, nodot: true, label: 'CL reduced by the static factor' }, { pts: pts(dyn), color: '#6F8196', w: 2.4, nodot: true, dash: [6, 4], label: 'inhibitor decays after one dose' }], { xl: 'time (h)', yl: 'plasma concentration (mg/L)', x0: 0, x1: tEnd, y0: 0 });
      var basic = x < 0.02;
      qs('.dv', rp).innerHTML = '<div style="display:flex;gap:26px;flex-wrap:wrap;margin-bottom:6px"><div><div class="ol">Static AUCR</div><b style="font-size:18px;color:' + C.red + '">' + st.toFixed(2) + '</b></div><div><div class="ol">Dynamic, constant inhibitor</div><b style="font-size:18px">' + (cst.auc / noI.auc).toFixed(2) + '</b> <span style="font-size:11.5px">(AUC 0 to 48 h, equal to the static value)</span></div><div><div class="ol">Dynamic, decaying inhibitor</div><b style="font-size:18px">' + (dyn.auc / noI.auc).toFixed(2) + '</b> <span style="font-size:11.5px">(AUC 0 to 48 h)</span></div><div><div class="ol">Cmax ratio, constant inhibitor</div><b style="font-size:18px">' + (cst.cmax / noI.cmax).toFixed(2) + '</b></div></div>' +
        '<div>ICH M12 (Step 4, 21 May 2024): the basic method excludes the risk of reversible inhibition if Cmax,u / Ki,u is below 0.02 (Section 2.1.2.1), where Cmax,u is the unbound Cmax at the highest recommended dose at steady state. At the selected ratio [I] / Ki = ' + pkcFmt(x) + ' the basic criterion is ' + (basic ? 'met' : 'not met') + '. When the model estimates an AUCR between 0.80 and 1.25 the guideline considers the risk of a clinically relevant interaction low (Section 7.5.1). The guideline states that mechanistic static models used with its gut and liver concentration equations can be overly conservative and give false positive results. Limits of the model shown here: one enzyme, reversible inhibition only, no intestinal term (Fg = 1), negligible extrahepatic clearance, one inhibitor concentration. Victim drug assumptions: CL ' + P.cl + ' L/h, V ' + P.V + ' L, ka ' + P.ka + ' per h, F ' + P.F + ', dose ' + P.dose + ' mg. The decaying inhibitor is a first-order decline from the selected [I] / Ki, an extension for illustration and not a model from the guideline.</div>';
    }
    ['pkc3fm', 'pkc3x', 'pkc3t'].forEach(function(id){ $(id).addEventListener('input', draw); });
    async function play(tok){
      $('pkc3fm').value = 0.8; $('pkc3x').value = 0.6; draw(); api.cap('For fm = 0.8 and [I] / Ki near 4, the static model predicts a rise in AUC of about 2.8-fold.'); await api.sleep(3600);
      await api.anim(2600, function(t){ $('pkc3fm').value = (0.8 - 0.6 * t).toFixed(2); draw(); }); api.cap('A low fm limits the effect: the other clearance pathways remain open.'); await api.sleep(3400);
      $('pkc3fm').value = 0.8; await api.anim(3000, function(t){ $('pkc3x').value = (0.6 - 2.6 * t).toFixed(2); draw(); }); api.cap('Below [I] / Ki of 0.02 the basic method excludes the risk of reversible inhibition.'); await api.sleep(3600); api.cap(null);
    }
    return { render: draw, play: play };
  } });
GUIDE.steps['pk/DDI models'] = { notice: 'The curves for a high fm rise steeply when [I] / Ki exceeds 1.', q: 'Why can the static model overstate an interaction?', a: 'It uses one inhibitor concentration, such as the steady-state maximum, and ignores that the inhibitor concentration falls between doses.' };

/* ---------- 4. Estimation ---------- */
/* Pooled objective: sum of squared residuals of the one-compartment oral model (F = 1, so CL/F and V/F) for one Theoph subject, with ka profiled out by a bounded
   golden-section search on log ka for each pair (CL/F, V/F). */
var PKC_TH = null;
async function pkcTheoph(){ if (PKC_TH) return PKC_TH; var r = await fetch('data/pk_public.json?v=1'); PKC_TH = (await r.json()).theoph; return PKC_TH; }
function pkcPred(dose, cl, v, ka, t){ var ke = cl / v; if (Math.abs(ka - ke) < 1e-7 * ka) return dose * ka * t * Math.exp(-ke * t) / v; return dose * ka / (v * (ka - ke)) * (Math.exp(-ke * t) - Math.exp(-ka * t)); }
function pkcSSE(S, cl, v, ka){ var s = 0, i; for (i = 0; i < S.t.length; i++) { var r = S.conc[i] - pkcPred(S.doseMg, cl, v, ka, S.t[i]); s += r * r; } return s; }
function pkcProfile(S, cl, v){
  var lo = Math.log(0.05), hi = Math.log(20), g = 0.6180339887, a = lo, b = hi, c = b - g * (b - a), d = a + g * (b - a), fc = pkcSSE(S, cl, v, Math.exp(c)), fd = pkcSSE(S, cl, v, Math.exp(d)), k;
  for (k = 0; k < 32; k++) { if (fc < fd) { b = d; d = c; fd = fc; c = b - g * (b - a); fc = pkcSSE(S, cl, v, Math.exp(c)); } else { a = c; c = d; fc = fd; d = a + g * (b - a); fd = pkcSSE(S, cl, v, Math.exp(d)); } }
  var lk = (a + b) / 2; return { sse: pkcSSE(S, cl, v, Math.exp(lk)), ka: Math.exp(lk) };
}
function pkcResid(S, cl, v){ var ka = pkcProfile(S, cl, v).ka, i, r = []; for (i = 0; i < S.t.length; i++) r.push(S.conc[i] - pkcPred(S.doseMg, cl, v, ka, S.t[i])); return r; }
LESSONS.pk.add({ level: 'advanced', short: 'Estimation', title: 'Parameter estimation',
  change: 'Select a subject, select an optimiser, click the map to set a start, and press Run.',
  watch: 'Estimating PK parameters from observed concentrations is an optimisation: choose the parameters that minimise the sum of squared residuals between the model and the data. Here a one-compartment oral model is fitted to one subject of the public Theoph data set (12 subjects, oral theophylline). The map shows the objective over CL/F and V/F, with ka chosen to minimise the objective at every point. Gradient descent on the logarithm of the objective follows the local slope with a fixed learning rate, as in lesson 09. Levenberg-Marquardt, the standard least-squares method, uses the Jacobian of the residuals and converges in a few iterations. The same objective, optimiser and local-minimum behaviour apply when the model is a neural network.',
  build: function(api){
    var mp = api.el('pn', 'left:28px;top:20px;width:440px;height:500px', '<h4>Objective: sum of squared residuals (log scale)</h4><canvas width="410" height="414"></canvas><div class="mm" style="font-size:11.5px;color:' + C.mid + ';margin-top:4px"></div>');
    var dp = api.el('pn', 'left:488px;top:20px;width:444px;height:236px', '<h4>Data and fit at the current iterate</h4><canvas width="414" height="186"></canvas>');
    var rp = api.el('pn', 'left:488px;top:266px;width:444px;height:254px;overflow:auto', '<div class="ev" style="font-size:12.5px;line-height:1.5;color:' + C.char + '"></div>');
    var ctl = api.addControl('Subject <select id="pkc4s"></select> Optimiser <span class="seg" data-opt><button type="button" data-v="0" class="on">Gradient descent</button><button type="button" data-v="1">Levenberg-Marquardt</button></span> Learning rate <input id="pkc4lr" type="range" min="-3" max="-0.5" step="0.1" value="-1.5" style="width:100px"> <b class="v1"></b> <button type="button" class="btn sm" data-run>Run</button> <button type="button" class="btn sm ghost" data-reset>Reset</button>');
    var TH = null, S = null, grid = null, opt = 0, start = { u: Math.log(0.9), v: Math.log(70) }, path = [], cur = null, fin = null, LIM = { u0: Math.log(0.3), u1: Math.log(12), v0: Math.log(6), v1: Math.log(160) };
    var NG = 56;
    function makeGrid(){
      grid = []; var min = Infinity, i, j;
      for (i = 0; i < NG; i++) { grid[i] = []; for (j = 0; j < NG; j++) { var u = LIM.u0 + (LIM.u1 - LIM.u0) * (j + 0.5) / NG, v = LIM.v0 + (LIM.v1 - LIM.v0) * (i + 0.5) / NG, s = pkcProfile(S, Math.exp(u), Math.exp(v)).sse; grid[i][j] = s; if (s < min) min = s; } }
      grid.min = min;
    }
    function obj(u, v){ return Math.log(pkcProfile(S, Math.exp(u), Math.exp(v)).sse); }
    function runGD(){
      var lr = Math.pow(10, +$('pkc4lr').value), u = start.u, v = start.v, P = [{ u: u, v: v, f: obj(u, v) }], k, h = 1e-4;
      for (k = 0; k < 150; k++) { var gu = (obj(u + h, v) - obj(u - h, v)) / (2 * h), gv = (obj(u, v + h) - obj(u, v - h)) / (2 * h); u -= lr * gu; v -= lr * gv; u = clamp(u, LIM.u0 - 1, LIM.u1 + 1); v = clamp(v, LIM.v0 - 1, LIM.v1 + 1); P.push({ u: u, v: v, f: obj(u, v) }); if (Math.hypot(gu, gv) < 1e-4) break; }
      return P;
    }
    function runLM(){
      var u = start.u, v = start.v, lam = 1e-2, P = [{ u: u, v: v, f: obj(u, v) }], k, h = 1e-4;
      for (k = 0; k < 40; k++) {
        var r0 = pkcResid(S, Math.exp(u), Math.exp(v)), ru = pkcResid(S, Math.exp(u + h), Math.exp(v)), rv = pkcResid(S, Math.exp(u), Math.exp(v + h)), n = r0.length, Ju = [], Jv = [], i;
        for (i = 0; i < n; i++) { Ju.push((ru[i] - r0[i]) / h); Jv.push((rv[i] - r0[i]) / h); }
        var a = 0, b = 0, c = 0, g1 = 0, g2 = 0; for (i = 0; i < n; i++) { a += Ju[i] * Ju[i]; b += Ju[i] * Jv[i]; c += Jv[i] * Jv[i]; g1 += Ju[i] * r0[i]; g2 += Jv[i] * r0[i]; }
        var sse0 = r0.reduce(function(s, x){ return s + x * x; }, 0), accepted = false, tries = 0;
        while (!accepted && tries < 12) { var A11 = a * (1 + lam), A22 = c * (1 + lam), det = A11 * A22 - b * b, du = -(A22 * g1 - b * g2) / det, dv = -(-b * g1 + A11 * g2) / det, nu = clamp(u + du, LIM.u0 - 1, LIM.u1 + 1), nv = clamp(v + dv, LIM.v0 - 1, LIM.v1 + 1), sse1 = pkcProfile(S, Math.exp(nu), Math.exp(nv)).sse; if (sse1 < sse0) { u = nu; v = nv; lam = Math.max(lam / 3, 1e-6); accepted = true; } else { lam *= 4; tries++; } }
        P.push({ u: u, v: v, f: Math.log(pkcProfile(S, Math.exp(u), Math.exp(v)).sse) }); if (!accepted || Math.abs(sse0 - Math.exp(P[P.length - 1].f)) < 1e-9 * sse0) break;
      }
      return P;
    }
    function covariance(cl, v){
      /* three-parameter Jacobian in log parameters at the estimate: ka free */
      var p = pkcProfile(S, cl, v), th = [Math.log(cl), Math.log(v), Math.log(p.ka)], n = S.t.length, J = [], i, k, h = 1e-5, f0 = [], j;
      function res(t){ var o = []; for (var q = 0; q < n; q++) o.push(S.conc[q] - pkcPred(S.doseMg, Math.exp(t[0]), Math.exp(t[1]), Math.exp(t[2]), S.t[q])); return o; }
      f0 = res(th); for (k = 0; k < 3; k++) { var t2 = th.slice(); t2[k] += h; var r2 = res(t2); J[k] = r2.map(function(x, q){ return (x - f0[q]) / h; }); }
      var M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]; for (i = 0; i < 3; i++) for (j = 0; j < 3; j++) { var s = 0; for (k = 0; k < n; k++) s += J[i][k] * J[j][k]; M[i][j] = s; }
      var s2 = f0.reduce(function(a, x){ return a + x * x; }, 0) / (n - 3), inv = [0, 1, 2].map(function(c){ var e = [0, 0, 0]; e[c] = 1; return pkcSolve(M, e); }), cov = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      for (i = 0; i < 3; i++) for (j = 0; j < 3; j++) cov[i][j] = s2 * inv[j][i];
      return { cov: cov, sd: [Math.sqrt(cov[0][0]), Math.sqrt(cov[1][1]), Math.sqrt(cov[2][2])], corr: [[1, cov[0][1] / Math.sqrt(cov[0][0] * cov[1][1]), cov[0][2] / Math.sqrt(cov[0][0] * cov[2][2])], [0, 1, cov[1][2] / Math.sqrt(cov[1][1] * cov[2][2])]], ka: p.ka, s2: s2 };
    }
    function drawMap(){
      var g = cv(qs('canvas', mp)), x = g.x, L = 44, R = g.w - 8, T = 6, B = g.h - 38, i, j; x.clearRect(0, 0, g.w, g.h); if (!grid) return;
      var cw = (R - L) / NG, ch = (B - T) / NG, nb = 14;
      for (i = 0; i < NG; i++) for (j = 0; j < NG; j++) { var q = Math.log10(grid[i][j] / grid.min), band = Math.min(nb - 1, Math.floor(q * 4)); var sh = Math.round(238 - band * 13); x.fillStyle = 'rgb(' + sh + ',' + (sh + 4) + ',' + (sh + 8) + ')'; x.fillRect(L + j * cw, B - (i + 1) * ch, cw + 0.6, ch + 0.6); }
      var px = function(u){ return L + (u - LIM.u0) / (LIM.u1 - LIM.u0) * (R - L); }, py = function(v){ return B - (v - LIM.v0) / (LIM.v1 - LIM.v0) * (B - T); };
      x.strokeStyle = C.char; x.lineWidth = 1; x.strokeRect(L, T, R - L, B - T); x.fillStyle = C.mid; x.font = '11px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'center';
      [0.5, 1, 2, 5, 10].forEach(function(c){ x.fillText(c, px(Math.log(c)), B + 14); }); x.fillText('CL/F (L/h, log scale)', (L + R) / 2, B + 30);
      x.textAlign = 'right'; [10, 20, 50, 100].forEach(function(c){ x.fillText(c, L - 4, py(Math.log(c)) + 4); }); x.save(); x.translate(10, (T + B) / 2); x.rotate(-Math.PI / 2); x.textAlign = 'center'; x.fillText('V/F (L, log scale)', 0, 0); x.restore();
      if (path.length) { x.strokeStyle = C.red; x.lineWidth = 2; x.beginPath(); path.forEach(function(p, k){ k ? x.lineTo(px(p.u), py(p.v)) : x.moveTo(px(p.u), py(p.v)); }); x.stroke(); x.fillStyle = C.red; path.forEach(function(p){ x.beginPath(); x.arc(px(p.u), py(p.v), 2.6, 0, 6.3); x.fill(); }); }
      x.fillStyle = '#fff'; x.strokeStyle = C.char; x.lineWidth = 1.6; x.beginPath(); x.arc(px(start.u), py(start.v), 6, 0, 6.3); x.fill(); x.stroke();
      if (cur) { x.fillStyle = C.char; x.beginPath(); x.moveTo(px(cur.u), py(cur.v) - 7); x.lineTo(px(cur.u) + 6, py(cur.v) + 5); x.lineTo(px(cur.u) - 6, py(cur.v) + 5); x.closePath(); x.fill(); }
      g.map = { L: L, R: R, T: T, B: B }; qs('canvas', mp)._map = g.map;
    }
    function drawFit(){
      var u = cur ? cur.u : start.u, v = cur ? cur.v : start.v, p = pkcProfile(S, Math.exp(u), Math.exp(v)), pts = [], i, tmax = 26;
      for (i = 0; i <= 100; i++) { var t = tmax * i / 100; pts.push([t, pkcPred(S.doseMg, Math.exp(u), Math.exp(v), p.ka, t)]); }
      var data = S.t.map(function(t, k){ return [t, S.conc[k]]; });
      var gx = qs('canvas', dp), g = lineChart(gx, [{ pts: pts, color: C.red, w: 2.8, nodot: true, label: 'model' }, { pts: data, color: C.char, w: 0.01, nodot: true, label: 'observed' }], { xl: 'time (h)', yl: 'concentration (mg/L)', x0: 0, x1: tmax, y0: 0 });
      data.forEach(function(p){ g.x.fillStyle = C.char; g.x.beginPath(); g.x.arc(g.sx(p[0]), g.sy(p[1]), 3.6, 0, 6.3); g.x.fill(); });
    }
    function readout(){
      var nca = PKE.nca(S.t, S.conc, S.doseMg), u = cur ? cur.u : start.u, v = cur ? cur.v : start.v, p = pkcProfile(S, Math.exp(u), Math.exp(v)), html = '<div class="ol">Current iterate</div><div>CL/F ' + pkcFmt(Math.exp(u)) + ' L/h, V/F ' + pkcFmt(Math.exp(v)) + ' L, ka ' + pkcFmt(p.ka) + ' per h, sum of squared residuals ' + p.sse.toFixed(2) + ' (mg/L)<sup>2</sup>. Iterations: ' + Math.max(0, path.length - 1) + '.</div>';
      if (fin) { var cc = covariance(Math.exp(fin.u), Math.exp(fin.v)); html += '<div class="ol" style="margin-top:8px">At the end of the run</div><div>Standard errors (asymptotic, log scale): CL/F ' + (cc.sd[0] * 100).toFixed(1) + '%, V/F ' + (cc.sd[1] * 100).toFixed(1) + '%, ka ' + (cc.sd[2] * 100).toFixed(1) + '%. Correlation of the estimates: CL/F with V/F ' + cc.corr[0][1].toFixed(2) + ', CL/F with ka ' + cc.corr[0][2].toFixed(2) + ', V/F with ka ' + cc.corr[1][2].toFixed(2) + '. Standard errors use the Jacobian at the estimate and one subject of 11 observations, so they are approximate.</div>'; }
      html += '<div class="ol" style="margin-top:8px">Non-compartmental analysis of the same subject</div><div>CL/F ' + pkcFmt(nca.clF) + ' L/h, Vz/F ' + pkcFmt(nca.vzF) + ' L, half-life ' + pkcFmt(nca.thalf) + ' h. The two methods use different assumptions and need not agree.</div>' +
        '<div class="ol" style="margin-top:8px">Link to lesson 09</div><div>Gradient descent here is the optimiser of lesson 09 applied to the logarithm of this objective. A learning rate that is too large overshoots and too small makes progress along the valley slowly. Levenberg-Marquardt rescales each direction with the Jacobian, as Adam rescales with gradient size.</div>';
      qs('.ev', rp).innerHTML = html;
    }
    function all(){ drawMap(); drawFit(); readout(); qs('.mm', mp).textContent = 'Shading: bands of ' + (1 / 4).toFixed(2) + ' in log10 of the objective above its minimum. Circle: start. Line: path. Triangle: current iterate. Click the map to set the start.'; }
    async function setSubject(i){ S = TH.subjects[i]; path = []; cur = null; fin = null; api.busy('Computing the objective'); await pkcTick(); makeGrid(); api.idle(); all(); }
    async function runOpt(){
      var tok = ++api.tok, P = opt === 0 ? runGD() : runLM(); path = []; fin = null; cur = null;
      var stepN = Math.max(1, Math.ceil(P.length / 60)), k;
      for (k = 0; k < P.length; k += stepN) { path = P.slice(0, k + 1); cur = P[k]; all(); await new Promise(function(r){ setTimeout(r, ANIM ? 30 : 0); }); if (api.tok !== tok) return; }
      path = P; cur = P[P.length - 1]; fin = cur; all();
    }
    qsa('[data-opt] button', ctl).forEach(function(b){ b.addEventListener('click', function(){ opt = +b.dataset.v; qsa('[data-opt] button', ctl).forEach(function(x){ x.classList.toggle('on', x === b); }); path = []; fin = null; cur = null; all(); }); });
    qs('[data-run]', ctl).addEventListener('click', runOpt);
    qs('[data-reset]', ctl).addEventListener('click', function(){ api.tok++; path = []; fin = null; cur = null; all(); });
    $('pkc4lr').addEventListener('input', function(){ qs('.v1', ctl).textContent = Math.pow(10, +this.value).toFixed(3); });
    qs('canvas', mp).addEventListener('click', function(e){ var c = qs('canvas', mp), m = c._map; if (!m) return; var r = c.getBoundingClientRect(), px = (e.clientX - r.left) / r.width * 410, py = (e.clientY - r.top) / r.height * 414; if (px < m.L || px > m.R || py < m.T || py > m.B) return; start = { u: LIM.u0 + (px - m.L) / (m.R - m.L) * (LIM.u1 - LIM.u0), v: LIM.v0 + (m.B - py) / (m.B - m.T) * (LIM.v1 - LIM.v0) }; path = []; cur = null; fin = null; all(); });
    async function render(){
      if (!TH) { TH = await pkcTheoph(); var sel = $('pkc4s'); sel.innerHTML = TH.subjects.map(function(s, i){ return '<option value="' + i + '">' + s.id + ' (' + s.wt + ' kg, ' + s.doseMg.toFixed(0) + ' mg)</option>'; }).join(''); sel.addEventListener('change', function(){ setSubject(+sel.value); }); qs('.v1', ctl).textContent = Math.pow(10, +$('pkc4lr').value).toFixed(3); await setSubject(0); } else all();
    }
    async function play(tok){
      await render(); path = []; cur = null; fin = null; all(); api.cap('Each point on the map is a pair (CL/F, V/F) with ka chosen to minimise the sum of squared residuals.'); await api.sleep(3400);
      opt = 0; qsa('[data-opt] button', ctl).forEach(function(x){ x.classList.toggle('on', +x.dataset.v === 0); }); api.cap('Gradient descent follows the slope with a fixed learning rate.'); await runOpt(); await api.sleep(1800);
      opt = 1; qsa('[data-opt] button', ctl).forEach(function(x){ x.classList.toggle('on', +x.dataset.v === 1); }); api.cap('Levenberg-Marquardt reaches the minimum in a few iterations from the same start.'); await runOpt(); await api.sleep(2600); api.cap(null);
    }
    return { render: render, play: play };
  } });
GUIDE.steps['pk/Estimation'] = { notice: 'Gradient descent progresses slowly along the valley. Levenberg-Marquardt reaches the minimum in a few iterations.', q: 'Why do the estimates of CL/F and V/F correlate?', a: 'The data constrain the terminal slope, which depends on their ratio, better than either parameter alone, so errors in the two move together.' };
