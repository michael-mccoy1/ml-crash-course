"use strict";
/* Lesson 11, additional steps centred on PK data. Written for PK-literate scientists who are new to machine learning.
   Steps: Sparse sampling (Intermediate), Population PK and ML (Advanced).
   Data: simulated populations made in the page (one-compartment oral model through PKE.oneComp). No measured data are used.
   Helpers from mlpk_b.js: mlbNormal, mlbBoost, mlbOLS, mlbFold, mlbSeg. Own helpers use the prefix mle. */

/* ---------- shared ---------- */
function mleLogTicks(lo, hi){
  var cand = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000], o = cand.filter(function(v){ return v >= lo * 0.999 && v <= hi * 1.001; });
  return o.length >= 2 ? o : [lo, hi];
}
function mleTickLab(v){ return v >= 1 ? String(+v.toFixed(0)) : String(+v.toFixed(2)); }
function mleFmt(v){ return v >= 100 ? v.toFixed(0) : (v >= 10 ? v.toFixed(1) : v.toFixed(2)); }

/* ---------- 1. Sparse sampling ---------- */
var MEE_DES = { 2: [[1, 4], [2, 12], [6, 24]], 3: [[0.5, 1.5, 4], [1, 4, 12], [4, 8, 24]], 4: [[0.5, 1, 2, 4], [0.5, 2, 6, 16], [4, 8, 12, 24]] };
var MEE_STY = '<style>.mt td{padding:3px 6px;font-size:12px;color:#34373F;line-height:1.25}</style>';
var MEE_DN = ['Early', 'Spread', 'Late'];
var ME1 = { cache: {}, tcache: {} };
/* one-compartment oral population. Concentrations and AUC are per 100 mg dose. */
function mleSim(n, seed){
  var z = mlbNormal(seed), r = rng(seed * 13 + 5), out = [], i;
  for (i = 0; i < n; i++) {
    var wt = clamp(75 * Math.exp(0.2 * z()), 40, 150), dose = [50, 100, 200][Math.floor(r() * 3)];
    var CL = 5 * Math.pow(wt / 70, 0.75) * Math.exp(0.3 * z()), V = 40 * (wt / 70) * Math.exp(0.25 * z()), ka = 1.2 * Math.exp(0.4 * z());
    var m = PKE.oneComp({ route: 'oral', dose: dose, F: 1, V: V, CL: CL, ka: ka });
    out.push({ wt: wt, dose: dose, CL: CL, V: V, ka: ka, conc: m.conc, aucN: 100 / CL, cmaxN: m.summary.cmax * 100 / dose, tmax: m.summary.tmax });
  }
  return out;
}
/* sampled concentrations per 100 mg with proportional lognormal assay error */
function mleMeasure(subj, times, cvA, seed){
  var z = mlbNormal(seed);
  return subj.map(function(s){ return times.map(function(t){ return s.conc(t) * 100 / s.dose * Math.exp(cvA * z()); }); });
}
/* NCA on the sampled points: linear-up, log-down trapezoids from (0, 0); the last two samples give the terminal slope when they decline */
function mleNCA(times, c){
  var n = times.length, auc = 0, cmax = 0, i, t0 = 0, c0 = 0;
  for (i = 0; i < n; i++) {
    var dt = times[i] - t0;
    if (c[i] < c0 && c[i] > 0 && c0 > 0) auc += (c0 - c[i]) * dt / Math.log(c0 / c[i]); else auc += (c0 + c[i]) / 2 * dt;
    t0 = times[i]; c0 = c[i]; if (c[i] > cmax) cmax = c[i];
  }
  var ext = false;
  if (n >= 2 && c[n - 1] < c[n - 2] && c[n - 1] > 0) { var lz = Math.log(c[n - 2] / c[n - 1]) / (times[n - 1] - times[n - 2]); auc += c[n - 1] / lz; ext = true; }
  return { auc: auc, cmax: cmax, ext: ext };
}
function mleRun1(n, di, ntr, cvA, seedI){
  var key = [n, di, ntr, cvA, seedI].join('|'); if (ME1.cache[key]) return ME1.cache[key];
  var times = MEE_DES[n][di], tk = 'test' + seedI; if (!ME1.tcache[tk]) ME1.tcache[tk] = mleSim(200, 9000 + seedI);
  var test = ME1.tcache[tk], train = mleSim(ntr, 100 + seedI);
  function feats(subj, C){ return subj.map(function(s, i){ return C[i].map(function(v){ return Math.log10(v); }).concat([s.wt]); }); }
  var Ctr = mleMeasure(train, times, cvA, 500 + seedI), Cte = mleMeasure(test, times, cvA, 700 + seedI);
  var Xtr = feats(train, Ctr), Xte = feats(test, Cte);
  var mA = mlbBoost(Xtr, train.map(function(s){ return Math.log10(s.aucN); }), { rounds: 150, lr: 0.1, depth: 3 });
  var mC = mlbBoost(Xtr, train.map(function(s){ return Math.log10(s.cmaxN); }), { rounds: 150, lr: 0.1, depth: 3 });
  var tA = test.map(function(s){ return s.aucN; }), tC = test.map(function(s){ return s.cmaxN; }), nca = Cte.map(function(c){ return mleNCA(times, c); });
  var pA = Xte.map(function(r){ return Math.pow(10, mA.predict(r)); }), pC = Xte.map(function(r){ return Math.pow(10, mC.predict(r)); });
  var lg = function(a){ return a.map(function(v){ return Math.log10(Math.max(v, 1e-6)); }); };
  var R = { n: n, di: di, times: times, test: test, Cte: Cte, tA: tA, tC: tC, nca: nca, pA: pA, pC: pC,
    mNA: mlbFold(lg(tA), lg(nca.map(function(q){ return q.auc; }))), mNC: mlbFold(lg(tC), lg(nca.map(function(q){ return q.cmax; }))), mMA: mlbFold(lg(tA), lg(pA)), mMC: mlbFold(lg(tC), lg(pC)),
    noExt: nca.filter(function(q){ return !q.ext; }).length / nca.length, ntr: ntr, cvA: cvA };
  return (ME1.cache[key] = R);
}
function mleScatter(g, tr, nc, ml, ttl, lab){
  var x = g.x, L = 44, T = 22, R = g.w - 12, B = g.h - 34, lo = Math.min.apply(null, tr) / 1.4, hi = Math.max.apply(null, tr) * 1.4, ll = Math.log(lo), lh = Math.log(hi), tk = mleLogTicks(lo, hi), i;
  function px(v){ return L + (Math.log(clamp(v, lo, hi)) - ll) / (lh - ll) * (R - L); }
  function py(v){ return B - (Math.log(clamp(v, lo, hi)) - ll) / (lh - ll) * (B - T); }
  x.clearRect(0, 0, g.w, g.h); x.lineWidth = 1; x.font = '10.5px Aptos, "Segoe UI", sans-serif';
  tk.forEach(function(v){ x.strokeStyle = C.light; x.beginPath(); x.moveTo(px(v), T); x.lineTo(px(v), B); x.stroke(); x.beginPath(); x.moveTo(L, py(v)); x.lineTo(R, py(v)); x.stroke(); x.fillStyle = C.mid; x.textAlign = 'center'; x.fillText(mleTickLab(v), px(v), B + 13); x.textAlign = 'right'; x.fillText(mleTickLab(v), L - 4, py(v) + 4); });
  x.setLineDash([4, 3]); x.strokeStyle = C.bg; [2, 0.5].forEach(function(f){ x.beginPath(); x.moveTo(px(lo), py(lo * f)); x.lineTo(px(hi), py(hi * f)); x.stroke(); }); x.setLineDash([]);
  x.strokeStyle = C.char; x.beginPath(); x.moveTo(px(lo), py(lo)); x.lineTo(px(hi), py(hi)); x.stroke();
  x.globalAlpha = 0.75; x.fillStyle = C.bg; for (i = 0; i < tr.length; i++) { x.beginPath(); x.arc(px(tr[i]), py(nc[i]), 2.6, 0, 6.2832); x.fill(); }
  x.fillStyle = C.red; for (i = 0; i < tr.length; i++) { x.beginPath(); x.arc(px(tr[i]), py(ml[i]), 2.2, 0, 6.2832); x.fill(); } x.globalAlpha = 1;
  x.fillStyle = C.char; x.font = '600 11.5px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'center'; x.fillText('True ' + lab, (L + R) / 2, g.h - 4);
  x.textAlign = 'left'; x.fillText(ttl, L, 12);
  x.save(); x.translate(11, (T + B) / 2); x.rotate(-Math.PI / 2); x.textAlign = 'center'; x.fillText('Estimated', 0, 0); x.restore();
  x.font = '10.5px Aptos, "Segoe UI", sans-serif'; x.fillStyle = C.bg; x.beginPath(); x.arc(R - 112, 13, 3, 0, 6.2832); x.fill(); x.fillStyle = C.char; x.textAlign = 'left'; x.fillText('NCA', R - 106, 16);
  x.fillStyle = C.red; x.beginPath(); x.arc(R - 66, 13, 3, 0, 6.2832); x.fill(); x.fillStyle = C.char; x.fillText('Boosted trees', R - 60, 16);
}
LESSONS.mlpk.add({ level: 'intermediate', short: 'Sparse sampling', title: 'Sparse sampling',
  change: 'Select the number of samples per subject and the sampling times. Set the number of training subjects and the assay error. Select Next subject to see one subject profile.',
  watch: 'Non-compartmental analysis (NCA) estimates AUC and Cmax from the sampled concentrations alone. With two to four samples per subject, which is typical of sparse sampling in a clinical study, NCA must extrapolate beyond the last sample, and the largest sampled concentration is usually below the true Cmax. A machine learning model can be trained on simulated subjects to map the sampled concentrations and the body weight to AUC and Cmax. The simulated population follows a one-compartment oral model: CL = 5 L/h times (WT/70)^0.75 times exp(eta), eta normal with SD 0.30; V = 40 L times (WT/70) times exp(eta), SD 0.25; ka = 1.2 per hour times exp(eta), SD 0.40; F = 1. Doses are 50, 100 or 200 mg. Weight is lognormal with median 75 kg and SD 0.2 in log units. Each sampled concentration carries a proportional lognormal assay error. Concentrations and AUC are expressed per 100 mg dose. NCA here uses linear-up, log-down trapezoids from time 0 with concentration 0, and extrapolates beyond the last sample with the slope of the last two samples when they decline. The boosted-trees model receives the log10 sampled concentrations and the weight, and predicts log10 AUC from zero to infinity and log10 Cmax. Both methods are scored on 200 test subjects that were not used in training. GMFE is the geometric mean fold error. Within 2-fold is the fraction of subjects with an estimate between half and twice the true value. The accuracy of the trained model is limited by its training simulator: it holds for subjects from the same parameter distribution measured at the same times, and not for a different absorption model, a two-compartment profile or other sampling times.',
  build: function(api){
    var nS = 3, di = 1, ntr = 300, cvA = 0.1, seedI = 1, si = 0;
    var sp = api.el('pn', 'left:28px;top:20px;width:444px;height:260px', '<h4 class="st">AUC from the sampled concentrations</h4><canvas width="416" height="198"></canvas>');
    var cp = api.el('pn', 'left:488px;top:20px;width:444px;height:260px', '<h4>Cmax from the sampled concentrations</h4><canvas width="416" height="198"></canvas>');
    var ep = api.el('pn', 'left:28px;top:292px;width:444px;height:228px', '<h4 class="et">One test subject</h4><canvas width="416" height="162"></canvas>');
    var tp = api.el('pn', 'left:488px;top:292px;width:444px;height:228px;overflow:auto', '<div class="tb" style="font-size:12.5px;line-height:1.45;color:' + C.char + '"></div>');
    var ctl = api.addControl('Samples ' + mlbSeg('data-n', ['2', '3', '4'], 1) + ' Times ' + mlbSeg('data-d', MEE_DN, 1) + ' Training subjects <input class="t" type="range" min="50" max="800" step="50" value="300" style="width:100px"> <b class="tv">300</b> Assay CV <input class="c" type="range" min="0.05" max="0.4" step="0.05" value="0.1" style="width:80px"> <b class="cv">10%</b> <button type="button" class="btn sm ghost" data-seed>New subjects</button> <button type="button" class="btn sm" data-next>Next subject</button>');
    var tI = qs('.t', ctl), cI = qs('.c', ctl);
    function drawCurve(R){
      var g = cv(qs('canvas', ep)), x = g.x, L = 40, T = 16, Rr = g.w - 10, B = g.h - 28, s = R.test[si], c = R.Cte[si], nc = R.nca[si], tmax = 24, ymax = 0, k;
      for (k = 0; k <= 96; k++) ymax = Math.max(ymax, s.conc(k / 4) * 100 / s.dose); ymax *= 1.15;
      function px(t){ return L + t / tmax * (Rr - L); } function py(v){ return B - v / ymax * (B - T); }
      x.clearRect(0, 0, g.w, g.h); x.lineWidth = 1; x.font = '10.5px Aptos, "Segoe UI", sans-serif';
      [0, 6, 12, 18, 24].forEach(function(t){ x.strokeStyle = C.light; x.beginPath(); x.moveTo(px(t), T); x.lineTo(px(t), B); x.stroke(); x.fillStyle = C.mid; x.textAlign = 'center'; x.fillText(String(t), px(t), B + 13); });
      for (k = 0; k <= 4; k++) { var v = ymax * k / 4; x.strokeStyle = C.light; x.beginPath(); x.moveTo(L, py(v)); x.lineTo(Rr, py(v)); x.stroke(); x.fillStyle = C.mid; x.textAlign = 'right'; x.fillText(mleFmt(v), L - 4, py(v) + 4); }
      x.strokeStyle = C.char; x.lineWidth = 1.8; x.beginPath(); for (k = 0; k <= 96; k++) { var t = k / 4, y = s.conc(t) * 100 / s.dose; if (k) x.lineTo(px(t), py(y)); else x.moveTo(px(t), py(y)); } x.stroke(); x.lineWidth = 1;
      x.setLineDash([3, 3]); x.strokeStyle = C.bg; R.times.forEach(function(t){ x.beginPath(); x.moveTo(px(t), T); x.lineTo(px(t), B); x.stroke(); }); x.setLineDash([]);
      x.fillStyle = C.red; R.times.forEach(function(t, i){ x.beginPath(); x.arc(px(t), py(c[i]), 4, 0, 6.2832); x.fill(); });
      x.fillStyle = C.char; x.textAlign = 'center'; x.font = '600 11.5px Aptos, "Segoe UI", sans-serif'; x.fillText('Time (h), concentration (mg/L per 100 mg)', (L + Rr) / 2, g.h - 3);
    }
    function draw(){
      var R = mleRun1(nS, di, ntr, cvA, seedI);
      mleScatter(cv(qs('canvas', sp)), R.tA, R.nca.map(function(q){ return q.auc; }), R.pA, 'AUC, mg h/L per 100 mg', 'AUC');
      mleScatter(cv(qs('canvas', cp)), R.tC, R.nca.map(function(q){ return q.cmax; }), R.pC, 'Cmax, mg/L per 100 mg', 'Cmax');
      qs('.st', sp).textContent = 'AUC from ' + nS + ' samples at ' + R.times.join(', ') + ' h';
      qs('.et', ep).textContent = 'Test subject ' + (si + 1) + ' of 200, WT ' + R.test[si].wt.toFixed(0) + ' kg, dose ' + R.test[si].dose + ' mg';
      drawCurve(R);
      var s = R.test[si], nc = R.nca[si];
      var f = function(m){ return m.gmfe.toFixed(2); }, w = function(m){ return Math.round(100 * m.within2) + '%'; };
      qs('.tb', tp).innerHTML = MEE_STY + '<table class="mt" style="width:100%;border-collapse:collapse"><tr style="font-weight:700;text-align:right"><td style="text-align:left">Test subjects</td><td>AUC GMFE</td><td>AUC within 2-fold</td><td>Cmax GMFE</td><td>Cmax within 2-fold</td></tr>' +
        '<tr style="text-align:right;border-top:1px solid ' + C.silver + '"><td style="text-align:left">NCA</td><td>' + f(R.mNA) + '</td><td>' + w(R.mNA) + '</td><td>' + f(R.mNC) + '</td><td>' + w(R.mNC) + '</td></tr>' +
        '<tr style="text-align:right;border-top:1px solid ' + C.silver + '"><td style="text-align:left">Boosted trees</td><td>' + f(R.mMA) + '</td><td>' + w(R.mMA) + '</td><td>' + f(R.mMC) + '</td><td>' + w(R.mMC) + '</td></tr></table>' +
        '<p style="margin:8px 0 4px">NCA could not extrapolate (the last two samples did not decline) in ' + Math.round(100 * R.noExt) + '% of test subjects. The model was trained on ' + R.ntr + ' simulated subjects with assay CV ' + Math.round(100 * R.cvA) + '%.</p>' +
        '<p style="margin:4px 0"><b>Subject ' + (si + 1) + '</b>, per 100 mg: true AUC ' + mleFmt(R.tA[si]) + ', NCA ' + mleFmt(nc.auc) + ', boosted trees ' + mleFmt(R.pA[si]) + '. True Cmax ' + mleFmt(R.tC[si]) + ', NCA ' + mleFmt(nc.cmax) + ', boosted trees ' + mleFmt(R.pC[si]) + '. True tmax ' + s.tmax.toFixed(1) + ' h.</p>' +
        '<p style="margin:4px 0;color:' + C.mid + '">Dashed lines in the scatter plots mark 2-fold error. Estimates outside the plotted range are drawn on its edge. All subjects are simulated.</p>';
    }
    function upd(){ ntr = +tI.value; cvA = +cI.value; qs('.tv', ctl).textContent = ntr; qs('.cv', ctl).textContent = Math.round(100 * cvA) + '%'; si = 0; draw(); }
    [tI, cI].forEach(function(e){ e.addEventListener('change', upd); e.addEventListener('input', function(){ qs('.tv', ctl).textContent = tI.value; qs('.cv', ctl).textContent = Math.round(100 * cI.value) + '%'; }); });
    qsa('[data-n] button', ctl).forEach(function(b){ b.addEventListener('click', function(){ nS = +b.textContent; qsa('[data-n] button', ctl).forEach(function(z){ z.classList.toggle('on', z === b); }); si = 0; wrap(); }); });
    qsa('[data-d] button', ctl).forEach(function(b){ b.addEventListener('click', function(){ di = +b.dataset.v; qsa('[data-d] button', ctl).forEach(function(z){ z.classList.toggle('on', z === b); }); si = 0; wrap(); }); });
    qs('[data-seed]', ctl).addEventListener('click', function(){ seedI++; si = 0; wrap(); });
    qs('[data-next]', ctl).addEventListener('click', function(){ si = (si + 1) % 200; draw(); });
    async function wrap(){ api.busy('Simulating subjects and training the model'); await tick(); try { draw(); } finally { api.idle(); } }
    function setUI(){
      qs('.tv', ctl).textContent = ntr; qs('.cv', ctl).textContent = Math.round(100 * cvA) + '%'; tI.value = ntr; cI.value = cvA;
      qsa('[data-n] button', ctl).forEach(function(z){ z.classList.toggle('on', +z.textContent === nS); }); qsa('[data-d] button', ctl).forEach(function(z){ z.classList.toggle('on', +z.dataset.v === di); });
    }
    function sum(){ var R = mleRun1(nS, di, ntr, cvA, seedI); return 'AUC GMFE ' + R.mNA.gmfe.toFixed(2) + ' for NCA and ' + R.mMA.gmfe.toFixed(2) + ' for boosted trees; Cmax GMFE ' + R.mNC.gmfe.toFixed(2) + ' and ' + R.mMC.gmfe.toFixed(2) + '.'; }
    async function render(){ await wrap(); }
    async function play(tok){
      nS = 3; di = 1; ntr = 300; cvA = 0.1; seedI = 1; si = 0; setUI(); await wrap();
      api.cap('Three samples per subject at ' + MEE_DES[3][1].join(', ') + ' h. ' + sum()); await api.sleep(4200);
      nS = 2; setUI(); await wrap(); api.cap('Two samples per subject. ' + sum()); await api.sleep(4200);
      nS = 2; di = 2; setUI(); await wrap(); api.cap('Two late samples, at ' + MEE_DES[2][2].join(' and ') + ' h, miss the absorption phase. ' + sum()); await api.sleep(4200);
      nS = 4; di = 1; ntr = 100; setUI(); await wrap(); api.cap('Four samples and 100 training subjects. ' + sum()); await api.sleep(4200); api.cap(null);
    }
    return { render: render, play: play };
  } });
GUIDE.steps['mlpk/Sparse sampling'] = { notice: 'NCA estimates at the edges of the scatter plots come from samples that do not cover the peak or the terminal decline.', q: 'Why does the trained model give a Cmax estimate when no sample falls at tmax?', a: 'The largest sampled concentration is a lower bound on Cmax. The model learned the relation between the sampled concentrations, the weight and Cmax from simulated subjects, so it adds the typical difference for that sampling schedule. The estimate is only as valid as the simulator that produced the training subjects.' };

/* ---------- 2. Population PK and ML ---------- */
var ME2 = { cache: {} };
/* covariates and individual log clearance. form 0: power law in CRCL. form 1: steep threshold function of CRCL, equal to 1 at CRCL 100. */
function mleG(c, form){
  if (!form) return Math.pow(c / 100, 0.3);
  var h = function(v){ var q = Math.pow(v / 80, 16); return 0.55 + 0.45 * q / (q + 1); }; return h(c) / h(100);
}
function mlePop(n, seed, form, sigma){
  var z = mlbNormal(seed), r = rng(seed * 7 + 3), X = [], lcl = [], lct = [], i;
  for (i = 0; i < n; i++) {
    var wt = clamp(75 * Math.exp(0.2 * z()), 40, 150), age = clamp(50 + 15 * z(), 18, 85), sex = r() < 0.5 ? 0 : 1, scr = 0.9 * Math.exp(0.2 * z()), alb = clamp(4.2 + 0.4 * z(), 2.5, 5.5);
    var crcl = (140 - age) * wt / (72 * scr) * (sex ? 0.85 : 1), typ = Math.log(5 * Math.pow(wt / 70, 0.75) * mleG(crcl, form));
    X.push([wt, age, sex, crcl, alb, z(), z()]); lct.push(typ); lcl.push(typ + 0.25 * z() + sigma * z());
  }
  return { X: X, y: lcl, typ: lct };
}
var MEE_NAMES = ['WT', 'AGE', 'SEX', 'CRCL', 'ALB', 'N1', 'N2'];
function mleParam(X, beta){ return X.map(function(r){ return beta[0] + beta[1] * Math.log(r[0] / 70) + beta[2] * Math.log(r[3] / 100); }); }
function mleFit2(ntr, form, sigma, seedI, pool){
  var tr = { X: pool.X.slice(0, ntr), y: pool.y.slice(0, ntr) };
  var ols = mlbOLS([tr.X.map(function(r){ return Math.log(r[0] / 70); }), tr.X.map(function(r){ return Math.log(r[3] / 100); })], tr.y);
  var gb = mlbBoost(tr.X, tr.y, { rounds: 150, lr: 0.08, depth: 3, minLeaf: 5 });
  return { ols: ols, beta: ols.beta, gb: gb };
}
function mleRun2(ntr, form, sigma, seedI){
  var key = [ntr, form, sigma, seedI].join('|'); if (ME2.cache[key]) return ME2.cache[key];
  var pk = [form, sigma, seedI].join('|'); if (!ME2.cache[pk]) ME2.cache[pk] = { pool: mlePop(800, 50 + seedI, form, sigma), test: mlePop(400, 6000 + seedI, form, sigma) };
  var P = ME2.cache[pk], M = mleFit2(ntr, form, sigma, seedI, P.pool), te = P.test;
  var pp = mleParam(te.X, M.beta), pg = te.X.map(function(r){ return M.gb.predict(r); });
  function rm(a, b){ return Math.sqrt(mean(a.map(function(v, i){ return (v - b[i]) * (v - b[i]); }))); }
  var res = { M: M, P: P, ntr: ntr, form: form, sigma: sigma, floor: Math.sqrt(0.25 * 0.25 + sigma * sigma),
    oP: rm(pp, te.y), oG: rm(pg, te.y), tP: rm(pp, te.typ), tG: rm(pg, te.typ), r2P: mlbR2(te.y, pp), r2G: mlbR2(te.y, pg), r2max: 1 - (0.25 * 0.25 + sigma * sigma) / (function(){ var m = mean(te.y); return mean(te.y.map(function(v){ return (v - m) * (v - m); })); })() };
  var gn = M.gb.gain(), tot = gn.reduce(function(a, b){ return a + b; }, 0) || 1; res.imp = gn.map(function(v){ return v / tot; });
  return (ME2.cache[key] = res);
}
var MEE_SIZES = [25, 50, 100, 200, 400, 800];
function mleCurve2(form, sigma, seedI){
  var key = 'lc|' + [form, sigma, seedI].join('|'); if (ME2.cache[key]) return ME2.cache[key];
  var o = MEE_SIZES.map(function(n){ var r = mleRun2(n, form, sigma, seedI); return { n: n, p: r.tP, g: r.tG }; }); return (ME2.cache[key] = o);
}
LESSONS.mlpk.add({ level: 'advanced', short: 'Population PK and ML', title: 'Population PK with ML covariate models',
  change: 'Select the covariate form and the number of training subjects. Set the estimation noise. Compare the two fitted curves with the true curve and read the learning curve.',
  watch: 'A population PK covariate model relates individual clearance to subject characteristics. The usual form is parametric: ln CL = theta0 + theta1 ln(WT/70) + theta2 ln(CRCL/100), with the covariates chosen in advance. A boosted-trees model receives all covariates and no functional form. The data are simulated, so the truth is known. Clearance is CL = 5 L/h times (WT/70)^0.75 times G(CRCL) times exp(eta), eta normal with SD 0.25. In the power-law form G = (CRCL/100)^0.3. In the threshold form G = h(CRCL)/h(100), with h(c) = 0.55 + 0.45 q/(1 + q) and q = (c/80)^16. Clearance is then about 45% lower below a CRCL of 80 mL/min than above it, with a steep transition, which a log-linear term cannot represent. The covariates are weight (WT), age (AGE), sex (SEX), creatinine clearance (CRCL, from the Cockcroft-Gault expression), albumin (ALB) and two noise variables (N1, N2). The simulated individual clearance estimate carries an additional error of the SD set by estimation noise. Between-subject variability and this noise are not explained by any covariate, so the expected error against the observed clearance cannot fall below sqrt(0.25^2 + noise^2) in ln units, for either model. On a finite test set the value fluctuates around that limit. The error against the true typical clearance, which is known only in a simulation, shows how close each model is to the covariate relation. Both models are scored on 400 test subjects. The parametric model returns exponents, which are interpretable. The boosted-trees model returns predictions and split-gain importance, which ranks covariates and does not give an exponent or a direction.',
  build: function(api){
    var ntr = 200, form = 1, sigma = 0.15, seedI = 1;
    var cp = api.el('pn', 'left:28px;top:20px;width:444px;height:260px', '<h4>Clearance against creatinine clearance</h4><canvas width="416" height="198"></canvas>');
    var lp = api.el('pn', 'left:488px;top:20px;width:444px;height:260px', '<h4>Error against true typical clearance</h4><canvas width="416" height="198"></canvas>');
    var ip = api.el('pn', 'left:28px;top:292px;width:444px;height:228px', '<h4>Importance in the boosted-trees model</h4><canvas width="416" height="162"></canvas>');
    var tp = api.el('pn', 'left:488px;top:292px;width:444px;height:228px;overflow:auto', '<div class="tb" style="font-size:12.5px;line-height:1.45;color:' + C.char + '"></div>');
    var ctl = api.addControl('Covariate form ' + mlbSeg('data-f', ['Power', 'Threshold'], 1) + ' Training subjects <input class="t" type="range" min="25" max="800" step="25" value="200" style="width:110px"> <b class="tv">200</b> Estimation noise <input class="s" type="range" min="0.05" max="0.4" step="0.05" value="0.15" style="width:80px"> <b class="sv">0.15</b> <button type="button" class="btn sm ghost" data-seed>New subjects</button>');
    var tI = qs('.t', ctl), sI = qs('.s', ctl);
    function drawCurve(R){
      var g = cv(qs('canvas', cp)), x = g.x, L = 40, T = 14, Rr = g.w - 10, B = g.h - 32, c, pts = R.P.pool, M = R.M, k, ymax = 8.5;
      function px(v){ return L + (v - 20) / 230 * (Rr - L); } function py(v){ return B - v / ymax * (B - T); }
      x.clearRect(0, 0, g.w, g.h); x.lineWidth = 1; x.font = '10.5px Aptos, "Segoe UI", sans-serif';
      [50, 100, 150, 200, 250].forEach(function(v){ x.strokeStyle = C.light; x.beginPath(); x.moveTo(px(v), T); x.lineTo(px(v), B); x.stroke(); x.fillStyle = C.mid; x.textAlign = 'center'; x.fillText(String(v), px(v), B + 13); });
      [0, 2, 4, 6, 8].forEach(function(v){ x.strokeStyle = C.light; x.beginPath(); x.moveTo(L, py(v)); x.lineTo(Rr, py(v)); x.stroke(); x.fillStyle = C.mid; x.textAlign = 'right'; x.fillText(String(v), L - 4, py(v) + 4); });
      x.fillStyle = C.bg; x.globalAlpha = 0.55; for (k = 0; k < Math.min(R.ntr, 300); k++) { var r = pts.X[k], cl = Math.exp(pts.y[k]) * Math.pow(70 / r[0], 0.75); if (r[3] < 20 || r[3] > 250) continue; x.beginPath(); x.arc(px(r[3]), py(Math.min(cl, ymax)), 2.2, 0, 6.2832); x.fill(); } x.globalAlpha = 1;
      function line(fn, col, w, dash){ x.strokeStyle = col; x.lineWidth = w; x.setLineDash(dash || []); x.beginPath(); for (c = 20; c <= 250; c += 2) { var y = fn(c); if (c > 20) x.lineTo(px(c), py(y)); else x.moveTo(px(c), py(y)); } x.stroke(); x.setLineDash([]); x.lineWidth = 1; }
      line(function(v){ return 5 * mleG(v, R.form); }, C.bg, 5);
      line(function(v){ return Math.exp(M.beta[0] + M.beta[2] * Math.log(v / 100)); }, C.char, 1.8, [5, 3]);
      line(function(v){ return Math.exp(M.gb.predict([70, 50, 0, v, 4.2, 0, 0])); }, C.red, 1.8);
      x.font = '10.5px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'left';
      [[C.bg, 'True typical CL'], [C.char, 'Log-linear fit'], [C.red, 'Boosted trees']].forEach(function(e, i){ x.fillStyle = e[0]; x.fillRect(L + 8, T + 6 + i * 13, 12, 3); x.fillStyle = C.char; x.fillText(e[1], L + 25, T + 10 + i * 13); });
      x.fillStyle = C.char; x.textAlign = 'center'; x.font = '600 11.5px Aptos, "Segoe UI", sans-serif'; x.fillText('CRCL (mL/min), CL (L/h) at WT 70 kg, age 50, ALB 4.2', (L + Rr) / 2, g.h - 4);
    }
    function drawLC(){
      var g = cv(qs('canvas', lp)), x = g.x, L = 44, T = 14, Rr = g.w - 12, B = g.h - 32, d = mleCurve2(form, sigma, seedI), ymax = Math.max.apply(null, d.map(function(e){ return Math.max(e.p, e.g); })) * 1.1, lx0 = Math.log(25), lx1 = Math.log(800), R0 = mleRun2(ntr, form, sigma, seedI);
      function px(n){ return L + (Math.log(n) - lx0) / (lx1 - lx0) * (Rr - L); } function py(v){ return B - v / ymax * (B - T); }
      x.clearRect(0, 0, g.w, g.h); x.lineWidth = 1; x.font = '10.5px Aptos, "Segoe UI", sans-serif';
      MEE_SIZES.forEach(function(n){ x.strokeStyle = C.light; x.beginPath(); x.moveTo(px(n), T); x.lineTo(px(n), B); x.stroke(); x.fillStyle = C.mid; x.textAlign = 'center'; x.fillText(String(n), px(n), B + 13); });
      var st = ymax > 0.5 ? 0.1 : 0.05; for (var v = 0; v <= ymax; v += st) { x.strokeStyle = C.light; x.beginPath(); x.moveTo(L, py(v)); x.lineTo(Rr, py(v)); x.stroke(); x.fillStyle = C.mid; x.textAlign = 'right'; x.fillText(v.toFixed(2), L - 4, py(v) + 4); }
      function line(key, col, dash){ x.strokeStyle = col; x.lineWidth = 2; x.setLineDash(dash || []); x.beginPath(); d.forEach(function(e, i){ if (i) x.lineTo(px(e.n), py(e[key])); else x.moveTo(px(e.n), py(e[key])); }); x.stroke(); x.setLineDash([]); x.fillStyle = col; d.forEach(function(e){ x.beginPath(); x.arc(px(e.n), py(e[key]), 3, 0, 6.2832); x.fill(); }); x.lineWidth = 1; }
      line('p', C.char, [5, 3]); line('g', C.red);
      x.strokeStyle = C.char; x.setLineDash([2, 3]); x.beginPath(); x.moveTo(px(R0.ntr), T); x.lineTo(px(R0.ntr), B); x.stroke(); x.setLineDash([]);
      [[C.char, 'Log-linear fit'], [C.red, 'Boosted trees']].forEach(function(e, i){ x.fillStyle = e[0]; x.fillRect(Rr - 98, T + 6 + i * 13, 12, 3); x.fillStyle = C.char; x.textAlign = 'left'; x.fillText(e[1], Rr - 82, T + 10 + i * 13); });
      x.fillStyle = C.char; x.textAlign = 'center'; x.font = '600 11.5px Aptos, "Segoe UI", sans-serif'; x.fillText('Training subjects', (L + Rr) / 2, g.h - 4); x.save(); x.translate(11, (T + B) / 2); x.rotate(-Math.PI / 2); x.fillText('RMSE, ln units', 0, 0); x.restore();
    }
    function drawImp(R){
      var g = cv(qs('canvas', ip)), x = g.x, n = MEE_NAMES.length, top = 8, bot = g.h - 8, rowH = (bot - top) / n, mx = Math.max.apply(null, R.imp) * 1.15, barL = 56, barW = g.w - barL - 56, tr = ['WT', 'CRCL'];
      x.clearRect(0, 0, g.w, g.h);
      MEE_NAMES.forEach(function(nm, j){
        var y0 = top + j * rowH, on = tr.indexOf(nm) >= 0, w = R.imp[j] / mx * barW;
        x.fillStyle = on ? C.dred : C.char; x.font = (on ? '700 ' : '') + '11px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'right'; x.fillText(nm, barL - 8, y0 + rowH * 0.66);
        x.fillStyle = on ? C.red : C.bg; x.fillRect(barL, y0 + 2, Math.max(0, w), rowH - 5);
        x.fillStyle = C.char; x.font = '10.5px Aptos, "Segoe UI", sans-serif'; x.textAlign = 'left'; x.fillText((100 * R.imp[j]).toFixed(0) + '%', barL + Math.max(0, w) + 5, y0 + rowH * 0.66);
      });
    }
    function draw(){
      var R = mleRun2(ntr, form, sigma, seedI); drawCurve(R); drawLC(); drawImp(R);
      var b = R.M.beta, tw = form ? 'no single true value' : '0.30';
      qs('.tb', tp).innerHTML = MEE_STY + '<table class="mt" style="width:100%;border-collapse:collapse"><tr style="font-weight:700;text-align:right"><td style="text-align:left">400 test subjects</td><td>RMSE vs observed</td><td>RMSE vs typical</td><td>R2 vs observed</td></tr>' +
        '<tr style="text-align:right;border-top:1px solid ' + C.silver + '"><td style="text-align:left">Log-linear fit</td><td>' + R.oP.toFixed(3) + '</td><td>' + R.tP.toFixed(3) + '</td><td>' + R.r2P.toFixed(2) + '</td></tr>' +
        '<tr style="text-align:right;border-top:1px solid ' + C.silver + '"><td style="text-align:left">Boosted trees</td><td>' + R.oG.toFixed(3) + '</td><td>' + R.tG.toFixed(3) + '</td><td>' + R.r2G.toFixed(2) + '</td></tr>' +
        '<tr style="text-align:right;border-top:1px solid ' + C.silver + ';color:' + C.mid + '"><td style="text-align:left">Expected with exact typical CL</td><td>' + R.floor.toFixed(3) + '</td><td>0</td><td>' + R.r2max.toFixed(2) + '</td></tr></table>' +
        '<p style="margin:8px 0 4px"><b>Log-linear fit returns</b> a WT exponent of ' + b[1].toFixed(2) + ' (simulated value 0.75) and a CRCL exponent of ' + b[2].toFixed(2) + ' (simulated value: ' + tw + '). <b>Boosted trees return</b> importance shares only, with no exponent.</p>' +
        '<p style="margin:4px 0;color:' + C.mid + '">Between-subject variability (SD 0.25) and the estimation noise (SD ' + sigma.toFixed(2) + ') remain in the observed values. Dots in the left plot are training subjects, with CL scaled to WT 70 kg by the simulated exponent 0.75.</p>';
    }
    function upd(){ ntr = +tI.value; sigma = +sI.value; qs('.tv', ctl).textContent = ntr; qs('.sv', ctl).textContent = sigma.toFixed(2); wrap(); }
    [tI, sI].forEach(function(e){ e.addEventListener('change', upd); e.addEventListener('input', function(){ qs('.tv', ctl).textContent = tI.value; qs('.sv', ctl).textContent = (+sI.value).toFixed(2); }); });
    qsa('[data-f] button', ctl).forEach(function(b){ b.addEventListener('click', function(){ form = +b.dataset.v; qsa('[data-f] button', ctl).forEach(function(z){ z.classList.toggle('on', z === b); }); wrap(); }); });
    qs('[data-seed]', ctl).addEventListener('click', function(){ seedI++; wrap(); });
    async function wrap(){ api.busy('Simulating subjects and fitting both models'); await tick(); try { draw(); } finally { api.idle(); } }
    function setUI(){
      qs('.tv', ctl).textContent = ntr; qs('.sv', ctl).textContent = sigma.toFixed(2); tI.value = ntr; sI.value = sigma; qsa('[data-f] button', ctl).forEach(function(z){ z.classList.toggle('on', +z.dataset.v === form); });
    }
    function sum(){ var R = mleRun2(ntr, form, sigma, seedI); return 'RMSE against the true typical CL: ' + R.tP.toFixed(3) + ' for the log-linear fit and ' + R.tG.toFixed(3) + ' for boosted trees.'; }
    async function render(){ await wrap(); }
    async function play(tok){
      form = 0; ntr = 200; sigma = 0.15; seedI = 1; setUI(); await wrap();
      api.cap('Power-law covariate form, 200 training subjects. ' + sum()); await api.sleep(4200);
      form = 1; setUI(); await wrap(); api.cap('Threshold form, which a log-linear term cannot represent. ' + sum()); await api.sleep(4200);
      ntr = 800; setUI(); await wrap(); api.cap('Threshold form, 800 training subjects. ' + sum()); await api.sleep(4200);
      ntr = 50; setUI(); await wrap(); api.cap('Threshold form, 50 training subjects. ' + sum()); await api.sleep(4200); api.cap(null);
    }
    return { render: render, play: play };
  } });
GUIDE.steps['mlpk/Population PK and ML'] = { notice: 'The error against the observed clearance changes little with training subjects, while the error against the typical clearance falls.', q: 'Why is the error against observed clearance close to a fixed value for both models?', a: 'Between-subject variability and estimation noise are not explained by any covariate. Their combined SD, the square root of 0.25 squared plus the noise squared, is the expected error in ln units for a model that matched the typical clearance exactly.' };
