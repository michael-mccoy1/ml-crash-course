"use strict";
/* Lesson 10, Pro steps: QSP, Uncertainty propagation, Credibility.
   Sources of fixed numbers and equations are given beside them.
   Data: the Biogen HLM set (public) through loadDataset('HLM'). The global endpoint is never read here. */

/* ---------- shared helpers (prefix pkd) ---------- */
function pkdSet(id, v){ var el = $(id); if (!el) return; el.value = v; var o = $('o-' + id); if (o) o.textContent = v; }
function pkdSeg(attr, items, cur){ return '<span class="seg" ' + attr + '>' + items.map(function(it, i){ return '<button type="button" data-v="' + i + '"' + (i === cur ? ' class="on"' : '') + '>' + esc(it) + '</button>'; }).join('') + '</span>'; }
function pkdPick(root, attr, k){ qsa('[' + attr + '] button', root).forEach(function(x){ x.classList.toggle('on', +x.dataset.v === k); }); }
function pkdSig(v){ v = +v; if (!isFinite(v)) return 'n/a'; var a = Math.abs(v); return a >= 100 ? v.toFixed(0) : (a >= 10 ? v.toFixed(1) : (a >= 1 ? v.toFixed(2) : v.toPrecision(2))); }
function pkdQuant(sorted, q){ var p = (sorted.length - 1) * q, i = Math.floor(p), f = p - i; return i + 1 < sorted.length ? sorted[i] * (1 - f) + sorted[i + 1] * f : sorted[i]; }
function pkdNormal(seed){ var r = rng(seed); return function(){ var u = 0; while (u === 0) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); }; }

/* A boosted-trees model of HLM log10 CLint fitted on the training compounds only (the lesson 02 workbench defaults). */
var PKD = { p: null, ds: null, F: null, rmse: null, q: null };
async function pkdReady(api){
  if (!PKD.p) PKD.p = (async function(){
    await tick(); var ds = await loadDataset('HLM'); await tick();
    var m = fitBoost(ds.BASE.slice(), ds.TRAIN, 150, 0.08, 3, 0, ds);
    PKD.ds = ds; PKD.F = m.F; PKD.rmse = rmse(ds.TEST, m.F, ds.Y); PKD.q = ds.TEST.filter(function(i){ return !ds.cens[i]; });
    var ys = PKD.q.map(function(i){ return ds.Y[i]; }).sort(function(a, b){ return a - b; }); PKD.med = ys[Math.floor(ys.length / 2)];
  })();
  api.busy('Fitting the HLM model on the training compounds'); try { await PKD.p; } finally { api.idle(); }
}

/* ---------- 1. QSP ---------- */
/* Indirect response models, the four basic types. Dayneka NL, Garg V, Jusko WJ. J Pharmacokinet Biopharm 1993; 21:457-478.
   With H = Imax C / (IC50 + C) (inhibition) or H = Emax C / (EC50 + C) (stimulation):
   I   dR/dt = kin (1 - H) - kout R      inhibition of production
   II  dR/dt = kin - kout (1 - H) R      inhibition of loss
   III dR/dt = kin (1 + H) - kout R      stimulation of production
   IV  dR/dt = kin - kout (1 + H) R      stimulation of loss
   Baseline R0 = kin / kout. The equations are in the form given in the Pumas indirect response tutorial, which cites the paper;
   the paper itself was not read. */
var PKD_IDR = [
  { n: 'I: inhibition of production', s: 1, f: function(H, R, kin, kout){ return kin * (1 - H) - kout * R; }, q: function(H, R0){ return R0 * (1 - H); } },
  { n: 'II: inhibition of loss', s: 1, f: function(H, R, kin, kout){ return kin - kout * (1 - H) * R; }, q: function(H, R0){ return R0 / Math.max(1 - H, 0.02); } },
  { n: 'III: stimulation of production', s: 0, f: function(H, R, kin, kout){ return kin * (1 + H) - kout * R; }, q: function(H, R0){ return R0 * (1 + H); } },
  { n: 'IV: stimulation of loss', s: 0, f: function(H, R, kin, kout){ return kin - kout * (1 + H) * R; }, q: function(H, R0){ return R0 / (1 + H); } }
];
LESSONS.pk.add({ level: 'pro', short: 'QSP', title: 'Quantitative systems pharmacology',
  change: 'Select the response type. Set Imax or Emax, IC50, kin, kout and the dose.',
  watch: 'Quantitative systems pharmacology (QSP) models describe the biological system that the drug acts on, in addition to the PK. The smallest example is the indirect response model: a response variable R is produced at rate kin and lost at rate kout, so its baseline is kin / kout, and the drug inhibits or stimulates either production or loss through a saturable function of concentration. The four types follow Dayneka, Garg and Jusko (1993). Here the PK is a one-compartment IV bolus with CL 10 L/h and V 50 L, fixed as an illustration. The response lags the concentration and recovers at the turnover rate kout, not at the elimination rate of the drug. The structure comes from biology and the parameters (Imax, IC50, kin, kout) are estimated from sparse biomarker data.',
  build: function(api){
    var CL = 10, V = 50, TEND = 72, type = 0, run = null;
    var pa = api.el('pn', 'left:28px;top:20px;width:270px;height:500px;overflow:auto', '<h4>Response model</h4>' + pkdSeg('data-ty', PKD_IDR.map(function(t){ return t.n.split(':')[0]; }), 0) + '<div class="tn" style="font-size:12px;color:' + C.mid + ';margin:4px 0 6px"></div>' + optRange('pkd1imax', 'Imax or Emax', 0.1, 1, 0.05, 0.9) + optRange('pkd1ic50', 'IC50 or EC50 (mg/L)', 0.05, 3, 0.05, 0.5) + optRange('pkd1kin', 'kin (response units/h)', 1, 20, 0.5, 5) + optRange('pkd1kout', 'kout (1/h)', 0.02, 0.5, 0.01, 0.1) + optRange('pkd1dose', 'Dose, IV bolus (mg)', 10, 500, 10, 100) + '<div class="ro" style="font-size:12.5px;line-height:1.5;color:' + C.char + ';margin-top:8px"></div>');
    var c1 = api.el('pn', 'left:316px;top:20px;width:616px;height:176px', '<h4>Plasma concentration, one-compartment PK</h4><canvas width="586" height="128"></canvas>');
    var c2 = api.el('pn', 'left:316px;top:204px;width:616px;height:196px', '<h4>Response relative to baseline</h4><canvas width="586" height="148"></canvas>');
    var tp = api.el('pn', 'left:316px;top:408px;width:616px;height:112px;overflow:auto', '<div class="tx" style="font-size:12.5px;line-height:1.5;color:' + C.char + '"></div>');
    function P(){ return { imax: +$('pkd1imax').value, ic50: +$('pkd1ic50').value, kin: +$('pkd1kin').value, kout: +$('pkd1kout').value, dose: +$('pkd1dose').value }; }
    function simulate(){
      var p = P(), M = PKD_IDR[type], R0 = p.kin / p.kout, conc = PKE.oneComp({ dose: p.dose, V: V, CL: CL, route: 'iv' }).conc;
      var f = function(t, y){ var Cc = conc(t), H = p.imax * Cc / (p.ic50 + Cc); return [M.f(H, y[0], p.kin, p.kout)]; };
      var s = PKE.rk4(f, [R0], 0, TEND, 0.1), ts = s.t, r = s.y.map(function(y){ return y[0] / R0; }), cs = ts.map(conc), q = ts.map(function(t, i){ var H = p.imax * cs[i] / (p.ic50 + cs[i]); return M.q(H, R0) / R0; });
      return { p: p, M: M, R0: R0, t: ts, r: r, c: cs, q: q };
    }
    function draw(){
      var S = simulate(), p = S.p, n = S.t.length, st = Math.max(1, Math.floor(n / 240)), pc = [], pr = [], pq = [], i, ext = 1, ti = 0;
      for (i = 0; i < n; i += st) { pc.push([S.t[i], S.c[i]]); pr.push([S.t[i], S.r[i]]); pq.push([S.t[i], Math.max(0, Math.min(S.q[i], 3))]); }
      for (i = 0; i < n; i++) if (Math.abs(S.r[i] - 1) > Math.abs(ext - 1)) { ext = S.r[i]; ti = S.t[i]; }
      lineChart(qs('canvas', c1), [{ pts: pc, color: C.char, w: 2.6, nodot: true, label: 'concentration' }], { xl: 'time (h)', yl: 'mg/L', x0: 0, x1: TEND, y0: 0 });
      var lo = Math.min(1, ext, 0.9), hi = Math.max(1, ext, 1.1), ymin = Math.floor(lo * 10) / 10, ymax = Math.ceil(Math.min(hi, 3) * 10) / 10;
      lineChart(qs('canvas', c2), [{ pts: pq, color: C.bg, w: 2, dash: [5, 4], nodot: true, label: 'response at steady state for the current concentration' }, { pts: pr, color: C.red, w: 3, nodot: true, label: 'response R / R0' }], { xl: 'time (h)', yl: 'R / R0', x0: 0, x1: TEND, y0: ymin, y1: ymax });
      var back = null; for (i = 0; i < n; i++) if (S.t[i] > ti && Math.abs(S.r[i] - 1) < 0.05) { back = S.t[i]; break; }
      qs('.tn', pa).textContent = S.M.n;
      qs('.ro', pa).innerHTML = 'Baseline R0 = kin / kout = <b>' + pkdSig(S.R0) + '</b> response units.<br>Turnover half-life ln 2 / kout = <b>' + pkdSig(Math.LN2 / p.kout) + ' h</b>.<br>Drug half-life ln 2 V / CL = <b>' + pkdSig(Math.LN2 * V / CL) + ' h</b>.<br>Largest change: <b>' + pkdSig(ext) + '</b> times baseline at <b>' + pkdSig(ti) + ' h</b>.<br>Within 5% of baseline again: <b>' + (back === null ? 'not within ' + TEND + ' h' : pkdSig(back) + ' h') + '</b>.';
      qs('.tx', tp).innerHTML = '<b>Structure and parameters.</b> The turnover structure (production, loss, saturable drug effect) is fixed by the biology. Imax, IC50, kin and kout are unknown for a new compound and are estimated by fitting biomarker data, which are usually sparse and noisy. Where the functional form of the drug effect is not known, a learned function can replace it inside the same ODE: see Hybrid mechanistic-ML models in lesson 11.<br><span style="color:' + C.mid + '">Equations: Dayneka, Garg and Jusko, J Pharmacokinet Biopharm 1993;21:457-478. PK parameters are illustrative.</span>';
    }
    function setType(k){ type = k; var el = $('pkd1imax'); el.max = PKD_IDR[k].s ? 1 : 5; if (PKD_IDR[k].s && +el.value > 1) pkdSet('pkd1imax', 1); pkdPick(pa, 'data-ty', k); draw(); }
    ['pkd1imax', 'pkd1ic50', 'pkd1kin', 'pkd1kout', 'pkd1dose'].forEach(function(id){ $(id).addEventListener('input', draw); });
    qsa('[data-ty] button', pa).forEach(function(b){ b.addEventListener('click', function(){ setType(+b.dataset.v); }); });
    async function play(tok){
      setType(0); pkdSet('pkd1kout', 0.1, 0); pkdSet('pkd1imax', 0.9); pkdSet('pkd1ic50', 0.5); pkdSet('pkd1dose', 100); draw();
      api.cap('Type I. The drug inhibits production, so R falls only after the existing R has turned over.'); await api.sleep(3200);
      await api.anim(2400, function(t){ pkdSet('pkd1kout', (0.1 - 0.08 * t).toFixed(2)); draw(); }); api.cap('A smaller kout lengthens the turnover half-life. The response then lags further behind the concentration and recovers more slowly.'); await api.sleep(3400);
      pkdSet('pkd1kout', 0.1); setType(3); api.cap('Type IV. The drug stimulates loss, which also lowers R. Compare the shape with type I.'); await api.sleep(3200); api.cap(null);
    }
    return { render: draw, play: play };
  } });
GUIDE.steps['pk/QSP'] = { notice: 'The response lags the concentration and recovers after the concentration has fallen.', q: 'Why does the response persist after the plasma concentration has fallen?', a: 'The drug changes the rate of production or loss of the response variable. The response returns to baseline at the turnover rate kout, not at the elimination rate of the drug.' };

/* ---------- 2. Uncertainty propagation ---------- */
/* Input distributions. CLint: lognormal in log10 units with the HLM test RMSE of the boosted model as the standard deviation (computed from the data).
   fup, ka: lognormal with an assumed geometric SD, F absorbed: logit-normal around 0.8 with an assumed SD on the logit scale.
   These three are assumptions chosen to illustrate propagation, not measured distributions; replace them with assay-specific values.
   Model: hepatic clearance by the well-stirred model (PKE.fromHLM, qh 20.7 mL/min/kg, 70 kg, bp 1) as the only clearance route, V 70 L (assumed),
   one-compartment first-order oral absorption, AUC = F dose / CL. F = Fabs times the fraction escaping hepatic first pass (1 - E). */
var PKD_N = 2000, PKD_SEED = 2026, PKD_OM_CL = 0.3, PKD_OM_V = 0.25;
LESSONS.pk.add({ level: 'pro', short: 'Uncertainty propagation', title: 'Uncertainty propagation',
  change: 'Set the point estimate of log10 CLint and the size of each input uncertainty. Read the intervals and the rank correlations.',
  watch: 'A predicted exposure carries the uncertainty of every input. Monte Carlo propagation draws each uncertain input from a distribution, runs the model for every draw and reports intervals of the output. Here 2,000 draws vary CLint (lognormal, with the HLM test RMSE of the boosted model as the standard deviation in log10 units), the fraction unbound in plasma, the absorption rate constant and the absorbed fraction; the last three distributions are assumptions. The model is a closed-form one-compartment oral model with hepatic clearance from the well-stirred model, because thousands of whole-body simulations are not practical in the browser. Parameter uncertainty is lack of knowledge about the typical value of a parameter for one compound. Between-subject variability is real variation among individuals. The grey histogram adds assumed between-subject variability of CL and V to the red parameter uncertainty alone; the two sources need separate sampling levels in a full analysis.',
  build: function(api){
    var DOSE = 100, V0 = 70, BP = 1, ready = false, res = null;
    var pa = api.el('pn', 'left:28px;top:20px;width:270px;height:500px;overflow:auto', '<h4>Inputs</h4>' + optRange('pkd2c', 'Point estimate, log10 CLint (mL/min/kg)', 0, 3, 0.05, 1.2) + optRange('pkd2sd', 'CLint error SD (log10 units)', 0, 1, 0.01, 0.55) + optRange('pkd2fu', 'fup geometric SD', 1, 2, 0.05, 1.3) + optRange('pkd2ka', 'ka geometric SD', 1, 3, 0.05, 1.5) + optRange('pkd2f', 'F absorbed, SD on the logit scale', 0, 1, 0.05, 0.3) + '<div class="ro" style="font-size:12px;line-height:1.5;color:' + C.char + ';margin-top:8px"></div>');
    var ha = api.el('pn', 'left:316px;top:20px;width:304px;height:262px', '<h4>AUC relative to the point estimate</h4><canvas width="274" height="214"></canvas>');
    var hb = api.el('pn', 'left:628px;top:20px;width:304px;height:262px', '<h4>Cmax relative to the point estimate</h4><canvas width="274" height="214"></canvas>');
    var ip = api.el('pn', 'left:316px;top:290px;width:616px;height:230px;overflow:auto', '<div class="ip" style="font-size:12.5px;line-height:1.5;color:' + C.char + '"></div>');
    function compute(){
      var c = +$('pkd2c').value, sd = +$('pkd2sd').value, gfu = Math.log(+$('pkd2fu').value), gka = Math.log(+$('pkd2ka').value), sf = +$('pkd2f').value;
      var z = pkdNormal(PKD_SEED), N = PKD_N, i, out = { lc: [], fup: [], ka: [], fa: [], bcl: [], bv: [], aucU: [], aucT: [], cmU: [], cmT: [] };
      function one(l, fup, ka, fa, ecl, ev){ var h = PKE.fromHLM(l, { fup: fup, bp: BP }), cl = h.clhPlasmaLh * ecl, F = fa * h.fh, v = V0 * ev, s = PKE.oneComp({ dose: DOSE, V: v, CL: cl, ka: ka, F: F, route: 'oral' }).summary; return [s.auc, s.cmax]; }
      var ref = one(c, 1, 1, 0.8, 1, 1);
      for (i = 0; i < N; i++) {
        var l = c + sd * z(), fup = Math.min(1, Math.exp(gfu * z()) * 1) , ka = Math.exp(gka * z()) * 1, fa = 1 / (1 + Math.exp(-(Math.log(0.8 / 0.2) + sf * z()))), ecl = Math.exp(PKD_OM_CL * z()), ev = Math.exp(PKD_OM_V * z());
        fup = Math.min(1, fup);
        var u = one(l, fup, ka, fa, 1, 1), t = one(l, fup, ka, fa, ecl, ev);
        out.lc.push(l); out.fup.push(fup); out.ka.push(ka); out.fa.push(fa); out.bcl.push(ecl); out.bv.push(ev); out.aucU.push(u[0]); out.cmU.push(u[1]); out.aucT.push(t[0]); out.cmT.push(t[1]);
      }
      out.ref = ref; return out;
    }
    function hist(canvas, U, T, ref, xl){
      var g = cv(canvas), x = g.x, box = { l: 44, r: g.w - 10, t: 10, b: g.h - 44 }, nb = 45, lo = -1.5, hi = 1.5, bins = function(a){ var b = new Array(nb).fill(0); a.forEach(function(v){ var k = Math.floor((Math.log10(v / ref) - lo) / (hi - lo) * nb); b[clamp(k, 0, nb - 1)]++; }); return b; }, bu = bins(U), bt = bins(T), mx = Math.max(Math.max.apply(null, bu), Math.max.apply(null, bt)), top = Math.ceil(mx / 100) * 100;
      var tk = [[-1, '0.1'], [-0.301, '0.5'], [0, '1'], [0.301, '2'], [1, '10']].map(function(a){ return { v: a[0], t: a[1] }; });
      axes(g, box, [lo, hi], [0, top], xl, 'draws', 4, { xticks: tk, ydec: 0 });
      var sx = function(v){ return box.l + (v - lo) / (hi - lo) * (box.r - box.l); }, sy = function(v){ return box.b - v / top * (box.b - box.t); }, bw = (box.r - box.l) / nb;
      x.fillStyle = 'rgba(111,129,150,0.45)'; bt.forEach(function(v, k){ x.fillRect(box.l + k * bw, sy(v), bw - 0.6, box.b - sy(v)); });
      x.strokeStyle = C.red; x.lineWidth = 2; x.beginPath(); bu.forEach(function(v, k){ var X0 = box.l + k * bw, Y = sy(v); if (k === 0) x.moveTo(X0, Y); else x.lineTo(X0, Y); x.lineTo(X0 + bw, Y); }); x.stroke();
      x.strokeStyle = C.char; x.lineWidth = 1; x.setLineDash([4, 3]); x.beginPath(); x.moveTo(sx(0), box.t); x.lineTo(sx(0), box.b); x.stroke(); x.setLineDash([]);
    }
    function draw(){
      if (!ready) return; res = compute(); var sd = +$('pkd2sd').value, ref = res.ref;
      hist(qs('canvas', ha), res.aucU, res.aucT, ref[0], 'AUC / point-estimate AUC (log scale)'); hist(qs('canvas', hb), res.cmU, res.cmT, ref[1], 'Cmax / point-estimate Cmax (log scale)');
      function iv(a){ var s = a.slice().sort(function(p, q){ return p - q; }); return [pkdQuant(s, 0.05), pkdQuant(s, 0.5), pkdQuant(s, 0.95)]; }
      var au = iv(res.aucU), at = iv(res.aucT), cu = iv(res.cmU), ct = iv(res.cmT), idx = res.lc.map(function(_, k){ return k; });
      var inputs = [['CLint', res.lc, 'U'], ['fup', res.fup, 'U'], ['ka', res.ka, 'U'], ['F absorbed', res.fa, 'U'], ['Between-subject CL', res.bcl, 'T'], ['Between-subject V', res.bv, 'T']];
      var rows = inputs.map(function(r){ var y = r[2] === 'U' ? res.aucU : res.aucT, y2 = r[2] === 'U' ? res.cmU : res.cmT; return { n: r[0], a: spearman(idx, r[1], y), c: spearman(idx, r[1], y2) }; });
      function bar(v){ var w = Math.min(1, Math.abs(v)) * 100; return '<div style="position:relative;height:12px;background:' + C.light + ';border-radius:3px"><i style="position:absolute;top:0;height:12px;border-radius:3px;background:' + (v >= 0 ? C.red : C.mid) + ';' + (v >= 0 ? 'left:50%;width:' + w / 2 + '%' : 'right:50%;width:' + w / 2 + '%') + '"></i><i style="position:absolute;left:50%;top:-2px;height:16px;width:1px;background:' + C.char + '"></i></div>'; }
      qs('.ip', ip).innerHTML = '<table style="width:100%;border-collapse:collapse;font-size:12.5px"><tr style="font-weight:700;border-bottom:1px solid ' + C.silver + '"><td></td><td>5th</td><td>Median</td><td>95th</td><td>95th / 5th</td></tr>' +
        [['AUC, parameter uncertainty', au], ['AUC, plus between-subject variability', at], ['Cmax, parameter uncertainty', cu], ['Cmax, plus between-subject variability', ct]].map(function(r){ return '<tr style="border-bottom:1px solid ' + C.light + '"><td style="padding:3px 0">' + r[0] + '</td><td>' + pkdSig(r[1][0]) + '</td><td>' + pkdSig(r[1][1]) + '</td><td>' + pkdSig(r[1][2]) + '</td><td><b>' + pkdSig(r[1][2] / r[1][0]) + '</b></td></tr>'; }).join('') + '</table>' +
        '<div style="font-size:11px;color:' + C.mid + ';margin:3px 0 8px">AUC in mg h/L and Cmax in mg/L for a 100 mg oral dose; 90% intervals of ' + PKD_N + ' draws, seed ' + PKD_SEED + '.</div>' +
        '<div class="ol">Spearman rank correlation with AUC (left) and Cmax (right)</div>' + rows.map(function(r){ return '<div style="display:grid;grid-template-columns:130px 1fr 44px 1fr 44px;gap:8px;align-items:center;margin:3px 0"><span>' + r.n + '</span>' + bar(r.a) + '<b style="text-align:right">' + r.a.toFixed(2) + '</b>' + bar(r.c) + '<b style="text-align:right">' + r.c.toFixed(2) + '</b></div>'; }).join('') +
        '<p style="margin:8px 0 0">Inputs: CLint error SD ' + sd.toFixed(2) + ' log10 units (HLM test RMSE of the boosted model: ' + PKD.rmse.toFixed(2) + '); fup, ka and F distributions and the between-subject variability (omega CL ' + PKD_OM_CL + ', omega V ' + PKD_OM_V + ') are assumptions. V is 70 L, bp is 1, and hepatic clearance is the only route. The correlations rank the inputs by their influence on the output.</p>';
      qs('.ro', pa).innerHTML = 'Point estimate: AUC <b>' + pkdSig(ref[0]) + '</b> mg h/L, Cmax <b>' + pkdSig(ref[1]) + '</b> mg/L, with fup 1, ka 1 per h and F absorbed 0.8.<br>The extraction ratio at the point estimate is <b>' + PKE.fromHLM(+$('pkd2c').value, { fup: 1, bp: BP }).extraction.toFixed(2) + '</b>.';
    }
    ['pkd2c', 'pkd2sd', 'pkd2fu', 'pkd2ka', 'pkd2f'].forEach(function(id){ $(id).addEventListener('input', draw); });
    async function render(){ await pkdReady(api); if (!ready) { ready = true; pkdSet('pkd2sd', PKD.rmse.toFixed(2)); } draw(); }
    async function play(tok){
      await render(); pkdSet('pkd2c', 1.2); pkdSet('pkd2sd', PKD.rmse.toFixed(2)); pkdSet('pkd2fu', 1.3); pkdSet('pkd2ka', 1.5); pkdSet('pkd2f', 0.3); draw();
      api.cap('Each draw is one set of input values. The red outline is the spread of exposure from parameter uncertainty alone.'); await api.sleep(3400);
      await api.anim(2200, function(t){ pkdSet('pkd2sd', (PKD.rmse * (1 - t)).toFixed(2)); draw(); }); api.cap('With no CLint error the remaining spread comes from fup, ka and F absorbed.'); await api.sleep(3000);
      pkdSet('pkd2sd', PKD.rmse.toFixed(2)); await api.anim(1800, function(t){ pkdSet('pkd2c', (1.2 + 1.2 * t).toFixed(2)); draw(); }); api.cap('At high CLint the extraction ratio approaches 1, so the same CLint error changes AUC less.'); await api.sleep(3200); api.cap(null);
    }
    return { render: render, play: play };
  } });
GUIDE.steps['pk/Uncertainty propagation'] = { notice: 'The grey histogram is wider than the red one because between-subject variability is added to the parameter uncertainty.', q: 'What is the difference between parameter uncertainty and between-subject variability?', a: 'Parameter uncertainty is lack of knowledge about the typical value of a parameter for one compound and shrinks with better data. Between-subject variability is real variation among individuals and does not shrink with better data.' };

/* ---------- 3. Credibility ---------- */
/* Terms and section numbers follow ICH M15, General principles for model-informed drug development (Step 4, adopted 29 January 2026; text read from the EMA Step 5 document).
   Sections: 2.1.1 question of interest, 2.1.2 context of use, 2.1.3 model influence, 2.1.4 consequence of wrong decision, 2.1.5 model risk, 2.1.6 model impact,
   2.2.1 technical criteria, 2.2.2 appropriateness of proposed MIDD, 2.2.3 evaluation of model and model outcomes, 2.2.4 outcome of the MIDD evidence assessment,
   3 model evaluation (verification, validation and applicability assessment), 4.1 model analysis plan, 4.2 model analysis report, Appendix 1 assessment table.
   M15 states no numerical acceptance criterion. The three contexts of use below are illustrations written for this lesson; they are not taken from M15. */
var PKD_CTX = [
  { n: 'First administration to humans', q: 'Which starting dose is selected for the first administration to humans?', u: 'A PBPK model whose hepatic clearance input is predicted by a machine learning model of HLM CLint predicts human exposure per dose. Nonclinical safety data and the exposure at the dose limit inform the answer.' },
  { n: 'Co-medication', q: 'Is the change in exposure caused by a co-medication within the range that requires no dose adjustment?', u: 'A PBPK model predicts the change in exposure of a drug when a second drug is given. Clinical data from other studies of the drug inform the answer.' },
  { n: 'Subgroup exposure', q: 'What is the exposure in a subgroup that was sampled sparsely?', u: 'A population PK model fitted to observed concentrations describes exposure in the subgroup. The observed concentrations are the primary evidence and the model outcome supports their interpretation.' }
];
var PKD_EVID = [
  { g: 'Verification (Section 3)', i: [
    ['Key user-generated code, equations and calculations are verified, and the verification is documented and available for review.', ''],
    ['The computerized system is reliable, reproducible and traceable, and software testing is documented.', ''],
    ['Quality assurance is applied to data management and modelling activities.', '']] },
  { g: 'Validation and applicability assessment (Section 3)', i: [
    ['The relevance of the data to the question of interest is justified, exclusions are justified and the potential for bias is assessed.', 'Applicability domain, lesson 07'],
    ['Model structure and parameters are consistent with knowledge of the drug, pharmacology, physiology and disease.', 'PBPK model structure, lesson 10'],
    ['Limitations of the data and the model are described.', ''],
    ['Key assumptions are identified, described and justified, and alternatives are considered.', ''],
    ['Method-specific issues are considered. M15 gives overfitting of an AI or ML model as an example.', 'Training and test performance, lesson 01'],
    ['Robustness is assessed, including sensitivity analysis, with the dependency on data, parameters and assumptions characterised.', 'Parameter sensitivity, lesson 10'],
    ['Performance (precision and bias) is assessed with graphical and numerical metrics, prioritising those that relate to the question of interest.', 'Evaluation metrics, lesson 01'],
    ['External validation with independent data is encouraged. Depending on the question of interest, context of use and model risk, it can be essential.', 'Training and test sets, lesson 01'],
    ['Simulation methods and scenarios are described and account for parameter and assumption uncertainties.', 'Uncertainty propagation, lesson 10']] },
  { g: 'Technical criteria and plan (Sections 2.2.1 and 4.1)', i: [
    ['Technical criteria, specific to the question of interest, are pre-defined and their rationale is given; the evaluation is commensurate with model risk.', ''],
    ['The planned analyses and evaluation activities are documented in a model analysis plan before the data are accessed, and changes are justified in the model analysis report.', '']] }
];
var PKD_CR = { p: null, d: null };
async function pkdCredData(api){
  if (!PKD_CR.p) PKD_CR.p = (async function(){
    await pkdReady(api); var ds = PKD.ds, DOSE = 100, fe = [], feC = [], feL = [], pa = [], ob = [];
    PKD.q.forEach(function(i){
      var a = PKE.fromHLM(PKD.F[i], { fup: 1, bp: 1 }), b = PKE.fromHLM(ds.Y[i], { fup: 1, bp: 1 }), aucP = DOSE / a.clhPlasmaLh, aucO = DOSE / b.clhPlasmaLh;
      fe.push(aucP / aucO); feC.push(Math.pow(10, PKD.F[i] - ds.Y[i])); feL.push(b.extraction); pa.push(aucP); ob.push(aucO);
    });
    function stats(v){ var l = v.map(function(x){ return Math.log10(x); }); return { gmfe: Math.pow(10, mean(l.map(Math.abs))), bias: Math.pow(10, mean(l)), w2: v.filter(function(x){ return x >= 0.5 && x <= 2; }).length / v.length, n: v.length }; }
    function grp(lo, hi){ var k = []; fe.forEach(function(x, j){ if (feL[j] >= lo && feL[j] < hi) k.push(x); }); return stats(k); }
    PKD_CR.d = { clint: stats(feC), auc: stats(fe), low: grp(0, 0.3), mid: grp(0.3, 0.7), high: grp(0.7, 1.01), pa: pa, ob: ob };
  })();
  api.busy('Computing the validation metrics'); try { await PKD_CR.p; } finally { api.idle(); }
}
LESSONS.pk.add({ level: 'pro', short: 'Credibility', title: 'Model credibility',
  change: 'Select a context of use. Set the model influence, the consequence of a wrong decision and the model impact. Tick the evidence that is planned.',
  watch: 'ICH M15 (adopted at Step 4 on 29 January 2026) gives a risk-based framework for assessing the evidence from a model. The question of interest is what the model-informed work must answer. The context of use states the role and scope of the model, the data used to build it, and the other evidence that informs the answer. Model influence is the intended weight of the model outcomes in the decision, relative to that other evidence. The consequence of a wrong decision is the potential negative effect of an incorrect decision. Model risk combines the two, and the model evaluation (Section 3) is to be commensurate with it. The three contexts of use here are illustrations written for this lesson. The validation example below is also an illustration: M15 states no numerical acceptance criterion, and technical criteria are pre-defined for each question of interest (Section 2.2.1).',
  build: function(api){
    var ctx = 0, inf = 1, con = 2, imp = 1, checked = {}, RATE = ['Low', 'Medium', 'High'];
    var ta = api.el('pn', 'left:28px;top:20px;width:438px;height:500px;overflow:auto', '<h4>Assessment table, key elements (Appendix 1)</h4><div class="sc" style="margin-bottom:6px"><div class="ol">Illustrative context of use</div>' + pkdSeg('data-cx', PKD_CTX.map(function(c){ return c.n; }), 0) + '</div><div class="rt"></div>');
    var ea = api.el('pn', 'left:484px;top:20px;width:448px;height:200px;overflow:auto', '<h4>Model evaluation evidence (Sections 2.2.1, 3 and 4.1)</h4><div class="ev"></div>');
    var va = api.el('pn', 'left:484px;top:228px;width:448px;height:292px;overflow:auto', '<h4>Illustration: exposure error from the HLM model</h4><div style="display:flex;gap:10px"><div class="vt" style="flex:1;font-size:12px;line-height:1.45;color:' + C.char + '"></div><canvas width="170" height="158"></canvas></div>');
    function risk(){ if (inf === 0 && con === 0) return ['Low', 'Both ratings are low.']; if (inf === 2 && con === 2) return ['High', 'Both ratings are high.']; return ['Not fixed by M15', 'The ratings differ or are both medium. M15 says the rating may be driven by the more influential of the two and that the justification records it. Rate it and justify the rating.']; }
    function ratingSeg(attr, cur){ return pkdSeg(attr, RATE, cur); }
    function drawTable(){
      var c = PKD_CTX[ctx], r = risk(); function row(a, b){ return '<tr style="border-top:1px solid ' + C.silver + '"><td style="padding:5px 8px 5px 0;width:116px;vertical-align:top;font-weight:700">' + a + '</td><td style="padding:5px 0">' + b + '</td></tr>'; }
      qs('.rt', ta).innerHTML = '<table style="width:100%;border-collapse:collapse;font-size:12.5px;color:' + C.char + ';line-height:1.45">' + row('Question of interest (2.1.1)', esc(c.q)) + row('Context of use (2.1.2)', esc(c.u)) + row('Model influence (2.1.3)', ratingSeg('data-inf', inf)) + row('Consequence of wrong decision (2.1.4)', ratingSeg('data-con', con)) +
        row('Model risk (2.1.5)', '<b style="color:' + C.red + '">' + r[0] + '</b><div style="font-size:11.5px;color:' + C.mid + '">' + r[1] + '</div>') + row('Model impact (2.1.6)', ratingSeg('data-imp', imp) + '<div style="font-size:11.5px;color:' + C.mid + '">How far the strategy varies from regulatory standards, or from expectations where there is no standard.</div>') +
        row('Planning stage', 'Technical criteria (2.2.1) and appropriateness of the proposed MIDD (2.2.2): written by the user, specific to the question of interest.') + row('Submission stage', 'Evaluation of model and model outcomes (2.2.3) and outcome of the MIDD evidence assessment (2.2.4).') + '</table>' +
        '<p style="font-size:11.5px;color:' + C.mid + ';margin:6px 0 0">A justification is expected for every rating (Section 2.1). The contexts and the default ratings shown here are illustrations, not entries from M15.</p>';
      qsa('[data-cx] button', ta).forEach(function(b){ b.onclick = function(){ ctx = +b.dataset.v; pkdPick(ta, 'data-cx', ctx); drawTable(); }; });
      [['data-inf', function(k){ inf = k; }], ['data-con', function(k){ con = k; }], ['data-imp', function(k){ imp = k; }]].forEach(function(a){ qsa('[' + a[0] + '] button', ta).forEach(function(b){ b.onclick = function(){ a[1](+b.dataset.v); drawTable(); }; }); });
    }
    function drawEv(){
      var n = 0, tot = 0; PKD_EVID.forEach(function(g){ g.i.forEach(function(it, k){ tot++; if (checked[g.g + k]) n++; }); });
      qs('.ev', ea).innerHTML = '<div style="font-size:12px;color:' + C.mid + ';margin-bottom:6px">Evidence items listed in M15. The depth of each is commensurate with model risk (2.1.5 and 3). <b style="color:' + C.red + '">' + n + ' of ' + tot + ' planned.</b></div>' + PKD_EVID.map(function(g){ return '<div class="ol" style="margin-top:6px">' + g.g + '</div>' + g.i.map(function(it, k){ return '<label style="display:grid;grid-template-columns:18px 1fr;gap:6px;align-items:start;font-size:12px;line-height:1.4;margin:3px 0;cursor:pointer"><input type="checkbox" data-e="' + esc(g.g + k) + '"' + (checked[g.g + k] ? ' checked' : '') + '><span>' + esc(it[0]) + (it[1] ? ' <span style="color:' + C.red + '">In this course: ' + esc(it[1]) + '.</span>' : '') + '</span></label>'; }).join(''); }).join('');
      qsa('input[data-e]', ea).forEach(function(i){ i.onchange = function(){ checked[i.dataset.e] = i.checked; drawEv(); }; });
    }
    function drawVal(){
      var d = PKD_CR.d; if (!d) return; function f(s){ return pkdSig(s.gmfe) + ' / ' + (s.w2 * 100).toFixed(0) + '%'; }
      qs('.vt', va).innerHTML = '<table style="width:100%;border-collapse:collapse"><tr style="font-weight:700;border-bottom:1px solid ' + C.silver + '"><td></td><td>GMFE</td><td>Within 2-fold</td></tr>' +
        [['CLint', d.clint], ['AUC, IV 100 mg', d.auc], ['AUC, extraction under 0.3', d.low], ['AUC, extraction 0.3 to 0.7', d.mid], ['AUC, extraction 0.7 and over', d.high]].map(function(r){ return '<tr style="border-bottom:1px solid ' + C.light + '"><td style="padding:2px 0">' + r[0] + ' (n = ' + r[1].n + ')</td><td><b>' + pkdSig(r[1].gmfe) + '</b></td><td><b>' + (r[1].w2 * 100).toFixed(0) + '%</b></td></tr>'; }).join('') + '</table>' +
        '<div style="font-size:11px;color:' + C.mid + ';margin-top:4px">GMFE is the geometric mean fold error, 10 to the mean absolute log10 ratio of predicted to reference. The reference is the AUC from the measured CLint, not an observed clinical exposure, so only the CLint error is measured. Well-stirred model, fup 1, bp 1, 70 kg. Quantified test compounds only. Illustration, not a threshold.</div>';
      var g = cv(qs('canvas', va)), x = g.x, box = { l: 36, r: g.w - 8, t: 8, b: g.h - 34 }, lo = -1.5, hi = 1.5, sx = function(v){ return box.l + (v - lo) / (hi - lo) * (box.r - box.l); }, sy = function(v){ return box.b - (v - lo) / (hi - lo) * (box.b - box.t); };
      axes(g, box, [lo, hi], [lo, hi], 'AUC from measured CLint', 'AUC predicted', 3, { xticks: [{ v: -1, t: '0.1' }, { v: 0, t: '1' }, { v: 1, t: '10' }], yticks: [{ v: -1, t: '0.1' }, { v: 0, t: '1' }, { v: 1, t: '10' }] });
      x.strokeStyle = C.char; x.lineWidth = 1; x.beginPath(); x.moveTo(sx(lo), sy(lo)); x.lineTo(sx(hi), sy(hi)); x.stroke(); x.setLineDash([4, 3]); x.strokeStyle = C.bg; [Math.log10(2), -Math.log10(2)].forEach(function(o){ x.beginPath(); x.moveTo(sx(lo), sy(lo + o)); x.lineTo(sx(hi), sy(hi + o)); x.stroke(); }); x.setLineDash([]);
      x.fillStyle = 'rgba(225,36,42,0.35)'; var step = Math.max(1, Math.floor(d.pa.length / 300)); for (var k = 0; k < d.pa.length; k += step) { x.beginPath(); x.arc(sx(clamp(Math.log10(d.ob[k]), lo, hi)), sy(clamp(Math.log10(d.pa[k]), lo, hi)), 2.2, 0, 6.3); x.fill(); }
    }
    async function render(){ drawTable(); drawEv(); await pkdCredData(api); drawVal(); }
    async function play(tok){
      await render(); ctx = 0; inf = 1; con = 2; drawTable(); api.cap('The question of interest and the context of use come first. Both are written before any rating.'); await api.sleep(3400);
      inf = 0; con = 0; drawTable(); api.cap('Both ratings low: M15 states that model risk is low.'); await api.sleep(3000);
      inf = 2; con = 2; drawTable(); api.cap('Both ratings high: M15 states that model risk is high.'); await api.sleep(3000);
      inf = 1; con = 2; drawTable(); api.cap('When the ratings differ, M15 does not fix the outcome. The rating is chosen and justified.'); await api.sleep(3400);
      api.cap('The evidence list is from Sections 3, 2.2.1 and 4.1. Each item can be addressed at a depth commensurate with the model risk.'); await api.sleep(3400);
      api.cap('The validation metric below is an illustration computed from the course data. M15 gives no numerical criterion.'); await api.sleep(3400); api.cap(null);
    }
    return { render: render, play: play };
  } });
GUIDE.steps['pk/Credibility'] = { notice: 'When model influence and consequence differ, M15 does not fix the model risk.', q: 'In ICH M15, what two ratings combine to give model risk?', a: 'Model influence, the intended weight of the model outcomes in the decision, and the consequence of a wrong decision. The model evaluation is to be commensurate with the model risk.' };
