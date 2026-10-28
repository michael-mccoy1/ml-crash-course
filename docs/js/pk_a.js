"use strict";
/* Lesson 10, Beginner steps. PK is treated as the interface between machine learning predictions and decisions.
   Sources of every fixed number are given beside it. */

function pkaSet(id, v, draw){ var el = $(id); if (!el) return; el.value = v; var o = $('o-' + id); if (o) o.textContent = v; if (draw) draw(); }
function pkaNice(v, d){ return v >= 100 ? v.toFixed(0) : (v >= 10 ? v.toFixed(1) : v.toFixed(d === undefined ? 2 : d)); }

/* ---------- 1. Concentration-time profile ---------- */
LESSONS.pk.add({ level: 'beginner', short: 'Concentration-time profile', title: 'Concentration-time profile',
  change: 'Set the dose and the four parameters. Switch the route and the axis.',
  watch: 'A one-compartment model reduces a concentration-time profile to four parameters: clearance (CL), volume of distribution (V), absorption rate constant (ka) and bioavailability (F). Machine learning models for PK are asked to predict these parameters, or quantities that determine them, for compounds that have not been dosed. The chart is computed from the closed-form solution of the model, and the readouts are computed from the same curve.',
  build: function(api){
    var pa = api.el('pn', 'left:28px;top:20px;width:270px;height:500px;overflow:auto', '<h4>Parameters</h4>' + optRange('pk1dose', 'Dose (mg)', 10, 500, 10, 100) + optRange('pk1cl', 'Clearance CL (L/h)', 1, 60, 1, 10) + optRange('pk1v', 'Volume V (L)', 10, 300, 5, 50) + optRange('pk1ka', 'Absorption rate ka (1/h)', 0.2, 5, 0.1, 1) + optRange('pk1f', 'Bioavailability F', 0.1, 1, 0.05, 0.8) + '<div class="og"><div class="ol">Route</div>' + inSeg(['IV bolus', 'Oral'], 0, 'data-route') + '</div><div class="og"><div class="ol">Concentration axis</div>' + inSeg(['Linear', 'Log'], 0, 'data-axis') + '</div>');
    var cp = api.el('pn', 'left:316px;top:20px;width:616px;height:304px', '<h4>Plasma concentration against time</h4><canvas width="586" height="256"></canvas>');
    var rp = api.el('pn', 'left:316px;top:336px;width:616px;height:184px;overflow:auto', '<div class="rd"></div>');
    var route = 0, logAxis = false;
    function vals(){ return { dose: +$('pk1dose').value, CL: +$('pk1cl').value, V: +$('pk1v').value, ka: +$('pk1ka').value, F: +$('pk1f').value }; }
    function draw(){
      var p = vals(), m = PKE.oneComp({ dose: p.dose, V: p.V, CL: p.CL, ka: p.ka, F: p.F, route: route ? 'oral' : 'iv' }), s = m.summary, tEnd = Math.min(168, Math.max(8, Math.ceil(5 * s.thalf / 2) * 2)), pts = [], n = 240, i, cmin = s.cmax / 100, ser;
      for (i = 0; i <= n; i++) { var t = tEnd * i / n, c = m.conc(t); pts.push([t, logAxis ? Math.log10(Math.max(c, cmin)) : c]); }
      var peak = logAxis ? Math.log10(s.cmax) : s.cmax;
      ser = [{ pts: pts, color: C.red, w: 3, nodot: true, label: route ? 'oral' : 'IV bolus' }, { pts: [[s.tmax, peak]], color: C.char, w: 1, label: 'Cmax' }];
      var o = { xl: 'time (h)', yl: logAxis ? 'plasma concentration (mg/L, log scale)' : 'plasma concentration (mg/L)', x0: 0, x1: tEnd };
      if (logAxis) { var lo = Math.floor(Math.log10(cmin)), hi = Math.ceil(Math.log10(s.cmax)), tk = []; for (var e = lo; e <= hi; e++) tk.push({ v: e, t: e >= 0 ? String(Math.pow(10, e)) : Math.pow(10, e).toFixed(-e) }); o.y0 = lo; o.y1 = hi; o.yticks = tk; } else { o.y0 = 0; }
      lineChart(qs('canvas', cp), ser, o);
      var rd = function(l, v, u){ return '<div style="min-width:104px"><div class="ol">' + l + '</div><div class="big" style="font-size:24px;color:' + C.red + ';margin:0">' + v + '</div><div style="font-size:11px;color:' + C.mid + '">' + u + '</div></div>'; };
      qs('.rd', rp).innerHTML = '<div style="display:flex;gap:14px;flex-wrap:wrap;margin-bottom:8px">' + rd('Cmax', pkaNice(s.cmax), 'mg/L') + rd('tmax', pkaNice(s.tmax), 'h') + rd('AUC', pkaNice(s.auc), 'mg h/L, F dose / CL') + rd('Half-life', pkaNice(s.thalf), 'h, ln 2 V / CL') + '</div>' +
        '<div style="font-size:12px;line-height:1.5;color:' + C.char + '"><b>Model.</b> ke = CL / V. IV: C(t) = dose / V exp(-ke t). Oral: C(t) = F dose ka / (V (ka - ke)) (exp(-ke t) - exp(-ka t)). AUC = F dose / CL.<br><b>Targets for ML.</b> CL, V, F and ka. The course HLM data inform CL through IVIVE (next step). V, F and ka need other data sets, which this course does not provide.</div>';
    }
    ['pk1dose', 'pk1cl', 'pk1v', 'pk1ka', 'pk1f'].forEach(function(id){ $(id).addEventListener('input', draw); });
    qsa('[data-route] button', pa).forEach(function(b){ b.addEventListener('click', function(){ route = +b.dataset.v; qsa('[data-route] button', pa).forEach(function(x){ x.classList.toggle('on', x === b); }); draw(); }); });
    qsa('[data-axis] button', pa).forEach(function(b){ b.addEventListener('click', function(){ logAxis = +b.dataset.v === 1; qsa('[data-axis] button', pa).forEach(function(x){ x.classList.toggle('on', x === b); }); draw(); }); });
    function pick(seg, k){ qsa(seg + ' button', pa).forEach(function(x){ x.classList.toggle('on', +x.dataset.v === k); }); }
    async function play(tok){
      route = 0; logAxis = false; pick('[data-route]', 0); pick('[data-axis]', 0); pkaSet('pk1cl', 10, null); pkaSet('pk1v', 50, null); draw();
      api.cap('IV bolus. The concentration starts at dose / V and decays with ke = CL / V.'); await api.sleep(3000);
      await api.anim(2000, function(t){ pkaSet('pk1cl', Math.round(10 + 10 * t), draw); }); api.cap('Doubling CL halves the AUC and the half-life at fixed V.'); await api.sleep(3000);
      logAxis = true; pick('[data-axis]', 1); draw(); api.cap('On a log axis the terminal phase is a straight line with slope -ke / 2.303.'); await api.sleep(3000);
      route = 1; logAxis = false; pick('[data-route]', 1); pick('[data-axis]', 0); draw(); api.cap('Oral dosing adds ka and F. Cmax and tmax now depend on absorption as well as elimination.'); await api.sleep(3200); api.cap(null);
    }
    return { render: draw, play: play };
  } });
GUIDE.steps['pk/Concentration-time profile'] = { notice: 'On the log axis the terminal phase is a straight line.', q: 'Which parameters of this model does a machine learning model predict for a new compound?', a: 'CL, V, F and ka, or the in vitro quantities from which they are derived.' };

/* ---------- 2. In vitro to in vivo clearance ---------- */
LESSONS.pk.add({ level: 'beginner', short: 'In vitro to in vivo clearance', title: 'In vitro to in vivo clearance',
  change: 'Set log10 CLint and the fraction unbound in plasma. Read the clearance and the extraction ratio.',
  watch: 'The Biogen HLM value is log10 intrinsic clearance in mL/min/kg. This step treats it as an intrinsic clearance already scaled to body weight: it applies no further scaling for microsomal protein or liver weight and no correction for binding in the incubation, because the public sources do not state how the value was scaled. The well-stirred model converts intrinsic clearance to hepatic blood clearance, CLh = Q fu CLint / (Q + fu CLint), with hepatic blood flow Q. CLh cannot exceed Q, so the extraction ratio saturates at high CLint. The fraction unbound is an assumption of this step, because the data set contains no fraction unbound.',
  build: function(api){
    var QH = 20.7, FLOOR = 0.6757, ds = null, meas = [], nc = 0;
    var ca = api.el('pn', 'left:28px;top:20px;width:590px;height:500px', '<h4>Hepatic clearance against intrinsic clearance, with the HLM compounds</h4><canvas width="560" height="430"></canvas>');
    var pa = api.el('pn', 'left:632px;top:20px;width:300px;height:500px;overflow:auto', '<h4>Settings</h4>' + optRange('pk2x', 'log10 CLint (mL/min/kg)', -1, 3.5, 0.05, 1.2) + optRange('pk2fu', 'Fraction unbound in plasma (assumed)', 0.01, 1, 0.01, 1) + '<div class="ro" style="font-size:12.5px;line-height:1.5;color:' + C.char + '"></div>');
    function mapAll(fup){ return meas.map(function(v){ return PKE.fromHLM(v, { fup: fup, qhPerKg: QH }); }); }
    function draw(){
      if (!ds) return; var x = +$('pk2x').value, fu = +$('pk2fu').value, g = cv(qs('canvas', ca)), c = g.x, L = 58, R = g.w - 14, T = 16, B = g.h - 56, X0 = -1, X1 = 3.5, YM = 24;
      var sx = function(v){ return L + (v - X0) / (X1 - X0) * (R - L); }, sy = function(v){ return B - v / YM * (B - T); };
      c.strokeStyle = C.silver; c.fillStyle = C.mid; c.font = '11px Aptos, "Segoe UI", sans-serif'; c.textAlign = 'right'; c.lineWidth = 1;
      [0, 5, 10, 15, 20].forEach(function(v){ c.beginPath(); c.moveTo(L, sy(v)); c.lineTo(R, sy(v)); c.stroke(); c.fillText(String(v), L - 6, sy(v) + 4); });
      c.textAlign = 'center'; [-1, 0, 1, 2, 3].forEach(function(v){ c.beginPath(); c.moveTo(sx(v), T); c.lineTo(sx(v), B); c.stroke(); c.fillText(String(v), sx(v), B + 14); });
      c.fillStyle = C.char; c.font = '600 12px Aptos, "Segoe UI", sans-serif'; c.fillText('log10 CLint (mL/min/kg)', (L + R) / 2, B + 32); c.save(); c.translate(14, (T + B) / 2); c.rotate(-Math.PI / 2); c.fillText('hepatic blood clearance CLh (mL/min/kg)', 0, 0); c.restore();
      /* histogram of the quantified compounds along the bottom of the plot */
      var nb = 24, lo = FLOOR, hi = Math.max.apply(null, meas), bw = (hi - lo) / nb, cnt = new Array(nb).fill(0), mx = 1; meas.forEach(function(v){ var k = Math.min(nb - 1, Math.floor((v - lo) / bw)); cnt[k]++; }); mx = Math.max.apply(null, cnt);
      cnt.forEach(function(n, k){ var hh = n / mx * 70; c.fillStyle = 'rgba(111,129,150,.55)'; c.fillRect(sx(lo + k * bw) + 1, B - hh, sx(lo + (k + 1) * bw) - sx(lo + k * bw) - 2, hh); });
      var bx = sx(FLOOR - 2 * bw), bwid = sx(FLOOR) - bx; c.fillStyle = 'rgba(225,36,42,.28)'; c.fillRect(bx, B - 70, bwid, 70); c.strokeStyle = C.red; c.setLineDash([4, 3]); c.strokeRect(bx, B - 70, bwid, 70); c.setLineDash([]);
      c.fillStyle = C.dred; c.font = '600 11px Aptos, "Segoe UI", sans-serif'; c.textAlign = 'center'; c.fillText('below floor', bx + bwid / 2, B - 76); c.fillText(nc.toLocaleString() + ' (' + Math.round(100 * nc / ds.N) + '%)', bx + bwid / 2, B - 88);
      c.fillStyle = C.mid; c.font = '11px Aptos, "Segoe UI", sans-serif'; c.textAlign = 'left'; c.fillText('Bars: quantified compounds. The dashed bar is the count at the assay floor; its height is not to scale.', L + 4, B + 48);
      /* ceiling and curve */
      c.strokeStyle = C.char; c.setLineDash([6, 4]); c.beginPath(); c.moveTo(L, sy(QH)); c.lineTo(R, sy(QH)); c.stroke(); c.setLineDash([]); c.fillStyle = C.char; c.font = '600 11px Aptos, "Segoe UI", sans-serif'; c.textAlign = 'left'; c.fillText('Hepatic blood flow Q = ' + QH + ' mL/min/kg', L + 6, sy(QH) - 5);
      c.strokeStyle = C.red; c.lineWidth = 3; c.beginPath(); for (var i = 0; i <= 200; i++) { var xv = X0 + (X1 - X0) * i / 200, h = PKE.wellStirred({ clint: Math.pow(10, xv), qh: QH, fu: fu, bp: 1 }).clh; i ? c.lineTo(sx(xv), sy(h)) : c.moveTo(sx(xv), sy(h)); } c.stroke();
      var r = PKE.fromHLM(x, { fup: fu, qhPerKg: QH }); c.strokeStyle = C.dred; c.lineWidth = 1.4; c.setLineDash([3, 3]); c.beginPath(); c.moveTo(sx(x), T); c.lineTo(sx(x), B); c.stroke(); c.setLineDash([]); c.fillStyle = C.dred; c.beginPath(); c.arc(sx(x), sy(r.clhBloodMLminKg), 5.5, 0, 6.3); c.fill();
      /* readouts */
      var all = mapAll(fu), share = function(th){ return Math.round(100 * all.filter(function(q){ return q.extraction > th; }).length / all.length); }, rf = PKE.fromHLM(FLOOR, { fup: fu, qhPerKg: QH });
      var up = PKE.fromHLM(x + Math.log10(2), { fup: fu, qhPerKg: QH }).clhBloodMLminKg / r.clhBloodMLminKg - 1, dn = PKE.fromHLM(x - Math.log10(2), { fup: fu, qhPerKg: QH }).clhBloodMLminKg / r.clhBloodMLminKg - 1;
var rw = function(l, v){ return '<div style="display:flex;justify-content:space-between;gap:8px;padding:2px 0;border-bottom:1px solid ' + C.light + '"><span>' + l + '</span><b>' + v + '</b></div>'; };
      qs('.ro', pa).innerHTML = '<div class="ol" style="margin-top:4px">At the selected point</div>' + rw('CLint', pkaNice(r.clintMLminKg) + ' mL/min/kg') + rw('CLh', pkaNice(r.clhBloodMLminKg) + ' mL/min/kg') + rw('CLh, 70 kg adult', pkaNice(r.clhBloodLh) + ' L/h') + rw('Extraction ratio', r.extraction.toFixed(2)) + rw('Sensitivity d ln CLh / d ln CLint', (1 - r.extraction).toFixed(2)) +
        '<p style="margin:8px 0 0">A 2-fold increase in CLint changes CLh by ' + (up >= 0 ? '+' : '') + (100 * up).toFixed(0) + '%. A 2-fold decrease changes it by ' + (100 * dn).toFixed(0) + '%.</p>' +
        '<div class="ol" style="margin-top:12px">Across the data set (fu = ' + fu.toFixed(2) + ')</div><p style="margin:2px 0">' + share(0.5) + '% of the quantified compounds have an extraction ratio above 0.5, and ' + share(0.8) + '% above 0.8.</p><p style="margin:2px 0">The ' + Math.round(100 * nc / ds.N) + '% at the assay floor have CLint of at most ' + pkaNice(rf.clintMLminKg) + ' mL/min/kg, so CLh of at most ' + pkaNice(rf.clhBloodMLminKg) + ' mL/min/kg. Their true values are lower by an unknown amount.</p>' +
        '<p style="margin:8px 0 0;color:' + C.mid + ';font-size:11.5px">Q = 20.7 mL/min/kg: Davies and Morris 1993. Blood to plasma ratio is taken as 1. Fraction unbound is an assumption.</p>';
    }
    async function render(){
      api.busy('Loading the HLM data'); ds = await loadDataset('HLM'); api.idle(); meas = []; nc = 0; for (var i = 0; i < ds.N; i++) { if (ds.cens && ds.cens[i]) nc++; else meas.push(ds.Y[i]); }
      var sorted = meas.slice().sort(function(a, b){ return a - b; }); if (!api._set) { api._set = true; pkaSet('pk2x', +sorted[Math.floor(sorted.length / 2)].toFixed(2), null); } draw();
    }
    ['pk2x', 'pk2fu'].forEach(function(id){ $(id).addEventListener('input', draw); });
    async function play(tok){
      await render(); pkaSet('pk2fu', 1, null); pkaSet('pk2x', 0, draw); api.cap('At low CLint, CLh is proportional to CLint. A 2-fold error in CLint is a 2-fold error in CLh.'); await api.sleep(3200);
      await api.anim(3200, function(t){ pkaSet('pk2x', +(0 + 3.4 * t).toFixed(2), draw); }); api.cap('At high CLint, CLh approaches hepatic blood flow and becomes insensitive to CLint.'); await api.sleep(3200);
      pkaSet('pk2fu', 0.1, null); pkaSet('pk2x', 1.5, draw); api.cap('A lower fraction unbound moves the curve to the right: the same CLint gives a lower CLh.'); await api.sleep(3200); api.cap(null);
    }
    return { render: render, play: play };
  } });
GUIDE.steps['pk/In vitro to in vivo clearance'] = { notice: 'The curve flattens at hepatic blood flow.', q: 'Why does a prediction error in CLint matter less for a high-extraction compound?', a: 'CLh approaches hepatic blood flow, so CLh changes little when CLint changes. For a low-extraction compound CLh is proportional to CLint.' };

/* ---------- 3. Where ML enters a PBPK workflow ---------- */
LESSONS.pk.add({ level: 'beginner', short: 'ML in the PBPK workflow', title: 'Where ML enters a PBPK workflow',
  change: 'Select a stage of the workflow.',
  watch: 'A PBPK workflow converts a structure to a decision through a chain of stages. Machine learning supplies the early stages: the predicted in vitro and physicochemical endpoints that become model parameters. Each stage carries its own uncertainty, and the uncertainty of the early stages propagates to the simulated exposure. Select a stage to see which model family in this course supplies it, how large its error is, and how it typically fails. The numbers for the HLM endpoint are computed from the course data.',
  build: function(api){
    var NODES = [
      { k: 'Structure', course: 'Input to the workflow.', unc: 'No model uncertainty. Ambiguity in the input propagates: salt form, protonation state, tautomer and stereochemistry.', fail: 'A structure that differs from the tested material, for example a salt or an undefined stereocentre, gives predictions for a different compound.' },
      { k: 'Descriptors and fingerprints', course: 'Lesson 01, Molecular representations and Descriptor contributions.', unc: 'Deterministic, but lossy: software differs in how it computes logP and TPSA, and fingerprints discard most three-dimensional information.', fail: 'Descriptors from a different toolkit or version than the training set shift the input distribution.' },
      { k: 'ML predictions', course: 'HLM CLint and logD: Lesson 01 (Stacking a logD model) and the model families in lessons 02 to 05 (trees, networks, pre-trained models). Fraction unbound, permeability and solubility: no data set in this course.', unc: null, fail: 'Compounds outside the applicability domain, and values at the assay floor (31% of the HLM compounds) that a regression model treats as measurements.' },
      { k: 'PBPK inputs', course: 'Tab 10, In vitro to in vivo clearance. Tab 11, Predicted parameters as PBPK inputs.', unc: null, fail: 'Placeholder defaults, such as blood to plasma ratio 1 and fraction unbound 1, left in place when no prediction exists.' },
      { k: 'Simulated profile', course: 'Tab 10, PBPK model structure. Tab 11, Surrogate models.', unc: 'Structural uncertainty (the whole-body model in this course is perfusion-limited and linear, with no transporters or permeability limitation) plus the propagated input error.', fail: 'A model structure that omits a process that matters for the compound, for example active transport or saturable clearance.' },
      { k: 'MIDD decision', course: 'Tab 10, MIDD decisions and ICH M15. Tab 11, Credibility of ML components.', unc: 'Model risk combines model influence and the consequence of a wrong decision (ICH M15, Section 2.1.5).', fail: 'A prediction used with a higher model influence than its validation supports.' }
    ], sel = 2, live = null, ref = null;
    var fp = api.el('pn', 'left:28px;top:20px;width:904px;height:150px;padding-top:10px', '<h4>Workflow</h4><div class="wf" style="display:flex;align-items:center;gap:6px;flex-wrap:nowrap"></div>');
    var dp = api.el('pn', 'left:28px;top:182px;width:904px;height:338px;overflow:auto', '<div class="dt" style="font-size:13px;line-height:1.5;color:' + C.char + '"></div>');
    function field(l, html){ return '<div style="margin-bottom:12px"><div class="ol">' + l + '</div><div>' + html + '</div></div>'; }
    function draw(){
      qs('.wf', fp).innerHTML = NODES.map(function(n, i){ return '<button type="button" data-n="' + i + '" style="flex:1;min-width:0;border:1.5px solid ' + (i === sel ? C.red : C.silver) + ';background:' + (i === sel ? '#FBE9E9' : '#fff') + ';border-radius:6px;padding:12px 6px;font:inherit;font-size:13px;font-weight:' + (i === sel ? 700 : 500) + ';color:' + C.char + ';cursor:pointer">' + esc(n.k) + '</button>' + (i < NODES.length - 1 ? '<span style="color:' + C.mid + ';font-size:18px">&rarr;</span>' : ''); }).join('');
      qsa('[data-n]', fp).forEach(function(b){ b.addEventListener('click', function(){ sel = +b.dataset.n; draw(); }); });
      var n = NODES[sel], unc = n.unc;
      if (n.k === 'ML predictions') unc = live ? 'HLM CLint, test compounds (scaffold split), log10 units. Boosted trees on 8 descriptors fitted in the browser: RMSE <b>' + live.rmse.toFixed(2) + '</b>, R2 <b>' + live.r2.toFixed(2) + '</b>. ' + (ref ? 'Saved reference runs with fingerprints and descriptors: LightGBM with predicted logD, RMSE <b>' + ref.lgbm.toFixed(2) + '</b>; Chemprop, RMSE <b>' + ref.cp.toFixed(2) + '</b>. ' : '') + 'An RMSE of ' + live.rmse.toFixed(2) + ' log10 units is a typical fold error of <b>' + Math.pow(10, live.rmse).toFixed(1) + '</b> in CLint.' : 'Fitting the course model.';
      if (n.k === 'PBPK inputs') unc = live ? 'Prediction error of each input, plus the error of the conversion itself. For CLint, an RMSE of ' + live.rmse.toFixed(2) + ' log10 units is a typical fold error of ' + Math.pow(10, live.rmse).toFixed(1) + '. The effect on CLh depends on the extraction ratio (previous step): it is largest for low-extraction compounds.' : 'Fitting the course model.';
      qs('.dt', dp).innerHTML = '<div style="font-size:18px;font-weight:700;margin-bottom:10px">' + esc(n.k) + '</div>' + field('Supplied by', esc(n.course)) + field('Uncertainty', unc) + field('Typical failure', esc(n.fail));
    }
    async function render(){
      draw(); if (!live) {
        api.busy('Fitting the course model on the HLM data'); await tick(); var ds = await loadDataset('HLM'), m = fitBoost(ds.BASE.slice(), ds.TRAIN, 150, 0.08, 4, 0, ds);
        live = { rmse: rmse(ds.TEST, m.F, ds.Y), r2: r2(ds.TEST, m.F, ds.Y) }; var ex = await loadExtras(), mt = ex.meta || {}; ref = (mt['LightGBM + predicted logD'] && mt['Chemprop (D-MPNN)']) ? { lgbm: mt['LightGBM + predicted logD'].RMSE, cp: mt['Chemprop (D-MPNN)'].RMSE } : null; api.idle(); draw();
      }
    }
    async function play(tok){ await render(); for (var k = 0; k < NODES.length; k++) { api.checkTok(tok); sel = k; draw(); api.cap(NODES[k].k + '.'); await api.sleep(2600); } sel = 2; draw(); api.cap(null); }
    return { render: render, play: play };
  } });
GUIDE.steps['pk/ML in the PBPK workflow'] = { notice: 'The error of the early stages propagates to the simulated exposure.', q: 'What does an RMSE of 0.56 log10 units mean for a predicted CLint?', a: 'A typical fold error of 10 to the power 0.56, which is 3.6, in CLint.' };

/* ---------- 4. MIDD decisions and ICH M15 ---------- */
LESSONS.pk.add({ level: 'beginner', short: 'MIDD decisions', title: 'MIDD decisions and ICH M15',
  change: 'Select a decision. Rate the model influence and the consequence of a wrong decision.',
  watch: 'ICH M15 defines model-informed drug development as the use of computational modelling and simulation that can include and integrate nonclinical data, clinical data, prior information and knowledge to generate evidence for decisions (Section 1.2). The guideline names population PK and PD, PBPK and biopharmaceutics, exposure-response, model-based meta-analysis, quantitative systems pharmacology and toxicology, agent-based models, disease progression models, and artificial intelligence and machine learning among the methods. It does not prescribe methods. It sets a risk-based framework for planning, evaluating and documenting the evidence: a question of interest, a context of use, model influence, the consequence of a wrong decision, and the model risk that follows from the last two (Section 2.1).',
  build: function(api){
    var DEC = [
      { k: 'First-in-human dose selection', q: 'Which starting dose gives a predicted human exposure below the exposure at the nonclinical no-observed-adverse-effect level?', m: ['PBPK', 'Population PK'] },
      { k: 'Drug-drug interaction assessment', q: 'Does co-administration with an inhibitor increase exposure beyond a stated threshold?', m: ['PBPK'] },
      { k: 'Dose selection', q: 'Which dose and regimen reach the target exposure in most patients?', m: ['Population PK', 'Exposure-response', 'QSP'] },
      { k: 'Special populations', q: 'Is a dose adjustment needed for organ impairment or for a paediatric age group?', m: ['PBPK', 'Population PK'] },
      { k: 'Trial design', q: 'Which sample size and sampling times distinguish the doses?', m: ['Population PK', 'Exposure-response'] }
    ], TYPES = ['PBPK', 'Population PK', 'Exposure-response', 'QSP'], sel = 0, infl = 0, cons = 0, LV = ['Low', 'Medium', 'High'];
    var lp = api.el('pn', 'left:28px;top:20px;width:300px;height:500px;overflow:auto', '<h4>Decision</h4><div class="dl"></div><div class="ol" style="margin-top:16px">Methods named in ICH M15, Section 1.2</div><div style="font-size:11.5px;line-height:1.45;color:' + C.char + '">Population pharmacokinetics and pharmacodynamics; physiologically based pharmacokinetics and biopharmaceutics; exposure-response; model-based meta-analysis; quantitative systems pharmacology and toxicology; agent-based models; disease progression models; artificial intelligence and machine learning.</div><p style="margin:14px 0 0;font-size:11.5px;line-height:1.5;color:' + C.mid + '">FDA MIDD Paired Meeting Program: a pilot from 2018 to 2022, continued under PDUFA VII (FDA, frequently asked questions).</p>');
    var mp = api.el('pn', 'left:346px;top:20px;width:586px;height:196px;overflow:auto', '<div class="ds" style="font-size:13px;line-height:1.5;color:' + C.char + '"></div>');
    var rp = api.el('pn', 'left:346px;top:228px;width:586px;height:292px;overflow:auto', '<h4>Model risk under ICH M15, Section 2.1.5</h4><div class="rk" style="font-size:13px;line-height:1.5;color:' + C.char + '"></div>');
    function risk(){
      if (infl === 0 && cons === 0) return ['Low', 'Both ratings are low, so the model risk is low.'];
      if (infl === 2 && cons === 2) return ['High', 'Both ratings are high, so the model risk is high.'];
      if (infl !== cons) return ['Not fixed', 'The ratings differ. M15 states that the model risk rating may be driven by the more influential of the two items, and that this consideration should be captured in the justification.'];
      return ['Not fixed', 'M15 states the outcome only when both ratings are low or both are high. The rating needs a justification.'];
    }
    function draw(){
      qs('.dl', lp).innerHTML = DEC.map(function(d, i){ return '<button type="button" data-d="' + i + '" style="display:block;width:100%;text-align:left;margin:0 0 5px;border:1.5px solid ' + (i === sel ? C.red : C.silver) + ';background:' + (i === sel ? '#FBE9E9' : '#fff') + ';border-radius:6px;padding:6px 11px;font:inherit;font-size:13px;font-weight:' + (i === sel ? 700 : 500) + ';color:' + C.char + ';cursor:pointer">' + esc(d.k) + '</button>'; }).join('');
      qsa('[data-d]', lp).forEach(function(b){ b.addEventListener('click', function(){ sel = +b.dataset.d; draw(); }); });
      var d = DEC[sel];
      qs('.ds', mp).innerHTML = '<div style="font-size:16px;font-weight:700;margin-bottom:8px">' + esc(d.k) + '</div><div class="ol">Example question of interest (illustrative)</div><p style="margin:2px 0 10px">' + esc(d.q) + '</p><div class="ol">Typical model types</div><div style="margin:4px 0 8px">' + TYPES.map(function(t){ var on = d.m.indexOf(t) >= 0; return '<span style="display:inline-block;margin:0 6px 6px 0;padding:3px 10px;border-radius:999px;border:1.5px solid ' + (on ? C.red : C.silver) + ';background:' + (on ? '#FBE9E9' : '#fff') + ';font-size:12px;font-weight:' + (on ? 700 : 400) + ';color:' + (on ? C.dred : C.mid) + '">' + t + '</span>'; }).join('') + '</div><p style="margin:0;font-size:11.5px;color:' + C.mid + '">The question and the typical model types are common practice, not part of ICH M15.</p>';
      var rs = risk();
      qs('.rk', rp).innerHTML = '<div style="display:grid;grid-template-columns:190px 1fr;gap:6px 12px;align-items:center;margin-bottom:8px"><div><b>Model influence</b></div><div>' + inSeg(LV, infl, 'data-i') + '</div><div><b>Consequence of wrong decision</b></div><div>' + inSeg(LV, cons, 'data-c') + '</div></div><div style="border:1.5px solid ' + C.silver + ';border-radius:6px;padding:10px 12px;background:' + C.light + '"><span class="ol">Model risk</span> <b style="color:' + C.red + ';font-size:17px;margin-left:6px">' + rs[0] + '</b><div style="margin-top:4px">' + rs[1] + '</div></div><p style="margin:8px 0 0;font-size:12px">Model influence is the weight of the model outcomes among the evidence. Consequence of wrong decision is the potential harm or lack of efficacy from an incorrect decision. Model evaluation (Section 3) should be commensurate with the model risk. For an artificial intelligence or machine learning model, Section 3 names overfitting as an issue to consider.</p>';
      qsa('[data-i] button', rp).forEach(function(b){ b.addEventListener('click', function(){ infl = +b.dataset.v; draw(); }); });
      qsa('[data-c] button', rp).forEach(function(b){ b.addEventListener('click', function(){ cons = +b.dataset.v; draw(); }); });
    }
    async function play(tok){
      sel = 0; infl = 0; cons = 0; draw(); api.cap('Each decision is a question of interest. The model types listed are those commonly applied to it.'); await api.sleep(3200);
      infl = 2; cons = 2; draw(); api.cap('High model influence and high consequence of a wrong decision give high model risk.'); await api.sleep(3200);
      infl = 0; cons = 2; draw(); api.cap('Differing ratings: the model risk may be driven by the more influential item, and the justification records it.'); await api.sleep(3400); api.cap(null);
    }
    return { render: draw, play: play };
  } });
GUIDE.steps['pk/MIDD decisions'] = { notice: 'Model risk follows from model influence and the consequence of a wrong decision.', q: 'What determines the required extent of model evaluation under ICH M15?', a: 'The model risk, which combines model influence and the consequence of a wrong decision. Evaluation should be commensurate with it.' };
