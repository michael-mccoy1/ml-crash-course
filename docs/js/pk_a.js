"use strict";
/* Lesson 10, Beginner steps. PK is treated as the interface between machine learning predictions and decisions.
   Sources of every fixed number are given beside it. */

function pkaSet(id, v, draw){ var el = $(id); if (!el) return; el.value = v; var o = $('o-' + id); if (o) o.textContent = v; if (draw) draw(); }
function pkaNice(v, d){ return v >= 100 ? v.toFixed(0) : (v >= 10 ? v.toFixed(1) : v.toFixed(d === undefined ? 2 : d)); }

/* ---------- 1. Concentration-time profile ---------- */
/* Clearance is taken from the HLM CLint of a test compound, once measured and once predicted by the boosted-trees model of lesson 02 fitted on the training compounds (mpReady in mlpk_a.js). V, ka and F are set by the sliders. */
function pkaCl(l10){ return PKE.fromHLM(l10, { bw: 70, fup: 1 }).clhPlasmaLh; }
function pkaTestStats(fup){ fup = fup === undefined ? 1 : fup;
  var ds = MP.ds, ids = ds.TEST.filter(function(i){ return !ds.cens[i]; }), s = 0, w = 0, bins = [{ n: 0, s: 0, w: 0 }, { n: 0, s: 0, w: 0 }, { n: 0, s: 0, w: 0 }];
  ids.forEach(function(i){
    var m = PKE.fromHLM(ds.Y[i], { bw: 70, fup: fup }), p = PKE.fromHLM(MP.F[i], { bw: 70, fup: fup }), e = Math.abs(Math.log10(p.clhPlasmaLh / m.clhPlasmaLh)), b = bins[m.extraction < 0.3 ? 0 : (m.extraction < 0.7 ? 1 : 2)];
    s += e; if (e <= Math.log10(2)) w++; b.n++; b.s += e; if (e <= Math.log10(2)) b.w++;
  });
  return { n: ids.length, gmfe: Math.pow(10, s / ids.length), within: w / ids.length, bins: bins.map(function(b){ return { n: b.n, gmfe: b.n ? Math.pow(10, b.s / b.n) : NaN, within: b.n ? b.w / b.n : NaN }; }) };
}
LESSONS.pk.add({ level: 'beginner', short: 'Concentration-time profile', title: 'Concentration-time profile',
  change: 'Choose a test compound. Set the dose, V, ka and F, and switch the route and the axis.',
  watch: 'A one-compartment model reduces a concentration-time profile to four parameters: clearance (CL), volume of distribution (V), absorption rate constant (ka) and bioavailability (F). In this step CL comes from the HLM intrinsic clearance of a test compound, which a boosted-trees model fitted on the training compounds predicts. The same compound is simulated with its measured CLint and with the predicted CLint, converted to hepatic plasma clearance with the well-stirred model (fraction unbound 1, blood to plasma ratio 1, hepatic blood flow 20.7 mL/min/kg, 70 kg). The two curves differ only in CL. V, ka and F are set by the sliders, since the course has no data to predict them. The AUC fold error is the clearance from the measured CLint divided by the clearance from the predicted CLint, or its inverse; it does not depend on V, ka or F.',
  build: function(api){
    var cur = -1, st = null, route = 0, logAxis = false, busy = false;
    var pa = api.el('pn', 'left:28px;top:20px;width:270px;height:500px;overflow:auto', '<h4>Set parameters</h4>' + optRange('pk1dose', 'Dose (mg)', 10, 500, 10, 100) + optRange('pk1v', 'Volume V (L)', 10, 300, 5, 100) + optRange('pk1ka', 'Absorption rate ka (1/h)', 0.2, 5, 0.1, 1) + optRange('pk1f', 'Bioavailability F', 0.1, 1, 0.05, 0.8) + '<div class="og"><div class="ol">Route</div>' + inSeg(['IV bolus', 'Oral'], 0, 'data-route') + '</div><div class="og"><div class="ol">Concentration axis</div>' + inSeg(['Linear', 'Log'], 0, 'data-axis') + '</div>');
    var cp = api.el('pn', 'left:316px;top:20px;width:616px;height:290px', '<h4 class="ct">Plasma concentration against time</h4><canvas width="586" height="244"></canvas>');
    var rp = api.el('pn', 'left:316px;top:322px;width:616px;height:198px;overflow:auto', '<div class="rd" style="font-size:12.5px;line-height:1.45;color:' + C.char + '"></div>');
    var ctl = api.addControl('Test compound <select class="sel" style="width:250px"></select> <button type="button" class="btn sm ghost" data-next>Next compound</button>');
    var sel = qs('.sel', ctl);
    function vals(){ return { dose: +$('pk1dose').value, V: +$('pk1v').value, ka: +$('pk1ka').value, F: +$('pk1f').value }; }
    function model(CL){ var p = vals(); return PKE.oneComp({ dose: p.dose, V: p.V, CL: CL, ka: p.ka, F: p.F, route: route ? 'oral' : 'iv' }); }
    function draw(){
      if (!MP.ds || cur < 0) return; var i = MP.list[cur].i, lm = MP.ds.Y[i], lp = MP.F[i], clM = pkaCl(lm), clP = pkaCl(lp), mM = model(clM), mP = model(clP), sM = mM.summary, sP = mP.summary;
      var tEnd = Math.min(168, Math.max(8, Math.ceil(5 * Math.max(sM.thalf, sP.thalf) / 2) * 2)), n = 240, k, ptsM = [], ptsP = [], cmin = Math.min(sM.cmax, sP.cmax) / 100;
      for (k = 0; k <= n; k++) { var t = tEnd * k / n, a = mM.conc(t), b = mP.conc(t); ptsM.push([t, logAxis ? Math.log10(Math.max(a, cmin)) : a]); ptsP.push([t, logAxis ? Math.log10(Math.max(b, cmin)) : b]); }
      var o = { xl: 'time (h)', yl: logAxis ? 'plasma concentration (mg/L, log scale)' : 'plasma concentration (mg/L)', x0: 0, x1: tEnd };
      if (logAxis) { var top = Math.max(sM.cmax, sP.cmax), lo = Math.floor(Math.log10(cmin)), hi = Math.ceil(Math.log10(top)), tk = []; for (var e = lo; e <= hi; e++) tk.push({ v: e, t: e >= 0 ? String(Math.pow(10, e)) : Math.pow(10, e).toFixed(-e) }); o.y0 = lo; o.y1 = hi; o.yticks = tk; } else { o.y0 = 0; }
      qs('.ct', cp).textContent = 'Plasma concentration, compound ' + (i + 1);
      lineChart(qs('canvas', cp), [{ pts: ptsM, color: C.char, w: 2.6, nodot: true, label: 'CL from measured CLint' }, { pts: ptsP, color: C.red, w: 2.6, nodot: true, label: 'CL from predicted CLint' }], o);
      var fe = clM / clP, ff = fe >= 1 ? fe : 1 / fe, row = function(l, a, b){ return '<tr style="border-top:1px solid ' + C.silver + '"><td style="padding:2px 4px">' + l + '</td><td style="text-align:right;padding:2px 4px">' + a + '</td><td style="text-align:right;padding:2px 4px;color:' + C.red + ';font-weight:700">' + b + '</td></tr>'; };
      qs('.rd', rp).innerHTML = '<div style="display:flex;gap:18px"><table style="border-collapse:collapse;flex:0 0 330px"><tr style="font-weight:700;font-size:11px;color:' + C.mid + '"><td></td><td style="text-align:right;padding:2px 4px">Measured</td><td style="text-align:right;padding:2px 4px">Predicted</td></tr>' +
        row('log10 CLint (mL/min/kg)', lm.toFixed(2), lp.toFixed(2)) + row('CL (L/h)', pkaNice(clM), pkaNice(clP)) + row('AUC (mg h/L)', pkaNice(sM.auc), pkaNice(sP.auc)) + row('Cmax (mg/L)', pkaNice(sM.cmax), pkaNice(sP.cmax)) + row('Half-life (h)', pkaNice(sM.thalf), pkaNice(sP.thalf)) + '</table>' +
        '<div style="flex:1"><div class="ol">This compound</div><b>AUC fold error ' + ff.toFixed(2) + '</b> (' + (clP < clM ? 'over' : 'under') + '-predicted), from an error of ' + (lp - lm >= 0 ? '+' : '') + (lp - lm).toFixed(2) + ' in log10 CLint.<div class="ol" style="margin-top:8px">All ' + st.n + ' quantified test compounds</div>Geometric mean AUC fold error <b>' + st.gmfe.toFixed(2) + '</b>. <b>' + Math.round(100 * st.within) + '%</b> within 2-fold. Test RMSE of log10 CLint ' + MP.rmseQ.toFixed(2) + '.</div></div>';
    }
    async function pick(k){ cur = k; sel.value = String(k); draw(); }
    async function render(){
      await mpReady(api); st = st || pkaTestStats();
      if (cur < 0) { var ae = MP.list.map(function(e, k){ return [Math.abs(MP.F[e.i] - MP.ds.Y[e.i]), k]; }).sort(function(a, b){ return a[0] - b[0]; }); cur = ae[Math.floor(ae.length / 2)][1]; }
      if (!sel.options.length) MP.list.forEach(function(e, k){ var o = document.createElement('option'); o.value = k; o.textContent = 'Compound ' + (e.i + 1) + ', measured log10 CLint ' + MP.ds.Y[e.i].toFixed(2); sel.appendChild(o); });
      sel.value = String(cur); draw();
    }
    sel.addEventListener('change', function(){ pick(+sel.value); });
    qs('[data-next]', ctl).addEventListener('click', function(){ if (MP.list) pick((cur + 1) % MP.list.length); });
    ['pk1dose', 'pk1v', 'pk1ka', 'pk1f'].forEach(function(id){ $(id).addEventListener('input', draw); });
    qsa('[data-route] button', pa).forEach(function(b){ b.addEventListener('click', function(){ route = +b.dataset.v; qsa('[data-route] button', pa).forEach(function(x){ x.classList.toggle('on', x === b); }); draw(); }); });
    qsa('[data-axis] button', pa).forEach(function(b){ b.addEventListener('click', function(){ logAxis = +b.dataset.v === 1; qsa('[data-axis] button', pa).forEach(function(x){ x.classList.toggle('on', x === b); }); draw(); }); });
    function pickSeg(seg, k){ qsa(seg + ' button', pa).forEach(function(x){ x.classList.toggle('on', +x.dataset.v === k); }); }
    async function play(tok){
      await render(); route = 0; logAxis = false; pickSeg('[data-route]', 0); pickSeg('[data-axis]', 0); draw();
      api.cap('A model fitted on the training compounds predicts log10 CLint for a test compound. Both values give a clearance, and each clearance gives a profile.'); await api.sleep(3600); api.checkTok(tok);
      logAxis = true; pickSeg('[data-axis]', 1); draw(); api.cap('On a log axis the terminal slope is -CL / V / 2.303, so the difference in slope is the difference in clearance.'); await api.sleep(3400); api.checkTok(tok);
      logAxis = false; route = 1; pickSeg('[data-axis]', 0); pickSeg('[data-route]', 1); draw(); api.cap('Oral dosing adds ka and F. The AUC fold error is unchanged, because it depends only on CL.'); await api.sleep(3400); api.checkTok(tok);
      pick((cur + 1) % MP.list.length); api.cap('Another test compound: the error in CLint differs, so the AUC fold error differs.'); await api.sleep(3200); api.cap(null);
    }
    return { render: render, play: play };
  } });
GUIDE.steps['pk/Concentration-time profile'] = { notice: 'The two curves differ only in the clearance taken from CLint.', q: 'Which parameters of this model does the HLM model predict, and which are set by hand?', a: 'The HLM model predicts CLint, from which CL is derived with the well-stirred model. V, ka and F are set by hand because the course has no data to predict them.' };

/* ---------- 2. In vitro to in vivo clearance ---------- */
LESSONS.pk.add({ level: 'beginner', short: 'In vitro to in vivo clearance', title: 'In vitro to in vivo clearance',
  change: 'Set log10 CLint and the fraction unbound in plasma. Read the clearance, the extraction ratio and the band set by the model error.',
  watch: 'The Biogen HLM value is log10 intrinsic clearance in mL/min/kg. This step treats it as an intrinsic clearance already scaled to body weight: it applies no further scaling for microsomal protein or liver weight and no correction for binding in the incubation, because the public sources do not state how the value was scaled. The well-stirred model converts intrinsic clearance to hepatic blood clearance, CLh = Q fu CLint / (Q + fu CLint), with hepatic blood flow Q. CLh cannot exceed Q, so the extraction ratio saturates at high CLint. The fraction unbound is an assumption of this step, because the data set contains no fraction unbound. The open circles are the test compounds: each is plotted at its measured CLint, with the CLh obtained from the CLint predicted by the boosted-trees model of lesson 02. The vertical distance of a circle from the curve is the effect of the prediction error on CLh. The shaded band is the selected CLint plus and minus the test RMSE of the model, and the table gives the fold error of CLh for the quantified test compounds, grouped by the extraction ratio of the measured value.',
  build: function(api){
    var QH = 20.7, FLOOR = 0.6757, ds = null, meas = [], nc = 0;
    var ca = api.el('pn', 'left:28px;top:20px;width:540px;height:500px', '<h4>Hepatic clearance against intrinsic clearance, with the HLM compounds</h4><canvas width="510" height="446"></canvas>');
    var pa = api.el('pn', 'left:582px;top:20px;width:350px;height:500px;overflow:auto', '<h4>Settings</h4>' + optRange('pk2x', 'log10 CLint (mL/min/kg)', -1, 3.5, 0.05, 1.2) + optRange('pk2fu', 'Fraction unbound in plasma (assumed)', 0.01, 1, 0.01, 1) + '<div class="ro" style="font-size:12.5px;line-height:1.5;color:' + C.char + '"></div>');
    function mapAll(fup){ return meas.map(function(v){ return PKE.fromHLM(v, { fup: fup, qhPerKg: QH }); }); }
    function draw(){
      if (!ds) return; var x = +$('pk2x').value, fu = +$('pk2fu').value, g = cv(qs('canvas', ca)), c = g.x, L = 58, R = g.w - 14, T = 16, B = g.h - 68, X0 = -1, X1 = 3.5, YM = 24;
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
      c.fillStyle = C.mid; c.font = '11px Aptos, "Segoe UI", sans-serif'; c.textAlign = 'left'; c.fillText('Bars: quantified compounds. Dashed bar: assay floor, height not to scale.', L + 4, B + 48); c.fillText('Circles: test compounds, CLh from predicted CLint. Shaded: plus or minus test RMSE.', L + 4, B + 62);
      var rs = MP.rmseQ, bx0 = sx(Math.max(X0, x - rs)), bx1 = sx(Math.min(X1, x + rs)); c.fillStyle = 'rgba(225,36,42,.10)'; c.fillRect(bx0, T, bx1 - bx0, B - T);
      MP.ds.TEST.forEach(function(i2){ if (MP.ds.cens[i2]) return; var yv = PKE.fromHLM(MP.F[i2], { fup: fu, qhPerKg: QH }).clhBloodMLminKg, xv2 = MP.ds.Y[i2]; c.strokeStyle = 'rgba(225,36,42,.55)'; c.lineWidth = 1; c.beginPath(); c.arc(sx(xv2), sy(yv), 2.4, 0, 6.3); c.stroke(); });
      /* ceiling and curve */
      c.strokeStyle = C.char; c.setLineDash([6, 4]); c.beginPath(); c.moveTo(L, sy(QH)); c.lineTo(R, sy(QH)); c.stroke(); c.setLineDash([]); c.fillStyle = C.char; c.font = '600 11px Aptos, "Segoe UI", sans-serif'; c.textAlign = 'left'; c.fillText('Hepatic blood flow Q = ' + QH + ' mL/min/kg', L + 6, sy(QH) - 5);
      c.strokeStyle = C.red; c.lineWidth = 3; c.beginPath(); for (var i = 0; i <= 200; i++) { var xv = X0 + (X1 - X0) * i / 200, h = PKE.wellStirred({ clint: Math.pow(10, xv), qh: QH, fu: fu, bp: 1 }).clh; i ? c.lineTo(sx(xv), sy(h)) : c.moveTo(sx(xv), sy(h)); } c.stroke();
      var r = PKE.fromHLM(x, { fup: fu, qhPerKg: QH }); c.strokeStyle = C.dred; c.lineWidth = 1.4; c.setLineDash([3, 3]); c.beginPath(); c.moveTo(sx(x), T); c.lineTo(sx(x), B); c.stroke(); c.setLineDash([]); c.fillStyle = C.dred; c.beginPath(); c.arc(sx(x), sy(r.clhBloodMLminKg), 5.5, 0, 6.3); c.fill();
      /* readouts */
      var ts = pkaTestStats(fu);
      var up = PKE.fromHLM(x + Math.log10(2), { fup: fu, qhPerKg: QH }).clhBloodMLminKg / r.clhBloodMLminKg - 1, dn = PKE.fromHLM(x - Math.log10(2), { fup: fu, qhPerKg: QH }).clhBloodMLminKg / r.clhBloodMLminKg - 1;
var rw = function(l, v){ return '<div style="display:flex;justify-content:space-between;gap:8px;padding:2px 0;border-bottom:1px solid ' + C.light + '"><span>' + l + '</span><b>' + v + '</b></div>'; };
      qs('.ro', pa).innerHTML = '<div class="ol" style="margin-top:4px">At the selected point</div>' + rw('CLint', pkaNice(r.clintMLminKg) + ' mL/min/kg') + rw('CLh', pkaNice(r.clhBloodMLminKg) + ' mL/min/kg') + rw('Extraction ratio', r.extraction.toFixed(2)) + 
        '<div class="ol" style="margin-top:12px">Model error at this CLint</div><p style="margin:2px 0">Test RMSE of log10 CLint ' + MP.rmseQ.toFixed(2) + '. CLint ' + pkaNice(Math.pow(10, x - MP.rmseQ)) + ' to ' + pkaNice(Math.pow(10, x + MP.rmseQ)) + ' gives CLh ' + pkaNice(PKE.fromHLM(x - MP.rmseQ, { fup: fu, qhPerKg: QH }).clhBloodMLminKg) + ' to ' + pkaNice(PKE.fromHLM(x + MP.rmseQ, { fup: fu, qhPerKg: QH }).clhBloodMLminKg) + ' mL/min/kg.</p>' +
        '<div class="ol" style="margin-top:10px">CLh fold error, ' + ts.n + ' test compounds</div><table style="width:100%;border-collapse:collapse;font-size:12px"><tr style="font-weight:700;font-size:11px;color:' + C.mid + '"><td style="padding:2px 4px">Extraction</td><td style="text-align:right;padding:2px 4px">n</td><td style="text-align:right;padding:2px 4px">GMFE</td><td style="text-align:right;padding:2px 4px">2-fold</td></tr>' + [['Below 0.3', 0], ['0.3 to 0.7', 1], ['0.7 and above', 2]].map(function(r2){ var b2 = ts.bins[r2[1]]; return '<tr style="border-top:1px solid ' + C.silver + '"><td style="padding:2px 4px">' + r2[0] + '</td><td style="text-align:right;padding:2px 4px">' + b2.n + '</td><td style="text-align:right;padding:2px 4px">' + (b2.n ? b2.gmfe.toFixed(2) : '') + '</td><td style="text-align:right;padding:2px 4px">' + (b2.n ? Math.round(100 * b2.within) + '%' : '') + '</td></tr>'; }).join('') + '<tr style="border-top:1px solid ' + C.silver + ';font-weight:700"><td style="padding:2px 4px">All</td><td style="text-align:right;padding:2px 4px">' + ts.n + '</td><td style="text-align:right;padding:2px 4px">' + ts.gmfe.toFixed(2) + '</td><td style="text-align:right;padding:2px 4px">' + Math.round(100 * ts.within) + '%</td></tr></table>' +
        '<p style="margin:6px 0 0;color:' + C.mid + ';font-size:11.5px">The ' + Math.round(100 * nc / ds.N) + '% of compounds at the assay floor are not in the table.</p>' +
        '<p style="margin:8px 0 0;color:' + C.mid + ';font-size:11.5px">Q = 20.7 mL/min/kg: Davies and Morris 1993. Blood to plasma ratio is taken as 1. Fraction unbound is an assumption.</p>';
    }
    async function render(){
      await mpReady(api); ds = MP.ds; meas = []; nc = 0; for (var i = 0; i < ds.N; i++) { if (ds.cens && ds.cens[i]) nc++; else meas.push(ds.Y[i]); }
      var sorted = meas.slice().sort(function(a, b){ return a - b; }); if (!api._set) { api._set = true; pkaSet('pk2x', +sorted[Math.floor(sorted.length / 2)].toFixed(2), null); } draw();
    }
    ['pk2x', 'pk2fu'].forEach(function(id){ $(id).addEventListener('input', draw); });
    async function play(tok){
      await render(); pkaSet('pk2fu', 1, null); pkaSet('pk2x', 0, draw); api.cap('At low CLint, CLh is proportional to CLint. A 2-fold error in CLint is a 2-fold error in CLh, and the circles scatter widely around the curve.'); await api.sleep(3400);
      await api.anim(3200, function(t){ pkaSet('pk2x', +(0 + 3.4 * t).toFixed(2), draw); }); api.cap('At high CLint, CLh approaches hepatic blood flow, the circles fall close to the curve and the band narrows in CLh.'); await api.sleep(3400);
      pkaSet('pk2fu', 0.1, null); pkaSet('pk2x', 1.5, draw); api.cap('A lower fraction unbound moves the curve to the right: the same CLint gives a lower CLh.'); await api.sleep(3200); api.cap(null);
    }
    return { render: render, play: play };
  } });
GUIDE.steps['pk/In vitro to in vivo clearance'] = { notice: 'The curve flattens at hepatic blood flow, so the same CLint error gives a smaller CLh error at high extraction.', q: 'Why does a prediction error in CLint matter less for a high-extraction compound?', a: 'CLh approaches hepatic blood flow, so CLh changes little when CLint changes. For a low-extraction compound CLh is proportional to CLint.' };

/* ---------- 3. Where ML enters a PBPK workflow ---------- */
LESSONS.pk.add({ level: 'beginner', short: 'ML in the PBPK workflow', title: 'Where ML enters a PBPK workflow',
  change: 'Select a stage of the workflow. Read the ML model, its test error and how it is validated.',
  watch: 'A PBPK workflow converts a structure to a decision through a chain of stages. Machine learning supplies the early stages: the predicted in vitro and physicochemical endpoints that become model parameters. Each stage carries its own uncertainty, and the uncertainty of the early stages propagates to the simulated exposure. Select a stage to see the machine learning model that supplies it, the size of its error, how that error is checked, and the typical failure. The numbers for the HLM endpoint are computed from the course data: the boosted-trees model is fitted in the page on the training compounds and scored on the scaffold-split test compounds.',
  build: function(api){
    var NODES = [
      { k: 'Structure', ml: 'None. The structure is the model input.', unc: 'No model error. Ambiguity in the input propagates: salt form, protonation state, tautomer and stereochemistry.', val: 'Compare the structure with the registered structure of the tested material.', fail: 'A structure that differs from the tested material, for example a salt or an undefined stereocentre, gives predictions for a different compound.' },
      { k: 'Descriptors and fingerprints', ml: 'The input features of the models in lessons 02 to 05: descriptors and Morgan fingerprints (lesson 01, Molecular representations).', unc: 'Deterministic, but lossy: software differs in how it computes logP and TPSA, and fingerprints discard most three-dimensional information.', val: 'Compute descriptors with the same toolkit and version as the training set.', fail: 'Descriptors from a different toolkit or version than the training set shift the input distribution.' },
      { k: 'ML predictions', ml: 'Boosted trees on descriptors (lesson 02), fitted on the HLM training compounds. Saved LightGBM and Chemprop runs are listed with it (lessons 02, 03 and 06). Fraction unbound, permeability and solubility: no data set in this course.', unc: null, val: 'RMSE and R2 on the scaffold-split test compounds, which were not used in fitting, together with an applicability-domain check (lesson 07).', fail: 'Compounds outside the applicability domain, and values at the assay floor (31% of the HLM compounds) that a regression model treats as measurements.' },
      { k: 'PBPK inputs', ml: 'The predicted log10 CLint becomes the intrinsic clearance input through IVIVE (tab 10, In vitro to in vivo clearance). Tab 11, Predicted parameters as PBPK inputs, runs the whole-body model on predicted and measured CLint.', unc: null, val: 'Predicted and measured CLint go through the same conversion, and the resulting CL or AUC is compared (tab 10, Concentration-time profile).', fail: 'Placeholder defaults, such as blood to plasma ratio 1 and fraction unbound 1, left in place when no prediction exists.' },
      { k: 'Simulated profile', ml: 'Optionally a surrogate: boosted trees or a network trained on whole-body model simulations (tab 11, Surrogate models).', unc: 'Structural error (the whole-body model in this course is perfusion-limited and linear, with no transporters or permeability limitation) plus the propagated input error. A surrogate adds its own approximation error.', val: 'Compare the surrogate with the whole-body model on held-out simulations, and include inputs outside the sampled range (tab 11, Surrogate models).', fail: 'A model structure that omits a process that affects the clearance or distribution of the compound, for example active transport or saturable clearance.' },
      { k: 'MIDD decision', ml: 'The ML component is one source of evidence in the model (tab 11, Credibility of ML components).', unc: 'Model risk combines model influence and the consequence of a wrong decision (ICH M15, Section 2.1.5).', val: 'The model risk sets the extent of the evaluation (ICH M15, Sections 2.1.5 and 3).', fail: 'A prediction used with a higher model influence than its validation supports.' }
    ], sel = 2, live = null, ref = null;
    var fp = api.el('pn', 'left:28px;top:20px;width:904px;height:150px;padding-top:10px', '<h4>Workflow</h4><div class="wf" style="display:flex;align-items:center;gap:6px;flex-wrap:nowrap"></div>');
    var dp = api.el('pn', 'left:28px;top:182px;width:904px;height:338px;overflow:auto', '<div class="dt" style="font-size:13px;line-height:1.5;color:' + C.char + '"></div>');
    function field(l, html){ return '<div style="margin-bottom:12px"><div class="ol">' + l + '</div><div>' + html + '</div></div>'; }
    function draw(){
      qs('.wf', fp).innerHTML = NODES.map(function(n, i){ return '<button type="button" data-n="' + i + '" style="flex:1;min-width:0;border:1.5px solid ' + (i === sel ? C.red : C.silver) + ';background:' + (i === sel ? '#FBE9E9' : '#fff') + ';border-radius:6px;padding:12px 6px;font:inherit;font-size:13px;font-weight:' + (i === sel ? 700 : 500) + ';color:' + C.char + ';cursor:pointer">' + esc(n.k) + '</button>' + (i < NODES.length - 1 ? '<span style="color:' + C.mid + ';font-size:18px">&rarr;</span>' : ''); }).join('');
      qsa('[data-n]', fp).forEach(function(b){ b.addEventListener('click', function(){ sel = +b.dataset.n; draw(); }); });
      var n = NODES[sel], unc = n.unc;
      if (n.k === 'ML predictions') unc = live ? 'HLM CLint, test compounds (scaffold split), log10 units. Boosted trees on 8 descriptors fitted in the browser: RMSE <b>' + live.rmse.toFixed(2) + '</b>, R2 <b>' + live.r2.toFixed(2) + '</b>. ' + (ref ? 'Saved reference runs with fingerprints and descriptors: LightGBM with predicted logD, RMSE <b>' + ref.lgbm.toFixed(2) + '</b>; Chemprop, RMSE <b>' + ref.cp.toFixed(2) + '</b>.' : '') : 'Fitting the course model.';
      if (n.k === 'PBPK inputs') unc = live ? 'An RMSE of ' + live.rmse.toFixed(2) + ' log10 units is a typical fold error of ' + Math.pow(10, live.rmse).toFixed(1) + ' in CLint. After conversion to hepatic clearance (fraction unbound 1), the geometric mean fold error is <b>' + live.gmfe.toFixed(2) + '</b> and <b>' + Math.round(100 * live.within) + '%</b> of the ' + live.n + ' quantified test compounds are within 2-fold. The effect on CLh is largest for low-extraction compounds (tab 10, In vitro to in vivo clearance).' : 'Fitting the course model.';
      qs('.dt', dp).innerHTML = '<div style="font-size:18px;font-weight:700;margin-bottom:10px">' + esc(n.k) + '</div>' + field('ML model', esc(n.ml)) + field('Test error', unc) + field('Validation', esc(n.val)) + field('Typical failure', esc(n.fail));
    }
    async function render(){
      draw(); if (!live) {
        await mpReady(api); var ds = MP.ds, ts = pkaTestStats(1); live = { rmse: MP.rmse, r2: r2(ds.TEST, MP.F, ds.Y), n: ts.n, gmfe: ts.gmfe, within: ts.within };
        var ex = await loadExtras(), mt = ex.meta || {}; ref = (mt['LightGBM + predicted logD'] && mt['Chemprop (D-MPNN)']) ? { lgbm: mt['LightGBM + predicted logD'].RMSE, cp: mt['Chemprop (D-MPNN)'].RMSE } : null; draw();
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
      { k: 'First-in-human dose selection', q: 'Which starting dose gives a predicted human exposure below the exposure at the nonclinical no-observed-adverse-effect level?', m: ['PBPK', 'Population PK'], ml: 'Before any in vivo data exist, a predicted CLint can be the only source of the clearance input. The ML prediction then carries a large share of the model influence, and the context of use names the training data and the test error.' },
      { k: 'Drug-drug interaction assessment', q: 'Does co-administration with an inhibitor increase exposure beyond a stated threshold?', m: ['PBPK'], ml: 'A predicted fm or inhibition constant enters the interaction ratio. The conclusion is reassessed across the error range of the prediction (tab 10, Drug-drug interaction models).' },
      { k: 'Dose selection', q: 'Which dose and regimen reach the target exposure in most patients?', m: ['Population PK', 'Exposure-response', 'QSP'], ml: 'An ML covariate model or a hybrid model supplies part of the clearance relation. The model influence is lower when clinical data constrain it (tab 11, Covariate screening).' },
      { k: 'Special populations', q: 'Is a dose adjustment needed for organ impairment or for a paediatric age group?', m: ['PBPK', 'Population PK'], ml: 'A model trained on adults does not cover the age range or impairment grade of the target group. The applicability domain is part of the context of use.' },
      { k: 'Trial design', q: 'Which sample size and sampling times distinguish the doses?', m: ['Population PK', 'Exposure-response'], ml: 'An ML surrogate can replace repeated simulations in the design search. It is validated against the full model on simulations not used for training (tab 11, Surrogate models).' }
    ], TYPES = ['PBPK', 'Population PK', 'Exposure-response', 'QSP'], sel = 0, infl = 0, cons = 0, LV = ['Low', 'Medium', 'High'];
    var lp = api.el('pn', 'left:28px;top:20px;width:300px;height:500px;overflow:auto', '<h4>Decision</h4><div class="dl"></div><div class="ol" style="margin-top:16px">Methods named in ICH M15, Section 1.2</div><div style="font-size:11.5px;line-height:1.45;color:' + C.char + '">Population pharmacokinetics and pharmacodynamics; physiologically based pharmacokinetics and biopharmaceutics; exposure-response; model-based meta-analysis; quantitative systems pharmacology and toxicology; agent-based models; disease progression models; artificial intelligence and machine learning.</div><p style="margin:14px 0 0;font-size:11.5px;line-height:1.5;color:' + C.mid + '">FDA MIDD Paired Meeting Program: a pilot from 2018 to 2022, continued under PDUFA VII (FDA, frequently asked questions).</p>');
    var mp = api.el('pn', 'left:346px;top:20px;width:586px;height:268px;overflow:auto', '<div class="ds" style="font-size:13px;line-height:1.5;color:' + C.char + '"></div>');
    var rp = api.el('pn', 'left:346px;top:300px;width:586px;height:220px;overflow:auto', '<h4>Model risk under ICH M15, Section 2.1.5</h4><div class="rk" style="font-size:13px;line-height:1.5;color:' + C.char + '"></div>');
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
      qs('.ds', mp).innerHTML = '<div style="font-size:16px;font-weight:700;margin-bottom:8px">' + esc(d.k) + '</div><div class="ol">Example question of interest (illustrative)</div><p style="margin:2px 0 10px">' + esc(d.q) + '</p><div class="ol">Typical model types</div><div style="margin:4px 0 8px">' + TYPES.map(function(t){ var on = d.m.indexOf(t) >= 0; return '<span style="display:inline-block;margin:0 6px 6px 0;padding:3px 10px;border-radius:999px;border:1.5px solid ' + (on ? C.red : C.silver) + ';background:' + (on ? '#FBE9E9' : '#fff') + ';font-size:12px;font-weight:' + (on ? 700 : 400) + ';color:' + (on ? C.dred : C.mid) + '">' + t + '</span>'; }).join('') + '</div><div class="ol">With an ML component (course interpretation)</div><p style="margin:2px 0 6px">' + esc(d.ml) + '</p><p style="margin:0;font-size:11.5px;color:' + C.mid + '">The question, the typical model types and the ML component example are common practice, not part of ICH M15.</p>';
      var rs = risk();
      qs('.rk', rp).innerHTML = '<div style="display:grid;grid-template-columns:190px 1fr;gap:6px 12px;align-items:center;margin-bottom:8px"><div><b>Model influence</b></div><div>' + inSeg(LV, infl, 'data-i') + '</div><div><b>Consequence of wrong decision</b></div><div>' + inSeg(LV, cons, 'data-c') + '</div></div><div style="border:1.5px solid ' + C.silver + ';border-radius:6px;padding:10px 12px;background:' + C.light + '"><span class="ol">Model risk</span> <b style="color:' + C.red + ';font-size:17px;margin-left:6px">' + rs[0] + '</b><div style="margin-top:4px">' + rs[1] + '</div></div><p style="margin:8px 0 0;font-size:12px">Model influence is the weight of the model outcomes among the evidence. Consequence of wrong decision is the potential harm or lack of efficacy from an incorrect decision. Model evaluation (Section 3) should be commensurate with the model risk. For an artificial intelligence or machine learning model, Section 3 names overfitting as an issue to consider. An ML component enters the context of use as a named model with its training data and test error. It enters the model influence rating through the share of the evidence that rests on the ML prediction.</p>';
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
