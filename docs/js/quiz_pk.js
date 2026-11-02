"use strict";
/* Quiz questions 76 onward: PK, PBPK, MIDD and machine learning in PK. Each question has a schematic that shows the setup and does not mark the answer.
   Numeric answers were computed with PKE (pkengine.js). ICH M15 wording is from the final guideline (Step 4, 29 January 2026), Section 2.1. */
(function(){
  var S = QZSV, C = S.C, B = QZ.bank;
  function cap(t, w, y){ return S.text(w / 2, y, t, { c: C.mid, i: true }); }
  function box(x, y, w, h, lines, o){ o = o || {}; var out = S.rect(x, y, w, h, { f: o.f || C.white, s: o.s || C.char, w: o.w || 1.4, d: o.d }), n = lines.length, i; for (i = 0; i < n; i++) out += S.text(x + w / 2, y + h / 2 + (i - (n - 1) / 2) * 15 + 4, lines[i], { b: i === 0 && o.b, s: o.s2 || 12, c: o.c }); return out; }
  function axes(x0, y0, w, h, xl, yl){ return S.line(x0, y0 + h, x0 + w, y0 + h) + S.line(x0, y0, x0, y0 + h) + S.text(x0 + w / 2, y0 + h + 26, xl, { s: 11.5 }) + '<g transform="translate(' + (x0 - 16) + ' ' + (y0 + h / 2) + ') rotate(-90)">' + S.text(0, 0, yl, { s: 11.5 }) + '</g>'; }
  function decayPath(x0, y0, w, h, ke, tmax, c0frac){ var d = '', i, n = 48; for (i = 0; i <= n; i++) { var t = tmax * i / n, c = Math.exp(-ke * t); d += (i ? ' L' : 'M') + (x0 + w * i / n).toFixed(1) + ' ' + (y0 + h - h * (c0frac || 0.92) * c).toFixed(1); } return d; }

  /* 76, 77, 80: the well-stirred liver, symbols only */
  QZFIG.liverDiagram = function(o){
    var W = 600, H = 210, g = '';
    g += S.rect(215, 40, 170, 100, { f: C.light, s: C.char, w: 2, r: 14 }) + S.text(300, 62, 'Liver (well-stirred)', { b: true });
    g += S.text(300, 84, 'unbound drug concentration', { s: 11 }) + S.text(300, 100, 'equal to the outflow', { s: 11 });
    g += S.text(300, 124, o.label || 'removal rate: fu times CLint', { s: 11.5, c: C.dred });
    g += S.arrow(60, 90, 213, 90, { w: 2.2 }) + S.text(138, 78, 'blood in, flow Q', { s: 11.5 });
    g += S.arrow(387, 90, 540, 90, { w: 2.2 }) + S.text(463, 78, 'blood out, flow Q', { s: 11.5 });
    g += S.arrow(300, 142, 300, 182, { c: C.red, w: 2 }) + S.text(300, 198, 'metabolised', { s: 11.5, c: C.dred });
    if (o.vals) g += S.text(138, 112, o.vals[0], { s: 11, c: C.mid }) + S.text(300, 20, o.vals[1], { s: 11.5, c: C.mid });
    return S.svg(W, H, g, 'A well-stirred liver compartment with blood inflow and outflow at flow Q and metabolic removal');
  };

  /* 78: one-compartment decay, parameters shown */
  QZFIG.decayParams = function(o){
    var W = 600, H = 222, g = '';
    g += box(30, 50, 150, 70, ['Central compartment', 'volume V = ' + o.v + ' L'], { f: C.light, b: true });
    g += S.arrow(182, 85, 262, 85, { c: C.red, w: 2 }) + S.text(222, 74, 'CL = ' + o.cl + ' L/h', { s: 11.5, c: C.dred });
    g += S.text(105, 40, 'IV bolus, ' + o.dose + ' mg', { s: 11.5 });
    g += axes(350, 30, 220, 130, 'time', 'plasma concentration') + S.path(decayPath(350, 30, 220, 130, 0.9, 4), { c: C.char, w: 2.2 });
    g += cap('First-order elimination from a single compartment.', W, 214);
    return S.svg(W, H, g, 'One compartment with volume and clearance and a declining plasma concentration over time');
  };

  /* 79: area under the curve, dose and clearance shown */
  QZFIG.aucArea = function(o){
    var W = 600, H = 222, g = '', x0 = 150, y0 = 26, w = 330, h = 140, d = decayPath(x0, y0, w, h, 0.8, 5);
    g += S.path(d + ' L' + (x0 + w) + ' ' + (y0 + h) + ' L' + x0 + ' ' + (y0 + h) + ' Z', { f: '#FBE9E9', c: 'none', w: 0 });
    g += axes(x0, y0, w, h, 'time', 'plasma concentration') + S.path(d, { c: C.char, w: 2.4 });
    g += S.text(x0 + 130, y0 + h - 30, 'AUC', { b: true, c: C.dred });
    g += S.text(70, 56, 'IV bolus', { s: 11.5 }) + S.text(70, 74, o.dose + ' mg', { b: true, s: 12 }) + S.text(70, 104, 'CL', { s: 11.5 }) + S.text(70, 122, o.cl + ' L/h', { b: true, s: 12 });
    g += cap('The shaded area is the AUC from zero to infinity.', W, 214);
    return S.svg(W, H, g, 'An IV bolus plasma concentration profile with the area under the curve shaded');
  };

  /* 81: repeated dosing at fixed interval */
  QZFIG.accumulationProfile = function(o){
    var W = 600, H = 226, g = '', x0 = 70, y0 = 26, w = 480, h = 140, ke = Math.LN2 / o.thalf, tEnd = o.tau * 5, d = '', i, n = 200, c = 0;
    for (i = 0; i <= n; i++) { var t = tEnd * i / n; c = 0; for (var k = 0; k * o.tau <= t; k++) c += Math.exp(-ke * (t - k * o.tau)); d += (i ? ' L' : 'M') + (x0 + w * i / n).toFixed(1) + ' ' + (y0 + h - h * 0.36 * c).toFixed(1); }
    g += axes(x0, y0, w, h, 'time (h)', 'plasma concentration') + S.path(d, { c: C.char, w: 2.2 });
    for (var j = 0; j < 5; j++) g += S.arrow(x0 + w * j / 5, y0 - 6, x0 + w * j / 5, y0 + 12, { c: C.red, w: 1.6 });
    g += S.text(x0 + w * 0.1, y0 + 26, 'doses every ' + o.tau + ' h', { s: 11.5, c: C.dred }) + S.text(x0 + w * 0.7, y0 + h - 8, 'half-life ' + o.thalf + ' h', { s: 11.5, c: C.mid });
    g += cap('Linear PK, intravenous bolus doses of equal size at a fixed interval.', W, 218);
    return S.svg(W, H, g, 'A plasma concentration profile for repeated doses at a fixed dosing interval');
  };

  /* 82, 83: perfusion-limited tissue */
  QZFIG.tissueFlow = function(o){
    var W = 600, H = 220, g = '';
    g += box(210, 50, 180, 90, ['Tissue', 'volume Vt'], { f: C.light, b: true });
    g += S.arrow(40, 80, 208, 80, { w: 2.2 }) + S.text(124, 68, 'arterial blood, flow Q', { s: 11.5 });
    g += S.arrow(392, 110, 560, 110, { w: 2.2 }) + S.text(476, 128, 'venous blood leaving the tissue', { s: 11.5 });
    if (o.mode === 'kp') {
      g += S.rect(430, 20, 22, 60, { f: C.bg, s: C.char }) + S.text(441, 96, 'plasma', { s: 11 }) + S.rect(470, 40, 22, 40, { f: C.white, s: C.char }) + S.text(481, 96, 'tissue', { s: 11 });
      g += S.text(461, 14, 'concentrations at equilibrium', { s: 11 });
      g += S.text(300, 170, 'perfusion-limited model, Kp from the compound and the tissue composition', { s: 11.5, c: C.mid });
    } else {
      g += S.text(300, 170, 'distribution in a perfusion-limited PBPK model', { s: 11.5, c: C.mid });
    }
    return S.svg(W, H, g, 'A tissue compartment supplied by arterial blood with venous blood leaving it');
  };

  /* 84: observed profile treated two ways, panels not labelled with the method names */
  QZFIG.ncaVsFit = function(){
    var W = 600, H = 230, g = '', i, pts = [[0.2, 0.55], [0.6, 0.8], [1.2, 0.72], [2.2, 0.5], [3.6, 0.3], [5.2, 0.17], [7, 0.1]];
    [0, 1].forEach(function(p){
      var x0 = 40 + p * 300, y0 = 44, w = 230, h = 120;
      g += S.text(x0 + w / 2, 24, p ? 'B: curve from a model structure' : 'A: segments between observations', { b: true, s: 11.5 }) + axes(x0, y0, w, h, 'time', 'concentration');
      var px = pts.map(function(q){ return [x0 + q[0] / 7.4 * w, y0 + h - q[1] * h]; });
      if (p) { var d = 'M' + (x0 + 2) + ' ' + (y0 + h); for (i = 0; i <= 40; i++) { var t = i / 40 * 7.4, c = 1.15 * (Math.exp(-0.45 * t) - Math.exp(-2.2 * t)); d += ' L' + (x0 + t / 7.4 * w).toFixed(1) + ' ' + (y0 + h - Math.max(0, c) * h).toFixed(1); } g += S.path(d, { c: C.red, w: 2 }); }
      else { var s = 'M' + px[0][0] + ' ' + px[0][1]; for (i = 1; i < px.length; i++) { g += '<polygon points="' + px[i - 1][0] + ',' + (y0 + h) + ' ' + px[i - 1][0] + ',' + px[i - 1][1] + ' ' + px[i][0] + ',' + px[i][1] + ' ' + px[i][0] + ',' + (y0 + h) + '" fill="#FBE9E9" stroke="#A1B1C4" stroke-width="0.8"/>'; s += ' L' + px[i][0] + ' ' + px[i][1]; } g += S.path(s, { c: C.char, w: 1.4 }); }
      px.forEach(function(q){ g += S.circle(q[0], q[1], 3.6, { f: C.char }); });
    });
    g += cap('The same observations analysed in two ways.', W, 222);
    return S.svg(W, H, g, 'Observed concentrations analysed by segments between observations and by a fitted curve');
  };

  /* 85: lognormal density with the median marked, mean not marked */
  QZFIG.lognormalDensity = function(o){
    var W = 600, H = 224, g = '', x0 = 60, y0 = 24, w = 480, h = 140, om = o.omega, xs = [], d = '', i, n = 160, xmax = 4.2, mx = 0;
    for (i = 1; i <= n; i++) { var x = xmax * i / n, dens = Math.exp(-Math.pow(Math.log(x), 2) / (2 * om * om)) / (x * om * Math.sqrt(2 * Math.PI)); xs.push([x, dens]); if (dens > mx) mx = dens; }
    xs.forEach(function(p, k){ d += (k ? ' L' : 'M') + (x0 + p[0] / xmax * w).toFixed(1) + ' ' + (y0 + h - p[1] / mx * h * 0.92).toFixed(1); });
    g += axes(x0, y0, w, h, 'individual parameter value (typical value = 1)', 'density') + S.path(d, { c: C.char, w: 2.2 });
    g += S.line(x0 + w / xmax, y0, x0 + w / xmax, y0 + h, { c: C.red, d: '5 4' }) + S.text(x0 + w / xmax + 6, y0 + 14, 'typical value (median)', { a: 'start', s: 11.5, c: C.dred });
    g += cap('Between-subject variability with a lognormal distribution, omega = ' + om + '.', W, 216);
    return S.svg(W, H, g, 'A lognormal distribution of an individual parameter with the typical value marked');
  };

  /* 86: spread across individuals against an interval around one estimate */
  QZFIG.varianceVsUncertainty = function(){
    var W = 600, H = 230, g = '', i;
    g += S.text(150, 22, 'Values for 40 individuals', { b: true, s: 11.5 }) + S.line(40, 150, 260, 150);
    var heights = [4, 9, 16, 24, 30, 32, 26, 18, 10, 5]; heights.forEach(function(v, k){ g += S.rect(48 + k * 21, 150 - v * 3.6, 18, v * 3.6, { f: C.bg, s: C.char, w: 0.8, r: 0 }); });
    g += S.text(150, 172, 'parameter value', { s: 11.5 });
    g += S.text(450, 22, 'One estimate with more data', { b: true, s: 11.5 }) + S.line(340, 150, 560, 150);
    [[0.55, 'n = 10', 70], [0.25, 'n = 100', 100], [0.1, 'n = 1000', 130]].forEach(function(r){ g += S.line(450 - r[0] * 100, r[2], 450 + r[0] * 100, r[2], { c: C.char, w: 2 }) + S.line(450 - r[0] * 100, r[2] - 5, 450 - r[0] * 100, r[2] + 5) + S.line(450 + r[0] * 100, r[2] - 5, 450 + r[0] * 100, r[2] + 5) + S.circle(450, r[2], 3.6, { f: C.red, s: C.red }) + S.text(574, r[2] + 4, r[1], { a: 'end', s: 11 }); });
    g += S.text(450, 172, 'parameter value', { s: 11.5 });
    g += cap('Two different sources of spread in a modelled parameter.', W, 214);
    return S.svg(W, H, g, 'A histogram of individual values and an interval around a single estimate that narrows with sample size');
  };

  /* 87: inputs, model and output with the places that carry error unmarked */
  QZFIG.uncertaintyChain = function(){
    var W = 600, H = 222, g = '';
    g += box(20, 78, 100, 60, ['Compound', 'structure']) + box(160, 78, 120, 60, ['ML-predicted', 'parameters'], { f: C.light }) + box(320, 78, 120, 60, ['PBPK model'], { f: C.light }) + box(480, 78, 100, 60, ['Predicted', 'exposure']);
    g += S.arrow(120, 108, 158, 108) + S.arrow(280, 108, 318, 108) + S.arrow(440, 108, 478, 108);
    g += box(160, 14, 120, 40, ['Training data', 'with assay values'], { s2: 11, f: C.white, d: '4 3' }) + S.arrow(220, 56, 220, 76, { d: '4 3' });
    g += box(320, 14, 120, 40, ['Physiology and', 'model assumptions'], { s2: 11, f: C.white, d: '4 3' }) + S.arrow(380, 56, 380, 76, { d: '4 3' });
    g += cap('Inputs to a PBPK prediction made from ML-predicted parameters.', W, 190);
    return S.svg(W, H, g, 'A workflow from structure through ML-predicted parameters and a PBPK model to a predicted exposure');
  };

  /* 88: simulator over the full range, training inputs sampled in one region only */
  QZFIG.surrogateRange = function(){
    var W = 600, H = 226, g = '', x0 = 70, y0 = 24, w = 470, h = 140, d = '', i, n = 80;
    g += S.rect(x0 + w * 0.18, y0, w * 0.5, h, { f: '#EAF5EE', s: '#2E7D4F', d: '4 3', w: 1.2, r: 0 }) + S.text(x0 + w * 0.43, y0 + 16, 'inputs sampled to train the surrogate', { s: 11.5, c: '#1E6B40' });
    for (i = 0; i <= n; i++) { var t = i / n; d += (i ? ' L' : 'M') + (x0 + w * t).toFixed(1) + ' ' + (y0 + h - h * (0.12 + 0.75 * (1 - Math.exp(-3 * t)) + 0.06 * Math.sin(t * 9))).toFixed(1); }
    g += axes(x0, y0, w, h, 'input parameter of the simulator', 'simulated exposure') + S.path(d, { c: C.char, w: 2.2 });
    g += S.arrow(x0 + w * 0.88, y0 + h - 24, x0 + w * 0.88, y0 + 40, { c: C.red, w: 1.6, d: '4 3' }) + S.text(x0 + w * 0.88, y0 + 30, 'new input', { s: 11.5, c: C.dred });
    g += cap('The simulator is defined over the full range of the input.', W, 218);
    return S.svg(W, H, g, 'A simulator output over an input range with the region sampled for training shaded and a new input outside it');
  };

  /* 89: ensemble members trained on the same data */
  QZFIG.ensembleSpread = function(){
    var W = 600, H = 220, g = '', i, xs = [250, 262, 268, 275, 281, 290];
    g += box(30, 62, 130, 60, ['Training data'], { f: C.light });
    for (i = 0; i < 5; i++) { g += box(220, 20 + i * 36, 100, 28, ['Model ' + (i + 1) + ', seed ' + (i + 1)], { s2: 11 }) + S.arrow(162, 92, 218, 34 + i * 36, { c: C.bg, w: 1 }); }
    g += S.line(380, 170, 570, 170) + S.text(475, 192, 'predicted value for one compound', { s: 11.5 });
    [0, 1, 2, 3, 4].forEach(function(k){ g += S.arrow(322, 34 + k * 36, 400 + xs[k] / 8 - 20 + k * 3, 160, { c: C.bg, w: 0.8 }) + S.circle(400 + xs[k] / 8 + k * 3, 170, 4.4, { f: C.char }); });
    g += S.text(475, 40, 'predictions of the members', { s: 11.5 });
    g += cap('Members of an ensemble differ only in their random seeds.', W, 212);
    return S.svg(W, H, g, 'Five models trained on the same data with different seeds and their predictions for one compound');
  };

  /* 90: mechanistic ODE and a hybrid where one term is learned */
  QZFIG.hybridOde = function(){
    var W = 600, H = 226, g = '';
    g += S.text(150, 20, 'Mechanistic model', { b: true, s: 11.5 }) + S.text(450, 20, 'Hybrid model', { b: true, s: 11.5 });
    [0, 1].forEach(function(p){
      var x = 20 + p * 300;
      g += box(x, 60, 80, 50, ['Gut', 'amount']) + box(x + 130, 60, 80, 50, ['Central', 'amount'], { f: C.light });
      g += S.arrow(x + 82, 85, x + 128, 85) + S.arrow(x + 212, 85, x + 262, 85, { c: C.red });
      g += S.text(x + 105, 74, 'absorption', { s: 11 });
      if (p) g += box(x + 218, 130, 70, 30, ['f(C): network'], { s2: 11, f: '#FBE9E9', s: C.red }) + S.arrow(x + 252, 130, x + 244, 90, { c: C.red, d: '4 3' }) + S.text(x + 252, 70, 'learned', { s: 11, c: C.dred });
      else g += S.text(x + 252, 70, 'CL times C', { s: 11, c: C.dred });
    });
    g += S.text(300, 192, 'Amounts follow mass-balance differential equations.', { s: 11.5, c: C.mid });
    return S.svg(W, H, g, 'A mechanistic compartment model and a hybrid model with one elimination term replaced by a neural network');
  };

  /* 91: random split against temporal split of compounds ordered in time */
  QZFIG.splitTimeline = function(){
    var W = 600, H = 226, g = '', i, seedv = 7;
    function r(){ seedv = (seedv * 16807) % 2147483647; return seedv / 2147483647; }
    [['A: random split', 0], ['B: split by date', 1]].forEach(function(p){
      var y = 40 + p[1] * 84; g += S.text(40, y - 12, p[0], { a: 'start', b: true, s: 11.5 }) + S.line(40, y + 28, 560, y + 28, { c: C.silver });
      for (i = 0; i < 40; i++) { var x = 46 + i * 12.6, test = p[1] ? i >= 32 : [3, 9, 14, 19, 24, 27, 31, 36].indexOf(i) >= 0; g += S.circle(x, y + 8 + (r() - 0.5) * 36, 4, { f: test ? C.red : C.char, s: test ? C.red : C.char }); }
    });
    g += S.arrow(40, 206, 560, 206, { c: C.mid, w: 1 }) + S.text(300, 222, 'time of synthesis', { s: 11.5 });
    g += S.circle(380, 14, 4, { f: C.char }) + S.text(390, 18, 'training', { a: 'start', s: 11 }) + S.circle(460, 14, 4, { f: C.red, s: C.red }) + S.text(470, 18, 'test', { a: 'start', s: 11 });
    return S.svg(W, H, g, 'Compounds ordered by date with test compounds chosen at random or from the latest dates');
  };

  /* 92, 94: the key assessment elements of ICH M15 Section 2.1 */
  QZFIG.m15Elements = function(o){
    var W = 600, H = 226, g = '', el = [['Question of interest', 20, 20], ['Context of use', 220, 20], ['Model influence', 20, 90], ['Consequence of wrong decision', 220, 90], ['Model risk', 420, 55], ['Model impact', 420, 125]];
    el.forEach(function(e){ var hl = o.focus === e[0]; g += box(e[1], e[2], 170, 46, [e[0]], { b: hl, f: hl ? '#FBE9E9' : C.light, s: hl ? C.red : C.char, w: hl ? 2 : 1.2 }); });
    g += S.text(300, 190, 'Key assessment elements in ICH M15, Section 2.1', { s: 11.5, c: C.mid }) + S.text(300, 208, 'Elements rated low, medium or high are accompanied by a justification.', { s: 11, c: C.mid });
    return S.svg(W, H, g, 'The six key assessment elements of the ICH M15 guideline');
  };

  /* 93: two ratings feeding a third */
  QZFIG.m15Risk = function(){
    var W = 600, H = 210, g = '';
    g += box(40, 40, 150, 46, ['Rating A', 'low, medium or high']) + box(40, 110, 150, 46, ['Rating B', 'low, medium or high']) + S.arrow(192, 63, 300, 98) + S.arrow(192, 133, 300, 112);
    g += box(302, 80, 130, 48, ['Model risk', 'low, medium or high'], { f: '#FBE9E9', s: C.red, w: 2 }) + S.arrow(434, 104, 490, 104);
    g += box(492, 80, 90, 48, ['Requirements', 'for model', 'evaluation'], { s2: 11 });
    g += cap('A third rating is derived from two others.', W, 192);
    return S.svg(W, H, g, 'Two ratings combined into a model risk rating that sets requirements for model evaluation');
  };

  /* 94: sources of evidence behind a decision */
  QZFIG.influenceBalance = function(){
    var W = 600, H = 200, g = '';
    g += S.text(300, 22, 'Evidence used to answer the question of interest', { b: true, s: 11.5 });
    [['A', 0.8], ['B', 0.45], ['C', 0.15]].forEach(function(p, k){ var y = 46 + k * 40; g += S.text(60, y + 15, 'Decision ' + p[0], { s: 11.5, a: 'end' }) + S.rect(70, y, 460, 24, { f: C.light, s: C.char, r: 0 }) + S.rect(70, y, 460 * p[1], 24, { f: C.red, s: C.char, r: 0 }); });
    g += S.rect(150, 176, 14, 12, { f: C.red, s: C.char, r: 0 }) + S.text(170, 186, 'model outcomes', { a: 'start', s: 11 }) + S.rect(300, 176, 14, 12, { f: C.light, s: C.char, r: 0 }) + S.text(320, 186, 'other data and evidence', { a: 'start', s: 11 });
    return S.svg(W, H, g, 'Three decisions with different shares of the evidence coming from model outcomes');
  };

  /* 95: generated code checked against a closed-form curve */
  QZFIG.codeCheck = function(){
    var W = 600, H = 226, g = '', x0 = 330, y0 = 30, w = 230, h = 120, i, n = 40, d1 = '', d2 = '';
    g += S.rect(24, 30, 250, 120, { f: C.light, s: C.char }) + S.text(34, 24, 'Generated model code', { a: 'start', b: true, s: 11.5 });
    ['dA/dt = -k * A', 'C = A / V', 'k = CL / V', 'simulate(t = 0 to 24 h)'].forEach(function(l, k){ g += '<text x="38" y="' + (56 + k * 24) + '" font-family="Consolas, monospace" font-size="12" fill="#34373F">' + esc(l) + '</text>'; });
    g += axes(x0, y0, w, h, 'time', 'concentration');
    for (i = 0; i <= n; i++) { var t = i / n; d1 += (i ? ' L' : 'M') + (x0 + w * t).toFixed(1) + ' ' + (y0 + h - h * 0.9 * Math.exp(-2.4 * t)).toFixed(1); d2 += (i ? ' L' : 'M') + (x0 + w * t).toFixed(1) + ' ' + (y0 + h - h * 0.9 * Math.exp(-1.7 * t)).toFixed(1); }
    g += S.path(d1, { c: C.char, w: 2 }) + S.path(d2, { c: C.red, w: 2, d: '5 4' });
    g += S.line(x0 + 10, y0 + 6, x0 + 34, y0 + 6, { c: C.char, w: 2 }) + S.text(x0 + 40, y0 + 10, 'output A', { a: 'start', s: 11 }) + S.line(x0 + 110, y0 + 6, x0 + 134, y0 + 6, { c: C.red, w: 2, d: '5 4' }) + S.text(x0 + 140, y0 + 10, 'output B', { a: 'start', s: 11 });
    g += cap('Two outputs for the same parameters, from different sources.', W, 214);
    return S.svg(W, H, g, 'Generated model code and two concentration curves for the same parameters');
  };

  /* 96: the pieces of an IVIVE chain, not in order */
  QZFIG.ivivePieces = function(){
    var W = 600, H = 220, g = '';
    g += box(20, 24, 160, 46, ['Predicted profile', 'concentration against time'], { s2: 11 }) + box(220, 24, 160, 46, ['Well-stirred model'], { f: C.light }) + box(420, 24, 160, 46, ['HLM intrinsic clearance', 'mL/min/kg'], { s2: 11 });
    g += box(20, 104, 160, 46, ['AUC and Cmax'], {}) + box(220, 104, 160, 46, ['PK or PBPK model', 'with distribution and absorption'], { f: C.light, s2: 11 }) + box(420, 104, 160, 46, ['Scaling to the whole liver', 'and body weight'], { s2: 11 });
    g += cap('Components of a prediction from an in vitro clearance, shown in no order.', W, 190);
    return S.svg(W, H, g, 'Six components of a prediction of exposure from microsomal clearance');
  };

  /* 97: clearance split into an inhibited pathway and other pathways */
  QZFIG.ddiPathways = function(o){
    var W = 600, H = 224, g = '';
    g += box(20, 76, 120, 60, ['Substrate', 'drug'], { f: C.light, b: true });
    g += S.arrow(142, 96, 250, 56) + S.arrow(142, 116, 250, 164);
    g += box(252, 30, 170, 50, ['Pathway through one enzyme', 'fraction fm = ' + o.fm], { s2: 11, f: '#FBE9E9', s: C.red }) + box(252, 140, 170, 50, ['All other pathways', 'fraction 1 minus fm'], { s2: 11 });
    g += box(450, 20, 130, 70, ['Competitive', 'reversible inhibitor', '[I]/Ki = ' + o.iki], { s2: 11 }) + S.arrow(450, 52, 424, 52, { c: C.red, d: '4 3' });
    g += cap('Clearance is the sum of the pathway clearances. Only one pathway is inhibited.', W, 214);
    return S.svg(W, H, g, 'Substrate clearance split between one inhibited enzyme pathway and other pathways');
  };

  var T1 = 'PK and PBPK', T2 = 'ML in PK and MIDD';
  function add(q, fig, o){ q.fig = { name: fig, o: o || {} }; B.push(q); }

  /* 76 */ add(qMC(T1, 'In the well-stirred liver model, what limits hepatic clearance when fu times CLint is much larger than hepatic blood flow?', 'Hepatic blood flow: the extraction ratio approaches 1 and clearance approaches Q.', ['Intrinsic clearance: clearance keeps increasing in proportion to CLint.', 'The fraction unbound alone, because CLint cancels.', 'The volume of distribution of the drug.'], 'Hepatic clearance is Q times fu times CLint divided by (Q + fu times CLint). When fu times CLint greatly exceeds Q, the ratio approaches Q.'), 'liverDiagram', {});
  /* 77 */ add(qMC(T1, 'In the well-stirred model, what does hepatic clearance approximate when fu times CLint is much smaller than hepatic blood flow?', 'Unbound intrinsic clearance, fu times CLint, so clearance is insensitive to blood flow.', ['Hepatic blood flow, so clearance is insensitive to CLint.', 'Zero, because the extraction ratio is 1.', 'The glomerular filtration rate.'], 'When fu times CLint is much smaller than Q, the denominator approaches Q and clearance approaches fu times CLint. The extraction ratio is low and clearance is capacity-limited.'), 'liverDiagram', { label: 'removal rate: fu times CLint' });
  /* 78 */ add(qNum(T1, 'A drug follows a one-compartment model with CL = 5 L/h and V = 35 L. What is the elimination half-life?', 4.85, 0.05, 'h', 'Half-life is ln 2 times V divided by CL: 0.6931 times 35 divided by 5 is 4.85 h.'), 'decayParams', { cl: 5, v: 35, dose: 100 });
  /* 79 */ add(qNum(T1, 'A 100 mg intravenous bolus is given to a subject with a clearance of 8 L/h. What is AUC from zero to infinity?', 12.5, 0.05, 'mg h/L', 'For an intravenous dose AUC equals dose divided by clearance: 100 mg divided by 8 L/h is 12.5 mg h/L.'), 'aucArea', { dose: 100, cl: 8 });
  /* 80 */ add(qNum(T1, 'Hepatic blood flow is 90 L/h, the fraction unbound in blood is 0.2 and the unbound intrinsic clearance of the whole liver is 450 L/h. What is hepatic clearance from the well-stirred model?', 45, 0.5, 'L/h', 'CLh = Q times fu times CLint divided by (Q + fu times CLint) = 90 times 0.2 times 450 divided by (90 + 90) = 45 L/h. The extraction ratio is 0.5.'), 'liverDiagram', { label: 'removal rate: fu times CLint', vals: ['Q = 90 L/h, fu = 0.2', 'CLint = 450 L/h'] });
  /* 81 */ add(qNum(T1, 'A drug with linear kinetics and a half-life of 6 h is given as an intravenous bolus every 12 h. What is the accumulation ratio at steady state?', 1.33, 0.02, '', 'The accumulation ratio is 1 divided by (1 - exp(-ke times tau)). With tau equal to two half-lives, exp(-ke times tau) is 0.25, so the ratio is 1 divided by 0.75, which is 1.33.'), 'accumulationProfile', { thalf: 6, tau: 12 });
  /* 82 */ add(qMC(T1, 'What does perfusion-limited distribution assume for a tissue in a PBPK model?', 'The tissue concentration is in instantaneous equilibrium with the venous blood leaving the tissue, so uptake is limited by blood flow.', ['Uptake is limited by the permeability of the cell membrane.', 'The tissue concentration equals the arterial concentration.', 'The drug does not enter the tissue.'], 'In a perfusion-limited tissue the rate of uptake is set by blood flow. Membrane permeability is not limiting, so the tissue and its venous outflow equilibrate immediately.'), 'tissueFlow', { mode: 'flow' });
  /* 83 */ add(qMC(T1, 'In a perfusion-limited PBPK model, what does the tissue-plasma partition coefficient Kp specify?', 'The ratio of the tissue concentration to the plasma concentration at equilibrium.', ['The rate constant of uptake into the tissue.', 'The fraction of cardiac output that perfuses the tissue.', 'The clearance of the drug by the tissue.'], 'Kp is an equilibrium ratio. Together with the tissue volume it sets the amount of drug the tissue holds, while blood flow sets how fast equilibrium is reached.'), 'tissueFlow', { mode: 'kp' });
  /* 84 */ add(qMC(T1, 'What distinguishes non-compartmental analysis from compartmental analysis?', 'Non-compartmental analysis estimates parameters from the observed profile without assuming a compartmental structure.', ['Non-compartmental analysis requires a PBPK model.', 'Non-compartmental analysis can only be applied to population data.', 'Compartmental analysis cannot estimate clearance.'], 'Non-compartmental analysis uses the observed concentrations directly, for example with the trapezoidal rule. Compartmental analysis fits a model structure to the observations.'), 'ncaVsFit', {});
  /* 85 */ add(qNum(T1, 'Clearance has a lognormal between-subject distribution with omega = 0.5 on the log scale. What is the ratio of the arithmetic mean to the typical value (the median)?', 1.13, 0.01, '', 'For a lognormal distribution the mean is the median times exp(omega squared divided by 2). With omega = 0.5 this is exp(0.125), which is 1.133.'), 'lognormalDensity', { omega: 0.5 });
  /* 86 */ add(qMC(T1, 'What is the difference between between-subject variability and parameter uncertainty?', 'Variability is real variation of a parameter across individuals. Uncertainty is limited knowledge of a parameter value, and it decreases with more data.', ['Variability decreases with more data and uncertainty does not.', 'Both are the same quantity under different names.', 'Variability is measurement error and uncertainty is biological variation.'], 'A larger sample narrows the interval around an estimate, but it does not narrow the spread of the parameter across individuals.'), 'varianceVsUncertainty', {});
  /* 87 */ add(qMulti(T2, 'Which sources of uncertainty affect a PBPK prediction made from machine-learning-predicted parameters? Select all that apply.', ['Error of each predicted input parameter.', 'Assay noise in the data the machine learning model was trained on.', 'Structural assumptions of the PBPK model.'], ['The programming language used to run the simulation.', 'The file format of the report.'], 'Prediction error, assay noise in the training data and model structure all contribute to the uncertainty of the predicted exposure. The language or file format does not change a correct simulation.'), 'uncertaintyChain', {});
  /* 88 */ add(qMC(T2, 'Why can a surrogate model trained on PBPK simulations fail for inputs outside the sampled parameter range?', 'It was fitted only to simulations inside the sampled region and carries no information about the simulator beyond it.', ['The simulator itself changes outside that range.', 'Surrogate models are limited to one output.', 'Neural networks cannot represent smooth functions.'], 'A surrogate approximates the simulator where it was sampled. Outside that region a tree model returns a constant and a network extrapolates in a way the training data do not constrain.'), 'surrogateRange', {});
  /* 89 */ add(qMC(T2, 'An ensemble of models trained on the same data gives a narrow spread of predictions for a compound. What does the spread not include?', 'Error shared by all members, such as assay noise and bias in the training data, so it underestimates the total prediction error.', ['Any effect of the random seeds.', 'The number of members in the ensemble.', 'The variance between members.'], 'Spread measures disagreement between members. Members trained on the same data share its noise and bias, so the spread understates the error against a measured value.'), 'ensembleSpread', {});
  /* 90 */ add(qMC(T2, 'What does a hybrid mechanistic-machine learning model keep from the mechanistic model?', 'The differential equation structure with its mass balance, with a learned function replacing only the unknown term.', ['Nothing: all terms are replaced by a network.', 'Only the parameter values, not the equations.', 'The training data used to fit the network.'], 'In a hybrid model the known terms and the mass balance are retained. Only the term that is not known mechanistically is represented by a learned function.'), 'hybridOde', {});
  /* 91 */ add(qMC(T2, 'Compounds in a project are made over several years and their chemistry drifts. Why does a random split overstate performance for predicting later compounds?', 'Random test compounds have close analogues from the same period in the training set, whereas later compounds differ from the training compounds.', ['Random splits remove the noise from the test set.', 'Random splits use fewer compounds for training.', 'A random split cannot place two compounds in the same set.'], 'A split by date mimics the use of a model on future compounds. A random split places analogues of the test compounds in the training set.'), 'splitTimeline', {});
  /* 92 */ add(qMC(T2, 'In ICH M15, what does the context of use describe?', 'The role and scope of the model or models used to answer the question of interest.', ['The statistical software used to fit the model.', 'The assay protocol used to generate the data.', 'The format of the regulatory submission.'], 'ICH M15, Section 2.1.2, describes the context of use as a concise, clear and explicit description of the role and scope of the models used to answer the question of interest.'), 'm15Elements', { focus: 'Context of use' });
  /* 93 */ add(qMC(T2, 'In ICH M15, which two ratings are combined to give model risk?', 'Model influence and consequence of wrong decision.', ['Model impact and model complexity.', 'Data volume and model accuracy.', 'Question of interest and context of use.'], 'ICH M15, Section 2.1.5, derives model risk by combining model influence and consequence of wrong decision. Model risk determines the requirements for model evaluation.'), 'm15Risk', {});
  /* 94 */ add(qMC(T2, 'In ICH M15, what is model influence?', 'The intended weight of the model outcomes in decision-making, considering the contribution of additional data or evidence.', ['The number of parameters in the model.', 'The precision of the parameter estimates.', 'The extent to which a strategy differs from regulatory standards.'], 'ICH M15, Section 2.1.3. The last option describes model impact, a separate element.'), 'influenceBalance', {});
  /* 95 */ add(qMC(T2, 'A language model drafts code for a one-compartment PK model. Which check is most likely to detect a unit or sign error?', 'Compare the simulated profile with the closed-form solution for chosen parameter values.', ['Read the code and judge whether it looks plausible.', 'Ask the language model to confirm that the code is correct.', 'Run the code once and confirm that it executes without an error.'], 'A numerical comparison with a known solution detects errors that run without exceptions. Plausibility, self-confirmation and successful execution do not.'), 'codeCheck', {});
  /* 96 */ add(qOrder(T1, 'Put the steps from a microsomal measurement to a predicted exposure in order.', ['Scale the HLM intrinsic clearance to the whole liver and body weight.', 'Convert it to hepatic clearance with the well-stirred model.', 'Combine it with distribution and absorption parameters in a PK or PBPK model.', 'Simulate the concentration-time profile and compute AUC and Cmax.'], 'In vitro to in vivo extrapolation scales the intrinsic clearance, converts it to hepatic clearance, and the PK model then predicts the profile and exposure metrics.'), 'ivivePieces', {});
  /* 97 */ add(qNum(T1, 'A substrate is cleared 90% (fm = 0.9) by one enzyme and 10% by other pathways. A competitive reversible inhibitor with [I]/Ki = 4 inhibits only that enzyme. Assuming clearance is the sum of independent pathways, what is the AUC ratio with and without the inhibitor?', 3.57, 0.05, '', 'Clearance with the inhibitor is fm divided by (1 + [I]/Ki) plus (1 - fm) of the original: 0.9 divided by 5 plus 0.1 is 0.28. The AUC ratio is the inverse, 1 divided by 0.28, which is 3.57. This is the static model for one inhibited pathway (Guest et al., Br J Clin Pharmacol 2011;71:72-87).'), 'ddiPathways', { fm: 0.9, iki: 4 });
})();
