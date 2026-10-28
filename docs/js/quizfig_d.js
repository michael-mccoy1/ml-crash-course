"use strict";
/* Quiz schematics for the questions on model comparison, assay noise, generative models, applicability,
   tree-algorithm details and the four ordering questions. Each figure shows the quantities in the question
   and does not mark the answer. */
(function(){
  var S = QZSV, C = S.C;
  function rng(seed){ var s = seed >>> 0; return function(){ s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
  function gauss(r){ var u = 0, v = 0; while (u === 0) u = r(); while (v === 0) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(6.2831853 * v); }
  function rtext(x, y, t, o){ o = o || {}; return '<text transform="translate(' + x + ' ' + y + ') rotate(-90)" text-anchor="middle" fill="' + (o.c || C.char) + '"' + (o.s ? ' font-size="' + o.s + '"' : '') + '>' + esc(t) + '</text>'; }
  function axes(x0, y0, x1, y1, xl, yl){ return S.line(x0, y0, x1, y0) + S.line(x0, y0, x0, y1) + S.text((x0 + x1) / 2, y0 + 30, xl, { s: 12 }) + rtext(x0 - 36, (y0 + y1) / 2, yl, { s: 12 }); }
  function tree(cx, y, w, depth, o){
    /* a complete binary tree drawn from circles; returns svg */
    o = o || {}; var out = '', gap = o.gap || 44, r = o.r || 8;
    (function rec(x, yy, wd, d){
      if (d > 0) { var xl = x - wd / 2, xr = x + wd / 2; out += S.line(x, yy, xl, yy + gap) + S.line(x, yy, xr, yy + gap); rec(xl, yy + gap, wd / 2, d - 1); rec(xr, yy + gap, wd / 2, d - 1); }
      out += S.circle(x, yy, r, { f: d === 0 ? (o.leaf || C.light) : C.white });
    })(cx, y, w, depth);
    return out;
  }

  /* 31: accuracy against training time for four models */
  QZFIG.paretoTime = function(){
    var pts = [['Model A', 150, 140], ['Model B', 250, 78], ['Model C', 345, 118], ['Model D', 470, 52]], g = '';
    [100, 200, 300, 400, 500].forEach(function(x){ g += S.line(x + 20, 30, x + 20, 180, { c: C.silver, w: 1 }); });
    [30, 80, 130, 180].forEach(function(y){ g += S.line(70, y, 560, y, { c: C.silver, w: 1 }); });
    pts.forEach(function(p){ g += S.circle(p[1], p[2], 7, { f: C.red, s: C.dred }) + S.text(p[1], p[2] - 14, p[0], { b: 1 }); });
    return S.svg(600, 230, axes(70, 180, 560, 20, 'Training time (logarithmic axis)', 'Test R2') + g + S.text(300, 14, 'Four models, each with a test R2 and a training time', { s: 12, c: C.mid }), 'Four models plotted by training time and test R2');
  };

  /* 32 and 35: repeat measurements of the same compounds */
  QZFIG.repeatScatter = function(o){
    var r = rng(o.seed || 7), sd = o.sd || 0.45, n = 46, a = '', b = '', i, x, y, px, py;
    for (i = 0; i < n; i++) {
      x = -1.6 + 3.2 * r(); y = x + sd * gauss(r);
      px = 60 + (x + 2) / 4 * 170; py = 190 - (Math.max(-2, Math.min(2, y)) + 2) / 4 * 160; a += S.circle(px, py, 3.4, { f: C.red, s: C.red, w: 0.5 });
      y = x + sd * gauss(r); px = 340 + (x + 2) / 4 * 170; py = 190 - (Math.max(-2, Math.min(2, y)) + 2) / 4 * 160; b += S.circle(px, py, 3.4, { f: C.char, s: C.char, w: 0.5 });
    }
    var inner = S.rect(60, 30, 170, 160, { f: C.white, s: C.silver, r: 0 }) + S.line(60, 190, 230, 30, { c: C.mid, d: '5 4', w: 1.2 }) + a + S.text(145, 214, 'First measurement', { s: 12 }) + rtext(40, 110, 'Repeat measurement', { s: 12 }) + S.text(145, 20, 'Same compounds measured twice', { b: 1, s: 12 }) +
      S.rect(340, 30, 170, 160, { f: C.white, s: C.silver, r: 0 }) + S.line(340, 190, 510, 30, { c: C.mid, d: '5 4', w: 1.2 }) + b + S.text(425, 214, 'Noise-free value', { s: 12 }) + rtext(320, 110, 'One measurement', { s: 12 }) + S.text(425, 20, 'Values with no noise against one measurement', { b: 1, s: 12 });
    return S.svg(600, 230, inner, 'Scatter of repeat measurements about the diagonal');
  };

  /* 33: language model output compared with source data */
  QZFIG.generatedText = function(){
    var lines = function(x, y, w, n){ var s = ''; for (var k = 0; k < (n || 4); k++) s += S.rect(x, y + k * 14, w * (k === (n || 4) - 1 ? 0.6 : 1), 7, { f: C.silver, s: C.silver, r: 2, w: 0.5 }); return s; };
    var inner = S.rect(20, 40, 110, 80, { f: C.light }) + S.text(75, 62, 'Prompt', { b: 1 }) + lines(34, 76, 82, 3) +
      S.arrow(132, 80, 190, 80) +
      S.rect(192, 30, 110, 100, { f: C.light }) + S.text(247, 56, 'Language', { b: 1 }) + S.text(247, 74, 'model', { b: 1 }) + S.text(247, 100, 'next-token', { s: 11, c: C.mid }) + S.text(247, 114, 'probabilities', { s: 11, c: C.mid }) +
      S.arrow(304, 80, 362, 80) +
      S.rect(364, 30, 130, 100, { f: C.white, s: C.red, w: 2 }) + S.text(429, 52, 'Generated text', { b: 1 }) + lines(378, 66, 102) +
      S.rect(364, 160, 130, 56, { f: C.white, d: '5 4' }) + S.text(429, 182, 'Measurements and', { s: 12 }) + S.text(429, 198, 'cited sources', { s: 12 }) +
      S.line(429, 132, 429, 158, { d: '4 3', c: C.mid });
    return S.svg(600, 230, inner, 'A language model produces text, and measurements and sources are separate from it');
  };

  /* 34: a fitted curve beyond the edge of the training range */
  QZFIG.trainingRange = function(){
    var r = rng(11), pts = '', i, x, y, step = '', cur = null;
    var fx = function(t){ return 60 + t * 490; }, fy = function(v){ return 190 - v * 150; };
    var xs = [], ys = [];
    for (i = 0; i < 36; i++) { x = i / 35; y = 0.12 + 0.78 * Math.sqrt(x) + 0.05 * gauss(r); xs.push(x); ys.push(y); }
    var edge = 0.52, fit = [], lastv = 0;
    [0, 0.1, 0.2, 0.3, 0.4, 0.52].forEach(function(t, k){ var v = 0.12 + 0.78 * Math.sqrt(Math.max(t - 0.05, 0)); fit.push([t, v]); lastv = v; });
    for (i = 0; i < xs.length; i++) pts += S.circle(fx(xs[i]), fy(ys[i]), 3.4, { f: xs[i] <= edge ? C.char : C.white, s: xs[i] <= edge ? C.char : C.red, w: xs[i] <= edge ? 0.5 : 1.4 });
    var d = 'M' + fx(0) + ' ' + fy(fit[0][1]); for (i = 1; i < fit.length; i++) d += ' L' + fx(fit[i][0]) + ' ' + fy(fit[i - 1][1]) + ' L' + fx(fit[i][0]) + ' ' + fy(fit[i][1]); d += ' L' + fx(1) + ' ' + fy(lastv);
    var inner = S.rect(60, 25, 490 * edge, 165, { f: 'rgba(161,177,196,0.28)', s: 'rgba(161,177,196,0.28)', r: 0 }) + axes(60, 190, 550, 25, 'Feature value', 'Measured value') + pts + S.path(d, { c: C.red, w: 2.4 }) +
      S.text(60 + 490 * edge / 2, 44, 'Training range', { b: 1 }) + S.text(60 + 490 * (1 + edge) / 2, 44, 'Outside the training range', { b: 1 }) +
      S.line(70, 14, 94, 14, { c: C.red, w: 2.4 }) + S.text(100, 18, 'fitted values', { a: 'start', s: 11 }) + S.circle(215, 14, 3.4, { f: C.char, s: C.char }) + S.text(223, 18, 'training compound', { a: 'start', s: 11 }) + S.circle(355, 14, 3.4, { f: C.white, s: C.red, w: 1.4 }) + S.text(363, 18, 'later measurement', { a: 'start', s: 11 });
    return S.svg(600, 230, inner, 'A fitted curve and measured values inside and outside the training range');
  };

  /* 36: squared residuals about one mean and about two group means (setup only) */
  QZFIG.splitSquares = function(){
    var v = [1.0, 1.4, 1.2, 1.7, 3.0, 3.5, 3.2, 3.7], m = v.reduce(function(a, b){ return a + b; }, 0) / v.length, k = 30, inner = '', i, sd, x, yb;
    inner += S.text(300, 18, 'Squares with the residual of each compound as the side', { s: 12, c: C.mid });
    inner += S.text(20, 56, 'One group', { a: 'start', b: 1 }) + S.text(20, 74, 'one mean', { a: 'start', s: 11, c: C.mid });
    for (i = 0; i < 8; i++) { sd = Math.abs(v[i] - m) * k; x = 110 + i * 56; yb = 100; inner += S.rect(x, yb - sd, sd, sd, { f: 'rgba(225,36,42,0.18)', s: C.red, r: 0, w: 1.2 }); }
    inner += S.line(100, 100, 560, 100, { c: C.mid, w: 1 });
    inner += S.text(20, 140, 'After a split', { a: 'start', b: 1 }) + S.text(20, 158, 'two means', { a: 'start', s: 11, c: C.mid });
    var g1 = v.slice(0, 4), g2 = v.slice(4), m1 = g1.reduce(function(a, b){ return a + b; }, 0) / 4, m2 = g2.reduce(function(a, b){ return a + b; }, 0) / 4;
    for (i = 0; i < 8; i++) { var mm = i < 4 ? m1 : m2; sd = Math.abs(v[i] - mm) * k; x = 110 + i * 56 + (i < 4 ? 0 : 24); yb = 190; inner += S.rect(x, yb - sd, sd, sd, { f: 'rgba(225,36,42,0.18)', s: C.red, r: 0, w: 1.2 }); }
    inner += S.line(100, 190, 560, 190, { c: C.mid, w: 1 }) + S.line(110 + 4 * 56 - 16, 120, 110 + 4 * 56 - 16, 200, { c: C.mid, d: '4 3', w: 1.2 }) + S.text(190, 214, 'Group 1', { b: 1 }) + S.text(430, 214, 'Group 2', { b: 1 });
    return S.svg(600, 230, inner, 'Squared residuals of eight compounds about one mean and about two group means');
  };

  /* 37: feature values, bins and the thresholds that bins allow */
  QZFIG.binsHistogram = function(){
    var r = rng(5), n = 40, vals = [], i, bins = 8, cnt = [], x0 = 70, x1 = 560, out = '';
    for (i = 0; i < n; i++) vals.push(Math.min(0.999, Math.max(0, 0.45 + 0.2 * gauss(r))));
    for (i = 0; i < bins; i++) cnt.push(0); vals.forEach(function(v){ cnt[Math.floor(v * bins)]++; });
    var bw = (x1 - x0) / bins, mx = Math.max.apply(null, cnt);
    for (i = 0; i < bins; i++) out += S.rect(x0 + i * bw + 2, 150 - cnt[i] / mx * 90, bw - 4, cnt[i] / mx * 90, { f: C.light, s: C.bg, r: 0, w: 1.2 });
    for (i = 0; i <= bins; i++) out += S.line(x0 + i * bw, 40, x0 + i * bw, 168, { c: C.red, d: i === 0 || i === bins ? '' : '5 4', w: i === 0 || i === bins ? 1 : 1.4 });
    vals.sort(function(a, b){ return a - b; }).forEach(function(v, k){ out += S.circle(x0 + v * (x1 - x0), 164 + (k % 3) * 5, 2.8, { f: C.char, s: C.char, w: 0.4 }); });
    return S.svg(600, 230, out + S.line(x0, 150, x1, 150) + S.text((x0 + x1) / 2, 214, 'Feature value (each dot is one compound)', { s: 12 }) + S.text(310, 20, 'Counts per bin, with the bin edges drawn as dashed lines', { s: 12, c: C.mid }) + rtext(36, 105, 'Compounds in the bin', { s: 11 }), 'Histogram of a feature with bin edges and the compounds in each bin');
  };

  /* 38: a tree with leaf counts and one candidate split of a small leaf */
  QZFIG.leafCounts = function(){
    var node = function(x, y, t, d, w){ w = w || 96; return S.rect(x - w / 2, y - 15, w, 30, { f: d ? C.white : C.light, d: d ? '5 4' : '', s: d ? C.red : C.char }) + S.text(x, y + 4, t, { s: 12 }); };
    var inner = S.line(260, 55, 130, 105) + S.line(260, 55, 390, 105) + S.line(390, 105, 320, 160) + S.line(390, 105, 460, 160) + S.line(460, 160, 420, 205, { c: C.red, d: '5 4' }) + S.line(460, 160, 504, 205, { c: C.red, d: '5 4' }) +
      node(260, 40, 'n = 30') + node(130, 105, 'n = 18') + node(390, 105, 'n = 12') + node(320, 160, 'n = 7') + node(460, 160, 'n = 5') + node(420, 205, 'n = 4', 1, 76) + node(504, 205, 'n = 1', 1, 76) +
      S.text(130, 136, 'leaf', { s: 11, c: C.mid }) + S.text(320, 191, 'leaf', { s: 11, c: C.mid }) +
      S.text(20, 30, 'n: compounds in the node', { a: 'start', s: 11, c: C.mid }) + S.text(20, 46, 'Dashed: a candidate split', { a: 'start', s: 11, c: C.red });
    return S.svg(600, 230, inner, 'A small tree with the number of compounds in each node and one candidate split of the node with five compounds');
  };

  /* 39: bootstrap sampling as a process (no outcome drawn) */
  QZFIG.bootstrapProcess = function(){
    var out = '', i, cols = 6;
    out += S.rect(30, 50, 170, 130, { f: C.light }) + S.text(115, 40, 'Training set, n compounds', { b: 1 });
    for (i = 0; i < 12; i++) out += S.circle(58 + (i % cols) * 25, 84 + Math.floor(i / cols) * 36, 8, { f: C.white });
    out += S.text(115, 150, 'n circles', { s: 11, c: C.mid }) ;
    out += S.arrow(205, 100, 395, 100) + S.text(300, 90, 'Draw one compound at random', { s: 12 }) + S.arrow(395, 140, 205, 140, { d: '5 4' }) + S.text(300, 160, 'Replace it in the training set', { s: 12 }) + S.text(300, 205, 'Repeat n times', { b: 1 });
    out += S.rect(400, 50, 170, 130, { f: C.white }) + S.text(485, 40, 'Bootstrap sample, n draws', { b: 1 });
    for (i = 0; i < 12; i++) out += S.rect(420 + (i % cols) * 24, 76 + Math.floor(i / cols) * 36, 18, 18, { f: C.light, s: C.bg, r: 2, w: 1 });
    return S.svg(600, 230, out, 'A training set sampled with replacement into a bootstrap sample of the same size');
  };

  /* 40: level-wise and leaf-wise growth, both ending with eight leaves */
  QZFIG.growthRules = function(){
    var out = '', r = 7;
    var lvl = function(cx, y){ var s = ''; (function rec(x, yy, wd, d){ if (d > 0) { var xl = x - wd / 2, xr = x + wd / 2; s += S.line(x, yy, xl, yy + 48) + S.line(x, yy, xr, yy + 48); rec(xl, yy + 48, wd / 2, d - 1); rec(xr, yy + 48, wd / 2, d - 1); } s += S.circle(x, yy, r, { f: d === 0 ? C.light : C.white }); })(cx, y, 120, 3); return s; };
    var leaf = function(cx, y){ var s = '', x = cx, yy = y, k; for (k = 0; k < 7; k++) { var lx = x - 26, rx = x + 26; s += S.line(x, yy, lx, yy + 24) + S.line(x, yy, rx, yy + 24); s += S.circle(rx, yy + 24, r, { f: C.light }); s += S.circle(x, yy, r, { f: C.white }); x = lx; yy += 24; } s += S.circle(x, yy, r, { f: C.light }); return s; };
    out += S.text(150, 22, 'Level-wise growth', { b: 1 }) + lvl(150, 42) + S.text(450, 22, 'Leaf-wise growth', { b: 1 }) + leaf(480, 42) + S.text(150, 214, '8 leaves', { s: 12, c: C.mid }) + S.text(450, 214, '8 leaves', { s: 12, c: C.mid });
    out += S.circle(20, 214, 6, { f: C.light }) + S.text(30, 218, 'leaf', { a: 'start', s: 11 });
    return S.svg(600, 230, out, 'Two trees with eight leaves each, one grown by depth and one grown by leaf');
  };

  /* 41: candidate settings that need to be compared, and the compound sets that exist */
  QZFIG.settingsToCompare = function(){
    var out = '', i, lab = ['Depth 2', 'Depth 4', 'Depth 6'];
    out += S.text(300, 18, 'Three candidate settings must be compared, and the test set is reserved for the final report', { s: 12, c: C.mid });
    out += S.rect(30, 40, 250, 56, { f: C.light }) + S.text(155, 62, 'Training compounds', { b: 1 }) + S.text(155, 82, 'used to fit each model', { s: 11, c: C.mid });
    out += S.rect(320, 40, 250, 56, { f: C.white, s: C.silver, w: 1.6 }) + S.text(445, 62, 'Test compounds', { b: 1 }) + S.text(445, 82, 'kept for the final score', { s: 11, c: C.mid });
    for (i = 0; i < 3; i++) out += S.rect(60 + i * 100, 130, 84, 34, { f: C.white }) + S.text(102 + i * 100, 152, lab[i], { b: 1 }) + S.arrow(102 + i * 100, 166, 102 + i * 100, 188, { d: '4 3' });
    out += S.rect(40, 190, 320, 32, { f: C.white, d: '5 4', s: C.red }) + S.text(200, 211, 'Which compounds give the comparison score?', { s: 12, c: C.dred });
    return S.svg(600, 230, out, 'Training and test compounds, three candidate settings and an unspecified comparison set');
  };

  /* 42: a tree output on a number line with the learning rate (setup only) */
  QZFIG.boostCorrection = function(){
    var x0 = 70, u = 130, out = '', i;
    out += S.line(x0, 120, x0 + 2.2 * u, 120, { w: 1.6 });
    for (i = 0; i <= 4; i++) { out += S.line(x0 + i * 0.5 * u, 114, x0 + i * 0.5 * u, 126, { w: 1.2 }) + S.text(x0 + i * 0.5 * u, 142, (i * 0.5).toFixed(1), { s: 11 }); }
    out += S.text(x0 + 1.1 * u, 164, 'Change in the prediction', { s: 12 }) + S.circle(x0, 120, 6, { f: C.red, s: C.dred }) + S.text(x0, 100, 'Prediction before', { s: 12, b: 1 });
    out += S.arrow(x0, 70, x0 + 2.0 * u, 70, { c: C.red, w: 2.2 }) + S.text(x0 + 1.0 * u, 58, 'Tree output for this compound: +2.0', { s: 12, c: C.dred, b: 1 });
    out += S.rect(400, 175, 170, 36, { f: C.light }) + S.text(485, 198, 'Learning rate: 0.1', { b: 1 });
    out += S.text(300, 24, 'Each round adds the tree output, scaled by the learning rate', { s: 12, c: C.mid });
    return S.svg(600, 230, out, 'A tree output of plus two on a number line and a learning rate of 0.1');
  };

  /* 43: a boosted ensemble with its parts named */
  QZFIG.boostedParts = function(){
    var out = '', xs = [70, 190, 310], lab = ['Tree 1', 'Tree 2', 'Tree 3'];
    xs.forEach(function(x, i){
      out += S.text(x, 22, lab[i], { b: 1 }) + S.line(x, 48, x - 30, 96) + S.line(x, 48, x + 30, 96) + S.circle(x, 48, 9, { f: C.white });
      out += S.rect(x - 56, 96, 48, 24, { f: C.light }) + S.text(x - 32, 112, '0.4', { s: 11 }) + S.rect(x + 8, 96, 48, 24, { f: C.light }) + S.text(x + 32, 112, '-0.7', { s: 11 });
      out += S.arrow(x, 124, x, 156, { c: C.mid });
    });
    out += S.text(86, 42, 'x at most 2.9', { a: 'start', s: 11, c: C.mid });
    out += S.rect(20, 158, 380, 32, { f: C.white }) + S.text(210, 179, 'Prediction = start value + scaled sum of the tree outputs', { s: 12 });
    out += S.line(20, 204, 400, 204, { c: C.char, w: 1.6 }) + S.text(210, 222, 'Number of rounds: how many trees are added', { s: 12 });
    out += S.line(374, 48, 374, 96, { c: C.red, w: 2 }) + S.text(384, 76, 'Tree depth', { a: 'start', s: 12 });
    out += S.arrow(440, 28, 322, 46, { c: C.mid }) + S.text(444, 32, 'Split threshold', { a: 'start', s: 12 });
    out += S.arrow(440, 112, 372, 110, { c: C.mid }) + S.text(444, 116, 'Leaf value', { a: 'start', s: 12 });
    out += S.arrow(440, 144, 318, 142, { c: C.mid }) + S.text(444, 148, 'Learning rate', { a: 'start', s: 12 });
    return S.svg(600, 230, out, 'Three trees of a boosted model with the parts that can be named: rounds, depth, learning rate, leaf values and split thresholds');
  };

  /* 55: the objects in one round of gradient boosting, shown without an order */
  QZFIG.boostObjects = function(){
    var out = '', r = rng(3), i, pts = [], p = function(i){ return 0.18 + 0.64 * i / 9; };
    var panel = function(x, y, title){ return S.rect(x, y, 270, 88, { f: C.white, s: C.silver, w: 1.4 }) + S.text(x + 135, y + 16, title, { b: 1, s: 12 }); };
    for (i = 0; i < 10; i++) pts.push(0.5 + 0.35 * Math.sin(i * 0.8) + 0.18 * (r() - 0.5));
    var curve = function(x, y, flat){ var d = '', k; for (k = 0; k < 10; k++) { var yy = flat ? 0.5 : 0.5 + 0.28 * Math.sin(k * 0.8); d += (k ? ' L' : 'M') + (x + 14 + k * 27) + ' ' + (y + 74 - yy * 44); } return S.path(d, { c: C.red, w: 2 }); };
    var dots = function(x, y){ var s = '', k; for (k = 0; k < 10; k++) s += S.circle(x + 14 + k * 27, y + 74 - pts[k] * 44, 3.4, { f: C.char, s: C.char, w: 0.4 }); return s; };
    out += panel(20, 20, 'Updated model') + dots(20, 20) + curve(20, 20, false);
    out += panel(310, 20, 'Residuals') + (function(){ var s = '', k; for (k = 0; k < 10; k++) { var yc = 20 + 74 - (0.5 * 44), yp = 20 + 74 - pts[k] * 44; s += S.line(310 + 14 + k * 27, yc, 310 + 14 + k * 27, yp, { c: C.red, w: 2 }) + S.circle(310 + 14 + k * 27, yp, 3, { f: C.char, s: C.char, w: 0.4 }); } return s + S.line(320, 20 + 74 - 22, 570, 20 + 74 - 22, { c: C.mid, d: '4 3', w: 1 }); })();
    out += panel(20, 122, 'Small tree') + S.line(155, 152, 125, 180) + S.line(155, 152, 185, 180) + S.circle(155, 152, 8, { f: C.white }) + S.rect(98, 180, 54, 22, { f: C.light }) + S.rect(158, 180, 54, 22, { f: C.light });
    out += panel(310, 122, 'Training data and current model') + dots(310, 122) + curve(310, 122, true);
    out += S.text(300, 226, 'Four objects used in one round of gradient boosting', { s: 11, c: C.mid });
    return S.svg(600, 230, out, 'Four objects used in one round of gradient boosting: data with the current model, residuals, a small tree and the updated model');
  };

  /* 56: the data sets, settings and score in a modelling workflow, shown without an order */
  QZFIG.workflowParts = function(){
    var out = '', box = function(x, y, w, h, t, t2, d){ return S.rect(x, y, w, h, { f: d ? C.white : C.light, d: d ? '5 4' : '' }) + S.text(x + w / 2, y + h / 2 - (t2 ? 2 : -4), t, { b: 1 }) + (t2 ? S.text(x + w / 2, y + h / 2 + 15, t2, { s: 11, c: C.mid }) : ''); };
    out += box(40, 30, 160, 56, 'Score on test compounds', 'reported once') + box(220, 30, 160, 56, 'Fitted model', 'weights or trees') + box(400, 30, 160, 56, 'All compounds', 'with measured targets');
    out += box(40, 130, 160, 56, 'Validation part', 'cut from training') + box(220, 130, 160, 56, 'Candidate settings', 'depth, learning rate') + box(400, 130, 160, 56, 'Test compounds', 'excluded from fitting');
    out += S.text(300, 216, 'Six components of a modelling workflow, with no order implied', { s: 12, c: C.mid });
    return S.svg(600, 230, out, 'Six components of a modelling workflow, with no order implied');
  };

  /* 57: the quantities in one training step of a network, without an order */
  QZFIG.networkQuantities = function(){
    var out = '', ins = [60, 100, 140], hid = [50, 85, 120, 155], i, j;
    for (i = 0; i < 3; i++) for (j = 0; j < 4; j++) out += S.line(150, ins[i] + 10, 290, hid[j] + 10, { c: C.silver, w: 1 });
    for (j = 0; j < 4; j++) out += S.line(290, hid[j] + 10, 410, 110, { c: C.silver, w: 1 });
    for (i = 0; i < 3; i++) out += S.circle(150, ins[i] + 10, 9, { f: C.white });
    for (j = 0; j < 4; j++) out += S.circle(290, hid[j] + 10, 9, { f: C.white });
    out += S.circle(410, 110, 11, { f: C.light });
    out += S.text(150, 38, 'Inputs', { b: 1 }) + S.text(290, 28, 'Hidden layer', { b: 1 }) + S.text(410, 82, 'Prediction', { b: 1 });
    out += S.rect(30, 170, 120, 40, { f: C.white }) + S.text(90, 195, 'Weights and biases', { s: 12 });
    out += S.rect(470, 40, 110, 40, { f: C.light }) + S.text(525, 65, 'Measured target', { s: 12, b: 1 });
    out += S.rect(470, 130, 110, 40, { f: C.white, s: C.red, w: 1.8 }) + S.text(525, 155, 'Loss', { b: 1 });
    out += S.text(300, 224, 'Quantities of one training step, with no order implied', { s: 12, c: C.mid });
    return S.svg(600, 230, out, 'A small network with its inputs, weights, prediction, measured target and loss');
  };

  /* 58: the objects a GPT works with when it extends a SMILES string, without an order */
  QZFIG.gptObjects = function(){
    var out = '', vocab = ['C', 'N', 'O', '(', ')', '=', '1', 'c'], k, sc = [2.1, 0.8, -0.4, 0.2, -1.2, 0.5, 0.9, 1.4], pr = [0.34, 0.09, 0.03, 0.06, 0.01, 0.05, 0.1, 0.17];
    var panel = function(x, y, title){ return S.rect(x, y, 270, 94, { f: C.white, s: C.silver, w: 1.4 }) + S.text(x + 135, y + 16, title, { b: 1, s: 12 }); };
    var bars = function(x, y, vals, neg, mx){ var s = '', base = y + 70 - (neg ? 14 : 0); for (k = 0; k < 8; k++) { var hgt = vals[k] / mx * 40; s += S.rect(x + 22 + k * 28, vals[k] >= 0 ? base - hgt : base, 16, Math.abs(hgt), { f: C.bg, s: C.mid, r: 0, w: 1 }) + S.text(x + 30 + k * 28, y + 88, vocab[k], { s: 11 }); } return s + S.line(x + 14, base, x + 250, base, { c: C.char, w: 1 }); };
    out += panel(20, 16, 'Probability of each token') + bars(20, 16, pr, false, 0.4);
    out += panel(310, 16, 'Next token') + S.rect(380, 52, 130, 36, { f: C.light, d: '5 4', s: C.red }) + S.text(445, 76, '?', { b: 1, s: 16, c: C.red });
    out += panel(20, 120, 'String so far') + (function(){ var toks = ['C', 'C', '(', '=', 'O', ')'], s = '', t; for (t = 0; t < toks.length; t++) s += S.rect(36 + t * 38, 160, 32, 30, { f: C.light }) + S.text(52 + t * 38, 180, toks[t], { b: 1 }); return s; })();
    out += panel(310, 120, 'Score of each token') + bars(310, 120, sc, true, 2.5);
    out += S.text(300, 226, 'Four objects used to extend a string by one token', { s: 11, c: C.mid });
    return S.svg(600, 230, out, 'Four objects used when a GPT extends a SMILES string by one token: the string, token scores, token probabilities and the chosen token');
  };

  (function(){
    var B = QZ.bank;
    function set(i, name, o){ B[i - 1].fig = { name: name, o: o || {} }; }
    set(31, 'paretoTime', {});
    set(32, 'repeatScatter', { sd: 0.5, seed: 7 });
    set(33, 'generatedText', {});
    set(34, 'trainingRange', {});
    set(35, 'repeatScatter', { sd: 0.32, seed: 19 });
    set(36, 'splitSquares', {});
    set(37, 'binsHistogram', {});
    set(38, 'leafCounts', {});
    set(39, 'bootstrapProcess', {});
    set(40, 'growthRules', {});
    set(41, 'settingsToCompare', {});
    set(42, 'boostCorrection', {});
    set(43, 'boostedParts', {});
    set(55, 'boostObjects', {});
    set(56, 'workflowParts', {});
    set(57, 'networkQuantities', {});
    set(58, 'gptObjects', {});
  })();
})();
