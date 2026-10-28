"use strict";
/* Quiz schematics for questions 1 to 15: data splits, R2, censoring, stacking, decision trees, forests and boosting.
   Each figure depicts the quantities in the question and does not mark the correct option. */
(function(){
  var S = QZSV, C = S.C;
  function rnd(seed){ var s = seed >>> 0; return function(){ s = (s + 0x6D2B79F5) >>> 0; var t = Math.imul(s ^ s >>> 15, 1 | s); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  function dots(x, y, w, h, n, seed, o){ o = o || {}; var r = rnd(seed), out = '', i; for (i = 0; i < n; i++) out += S.circle(x + 6 + r() * (w - 12), y + 6 + r() * (h - 12), o.r || 3.4, { f: o.f || C.char, s: o.s || o.f || C.char, w: o.w || 1 }); return out; }
  function cap(t, w, y){ return S.text(w / 2, y, t, { c: C.mid, i: true }); }
  function cap2(a, b, w, y){ return S.text(w / 2, y, a, { c: C.mid, i: true }) + S.text(w / 2, y + 15, b, { c: C.mid, i: true }); }
  function tree(x, y, depth, dx, dy){ /* binary tree from the root at (x, y): lines, circles for the internal nodes, and the leaf positions */
    var out = '', lv, k, nodes = [[x, y]], all = [];
    for (lv = 0; lv < depth; lv++) { var next = []; for (k = 0; k < nodes.length; k++) { var px = nodes[k][0], py = nodes[k][1], off = dx / Math.pow(2, lv); [-1, 1].forEach(function(sg){ var cx = px + sg * off, cy = py + dy; out += S.line(px, py, cx, cy, { c: C.bg, w: 1.4 }); next.push([cx, cy]); }); all.push(nodes[k]); } nodes = next; }
    all.forEach(function(p){ out += S.circle(p[0], p[1], 5, { f: C.char }); });
    return { svg: out, leaves: nodes };
  }

  /* 1. split of compounds into a training set and a test set */
  QZFIG.splitDots = function(){
    var W = 600, H = 220, g = '';
    g += S.rect(10, 40, 130, 110, { f: C.light }) + S.text(75, 32, 'All compounds', { b: true }) + dots(10, 40, 130, 110, 26, 3);
    g += S.arrow(145, 95, 195, 60) + S.arrow(145, 95, 195, 140);
    g += S.rect(200, 14, 150, 86, { f: C.light }) + S.text(275, 11, 'Training set', { b: true }) + dots(200, 14, 150, 86, 17, 5);
    g += S.rect(200, 118, 150, 72, { f: C.light, s: C.red }) + S.text(275, 112, 'Test set', { b: true, c: C.dred }) + dots(200, 118, 150, 72, 8, 9, { f: C.red });
    g += S.rect(420, 24, 150, 56, { f: C.white, s: C.char, w: 2 }) + S.text(495, 48, 'Model', { b: true }) + S.text(495, 66, 'fitted to the training set', { s: 11 });
    g += S.arrow(352, 52, 418, 52) + S.text(385, 44, 'fit', { s: 11 });
    g += S.line(495, 82, 495, 154, { c: C.char }) + S.arrow(495, 154, 353, 154) + S.text(528, 120, 'predict', { s: 11 });
    g += cap('Compounds are split into a training set and a test set before the model is fitted.', W, 212);
    return S.svg(W, H, g, 'Compounds split into a training set and a test set; the model is fitted to the training set and applied to the test set');
  };

  /* 2. fit through training points, test points off the curve, and the two R2 values */
  QZFIG.trainTestFit = function(o){
    var W = 600, H = 232, g = '', r = rnd(11), tr = o.train === undefined ? 0.95 : o.train, te = o.test === undefined ? 0.2 : o.test, i, px = [], py = [], x0 = 30, y0 = 28, pw = 290, ph = 150;
    g += S.rect(x0, y0, pw, ph, { f: C.white, s: C.silver }) + S.text(x0 + pw / 2, 20, 'Fitted function and compounds', { b: true });
    for (i = 0; i < 9; i++) { var x = x0 + 18 + i * (pw - 36) / 8, y = y0 + ph / 2 + Math.sin(i * 0.7) * 18 + (r() - 0.5) * 70; px.push(x); py.push(Math.max(y0 + 12, Math.min(y0 + ph - 12, y))); }
    var d = 'M' + px[0] + ' ' + py[0]; for (i = 1; i < px.length; i++) d += ' L' + px[i] + ' ' + py[i];
    g += S.path(d, { c: C.char, w: 1.6 });
    for (i = 0; i < px.length; i++) g += S.circle(px[i], py[i], 4, { f: C.char });
    for (i = 0; i < 5; i++) { var tx = x0 + 30 + i * 58 + (r() - 0.5) * 10, seg = Math.min(7, Math.max(0, Math.floor((tx - px[0]) / ((px[8] - px[0]) / 8)))), fr = (tx - px[seg]) / (px[seg + 1] - px[seg]), cy = py[seg] + (py[seg + 1] - py[seg]) * fr, ty = cy + (i % 2 ? 1 : -1) * (30 + r() * 16); g += S.circle(tx, Math.max(y0 + 12, Math.min(y0 + ph - 12, ty)), 4, { f: C.white, s: C.red, w: 2 }); }
    g += S.circle(x0 + 20, y0 + ph + 22, 4, { f: C.char }) + S.text(x0 + 30, y0 + ph + 26, 'training compound', { a: 'start', s: 11 }) + S.circle(x0 + 160, y0 + ph + 22, 4, { f: C.white, s: C.red, w: 2 }) + S.text(x0 + 170, y0 + ph + 26, 'test compound', { a: 'start', s: 11 });
    var bx = 380, by = 28, bh = 150;
    g += S.text(bx + 110, 20, 'R2', { b: true }) + S.line(bx, by, bx, by + bh, { c: C.silver }) + S.line(bx, by + bh, bx + 220, by + bh, { c: C.silver });
    [0, 0.5, 1].forEach(function(v){ var yy = by + bh - v * bh; g += S.line(bx - 4, yy, bx, yy, { c: C.mid }) + S.text(bx - 8, yy + 4, String(v), { a: 'end', s: 11 }); });
    g += S.rect(bx + 30, by + bh - tr * bh, 60, tr * bh, { f: C.char, s: C.char, r: 0 }) + S.text(bx + 60, by + bh - tr * bh - 6, tr.toFixed(2), { b: true }) + S.text(bx + 60, by + bh + 16, 'Training set', { s: 11 });
    g += S.rect(bx + 130, by + bh - te * bh, 60, te * bh, { f: C.red, s: C.red, r: 0 }) + S.text(bx + 160, by + bh - te * bh - 6, te.toFixed(2), { b: true, c: C.dred }) + S.text(bx + 160, by + bh + 16, 'Test set', { s: 11 });
    g += cap('The same model scored on the compounds it was fitted to and on compounds held out.', W, 224);
    return S.svg(W, H, g, 'A fitted function through training compounds, test compounds, and the training and test R2');
  };

  /* 3. R2 as residuals about the model compared with residuals about the mean */
  QZFIG.r2Residuals = function(){
    var W = 600, H = 236, g = '', r = rnd(21), i, pts = [], bx = 40, by = 30, bw = 230, bh = 118, bx2 = 340;
    for (i = 0; i < 12; i++) pts.push([0.08 + i * 0.075 + (r() - 0.5) * 0.04, 0.5 + (r() - 0.5) * 0.7]);
    function px(u, ox){ return ox + u * bw; } function py(v){ return by + bh - v * bh; }
    g += S.text(bx + bw / 2, 20, 'Residuals about the model', { b: true }) + S.rect(bx, by, bw, bh, { f: C.white, s: C.silver });
    g += S.line(bx, by + bh, bx + bw, by, { c: C.bg, d: '5 4' });
    pts.forEach(function(p){ var t = p[0], mp = Math.max(0.05, Math.min(0.95, t + (p[1] - 0.5) * 0.25 + (r() - 0.5) * 0.03)); g += S.line(px(t, bx), py(p[1]), px(t, bx), py(mp), { c: C.red, w: 1.6 }) + S.circle(px(t, bx), py(p[1]), 3.6, { f: C.char }); });
    g += S.text(bx + bw / 2, by + bh + 16, 'measured value (x), predicted value (y)', { s: 11 });
    g += S.text(bx2 + bw / 2, 20, 'Residuals about the mean', { b: true }) + S.rect(bx2, by, bw, bh, { f: C.white, s: C.silver });
    var my = by + bh / 2; g += S.line(bx2, my, bx2 + bw, my, { c: C.char, w: 1.6, d: '6 4' });
    pts.forEach(function(p){ g += S.line(px(p[0], bx2), py(p[1]), px(p[0], bx2), my, { c: C.red, w: 1.6 }) + S.circle(px(p[0], bx2), py(p[1]), 3.6, { f: C.char }); });
    g += S.text(bx2 + bw / 2, by + bh + 16, 'measured value (x), measured value (y)', { s: 11 });
    g += S.line(bx2 + 20, by + bh + 34, bx2 + 44, by + bh + 34, { c: C.char, w: 1.6, d: '6 4' }) + S.text(bx2 + 50, by + bh + 38, 'mean of the measured values', { a: 'start', s: 11 });
    g += S.line(bx + 20, by + bh + 34, bx + 44, by + bh + 34, { c: C.red, w: 1.8 }) + S.text(bx + 50, by + bh + 38, 'residual', { a: 'start', s: 11 });
    g += S.text(W / 2, 205, 'R2 = 1 minus A divided by B', { b: true, s: 13 }) + S.text(W / 2, 222, 'A: sum of squared residuals of the model.  B: sum of squared residuals about the mean.', { s: 11.5 });
    return S.svg(W, H, g, 'Residuals about the model and residuals about the mean of the measured values, with the definition of R2');
  };

  /* 4. compounds grouped by ring scaffold */
  QZFIG.scaffoldGroups = function(){
    var W = 600, H = 242, g = '', names = ['A', 'B', 'C', 'D', 'E', 'F'], k, cw = 170, ch = 78;
    for (k = 0; k < 6; k++) {
      var cx = 20 + (k % 3) * 190, cy = 22 + Math.floor(k / 3) * 98, hx = cx + 20, hy = cy + 22, pts = '';
      g += S.rect(cx, cy, cw, ch, { f: C.light, s: C.bg });
      for (var a = 0; a < 6; a++) pts += (hx + 13 * Math.cos(a * Math.PI / 3 + 0.5)).toFixed(1) + ',' + (hy + 13 * Math.sin(a * Math.PI / 3 + 0.5)).toFixed(1) + ' ';
      g += '<polygon points="' + pts + '" fill="none" stroke="' + C.char + '" stroke-width="1.6"/>' + S.text(cx + 44, cy + 22 + 4, 'Scaffold ' + names[k], { a: 'start', b: true, s: 12 });
      g += dots(cx + 70, cy + 30, cw - 80, ch - 38, 4, 40 + k, { f: C.mid }) + dots(cx + 14, cy + 44, 60, ch - 52, 3, 90 + k, { f: C.mid });
    }
    g += cap2('Each box holds the compounds that share one ring scaffold.', 'A split assigns compounds to the training set or the test set.', W, 218);
    return S.svg(W, H, g, 'Six groups of compounds, each sharing a ring scaffold');
  };

  /* 5. histogram of measured values with the pile at the assay limit */
  QZFIG.censoredHist = function(){
    var W = 600, H = 236, g = '', bx = 70, by = 34, bw = 480, bh = 150, heights = [1, 0.12, 0.16, 0.2, 0.22, 0.24, 0.24, 0.22, 0.2, 0.17, 0.14, 0.11, 0.08, 0.06, 0.04, 0.03, 0.02], n = heights.length, k, bwd = bw / n;
    g += S.line(bx, by + bh, bx + bw, by + bh, { c: C.mid }) + S.line(bx, by, bx, by + bh, { c: C.mid });
    for (k = 0; k < n; k++) { var hh = heights[k] * bh * 0.95; g += S.rect(bx + k * bwd + 1, by + bh - hh, bwd - 2, hh, { f: k === 0 ? C.red : C.bg, s: k === 0 ? C.dred : C.mid, r: 0, w: 1 }); }
    g += S.line(bx, by - 4, bx, by + bh + 6, { c: C.char, w: 2, d: '4 3' }) + S.text(bx, by - 10, 'assay lower limit', { a: 'middle', s: 11, b: true });
    g += S.text(bx + bwd + 12, by + 28, '31% of the compounds are reported at the lowest value', { a: 'start', s: 11, c: C.dred, b: true });
    g += S.text(bx + bw / 2, by + bh + 22, 'measured log10 HLM clearance', { b: true, s: 12 });
    g += '<text transform="translate(18 ' + (by + bh / 2) + ') rotate(-90)" text-anchor="middle" fill="' + C.char + '" font-weight="700">compounds</text>';
    g += cap('Distribution of the reported HLM clearance values.', W, 224);
    return S.svg(W, H, g, 'Histogram of reported clearance values with a tall bar at the assay lower limit');
  };

  /* 6. one bar for all compounds with the floor share marked */
  QZFIG.floorBar = function(o){
    var W = 600, H = 150, g = '', x0 = 40, w = 520, f = o.share === undefined ? 0.31 : o.share, y = 56, hh = 40, k;
    g += S.text(x0, 40, o.total || '3,087 compounds', { a: 'start', b: true }) + S.line(x0, 46, x0 + w, 46, { c: C.mid }) + S.line(x0, 42, x0, 50, { c: C.mid }) + S.line(x0 + w, 42, x0 + w, 50, { c: C.mid });
    g += S.rect(x0, y, w * f, hh, { f: C.red, s: C.dred, r: 0 }) + S.rect(x0 + w * f, y, w * (1 - f), hh, { f: C.bg, s: C.mid, r: 0 });
    g += S.text(x0 + w * f / 2, y + hh / 2 + 4, '31% at the assay floor', { c: C.white, b: true, s: 11 }) + S.text(x0 + w * f + w * (1 - f) / 2, y + hh / 2 + 4, 'above the floor', { b: true, s: 11 });
    for (k = 0; k <= 4; k++) { var xx = x0 + w * k / 4; g += S.line(xx, y + hh, xx, y + hh + 6, { c: C.mid }) + S.text(xx, y + hh + 20, (k * 25) + '%', { s: 11 }); }
    g += cap('The bar has the length of the whole data set.', W, 142);
    return S.svg(W, H, g, 'A bar for the whole data set with the share at the assay floor shaded');
  };

  /* 7. the same compounds ordered by measured and by predicted value */
  QZFIG.rankColumns = function(){
    var W = 600, H = 232, g = '', xl = 200, xr = 400, top = 36, bot = 190, names = ['A', 'B', 'C', 'D', 'E', 'F'], meas = [0.6, 1.0, 1.4, 1.9, 2.3, 2.7], pred = [1.15, 1.3, 1.45, 1.55, 1.7, 1.85];
    function yy(v, lo, hi){ return bot - (v - lo) / (hi - lo) * (bot - top); }
    g += S.text(xl, 22, 'Measured value', { b: true }) + S.text(xr, 22, 'Predicted value', { b: true }) + S.line(xl, top, xl, bot, { c: C.mid }) + S.line(xr, top, xr, bot, { c: C.mid });
    names.forEach(function(n, k){ var ya = yy(meas[k], 0.4, 2.9), yb = yy(pred[k], 1.0, 2.0); g += S.line(xl, ya, xr, yb, { c: C.bg, w: 1.6 }) + S.circle(xl, ya, 5, { f: C.char }) + S.circle(xr, yb, 5, { f: C.red, s: C.dred }) + S.text(xl - 14, ya + 4, n, { a: 'end', b: true }) + S.text(xl - 28, ya + 4, '', {}) + S.text(xr + 14, yb + 4, n, { a: 'start', b: true }); });
    g += S.text(xl - 40, bot + 4, 'low', { a: 'end', s: 11, c: C.mid }) + S.text(xl - 40, top + 4, 'high', { a: 'end', s: 11, c: C.mid });
    g += S.text(xr + 60, bot + 4, 'low', { a: 'start', s: 11, c: C.mid }) + S.text(xr + 60, top + 4, 'high', { a: 'start', s: 11, c: C.mid });
    g += cap('Six compounds placed on a measured scale and on a predicted scale, joined by compound.', W, 222);
    return S.svg(W, H, g, 'Six compounds on a measured scale and a predicted scale, joined by lines');
  };

  /* 8. a logD model feeding an HLM model */
  QZFIG.stackFlow = function(){
    var W = 600, H = 226, g = '';
    g += S.rect(14, 22, 150, 52, { f: C.light }) + S.text(89, 44, 'logD data set', { b: true }) + S.text(89, 62, '4,200 measured values', { s: 11 });
    g += S.arrow(166, 48, 216, 48) + S.rect(218, 22, 150, 52, { f: C.white, w: 2 }) + S.text(293, 44, 'logD model', { b: true }) + S.text(293, 62, 'trained on logD data', { s: 11 });
    g += S.arrow(370, 48, 420, 48) + S.rect(422, 22, 164, 52, { f: C.light, s: C.red, w: 2 }) + S.text(504, 44, 'Predicted logD', { b: true, c: C.dred }) + S.text(504, 62, 'for each HLM compound', { s: 11 });
    g += S.line(504, 76, 504, 112, { c: C.char }) + S.arrow(504, 112, 504, 128, { c: C.char });
    g += S.rect(14, 128, 200, 60, { f: C.light }) + S.text(114, 150, 'HLM descriptors', { b: true }) + S.text(114, 170, 'logP, TPSA, MW and others', { s: 11 });
    g += S.text(228, 163, '+', { b: true, s: 20 }) + S.rect(246, 128, 150, 60, { f: C.light, s: C.red, w: 2 }) + S.text(321, 150, 'logD_pred', { b: true, c: C.dred }) + S.text(321, 170, 'one more feature', { s: 11 });
    g += S.line(396, 158, 420, 158, { c: C.char }) + S.arrow(420, 158, 440, 158) + S.rect(442, 128, 144, 60, { f: C.white, w: 2 }) + S.text(514, 154, 'HLM model', { b: true }) + S.text(514, 172, 'predicts clearance', { s: 11 });
    g += S.line(504, 128, 504, 128, {}) + cap('Two models trained on two data sets, connected through one predicted value.', W, 214);
    return S.svg(W, H, g, 'A logD model whose prediction becomes an input feature of an HLM model');
  };

  /* 9. data sets and pipeline operations */
  QZFIG.leakPipeline = function(){
    var W = 600, H = 232, g = '';
    g += S.text(20, 20, 'Data sets', { a: 'start', b: true });
    g += S.rect(20, 28, 370, 76, { f: C.light }) + S.text(34, 46, 'Training set', { a: 'start', b: true }) + S.rect(40, 54, 150, 40, { f: C.white, d: '5 3', s: C.mid }) + S.text(115, 78, 'Validation set', { b: true, s: 12 }) + S.text(202, 78, 'cut from the training compounds', { a: 'start', s: 11 });
    g += S.rect(420, 28, 160, 76, { f: C.light, s: C.red, w: 2 }) + S.text(500, 72, 'Test set', { b: true, c: C.dred });
    g += S.text(20, 134, 'Operations in a modelling workflow', { a: 'start', b: true });
    ['Standardise features', 'Fit the model', 'Choose settings', 'Report the score'].forEach(function(t, k){ g += S.rect(20 + k * 143, 144, 132, 44, { f: C.white, w: 1.6 }) + S.text(86 + k * 143, 171, t, { b: true, s: 11.5 }); });
    g += cap('Each operation uses one or more of the data sets.', W, 214);
    return S.svg(W, H, g, 'A training set containing a validation set, a separate test set, and four modelling operations');
  };

  /* 10. a compound routed down a decision tree to a leaf */
  QZFIG.treeRoute = function(){
    var W = 600, H = 240, g = '', hi = C.red;
    function node(x, y, t, on){ return S.rect(x - 62, y - 15, 124, 30, { f: C.white, s: on ? hi : C.char, w: on ? 2.4 : 1.4 }) + S.text(x, y + 4, t, { s: 11.5, b: true }); }
    function leaf(x, y, vals, on){ return S.rect(x - 56, y - 18, 112, 50, { f: on ? '#FBE9E9' : C.light, s: on ? hi : C.bg, w: on ? 2.4 : 1.4 }) + S.text(x, y - 3, 'leaf', { s: 11, b: true }) + S.text(x, y + 14, vals, { s: 11 }); }
    g += S.text(20, 22, 'Compound: logP 2.1, TPSA 74', { a: 'start', s: 11.5, b: true });
    g += S.line(300, 50, 190, 108, { c: hi, w: 2.4 }) + S.line(300, 50, 450, 108, { c: C.bg, w: 1.4 }) + S.line(190, 108, 110, 168, { c: C.bg, w: 1.4 }) + S.line(190, 108, 270, 168, { c: hi, w: 2.4 });
    g += node(300, 36, 'logP at most 2.9?', true) + S.text(228, 74, 'yes', { s: 11, c: C.dred, b: true }) + S.text(392, 74, 'no', { s: 11 });
    g += node(190, 108, 'TPSA at most 60?', true) + S.text(130, 142, 'yes', { s: 11 }) + S.text(252, 142, 'no', { s: 11, c: C.dred, b: true });
    g += leaf(110, 186, '0.8  0.9  1.1', false) + leaf(270, 186, '1.2  1.5  1.6  1.7', true) + leaf(450, 140, '1.9  2.2  2.4', false);
    g += S.arrow(328, 204, 452, 204, { c: C.char }) + S.rect(454, 184, 126, 40, { f: C.white, w: 2 }) + S.text(517, 202, 'Prediction', { b: true }) + S.text(517, 217, 'for the compound', { s: 11 });
    g += cap('Numbers in a leaf are measured values of training compounds.', W, 236);
    return S.svg(W, H, g, 'A compound routed down a decision tree to a leaf that contains measured values of training compounds');
  };

  /* 11. a tree of depth 4 and an empty R2 axis with two depths marked */
  QZFIG.treeDepths = function(){
    var W = 600, H = 232, g = '', t4 = tree(150, 36, 4, 72, 36), ax = 350, ay = 30, aw = 220, ah = 140;
    g += S.text(150, 18, 'Tree of depth 4', { b: true }) + t4.svg;
    t4.leaves.forEach(function(p){ g += S.rect(p[0] - 5, p[1] - 5, 10, 10, { f: C.bg, s: C.mid, r: 1, w: 1 }); });
    g += S.text(150, 214, 'Four questions in a row', { s: 11 });
    g += S.text(ax + aw / 2, 18, 'R2 against tree depth', { b: true }) + S.rect(ax, ay, aw, ah, { f: C.white, s: C.silver }) + S.line(ax + 50, ay, ax + 50, ay + ah, { c: C.red, d: '4 3', w: 1.6 }) + S.line(ax + 170, ay, ax + 170, ay + ah, { c: C.red, d: '4 3', w: 1.6 });
    g += S.text(ax + 50, ay + ah + 18, 'depth 4', { s: 11, b: true }) + S.text(ax + 170, ay + ah + 18, 'depth 12', { s: 11, b: true }) + S.text(ax + aw / 2, ay + ah + 36, 'tree depth', { s: 11 });
    g += '<text transform="translate(' + (ax - 12) + ' ' + (ay + ah / 2) + ') rotate(-90)" text-anchor="middle" fill="' + C.char + '" font-size="11">R2 (training and test)</text>';
    return S.svg(W, H, g, 'A tree of depth 4 and an R2 axis with depth 4 and depth 12 marked');
  };

  /* 12. bootstrap samples, one tree each, predictions on a number line */
  QZFIG.baggingFlow = function(){
    var W = 600, H = 232, g = '', preds = [1.15, 1.95, 1.45], lo = 0.5, hi = 2.5, nx = 400, nw = 180, k;
    function vx(v){ return nx + (v - lo) / (hi - lo) * nw; }
    g += S.rect(10, 20, 100, 170, { f: C.light }) + S.text(60, 14, 'Training set', { b: true, s: 11.5 }) + dots(10, 20, 100, 170, 22, 61);
    for (k = 0; k < 3; k++) {
      var y = 24 + k * 58;
      g += S.arrow(112, 105, 150, y + 22) + S.rect(152, y, 88, 44, { f: C.light, s: C.mid }) + dots(152, y, 88, 44, 7 + (k % 2), 70 + k) + S.text(196, y - 4, 'Sample ' + (k + 1), { s: 11 });
      g += S.arrow(242, y + 22, 276, y + 22) + S.rect(278, y + 4, 80, 36, { f: C.white, w: 1.6 }) + S.text(318, y + 27, 'Tree ' + (k + 1), { b: true, s: 12 });
      g += S.arrow(360, y + 22, 392, y + 22) + S.line(nx, y + 22, nx + nw, y + 22, { c: C.silver }) + S.circle(vx(preds[k]), y + 22, 6, { f: C.red, s: C.dred }) + S.text(vx(preds[k]) + (k === 2 ? -14 : 0), y + (k === 2 ? 40 : 8), 'tree ' + (k + 1), { s: 11 });
    }
    g += S.line(vx(1.5), 12, vx(1.5), 188, { c: C.char, w: 1.6, d: '5 3' }) + S.text(vx(1.5), 10, 'measured', { s: 11, b: true });
    [0.5, 1.5, 2.5].forEach(function(v){ g += S.line(vx(v), 186, vx(v), 192, { c: C.mid }) + S.text(vx(v), 204, v.toFixed(1), { s: 11 }); });
    g += S.rect(392, 210, 196, 20, { f: C.white, s: C.mid, w: 1 }) + S.text(490, 224, 'Forest: mean of the trees', { s: 11, b: true });
    return S.svg(W, H, g, 'Three bootstrap samples of the training set, one tree fitted to each, and the three tree predictions for one compound');
  };

  /* 13. boosting rounds: model, data and residual bars */
  QZFIG.boostRounds = function(){
    var W = 600, H = 226, g = '', r = rnd(5), n = 10, xs = [], ys = [], i, trend = function(t){ return 0.5 + 0.3 * Math.sin(t * 5.2); };
    for (i = 0; i < n; i++) { xs.push((i + 0.5) / n); ys.push(Math.max(0.05, Math.min(0.95, trend((i + 0.5) / n) + (r() - 0.5) * 0.2))); }
    var panels = [{ t: 'Training mean', lv: 1 }, { t: 'After tree 1', lv: 2 }, { t: 'After tree 2', lv: 4 }], pw = 168, ph = 130, py0 = 40;
    panels.forEach(function(p, k){
      var ox = 12 + k * 200, model = function(t){ var seg = Math.min(p.lv - 1, Math.floor(t * p.lv)), a = seg / p.lv, b = (seg + 1) / p.lv, m = 0, c = 0; for (var q = 0; q < n; q++) if (xs[q] >= a && xs[q] < b) { m += ys[q]; c++; } return c ? m / c : 0.5; };
      g += S.text(ox + pw / 2, 28, p.t, { b: true }) + S.rect(ox, py0, pw, ph, { f: C.white, s: C.silver });
      var d = '';
      for (var s = 0; s < p.lv; s++) { var xa = ox + (s / p.lv) * pw, xb = ox + ((s + 1) / p.lv) * pw, my = py0 + ph - model((s + 0.5) / p.lv) * ph; d += (s ? ' L' : 'M') + xa + ' ' + my + ' L' + xb + ' ' + my; }
      for (i = 0; i < n; i++) { var px = ox + xs[i] * pw, py = py0 + ph - ys[i] * ph, my2 = py0 + ph - model(xs[i]) * ph; g += S.line(px, py, px, my2, { c: C.red, w: 1.6 }) + S.circle(px, py, 3.4, { f: C.char }); }
      g += S.path(d, { c: C.char, w: 1.8 });
      if (k < 2) g += S.arrow(ox + pw + 4, py0 + ph / 2, ox + pw + 28, py0 + ph / 2) + S.text(ox + pw + 16, py0 + ph / 2 - 8, 'tree ' + (k + 1), { s: 11 });
    });
    g += S.line(40, 190, 56, 190, { c: C.char, w: 1.8 }) + S.text(62, 194, 'model', { a: 'start', s: 11 }) + S.line(120, 190, 136, 190, { c: C.red, w: 1.8 }) + S.text(142, 194, 'distance from each compound to the model', { a: 'start', s: 11 });
    g += cap('The model after each round of boosting, with the measured values.', W, 216);
    return S.svg(W, H, g, 'Three panels showing a boosted model after zero, one and two trees, with measured values and vertical segments to the model');
  };

  /* 14. the update rule of boosting */
  QZFIG.boostUpdate = function(){
    var W = 600, H = 222, g = '', bx = [{ x: 10, w: 114, t: 'Prediction after', s: 'this round', c: C.red }, { x: 160, w: 114, t: 'Prediction before', s: 'this round', c: C.char }, { x: 318, w: 108, t: 'Learning rate', s: 'set before training', c: C.char }, { x: 472, w: 118, t: 'Tree output', s: 'new tree correction', c: C.char }];
    bx.forEach(function(b){ g += S.rect(b.x, 40, b.w, 62, { f: C.light, s: b.c, w: 2 }) + S.text(b.x + b.w / 2, 66, b.t, { b: true, s: 12 }) + S.text(b.x + b.w / 2, 86, b.s, { s: 11 }); });
    g += S.text(142, 78, '=', { b: true, s: 22 }) + S.text(296, 78, '+', { b: true, s: 22 }) + S.text(449, 76, 'times', { s: 11.5, b: true });
    g += S.text(W / 2, 28, 'Update of the prediction for one compound in one round', { b: true });
    var k; for (k = 0; k < 5; k++) { g += S.circle(70 + k * 115, 150, 15, { f: k === 0 ? C.light : C.white, w: 1.6 }) + S.text(70 + k * 115, 155, String(k + 1), { b: true }); if (k < 4) g += S.arrow(87 + k * 115, 150, 53 + (k + 1) * 115, 150); }
    g += S.text(W / 2, 184, 'Rounds: the same update is applied in each round, with a new tree each time', { s: 11.5 });
    g += cap('Quantities in the update rule of gradient boosting.', W, 210);
    return S.svg(W, H, g, 'The boosting update: prediction after equals prediction before plus learning rate times tree output');
  };

  /* 15. a tree drawn to a given number of levels */
  QZFIG.treeLevels = function(o){
    var W = 600, H = 236, g = '', d = o.depth || 3, t = tree(300, 36, d, 130, 46), lv;
    g += t.svg;
    for (lv = 0; lv < d; lv++) g += S.text(20, 40 + lv * 46, 'question ' + (lv + 1), { a: 'start', s: 11, c: C.mid });
    t.leaves.forEach(function(p){ g += S.rect(p[0] - 11, p[1] - 8, 22, 16, { f: C.light, s: C.char, w: 1.4, r: 2 }); });
    g += S.rect(20, 36 + d * 46 - 8, 22, 16, { f: C.light, s: C.char, w: 1.4, r: 2 }) + S.text(50, 40 + d * 46, 'leaf', { a: 'start', s: 11, c: C.mid });
    g += S.text(W / 2, 36 + d * 46 + 34, 'Depth ' + d + ': ' + d + ' questions in a row. Each question splits a group in two.', { b: true, s: 12 });
    g += cap('A tree drawn to depth ' + d + '.', W, 36 + d * 46 + 56);
    return S.svg(W, H, g, 'A binary tree drawn to ' + d + ' levels');
  };

  (function(){ var B = QZ.bank; function set(i, name, o){ B[i - 1].fig = { name: name, o: o || {} }; }
    set(1, 'splitDots'); set(2, 'trainTestFit', { train: 0.95, test: 0.2 }); set(3, 'r2Residuals'); set(4, 'scaffoldGroups'); set(5, 'censoredHist'); set(6, 'floorBar', { total: '3,087 compounds', share: 0.31 });
    set(7, 'rankColumns'); set(8, 'stackFlow'); set(9, 'leakPipeline'); set(10, 'treeRoute'); set(11, 'treeDepths'); set(12, 'baggingFlow'); set(13, 'boostRounds'); set(14, 'boostUpdate'); set(15, 'treeLevels', { depth: 3 });
  })();
})();
