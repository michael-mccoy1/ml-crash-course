"use strict";
/* Quiz schematics for the neural network and optimiser questions (bank indices 16 to 22 and 44 to 54).
   Each figure shows the quantities in the question. None of them states the answer. */
(function(){
  var S = QZSV, C = S.C;

  /* drawing helpers local to these figures */
  function arr(x1, y1, x2, y2, o){
    o = o || {}; var c = o.c || C.char, w = o.w || 1.6, a = Math.atan2(y2 - y1, x2 - x1), h = o.h || 8, bx = x2 - h * Math.cos(a), by = y2 - h * Math.sin(a);
    return '<line x1="' + x1 + '" y1="' + y1 + '" x2="' + bx.toFixed(1) + '" y2="' + by.toFixed(1) + '" stroke="' + c + '" stroke-width="' + w + '"' + (o.d ? ' stroke-dasharray="' + o.d + '"' : '') + '/>' +
      '<polygon points="' + x2 + ',' + y2 + ' ' + (bx - h * 0.45 * Math.sin(a)).toFixed(1) + ',' + (by + h * 0.45 * Math.cos(a)).toFixed(1) + ' ' + (bx + h * 0.45 * Math.sin(a)).toFixed(1) + ',' + (by - h * 0.45 * Math.cos(a)).toFixed(1) + '" fill="' + c + '"/>';
  }
  /* text with a halo so that it stays legible over lines */
  function lab(x, y, t, o){ o = o || {}; return '<text x="' + x + '" y="' + y + '" text-anchor="' + (o.a || 'middle') + '" fill="' + (o.c || C.char) + '" font-size="' + (o.s || 12) + '"' + (o.b ? ' font-weight="700"' : '') + ' stroke="' + C.light + '" stroke-width="3.5" paint-order="stroke" stroke-linejoin="round">' + esc(t) + '</text>'; }
  function curve(fn, x0, x1, px0, px1, y0, y1, py0, py1, n){
    var d = '', i, x, y, X, Y; n = n || 80;
    for (i = 0; i <= n; i++) { x = x0 + (x1 - x0) * i / n; y = fn(x); y = Math.max(y0, Math.min(y1, y)); X = px0 + (x - x0) / (x1 - x0) * (px1 - px0); Y = py1 - (y - y0) / (y1 - y0) * (py1 - py0); d += (i ? 'L' : 'M') + X.toFixed(1) + ' ' + Y.toFixed(1); }
    return d;
  }
  function frame(px0, py0, px1, py1, xl, yl, xo){
    return arr(px0, py1, px1 + 10, py1, { c: C.char, w: 1.2, h: 7 }) + arr(px0, py1, px0, py0 - 10, { c: C.char, w: 1.2, h: 7 }) + S.text((px0 + px1) / 2, py1 + (xo || 28), xl, { s: 12 }) +
      '<text transform="translate(' + (px0 - 16) + ' ' + ((py0 + py1) / 2) + ') rotate(-90)" text-anchor="middle" font-size="12" fill="' + C.char + '">' + esc(yl) + '</text>';
  }
  function legend(x, y, items){
    return items.map(function(it, k){ var yy = y + k * 18; return S.line(x, yy, x + 22, yy, { c: it[0], w: 2.6, d: it[2] }) + S.text(x + 30, yy + 4, it[1], { a: 'start', s: 12 }); }).join('');
  }

  /* layered network: lines are weights, circles are units */
  function layered(cx0, cx1, top, bot, sizes){
    var k, i, j, xs = [], pos = [], out = '', n = sizes.length, maxn = Math.max.apply(null, sizes);
    for (k = 0; k < n; k++) xs.push(n === 1 ? (cx0 + cx1) / 2 : cx0 + (cx1 - cx0) * k / (n - 1));
    for (k = 0; k < n; k++) { var m = sizes[k], sp = m > 1 ? Math.min(24, (bot - top) / (m - 1)) : 0, mid = (top + bot) / 2, col = []; for (i = 0; i < m; i++) col.push([xs[k], mid + (i - (m - 1) / 2) * sp]); pos.push(col); }
    var r = Math.max(3.5, Math.min(10, ((bot - top) / Math.max(1, maxn - 1)) * 0.42));
    for (k = 0; k < n - 1; k++) for (i = 0; i < sizes[k]; i++) for (j = 0; j < sizes[k + 1]; j++) out += S.line(pos[k][i][0], pos[k][i][1], pos[k + 1][j][0], pos[k + 1][j][1], { c: C.silver, w: 0.9 });
    for (k = 0; k < n; k++) for (i = 0; i < sizes[k]; i++) out += S.circle(pos[k][i][0], pos[k][i][1], r, { f: k === 0 ? C.light : (k === n - 1 ? C.red : C.bg), s: C.char, w: 1.2 });
    return { svg: out, xs: xs };
  }

  /* 16, 17: parameter counting setup */
  QZFIG.mlpLayers = function(o){
    var sz = o.sizes, L = layered(90, 510, 28, 188, sz), k, t = '';
    for (k = 0; k < sz.length; k++) t += S.text(L.xs[k], 208, (k === 0 ? 'Input' : (k === sz.length - 1 ? 'Output' : 'Hidden ' + k)) + ': ' + sz[k] + (sz[k] === 1 ? ' unit' : ' units'), { s: 12, b: true });
    t += S.text(300, 226, 'Each line is one weight. Each hidden and output unit has one bias.', { s: 11.5, c: C.mid });
    return S.svg(600, 232, L.svg + t, 'Fully connected network with layer sizes ' + sz.join(', '));
  };

  /* 18: a function between two linear layers, and two candidate functions */
  QZFIG.activationCompare = function(){
    var t = '', bx = 14, by = 28;
    t += S.text(190, 22, 'Two layers with a function between them', { b: true, s: 12 });
    t += S.rect(bx, by + 40, 70, 34, { f: C.light }) + S.text(bx + 35, by + 62, 'Input x', { s: 12 });
    t += arr(bx + 70, by + 57, bx + 94, by + 57);
    t += S.rect(bx + 94, by + 40, 70, 34, { f: C.bg }) + S.text(bx + 129, by + 62, 'Layer 1', { s: 12 });
    t += arr(bx + 164, by + 57, bx + 174, by + 57, { h: 6 });
    t += S.rect(bx + 174, by + 40, 58, 34, { f: C.white, s: C.red, d: '4 3', w: 1.6 }) + S.text(bx + 203, by + 62, 'f(z)', { s: 12, c: C.red, b: true });
    t += arr(bx + 232, by + 57, bx + 246, by + 57, { h: 6 });
    t += S.rect(bx + 246, by + 40, 70, 34, { f: C.bg }) + S.text(bx + 281, by + 62, 'Layer 2', { s: 12 });
    t += arr(bx + 316, by + 57, bx + 338, by + 57) + S.text(bx + 350, by + 62, 'y', { s: 12 });
    t += S.text(bx + 176, by + 112, 'Each layer computes a weighted sum plus a bias.', { s: 11.5, c: C.mid });
    t += S.text(bx + 176, by + 130, 'f(z) is a function applied between the layers.', { s: 11.5, c: C.mid });
    var px0 = 430, px1 = 570, py0 = 40, py1 = 150, zx = px0 + (px1 - px0) / 2, zy = py0 + (py1 - py0) / 2;
    t += arr(px0 - 6, zy, px1 + 10, zy, { w: 1.2, h: 7 }) + arr(zx, py1 + 6, zx, py0 - 8, { w: 1.2, h: 7 }) + S.text(px1 + 10, zy + 16, 'z', { s: 12 }) + S.text(zx + 14, py0 - 4, 'f(z)', { s: 12 });
    t += S.path(curve(function(z){ return z; }, -2, 2, px0, px1, -2, 2, py0, py1), { c: C.mid, d: '5 4', w: 2 });
    t += S.path(curve(function(z){ return Math.max(0, z); }, -2, 2, px0, px1, -2, 2, py0, py1), { c: C.red, w: 2.8 });
    t += legend(440, 182, [[C.mid, 'f(z) = z', '5 4'], [C.red, 'f(z) = max(0, z)']]);
    return S.svg(600, 232, t, 'Two layers with a function between them, and the plot of two candidate functions');
  };

  /* 19: training and validation loss; the minimum of the validation loss is not marked */
  QZFIG.lossCurves = function(){
    var px0 = 70, px1 = 540, py0 = 24, py1 = 170, t = '';
    t += frame(px0, py0, px1, py1, 'Epoch', 'Loss');
    t += S.path(curve(function(e){ return 0.12 + 0.95 * Math.exp(-e / 16); }, 0, 80, px0, px1, 0, 1.1, py0, py1), { c: C.char, w: 2.8 });
    t += S.path(curve(function(e){ return 0.38 + 0.6 * Math.exp(-e / 9) + 0.000075 * Math.pow(Math.max(0, e - 20), 2); }, 0, 80, px0, px1, 0, 1.1, py0, py1), { c: C.red, w: 2.8 });
    t += legend(380, 40, [[C.char, 'Training loss'], [C.red, 'Validation loss']]);
    t += S.text(300, 224, 'Loss on the training compounds and on compounds held out during training', { s: 11.5, c: C.mid });
    return S.svg(600, 232, t, 'Training loss and validation loss against epoch');
  };

  /* 20: a network with two hidden layers and a dropout rate; no unit is marked */
  QZFIG.dropoutNet = function(){
    var sz = [4, 6, 6, 1], L = layered(90, 510, 50, 180, sz), t = '', names = ['Input', 'Hidden 1', 'Hidden 2', 'Output'], k;
    for (k = 0; k < 4; k++) t += S.text(L.xs[k], 200, names[k], { s: 12, b: true });
    t += S.rect(L.xs[1] - 44, 14, L.xs[2] - L.xs[1] + 88, 18, { f: C.white, s: C.red, d: '4 3', w: 1.4 }) + S.text((L.xs[1] + L.xs[2]) / 2, 27, 'dropout rate 0.2 on both hidden layers', { s: 12, c: C.red, b: true });
    t += S.text(300, 222, 'Network used during training.', { s: 11.5, c: C.mid });
    return S.svg(600, 232, L.svg + t, 'Network with two hidden layers and a dropout rate of 0.2');
  };

  /* shared loss curve of one weight */
  function valley(extra){
    var px0 = 90, px1 = 510, py0 = 26, py1 = 160, t = '', f = function(w){ return 0.5 * w * w + 0.15; };
    t += frame(px0, py0, px1, py1, 'Weight w', 'Loss L', 26);
    t += S.path(curve(f, -3, 3, px0, px1, 0, 4.7, py0, py1, 100), { c: C.red, w: 2.8 });
    t += extra(function(w){ return px0 + (w + 3) / 6 * (px1 - px0); }, function(l){ return py1 - l / 4.7 * (py1 - py0); }, f);
    return t;
  }

  /* 44: the slope of the loss at the current weight; the gradient arrow is not drawn */
  QZFIG.gradientValley = function(){
    var t = valley(function(X, Y, f){
      var w = -1.9, m = w, x0 = w - 0.8, x1 = w + 0.9, s = '';
      s += S.line(X(x0), Y(f(w) + m * (x0 - w)), X(x1), Y(f(w) + m * (x1 - w)), { c: C.char, w: 1.6, d: '5 4' });
      s += S.circle(X(w), Y(f(w)), 6, { f: C.char, s: C.char });
      s += lab(X(w) - 12, Y(f(w)) + 22, 'current weight', { a: 'end' });
      s += S.text(300, 206, 'Dashed line: tangent at the current weight. Its slope is dL/dw.', { s: 12 });
      s += arr(300, 224, 372, 224) + S.text(380, 228, 'w increases', { a: 'start', s: 12 });
      s += arr(300, 224, 228, 224) + S.text(220, 228, 'w decreases', { a: 'end', s: 12 });
      return s;
    });
    return S.svg(600, 232, t, 'Loss against one weight, with the tangent at the current weight');
  };

  /* 21: one update; its size is the quantity asked about, so the new weight is only indicated */
  QZFIG.valleyStep = function(){
    var t = valley(function(X, Y, f){
      var w = -2.2, w2 = -0.4, s = '';
      s += S.circle(X(w), Y(f(w)), 6, { f: C.char, s: C.char }) + lab(X(w) + 6, Y(f(w)) + 26, 'current w', { a: 'end', s: 12 });
      s += arr(X(w) + 8, Y(f(w)) - 6, X(w2), Y(f(w)) - 6, { c: C.red, w: 2 });
      s += lab((X(w) + X(w2)) / 2, Y(f(w)) - 16, 'update to w', { c: C.red });
      s += S.text(300, 218, 'update = learning rate times gradient', { s: 12, b: true });
      return s;
    });
    return S.svg(600, 232, t, 'One gradient descent update on a loss curve');
  };

  /* 45: successive updates of one weight */
  QZFIG.valleySteps = function(){
    var t = valley(function(X, Y, f){
      var ws = [-2.6, -1.7, -1.05, -0.6, -0.3], s = '', i;
      for (i = 0; i < ws.length - 1; i++) s += arr(X(ws[i]) + 5, Y(f(ws[i])) - 7, X(ws[i + 1]) - 3, Y(f(ws[i + 1])) - 9, { c: C.char, w: 1.5, h: 6 });
      for (i = 0; i < ws.length; i++) s += S.circle(X(ws[i]), Y(f(ws[i])), 5, { f: i ? C.white : C.char, s: C.char });
      for (i = 0; i < ws.length; i++) s += lab(X(ws[i]) - 12, Y(f(ws[i])) + 4 - (i ? 0 : 2), 'w' + i, { a: 'end', s: 12 });
      s += S.text(300, 218, 'Gradient descent: successive updates of one weight', { s: 12, c: C.mid });
      return s;
    });
    return S.svg(600, 232, t, 'Successive updates of one weight on a loss curve');
  };

  /* elongated contours for 46 and 47 */
  function contours(extra){
    var t = '', k, cx = 330, cy = 100;
    for (k = 1; k <= 4; k++) t += '<ellipse cx="' + cx + '" cy="' + cy + '" rx="' + (k * 62) + '" ry="' + (k * 15) + '" fill="none" stroke="' + C.silver + '" stroke-width="1.8"/>';
    t += S.circle(cx, cy, 4.5, { f: C.char, s: C.char }) + lab(cx, cy - 10, 'minimum', { s: 12 });
    t += extra(cx, cy);
    return t;
  }
  /* 46: the gradient has a different size along each direction */
  QZFIG.contourGrad = function(){
    var t = contours(function(){
      var x = 150, y = 128, s = '';
      s += S.circle(x, y, 5, { f: C.char, s: C.char });
      s += arr(x, y, x, y + 56, { c: C.red, w: 2.4 }) + lab(x + 10, y + 36, 'gradient component across the valley', { a: 'start', c: C.red });
      s += arr(x, y, x + 24, y, { c: C.mid, w: 2.4 }) + lab(x + 32, y - 6, 'component along the valley', { a: 'start', c: C.mid });
      return s;
    });
    t += S.text(300, 222, 'Contours of a loss over two weights. The gradient at a point has a component along each weight.', { s: 11.5, c: C.mid });
    return S.svg(600, 232, t, 'Elongated loss contours with the components of the gradient at one point');
  };
  /* 47: successive plain gradient steps across a narrow valley */
  QZFIG.contourPath = function(){
    var pts = [[100, 140], [132, 62], [164, 128], [192, 74], [218, 118], [242, 82], [262, 112], [280, 90]], i, s = '';
    var t = contours(function(){
      for (i = 0; i < pts.length - 1; i++) s += arr(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], { c: C.red, w: 1.8, h: 6 });
      for (i = 0; i < pts.length; i++) s += S.circle(pts[i][0], pts[i][1], 3.6, { f: i ? C.white : C.char, s: C.char });
      s += lab(pts[0][0], pts[0][1] + 22, 'start', { s: 12 });
      return s;
    });
    t += S.text(300, 222, 'Plain gradient descent on a loss over two weights: successive steps', { s: 11.5, c: C.mid });
    return S.svg(600, 232, t, 'Successive gradient descent steps across a narrow valley');
  };

  /* 48: what the backward pass has to deliver */
  QZFIG.backpropSetup = function(){
    var t = '', names = ['Input x', 'Layer 1', 'Layer 2', 'Output'], bx = [20, 150, 280, 410], i, w = [['W1', 150], ['W2', 280], ['W3', 410]];
    for (i = 0; i < 4; i++) t += S.rect(bx[i], 56, 88, 40, { f: i === 0 ? C.light : C.bg }) + S.text(bx[i] + 44, 80, names[i], { s: 12, b: true });
    for (i = 0; i < 3; i++) t += arr(bx[i] + 88, 76, bx[i + 1], 76, { w: 1.8 });
    t += arr(bx[3] + 88, 76, 520, 76, { w: 1.8 }) + S.rect(520, 56, 66, 40, { f: C.red, s: C.dred }) + S.text(553, 80, 'Loss L', { s: 12, b: true, c: C.white });
    t += S.text(300, 38, 'Forward pass: inputs to prediction to loss', { s: 12, c: C.mid });
    for (i = 0; i < 3; i++) t += S.text(w[i][1] + 44, 124, w[i][0], { s: 12, b: true }) + S.text(w[i][1] + 44, 146, 'dL/d' + w[i][0] + ' = ?', { s: 12, c: C.red, b: true });
    t += S.text(300, 190, 'Training needs the gradient of the loss with respect to every weight.', { s: 12 });
    t += S.text(300, 210, 'W1, W2 and W3 are the weights of the three layers.', { s: 11.5, c: C.mid });
    return S.svg(600, 232, t, 'Forward pass through three layers to the loss, with the gradient needed for each weight');
  };

  /* 49: product of local slopes through depth */
  QZFIG.deepChain = function(){
    var t = '', n = 6, i, x0 = 60, dx = 96, y = 66, x;
    t += S.text(300, 28, 'Forward direction', { s: 12, c: C.mid });
    for (i = 0; i < n; i++) {
      x = x0 + i * dx;
      t += S.rect(x - 30, y - 20, 60, 40, { f: C.bg }) + S.text(x, y + 4, 'Layer ' + (i + 1), { s: 12, b: true });
      t += S.text(x, y + 40, 's' + (i + 1), { s: 12, c: C.red, b: true });
      if (i < n - 1) t += arr(x + 30, y, x + dx - 30, y, { w: 1.6, h: 7 });
    }
    t += arr(x0 + (n - 1) * dx, 126, x0, 126, { c: C.red, w: 2, d: '6 4' });
    t += S.text(300, 150, 'Backward direction: the gradient passes through every layer', { s: 12, c: C.red });
    t += S.text(300, 182, 's_k is the local slope of layer k.', { s: 12 });
    t += S.text(300, 204, 'The gradient at layer 1 is the loss slope times s6, s5, s4, s3, s2 and s1.', { s: 12 });
    return S.svg(600, 232, t, 'Six layers in sequence, each with a local slope, and the backward direction');
  };

  /* 50: the penalty term */
  QZFIG.penaltyCurve = function(){
    var px0 = 110, px1 = 490, py0 = 26, py1 = 160, t = '';
    t += frame(px0, py0, px1, py1, 'Weight w', 'Penalty', 26);
    t += S.path(curve(function(w){ return w * w; }, -3, 3, px0, px1, 0, 9, py0, py1, 100), { c: C.red, w: 2.8 });
    t += S.text(px0 + 24, py0 + 14, 'penalty = strength times w squared', { a: 'start', s: 12, b: true });
    t += S.text(300, 214, 'The penalty is added to the loss for every weight.', { s: 12, c: C.mid });
    return S.svg(600, 232, t, 'A penalty on the square of a weight against the weight');
  };

  /* 51: perplexity as a function of the loss; the loss is marked and no value is read off */
  QZFIG.perplexityCurve = function(){
    var px0 = 110, px1 = 490, py0 = 26, py1 = 150, t = '', lx = px0 + 0.693 / 3 * (px1 - px0);
    t += frame(px0, py0, px1, py1, 'Next-token loss', 'Perplexity', 26);
    t += S.path(curve(function(l){ return Math.exp(l); }, 0, 3, px0, px1, 0, 20.1, py0, py1, 100), { c: C.red, w: 2.8 });
    t += S.line(lx, py0 + 4, lx, py1, { c: C.char, w: 1.4, d: '5 4' }) + S.text(lx + 6, py0 + 16, 'loss 0.693', { a: 'start', s: 12, b: true });
    t += S.text(300, 198, 'perplexity = e to the power of the loss', { s: 12, b: true });
    t += S.text(300, 218, 'The axes show the relation between the two quantities only.', { s: 11.5, c: C.mid });
    return S.svg(600, 232, t, 'Perplexity against next-token loss with the given loss marked');
  };

  /* 52: loss against the probability of the correct token */
  QZFIG.surprisalCurve = function(){
    var px0 = 110, px1 = 490, py0 = 26, py1 = 150, t = '', x5 = px0 + 0.5 * (px1 - px0), k;
    t += frame(px0, py0, px1, py1, 'Probability p of the correct token', 'Cross-entropy loss', 38);
    t += S.path(curve(function(p){ return -Math.log(Math.max(p, 0.02)); }, 0.02, 1, px0 + 0.02 * (px1 - px0), px1, 0, 3.4, py0, py1, 100), { c: C.red, w: 2.8 });
    for (k = 0; k <= 4; k++) t += S.line(px0 + k / 4 * (px1 - px0), py1, px0 + k / 4 * (px1 - px0), py1 + 5, { c: C.char, w: 1.2 }) + S.text(px0 + k / 4 * (px1 - px0), py1 + 18, (k / 4).toFixed(2).replace(/0+$/, '').replace(/\.$/, ''), { s: 11 });
    t += S.line(x5, py0 + 4, x5, py1, { c: C.char, w: 1.4, d: '5 4' }) + S.text(x5 + 6, py0 + 16, 'p = 0.5', { a: 'start', s: 12, b: true });
    t += S.text(300, 208, 'loss = minus the natural log of p', { s: 12, b: true });
    return S.svg(600, 232, t, 'Cross-entropy loss against the probability given to the correct token, with p = 0.5 marked');
  };

  /* 22: the generalisation gap against model flexibility; no remedy is shown */
  QZFIG.gapCurves = function(){
    var px0 = 80, px1 = 530, py0 = 24, py1 = 160, t = '', xm = 7.2, fTr = function(x){ return 1.0 * Math.exp(-x / 3.2) + 0.05; }, fTe = function(x){ return 0.5 + 0.55 * Math.exp(-x / 2.5) + 0.0065 * Math.pow(Math.max(0, x - 3), 2.2); };
    t += frame(px0, py0, px1, py1, 'Model flexibility', 'Error', 26);
    t += S.path(curve(fTr, 0, 10, px0, px1, 0, 1.15, py0, py1), { c: C.char, w: 2.8 });
    t += S.path(curve(fTe, 0, 10, px0, px1, 0, 1.15, py0, py1), { c: C.red, w: 2.8 });
    var X = px0 + xm / 10 * (px1 - px0), yTr = py1 - fTr(xm) / 1.15 * (py1 - py0), yTe = py1 - fTe(xm) / 1.15 * (py1 - py0);
    t += S.line(X, yTr, X, yTe, { c: C.char, w: 1.6, d: '3 3' }) + lab(X + 8, (yTr + yTe) / 2 + 4, 'gap: test minus training error', { a: 'start' });
    t += legend(300, 36, [[C.char, 'Training error'], [C.red, 'Test error']]);
    t += S.text(300, 214, 'Overfitting is a gap between training error and test error.', { s: 11.5, c: C.mid });
    return S.svg(600, 232, t, 'Training and test error against model flexibility with the gap marked');
  };

  /* 53, 54: design of the saved experiment (network sizes and training set size) */
  QZFIG.experimentDesign = function(o){
    var params = [15, 105, 1051, 10095, 101437], t = '', i, x, r, hl = o.hl || [], n = o.n, hi;
    t += S.text(300, 22, 'Five network sizes, each trained on ' + n.toLocaleString() + ' points and scored on held-out points', { s: 12, b: true });
    for (i = 0; i < 5; i++) {
      x = 70 + i * 115; r = 7 + (Math.log10(params[i]) - 1) * 6.6; hi = hl.indexOf(i) >= 0;
      t += S.circle(x, 76, r, { f: hi ? C.red : C.bg, s: hi ? C.dred : C.char, w: hi ? 2 : 1.4 });
      t += S.text(x, 76 + 36 + 14, params[i].toLocaleString(), { s: 12, b: hi }) + S.text(x, 76 + 36 + 29, 'parameters', { s: 11.5, c: C.mid });
    }
    t += S.text(300, 172, 'Training set: ' + n.toLocaleString() + ' labelled points in 3D', { s: 12 });
    t += S.text(300, 191, 'Test set: 2,000 held-out points near the surface', { s: 12 });
    t += S.text(300, 215, 'Circle area increases with the number of parameters.', { s: 11.5, c: C.mid });
    return S.svg(600, 232, t, 'Five network sizes trained on ' + n + ' points');
  };

  (function(){
    var B = QZ.bank; function set(i, name, o){ B[i - 1].fig = { name: name, o: o || {} }; }
    set(16, 'mlpLayers', { sizes: [8, 16, 1] });
    set(17, 'mlpLayers', { sizes: [3, 4, 4, 1] });
    set(18, 'activationCompare');
    set(19, 'lossCurves');
    set(20, 'dropoutNet');
    set(21, 'valleyStep');
    set(22, 'gapCurves');
    set(44, 'gradientValley');
    set(45, 'valleySteps');
    set(46, 'contourGrad');
    set(47, 'contourPath');
    set(48, 'backpropSetup');
    set(49, 'deepChain');
    set(50, 'penaltyCurve');
    set(51, 'perplexityCurve');
    set(52, 'surprisalCurve');
    set(53, 'experimentDesign', { n: 100, hl: [] });
    set(54, 'experimentDesign', { n: 10000, hl: [3, 4] });
  })();
})();
