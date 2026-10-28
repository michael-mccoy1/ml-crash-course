"use strict";
/* Quiz schematics for questions 23 to 30: tokens, next-token prediction, the causal mask, temperature,
   SMILES validity, self-supervised learning, fine-tuning and pre-training data. */
(function(){
  var S = QZSV, C = S.C, MONO = 'font-family="Cascadia Mono,Consolas,monospace"';
  function mono(x, y, t, o){ o = o || {}; return '<text x="' + x + '" y="' + y + '" text-anchor="' + (o.a || 'middle') + '" fill="' + (o.c || C.char) + '" ' + MONO + (o.s ? ' font-size="' + o.s + '"' : '') + (o.b ? ' font-weight="700"' : '') + '>' + esc(t) + '</text>'; }
  var VOCAB = ['[pad]', '[start]', '[end]', '#', '(', ')', '-', '.', '/', '1', '2', '3', '4', '5', '6', '=', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', '[C@@H]', '[nH]', '[N+]', '[O-]', 'c', 'n', 'o', 's'];
  function tokens(smi){ return smi.match(/\[[^\]]+\]|Br|Cl|%\d\d|./g) || []; }
  function softmax(z, T){ var m = Math.max.apply(null, z), e = z.map(function(v){ return Math.exp((v - m) / T); }), s = e.reduce(function(a, b){ return a + b; }, 0); return e.map(function(v){ return v / s; }); }
  function bars(x0, y0, w, hh, labels, probs, o){
    o = o || {}; var n = labels.length, bw = w / n, out = '';
    out += S.line(x0, y0 + hh, x0 + w, y0 + hh, { c: C.char, w: 1.2 }) + S.line(x0, y0, x0, y0 + hh, { c: C.char, w: 1.2 });
    [0, 0.5, 1].forEach(function(v){ var y = y0 + hh - v * hh; out += S.line(x0 - 3, y, x0, y, { c: C.char, w: 1 }) + S.text(x0 - 6, y + 4, String(v), { a: 'end', s: 11 }); });
    probs.forEach(function(p, i){ var bh = p * hh, bx = x0 + i * bw + bw * 0.16; out += '<rect x="' + bx + '" y="' + (y0 + hh - bh) + '" width="' + (bw * 0.68) + '" height="' + bh + '" fill="' + (o.f || C.red) + '"/>' + mono(bx + bw * 0.34, y0 + hh + 15, labels[i], { s: 12 }); });
    return out;
  }

  /* 23: a SMILES string becomes a sequence of integer IDs, the only input the model receives */
  QZFIG.tokenPipeline = function(o){
    var smi = o.smiles || 'CC(=O)Nc1ccc(O)cc1', t = tokens(smi), n = t.length, bw = Math.min(30, 560 / n), x0 = 300 - n * bw / 2, out = '';
    out += mono(300, 28, smi, { s: 18, b: true }) + S.text(300, 48, 'SMILES string (text)', { c: C.mid });
    out += S.arrow(300, 58, 300, 92) + S.rect(236, 62, 128, 24, { f: C.light, r: 3 }) + S.text(300, 78, 'Tokeniser', { b: true });
    out += S.text(300, 112, 'Input to the model: one integer ID per token', { c: C.mid });
    t.forEach(function(k, i){ var id = VOCAB.indexOf(k); out += S.rect(x0 + i * bw + 1, 122, bw - 2, 30, { f: C.white, r: 2, w: 1.2 }) + S.text(x0 + i * bw + bw / 2, 142, String(id < 0 ? '?' : id), { s: 12, b: true }); });
    out += S.text(300, 178, 'Vocabulary: ' + VOCAB.length + ' entries, each with a fixed ID', { c: C.mid });
    out += S.text(300, 198, 'The model receives the IDs and not the characters.', { c: C.char });
    return S.svg(600, 212, out, 'A SMILES string is converted to a sequence of integer IDs');
  };

  /* 24: the quantities in one training example: a token prefix goes in, one probability per vocabulary token comes out */
  QZFIG.prefixToProbabilities = function(o){
    var pre = o.prefix || ['[start]', 'C', 'C', '(', '=', 'O', ')'], labels = o.labels || ['N', 'c', 'C', 'O', '(', '1'], probs = o.probs || [0.46, 0.22, 0.14, 0.09, 0.06, 0.03], out = '', x = 20, i;
    out += S.text(20, 20, 'Input: tokens of one training string', { a: 'start', c: C.mid });
    for (i = 0; i < pre.length; i++) { var w = Math.max(26, pre[i].length * 8 + 12); out += S.rect(x, 28, w, 28, { f: C.light, r: 3 }) + mono(x + w / 2, 47, pre[i], { s: 13 }); x += w + 5; }
    out += S.arrow(110, 58, 110, 96) + S.rect(50, 96, 120, 32, { f: C.char, s: C.char, r: 4 }) + S.text(110, 117, 'GPT', { c: C.white, b: true });
    out += S.arrow(170, 112, 300, 112);
    out += S.text(330, 20, 'Output: one probability per vocabulary token', { a: 'start', c: C.mid });
    out += bars(340, 44, 240, 126, labels, probs);
    out += S.text(460, 206, 'Tokens shown: 6 of ' + VOCAB.length, { c: C.mid, s: 11 });
    return S.svg(600, 220, out, 'A token prefix is the input and the output is a probability for each token');
  };

  /* 25: the score matrix of attention, one row per position being processed; the row of one token is marked with the positions after it */
  QZFIG.attentionScores = function(o){
    var toks = o.tokens || ['C', 'C', '(', '=', 'O', ')', 'N', 'c'], n = toks.length, cs = 19, x0 = 230, y0 = 60, hi = o.row === undefined ? 4 : o.row, out = '', r, c;
    out += S.text(300, 16, 'Attention score matrix', { b: true });
    out += S.text(x0 + n * cs / 2, 34, 'Position in the sequence (column)', { c: C.mid, s: 11 });
    for (c = 0; c < n; c++) out += mono(x0 + c * cs + cs / 2, y0 - 6, toks[c], { s: 12 });
    for (r = 0; r < n; r++) {
      out += mono(x0 - 8, y0 + r * cs + cs / 2 + 4, toks[r], { a: 'end', s: 12 });
      for (c = 0; c < n; c++) out += S.rect(x0 + c * cs, y0 + r * cs, cs - 2, cs - 2, { f: C.bg, s: C.mid, r: 2, w: 1 });
    }
    out += '<rect x="' + (x0 - 2) + '" y="' + (y0 + hi * cs - 2) + '" width="' + (n * cs + 2) + '" height="' + (cs + 2) + '" fill="none" stroke="' + C.red + '" stroke-width="2.4" rx="3"/>';
    out += S.text(x0 - 34, y0 + n * cs / 2, 'Position being processed (row)', { c: C.mid, s: 11, a: 'end' });
    out += S.text(x0 + n * cs + 14, y0 + hi * cs + 14, 'row of "' + toks[hi] + '"', { a: 'start', c: C.red, b: true, s: 11 });
    var bx = x0 + (hi + 1) * cs - 1, bw = (n - hi - 1) * cs;
    if (bw > 0) out += S.line(bx, y0 + hi * cs - 8, bx + bw - 2, y0 + hi * cs - 8, { c: C.red, w: 1.6 }) + S.line(bx, y0 + hi * cs - 12, bx, y0 + hi * cs - 4, { c: C.red, w: 1.6 }) + S.line(bx + bw - 2, y0 + hi * cs - 12, bx + bw - 2, y0 + hi * cs - 4, { c: C.red, w: 1.6 });
    out += S.text(x0 + n * cs + 14, y0 + hi * cs - 6, 'columns after "' + toks[hi] + '"', { a: 'start', c: C.red, s: 11 });
    return S.svg(600, y0 + n * cs + 12, out, 'Attention score matrix with a row for each position being processed');
  };

  /* 26: one set of logits, shown as probabilities at two temperatures */
  QZFIG.temperatureBars = function(o){
    var labels = o.labels || ['N', 'c', 'C', 'O', '(', ')'], z = o.logits || [2.2, 1.4, 1.0, 0.2, -0.3, -1.0], temps = o.temps || [0.5, 1.5], out = '';
    out += S.text(300, 18, 'Next-token probabilities from the same logits', { b: true });
    temps.forEach(function(T, k){ var x0 = 50 + k * 290; out += S.text(x0 + 110, 42, 'Temperature ' + T, { b: true, c: C.char }); out += bars(x0, 54, 220, 110, labels, softmax(z, T), { f: k ? C.mid : C.red }); });
    out += S.text(300, 208, 'probability = softmax(logit / temperature)', { c: C.mid });
    return S.svg(600, 220, out, 'Probabilities over six tokens at two temperatures for the same logits');
  };

  /* 27: a generated string with an unclosed ring digit and the result of parsing it */
  QZFIG.smilesParse = function(o){
    var smi = o.smiles || 'CC(=O)Nc1ccc(O)cc', t = tokens(smi), bw = 28, x0 = 300 - t.length * bw / 2, out = '', i, open = {};
    out += S.text(300, 20, 'String generated by sampling tokens', { c: C.mid });
    t.forEach(function(k, i2){ if (/^\d$/.test(k)) open[k] = (open[k] || 0) + 1; });
    for (i = 0; i < t.length; i++) { var ring = /^\d$/.test(t[i]) && open[t[i]] % 2 === 1; out += S.rect(x0 + i * bw + 1, 30, bw - 2, 30, { f: ring ? '#FBE9E9' : C.white, s: ring ? C.red : C.char, r: 3, w: ring ? 2.2 : 1.2 }) + mono(x0 + i * bw + bw / 2, 51, t[i], { s: 14, b: ring }); }
    var rk = Object.keys(open).filter(function(k){ return open[k] % 2 === 1; });
    out += S.text(300, 80, rk.length ? 'Ring digit ' + rk[0] + ' appears once, at the opening atom only' : 'Every ring digit appears twice', { c: rk.length ? C.red : C.mid, s: 12 });
    out += S.arrow(300, 86, 300, 98) + S.rect(236, 98, 128, 24, { f: C.light, r: 3 }) + S.text(300, 114, 'RDKit parse', { b: true }) + S.arrow(300, 122, 300, 132);
    out += S.rect(150, 132, 300, 44, { f: C.white, s: C.red, r: 4, w: 1.8 }) + S.text(300, 150, rk.length ? 'Parse error' : 'Valid molecule', { b: true, c: rk.length ? C.red : C.char }) + S.text(300, 168, rk.length ? 'Ring closure ' + rk[0] + ' is not closed' : 'Structure perceived', { c: C.char, s: 11 });
    return S.svg(600, 192, out, 'A generated SMILES string is checked by RDKit and fails because a ring is not closed');
  };

  /* 28: masked-token prediction: masked string in, predicted token distribution out, target token beside it */
  QZFIG.maskedToken = function(o){
    var t = (o.tokens || ['C', 'C', '(', '=', 'O', ')', 'N', 'c', '1', 'c', 'c', 'c']).slice(), m = o.mask === undefined ? 6 : o.mask, target = t[m], bw = 34, x0 = 300 - t.length * bw / 2, out = '', labels = o.labels || ['N', 'C', 'O', 'c', 'S'], probs = o.probs || [0.71, 0.12, 0.09, 0.06, 0.02], i;
    out += S.text(300, 16, 'Input: one atom token is masked', { c: C.mid });
    for (i = 0; i < t.length; i++) { var mk = i === m; out += S.rect(x0 + i * bw + 1, 22, bw - 2, 30, { f: mk ? '#FBE9E9' : C.white, s: mk ? C.red : C.char, r: 3, w: mk ? 2.2 : 1.2 }) + mono(x0 + i * bw + bw / 2, 42, mk ? '?' : t[i], { s: 14, b: mk }); }
    out += S.arrow(300, 54, 300, 74) + S.rect(240, 74, 120, 28, { f: C.char, s: C.char, r: 4 }) + S.text(300, 93, 'GPT', { c: C.white, b: true });
    out += S.arrow(300, 102, 300, 126);
    out += bars(210, 132, 180, 44, labels, probs);
    out += S.text(300, 208, 'Predicted probability at the masked position', { c: C.mid, s: 11 });
    out += S.rect(436, 132, 130, 44, { f: C.white, s: C.char, r: 4 }) + S.text(501, 150, 'Target token', { c: C.mid, s: 11 }) + mono(501, 169, target, { s: 16, b: true });
    out += S.arrow(436, 154, 396, 154, { d: '4 3' });
    return S.svg(600, 220, out, 'A masked token is predicted from the other tokens in the string');
  };

  /* 29: an encoder and a prediction head, trained on labelled compounds with a loss */
  QZFIG.transferModel = function(o){
    var out = '', n = o.n || '1,000';
    out += S.rect(20, 26, 110, 52, { f: C.light, r: 4 }) + S.text(75, 48, 'Molecule', { b: true }) + S.text(75, 66, 'SMILES or graph', { s: 11, c: C.mid });
    out += S.arrow(130, 52, 160, 52);
    out += S.rect(162, 18, 140, 68, { f: C.bg, s: C.char, r: 4 }) + S.text(232, 44, 'Pre-trained', { b: true }) + S.text(232, 60, 'encoder', { b: true }) + S.text(232, 77, 'weights from pre-training', { s: 10.5 });
    out += S.arrow(302, 52, 354, 52) + S.text(328, 44, 'vector', { s: 11, c: C.mid });
    out += S.rect(356, 26, 100, 52, { f: C.white, r: 4 }) + S.text(406, 48, 'Prediction', { b: true }) + S.text(406, 64, 'head', { b: true });
    out += S.arrow(456, 52, 486, 52);
    out += S.rect(488, 26, 94, 52, { f: C.white, r: 4 }) + S.text(535, 48, 'Predicted', { b: true }) + S.text(535, 64, 'clearance', { b: true });
    out += S.rect(356, 128, 226, 52, { f: '#FBE9E9', s: C.red, r: 4, w: 1.8 }) + S.text(469, 150, 'Labelled compounds, n = ' + n, { b: true }) + S.text(469, 168, 'measured clearance for each', { s: 11 });
    out += S.arrow(535, 128, 535, 80, { c: C.red }) + S.text(543, 108, 'loss', { a: 'start', c: C.red, b: true, s: 11 });
    out += S.arrow(406, 128, 406, 80, { c: C.red, d: '5 4' });
    out += S.line(356, 154, 232, 154, { c: C.red, d: '5 4' }) + S.arrow(232, 154, 232, 88, { c: C.red, d: '5 4' }) + S.text(224, 124, 'weight updates', { a: 'end', c: C.red, s: 11 });
    out += S.text(300, 210, 'The loss compares each prediction with the measured value.', { c: C.mid });
    return S.svg(600, 224, out, 'An encoder with a prediction head, trained on labelled compounds with a loss');
  };

  /* 30: sizes of the unlabelled and labelled data sets on a log axis */
  QZFIG.dataScale = function(o){
    var rows = o.rows || [['Structures without measured values', 10000000, C.bg], ['Labelled logD compounds', 4200, C.red], ['Labelled HLM compounds', 3087, C.red]], x0 = 200, w = 360, out = '', lo = 2, hi = 7, i;
    function sx(v){ return x0 + (Math.log10(v) - lo) / (hi - lo) * w; }
    out += S.text(300, 18, 'Number of molecules available, log scale', { b: true });
    for (i = lo; i <= hi; i++) out += S.line(sx(Math.pow(10, i)), 36, sx(Math.pow(10, i)), 156, { c: C.silver, w: 1 }) + S.text(sx(Math.pow(10, i)), 172, i === 2 ? '100' : '10^' + i, { s: 11, c: C.mid });
    rows.forEach(function(r, k){ var y = 46 + k * 36; out += S.text(x0 - 10, y + 17, r[0], { a: 'end', s: 12 }) + '<rect x="' + x0 + '" y="' + y + '" width="' + (sx(r[1]) - x0) + '" height="24" fill="' + r[2] + '"/>' + (sx(r[1]) + 90 > 598 ? S.text(sx(r[1]) - 6, y + 17, r[1].toLocaleString('en-US'), { a: 'end', b: true, s: 12 }) : S.text(sx(r[1]) + 6, y + 17, r[1].toLocaleString('en-US'), { a: 'start', b: true, s: 12 })); });
    out += S.text(300, 204, 'A measured value requires an assay for each compound. A structure does not.', { c: C.mid });
    return S.svg(600, 216, out, 'Number of unlabelled structures compared with labelled compounds');
  };

  (function(){ var B = QZ.bank; function set(i, name, o){ B[i - 1].fig = { name: name, o: o || {} }; }
    set(23, 'tokenPipeline', { smiles: 'CC(=O)Nc1ccc(O)cc1' });
    set(24, 'prefixToProbabilities', {});
    set(25, 'attentionScores', {});
    set(26, 'temperatureBars', {});
    set(27, 'smilesParse', { smiles: 'CC(=O)Nc1ccc(O)cc' });
    set(28, 'maskedToken', {});
    set(29, 'transferModel', { n: '1,000' });
    set(30, 'dataScale', {});
  })();
})();
