"use strict";
/* Lesson 6: Compare */
var L5 = new Lesson('compare', 'Compare');
L5.optsHTML = function(){ return optGroup('Endpoint', seg([['HLM', 'HLM clearance'], ['logD', 'logD']], EP, 'data-ep'), 'Scores are kept per endpoint.') + '<div class="og"><div class="ol">What is listed</div><div class="hint">Everything you trained in this session on the selected endpoint, plus reference runs from the course notebooks. Train models in the Tree models, Neural networks and GPT lessons first.</div><div class="status" id="cst"></div></div>'; };
L5.onData = function(){ };
function cmpRows(){
  var rows = [], names = Object.keys(D.scores); names.forEach(function(k){ var m = D.scores[k]; rows.push({ n: k + (m.params ? ' (' + m.params + ')' : ''), v: m.r2, t: m.secs, ref: false, m: m }); });
  var refs = (EXTRAS && EXTRAS.refs && EXTRAS.refs[EP]) || []; refs.forEach(function(r){ rows.push({ n: r[0], v: r[1], t: r[2], ref: true }); }); return rows.sort(function(a, b){ return b.v - a.v; });
}
function gptRow(){ return G && G.trained ? { n: 'GPT (' + G.NL + ' layers, ' + G.steps + ' steps)', t: G.secs } : null; }

L5.add({ short: 'Leaderboard', title: 'Accuracy and training time for every model you trained',
  change: 'Press Play. Train more models in the other lessons and come back.',
  watch: 'Red bars are models trained in this session. Blue-grey bars are reference runs from the notebooks, which use about 2,000 more features.',
  build: function(api){
    var pn = api.el('pn', 'left:28px;top:28px;width:904px;height:408px;overflow:auto', '<div class="lb"></div><p class="note" style="font-size:12px;margin-top:12px">R2 is measured on unseen compounds. Time covers fitting the model only, in your browser or on the notebook machine. The GPT predicts tokens, not a property, so it has a time but no R2.</p>');
    function draw(prog){
      var rows = cmpRows(), g = gptRow(), mx = Math.max(0.5, Math.max.apply(null, rows.map(function(r){ return r.v; }))), html = '<div class="hd">Model</div><div class="hd">R2 on unseen compounds</div><div class="hd" style="text-align:right">R2</div><div class="hd" style="text-align:right">Training time</div>';
      rows.forEach(function(r, i){ var vis = clamp(prog * rows.length - i * 0.6, 0, 1); html += '<div class="nm" style="opacity:' + (vis > 0.05 ? 1 : 0.15) + '">' + esc(r.n) + '</div><div class="tr"><i class="' + (r.ref ? 'ref' : '') + '" style="width:' + (clamp(r.v, 0, mx) / mx * 100 * vis) + '%;transition:none"></i></div><div class="vv">' + r.v.toFixed(2) + '</div><div class="vv" style="font-weight:500;color:#555B66">' + (r.t === null || r.t === undefined ? 'n/a' : fmtT(r.t)) + '</div>'; });
      if (g) html += '<div class="nm">' + esc(g.n) + '</div><div class="tr"></div><div class="vv">n/a</div><div class="vv" style="font-weight:500;color:#555B66">' + fmtT(g.t) + '</div>';
      qs('.lb', pn).innerHTML = html; var n = Object.keys(D.scores).length; var st = $('cst'); if (st) st.textContent = n ? n + ' model' + (n > 1 ? 's' : '') + ' trained on ' + EP + (g ? ' and a GPT' : '') + '.' : 'Nothing trained on ' + EP + ' yet.'; }
    function render(){ draw(1); }
    async function play(tok){ draw(0); var rows = cmpRows(); await api.beat(Object.keys(D.scores).length ? 'These are the models you trained, next to the notebook references.' : 'You have not trained any model on this endpoint yet. Only the notebook references are shown.', 2600); await api.anim(ANIM ? 3200 : 1, function(t){ draw(t); }); api.cap('Compare the bars first, then ask what each extra bit of accuracy cost in training time.'); await api.sleep(3600); api.key('The best model is the one that is accurate enough on unseen compounds for the time and data you have.'); }
    return { render: render, play: play };
  } });

L5.add({ short: 'Accuracy against time', title: 'Accuracy rarely comes free',
  change: 'Press Play. Train other settings in the earlier lessons and watch the points move.',
  watch: 'Each point is a model. Higher is more accurate. Further left is faster. Models up and to the left dominate the others.',
  build: function(api){
    var pp = api.el('pn', 'left:28px;top:28px;width:640px;height:408px', '<h4>Unseen R2 against training time (log scale)</h4><canvas width="618" height="350"></canvas>');
    var sp = api.el('pn', 'left:688px;top:28px;width:244px;height:408px', '<h4>How to read it</h4><div class="rd" style="font-size:12.5px;color:#4a4f5a"></div>');
    function pts(){ var out = []; Object.keys(D.scores).forEach(function(k){ var m = D.scores[k]; if (m.secs > 0) out.push({ n: k, t: Math.max(m.secs, 0.003), v: m.r2, ref: false }); }); ((EXTRAS && EXTRAS.refs && EXTRAS.refs[EP]) || []).forEach(function(r){ if (r[2]) out.push({ n: r[0].split(',')[0], t: r[2], v: r[1], ref: true }); }); return out; }
    function draw(prog){
      var P = pts(), g = cv(qs('canvas', pp)), x = g.x, box = { l: 52, r: g.w - 20, t: 14, b: g.h - 44 }; if (!P.length) { x.fillStyle = C.mid; x.fillText('Train a model first', 60, 40); return; }
      var lt = P.map(function(p){ return Math.log10(p.t); }), lo = Math.floor(Math.min.apply(null, lt)) - 0.2, hi = Math.ceil(Math.max.apply(null, lt)) + 0.2, vmx = Math.max(0.5, Math.max.apply(null, P.map(function(p){ return p.v; })) * 1.1), vmn = Math.min(0, Math.min.apply(null, P.map(function(p){ return p.v; })));
      var sx = function(l){ return box.l + (l - lo) / (hi - lo) * (box.r - box.l); }, sy = function(v){ return box.b - (v - vmn) / (vmx - vmn) * (box.b - box.t); }; x.strokeStyle = C.silver; x.fillStyle = C.mid; x.textAlign = 'right'; x.font = '12px Aptos, Segoe UI, sans-serif';
      for (var k = 0; k <= 4; k++) { var v = vmn + (vmx - vmn) * k / 4, y = sy(v); x.beginPath(); x.moveTo(box.l, y); x.lineTo(box.r, y); x.stroke(); x.fillText(v.toFixed(2), box.l - 6, y + 4); } x.textAlign = 'center'; for (var e = Math.ceil(lo); e <= hi; e++) { x.beginPath(); x.moveTo(sx(e), box.t); x.lineTo(sx(e), box.b); x.stroke(); var s = Math.pow(10, e); x.fillText(s >= 1 ? s + ' s' : s + ' s', sx(e), box.b + 16); }
      x.fillStyle = C.char; x.font = '600 12px Aptos, Segoe UI, sans-serif'; x.fillText('training time', (box.l + box.r) / 2, g.h - 6); x.save(); x.translate(13, (box.t + box.b) / 2); x.rotate(-Math.PI / 2); x.fillText('unseen R2', 0, 0); x.restore();
      P.forEach(function(p, i){ var vis = clamp(prog * P.length - i * 0.7, 0, 1); if (vis <= 0) return; var X = sx(Math.log10(p.t)), Y = sy(p.v); x.globalAlpha = vis; x.fillStyle = p.ref ? C.bg : C.red; x.beginPath(); x.arc(X, Y, 7 * (0.5 + 0.5 * vis), 0, 6.3); x.fill(); x.fillStyle = C.char; x.font = '600 11px Aptos, Segoe UI, sans-serif'; x.textAlign = X > box.r - 130 ? 'right' : 'left'; x.fillText(p.n.length > 26 ? p.n.slice(0, 26) : p.n, X + (X > box.r - 130 ? -11 : 11), Y + 4); x.globalAlpha = 1; });
      qs('.rd', sp).innerHTML = '<p><b style="color:' + C.red + '">Red</b>: trained in this session.</p><p><b style="color:#6B7B8C">Blue grey</b>: notebook runs that report a time.</p><p>Moving right costs time. Moving up buys accuracy. A point that is both slower and less accurate than another is not worth using.</p><p class="note" style="color:#6B7280">The time axis is logarithmic, so each gridline is ten times slower.</p>'; }
    function render(){ draw(1); }
    async function play(tok){ draw(0); await api.beat('Every model sits at its accuracy and its training time.', 2400); await api.anim(ANIM ? 3000 : 1, function(t){ draw(t); }); api.cap('The ideal model is in the top left corner: accurate and fast.'); await api.sleep(3200); api.key('Pick the cheapest model that is accurate enough, then check it on compounds it has not seen.'); }
    return { render: render, play: play };
  } });

L5.add({ short: 'Discussion', title: 'Questions to settle before trusting a result',
  change: 'Press Play and discuss each question before the next appears.',
  watch: 'Use your own scores from the earlier lessons as evidence.',
  build: function(api){
    var qs_ = ['Would the same model win on another split of the data? What would you run to find out?', 'Why do tree ensembles often match or beat neural networks on a few thousand compounds?', 'Which score did you report, and on which compounds? Would a colleague get the same answer from the same data?', 'Does handling of censored values change which model looks best? Look at the data quality lesson.', 'How does assay noise limit the best possible R2? How would you estimate that limit?', 'The GPT has no R2. What would you do with it in a project: predict, generate, or embed compounds?'];
    var els = qs_.map(function(q, i){ return api.el('pn fd', 'left:' + (28 + (i % 2) * 456) + 'px;top:' + (28 + Math.floor(i / 2) * 134) + 'px;width:448px;height:122px', '<h4>Question ' + (i + 1) + '</h4><div style="font-size:15px;line-height:1.45;color:#34373F">' + esc(q) + '</div>'); });
    function render(){ els.forEach(function(e){ e.classList.add('on'); }); }
    async function play(tok){ els.forEach(function(e){ e.classList.remove('on'); }); for (var i = 0; i < els.length; i++) { els[i].classList.add('on'); await api.sleep(1200); } api.key('A model is only as trustworthy as the way it was tested. Ask these questions of every result.'); }
    return { render: render, play: play };
  } });

L5.glossary([
  ['R2', 'Share of the spread in the answers explained on unseen compounds. 1 is perfect, 0 is no better than the average.'],
  ['Spearman', 'How well the model ranks compounds, ignoring exact values.'], ['RMSE', 'Typical prediction error in the units of the endpoint.'],
  ['Training time', 'How long fitting the model took, excluding drawing and animation.'], ['Reference run', 'A result from the course notebooks, with about 2,000 more features and a different machine.'],
  ['Dominated model', 'A model that another model beats on both accuracy and speed.'], ['Unseen compounds', 'Compounds held back from training, chosen by scaffold.'],
  ['Assay noise', 'Measurement error in the lab data. It limits the R2 any model can reach.']
]);
