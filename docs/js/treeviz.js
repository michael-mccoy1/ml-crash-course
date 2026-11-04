"use strict";
/* Live tree-model learning of a thick double helix: engine (no DOM) and surface extraction.
   Everything is in the box [-1, 1]^3. The target is the signed distance to the helix surface, negative inside, clipped to +-TV.CLIP. */
var TV = (function(){
  var TAU = 6.283185307179586, HR = 0.45, HS = 0.2, HK = TAU * 1.5 / 1.7, HZ = 0.85, CLIP = 0.25, B = 32;
  function mkrng(seed){ var s = seed >>> 0; return function(){ s = (s + 0x6D2B79F5) >>> 0; var t = s; t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61); return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  /* signed distance to the helix: two tubes of radius HS around spiral curves, plus rods of radius 0.075 across the helix every 0.3 */
  function sdf(x, y, z){
    var th = Math.atan2(y, x), best = 1e9, s, n, ph, zz, dx, dy, dz, d;
    for (s = 0; s < 2; s++) {
      ph = s * Math.PI;
      for (n = -3; n <= 3; n++) {
        zz = (th - ph + TAU * n) / HK; if (zz < -HZ) zz = -HZ; else if (zz > HZ) zz = HZ;
        dx = x - HR * Math.cos(HK * zz + ph); dy = y - HR * Math.sin(HK * zz + ph); dz = z - zz;
        d = dx * dx + dy * dy + dz * dz; if (d < best) best = d;
      }
    }
    best = Math.sqrt(best) - HS;
    var j = Math.round((z + 0.6) / 0.3); if (j < 0) j = 0; else if (j > 4) j = 4;
    var zj = -0.6 + 0.3 * j, ax = HR * Math.cos(HK * zj), ay = HR * Math.sin(HK * zj), t = (x * ax + y * ay) / (ax * ax + ay * ay); if (t < -1) t = -1; else if (t > 1) t = 1;
    var px = x - t * ax, py = y - t * ay, rd = Math.sqrt(px * px + py * py + (z - zj) * (z - zj)) - 0.075;
    return rd < best ? rd : best;
  }
  function clip(v){ return v < -CLIP ? -CLIP : (v > CLIP ? CLIP : v); }
  /* training points: half uniform in the box, half in the cylinder around the helix */
  function makeData(N, seed){
    var r = mkrng(1000 + seed * 7919 + N), X = new Float32Array(N * 3), y = new Float32Array(N), i, x, yy, z, a, rr;
    for (i = 0; i < N; i++) {
      var u = r(), dd;
      if (u < 0.2) { x = r() * 2 - 1; yy = r() * 2 - 1; z = r() * 2 - 1; dd = sdf(x, yy, z); }
      else if (u < 0.4) { a = r() * TAU; rr = 0.85 * Math.sqrt(r()); x = rr * Math.cos(a); yy = rr * Math.sin(a); z = (r() * 2 - 1) * 0.95; dd = sdf(x, yy, z); }
      else { do { a = r() * TAU; rr = 0.8 * Math.sqrt(r()); x = rr * Math.cos(a); yy = rr * Math.sin(a); z = (r() * 2 - 1) * 0.92; dd = sdf(x, yy, z); } while (dd > 0.18 || dd < -0.18); }
      X[3 * i] = x; X[3 * i + 1] = yy; X[3 * i + 2] = z; y[i] = clip(dd);
    }
    return { X: X, y: y, n: N };
  }
  /* held-out points within 0.15 of the surface, half inside and half outside, so a constant prediction scores 0.5 */
  function makeTest(seed){
    var r = mkrng(555 + seed), T = 3000, X = new Float32Array(T * 2 * 3), nin = 0, nout = 0, tries = 0;
    while ((nin < T || nout < T) && tries < 4000000) {
      tries++; var a = r() * TAU, rr = 0.8 * Math.sqrt(r()), x = rr * Math.cos(a), y = rr * Math.sin(a), z = (r() * 2 - 1) * 0.92, d = sdf(x, y, z), k = -1;
      if (d < 0 && d > -0.15 && nin < T) { k = nin++; } else if (d >= 0 && d < 0.15 && nout < T) { k = T + nout++; }
      if (k >= 0) { X[3 * k] = x; X[3 * k + 1] = y; X[3 * k + 2] = z; }
    }
    return { X: X, n: T * 2, half: T };
  }
  function balancedAcc(pred, half){ var tp = 0, tn = 0, i; for (i = 0; i < half; i++) if (pred[i] < 0) tp++; for (i = half; i < 2 * half; i++) if (pred[i] >= 0) tn++; return 0.5 * (tp / half + tn / half); }

  /* ---------- regression tree on three features, histogram splits within the node's cell ---------- */
  function buildTree(X, y, idx, n, o){
    var maxDepth = o.maxDepth, minLeaf = o.minLeaf || 1, mtry = o.mtry || 3, rand = o.rand || Math.random;
    var feat = [], thr = [], left = [], right = [], val = [], cnt = [], dep = [], lo = [], hi = [], neg = [];
    var sumH = new Float64Array(B), cntH = new Int32Array(B), negH = new Int32Array(B);
    var stack = [[0, n, -1, 0, [-1, -1, -1], [1, 1, 1]]], f, i, k;
    while (stack.length) {
      var it = stack.pop(), a = it[0], b = it[1], par = it[2], side = it[3], nlo = it[4], nhi = it[5], m = b - a, id = feat.length, s = 0, ng = 0;
      for (i = a; i < b; i++) { s += y[idx[i]]; if (y[idx[i]] < 0) ng++; }
      feat.push(-1); thr.push(0); left.push(-1); right.push(-1); val.push(s / m); cnt.push(m); neg.push(ng / m); lo.push(nlo[0], nlo[1], nlo[2]); hi.push(nhi[0], nhi[1], nhi[2]);
      var depth = par < 0 ? 0 : dep[par] + 1; dep.push(depth);
      if (par >= 0) { if (side === 0) left[par] = id; else right[par] = id; }
      if (depth >= maxDepth || m < 2 * minLeaf || m < 2) continue;
      var fs = [0, 1, 2]; if (mtry < 3) { for (i = 2; i > 0; i--) { var jx = Math.floor(rand() * (i + 1)), tt = fs[i]; fs[i] = fs[jx]; fs[jx] = tt; } fs = fs.slice(0, mtry); }
      var bestG = 1e-12, bestF = -1, bestK = -1, bestThr = 0, tot2 = s * s / m;
      for (var q = 0; q < fs.length; q++) {
        f = fs[q]; var l0 = nlo[f], w = nhi[f] - nlo[f]; if (w <= 1e-9) continue;
        sumH.fill(0); cntH.fill(0);
        var sc = B / w;
        for (i = a; i < b; i++) { var p = idx[i], bi = Math.floor((X[3 * p + f] - l0) * sc); if (bi < 0) bi = 0; else if (bi >= B) bi = B - 1; sumH[bi] += y[p]; cntH[bi]++; }
        var cl = 0, sl = 0;
        for (k = 0; k < B - 1; k++) {
          cl += cntH[k]; sl += sumH[k]; var cr = m - cl; if (cl < minLeaf || cr < minLeaf) continue;
          var g = sl * sl / cl + (s - sl) * (s - sl) / cr - tot2; if (g > bestG) { bestG = g; bestF = f; bestK = k; bestThr = l0 + (k + 1) / sc; }
        }
      }
      if (bestF < 0) continue;
      var pa = a, pb = b - 1;
      while (pa <= pb) { if (X[3 * idx[pa] + bestF] < bestThr) pa++; else { var tmp = idx[pa]; idx[pa] = idx[pb]; idx[pb] = tmp; pb--; } }
      var nl = pa - a; if (nl < 1 || nl >= m) continue;
      feat[id] = bestF; thr[id] = bestThr;
      var rlo = nlo.slice(), lhi = nhi.slice(); rlo[bestF] = bestThr; lhi[bestF] = bestThr;
      stack.push([a + nl, b, id, 1, rlo, nhi]); stack.push([a, a + nl, id, 0, nlo, lhi]);
    }
    return { feat: Int8Array.from(feat), thr: Float32Array.from(thr), left: Int32Array.from(left), right: Int32Array.from(right), val: Float32Array.from(val), cnt: Int32Array.from(cnt), dep: Uint8Array.from(dep), lo: Float32Array.from(lo), hi: Float32Array.from(hi), neg: Float32Array.from(neg), nodes: feat.length };
  }
  function predictOne(t, x, y, z, cap){ var nd = 0; while (t.feat[nd] >= 0 && t.dep[nd] < cap) { var v = t.feat[nd] === 0 ? x : (t.feat[nd] === 1 ? y : z); nd = v < t.thr[nd] ? t.left[nd] : t.right[nd]; } return t.val[nd]; }
  function predictSet(t, X, n, cap, out, w){ for (var i = 0; i < n; i++) out[i] += w * predictOne(t, X[3 * i], X[3 * i + 1], X[3 * i + 2], cap); }
  function depthOf(t){ var m = 0; for (var i = 0; i < t.nodes; i++) if (t.dep[i] > m) m = t.dep[i]; return m; }
  /* add w * (tree prediction) to a G^3 grid by filling each leaf box; cells are indexed (k * G + j) * G + i, centres at (i + 0.5) / G * 2 - 1 */
  function fillGrid(t, G, F, w, cap){
    for (var nd = 0; nd < t.nodes; nd++) {
      if (!(t.feat[nd] < 0 || t.dep[nd] === cap)) continue;
      var i0 = Math.ceil((t.lo[3 * nd] + 1) * G / 2 - 0.5), i1 = Math.ceil((t.hi[3 * nd] + 1) * G / 2 - 0.5), j0 = Math.ceil((t.lo[3 * nd + 1] + 1) * G / 2 - 0.5), j1 = Math.ceil((t.hi[3 * nd + 1] + 1) * G / 2 - 0.5), k0 = Math.ceil((t.lo[3 * nd + 2] + 1) * G / 2 - 0.5), k1 = Math.ceil((t.hi[3 * nd + 2] + 1) * G / 2 - 0.5);
      if (i0 < 0) i0 = 0; if (j0 < 0) j0 = 0; if (k0 < 0) k0 = 0; if (i1 > G) i1 = G; if (j1 > G) j1 = G; if (k1 > G) k1 = G;
      var v = w * t.val[nd];
      for (var k = k0; k < k1; k++) for (var j = j0; j < j1; j++) { var base = (k * G + j) * G; for (var i = i0; i < i1; i++) F[base + i] += v; }
    }
  }
  /* ---------- the three model types, grown one tree at a time ---------- */
  function Model(kind, data, test, o){
    this.kind = kind; this.o = o; this.data = data; this.test = test; this.trees = []; this.G = o.G || 64; this.rand = mkrng(o.seed * 31 + 7);
    this.tpred = new Float32Array(test.n); this.gridSum = new Float32Array(this.G * this.G * this.G);
    this.snaps = []; this.acc = []; this.cum = new Float32Array(test.n);
    var N = data.n; this.N = N;
    if (kind === 'boost') { var m = 0; for (var i = 0; i < N; i++) m += data.y[i]; this.base = m / N; this.F = new Float32Array(N).fill(this.base); this.res = new Float32Array(N); this.idx = new Int32Array(N); this.gridSum.fill(this.base); this.cum.fill(this.base); }
    this.counts = [];
  }
  Model.prototype.addTree = function(){
    var o = this.o, d = this.data, N = this.N, kind = this.kind, i, t;
    if (kind === 'boost') {
      for (i = 0; i < N; i++) { this.res[i] = d.y[i] - this.F[i]; this.idx[i] = i; }
      t = buildTree(d.X, this.res, this.idx, N, { maxDepth: o.depth, minLeaf: o.minLeaf, mtry: 3, rand: this.rand });
      for (i = 0; i < N; i++) this.F[i] += o.lr * predictOne(t, d.X[3 * i], d.X[3 * i + 1], d.X[3 * i + 2], 99);
      predictSet(t, this.test.X, this.test.n, 99, this.cum, o.lr);
      fillGrid(t, this.G, this.gridSum, o.lr, 99); this.trees.push(t);
    } else {
      var n = Math.min(N, o.bagCap || 40000), idx = new Int32Array(n);
      if (kind === 'forest') for (i = 0; i < n; i++) idx[i] = Math.floor(this.rand() * N); else for (i = 0; i < n; i++) idx[i] = i;
      t = buildTree(d.X, d.y, idx, n, { maxDepth: o.depth, minLeaf: o.minLeaf, mtry: kind === 'forest' ? o.mtry : 3, rand: this.rand });
      this.trees.push(t);
      if (kind === 'forest') { predictSet(t, this.test.X, this.test.n, 99, this.cum, 1); fillGrid(t, this.G, this.gridSum, 1, 99); }
    }
    var cnt = this.trees.length, tp = new Float32Array(this.test.n);
    if (kind === 'forest') for (i = 0; i < tp.length; i++) tp[i] = this.cum[i] / cnt; else if (kind === 'boost') tp.set(this.cum); else predictSet(t, this.test.X, this.test.n, 99, tp, 1);
    this.acc.push(balancedAcc(tp, this.test.half));
    if (this.isCheck(cnt)) this.snapshot(cnt);
    return t;
  };
  Model.prototype.isCheck = function(c){ var cs = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]; return cs.indexOf(c) >= 0 || c === this.o.trees; };
  Model.prototype.snapshot = function(c){ var f = new Float32Array(this.gridSum.length), s = this.kind === 'forest' ? 1 / c : 1; for (var i = 0; i < f.length; i++) f[i] = this.gridSum[i] * s; this.snaps.push({ count: c, F: f }); };
  /* the single tree: field at a given depth cap */
  Model.prototype.treeField = function(cap){ var F = new Float32Array(this.G * this.G * this.G); fillGrid(this.trees[0], this.G, F, 1, cap); return F; };
  Model.prototype.treeAcc = function(cap){ var p = new Float32Array(this.test.n); predictSet(this.trees[0], this.test.X, this.test.n, cap, p, 1); return balancedAcc(p, this.test.half); };

  /* ---------- surface points: sign changes of a field on a G^3 grid, with positions interpolated and normals from the smoothed inside map ---------- */
  function surface(F, G){
    var n3 = G * G * G, O = new Float32Array(n3), T = new Float32Array(n3), i, j, k, a;
    for (i = 0; i < n3; i++) O[i] = F[i] < 0 ? 1 : 0;
    function blur(src, dst, stride){ for (var q = 0; q < n3; q++) { var c = Math.floor(q / stride) % G; var lo = c > 0 ? src[q - stride] : src[q], hi2 = c < G - 1 ? src[q + stride] : src[q]; dst[q] = 0.25 * lo + 0.5 * src[q] + 0.25 * hi2; } }
    for (var rep = 0; rep < 2; rep++) { blur(O, T, 1); blur(T, O, G); blur(O, T, G * G); O.set(T); }
    var px = [], py = [], pz = [], nx = [], ny = [], nz = [], h = 2 / G;
    function emit(c0, c1, ax, i0, j0, k0, t){
      var x = (i0 + 0.5) * h - 1, y = (j0 + 0.5) * h - 1, z = (k0 + 0.5) * h - 1;
      if (ax === 0) x += t * h; else if (ax === 1) y += t * h; else z += t * h;
      var c = t < 0.5 ? c0 : c1, ci = c % G, cj = Math.floor(c / G) % G, ck = Math.floor(c / (G * G));
      var gx = (ci < G - 1 ? O[c + 1] : O[c]) - (ci > 0 ? O[c - 1] : O[c]), gy = (cj < G - 1 ? O[c + G] : O[c]) - (cj > 0 ? O[c - G] : O[c]), gz = (ck < G - 1 ? O[c + G * G] : O[c]) - (ck > 0 ? O[c - G * G] : O[c]), l = Math.sqrt(gx * gx + gy * gy + gz * gz) || 1;
      px.push(x); py.push(y); pz.push(z); nx.push(-gx / l); ny.push(-gy / l); nz.push(-gz / l);
    }
    for (k = 0; k < G; k++) for (j = 0; j < G; j++) for (i = 0; i < G; i++) {
      var c0 = (k * G + j) * G + i, v0 = F[c0];
      if (i < G - 1) { var v1 = F[c0 + 1]; if ((v0 < 0) !== (v1 < 0)) emit(c0, c0 + 1, 0, i, j, k, v0 / (v0 - v1)); }
      if (j < G - 1) { var v2 = F[c0 + G]; if ((v0 < 0) !== (v2 < 0)) emit(c0, c0 + G, 1, i, j, k, v0 / (v0 - v2)); }
      if (k < G - 1) { var v3 = F[c0 + G * G]; if ((v0 < 0) !== (v3 < 0)) emit(c0, c0 + G * G, 2, i, j, k, v0 / (v0 - v3)); }
    }
    return { n: px.length, px: Float32Array.from(px), py: Float32Array.from(py), pz: Float32Array.from(pz), nx: Float32Array.from(nx), ny: Float32Array.from(ny), nz: Float32Array.from(nz) };
  }
  function targetField(G){ var F = new Float32Array(G * G * G), h = 2 / G; for (var k = 0; k < G; k++) for (var j = 0; j < G; j++) for (var i = 0; i < G; i++) F[(k * G + j) * G + i] = sdf((i + 0.5) * h - 1, (j + 0.5) * h - 1, (k + 0.5) * h - 1); return F; }
  return { sdf: sdf, CLIP: CLIP, makeData: makeData, makeTest: makeTest, balancedAcc: balancedAcc, buildTree: buildTree, predictOne: predictOne, predictSet: predictSet, depthOf: depthOf, fillGrid: fillGrid, Model: Model, surface: surface, targetField: targetField, mkrng: mkrng };
})();

/* ---------- interface: schematic, 3D views, slices, charts, controls ---------- */
TV.mount = function(root, opts){
  opts = opts || {};
  var CW = 904, CH = 500, dpr = Math.min(window.devicePixelRatio || 1, 1.5), G = 64, GT = 80, CLIP = TV.CLIP;
  var OR = [216, 90, 48], BL = [55, 138, 221], GR = [104, 112, 128];
  var TXT = opts.txt || '#2a2c33', TX2 = opts.tx2 || '#5f636d', BRD = opts.brd || '#c3c6cd', BG = opts.bg || '#ffffff';
  root.innerHTML = '<canvas class="tvc" style="display:block;width:' + CW + 'px;height:' + CH + 'px;touch-action:none"></canvas>' + (opts.ctl ? '' : '<div class="tvctl" style="display:flex;flex-wrap:wrap;gap:6px 14px;align-items:center;font-size:12.5px;color:' + TX2 + ';padding:7px 2px 0"></div>');
  var cvs = root.querySelector('canvas'), ctl = opts.ctl || root.querySelector('.tvctl'), g = cvs.getContext('2d'), bs = 0;
  var S = { kind: opts.kind || 'boost', N: opts.N || 10000, depth: opts.depth || 8, trees: opts.trees || 150, lr: opts.lr || 0.15, minLeaf: opts.minLeaf || 3, mtry: opts.mtry || 3, seed: 1, boxes: false, z: 0 };
  var model = null, data = null, test = TV.makeTest(1), shown = 0, sel = 0, follow = true, training = false, playing = false, probe = [0.45, 0, 0], sweepRes = {}, sweeping = '', ang = 0.7, tilt = 0.5, drag = null, dirty = true, accTree = [], tgtSurf = null, lrnSurf = null, tgtSlice = null, lrnSlice = null, stripCache = {}, lastFrame = 0, vis = true, rotOn = true, vDirty = true, lastRot = 0, cost = 0;
  function sel1(id, label, opts2, val){ return '<label>' + label + ' <select id="' + id + '" style="height:28px;font-size:12.5px">' + opts2.map(function(o){ return '<option value="' + o[0] + '"' + (String(o[0]) === String(val) ? ' selected' : '') + '>' + o[1] + '</option>'; }).join('') + '</select></label>'; }
  var rng20 = []; for (var d0 = 1; d0 <= 20; d0++) rng20.push([d0, d0]);
  if (opts.ctl) { ctl.style.cssText = 'display:flex;flex-wrap:wrap;gap:6px 14px;align-items:center;font-size:12.5px;padding:6px 0;flex-basis:100%;width:100%'; }
  ctl.innerHTML =
    sel1('tvk', 'Model', [['tree', 'Decision tree'], ['forest', 'Random forest'], ['boost', 'Gradient boosting']], S.kind) +
    sel1('tvn', 'Training points', [[10, '10'], [100, '100'], [1000, '1,000'], [10000, '10,000'], [30000, '30,000']], S.N) +
    sel1('tvd', 'Maximum depth', rng20, S.depth) +
    sel1('tvt', 'Trees or rounds', [[1, '1'], [3, '3'], [10, '10'], [30, '30'], [60, '60'], [100, '100'], [150, '150'], [200, '200'], [300, '300']], S.trees) +
    sel1('tvl', 'Learning rate', [[0.05, '0.05'], [0.1, '0.1'], [0.15, '0.15'], [0.3, '0.3'], [1, '1']], S.lr) +
    sel1('tvm', 'Minimum leaf', [[1, '1'], [2, '2'], [3, '3'], [10, '10'], [30, '30']], S.minLeaf) +
    sel1('tvf', 'Features per split', [[1, '1'], [2, '2'], [3, '3']], S.mtry) +
    '<span style="display:flex;gap:8px;align-items:center;flex-basis:100%"><button type="button" id="tvp" style="height:28px">Play</button><button type="button" id="tvr" style="height:28px">Restart</button><button type="button" id="tvs" style="height:28px">New random seed</button>' +
    '<label id="tvpl">Steps <input type="range" id="tvg" min="0" max="0" value="0" style="width:150px;vertical-align:middle"> <span id="tvgo"></span></label>' +
    '<label>Slice height <input type="range" id="tvz" min="-85" max="85" value="0" style="width:110px;vertical-align:middle"></label>' +
    '<label><input type="checkbox" id="tvb"> Show partition boxes</label><label><input type="checkbox" id="tvo" checked> Rotate</label></span>' +
    '<span style="display:flex;gap:8px;align-items:center;flex-basis:100%">Presets: <button type="button" id="tvc" style="height:28px">Crisp helix</button><button type="button" id="tvh" style="height:28px">Few points, shallow tree</button><button type="button" id="tvw" style="height:28px">Run the training-points sweep</button><span id="tvst"></span></span>';
  function $(id){ return ctl.querySelector('#' + id); }
  function setSel(id, v){ $(id).value = String(v); }
  function mkView(P){ var c = document.createElement('canvas'); c.width = P; c.height = P; return { c: c, x: c.getContext('2d'), P: P, img: null, zb: null }; }
  var vT = null, vL = null, VS = 220;
  function ensure(){
    var r = cvs.getBoundingClientRect(), vs = r.width ? r.width / CW : 1, want = Math.max(1, Math.ceil(dpr * Math.max(vs, 1) * 4) / 4);
    if (want !== bs) { bs = want; cvs.width = Math.round(CW * bs); cvs.height = Math.round(CH * bs); var P = Math.round(VS * bs); vT = mkView(P); vL = mkView(P); vT.img = vT.x.createImageData(P, P); vL.img = vL.x.createImageData(P, P); vT.zb = new Float32Array(P * P); vL.zb = new Float32Array(P * P); dirty = true; }
  }
  function splatView(v, surf, base, Gs){
    var P = v.P, img = v.img.data, zb = v.zb, i, sc = P * 0.43, ca = Math.cos(ang), sa = Math.sin(ang), ct = Math.cos(tilt), st = Math.sin(tilt), sq = Math.max(2, Math.ceil(2 / Gs * sc * 1.5)), lx = -0.45, lz = 0.7;
    img.fill(0); zb.fill(1e9);
    for (i = 0; i < surf.n; i++) {
      var x = surf.px[i] * ca - surf.py[i] * sa, y1 = surf.px[i] * sa + surf.py[i] * ca, z = surf.pz[i], y2 = y1 * ct - z * st, z2 = y1 * st + z * ct;
      var sx = P / 2 + x * sc, sy = P / 2 - z2 * sc;
      var nx0 = surf.nx[i] * ca - surf.ny[i] * sa, ny1 = surf.nx[i] * sa + surf.ny[i] * ca, nz0 = surf.nz[i], ny2 = ny1 * ct - nz0 * st, nz2 = ny1 * st + nz0 * ct;
      var sh = 0.3 + 0.7 * Math.max(0, nx0 * lx + nz2 * lz - ny2 * 0.35), fog = 0.8 + 0.2 * (1 - (y2 + 1.1) / 2.2);
      var r = Math.min(255, base[0] * sh * fog + 18 * sh), gg = Math.min(255, base[1] * sh * fog + 18 * sh), b = Math.min(255, base[2] * sh * fog + 18 * sh);
      var x0 = Math.round(sx - sq / 2), y0 = Math.round(sy - sq / 2);
      for (var yy = y0; yy < y0 + sq; yy++) { if (yy < 0 || yy >= P) continue; for (var xx = x0; xx < x0 + sq; xx++) { if (xx < 0 || xx >= P) continue; var o = yy * P + xx; if (y2 < zb[o]) { zb[o] = y2; var q = o * 4; img[q] = r; img[q + 1] = gg; img[q + 2] = b; img[q + 3] = 255; } } }
    }
    v.x.putImageData(v.img, 0, 0);
  }
  function sliceImg(size, fn, col){
    var c = document.createElement('canvas'); c.width = size; c.height = size; var x = c.getContext('2d'), im = x.createImageData(size, size), i, j;
    for (j = 0; j < size; j++) for (i = 0; i < size; i++) { var v = fn(i, j), t = v < 0 ? Math.min(1, -v / 0.1) : 0, o = (j * size + i) * 4; im.data[o] = 244 + (col[0] - 244) * t; im.data[o + 1] = 244 + (col[1] - 244) * t; im.data[o + 2] = 240 + (col[2] - 240) * t; im.data[o + 3] = 255; }
    x.putImageData(im, 0, 0); return c;
  }
  function makeTgtSlice(){ tgtSlice = sliceImg(100, function(i, j){ return TV.sdf((i + 0.5) / 100 * 2 - 1, (j + 0.5) / 100 * 2 - 1, S.z); }, GR); }
  function currentField(){
    if (!model) return null;
    if (S.kind === 'tree') return model.treeField(shown);
    if (!model.snaps.length) return null;
    return model.snaps[Math.min(shown, model.snaps.length - 1)].F;
  }
  function refreshLearned(){
    var F = currentField(); if (!F) { lrnSurf = null; lrnSlice = null; dirty = true; return; }
    lrnSurf = TV.surface(F, G); var k = Math.max(0, Math.min(G - 1, Math.round((S.z + 1) / 2 * G - 0.5)));
    lrnSlice = sliceImg(G, function(i, j){ return F[(k * G + j) * G + i]; }, OR); stripCache = {}; dirty = true;
  }
  function updateSlider(){
    var sl = $('tvg');
    if (S.kind === 'tree') { var md = model ? TV.depthOf(model.trees[0]) : 1; sl.min = 1; sl.max = Math.max(1, md); sl.value = shown; $('tvgo').textContent = 'depth ' + shown + ' of ' + Math.max(1, md); }
    else { var n = model ? model.snaps.length : 0; sl.min = 0; sl.max = Math.max(0, n - 1); sl.value = Math.min(shown, Math.max(0, n - 1)); $('tvgo').textContent = n ? (model.snaps[Math.min(shown, n - 1)].count + ' of ' + S.trees + (S.kind === 'forest' ? ' trees' : ' rounds')) : ''; }
    $('tvf').disabled = S.kind !== 'forest'; $('tvl').disabled = S.kind !== 'boost'; $('tvt').disabled = S.kind === 'tree';
  }
  function train(){
    data = TV.makeData(S.N, S.seed); stripCache = {}; sel = 0; follow = true; playing = false; $('tvp').textContent = 'Pause';
    model = new TV.Model(S.kind, data, test, { seed: S.seed, depth: S.depth, minLeaf: S.minLeaf, mtry: S.mtry, lr: S.lr, trees: S.trees, G: G });
    if (S.kind === 'tree') { model.addTree(); training = false; $('tvp').textContent = 'Play'; shown = TV.depthOf(model.trees[0]); accTree = []; for (var d = 1; d <= shown; d++) accTree.push(model.treeAcc(d)); }
    else { training = true; shown = 0; accTree = []; }
    updateSlider(); refreshLearned(); dirty = true;
  }
  function stepTraining(){
    if (!training || !model) return; var t0 = performance.now(), before = model.snaps.length;
    while (performance.now() - t0 < 14 && model.trees.length < S.trees) model.addTree();
    if (model.trees.length >= S.trees) { training = false; $('tvp').textContent = 'Play'; }
    if (model.snaps.length !== before) { if (follow) shown = model.snaps.length - 1; updateSlider(); refreshLearned(); }
    dirty = true;
  }
  function mix(c, t, base){ return [Math.round(base[0] + (c[0] - base[0]) * t), Math.round(base[1] + (c[1] - base[1]) * t), Math.round(base[2] + (c[2] - base[2]) * t)]; }
  function rgb(c, f){ f = f || 1; return 'rgb(' + Math.min(255, Math.round(c[0] * f)) + ',' + Math.min(255, Math.round(c[1] * f)) + ',' + Math.min(255, Math.round(c[2] * f)) + ')'; }
  function drawCell(x, cx, cy, s, lo, hi, val){
    var k = s / 2.95, ox = 0.45, oy = 0.3;
    function P(px, py, pz){ return [cx + (px + ox * py) * k, cy - (pz + oy * py) * k]; }
    function poly(pts, fill, stroke){ x.beginPath(); x.moveTo(pts[0][0], pts[0][1]); for (var i = 1; i < pts.length; i++) x.lineTo(pts[i][0], pts[i][1]); x.closePath(); if (fill) { x.fillStyle = fill; x.fill(); } if (stroke) { x.strokeStyle = stroke; x.lineWidth = 0.7; x.stroke(); } }
    var wire = '#c3c6cd', a, b;
    poly([P(-1, -1, -1), P(1, -1, -1), P(1, -1, 1), P(-1, -1, 1)], null, wire); poly([P(-1, 1, 1), P(1, 1, 1), P(1, 1, -1)], null, wire);
    x.beginPath(); a = P(-1, -1, 1); b = P(-1, 1, 1); x.moveTo(a[0], a[1]); x.lineTo(b[0], b[1]); a = P(1, -1, 1); b = P(1, 1, 1); x.moveTo(a[0], a[1]); x.lineTo(b[0], b[1]); a = P(1, -1, -1); b = P(1, 1, -1); x.moveTo(a[0], a[1]); x.lineTo(b[0], b[1]); x.strokeStyle = wire; x.lineWidth = 0.7; x.stroke();
    var t = Math.max(-1, Math.min(1, -val / 0.15)), c = t >= 0 ? mix(OR, t, [232, 233, 236]) : mix(GR, -t * 0.5, [232, 233, 236]);
    var x0 = lo[0], x1 = hi[0], y0 = lo[1], y1 = hi[1], z0 = lo[2], z1 = hi[2];
    poly([P(x0, y0, z0), P(x1, y0, z0), P(x1, y0, z1), P(x0, y0, z1)], rgb(c, 1), 'rgba(40,44,52,0.55)');
    poly([P(x0, y0, z1), P(x1, y0, z1), P(x1, y1, z1), P(x0, y1, z1)], rgb(c, 1.1), 'rgba(40,44,52,0.55)');
    poly([P(x1, y0, z0), P(x1, y1, z0), P(x1, y1, z1), P(x1, y0, z1)], rgb(c, 0.8), 'rgba(40,44,52,0.55)');
  }
  function pathOf(t, cap){ var nd = 0, out = [0]; while (t.feat[nd] >= 0 && t.dep[nd] < cap) { var v = probe[t.feat[nd]]; nd = v < t.thr[nd] ? t.left[nd] : t.right[nd]; out.push(nd); } return out; }
  function drawTree(x, t, cap){
    var yc = [52, 116, 180, 244, 304], sz = [50, 42, 35, 29, 25], names = ['x', 'y', 'z'], onPath = {}, pp = pathOf(t, cap), i;
    for (i = 0; i < pp.length; i++) onPath[pp[i]] = 1;
    var items = [{ nd: 0, L: 0, i: 0 }], pos = {}, q;
    for (q = 0; q < items.length; q++) {
      var it = items[q]; it.x = 10 + (it.i + 0.5) * 424 / Math.pow(2, it.L); it.y = yc[it.L]; pos[it.nd] = it;
      if (it.L < 4 && t.feat[it.nd] >= 0 && t.dep[it.nd] < cap) { items.push({ nd: t.left[it.nd], L: it.L + 1, i: it.i * 2, par: it.nd }); items.push({ nd: t.right[it.nd], L: it.L + 1, i: it.i * 2 + 1, par: it.nd }); }
    }
    for (q = 1; q < items.length; q++) { var c = items[q], p = pos[c.par], w = 0.7 + 2.4 * Math.sqrt(t.cnt[c.nd] / t.cnt[c.par]), hot = onPath[c.nd] && onPath[c.par]; x.strokeStyle = hot ? rgb(OR) : '#b9bcc4'; x.lineWidth = hot ? Math.max(2, w) : w; x.beginPath(); x.moveTo(p.x, p.y + sz[p.L] / 2 + 7); x.lineTo(c.x, c.y - sz[c.L] / 2 - 2); x.stroke(); }
    x.textAlign = 'left';
    for (q = 0; q < items.length; q++) {
      var n = items[q], s = sz[n.L], nd = n.nd, lo = [t.lo[3 * nd], t.lo[3 * nd + 1], t.lo[3 * nd + 2]], hi = [t.hi[3 * nd], t.hi[3 * nd + 1], t.hi[3 * nd + 2]];
      if (onPath[nd]) { x.strokeStyle = rgb(OR); x.lineWidth = 1.6; x.strokeRect(n.x - s / 2 - 2, n.y - s / 2 - 2, s + 4, s + 4); }
      drawCell(x, n.x, n.y, s, lo, hi, t.val[nd]);
      x.fillStyle = '#e3e5ea'; x.fillRect(n.x - s / 2, n.y + s / 2 + 2, s, 3); x.fillStyle = rgb(OR); x.fillRect(n.x - s / 2, n.y + s / 2 + 2, s * t.neg[nd], 3);
      x.fillStyle = TX2; x.font = '10.5px Aptos,Segoe UI,sans-serif';
      if (n.L <= 2) { var internal = t.feat[nd] >= 0 && t.dep[nd] < cap; x.fillText(internal ? names[t.feat[nd]] + ' < ' + t.thr[nd].toFixed(2) : 'leaf', n.x + s / 2 + 5, n.y - 1); x.fillText(t.cnt[nd].toLocaleString() + (n.L <= 1 ? ' points' : ''), n.x + s / 2 + 5, n.y + 11); }
      if (n.L === 4 && t.feat[nd] >= 0 && t.dep[nd] < cap) { x.strokeStyle = '#b9bcc4'; x.lineWidth = 0.8; x.beginPath(); x.moveTo(n.x, n.y + s / 2 + 5); x.lineTo(n.x - 5, n.y + s / 2 + 12); x.moveTo(n.x, n.y + s / 2 + 5); x.lineTo(n.x + 5, n.y + s / 2 + 12); x.stroke(); }
    }
    x.textAlign = 'left';
    var nodes = 0, leaves = 0, byDepth = [], mxd = 0;
    for (i = 0; i < t.nodes; i++) { if (t.dep[i] > cap) continue; nodes++; var isLeaf = t.feat[i] < 0 || t.dep[i] === cap; if (isLeaf) { leaves++; byDepth[t.dep[i]] = (byDepth[t.dep[i]] || 0) + 1; } if (t.dep[i] > mxd) mxd = t.dep[i]; }
    return { nodes: nodes, leaves: leaves, byDepth: byDepth, maxDepth: mxd, path: pp };
  }
  function line(x, pts, col, w){ x.strokeStyle = col; x.lineWidth = w; x.beginPath(); for (var i = 0; i < pts.length; i++) i ? x.lineTo(pts[i][0], pts[i][1]) : x.moveTo(pts[i][0], pts[i][1]); x.stroke(); }
  function chart(x, cx, cy, cw, ch, title, series, xmin, xmax, logx, marks, xlab){
    x.strokeStyle = BRD; x.lineWidth = 0.7; x.strokeRect(cx, cy, cw, ch); x.fillStyle = TX2; x.font = '11px Aptos,Segoe UI,sans-serif'; x.textAlign = 'left'; x.fillText(title, cx, cy - 6);
    x.fillText('1.0', cx - 22, cy + 8); x.fillText('0.5', cx - 22, cy + ch); x.setLineDash([3, 3]); line(x, [[cx, cy + ch * 0.5], [cx + cw, cy + ch * 0.5]], '#d5d8de', 0.7); x.setLineDash([]);
    function fx(v){ return cx + (logx ? (Math.log(v) - Math.log(xmin)) / (Math.log(xmax) - Math.log(xmin)) : (v - xmin) / (xmax - xmin || 1)) * cw; }
    function fy(v){ return cy + ch - (Math.max(0.5, v) - 0.5) * 2 * ch; }
    series.forEach(function(s){ if (s.pts.length) { line(x, s.pts.map(function(p){ return [fx(p[0]), fy(p[1])]; }), s.col, 2); if (s.dots) s.pts.forEach(function(p){ x.fillStyle = s.col; x.beginPath(); x.arc(fx(p[0]), fy(p[1]), 3, 0, 6.2832); x.fill(); }); } });
    (marks || []).forEach(function(m){ x.strokeStyle = TXT; x.lineWidth = 0.8; x.setLineDash([2, 2]); x.beginPath(); x.moveTo(fx(m), cy); x.lineTo(fx(m), cy + ch); x.stroke(); x.setLineDash([]); });
    if (xlab) { x.fillStyle = TX2; x.fillText(xlab, cx + cw - x.measureText(xlab).width, cy + ch + 11); }
  }
  function stripTile(x, tr, idx, cx, cy, s){
    var key = idx + ':' + S.z + ':' + S.kind, c = stripCache[key];
    if (!c) { c = document.createElement('canvas'); c.width = 24; c.height = 24; var q = c.getContext('2d'), im = q.createImageData(24, 24), amp = CLIP, vals = new Float32Array(576), mxa = 1e-6, j, i; for (j = 0; j < 24; j++) for (i = 0; i < 24; i++) { var v0 = TV.predictOne(tr, (i + 0.5) / 12 - 1, (j + 0.5) / 12 - 1, S.z, 99); if (S.kind === 'boost') v0 *= S.lr; vals[j * 24 + i] = v0; if (Math.abs(v0) > mxa) mxa = Math.abs(v0); } if (S.kind === 'boost') amp = mxa; for (j = 0; j < 24; j++) for (i = 0; i < 24; i++) { var v = vals[j * 24 + i]; var tt = Math.max(-1, Math.min(1, -v / amp)), col = tt >= 0 ? mix(OR, tt, [244, 244, 240]) : (S.kind === 'boost' ? mix(BL, -tt, [244, 244, 240]) : mix(GR, -tt * 0.18, [244, 244, 240])), o = (j * 24 + i) * 4; im.data[o] = col[0]; im.data[o + 1] = col[1]; im.data[o + 2] = col[2]; im.data[o + 3] = 255; } q.putImageData(im, 0, 0); stripCache[key] = c; }
    x.drawImage(c, cx, cy, s, s); x.strokeStyle = idx === sel ? rgb(OR) : BRD; x.lineWidth = idx === sel ? 2 : 0.6; x.strokeRect(cx, cy, s, s);
  }
  var layer = document.createElement('canvas'), lg = null;
  function drawStatic(){
    if (!lg || layer.width !== cvs.width) { layer.width = cvs.width; layer.height = cvs.height; lg = layer.getContext('2d'); }
    var x = lg; x.setTransform(1, 0, 0, 1, 0, 0); x.clearRect(0, 0, layer.width, layer.height); x.setTransform(bs, 0, 0, bs, 0, 0);
    x.fillStyle = BG; x.fillRect(0, 0, CW, CH); x.font = '11.5px Aptos,Segoe UI,sans-serif'; x.textBaseline = 'alphabetic';
    var kindName = S.kind === 'tree' ? 'Decision tree' : (S.kind === 'forest' ? 'Random forest' : 'Gradient boosting'), info = null, cnt = 0, tr = null, cap = 99;
    if (model && model.trees.length) {
      if (S.kind === 'tree') { tr = model.trees[0]; cap = shown; } else { cnt = model.snaps.length ? model.snaps[Math.min(shown, model.snaps.length - 1)].count : 0; if (sel >= cnt) sel = Math.max(0, cnt - 1); tr = model.trees[Math.min(sel, model.trees.length - 1)]; }
    }
    x.fillStyle = TXT; x.font = '12.5px Aptos,Segoe UI,sans-serif';
    x.fillText(kindName + (S.kind === 'tree' ? ', depth ' + shown : ' of ' + (cnt || 0) + (S.kind === 'forest' ? ' trees' : ' rounds')) + (S.kind !== 'tree' && tr ? '. Tree ' + (sel + 1) + ' drawn.' : ''), 6, 14);
    if (tr) info = drawTree(x, tr, cap);
    x.fillStyle = TX2; x.font = '11px Aptos,Segoe UI,sans-serif';
    if (S.kind === 'tree' && info) {
      x.fillText('Nodes ' + info.nodes.toLocaleString() + ', leaves ' + info.leaves.toLocaleString() + ', depth ' + info.maxDepth + '. Leaves by depth:', 6, 354);
      var mx = 1; info.byDepth.forEach(function(v){ if (v > mx) mx = v; }); var bw = Math.min(18, 410 / Math.max(1, info.byDepth.length));
      for (var dd = 0; dd < info.byDepth.length; dd++) { var h2 = ((info.byDepth[dd] || 0) / mx) * 38; x.fillStyle = dd < 5 ? rgb(GR) : rgb(OR); x.fillRect(8 + dd * (bw + 2), 404 - h2, bw, h2); x.fillStyle = TX2; if (dd % 2 === 0 || bw > 14) x.fillText(String(dd), 8 + dd * (bw + 2) + 2, 416); }
      x.fillStyle = TX2; x.fillText('Grey bars: levels drawn above. Orange: deeper levels, not drawn.', 6, 430);
    } else if (info) {
      var first = Math.max(0, cnt - 14), tsz = 26; x.fillText(S.kind === 'forest' ? 'Each tile: one tree on the slice. Click one to draw it.' : 'Each tile: the correction of one round on the slice. Click one to draw it.', 6, 354);
      for (var tt2 = first; tt2 < cnt; tt2++) stripTile(x, model.trees[tt2], tt2, 8 + (tt2 - first) * (tsz + 4), 364, tsz);
      if (cnt > 14) { x.fillStyle = TX2; x.fillText('showing the last 14 of ' + cnt, 8, 406); }
      if (lrnSlice) { x.drawImage(lrnSlice, 380, 364, 38, 38); x.strokeStyle = BRD; x.strokeRect(380, 364, 38, 38); x.fillStyle = TX2; x.fillText(S.kind === 'forest' ? 'mean' : 'sum', 384, 414); }
    }
    var series = [], marks = [], accs = S.kind === 'tree' ? accTree : (model ? model.acc : []);
    series.push({ pts: accs.map(function(a, i){ return [i + 1, a]; }), col: rgb(OR) });
    if (S.kind === 'tree') { marks.push(shown); chart(x, 24, 452, 396, 38, 'Held-out accuracy against depth', series, 1, Math.max(2, accTree.length), false, marks, 'depth'); }
    else { if (cnt) marks.push(cnt); chart(x, 24, 452, 396, 38, 'Held-out accuracy against ' + (S.kind === 'forest' ? 'number of trees' : 'rounds') + ' (log axis)', series, 1, Math.max(2, S.trees), true, marks, S.kind === 'forest' ? 'trees' : 'rounds'); }
    x.fillStyle = TX2; x.font = '11.5px Aptos,Segoe UI,sans-serif'; x.fillText('Target helix', 452, 14); x.fillText('Learned by the model', 682, 14);
    x.fillText('Target slice, z = ' + S.z.toFixed(2), 452, 270); x.fillText('Learned slice', 566, 270);
    if (tgtSlice) { x.drawImage(tgtSlice, 452, 276, 100, 100); x.strokeStyle = BRD; x.lineWidth = 0.7; x.strokeRect(452, 276, 100, 100); }
    if (lrnSlice) { x.imageSmoothingEnabled = false; x.drawImage(lrnSlice, 566, 276, 100, 100); x.imageSmoothingEnabled = true; x.strokeStyle = BRD; x.strokeRect(566, 276, 100, 100); }
    [452, 566].forEach(function(ox){ var px = ox + (probe[0] + 1) / 2 * 100, py = 276 + (probe[1] + 1) / 2 * 100; x.strokeStyle = TXT; x.lineWidth = 1; x.beginPath(); x.moveTo(px - 5, py); x.lineTo(px + 5, py); x.moveTo(px, py - 5); x.lineTo(px, py + 5); x.stroke(); });
    var acc = S.kind === 'tree' ? (accTree.length ? accTree[Math.min(shown, accTree.length) - 1] : 0.5) : (model && model.acc.length && cnt ? model.acc[cnt - 1] : 0.5);
    x.fillStyle = TXT; x.font = '12px Aptos,Segoe UI,sans-serif'; var L2 = ['Balanced accuracy ' + acc.toFixed(3), 'Chance level 0.500', 'Training points ' + S.N.toLocaleString(), 'Held-out points ' + test.n.toLocaleString(), 'within 0.15 of the surface']; if (info) L2.push((S.kind === 'tree' ? 'Leaves ' : 'Leaves in the drawn tree ') + info.leaves.toLocaleString()); if (S.kind !== 'tree' && cnt) L2.push((S.kind === 'forest' ? 'Trees ' : 'Rounds ') + cnt + ' of ' + S.trees + (training ? ', training' : ''));
    L2.forEach(function(s, i){ x.fillText(s, 684, 290 + i * 17); });
    x.fillStyle = TX2; x.font = '11px Aptos,Segoe UI,sans-serif'; x.fillText('Cross at (' + probe[0].toFixed(2) + ', ' + probe[1].toFixed(2) + ', ' + S.z.toFixed(2) + ').', 452, 392); x.fillText('Click a slice to move the cross. Its path through the tree is outlined.', 452, 406);
    var kinds = ['tree', 'forest', 'boost'], cols = { tree: rgb(GR), forest: rgb(BL), boost: rgb(OR) }, nm = { tree: 'tree', forest: 'forest', boost: 'boosting' }, ser = [];
    kinds.forEach(function(k){ if (sweepRes[k]) ser.push({ pts: sweepRes[k].map(function(p){ return [p[0], p[1]]; }), col: cols[k], dots: true }); });
    chart(x, 470, 452, 410, 38, 'Held-out accuracy against training points (sweep, log axis)', ser, 10, 30000, true, [], 'points');
    var lx = 480; kinds.forEach(function(k){ if (sweepRes[k]) { x.fillStyle = cols[k]; x.fillRect(lx, 457, 8, 3); x.fillStyle = TX2; x.font = '10.5px Aptos,Segoe UI,sans-serif'; x.fillText(nm[k], lx + 11, 462); lx += 70; } });
    dirty = false;
  }
  function boxes(x, tr, cap){
    var sc = VS * 0.43, ca = Math.cos(ang), sa = Math.sin(ang), ct = Math.cos(tilt), st = Math.sin(tilt), ox = 682 + VS / 2, oy = 22 + VS / 2, n = 0;
    function P(px, py, pz){ var xx = px * ca - py * sa, y1 = px * sa + py * ca, z2 = y1 * st + pz * ct; return [ox + xx * sc, oy - z2 * sc]; }
    x.strokeStyle = 'rgba(30,34,40,0.45)'; x.lineWidth = 0.6;
    for (var nd = 0; nd < tr.nodes && n < 400; nd++) {
      if (!(tr.feat[nd] < 0 || tr.dep[nd] === cap)) continue; if (S.kind === 'boost' || tr.val[nd] > -0.04) continue;
      var lo = [tr.lo[3 * nd], tr.lo[3 * nd + 1], tr.lo[3 * nd + 2]], hi = [tr.hi[3 * nd], tr.hi[3 * nd + 1], tr.hi[3 * nd + 2]]; if ((hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]) > 0.25) continue; n++;
      var c = [[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]], [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]], [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]].map(function(q){ return P(q[0], q[1], q[2]); });
      x.beginPath(); [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]].forEach(function(e){ x.moveTo(c[e[0]][0], c[e[0]][1]); x.lineTo(c[e[1]][0], c[e[1]][1]); }); x.stroke();
    }
  }
  function frame(now){
    if (!vis || document.hidden || !cvs.offsetParent) { lastRot = 0; setTimeout(function(){ requestAnimationFrame(frame); }, 250); return; }
    ensure();
    var rot = rotOn && !drag && now - lastRot > Math.max(50, cost * 3);
    if (rot) { ang += 0.00048 * (lastRot ? Math.min(200, now - lastRot) : 16); lastRot = now; }
    stepTraining();
    if (playing && !training && model) { if (now - lastFrame > (S.kind === 'tree' ? 700 : 260)) { lastFrame = now; var mx = parseInt($('tvg').max, 10); if (shown >= mx) { playing = false; $('tvp').textContent = 'Play'; } else { shown++; follow = false; updateSlider(); refreshLearned(); } } }
    if (rot || dirty || vDirty) {
      var t0 = performance.now(); vDirty = false;
      if (dirty) drawStatic();
      g.setTransform(1, 0, 0, 1, 0, 0); g.drawImage(layer, 0, 0); g.setTransform(bs, 0, 0, bs, 0, 0);
      if (tgtSurf) { splatView(vT, tgtSurf, [118, 126, 142], GT); g.drawImage(vT.c, 452, 22, VS, VS); }
      g.strokeStyle = BRD; g.lineWidth = 0.7; g.strokeRect(452, 22, VS, VS); g.strokeRect(682, 22, VS, VS);
      if (lrnSurf) { splatView(vL, lrnSurf, [224, 104, 56], G); g.drawImage(vL.c, 682, 22, VS, VS); } else { g.fillStyle = TX2; g.font = '12px Aptos,Segoe UI,sans-serif'; g.fillText(model && (S.kind === 'tree' || model.snaps.length) ? 'The model calls every point outside,' : 'Training', 690, 130); if (model && (S.kind === 'tree' || model.snaps.length)) g.fillText('so there is no surface to draw.', 690, 146); }
      if (S.boxes && model && model.trees.length) boxes(g, S.kind === 'tree' ? model.trees[0] : model.trees[Math.min(sel, model.trees.length - 1)], S.kind === 'tree' ? shown : 99);
      cost = cost ? cost * 0.7 + (performance.now() - t0) * 0.3 : performance.now() - t0;
      if (cost > 45 && rotOn) { rotOn = false; $('tvo').checked = false; }
    }
    requestAnimationFrame(frame);
  }
  function readParams(){ S.kind = $('tvk').value; S.N = +$('tvn').value; S.depth = +$('tvd').value; S.trees = +$('tvt').value; S.lr = +$('tvl').value; S.minLeaf = +$('tvm').value; S.mtry = +$('tvf').value; }
  ['tvk', 'tvn', 'tvd', 'tvt', 'tvl', 'tvm', 'tvf'].forEach(function(id){ $(id).addEventListener('change', function(){ readParams(); train(); }); });
  $('tvr').addEventListener('click', function(){ train(); });
  $('tvs').addEventListener('click', function(){ S.seed++; train(); });
  $('tvp').addEventListener('click', function(){
    if (training) { training = false; $('tvp').textContent = 'Resume'; return; }
    if (model && S.kind !== 'tree' && model.trees.length < S.trees) { training = true; $('tvp').textContent = 'Pause'; return; }
    playing = !playing; if (playing) { shown = parseInt($('tvg').min, 10); follow = false; updateSlider(); refreshLearned(); } $('tvp').textContent = playing ? 'Stop' : 'Play';
  });
  $('tvg').addEventListener('input', function(){ shown = parseInt(this.value, 10); follow = S.kind === 'tree' ? false : shown >= (model ? model.snaps.length - 1 : 0); playing = false; $('tvp').textContent = 'Play'; updateSlider(); refreshLearned(); });
  $('tvz').addEventListener('input', function(){ S.z = this.value / 100; makeTgtSlice(); refreshLearned(); });
  $('tvb').addEventListener('change', function(){ S.boxes = this.checked; vDirty = true; });
  $('tvo').addEventListener('change', function(){ rotOn = this.checked; cost = 0; });
  if ('IntersectionObserver' in window) new IntersectionObserver(function(es){ vis = es[es.length - 1].isIntersecting; if (vis) vDirty = true; }).observe(cvs);
  function applyPreset(p){ S.kind = p.kind; S.N = p.N; S.depth = p.depth; S.trees = p.trees; S.lr = p.lr; S.minLeaf = p.minLeaf; S.mtry = p.mtry || 3; setSel('tvk', S.kind); setSel('tvn', S.N); setSel('tvd', S.depth); setSel('tvt', S.trees); setSel('tvl', S.lr); setSel('tvm', S.minLeaf); setSel('tvf', S.mtry); train(); }
  $('tvc').addEventListener('click', function(){ applyPreset({ kind: 'boost', N: 30000, depth: 10, trees: 200, lr: 0.15, minLeaf: 3 }); });
  $('tvh').addEventListener('click', function(){ applyPreset({ kind: 'tree', N: 1000, depth: 8, trees: 1, lr: 0.15, minLeaf: 1 }); });
  $('tvw').addEventListener('click', function(){
    if (sweeping) return; var kind = S.kind, Ns = [10, 100, 1000, 10000, 30000], res = [], i = 0; sweeping = kind; sweepRes[kind] = res; var st = $('tvst');
    (function next(){
      if (i >= Ns.length) { sweeping = ''; st.textContent = ''; dirty = true; return; }
      st.textContent = 'Sweep, ' + kind + ': ' + Ns[i].toLocaleString() + ' points';
      setTimeout(function(){
        var d2 = TV.makeData(Ns[i], S.seed), m2 = new TV.Model(kind, d2, test, { seed: S.seed, depth: S.depth, minLeaf: S.minLeaf, mtry: S.mtry, lr: S.lr, trees: S.trees, G: 8 }), n = kind === 'tree' ? 1 : S.trees, j = 0;
        (function grow(){ var t0 = performance.now(); while (j < n && performance.now() - t0 < 25) { m2.addTree(); j++; } if (j < n) { setTimeout(grow, 0); return; } res.push([Ns[i], kind === 'tree' ? m2.treeAcc(S.depth) : m2.acc[m2.acc.length - 1]]); i++; dirty = true; next(); })();
      }, 20);
    })();
  });
  function pos(ev){ var r = cvs.getBoundingClientRect(); return [(ev.clientX - r.left) / r.width * CW, (ev.clientY - r.top) / r.height * CH]; }
  cvs.addEventListener('pointerdown', function(ev){
    var p = pos(ev);
    if (p[0] >= 452 && p[0] <= 666 && p[1] >= 276 && p[1] <= 376) { if (p[0] <= 552) probe = [(p[0] - 452) / 100 * 2 - 1, (p[1] - 276) / 100 * 2 - 1, S.z]; else if (p[0] >= 566) probe = [(p[0] - 566) / 100 * 2 - 1, (p[1] - 276) / 100 * 2 - 1, S.z]; dirty = true; return; }
    if (S.kind !== 'tree' && p[1] >= 364 && p[1] <= 390 && p[0] >= 8 && p[0] < 8 + 14 * 30 && model) { var cnt = model.snaps.length ? model.snaps[Math.min(shown, model.snaps.length - 1)].count : 0, first = Math.max(0, cnt - 14), k = first + Math.floor((p[0] - 8) / 30); if (k < cnt) { sel = k; dirty = true; } return; }
    if (p[0] >= 452 && p[0] <= 902 && p[1] >= 22 && p[1] <= 242) { drag = [ev.clientX, ev.clientY, ang, tilt]; cvs.setPointerCapture(ev.pointerId); }
  });
  cvs.addEventListener('pointermove', function(ev){ if (!drag) return; ang = drag[2] + (ev.clientX - drag[0]) * 0.01; tilt = Math.max(-0.2, Math.min(1.4, drag[3] + (ev.clientY - drag[1]) * 0.008)); vDirty = true; });
  cvs.addEventListener('pointerup', function(){ drag = null; });
  ensure(); makeTgtSlice();
  setTimeout(function(){ tgtSurf = TV.surface(TV.targetField(GT), GT); dirty = true; train(); requestAnimationFrame(frame); }, 30);
  return { state: S, getModel: function(){ return model; }, setPreset: applyPreset, getAcc: function(){ return S.kind === 'tree' ? accTree[shown - 1] : (model && model.acc.length ? model.acc[model.acc.length - 1] : 0); }, isTraining: function(){ return training; } };
};

if (typeof module !== 'undefined') module.exports = TV;
