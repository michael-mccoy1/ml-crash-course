"use strict";
/* Shared by the 3D shape steps: a torus, sampled points, accuracy on held-out points and a rotatable 3D view. */
var SH = { R: 1, r: 0.4, L: 1.7, GRID: 26 };
/* signed distance to the torus surface: negative inside, positive outside */
function shSD(x, y, z){ var q = Math.sqrt(x * x + y * y) - SH.R; return Math.sqrt(q * q + z * z) - SH.r; }
/* half of the points anywhere in the cube, half within 0.6 of the surface */
function shTrain(n, seed){
  var rnd = rng(seed), X = new Float32Array(n * 3), T = new Float32Array(n), k = 0;
  while (k < n) { var x = (rnd() * 2 - 1) * SH.L, y = (rnd() * 2 - 1) * SH.L, z = (rnd() * 2 - 1) * SH.L, d = shSD(x, y, z); if (k % 2 === 0 || Math.abs(d) < 0.6) { X[k * 3] = x; X[k * 3 + 1] = y; X[k * 3 + 2] = z; T[k] = clamp(d, -0.4, 0.4); k++; } }
  return { X: X, T: T, n: n };
}
/* held-out points within 0.5 of the surface, half inside and half outside */
function shTest(n, seed){
  var rnd = rng(seed), X = new Float32Array(n * 3), inside = new Uint8Array(n), k = 0, ni = 0, no = 0;
  while (k < n) { var x = (rnd() * 2 - 1) * SH.L, y = (rnd() * 2 - 1) * SH.L, z = (rnd() * 2 - 1) * SH.L, d = shSD(x, y, z); if (Math.abs(d) > 0.5) continue; var ins = d < 0 ? 1 : 0; if (ins ? ni >= n / 2 : no >= n / 2) continue; X[k * 3] = x; X[k * 3 + 1] = y; X[k * 3 + 2] = z; inside[k] = ins; if (ins) ni++; else no++; k++; }
  return { X: X, inside: inside, n: n };
}
/* balanced accuracy: the average of the share of inside points called inside and outside points called outside */
function shAccuracy(test, predict){
  var tp = 0, tn = 0, pos = 0, neg = 0;
  for (var i = 0; i < test.n; i++) { var p = predict(test.X[i * 3], test.X[i * 3 + 1], test.X[i * 3 + 2]) < 0 ? 1 : 0; if (test.inside[i]) { pos++; tp += p; } else { neg++; tn += 1 - p; } }
  return (tp / pos + tn / neg) / 2;
}
/* points that a model calls inside, on a regular grid */
function shGridInside(predict){
  var out = [], G = SH.GRID, i, j, k; for (i = 0; i < G; i++) for (j = 0; j < G; j++) for (k = 0; k < G; k++) { var x = (i / (G - 1) * 2 - 1) * SH.L, y = (j / (G - 1) * 2 - 1) * SH.L, z = (k / (G - 1) * 2 - 1) * SH.L; if (predict(x, y, z) < 0) out.push([x, y, z]); }
  return out;
}
function shSurface(){
  if (shSurface.pts) return shSurface.pts; var pts = [], u, v; for (u = 0; u < 44; u++) for (v = 0; v < 14; v++) { var a = u / 44 * 6.2832, b = v / 14 * 6.2832; pts.push([(SH.R + SH.r * Math.cos(b)) * Math.cos(a), (SH.R + SH.r * Math.cos(b)) * Math.sin(a), SH.r * Math.sin(b)]); }
  return (shSurface.pts = pts);
}
/* draws the true surface in grey and the model's inside points in red. Drag to rotate. */
function shDraw(canvas, st){
  var g = cv(canvas), x = g.x, cy = Math.cos(st.yaw), sy = Math.sin(st.yaw), cp = Math.cos(st.pitch), sp = Math.sin(st.pitch), S = g.h / 4.1, ox = g.w / 2, oy = g.h / 2;
  function proj(p){ var a = p[0] * cy - p[1] * sy, b = p[0] * sy + p[1] * cy, c = p[2], d = b * cp - c * sp, e = b * sp + c * cp; return [ox + a * S, oy - e * S, d]; }
  var items = []; shSurface().forEach(function(p){ var q = proj(p); items.push([q[0], q[1], q[2], 0]); }); (st.pts || []).forEach(function(p){ var q = proj(p); items.push([q[0], q[1], q[2], 1]); });
  items.sort(function(a, b){ return a[2] - b[2]; });
  items.forEach(function(it){ if (it[3] === 0) { x.fillStyle = 'rgba(52,55,63,.55)'; x.fillRect(it[0] - 1.1, it[1] - 1.1, 2.2, 2.2); } else { x.fillStyle = 'rgba(225,36,42,.5)'; x.fillRect(it[0] - 2, it[1] - 2, 4, 4); } });
  x.fillStyle = C.char; x.font = '12px Aptos, Segoe UI, sans-serif'; x.textAlign = 'left'; x.fillText('Drag to rotate. Grey: the true surface. Red: grid points the model calls inside.', 8, g.h - 8);
}
function shRotatable(canvas, st){
  var drag = false, lx = 0, ly = 0;
  canvas.addEventListener('pointerdown', function(e){ drag = true; lx = e.clientX; ly = e.clientY; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener('pointermove', function(e){ if (!drag) return; st.yaw += (e.clientX - lx) * 0.01; st.pitch = clamp(st.pitch + (e.clientY - ly) * 0.01, -1.5, 1.5); lx = e.clientX; ly = e.clientY; shDraw(canvas, st); });
  canvas.addEventListener('pointerup', function(){ drag = false; });
}
