"""Export the compact dataset used by the website -> results/applet_data.json.

Needs results/logd_pred.csv (run notebook 05 first). Optional: results/results.json (notebooks 05 and 06 add scores).
"""
import json, os, sys
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, rdMolDescriptors as R

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as c

df = c.load("HLM")
tr, te = c.split(df, "scaffold")
fns = {"logP": Crippen.MolLogP, "TPSA": R.CalcTPSA, "MW": Descriptors.MolWt, "HBD": R.CalcNumHBD, "HBA": R.CalcNumHBA,
       "RotB": R.CalcNumRotatableBonds, "Fsp3": R.CalcFractionCSP3, "AromRings": R.CalcNumAromaticRings}
X = np.array([[f(m) for f in fns.values()] for m in df.mol])
logd = pd.read_csv(os.path.join(c.ROOT, "results", "logd_pred.csv")).logD_pred.values
assert len(logd) == len(df)
is_te = np.zeros(len(df), int); is_te[te] = 1
rows = [[*[round(float(v), 2) for v in x], round(float(l), 2), round(float(y), 2), int(t)] for x, l, y, t in zip(X, logd, df.y, is_te)]
smiles = [s for s, t in zip(df.smiles, is_te) if not t and len(s) <= 55][:1500]

# a few molecule graphs for the message-passing animation
graphs = []
for s, m in zip(df.smiles, df.mol):
    n = m.GetNumHeavyAtoms()
    if not (13 <= n <= 20) or len(graphs) >= 6:
        continue
    mm = Chem.Mol(m); AllChem.Compute2DCoords(mm)
    pos = mm.GetConformer().GetPositions()[:, :2]
    lo, hi = pos.min(0), pos.max(0); span = max(hi - lo)
    xy = (pos - lo) / span
    graphs.append({"smi": s, "atoms": [[a.GetSymbol(), round(float(xy[a.GetIdx()][0]), 3), round(float(1 - xy[a.GetIdx()][1]), 3)] for a in mm.GetAtoms()],
                   "bonds": [[b.GetBeginAtomIdx(), b.GetEndAtomIdx(), int(b.GetBondTypeAsDouble() if b.GetBondTypeAsDouble() in (1.0, 2.0, 3.0) else 1)] for b in mm.GetBonds()],
                   "aspect": [round(float((hi - lo)[0] / span), 3), round(float((hi - lo)[1] / span), 3)]})

# real structures for the overview strip: 150 unseen compounds, kekulised, ~1 unit per bond
try:
    from rdkit.Chem import rdCoordGen
except Exception:
    rdCoordGen = None
rs = np.random.RandomState(4)
cmp_list = []
for i in rs.permutation(te):
    m = df.mol[i]
    if m.GetNumHeavyAtoms() > 36 or len(cmp_list) >= 150:
        continue
    mk = Chem.Mol(m)
    Chem.Kekulize(mk, clearAromaticFlags=True)
    if rdCoordGen is not None:
        rdCoordGen.AddCoords(mk)
    else:
        AllChem.Compute2DCoords(mk)
    pos = mk.GetConformer().GetPositions()[:, :2]
    bl = np.mean([np.linalg.norm(pos[b.GetBeginAtomIdx()] - pos[b.GetEndAtomIdx()]) for b in mk.GetBonds()]) or 1.0
    pos = pos / bl
    cmp_list.append({"i": int(i), "smi": df.smiles[i],
                     "atoms": [[a.GetSymbol(), round(float(pos[a.GetIdx()][0]), 2), round(float(-pos[a.GetIdx()][1]), 2)] for a in mk.GetAtoms()],
                     "bonds": [[b.GetBeginAtomIdx(), b.GetEndAtomIdx(), int(round(b.GetBondTypeAsDouble())) if b.GetBondTypeAsDouble() in (1.0, 2.0, 3.0) else 1] for b in mk.GetBonds()]})

meta = {}
rp = c.RESULTS
if os.path.exists(rp):
    res = json.load(open(rp))
    for k, v in res.items():
        meta[k.split("|")[2]] = v

out = {"cols": list(fns) + ["logD_pred", "y", "test"], "rows": rows, "smiles": smiles, "graphs": graphs, "cmp": cmp_list, "meta": meta}
json.dump(out, open(os.path.join(c.ROOT, "results", "applet_data.json"), "w"), separators=(",", ":"))
print("wrote applet_data.json", len(rows), "rows,", len(graphs), "graphs,", len(cmp_list), "structures, meta keys:", list(meta))
