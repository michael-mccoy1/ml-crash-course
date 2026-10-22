# %% [markdown]
# # 02: Neural Networks (25 min)
# **Idea:** stack layers of `output = activation(W times input + b)`; learn all `W`,`b` by gradient descent on the prediction error.
#
# Three flavours we touch:
# 1. **MLP** on a fixed feature table (fingerprints/descriptors) - the "plain" NN.
# 2. **Graph neural network (GNN)** - learns features by passing messages along bonds. (Chemprop is the standard tool.)
# 3. **Transformer/GPT** - next block.
#
# Key lesson: NNs need *more care* than trees (scaling, regularisation, early stopping) and don't automatically win on small tabular data.

# %%
import os, sys, subprocess
if "google.colab" in sys.modules:
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "rdkit", "torch_geometric"], check=True)
    os.chdir("ml-crash-course/notebooks")
sys.path.insert(0, "..")
import numpy as np, matplotlib.pyplot as plt, torch, torch.nn as nn
import common as c
torch.manual_seed(0); np.random.seed(0)

df = c.load("HLM"); fp, desc, names = c.get_features(df)
tr, te = c.split(df, "scaffold"); y = df.y.values.astype(np.float32)

# validation set carved from train (for early stopping - never touch test!)
rng = np.random.RandomState(0); perm = rng.permutation(tr)
va, tr_in = perm[:len(perm)//8], perm[len(perm)//8:]

# Input: log1p(ECFP counts) + standardised descriptors (robustly clipped)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(desc[tr_in])
Xd = np.clip(sc.transform(desc), -5, 5).astype(np.float32)
X = np.hstack([np.log1p(fp), Xd]).astype(np.float32)
print("input shape:", X.shape)

# %% [markdown]
# ## 1. MLP from scratch (PyTorch)
# Watch the **train vs validation loss**: the gap is overfitting.

# %%
def make_mlp(d_in, hidden=256, dropout=0.3):
    return nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout),
                         nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                         nn.Linear(hidden, 1))

def fit_mlp(X, y, tr_idx, va_idx, epochs=60, lr=1e-3, wd=1e-4, dropout=0.3, hidden=256, patience=10):
    model = make_mlp(X.shape[1], hidden, dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    Xt, yt = torch.tensor(X), torch.tensor(y)
    hist, best, best_state, bad = [], 1e9, None, 0
    for ep in range(epochs):
        model.train(); p = tr_idx[np.random.permutation(len(tr_idx))]
        for i in range(0, len(p), 64):
            b = p[i:i+64]; opt.zero_grad()
            loss = nn.functional.mse_loss(model(Xt[b]).squeeze(1), yt[b]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            l_tr = nn.functional.mse_loss(model(Xt[tr_idx]).squeeze(1), yt[tr_idx]).item()
            l_va = nn.functional.mse_loss(model(Xt[va_idx]).squeeze(1), yt[va_idx]).item()
        hist.append((l_tr, l_va))
        if l_va < best: best, bad, best_state = l_va, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state)       # early stopping: roll back to best epoch
    return model, np.array(hist)

def predict(model, X, idx):
    model.eval()
    with torch.no_grad(): return model(torch.tensor(X[idx])).squeeze(1).numpy()

model, hist = fit_mlp(X, y, tr_in, va)
plt.plot(hist[:, 0], label="train MSE"); plt.plot(hist[:, 1], label="val MSE")
plt.xlabel("epoch"); plt.legend(); plt.show()
c.log_result("MLP (desc+ECFP)", c.metrics(y[te], predict(model, X, te)))

# %% [markdown]
# **Your turn #1:** rerun `fit_mlp` with `dropout=0.0, wd=0` and then `hidden=1024`. Does the train/val gap grow? Does test R² improve?

# %%
m2, h2 = fit_mlp(X, y, tr_in, va, dropout=0.0, wd=0.0, hidden=1024, patience=100, epochs=40)
plt.plot(h2[:, 0], label="train"); plt.plot(h2[:, 1], label="val"); plt.legend(); plt.title("no regularisation"); plt.show()
print({k: round(v, 3) for k, v in c.metrics(y[te], predict(m2, X, te)).items()})

# %% [markdown]
# ## 2. A graph neural network (stretch / demo)
# Instead of a fixed fingerprint, atoms exchange information with bonded neighbours for *k* rounds, then a pooled vector predicts `y`.
# (Short on time? Run it and read the result; no need to tweak.)

# %%
try:
    from torch_geometric.utils import from_smiles
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool
    HAVE_PYG = True
except Exception as e:
    HAVE_PYG = False; print("torch_geometric not available - skip:", e)

# %%
if HAVE_PYG:
    graphs = []
    for s, t in zip(df.smiles, y):
        g = from_smiles(s); g.x = g.x.float(); g.edge_attr = g.edge_attr.float(); g.y = torch.tensor([t]); graphs.append(g)

    class GNN(nn.Module):
        def __init__(s, d_node=9, d_edge=3, h=128, layers=3):
            super().__init__()
            s.emb = nn.Linear(d_node, h); s.eemb = nn.Linear(d_edge, h)
            s.convs = nn.ModuleList([GINEConv(nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h)), edge_dim=h) for _ in range(layers)])
            s.head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.2), nn.Linear(h, 1))
        def forward(s, d):
            h = s.emb(d.x); e = s.eemb(d.edge_attr)
            for conv in s.convs: h = torch.relu(conv(h, d.edge_index, e)) + h     # message passing + residual
            return s.head(global_mean_pool(h, d.batch)).squeeze(1)

    # normalise atom features crudely (they are small integer codes)
    allx = torch.cat([g.x for g in graphs]); mu, sd = allx.mean(0), allx.std(0) + 1e-6
    ae = torch.cat([g.edge_attr for g in graphs]); emu, esd = ae.mean(0), ae.std(0) + 1e-6
    for g in graphs: g.x = (g.x - mu) / sd; g.edge_attr = (g.edge_attr - emu) / esd

    loaders = {k: DataLoader([graphs[i] for i in idx], batch_size=64, shuffle=(k == "tr"))
               for k, idx in [("tr", tr_in), ("va", va), ("te", te)]}
    gnn = GNN(); opt = torch.optim.AdamW(gnn.parameters(), lr=1e-3, weight_decay=1e-4)
    def run(loader):
        gnn.eval(); P, T = [], []
        with torch.no_grad():
            for b in loader: P.append(gnn(b)); T.append(b.y)
        return torch.cat(P).numpy(), torch.cat(T).numpy()
    best, best_state, bad = 1e9, None, 0
    for ep in range(40):
        gnn.train()
        for b in loaders["tr"]:
            opt.zero_grad(); nn.functional.mse_loss(gnn(b), b.y).backward(); opt.step()
        p, t = run(loaders["va"]); l = float(((p - t) ** 2).mean())
        if l < best: best, bad, best_state = l, 0, {k: v.clone() for k, v in gnn.state_dict().items()}
        else:
            bad += 1
            if bad >= 8: break
    gnn.load_state_dict(best_state); p, t = run(loaders["te"])
    c.log_result("GNN (GINE, from SMILES)", c.metrics(t, p))
