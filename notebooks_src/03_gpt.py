# %% [markdown]
# # 03: GPT (20 min)
# **GPT = Generative Pre-trained Transformer**: a *decoder-only* transformer trained on one task, **predict the next token** given all previous tokens.
#
# Ingredients you will see in ~60 lines of code:
# 1. **Tokenizer** - text to integer IDs (here: SMILES atoms/bonds/rings).
# 2. **Embeddings** - token ID to vector, plus position information.
# 3. **Causal self-attention** - each token looks at *earlier* tokens only, weighting them by relevance.
# 4. **Next-token loss** - cross-entropy against the sequence shifted by one.
#
# ChatGPT-scale models are this same recipe with trillions of tokens, billions of parameters, plus instruction tuning. Here we train a **tiny chemistry GPT on SMILES** from scratch on CPU, then:
# * **Generate** new molecules (the "G" in GPT)
# * **Reuse** what it learned (embeddings) to predict HLM clearance (transfer learning)

# %%
import os, sys, re, subprocess
if "google.colab" in sys.modules:
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "rdkit"], check=True)
    os.chdir("ml-crash-course/notebooks")
sys.path.insert(0, "..")
import numpy as np, matplotlib.pyplot as plt, torch, torch.nn as nn, torch.nn.functional as F
import common as c
from rdkit import Chem
torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(max(1, os.cpu_count() // 2))

df = c.load("HLM"); tr, te = c.split(df, "scaffold"); y = df.y.values

# %% [markdown]
# ## 1. Tokenize SMILES
# Treat `Cl`, `Br`, `[C@@H]` etc. as single tokens (the usual regex from the Molecular Transformer paper).

# %%
PATTERN = re.compile(r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])")
tokenize = lambda s: PATTERN.findall(s)
print(tokenize("CNc1cc(Cl)ccc1[C@@H](O)C"))

# vocab built from TRAIN only (no peeking at test molecules)
train_smiles = list(df.smiles[tr])
vocab = ["<pad>", "<bos>", "<eos>"] + sorted({t for s in train_smiles for t in tokenize(s)})
stoi = {t: i for i, t in enumerate(vocab)}; itos = dict(enumerate(vocab))
BLOCK = 100                                       # max sequence length
def encode(s):
    ids = [stoi["<bos>"]] + [stoi.get(t, 0) for t in tokenize(s)][:BLOCK - 2] + [stoi["<eos>"]]
    return ids + [0] * (BLOCK - len(ids))
Xtok = torch.tensor([encode(s) for s in df.smiles])
print("vocab size:", len(vocab), "| sequence length:", BLOCK)

# %% [markdown]
# ## 2. The model
# Read `Block.forward`: *attention* mixes information **between** tokens; the *MLP* processes each token **individually**. Residual connections (`x + ...`) keep gradients flowing.

# %%
class CausalSelfAttention(nn.Module):
    def __init__(s, d, heads):
        super().__init__(); s.h = heads; s.qkv = nn.Linear(d, 3 * d); s.proj = nn.Linear(d, d)
    def forward(s, x):
        B, T, D = x.shape
        q, k, v = s.qkv(x).view(B, T, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
        # is_causal=True masks the future: token t may only attend to tokens <= t
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return s.proj(out.transpose(1, 2).reshape(B, T, D))

class Block(nn.Module):
    def __init__(s, d, heads):
        super().__init__(); s.ln1 = nn.LayerNorm(d); s.attn = CausalSelfAttention(d, heads)
        s.ln2 = nn.LayerNorm(d); s.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(s, x):
        x = x + s.attn(s.ln1(x)); return x + s.mlp(s.ln2(x))

class TinyGPT(nn.Module):
    def __init__(s, vocab, d=128, heads=4, layers=3, block=BLOCK):
        super().__init__(); s.tok = nn.Embedding(vocab, d); s.pos = nn.Embedding(block, d)
        s.blocks = nn.Sequential(*[Block(d, heads) for _ in range(layers)]); s.ln = nn.LayerNorm(d)
        s.head = nn.Linear(d, vocab, bias=False)
    def hidden(s, idx):
        return s.ln(s.blocks(s.tok(idx) + s.pos(torch.arange(idx.shape[1]))))
    def forward(s, idx): return s.head(s.hidden(idx))

gpt = TinyGPT(len(vocab))
print(f"{sum(p.numel() for p in gpt.parameters())/1e6:.2f}M parameters   (GPT-3 has 175,000M)")

# %% [markdown]
# ## 3. Pre-train: predict the next token
# Input = tokens `[0..T-1]`, target = tokens `[1..T]`. Trained only on **training-set SMILES** (no `y` labels!). ~3 min on a laptop CPU.

# %%
import time
opt = torch.optim.AdamW(gpt.parameters(), lr=1e-3, weight_decay=0.01)
train_ids = Xtok[tr]; losses = []; t0 = time.time()
EPOCHS = 40
for ep in range(EPOCHS):
    gpt.train(); perm = torch.randperm(len(train_ids)); tot = 0; n = 0
    for i in range(0, len(perm), 64):
        b = train_ids[perm[i:i+64]]; inp, tgt = b[:, :-1], b[:, 1:]
        logits = gpt(inp)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=0)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(gpt.parameters(), 1.0); opt.step()
        tot += loss.item(); n += 1
    losses.append(tot / n); print(f"epoch {ep+1:2d}  loss {losses[-1]:.3f}  ({time.time()-t0:.0f}s)")
plt.plot(losses); plt.xlabel("epoch"); plt.ylabel("next-token loss"); plt.show()

# %% [markdown]
# ## 4. Generate molecules (sampling)
# Start from `<bos>`, sample a token, append, repeat. **Temperature** controls creativity: low = conservative, high = wild.

# %%
@torch.no_grad()
def generate(n=200, temperature=1.0, max_len=BLOCK - 1):
    gpt.eval(); idx = torch.full((n, 1), stoi["<bos>"])
    for _ in range(max_len - 1):
        logits = gpt(idx)[:, -1, :] / temperature
        logits[:, 0] = -1e9; logits[:, stoi["<bos>"]] = -1e9
        nxt = torch.multinomial(F.softmax(logits, -1), 1); idx = torch.cat([idx, nxt], 1)
        if (idx == stoi["<eos>"]).any(1).all(): break
    out = []
    for row in idx.tolist():
        toks = []
        for t in row[1:]:
            if t == stoi["<eos>"]: break
            toks.append(itos[t])
        out.append("".join(toks))
    return out

train_set = set(train_smiles)
for T in [0.7, 1.0, 1.3]:
    gen = generate(200, T)
    valid = [s for s in gen if s and Chem.MolFromSmiles(s) is not None]
    uniq = set(valid); novel = uniq - train_set
    print(f"T={T}: valid {len(valid)/len(gen):.0%} | unique {len(uniq)/max(1,len(valid)):.0%} | novel {len(novel)/max(1,len(uniq)):.0%}")

# %%
from rdkit.Chem import Draw
good = [s for s in generate(100, 0.8) if s and Chem.MolFromSmiles(s) is not None][:9]
Draw.MolsToGridImage([Chem.MolFromSmiles(s) for s in good], molsPerRow=3, subImgSize=(250, 200))

# %% [markdown]
# **Your turn:** the model never saw a rule for valence or ring closure. Where does its "chemistry" come from? What happens to validity if you train for 2 epochs instead of 40?

# %% [markdown]
# ## 5. Reuse the GPT for prediction
# The GPT's hidden states are a learned representation of each molecule. Mean-pool them to feature vector to fit a simple **ridge regression** to `y`. This is the same pattern as using LLM embeddings for downstream tasks.

# %%
@torch.no_grad()
def embed(ids):
    gpt.eval(); out = []
    for i in range(0, len(ids), 256):
        b = ids[i:i+256]; h = gpt.hidden(b); mask = (b != 0).unsqueeze(-1).float()
        out.append(((h * mask).sum(1) / mask.sum(1)).numpy())
    return np.vstack(out)
E = embed(Xtok)

from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(E[tr])
ridge = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(sc.transform(E[tr]), y[tr])
c.log_result("GPT embeddings + ridge", c.metrics(y[te], ridge.predict(sc.transform(E[te]))))

# %% [markdown]
# **Discuss:** this tiny GPT saw only ~2,500 molecules. Published chemistry language models (ChemBERTa, MolFormer, ...) pre-train on 10⁷-10⁹ molecules. What would you expect to change? And how does this differ from *prompting* a general LLM like ChatGPT/Claude with a SMILES string?

# %% [markdown]
# ### Optional: use a real pre-trained chemistry language model
# Needs internet + `pip install transformers`. Uncomment to try ChemBERTa embeddings with the same ridge recipe.

# %%
# from transformers import AutoTokenizer, AutoModel
# tok = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1"); mdl = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1").eval()
# def chemberta(smiles, bs=64):
#     out = []
#     with torch.no_grad():
#         for i in range(0, len(smiles), bs):
#             enc = tok(list(smiles[i:i+bs]), padding=True, truncation=True, max_length=128, return_tensors="pt")
#             h = mdl(**enc).last_hidden_state; m = enc["attention_mask"].unsqueeze(-1)
#             out.append(((h * m).sum(1) / m.sum(1)).numpy())
#     return np.vstack(out)
# E2 = chemberta(df.smiles.tolist())
# sc2 = StandardScaler().fit(E2[tr]); r2 = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(sc2.transform(E2[tr]), y[tr])
# c.log_result("ChemBERTa embeddings + ridge", c.metrics(y[te], r2.predict(sc2.transform(E2[te]))))
