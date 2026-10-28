# %% [markdown]
# # 00: Setup and data
# **Course anchor paper:** Fang et al., *Prospective Validation of Machine Learning Algorithms for ADME Prediction: An Industrial Perspective*, J. Chem. Inf. Model. 2023. Biogen measured ADME endpoints for ~3,500 compounds and released them publicly.
#
# **Question:** *given a molecule's structure, can a model predict how fast human liver microsomes (HLM) clear it, and does the model type matter?*
#
# | Notebook | Model family |
# |---|---|
# | 00 | Data, features, splits |
# | 01 | Trees (decision tree, random forest, gradient boosting) |
# | 02 | Neural networks (MLP, then a graph neural network) |
# | 03 | GPT (decoder-only transformer) |
# | 04 | Comparison and discussion |

# %%
import os, sys, subprocess
REPO_URL = "https://github.com/michael-mccoy1/ml-crash-course.git"
if "google.colab" in sys.modules:  # running in Colab
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "rdkit", "lightgbm"], check=True)
    if not os.path.exists("ml-crash-course"):
        subprocess.run(["git", "clone", REPO_URL], check=True)
    os.chdir("ml-crash-course/notebooks")
sys.path.insert(0, "..")

import numpy as np, pandas as pd, matplotlib.pyplot as plt
import common as c
from rdkit.Chem import Draw

# %% [markdown]
# ## The data
# Each row is a compound with a measured **log10 HLM intrinsic clearance** (mL/min/kg). Higher = metabolised faster = shorter half-life.

# %%
df = c.load("HLM")
print(len(df), "compounds with HLM measurements")
df[["smiles", "y"]].head()

# %%
plt.hist(df.y, bins=40); plt.xlabel("log10 HLM CLint"); plt.ylabel("count"); plt.show()
Draw.MolsToGridImage(list(df.mol[:6]), legends=[f"y={v:.2f}" for v in df.y[:6]], molsPerRow=3)

# %% [markdown]
# ## Molecular representations
# A model needs a numeric table `X`. Two common choices:
# * **ECFP counts** - 2048 columns, "how many times does each circular substructure occur?"
# * **RDKit descriptors** - ~200 computed physicochemical descriptors (logP, TPSA, MW, ...)
# 
# (Neural networks can also learn features directly from the molecular graph or the SMILES string, in notebooks 02 and 03.)

# %%
fp, desc, names = c.get_features(df)   # cached after first call
print("ECFP:", fp.shape, " descriptors:", desc.shape)

# %% [markdown]
# ## Splits matter as much as models
# * **Random split** - test compounds have close analogs in training. Optimistic.
# * **Scaffold split** - whole chemical series are held out. Closer to prospective prediction on a new series, which the paper argues is what matters.
#
# We use the **scaffold split** as the main benchmark throughout.

# %%
tr, te = c.split(df, "scaffold")
print(f"scaffold split: {len(tr)} train / {len(te)} test")
print("test y mean/std:", df.y[te].mean().round(2), df.y[te].std().round(2),
      "| train:", df.y[tr].mean().round(2), df.y[tr].std().round(2))

# %% [markdown]
# ## Baseline: predicting the training mean
# Predicting the training mean gives an R2 of about 0. A model has to exceed this.

# %%
base = c.metrics(df.y[te], np.full(len(te), df.y[tr].mean()))
base["Spearman"] = 0.0
c.log_result("Baseline (train mean)", base)
