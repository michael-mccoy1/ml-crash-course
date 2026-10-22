# %% [markdown]
# # 06: Chemprop (graph neural network)
# [Chemprop](https://github.com/chemprop/chemprop) is the standard message-passing network for molecules. Instead of fingerprints it reads the **molecular graph**: atoms are nodes, bonds are edges. For a few rounds each bond collects messages from neighbouring bonds, then all atoms are pooled into one vector and a small network predicts the endpoint.
#
# Same scaffold split as every other notebook, so the score is directly comparable. CPU training takes a few minutes. Use a GPU runtime in Colab for speed.

# %%
import os, sys, subprocess
if "google.colab" in sys.modules:
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "rdkit", "chemprop"], check=True)
    os.chdir("ml-crash-course/notebooks")
sys.path.insert(0, "..")
import numpy as np, pandas as pd
import common as c

df = c.load("HLM"); tr, te = c.split(df, "scaffold")
rng = np.random.RandomState(0); perm = rng.permutation(tr); va, tr_in = perm[:len(perm) // 8], perm[len(perm) // 8:]
out = os.path.join(c.ROOT, "results", "chemprop"); os.makedirs(out, exist_ok=True)
for name, idx in [("train", tr_in), ("val", va), ("test", te)]:
    df.iloc[idx][["smiles", "y"]].to_csv(os.path.join(out, name + ".csv"), index=False)
print(len(tr_in), len(va), len(te))

# %% [markdown]
# ## Train
# `--epochs 30` is enough to see the effect. Published settings usually use more epochs and an ensemble.

# %%
import time

def chemprop(*args):
    """Run the Chemprop CLI in a subprocess (works in Colab and locally)."""
    code = "import sys; from chemprop.cli.main import main; sys.argv = ['chemprop'] + sys.argv[1:]; main()"
    subprocess.run([sys.executable, "-c", code, *args], check=True)

t0 = time.time()
cmd = ["train",
       "--data-path", os.path.join(out, "train.csv"), os.path.join(out, "val.csv"), os.path.join(out, "test.csv"),
       "--task-type", "regression", "--smiles-columns", "smiles", "--target-columns", "y",
       "--epochs", "30", "--save-dir", os.path.join(out, "model"), "--num-workers", "0", "--accelerator", "auto"]
chemprop(*cmd)
train_secs = time.time() - t0
print(f"trained in {train_secs:.0f} s")

# %% [markdown]
# ## Predict the unseen scaffolds and score them

# %%
import glob
found = sorted(glob.glob(os.path.join(out, "model", "**", "best.pt"), recursive=True)) or sorted(glob.glob(os.path.join(out, "model", "**", "*.ckpt"), recursive=True))
ckpt = found[-1]; print("using", ckpt)
pred_path = os.path.join(out, "test_pred.csv")
chemprop("predict", "--test-path", os.path.join(out, "test.csv"), "--smiles-columns", "smiles",
         "--model-paths", ckpt, "--preds-path", pred_path, "--num-workers", "0")
pred = pd.read_csv(pred_path).iloc[:, -1].values
m = c.metrics(df.y.values[te], pred); m["train_secs"] = train_secs
c.log_result("Chemprop (D-MPNN)", m)

# %% [markdown]
# **Discuss:** Chemprop learns its own features from the graph. Compare it with LightGBM on fingerprints and descriptors (notebook 01). On a few thousand compounds, which wins? What changes with 100,000 compounds, or with a pretrained model?
