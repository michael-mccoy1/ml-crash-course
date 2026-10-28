# %% [markdown]
# # 08: Fine-tuning Chemprop
# Notebook 06 trained Chemprop from scratch on HLM. Here Chemprop is **pre-trained on logD** (4,200 AstraZeneca compounds) and then **fine-tuned on a small number of HLM compounds**. This is the same idea as the Pre-trained models lesson on the website, with a real graph network.
#
# Three ways to start the HLM model are compared on the same scaffold split:
#
# * **From scratch**: random weights, trained only on the HLM compounds.
# * **Frozen encoder**: start from the logD model, keep the message-passing layers fixed, and train only the small network on top (`--freeze-encoder`).
# * **Full fine-tuning**: start from the logD model and let every weight move (`--checkpoint` alone).
#
# The notebook repeats each setting for several numbers of labelled HLM compounds, because pre-training is meant to help when labels are scarce. Many small runs are needed, so use a GPU runtime in Colab if you have one.

# %%
import os, sys, subprocess, glob, json, time
if "google.colab" in sys.modules:
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "rdkit", "chemprop"], check=True)
    os.chdir("ml-crash-course/notebooks")
sys.path.insert(0, ".."); sys.path.insert(0, ".")
import numpy as np, pandas as pd
import common as c

OUT = os.path.join(c.ROOT, "results", "chemprop_ft"); os.makedirs(OUT, exist_ok=True)

def chemprop(*args):
    """Run the Chemprop CLI in a subprocess (works in Colab and locally)."""
    code = "import sys; from chemprop.cli.main import main; sys.argv = ['chemprop'] + sys.argv[1:]; main()"
    subprocess.run([sys.executable, "-c", code, *args], check=True, stdout=subprocess.DEVNULL)

def find_ckpt(d):
    found = sorted(glob.glob(os.path.join(d, "**", "best.pt"), recursive=True)) or sorted(glob.glob(os.path.join(d, "**", "*.ckpt"), recursive=True))
    return found[-1]

# %% [markdown]
# ## 1. Pre-train on logD

# %%
az = c.load_logd(); tr_a, te_a = c.split(az, "scaffold")
rs = np.random.RandomState(0); perm = rs.permutation(tr_a); va_a, tr_in_a = perm[:len(perm) // 8], perm[len(perm) // 8:]
for name, idx in [("train", tr_in_a), ("val", va_a), ("test", te_a)]:
    az.iloc[idx][["smiles", "y"]].to_csv(os.path.join(OUT, "az_" + name + ".csv"), index=False)
pre_dir = os.path.join(OUT, "pretrain")
t0 = time.time()
chemprop("train", "--data-path", *[os.path.join(OUT, f"az_{n}.csv") for n in ("train", "val", "test")], "--task-type", "regression",
         "--smiles-columns", "smiles", "--target-columns", "y", "--epochs", "30", "--save-dir", pre_dir, "--num-workers", "0", "--accelerator", "auto")
pre_secs = time.time() - t0
pre_ckpt = find_ckpt(pre_dir)
chemprop("predict", "--test-path", os.path.join(OUT, "az_test.csv"), "--smiles-columns", "smiles", "--model-paths", pre_ckpt, "--preds-path", os.path.join(OUT, "az_pred.csv"), "--num-workers", "0")
pre_r2 = c.metrics(az.y.values[te_a], pd.read_csv(os.path.join(OUT, "az_pred.csv")).iloc[:, -1].values)["R2"]
print(f"logD pre-training: {pre_secs:.0f} s, test R2 {pre_r2:.3f}")

# %% [markdown]
# ## 2. Fine-tune on HLM with few labels
# A fixed set of 100 HLM compounds is held aside for choosing the best epoch. Training sets of 50 to 1,000 compounds are drawn from the rest. Each setting is repeated with two different random draws.

# %%
hlm = c.load("HLM"); tr_h, te_h = c.split(hlm, "scaffold")
perm_h = np.random.RandomState(1).permutation(tr_h); val_h, pool = perm_h[:100], perm_h[100:]
hlm.iloc[val_h][["smiles", "y"]].to_csv(os.path.join(OUT, "hlm_val.csv"), index=False)
hlm.iloc[te_h][["smiles", "y"]].to_csv(os.path.join(OUT, "hlm_test.csv"), index=False)

SIZES = [50, 100, 200, 500, 1000]; SEEDS = [0, 1]; MODES = {"scratch": [], "frozen": ["--checkpoint", pre_ckpt, "--freeze-encoder"], "full": ["--checkpoint", pre_ckpt]}
results = []
for n in SIZES:
    for seed in SEEDS:
        draw = np.random.RandomState(100 + seed).permutation(pool)[:n]
        tp = os.path.join(OUT, f"hlm_train_{n}_{seed}.csv"); hlm.iloc[draw][["smiles", "y"]].to_csv(tp, index=False)
        for mode, extra in MODES.items():
            d = os.path.join(OUT, f"run_{mode}_{n}_{seed}"); t0 = time.time()
            chemprop("train", "--data-path", tp, os.path.join(OUT, "hlm_val.csv"), os.path.join(OUT, "hlm_test.csv"), "--task-type", "regression", "--smiles-columns", "smiles", "--target-columns", "y",
                     "--epochs", "30", "--save-dir", d, "--num-workers", "0", "--accelerator", "auto", "--data-seed", str(seed), *extra)
            secs = time.time() - t0; ck = find_ckpt(d); pp = os.path.join(d, "pred.csv")
            chemprop("predict", "--test-path", os.path.join(OUT, "hlm_test.csv"), "--smiles-columns", "smiles", "--model-paths", ck, "--preds-path", pp, "--num-workers", "0")
            m = c.metrics(hlm.y.values[te_h], pd.read_csv(pp).iloc[:, -1].values)
            results.append({"n": n, "seed": seed, "mode": mode, "r2": round(float(m["R2"]), 4), "secs": round(secs, 1)})
            print(n, seed, mode, results[-1]["r2"], f"{secs:.0f} s", flush=True)
            json.dump({"pretrain": {"endpoint": "logD", "r2": round(float(pre_r2), 3), "secs": round(pre_secs, 1), "epochs": 30}, "runs": results}, open(os.path.join(c.ROOT, "results", "chemprop_finetune.json"), "w"))

# %% [markdown]
# ## 3. Read the result

# %%
tab = pd.DataFrame(results).groupby(["n", "mode"]).r2.mean().unstack()[["scratch", "frozen", "full"]].round(3)
print(tab)

# %% [markdown]
# **Discuss:** Where does pre-training help most, with 50 compounds or with 1,000? Is freezing the encoder safer than updating everything when labels are scarce? logD and HLM are different endpoints, so the pre-trained encoder had to learn something that transfers. What if the pre-training task had been unrelated to your endpoint?
