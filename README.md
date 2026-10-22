# ML Crash Course: Trees, Neural Networks and GPT for ADME

Anchored on the public Biogen ADME set (Fang et al., *J. Chem. Inf. Model.* 2023; data from
[molecularinformatics/Computational-ADME](https://github.com/molecularinformatics/Computational-ADME)).
Task: predict log10 human liver microsome clearance (HLM) from structure. A second public set (AstraZeneca logD, MoleculeNet "Lipophilicity") feeds a predicted logD into the HLM models.

## Website (nothing to install)

**https://michael-mccoy1.github.io/ml-crash-course/**

| Tab | Min | Content |
|---|---|---|
| What is ML? | 10 | Overview, underfitting-to-overfitting demo, logD to HLM stacking, glossary |
| Trees | 25 | Live decision tree, forest vote, boosting replay |
| Neural networks | 25 | Live network, one compound through it, one neuron's arithmetic, Chemprop message passing |
| GPT | 20 | Pipeline, per-token confidence, attention, next-token odds |
| Compare | 10 | R² and training time per model |

Features can be toggled on every tab. Training happens in the browser (TensorFlow.js).

## Notebooks (Colab or local)

| Notebook | Content |
|---|---|
| `00_setup_and_data` | Data, features, scaffold split, baseline |
| `01_tree_models` | Tree, forest, LightGBM, importances |
| `02_neural_networks` | MLP, GNN |
| `03_gpt` | Tiny GPT, generation, embeddings |
| `04_compare_and_discuss` | Results table, variance over splits |
| `05_logd_to_hlm` | AstraZeneca logD model, predicted logD as an HLM feature |
| `06_chemprop` | Chemprop D-MPNN on the same split |

Reference scores (scaffold split, HLM R²): LightGBM 0.51, Chemprop 0.46, MLP 0.43, random forest 0.40, GNN 0.29, GPT embeddings 0.16. Adding predicted logD: +0.06 R² with eight descriptors, +0.005 with full fingerprints (five splits).

Local run: `pip install rdkit scikit-learn lightgbm torch torch_geometric chemprop matplotlib pandas scipy`.

## Publishing

Settings → Pages → Deploy from a branch → `main` / `/docs`.

## Rebuilding

```
python build_notebooks.py        # notebooks_src/*.py -> notebooks/*.ipynb
python build_data.py             # needs results/logd_pred.csv (run notebook 05)
python build_applet.py           # applet/site_template.html + data -> docs/index.html
```

## Layout

```
docs/index.html              website (generated)   docs/assets/  Takeda curve, pattern
applet/site_template.html    website source
build_*.py                   builders
notebooks_src/ notebooks/    notebook sources and .ipynb
common.py                    shared helpers
data/                        biogen_adme.csv, lipophilicity_az.csv
results/applet_data.json     data the website loads
```
