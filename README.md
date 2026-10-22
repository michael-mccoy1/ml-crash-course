# Model Lab: AI/ML for ADME

A 90-minute course on tree models, neural networks and GPTs, built on two public ADME endpoints:

* **HLM clearance**: Biogen ADME set (Fang et al., J. Chem. Inf. Model. 2023), 3,087 compounds. 31% of the values sit at the assay floor.
* **logD**: AstraZeneca set from ChEMBL via MoleculeNet, 4,200 compounds. A logD model can feed its prediction into the HLM model.

Structures, descriptors, scaffolds and similarity statistics come from RDKit. RDKit also runs in the browser (loaded on demand) to draw and validate molecules.

## Website

**https://michael-mccoy1.github.io/ml-crash-course/**

Nothing to install. Each lesson has a fixed options panel on the left and a scrolling sequence of steps on the right. Each step has a short instruction (what to change, what to watch), an animated stage that follows the options, and a key point. Every lesson ends with its own glossary.

| Lesson | Minutes | Content |
|---|---|---|
| 01 Data and ML basics | 12 | Structure to prediction, descriptor maps, the unseen set, over and underfitting, scoring, logD feeding HLM, predict your own SMILES |
| 02 Data quality | 12 | Censored values, outliers, property ranges, the test-set trap, a combined comparison |
| 03 Tree models | 20 | Split search, depth, one compound down the tree, forest, boosting, feature importance |
| 04 Neural networks | 20 | One neuron, layers, learning rate, live training, dropout, one compound through the network, one neuron opened, what it learned, Chemprop |
| 05 GPT | 18 | Tokens, embeddings, the transformer stack, training, attention, next-token odds, generation checked by RDKit |
| 06 Compare | 8 | Leaderboard, accuracy against training time, discussion questions |

The endpoint (HLM or logD) and the feature toggles are shared by all lessons. The GPT always reads the Biogen SMILES.

## Notebooks (Colab or local)

| Notebook | Content |
|---|---|
| `00_setup_and_data` | Data, features, scaffold split, baseline |
| `01_tree_models` | Tree, forest, LightGBM, importances |
| `02_neural_networks` | MLP and a graph network |
| `03_gpt` | Tiny GPT from scratch, generation, embeddings |
| `04_compare_and_discuss` | Results table and variance over splits |
| `05_logd_to_hlm` | logD model trained on AstraZeneca data, predicted logD as an HLM feature |
| `06_chemprop` | Chemprop on the same scaffold split |
| `07_data_quality` | Three ways to treat censored HLM values |

Reference results on the scaffold split with fingerprints and descriptors: LightGBM R2 0.51, Chemprop 0.46, MLP 0.43, random forest 0.40, graph network 0.29, GPT embeddings with ridge 0.16. Predicted logD adds about +0.06 R2 to eight descriptors and +0.005 to full fingerprints. Dropping censored values from training raises R2 on measurable compounds from 0.29 to 0.34 and lowers it on all unseen compounds.

Local run: `pip install rdkit scikit-learn lightgbm torch torch_geometric chemprop matplotlib pandas scipy`.

## Publishing

The site is the `docs` folder. A GitHub Actions workflow (`.github/workflows/pages.yml`) deploys it on every push to `main`. In the repository settings, set Pages source to GitHub Actions.

## Rebuilding

```
python build_data.py        # docs/data/*.json from RDKit
python build_notebooks.py   # notebooks_src/*.py to notebooks/*.ipynb
```

To preview the site, run `python -m http.server` inside `docs` and open http://localhost:8000.

## Layout

```
docs/index.html, css/site.css, js/*.js    the website
docs/data/                                hlm.json, logd.json, structs.json (RDKit drawings), extras.json
docs/assets/                              Takeda curve and pattern images
build_data.py, build_notebooks.py         builders
common.py                                 shared data, feature, split and metric helpers
data/                                     biogen_adme.csv, lipophilicity_az.csv
notebooks_src/, notebooks/                notebook sources and .ipynb files
```
