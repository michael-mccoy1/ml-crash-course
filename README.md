# Model Lab: AI/ML for ADME

A course on tree models, neural networks and GPTs for ADME modelling, written for scientists who are new to machine learning. Two public endpoints:

* **HLM clearance**: Biogen set (Fang et al., J. Chem. Inf. Model. 2023), 3,087 compounds. 31% sit at the assay floor.
* **logD**: AstraZeneca set from ChEMBL via MoleculeNet, 4,200 compounds.

RDKit builds the data and also runs in the browser.

## Website

**https://michael-mccoy1.github.io/ml-crash-course/**

Nothing to install. Each lesson has objectives, a contents table, an options panel, steps with a Notice line and a pop quiz, and a glossary. The tree, network and GPT lessons have a workbench: change an option and the figure updates, select Train and the fit replays. Every lesson has training-set filters in its options.

Each lesson has four sections that open and close: **Beginner** (core machine learning concepts for scientists new to the field; chemistry is assumed), **Intermediate**, **Advanced** and **Pro**. Only the first section is open at the start. Numbered markers at the right edge of the page jump to any step and open its section. A section assumes only the Beginner sections of the earlier lessons. Step titles are set in `docs/js/titles.js`, and the writing conventions are in `STYLE.md`.

| Lesson | Content |
|---|---|
| 01 Data and ML basics | Molecular representations, dataset structure, fitting a model, training and test sets, training and test performance, evaluation metrics, descriptors, target distribution, model complexity, censored measurements, stacking a logD model, prediction of a new compound |
| 02 Tree models | Decision tree structure, split selection, random forests, gradient boosting, workbench, feature importance, filters, seeds and ensembles |
| 03 Neural networks | Linear neuron, hidden layer, nonlinearity, network capacity, workbench, learning rate, regularisation, partial dependence, message passing, comparison with trees, seeds and ensembles |
| 04 GPT | Tokenisation, token embeddings, next-token prediction, context length, workbench, sampling and temperature, validity, transformer architecture, attention weights, design test |
| 05 Pre-trained models | Labelled and unlabelled data, masked prediction, effect of pre-training, transfer strategies, GROVER, frozen embeddings, fine-tuning a GPT and Chemprop |
| 06 Model comparison | Metric behaviour, assay repeatability ceiling, accuracy and training time, leaderboard, reading a result |
| 07 Role of AI | AI, machine learning and generative AI, prediction and generation, applicability domain, assessing a prediction, extrapolation, assay noise, failures of generative models, checks before use |
| 08 Tree algorithms | Split criterion, histogram binning, bootstrap sampling, exact and binned search, depth and leaf size, bagging, boosting losses, tuning, library designs, fitting a 3D double helix, data requirements |
| 09 Network algorithms | Gradient descent, finite-difference gradients, parameter count, optimisers, backpropagation, initialisation, weight decay, attention, cross-entropy, network size against data |
| 10 PK and PBPK | Concentration-time profiles, in vitro to in vivo clearance, where ML enters a PBPK workflow, MIDD decisions and ICH M15, PBPK model structure, NCA and compartmental fits of public Theoph and Indometh data, population variability, parameter sensitivity, absorption and distribution, drug-drug interaction models, parameter estimation, quantitative systems pharmacology, uncertainty propagation, model credibility |
| 11 ML in PK and MIDD | Model families and PK tasks, predicted parameters as PBPK inputs, error propagation to exposure, surrogate models, covariate screening, ensembles for parameter uncertainty, hybrid mechanistic-ML models, sequence models, transfer from in vitro to in vivo, graph networks for PK parameters, LLMs in the workflow, generative design, credibility of ML components, validation design |
| 12 Quiz | 97 questions, 10 drawn at random each round: multiple choice, select all, numeric, ordering, and challenges that train a boosting model, forest or tree against a goal. Each question has a schematic. 22 questions cover PK, PBPK, MIDD and machine learning in PK |

## Notebooks

`notebooks/` covers the same ideas in Python (Colab or local): setup, trees, networks, GPT, comparison, logD to HLM, Chemprop, data quality, and Chemprop fine-tuning.

Reference results (scaffold split, fingerprints and descriptors): LightGBM R2 0.51, Chemprop 0.46, MLP 0.43, random forest 0.40, graph network 0.29. Chemprop pre-trained on logD, then fine-tuned on 1,000 HLM compounds: R2 0.41 for full fine-tuning, 0.28 frozen encoder, 0.24 from scratch.

Install: `pip install rdkit scikit-learn lightgbm torch torch_geometric chemprop matplotlib pandas scipy`

## Publishing

The site is `docs/`. The workflow in `.github/workflows/pages.yml` deploys it on every push to `main`. Set Pages source to GitHub Actions.

## Rebuilding

```
python build_data.py        # docs/data/*.json
python build_notebooks.py   # notebooks_src/*.py to notebooks/*.ipynb
```

Preview: run `python -m http.server` in `docs` and open http://localhost:8000.

## Layout

```
docs/            the website (index.html, css, js, data, vendor, assets)
build_data.py    builds docs/data from RDKit
common.py        shared data, feature, split and metric helpers
data/            input CSVs
notebooks_src/   notebook sources
notebooks/       built notebooks
```
