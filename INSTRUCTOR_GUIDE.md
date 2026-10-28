# Instructor guide

## Before the session

* Open the site and hard refresh (Ctrl+F5). Open each lesson tab once so that its data and scripts load.
* Nothing needs installing. Every model trains in the browser.

## Structure of the site

* Each lesson opens with its objectives and a contents table. The contents table lists every step by section and links to it.
* Each lesson has four sections: Beginner, Intermediate, Advanced and Pro. Each section opens and closes. Only the first is open at the start.
* Beginner assumes chemistry and statistics but no machine learning. Each later section assumes only the Beginner sections of the earlier lessons.
* The numbered markers at the right edge of the page jump to any step and open its section.
* The options panel at the left sets the endpoint, the features and the training-set filters for the Intermediate steps and above. It is collapsed in lessons that start with a Beginner section.
* The tree, network and GPT lessons have a workbench: set options, then select Train. Slow replay shows every step. Fast skips ahead.
* Each step has a Notice line and a pop quiz. Pose the question before opening the answer.

## Lessons

| Lesson | Beginner | Later sections |
|---|---|---|
| 01 Data and ML basics | Molecular representations, dataset structure, fitting a model, training and test sets, training and test performance, evaluation metrics | Descriptor contributions, target distribution, model complexity, censored measurements, stacking a logD model, predicting a new compound |
| 02 Tree models | Decision tree structure, split selection, random forests, gradient boosting | Workbench, single split, prediction path, forest size and depth, boosting rounds, feature importance, training set filters, seeds and ensembles |
| 03 Neural networks | Linear neuron, hidden layer, nonlinearity, network capacity | Weight and bias, neuron activation, workbench, network size, learning rate, regularisation, partial dependence, message passing |
| 04 GPT | Tokenisation, token embeddings, next-token prediction, context length | Vocabulary, workbench, sampling and temperature, generation and validity, transformer architecture, attention weights, design test |
| 05 Pre-trained models | Labelled and unlabelled data, masked prediction, effect of pre-training, transfer strategies | Learning curve from scratch, self-supervised task, functional group prediction, GROVER, frozen embeddings, fine-tuning the GPT and Chemprop |
| 06 Model comparison | Metric behaviour, assay repeatability ceiling, accuracy and training time | Leaderboard, accuracy against training time, reading a result |
| 07 Role of AI | AI, machine learning and generative AI, prediction and generation, applicability domain, assessing a prediction | Extrapolation, assay noise, generative model failures, capabilities and limits, checks before use |
| 08 Tree algorithms | Split criterion, histogram binning, bootstrap sampling | Exact and binned search, depth and leaf size, bagging and variance, fitting a 3D shape, data requirements, boosting losses, hyperparameter tuning, boosting libraries |
| 09 Network algorithms | Gradient descent in one dimension, finite-difference gradient, parameter count | Optimisers, backpropagation, parameter count of a GPT, network size and data, weight initialisation, weight decay, scaled dot-product attention, cross-entropy and perplexity |
| 10 Quiz | Ten random questions from a bank of 75, including challenges that train real models |  |

## Suggested activities

* **01:** compare two molecules in Molecular representations and read the Tanimoto similarity. Switch between random and scaffold splits in Training and test sets. Ask why a random split gives an optimistic estimate.
* **02:** set depth 2 against 12 in the workbench and select Train. Ask why the trees in a forest differ.
* **03:** fit the linear neuron, then train the workbench with 120 epochs. Ask why dropout is used when it raises training error.
* **04:** compare the output before and after Train. Ask why generated molecules are checked with RDKit.
* **05:** compare Head only with Everything in the fine-tuning workbench. Read the Chemprop curves at 50 and 1,000 compounds.
* **06:** read the leaderboard. Ask which result is least trustworthy and why.
* **07:** drag the training range in Extrapolation until the score outside it falls to zero. Raise the assay noise and read the ceiling.
* **08:** compare exact and binned search. Run the tuning grid and compare the cell chosen on validation with the best cell on test.
* **09:** run the three optimisers on the raw and the centred feature. Step through backpropagation. Compare the network sizes at 100, 1,000 and 10,000 points.
* **10:** run a round as a group, or individually. The challenge goals count leaves, the units of tree-model size, so a smaller model fits faster on any computer. On these eight descriptors the test R2 is limited to about 0.3 on HLM and 0.36 on logD, and the goals are set near those limits.

## Points to watch

* R2 of 0 means no better than predicting the training mean. It does not indicate a broken model.
* A high training score is not a good result without a matching test score.
* The demonstration GPT and the attention and parameter steps use small sizes so that the numbers can be read. Real models are far larger.
* The boosting loss step is simplified. Real libraries use second-order steps.
* A test score below the assay noise ceiling is not a failure. It may be as good as the assay allows.

## If something goes wrong

* Blank page: open the site over http, not from a file.
* GPT steps blank: an ad blocker or policy may block a script. Open the browser console (F12) and read the messages.
* RDKit steps fail: RDKit is about 7 MB. Wait for it to load.
* Slow laptop: switch animations off and lower the GPT training steps.
