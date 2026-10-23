# Instructor Guide

## Before the session

* Open the site and hard refresh (Ctrl+F5). Click each lesson tab once.
* Nothing needs installing.

## How the site works

* Each lesson opens with a card: goals and words.
* The left panel changes every stage. Have people change something before reading.
* The tree, network and GPT lessons have a workbench. Change an option, then press Train. Slow shows every step. Fast skips ahead.
* Each step has a Notice line and a Pop quiz. Ask the question before anyone opens the answer.

## Lessons

| Lesson | Cover |
|---|---|
| 01 Data and ML basics | Fitting a line, features, the unseen set, assay floor, overfitting |
| 02 Tree models | One cut, workbench, forest, boosting, the failure test |
| 03 Neural networks | One weight, one neuron, workbench, learning rate, dropout, the comparison test |
| 04 GPT | Tokens, attention, workbench, the design test |
| 05 Pre-trained models | Fill in the blank, GROVER, fine-tuning a GPT and Chemprop |
| 06 Compare | Leaderboard, discussion |

### 01 Data and ML basics
* **Say:** a model is a rule fitted to examples.
* **Do:** drag the weight and bias, then press Find the best weights. Switch the split in The unseen set.
* **Ask:** why is a random split too optimistic?
* **Watch for:** R2 of 0 means no better than the average, not a broken model.

### 02 Tree models
* **Say:** a tree is yes or no questions. A forest averages trees. Boosting fixes mistakes in turn.
* **Do:** guess depth 2 against 12, then Train. Click in the shaded zones of the last test.
* **Ask:** why do the trees in a forest differ?
* **Watch for:** a high training score is not a good result.

### 03 Neural networks
* **Say:** a weight is how hard one input pushes. Training nudges all of them.
* **Do:** one weight first, then the workbench with Epochs at 120. Run the last test.
* **Ask:** why use dropout when it worsens training error?
* **Watch for:** the lines in the diagram are weights. Hover to read one.

### 04 GPT
* **Say:** a GPT guesses the next token. Here the text is a molecule.
* **Do:** click tokens in Attention. Compare what it writes before and after Train. Run the last test.
* **Ask:** why check its molecules with RDKit?
* **Watch for:** the demo model is tiny.

### 05 Pre-trained models
* **Say:** pre-training learns from molecules with no lab values. Fine-tuning adds your labels.
* **Do:** play the fill-in-the-blank game. Compare Head only with Everything. Read the Chemprop lines at 50 and 1,000.
* **Ask:** when would you freeze the model?
* **Watch for:** pre-training does not always help. Grey is the baseline.

### 06 Compare
* Walk the leaderboard. Ask which result people trust least.

## If something goes wrong

* Blank page: open the site over http, not from a file.
* GPT steps blank: an ad blocker or policy may block a script. Press F12 and read the Console.
* RDKit steps fail: RDKit is about 7 MB. Wait for it to load.
* Slow laptop: switch animations off and lower the GPT training steps.
