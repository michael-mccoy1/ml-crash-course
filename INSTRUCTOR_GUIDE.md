# Instructor Guide (90 minutes, no machine learning background needed)

## Before the session

* Open the site and hard refresh (Ctrl+F5). Click each lesson tab once so the pages are built, then return to lesson 01.
* The GPT lesson loads a small demo model by itself. If the Pre-trained models lesson shows a gate asking for the GPT, open the GPT tab once and wait a few seconds.
* Nothing needs to be installed. Notebooks are optional homework.

## How the site teaches

* Each lesson opens with a card: how long it takes, what participants will be able to do, and the words they will meet. Read the three goals aloud.
* The options panel on the left drives every stage. Tell participants to change something before reading the explanation: the picture answers first.
* The three model lessons have a workbench: drag the options, then press Train and watch a slowed-down fit. Slow shows every step, Fast skips ahead. The reported training time is the real compute time.
* Under every stage there is a line that says what to notice and a question, Check yourself, with a hidden answer. Use the question as the discussion prompt before anyone opens the answer.
* Filter training data is in the options of every lesson, so participants can see how the choice of data helps or hurts each kind of model.

## Run of show

| Minutes | Lesson | Plan |
|---|---|---|
| 0 to 14 | 01 Data and ML basics | Learning as fitting a line, features, the unseen set, assay floor, three ways to treat it, overfitting. |
| 14 to 32 | 02 Tree models | One tree choosing a question, then the workbench with tree, forest and boosting. |
| 32 to 52 | 03 Neural networks | What a weight is, one neuron, the workbench, learning rate, dropout. |
| 52 to 68 | 04 GPT | Tokens, embeddings, the stack, the workbench, attention. |
| 68 to 84 | 05 Pre-trained models | Why pre-train, fill in the blank, GROVER, fine-tuning a GPT and Chemprop. |
| 84 to 90 | 06 Compare | Leaderboard and the discussion questions. |

## Lesson by lesson

### 01 Data and ML basics
* **Say:** a model is a rule that turns numbers about a compound into a predicted lab value, and learning means choosing the rule that misses by the least.
* **Do:** in What learning means, let participants drag the weight and bias before pressing Find the best weights. In The unseen set, switch the split and compare the numbers.
* **Ask:** why is a random split too optimistic? What does the tall bar at the assay floor mean?
* **Common confusion:** R2 of 0 does not mean the model is broken. It means no better than guessing the average.

### 02 Tree models
* **Say:** a tree is a flowchart of yes or no questions. A forest averages many trees. Boosting adds small trees that fix earlier mistakes.
* **Do:** before pressing Train on the decision tree, ask the room what depth 2 versus 12 will do. Then compare the red unseen curve.
* **Ask:** why do the trees in a forest differ? What goes wrong with a very large learning rate in boosting?
* **Common confusion:** a high score on the training compounds is not a good result. Always read the unseen score.

### 03 Neural networks
* **Say:** a weight is a number that says how strongly one input pushes the prediction. A network is many weights, and training nudges all of them to reduce the error.
* **Do:** the first step has one weight: drag it, then press Find the best weights. In the workbench raise Epochs to 120 and watch the validation curve.
* **Ask:** why use dropout when it makes the training error worse? Why does a huge learning rate fail?
* **Common confusion:** the lines in the network diagram are weights, not data. Hover a line to read its value.

### 04 GPT
* **Say:** a GPT reads text as tokens and learns to guess the next one. Here the text is a molecule written as SMILES.
* **Do:** in Attention click different tokens and change the head. In the workbench compare what it writes before and after Train.
* **Ask:** can a GPT look at later tokens? Why check its molecules with RDKit?
* **Common confusion:** the demo model is tiny. It shows the mechanism, not the quality of a large language model.

### 05 Pre-trained models
* **Say:** pre-training learns chemistry from molecules with no lab values. Fine-tuning then teaches a model your labels. GROVER and Chemprop are examples.
* **Do:** play the fill-in-the-blank game. In Fine-tune, compare Head only with Everything at few labelled compounds, then change the endpoint to logD. In Fine-tune Chemprop, read the three starts at 50 and at 1000 compounds.
* **Ask:** when would you freeze the pre-trained model? Which Chemprop start wins at 50 compounds, and which at 1,000?
* **Common confusion:** pre-training does not always help. A model that is too small, or trained on chemistry unlike yours, may add nothing. The grey line shows the baseline.

### 06 Compare
* Walk the leaderboard. Use the discussion questions. Ask participants which of their results they would least trust, and why.

## Points to land

* The score depends on how the unseen set was chosen and how the data were treated. Report both.
* Censoring changes which model looks best.
* Trees need no scaling and are hard to beat on a few thousand compounds. Neural networks need care: scaling, validation, dropout, a stopping point.
* A GPT is a generative model. Predicting clearance is not its job. It copies the range of the data it read, and RDKit checks what it writes.
* Pre-training and fine-tuning help most when labels are scarce, and only when the pre-training data resemble yours.

## If something goes wrong

* A blank page or no data: the site must be opened over http (GitHub Pages or `python -m http.server`), not from a file.
* The GPT or the fine-tuning workbench does not run: the TensorFlow.js script may be blocked by a network proxy. The rest of the site still works, and the notebooks cover the same ideas.
* RDKit does not load (generation, group matching and predict-your-own-SMILES): it comes from a public CDN and is about 7 MB.
* Slow laptop: switch animations off in the header, lower the GPT training steps to 150, and use one transformer layer.
