# Instructor Guide

## Before the session

* Open the site and hard refresh (Ctrl+F5). Click each lesson tab once.
* Nothing needs installing.

## How the site works

* Each lesson opens with its goals.
* The options panel changes every stage. Have people change something before reading.
* The tree, network and GPT lessons have a workbench. Change an option, then press Train. Slow shows every step. Fast skips ahead.
* Each step has a Notice line and a Pop quiz. Ask the question before anyone opens the answer.

## Lessons

| Lesson | Cover |
|---|---|
| 01 Data and ML basics | Fitting a line, features, the test set, assay floor, overfitting |
| 02 Tree models | One cut, workbench, forest, boosting, all three side by side, seeds |
| 03 Neural networks | One weight, one neuron, workbench, learning rate, dropout, the comparison test, seeds |
| 04 GPT | Tokens, attention, workbench, the design test |
| 05 Pre-trained models | Fill in the blank, GROVER, fine-tuning a GPT and Chemprop |
| 06 Compare | Leaderboard, discussion |
| 07 Role of AI | Machine learning against generative AI, which fits which job, the training range, lab noise, how generative AI fails, can and cannot, five checks |
| 08 Tree algorithms | Split search, growth limits, bagging, boosting losses, tuning, library designs, fitting a double helix, trees against data |
| 09 Network algorithms | Gradient descent, backpropagation, initialisation, weight decay, attention, next-token loss, parameter counts, network size against data |
| 10 Quiz | Ten random questions from a bank of 75, including challenges that train real models |

### 01 Data and ML basics
* **Say:** a model is a rule fitted to examples.
* **Do:** drag the weight and bias, then press Find the best weights. Switch the split in The test set.
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

### 07 Role of AI
* **Say:** machine learning predicts a number you can score. Generative AI produces content you must check. People decide.
* **Do:** click through the layers, then the six jobs and both tools. Open each of the five failures. Drag the training range until the score outside it falls to zero. Raise the lab noise. Sort the ten statements.
* **Ask:** which of the five checks would have caught the weakest result you saw today?
* **Watch for:** a score below the noise ceiling is not a failure. It may be as good as the assay allows.

### 08 Tree algorithms
* **Say:** a split is the threshold with the largest drop in squared error. Everything else, forests, boosting and tuning, is built on that search.
* **Do:** compare exact and binned search. Raise the depth and watch test R2 peak and fall. Fit forests with 1 and 8 features per split. Run the tuning grid. Play the helix fit and rotate it. Play the data step and read the final chart.
* **Ask:** why must the test set stay out of the choice of settings?
* **Watch for:** the boosting step is simplified. Real libraries use second-order steps.

### 09 Network algorithms
* **Say:** training is gradient descent, and backpropagation is how the gradients are found.
* **Do:** run the three optimisers on the raw and the centred feature. Step through the backward pass. Move the weight scale in the initialisation step. Drag the attention query. Play the scaling step with 100, 1,000 and 10,000 points and compare the five network sizes.
* **Ask:** why does a sigmoid network lose its gradient in a deep stack?
* **Watch for:** the attention and parameter steps use tiny sizes so the numbers can be read. Real models are far larger.

### 10 Quiz
* **Say:** each round is ten questions chosen at random from 75, so a second round is different from the first.
* **Do:** run a round as a group, or let people work alone. In the challenge questions the goal counts leaves, the pieces a tree model builds, so a smaller model is a faster one on any computer.
* **Ask:** which setting did you change first in the challenge, and why?
* **Watch for:** the data allow a test R2 of only about 0.3 on HLM and 0.36 on logD with these eight descriptors, so the challenge goals are set near those limits.

## If something goes wrong

* Blank page: open the site over http, not from a file.
* GPT steps blank: an ad blocker or policy may block a script. Press F12 and read the Console.
* RDKit steps fail: RDKit is about 7 MB. Wait for it to load.
* Slow laptop: switch animations off and lower the GPT training steps.
