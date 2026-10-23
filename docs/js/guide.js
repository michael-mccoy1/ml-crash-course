"use strict";
/* Lesson cards, what to notice in each step, and a pop quiz question with a hidden answer. */
var GUIDE = {
  intro: {
    start: { what: 'We predict how fast the liver breaks a molecule down (HLM clearance) and how oily it is (logD).', goals: ['Turn a molecule into numbers.', 'Test a model on compounds it has not seen.', 'Spot three data problems.'], words: ['feature', 'unseen set', 'R2', 'censoring', 'overfitting'], prior: 'No background needed.', map: [['01', 'Data and ML basics', '', 'Learning, features, fair tests, data problems.'], ['02', 'Tree models', '', 'Yes or no questions, forests, boosting.'], ['03', 'Neural networks', '', 'Weights, layers, training.'], ['04', 'GPT', '', 'Tokens, attention, writing molecules.'], ['05', 'Pre-trained models', '', 'Learn from unlabelled data, then fine-tune.'], ['06', 'Compare', '', 'Accuracy against training time.']] },
    trees: { goals: ['See how a tree predicts from yes or no questions.', 'See what forests and boosting add.', 'See where trees fail.'], words: ['tree', 'depth', 'forest', 'boosting', 'learning rate'], prior: 'Do lesson 01 first.' },
    nn: { goals: ['Learn what a weight is.', 'See training adjust many weights.', 'See overfitting and dropout.'], words: ['weight', 'neuron', 'layer', 'epoch', 'dropout'], prior: 'Do lesson 01 first.' },
    gpt: { what: 'A GPT is a model that writes text one piece at a time by guessing the next piece.', goals: ['See a molecule become tokens and numbers.', 'See a GPT learn to guess the next token.', 'Check its output with RDKit, a chemistry toolkit.'], words: ['token', 'embedding', 'attention', 'temperature', 'SMILES'], prior: 'Do lesson 03 first. The model is tiny.' },
    pre: { goals: ['Learn from molecules with no lab values.', 'Fine-tune with a few labels.', 'Freeze or update a model.'], words: ['pre-training', 'fine-tuning', 'frozen', 'GROVER', 'labelled'], prior: 'Do lessons 03 and 04 first.' },
    compare: { goals: ['Compare models on accuracy and time.', 'Decide what a fair comparison needs.'], words: ['unseen R2', 'training time'], prior: 'Train a model or two first.' }
  },
  steps: {
    'start/What learning means': { notice: 'The error number falls as the line fits better.', q: 'Can one straight line match every compound?', a: 'No. Compounds scatter around any line. Learning finds the line that misses least.' },
    'start/Structure to prediction': { notice: 'The model sees only the feature numbers.', q: 'Why can a few numbers such as logP give a rough prediction?', a: 'They describe how oily and how polar a compound is, which affects how fast the liver clears it. Other details are missing, so it stays rough.' },
    'start/Where features come from': { notice: 'Darker atoms add more.', q: 'What usually happens to logP when you add a carbon?', a: 'It rises.' },
    'start/The unseen set': { notice: 'Compare how many unseen compounds have a close relative in training.', q: 'Why is a random split too optimistic?', a: 'Unseen compounds often have near twins in training, so the model scores well by recognising twins.' },
    'start/The target': { notice: 'The tall red bar is the assay floor.', q: 'What does the tall bar say about the data?', a: 'Those values are not real measurements. A model cannot learn real differences among them.' },
    'start/Handling censoring': { notice: 'Compare the two bars on each card.', q: 'To rank measurable compounds, which option would you pick?', a: 'Drop censored values or use two models. Both score better on measured compounds.' },
    'start/Underfitting and overfitting': { notice: 'Training error keeps falling. Unseen error eventually rises.', q: 'Which depth would you pick?', a: 'The depth with the lowest unseen error, not the lowest training error.' },
    'start/Scoring a model': { notice: 'Watch R2 and Spearman as the dots move.', q: 'Can Spearman be good while R2 is poor?', a: 'Yes. Predictions can be in the right order but squeezed or shifted, so errors are large.' },
    'start/logD feeds HLM': { notice: 'Compare the two scores.', q: 'Why can a predicted logD help?', a: 'It was learned from thousands of logD measurements that the small clearance set lacks.' },
    'start/Predict a compound': { notice: 'logD is predicted first, then used for clearance.', q: 'When would you distrust this prediction?', a: 'For compounds unlike the training compounds.' },

    'trees/Choosing a question': { notice: 'The horizontal lines are each group\'s average, its prediction.', q: 'Drag the cut to the far left. What happens to the miss score?', a: 'It returns to about the no-cut value. A cut that separates nothing tells the tree nothing.' },
    'trees/Workbench': { notice: 'Watch the red unseen curve while the model grows.', q: 'Train a tree at depth 2 and at depth 12. Which scores better?', a: 'Usually a moderate depth. The deep tree memorises the training compounds.' },
    'trees/One compound': { notice: 'One yes or no answer at each box.', q: 'Do two compounds in the same leaf get the same prediction?', a: 'Yes. The prediction is the leaf average.' },
    'trees/Forest': { notice: 'The trees disagree. Their average is steadier.', q: 'Why do the trees in a forest differ?', a: 'Each learned from a different random draw of the compounds.' },
    'trees/Boosting': { notice: 'Dots move toward the diagonal as trees are added.', q: 'What goes wrong with a very large learning rate?', a: 'Each tree overcorrects, so the score bounces or worsens.' },
    'trees/Which features matter': { notice: 'Compare the two charts.', q: 'Why can a feature rank high on the left but cost little on the right?', a: 'A correlated feature can take its place.' },
    'trees/Filtering the data': { notice: 'Grey bars are the compounds removed.', q: 'Dropping censored values lowers one score and raises another. Why?', a: 'The model stops learning the pile at the floor, so it misses floor compounds but ranks measured ones better.' },
    'trees/Results': { notice: 'Cards marked current options use the options panel values.', q: 'Which model if training time matters most?', a: 'The single tree, if its unseen score is good enough.' },
    'trees/Seeds and ensembles': { notice: 'Grey bars are single runs. The red line averages the first runs.', q: 'Why does an average of runs usually beat a typical single run?', a: 'Each run makes different random mistakes. Averaging cancels part of them.' },
    'trees/Is it useful': { notice: 'Blocky, smoother, closest. All three go flat outside the data.', q: 'Which model would you trust for a compound far outside the data?', a: 'None blindly. All three repeat the last value they saw. Only a model that knows the trend could follow it.' },

    'nn/What a weight is': { notice: 'The thin lines show the misses.', q: 'Set the weight to zero. What does the model predict?', a: 'The bias for every compound. The input has no effect.' },
    'nn/One neuron': { notice: 'Red adds to the total. Grey subtracts.', q: 'What does ReLU do to a negative total?', a: 'It turns it into zero.' },
    'nn/Workbench': { notice: 'Each line is a weight. Watch them go from random to organised.', q: 'Raise Epochs to 120 and Train. What happens to the curves?', a: 'Training error keeps falling. Validation error stops falling or rises. That is overfitting.' },
    'nn/Layers and neurons': { notice: 'The table counts every weight.', q: 'Doubling the neurons per layer multiplies the weights between two hidden layers by how much?', a: 'Four.' },
    'nn/Learning rate': { notice: 'All three start from the same line.', q: 'Why does the largest rate fail?', a: 'Each step overshoots, so the error bounces or grows.' },
    'nn/Dropout and early stopping': { notice: 'Compare the two curves and their best epochs.', q: 'Dropout worsens training error. Why use it?', a: 'The network cannot lean on one neuron, so unseen error is often lower.' },
    'nn/Filtering the data': { notice: 'The network trains twice.', q: 'A filter removes 30% of compounds and the score barely moves. Why?', a: 'Those compounds added little, or others covered the same chemistry.' },
    'nn/One compound': { notice: 'Darker red is more active.', q: 'Where is the prediction made?', a: 'In the last circle.' },
    'nn/Inside a neuron': { notice: 'The table lists each learned weight.', q: 'What does a negative weight mean?', a: 'A larger input lowers this neuron\'s total.' },
    'nn/What it learned': { notice: 'The red line is the average effect.', q: 'What does a curved line show?', a: 'A non-linear effect.' },
    'nn/Chemprop': { notice: 'The red zone grows each round.', q: 'After three rounds, which atoms can affect the chosen atom?', a: 'Atoms up to three bonds away.' },
    'nn/Is it useful': { notice: 'Compare the lines at the smallest and largest sizes.', q: 'When is a network worth the effort?', a: 'Read your numbers. A network usually needs more data than a tree to catch up.' },

    'nn/Seeds and ensembles': { notice: 'Same network, different seeds, different scores.', q: 'Is an ensemble the same as combining different model types?', a: 'No. It averages several runs of the same model with different seeds.' },
    'gpt/Tokens': { notice: 'Two-letter atoms such as Cl stay together.', q: 'Why does the model need ID numbers?', a: 'Models work with numbers. Each token needs one to look up its values.' },
    'gpt/Embeddings': { notice: 'Switch between Untrained and Trained.', q: 'Are the untrained numbers meaningful?', a: 'No. They are random until training.' },
    'gpt/The stack': { notice: 'Every block has the same shape.', q: 'What does adding a layer do to the parameter count?', a: 'It adds the same amount each time.' },
    'gpt/Workbench': { notice: 'Compare what it writes before and after Train.', q: 'Does a low error guarantee valid molecules?', a: 'No. RDKit checks validity.' },
    'gpt/Attention': { notice: 'Click another token.', q: 'Why can it not look at later tokens?', a: 'It is trained to guess the next token. Seeing it would be cheating.' },
    'gpt/Next token': { notice: 'Drag Temperature and watch the bars.', q: 'Which temperature is safest but least varied?', a: 'A low one.' },
    'gpt/Generate and check': { notice: 'Valid molecules have a red outline.', q: 'Why check with RDKit?', a: 'The GPT learned text patterns. Only RDKit can say if the text is a valid molecule.' },
    'gpt/Filtering the data': { notice: 'Compare its lengths with the corpus.', q: 'Trained only on small molecules, what does it write?', a: 'Mostly small molecules.' },
    'gpt/Is it useful': { notice: 'Compare valid and new at each temperature.', q: 'Would you hand these to a chemist?', a: 'No. Some are invalid and valid ones still need checks. A GPT proposes candidates.' },

    'pre/Why pre-train': { notice: 'The curve is steep on the left.', q: 'What does pre-training try to fix?', a: 'The slow start with few labelled compounds.' },
    'pre/Learning without labels': { notice: 'The answer comes from the molecule.', q: 'Why is this called self-supervised?', a: 'The labels come from the data itself.' },
    'pre/Which groups are present': { notice: 'Click a group to highlight it.', q: 'Why is naming groups useful for pre-training?', a: 'The model must learn what groups look like.' },
    'pre/The model and its size': { notice: 'GROVER reads a graph, not text.', q: 'Why can GROVER not train in this page?', a: 'About 100 million weights and 10 million molecules is too much for a browser tab.' },
    'pre/Fixed features': { notice: 'The gap between red and grey is what pre-training added.', q: 'If red and grey are equal, what does that say?', a: 'This GPT\'s pre-training added nothing useful here.' },
    'pre/Fine-tune': { notice: 'Compare red with grey and watch the share of weights moved.', q: 'When choose Head only over Everything?', a: 'With few labels: the pre-trained model cannot be damaged.' },
    'pre/Fine-tune Chemprop': { notice: 'Hide lines to compare two at a time.', q: 'Which start wins at 1,000 compounds, and what happens to the frozen start at 50?', a: 'Full fine-tuning wins (0.41 against 0.24 from scratch). Frozen is worse than scratch at 50 and better from 100 up.' },
    'pre/Using it in practice': { notice: 'Four stages.', q: 'What must stay the same when comparing with a descriptor model?', a: 'The split and the unseen compounds.' },

    'compare/Leaderboard': { notice: 'Red bars are yours.', q: 'Why are your scores lower than the reference runs?', a: 'They use thousands of extra features and train longer.' },
    'compare/Accuracy against time': { notice: 'Up is more accurate. Left is faster.', q: 'What if a model is both slower and less accurate?', a: 'There is no reason to choose it.' },
    'compare/Discussion': { notice: 'Answer from your own scores.', q: 'Which result would you least trust?', a: 'Any score from one split or a small unseen set.' }
  }
};
