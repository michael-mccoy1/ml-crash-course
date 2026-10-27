"use strict";
/* Tab 10: a quiz. A bank of 75 questions, ten drawn at random each round.
   Knowledge questions are multiple choice, select-all, numeric or ordering.
   Challenge questions train a real model in the browser and check the result against a goal. */
var QZ = { bank: [], cur: [], i: 0, score: 0, log: [], root: null, drawn: false };
var QZ_DRAW = 10;

/* ---------- question builders ---------- */
function qMC(topic, text, right, wrong, why){ return { t: 'mc', topic: topic, text: text, opts: [right].concat(wrong), ans: 0, why: why }; }
function qMulti(topic, text, right, wrong, why){ return { t: 'multi', topic: topic, text: text, opts: right.concat(wrong), ans: right.map(function(_, i){ return i; }), why: why }; }
function qNum(topic, text, ans, tol, unit, why){ return { t: 'num', topic: topic, text: text, ans: ans, tol: tol, unit: unit || '', why: why }; }
function qOrder(topic, text, steps, why){ return { t: 'order', topic: topic, text: text, steps: steps, why: why }; }
function qTune(topic, o){ o.t = 'tune'; o.topic = topic; return o; }

/* ---------- knowledge questions ---------- */
(function(){
  var B = QZ.bank;
  /* 01 Data and ML basics */
  B.push(qMC('Data and ML basics', 'Why do we hold back test compounds?', 'To check that the model learned a real pattern and did not just memorise the training compounds.', ['To make training faster.', 'To give the model more compounds to learn from.', 'To remove compounds with unusual values.'], 'The model never sees test compounds while it learns, so its score on them shows how it will do on new compounds.'));
  B.push(qMC('Data and ML basics', 'A model scores R2 0.95 on its training compounds and 0.20 on its test compounds. What has happened?', 'It has overfitted: it memorised the training compounds, noise included.', ['It is an excellent model, and the test set is just hard.', 'The test set is too large.', 'The model is underfitted.'], 'A big gap between training and test scores is the sign of overfitting. The test score is the one that matters.'));
  B.push(qMC('Data and ML basics', 'What does a test R2 of 0 mean?', 'The model is no better than predicting the average every time.', ['The model is perfect.', 'The model got every compound wrong.', 'The model needs more features.'], 'R2 is the share of the spread in the answers that the model explains. Zero means it explains none of it.'));
  B.push(qMC('Data and ML basics', 'What is a scaffold split?', 'The test compounds share no ring skeleton with the training compounds.', ['The test compounds are the largest molecules.', 'The test compounds are picked by their clearance values.', 'The training compounds are split into two equal groups.'], 'A scaffold is the ring skeleton of a compound. Splitting by scaffold tests whether a model works on new chemical series, not just near copies of training compounds.'));
  B.push(qMC('Data and ML basics', 'About 31% of the HLM compounds sit exactly at the assay floor. What does that mean?', 'The assay cannot report clearance below that value, so the true values are lower by an unknown amount.', ['Those compounds were measured twice.', 'Those compounds are all the same molecule.', 'Those values are errors and should be deleted.'], 'This is called censoring. The value written down is a limit, not a measurement.'));
  B.push(qNum('Data and ML basics', 'The HLM set has 3,087 compounds and 31% of them are at the assay floor. About how many compounds is that?', 957, 25, 'compounds', '0.31 times 3,087 is about 957.'));
  B.push(qMC('Data and ML basics', 'What does Spearman correlation measure?', 'How well the model ranks compounds, whatever the exact values.', ['The typical error in the units of the endpoint.', 'The share of the spread explained.', 'How long the model took to train.'], 'Ranking matters when you only need to pick the best compounds. RMSE is the error in the endpoint units.'));
  B.push(qMC('Data and ML basics', 'What is logD?', 'logP corrected for ionisation at pH 7.4.', ['The distance between two atoms.', 'The log of the dose.', 'The same thing as molecular weight.'], 'A compound that is charged at pH 7.4 prefers water more than its logP suggests. logD accounts for that.'));
  B.push(qMulti('Data and ML basics', 'Which of these let information from the test compounds leak into the model? Select all that apply.', ['Choosing settings by repeatedly checking the test R2.', 'Adding the test compounds to the training set.'], ['Choosing settings with a validation set cut from the training compounds.', 'Standardising features with the mean of the training compounds only.'], 'Anything that lets the test compounds guide the model makes the test score too optimistic. A validation set cut from training compounds is the safe way to compare settings.'));

  /* 02 Tree models */
  B.push(qMC('Tree models', 'How does a decision tree predict for a compound?', 'It follows yes or no questions to a leaf and gives the average of the training compounds in that leaf.', ['It multiplies each feature by a weight and adds them.', 'It finds the single most similar training compound.', 'It averages every training compound.'], 'The leaf average is the prediction, so every compound that lands in the same leaf gets the same answer.'));
  B.push(qMC('Tree models', 'You deepen a single tree from 4 levels to 12. What usually happens?', 'Training R2 rises, and test R2 stops rising or falls.', ['Both training and test R2 rise.', 'Training R2 falls and test R2 rises.', 'Nothing changes.'], 'A deep tree makes tiny leaves and memorises noise. It looks better on training compounds and worse on new ones.'));
  B.push(qMC('Tree models', 'Why does a random forest average many trees, each trained on a different random draw of the compounds?', 'The errors of individual trees partly cancel when they are averaged.', ['Each tree fixes the mistakes of the one before it.', 'It makes each tree deeper.', 'It removes the need for a test set.'], 'Trees trained on different draws make different mistakes. Their average is steadier than any one tree.'));
  B.push(qMC('Tree models', 'In gradient boosting, what is each new tree trained to predict?', 'The mistakes the earlier trees left behind.', ['The original target again.', 'A random draw of the compounds.', 'The average of all earlier trees.'], 'Each tree adds a correction. Boosting is a sequence, where a forest is a crowd.'));
  B.push(qMC('Tree models', 'What does a small learning rate do in boosting?', 'Each tree adds only a small correction, so more rounds are needed.', ['It makes each tree shallower.', 'It removes noise from the target.', 'It makes training faster.'], 'Small steps are careful but slow. A large rate can fit quickly and overshoot into noise.'));
  B.push(qNum('Tree models', 'A tree may ask at most 5 questions in a row (depth 5). What is the largest number of leaves it can have?', 32, 0, 'leaves', 'Each level can double the number of groups: 2 to the power 5 is 32.'));

  /* 03 Neural networks */
  B.push(qNum('Neural networks', 'A network has 8 inputs, one hidden layer of 16 units and 1 output. How many parameters (weights plus biases) does it have?', 161, 0, 'parameters', 'Inputs to hidden: 8 times 16 weights plus 16 biases is 144. Hidden to output: 16 weights plus 1 bias is 17. Total 161.'));
  B.push(qNum('Neural networks', 'A network has 3 inputs, two hidden layers of 4 units each, and 1 output. How many parameters does it have?', 41, 0, 'parameters', '3 times 4 plus 4 is 16. 4 times 4 plus 4 is 20. 4 times 1 plus 1 is 5. Total 41.'));
  B.push(qMC('Neural networks', 'Why does a network need an activation function such as ReLU between layers?', 'Without it the layers collapse into one straight-line formula, so the network could not learn bends.', ['It makes the weights smaller.', 'It speeds up each epoch.', 'It chooses the learning rate.'], 'The bend is what lets a network fit curved patterns.'));
  B.push(qMC('Neural networks', 'Training loss keeps falling, but loss on the validation compounds starts to rise. What should you do?', 'Stop training, because the network has started to memorise.', ['Train for longer.', 'Raise the learning rate.', 'Remove the validation compounds.'], 'That is the signal early stopping uses: keep the weights from the epoch with the lowest validation loss.'));
  B.push(qMC('Neural networks', 'What does dropout do during training?', 'It switches off random units, so no single unit can be relied on.', ['It removes the lowest-scoring compounds.', 'It deletes the last layer.', 'It lowers the learning rate.'], 'Dropout is a way to fight overfitting. It is off when the trained network makes predictions.'));
  B.push(qMC('Neural networks', 'What usually happens if the learning rate is far too high?', 'The loss bounces around or blows up instead of falling.', ['Training becomes more accurate.', 'The network ignores the learning rate.', 'Training finishes sooner with a better score.'], 'Each step overshoots the bottom of the valley.'));
  B.push(qMulti('Neural networks', 'Which of these reduce overfitting in a neural network? Select all that apply.', ['Dropout.', 'Weight decay.', 'Early stopping.', 'More training compounds.'], ['Training for many more epochs with no checks.', 'Making the network much larger with the same data.'], 'Each of the first four limits how closely the network can memorise. The last two make memorising easier.'));

  /* 04 GPT */
  B.push(qMC('GPT', 'In the course GPT, what is a token?', 'An atom, a bond, a bracket or a ring digit in a SMILES string.', ['A whole molecule.', 'A measured clearance value.', 'A password.'], 'The vocabulary is the set of tokens the model knows.'));
  B.push(qMC('GPT', 'What is a GPT trained to do?', 'Predict the next token.', ['Predict the clearance of a compound.', 'Sort compounds into groups.', 'Draw the structure of a molecule.'], 'Writing text is repeated next-token prediction: predict, pick a token, add it, repeat.'));
  B.push(qMC('GPT', 'What does the causal mask do?', 'It stops a token looking at later tokens, so the model only uses the past.', ['It hides the test compounds.', 'It removes rare tokens.', 'It lowers the temperature.'], 'Without the mask, the model could read the answer it is supposed to predict.'));
  B.push(qMC('GPT', 'You lower the sampling temperature. What happens to the text the GPT writes?', 'It picks the likeliest token more often, so the output is less varied.', ['The output becomes more random.', 'The model forgets its training.', 'The vocabulary shrinks.'], 'High temperature gives more varied output. Low temperature is safer and more repetitive.'));
  B.push(qMC('GPT', 'Why can a GPT write a SMILES string that is not a valid molecule?', 'It learns patterns of text. Nothing in training checks the chemistry.', ['Because it ran out of memory.', 'Because RDKit changed the string.', 'Because SMILES strings are never valid.'], 'RDKit is used to check validity after the model writes. The share of valid strings is a measure of how well it learned the patterns.'));

  /* 05 Pre-trained models */
  B.push(qMC('Pre-trained models', 'What does self-supervised mean?', 'The labels come from the data itself, such as a hidden atom whose identity is known, so no experiment is needed.', ['A person labels every compound by hand.', 'The model grades its own test score.', 'The model trains with no data.'], 'This is how a model can learn from millions of molecules that have no lab values.'));
  B.push(qMC('Pre-trained models', 'What is fine-tuning?', 'Adjusting a pre-trained model, or a small model on top of it, using your own labelled compounds.', ['Training a new model from random weights.', 'Making the test set smaller.', 'Lowering the temperature.'], 'The general knowledge is already there, so a small labelled set can be enough.'));
  B.push(qMC('Pre-trained models', 'Why pre-train on millions of molecules that have no lab values?', 'The model learns general patterns of chemistry, so a small labelled set goes further.', ['Because lab values are always wrong.', 'Because it makes the model smaller.', 'Because it removes the need for a test set.'], 'Labelled data is expensive. Unlabelled structures are cheap.'));

  /* 06 Compare */
  B.push(qMC('Compare', 'What is a dominated model?', 'A model that another model beats on both accuracy and speed.', ['The model with the most parameters.', 'The model trained on the most compounds.', 'The model that was trained first.'], 'You would never pick a dominated model, since another one is better in both ways.'));
  B.push(qMC('Compare', 'What does assay noise do to the scores of every model?', 'It limits the R2 any model can reach.', ['It makes tree models better than networks.', 'It only affects the training set.', 'Nothing, because models average it out.'], 'If a repeat measurement of the same compound gives a different value, no model can predict the noise.'));

  /* 07 Role of AI */
  B.push(qMC('Role of AI', 'What is a hallucination?', 'Text from a language model that sounds right and is false.', ['A model that scores R2 of 1.', 'A compound with no measured value.', 'A picture of a molecule.'], 'A language model writes likely text, and likely is not the same as true.'));
  B.push(qMC('Role of AI', 'Why does a model often do badly on compounds outside its training range?', 'It has no examples there, so it can only stretch a pattern learned from the range it saw.', ['Because outside values are always noisy.', 'Because models forget old compounds.', 'Because the test set is smaller there.'], 'A tree predicts a leaf average, so it goes flat outside the range. A network may extend a line that has no support.'));
  B.push(qMC('Role of AI', 'Two lab repeats of the same compounds agree with R2 of about 0.8. What is a sensible expectation for a model scored against that assay?', 'Its test R2 is unlikely to be much above about 0.8, because the assay itself is that noisy.', ['It should reach 1.0 with enough data.', 'It should reach 0.8 only if it is a deep network.', 'The assay noise does not matter to a model.'], 'A model cannot predict the random part of a measurement.'));

  /* 08 Tree algorithms */
  B.push(qNum('Tree algorithms', 'A group of compounds has a sum of squared errors (SSE) of 100. A split sends them into two groups with SSE 30 and 25. What is the gain?', 45, 0, '', 'Gain is the SSE before minus the SSE after: 100 minus 55 is 45.'));
  B.push(qMC('Tree algorithms', 'Why do tree libraries group feature values into bins before searching for splits?', 'There are far fewer candidate thresholds to test, so the search is much faster.', ['It makes every tree shallower.', 'It removes outliers.', 'It makes the test set bigger.'], 'With 64 bins there are at most 63 thresholds per feature, however many compounds there are.'));
  B.push(qMC('Tree algorithms', 'What is the minimum leaf size?', 'The fewest compounds a leaf may hold. A split that would break it is rejected.', ['The smallest feature value.', 'The depth of the shallowest leaf.', 'The number of trees in a forest.'], 'Raising it stops a tree making tiny leaves that memorise single compounds.'));
  B.push(qNum('Tree algorithms', 'A bootstrap sample draws as many compounds as the training set, at random, with replacement. About what percent of the training compounds are left out of a given sample?', 37, 3, 'percent', 'The chance a compound is never drawn is about 1 divided by e, which is 36.8%. Those compounds are the out-of-bag set for that tree.'));
  B.push(qMC('Tree algorithms', 'Which growth rule can overfit small data more easily?', 'Leaf-wise growth, which always splits the leaf with the largest gain.', ['Level-wise growth, which splits every node at one depth first.', 'They are identical.', 'Neither can overfit.'], 'Leaf-wise growth can build a very deep, lopsided tree in one place.'));
  B.push(qMC('Tree algorithms', 'What is a validation set?', 'Training compounds held back to compare settings. It is not the test set.', ['The test compounds.', 'The compounds with the highest values.', 'The compounds a tree gets wrong.'], 'You can tune on a validation set as often as you like without spoiling the test score.'));
  B.push(qNum('Tree algorithms', 'In boosting, the learning rate is 0.1 and a new tree predicts a correction of +2.0 for a compound. By how much does the prediction for that compound move?', 0.2, 0.001, '', 'The correction is multiplied by the learning rate: 0.1 times 2.0 is 0.2.'));
  B.push(qMulti('Tree algorithms', 'Which of these are hyperparameters of gradient boosting? Select all that apply.', ['The number of rounds.', 'The tree depth.', 'The learning rate.'], ['The average value stored in each leaf.', 'The threshold of each split.'], 'You set the first three before training. The leaf averages and thresholds are found by the algorithm.'));

  /* 09 Network algorithms */
  B.push(qMC('Network algorithms', 'The gradient of the loss with respect to a weight points which way?', 'Uphill, so a step goes the opposite way.', ['Downhill, so a step follows it.', 'Toward the test set.', 'It has no direction.'], 'Gradient descent subtracts the gradient times the learning rate.'));
  B.push(qMC('Network algorithms', 'What does the learning rate set in gradient descent?', 'The size of each weight update, as a multiple of the gradient.', ['The number of layers.', 'The number of compounds in a batch.', 'The share of weights set to zero.'], 'A step is the learning rate times the slope. Too small is slow, and too large overshoots.'));
  B.push(qMC('Network algorithms', 'What does Adam do that plain gradient descent does not?', 'It divides each step by the recent size of the gradient.', ['It removes the test set.', 'It makes the network wider.', 'It ignores the loss.'], 'That rescales each weight, so a flat direction and a steep one both make progress.'));
  B.push(qMC('Network algorithms', 'What is momentum?', 'A running sum of earlier gradients, so steps keep a steady direction.', ['The size of the batch.', 'A penalty on large weights.', 'The number of epochs.'], 'Momentum smooths out steps that zigzag across a narrow valley.'));
  B.push(qMC('Network algorithms', 'What is backpropagation?', 'Finding every gradient by applying the chain rule from the loss back through the layers.', ['Running the network in reverse to make compounds.', 'Training for fewer epochs.', 'Copying weights between networks.'], 'It gives all the gradients in about the cost of one extra pass.'));
  B.push(qMC('Network algorithms', 'What is a vanishing gradient?', 'A gradient that shrinks toward zero as it travels back through many layers.', ['A gradient that becomes infinite.', 'A weight that is deleted.', 'A loss that is zero.'], 'The early layers then barely learn.'));
  B.push(qMC('Network algorithms', 'What does weight decay do?', 'It pulls each weight toward zero a little at every step, which discourages memorising.', ['It makes the learning rate smaller over time.', 'It deletes the smallest weights.', 'It adds more layers.'], 'Large weights let a network bend sharply around single compounds.'));
  B.push(qNum('Network algorithms', 'A language model has a next-token loss of 0.693, which is the natural log of 2. What is its perplexity?', 2, 0.05, '', 'Perplexity is e raised to the loss. It is the number of equally likely tokens the model is choosing between.'));
  B.push(qNum('Network algorithms', 'The model gives the correct next token a probability of 0.5. What is the cross-entropy loss for that token?', 0.69, 0.02, '', 'Cross-entropy is minus the natural log of the probability given to the correct token: minus ln 0.5 is 0.693.'));
  B.push(qMC('Network algorithms', 'You train networks of different sizes on only 100 points. What do you see?', 'Even the biggest network scores about the same on held-out points as a small one, while it fits its 100 training points almost perfectly.', ['The biggest network always scores highest.', 'Every network scores perfectly on held-out points.', 'The smallest network overfits the most.'], 'A network needs data to put its parameters to use. With 100 points it memorises them.'));
  B.push(qMC('Network algorithms', 'With 10,000 points, a network with about 100,000 parameters scores only a little above one with about 10,000. What does that show?', 'Returns diminish: ten times the parameters buys a small gain.', ['Large networks never help.', 'The data has no noise.', 'The small network has overfitted.'], 'More parameters help until the data runs out of detail to teach them.'));

  /* orderings */
  B.push(qOrder('Tree models', 'Put the steps of one round of gradient boosting in order.', ['Work out the mistake left on each training compound.', 'Fit a small tree to those mistakes.', 'Add the tree to the model, scaled by the learning rate.', 'Start the next round with the new mistakes.'], 'Each round fits a small tree to what is still wrong, adds a scaled copy of it, and then looks again.'));
  B.push(qOrder('Data and ML basics', 'Put these steps of a sound modelling workflow in order.', ['Split the compounds into training and test sets.', 'Fit the model on the training compounds.', 'Compare settings on a validation set cut from the training compounds.', 'Score the final model once on the test compounds.'], 'The test compounds are touched last and only once, so the score is an honest estimate.'));
  B.push(qOrder('Network algorithms', 'Put one training step of a neural network in order.', ['Run the inputs forward to get a prediction.', 'Compute the loss.', 'Send gradients back through the layers.', 'Update every weight against its gradient.'], 'Forward pass, loss, backpropagation, update.'));
  B.push(qOrder('GPT', 'Put the steps a GPT repeats to write a SMILES string in order.', ['Feed in the tokens written so far.', 'Compute a score for every token in the vocabulary.', 'Turn the scores into probabilities with softmax.', 'Pick the next token and add it to the string.'], 'The loop repeats until the model picks an end token.'));
})();

/* ---------- challenge questions: train a real model and meet a goal ---------- */
/* goal types: budget (test R2 at least r2 with at most leaves), gap (training R2 minus test R2 at least gap),
   low (training R2 at most x), best (test R2 at least r2 with at most leaves, any settings) */
(function(){
  var B = QZ.bank;
  B.push(qTune('Challenge: gradient boosting', { kind: 'boost', ds: 'HLM', init: { rounds: 100, depth: 5, lr: 0.1 }, goal: { type: 'budget', r2: 0.27, leaves: 80 },
    text: 'On the HLM data, find boosting settings with a test R2 of at least 0.27 using no more than 80 leaves in total. Fewer leaves is a faster fit, on any computer.', hint: 'The score levels off long before 100 rounds. Try a high learning rate with shallow trees.',
    why: 'Shallow trees with a high learning rate reach the plateau in a handful of rounds. Most of the extra leaves in a big model buy almost nothing.' }));
  B.push(qTune('Challenge: gradient boosting', { kind: 'boost', ds: 'HLM', init: { rounds: 300, depth: 6, lr: 0.05 }, goal: { type: 'budget', r2: 0.28, leaves: 600 },
    text: 'On the HLM data, find boosting settings with a test R2 of at least 0.28 using no more than 600 leaves in total.', hint: 'Depth 6 overfits here. Depth 3 or 4 does better, and needs far fewer leaves.',
    why: 'Deeper trees score worse on test compounds here, and cost more leaves.' }));
  B.push(qTune('Challenge: gradient boosting', { kind: 'boost', ds: 'logD', init: { rounds: 20, depth: 2, lr: 0.1 }, goal: { type: 'budget', r2: 0.33, leaves: 1600 },
    text: 'On the logD data, find boosting settings with a test R2 of at least 0.33 using no more than 1,600 leaves in total.', hint: 'Start with 100 rounds, depth 4 and a learning rate of 0.1.',
    why: 'The logD data has more compounds and more signal, so it supports more rounds and deeper trees than HLM.' }));
  B.push(qTune('Challenge: gradient boosting', { kind: 'boost', ds: 'logD', init: { rounds: 100, depth: 6, lr: 0.3 }, goal: { type: 'budget', r2: 0.31, leaves: 300 },
    text: 'On the logD data, find the cheapest boosting model you can: a test R2 of at least 0.31 with no more than 300 leaves in total.', hint: 'Try about 30 rounds at depth 2 with a learning rate of 0.3.',
    why: 'A third of the leaves of the default settings reach nearly the same score.' }));
  B.push(qTune('Challenge: memorising', { kind: 'boost', ds: 'HLM', init: { rounds: 20, depth: 2, lr: 0.1 }, goal: { type: 'gap', gap: 0.5 },
    text: 'On the HLM data, make a boosting model memorise: find settings where the training R2 is at least 0.5 higher than the test R2.', hint: 'Deep trees, many rounds and a high learning rate all help a model memorise.',
    why: 'Deep trees give tiny leaves that fit single compounds, noise included. The training score soars and the test score falls.' }));
  B.push(qTune('Challenge: underfitting', { kind: 'boost', ds: 'HLM', init: { rounds: 100, depth: 4, lr: 0.1 }, goal: { type: 'low', train: 0.1 },
    text: 'On the HLM data, make a boosting model underfit: find settings where even the training R2 is below 0.1.', hint: 'Use very few rounds and a very small learning rate.',
    why: 'A model that is too weak or too short-trained misses the pattern even in the compounds it was shown.' }));
  B.push(qTune('Challenge: random forest', { kind: 'forest', ds: 'HLM', init: { trees: 5, depth: 3 }, goal: { type: 'budget', r2: 0.29, leaves: 1500 },
    text: 'On the HLM data, find random forest settings with a test R2 of at least 0.29 using no more than 1,500 leaves in total.', hint: 'Try 20 trees of depth 6 or 7, then see what depth 8 does.',
    why: 'A forest of moderately deep trees beats a very large one on cost, and the score stops rising after a few dozen trees.' }));
  B.push(qTune('Challenge: random forest', { kind: 'forest', ds: 'logD', init: { trees: 5, depth: 3 }, goal: { type: 'budget', r2: 0.34, leaves: 4000 },
    text: 'On the logD data, find random forest settings with a test R2 of at least 0.34 using no more than 4,000 leaves in total.', hint: 'Between 5 and 20 trees of depth 8 do it.',
    why: 'Each forest tree can be deep, because averaging cancels their noise. A single deep tree could not do this.' }));
  B.push(qTune('Challenge: random forest', { kind: 'forest', ds: 'HLM', init: { trees: 100, depth: 12 }, goal: { type: 'budget', r2: 0.285, leaves: 700 },
    text: 'On the HLM data, find the cheapest random forest you can: a test R2 of at least 0.285 with no more than 700 leaves in total.', hint: 'Fewer, shallower trees. Try 10 trees of depth 6.',
    why: 'A few shallow trees get most of the benefit of averaging.' }));
  B.push(qTune('Challenge: single tree', { kind: 'tree', ds: 'HLM', init: { depth: 10, minleaf: 5 }, goal: { type: 'budget', r2: 0.24, leaves: 20 },
    text: 'On the HLM data, find a single tree with a test R2 of at least 0.24 using no more than 20 leaves.', hint: 'Depth 4 is the sweet spot. A deeper tree scores worse.',
    why: 'A tree of depth 4 has at most 16 leaves and scores best. Deeper trees fit noise.' }));
  B.push(qTune('Challenge: single tree', { kind: 'tree', ds: 'logD', init: { depth: 2, minleaf: 5 }, goal: { type: 'budget', r2: 0.26, leaves: 80 },
    text: 'On the logD data, find a single tree with a test R2 of at least 0.26 using no more than 80 leaves.', hint: 'Try depth 6.',
    why: 'logD has more compounds, so a deeper single tree pays off before it starts to memorise.' }));
  B.push(qTune('Challenge: single tree', { kind: 'tree', ds: 'HLM', init: { depth: 3, minleaf: 5 }, goal: { type: 'gap', gap: 0.5 },
    text: 'On the HLM data, grow a single tree that memorises: a training R2 at least 0.5 higher than its test R2.', hint: 'Remove the limits: a deep tree with a minimum leaf size of 1.',
    why: 'With a minimum leaf size of 1 and enough depth, every training compound gets its own leaf. Training R2 reaches almost 1, and test R2 collapses.' }));
  B.push(qTune('Challenge: minimum leaf size', { kind: 'tree', ds: 'HLM', init: { depth: 12, minleaf: 1 }, goal: { type: 'budget', r2: 0.23, leaves: 200 },
    text: 'On the HLM data, a depth 12 tree with a minimum leaf size of 1 overfits. Fix it by changing the settings: reach a test R2 of at least 0.23 with no more than 200 leaves.', hint: 'Raise the minimum leaf size, or lower the depth.',
    why: 'A larger minimum leaf size stops the tree cutting tiny groups, which is a second way to control the depth.' }));
  B.push(qTune('Challenge: gradient boosting', { kind: 'boost', ds: 'HLM', init: { rounds: 400, depth: 8, lr: 0.3 }, goal: { type: 'budget', r2: 0.285, leaves: 500 },
    text: 'On the HLM data, find boosting settings with a test R2 of at least 0.285 using no more than 500 leaves. This is close to the best score the data allows.', hint: 'Try about 50 rounds of depth 3 with a learning rate of 0.1.',
    why: 'The best boosted models on these eight features score about 0.29. Very little is gained after that, whatever the settings.' }));
  B.push(qTune('Challenge: learning rate', { kind: 'boost', ds: 'logD', init: { rounds: 10, depth: 4, lr: 0.01 }, goal: { type: 'budget', r2: 0.3, leaves: 200 },
    text: 'On the logD data, the learning rate is far too small, so 10 rounds get nowhere. Reach a test R2 of at least 0.30 using no more than 200 leaves.', hint: 'Raise the learning rate. A rate of 0.3 with about 10 rounds is enough.',
    why: 'A small learning rate needs many rounds. A larger one gets the same distance in fewer.' }));
  B.push(qTune('Challenge: random forest', { kind: 'forest', ds: 'logD', init: { trees: 50, depth: 8 }, goal: { type: 'low', train: 0.3 },
    text: 'On the logD data, build a random forest that underfits: a training R2 below 0.3.', hint: 'Shallow trees cannot learn much, however many you average.',
    why: 'Averaging many trees that are each too shallow does not make a model that fits.' }));
  B.push(qTune('Challenge: random forest', { kind: 'forest', ds: 'HLM', init: { trees: 5, depth: 5 }, goal: { type: 'gap', gap: 0.35 },
    text: 'On the HLM data, build a random forest whose training R2 is at least 0.35 higher than its test R2.', hint: 'Make every tree very deep.',
    why: 'Even a forest memorises when its trees are deep enough, although averaging softens it.' }));
})();

/* ---------- running a challenge model ---------- */
function qzLeaves(n){ return n.f === undefined ? 1 : qzLeaves(n.l) + qzLeaves(n.r); }
async function qzRun(q, s){
  var ds = await loadDataset(q.ds), keep = D; D = ds;
  try {
    var f = ds.BASE.slice(), t0 = performance.now(), pred, leaves = 0;
    if (q.kind === 'boost') { var m = fitBoost(f, ds.TRAIN, s.rounds, s.lr, s.depth); pred = m.F; m.trees.forEach(function(t){ leaves += qzLeaves(t); }); }
    else if (q.kind === 'forest') { var fr = fitForest(f, ds.TRAIN, s.trees, s.depth, 7); pred = fr.pred; fr.trees.forEach(function(t){ leaves += qzLeaves(t); }); }
    else { var tr = fitTree(f, ds.TRAIN, s.depth, s.minleaf, 1); pred = treeAll(tr, ds.N); leaves = qzLeaves(tr); }
    return { te: r2(ds.TEST, pred), tr: r2(ds.TRAIN, pred), leaves: leaves, ms: performance.now() - t0 };
  } finally { D = keep; }
}
function qzPass(g, r){
  if (g.type === 'budget') return r.te >= g.r2 && r.leaves <= g.leaves;
  if (g.type === 'gap') return r.tr - r.te >= g.gap;
  return r.tr <= g.train;
}
function qzGoalText(g){
  if (g.type === 'budget') return 'Goal: test R2 at least ' + g.r2.toFixed(2) + ' with at most ' + g.leaves.toLocaleString() + ' leaves.';
  if (g.type === 'gap') return 'Goal: training R2 minus test R2 at least ' + g.gap.toFixed(2) + '.';
  return 'Goal: training R2 at most ' + g.train.toFixed(2) + '.';
}

/* ---------- the page ---------- */
function qzShuffle(a, rnd){ var o = a.slice(); for (var i = o.length - 1; i > 0; i--) { var j = Math.floor(rnd() * (i + 1)), t = o[i]; o[i] = o[j]; o[j] = t; } return o; }
function qzDraw(){
  var rnd = Math.random, tunes = QZ.bank.filter(function(q){ return q.t === 'tune'; }), other = QZ.bank.filter(function(q){ return q.t !== 'tune'; });
  var nt = 2 + Math.floor(rnd() * 3), picks = qzShuffle(tunes, rnd).slice(0, nt).concat(qzShuffle(other, rnd).slice(0, QZ_DRAW - nt));
  QZ.cur = qzShuffle(picks, rnd).map(function(q){ var c = Object.assign({}, q); if (c.t === 'mc' || c.t === 'multi') { var idx = qzShuffle(c.opts.map(function(_, i){ return i; }), rnd); c.view = idx; } if (c.t === 'order') { var o; do { o = qzShuffle(c.steps.map(function(_, i){ return i; }), rnd); } while (o.every(function(v, i){ return v === i; })); c.view = o; } return c; });
  QZ.i = 0; QZ.score = 0; QZ.log = [];
}
function qzTopbar(){
  var dots = QZ.cur.map(function(_, i){ var st = QZ.log[i]; return '<i class="' + (st !== undefined ? (st ? 'ok' : 'no') : (i === QZ.i && QZ.i < QZ.cur.length ? 'now' : '')) + '"></i>'; }).join('');
  return '<div class="qtop"><div class="qdots">' + dots + '</div><div class="qsc">' + (QZ.i < QZ.cur.length ? 'Question ' + (QZ.i + 1) + ' of ' + QZ.cur.length : 'Finished') + '</div></div>';
}
function qzRender(){
  var root = QZ.root; if (!root) return; var q = QZ.cur[QZ.i];
  if (!q) return qzFinish();
  var html = '<div class="qwrap">' + qzTopbar() + '<div class="qcard"><div class="qtopic">' + esc(q.topic) + '</div><h3 class="qtext">' + esc(q.text) + '</h3><div class="qbody"></div><div class="qfb" hidden></div><div class="qact"></div></div></div>';
  root.innerHTML = '<div class="qpage">' + html + '</div>';
  var body = qs('.qbody', root), act = qs('.qact', root), fb = qs('.qfb', root), done = false;
  function finish(ok, extra){ if (done) return; done = true; QZ.log[QZ.i] = ok; if (ok) QZ.score++; fb.hidden = false; fb.className = 'qfb ' + (ok ? 'ok' : 'no'); fb.innerHTML = '<b>' + (ok ? 'Correct.' : 'Not quite.') + '</b> ' + (extra || '') + ' ' + esc(q.why || ''); act.innerHTML = '<button type="button" class="btn" data-next>' + (QZ.i === QZ.cur.length - 1 ? 'See my score' : 'Next question') + '</button>'; qs('[data-next]', act).addEventListener('click', function(){ QZ.i++; qzRender(); window.scrollTo({ top: 0 }); }); qs('.qtop', root).outerHTML = qzTopbar(); }
  if (q.t === 'mc' || q.t === 'multi') {
    var multi = q.t === 'multi'; body.innerHTML = (multi ? '<div class="qhint">Select every answer that applies.</div>' : '') + q.view.map(function(oi, k){ return '<label class="qopt"><input type="' + (multi ? 'checkbox' : 'radio') + '" name="qo" value="' + oi + '"><span>' + esc(q.opts[oi]) + '</span></label>'; }).join('');
    act.innerHTML = '<button type="button" class="btn" data-check disabled>Check answer</button>'; var cb = qs('[data-check]', act);
    body.addEventListener('change', function(){ cb.disabled = !qsa('input:checked', body).length; });
    cb.addEventListener('click', function(){
      var chosen = qsa('input:checked', body).map(function(i){ return +i.value; }), want = multi ? q.ans : [0], ok = chosen.length === want.length && want.every(function(w){ return chosen.indexOf(w) >= 0; });
      qsa('.qopt', body).forEach(function(l){ var i = qs('input', l), v = +i.value, right = want.indexOf(v) >= 0; i.disabled = true; l.classList.toggle('right', right); l.classList.toggle('wrong', i.checked && !right); });
      finish(ok, ok ? '' : (multi ? 'The answers that apply are marked in green.' : 'The right answer is marked in green.'));
    });
  } else if (q.t === 'num') {
    body.innerHTML = '<div class="qnum"><input type="text" inputmode="decimal" placeholder="Your answer" aria-label="Your answer"> <span>' + esc(q.unit) + '</span></div>' + (q.tol ? '<div class="qhint">Any answer within ' + q.tol + ' of the exact value counts.</div>' : '');
    act.innerHTML = '<button type="button" class="btn" data-check disabled>Check answer</button>'; var inp = qs('input', body), cb2 = qs('[data-check]', act);
    inp.addEventListener('input', function(){ cb2.disabled = !String(inp.value).trim() || isNaN(parseFloat(String(inp.value).replace(/,/g, ''))); });
    function go(){ if (cb2.disabled) return; var v = parseFloat(String(inp.value).replace(/,/g, '')), ok = Math.abs(v - q.ans) <= q.tol + 1e-9; inp.disabled = true; finish(ok, ok ? '' : 'The answer is ' + q.ans + (q.unit ? ' ' + q.unit : '') + '.'); }
    cb2.addEventListener('click', go); inp.addEventListener('keydown', function(e){ if (e.key === 'Enter') go(); });
  } else if (q.t === 'order') {
    var order = q.view.slice();
    function drawOrder(lock){ body.innerHTML = '<div class="qhint">Use the arrows to put the steps in order, first step at the top.</div><ol class="qord">' + order.map(function(si, k){ return '<li><span>' + esc(q.steps[si]) + '</span>' + (lock ? '' : '<button type="button" class="btn sm ghost" data-up="' + k + '" aria-label="Move up"' + (k === 0 ? ' disabled' : '') + '>&uarr;</button><button type="button" class="btn sm ghost" data-dn="' + k + '" aria-label="Move down"' + (k === order.length - 1 ? ' disabled' : '') + '>&darr;</button>') + '</li>'; }).join('') + '</ol>'; }
    drawOrder(false);
    body.addEventListener('click', function(e){ var b = e.target.closest('button'); if (!b || done) return; var k = b.dataset.up !== undefined ? +b.dataset.up : (b.dataset.dn !== undefined ? +b.dataset.dn : -1); if (k < 0) return; var j = b.dataset.up !== undefined ? k - 1 : k + 1, t = order[k]; order[k] = order[j]; order[j] = t; drawOrder(false); });
    act.innerHTML = '<button type="button" class="btn" data-check>Check order</button>';
    qs('[data-check]', act).addEventListener('click', function(){ var ok = order.every(function(v, i){ return v === i; }); drawOrder(true); finish(ok, ok ? '' : 'The right order is: ' + q.steps.map(function(s, i){ return (i + 1) + ', ' + s; }).join(' ')); });
  } else if (q.t === 'tune') { qzTune(q, body, act, finish, fb); }
}

function qzTune(q, body, act, finish, fb){
  var s = Object.assign({}, q.init), tries = [], passed = false, hintShown = false, ctl = [];
  if (q.kind === 'boost') ctl = [['rounds', 'Rounds', 1, 400, 1], ['depth', 'Tree depth', 1, 8, 1], ['lr', 'Learning rate', 0.01, 1, 0.01]];
  else if (q.kind === 'forest') ctl = [['trees', 'Trees', 1, 150, 1], ['depth', 'Tree depth', 1, 14, 1]];
  else ctl = [['depth', 'Tree depth', 1, 14, 1], ['minleaf', 'Minimum leaf size', 1, 50, 1]];
  body.innerHTML = '<div class="qgoal">' + esc(qzGoalText(q.goal)) + '</div><div class="qctl">' + ctl.map(function(c){ return '<label>' + c[1] + ' <output data-o="' + c[0] + '">' + s[c[0]] + '</output><input type="range" data-k="' + c[0] + '" min="' + c[2] + '" max="' + c[3] + '" step="' + c[4] + '" value="' + s[c[0]] + '"></label>'; }).join('') + '</div><div class="qrunrow"><button type="button" class="btn" data-run>Train this model</button><span class="qrs"></span></div><div class="qtries"></div><div class="qhintbox" hidden></div>';
  qsa('input[type=range]', body).forEach(function(r){ r.addEventListener('input', function(){ s[r.dataset.k] = +r.value; qs('[data-o="' + r.dataset.k + '"]', body).textContent = r.value; }); });
  var run = qs('[data-run]', body), st = qs('.qrs', body), box = qs('.qtries', body);
  function drawTries(){ box.innerHTML = tries.length ? '<table class="qt"><thead><tr><th>Settings</th><th>Test R2</th><th>Training R2</th><th>Leaves</th><th></th></tr></thead><tbody>' + tries.slice().reverse().map(function(t){ return '<tr class="' + (t.ok ? 'ok' : '') + '"><td>' + esc(t.label) + '</td><td>' + t.r.te.toFixed(3) + '</td><td>' + t.r.tr.toFixed(3) + '</td><td>' + t.r.leaves.toLocaleString() + '</td><td>' + (t.ok ? 'meets the goal' : 'not yet') + '</td></tr>'; }).join('') + '</tbody></table>' : ''; }
  function label(){ return q.kind === 'boost' ? s.rounds + ' rounds, depth ' + s.depth + ', rate ' + (+s.lr).toFixed(2) : (q.kind === 'forest' ? s.trees + ' trees, depth ' + s.depth : 'depth ' + s.depth + ', min leaf ' + s.minleaf); }
  run.addEventListener('click', async function(){
    if (passed) return; run.disabled = true; st.textContent = 'Training'; await tick();
    try { var r = await qzRun(q, s), ok = qzPass(q.goal, r); tries.push({ label: label(), r: r, ok: ok }); drawTries(); st.textContent = ok ? '' : 'Not there yet. Change a setting and train again.';
      if (ok) { passed = true; run.disabled = true; act.innerHTML = ''; finish(true, 'Your model met the goal with ' + r.leaves.toLocaleString() + ' leaves in ' + Math.max(1, Math.round(r.ms)) + ' ms.'); } else run.disabled = false;
    } catch (e) { console.error(e); st.textContent = 'Something went wrong. Try again.'; run.disabled = false; }
  });
  act.innerHTML = '<button type="button" class="btn ghost" data-hint>Show a hint</button> <button type="button" class="btn ghost" data-give>Give up and see why</button>';
  qs('[data-hint]', act).addEventListener('click', function(){ var hb = qs('.qhintbox', body); hb.hidden = false; hb.textContent = 'Hint: ' + q.hint; hintShown = true; });
  qs('[data-give]', act).addEventListener('click', function(){ if (passed) return; passed = true; run.disabled = true; act.innerHTML = ''; finish(false, 'You can keep this model in mind: ' + q.hint); });
}

function qzFinish(){
  var root = QZ.root, n = QZ.cur.length, pc = Math.round(100 * QZ.score / n);
  var msg = QZ.score === n ? 'A perfect round.' : (QZ.score >= 8 ? 'A strong round.' : (QZ.score >= 5 ? 'A solid start. The review below shows where to look again.' : 'Worth another look at the lessons. The review below shows where.'));
  var rev = QZ.cur.map(function(q, i){ return '<li class="' + (QZ.log[i] ? 'ok' : 'no') + '"><b>' + (QZ.log[i] ? 'Right' : 'Missed') + '</b> <span class="qtp">' + esc(q.topic) + '</span><div>' + esc(q.text) + '</div>' + (QZ.log[i] ? '' : '<div class="qwhy">' + esc(q.why || '') + '</div>') + '</li>'; }).join('');
  root.innerHTML = '<div class="qpage"><div class="qwrap">' + qzTopbar() + '<div class="qcard qend"><div class="qtopic">Your score</div><div class="qbig">' + QZ.score + ' of ' + n + '</div><p>' + pc + '% right. ' + msg + '</p><div class="qact"><button type="button" class="btn" data-again>New random ' + n + '</button></div></div><ol class="qrev">' + rev + '</ol></div></div>';
  qs('[data-again]', root).addEventListener('click', function(){ qzDraw(); qzRender(); window.scrollTo({ top: 0 }); });
}

var LQ = new Lesson('quiz', 'Quiz');
LQ.custom = function(root){
  QZ.root = root; root.classList.add('qz');
  var intro = '<div class="qpage"><div class="qwrap"><div class="qcard qintro"><div class="qtopic">Quiz</div><h3 class="qtext">Ten questions, drawn at random from a bank of ' + QZ.bank.length + '</h3><p>Some questions are multiple choice, some ask for a number or an order, and some are challenges where you set the parameters of a real model and train it. Challenge models train in your browser on the HLM or logD data, and the goal always counts leaves, the pieces a tree model builds, so a smaller model is a faster one on any computer.</p><div class="qact"><button type="button" class="btn" data-start>Start the quiz</button></div></div></div></div>';
  root.innerHTML = intro; qs('[data-start]', root).addEventListener('click', function(){ qzDraw(); qzRender(); window.scrollTo({ top: 0 }); });
};
