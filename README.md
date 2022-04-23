# NeMo: Neural Modeling toolbox

This repository contains a series of tools to model neural activity.

Two modeling algorithms are made available for now:
- a Sklearn-based Ridge-regression algorithm;
- a Tensorflow-based linear regression algorithm using early stopping as a regularization method and a choice of stochastic, mini-batch or batch Gradient Descent as a training method.

Both algorithms can be used to obtain Spectro-Temporal Receptive Fields (STRFs), as well as to decode auditory stimuli from elicited neural activity.
