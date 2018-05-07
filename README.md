# auditory-neural-predictive-modeling

This repository contains a series of tools to model relationships between auditory stimuli and neural activity.

Two algorithms are made available for now:
- a Sklearn-based Ridge-regression algorithm;
- a Tensorflow-based linear regression algorithm using early stopping as a regularization strategy and a choice of stochastic, mini-batch or batch Gradient Descent.

Both algorithms can be used to obtain Spectro-Temporal Receptive Fields (STRFs), as well as to decode auditory stimuli from elicited neural activity.
