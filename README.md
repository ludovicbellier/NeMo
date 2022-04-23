# NeMo: Neural Modeling toolbox

- This repository contains a series of Python tools and algorithms for applying predictive models to neural activity.

- Specifically developed for the estimation of Spectro-Temporal Receptive Fields (encoding models) and for stimulus reconstruction approaches (decoding models).

- Takes as inputs any continuous representation of the stimulus (e.g., an auditory spectrogram), along with the elicited neural activity (e.g., High-Frequency Activity, 70-150Hz).

- Organized along 5 axes:
  - data preparation, or preprocessing, with tools for assembling the feature lag matrix, creating groups and classes, fixing artifacts and scaling;
  - model selection, with tools to perform StratifiedGroupShuffleSplit, grid search and best of N strategies, and to yield model outputs;
  - core models, with a Robust Multiple Linear Regression with Early Stopping estimator based on Tensorflow, and a MultiLayer Perceptron with Custom Early Stopping estimator based on sklearn;
  - visualizations, with tools to observe data at different steps of the modeling process;
  - other tools.

- Allows for unique controllability of the splitting and model selection, and provides cutting-edge estimators to compute both encoding and decoding models.
