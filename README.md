# NeMo: Neural Modeling toolbox

- This repository contains a series of Python tools and algorithms for applying predictive models to neural activity.

- Specifically developed for the estimation of Spectro-Temporal Receptive Fields (encoding models) and for stimulus reconstruction approaches (decoding models).

- Takes as inputs any continuous representation of the stimulus (e.g., an auditory spectrogram), along with the elicited neural activity (e.g., High-Frequency Activity, 70-150Hz).

- Organized along 6 axes:
  - data preparation, with tools for creating classes and groups, fixing artifacts, and assembling the feature lag matrix;
  - data splitting and scaling, to perform a unique StratifiedGroupShuffleSplit, and to scale data;
  - core models, with a Robust Multiple Linear Regression with Early Stopping estimator based on Tensorflow, and a MultiLayer Perceptron with Custom Early Stopping estimator based on sklearn;
  - model outputs, to yield predicted sets, model coefficients and performance metrics;
  - model selection, with tools to perform a custom grid search and a best of N strategy;
  - visualizations, with tools to observe data at different steps of the modeling process.

- Allows for unique controllability of the splitting and model selection, and provides cutting-edge estimators to compute both encoding and decoding models.
