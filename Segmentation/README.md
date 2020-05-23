# Alpha-Garden-Segmentation

##### Contributors: Paul Shao, Mark Presten

This repository contains the codes for training a UNet model for the segmentation task for Alpha Garden.

Here's a brief summary for each of the code files / directories:

- `images/` :
  - `holdout/`: contains the held-out images (excluded from the train-validation process)
  - `train/`: contains the images for training and validating the model
- `models/`: contains the baseline model and the newly saved models from the training process
- `constants.py`: contains constants used throughout the training process; configurable to allow different training parameters.
- `data_utils.py`: contains util functions to help with loading and generating design matrix and response vector (`X_train, y_train, X_valid, y_valid`)
- `eval_utils.py`: contains util functions for evaluating a trained model and predicting masks.
- `train.py`: the **main script** that should be run to train a segmentaion model.

Codes modularized by Paul