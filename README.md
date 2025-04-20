This repository contains files to train, search hyper-parameters for ML models and submission files to the zindi-canpoy-classification challenge. 

Files
1. [/scr/hyperparameter_search.py](https://github.com/ManojGopale/zindi-canopy-crop-classification/blob/v1/scr/hyperparameter_search.py)
   Creates an optuna study to retreive optimal parameters from an extensive search-space of hyper-parameters. Once study is done, we can pick the best obtained parameters along with the entire data and train the final model for submission.
2. [/scr/main.py](https://github.com/ManojGopale/zindi-canopy-crop-classification/blob/v1/scr/main.py)
   Scipt runs the model with given hyper-paramters and creates a submission file for the competition.

Models:
1. LighGBM - Model is tuned and submision is made to the competition.
2. Prithvi (IBM) - Next model to tune for the competition.
